from typing import Optional, Callable
from typing_extensions import Unpack, Tuple
import time
import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache
from .utils import build_target_layer_ids, extract_context_feature, sample

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get("target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers))
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)
    
    def _sync(self, device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        mask_token_id: int,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
        print_stats: bool = True,
        return_stats: bool = False,
    ):
        self.eval()
        device = next(self.parameters()).device
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # --- Stage timings (wall-clock ms) ---
        t_prefill_target = 0.0
        t_embed_list = []
        t_draft_list = []
        t_lm_head_list = []
        t_target_verify_list = []
        t_postproc_list = []
        t_total_start = time.perf_counter()

        # Prefill stage
        self._sync(device)
        t0 = time.perf_counter()
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        self._sync(device)
        t_prefill_target = (time.perf_counter() - t0) * 1000.0

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature)
        target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        num_decode_steps = 0
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]

            # embed
            self._sync(device)
            t0 = time.perf_counter()
            noise_embedding = target.model.embed_tokens(block_output_ids)
            self._sync(device)
            t_embed_list.append((time.perf_counter() - t0) * 1000.0)

            # draft (dflash forward)
            self._sync(device)
            t0 = time.perf_counter()
            draft_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size+1:, :]
            self._sync(device)
            t_draft_list.append((time.perf_counter() - t0) * 1000.0)

            # lm_head
            self._sync(device)
            t0 = time.perf_counter()
            draft_logits = target.lm_head(draft_hidden)
            self._sync(device)
            t_lm_head_list.append((time.perf_counter() - t0) * 1000.0)

            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)

            # target verify
            self._sync(device)
            t0 = time.perf_counter()
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            self._sync(device)
            t_target_verify_list.append((time.perf_counter() - t0) * 1000.0)

            # postproc (sample + acceptance)
            self._sync(device)
            t0 = time.perf_counter()
            posterior = sample(output.logits, temperature)
            acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(output.hidden_states, self.target_layer_ids)[:, :acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            self._sync(device)
            t_postproc_list.append((time.perf_counter() - t0) * 1000.0)

            num_decode_steps += 1
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
            ):
                break

        t_total_end = time.perf_counter()
        total_generate_ms = (t_total_end - t_total_start) * 1000.0

        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_t).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

        generated_tokens = output_ids.shape[1] - num_input_tokens
        draft_accepted = sum(acceptance_lengths) - len(acceptance_lengths)  # draft tokens only (exclude 1 target per step)
        target_only_steps = sum(1 for al in acceptance_lengths if al == 1)  # steps where only target token taken
        avg_accept_per_block = sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0
        ttft_ms = t_prefill_target
        tpot_ms = total_generate_ms / generated_tokens if generated_tokens else 0.0
        throughput = (generated_tokens / (total_generate_ms / 1000.0)) if total_generate_ms > 0 else 0.0

        stats = {
            "generated_tokens": generated_tokens,
            "total_generate_ms": total_generate_ms,
            "throughput": throughput,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
            "acceptance_lengths": acceptance_lengths,
        }

        if print_stats:
            print(f"dflash_cfg.block_size is {block_size}")
            print(f"target_layer_ids is")
            print(" ".join(map(str, self.target_layer_ids)))
            print(
                f"[Tokens] prompt={num_input_tokens}, generated={generated_tokens}, "
                f"draft_accepted={draft_accepted}, target_only={target_only_steps}, "
                f"avg_accept_per_block={avg_accept_per_block:.3f}"
            )
            print(
                f"[Latency] TTFT={ttft_ms:.3f} ms, TPOT={tpot_ms:.3f} ms/token, "
                f"total_generate={total_generate_ms:.3f} ms, throughput={throughput:.3f} tokens/s"
            )
            sum_embed = sum(t_embed_list)
            sum_draft = sum(t_draft_list)
            sum_lm_head = sum(t_lm_head_list)
            sum_target_verify = sum(t_target_verify_list)
            sum_postproc = sum(t_postproc_list)
            n_run = len(t_embed_list)
            print("[Stage timings] wall-clock (ms):")
            print(f"  prefill target: wall {t_prefill_target:.3f} ms (max {t_prefill_target:.3f}, 1 runs)")
            print(f"  target ctx: wall 0.000 ms (max 0.000, 0 runs)")
            max_embed = max(t_embed_list) if t_embed_list else 0.0
            max_draft = max(t_draft_list) if t_draft_list else 0.0
            max_lm = max(t_lm_head_list) if t_lm_head_list else 0.0
            max_verify = max(t_target_verify_list) if t_target_verify_list else 0.0
            max_post = max(t_postproc_list) if t_postproc_list else 0.0
            print(f"  embed: wall {sum_embed:.3f} ms (max {max_embed:.3f}, {n_run} runs)")
            print(f"  draft: wall {sum_draft:.3f} ms (max {max_draft:.3f}, {n_run} runs)")
            print(f"  lm_head: wall {sum_lm_head:.3f} ms (max {max_lm:.3f}, {n_run} runs)")
            print(f"  target verify: wall {sum_target_verify:.3f} ms (max {max_verify:.3f}, {len(t_target_verify_list)} runs)")
            print("  prep: wall 0.000 ms (max 0.000, 0 runs)")
            print(f"  postproc: wall {sum_postproc:.3f} ms (max {max_post:.3f}, {n_run} runs)")
            print("  kv trim: wall 0.000 ms (max 0.000, 0 runs)")
            print("  hidden append: wall 0.000 ms (max 0.000, 0 runs)")
            print("  set_tensor: wall 0.000 ms (max 0.000, 0 runs)")
            print("  get_tensor: wall 0.000 ms (max 0.000, 0 runs)")
            print("  argmax: wall 0.000 ms (max 0.000, 0 runs)")
            print("  make_tensor: wall 0.000 ms (max 0.000, 0 runs)")
            print("  other (untracked): wall 0.000 ms (max 0.000, 0 runs)")
            print(f"[Draft acceptance per step] {acceptance_lengths}")

        return (output_ids, stats) if return_stats else output_ids
