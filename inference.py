import time
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# 1. Load the DFlash Draft Model
# Note: trust_remote_code=True is required for DFlash. We recommend run on one GPU currently.
model = AutoModel.from_pretrained(
    "./models/Qwen3-4B-DFlash-b16", 
    trust_remote_code=True, 
    dtype="bfloat16", 
    device_map="cuda"
).eval()

# 2. Load the Target Model
target = AutoModelForCausalLM.from_pretrained(
    "./models/Qwen3-4B", 
    dtype="bfloat16", 
    device_map="cuda"
).eval()

# 3. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./models/Qwen3-4B")
# Essential: Add the mask token required for diffusion steps
tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

# 4. Prepare Input
prompt = "introduce ffmpeg in details."
messages = [
    {"role": "user", "content": prompt}
]
# Note: this draft model is used for thinking mode disabled
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
max_new_tokens = 100

# Debug: token ids
print(f"[DEBUG] mask_token_id={tokenizer.mask_token_id} eos_token_id={tokenizer.eos_token_id}")

# 5. Run Speculative Decoding (DFlash)
generate_ids, dflash_stats = model.spec_generate(
    input_ids=model_inputs["input_ids"],
    max_new_tokens=max_new_tokens,
    temperature=0.0,
    target=target,
    mask_token_id=tokenizer.mask_token_id,
    stop_token_ids=[tokenizer.eos_token_id],
    return_stats=True,
)

print(tokenizer.decode(generate_ids[0], skip_special_tokens=True))

# 6. Run Baseline (target-only) for comparison
print("\n=====Run Baseline model for comparison . =====\n")
target.eval()
with torch.inference_mode():
    if hasattr(target, "device"):
        dev = target.device
    else:
        dev = next(target.parameters()).device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_start = time.perf_counter()
    baseline_ids = target.generate(
        model_inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_end = time.perf_counter()
baseline_total_ms = (t_end - t_start) * 1000.0
baseline_generated = baseline_ids.shape[1] - model_inputs["input_ids"].shape[1]
baseline_throughput = baseline_generated / (baseline_total_ms / 1000.0) if baseline_total_ms > 0 else 0.0
baseline_tpot_ms = baseline_total_ms / baseline_generated if baseline_generated else 0.0
# Baseline has no separate prefill timing; use total/generated as TPOT, same as TTFT placeholder
baseline_ttft_ms = baseline_tpot_ms
print(
    f"[Target-only] generated={baseline_generated}, TTFT={baseline_ttft_ms:.3f} ms, "
    f"TPOT={baseline_tpot_ms:.3f} ms/token, total_generate={baseline_total_ms:.3f} ms, "
    f"throughput={baseline_throughput:.3f} tokens/s"
)
print("[Target-only stage timings] wall-clock (ms): (not instrumented in this script)")
print(tokenizer.decode(baseline_ids[0], skip_special_tokens=True))

# Compare
dflash_throughput = dflash_stats["throughput"]
ratio = dflash_throughput / baseline_throughput if baseline_throughput > 0 else 0.0
print(f"\n[Compare] dflash / baseline throughput ratio={ratio:.3f}")
