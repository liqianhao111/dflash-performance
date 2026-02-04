LOG in nvidia RTX 5080:
[DEBUG] mask_token_id=151669 eos_token_id=151645
dflash_cfg.block_size is 16
target_layer_ids is
1 9 17 25 33
[Tokens] prompt=18, generated=100, draft_accepted=56, target_only=16, avg_accept_per_block=2.273
[Latency] TTFT=189.333 ms, TPOT=12.355 ms/token, total_generate=1235.463 ms, throughput=80.941 tokens/s
[Stage timings] wall-clock (ms):
  prefill target: wall 189.333 ms (max 189.333, 1 runs)
  target ctx: wall 0.000 ms (max 0.000, 0 runs)
  embed: wall 27.114 ms (max 26.055, 44 runs)
  draft: wall 118.566 ms (max 5.431, 44 runs)
  lm_head: wall 40.262 ms (max 1.091, 44 runs)
  target verify: wall 812.939 ms (max 30.842, 44 runs)
  prep: wall 0.000 ms (max 0.000, 0 runs)
  postproc: wall 33.346 ms (max 24.355, 44 runs)
  kv trim: wall 0.000 ms (max 0.000, 0 runs)
  hidden append: wall 0.000 ms (max 0.000, 0 runs)
  set_tensor: wall 0.000 ms (max 0.000, 0 runs)
  get_tensor: wall 0.000 ms (max 0.000, 0 runs)
  argmax: wall 0.000 ms (max 0.000, 0 runs)
  make_tensor: wall 0.000 ms (max 0.000, 0 runs)
  other (untracked): wall 0.000 ms (max 0.000, 0 runs)
[Draft acceptance per step] [5, 1, 1, 1, 2, 1, 2, 4, 2, 3, 1, 1, 2, 4, 4, 1, 1, 2, 1, 1, 4, 3, 2, 1, 1, 2, 1, 2, 6, 6, 3, 2, 3, 3, 1, 3, 1, 2, 2, 1, 2, 4, 2, 3]
user
introduce ffmpeg in details.
assistant
<think>

</think>

FFmpeg is a powerful, cross-platform open-source software project that is used for **record, convert, and stream audio and video**. It is widely used in the media processing industry, from streaming services to video editing software, and is a key tool for developers and media professionals.

---

## 🧩 What is FFmpeg?

FFmpeg is a collection of libraries and command-line tools that can:

- **Decode** audio and video streams from various formats.
- **Encode** audio and video

=====Run Baseline model for comparison . =====

The following generation flags are not valid and may be ignored: ['temperature', 'top_p', 'top_k']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
[Target-only] generated=100, TTFT=15.470 ms, TPOT=15.470 ms/token, total_generate=1547.027 ms, throughput=64.640 tokens/s
[Target-only stage timings] wall-clock (ms): (not instrumented in this script)
user
introduce ffmpeg in details.
assistant
<think>

</think>

FFmpeg is a powerful, cross-platform open-source software project that is used for **record, convert, and stream audio and video**. It is widely used in the media processing industry and is a key tool for developers, content creators, and system administrators who need to handle multimedia files.

---

## 🧩 What is FFmpeg?

FFmpeg is a collection of libraries and command-line tools that can:

- **Decode** audio and video streams from various formats.
- **Encode** audio and

[Compare] dflash / baseline throughput ratio=1.252