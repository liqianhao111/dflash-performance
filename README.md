# DFlash Performance

基于 **NVIDIA RTX 5080** 的 DFlash 推测解码性能记录。

---

## 环境与配置

| 项目 | 值 |
|------|-----|
| 设备 | NVIDIA RTX 5080 |
| `mask_token_id` | 151669 |
| `eos_token_id` | 151645 |
| `block_size` | 16 |
| `target_layer_ids` | 1, 9, 17, 25, 33 |

---

## DFlash 推理结果

### Token 统计

| 指标 | 数值 |
|------|------|
| prompt tokens | 18 |
| generated tokens | 100 |
| draft accepted | 56 |
| target-only steps | 16 |
| avg accept per block | 2.273 |

### 延迟与吞吐

| 指标 | 数值 |
|------|------|
| **TTFT** | 189.333 ms |
| **TPOT** | 12.355 ms/token |
| **total generate** | 1235.463 ms |
| **throughput** | **80.941 tokens/s** |

### 各阶段耗时 (wall-clock ms)

| Stage | Total | Max | Runs |
|-------|-------|-----|-----|
| prefill target | 189.333 | 189.333 | 1 |
| embed | 27.114 | 26.055 | 44 |
| draft | 118.566 | 5.431 | 44 |
| lm_head | 40.262 | 1.091 | 44 |
| target verify | 812.939 | 30.842 | 44 |
| postproc | 33.346 | 24.355 | 44 |
| 其余 | 0.000 | — | — |

### Draft acceptance per step

```
[5, 1, 1, 1, 2, 1, 2, 4, 2, 3, 1, 1, 2, 4, 4, 1, 1, 2, 1, 1, 4, 3, 2, 1, 1, 2, 1, 2, 6, 6, 3, 2, 3, 3, 1, 3, 1, 2, 2, 1, 2, 4, 2, 3]
```

---

## Baseline（仅 Target 模型）

| 指标 | 数值 |
|------|------|
| generated | 100 |
| TTFT | 15.470 ms |
| TPOT | 15.470 ms/token |
| total generate | 1547.027 ms |
| **throughput** | **64.640 tokens/s** |

---

## 对比

| 项目 | 数值 |
|------|------|
| **dflash / baseline throughput ratio** | **1.252** |

DFlash 推测解码相比单跑 target 约 **25.2%** 吞吐提升。

---

## 示例输出（DFlash）

**Prompt:** *introduce ffmpeg in details.*

> FFmpeg is a powerful, cross-platform open-source software project that is used for **record, convert, and stream audio and video**. It is widely used in the media processing industry, from streaming services to video editing software, and is a key tool for developers and media professionals.
>
> ## What is FFmpeg?
>
> FFmpeg is a collection of libraries and command-line tools that can:
>
> - **Decode** audio and video streams from various formats.
> - **Encode** audio and video …

---

## 示例输出（Baseline）

> FFmpeg is a powerful, cross-platform open-source software project that is used for **record, convert, and stream audio and video**. It is widely used in the media processing industry and is a key tool for developers, content creators, and system administrators who need to handle multimedia files.
>
> ## What is FFmpeg?
>
> FFmpeg is a collection of libraries and command-line tools that can:
>
> - **Decode** audio and video streams from various formats.
> - **Encode** audio and …
