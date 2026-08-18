# LLM benchmarks: Nothing Phone (2) — CPU vs GPU vs NPU

On-device LLM inference measured on a **Nothing Phone (2)** (Snapdragon **8+ Gen 1** / SM8475, 12 GB RAM, Android 16, stock ROM, **no root**), August 2026. Every engine the SoC has, tested against the same models where possible:

| Engine | Backend | Status |
|---|---|---|
| CPU (Cortex-X2 + A710, armv8.2 `dotprod+i8mm+fp16`) | llama.cpp | ✅ measured |
| GPU (Adreno 730) | llama.cpp OpenCL (Qualcomm Adreno kernels) | ✅ measured |
| NPU (Hexagon HTP **v69**) | ExecuTorch QNN | 🔄 in progress |

**TL;DR: on this SoC generation the marketing hierarchy (NPU > GPU > CPU) is backwards for LLMs. The CPU wins nearly everything.** The Adreno 730 loses token generation by 5–10×, and only approaches parity on large-model prompt prefill. The Hexagon v69 is unsupported by every mainstream NPU framework (Qualcomm Genie, llama.cpp's Hexagon backend, and MNN all require v73+); the only working path is ExecuTorch QNN, and it's limited to ~1B-class models.

## Results

### Text: GPU (OpenCL, `-ngl 99`) vs CPU (`-ngl 0 -t 4`), same GGUF file

`llama-bench -p 512 -n 128 -r 3` — tokens/sec, mean ± σ. pp = prompt prefill, tg = generation ("typing speed").

| Model | Quant | GPU pp512 | CPU pp512 | GPU tg128 | CPU tg128 |
|---|---|---:|---:|---:|---:|
| Qwen3.5-2B | Q4_0 | 55.8 ± 1.0 | **99.8 ± 1.9** | 2.68 ± 0.08 | **15.0 ± 0.7** |
| Qwen3.5-2B | Q4_K_M | 44.1 ± 0.1 | **56.4 ± 4.2** | 2.65 ± 0.15 | **11.6 ± 1.0** |
| Qwen3.5-9B | Q4_0 | **23.6 ± 0.0** | 18.4 ± 0.1 | 1.57 ± 0.13 | **3.49 ± 0.14** |
| Llama-3.1-8B | Q4_K_M | 10.2 ± 0.0 | **14.9 ± 0.1** | 1.99 ± 0.00 | **4.11 ± 0.09** |

**The GPU's single win** (bold left column, row 3): big-model prompt prefill *in the Adreno-optimized Q4_0 format* — 9B prefill 28% faster than CPU. With a non-optimized quant (8B Q4_K_M) it loses even that. CPU wins decode everywhere, 2.1–5.6×.

### Vision: SmolVLM2-500M Q8_0, 1024px screenshot (`llama-mtmd-cli`)

| Engine | Image encode (per chunk) | Result quality |
|---|---:|---|
| **CPU** (`--no-mmproj-offload -t 4`) | **1.83 s** | Correct: read clock, date |
| GPU (`-ngl 99`, encoder on GPU) | 7.54 s | Correct: read clock, date, "84% — Charging rapidly" |

**CPU encodes vision 4.1× faster than the Adreno 730.** SmolVLM2 ran clean on both engines — the earlier kernel panic was specific to Qwen3-VL's mmproj allocation bug, not vision per se.

Earlier CPU-only round (same phone, same llama.cpp flags):

| Model | Quant | CPU pp512 | CPU tg128 | Best threads |
|---|---|---:|---:|---|
| Qwen2.5-1.5B | Q4_K_M | 94.9 | 21.2 | 4 |
| Llama-3.1-8B | Q6_K | 10.7 | 2.37 | 4 |

More threads than 4 is **slower** (big.LITTLE: the 4 big cores win; adding LITTLE cores adds sync overhead).

### Findings

1. **CPU beats GPU on everything at 2B** — decode by 5.6×, prefill by 1.8×. The Adreno 730's OpenCL decode suffers per-token synchronization overhead that no kernel tuning fixes on this driver generation. (llama.cpp's own verified-device table starts at Adreno 750; A730 runs, but this is why it isn't listed.)
2. **Q4_0 is the right quant for BOTH engines on ARM.** CPU prefill jumps 77% over Q4_K_M (99.8 vs 56.4) because llama.cpp runtime-repacks Q4_0 into ARM-interleaved layout. It's also the format Qualcomm's Adreno kernels are optimized for. Use Q4_0 on phones.
3. **Vision on GPU is broken, and on CPU it's dangerous** *(as of llama.cpp b~6150 / commit adb55e5)*: Qwen3-VL-2B's mmproj encoder computed a **13.6 GB** allocation for a 1080×2412 screenshot — segfault on OpenCL, and on CPU the attempt **OOM'd the device into a kernel panic and rebooted the phone** (`sys.boot.reason=kernel_panic,...oom`). Cap image resolution before the encoder, always. (Vision re-test with SmolVLM2-500M pending.)
4. **The Hexagon v69 NPU is orphaned by the ecosystem.** llama.cpp's ggml-hexagon backend (merged Oct 2025) ships kernels for v73/v75/v79/v81 only. Qualcomm Genie requires v73+. The one working path for LLM-on-NPU on 8/8+ Gen 1 is **ExecuTorch's QNN backend** (officially supports SM8475); community-reported ~31 tok/s decode for Qwen3-0.6B on v69. **We did not verify this** — see Status. Treat the 31 tok/s figure as someone else's claim, not a result of ours.
5. **Screen-off WiFi power-save throttles adb to ~100 KB/s** (from 2–6 MB/s screen-on, 39 MB/s USB). If you benchmark over wireless adb, keep the screen awake or use USB.

## Method

- **Same file, both engines**: every GPU-vs-CPU pair ran the identical GGUF — the delta is purely the execution engine.
- **Thermal gating**: ≥45 s between rounds, then poll `dumpsys thermalservice` until throttle status ≤ 1. No run starts hot.
- 3 repetitions per metric (σ reported); screen off during runs; battery ≥ 15% enforced; device on charger.
- No root ⇒ no fixed clocks: these are as-experienced numbers, not lab numbers. The gating + σ keep them honest.

## Reproduce

Build llama.cpp for Android with the OpenCL (Adreno) backend — see [`scripts/build-llama-opencl-android.sh`](scripts/build-llama-opencl-android.sh) (NDK r27, OpenCL headers + ICD loader into the NDK sysroot, then `-DGGML_OPENCL=ON -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+i8mm+fp16`).

```bash
adb push build/bin/llama-bench /data/local/tmp/
adb shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/vendor/lib64 \
  ./llama-bench -m model.gguf -p 512 -n 128 -ngl 99 -r 3'   # GPU
#                                              -ngl 0 -t 4  # CPU
```

`LD_LIBRARY_PATH=/vendor/lib64` lets the binary find the vendor's `libOpenCL.so`. Verify GPU detection with `llama-cli --list-devices` → `GPUOpenCL: QUALCOMM Adreno(TM)`.

Driver used for the full autonomous matrix (waits for models, benches both engines, thermal-gates): [`scripts/bench-driver.sh`](scripts/bench-driver.sh).

### Models tested

| Model | File | Source |
|---|---|---|
| Qwen3.5-2B | Q4_0 / Q4_K_M | [bartowski/Qwen_Qwen3.5-2B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF) |
| Qwen3.5-9B | Q4_0 | [bartowski/Qwen_Qwen3.5-9B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF) |
| Llama-3.1-8B | Q4_K_M | [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) |
| Qwen3-VL-2B (vision) | Q4_K_M + mmproj Q8_0 | [Qwen/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF) |
| SmolVLM2-500M (vision) | Q8_0 + mmproj | [ggml-org/SmolVLM2-500M-Video-Instruct-GGUF](https://huggingface.co/ggml-org/SmolVLM2-500M-Video-Instruct-GGUF) |

Raw logs in [`results/`](results/).

## Status

- [x] CPU baseline (2 models × 3 thread counts)
- [x] GPU vs CPU: Qwen3.5-2B (Q4_0 + Q4_K_M)
- [x] GPU vs CPU: Qwen3.5-9B Q4_0
- [x] GPU vs CPU: Llama-3.1-8B Q4_K_M
- [x] Vision: SmolVLM2-500M, CPU + GPU (Qwen3-VL parked — encoder allocation bug, see finding 3)
- [ ] NPU: ExecuTorch QNN, Qwen3-0.6B on Hexagon v69 — **not run.** Artifacts were staged (`.pte` + QNN libs, run commands fixed) but the transfer was cut short, and the phone was then restored to stock Nothing OS and relocked, ending device access. Nothing here is an NPU measurement.

### Verdict

For LLMs on the Snapdragon 8+ Gen 1, **run everything on the CPU with `dotprod+i8mm+fp16` and Q4_0 weights.** The GPU earns exactly one narrow lane (Q4_0 big-model prefill, +28% at 9B) that rarely justifies its 2–5× decode penalty; a hybrid "GPU prefill → CPU decode" split is theoretically optimal but llama.cpp doesn't support it per-phase. Vision belongs on the CPU too. The NPU — if the v69 ExecuTorch path verifies — is the efficiency lane for sub-1B models, not a general accelerator.
## Continuing this work

**Device access first.** The phone was restored to stock Nothing OS (PongIND
B4.1) and the bootloader **relocked** after these runs, so re-testing means
unlocking again — which wipes userdata. Everything in this repo was measured on
a stock, unrooted device, and the CPU/GPU numbers do not need root to reproduce.

**The NPU lane, concretely.** Hexagon **v69** on SM8475 is the constraint:
llama.cpp's `ggml-hexagon` ships kernels for v73/v75/v79/v81, and Qualcomm Genie
wants v73+, so neither runs here. The only viable path is **ExecuTorch's QNN
backend**, which officially lists SM8475. To finish it you need, on-device:
the exported `.pte`, the QNN runtime libs (`libQnnHtp*.so` and the v69 skel),
and `LD_LIBRARY_PATH` plus `ADSP_LIBRARY_PATH` pointed at them. Export the model
with the QNN partitioner on a Linux host — the arm64 runner and the exporter are
separate steps, and the export is the slow one.

**Method note that cost us a phone reboot.** Cap image resolution *before* the
vision encoder. Qwen3-VL-2B's mmproj computed a 13.6 GB allocation for a
1080×2412 screenshot and OOM'd the device into a kernel panic (finding 3). Any
future vision run should assert the computed allocation is sane before calling
the encoder, not after.

**Trust functional checks over declared support.** The pattern that held
throughout: a backend's own compatibility table (llama.cpp's verified-device
list starting at Adreno 750) predicted the result better than "it runs".
Measure the thing; don't infer it from the fact that it launched.

