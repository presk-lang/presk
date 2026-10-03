> The PyTorch baseline at 0.10.0 step 7 (D97: `make pytorch-baseline` on presk `81bae21`, the manual's edits set aside, begun 2026-10-01 at 19:13 on a quiet machine, the run of record), copied from `out/pytorch-baseline/report.md`: the plans `presk plan` gives now state `Linear.apply/%3`'s bias as its epilogue (`dot+add`), and the C programs are built `-O2`, by their Makefiles. T2 and T4 measured again. Beside it, the same measurement with the C programs built `-O3 -mcpu=native` and `-O3` (`docs/pytorch-baseline-0.10.0-step-7-o3-native.md`, `-o3.md`), and in D97 an interleaved comparison with `main`'s programs (948521f) that splits what the epilogue bought from what GCC's inlining did.

# The course GPT on three sides: presk 0.10.0-dev against PyTorch

The course GPT (`corpus/nlp/05_gpt.psk`, D = 384, 6 heads, 6 blocks, V = 92) at B = 16, T = 256, `f32` on every side, with the checkpoint `out/course_probe_f32/final.psk.d` and the token indices the C export synthesizes; 2026-10-01, presk `81bae21`, `experiments/pytorch_baseline.sh`. Three sides: presk's native backend (`presk bench --plan-only`, `presk train`), the C program `presk export --target c` writes (GCC 13, `-O2`, OpenMP), and PyTorch 2.14.0+cpu running the model `presk export --target pytorch` writes. Each time is the median of seven calls or steps, the first call or step, which compiles natively, left out on the native and PyTorch sides; the C program's seven calls are all counted.

**Operations.** The forward pass is 96.93 GFLOP of contractions, a training step 290.78 (each contraction and its two derivatives); PyTorch's counter (`torch.utils.flop_counter`) gives 96.93 and 290.78. The rates below divide these by the median, whatever a side executes: presk's gradient recomputes the forward pass's products (below, where the C program's time goes).

## The forward pass

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 1 392.9 | 1 392.3 | 69.6 | 8.65 |
| 10 (ten X925, pinned) | native | plan | 321.5 | 321.0 | 301.5 | 2.00 |
| 10 (ten X925, pinned) | C | default | 896.4 | 886.9 | 108.1 | 5.57 |
| 10 (ten X925, pinned) | C | plan | 277.0 | 274.0 | 349.9 | 1.72 |
| 10 (ten X925, pinned) | PyTorch | its own | 161.0 | 146.4 | 602.1 | 1.00 |
| 20 (twenty, unpinned) | native | default | 2 117.5 | 2 065.7 | 45.8 | 8.31 |
| 20 (twenty, unpinned) | native | plan | 321.4 | 320.7 | 301.5 | 1.26 |
| 20 (twenty, unpinned) | C | default | 1 480.9 | 1 410.3 | 65.4 | 5.81 |
| 20 (twenty, unpinned) | C | plan | 279.7 | 268.6 | 346.5 | 1.10 |
| 20 (twenty, unpinned) | PyTorch | its own | 254.7 | 237.8 | 380.5 | 1.00 |

## One training step

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native, `presk train` | default | 13 955.1 | 20.8 | 14.03 |
| 10 (ten X925, pinned) | native, `presk train` | plan | 1 429.0 | 203.5 | 1.44 |
| 10 (ten X925, pinned) | native, `loss_grad` 13 889 + `adamw.step` 20 | default | 13 909.4 | 20.9 | 13.98 |
| 10 (ten X925, pinned) | native, `loss_grad` 1 311 + `adamw.step` 16 | plan | 1 326.8 | 219.2 | 1.33 |
| 10 (ten X925, pinned) | C, `loss_grad` 11 689 + `adamw.step` 17 | default | 11 706.2 | 24.8 | 11.77 |
| 10 (ten X925, pinned) | C, `loss_grad` 1 002 + `adamw.step` 15 | plan | 1 017.0 | 285.9 | 1.02 |
| 10 (ten X925, pinned) | PyTorch, `train.py` | its own | 994.8 | 292.3 | 1.00 |
| 20 (twenty, unpinned) | native, `presk train` | default | 13 900.6 | 20.9 | 10.24 |
| 20 (twenty, unpinned) | native, `presk train` | plan | 1 438.7 | 202.1 | 1.06 |
| 20 (twenty, unpinned) | native, `loss_grad` 13 641 + `adamw.step` 24 | default | 13 665.5 | 21.3 | 10.07 |
| 20 (twenty, unpinned) | native, `loss_grad` 1 314 + `adamw.step` 22 | plan | 1 335.8 | 217.7 | 0.98 |
| 20 (twenty, unpinned) | C, `loss_grad` 11 782 + `adamw.step` 19 | default | 11 801.3 | 24.6 | 8.70 |
| 20 (twenty, unpinned) | C, `loss_grad` 1 054 + `adamw.step` 20 | plan | 1 074.2 | 270.7 | 0.79 |
| 20 (twenty, unpinned) | PyTorch, `train.py` | its own | 1 357.2 | 214.2 | 1.00 |

The C step is its two programs' medians added, and so is the native step by `presk bench` (since 0.10.0 step 6, T4's step like for like); neither drops (the export drops keys, so its dropout is the expectation, and `presk bench` does not train), where `presk train` and `train.py` drop with their own masks. `train.py`'s step builds the module `GPT(256)` at every call of the exported `loss` (`torch.func.functional_call` on a fresh module, its initializers drawing every parameter): 21.5 ms on ten threads, inside its time.

Every layout and thread count gives the same `f32` logits and the same gradient, bit for bit, and the native training run on ten threads under the plan ends on the default layout's weights byte for byte: yes.

## T4: the native backend against the C program

Plan 0.10.0's statement T4 (§3.5, as amended after step 5): the native backend's forward pass and training step under the plans take at most 1.3× the C program's, same plans, same cores; the step compared like for like, the native `loss_grad` and `adamw.step` timed by `presk bench` against the C program's, neither drawing dropout masks, `presk train`'s step reported beside it. Judged on the ten X925 cores pinned, the row of record; every core unpinned reported.

| threads | workload | native ms | C ms | native ÷ C | T4 (≤ 1.3×) |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | the forward pass | 321.5 | 277.0 | 1.16 | holds |
| 10 (ten X925, pinned) | the step: `loss_grad` + `adamw.step` | 1 326.8 | 1 017.0 | 1.30 | fails |
| 10 (ten X925, pinned) | beside: `presk train`'s step, drawing its masks | 1 429.0 | 1 017.0 | 1.41 | — |
| 20 (twenty, unpinned) | the forward pass | 321.4 | 279.7 | 1.15 | (reported) |
| 20 (twenty, unpinned) | the step: `loss_grad` + `adamw.step` | 1 335.8 | 1 074.2 | 1.24 | (reported) |
| 20 (twenty, unpinned) | beside: `presk train`'s step, drawing its masks | 1 438.7 | 1 074.2 | 1.34 | — |

## The linear products alone (ten threads, pinned)

`Linear{In,Out}.apply` of the course GPT, `x · w + b` at x : [16, 256, In], under the layout the forward plan gives `Linear.apply` (one layout for its three shapes); PyTorch `x @ w + b`.

A call's time on each side, its product and its bias together; for the native backend also its kernels' alone (`presk bench --calibrate`: the product's and the bias's, or one when the plan states the bias as the product's epilogue, plan 0.10.0 step 7), the rest of its call being its host's buffers and copies. GFLOP/s: the product's operations over the call's time, or over the kernels'.

| product | GFLOP | native ms | its kernels ms | C ms | PyTorch ms | native GFLOP/s | its kernels | C | PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| [16, 256, 384] · [384, 384] | 1.21 | 3.10 | 3.69 | 2.08 | 1.25 | 390 | 328 | 580 | 969 |
| [16, 256, 384] · [384, 1536] | 4.83 | 12.67 | 13.47 | 8.10 | 6.45 | 381 | 359 | 597 | 749 |
| [16, 256, 1536] · [1536, 384] | 4.83 | 12.56 | 12.94 | 8.29 | 4.84 | 385 | 374 | 583 | 997 |

## Kernels and host, the forward pass under its plan (ten threads, pinned)

The same kernels on both presk sides (C1), so what differs is who compiles them (Cranelift against GCC `-O2`) and what runs around them. The native kernels' time is `presk bench --calibrate`'s (every launch timed, its threads' start included); the C kernels' is the instrumented program's (below, three calls after a first; the program's median call is uninstrumented). The host is the call's median less its kernels. Under the plan only: `--calibrate` runs a nest no plan states scalar, where the native backend gives it its natural lanes, so under the default layout it times other kernels; the forward plan leaves only the host's small kernels (the causal mask, the positions) unstated.

| side | the call, ms | its kernels, ms | the rest, the host's, ms |
|---|---|---|---|
| native | 321.5 | 320.7 | 0.8 |
| C | 277.0 | 264.7 | 12.3 |

## Distances to the native backend's `f64` logits

| side | ‖Δ‖₂ / ‖y‖₂ | max \|Δ\| | max \|Δ\| / max \|y\| | positions whose largest logit moves |
|---|---|---|---|---|
| PyTorch float64 (a check of the comparison) | 6.51e-16 | 1.08e-17 | 2.40e-15 | 0 of 4096 |
| native f32, and the C program | 3.26e-07 | 6.70e-09 | 1.49e-06 | 1 of 4096 |
| PyTorch float32 | 1.62e-07 | 2.74e-09 | 6.08e-07 | 0 of 4096 |

The checkpoint is a probe a few steps from the initialization (its head's largest weight is 1.8·10⁻⁵), so the logits are of order 10⁻³ and a position whose largest logit moves is a near-tie.

## Where the C program's time goes (ten threads, pinned, three calls after a first)

A copy of `model.c` with every launch timed on the thread that makes it, a kernel's call or the OpenMP region that divides it, its team's start and its wait for the slowest thread included (`experiments/kernel_time.py`, `kernel_time.c`); the kernels are the program's, and its outputs are unchanged. Four calls, the first left out. GFLOP counts every launch, recomputations included.

### The forward pass, default layout

The instrumented program: 831.0 ms a call (the median of its four); its kernels 813.3 ms; the rest, the host's, 17.7 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.684 | 142 | 84.2 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 173 | 0.0 | 0.129 |  | 15.8 % |
| every kernel | 222 | | **0.813** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 122 | 0.237 | 29.1 % |
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 130 | 0.223 | 27.5 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 223 | 0.130 | 16.0 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 84 | 0.057 | 7.0 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 136 | 0.035 | 4.4 % |
| `k16`: `softmax axis seq2 (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, heads: 6, seq: …` | 6 |  | 0.031 | 3.8 % |
| `k23`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.020 | 2.4 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.019 | 2.3 % |

### The forward pass under its plan

The instrumented program: 281.6 ms a call (the median of its four); its kernels 264.7 ms; the rest, the host's, 16.9 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.177 | 549 | 66.7 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 143 | 0.0 | 0.088 |  | 33.3 % |
| every kernel | 192 | | **0.265** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k10`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 2…` | 24 | 538 | 0.054 | 20.4 % |
| `k23`: `dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq:…` | 6 | 562 | 0.052 | 19.5 % |
| `k21`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq:…` | 6 | 599 | 0.048 | 18.3 % |
| `k16`: `softmax axis seq2, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, hea…` | 6 |  | 0.020 | 7.7 % |
| `k22`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.016 | 5.9 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 444 | 0.011 | 4.1 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 449 | 0.011 | 4.1 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.010 | 3.9 % |

### The gradient, `loss_grad`, default layout

The instrumented program: 11 744.9 ms a call (the median of its four); its kernels 11 538.5 ms; the rest, the host's, 206.4 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.875 | 133 | 7.6 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 10.348 | 17 | 89.7 % |
| every other kernel | 865 | 0.0 | 0.316 |  | 2.7 % |
| every kernel | 1012 | | **11.538** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k58`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 9 | 3.252 | 28.2 % |
| `k53`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 9 | 3.107 | 26.9 % |
| `k63`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 17 | 1.722 | 14.9 % |
| `k57`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 33 | 0.873 | 7.6 % |
| `k62`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 42 | 0.693 | 6.0 % |
| `k52`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 43 | 0.681 | 5.9 % |
| `k32`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 122 | 0.238 | 2.1 % |
| `k29`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 126 | 0.230 | 2.0 % |

### The gradient under its plan

The instrumented program: 1 023.0 ms a call (the median of its four); its kernels 882.7 ms; the rest, the host's, 140.3 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.255 | 455 | 28.9 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 0.301 | 579 | 34.2 % |
| every other kernel | 878 | 0.0 | 0.326 |  | 36.9 % |
| every kernel | 1025 | | **0.883** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k18`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 2…` | 24 | 538 | 0.054 | 6.1 % |
| `k65`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 557 | 0.052 | 5.9 % |
| `k31`: `dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq:…` | 6 | 557 | 0.052 | 5.9 % |
| `k66`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 561 | 0.052 | 5.9 % |
| `k59`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 562 | 0.052 | 5.8 % |
| `k60`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 573 | 0.051 | 5.7 % |
| `k53`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 591 | 0.049 | 5.6 % |
| `k29`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq:…` | 6 | 611 | 0.047 | 5.4 % |

## The machine and PyTorch's libraries

```
date 2026-10-01T19:13:23+02:00
presk 81bae21, presk 0.10.0-dev
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Vitesse maximale du processeur en MHz : 3900,0000
Nom de modèle : Cortex-A725
Vitesse maximale du processeur en MHz : 2808,0000
load 1.15 7.19 9.18
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
cflags -O2 -Wall -Wextra
OpenBLAS 0.3.34 DYNAMIC_ARCH NO_AFFINITY USE_OPENMP neoversev2 MAX_THREADS=256
arm_compute_version=v53.2.0
```

What PyTorch's forward pass hands to oneDNN, one call (`ONEDNN_VERBOSE=1`: count, primitive, implementation, shapes); the linear products are not among them, so they are the wheel's OpenBLAS:

```
     12 eltwise jit:sve 16x256x1536
     12 matmul gemm:acl 96x256x256:96x256x64
     12 matmul gemm:acl 96x256x64:96x64x256
     12 reorder jit:uni 96x256x64
     12 reorder jit:uni 96x64x256
```
