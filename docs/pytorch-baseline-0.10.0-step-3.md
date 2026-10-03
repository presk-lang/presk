> The PyTorch baseline at 0.10.0 step 3 (D91: `make pytorch-baseline` on presk `dc386d8`, begun 2026-09-30 at 10:23, one run), copied from `out/pytorch-baseline/report.md`: T2's second reading, the dead code after differentiation removed. Step 1's reading, to compare with, is `docs/pytorch-baseline-0.10.0-step-1.md` (D89).

# The course GPT on three sides: presk 0.10.0-dev against PyTorch

The course GPT (`corpus/nlp/05_gpt.psk`, D = 384, 6 heads, 6 blocks, V = 92) at B = 16, T = 256, `f32` on every side, with the checkpoint `out/course_probe_f32/final.psk.d` and the token indices the C export synthesizes; 2026-09-30, presk `dc386d8`, `experiments/pytorch_baseline.sh`. Three sides: presk's native backend (`presk bench --plan-only`, `presk train`), the C program `presk export --target c` writes (GCC 13, `-O2`, OpenMP), and PyTorch 2.14.0+cpu running the model `presk export --target pytorch` writes. Each time is the median of seven calls or steps, the first call or step, which compiles natively, left out on the native and PyTorch sides; the C program's seven calls are all counted.

**Operations.** The forward pass is 96.93 GFLOP of contractions, a training step 290.78 (each contraction and its two derivatives); PyTorch's counter (`torch.utils.flop_counter`) gives 96.93 and 290.78. The rates below divide these by the median, whatever a side executes: presk's gradient recomputes the forward pass's products (below, where the C program's time goes).

## The forward pass

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 1 769.4 | 1 722.0 | 54.8 | 11.57 |
| 10 (ten X925, pinned) | native | plan | 700.0 | 684.8 | 138.5 | 4.58 |
| 10 (ten X925, pinned) | C | default | 853.3 | 827.7 | 113.6 | 5.58 |
| 10 (ten X925, pinned) | C | plan | 285.5 | 278.6 | 339.5 | 1.87 |
| 10 (ten X925, pinned) | PyTorch | its own | 152.9 | 143.1 | 633.8 | 1.00 |
| 20 (twenty, unpinned) | native | default | 1 929.4 | 1 919.3 | 50.2 | 7.58 |
| 20 (twenty, unpinned) | native | plan | 1 158.2 | 1 149.2 | 83.7 | 4.55 |
| 20 (twenty, unpinned) | C | default | 1 510.5 | 1 426.9 | 64.2 | 5.93 |
| 20 (twenty, unpinned) | C | plan | 292.9 | 280.9 | 330.9 | 1.15 |
| 20 (twenty, unpinned) | PyTorch | its own | 254.5 | 227.8 | 380.8 | 1.00 |

## One training step

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native, `presk train` | default | 24 304.0 | 12.0 | 24.08 |
| 10 (ten X925, pinned) | native, `presk train` | plan | 9 902.1 | 29.4 | 9.81 |
| 10 (ten X925, pinned) | C, `loss_grad` 13 372 + `adamw.step` 17 | default | 13 389.4 | 21.7 | 13.27 |
| 10 (ten X925, pinned) | C, `loss_grad` 1 863 + `adamw.step` 15 | plan | 1 877.8 | 154.8 | 1.86 |
| 10 (ten X925, pinned) | PyTorch, `train.py` | its own | 1 009.2 | 288.1 | 1.00 |
| 20 (twenty, unpinned) | native, `presk train` | default | 23 215.2 | 12.5 | 16.88 |
| 20 (twenty, unpinned) | native, `presk train` | plan | 10 585.7 | 27.5 | 7.70 |
| 20 (twenty, unpinned) | C, `loss_grad` 14 684 + `adamw.step` 20 | default | 14 704.6 | 19.8 | 10.69 |
| 20 (twenty, unpinned) | C, `loss_grad` 1 921 + `adamw.step` 17 | plan | 1 938.1 | 150.0 | 1.41 |
| 20 (twenty, unpinned) | PyTorch, `train.py` | its own | 1 375.0 | 211.5 | 1.00 |

The C step is its two programs' medians added; neither drops (the export drops keys, so its dropout is the expectation), where `presk train` and `train.py` drop with their own masks. `train.py`'s step builds the module `GPT(256)` at every call of the exported `loss` (`torch.func.functional_call` on a fresh module, its initializers drawing every parameter): 21.7 ms on ten threads, inside its time.

Every layout and thread count gives the same `f32` logits and the same gradient, bit for bit, and the native training run on ten threads under the plan ends on the default layout's weights byte for byte: yes.

## The linear products alone (ten threads, pinned)

`Linear{In,Out}.apply` of the course GPT, `x · w + b` at x : [16, 256, In], under the layout the forward plan gives `Linear.apply` (one layout for its three shapes); PyTorch `x @ w + b`.

A call's time on each side, its product and its bias together; for the native backend also its two kernels' alone (`presk bench --calibrate`), the rest of its call being its host's buffers and copies. GFLOP/s: the product's operations over the call's time, or over the kernels'.

| product | GFLOP | native ms | its kernels ms | C ms | PyTorch ms | native GFLOP/s | its kernels | C | PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| [16, 256, 384] · [384, 384] | 1.21 | 20.30 | 5.09 | 2.31 | 1.25 | 60 | 237 | 523 | 966 |
| [16, 256, 384] · [384, 1536] | 4.83 | 48.61 | 17.43 | 9.19 | 6.08 | 99 | 277 | 526 | 794 |
| [16, 256, 1536] · [1536, 384] | 4.83 | 80.09 | 13.98 | 9.25 | 4.51 | 60 | 346 | 522 | 1071 |

## Kernels and host, the forward pass under its plan (ten threads, pinned)

The same kernels on both presk sides (C1), so what differs is who compiles them (Cranelift against GCC `-O2`) and what runs around them. The native kernels' time is `presk bench --calibrate`'s (every launch timed, its threads' start included); the C kernels' is the instrumented program's (below, three calls after a first; the program's median call is uninstrumented). The host is the call's median less its kernels. Under the plan only: `--calibrate` runs a nest no plan states scalar, where the native backend gives it its natural lanes, so under the default layout it times other kernels; the forward plan leaves only the host's small kernels (the causal mask, the positions) unstated.

| side | the call, ms | its kernels, ms | the rest, the host's, ms |
|---|---|---|---|
| native | 700.0 | 362.9 | 337.1 |
| C | 285.5 | 276.5 | 9.0 |

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

The instrumented program: 878.2 ms a call (the median of its four); its kernels 858.4 ms; the rest, the host's, 19.8 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.731 | 133 | 85.2 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 173 | 0.0 | 0.127 |  | 14.8 % |
| every kernel | 222 | | **0.858** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 111 | 0.260 | 30.3 % |
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 118 | 0.246 | 28.7 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 223 | 0.130 | 15.2 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 84 | 0.058 | 6.7 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 137 | 0.035 | 4.1 % |
| `k16`: `softmax axis seq2 (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, heads: 6, seq: …` | 6 |  | 0.031 | 3.6 % |
| `k23`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.019 | 2.2 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.019 | 2.2 % |

### The forward pass under its plan

The instrumented program: 294.8 ms a call (the median of its four); its kernels 276.5 ms; the rest, the host's, 18.3 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.173 | 560 | 62.6 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 179 | 0.0 | 0.103 |  | 37.4 % |
| every kernel | 228 | | **0.276** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 545 | 0.053 | 19.3 % |
| `k25`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 580 | 0.050 | 18.1 % |
| `k22`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 611 | 0.047 | 17.2 % |
| `k17`: `softmax axis seq2, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, hea…` | 6 |  | 0.022 | 7.9 % |
| `k24`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.016 | 5.9 % |
| `k19`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 441 | 0.011 | 4.0 % |
| `k14`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 459 | 0.011 | 3.8 % |
| `k11`: `add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 30 |  | 0.010 | 3.7 % |

### The gradient, `loss_grad`, default layout

The instrumented program: 13 618.2 ms a call (the median of its four); its kernels 13 407.9 ms; the rest, the host's, 210.3 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 193 | 311.9 | 2.408 | 130 | 18.0 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 10.313 | 17 | 76.9 % |
| every other kernel | 1349 | 0.0 | 0.686 |  | 5.1 % |
| every kernel | 1616 | | **13.408** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k54`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 9 | 3.231 | 24.1 % |
| `k49`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 9 | 3.105 | 23.2 % |
| `k60`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 17 | 1.713 | 12.8 % |
| `k53`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 33 | 0.873 | 6.5 % |
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 18 | 109 | 0.798 | 6.0 % |
| `k59`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 42 | 0.690 | 5.1 % |
| `k48`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 43 | 0.681 | 5.1 % |
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 12 | 103 | 0.561 | 4.2 % |

### The gradient under its plan

The instrumented program: 1 895.6 ms a call (the median of its four); its kernels 1 763.3 ms; the rest, the host's, 132.3 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 193 | 311.9 | 0.780 | 400 | 44.2 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 0.361 | 483 | 20.5 % |
| every other kernel | 1416 | 0.0 | 0.622 |  | 35.3 % |
| every kernel | 1683 | | **1.763** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 90 | 413 | 0.263 | 14.9 % |
| `k22`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 18 | 437 | 0.199 | 11.3 % |
| `k25`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 12 | 435 | 0.133 | 7.6 % |
| `k17`: `softmax axis seq2, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, hea…` | 24 |  | 0.076 | 4.3 % |
| `k64`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 396 | 0.073 | 4.2 % |
| `k19`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 30 | 340 | 0.071 | 4.0 % |
| `k51`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 427 | 0.068 | 3.8 % |
| `k57`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 432 | 0.067 | 3.8 % |

## The machine and PyTorch's libraries

```
date 2026-09-30T10:23:26+02:00
presk dc386d8, presk 0.10.0-dev
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Vitesse maximale du processeur en MHz : 3900,0000
Nom de modèle : Cortex-A725
Vitesse maximale du processeur en MHz : 2808,0000
load 0.47 3.04 5.90
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
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
