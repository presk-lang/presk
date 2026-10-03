> The PyTorch baseline at 0.10.0 step 1 (D89: `make pytorch-baseline` on presk `09aa46f`, begun 2026-09-28 at 23:07, one run), copied from `out/pytorch-baseline/report.md`: T2's first reading, the derivatives of `dot` in the layout space. The version's starting point, to compare with, is `docs/pytorch-baseline-2026-09-28.md` (D88).

# The course GPT on three sides: presk 0.10.0-dev against PyTorch

The course GPT (`corpus/nlp/05_gpt.psk`, D = 384, 6 heads, 6 blocks, V = 92) at B = 16, T = 256, `f32` on every side, with the checkpoint `out/course_probe_f32/final.psk.d` and the token indices the C export synthesizes; 2026-09-28, presk `09aa46f`, `experiments/pytorch_baseline.sh`. Three sides: presk's native backend (`presk bench --plan-only`, `presk train`), the C program `presk export --target c` writes (GCC 13, `-O2`, OpenMP), and PyTorch 2.14.0+cpu running the model `presk export --target pytorch` writes. Each time is the median of seven calls or steps, the first call or step, which compiles natively, left out on the native and PyTorch sides; the C program's seven calls are all counted.

**Operations.** The forward pass is 96.93 GFLOP of contractions, a training step 290.78 (each contraction and its two derivatives); PyTorch's counter (`torch.utils.flop_counter`) gives 96.93 and 290.78. The rates below divide these by the median, whatever a side executes: presk's gradient recomputes the forward pass's products (below, where the C program's time goes).

## The forward pass

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 1 743.8 | 1 718.0 | 55.6 | 11.02 |
| 10 (ten X925, pinned) | native | plan | 683.0 | 668.1 | 141.9 | 4.32 |
| 10 (ten X925, pinned) | C | default | 865.1 | 829.0 | 112.0 | 5.47 |
| 10 (ten X925, pinned) | C | plan | 287.9 | 274.6 | 336.6 | 1.82 |
| 10 (ten X925, pinned) | PyTorch | its own | 158.2 | 150.6 | 612.5 | 1.00 |
| 20 (twenty, unpinned) | native | default | 2 046.3 | 2 026.2 | 47.4 | 8.61 |
| 20 (twenty, unpinned) | native | plan | 1 098.4 | 1 084.8 | 88.2 | 4.62 |
| 20 (twenty, unpinned) | C | default | 1 491.7 | 1 396.0 | 65.0 | 6.27 |
| 20 (twenty, unpinned) | C | plan | 284.6 | 277.2 | 340.6 | 1.20 |
| 20 (twenty, unpinned) | PyTorch | its own | 237.8 | 234.9 | 407.6 | 1.00 |

## One training step

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native, `presk train` | default | 30 486.1 | 9.5 | 29.49 |
| 10 (ten X925, pinned) | native, `presk train` | plan | 12 824.1 | 22.7 | 12.40 |
| 10 (ten X925, pinned) | C, `loss_grad` 15 664 + `adamw.step` 18 | default | 15 681.5 | 18.5 | 15.17 |
| 10 (ten X925, pinned) | C, `loss_grad` 2 841 + `adamw.step` 12 | plan | 2 852.9 | 101.9 | 2.76 |
| 10 (ten X925, pinned) | PyTorch, `train.py` | its own | 1 033.8 | 281.3 | 1.00 |
| 20 (twenty, unpinned) | native, `presk train` | default | 30 628.9 | 9.5 | 22.76 |
| 20 (twenty, unpinned) | native, `presk train` | plan | 15 973.5 | 18.2 | 11.87 |
| 20 (twenty, unpinned) | C, `loss_grad` 19 219 + `adamw.step` 19 | default | 19 238.6 | 15.1 | 14.29 |
| 20 (twenty, unpinned) | C, `loss_grad` 2 891 + `adamw.step` 17 | plan | 2 908.8 | 100.0 | 2.16 |
| 20 (twenty, unpinned) | PyTorch, `train.py` | its own | 1 345.9 | 216.0 | 1.00 |

The C step is its two programs' medians added; neither drops (the export drops keys, so its dropout is the expectation), where `presk train` and `train.py` drop with their own masks. `train.py`'s step builds the module `GPT(256)` at every call of the exported `loss` (`torch.func.functional_call` on a fresh module, its initializers drawing every parameter): 22.1 ms on ten threads, inside its time.

Every layout and thread count gives the same `f32` logits and the same gradient, bit for bit, and the native training run on ten threads under the plan ends on the default layout's weights byte for byte: yes.

## The linear products alone (ten threads, pinned)

`Linear{In,Out}.apply` of the course GPT, `x · w + b` at x : [16, 256, In], under the layout the forward plan gives `Linear.apply` (one layout for its three shapes); PyTorch `x @ w + b`.

A call's time on each side, its product and its bias together; for the native backend also its two kernels' alone (`presk bench --calibrate`), the rest of its call being its host's buffers and copies. GFLOP/s: the product's operations over the call's time, or over the kernels'.

| product | GFLOP | native ms | its kernels ms | C ms | PyTorch ms | native GFLOP/s | its kernels | C | PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| [16, 256, 384] · [384, 384] | 1.21 | 21.31 | 5.05 | 2.23 | 1.25 | 57 | 239 | 542 | 966 |
| [16, 256, 384] · [384, 1536] | 4.83 | 54.14 | 17.44 | 8.44 | 6.08 | 89 | 277 | 572 | 794 |
| [16, 256, 1536] · [1536, 384] | 4.83 | 74.09 | 13.97 | 8.71 | 4.68 | 65 | 346 | 555 | 1032 |

## Kernels and host, the forward pass under its plan (ten threads, pinned)

The same kernels on both presk sides (C1), so what differs is who compiles them (Cranelift against GCC `-O2`) and what runs around them. The native kernels' time is `presk bench --calibrate`'s (every launch timed, its threads' start included); the C kernels' is the instrumented program's (below, three calls after a first; the program's median call is uninstrumented). The host is the call's median less its kernels. Under the plan only: `--calibrate` runs a nest no plan states scalar, where the native backend gives it its natural lanes, so under the default layout it times other kernels; the forward plan leaves only the host's small kernels (the causal mask, the positions) unstated.

| side | the call, ms | its kernels, ms | the rest, the host's, ms |
|---|---|---|---|
| native | 683.0 | 362.3 | 320.7 |
| C | 287.9 | 280.2 | 7.8 |

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

The instrumented program: 882.8 ms a call (the median of its four); its kernels 848.8 ms; the rest, the host's, 34.0 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.719 | 135 | 84.7 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 173 | 0.0 | 0.130 |  | 15.3 % |
| every kernel | 222 | | **0.849** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 117 | 0.249 | 29.3 % |
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 117 | 0.247 | 29.1 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 223 | 0.130 | 15.3 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 85 | 0.057 | 6.7 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 139 | 0.035 | 4.1 % |
| `k16`: `softmax axis seq2 (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, heads: 6, seq: …` | 6 |  | 0.031 | 3.7 % |
| `k23`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.020 | 2.3 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.019 | 2.2 % |

### The forward pass under its plan

The instrumented program: 299.0 ms a call (the median of its four); its kernels 280.2 ms; the rest, the host's, 18.8 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.174 | 556 | 62.2 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 179 | 0.0 | 0.106 |  | 37.8 % |
| every kernel | 228 | | **0.280** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 532 | 0.055 | 19.5 % |
| `k25`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 590 | 0.049 | 17.5 % |
| `k22`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 603 | 0.048 | 17.2 % |
| `k17`: `softmax axis seq2, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, hea…` | 6 |  | 0.020 | 7.1 % |
| `k24`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.015 | 5.5 % |
| `k11`: `add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 30 |  | 0.013 | 4.8 % |
| `k19`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 422 | 0.011 | 4.1 % |
| `k14`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 470 | 0.010 | 3.7 % |

### The gradient, `loss_grad`, default layout

The instrumented program: 16 004.0 ms a call (the median of its four); its kernels 15 742.4 ms; the rest, the host's, 261.6 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 303 | 590.4 | 4.380 | 135 | 27.8 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 10.340 | 17 | 65.7 % |
| every other kernel | 1843 | 0.0 | 1.023 |  | 6.5 % |
| every kernel | 2220 | | **15.742** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k60`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 9 | 3.246 | 20.6 % |
| `k54`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 9 | 3.101 | 19.7 % |
| `k66`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 17 | 1.726 | 11.0 % |
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 36 | 118 | 1.475 | 9.4 % |
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 36 | 119 | 1.467 | 9.3 % |
| `k59`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 33 | 0.875 | 5.6 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 144 | 221 | 0.787 | 5.0 % |
| `k65`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 42 | 0.691 | 4.4 % |

### The gradient under its plan

The instrumented program: 2 858.2 ms a call (the median of its four); its kernels 2 717.1 ms; the rest, the host's, 141.1 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 303 | 590.4 | 1.438 | 411 | 52.9 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 0.360 | 485 | 13.3 % |
| every other kernel | 1916 | 0.0 | 0.919 |  | 33.8 % |
| every kernel | 2293 | | **2.717** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 144 | 411 | 0.423 | 15.6 % |
| `k25`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 36 | 436 | 0.399 | 14.7 % |
| `k22`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 36 | 438 | 0.397 | 14.6 % |
| `k17`: `softmax axis seq2, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, hea…` | 30 |  | 0.095 | 3.5 % |
| `k19`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 36 | 330 | 0.088 | 3.2 % |
| `k14`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 36 | 375 | 0.077 | 2.8 % |
| `k24`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 30 |  | 0.077 | 2.8 % |
| `k70`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 396 | 0.073 | 2.7 % |

## The machine and PyTorch's libraries

```
date 2026-09-28T23:07:52+02:00
presk 09aa46f, presk 0.10.0-dev
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Vitesse maximale du processeur en MHz : 3900,0000
Nom de modèle : Cortex-A725
Vitesse maximale du processeur en MHz : 2808,0000
load 3.41 13.90 9.57
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
