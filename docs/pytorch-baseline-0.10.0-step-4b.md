> The PyTorch baseline at 0.10.0 step 4b (D94: `make pytorch-baseline` on presk `5fca917`, the step's code commit before its branch was rebased onto the re-measurement's, D93, the manual's edits set aside, begun 2026-09-30 at 22:11, the run of record), copied from `out/pytorch-baseline/report.md`: T2 again, the gradient keeping its fusion groups only where a realization can see them, so not in `f32`. A first run on the same tree, begun at 21:50, twenty seconds after the last timed run of D93, gave the C step 1 148.5 ms (1.13×), `presk train` under the plan 8 037.2 ms, and the C forward pass under its plan 303.1 ms, which this step cannot move. Step 4's reading, to compare with, is `docs/pytorch-baseline-0.10.0-step-4.md` (D92).

# The course GPT on three sides: presk 0.10.0-dev against PyTorch

The course GPT (`corpus/nlp/05_gpt.psk`, D = 384, 6 heads, 6 blocks, V = 92) at B = 16, T = 256, `f32` on every side, with the checkpoint `out/course_probe_f32/final.psk.d` and the token indices the C export synthesizes; 2026-09-30, presk `5fca917`, `experiments/pytorch_baseline.sh`. Three sides: presk's native backend (`presk bench --plan-only`, `presk train`), the C program `presk export --target c` writes (GCC 13, `-O2`, OpenMP), and PyTorch 2.14.0+cpu running the model `presk export --target pytorch` writes. Each time is the median of seven calls or steps, the first call or step, which compiles natively, left out on the native and PyTorch sides; the C program's seven calls are all counted.

**Operations.** The forward pass is 96.93 GFLOP of contractions, a training step 290.78 (each contraction and its two derivatives); PyTorch's counter (`torch.utils.flop_counter`) gives 96.93 and 290.78. The rates below divide these by the median, whatever a side executes: presk's gradient recomputes the forward pass's products (below, where the C program's time goes).

## The forward pass

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 1 710.3 | 1 675.2 | 56.7 | 10.85 |
| 10 (ten X925, pinned) | native | plan | 650.3 | 642.5 | 149.1 | 4.12 |
| 10 (ten X925, pinned) | C | default | 805.0 | 802.7 | 120.4 | 5.11 |
| 10 (ten X925, pinned) | C | plan | 286.6 | 282.4 | 338.2 | 1.82 |
| 10 (ten X925, pinned) | PyTorch | its own | 157.7 | 148.7 | 614.7 | 1.00 |
| 20 (twenty, unpinned) | native | default | 1 962.9 | 1 872.4 | 49.4 | 7.72 |
| 20 (twenty, unpinned) | native | plan | 1 141.9 | 1 121.5 | 84.9 | 4.49 |
| 20 (twenty, unpinned) | C | default | 1 476.1 | 1 345.6 | 65.7 | 5.81 |
| 20 (twenty, unpinned) | C | plan | 282.2 | 276.7 | 343.5 | 1.11 |
| 20 (twenty, unpinned) | PyTorch | its own | 254.1 | 240.3 | 381.4 | 1.00 |

## One training step

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native, `presk train` | default | 20 537.6 | 14.2 | 19.87 |
| 10 (ten X925, pinned) | native, `presk train` | plan | 8 100.7 | 35.9 | 7.84 |
| 10 (ten X925, pinned) | C, `loss_grad` 11 677 + `adamw.step` 18 | default | 11 694.6 | 24.9 | 11.32 |
| 10 (ten X925, pinned) | C, `loss_grad` 1 122 + `adamw.step` 12 | plan | 1 134.0 | 256.4 | 1.10 |
| 10 (ten X925, pinned) | PyTorch, `train.py` | its own | 1 033.4 | 281.4 | 1.00 |
| 20 (twenty, unpinned) | native, `presk train` | default | 20 353.5 | 14.3 | 14.59 |
| 20 (twenty, unpinned) | native, `presk train` | plan | 9 327.8 | 31.2 | 6.69 |
| 20 (twenty, unpinned) | C, `loss_grad` 11 651 + `adamw.step` 20 | default | 11 671.1 | 24.9 | 8.37 |
| 20 (twenty, unpinned) | C, `loss_grad` 1 165 + `adamw.step` 18 | plan | 1 183.1 | 245.8 | 0.85 |
| 20 (twenty, unpinned) | PyTorch, `train.py` | its own | 1 395.2 | 208.4 | 1.00 |

The C step is its two programs' medians added; neither drops (the export drops keys, so its dropout is the expectation), where `presk train` and `train.py` drop with their own masks. `train.py`'s step builds the module `GPT(256)` at every call of the exported `loss` (`torch.func.functional_call` on a fresh module, its initializers drawing every parameter): 21.0 ms on ten threads, inside its time.

Every layout and thread count gives the same `f32` logits and the same gradient, bit for bit, and the native training run on ten threads under the plan ends on the default layout's weights byte for byte: yes.

## The linear products alone (ten threads, pinned)

`Linear{In,Out}.apply` of the course GPT, `x · w + b` at x : [16, 256, In], under the layout the forward plan gives `Linear.apply` (one layout for its three shapes); PyTorch `x @ w + b`.

A call's time on each side, its product and its bias together; for the native backend also its two kernels' alone (`presk bench --calibrate`), the rest of its call being its host's buffers and copies. GFLOP/s: the product's operations over the call's time, or over the kernels'.

| product | GFLOP | native ms | its kernels ms | C ms | PyTorch ms | native GFLOP/s | its kernels | C | PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| [16, 256, 384] · [384, 384] | 1.21 | 18.29 | 4.91 | 2.31 | 1.25 | 66 | 246 | 523 | 966 |
| [16, 256, 384] · [384, 1536] | 4.83 | 49.17 | 17.47 | 10.05 | 6.15 | 98 | 277 | 481 | 786 |
| [16, 256, 1536] · [1536, 384] | 4.83 | 74.83 | 13.93 | 8.89 | 4.59 | 65 | 347 | 543 | 1052 |

## Kernels and host, the forward pass under its plan (ten threads, pinned)

The same kernels on both presk sides (C1), so what differs is who compiles them (Cranelift against GCC `-O2`) and what runs around them. The native kernels' time is `presk bench --calibrate`'s (every launch timed, its threads' start included); the C kernels' is the instrumented program's (below, three calls after a first; the program's median call is uninstrumented). The host is the call's median less its kernels. Under the plan only: `--calibrate` runs a nest no plan states scalar, where the native backend gives it its natural lanes, so under the default layout it times other kernels; the forward plan leaves only the host's small kernels (the causal mask, the positions) unstated.

| side | the call, ms | its kernels, ms | the rest, the host's, ms |
|---|---|---|---|
| native | 650.3 | 361.4 | 288.9 |
| C | 286.6 | 268.4 | 18.2 |

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

The instrumented program: 872.9 ms a call (the median of its four); its kernels 856.3 ms; the rest, the host's, 16.6 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.728 | 133 | 85.0 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 173 | 0.0 | 0.129 |  | 15.0 % |
| every kernel | 222 | | **0.856** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 114 | 0.254 | 29.6 % |
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 117 | 0.248 | 29.0 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 220 | 0.132 | 15.4 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 84 | 0.057 | 6.7 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 138 | 0.035 | 4.1 % |
| `k16`: `softmax axis seq2 (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, heads: 6, seq: …` | 6 |  | 0.031 | 3.6 % |
| `k23`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.020 | 2.3 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.019 | 2.2 % |

### The forward pass under its plan

The instrumented program: 285.8 ms a call (the median of its four); its kernels 268.4 ms; the rest, the host's, 17.4 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.173 | 560 | 64.5 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 179 | 0.0 | 0.095 |  | 35.5 % |
| every kernel | 228 | | **0.268** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 543 | 0.053 | 19.9 % |
| `k25`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 585 | 0.050 | 18.5 % |
| `k22`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 608 | 0.048 | 17.8 % |
| `k17`: `softmax axis seq2, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, hea…` | 6 |  | 0.019 | 7.3 % |
| `k24`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.016 | 5.9 % |
| `k14`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 439 | 0.011 | 4.1 % |
| `k19`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 457 | 0.011 | 3.9 % |
| `k11`: `add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 30 |  | 0.009 | 3.4 % |

### The gradient, `loss_grad`, default layout

The instrumented program: 11 755.3 ms a call (the median of its four); its kernels 11 564.7 ms; the rest, the host's, 190.6 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.882 | 132 | 7.6 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 10.359 | 17 | 89.6 % |
| every other kernel | 865 | 0.0 | 0.324 |  | 2.8 % |
| every kernel | 1012 | | **11.565** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k58`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 9 | 3.253 | 28.1 % |
| `k53`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 9 | 3.107 | 26.9 % |
| `k63`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 17 | 1.732 | 15.0 % |
| `k57`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 33 | 0.873 | 7.6 % |
| `k62`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 42 | 0.694 | 6.0 % |
| `k52`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 43 | 0.682 | 5.9 % |
| `k32`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 120 | 0.242 | 2.1 % |
| `k29`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 126 | 0.231 | 2.0 % |

### The gradient under its plan

The instrumented program: 1 135.2 ms a call (the median of its four); its kernels 991.3 ms; the rest, the host's, 143.9 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.301 | 386 | 30.4 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 0.357 | 488 | 36.0 % |
| every other kernel | 914 | 0.0 | 0.333 |  | 33.6 % |
| every kernel | 1061 | | **0.991** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k67`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 403 | 0.072 | 7.3 % |
| `k18`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 421 | 0.069 | 7.0 % |
| `k55`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 431 | 0.067 | 6.8 % |
| `k61`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 435 | 0.067 | 6.7 % |
| `k33`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 438 | 0.066 | 6.7 % |
| `k30`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 443 | 0.065 | 6.6 % |
| `k68`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 555 | 0.052 | 5.3 % |
| `k62`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 569 | 0.051 | 5.1 % |

## The machine and PyTorch's libraries

```
date 2026-09-30T22:11:17+02:00
presk 5fca917, presk 0.10.0-dev
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Vitesse maximale du processeur en MHz : 3900,0000
Nom de modèle : Cortex-A725
Vitesse maximale du processeur en MHz : 2808,0000
load 4.01 6.22 6.77
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
