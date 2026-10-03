> The PyTorch baseline at 0.10.0 step 6 (D96: `make pytorch-baseline` on presk `ceade37`, the manual's edits set aside, begun 2026-10-01 at 16:02 right after `make host-profile`, the run of record), copied from `out/pytorch-baseline/report.md`: T4 judged, the native backend against the C program on the forward pass and on the step like for like, its `loss_grad` and `adamw.step` timed by `presk bench` (rows and a section this step adds), `presk train` beside; T2 again. In the table "Kernels and host", the native host is below the comparison's resolution: `presk bench --calibrate`'s timing of each launch costs more than what is left of it (D96).

# The course GPT on three sides: presk 0.10.0-dev against PyTorch

The course GPT (`corpus/nlp/05_gpt.psk`, D = 384, 6 heads, 6 blocks, V = 92) at B = 16, T = 256, `f32` on every side, with the checkpoint `out/course_probe_f32/final.psk.d` and the token indices the C export synthesizes; 2026-10-01, presk `ceade37`, `experiments/pytorch_baseline.sh`. Three sides: presk's native backend (`presk bench --plan-only`, `presk train`), the C program `presk export --target c` writes (GCC 13, `-O2`, OpenMP), and PyTorch 2.14.0+cpu running the model `presk export --target pytorch` writes. Each time is the median of seven calls or steps, the first call or step, which compiles natively, left out on the native and PyTorch sides; the C program's seven calls are all counted.

**Operations.** The forward pass is 96.93 GFLOP of contractions, a training step 290.78 (each contraction and its two derivatives); PyTorch's counter (`torch.utils.flop_counter`) gives 96.93 and 290.78. The rates below divide these by the median, whatever a side executes: presk's gradient recomputes the forward pass's products (below, where the C program's time goes).

## The forward pass

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 1 403.4 | 1 400.2 | 69.1 | 9.43 |
| 10 (ten X925, pinned) | native | plan | 329.9 | 329.7 | 293.8 | 2.22 |
| 10 (ten X925, pinned) | C | default | 864.9 | 861.4 | 112.1 | 5.81 |
| 10 (ten X925, pinned) | C | plan | 284.5 | 277.2 | 340.7 | 1.91 |
| 10 (ten X925, pinned) | PyTorch | its own | 148.8 | 144.1 | 651.6 | 1.00 |
| 20 (twenty, unpinned) | native | default | 1 985.5 | 1 977.3 | 48.8 | 8.09 |
| 20 (twenty, unpinned) | native | plan | 329.9 | 329.2 | 293.8 | 1.34 |
| 20 (twenty, unpinned) | C | default | 1 495.4 | 1 461.9 | 64.8 | 6.09 |
| 20 (twenty, unpinned) | C | plan | 281.5 | 277.8 | 344.3 | 1.15 |
| 20 (twenty, unpinned) | PyTorch | its own | 245.4 | 239.0 | 395.0 | 1.00 |

## One training step

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native, `presk train` | default | 13 957.9 | 20.8 | 13.81 |
| 10 (ten X925, pinned) | native, `presk train` | plan | 1 428.3 | 203.6 | 1.41 |
| 10 (ten X925, pinned) | native, `loss_grad` 13 804 + `adamw.step` 19 | default | 13 823.2 | 21.0 | 13.67 |
| 10 (ten X925, pinned) | native, `loss_grad` 1 320 + `adamw.step` 16 | plan | 1 335.5 | 217.7 | 1.32 |
| 10 (ten X925, pinned) | C, `loss_grad` 11 666 + `adamw.step` 16 | default | 11 682.5 | 24.9 | 11.56 |
| 10 (ten X925, pinned) | C, `loss_grad` 1 126 + `adamw.step` 12 | plan | 1 137.8 | 255.6 | 1.13 |
| 10 (ten X925, pinned) | PyTorch, `train.py` | its own | 1 011.0 | 287.6 | 1.00 |
| 20 (twenty, unpinned) | native, `presk train` | default | 14 067.4 | 20.7 | 10.15 |
| 20 (twenty, unpinned) | native, `presk train` | plan | 1 448.3 | 200.8 | 1.04 |
| 20 (twenty, unpinned) | native, `loss_grad` 13 742 + `adamw.step` 24 | default | 13 766.2 | 21.1 | 9.93 |
| 20 (twenty, unpinned) | native, `loss_grad` 1 317 + `adamw.step` 22 | plan | 1 339.1 | 217.1 | 0.97 |
| 20 (twenty, unpinned) | C, `loss_grad` 11 749 + `adamw.step` 21 | default | 11 769.5 | 24.7 | 8.49 |
| 20 (twenty, unpinned) | C, `loss_grad` 1 174 + `adamw.step` 17 | plan | 1 190.8 | 244.2 | 0.86 |
| 20 (twenty, unpinned) | PyTorch, `train.py` | its own | 1 386.6 | 209.7 | 1.00 |

The C step is its two programs' medians added, and so is the native step by `presk bench` (since 0.10.0 step 6, T4's step like for like); neither drops (the export drops keys, so its dropout is the expectation, and `presk bench` does not train), where `presk train` and `train.py` drop with their own masks. `train.py`'s step builds the module `GPT(256)` at every call of the exported `loss` (`torch.func.functional_call` on a fresh module, its initializers drawing every parameter): 22.0 ms on ten threads, inside its time.

Every layout and thread count gives the same `f32` logits and the same gradient, bit for bit, and the native training run on ten threads under the plan ends on the default layout's weights byte for byte: yes.

## T4: the native backend against the C program

Plan 0.10.0's statement T4 (§3.5, as amended after step 5): the native backend's forward pass and training step under the plans take at most 1.3× the C program's, same plans, same cores; the step compared like for like, the native `loss_grad` and `adamw.step` timed by `presk bench` against the C program's, neither drawing dropout masks, `presk train`'s step reported beside it. Judged on the ten X925 cores pinned, the row of record; every core unpinned reported.

| threads | workload | native ms | C ms | native ÷ C | T4 (≤ 1.3×) |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | the forward pass | 329.9 | 284.5 | 1.16 | holds |
| 10 (ten X925, pinned) | the step: `loss_grad` + `adamw.step` | 1 335.5 | 1 137.8 | 1.17 | holds |
| 10 (ten X925, pinned) | beside: `presk train`'s step, drawing its masks | 1 428.3 | 1 137.8 | 1.26 | — |
| 20 (twenty, unpinned) | the forward pass | 329.9 | 281.5 | 1.17 | (reported) |
| 20 (twenty, unpinned) | the step: `loss_grad` + `adamw.step` | 1 339.1 | 1 190.8 | 1.12 | (reported) |
| 20 (twenty, unpinned) | beside: `presk train`'s step, drawing its masks | 1 448.3 | 1 190.8 | 1.22 | — |

## The linear products alone (ten threads, pinned)

`Linear{In,Out}.apply` of the course GPT, `x · w + b` at x : [16, 256, In], under the layout the forward plan gives `Linear.apply` (one layout for its three shapes); PyTorch `x @ w + b`.

A call's time on each side, its product and its bias together; for the native backend also its two kernels' alone (`presk bench --calibrate`), the rest of its call being its host's buffers and copies. GFLOP/s: the product's operations over the call's time, or over the kernels'.

| product | GFLOP | native ms | its kernels ms | C ms | PyTorch ms | native GFLOP/s | its kernels | C | PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| [16, 256, 384] · [384, 384] | 1.21 | 3.28 | 3.74 | 2.19 | 1.25 | 368 | 323 | 552 | 964 |
| [16, 256, 384] · [384, 1536] | 4.83 | 13.32 | 15.10 | 9.48 | 6.28 | 363 | 320 | 510 | 770 |
| [16, 256, 1536] · [1536, 384] | 4.83 | 12.73 | 13.38 | 9.22 | 4.51 | 380 | 361 | 524 | 1071 |

## Kernels and host, the forward pass under its plan (ten threads, pinned)

The same kernels on both presk sides (C1), so what differs is who compiles them (Cranelift against GCC `-O2`) and what runs around them. The native kernels' time is `presk bench --calibrate`'s (every launch timed, its threads' start included); the C kernels' is the instrumented program's (below, three calls after a first; the program's median call is uninstrumented). The host is the call's median less its kernels. Under the plan only: `--calibrate` runs a nest no plan states scalar, where the native backend gives it its natural lanes, so under the default layout it times other kernels; the forward plan leaves only the host's small kernels (the causal mask, the positions) unstated.

| side | the call, ms | its kernels, ms | the rest, the host's, ms |
|---|---|---|---|
| native | 329.9 | 331.6 | -1.7 |
| C | 284.5 | 265.6 | 18.9 |

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

The instrumented program: 868.5 ms a call (the median of its four); its kernels 835.3 ms; the rest, the host's, 33.3 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.710 | 137 | 85.0 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 173 | 0.0 | 0.125 |  | 15.0 % |
| every kernel | 222 | | **0.835** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 118 | 0.246 | 29.5 % |
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 121 | 0.239 | 28.6 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 223 | 0.130 | 15.6 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 84 | 0.057 | 6.9 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 135 | 0.036 | 4.3 % |
| `k16`: `softmax axis seq2 (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, heads: 6, seq: …` | 6 |  | 0.031 | 3.8 % |
| `k23`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.019 | 2.3 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.019 | 2.2 % |

### The forward pass under its plan

The instrumented program: 285.5 ms a call (the median of its four); its kernels 265.6 ms; the rest, the host's, 20.0 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.171 | 565 | 64.6 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 179 | 0.0 | 0.094 |  | 35.4 % |
| every kernel | 228 | | **0.266** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 545 | 0.053 | 20.0 % |
| `k25`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 590 | 0.049 | 18.5 % |
| `k22`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 614 | 0.047 | 17.8 % |
| `k17`: `softmax axis seq2, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, hea…` | 6 |  | 0.020 | 7.5 % |
| `k24`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.015 | 5.8 % |
| `k19`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 437 | 0.011 | 4.2 % |
| `k14`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 468 | 0.010 | 3.9 % |
| `k11`: `add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 30 |  | 0.010 | 3.6 % |

### The gradient, `loss_grad`, default layout

The instrumented program: 11 653.1 ms a call (the median of its four); its kernels 11 511.3 ms; the rest, the host's, 141.8 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.856 | 136 | 7.4 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 10.338 | 17 | 89.8 % |
| every other kernel | 865 | 0.0 | 0.317 |  | 2.8 % |
| every kernel | 1012 | | **11.511** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k58`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 9 | 3.248 | 28.2 % |
| `k53`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 9 | 3.105 | 27.0 % |
| `k63`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 17 | 1.718 | 14.9 % |
| `k57`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 33 | 0.873 | 7.6 % |
| `k62`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 42 | 0.692 | 6.0 % |
| `k52`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 43 | 0.682 | 5.9 % |
| `k32`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 126 | 0.230 | 2.0 % |
| `k29`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 132 | 0.219 | 1.9 % |

### The gradient under its plan

The instrumented program: 1 118.0 ms a call (the median of its four); its kernels 988.7 ms; the rest, the host's, 129.2 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.303 | 384 | 30.6 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 0.360 | 485 | 36.4 % |
| every other kernel | 914 | 0.0 | 0.326 |  | 32.9 % |
| every kernel | 1061 | | **0.989** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k67`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 402 | 0.072 | 7.3 % |
| `k18`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 419 | 0.069 | 7.0 % |
| `k55`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 430 | 0.067 | 6.8 % |
| `k61`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 430 | 0.067 | 6.8 % |
| `k33`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 434 | 0.067 | 6.8 % |
| `k30`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 440 | 0.066 | 6.7 % |
| `k68`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 548 | 0.053 | 5.4 % |
| `k62`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 568 | 0.051 | 5.2 % |

## The machine and PyTorch's libraries

```
date 2026-10-01T16:02:21+02:00
presk ceade37, presk 0.10.0-dev
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Vitesse maximale du processeur en MHz : 3900,0000
Nom de modèle : Cortex-A725
Vitesse maximale du processeur en MHz : 2808,0000
load 8.40 7.89 8.25
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
