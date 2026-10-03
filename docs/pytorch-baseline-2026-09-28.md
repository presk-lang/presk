> The second of the two runs of record of the PyTorch baseline at 0.9.0 (D87: `make pytorch-baseline` on presk `2ea860d`, begun 2026-09-28 at 20:41), copied from `out/pytorch-baseline/report.md`; the first run's medians are in D87, beside this one's. It is 0.10.0's starting point: plan 0.10.0 re-runs `make pytorch-baseline` at every step that could move T2 or T4 and compares with it (D88).

# The course GPT on three sides: presk 0.9.0 against PyTorch

The course GPT (`corpus/nlp/05_gpt.psk`, D = 384, 6 heads, 6 blocks, V = 92) at B = 16, T = 256, `f32` on every side, with the checkpoint `out/course_probe_f32/final.psk.d` and the token indices the C export synthesizes; 2026-09-28, presk `2ea860d`, `experiments/pytorch_baseline.sh`. Three sides: presk's native backend (`presk bench --plan-only`, `presk train`), the C program `presk export --target c` writes (GCC 13, `-O2`, OpenMP), and PyTorch 2.14.0+cpu running the model `presk export --target pytorch` writes. Each time is the median of seven calls or steps, the first call or step, which compiles natively, left out on the native and PyTorch sides; the C program's seven calls are all counted.

**Operations.** The forward pass is 96.93 GFLOP of contractions, a training step 290.78 (each contraction and its two derivatives); PyTorch's counter (`torch.utils.flop_counter`) gives 96.93 and 290.78. The rates below divide these by the median, whatever a side executes: presk's gradient recomputes the forward pass's products (below, where the C program's time goes).

## The forward pass

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 1 807.0 | 1 757.2 | 53.6 | 11.81 |
| 10 (ten X925, pinned) | native | plan | 710.0 | 663.8 | 136.5 | 4.64 |
| 10 (ten X925, pinned) | C | default | 879.8 | 864.2 | 110.2 | 5.75 |
| 10 (ten X925, pinned) | C | plan | 284.9 | 280.8 | 340.2 | 1.86 |
| 10 (ten X925, pinned) | PyTorch | its own | 153.0 | 142.2 | 633.6 | 1.00 |
| 20 (twenty, unpinned) | native | default | 2 099.1 | 2 011.3 | 46.2 | 8.33 |
| 20 (twenty, unpinned) | native | plan | 1 194.7 | 1 161.0 | 81.1 | 4.74 |
| 20 (twenty, unpinned) | C | default | 1 497.4 | 1 402.2 | 64.7 | 5.94 |
| 20 (twenty, unpinned) | C | plan | 285.1 | 277.4 | 340.0 | 1.13 |
| 20 (twenty, unpinned) | PyTorch | its own | 252.0 | 235.3 | 384.7 | 1.00 |

## One training step

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native, `presk train` | default | 29 895.3 | 9.7 | 29.51 |
| 10 (ten X925, pinned) | native, `presk train` | plan | 22 291.8 | 13.0 | 22.00 |
| 10 (ten X925, pinned) | C, `loss_grad` 16 375 + `adamw.step` 17 | default | 16 391.5 | 17.7 | 16.18 |
| 10 (ten X925, pinned) | C, `loss_grad` 12 232 + `adamw.step` 15 | plan | 12 247.2 | 23.7 | 12.09 |
| 10 (ten X925, pinned) | PyTorch, `train.py` | its own | 1 013.1 | 287.0 | 1.00 |
| 20 (twenty, unpinned) | native, `presk train` | default | 31 060.1 | 9.4 | 22.77 |
| 20 (twenty, unpinned) | native, `presk train` | plan | 27 387.7 | 10.6 | 20.08 |
| 20 (twenty, unpinned) | C, `loss_grad` 19 298 + `adamw.step` 20 | default | 19 317.2 | 15.1 | 14.16 |
| 20 (twenty, unpinned) | C, `loss_grad` 12 296 + `adamw.step` 18 | plan | 12 314.2 | 23.6 | 9.03 |
| 20 (twenty, unpinned) | PyTorch, `train.py` | its own | 1 363.9 | 213.2 | 1.00 |

The C step is its two programs' medians added; neither drops (the export drops keys, so its dropout is the expectation), where `presk train` and `train.py` drop with their own masks. `train.py`'s step builds the module `GPT(256)` at every call of the exported `loss` (`torch.func.functional_call` on a fresh module, its initializers drawing every parameter): 21.5 ms on ten threads, inside its time.

Every layout and thread count gives the same `f32` logits and the same gradient, bit for bit, and the native training run on ten threads under the plan ends on the default layout's weights byte for byte: yes.

## The linear products alone (ten threads, pinned)

`Linear{In,Out}.apply` of the course GPT, `x · w + b` at x : [16, 256, In], under the layout the forward plan gives `Linear.apply` (one layout for its three shapes); PyTorch `x @ w + b`.

A call's time on each side, its product and its bias together; for the native backend also its two kernels' alone (`presk bench --calibrate`), the rest of its call being its host's buffers and copies. GFLOP/s: the product's operations over the call's time, or over the kernels'.

| product | GFLOP | native ms | its kernels ms | C ms | PyTorch ms | native GFLOP/s | its kernels | C | PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| [16, 256, 384] · [384, 384] | 1.21 | 19.56 | 4.95 | 2.33 | 1.26 | 62 | 244 | 517 | 956 |
| [16, 256, 384] · [384, 1536] | 4.83 | 53.73 | 17.29 | 8.51 | 6.78 | 90 | 280 | 568 | 713 |
| [16, 256, 1536] · [1536, 384] | 4.83 | 69.28 | 13.93 | 9.48 | 5.16 | 70 | 347 | 510 | 937 |

## Kernels and host, the forward pass under its plan (ten threads, pinned)

The same kernels on both presk sides (C1), so what differs is who compiles them (Cranelift against GCC `-O2`) and what runs around them. The native kernels' time is `presk bench --calibrate`'s (every launch timed, its threads' start included); the C kernels' is the instrumented program's (below, three calls after a first; the program's median call is uninstrumented). The host is the call's median less its kernels. Under the plan only: `--calibrate` runs a nest no plan states scalar, where the native backend gives it its natural lanes, so under the default layout it times other kernels; the forward plan leaves only the host's small kernels (the causal mask, the positions) unstated.

| side | the call, ms | its kernels, ms | the rest, the host's, ms |
|---|---|---|---|
| native | 710.0 | 361.8 | 348.3 |
| C | 284.9 | 268.0 | 16.9 |

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

The instrumented program: 879.6 ms a call (the median of its four); its kernels 861.6 ms; the rest, the host's, 18.0 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the layout space's contractions (`dot`, `ein`) | 49 | 96.9 | 0.733 | 132 | 85.1 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 173 | 0.0 | 0.128 |  | 14.9 % |
| every kernel | 222 | | **0.862** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 111 | 0.260 | 30.2 % |
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 117 | 0.248 | 28.8 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 222 | 0.130 | 15.1 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 84 | 0.057 | 6.7 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 138 | 0.035 | 4.1 % |
| `k16`: `softmax axis seq2 (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, heads: 6, seq: …` | 6 |  | 0.031 | 3.6 % |
| `k23`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.019 | 2.2 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.019 | 2.2 % |

### The forward pass under its plan

The instrumented program: 284.7 ms a call (the median of its four); its kernels 268.0 ms; the rest, the host's, 16.6 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the layout space's contractions (`dot`, `ein`) | 49 | 96.9 | 0.171 | 567 | 63.8 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 179 | 0.0 | 0.097 |  | 36.2 % |
| every kernel | 228 | | **0.268** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 555 | 0.052 | 19.5 % |
| `k25`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 588 | 0.049 | 18.4 % |
| `k22`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 607 | 0.048 | 17.8 % |
| `k17`: `softmax axis seq2, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, hea…` | 6 |  | 0.020 | 7.5 % |
| `k24`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.016 | 5.9 % |
| `k19`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 438 | 0.011 | 4.1 % |
| `k14`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 494 | 0.010 | 3.7 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.009 | 3.5 % |

### The gradient, `loss_grad`, default layout

The instrumented program: 15 796.8 ms a call (the median of its four); its kernels 15 626.9 ms; the rest, the host's, 169.9 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the layout space's contractions (`dot`, `ein`) | 303 | 590.4 | 4.305 | 137 | 27.6 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 10.314 | 17 | 66.0 % |
| every other kernel | 1843 | 0.0 | 1.008 |  | 6.4 % |
| every kernel | 2220 | | **15.627** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k60`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 9 | 3.230 | 20.7 % |
| `k54`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 9 | 3.102 | 19.9 % |
| `k66`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 17 | 1.717 | 11.0 % |
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 36 | 120 | 1.455 | 9.3 % |
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 36 | 124 | 1.408 | 9.0 % |
| `k59`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 33 | 0.874 | 5.6 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 144 | 219 | 0.794 | 5.1 % |
| `k65`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 42 | 0.691 | 4.4 % |

### The gradient under its plan

The instrumented program: 12 239.1 ms a call (the median of its four); its kernels 12 097.8 ms; the rest, the host's, 141.3 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the layout space's contractions (`dot`, `ein`) | 303 | 590.4 | 1.443 | 409 | 11.9 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 9.715 | 18 | 80.3 % |
| every other kernel | 1879 | 0.0 | 0.940 |  | 7.8 % |
| every kernel | 2256 | | **12.098** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k61`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 9 | 3.128 | 25.9 % |
| `k55`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 9 | 3.111 | 25.7 % |
| `k67`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 18 | 1.641 | 13.6 % |
| `k60`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 41 | 0.710 | 5.9 % |
| `k66`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 52 | 0.561 | 4.6 % |
| `k54`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 53 | 0.547 | 4.5 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 144 | 406 | 0.428 | 3.5 % |
| `k22`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 36 | 435 | 0.400 | 3.3 % |

## The machine and PyTorch's libraries

```
date 2026-09-28T20:41:39+02:00
presk 2ea860d, presk 0.9.0
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Vitesse maximale du processeur en MHz : 3900,0000
Nom de modèle : Cortex-A725
Vitesse maximale du processeur en MHz : 2808,0000
load 5.21 6.52 7.01
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
