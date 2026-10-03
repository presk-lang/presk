> The PyTorch baseline after 0.10.0 step 7's follow-up (D97's addendum: `make pytorch-baseline` on presk `14c24a0`, the manual's edits set aside, begun 2026-10-01 at 22:10 on a quiet machine, the run of record), copied from `out/pytorch-baseline-followup/report.md`, regenerated from its results by `122c74d`, which moves one sentence out of a table: the C programs always inline their kernels and are compiled on GCC without scheduling before register allocation, and every C program is timed twice, under OpenMP's default waiting and with `GOMP_SPINCOUNT=8M`, its workers then spinning about 2 ms as the native backend's do. T2 and T4 measured again; T4 is judged at step 8 (the user's answers on presk !47).

# The course GPT on three sides: presk 0.10.0-dev against PyTorch

The course GPT (`corpus/nlp/05_gpt.psk`, D = 384, 6 heads, 6 blocks, V = 92) at B = 16, T = 256, `f32` on every side, with the checkpoint `out/course_probe_f32/final.psk.d` and the token indices the C export synthesizes; 2026-10-01, presk `14c24a0`, `experiments/pytorch_baseline.sh`. Three sides: presk's native backend (`presk bench --plan-only`, `presk train`), the C program `presk export --target c` writes (GCC 13, `-O2`, OpenMP), and PyTorch 2.14.0+cpu running the model `presk export --target pytorch` writes. Each time is the median of seven calls or steps, the first call or step, which compiles natively, left out on the native and PyTorch sides; the C program's seven calls are all counted.

**The C program's threads.** Each C program is timed twice: under the OpenMP runtime's default, by which an idle worker sleeps after about 0.1 ms (libgomp's 300 000 spins on these cores), and with `GOMP_SPINCOUNT=8M`, by which it spins about 2 ms before it sleeps, as the native backend's pool does (presk D97's addendum). T4 compares the native backend with the second, both sides waiting alike; T2 compares PyTorch, at its own runtime's default, with the first. The same bits either way.

**Operations.** The forward pass is 96.93 GFLOP of contractions, a training step 290.78 (each contraction and its two derivatives); PyTorch's counter (`torch.utils.flop_counter`) gives 96.93 and 290.78. The rates below divide these by the median, whatever a side executes: presk's gradient recomputes the forward pass's products (below, where the C program's time goes).

## The forward pass

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 1 393.1 | 1 391.9 | 69.6 | 8.66 |
| 10 (ten X925, pinned) | native | plan | 323.4 | 322.7 | 299.7 | 2.01 |
| 10 (ten X925, pinned) | C | default | 854.4 | 852.6 | 113.4 | 5.31 |
| 10 (ten X925, pinned) | C | plan | 266.8 | 263.8 | 363.3 | 1.66 |
| 10 (ten X925, pinned) | C, `GOMP_SPINCOUNT=8M` | default | 775.3 | 774.5 | 125.0 | 4.82 |
| 10 (ten X925, pinned) | C, `GOMP_SPINCOUNT=8M` | plan | 219.9 | 219.5 | 440.8 | 1.37 |
| 10 (ten X925, pinned) | PyTorch | its own | 160.9 | 149.0 | 602.4 | 1.00 |
| 20 (twenty, unpinned) | native | default | 1 964.2 | 1 945.6 | 49.3 | 7.82 |
| 20 (twenty, unpinned) | native | plan | 324.3 | 323.6 | 298.9 | 1.29 |
| 20 (twenty, unpinned) | C | default | 1 535.1 | 1 459.3 | 63.1 | 6.11 |
| 20 (twenty, unpinned) | C | plan | 268.7 | 263.4 | 360.7 | 1.07 |
| 20 (twenty, unpinned) | C, `GOMP_SPINCOUNT=8M` | default | 1 589.6 | 1 588.0 | 61.0 | 6.32 |
| 20 (twenty, unpinned) | C, `GOMP_SPINCOUNT=8M` | plan | 221.1 | 220.4 | 438.5 | 0.88 |
| 20 (twenty, unpinned) | PyTorch | its own | 251.3 | 244.6 | 385.6 | 1.00 |

## One training step

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native, `presk train` | default | 14 013.3 | 20.8 | 14.01 |
| 10 (ten X925, pinned) | native, `presk train` | plan | 1 443.0 | 201.5 | 1.44 |
| 10 (ten X925, pinned) | native, `loss_grad` 13 916 + `adamw.step` 21 | default | 13 936.9 | 20.9 | 13.94 |
| 10 (ten X925, pinned) | native, `loss_grad` 1 324 + `adamw.step` 16 | plan | 1 340.6 | 216.9 | 1.34 |
| 10 (ten X925, pinned) | C, `loss_grad` 11 662 + `adamw.step` 19 | default | 11 680.4 | 24.9 | 11.68 |
| 10 (ten X925, pinned) | C, `loss_grad` 992 + `adamw.step` 15 | plan | 1 006.9 | 288.8 | 1.01 |
| 10 (ten X925, pinned) | C, `GOMP_SPINCOUNT=8M`, `loss_grad` 11 645 + `adamw.step` 14 | default | 11 658.9 | 24.9 | 11.66 |
| 10 (ten X925, pinned) | C, `GOMP_SPINCOUNT=8M`, `loss_grad` 841 + `adamw.step` 12 | plan | 852.8 | 341.0 | 0.85 |
| 10 (ten X925, pinned) | PyTorch, `train.py` | its own | 1 000.1 | 290.7 | 1.00 |
| 20 (twenty, unpinned) | native, `presk train` | default | 13 912.3 | 20.9 | 10.10 |
| 20 (twenty, unpinned) | native, `presk train` | plan | 1 437.5 | 202.3 | 1.04 |
| 20 (twenty, unpinned) | native, `loss_grad` 13 779 + `adamw.step` 24 | default | 13 803.4 | 21.1 | 10.02 |
| 20 (twenty, unpinned) | native, `loss_grad` 1 318 + `adamw.step` 22 | plan | 1 339.7 | 217.0 | 0.97 |
| 20 (twenty, unpinned) | C, `loss_grad` 11 882 + `adamw.step` 20 | default | 11 902.1 | 24.4 | 8.64 |
| 20 (twenty, unpinned) | C, `loss_grad` 1 123 + `adamw.step` 16 | plan | 1 139.2 | 255.2 | 0.83 |
| 20 (twenty, unpinned) | C, `GOMP_SPINCOUNT=8M`, `loss_grad` 11 875 + `adamw.step` 16 | default | 11 891.0 | 24.5 | 8.63 |
| 20 (twenty, unpinned) | C, `GOMP_SPINCOUNT=8M`, `loss_grad` 844 + `adamw.step` 15 | plan | 858.9 | 338.6 | 0.62 |
| 20 (twenty, unpinned) | PyTorch, `train.py` | its own | 1 378.1 | 211.0 | 1.00 |

The C step is its two programs' medians added, and so is the native step by `presk bench` (since 0.10.0 step 6, T4's step like for like); neither drops (the export drops keys, so its dropout is the expectation, and `presk bench` does not train), where `presk train` and `train.py` drop with their own masks. `train.py`'s step builds the module `GPT(256)` at every call of the exported `loss` (`torch.func.functional_call` on a fresh module, its initializers drawing every parameter): 22.0 ms on ten threads, inside its time.

Every layout and thread count gives the same `f32` logits and the same gradient, bit for bit, and the native training run on ten threads under the plan ends on the default layout's weights byte for byte: yes.

## T4: the native backend against the C program

Plan 0.10.0's statement T4 (§3.5, as amended after steps 5 and 7): the native backend's forward pass and training step under the plans take at most 1.3× the C program's, same plans, same cores; the step compared like for like, the native `loss_grad` and `adamw.step` timed by `presk bench` against the C program's, neither drawing dropout masks, `presk train`'s step reported beside it. Judged on the ten X925 cores pinned, the C program's workers waiting as the native pool's (`GOMP_SPINCOUNT=8M`), the row of record; the C program's default waiting and every core unpinned reported.

| threads | the C program's waiting | workload | native ms | C ms | native ÷ C | T4 (≤ 1.3×) |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | `GOMP_SPINCOUNT=8M` | the forward pass | 323.4 | 219.9 | 1.47 | fails |
| 10 (ten X925, pinned) | `GOMP_SPINCOUNT=8M` | the step: `loss_grad` + `adamw.step` | 1 340.6 | 852.8 | 1.57 | fails |
| 10 (ten X925, pinned) | `GOMP_SPINCOUNT=8M` | beside: `presk train`'s step, drawing its masks | 1 443.0 | 852.8 | 1.69 | — |
| 10 (ten X925, pinned) | OpenMP's default | the forward pass | 323.4 | 266.8 | 1.21 | (reported) |
| 10 (ten X925, pinned) | OpenMP's default | the step: `loss_grad` + `adamw.step` | 1 340.6 | 1 006.9 | 1.33 | (reported) |
| 10 (ten X925, pinned) | OpenMP's default | beside: `presk train`'s step, drawing its masks | 1 443.0 | 1 006.9 | 1.43 | — |
| 20 (twenty, unpinned) | `GOMP_SPINCOUNT=8M` | the forward pass | 324.3 | 221.1 | 1.47 | (reported) |
| 20 (twenty, unpinned) | `GOMP_SPINCOUNT=8M` | the step: `loss_grad` + `adamw.step` | 1 339.7 | 858.9 | 1.56 | (reported) |
| 20 (twenty, unpinned) | `GOMP_SPINCOUNT=8M` | beside: `presk train`'s step, drawing its masks | 1 437.5 | 858.9 | 1.67 | — |
| 20 (twenty, unpinned) | OpenMP's default | the forward pass | 324.3 | 268.7 | 1.21 | (reported) |
| 20 (twenty, unpinned) | OpenMP's default | the step: `loss_grad` + `adamw.step` | 1 339.7 | 1 139.2 | 1.18 | (reported) |
| 20 (twenty, unpinned) | OpenMP's default | beside: `presk train`'s step, drawing its masks | 1 437.5 | 1 139.2 | 1.26 | — |

## The linear products alone (ten threads, pinned)

`Linear{In,Out}.apply` of the course GPT, `x · w + b` at x : [16, 256, In], under the layout the forward plan gives `Linear.apply` (one layout for its three shapes); PyTorch `x @ w + b`.

A call's time on each side, its product and its bias together; for the native backend also its kernels' alone (`presk bench --calibrate`: the product's and the bias's, or one when the plan states the bias as the product's epilogue, plan 0.10.0 step 7), the rest of its call being its host's buffers and copies. GFLOP/s: the product's operations over the call's time, or over the kernels'.

| product | GFLOP | native ms | its kernels ms | C ms | PyTorch ms | native GFLOP/s | its kernels | C | PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| [16, 256, 384] · [384, 384] | 1.21 | 3.10 | 3.68 | 2.05 | 1.24 | 390 | 328 | 588 | 971 |
| [16, 256, 384] · [384, 1536] | 4.83 | 12.67 | 13.41 | 8.08 | 6.18 | 381 | 360 | 598 | 781 |
| [16, 256, 1536] · [1536, 384] | 4.83 | 12.56 | 13.20 | 7.96 | 5.21 | 385 | 366 | 607 | 927 |

## Kernels and host, the forward pass under its plan (ten threads, pinned)

The same kernels on both presk sides (C1), so what differs is who compiles them (Cranelift against GCC `-O2`) and what runs around them. The native kernels' time is `presk bench --calibrate`'s (every launch timed, its threads' start included); the C kernels' is the instrumented program's (below, three calls after a first; the program's median call is uninstrumented). The host is the call's median less its kernels. Under the plan only: `--calibrate` runs a nest no plan states scalar, where the native backend gives it its natural lanes, so under the default layout it times other kernels; the forward plan leaves only the host's small kernels (the causal mask, the positions) unstated.

The C program's call is its time with `GOMP_SPINCOUNT=8M`, under which its kernels were timed.

| side | the call, ms | its kernels, ms | the rest, the host's, ms |
|---|---|---|---|
| native | 323.4 | 323.5 | -0.1 |
| C | 219.9 | 207.8 | 12.1 |

## Distances to the native backend's `f64` logits

| side | ‖Δ‖₂ / ‖y‖₂ | max \|Δ\| | max \|Δ\| / max \|y\| | positions whose largest logit moves |
|---|---|---|---|---|
| PyTorch float64 (a check of the comparison) | 6.51e-16 | 1.08e-17 | 2.40e-15 | 0 of 4096 |
| native f32, and the C program | 3.26e-07 | 6.70e-09 | 1.49e-06 | 1 of 4096 |
| PyTorch float32 | 1.62e-07 | 2.74e-09 | 6.08e-07 | 0 of 4096 |

The checkpoint is a probe a few steps from the initialization (its head's largest weight is 1.8·10⁻⁵), so the logits are of order 10⁻³ and a position whose largest logit moves is a near-tie.

## Where the C program's time goes (ten threads, pinned, three calls after a first)

A copy of `model.c` with every launch timed on the thread that makes it, a kernel's call or the OpenMP region that divides it, its team's start and its wait for the slowest thread included (`experiments/kernel_time.py`, `kernel_time.c`); the kernels are the program's, and its outputs are unchanged. Four calls, the first left out, with `GOMP_SPINCOUNT=8M`. GFLOP counts every launch, recomputations included.

### The forward pass, default layout

The instrumented program: 783.7 ms a call (the median of its four); its kernels 767.6 ms; the rest, the host's, 16.1 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.672 | 144 | 87.6 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 173 | 0.0 | 0.095 |  | 12.4 % |
| every kernel | 222 | | **0.768** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 125 | 0.233 | 30.3 % |
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 131 | 0.221 | 28.8 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 228 | 0.127 | 16.6 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 86 | 0.056 | 7.3 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 142 | 0.034 | 4.4 % |
| `k16`: `softmax axis seq2 (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, heads: 6, seq: …` | 6 |  | 0.027 | 3.5 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.019 | 2.4 % |
| `k23`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.016 | 2.1 % |

### The forward pass under its plan

The instrumented program: 223.1 ms a call (the median of its four); its kernels 207.8 ms; the rest, the host's, 15.2 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.155 | 624 | 74.7 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 143 | 0.0 | 0.053 |  | 25.3 % |
| every kernel | 192 | | **0.208** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k21`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq:…` | 6 | 618 | 0.047 | 22.6 % |
| `k23`: `dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq:…` | 6 | 628 | 0.046 | 22.2 % |
| `k10`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 2…` | 24 | 650 | 0.045 | 21.5 % |
| `k16`: `softmax axis seq2, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) -> %1 : f32[batch: 16, hea…` | 6 |  | 0.017 | 8.1 % |
| `k22`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.014 | 6.6 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 555 | 0.009 | 4.2 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 584 | 0.008 | 4.0 % |
| `k17`: `mul (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) …` | 6 |  | 0.004 | 1.9 % |

### The gradient, `loss_grad`, default layout

The instrumented program: 11 683.9 ms a call (the median of its four); its kernels 11 519.5 ms; the rest, the host's, 164.4 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.824 | 141 | 7.2 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 10.413 | 17 | 90.4 % |
| every other kernel | 865 | 0.0 | 0.283 |  | 2.5 % |
| every kernel | 1012 | | **11.519** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k58`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 9 | 3.303 | 28.7 % |
| `k53`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 9 | 3.131 | 27.2 % |
| `k63`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 17 | 1.710 | 14.8 % |
| `k57`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 33 | 0.872 | 7.6 % |
| `k62`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 42 | 0.692 | 6.0 % |
| `k52`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 42 | 0.687 | 6.0 % |
| `k29`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 128 | 0.226 | 2.0 % |
| `k32`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 143 | 0.202 | 1.8 % |

### The gradient under its plan

The instrumented program: 819.9 ms a call (the median of its four); its kernels 691.7 ms; the rest, the host's, 128.2 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.213 | 547 | 30.7 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 0.276 | 633 | 39.9 % |
| every other kernel | 878 | 0.0 | 0.203 |  | 29.4 % |
| every kernel | 1025 | | **0.692** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k60`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 586 | 0.049 | 7.2 % |
| `k29`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq:…` | 6 | 627 | 0.046 | 6.7 % |
| `k66`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 630 | 0.046 | 6.7 % |
| `k53`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 639 | 0.045 | 6.6 % |
| `k31`: `dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq:…` | 6 | 639 | 0.045 | 6.6 % |
| `k59`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 642 | 0.045 | 6.5 % |
| `k54`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 647 | 0.045 | 6.5 % |
| `k65`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 662 | 0.044 | 6.3 % |

## The machine and PyTorch's libraries

```
date 2026-10-01T22:10:34+02:00
presk 14c24a0, presk 0.10.0-dev
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Vitesse maximale du processeur en MHz : 3900,0000
Nom de modèle : Cortex-A725
Vitesse maximale du processeur en MHz : 2808,0000
load 1.94 7.87 7.96
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
cflags -O2 -Wall -Wextra
c_wait GOMP_SPINCOUNT=8M
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
