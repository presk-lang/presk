> The PyTorch baseline at version 0.10.0 (plan 0.10.0 step 8, D100: `make pytorch-baseline` on presk `8827c52`, the version's committed tree, begun 2026-10-02 at 15:19 on a quiet machine, alone, a one-minute load of 1.8 left by `make tutorial`, the run of record), copied from `out/pytorch-baseline/report.md`. T2, T4 and T5 are judged on it: the C training step under the plans 0.99× PyTorch's (T2 holds); the native backend 1.52× the C program on the forward pass and 1.60× on the step like for like, the C program's threads waiting as the native ones (T4 fails, by the kernels' code).

# The course GPT on three sides: presk 0.10.0 against PyTorch

The course GPT (`corpus/nlp/05_gpt.psk`, D = 384, 6 heads, 6 blocks, V = 92) at B = 16, T = 256, `f32` on every side, with the checkpoint `out/course_probe_f32/final.psk.d` and the token indices the C export synthesizes; 2026-10-02, presk `8827c52`, `experiments/pytorch_baseline.sh`. Three sides: presk's native backend (`presk bench --plan-only`, `presk train`), the C program `presk export --target c` writes (GCC 13, `-O2`, OpenMP), and PyTorch 2.14.0+cpu running the model `presk export --target pytorch` writes. Each time is the median of seven calls or steps, the first call or step, which compiles natively, left out on the native and PyTorch sides; the C program's seven calls are all counted.

**The C program's threads.** Each C program is timed twice: under the OpenMP runtime's default, by which an idle worker sleeps after about 0.1 ms (libgomp's 300 000 spins on these cores), and with `GOMP_SPINCOUNT=8M`, by which it spins about 2 ms before it sleeps, as the native backend's pool does (presk D97's addendum). T4 compares the native backend with the second, both sides waiting alike; T2 compares PyTorch, at its own runtime's default, with the first. The same bits either way.

**Operations.** The forward pass is 96.93 GFLOP of contractions, a training step 290.78 (each contraction and its two derivatives); PyTorch's counter (`torch.utils.flop_counter`) gives 96.93 and 290.78. The rates below divide these by the median, whatever a side executes: presk's gradient recomputes the forward pass's products (below, where the C program's time goes).

## The forward pass

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 1 428.6 | 1 423.3 | 67.8 | 9.75 |
| 10 (ten X925, pinned) | native | plan | 340.4 | 339.9 | 284.7 | 2.32 |
| 10 (ten X925, pinned) | C | default | 832.4 | 800.5 | 116.4 | 5.68 |
| 10 (ten X925, pinned) | C | plan | 272.9 | 264.1 | 355.2 | 1.86 |
| 10 (ten X925, pinned) | C, `GOMP_SPINCOUNT=8M` | default | 768.9 | 765.6 | 126.1 | 5.25 |
| 10 (ten X925, pinned) | C, `GOMP_SPINCOUNT=8M` | plan | 223.5 | 223.2 | 433.6 | 1.53 |
| 10 (ten X925, pinned) | PyTorch | its own | 146.5 | 130.0 | 661.6 | 1.00 |
| 20 (twenty, unpinned) | native | default | 2 006.0 | 2 002.3 | 48.3 | 7.84 |
| 20 (twenty, unpinned) | native | plan | 340.5 | 340.1 | 284.7 | 1.33 |
| 20 (twenty, unpinned) | C | default | 1 534.1 | 1 452.4 | 63.2 | 6.00 |
| 20 (twenty, unpinned) | C | plan | 272.0 | 261.0 | 356.3 | 1.06 |
| 20 (twenty, unpinned) | C, `GOMP_SPINCOUNT=8M` | default | 1 569.5 | 1 560.7 | 61.8 | 6.13 |
| 20 (twenty, unpinned) | C, `GOMP_SPINCOUNT=8M` | plan | 226.0 | 225.6 | 428.9 | 0.88 |
| 20 (twenty, unpinned) | PyTorch | its own | 255.8 | 245.1 | 378.9 | 1.00 |

## One training step

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native, `presk train` | default | 14 004.8 | 20.8 | 13.89 |
| 10 (ten X925, pinned) | native, `presk train` | plan | 1 425.2 | 204.0 | 1.41 |
| 10 (ten X925, pinned) | native, `loss_grad` 13 890 + `adamw.step` 19 | default | 13 909.2 | 20.9 | 13.79 |
| 10 (ten X925, pinned) | native, `loss_grad` 1 325 + `adamw.step` 16 | plan | 1 340.6 | 216.9 | 1.33 |
| 10 (ten X925, pinned) | C, `loss_grad` 11 650 + `adamw.step` 17 | default | 11 667.4 | 24.9 | 11.57 |
| 10 (ten X925, pinned) | C, `loss_grad` 981 + `adamw.step` 14 | plan | 995.4 | 292.1 | 0.99 |
| 10 (ten X925, pinned) | C, `GOMP_SPINCOUNT=8M`, `loss_grad` 11 599 + `adamw.step` 14 | default | 11 612.6 | 25.0 | 11.51 |
| 10 (ten X925, pinned) | C, `GOMP_SPINCOUNT=8M`, `loss_grad` 824 + `adamw.step` 13 | plan | 836.8 | 347.5 | 0.83 |
| 10 (ten X925, pinned) | PyTorch, `train.py` | its own | 1 008.6 | 288.3 | 1.00 |
| 20 (twenty, unpinned) | native, `presk train` | default | 14 247.7 | 20.4 | 10.33 |
| 20 (twenty, unpinned) | native, `presk train` | plan | 1 451.2 | 200.4 | 1.05 |
| 20 (twenty, unpinned) | native, `loss_grad` 13 682 + `adamw.step` 24 | default | 13 705.8 | 21.2 | 9.93 |
| 20 (twenty, unpinned) | native, `loss_grad` 1 332 + `adamw.step` 22 | plan | 1 353.3 | 214.9 | 0.98 |
| 20 (twenty, unpinned) | C, `loss_grad` 11 808 + `adamw.step` 18 | default | 11 826.0 | 24.6 | 8.57 |
| 20 (twenty, unpinned) | C, `loss_grad` 1 037 + `adamw.step` 18 | plan | 1 055.1 | 275.6 | 0.76 |
| 20 (twenty, unpinned) | C, `GOMP_SPINCOUNT=8M`, `loss_grad` 11 936 + `adamw.step` 17 | default | 11 953.0 | 24.3 | 8.66 |
| 20 (twenty, unpinned) | C, `GOMP_SPINCOUNT=8M`, `loss_grad` 847 + `adamw.step` 15 | plan | 861.9 | 337.4 | 0.62 |
| 20 (twenty, unpinned) | PyTorch, `train.py` | its own | 1 379.8 | 210.7 | 1.00 |

The C step is its two programs' medians added, and so is the native step by `presk bench` (since 0.10.0 step 6, T4's step like for like); neither drops (the export drops keys, so its dropout is the expectation, and `presk bench` does not train), where `presk train` and `train.py` drop with their own masks. `train.py`'s step builds the module `GPT(256)` at every call of the exported `loss` (`torch.func.functional_call` on a fresh module, its initializers drawing every parameter): 22.1 ms on ten threads, inside its time.

Every layout and thread count gives the same `f32` logits and the same gradient, bit for bit, and the native training run on ten threads under the plan ends on the default layout's weights byte for byte: yes.

## T4: the native backend against the C program

Plan 0.10.0's statement T4 (§3.5, as amended after steps 5 and 7): the native backend's forward pass and training step under the plans take at most 1.3× the C program's, same plans, same cores; the step compared like for like, the native `loss_grad` and `adamw.step` timed by `presk bench` against the C program's, neither drawing dropout masks, `presk train`'s step reported beside it. Judged on the ten X925 cores pinned, the C program's workers waiting as the native pool's (`GOMP_SPINCOUNT=8M`), the row of record; the C program's default waiting and every core unpinned reported.

| threads | the C program's waiting | workload | native ms | C ms | native ÷ C | T4 (≤ 1.3×) |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | `GOMP_SPINCOUNT=8M` | the forward pass | 340.4 | 223.5 | 1.52 | fails |
| 10 (ten X925, pinned) | `GOMP_SPINCOUNT=8M` | the step: `loss_grad` + `adamw.step` | 1 340.6 | 836.8 | 1.60 | fails |
| 10 (ten X925, pinned) | `GOMP_SPINCOUNT=8M` | beside: `presk train`'s step, drawing its masks | 1 425.2 | 836.8 | 1.70 | — |
| 10 (ten X925, pinned) | OpenMP's default | the forward pass | 340.4 | 272.9 | 1.25 | (reported) |
| 10 (ten X925, pinned) | OpenMP's default | the step: `loss_grad` + `adamw.step` | 1 340.6 | 995.4 | 1.35 | (reported) |
| 10 (ten X925, pinned) | OpenMP's default | beside: `presk train`'s step, drawing its masks | 1 425.2 | 995.4 | 1.43 | — |
| 20 (twenty, unpinned) | `GOMP_SPINCOUNT=8M` | the forward pass | 340.5 | 226.0 | 1.51 | (reported) |
| 20 (twenty, unpinned) | `GOMP_SPINCOUNT=8M` | the step: `loss_grad` + `adamw.step` | 1 353.3 | 861.9 | 1.57 | (reported) |
| 20 (twenty, unpinned) | `GOMP_SPINCOUNT=8M` | beside: `presk train`'s step, drawing its masks | 1 451.2 | 861.9 | 1.68 | — |
| 20 (twenty, unpinned) | OpenMP's default | the forward pass | 340.5 | 272.0 | 1.25 | (reported) |
| 20 (twenty, unpinned) | OpenMP's default | the step: `loss_grad` + `adamw.step` | 1 353.3 | 1 055.1 | 1.28 | (reported) |
| 20 (twenty, unpinned) | OpenMP's default | beside: `presk train`'s step, drawing its masks | 1 451.2 | 1 055.1 | 1.38 | — |

## The linear products alone (ten threads, pinned)

`Linear{In,Out}.apply` of the course GPT, `x · w + b` at x : [16, 256, In], under the layout the forward plan gives `Linear.apply` (one layout for its three shapes); PyTorch `x @ w + b`.

A call's time on each side, its product and its bias together; for the native backend also its kernels' alone (`presk bench --calibrate`: the product's and the bias's, or one when the plan states the bias as the product's epilogue, plan 0.10.0 step 7), the rest of its call being its host's buffers and copies. GFLOP/s: the product's operations over the call's time, or over the kernels'.

| product | GFLOP | native ms | its kernels ms | C ms | PyTorch ms | native GFLOP/s | its kernels | C | PyTorch |
|---|---|---|---|---|---|---|---|---|---|
| [16, 256, 384] · [384, 384] | 1.21 | 3.10 | 3.61 | 2.00 | 1.26 | 390 | 334 | 605 | 955 |
| [16, 256, 384] · [384, 1536] | 4.83 | 12.59 | 13.15 | 8.41 | 6.17 | 384 | 367 | 574 | 783 |
| [16, 256, 1536] · [1536, 384] | 4.83 | 12.47 | 13.12 | 7.76 | 4.68 | 387 | 368 | 623 | 1033 |

## Kernels and host, the forward pass under its plan (ten threads, pinned)

The same kernels on both presk sides (C1), so what differs is who compiles them (Cranelift against GCC `-O2`) and what runs around them. The native kernels' time is `presk bench --calibrate`'s (every launch timed, its threads' start included); the C kernels' is the instrumented program's (below, three calls after a first; the program's median call is uninstrumented). The host is the call's median less its kernels. Under the plan only: `--calibrate` runs a nest no plan states scalar, where the native backend gives it its natural lanes, so under the default layout it times other kernels; the forward plan leaves only the host's small kernels (the causal mask, the positions) unstated.

The C program's call is its time with `GOMP_SPINCOUNT=8M`, under which its kernels were timed.

| side | the call, ms | its kernels, ms | the rest, the host's, ms |
|---|---|---|---|
| native | 340.4 | 339.9 | 0.6 |
| C | 223.5 | 211.7 | 11.8 |

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

The instrumented program: 791.2 ms a call (the median of its four); its kernels 774.4 ms; the rest, the host's, 16.8 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.681 | 142 | 87.9 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 173 | 0.0 | 0.093 |  | 12.1 % |
| every kernel | 222 | | **0.774** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k21`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 123 | 0.236 | 30.4 % |
| `k24`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 127 | 0.229 | 29.5 % |
| `k10`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 230 | 0.126 | 16.3 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 87 | 0.056 | 7.2 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 143 | 0.034 | 4.4 % |
| `k16`: `softmax axis seq2 where (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : i64[seq: 256, seq2: 256]) -…` | 6 |  | 0.026 | 3.4 % |
| `k6`: `mean axis model (%0 : f32[batch: 16, seq: 256, model: 384]) -> %1 : f32[batch: 16, seq: 256]` | 26 |  | 0.019 | 2.4 % |
| `k23`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.017 | 2.1 % |

### The forward pass under its plan

The instrumented program: 227.0 ms a call (the median of its four); its kernels 211.7 ms; the rest, the host's, 15.2 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 49 | 96.9 | 0.154 | 629 | 72.8 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 0 | 0.0 | 0.000 |  | 0.0 % |
| every other kernel | 143 | 0.0 | 0.058 |  | 27.2 % |
| every kernel | 192 | | **0.212** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k21`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq:…` | 6 | 623 | 0.047 | 22.0 % |
| `k23`: `dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq:…` | 6 | 636 | 0.046 | 21.5 % |
| `k10`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 2…` | 24 | 654 | 0.044 | 20.9 % |
| `k16`: `softmax axis seq2 where, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : i64[seq: 256, s…` | 6 |  | 0.021 | 9.8 % |
| `k22`: `gelu (%0 : f32[batch: 16, seq: 256, 1536]) -> %1 : f32[batch: 16, seq: 256, 1536]` | 6 |  | 0.014 | 6.5 % |
| `k13`: `ein[batch, heads, seq, seq2] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, …` | 6 | 546 | 0.009 | 4.2 % |
| `k18`: `ein[batch, heads, seq, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6,…` | 6 | 584 | 0.008 | 3.9 % |
| `k17`: `mul (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : f32[batch: 16, heads: 6, seq: 256, seq2: 256]) …` | 6 |  | 0.004 | 2.0 % |

### The gradient, `loss_grad`, default layout

The instrumented program: 11 667.2 ms a call (the median of its four); its kernels 11 510.1 ms; the rest, the host's, 157.1 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.862 | 135 | 7.5 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 10.377 | 17 | 90.2 % |
| every other kernel | 865 | 0.0 | 0.272 |  | 2.4 % |
| every kernel | 1012 | | **11.510** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k58`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 9 | 3.276 | 28.5 % |
| `k53`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 9 | 3.131 | 27.2 % |
| `k63`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 17 | 1.701 | 14.8 % |
| `k57`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 33 | 0.872 | 7.6 % |
| `k62`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 42 | 0.690 | 6.0 % |
| `k52`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 42 | 0.688 | 6.0 % |
| `k29`: `dot (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 122 | 0.237 | 2.1 % |
| `k32`: `dot (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 123 | 0.235 | 2.0 % |

### The gradient under its plan

The instrumented program: 842.6 ms a call (the median of its four); its kernels 702.9 ms; the rest, the host's, 139.7 ms.

| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |
|---|---|---|---|---|---|
| the contractions `dot` and `ein` | 73 | 116.3 | 0.216 | 539 | 30.7 % |
| the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`) | 74 | 174.5 | 0.276 | 633 | 39.2 % |
| every other kernel | 878 | 0.0 | 0.211 |  | 30.1 % |
| every kernel | 1025 | | **0.703** | | 100 % |

| kernel | launches a call | GFLOP/s | s a call | share |
|---|---|---|---|---|
| `k60`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 598 | 0.048 | 6.9 % |
| `k29`: `dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq:…` | 6 | 624 | 0.046 | 6.6 % |
| `k59`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 633 | 0.046 | 6.5 % |
| `k31`: `dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq:…` | 6 | 633 | 0.046 | 6.5 % |
| `k66`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 634 | 0.046 | 6.5 % |
| `k53`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 634 | 0.046 | 6.5 % |
| `k54`: `dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 648 | 0.045 | 6.4 % |
| `k65`: `dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 655 | 0.044 | 6.3 % |

## The machine and PyTorch's libraries

```
date 2026-10-02T15:19:49+02:00
presk 8827c52, presk 0.10.0
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Vitesse maximale du processeur en MHz : 3900,0000
Nom de modèle : Cortex-A725
Vitesse maximale du processeur en MHz : 2808,0000
load 1.81 6.22 6.95
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
