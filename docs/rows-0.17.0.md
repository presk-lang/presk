# The rows of record at 0.17.0 (plan 0.17.0, step 8)

Every row the version measures, on the version's binary
(`out/bin/presk-0.17.0-step8a`, the tree of steps 1 to 7, presk `49840da`;
step 7's fix, `ced194e`, changes no bit and no timed path, D180). No row of
record is solved or timed again (plan §12).
The decision is D181. Two phases:
- **Phase A, untimed** (`out/step8-0.17.0/phaseA.sh`): the solves of QM1
  and QM2, side by side, and the margins of QM3. The envelopes of record
  are reused, since the version changes the solver's menu and not a
  calibration (QM0): 0.16.1's for V6 and the course logits (D170), the
  trained curve's (D132), BERT-large's two blocks' at 0.16.0
  (`make compile-times`). The exact logits and every row of 0.16.1 are
  linked from their directories of record (D163's `link_rows`), so the
  margins ran only the new realizations, on the Code civil's 461 held-out
  windows and the Code pénal's first 461 (D168's judge set).
- **Phase B, timed, alone on falicon** (`out/step8-0.17.0/phaseB.sh`), by
  D127's protocol: every round starts when the one-minute load is under
  1.5, and the load and the busiest processes are logged at each start.
  It ran from 11:42 to 15:06 on 2026-10-29, every round's load
  between 0.83 and 1.49 (`out/step8-0.17.0/phaseB.log`).

**How the phases ran.**
- **The first night.** Phase A started on 2026-10-28 at 20:50. At 21:39
  step 7's first full-size run was OOM-killed beside it. systemd then
  stopped the session's whole scope, and with it phase A's course and
  BERT solves and the chain that was to start phase B (D180).
- **The re-run.** Those solves ran again from 2026-10-29 04:58, in a scope
  of their own (`PARTS="course bert"`). The scope was raised from 70 to
  95 GB at 07:51, when the six BERT-large solves reached 9.4 GB each.
- **BERT's four `--narrow measured` solves** were stopped by hand at
  08:19, after 3 h 20. Each of their measurements runs on the kernel
  interpreter, minutes at BERT-large's size, and they make up to about
  eighty: the course's took 55 measurements and 6 h 25. They are reported
  beside QM2, not judged, and ran again after phase B, from 15:06 to 23:19
  (§1, D181's addendum).
- **Beside phase A** ran step 7's RW4 and RW3, each heavy run in a memory
  scope of its own. Phase A ended at 11:42, and phase B started the same
  minute, alone.

## 1. QM1 and QM2: the one menu

`presk solve --menu f64,f32,bf16,int8,int4` (the one menu) and `presk
solve --quant` (the floats, then the integers in a second pass) on the
same envelopes and budgets. Each was run under the solver's defaults, then
under `--narrow measured`, then with `--margin 1` as well (the user,
2026-10-28). The figures are `out/step8-0.17.0/qm.py`'s
(`out/step8-0.17.0/qm.md`). A traffic ratio is all-`f64`'s traffic over
the realization's. A gap is measured against the exact reference on the
envelope's samples.

| row | setting | values away from f64 | parameters | traffic ratio | measured | budget | inside |
|---|---|---|---|---|---|---|---|
| the curve at D = 64, 1e-1 | the defaults | 19 of 19 | 0 f32, 0 bf16, 7 int8, 2 int4 | 3.55× | 1.720e-04 | 2.788e-01 | yes |
| the curve at D = 64, 1e-2 | the defaults | 19 of 19 | 5 f32, 2 bf16, 2 int8, 0 int4 | 2.55× | 7.182e-05 | 2.788e-02 | yes |
| the curve at D = 64, 1e-3 | the defaults | 19 of 19 | 8 f32, 1 bf16 | 2.03× | 3.369e-07 | 2.788e-03 | yes |
| the curve at D = 64, 1e-4 | the defaults | 19 of 19 | 9 f32, 0 bf16 | 1.99× | 1.242e-08 | 2.788e-04 | yes |
| the curve at D = 64, 1e-5 | the defaults | 18 of 19 | 9 f32, 0 bf16 | 1.96× | 1.649e-08 | 2.788e-05 | yes |
| the curve at D = 64, 1e-6 | the defaults | 15 of 19 | 9 f32, 0 bf16 | 1.67× | 2.275e-08 | 2.788e-06 | yes |
| the curve at D = 64, 1e-7 | the defaults | 5 of 19 | 9 f32, 0 bf16 | 1.30× | 1.576e-09 | 2.788e-07 | yes |
| V6, the logits at D = 64, ε* = 4.67·10⁻⁴ | the defaults | 18 of 19 | 9 f32, 0 bf16 | 1.96× | 2.138e-06 | 4.670e-04 | yes |
| V6, the logits at D = 64, ε* = 4.67·10⁻⁴ | `--narrow measured` | 19 of 19, narrowed 1 in 2 pass(es) | 9 f32, 0 bf16 | 1.99× | 2.268e-06 | 4.670e-04 | yes |
| V6, the logits at D = 64, ε* = 4.67·10⁻⁴ | `--narrow measured --margin 1` | 19 of 19, narrowed 1 in 2 pass(es) | 9 f32, 0 bf16 | 1.99× | 2.268e-06 | 4.670e-04 | yes |
| the course logits, ε* = 4.96·10⁻⁴ | the defaults | 17 of 18 | 9 f32, 0 bf16 | 1.98× | 3.017e-06 | 4.960e-04 | yes |
| the course logits, ε* = 4.96·10⁻⁴ | `--narrow measured` | 18 of 18, narrowed 1 in 2 pass(es) | 9 f32, 0 bf16 | 1.99× | 2.526e-06 | 4.960e-04 | yes |
| the course logits, ε* = 4.96·10⁻⁴ | `--narrow measured --margin 1` | 18 of 18, narrowed 1 in 2 pass(es) | 9 f32, 0 bf16 | 1.99× | 2.526e-06 | 4.960e-04 | yes |
| BERT-large, two blocks, 10⁻³ | the defaults | 19 of 19 | 8 f32, 0 bf16 | 2.00× | 1.059e-05 | 6.254e-03 | yes |
| BERT-large, two blocks, 10⁻³ | `--narrow measured` | 19 of 19, narrowed 38 in 3 pass(es) | 0 f32, 0 bf16, 4 int8, 4 int4 | 6.73× | 2.213e-04 | 6.254e-03 | yes |
| BERT-large, two blocks, 10⁻³ | `--narrow measured --margin 1` | 19 of 19, narrowed 37 in 3 pass(es) | 0 f32, 0 bf16, 5 int8, 3 int4 | 8.41× | 5.912e-03 | 6.254e-03 | yes |

**QM1 holds.** Every one-menu solve is inside its budget by measurement,
with no widening round:
- the curve at D = 64, at all seven budgets;
- V6 at ε\* = 4.67·10⁻⁴;
- the course logits at ε\* = 4.96·10⁻⁴;
- BERT-large's two blocks at 10⁻³;
- every `--narrow measured` row, BERT-large's included (8.41× with
  `--margin 1`, measured 5.91·10⁻³).

**The curve under the one menu** against 0.16.1's floating-point curve
(D170, the same envelope):

| budget | 0.16.1 | the one menu | integers taken |
|---|---|---|---|
| 10⁻¹ | 3.33× | **3.55×** | 7 `int8`, 2 `int4` |
| 10⁻² | 2.51× | **2.55×** | 2 `int8` |
| 10⁻³ | 2.03× | 2.03× | none |
| 10⁻⁴ | 1.99× | 1.99× | none |
| 10⁻⁵ | 1.96× | 1.96× | none |
| 10⁻⁶ | 1.67× | 1.67× | none |
| 10⁻⁷ | 1.30× | 1.30× | none |

From 10⁻³ down, the one menu's realizations are 0.16.1's, with the same
measured gaps. At 10⁻¹ and 10⁻² the integers buy 7 % and 2 % more
traffic. Those two realizations are not certified at 10⁻³, and neither are
0.16.1's: on the Code civil's 461 windows, 918 and 72 decisions flip,
where 0.16.1's flip 475 and 78.

| row | setting | one menu: parameters | ratio | measured | two passes: matrices quantized | ratio | measured | one ≥ two | verdict |
|---|---|---|---|---|---|---|---|---|---|
| V6, the logits at D = 64, ε* = 4.67·10⁻⁴ | the defaults | 9 f32, 0 bf16 | 1.96× | 2.138e-06 | 0 of 9 (0 int8, 0 int4) | 1.96× | 2.138e-06 | yes | holds |
| V6, the logits at D = 64, ε* = 4.67·10⁻⁴ | `--narrow measured` | 9 f32, 0 bf16 | 1.99× | 2.268e-06 | 0 of 9 (0 int8, 0 int4) | 1.99× | 2.268e-06 | yes | (reported) |
| V6, the logits at D = 64, ε* = 4.67·10⁻⁴ | `--narrow measured --margin 1` | 9 f32, 0 bf16 | 1.99× | 2.268e-06 | 0 of 9 (0 int8, 0 int4) | 1.99× | 2.268e-06 | yes | (reported) |
| the course logits, ε* = 4.96·10⁻⁴ | the defaults | 9 f32, 0 bf16 | 1.98× | 3.017e-06 | 0 of 9 (0 int8, 0 int4) | 1.98× | 3.017e-06 | yes | holds |
| the course logits, ε* = 4.96·10⁻⁴ | `--narrow measured` | 9 f32, 0 bf16 | 1.99× | 2.526e-06 | 0 of 9 (0 int8, 0 int4) | 1.99× | 2.526e-06 | yes | (reported) |
| the course logits, ε* = 4.96·10⁻⁴ | `--narrow measured --margin 1` | 9 f32, 0 bf16 | 1.99× | 2.526e-06 | 0 of 9 (0 int8, 0 int4) | 1.99× | 2.526e-06 | yes | (reported) |
| BERT-large, two blocks, 10⁻³ | the defaults | 8 f32, 0 bf16 | 2.00× | 1.059e-05 | 10 of 10 (6 int8, 4 int4) | 4.93× | 3.821e-03 | no | fails |
| BERT-large, two blocks, 10⁻³ | `--narrow measured` | 0 f32, 0 bf16, 4 int8, 4 int4 | 6.73× | 2.213e-04 | 10 of 10 (6 int8, 4 int4) | 6.80× | 4.637e-03 | no | (reported) |
| BERT-large, two blocks, 10⁻³ | `--narrow measured --margin 1` | 0 f32, 0 bf16, 5 int8, 3 int4 | 8.41× | 5.912e-03 | 10 of 10 (6 int8, 4 int4) | 6.80× | 4.637e-03 | yes | (reported) |

**QM2 fails, on BERT-large.**
- **V6 and the course logits.** The one menu holds at equality under the
  defaults: it takes no integer at ε\*, and its realizations are 0.16.1's,
  byte for byte. Under `--narrow measured` both solves reach 1.99× on the
  course logits, after 55 measurements each.
- **BERT-large's two blocks at 10⁻³.** The one menu takes no integer, at
  2.00×. The two-pass solve quantizes the ten matrices (six `int8`, four
  `int4`), at 4.93×, measured 3.82·10⁻³ inside the budget of
  6.25·10⁻³.
- **Why.** Every integer term, `s · max |ŵ − w|`, prices a matrix's
  quantization at its largest single move times the 1-norm of its
  sensitivity. That is about the whole budget at 10⁻³ (D177's finding on
  the trained GPT). The second pass prices by the root-sum-square of the
  moves (D49), and its measurement confirms it with room to spare.
- **BERT's `--narrow measured` rows** (D181's addendum) ran again after
  phase B, from 15:06 to 23:19, beside step 9's checks. The narrowing
  corrects the term. The one menu takes four `int8` and four `int4`
  matrices, at 6.73× against the two passes' 6.80×, 1 % short at the
  default margin. With `--margin 1` it takes five `int8` and three `int4`,
  at 8.41×, measured 5.91·10⁻³ inside 6.25·10⁻³, where the two passes
  stay at 6.80×. The two passes narrow the floats under the margin and
  quantize after it (8.32·10⁻⁴, then 4.64·10⁻³), so the margin does not
  move them. Each one-menu solve took 43 measurements and 8 h 13 min, each
  two-pass one 27 and 6 h 15 min.

## 2. QM3: certified, and timed

The one menu's realization of the course logits at ε\* is 0.16.1's, byte
for byte (no integer taken), so its certificate is 0.16.1's (D170). The
margins were taken again on it, by `margins_course.sh`, and so was the
`--narrow measured` realization beside it:

| realization | the Code civil's 461: gap on the logits | flips | certified at 10⁻³ | the Code pénal's 461: gap | flips |
|---|---|---|---|---|---|
| 0.16.1's, at ε\* = 4.96·10⁻⁴ | 8.695·10⁻⁶ | 0 | yes | 2.238·10⁻⁵ | 0 |
| the one menu's | 8.695·10⁻⁶ | 0 | yes | 2.238·10⁻⁵ | 0 |
| the one menu's, `--narrow measured` | 8.871·10⁻⁶ | 0 | yes | 2.210·10⁻⁵ | 0 |

At D = 64, beside: V6's one-menu realization is 2.883·10⁻⁶ from the
exact logits on the Code civil and 3.081·10⁻⁶ on the Code pénal, with no
flip, as 0.16.1's.

**QM3's first part holds:** certified on the Code civil, and Y1 on the
Code pénal (Z2's (ii) and (iii)). **Its time** (phase B, `presk bench experiments/margins.psk --entry
logits` at the course constants, B = 16, T = 256, the default layout, ten
X925 pinned, five processes, the medians): 0.16.1's realization 875.9 ms,
the one menu's, the same realization, 892.8 ms (1.019×, the processes'
spread), its `--narrow measured` one 876.4 ms (1.001×), all-`f32` 866.3 ms
(0.989×). **QM3 holds**, its time reported.

## 3. NB2 and NB3: T2 and T4 with the LLVM backend

`experiments/pytorch_baseline.sh` with `LLVM=1`, five rounds, each
round a process of every side (D127's protocol). The course GPT at
B = 16, T = 256 in `f32`, ten X925 pinned, ten threads, under the frozen
profile's plans. The C program waits as the native backend's pool does
(`GOMP_SPINCOUNT=8M`, D97's addendum). The medians of the five processes
(`out/retime-0.17.0/table4.md`):

| workload | C program (GCC `-O2`) | Cranelift, level 2 | ÷ C | LLVM | ÷ C |
|---|---|---|---|---|---|
| the forward pass `GPT.apply` | 206.5 ms | 246.4 ms | 1.19× | 205.5 ms | 0.995× |
| the step, `loss_grad` + `adamw.step` | 808.1 ms | 803.4 ms | 0.994× | 703.7 ms | 0.871× |

- **NB2 holds.** Cranelift with level 2 takes 1.19× the C program's
  time on the forward pass and 0.994× on the step, under T4's
  threshold of 1.3×. At 0.16.0, T4 failed at 1.63× and 1.67× (D166).
- **NB3 holds.** The LLVM backend takes 0.995× and 0.871× of it, so
  the compiler's own backend is at GCC on its kernels, C1 and H kept
  (NB0, D176).
- **The step's 0.871× is mostly the C program's host.** Kernel for kernel
  (§4), LLVM's step is 683 ms against GCC's 695 ms, 0.98×. The C program's
  gradient spends about 110 ms outside its kernels, as D174 found, and the
  native host about 20 ms.
- **The step is now at the C program's level on both native backends.**
  The forward pass is not on Cranelift: what remains there is the
  register tile's broadcast, which level 3 does without (D175).
- **The LLVM backend's training run** gives `presk train`'s final
  checkpoint, byte for byte, in all five rounds (`train.llvm.same`).


## 4. NB4: the kernels per backend, and LLVM's compile time

`experiments/kernel_shares.sh` (D174's method), five processes of each
side, with Cranelift as at `v0.16.1` (step 1's binary), Cranelift with
level 2 and LLVM (`BENCH=--llvm`); the C program's sides are the same
programs, under GCC and clang. The sums of the kernels' median launches, ms
a call (`out/kernel-shares-0.17.0-rows/`):

| entry | setting | Cranelift, `v0.16.1` | Cranelift, level 2 | LLVM | C, GCC `-O2` | C, clang `-O2` | C, clang `-O3` |
|---|---|---|---|---|---|---|---|
| the forward pass `GPT.apply` | ten X925, ten threads | 331.40 | 239.90 | 202.25 | 202.44 | — | — |
| the gradient `loss_grad` | ten X925, ten threads | 1291.56 | 766.01 | 671.82 | 679.88 | — | — |
| the optimizer `adamw.step` | ten X925, ten threads | 15.08 | 12.32 | 11.32 | 15.51 | — | — |
| the forward pass `GPT.apply` | one X925, one thread | 3035.59 | 2045.39 | 1616.18 | 2182.94 | 1663.95 | 1660.24 |
| the gradient `loss_grad` | one X925, one thread | 10099.58 | 6040.29 | 4996.54 | 6725.13 | 6002.82 | 5977.66 |
| the optimizer `adamw.step` | one X925, one thread | 34.41 | 25.00 | 24.96 | 25.63 | 25.60 | 25.65 |

- On ten cores, level 2 takes the forward pass's kernels from 331 to 240 ms
  and the gradient's from 1 292 to 766 ms; LLVM's are at GCC's on the
  forward pass (202.3 against 202.4 ms) and under it on the gradient (672
  against 680 ms).
- On one core, LLVM's kernels are faster than every build of the exported
  C (1 616 ms on the forward pass, clang `-O3` 1 660, GCC 2 183): they are
  compiled with `-mcpu=native`, the C program's builds without it.

**The LLVM backend's compile time** (`out/llvm-compile-0.17.0/`):

The wall time of one `presk bench … --warmup 0 --repeat 1` (the front end, the kernels' compilation and one call), ten X925 pinned: on LLVM with an empty cache, again with the cache it filled, and on Cranelift. Five processes each; the compile time is cold less cached.

| program | kernels compiled | cold (s) | cached (s) | Cranelift (s) | compiling (s) | a kernel (ms) |
|---|---|---|---|---|---|---|
| fwd | 23 | 2.30 | 1.21 | 1.22 | 1.10 | 48 |
| grad | 90 | 5.72 | 1.96 | 2.06 | 3.77 | 42 |
| adam | 49 | 5.30 | 3.50 | 3.57 | 1.80 | 37 |

A kernel takes 37 to 48 ms to compile through `clang`; cached, a program
starts as fast as on Cranelift.

## 5. TS2: the step at course scale

Per step: a round's four-step time minus its one-step time, over three; each a `presk train` process of its own, at the course constants, on every core (`j_step.sh`'s setting).

| realization | round 1 | round 2 | round 3 | round 4 | round 5 | median (s) | fastest | ÷ all-`f64` |
|---|---|---|---|---|---|---|---|---|
| all-`f64` | 14.85 | 14.83 | 15.23 | 15.01 | 15.20 | **15.01** | 14.83 | 1.00× |
| J′, by function (`out/step_1e-3.json`) | 7.00 | 7.06 | 6.99 | 6.95 | 6.95 | **6.99** | 6.95 | 2.15× |
| by value (D178) | 7.84 | 7.87 | 7.77 | 7.74 | 7.80 | **7.80** | 7.74 | 1.92× |
| J′ with `θ′`, `m′`, `v′` in `f64` (beside) | 7.60 | 7.57 | 7.59 | 7.58 | 7.50 | **7.58** | 7.50 | 1.98× |

**TS2 fails, on its second half.** The per-value realization's step is
1.92× faster than all-`f64`'s (at least 1.3×: yes), but slower than J′'s
per-function realization (7.80 s against 6.99 s, at most: no). D178
predicted 0.11 % slower (TS1); it is 11.6 % slower. J′ with the
optimizer's state kept in `f64`, which the per-value space keeps by the
user's decision (the plan's question 6), is already 8.4 % slower than J′
(7.58 s): most of the gap is that state.

## 6. TS3: the conversions' share of the step

Four steps, the first left out; three processes each, the median.

| scale | realization | the step (ms) | `buffers.convert` (ms) | its share |
|---|---|---|---|---|
| D = 64 | all-`f64` | 214.3 | 0.07 | 0.03 % |
| D = 64 | J's (`out/h6-0.16.0/solved-train.json`) | 319.2 | 39.89 | 12.50 % |
| D = 64 | J′'s | 133.0 | 0.92 | 0.71 % |
| D = 64 | the per-value one | 175.3 | 22.35 | 12.75 % |
| course | all-`f64` | 14920.5 | 0.31 | 0.00 % |
| course | J's (`out/h6-0.16.0/solved-train.json`) | 15251.6 | 235.39 | 1.54 % |
| course | J′'s | 6931.4 | 68.66 | 0.99 % |
| course | the per-value one | 7879.7 | 245.60 | 3.12 % |

- **J at D = 64 is explained in part.** Its realization's step spends
  12.5 % of its time converting buffers (40 ms of 319 ms), where all-`f64`
  spends none; that is 38 % of its 105 ms over all-`f64` (D166's 0.74×).
- **The per-value realization** converts as much at D = 64 (12.8 %) and
  3.1 % of its step at course scale (246 ms, J′'s 69 ms): the
  conversions are 177 ms of its 948 ms over J′, the state kept wide most
  of the rest (TS2).
