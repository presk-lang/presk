# Changelog

## 0.7.0 — 2026-09-24

Ranking, not speed (`presk-specs/implementation/presk-plan-0.7.0.md`).
Three versions have measured precision and speed; none has measured the
thesis. RA4's hypothesis is that on regular tensor kernels a static cost
model orders more than 90 % of the pairs of variants autotuning
distinguishes, at a decision cost lower by orders of magnitude. 0.7.0 tests
it on the CPU backend that exists, with attention as the case study: a
layout space over each nest whose every point is exact by construction and
asserted bit-identical to the scalar sequential kernel, a plan object that
carries the layout beside the realization, a cost model that reads the
layout (reuse-aware traffic, threads, register pressure, an uncertainty),
calibrated on a GEMM the attention nests never enter, and an exhaustive
sweep pinned to the ten X925 cores and repeated (four hours later, where
the plan asked for a second day). Statements L to P. The version does not
promise to close the 22× against PyTorch.

**What it shows.** Exactness survives a space of variants: 17 879 points of
the course GPT's attention, a GEMM and a BERT-shaped sample — register
tiles, k-blocks, loop orders, lanes, packs, thread splits at any depth — are
every one bit-identical to the scalar sequential kernel, natively and on the
kernel interpreter (L). The space pays: its best points run the attention's
`ein`s 9.2× and 16.5× faster than 0.6.0 at batch 16 and 47× and 77× at batch
1, and a 1024³ `f32` product at 460 GFLOP/s, 5.49× 0.6.0's point (P). **What
it does not show** is RA4. The static model orders 47.9 % and 43.9 % of the
attention `ein`s' distinguished close pairs right where the hypothesis asks
more than 90 % (M), chooses 582–717 times faster than the sweep at batch 16
where it asks 10³ (N), and picks points 2.18–3.01 times slower than the best
(O). The cause is named: the model gives a load no weight and counts traffic
from L2, so it ranks lanes and thread splits (93–97 %) and cannot see a
register tile. Its candidates within `1 + u` still hold the best point, or
one within 1.08× of it, at 2.4–15 % of the space: a pruner, not a chooser.
The version closes on these results; the load term is a design for a later
one, to be tested on a fresh sweep.

- Step 7: documentation, tutorial step 17, version 0.7.0. The manual brought
  to 0.7.0 as implemented: `realize` gains a table of the layout decisions
  (domains, defaults, what each changes), their well-formedness and how a
  plan is chosen, and loses a paragraph that said this version lowers few
  of them; `native` a packed, tiled and split kernel as `dump --stage kernel`
  prints it, the GEMM's best point beside 0.6.0's and the model's pick, and
  what the split at a depth buys at batch one; `optimization` what `u` is for
  and what the layout model cannot see (D62); `cli` checked against `presk
  --help` and each command's (`--seed` added to the synopses of `plan` and
  `autotune`, a section name corrected); `experiments` §7, the statements of
  0.7.0, and a closing section that no longer says RA4 is untested or quotes
  the 94×; `status` closed with L to P, the verification and what the version
  has and has not; the index, `install` and the chapter summaries. Tutorial
  step 17, "A plan for attention": the layout model calibrated on a GEMM,
  then `presk plan`, `presk run --plan` (the same output), `presk autotune
  --space small` twice and `presk rank` on the tutorial's attention, in
  `make tutorial`, which now runs steps 0 to 17. Its quoted outputs were
  compared with a run: they agree but for timings, `presk --help` (the three
  commands of 0.7.0), and step 16's `bf16` and recipe rows, which moved with
  D58's `bf16` `dot` (the recipe 8.49·10⁻³ → 8.66·10⁻³ from the exact step;
  every choice of the solver unchanged). The register-load term is recorded
  as a design in `docs/kernels-roadmap.md`, with the evidence for it (thirty
  points of the values `ein` that the model predicts equal, ordered by their
  loads per vector multiply-add from 6.9–7.3 ms down to 2.5–2.9 ms), what it
  will not fix alone, and how to test it on a sweep it was not designed on.
  Spec questions 63–70 copied to `presk-specs`
  (`presk-spec-questions-0.7.0.md`), 63, 64, 66, 67 and 68 marked as in the
  reference, 65 and 69 given the measurement's numbers. Version 0.7.0; `presk
  --help` lists `plan`, `autotune` and `rank` without their step numbers and
  names layout plans among the compiler's parts. Found and fixed: `make
  ra4-small` ranked its sweep with `presk rank`'s default entry,
  `SelfAttention.apply`, so its report left M, N and O "not measured";
  `experiments/ra4.sh` now passes its entry. Verification on falicon: `make
  test` (140 tests), `check-examples`, `plan-check`, `ra4-small` (every point
  exact on both executors), `tutorial` (twice), `conformance` and `cargo doc
  -p presk-manual` pass; §5's `presk plan` on the course GPT's attention and
  `presk run --plan` give the default's output byte for byte. The measurement
  itself is step 6's and was not run again.
- Step 6: the measurement (D62). One night on falicon ran four things:
  - two pinned sweeps of the course GPT's attention (B = 1 and 16, `f32`
    and `f64`, 16 324 points each), the GEMM at 1024³ and a BERT-shaped
    sample, four hours apart;
  - a 20-core sweep with its own calibration;
  - statement L's check on the kernel interpreter, in 130 shards on every
    core.

  **L holds**: 17 879 points, none differing. **M fails**: the attention
  `ein`s order 47.9 % and 43.9 % of their distinguished close pairs right,
  37–46 % of them tied. **N fails**: 582–717× at B = 16 and 75–89× at
  B = 1. **O fails**: top-1 regret 2.18–3.01. **P holds**: the best GEMM
  point is 5.49× 0.6.0's, 460 GFLOP/s on ten cores.

  The model is right on lanes and threads, and blind to register tiles,
  loop orders and k-blocks: a load weighs nothing and its traffic starts
  at L2, so every point is predicted compute-bound. Its candidates within
  `1 + u` always hold the best point, at 2–15 % of the space.

  The tools gained what the measurement needed:
  - `autotune --top/--random` (a sample), `--repeat 0 --shard K/N` (the
    check split over processes) and 0.6.0's point beside every space;
  - `rank --check` and the statements decided in the report;
  - `experiments/ra4_check.sh`, `make ra4-check` and `make ra4-20`.

  `rank` counts a tie as not right and takes `presk plan`'s pick (question
  70).
- Step 5: `presk plan`, `presk autotune`, `presk rank` (D61). The entry runs
  once natively, and the first launch of every nest at every shape is
  captured with its input buffers. Each nest is lowered again under every
  point of its space, the thread splits included and checked by condition 8.
  `presk plan` costs the points with the profile's layout model and writes a
  plan. `presk autotune` times every point on the captured buffers and
  checks its output bit-identical to point 0's (the default layout, scalar,
  one worker), natively and with `--interp` on the kernel interpreter. It
  writes a JSON line per point with the prediction's terms. `presk rank`
  applies plan §3.6's rule to two days: close pairs, distinguished pairs,
  rank accuracy overall and on decided pairs, Spearman, top-1 and top-k
  regret, per class, and the decision cost. `experiments/ra4.sh` runs them
  pinned. `make ra4-small` takes 55 s on the tutorial's attention, every
  point exact. At plan §3.4's shapes the space is 3 305 points per type at
  B = 1 and 4 857 at B = 16. The scores nest's full sweep takes 18 s and
  348 s, 57 and 360 times its static choice, where statement N asks 10³.
  Checking every B = 16 point on the interpreter would take about 30 hours,
  so `make ra4` checks B = 1 there by default (`RA4_INTERP`).
- Step 4: the layout cost model and its calibration (D60). `planned_cost`
  reads one worker's part of a kernel: its work, its traffic from L2, L3 and
  memory (footprints from the affine indices against each cache), and its
  register pressure. The machine is read from the cores the process runs on
  (pinned to an X925: 2 MiB of L2, where `cpu0`, an A725, has 512 KiB). The
  model `λ + spawn·[p>1] + max(W/π_g, Q_L2/β_L2, p·Q_L3/β_L3, p·Q_mem/β_mem)`
  is fitted by coordinate descent into the profile's `layout`, with `π` per
  (type, lanes, workers). `presk bench --calibrate FILE --space calib` times
  a GEMM under 96 layouts; `make calibrate-ra4` runs it at three sizes in
  `f32` and `f64`, pinned. On falicon, u = 0.185 over 576 samples, Spearman
  0.980. In those samples the best 1024³ `f32` layout runs at 405 GFLOP/s
  against 77 for 0.6.0's default (a preview; step 6 decides P).
- Step 3: the pack, two softmax passes and a split at a depth (D59). Every
  decision of the reference's table is now lowered, and `E7002` means an
  ill-formed layout. A pack is two launches, a copy of operand 1 to an NN
  panel, then the contraction on it; the softmax may take two passes, each
  exponential computed once. A plan's split divides the loop at a depth over
  its workers, whatever the work: `Kernel::parallel_at` follows the
  reference's definition, the Cranelift kernel is compiled with that loop
  ranged and the enclosing ones in full, and conditions 5 and 8 are checked
  against the machine and the kernel. The default split is 0.6.0's.
  Every point of four nests (1 491, the pack included) is bit-identical to the
  default on both executors, and 1 618 split points natively; one worker and
  many agree at depths 0–2 and at batch 1. The reference's example plan runs
  the course attention with the default's output.
- Step 2: one contraction builder for `dot` and `ein` (D58). The contraction
  nest is recognized once for both, and lowered in the form (row-wise or
  tiled), register tile, k-block, loop order and lanes its plan states; only
  the pack, two softmax passes and thread splits are still refused. The
  default is 0.6.0's kernel: every `f64` and `f32` `dot` text-identical, the
  attention `ein`s less an exact multiplication by one. A `bf16` `dot` whose
  width is not a multiple of eight is tiled `[1, 1]` and accumulates in
  `f32`, 2.4·10⁻² → 2.7·10⁻³ from the exact product at 1024 terms (D56).
  Every point of three nests, 1 142 in all, each its own kernel, is
  bit-identical to the default in `f64`, `f32` and `bf16` on both executors
  (`tests/layout.rs`). Lanes on an NT contraction need the pack, a condition
  added to the reference. The conformance suite asserts its bands on the
  examples in every realization and on the test programs in `f64` and `f32`,
  reporting their `bf16` gaps. `stack_prefix_grad` had crossed the `bf16`
  band on 6 input seeds of 20 with 0.6.0's kernels and passed at seed 0 by
  luck; the user decided.
- Step 1: `Layout`, `Plan` and `--plan` (D57). A realization plan
  (`{ "realization", "profile", "layout" }`, language reference §6.6) is
  read by `--plan @P.json` on `run`, `bench`, `train` and `dump`, exclusive
  with `--realize`, which is now the plan with the default layout. The kernel
  backend carries the plan and reads each nest's layout at its launch: the
  lanes are lowered as stated, every other decision is accepted at the value
  0.6.0's kernel has and refused otherwise, as is any thread split, and a
  layout ill-formed at the nest's shapes (the reference's conditions 3, 4, 7
  and 9) is refused too — all with the new `E7002`, never ignored. The layout
  is part of the kernel cache's key. `dump --stage realize --plan` lists the
  plan per nest; `bench --plan` adds a row for the plan's layout. `make
  plan-check`: a default plan gives byte-identical `dump --stage kernel` on
  ten entries, identical also to `v0.6.0`'s in `f64`, `f32` and `bf16`. The
  backend's `lanes` field became its default layout's. Found on the way: the
  `realize` chapter still said "named PSK-Realize until 0.6.0" and "`--stage
  realize` is an alias of `--stage realize`", both from the rename; fixed.
- Step 0: definitions and bookkeeping (D56). The language reference (draft
  v0.2) gives the layout decisions their domains — contraction form,
  register tile, k-block, loop order, lanes, thread split at a depth, pack,
  softmax passes — with the exactness rule, the plan object, well-formedness
  conditions 7 to 9, the parallel loop at any depth and the tile loops of
  the kernel grammar. Checking that every default reproduces 0.6.0's kernel
  found that it did not as the plan had it: an `ein` is the tile `(1, 1)`
  with one lane, not `(1, 8)`; the row-wise `dot` is taken whenever J is
  not a multiple of eight, and in `bf16` it keeps its partial sums in `bf16`
  (seven times further from the exact product than the tiled form, at 1024
  terms), which the realization does not declare; `embed`'s derivative and
  `sum_to` do the same; and no kernel reads the realization's `accum`. The
  reference, the plan's §3 and the manual's `native` chapter say so. Version
  `0.7.0-dev`; spec questions 63–69; the X925 cores recorded as
  `RA4_CORES`.

Before the plan, on 2026-09-23:

- **PSK-Sched is renamed PSK-Realize**, as the language reference renamed it
  on 2026-09-23: crate `presk-realize`, `presk dump --stage realize`, `make
  realize`, manual chapter `realize`, with `sched` kept as an alias of the
  stage and of the target. The IR name and the CLI only: the realization
  files keep "site" for the precision vocabulary, and the dated records
  (plans, decisions, spec questions, past entries here) keep the old name.
- **PyTorch re-measured on falicon** (D55): one training step of the course
  GPT is 29.96 s in presk `f32` against 1.34 s in PyTorch 2.14 `float32`,
  twenty threads and dropout on both sides, three repetitions — **22×**,
  where the `experiments` chapter quoted 0.4.0's 94× from another machine.
  Fourteen of the twenty-two is the contraction: a 1024³ `f32` product runs
  at 710 GFLOP/s under PyTorch (its own OpenBLAS 0.3.34, oneDNN 3.12 over
  Arm Compute Library 53.2) and at 49.6 under presk. Neither side moved for
  a reason inside the compiler.
- **The PyTorch export's `training` switch** (D54): every `uses Rand`
  function takes `training=False` beside `generator=None`, both passed on
  through every call to a `uses Rand` function; `dropout` drops under the
  switch and is the expectation without it. `train.py` calls a keyed
  objective with `None` for the key and `training=True`, so it drops as
  `presk train` does, with masks PyTorch draws. `make experiment-gpt` runs
  again (it had failed since 0.5.0 step 1), with the same steps, batches and
  threads on both sides.
- The stacked loop over a tuple accumulator is exported as `block((h, k),
  mask)`; it splatted the pattern into two arguments.
- `presk train` prints the median step time, the first step excluded, on
  standard error, and `--json` carries `ms` per step and `step_ms`;
  `train.py` prints the same, takes `--threads` (default: the machine's
  cores) and prints its thread count.

## 0.6.0 — 2026-09-22

Make it pay (`presk-specs/implementation/presk-plan-0.6.0.md`). Two versions
have ended with the same sentence: 0.4.0 showed the exactness discipline
affordable and the speed not there, 0.5.0 showed precision declarable, solved
and verified and the speed still not there — statement D holds by
measurement, statement E fails at 0.96×. A budget that buys nothing is a
guarantee nobody will pay for, and the MLPerf comparison needs one
realization that is both inside a budget *and* faster.

0.5.0 established which three things must become true: arithmetic width must
buy throughput (vector lanes, which D42 showed are **bit-exact** in the shape
the blocked contraction already has), the objective must be time and not
traffic, and the menu must be wide enough for the choice to matter (int8/int4
with per-channel scales). The version also settles whether the condition
number rescues the deterministic bound, and gives the ε curve its quality
axis.

**What it shows.** A narrower type buys time, twice, each time inside a
declared budget measured against the exact denotation. A training step
solved under a 10⁻³ budget on its update is **1.99×** faster than all-`f64`
at course scale (statement J′), once the budget was given the whole step as
its scope — a budget on the loss alone had given 1.13× (J). And `int8`
weights run BERT-large inference at MLPerf's own constants **2.2×** faster
than `f32`, 5.7·10⁻² from the exact logits: inside a budget of 10⁻¹, not
10⁻². Vector lanes are bit-identical to the scalar kernels (H); `int8`
weights stay inside a 10⁻² budget on the corpus GPT's loss (K); a quantised
gap splits into the weights' rounding and the arithmetic, which only an
exact reference can separate; and MLPerf's accuracy rule admits all eight
realizations of the corpus GPT that a budget separates over twelve orders of
magnitude. **What it does not show** is a realization inside a budget and
faster than FP32, MLPerf's reference: the solved step is `f32` in every
function and level with all-`f32` (30.0 against 31.0 s). Lanes buy little on
a matmul where 3× was asked (I, retired: the factor is an instruction set's).
The condition number does not rescue the deterministic bound, so a budget is
discharged by measurement (question 56, option (c)); the quality axis is
flat at this scale; and the 94× against PyTorch is still 0.4.0's, measured
on another machine.

- Step 9: documentation, tutorial step 16, version 0.6.0. The manual brought
  to 0.6.0 as implemented: `status` closed with H, I, J, J′ and K and with
  what the version has and has not; `experiments` §6, the statements of
  0.6.0, and a rewritten "What these experiments do not show"; `cli`,
  `native`, `training`, `serve`, `optimization`, `precision`, `language`,
  `install`, `data` and the index corrected where they still described
  0.3.0 to 0.5.0 (vectorization "not in this version", `solve` minimising
  traffic, `train` without its native flags, per-layer keys before D37, lanes
  said to make every run faster). Tutorial step 16, "A budget on the training
  step": `presk calibrate --step`, `presk solve --step` and `presk train
  --realize` on the tutorial's tower, where `bf16` loses the step to its
  storage rounding, the recipe is 8.5·10⁻³ from the exact step, and the
  solved step's run ends four thousand times closer to the `f64` run than a
  change of seed. `make tutorial` now runs every command of the chapter,
  steps 0 to 16 — step 0's, step 9's and step 15's were outside it — and
  every quoted output was regenerated: step 0 still quoted the `presk --help`
  of 0.2.0, step 12 a calibration from before fusion, step 15 a `traffic`
  line `solve` no longer prints. `presk --help` takes its version from the
  workspace (it said 0.4.0), and two stale help lines are fixed. Spec
  questions 57–62 copied to `presk-specs` (`presk-spec-questions-0.6.0.md`).
  The plan's §5 verification, on falicon: `make test` (120 tests),
  `check-examples`, `conformance`, `precision`, `curve` (its eight training
  runs redone from scratch, byte-identical), `coverage`, `experiment` and
  `tutorial` pass; `quant-table`, `step-table` and `foil` give their tables
  again; J′ re-measured at 2.02×; `--realize f64` byte-identical to 0.4.0's
  on thirty outputs; every golden of 0.1.0–0.5.0 unchanged. Too long to run: one
  course-scale epoch (two steps ran, and `make j-step`) and `make bert` at 24
  blocks (run at two). `make experiment-gpt` fails, and has since 0.5.0 step
  1: the exported `train.py` omits the dropout key the corpus GPT's objective
  takes. Also found, and fixed: the day an envelope records was a constant
  in the source (it is now the day of measurement, in UTC, with a test), and
  `make curve`'s bound column printed the rule's name instead of the bound.
  The 0.5.0 heading of this file said "Unreleased" although `v0.5.0` is
  tagged.
- Step 8: the MLPerf comparison, with numbers (D53). The foil rests on what
  holds, each result stated as what it is. **Both rules on one model**
  (`make foil`, `experiments/foil.sh`): MLPerf's rule — task accuracy within
  99 % or 99.9 % of FP32 over the whole validation set — and a budget, applied
  to the same eight realizations of the corpus GPT, on D50's all-`f64`
  checkpoint and its held-out tenth (461 windows, 118 016 predictions; a new
  entry of `05_gpt.psk`, `accuracy`, scored in `f64` as MLPerf's harness
  scores from outside). The accuracy rule admits all eight, at 99.9 % as at
  99 %, seven predictions being the largest change; the budget separates them
  over twelve orders of magnitude from two windows and attributes every
  quantised gap to the weights. A check takes 0.1 s, a pass 4 s, the exact
  reference 7.5 s once. K restated: on held-out windows `int8` weights are
  1.46·10⁻⁴ from the exact loss, not 3.23·10⁻⁶ — still inside 10⁻² by
  seventy. **BERT-large runs** (`make bert`, `experiments/bert.sh`): inference
  at MLPerf's constants on its initialization and SQuAD-shaped features
  (`experiments/bert_features.py`, a new entry `squad`); the exact reference
  of one feature in 1 h 57 min and 44 GB; on the logits, `int8` weights 5.7·10⁻²
  from it (outside 10⁻²) and 2.2× faster than `f32` at batch 1 and 20, the first type
  narrower than `f32` to buy time on a model; at batch 1 one core does all the
  work. The exact interpreter **shares tuples and records** instead of
  cloning them at every use, and its backend consumes its arguments: four
  full-width blocks 40 → 13.5 GB, byte-identical, every golden unchanged. The
  tiny BERT joins the native conformance suite; `make coverage` runs it. The
  foil's 0.5.0 sentences that overclaimed are withdrawn; question 62, new.
- Step 7b: a budget on the training step, and J again (D52). A budget in a
  `training run` binds **one optimizer step's update**,
  `‖θ̂′ − θ′‖₂ / ‖θ′ − θ‖₂`, against the exact step from the same state.
  `presk calibrate --step` builds the state (a checkpoint plus `--warmup`
  steps in `f64`), records the envelope of the exact step (the `vjp`'s and
  `adamw.step`'s sites), and prints per `--realize` the update gap, the
  **storage floor** (the exact `θ′` merely rounded to its storage type), the
  gradient gap and the sampling noise between two batches; `recipe` is mixed
  precision as practitioners use it, and `--reference f64` measures where the
  exact step is not affordable. `presk solve --step` measures every function
  of the step alone in `f32` and `bf16`, narrows greedily by predicted time
  per unit of gap under a root-sum-square, verifies and widens; a
  realization gains a `functions` map (a type per function). At `D = 64`: the
  `f32` update is 2.2·10⁻⁴ from the exact one while its gradient is 6.4·10⁻⁷
  — AdamW multiplies the error by 340, so **question 60 is answered with the
  update, not the gradient**; the recipe is 2.9·10⁻², outside 10⁻²; the
  solver takes `f32` for all sixteen functions and `bf16` nowhere, the profile
  predicting it slower. **J′ holds at course scale: 1.99×** per step (`f64`
  59.7 s, solved 30.0 s, all-`f32` 31.0 s, recipe 34.6 s), the solved step
  6.7·10⁻⁵ from `f64` there and 2.2·10⁻⁴ from the exact step at `D = 64`.
  Trained from scratch for 600 steps under the solved step, the GPT ends
  2.9·10⁻⁴ from all-`f64`'s validation loss and 5.0·10⁻³ from its weights,
  where two seeds end 1.22 apart. The storage floor grows as the learning rate falls,
  so a relative update budget is stated at a state (question 61, new). New
  `presk data diff`; `make step-table`, `make j-step`, `make step-train`.
- Step 6: the condition number, and the answer about the bound (D51).
  `presk calibrate` records at every sum and contraction `Σ|xᵢwᵢ|`,
  `√Σ(xᵢwᵢ)²` and each operand's normwise factor (by re-evaluating it on
  `|operands|`, `operands²` and ones), prints the worst condition numbers, and
  takes `--weights`; the envelope records its weights and data slice. The
  propagation has four rules for sums — `k` (0.5.0's), `normwise` (sound),
  `cond` (the condition number, now the default, sound in every case tested)
  and `rss` (an estimate) — chosen by `presk check --precision --model`, and
  `--measure` prints the gap measured natively on the envelope's inputs
  beside each. Median bound/gap on the conformance entries 1.8·10³ → 18; on
  the trained corpus GPT 10³³ → 10¹⁰ in `f32`: **the condition number does
  not rescue the bound**. The root-sum-square estimate is 3.7–10× the
  measured gap in `f64`/`f32`/`bf16`, 2.4·10³× on a mixed realization.
  Recommended to the specification: question 56 (c), a budget is discharged
  by measurement; question 58 yes, a statistical model to propose and not to
  discharge. `presk solve` runs on the envelope's data slice; `make
  precision` measures the GPT on trained weights and text.
- Step 7: the quality axis, and J at course scale (D50). `experiments/
  curve.sh` gains a validation-loss column: each of the seven solved
  realizations trains the corpus GPT (`D = 64`) from scratch, 600 steps on
  90 % of the Code civil, evaluated in `f64` on the other 10 %. **The axis is
  flat**: every run within 1.2·10⁻³ of all-`f64` (3.0263), no trend, while
  changing the seed moves it by 0.13 — no knee in quality at this scale. The
  solver columns are re-measured on trained weights at `T = 256` (D48): all
  inside, 3.22× to 1.20× less traffic. **J fails at course scale: 1.13×** per
  training step (`f64` 56.9 s, solved 50.2 s), while all-`f32` is 1.87×; the
  solved realization covers the loss's sites, and the reverse sweep and the
  optimizer, most of a step, stay at its `f64` default — question 60. The
  sensitivity ranking re-measured on trained weights **reverses 0.5.0's
  finding**: per element the token embedding is the most sensitive
  parameter and the head the second, twenty times the down-projection, as
  the folklore says. Statement D re-measured: 4.48·10⁻⁶ against a 10⁻³
  budget. New `presk run --weights` and `--from`; the curve script reads
  `solve`'s `cost` line again.
- Step 5d: which weights to quantise, and statement K (D49). `presk solve
  --quant [--scheme pow2]` offers every matrix parameter — by *declared*
  rank, so a stacked bias is not one — as `int8` or `int4` on top of the
  float assignment, priced by the gradient the reverse sweep already
  computes and accepted by measurement. The signed first-order prediction
  `Σ ∂L/∂w·(ŵ − w)` came within ten per cent of the measured gap where the
  bound `Σ|g|·|δ|` was 230× too large and quantised nothing; the proposal
  uses the root-sum-square. At 10⁻³ it quantises all nine matrices (five
  `int8`, four `int4`, 11× fewer weight bytes), measured 3.11·10⁻³. `presk
  quantize` gained `--data`, `--list`, and float realizations;
  `make quant-table` and `make gpt-trained`. **Statement K holds**: `int8`
  weights 3.23·10⁻⁶ from the exact loss on text, against a 10⁻² budget;
  `int4` 8.32·10⁻⁵, `bf16` everywhere 4.27·10⁻⁴.
- **A checkpoint of other dimensions was read from shifted offsets (D48).**
  A `params` file records no shapes, and the reader only checked it was not
  too short: the corpus GPT trained at `T = 256` and opened at `T = 32` —
  which every "trained weights" measurement since 0.5.0 step 5 did — read
  every leaf after `pos` from the wrong place, and `solve` ignored load
  errors. Now refused (`E9005`). 0.5.0's statement D number, its sensitivity
  ranking (and the finding against the folklore), the ε curve and 0.6.0 step
  4's assignments were measured on those weights; they are to be redone on
  properly trained ones in step 7. `make precision` and the curve now run at
  `T = 256`.
- Step 5c: `int4` kernels (D47). Both executors load a nibble as the
  packing stores it; the JIT reads four lanes as one 32-bit word, shifted by
  the first element's parity, each lane's nibble moved to the top of its
  `i32` by a per-lane multiplication and shifted back arithmetically. `int4`
  buffers carry four bytes of padding for that word, and a layer's slice
  stays quantised when it starts on a byte. The bit-identity test covers
  `int4` in both schemes, and a program of odd widths (`tests/quant/odd.psk`)
  makes half its rows start mid-byte. Measured: the 16 MB contraction 52.9 →
  10.1 ms (5.2×, level with `int8`); the GPT forward level at `D = 256`,
  0.82× at `D = 64`.
- Step 5b: integer kernels (D45). Integer weights cross the runtime seam as
  `Buf::Quant` — integers and scales, shared rather than copied — and stay
  integers through a `stack{L}` fold's slicing. The lowering is unchanged:
  the host marks a kernel input that receives them (`BufKind::Quant`, part of
  the cache key), and both executors load a marked input as `f32(q)·s`, the
  JIT in four lanes at once (four bytes, sign-extended twice, converted,
  times four scales). Running on the integers is **bit-identical** to running
  on their dequantised values, on both executors, scalar and in lanes, in
  `f32` and `f64`, which a test asserts; and with `pow2` scales, applying the
  scale after the sum is bit-identical to applying it on load, which another
  asserts. Measured (lanes on): a 64 × 1024 × 4096 contraction 47.6 → 10.7 ms
  (4.4×), the GPT forward at `D = 256` 28.6 → 26.0 ms (1.10×), at `D = 64`
  unchanged. `int4` is still read as values (5c).
- **`presk bench` timed the loading of its inputs (D46).** Its inputs come
  from the reference interpreter as exact reals, and every native row rounded
  them to its machine type on every call, inside the timing; now each row
  rounds (and quantises) once, outside it. On this machine that conversion
  was most of a 1024³ matmul — `f64` 183 → 68 ms — so every `bench` figure
  before this entry carries a constant on both sides and its ratios are
  diluted, statements I and J included. Calibrated profiles are unaffected:
  they time kernels, and the conversion was outside every kernel.
- Step 5a: integer weights, the reference path (D44). A realization gains a
  `quant` map: parameters by path (`blocks.attn.wq.w`), each with `bits` (8
  or 4), the axis it is contracted over (`over`, a name or a position from
  the end, since module parameters are written without axis names), and a
  `scheme` (`absmax`, or `pow2` for exact dequantisation). Scales are
  computed from the weights at load and never written into the realization
  (question 57). `run`, `run --data` and `serve` run on the dequantised
  parameters, `train` refuses a quantised realization, and a misspelt path is
  an error rather than a no-op. New `presk quantize` splits the gap of the
  output into its **representation** part (exact on `ŵ` against exact on `w`)
  and its **arithmetic** part (the machine run on `ŵ` against exact on `ŵ`),
  which only an exact reference can separate. On the corpus GPT at `D = 64`
  after 12 steps, blocks and head quantised: `int8` 2.52·10⁻⁶ relative
  (arithmetic 1.81·10⁻⁷), `int4` 1.06·10⁻⁴ (arithmetic 4.4·10⁻⁸), `pow2`
  `int8` 4.40·10⁻⁶ — rounding the weights dominates by one to three orders.
  No integer kernels yet (5b): the numbers are right and the bytes are not
  saved at run time.
- Vector lanes are **on by default** on the native backend, because they are
  bit-exact: no accuracy to trade, no realization to declare, no gap to
  measure, and a transformation that is free and exact has no reason to be
  opt-in. It does not close statement J — it raises both sides, and the
  solved realization is still 1.06× — but on the development machine it
  made every native run faster for nothing (not everywhere: on falicon,
  BERT-large at batch 20 is 15 % slower in lanes in `f32`, step 8). Turning it on exposed a bug: the lane width was taken from the
  *realization's default*, so a mixed assignment sent a two-lane request to a
  `bf16` kernel that cannot take one. The width belongs to each kernel's own
  type.
- Step 4: time as the solver's objective. `presk solve --profile` prices each
  site with the roofline's own prediction, the dtype in `Q` and not in `W`,
  and `W` from the site's elements and the weight of its primitive, which the
  envelope now records. Under traffic the solver reported 2.67× less; under
  time it reports 1.19× faster and the **relaxation floor rises above the
  chosen cost** — narrowing everything would be slower, which the traffic
  objective could not express. **Statement J fails at 1.06×** against 1.3×,
  but the sign has turned: 0.5.0's traffic-solved assignment was 0.96×,
  *slower* than doing nothing. Conversion costs at kernel edges are **not**
  in the model, and the solved realization is not run with lanes; the uniform
  rows show both are worth more than the assignment is.
- Step 3: the machine profile fitted per realization. One profile now holds
  the constants of every machine type and lane width on that machine, because
  a narrow type's `π` is not a wide type's — which is the whole reason lanes
  exist — and a cost model meant to compare types cannot have one `π`. Files
  written before this step still load and still mean what they said.
  Measured on the corpus GPT: `f64` π 6.36e9, `f32` 8.03e9, **`f32` with four
  lanes 1.59e10**, two and a half times the `f64` figure. A bug of step 1 was
  caught by this: the cost model counted a vector statement as *one*
  element's work, which made the fitted π of a vector kernel come out below
  its scalar form's — the opposite of what the hardware does.
- Step 2: Cranelift vector emission. The lane count is the machine type's —
  four `f32` or two `f64`, since Cranelift's vector types are 128 bits — and
  anything else is refused rather than quietly rounded. **Statement H holds
  on both paths**, asserted as an equality: the emitted vector kernel is
  bit-identical to the emitted scalar one. **Statement I fails**: on a 1024³
  matmul the lanes give 1.25× in `f64` (104.1 → 83.3 ms) and 1.15× in `f32`
  (87.2 → 75.8), where it asked for 3×. The gain is real and free, the same
  bits faster, but the blocked form already ran eight independent accumulator
  chains so most of the parallelism was there. **Packed panels are not
  written** and are the first thing to try before concluding anything about
  the ceiling.
- Step 1: the vector form in PSK-Kernel. Vector variables of `lanes` reals,
  a contiguous load, a broadcast and lane-wise arithmetic, executed by the
  kernel interpreter; the blocked contraction emits them when asked.
  **Statement H holds on the interpreter**: a vector kernel is
  *bit-identical* to the scalar one at two and four lanes in `f64` and `f32`,
  which a test asserts as an equality rather than a tolerance. A lane is a
  different output column, never a partial sum of one, so every output still
  accumulates in increasing order — a layout change, not a reassociation, and
  it owes no declared order and no gap. The Cranelift backend refuses a
  vector kernel with a message; emitting them is step 2.
- Step 0: version `0.6.0-dev`; **statement G retired in favour of H** (D43),
  because G asked the wrong question — the kernel it describes is exact, so
  the claim to make is bit-identity, which is stronger than any tolerance;
  spec questions 57 to 59.

## 0.5.0 — 2026-09-20

Precision as a resource
(`presk-specs/implementation/presk-plan-0.5.0.md`): RA3 of the research
programme, which the two previous plans deferred for one stated reason — the
solver could not be evaluated on programs that take minutes a step — and
which 0.4.0 made reachable by getting a course-scale step to 39.8 s. The
version turns `--realize` from a two-valued flag into a solved object
carrying a storage and an accumulator type per fusion group and a declared
reduction order, adds a measured magnitude envelope, a first-order error
propagation that exploits known conditioning, a sensitivity analysis by the
existing differentiation, and the solver that turns `budget(rel_error = ε)`
into an assignment minimising predicted time. One vectorized GEMM ships under
that contract, the first item of `docs/kernels-roadmap.md`, because without
vector lanes precision buys no measurable time and without the contract the
vector kernel reassociates a sum it is not allowed to reassociate. The
primary deliverable is the ε-to-quality curve the roadmap asks for.

**What it shows.** A budget declared in a program becomes an assignment whose
distance from the exact denotation is *measured*, not asserted: 1.05e-7
against a budget of 4.52e-3 on the corpus GPT (statement D). A sweep of that
budget is a clean one-parameter curve, four times less traffic at 1e-1 and
1.69× at 1e-7, every point verified. **What it does not show** is that any of
it is faster: the solved realization runs at 0.96× the speed of all-`f64`
(statement E), because the solver minimises traffic and these kernels are
bound by arithmetic. **The vectorized kernel is deferred to 0.6.0**, as 0.4.0
deferred vectorization before it and for a reason now better understood:
vectorizing across output columns is bit-exact (D42) and needs a vector form
in PSK-Kernel, which this version did not write. Statement G is about that
kernel, is not askable, and needs restating.

- Step 9: the four statements answered by measurement, and the MLPerf
  comparison. **D holds** — a declared budget becomes an assignment whose
  measured distance from the exact denotation is 1.05e-7 against 4.52e-3,
  and no other compiler can run that check. **E fails**: the solved
  realization is 5.63 ms against all-`f64`'s 5.39, **0.96×**, because the
  solver minimises traffic and these kernels are not traffic-bound —
  optimising the right proxy in the wrong regime. **F fails narrowly**, 10 of
  21 entries within 1e3, median 1.8e3. **G is not askable**: it is about a
  vectorized GEMM that does not exist and, by D42, should be built under a
  sequential order where it would have no gap to measure. So the claim that
  precision can be declared and verified is demonstrated; the claim that it
  pays is not.
- Step 8 (ε axis done): `make curve` sweeps the budget on the corpus GPT —
  one program, one parameter varying, which is the sweep the specification
  says cannot be done cleanly elsewhere. **Every point is inside its budget
  by measurement**, and traffic rises monotonically as the budget tightens:
  4.00× less than all-`f64` at 1e-1, 2.67× at 1e-3, 1.69× at 1e-7, with a
  knee between 1e-2 and 1e-3. The measured gap is not monotone, and the
  propagated bound at these constants is 7.15e11 — vacuous, which is why the
  measured columns are the result. **The quality axis is missing**: a
  validation loss per point needs a training run per point, which this
  version has not been able to afford.
- Step 7 (partly landed): `Stmt::SetL` assigns a loop variable in PSK-Kernel
  and both executors, and on it the interleaved contraction is emitted; the
  declared order is part of the kernel cache key and reaches the kernel,
  which a test asserts — the MLP at `D = 32` in `f32` gives 1.5494239330292
  sequentially and 1.5494240522385 with eight partials, both executors
  agreeing to the last digit. `presk bench --native-only` makes a large
  benchmark affordable. **The measurement is a negative result and a useful
  one**: on a 1024³ matmul the reassociating order is **2.7× slower** (239.0
  ms against 87.3), because it cannot use the blocked form and falls back to
  the naive contraction. For a CPU contraction there is no reason to
  reassociate: the exact form is faster *and* exact. The order stays
  implemented and measured for the coprocessor and the GPU. **The finding is worth more than the code**
  (D42): the blocked form's eight accumulators are eight different output
  *columns*, each still summing in order, so putting them in vector lanes is
  **bit-exact**. Vectorization does not need a new realization after all,
  which is a better position than D31 assumed; the contract is still needed
  for the coprocessor and the GPU, where a reduction crosses lanes.
- Step 6: the solver. `presk solve --precision ENV.json --budget ε [--weights
  ckpt] --out R.json` turns a budget into a realization, which `--realize
  @R.json` then runs. Propose greedily by bytes saved per unit of
  sensitivity-weighted error, **verify by measurement against the exact
  reference** rather than by the propagated bound (D41) — the bound is
  vacuous on a transformer, and measuring is the stronger check and one only
  this compiler can make — then widen the worst site one step at a time. The
  objective is traffic, not time, because `bf16` is slower than `f32` on a
  compute-bound kernel. On the corpus GPT at `D = 64`, trained, budget 1e-3:
  **2.67× less traffic**, zero widening rounds, measured gap 1.05e-7 against
  a budget of 4.5e-3. On a small program the greedy proposal is exactly
  optimal at 1e-3 and 1.057× off at 1e-6.
- Step 5: sensitivity by differentiation. `presk dump --stage sensitivity`
  prints `‖∂L/∂v‖₁` at every site and per module, for one reverse sweep: the
  adjoints already exist, so the step keeps the correspondence between a
  forward value and its adjoint rather than differentiating again (D40). Two
  findings. A ranking at initialization is vacuous, because a zero-
  initialized head stops every upstream gradient, so `--weights` is needed.
  And half the mixed-precision folklore holds: the output head dominates at
  5.86 against 2.46 for the whole block stack, while the token embedding
  table scores 0.28 — below the down-projection practice narrows freely — so
  keeping embeddings wide is not explained by sensitivity.
- Step 4: first-order error propagation over PSK-Core.
  `presk check --precision ENVELOPE.json [--sites N]` bounds an entry's
  distance from its denotation under a realization. Quotients, roots and
  logarithms propagate in *relative* terms, which is why a normaliser costs
  nothing; a residual adds; a softmax is 1-Lipschitz in the ∞ norm; and a
  contraction's constant is a function of the declared reduction order, so
  blocking leaves a bound untouched where interleaved partials change it.
  **Sound and, on a real model, vacuous** (D39): the bound exceeds the
  measured gap on every conformance entry in every realization, median ratio
  1.8e3 and 27 on a normaliser, but 1e20 on the corpus GPT — a contraction of
  `k` terms multiplies error by `k` where committed errors grow like `√k`.
  The tightening is named: each sum's condition number, measured. New
  `presk-precision::propagate`, new tests for soundness and for the reduction
  order's effect, `make precision` now bounds as well as calibrates.
- Step 3: the magnitude envelope. `presk calibrate --precision` records the
  largest `|x|` at each site on the reference interpreter, since an absolute
  error means nothing without a scale and a relative one is unbounded near a
  zero. Static intervals cannot supply that scale for a network, so it is
  measured and every bound quoting it quotes its sample count (D35). The
  step's content is the check that the envelope is a property of the program
  rather than of the sample: on the corpus GPT, disjoint slices of the civil
  code agree within **1.241×** and far-apart ones within **1.128×**, over 77
  compared sites with none present in only one. New manual chapter
  `precision`, new target `make precision`.
- Step 2: the realization becomes an object. `Realize`, a flag with two
  values, is now `Realization`: a storage and an accumulator type per site, a
  declared reduction order, and the magnitude envelope any bound assumed
  (D34). `--realize` takes `f64`, `f32`, `bf16` or `@FILE`. A site is a
  fusion group's root, and a buffer crossing between sites of different types
  is converted at the kernel's edge. `bf16` is storage only — a shift in, one
  round to nearest even out, written once in Rust and again as Cranelift
  integer instructions, with the two executors agreeing digit for digit on a
  gradient (D38). The reduction order is the field D31 and D32 both needed
  and had nowhere to write down: `blocked` is bit-identical to `sequential`,
  `interleaved` reassociates. **Nothing moved**: every `f64` and `f32` entry
  of the conformance table is what 0.4.0 measured, to the last digit, and the
  suite now reports a third column, `bf16`, worst 3.3e-2.
- Step 1: the axis `split`, the declared inverse of `merge` — one axis read
  as two, row-major, a pure retag whose adjoint is `reshape_like`, checked
  against finite differences at 512 bits and identical on all three backends.
  Unlike `merge` it names both halves, so **splits chain where merges cannot**
  (D36), and that is what lets `coverage/llama.psk` transcribe grouped-query
  attention exactly — with neither `split_heads` nor `merge_heads`, and with
  no transpose, because `ein` contracts by name. Per-layer randomness inside
  a `stack{L}` fold needed nothing new: the key rides in the accumulator,
  which is already threaded, and the index route was dropped after
  establishing that AD refuses to differentiate an indexed fold (D37). The
  corpus GPT gains the course's dropout at its three sites. Two corrections
  to 0.4.0's account: rotary positions were never blocked on the split but on
  `concat`, a way to build a tensor along an axis, and question 51's
  provisional answer was wrong.
- Step 0: version `0.5.0-dev`; crate `presk-precision` scaffolded, with the
  two quantities it must not confuse written into its own documentation — a
  bound guarantees and a sensitivity ranks; decisions D34 (the realization is
  an object, not a flag) and D35 (a bound is conditional on a measured
  envelope, and says so); spec questions 50 to 55.

## 0.4.0 — 2026-09-20

Real models, real speed
(`presk-specs/implementation/presk-plan-0.4.0.md`): a corpus of models from
an NLP course as the forcing function, then the scheduling and backend work
it forces. One training step of the course's GPT went from 289 s to 39.8 s,
seven times faster, **without changing a single printed digit of the loss** —
every transformation in this version is exact over ℝ and reassociates
nothing. Measured against PyTorch the tool is still 94× slower a step, and
the `experiments` chapter says where that goes: about twenty-six of it is
PyTorch reaching a matrix coprocessor that presk does not emit for, and the
reason it does not is that both routes there reassociate a sum, which in this
system is a different realization owing a declared order and a measured gap.
Steps in reverse order:

- Step 1 (in progress): `corpus/nlp/`, the ladder of the NLP course on the
  words and text of the Code civil (`data/prepare.py`, `make corpus`,
  `EXPECTED.md`). Rung 1 `01_bigram_count.psk`: the counting bigram in closed
  form, as one `ein` over two `one_hot` tensors — 2.343771504 on all 67652
  pairs, identical on the reference interpreter and the native backend, and
  equal to the NumPy reference. Rung 2 `02_bigram_neural.psk`: the same
  function fitted by AdamW, 3960 steps in 3.6 s natively, batch losses
  straddling the closed-form value, which is the course's own sanity check.
- Step 10: version 0.4.0. Tutorial step 13 "A model that is not a toy" walks
  the corpus — text to dataset, the counting bigram in closed form on 67 652
  real pairs, the same model fitted, the MLP that needed `merge`, and the
  course GPT checking at its own constants — with every quoted output
  produced by `make tutorial`. New manual chapters `scheduling`, `corpus` and
  `coverage`; `status` closed with the verification list and with what this
  version does not have; README and the version itself. Spec questions 45–49
  copied to `presk-specs/implementation/presk-spec-questions-0.4.0.md`, and
  the plan carries a status note listing every deviation. WaveNet and dilated
  convolution are deferred with their reason: step 9 established convolution
  is a type-system extension, not a primitive.
- Step 9: coverage of the shape fragment, measured on the MLPerf
  architectures — the roadmap's own metric for the type system, and the only
  thing this project takes from MLCommons (D27). `coverage/bert.psk` is
  BERT-large as the suite fixes it, 24 post-normalized blocks with the three
  embedding tables, a padding mask that broadcasts by name and the SQuAD span
  head: it checks whole. `coverage/llama.psk` is the decoder Llama, GPT-J and
  Mixtral share, with `RMSNorm`, SwiGLU and `silu` written from `sigmoid`: it
  checks with two substitutions, rotary positions and grouped-query
  attention, both blocked on the same missing construct — splitting one axis
  into two, whose three workarounds are closed (`split` unresolved, `concat`
  not implemented, range slicing not syntax). Of seven model families, two
  are expressible and they are the transformer families; convolution blocks
  four benchmarks, segmented reductions two, top-*k* with a runtime parameter
  gather one, and non-maximum suppression is outside a shape language
  altogether. Manual chapter `coverage`, `make coverage`.
- Step 8: the decisive experiment, answered by measurement. The course GPT
  (`D=384`, 6 heads, 6 blocks, context 256, batch 16, character-level Code
  civil, `f32`, twelve cores) trained by `presk train` and by the `train.py`
  the export writes. Of the three statements this version set: **A fails**
  (94× PyTorch a step, 82× with `--opt`, against the 3× asked), **B fails**
  (`--opt --profile` is worth 1.14×, against 1.30× asked), **C holds** (the
  profile predicted 1.087× where 1.14× was measured, 5 % off against the
  30 % band). One step is 266 GFLOP: PyTorch 0.405 s at 656 GFLOP/s, presk
  37.9 s at 7.0.
  The gap is diagnosed rather than lamented: PyTorch's CPU matmul reaches
  Apple's matrix coprocessor through the platform BLAS at 1309–1454 GFLOP/s
  where presk's scalar blocked kernel does 47–56, a factor of twenty-six that
  is instruction-set access and not compiler cleverness; the remaining three
  is presk's own non-contraction work. Recorded in the `experiments` chapter
  with what it says about the thesis: this version bought 7.3× and every
  transformation it shipped is exact — the losses are identical to every
  digit printed — so the discipline is affordable, and the next order of
  magnitude needs a new declared realization rather than a faster loop
  (D31, D32).
- Step 7: a rewrite the cost model arbitrates. The rule this version's plan
  named, reassociating `(q·k)·v`, has nowhere to fire: counted on the corpus
  GPT after differentiation, no contraction takes another contraction as an
  operand, because a transformer puts a softmax, a GELU or a normalization
  between every pair. The rule that does fire moves a scalar across a
  contraction — `(Σ aᵢbᵢ)/s = Σ(aᵢ/s)bᵢ`, exact over ℝ — and the e-graph
  proposes scaling every operand so the cost can choose. Attention divides
  6 291 456 scores by `√hd` where it could divide 1 572 864 queries; the
  calibrated roofline prices the two forms apart (`SelfAttention.apply`
  10 205 µs against 9 489 µs) while operation counts price them the same.
  Measured over forty training steps of the corpus GPT: 0.710 s a step
  without rewriting, 0.626 with counts, 0.617 with the profile, all three
  giving identical losses. The profile's predictions rank far better than
  counts (Spearman 0.96 against 0.59) but its decisions barely do, and its
  extraction costs 2.54 s against 0.41 s, so it takes some 230 steps to repay
  itself — a cost model can rank kernels well and still not change which
  program is extracted. D33.
- Step 6: contractions in a blocked form. `dot` and `matmul` no longer walk a
  column of the right operand per output element: eight columns of the result
  are held in scalar accumulators across the whole contraction, and the inner
  loop loads one element of the left operand and a contiguous run of eight
  from the right, storing nothing. A width that is not a multiple of eight
  takes a row-wise form that sweeps a whole row and handles any width. Both
  are **exact** — each output still accumulates `k` in increasing order, only
  the memory the loop touches changes — and a test asserts that the two forms
  agree bit for bit in `f64` and `f32`, with the conformance gaps unchanged to
  every digit. A 1024³ `f32` matmul goes from 3.7 to 9.9 GFLOP/s on one thread
  and reaches 47–56 GFLOP/s on twelve, about two thirds of this machine's
  scalar peak. **One training step of the corpus GPT at the course's own
  constants now takes 39.8 s, against 289 s before this version's scheduling
  and backend work — seven times faster, with the loss unchanged to the last
  digit.** D32.
- Step 5: threads. A kernel whose outermost loop is parallel — the body is
  one loop nest and every store indexes the output with that loop's variable,
  so the iterations write disjoint elements — is split across workers by the
  Cranelift backend; a reduction over the outer axis and a scatter such as
  `embed_vjp` fail that test and stay sequential. Compiled kernels now take
  the half-open range of their outermost loop, so one function serves the
  sequential call and every worker. Only kernels above four million weighted
  operations are split, since spawning costs tens of microseconds.
  `--threads N` on `run`, `train` and `bench`, defaulting to the machine's
  cores; `PRESK_THREADS` does the same for a library caller.
  The split is *exact*: each output element is computed by the same inner
  loops in the same order, so one worker and twelve give bit-identical
  buffers in `f64` and `f32`, asserted element by element. Measured on twelve
  cores, one training step of the corpus GPT: 4.00 s → 0.99 s at `D=64`, and
  **289 s → 58.3 s at the course's own constants**, a five-fold gain for an
  unchanged loss. Vectorization is not in this version, and the reason is
  recorded: the kernels that would vectorize exactly are not the bottleneck,
  and vectorizing a contraction's inner sum reassociates it, which is a new
  realization needing its own declared order and measured gap. D31.
- Step 4: PSK-Sched v0, fusion groups. New crate `presk-sched` partitions
  each PSK-Core body into connected sets of elementwise nodes of one result
  type whose interior values are read only inside the set; PSK-Kernel lowers
  a group to a single loop nest, so the intermediates stay in registers and
  only the result is written. A group whose type is a parameter tree is
  mapped over the leaves, one launch per leaf for the whole chain, which is
  where most of the saving is. A value read twice inside a group is computed
  twice rather than stored, bounded at sixty-four expression nodes. The
  partition is structural, so it is computed once per function and serves
  every shape, realization and backend. `presk dump --stage sched`; the
  safety invariants are tested on the corpus and the examples.
  A whole training step of the corpus GPT goes from 1515 launches to 884,
  42 % removed (`presk train` reports it); the plan asked for half, and the
  gap is the forward and backward passes, whose chains reductions cut.
  On the corpus GPT the optimizer step goes from 414 launches to 111 and,
  on the tiny MiniGPT, from 1.54 ms to 1.32 ms — about 15 %, outside the
  ±4 % spread, and the first time a pass in this project has moved the clock
  outside the noise. The saving is traffic, not launch overhead: 303 launches
  at the measured λ of 50 ns would be 1 % of that time. The forward pass
  loses a quarter of its launches and no time, being contraction-bound, and
  every golden output is byte-identical since fusion reassociates nothing.
  Not done, and recorded: multi-output kernels, reduction fusion, and an
  extraction that knows about the partition (question 49). Manual chapter
  `scheduling`; D30.
- Step 3: `dropout`, with its key in the signature. `dropout(rate, key, x)`
  multiplies `x` by an inverted-dropout mask drawn from `key` and nothing
  else, and `split_key(key) → (key, key)` gives independent keys so a
  function can drop more than once. The key is explicit for a correctness
  reason, not a stylistic one: differentiation recomputes the forward pass,
  so a mask drawn from a hidden stream would differ between the forward and
  the backward and the gradient would be silently wrong. Being a function of
  the key, the mask recomputes identically, and `dropout` needs no
  differentiation rule of its own — it lowers to `x · dropout_mask(rate,
  key, x)`, whose adjoint is the adjoint of a product. Checked against
  central finite differences at 512 bits *while dropping*
  (`tests/data/dropout_grad.psk`), with the same key giving the same mask and
  another key another one.
  An objective may take a per-step key between `θ` and the batch (`fn loss(θ
  : params M, key : key, batch : …)`), which `presk train` derives from the
  run's seed as the step's child key (`Key::child`, constant time);
  `tests/train/dropout_reg.psk` trains reproducibly from its seed. Outside a
  training run the mask is all ones, the expectation of the random function,
  so `run`, `serve` and the export evaluate the model's mean: the seam gains
  `Backend::set_training`, and the export emits `F.dropout(x, p,
  training=False)` because a Presk key cannot drive PyTorch's generator.
  Answers spec questions 33 and 48, except for one part recorded there: a
  `stack{L}` fold passes the same extras to every iteration, so the blocks of
  a stacked module share one key.
- Step 2: text at scale. `presk data chars` builds a character vocabulary
  from a UTF-8 text (every distinct character, sorted, as a JSON array of
  tokens) and `presk data encode` turns a text into the token stream
  `lm-windows` cuts into windows; a character the vocabulary does not hold is
  named with its code point, and a sub-word vocabulary is refused rather than
  guessed at (`presk_data::{char_vocab, encode, decode_text}`, round-trip
  test on accents, apostrophes and newlines). The Code civil becomes 1182121
  tokens over 92 characters and 4617 non-overlapping windows of 256 in a
  2.4 MB file. Corpus rung 5 `05_gpt.psk`, the course GPT, passes `presk
  check` at the course's own constants (`D=384, H=6, L=6, T=256`) with the
  data binding `T`; a training step takes 289 s at those constants (1.87 GB
  resident, about 0.9 GFLOP/s) and 4.2 s reduced to `D=64, H=2, L=2`, which
  is the wall steps 4 to 6 exist to break. Resident memory is identical at 3 and at 12 steps, because
  the reader already seeks to each sample as the batch asks for it: the
  streaming the plan asked for was already there, and only overlapping
  windows would need lazy windowing (`make corpus-gpt`). `presk data bpe` is
  deferred: the course's GPT is character-level, so nothing in the corpus
  needs sub-word tokens yet.
- New builtin `merge(axis a, axis b, x)`, the answer to spec question 45: two
  adjacent axes become one of length `|a|·|b|`, row-major. Adjacency is
  required (joining distant axes would also transpose) and at least one length
  must be constant (`E1007` otherwise, two `tests/diag/shape_merge_*.psk`
  fixtures); the joined axis is unnamed and is named by the signature it is
  passed to. Since row-major buffers make it metadata-only, it launches no
  kernel on any backend and costs nothing in the model; its adjoint is the
  retag back (`reshape_like`), checked against central finite differences at
  512 bits (`tests/data/merge_grad.psk`). Exported as `.flatten(k, k+1)`, with
  the conformance test passing.
- `embed` now carries the name of its table's second axis into the result
  (`param tok : ℝ[vocab: V, emb: E]` gives `ℝ[…, emb: E]`), so the new axis
  can be referred to by name; an unnamed table column still gives an unnamed
  axis, so nothing that existed changes.
- Corpus rung 3 `03_mlp.psk`: the embedding MLP of the course (Bengio 2003),
  three characters of context through `merge`, 2112 AdamW steps in 35 s
  natively, loss 3.71 → 1.73, well below the bigram's 2.34.
- New builtin `one_hot(t : idx[V][S]) → ℝ[S, V]`, the indicator vectors whose
  contraction with a table `embed` denotes: typing, lowering, regularity
  (`Disc`, like every index argument), differentiation (no adjoint flows
  through it), the reference interpreter, a PSK-Kernel v0 kernel, the
  PyTorch export (`F.one_hot`) and the cost model. Identical results on the
  interpreter, the kernel interpreter and the Cranelift backend.
- `presk run --data FILE.psk.d [--samples N]`: the entry's last argument is
  taken from a data file instead of synthesized, with the `batch` axis of
  that argument bound to the number of samples read, so an objective can be
  evaluated on its own dataset (`presk_train::run_on_data`;
  `presk_interp::run::resolve_entry` is now public).

## 0.3.0 — 2026-09-19

Native backend first, then e-graph rewriting
(`presk-specs/implementation/presk-plan-0.3.0.md`): a runtime seam
(`Backend`), PSK-Kernel v0 and its interpreter, a Cranelift JIT behind it
(`--backend native --realize f64|f32`, the tiny MiniGPT forward 27× the
reference, twenty training steps 34 s → 0.4 s), training and serving on it
with their gaps to the exact reference measured, `inference serve` and
`presk serve`, the e-graph optimizer `presk-opt` with rules exact over ℝ and
the interpreter as oracle, and a cost model whose roofline constants are
fitted on measured kernels. Steps in reverse order:

- Step 8: version 0.3.0; tutorial step 12 "Run it natively, then let the
  compiler rewrite" (`make tutorial` runs its commands: bench, native
  training with the same log as the reference, `--opt` with the same
  gradient, `--calibrate`, `--profile`); README, this changelog, `status`
  closed with the acceptance checks of the plan; spec questions 35–44 copied
  to `presk-specs/implementation/presk-spec-questions-0.3.0.md`; the
  kernel and native backends report their run-time errors as `E7001` (no
  separate `E10xxx` range was needed).
- Step 7: cost model v0.1, calibrated — `presk_kernel::cost` computes the
  work (weighted operation count of the loop nest) and traffic (bytes read
  and written) of every kernel from its IR and `TimedExec` times every
  launch; `presk bench --calibrate FILE.json [--realize f64|f32]` fits the
  roofline `λ + max(W/π, Q/β)` on the measured kernels of an entry (relative
  least squares on a refined log grid), merges with an existing profile and
  writes it with Spearman's rank correlation of three predictors (operation
  counts 0.15, work 0.91, fitted roofline 0.95 on the tiny MiniGPT's three
  entries in `f64`; 0.37 / 0.77 / 0.89 in `f32`); `presk_opt::model` derives
  the same work and traffic for a PSK-Core node from its type, the run's
  `--dim`s bound by name and 64 for the other dimensions; `--opt --profile
  FILE.json` on `run`/`train`/`serve`/`bench`/`export`/`dump --stage
  core-opt` extracts by predicted seconds (the report then prints
  microseconds). The run goldens are unchanged under a profile; the three
  MiniGPT entries time the same with and without the profile, within the
  ±4 % run-to-run spread. `make calibrate`. Manual: `optimization` gains
  "The cost model, calibrated"; D25.
- Step 6: crate `presk-opt` — e-graph rewriting of PSK-Core with `egg`:
  one e-graph per function body, the PSK-Core type of every class as an
  analysis (rules fire only between compatible types, the rewritten program
  is typed), rules exact over ℝ in three families (algebra: commutativity,
  associativity, neutral and absorbing elements, `neg`, `pow`, `exp`/`log`,
  `sqrt`, distributivity; reductions: scalar factors out of `sum`/`mean`;
  structure: `sum_to`/`broadcast_like` identities between equal types,
  `zeros_like` in sums, `field`/`set_field` chains and their sums into a
  single `record`, `proj` of tuples), extraction by operation counts,
  write-back in ANF (dead definitions, such as the forward recomputation in
  adjoints, disappear); `--opt` on `presk run`, `train`, `serve`, `bench`
  and the conformance path of `export`; `presk dump --stage core-opt` with a
  report (functions rewritten, lets and cost before and after, stop
  reason); a function that hits a limit is extracted from the partial
  e-graph and reported (D24). Oracle: the eight `tests/run` goldens and the
  20-step MiniGPT training golden are unchanged under `--opt`, a property
  test compares random PSK-Core expressions before and after rewriting on
  the interpreter at 128 bits (2⁻¹⁰⁰), and the native backend gives the
  same forward, gradient, training and serving outputs with and without
  `--opt` (`make opt-examples`). `Linear.apply.vjp` goes from 14 to 7 lets,
  `mlp.psk` from 89 to 71; no measurable speed change on the tiny shapes.
  `PrimOp` and `Lit` gain `Eq`/`Hash`; `presk dump --stage core|core-after-ad`
  honour `--const`. Manual chapter `optimization`; question 44 (axis names
  in the equality of values).
- Step 5: serving — the `inference serve` block resolved by the checker
  (model, `weights` path, the generation function found by its signature
  `(θ : params M, ctx : idx[V][batch: B, seq: T], …) -> idx[V][batch: B]`,
  `W0003` without one, `--summary`); crate `presk-serve`: the generation
  loop over the `Backend` seam, sampling as written or greedy through a copy
  of the function with `sample` replaced by `argmax`, a sliding window of
  `T` tokens when the parameters fix the context length (`pos : ℝ[T, D]`,
  question 43); `presk serve`; `presk export --serve` writes `serve.py` and a
  `pskd.read_params` that decodes a `params M` checkpoint with the parameter
  tree's layout generated into the script; on the tiny MiniGPT the greedy
  tokens are identical on the reference, natively in `f64` and `f32`, and in
  `serve.py` (`make serve-examples`, `make serve-pt`); manual chapter `serve`.
- Step 4: training natively — `presk train --backend native --realize
  f64|f32` compared with the reference (regression, 200 steps: 4.4e-16 in
  `f64`, 1.6e-7 in `f32`; test in `presk-cranelift`); `presk export
  --conformance-entry train:K` also runs the `K` steps natively and writes
  `conformance/native_{f64,f32}.json`; `experiments/exact_vs_almost.py` and
  the `experiments` chapter measure presk's own realizations next to
  PyTorch's (tiny MiniGPT, 20 steps: 3.7e-7 / 6.7e-16 against 9.6e-7 /
  8.9e-16); `make train-examples` trains natively too; `V=64 D=64 H=4 L=4`
  trains 500 steps in 159 s.
- Step 3: crate `presk-cranelift` — every PSK-Kernel v0 kernel compiled to
  machine code by Cranelift's JIT on first use (loops as blocks, `libm`
  symbols for the transcendental functions), behind the kernel backend's
  host program; `--backend native` on `presk run` and `presk train`; the
  conformance suite passes natively in both realizations; `presk bench`
  times an entry on every backend (tiny MiniGPT forward: 16 ms on the
  reference at 64 bits, 0.59 ms native; 20 training steps 34 s → 0.4 s).
- Step 2: crate `presk-kernel` — PSK-Kernel v0: every tensor primitive of
  PSK-Core lowered to a loop-nest kernel at concrete shapes and one machine
  type (specialized on first use, cached), a textual form (`presk dump
  --stage kernel`), an interpreter of the kernel IR in `f64` or `f32`, and
  the host program evaluating PSK-Core bodies over the kernels (structural
  primitives, initializers and sampling on the host with the reference's
  keys); `--backend kernel` on `presk run` and `presk train`; the conformance
  suite against the interpreter (41 primitive cases, 11 example entries,
  gaps ≤ 2.6e-14 in f64 and ≤ 5.3e-6 in f32).
- Step 1: crate `presk-rt` — runtime values (buffers of exact reals, `f64`,
  `f32` or `i64` with named axes; tuples and records), the `Backend` trait
  (`prepare` an entry at bound dimensions and a realization, `call` it),
  `InterpBackend` (the reference interpreter behind the trait, exact values
  passing through untouched), the parameter-tree I/O moved from
  `presk-train`; `presk run` and `presk train` run over the trait
  (`--backend interp`, the only backend so far); the interpreter's input
  synthesis is exposed as `prepare_run`.
- Step 0: version `0.3.0-dev`; spec questions 35–42; decision D21.

## 0.2.0 — 2026-09-19

Training end to end (`presk-specs/implementation/presk-plan-0.2.0.md`): a
data format, the optimizer as PSK-Core, `stack{L}` as a fold, `presk train`
on the interpreter, the training run and the spec's graph mode in the PyTorch
export, conformance of one optimizer step and of 20 training steps against
PyTorch. Steps in reverse order:

- Step 7: manual chapters `training`, `data` and `experiments` (`make
  experiment`: a silent broadcasting bug of plain PyTorch that the types rule
  out; the gap of float32/float64 training to the exact reference, 3e-7 and
  4e-16 after ten steps), tutorial step 11 (train a model), README, this
  changelog; interpreter errors name the function and
  the binding (`in \`RMSNorm.apply\` (%8): …`); the constant `pi`; an
  unconstrained dimension argument of a call is left unbound instead of
  failing.

- Step 1: the `.psk.d` data container (crate `presk-data`): samples of one
  Presk type, `f64` reals injected exactly, `idx` at the width of their
  bound; `presk data pack | lm-windows | info | export`; the training
  pipeline (`shuffle`, `batch`, `take`) over sample indices; error codes
  `E9xxx`; a standalone type parser in `presk-syntax`.
- Step 6: PyTorch export of the training run: `--train` writes `pskd.py` (a
  `.psk.d` `Dataset`), `train.py` (the run as a PyTorch loop: AdamW with the
  block's hyperparameters, the schedule, clipping, checkpoints) and the data
  file; `--conformance-entry train:K` replays `K` interpreter steps in PyTorch
  on the same batches and compares the parameters (tiny MiniGPT, 20 steps:
  worst relative gap 1.3e-10); `--mode graph` (the spec's tool-level mode,
  light form): `export.py` saves `model.pt2` and `model.onnx`, tested under
  ONNX Runtime; `make venv` adds onnx, onnxscript, onnxruntime.
- Step 5: `presk train` (crate `presk-train`): the training loop on the
  interpreter — initializer, data pipeline, objective and its `vjp`, the
  PSK-Core optimizer, schedule horizon, log (`--json`), checkpoints as
  `params M` files (`final.psk.d`); `--steps`, `--epochs`, `--data`, `--ckpt`,
  `--log-every`; `tests/train/mlp_regression.psk` converges; the tiny MiniGPT's
  20-step log is a golden; `presk run` on a training file exits 2 pointing to
  `presk train` (exit code 3 retired).
- Step 4: `stack{L}` lowered to a `fold` in PSK-Core (program size independent
  of `L`), interpreted as a loop and differentiated as a backward fold over the
  scanned accumulators; new primitive `set_slice`; `presk run|dump
  --unroll-stack` keeps 0.1.0's unrolled form; golden outputs unchanged; the
  gradient through a stacked module checked against finite differences.
- Step 3: the optimizer as PSK-Core (`adamw.init`, `adamw.lr`, `adamw.step`
  over `params M`): AdamW as PyTorch defines it, `constant`/`linear`/`cosine`
  schedules, global-norm clipping, decoupled weight decay; new primitive
  `sum_squares`; scalar-against-tree arithmetic in the interpreter;
  `presk run --entry adamw.step`; `--conformance-entry adamw.step` generates
  a test against `torch.optim.AdamW` (passes at rtol 1e-9); the block's name
  is the algorithm (`adamw` only, `E2010` otherwise).
- Step 2: the `training run` block typed against its objective: sample
  type = batch type without the `batch` axis, `batch(B = n)` binds `B`, the
  data file's type is matched when the file exists (its dimensions bind the
  objective's), `W0002` when it is missing; the `TrainingRun` record in the
  typed program, printed by `presk check --summary`; `tests/diag/train_*.psk`.
  The example `train_minigpt.psk` now has its data
  (`examples/minigpt/data/tokens.psk.d`, 510 windows of a 64-token stream);
  a file whose index bound is below the program's is accepted.
- Step 0: version `0.2.0-dev`; spec questions 28–34.

## 0.1.0 — 2026-09-19

The first version of the Presk compiler: the front-end, an arbitrary-precision
reference interpreter and a readable PyTorch export, built in eight steps
(`presk-specs/implementation/presk-plan-0.1.0.md`). The language is the one of
`presk-specs/examples/*.psk`, which are the acceptance tests and pass every
command below.

### `presk check`

- Lexer with Unicode identifiers (`θ`, `x̂`), ASCII synonyms of the five
  symbols (`Real`, `\`, `@`, `><`, `<<`), newline layout; recursive-descent
  parser with recovery; diagnostics rendered by ariadne, phrased in source
  terms, with codes (`E0xxx` syntax, `E1xxx` elaboration, `E2xxx` typing,
  `E3xxx` regularity, `E4xxx` effects, `E5xxx` lowering, `E6xxx`
  differentiation, `E7xxx` run, `E8xxx` export, `W0001`).
- Elaboration: imports, exact constants with `--const` overrides (reaching
  imported files), `dim` bounds, type aliases, generic modules with structural
  instances, parameter trees, `static assert` decided for all values in ℕ by
  an in-house procedure for the Presburger fragment (property-tested).
- Shape typing of function bodies: named axes, `...` prefixes, broadcasting,
  `idx`, `params`, `ein` with the `seq`/`seq2` relabeling rule, the four call
  argument forms, module members with implicit or explicit θ, `grad`, a
  builtin table; configuration blocks against schemas.
- Regularity classes per parameter (`C∞ < C⁰⁺ < Disc`) with the culprit
  named on `grad` errors; effects against `uses`.
- `--summary` (types, `let`s, classes, effects), `--json`.

### `presk run`

- PSK-Core: a dimension-polymorphic ANF IR with explicit parameter trees,
  unrolled `stack`, lifted lambdas, per-module initializers, structural
  verification.
- Reverse-mode automatic differentiation as a source-to-source
  transformation (`f.vjp`, `f.grad`), with structural adjoints of parameter
  trees; validated against central finite differences at 512 bits.
- Interpreter over MPFR (`rug`, `--bits`), tensors with named axes,
  reproducible randomness from `--seed`; `--entry`, `--dim`, `--const`,
  `--digits`, `--json`, `--verbose`. Golden outputs in `tests/run/`.

### `presk export --target pytorch --mode readable`

- One `nn.Module` per module, `forward` for `apply`, shapes as comments or
  `--shapes=check` asserts, `stack` as `nn.ModuleList`, `torch.func` for
  parameter trees and `grad`, a generated `test_conformance.py` fed with the
  interpreter's data (`--conformance-entry`). Verified against PyTorch 2.14
  in float64 (`make venv`, `make conformance`): `mse_grad`,
  `LayerNorm{8}.apply`, `MiniGPT.apply` and the training `loss` agree with
  the MPFR reference within rtol 1e-9.

### Also

- A tutorial chapter in the manual, with its programs in `tutorial/` and
  `make tutorial` running every command of it (found and fixed: file stems
  that are not Python identifiers, a stacked call nested in an expression,
  `dim` bounds not enforced by `presk run`).
- `presk dump --stage tokens|ast|elab|core|core-after-ad`.
- A rustdoc user manual (`make doc`), architecture decisions
  (`docs/decisions.md`), the questions sent to the specification
  (`docs/spec-questions.md`), and the tooling rationale
  (`presk-specs/implementation/presk-tooling.md`).

### Not in 0.1.0

E-graph rewriting and cost models, code generation, PSK-Sched/PSK-Kernel, the
`.pskk`/`.pskb`/`.pskt` formats, the Zig runtime, Lean 4 certification,
execution of `training run`, `--mode graph`, second-order differentiation,
`vjp`/`jvp`/`concat`/`dropout`/`clamp`/`max`/`min`.
