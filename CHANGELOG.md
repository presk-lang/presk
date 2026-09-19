# Changelog

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
