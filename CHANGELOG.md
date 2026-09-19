# Changelog

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
