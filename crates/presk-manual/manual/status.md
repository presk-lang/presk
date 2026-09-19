# Status

Version 0.1.0 was built in eight steps (plan:
`presk-specs/implementation/presk-plan-0.1.0.md`); all are done and the
acceptance criteria below hold. `CHANGELOG.md` lists the contents of the
version.

| Step | Contents | Status |
|---|---|---|
| 0 | Workspace scaffold, CLI surface, Makefile, examples copied from `presk-specs`, decision and spec-question logs | done (2026-09-18) |
| 1 | Lexer (Unicode identifiers, ASCII synonyms, layout newlines), parser, AST, diagnostics rendering, `presk dump --stage tokens\|ast`, syntax errors in `presk check` | done (2026-09-18) |
| 2 | Dimension expressions as linear forms, the Presburger-fragment solver (equalities, bounds, divisibility, non-zero divisor obligations), elaboration: `import` resolution, `const` evaluation, `static assert`, module instantiation, `--const` overrides, `presk dump --stage elab` | done (2026-09-19) |
| 3 | Shape typing: named axes, `...` prefixes, broadcasting, `idx`, `params`, `ein`, call argument forms, configuration-block schemas; `presk check` complete, `--summary` | done (2026-09-19) |
| 4 | PSK-Core IR (ANF over exact real tensors), lowering with explicit parameter trees, `stack{L}` unrolling, `presk dump --stage core` | done (2026-09-19) |
| 5 | Regularity classes (`C∞ < C⁰⁺ < Disc`, per argument) and effect checking; `grad` legality with a named culprit | done (2026-09-19) |
| 6 | Automatic differentiation (reverse mode, source to source on PSK-Core) | done (2026-09-19) |
| 7 | Reference interpreter in arbitrary precision (`rug`/MPFR), `presk run` | done (2026-09-19) |
| 7b | Readable PyTorch export and generated conformance test, `presk export` | done (2026-09-19) |
| 8 | Hardening, `--json` for `check`, `CHANGELOG.md`, README, spec questions sent to `presk-specs`, tag `v0.1.0` | done (2026-09-19) |
| — | Tutorial chapter (`tutorial/`, `make tutorial`); PyTorch conformance verified with `make conformance` | done (2026-09-19) |

## Acceptance criteria of 0.1.0

- The five example programs (`tests/examples/{mlp,layernorm}/*.psk`,
  `tests/examples/minigpt/*.psk`) pass `presk check`.
  *Done: all five pass the complete `presk check` (shapes, regularity,
  effects) with only `W0001` implicit-dimension warnings (`make check-examples`);
  since 0.2.0 step 2 the `training run` of `train_minigpt.psk` is checked
  against `examples/minigpt/data/tokens.psk.d`.*
- `mlp.psk` (`mse`, `mse_grad`), `layernorm.psk` (`LayerNorm.apply`) and the
  `minigpt.psk` forward pass with reduced constants evaluate with `presk run`.
  *Done (`make run-examples`); golden outputs in `tests/run/`, and the AD
  gradient of `mse` agrees with finite differences at 512 bits to 2⁻¹⁰⁰.*
- The three model files export to PyTorch and the generated code matches the
  golden snapshots. *Done: the five files export, the generated Python
  byte-compiles, and `test_conformance.py` is generated with the
  interpreter's data (to be run where PyTorch is installed).*

## Version 0.2.0

Training end to end, planned in
`presk-specs/implementation/presk-plan-0.2.0.md` and built the same day
(2026-09-19); the acceptance criteria of the plan hold (`make train-examples`,
`make conformance`, `make tutorial`). E-graph rewriting and Cranelift code
generation open 0.3.0.

| Step | Contents | Status |
|---|---|---|
| 0 | Version `0.2.0-dev`, CHANGELOG "Unreleased", spec questions 28–34 logged | done (2026-09-19) |
| 1 | The `.psk.d` container (crate `presk-data`), `presk data pack \| lm-windows \| info \| export`, the pipeline over sample indices, `E9xxx` | done (2026-09-19) |
| 2 | The `data` pipeline typed against the objective (sample type = batch type minus `batch`, `B` bound by `batch(B = n)`, the file's type matched and its dimensions bound); `TrainingRun` record in the typed program; `--summary`; `W0002`; `tests/diag/train_*.psk` | done (2026-09-19) |
| 3 | The optimizer lowered to PSK-Core (`adamw.init`, `adamw.lr`, `adamw.step`; AdamW as PyTorch defines it, three schedules, clipping); `sum_squares`; scalar-over-tree arithmetic in the interpreter; `presk run --entry adamw.step`; one-step conformance against `torch.optim.AdamW` (`make conformance`) | done (2026-09-19) |
| 4 | `stack{L}` as a `fold` in PSK-Core (`Rhs::Fold`, scan/reverse/index forms, `set_slice`), the interpreter and AD (backward fold with a generated `fold_bwd` body); `--unroll-stack` for 0.1.0's form; golden outputs unchanged; finite differences through a stacked module (`tests/data/stack_grad.psk`) | done (2026-09-19) |
| 5 | `presk train` (crate `presk-train`): the loop over the interpreter with the objective's `vjp`, the PSK-Core optimizer and the data pipeline; `--steps`, `--epochs`, `--data`, `--ckpt`, `--log-every`, `--json`; checkpoints as `params M` files; regression converges, tiny MiniGPT 20 steps golden; `presk run` on a training file points to `presk train` | done (2026-09-19) |
| 6 | PyTorch export: `--train` writes `pskd.py`, `train.py` and the data file; `--conformance-entry train:K` replays `K` interpreter steps in PyTorch (20 steps of the tiny MiniGPT: worst relative gap 1.3e-10); `--mode graph` writes `export.py` producing `model.pt2` and `model.onnx`, tested under ONNX Runtime; `make venv` gains onnx, onnxscript, onnxruntime | done (2026-09-19) |
| 7 | Manual chapters `training`, `data` and `experiments` (`make experiment`), tutorial step 11 (train a model), README, CHANGELOG 0.2.0, spec questions copied to `presk-specs`, version 0.2.0; tag `v0.2.0` by the user | done (2026-09-19) |
| 8 | Optional Cranelift spike | not done (open for 0.3.0) |

## Version 0.3.0

Native backend first, then e-graph rewriting, planned in
`presk-specs/implementation/presk-plan-0.3.0.md` and built the same day
(2026-09-19); the verification list of the plan (§5) holds, see below.

| Step | Contents | Status |
|---|---|---|
| 0 | Version `0.3.0-dev`, CHANGELOG "Unreleased", spec questions 35–42, decision D21 (the `Backend` seam) | done (2026-09-19) |
| 1 | Crate `presk-rt`: runtime values (exact, f64, f32, i64 buffers with named axes; trees), the `Backend` trait and `Prepared` entries, `InterpBackend`, parameter I/O, `run_with`; `presk run --backend interp` and `presk train --backend interp` over the trait, goldens unchanged | done (2026-09-19) |
| 2 | PSK-Kernel v0 (crate `presk-kernel`): one kernel per tensor primitive at concrete shapes, specialized on first use; loop-level IR with a textual form (`presk dump --stage kernel`); kernel-IR interpreter in `f64`/`f32`; the host program over PSK-Core bodies; `--backend kernel`; the conformance suite (41 primitive cases, 11 example entries, both realizations) with measured gaps | done (2026-09-19) |
| 3 | Cranelift backend (crate `presk-cranelift`): every kernel JIT-compiled on first use, `libm` symbols, `--backend native` on `run`/`train`; the native conformance suite; `presk bench` and the first performance table (tiny MiniGPT forward 27× the reference, 20 training steps in 0.4 s) | done (2026-09-19) |
| 4 | Training natively: `presk train --backend native --realize f64\|f32` measured against the reference (regression 200 steps: 4.4e-16 in f64, 1.6e-7 in f32, a test); `train:K` exports the native `θ_K` too and the experiments chapter's realization table gains presk's `f64`/`f32` points; `make train-examples` extended; `V=64 D=64 H=4 L=4` trains 500 steps in 159 s with the loss decreasing | done (2026-09-19) |
| 5 | `inference serve` resolved in the checker (model, weights path, generation function by signature, `W0003`, `--summary`); crate `presk-serve` (the loop over `Backend`, greedy copy `sample → argmax`, sliding window when the parameters fix `T`); `presk serve`; `presk export --serve` (`serve.py`, `pskd.read_params` with the generated layout); greedy tokens identical on the reference, native f64/f32 and `serve.py` float64/float32; manual chapter `serve`; question 43 | done (2026-09-19) |
| 6 | Crate `presk-opt` over `egg`: one e-graph per PSK-Core function body with the types as class analysis, rules exact over ℝ (algebra, reductions, the adjoint cleanup: `sum_to`/`broadcast_like` identities, `zeros_like`, `field`/`set_field` chains into records, `proj` of tuples), extraction by operation counts, ANF write-back (dead code disappears); `--opt` on `run`/`train`/`serve`/`bench`/`export`, `presk dump --stage core-opt` with a report; the eight run goldens and the 20-step training golden unchanged, a property test against the interpreter at 128 bits, native outputs identical with and without `--opt`; `Linear.apply.vjp` 14 → 7 lets; manual chapter `optimization`; D24, question 44 | done (2026-09-19) |
| 7 | Cost model v0.1: work and traffic of every kernel from its IR (`presk_kernel::cost`) and of every PSK-Core node from its type (`presk_opt::model`); `presk bench --calibrate FILE.json` times every kernel natively (`TimedExec`), fits the roofline `λ + max(W/π, Q/β)` by relative least squares and writes the profile with Spearman correlations (counts 0.15, work 0.91, roofline 0.95 on the tiny MiniGPT in `f64`); `--opt --profile` extracts by predicted seconds; goldens unchanged under the profile; bench with and without the profile within the noise; manual section in `optimization`; D25 | done (2026-09-19) |
| 8 | Version 0.3.0; tutorial step 12 (native run and training, `--opt`, `--calibrate`, `--profile`, in `make tutorial`); README, CHANGELOG 0.3.0, spec questions 35–44 copied to `presk-specs`; tag `v0.3.0` by the user | done (2026-09-19) |

### Verification of 0.3.0 (plan §5)

- `make test`: the whole workspace, including the kernel and native
  conformance suites, the e-graph oracle tests (the run goldens and the
  training golden under `--opt`, with and without a profile) and the
  property test of the rewriting against the interpreter at 128 bits.
- `make run-examples`: the interpreter, unchanged; `presk run … --backend
  native --realize f32` on the tiny MiniGPT forward pass (gap 0 on the
  argmax of the example, `native` chapter).
- `presk train … --backend native` on the tiny MiniGPT, `presk serve … --backend
  native` on its checkpoint (`make train-examples`, `make serve-examples`):
  greedy tokens identical on every backend.
- `presk bench` on forward, loss and optimizer step (`make bench`); `presk
  train --backend native --opt --profile` on the same goldens, timed within
  the noise (`make opt-examples`, `make calibrate`).
- `make conformance`: the PyTorch side plus the native K-step line
  (`conformance/native_{f64,f32}.json`); `make experiment`: the realization
  table with presk's `f64`/`f32` points; `make tutorial`: steps 0–12.

Not done in 0.3.0, carried forward: `ein` reassociation and fusion rules
(the `(q·k)·v` decision), a fusion pass making `Q` non-additive, the
`.pskk`/`.pskb` bundle and the Zig runtime (questions 35–36), the precision
solver behind `--realize`, the native StableHLO emission of `--mode graph`.

## Out of 0.1.0

E-graph rewriting, cost models, code generation (backend to be chosen in
0.2), PSK-Sched and PSK-Kernel, the `.pskk`/`.pskb`/`.pskt` formats, the Zig
runtime, Lean 4 certification, execution of a `training run`, `presk export
--mode graph` and `--precision=solved`.
