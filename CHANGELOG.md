# Changelog

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
