# Questions for the specification

Points where the examples or the spec chapter (`presk-specs`,
`report/chapters/02-spec.tex`, `presk-draft-spec.md`) leave the compiler a
choice. Each entry records the provisional answer implemented in 0.1.0; the
answer is to be settled in the spec, after which this file is updated.

| # | Question | Provisional answer in 0.1.0 | Status |
|---|---|---|---|
| 1 | `param x : param ℝ[…]` (minigpt.psk) vs `param w : ℝ[…]` (mlp.psk, layernorm.psk) | both accepted; `param` on a type is a no-op qualifier | open |
| 2 | No EBNF for `optimizer`, `training run`, `inference serve`, `model`, `import`, `type`, `static assert`, `dim … <=` | schema table in `presk-types::check::config` | open |
| 3 | Layout of `let` without `in` | expression ends at a newline unless the next token is a binary operator that cannot start an expression; `-` never continues a line | open |
| 4 | `ein` with axis names duplicated across operands (`q * k` both carry `seq`) and the `seq2` convention | relabeling rule: a name `n` carried by both operands is renamed, in the right operand, to the unique unmatched copy `n<digits>` (in the output but in no operand, or in one operand but not in the output) of equal length; ambiguity is an error | open |
| 5 | `cross_entropy -> ℝ[]` in the spec vs `mean(cross_entropy(…))` in train_minigpt.psk | typed `ℝ[...]` (per position) | open |
| 6 | Free dimensions (`B` in mlp.psk with no `dim B`) | implicit fn-level dimension, warning `W0001` | open |
| 7 | `.apply` arity: `net.apply(θ, x)` outside a module, `l1.apply(x)` inside | θ is the explicit first parameter of every module fn in PSK-Core; implicit inside a module body | open |
| 8 | Module-typed fn parameters (`net : MLP{D, 4*D}`) | compile-time only: a call must pass a module path | open |
| 9 | `param pos : ℝ[T, D]` with `dim T <= 1024` free | accepted for typing; `presk run` needs `--dim T`; suggest `ℝ[Tmax, D]` + slice | open |
| 10 | Empty axes (ℕ ∋ 0) and `mean` over a possibly empty axis | runtime error only; no static obligation | open |
| 11 | Regularity per argument (needed for `loss`: C∞ in θ, Disc in the batch) vs the spec's "per function" wording | per argument | open |
| 12 | `grad` over pytrees (`params net`) and over `Rand` functions | allowed; `grad : (P -> ℝ[]) -> (P -> P)` for any pytree `P` | open |
| 13 | Unification of an unnamed axis with a named one (`Linear.apply` result feeding `split_heads`) | allowed; names are never inferred into signatures | open |
| 14 | `stack{L}` semantics and the `.psk.d` data format | `stack` unrolled for constant `L`; no `.psk.d` reader, so `training run` is not executable | open |
| 15 | In a call, `name: expr` (named axis, `positions(seq: T)`) vs `name : ℝ` (ascription, `sqrt(HD : ℝ)`) share the syntax `IDENT :` | ascription only when a type keyword (`ℝ`, `idx`, `param`, `params`) follows the colon; otherwise a named axis | open |
| 16 | Reserved words vs contextual ones: `model` is a keyword in the spec (`model GPTMini = …`) but also an axis name (`model: D`, `axis model`) and a config field | `model`, `run`, `serve`, `stream`, `checkpoint`, `every`, `steps`, `to` are contextual (identifiers recognized by position); the others (`module param params use fn const dim type import as static assert optimizer training inference effect let uses init axis idx ein`) are reserved; config field names may be keywords (`optimizer = adamw`) | open |
| 17 | `-` at the start of a line never continues the previous expression (it would be ambiguous with a negated expression); every other binary operator may start a continuation line | as stated | open |
| 18 | Does a `dim T <= N` bound of the callee's file constrain the caller's argument (`T = 5000 exceeds the bound` in the spec's Python binding)? | not checked at call sites in 0.1.0 | open |
| 19 | `split_heads`/`merge_heads` are "provided by the standard library" but their head count is not an argument | they read `const H` and `const HD` of the calling file; error if absent | open |
| 20 | The result of `Linear.apply : ℝ[..., Out]` loses the name of its last axis (`model`), which the next call re-attaches by unifying unnamed with named | accepted; names are never inferred into signatures (see 13) | open |
| 21 | Functions without a declared return type (`fn apply(x, mask) = …`) | typed from the body; must be checked before their callers, else E2012 | open |
| 22 | `sample` is `uses Rand` but takes no key: where does its randomness come from? | the interpreter draws from a stream seeded by `--seed`, independent of the initializers' keys | open |
| 23 | Which constants does an override (`--const`) reach when a model's constants live in an imported file? | every file that declares the constant | open |
| 24 | How is a function with `...` parameters (`Linear.apply`, `LayerNorm.apply`) run stand-alone: what are the prefix axes? | extra `--dim NAME=N` names become the prefix axes, in order | open |
| 25 | An initializer of module `M` in file `A` mentions `A`'s `dim T`; when called from a function of file `B` whose own implicit `T` is a different variable, how are they related? | bound by name at run time (`T` of `B` supplies `T` of `A`), the run-time counterpart of question 6 | open |
| 26 | A `param` whose shape mentions a file-level `dim` (`pos : ℝ[T, D]` with `dim T <= 1024`) has no concrete shape at construction in a framework like PyTorch | the export makes `T` a constructor argument of the module and of every module that uses it (`MiniGPT(T)`); the spec may prefer `ℝ[Tmax, D]` (question 9) | open |
| 27 | The grammar has `param … [init = expr]` but never names the admissible initializers, nor whether `init` may be omitted | four forms accepted (`zeros`, `ones`, `xavier_uniform`, `normal(std = literal)`), `E2013` otherwise; `xavier_uniform` uses `fan_in` = first axis, `fan_out` = last axis; an omitted `init` means `zeros`; the spec should list the initializers and their distributions | open |
