# Diagnostics

Every problem is reported as a diagnostic with a code, a severity, a message,
one or more labelled source spans, and optionally notes and a help line. The
message and the labels are phrased in the terms of the source: axis names,
dimension names, function names — never internal identifiers.

Rendering (on a terminal, with colors; below, the plain form used in tests):

```text
[E0010] Error: expected `=`, found identifier `gelu`
   ,-[ tests/diag/parse_missing_eq.psk:3:3 ]
   |
 3 |   gelu(x)
   |   ^^|^
   |     `--- expected `=` here
---'
```

Locations are `file:line:column`; columns count characters, so a Unicode
identifier counts as one column per character.

Several errors may be reported for one file: after an error inside a
declaration, parsing resumes at the next line that starts, at column 0, with a
declaration keyword.

## Code ranges

| Range | Phase | Status |
|---|---|---|
| `E00xx` | lexing and parsing | implemented |
| `E1xxx` | elaboration (names, imports, constants, module instantiation, assertions) | implemented |
| `E2xxx` | shape typing, calls, configuration blocks | implemented |
| `E3xxx` | regularity classes (`grad` legality) | implemented |
| `E4xxx` | effects | implemented |
| `E5xxx` | lowering to PSK-Core | implemented |
| `E6xxx` | automatic differentiation | implemented |
| `E7xxx` | `presk run`, `train`, `serve`, `bench` (entry resolution, input synthesis, runtime — on every backend: the kernel and native backends report through `E7001` too; a realization plan's layout, `E7002`; an entry undefined at its dimensions, `E7003`, 0.10.0 step 7b) | implemented |
| `E8xxx` | `presk export`: the PyTorch export (`E8001`); the C export (`E8003`–`E8009`, 0.9.0; `E8002` no longer emitted since 0.9.0 step 5, `E8005` since step 6) | implemented |
| `E9xxx` | `.psk.d` data files (`presk data`, the training pipeline) | implemented (0.2.0 step 1) |
| `W0xxx` | warnings | implemented (W0001, W0002, W0003, W0004) |

## Codes

| Code | Message | Cause |
|---|---|---|
| `E0001` | unterminated string literal | a `"` is not closed before the end of the line |
| `E0002` | unexpected character `c` (U+XXXX) | a character that belongs to no token, e.g. `§`; the character is skipped |
| `E0010` | expected X, found Y | the general syntax error: the parser expected a token or construct (`=`, `]`, an expression, a type, end of line…) and found something else |
| `E0011` | (construct-specific message) | a `let` without a body on the next line, or a `let` inside parentheses or brackets |

Every code has at least one reference input in `tests/diag/` whose rendered
report is a snapshot test; adding a code means adding such a file.

## Elaboration codes

| Code | Message | Cause |
|---|---|---|
| `E1001` | unresolved name `X` in a dimension / unresolved type / unresolved module | a name in a type or dimension position is not a `const`, a `dim`, a type parameter, a type alias or a module in scope (imports are reached through their alias: `m.V`) |
| `E1002` | cannot import / import cycle | the imported file does not exist, cannot be read, or imports (transitively) the importing file |
| `E1003` | `M` takes N dimension arguments, K given | wrong arity of a module instantiation, or a malformed `stack{N}(M)` |
| `E1004` | static assertion fails / cannot be proved | the assertion is false with the file's constants, false for some value of a symbolic dimension (the counterexample is shown), or outside what the decision procedure handles |
| `E1005` | `X` is not a natural number | a `const` evaluates to a non-integer or a negative (constants are exact: `768 / 7` is `768/7`), or a literal exceeds 64 bits |
| `E1006` | `e` may not be divisible by k | `e / k` with a symbolic `e` and no fact proving `k \| e`. The help suggests `static assert e % k == 0`, which the checker then refuses for a symbolic `e` (`E1004`: an assertion is decided for every value, and adds no fact); write `e` as a multiple of `k` instead (`dim M`, `k * M`) |
| `E1007` | outside the decidable fragment | a product of two symbolic dimensions, a division by a non-constant, `%` outside `static assert … % k == 0`, or an unsupported assertion form |
| `E1008` | duplicate X | two declarations or members with the same name in a file or module |
| `E1009` | not a module / a `param` must be a real tensor / only `nat` constants | a kind mismatch: `params x` where `x` is not a module, a `param` of non-tensor type, a `const` declared with a type other than `nat` |
| `E1010` | `...` must be the first axis / kind `Shape` is not supported | an ellipsis in a non-leading position; a `{S : Shape}` type parameter (specification feature, not implemented) |
| `E1011` | `--const NAME` names no `const` of this file | the override does not match a declaration of the main file |

## Typing codes

| Code | Message | Cause |
|---|---|---|
| `E2001` | unresolved name / unresolved function | a name in a body that is no variable, parameter, module member, constant, function or builtin in scope |
| `E2002` | type mismatch: expected `T`, found `U` | the general unification failure (also: module mismatch, wrong `grad` argument, cannot negate / inject / raise this type, tuple pattern against a non-tuple; `sample` given no key, a `softmax` mask that is not `bool[…]`, a mask in arithmetic: 0.10.0 step 7b) |
| `E2003` | shape mismatch / cannot broadcast / axis name mismatch | ranks differ; no trailing or by-name alignment for an elementwise operation; two named axes with different names where one is expected (`seq` vs `seq2`); `·` without the needed axes |
| `E2004` | wrong number of arguments / missing argument / unknown named argument | call arity, a parameter without default not given, a named argument that matches no parameter, a missing `θ` when calling a module function outside its module |
| `E2005` | not callable / not a member | calling a non-function, a function name that is not a member of the module, field access on an expression, a module instance used as a value |
| `E2006` | dimension mismatch: `a` vs `b` | two dimension expressions that must be equal are not (a counterexample is shown when the procedure finds one). A `split` whose two lengths are both known says so as arithmetic instead: *`split` of `a` does not account for every element*, with the product and the axis length |
| `E2007` | `ein` errors | an operand with an unnamed axis or an open prefix, an output axis absent from all operands, an ambiguous relabeling |
| `E2008` | no axis named `a` | `axis a`, `[a: i]` on a tensor whose type has no axis of that name, or a positional index on an open shape |
| `E2009` | index tensor mixed with reals | `+ - * /` between an `idx` tensor and a real; use `real(…)` |
| `E2010` | configuration block errors | unknown or duplicate field, missing required field, value not of the expected form, `objective` whose parameter tree is not the model's or that has no batch parameter, a batch tensor without a `batch` axis, a data file whose sample type does not match the objective's (both types and the offending axis are shown), an unreadable data file, an `optimizer` block whose name is not `adamw` (the name is the algorithm), unknown optimizer name; an `inference serve`'s `generate` that names no top-level function of the file, or one whose signature is not a generation function's (since 0.10.0 step 8) |
| `E2011` | `f` returns `U` but is declared `-> T` | the body's type does not unify with the declared return type (the unification error is reported first) |
| `E2012` | not implemented in 0.1.0 | `vjp`, `jvp`, `concat`, `clamp`, `max`, `min`; ascription to a type other than `ℝ`; a function used as a value; `split_heads`/`merge_heads` without `const H` and `const HD` in scope; a function without return type called before it is checked |
| `E2013` | invalid initializer | `init =` is not one of `xavier_uniform`, `zeros`, `ones`, `normal(std = f)` with `f` a decimal literal (the vocabulary of the clause, see the `language` chapter, "Initializers"); a `param` without `init =` is zero-initialized |
| `E2015` | not part of the core language | a string, quantity, `|>`, `|`, `⊗`, `∘` or `%` inside a function body |
| `E2016` | `f` calls itself / module `M` includes itself | a function calls itself, directly or through other functions, or a module includes itself through its `use` members, directly or through other modules (the reference's requirement "No recursion"; iteration is `stack{N}(M)`); every function of the cycle is named and each call or `use` labelled. Refused once the calls are resolved, before anything walks them (`presk dump --stage elab` refuses a module that includes itself too); until 0.10.0 step 8 such a program overflowed the checker's stack |

## Regularity and effect codes

| Code | Message | Cause |
|---|---|---|
| `E3001` | cannot differentiate through `p`: the function is discontinuous (Disc) in `x` | a `grad` whose function is `Disc` in the differentiated parameter; the primitive responsible (`argmax`, `sample`, the index argument of `embed`/`cross_entropy`, `real` of an index) is labelled, with a suggested substitute |
| `E4001` | `f` uses the effect `Rand` without declaring it | a function (or a lambda inside it) draws randomness (`sample`, initializers) or calls a function that does, without `uses Rand` |
| `E4002` | unknown effect `E` | a `uses` clause naming an effect other than `Rand` |

## Lowering codes

| Code | Message | Cause |
|---|---|---|
| `E5001` | not lowered / not a value of the core language | an expression the checker accepts but PSK-Core has no form for (a comparison in a body, a module used as a value, a function value applied to named arguments); internal inconsistencies are reported with this code as well |
| `E5002` | `stack` with a symbolic count is not supported | the *initializer* of `stack{L}(M)` splits its key `L` times, so `L` must be a constant there (and everywhere with `--unroll-stack`); `apply` through a stack is a `fold` and accepts a symbolic `L` |

## Differentiation codes

| Code | Message | Cause |
|---|---|---|
| `E6001` | `grad` of a function value that is not a lambda / nested `grad` / differentiating a backward fold / application of an unknown function value | `grad(λ θ. …)` is differentiated where the lambda is defined in the same body. Refused: second-order differentiation (`grad` inside a differentiated function, or the backward fold a stacked module's gradient contains), and applying, inside a differentiated function, a function value that is not a lambda of that function's own body, such as a lambda of an enclosing body that a lambda captures (`let g = λ a. sq(a)`, `let k = λ b. g(b)`, `k(x)`). A lambda of the body, applied there, is differentiated through the function it calls (before 0.10.0 step 3 it was refused with this code when it was the body's only lambda, and differentiated as another lambda of the body, without a diagnostic, when there were several) |
| `E6002` | cannot differentiate `p` | an adjoint primitive appears on a differentiated path (second-order derivative of adjoint code) |

## Run-time codes

| Code | Message | Cause |
|---|---|---|
| `E7001` | (message) | `presk run`: unknown entry or module, wrong number of module arguments, an unbound dimension (`give --dim T=N`), a `--dim` value that violates a file bound (`dim T <= 1024`), an unknown `--dim` name, a value that cannot be synthesized (a module-typed or function-typed parameter), or a runtime error of the interpreter (an index out of range, a shape the primitives cannot align), labelled with the source of the failing binding; on `--backend kernel\|native`, a shape with an open prefix that cannot be materialized, a dimension that does not evaluate to a natural number, or a kernel the lowering does not support; a realization whose `quant` map names a path that is not a parameter, an axis the parameter does not have (or a name two of its axes share), a width other than 8 or 4 bits, an entry whose first argument is not a parameter tree, or `presk train` with a quantised realization (0.6.0 step 5a); a realization whose `params` names a path that is not a parameter, a path `quant` names too, or a type other than `f32` or `bf16`, and `presk train`, `calibrate --step` or `solve --step` with a realization that types parameters (0.12.0 step 1) |
| `E7002` | the plan's layout of `NEST`: … / the realization accumulates `NEST` in T, and the kernels accumulate its S storage in A | a realization plan (`--plan @FILE`, `realize` chapter; 0.7.0 step 1) that names a nest the program does not have (no such function or binding, a binding inside a fusion group, or one that is not a primitive), or states a decision the nest cannot have (a tile on a nest that is not a `dot`, one of its derivatives `dot_vjp_l` and `dot_vjp_r`, or an `ein`, passes on one that is not a `softmax`) — both reported before anything runs; or, when the nest is launched, a layout that is ill-formed at its shapes and types (the reference's conditions 3, 4, 7 and 9, named in the message; on a `dot_vjp_r` summed over several leading axes, a k-block must divide the innermost of them, since 0.10.0 step 1) or, for a thread split, against its kernel and the machine: a loop at that depth that is not parallel or has one iteration (condition 8), a worker count other than the machine's or half of it on the native backend (condition 5); or, on the kernel and native backends, a realization whose declared accumulation type at a nest (`accum`) is not its storage's arithmetic type, `bf16` accumulated in `bf16` or `f32` in `f64`, which no kernel honours (refused by the backend's `prepare` before anything runs, since 0.10.0 step 8; the reference interpreter runs it); or, under `--opt` (`run`, `train`, `serve`, `bench`, `explain`), a realization that types a site, or a plan that lays one out, of a function the optimizer wrote back with new numbers: "--opt writes each function it optimizes back with new numbers … it types `Linear.apply/%3`, now `Linear.apply/%7`" (since 0.13.0 step 1, D134) |
| `E7003` | `F` is refused at the dimensions given: … | an entry undefined at the dimensions it is given (0.10.0 step 7b; the reference's definition "Undefined at dimensions" and requirement "Entries"), refused before anything is computed, on every executor, by `presk run`, `train`, `serve`, `export`, `bench`, `plan`, `autotune`, `dump`, `calibrate` and `quantize` alike: an operation of its body, of a function it calls, of a closure it applies or of a `fold` of count at least one is undefined there — `mean` over no element, `mean axis a` over an axis of length 0, a selection `x[a: i]` outside its axis, `embed`, `cross_entropy`, `argmax` or `sample` with no index below `V`, or the derivative of one of these — or a parameter's type has no value (indices below 0 in a tensor with an element). The message names the operation, the function it is in, and the dimension and its value; the label points at the operation. What depends on values stays an error of the run (`E7001`): a division by zero, a masked `softmax` whose mask empties a row. A softmax over an axis of length 0 is defined, and empty |

## Export codes

The C export's codes, `E8003`–`E8009`, are reported when the entry is
exported, not by `presk check`: `crates/presk-cli/tests/export_c.rs` runs
`presk export --target c` on `tests/diag/export_c_key.psk`,
`tests/diag/export_c_scalar.psk`, `tests/diag/export_c_quant.psk`,
`tests/export_c/serve_untied.psk` and the examples, and asserts each code
and message.

| Code | Message | Cause |
|---|---|---|
| `E8001` | not exported / builtin has no PyTorch export | a construct with no readable PyTorch form (a `let` inside an expression, a builtin outside the table) |
| `E8002` | (no longer emitted) | refused `presk export --target c` as a whole in 0.9.0 steps 0–2, then its options `--plan` and `--threads` in steps 3–4, exit code 2; both are implemented since step 5, and the code is not reused |
| `E8003` | `E` is exported at bound dimensions only: unbound dimension … | `presk export --target c` without a `--dim` for one of the entry's dimensions, or with a shape left open (`...`): the C program has literal shapes only |
| `E8004` | the entry calls the initializer `f`, which the C runtime does not provide | the exported entry runs an initializer (`normal`, `xavier_uniform`), as `M.init` does: an exported model reads its parameters |
| `E8005` | (no longer emitted) | refused an entry that calls `sample` in 0.9.0 steps 3–5; since step 6 `sample` is exported, drawing since 0.10.0 step 7b from its key, and the code is not reused |
| `E8006` | a key reaches …: the C program holds the entry's keys only to draw from them | a key the C program would have to compute with: in the entry's result, in a kernel's operand, or given to `sample` when it is not an argument of the entry (split, or stacked by a fold). A key may be split (`split_key`), given to a dropout mask outside training, which reads none, or given to `sample` as an argument of the entry, which is then a `presk_key` of the program's `_in` struct (0.10.0 step 7b); a key nothing draws from is dropped |
| `E8007` | the entry's result holds integer weights / `merge a, b` reshapes integer weights | integer weights where the C program cannot follow them: in the exported entry's result (the program writes the reals it computes, and the weights are read), under a reshape (`merge`, `split`, `reshape_like`, which keep the integers with their scales' axis under a new shape), or drawn from by `sample`. Every integer weight was refused until 0.9.0 step 7; since then the program reads them, quantizing them as it reads them, slices them and hands them to its kernels |
| `E8008` | the entry's result holds a nat | a scalar (`nat`, `int`, `f64`, `bool`) in the exported entry's result: the program writes its outputs as a `.psk.d` file, which holds tensors only |
| `E8009` | `E` is served over the whole context, which grows by a token at every step | `presk export --target c --serve` of a generation function whose model's parameters do not fix the context's length: `presk serve` then passes the whole context, one token longer at every step, and a C program has one shape. The rule is `presk serve`'s own (the initializer's dimensions include the context's), so `generate` slides a window exactly where `presk serve` does (0.9.0 step 6) |

## Data codes

| Code | Message | Cause |
|---|---|---|
| `E9001` | not a data type | the sample type of a `.psk.d` is not closed (a symbolic dimension, an open prefix `...`), is `idx[0]`, a named type, or `params M` opened without the program |
| `E9002` | not a .psk.d file | bad magic, unsupported version, unreadable path |
| `E9003` | truncated | fewer bytes than the header announces, a sample index beyond the count, an I/O error |
| `E9004` | value out of bound / not an index / not a real | an index at or above its `idx[N]` bound, an unparsable text value, too few or too many values on a line of `presk data pack` |
| `E9005` | sample does not match the type | leaf count, element count or kind (reals where indices are expected); a `params` checkpoint larger than the program's layout at the given dimensions — a model of other dimensions, such as `pos : ℝ[T, D]` trained at another `T` (0.6.0 step 5d) |

## Warnings

| Code | Message | Cause |
|---|---|---|
| `W0001` | implicit dimension `B` | an undeclared name in a dimension position of a function signature is taken as a dimension of that function; declare it with `dim B` or `{B : dim}` to silence the warning |
| `W0002` | data file `path` not found | the `data` file of a `training run` is not readable at check time; its sample type is checked by `presk train` |
| `W0003` | no generation function for this `inference serve` | the block has no `generate` field and the file no top-level `fn (θ : params M, [key : key,] ctx : idx[V][batch: B, seq: T], …) -> idx[V][batch: B]`; `presk serve --entry FN` names one (`serve` chapter) |
| `W0004` | the realization types `g.vjp`, which this stored gradient only calls | under `--remat store` a realization names a `.vjp`, by function or by one of its sites, whose sweep the program runs as `g.bwd` after `g.fwd`, and names neither of them: the type applies to nothing (a realization solved under `--remat recompute`, or before 0.10.0 step 4; one `presk solve --step` writes under `store` names both, and is not warned of). Run it with `--remat recompute`, or solve again under `store`, which names `.fwd` and `.bwd` (0.10.0 step 4b, D94; `language` chapter, "Residuals") |

## Precision reports

`presk check --precision` prints no coded diagnostic, but four lines of its
report say that part of the bound is not what it claims to be:

| Line | Meaning |
|---|---|
| `uncovered` | sites the envelope does not mention; their magnitude was guessed from their inputs, which usually means the envelope was measured on another entry |
| `unbounded` | a quotient whose divisor reached zero in the samples, so it has no relative bound; the bound is infinite and says so |
| `fallback` | sums and contractions whose terms the envelope did not record, so the rule of `--model` fell back to 0.5.0's `k` one there — an envelope written before 0.6.0 step 6; calibrate again |
| `estimate rss` | the root-sum-square propagation is an **estimate**, not a bound, and is printed under that word (D51) |

The reports of a training step, `presk calibrate --step` and `presk solve
--step` (0.6.0 step 7b), have three columns that qualify the gap:

| Column | Meaning |
|---|---|
| `storage floor` | the update gap of the exact `θ′` merely rounded to the type the optimizer stores it in: the part of the gap no arithmetic can remove. When the update gap equals it, the realization's arithmetic is not the problem; when the floor alone exceeds the budget, no realization with that storage can meet it (question 61) |
| `grad gap` | where the error enters; it is **not** what the budget binds, and it can be hundreds of times smaller than the update gap, since AdamW divides by `√v` (D52) |
| `noise` | the sampling noise between two batches of the same size, the scale a rounding error is to be compared with; not a threshold |

`presk solve --step` prints `OUTSIDE the budget` when widening has run out
of narrowed functions and the measurement is still above the budget; it
still exits with code 0. An envelope measured on an entry rather than a step
exits with code 2; `presk calibrate --step` without `--weights` exits with
code 2 and names the reason (a step from initialisation measures an
untrained network).

`--model` other than `cond`, `normwise`, `k`, `rss` or `all` exits with code 2;
`--measure` on an envelope whose weights or data file cannot be read exits
with code 1 and names the file (a checkpoint of other dimensions is `E9005`).
