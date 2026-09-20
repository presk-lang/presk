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
| `E7xxx` | `presk run`, `train`, `serve`, `bench` (entry resolution, input synthesis, runtime — on every backend: the kernel and native backends report through `E7001` too) | implemented |
| `E8xxx` | PyTorch export | implemented |
| `E9xxx` | `.psk.d` data files (`presk data`, the training pipeline) | implemented (0.2.0 step 1) |
| `W0xxx` | warnings | implemented (W0001, W0002, W0003) |

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
| `E1006` | `e` may not be divisible by k | `e / k` with a symbolic `e` and no fact proving `k \| e`; add `static assert e % k == 0` |
| `E1007` | outside the decidable fragment | a product of two symbolic dimensions, a division by a non-constant, `%` outside `static assert … % k == 0`, or an unsupported assertion form |
| `E1008` | duplicate X | two declarations or members with the same name in a file or module |
| `E1009` | not a module / a `param` must be a real tensor / only `nat` constants | a kind mismatch: `params x` where `x` is not a module, a `param` of non-tensor type, a `const` declared with a type other than `nat` |
| `E1010` | `...` must be the first axis / kind `Shape` is not supported | an ellipsis in a non-leading position; a `{S : Shape}` type parameter (specification feature, not implemented) |
| `E1011` | `--const NAME` names no `const` of this file | the override does not match a declaration of the main file |

## Typing codes

| Code | Message | Cause |
|---|---|---|
| `E2001` | unresolved name / unresolved function | a name in a body that is no variable, parameter, module member, constant, function or builtin in scope |
| `E2002` | type mismatch: expected `T`, found `U` | the general unification failure (also: module mismatch, wrong `grad` argument, cannot negate / inject / raise this type, tuple pattern against a non-tuple) |
| `E2003` | shape mismatch / cannot broadcast / axis name mismatch | ranks differ; no trailing or by-name alignment for an elementwise operation; two named axes with different names where one is expected (`seq` vs `seq2`); `·` without the needed axes |
| `E2004` | wrong number of arguments / missing argument / unknown named argument | call arity, a parameter without default not given, a named argument that matches no parameter, a missing `θ` when calling a module function outside its module |
| `E2005` | not callable / not a member | calling a non-function, a function name that is not a member of the module, field access on an expression, a module instance used as a value |
| `E2006` | dimension mismatch: `a` vs `b` | two dimension expressions that must be equal are not (a counterexample is shown when the procedure finds one) |
| `E2007` | `ein` errors | an operand with an unnamed axis or an open prefix, an output axis absent from all operands, an ambiguous relabeling |
| `E2008` | no axis named `a` | `axis a`, `[a: i]` on a tensor whose type has no axis of that name, or a positional index on an open shape |
| `E2009` | index tensor mixed with reals | `+ - * /` between an `idx` tensor and a real; use `real(…)` |
| `E2010` | configuration block errors | unknown or duplicate field, missing required field, value not of the expected form, `objective` whose parameter tree is not the model's or that has no batch parameter, a batch tensor without a `batch` axis, a data file whose sample type does not match the objective's (both types and the offending axis are shown), an unreadable data file, an `optimizer` block whose name is not `adamw` (the name is the algorithm), unknown optimizer name |
| `E2011` | `f` returns `U` but is declared `-> T` | the body's type does not unify with the declared return type (the unification error is reported first) |
| `E2012` | not implemented in 0.1.0 | `vjp`, `jvp`, `concat`, `clamp`, `max`, `min`; ascription to a type other than `ℝ`; a function used as a value; `split_heads`/`merge_heads` without `const H` and `const HD` in scope; a function without return type called before it is checked |
| `E2013` | invalid initializer | `init =` is not one of `xavier_uniform`, `zeros`, `ones`, `normal(std = f)` with `f` a decimal literal (the vocabulary of the clause, see the `language` chapter, "Initializers"); a `param` without `init =` is zero-initialized |
| `E2015` | not part of the core language | a string, quantity, `|>`, `|`, `⊗`, `∘` or `%` inside a function body |

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
| `E6001` | `grad` of a function value that is not a lambda / nested `grad` / unknown function value | 0.1.0 differentiates `grad(λ θ. …)` where the lambda is defined in the same body; second-order differentiation (`grad` inside a differentiated function) and applying an unknown function value are not supported |
| `E6002` | cannot differentiate `p` | an adjoint primitive appears on a differentiated path (second-order derivative of adjoint code) |

## Run-time codes

| Code | Message | Cause |
|---|---|---|
| `E7001` | (message) | `presk run`: unknown entry or module, wrong number of module arguments, an unbound dimension (`give --dim T=N`), a `--dim` value that violates a file bound (`dim T <= 1024`), an unknown `--dim` name, a value that cannot be synthesized (a module-typed or function-typed parameter), or a runtime error of the interpreter (an index out of range, a shape the primitives cannot align), labelled with the source of the failing binding; on `--backend kernel\|native`, a shape with an open prefix that cannot be materialized, a dimension that does not evaluate to a natural number, or a kernel the lowering does not support |

## Export codes

| Code | Message | Cause |
|---|---|---|
| `E8001` | not exported / builtin has no PyTorch export | a construct with no readable PyTorch form (a `let` inside an expression, a builtin outside the table) |

## Data codes

| Code | Message | Cause |
|---|---|---|
| `E9001` | not a data type | the sample type of a `.psk.d` is not closed (a symbolic dimension, an open prefix `...`), is `idx[0]`, a named type, or `params M` opened without the program |
| `E9002` | not a .psk.d file | bad magic, unsupported version, unreadable path |
| `E9003` | truncated | fewer bytes than the header announces, a sample index beyond the count, an I/O error |
| `E9004` | value out of bound / not an index / not a real | an index at or above its `idx[N]` bound, an unparsable text value, too few or too many values on a line of `presk data pack` |
| `E9005` | sample does not match the type | leaf count, element count or kind (reals where indices are expected) |

## Warnings

| Code | Message | Cause |
|---|---|---|
| `W0001` | implicit dimension `B` | an undeclared name in a dimension position of a function signature is taken as a dimension of that function; declare it with `dim B` or `{B : dim}` to silence the warning |
| `W0002` | data file `path` not found | the `data` file of a `training run` is not readable at check time; its sample type is checked by `presk train` |
| `W0003` | no generation function for this `inference serve` | the file has no top-level `fn (θ : params M, ctx : idx[V][batch: B, seq: T], …) -> idx[V][batch: B]`; `presk serve --entry FN` names one (`serve` chapter) |
