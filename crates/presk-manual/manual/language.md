# The language, as accepted by this version

This chapter describes the syntax the parser accepts and the static
semantics implemented: elaboration (names, constants, dimensions,
assertions, module instances, signatures), shape typing of function bodies
("Typing"), regularity classes and effects, and the intermediate
representation ("PSK-Core"). The reference programs are in
`tests/examples/`; the reference text is the specification chapter of
`presk-specs`. Where the grammar of the specification is only sketched, the
choice made here is recorded in `docs/spec-questions.md`.

## Lexical structure

**Comments** start with `--` and run to the end of the line. There are no
block comments.

**Identifiers** follow Unicode XID: a letter or `_`, then letters, digits,
`_` and combining marks. Greek letters and accented names are ordinary
identifiers (`θ`, `γ`, `μ`, `σ2`, `ŷ`, `x̂`); names are compared after NFC
normalization, so a precomposed and a decomposed spelling of the same name
denote the same identifier. `·` is never part of an identifier.

**Reserved words:** `module param params use fn const dim type import as
static assert optimizer training inference effect let uses init axis idx ein`,
plus `ℝ` and `λ`. The words `model run serve stream checkpoint every steps to`
are *contextual*: they are ordinary identifiers that the parser recognizes by
position (`training run`, `model = …`, `axis model`, `checkpoint every 1000
steps to "ckpt/"` all work).

**Unicode symbols and their ASCII synonyms.** Both spellings are accepted
everywhere and mean exactly the same; the Unicode form is canonical and is
what diagnostics print.

| Unicode | ASCII | Meaning |
|---|---|---|
| `ℝ` | `Real` | the real tensor type |
| `λ` | `\` | lambda |
| `·` | `@` | contraction of the last axis of the left operand with the first of the right |
| `⊗` | `><` | product of parallelization strategies |
| `∘` | `<<` | module composition |

**Literals.**

| Form | Examples | Token |
|---|---|---|
| natural | `768`, `0` | `Nat` (arbitrary size; the text is kept) |
| decimal | `0.02`, `3e-4`, `1e-5` | `Float` (the text is kept; no rounding at parse time) |
| quantity | `5ms`, `3MiB`, `204GB/s`, `192KiB/sm` | `Quantity { num, unit }` — a number immediately followed by a unit |
| string | `"data/tokens.psk.d"` | `Str` (escapes `\"`, `\\`, `\n`, `\t`) |

A leading `-` is the unary minus operator, not part of the literal
(`logits[seq: -1]` parses as a negated `1`).

**Layout.** Line breaks are significant. A declaration, a module member or a
configuration field ends at the end of its line. An expression ends at the
end of its line unless the *next* line starts with a binary operator that
cannot itself begin an expression (`|>`, `|`, `⊗`, `∘`, `·`, `*`, `/`, `%`,
`^`, `+`, `==`, `<=`, …); such a line continues the expression. `-` never
continues a line, since `-x` is an expression. Inside parentheses, brackets
and dimension-argument braces, line breaks are ignored, which is how a long
function signature or call spans several lines. A `let` binding takes the rest
of its line and its body starts on the next line (there is no `in`).

## Program structure

A file is a sequence of declarations.

```text
import "../mlp/mlp.psk" as ff          -- another file, under an alias
const D : nat = 768                     -- hyperparameter (the type is optional)
const HD : nat = D / H                  -- computed constant
static assert D % H == 0                -- checked at compile time (step 2)
dim B                                   -- symbolic dimension
dim T <= 1024                           -- bounded symbolic dimension
type Acts = ℝ[batch: B, seq: T, model: D]
effect Rand                             -- (spec only) declares an effect
model GPT = stack{12}(Block) ∘ Embedding{V, D}   -- (spec only) module composition
module … { … }
fn … = …
optimizer NAME { … }
training run { … }
inference serve { … }
```

### Modules

```text
module Linear{In Out : dim} {
  param w : ℝ[In, Out]  init = xavier_uniform
  param b : ℝ[Out]      init = zeros
  fn apply(x : ℝ[..., In]) -> ℝ[..., Out] =
    x · w + b
}
```

- **Type parameters** `{In Out : dim}`, `{D : dim, hidden : dim}`, `{S :
  Shape}`: groups of names separated by spaces, each group followed by `:`
  and a kind (`dim` or `Shape`), groups separated by `,`. A module or
  function without type parameters simply omits the braces.
- **`param names : type [init = initializer]`** declares learnable tensors.
  Several names may share a declaration (`param a, b : ℝ[D]`). Both `param w
  : ℝ[…]` and `param w : param ℝ[…]` are accepted (the specification's
  examples use both). The `init =` clause names one of the four
  initializers below; without it the parameter starts at zero.
- **`use names : module-expr`** includes sub-modules: `use l1 : Linear{D,
  hidden}`, `use wq, wk, wv, wo : ff.Linear{D, D}`, `use blocks :
  stack{L}(Block)`. A module expression is a path, optional dimension
  arguments in braces, optional module arguments in parentheses, and may be
  composed with `∘`.
- **`fn`** members are functions of the module; inside a module body the
  module's parameters and sub-modules are in scope (`l1.apply(x)`).

### Initializers

The `init =` clause of a `param` says how the parameter is filled before
training. Its right-hand side is not an arbitrary expression and the names
are not keywords of the language: they are the fixed vocabulary of this
clause, and anything else is rejected with `E2013`. Version 0.1.0 has four:

| Clause | Value of the parameter | Random? |
|---|---|---|
| `init = zeros` | every entry is 0 | no |
| `init = ones` | every entry is 1 | no |
| `init = normal(std = s)` | independent draws from the normal distribution of mean 0 and standard deviation `s` (`s` a decimal literal, e.g. `0.02`) | yes |
| `init = xavier_uniform` | independent draws from the uniform distribution on `[−a, a]` with `a = √(6 / (fan_in + fan_out))`, where `fan_in` is the length of the first axis and `fan_out` that of the last (the two are the same axis for a vector, 1 and 1 for a scalar): Glorot and Bengio's initialization, which keeps the variance of activations and gradients of a linear layer roughly constant across depth | yes |
| *(no clause)* | as `zeros` | no |

Rules:

- **Randomness is the `Rand` effect.** The random initializers make the
  module's initializer function `M.init(key) -> params M` (generated for
  every module, see "PSK-Core") `uses Rand`; deterministic ones do not add
  the effect. The key passed to `M.init` is split once per `param` and per
  `use`, in declaration order, so every parameter draws from its own key:
  changing one initializer does not change the values of the others, and
  the same `--seed` gives the same parameters on every machine.
- **Exact values.** The draws are made in double precision and injected
  exactly into ℝ (the same rule as hyperparameters): the initializer denotes
  the real numbers that were drawn, and the interpreter carries them at
  `--bits`.
- **Where they run.** `presk run` calls `M.init` to synthesize a `params M`
  argument; `presk export` maps them to `torch.zeros`, `torch.ones`,
  `nn.init.normal_(std=…)` and `nn.init.xavier_uniform_` (PyTorch swaps
  the two fans of a matrix, which leaves `a` unchanged).
- **Not in 0.1.0**: any other distribution (`kaiming`, `orthogonal`, a
  truncated normal), a `std` given as a constant or an expression, or an
  arbitrary expression of the module's dimensions. The specification's
  grammar writes `init = expr` without naming the admissible expressions;
  the list above is this implementation's choice (`docs/spec-questions.md`,
  question 27).

### Functions

```text
fn mse{D : dim}(net : MLP{D, 4*D}, θ : params net, x : ℝ[batch: B, model: D],
                y : ℝ[batch: B, model: D]) -> ℝ[] =
  let ŷ = net.apply(θ, x)
  mean((ŷ - y)^2)

fn next_token(θ : params m.MiniGPT, ctx : idx[m.V][batch: B, seq: T],
              temperature : f64 = 0.8) -> idx[m.V][batch: B] uses Rand =
  …
```

`fn name[{typarams}](params) [-> type] [uses effects] = body`. A parameter is
`[axis] name [: type] [= default]`; the type may be omitted (`fn apply(x,
mask) = …`), a default value makes the parameter optional, and `axis i`
declares an axis-name parameter (used by the standard library's `concat`).
The return type and the `uses` clause may sit on their own lines. The body is
one expression.

### Configuration blocks

```text
optimizer adamw {
  lr           : f64 = 3e-4
  betas        : (f64, f64) = (0.9, 0.95)
  schedule     : cosine(warmup = 2000)
}

training run {
  model     = m.MiniGPT
  objective = loss
  optimizer = adamw
  data      = stream "data/tokens.psk.d" |> shuffle(seed = 1234) |> batch(B = 32)
  epochs    : nat = 10
  parallel  = data_parallel(replicas = auto) ⊗ tensor_parallel(axis model)
  precision = budget(rel_error = 1e-3)
  checkpoint every 1000 steps to "ckpt/"
}

inference serve {
  model     = m.MiniGPT
  weights   = "ckpt/final"
  precision = budget(abs_error = 1e-2) | quantize(int8, calibration = "data/calib.psk.d")
  target    = gpu(sm_90) | cpu(avx512)
  batching  = dynamic(max_batch = 64, timeout = 5ms)
  cache     = kv_cache(axis seq)
}
```

Items are `name = expr`, `name : type = expr`, or `name : expr` (the parser
tries a type first and keeps it only if `=` follows), one per line, plus
`checkpoint every N steps to "path"`. Field names may coincide with keywords
(`optimizer = adamw`). `stream "path"` is sugar for `stream("path")`. Field
names and value forms are checked against a schema per block (`E2010`, see
"Typing" below).

**The `training run` block against its objective** (0.2.0, step 2). Beyond
the schema, the checker resolves the block into a typed record (printed by
`presk check --summary` under `-- training --`):

- `objective` is a function of the file whose first parameter is `θ :
  params M` with `M` the `model`, and whose second parameter is the
  **batch**; any further parameter must have a default.
- The **sample type** is the batch type with the axis named `batch` removed
  from every tensor leaf: for `batch : (idx[V][batch: B, seq: T], idx[V][batch:
  B, seq: T])` it is `(idx[V][seq: T], idx[V][seq: T])`. A leaf without a
  `batch` axis, or with an open prefix, is an error.
- `data = stream "path" |> shuffle(seed = s) |> batch(B = n) [|> take(n =
  k)]`: `batch(B = n)` binds the `batch` axis of every leaf to `n` (so the
  objective's `B` is `n`); the path is relative to the `.psk` file. When
  the file exists it is opened and its sample type is matched against the
  objective's: literal dimensions must agree, symbolic ones (`T`) are bound
  from the file; a literal index bound (`V`) must be at least the file's,
  since samples of a smaller vocabulary are valid samples of a larger one,
  and a symbolic one is bound to it; a mismatch is `E2010` with both
  types and the offending axis. When the file is missing the check is
  deferred to `presk train` with the warning `W0002`.
- `epochs` and `checkpoint every N steps to "dir"` are recorded for the loop.

The `data` chapter defines the file format and the pipeline's semantics.

**The optimizer** (0.2.0, step 3). The name of an `optimizer` block is the
algorithm; `adamw` is the only one in 0.2.0 (`E2010` otherwise). For every
`training run`, the block is lowered to three ordinary PSK-Core functions
over `P = params M`, visible with `presk dump --stage core` and runnable
with `presk run --entry adamw.step`:

```text
adamw.init(θ : P) -> (P, P)                              -- moments at zero
adamw.lr(t : nat, total : nat) -> ℝ[]                    -- the schedule
adamw.step(θ : P, s : (P, P), g : P, t : nat, total : nat) -> (P, (P, P))
```

`step` is AdamW as PyTorch defines it: with `grad_clip = c` the gradient is
first scaled by `min(1, c / (‖g‖₂ + 1e-6))` (the global norm over the tree,
as `clip_grad_norm_`); then `m' = β₁ m + (1 − β₁) g`, `v' = β₂ v + (1 − β₂)
g²`, `m̂ = m' / (1 − β₁ᵗ)`, `v̂ = v' / (1 − β₂ᵗ)`, and `θ' = θ − lr(t, total) ·
(m̂ / (√v̂ + ε) + λ θ)` with the decoupled weight decay `λ`. `t` counts steps
from 1 and `total = epochs × batches per epoch`. Schedules: `constant()`;
`linear(warmup = w)`, `lr · t/w` for `t ≤ w` then `lr · (total − t)/(total −
w)`; `cosine(warmup = w)`, `lr · t/w` then `lr · ½(1 + cos(π (t − w)/(total −
w)))`, both reaching 0 at `total`. Defaults: `betas = (0.9, 0.999)`, `eps =
1e-8`, `weight_decay = 0`, no clipping, `constant()`. Hyperparameters are
decimal literals injected exactly into ℝ; the piecewise forms use `relu`
(`min(1, x) = 1 − relu(1 − x)`); `π` is the primitive `pi`, correctly
rounded at `--bits` (so the cosine schedule ends at 0 up to the working
precision). The functions use only primitives of the language plus
`sum_squares` (the squared norm of a tree) and `pi`, so the interpreter runs
them at `--bits` and the export compares
them with `torch.optim.AdamW` (`--conformance-entry adamw.step`). Tree-level
arithmetic (`β₁ · m`, `θ − lr · d`) is structural: a scalar applies to every
leaf of a parameter tree.

## Types

| Syntax | Meaning |
|---|---|
| `ℝ[axes]`, `ℝ[]`, `ℝ` | real tensor; `ℝ[]` and bare `ℝ` are the real scalar |
| `idx[N][axes]`, `idx[N]` | tensor of integers bounded by `0 ≤ i < N` |
| `param T` | a learnable tensor of type `T` |
| `params M` | the parameter tree of module `M` (a path: `params net`, `params m.MiniGPT`) |
| `nat`, `int`, `rat`, `f64`, `bool`, `key` | hyperparameter scalars (ordinary named types) |
| `Tokens`, `MLP{D, 4*D}`, `ff.Linear{D, D}` | a type alias or a module type, with optional dimension arguments |
| `(A, B)` | tuple |

An **axis list** is comma-separated; each axis is `dimexpr`, `name: dimexpr`,
or `...` (any number of leading axes): `ℝ[In, Out]`, `ℝ[batch: B, seq: T]`,
`ℝ[..., model: D]`. A dimension expression is an ordinary expression (`D`,
`4*D`, `D / H`, `m.V`); its arithmetic is interpreted in step 2.

## Expressions

Precedence, from loosest to tightest:

| Level | Operators | Notes |
|---|---|---|
| 1 | `\|>` | pipeline |
| 2 | `\|` | alternation / ordered combination |
| 3 | `⊗` `∘` | |
| 4 | `==` `<=` `>=` `<` `>` | |
| 5 | `+` `-` | |
| 6 | `*` `/` `%` `·` | |
| 7 | unary `-` | |
| 8 | `^` | right-associative, binds tighter than unary minus: `-x^2` is `-(x^2)` |
| 9 | postfix | call `f(…)`, index `t[…]`, dimension arguments `M{…}`, field `.name` |

All binary operators are left-associative except `^`.

**Forms.**

- **Names and paths:** `x`, `l1.apply`, `m.MiniGPT.apply`, `m.V`.
- **Calls** `f(args)` with four argument forms: positional `f(x)`; named
  `normal(std = 0.02)`; axis `mean(axis model, x)`, `softmax(axis seq2, s)`;
  named axis `positions(seq: T)`. `name : ℝ` after an identifier is an
  ascription, not a named axis: `sqrt(HD : ℝ)`.
- **`let pat = expr`** followed by the body on the next line; `pat` is a name
  or a tuple of names (`let (x, y) = batch`). Rebinding a name is allowed
  (`let x = x + …`).
- **`λ pat. body`** (or `\pat. body`), extending as far right as possible:
  `grad(λ θ. mse(net, θ, x, y))(θ)`.
- **`ein[axes](body)`**: Einstein summation with the result axes named:
  `ein[batch, heads, seq, seq2](q * k)`.
- **Indexing** `t[sel, …]` with positional `t[i]` or named `logits[seq: -1]`
  selectors.
- **Ascription** `(expr : type)`.
- **Tuples** `(a, b)`, `()`.
- **Dimension arguments** on a name: `Linear{D, D}`.
- Bare names such as `auto`, `int8`, `sm_90`, `xavier_uniform`, `zeros` are
  ordinary identifiers; their meaning is a matter of typing.

## Elaboration: what is resolved and checked today

After parsing, `presk check` elaborates the file and its imports. This is
the part of the semantics implemented so far; function *bodies* are still
untyped.

**Imports.** `import "path" as m` loads `path` relative to the importing
file, elaborates it first, and makes its constants, dimensions, type aliases
and modules available as `m.V`, `m.MiniGPT`, … A file is loaded once even if
imported twice; an import cycle is an error.

**Constants.** `const NAME [: nat] = expr` is evaluated exactly, in order of
declaration, from literals, earlier constants (own or imported) and `+ - * /`.
The result must be a natural number: `768 / 7` is rejected, `768 / 12` is
`64`. `--const NAME=N` on the command line replaces the value of a constant of
the main file. Constants are substituted everywhere they appear in
dimensions; they never remain symbolic.

**Dimensions.** A dimension expression is a *linear form* over the
symbolic dimensions in scope — `dim` declarations of the file, type
parameters `{D : dim}` of the enclosing module or function — with rational
coefficients: `4*D`, `D / 12`, `m + n`, `2*D - 1`. The product of two symbolic
dimensions is outside the fragment and rejected; so is a division by a
symbolic dimension (ℕ includes 0). Dividing a symbolic expression by a
constant `k` requires the fact `k | e` to be provable (typically from
`static assert e % k == 0`). In a *function* signature, an undeclared name
in a dimension position is accepted as an implicit dimension of that function,
with warning `W0001`; in a module it is an error.

**Facts and assertions.** `dim T <= N` records the fact `T ≤ N`.
`static assert` accepts `a == b`, `a <= b`, `a < b`, `a >= b`, `a > b` and
`a % k == 0` (`k` a positive constant) over dimension expressions. It is
decided for *all* values in ℕ of the symbolic dimensions: a failing assertion
reports a counterexample (`fails for T = 1`); an assertion the procedure
cannot decide is an error rather than a silent pass. A discharged symbolic
assertion becomes a fact for the rest of the file.

**Types.** `ℝ[axes]` and `idx[N][axes]` become shapes whose axis lengths are
linear forms; `...` must be the first axis. `nat int rat f64 bool key` are the
scalar types; `(A, B)` tuples; a type alias is substituted; `M{args}` names a
module instance, whose arity is checked; `param T` is `T`; `params m` denotes
the parameter tree of a module — either a module definition (`params
m.MiniGPT`) or a module-typed parameter declared earlier in the same
signature (`net : MLP{D, 4*D}, θ : params net`). `{S : Shape}` type
parameters are not supported.

**Modules and instances.** A module definition is elaborated once, with its
type parameters as symbolic dimensions: parameters must be real tensors,
`use` members are module expressions (`Linear{D, hidden}`, `ff.MLP{D, 4*D}`,
`stack{L}(Block)`, `A ∘ B`), member functions see the module's type
parameters. An *instance* is a module applied to dimension arguments; equal
instances are shared. The parameter tree of an instance substitutes the
arguments recursively through the `use`s, and `stack{L}(M)` prepends an axis
of length `L` to every parameter of `M`: `presk dump --stage elab` lists,
for `MiniGPT`, `blocks.attn.wq.w : ℝ[12, 768, 768]`.

**Configuration blocks** and `model` declarations are recorded but their
contents are not checked yet.

## Typing: what is checked in function bodies

Every function is checked once, generically: its type parameters, the file's
`dim`s and its `...` prefix are fixed unknowns. At a call, the callee's
signature is instantiated with fresh unknowns that unification then solves
from the arguments.

**Shapes.** Two shapes unify axis by axis. A *named* axis unifies only with
an axis of the same name or with an unnamed one; two named axes with
different names never unify even if their lengths agree (`seq` and `seq2`
are different axes). Lengths are dimension expressions: an inference
variable is solved, otherwise the decision procedure must prove the
equality (`D` vs `4*D` fails with a counterexample). A `...` prefix of a
callee absorbs the leading axes of the argument: `Linear.apply` with
`x : ℝ[..., In]` applied to `ℝ[batch: B, seq: T, model: 768]` binds the
prefix to `[batch: B, seq: T]`, `In` to `768`, and returns
`ℝ[batch: B, seq: T, Out]` (the result's last axis is unnamed; it unifies
with `model: D` at the next call). Inside the callee, `...` is one fixed
prefix shared by every occurrence in the signature, so
`mean(axis model, x) : ℝ[...]` and `x - μ` broadcast back to `ℝ[..., model: D]`.

**Broadcasting** (`+ - * / ^` between real tensors): a scalar `ℝ[]` combines
with anything; otherwise the axes are aligned from the last one (`x̂ * γ`
with `γ : ℝ[D]`, `s + mask` with `mask : ℝ[seq: T, seq2: T]`) and, if that
fails because of names, by name (every axis of the smaller operand named and
present in the larger). The result is the larger shape; names are kept. There
is no length-1 broadcasting.

**Operators.** `·` contracts the last axis of the left operand with the first
of the right; `^ n` takes a natural exponent; `- * + /` between numbers of
the same hyperparameter type (`nat`, `f64`, …) stay in that type; an `idx`
tensor never mixes with reals (`real(t)` injects it). Comparisons give `bool`.

**`ein[out](a * b)`.** The body is a product; every axis of every operand
must be named. Output names must occur in the operands; the other names are
contracted; equal names across operands must have equal lengths. When both
operands carry the same name `n` (`q * k` both have `seq`) and exactly one
*copy* `n<digits>` of it is unmatched — in the output but in no operand
(`seq2` for the scores) or in one operand but not in the output (`seq2` of
the attention weights against `v`) — the right operand's `n` is renamed to
that copy; more than one candidate is an error. This is the provisional
reading of the specification's `seq`/`seq2` convention.

**Calls.** Arguments are positional or named (`name = e`); a parameter with
a default may be omitted; `axis a` and `name: dim` are the axis forms.
`l1.apply(x)` inside a module passes no parameter tree (the sub-module's
parameters are part of the enclosing θ); `net.apply(θ, x)` on a module-typed
parameter and `m.MiniGPT.apply(θ, x)` on a module name pass θ explicitly as
first argument, of type `params M`. Through `stack{L}(M)`, `blocks.apply(x, …)`
is `M.apply` applied `L` times, so its result must have the type of its first
argument. A function without a declared return type gets its body's type, and
must be checked before its callers (declare the type otherwise).
`grad(λ θ. e)` takes a one-argument function returning `ℝ[]` and returns a
function of the same argument type; `grad(f)(θ)` applies it. Untyped lambda
parameters are inferred from use.

**Builtins** (with their typing): elementwise `gelu relu exp tanh sqrt abs
log sigmoid sin cos` (`ℝ[S] → ℝ[S]`); `mean(x)`, `sum(x)` (`→ ℝ[]`);
`mean(axis a, x)`, `sum(axis a, x)` (remove axis `a`); `softmax(axis a, x)`;
`real(n)` (`nat`/`f64` → `ℝ[]`) and `real(t)` (`idx[N][S] → ℝ[S]`);
`embed(w : ℝ[V, D], t : idx[V][S]) → ℝ[S, D]`; `cross_entropy(logits :
ℝ[S, V], y : idx[V][S]) → ℝ[S]`; `argmax`, `sample` (`ℝ[S, V] → idx[V][S]`);
`split_heads` (`ℝ[…, seq, model] → ℝ[…, heads: H, seq, hd: HD]`) and
`merge_heads`, which read the constants `H` and `HD` of the file;
`positions(seq: T) → idx[T][seq: T]`; `causal_mask(seq: T) → ℝ[seq: T, seq2: T]`;
`matmul`; `grad`. Not implemented: `vjp`, `jvp`, `concat`, `dropout`,
`clamp`, `max`, `min` (error `E2012`). Initializers are not builtins but
the vocabulary of the `init =` clause: see "Initializers" above.

**Configuration blocks** are checked against schemas: `optimizer` fields
`lr betas weight_decay grad_clip schedule eps`; `training run` fields `model
objective optimizer data epochs parallel precision` plus `checkpoint every N
steps to "path"` (`model`, `objective`, `optimizer` required; `objective`
must be a top-level function whose first parameter is `params <model>`;
`optimizer` must name an `optimizer` block); `inference serve` fields `model
weights precision target batching cache` (`model` required). Value forms:
`cosine(warmup = N)`, `stream "path" |> shuffle(seed = N) |> batch(B = N)`,
`data_parallel(replicas = auto|N) ⊗ tensor_parallel(axis a)`,
`budget(rel_error|abs_error = ε) | quantize(int8, calibration = "path")`,
`gpu(NAME) | cpu(NAME)`, `dynamic(max_batch = N, timeout = 5ms)`,
`kv_cache(axis a)`. Their semantics is not used.

**Not checked in 0.1.0:** the bound of a `dim T <= N` at a call site (the
callee's bound is not imposed on the caller's dimension).

## Regularity classes and effects

Both analyses run on PSK-Core, after lowering, and are part of `presk check`.

**Regularity.** Every function's result gets a class *per parameter*, on the
lattice `C∞ < C⁰⁺ < Disc`: the worst class of any operation on a path from
that parameter to the result. Primitives contribute per argument: `argmax`
and `sample` are `Disc` in everything; `embed` and `cross_entropy` are `C∞`
in the table/logits and `Disc` in the indices; `real(t)` of an index tensor
is `Disc`; `relu` and `abs` are `C⁰⁺`; everything else (`+ · ein exp tanh
softmax gelu sqrt mean …`) is `C∞`. Calls use the callee's summary, so
`loss` in `train_minigpt.psk` is `C∞` in `θ` and `Disc` in `batch`, and
`next_token` is `Disc` in all its parameters (it returns a sample). The
analysis is syntactic and sound, not exact: `relu(x) - relu(-x)` is the
identity but is classified `C⁰⁺`.

`grad(λ θ. e)` is legal only if `e` is at least `C⁰⁺` in `θ` (a gradient
defined almost everywhere). Otherwise `E3001` names the primitive
responsible and where it is, and suggests a substitute:

```text
[E3001] Error: cannot differentiate through `argmax`: the function is discontinuous (Disc) in `l`
   ,-[ tests/diag/reg_argmax.psk:6:3 ]
 4 |   mean(real(argmax(logits)))
   |             `------ `argmax` makes the result discontinuous in `l`
 6 |   grad(λ l. pick(l))(logits)
   |   `------------------------- `grad` here, with respect to `l`
   | Help: use `softmax` for a smooth surrogate, or an explicit straight-through relaxation
```

**Effects.** The only effect of 0.1.0 is `Rand`. A function *uses* `Rand`
if it calls `sample` or an initializer, or calls a function that does. The
inferred effects must be declared with `uses Rand` (`E4001` otherwise,
labelling the first use); a `uses` clause naming an unknown effect is
`E4002`. Initializers (`M.init`) are implicitly `uses Rand`; a lambda is
checked against the declaration of the function it appears in; `grad` of a
`Rand` function is allowed.

## PSK-Core: the intermediate representation

`presk dump --stage core` shows the program after *lowering*, the form on
which the interpreter (step 7) and the later analyses work. PSK-Core is a
first-order functional IR in A-normal form: every function body is a list of
`name : type = operation` bindings followed by `return`. Excerpt for
`mlp.psk`:

```text
fn Linear.apply{In Out}(θ : params Linear{In, Out}, x : ℝ[..., In]) -> ℝ[..., Out] {
  w : ℝ[In, Out] = θ.w
  %3 : ℝ[..., Out] = dot x, w
  b : ℝ[Out] = θ.b
  %5 : ℝ[..., Out] = add %3, b
  return %5
}
fn mse{D B}(θ : params MLP{D, 4*D}, x : ℝ[batch: B, model: D], y : ℝ[batch: B, model: D]) -> ℝ[] {
  ŷ : ℝ[batch: B, D] = call MLP.apply{D = D, hidden = 4*D}(θ, x)
  …
fn mse_grad{D B}(θ : params MLP{D, 4*D}, x : …, y : …) -> params MLP{D, 4*D} {
  %3 : (params MLP{D, 4*D}) -> ℝ[] = closure mse_grad.lambda0[x, y]
  %4 : (params MLP{D, 4*D}) -> params MLP{D, 4*D} = grad %3
  %5 : params MLP{D, 4*D} = apply %4(θ)
  return %5
}
```

What lowering does:

- **Functions stay generic in their dimensions** (`{In Out}`, `{D B}`):
  types mention the symbolic dimensions of the signature, and every `call`
  carries the instantiation of the callee's dimensions by the caller's
  expressions (`Linear.apply{In = D, Out = hidden}`). Shapes are therefore
  never expanded; the interpreter binds dimensions at run time.
- **The parameter tree is explicit.** A module function takes `θ : params M`
  as first parameter; inside the body a `param` is `θ.w` and a `use` member
  is `θ.l1`, passed as the `θ` of the sub-module's function. At top level
  `net.apply(θ, x)` passes θ itself. Module-typed parameters (`net : MLP{D,
  4*D}`) and arguments are compile-time and disappear.
- **`stack{L}(M)` is a `fold`** (0.2.0, step 4): `blocks.apply(h0, mask)`
  becomes `fold L Block.apply(stacked θ.blocks; acc h0; mask)`, a loop whose
  iteration `i` calls the body on slice `i` of the stacked parameter tree,
  the accumulator and the extra arguments; the program's size no longer
  depends on `L`, which may be symbolic in the IR. `presk dump --unroll-stack`
  and `presk run --unroll-stack` give 0.1.0's form, `L` calls on
  `slice_params i`, for comparison. A stacked module's *initializer* still
  splits its key `L` times and stacks `L` trees, so `L` must be a constant
  there (`E5002`).
- **Lambdas are lifted** (closure conversion): `grad(λ θ. mse(net, θ, x, y))`
  becomes a function `mse_grad.lambda0(x, y, θ)` whose free variables come
  first, a `closure` over `x, y`, a `grad` of that function value, and an
  `apply`. Automatic differentiation (step 6) rewrites `grad`.
- **One `init` function per module**, `M.init(key) -> params M`, built from
  the `init =` clauses (`zeros`, `ones`, `xavier_uniform`, `normal std=…`)
  and the sub-modules' `init`s, with the random key split once per field
  (`split_key n`); stacked sub-modules are initialized `L` times and
  combined with `stack_params L`.
- **Builtins become primitives** whose axes are designated by *name*
  (`mean axis model`, `softmax axis seq2`, `index seq: -1`), because runtime
  tensors carry the axis names of their types; `ein` records the relabeled
  axis names of each operand and of the output; `split_heads 12x64` carries
  the head constants; `real` injects a hyperparameter into `ℝ[]`.
- **Verification.** After lowering, a structural check (`presk_core::verify`)
  ensures every variable is defined once and before use, every call has the
  callee's arity and instantiates all its dimensions, and every block returns
  a defined variable. It runs before `dump --stage core` prints anything and
  in the tests.

## Automatic differentiation

`grad` is realized as a source-to-source transformation on PSK-Core
(`presk dump --stage core-after-ad` shows the result). A `fold` is
differentiated as a loop: the forward copy *scans* (it also returns the stack
of the accumulators before each iteration), and the reverse sweep emits a
backward `fold` running from `L − 1` down to `0` whose body, a generated
`….fold_bwd`, applies the body's `vjp` at iteration `i` on the saved
accumulator, passes the accumulator's cotangent to the previous iteration,
writes the stacked inputs' cotangent into slice `i` (`set_slice`) and sums
the extras'. It is *reverse mode*:
for a function `f(params…) -> ℝ[]`, a function `f.vjp(params…, ct)` returns
the adjoint of every parameter given the cotangent `ct` of the result, by
copying the forward body and appending the reverse sweep — for each binding,
from last to first, the cotangents of its arguments are computed from the
cotangent of its result and *accumulated* (a variable used twice receives the
sum). `grad v`, where `v = closure f[captured…]`, becomes
`closure f.grad[captured…]` with `f.grad(captured…, θ) = f.vjp(captured…, θ, 1).θ`.
The function values are then applied as before, so nothing but ordinary
PSK-Core remains and `verify` applies to the result.

Excerpt of `mlp.psk` after differentiation:

```text
fn Linear.apply.vjp{In Out}(θ : params Linear{In, Out}, x : ℝ[..., In], ct : ℝ[..., Out])
    -> (params Linear{In, Out}, ℝ[..., In]) {
  w : ℝ[In, Out] = θ.w
  %4 : ℝ[..., Out] = dot x, w
  b : ℝ[Out] = θ.b
  %6 : ℝ[..., Out] = add %4, b
  %7 : ℝ[..., Out] = sum_to ct, %4        -- adjoint of the broadcast add
  %8 : ℝ[Out] = sum_to ct, b
  %9 : params Linear{In, Out} = zeros_like θ
  %10 : params Linear{In, Out} = set_field b %9, %8
  %11 : ℝ[..., In] = dot_vjp_l %7, w       -- ∂x
  %12 : ℝ[In, Out] = dot_vjp_r x, %7       -- ∂w
  %13 : params Linear{In, Out} = zeros_like θ
  %14 : params Linear{In, Out} = set_field w %13, %12
  %15 : params Linear{In, Out} = add %10, %14
  %16 : (params Linear{In, Out}, ℝ[..., In]) = (%15, %11)
  return %16
}
```

- **Callees** get their own `.vjp`, generated on demand and once
  (`mse.vjp` calls `MLP.apply.vjp`, which calls `Linear.apply.vjp`); functions
  off the differentiated path (`Linear.init`) get none.
- **Parameter trees** are differentiated structurally: the adjoint of `θ.w`
  is `zeros_like θ` with field `w` set (`set_field`), and adjoints of trees
  are added with the structural `add`; the adjoint of a projection uses
  `set_elem`; `slice_params` and `stack_params` have `scatter_params` and
  `slice_params` as adjoints.
- **Broadcasting** is undone by `sum_to` (sum down to the operand's shape) and
  reductions by `broadcast_like`/`mean_vjp`; `softmax`'s rule is
  `y · (ct − Σ_axis(ct · y))`; `ein`'s adjoint with respect to one operand is
  another `ein` of the cotangent with the other operands; `embed` and
  `cross_entropy` use dedicated `embed_vjp`/`cross_entropy_vjp`
  (scatter-add and `softmax − onehot`); `gelu`, `relu`, `abs` use `gelu'`,
  `heaviside`, `sign`.
- **No adjoint** flows into hyperparameters (`nat`, `f64`), indices, keys, or
  through `argmax`/`sample`; the regularity analysis has already rejected
  `grad` where that would be a mistake.
- **Not supported in 0.1.0** (`E6001`/`E6002`): second-order differentiation
  (`grad` inside a differentiated function, derivatives of the adjoint
  primitives), `grad` of a function value that is not a lambda of the same
  body, `vjp`/`jvp` as source-level operators.

The interpreter (step 7) executes the adjoint primitives; until then the
transformation is validated by `verify` and by the snapshot of `mlp.psk`.

## Running programs: the reference interpreter

`presk run` evaluates PSK-Core after automatic differentiation on an
interpreter over arbitrary-precision reals (MPFR through `rug`, `--bits`,
default 256). It is the executable approximation of the exact denotation: the
error of one operation is about 2⁻²⁵⁶ and every operation is correctly
rounded, so results are deterministic across machines and serve as the
reference for everything else (golden tests, the coming PyTorch conformance
test).

- **Values** are tensors with *named axes* over MPFR reals, index tensors
  (`i64` with their bound), hyperparameters (`nat`, `f64`), keys, tuples,
  parameter records and function values. After every binding the tensor's axis
  names are set from the binding's type, so primitives select axes by name
  (`mean axis model`, `softmax axis seq2`, `index seq: -1`) exactly as the
  types say.
- **Dimensions** are runtime values: a call instantiates the callee's
  dimensions from `dim_args`; the entry's dimensions come from `--dim`;
  initializers (`zeros`, `positions`, `causal_mask`) materialize shapes from
  their types under that environment.
- **Broadcasting** mirrors the type checker: the smaller operand aligns with
  the larger from the trailing axis when names agree, else by name; the
  adjoint `sum_to` reduces along the same alignment.
- **Exact functions**: `gelu` is `x·Φ(x)` with `Φ` from `erf`; `softmax`
  subtracts the row maximum before `exp` (a value-preserving rewrite, since
  the semantics is exact); `causal_mask` uses `−∞`, which `exp` maps to 0;
  `cross_entropy` is `−log softmax(logits)[y]` per position.
- **Dimension arguments** of a call or a fold are evaluated in the caller;
  one the checker left unconstrained (a file-level `dim` that the callee's
  types do not mention) is left unbound, and only a use of it fails.
- **Randomness** is reproducible: `--seed` derives a master ChaCha20 key,
  split into one key per entry parameter; `normal(std)` and `xavier_uniform`
  draw from their key; `sample` draws from a stream seeded by `--seed`.
- **`training run`** is not executable (no data reader, no optimizer
  semantics in 0.1.0): `presk run` without `--entry` on such a file exits
  with 3. Its functions (`loss`) run with `--entry`.

Sanity checks the interpreter passes: on the tiny MiniGPT (`--const V=64 …`)
`loss` evaluates to `4.15888308336… = ln 64`, the cross-entropy of uniform
logits, since the output head is initialized to zero; the gradient of `mse`
computed by AD agrees with central finite differences at 512 bits
(`h = 2⁻¹²⁰`) to 2⁻¹⁰⁰ on every component of the MLP's parameters.

## PyTorch export

`presk export --target pytorch --mode readable` is a printer over the
*typed program* (not over PSK-Core), so the Python is isomorphic to the
source. From `mlp.psk`:

```python
class Linear(nn.Module):
    """Generated from `module Linear` (mlp.psk:5). Do not edit by hand."""

    def __init__(self, In, Out):
        super().__init__()
        self.In = In
        self.Out = Out
        self.w = nn.init.xavier_uniform_(nn.Parameter(torch.empty(self.In, self.Out)))  # ℝ[In, Out]
        self.b = nn.Parameter(torch.zeros(self.Out))  # ℝ[Out]

    def forward(self, x):
        """Generated from `fn apply` (mlp.psk:9)."""
        # x : ℝ[..., In]
        return x @ self.w + self.b

def mse(net, θ, x, y):
    # net : MLP{D, 4*D}  (a module, passed as an nn.Module)
    # θ : params MLP{D, 4*D}
    B = x.shape[0]
    D = x.shape[1]
    ŷ = torch.func.functional_call(net, θ, (x,))  # ℝ[batch: B, D]
    return ((ŷ - y) ** 2).mean()
```

Correspondences: `const` → module-level constant (with `--const` overrides
applied); `static assert` → `assert`; `dim`/`type` → comments; module type
parameters → constructor arguments stored as attributes; `param … init = …`
→ `nn.Parameter` with `nn.init.*`; `use` → sub-module attribute; `stack{L}(M)`
→ `nn.ModuleList` and a `for` (or `L` calls with `--stack=unroll`), hoisted into a temporary when the stacked call is nested in an expression; `param`s
whose shape mentions a file dimension (`pos : ℝ[T, D]`) make that dimension a
constructor argument, propagated through the modules that use them
(`MiniGPT(T)`); `fn apply` → `forward`; module-typed parameters stay Python
parameters (an `nn.Module`); `params M` → a state dict, and `net.apply(θ, x)`
→ `torch.func.functional_call(net, θ, (x,))`; `grad(λ θ. e)(θ)` →
`torch.func.grad(lambda θ: e)(θ)`; `uses Rand` → a `generator=None` keyword.
Symbolic dimensions of a function are bound from the arguments' shapes
(`T = ids.shape[1]`) where a value is needed (`torch.arange(T)`, the causal
mask, shape assertions).

Expressions: `·` → `@`; `^n` → `** n`; `ein[…](a * b)` → `torch.einsum` with
one letter per axis name (after the relabeling of the type checker);
`mean(axis a, x)` → `x.mean(dim=k)` with `k` from the axis position, counted
from the end when the type has an open prefix; `softmax` likewise; `embed` →
indexing; `positions(seq: T)` → `torch.arange(T)`; `causal_mask` →
`torch.triu(torch.full(…, -inf), 1)`; `split_heads`/`merge_heads` → two small
helpers emitted in every file; `cross_entropy` → `F.cross_entropy(…,
reduction="none")` per position; `argmax`, `sample` (`torch.multinomial`);
`real(t)` of an index tensor → `.to(default dtype)`; a hyperparameter
injected into `ℝ` (`sqrt(HD : ℝ)`) stays a Python number (`math.sqrt`).
Presk broadcasts by leading prefix or by name where PyTorch broadcasts from
the trailing axis: the exporter inserts `unsqueeze`s where the two differ
(`x - μ` with `μ : ℝ[...]` becomes `x - μ.unsqueeze(-1)`).

Not exported: `training run`, `optimizer`, `inference serve` are reproduced
as a commented block so that nothing is silently dropped; the error budget is
a comment; dtypes are those of the inputs (PyTorch's default `float32`, or
`float64` in the conformance test). `--mode graph` and `--precision=solved`
are not in 0.1.0.

## What the parser accepts but the specification leaves open

See `docs/spec-questions.md` in the repository: the two `param` forms, the
grammar of configuration blocks, the layout rule, the `name: e` / `name : ℝ`
distinction in calls, the contextual keywords, and the rule that `-` never
continues a line.
