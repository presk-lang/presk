# The language, as accepted by this version

This chapter describes the syntax the parser accepts and the static
semantics implemented: elaboration (names, constants, dimensions,
assertions, module instances, signatures), shape typing of function bodies
("Typing"), regularity classes and effects, and the intermediate
representation ("PSK-Core"). The reference programs are in
`examples/spec/`; the normative text is the language reference of
`presk-specs` (`reference/presk-reference.tex`), whose Status paragraphs
say where this compiler departs from it. Where the specification left a
choice, the one made here is recorded in `docs/spec-questions.md`, and the
reference's chapter 8 states what became of each.

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
(`logits[seq: -1]` parses as a negated `1`). `..` is the delimiter of a
slice's bounds (0.17.0 step 6): since `1.` is no literal, `0..1` is three
tokens, `0`, `..` and `1`, and `...` stays one.

**Layout.** Line breaks are significant. A declaration, a module member or a
configuration field ends at the end of its line. An expression ends at the
end of its line unless the *next* line starts with a binary operator that
cannot itself begin an expression (`|>`, `|`, `⊗`, `∘`, `·`, `*`, `/`, `%`,
`+`, `==`, `<=`, …); such a line continues the expression. `-` never
continues a line, since `-x` is an expression, and neither does `^`, which
the parser reads with the unary operators and not as a binary operator (the
reference's requirement on line breaks, chapter 1; this chapter listed it
until 0.10.0). Inside parentheses, brackets
and dimension-argument braces, line breaks are ignored, which is how a long
function signature or call spans several lines. A `let` binding takes the rest
of its line and its body starts on the next line (there is no `in`).

## Program structure

A file is a sequence of declarations.

<!-- psk: none (an overview of the declarations; `effect` and `model` are the specification's only) -->
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

<!-- psk: examples/tutorial/01_linear.psk#Linear -->
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

<!-- psk: none (two signatures, the body of `next_token` elided) -->
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

<!-- psk: none (abridged from examples/spec/minigpt/train_minigpt.psk and serve_minigpt.psk: fields left out) -->
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
- `parallel` and `precision` pass the schema and are not read. Since 0.6.0
  step 7b, `precision = budget(rel_error = ε)` has a meaning — it binds one
  optimizer step's parameter update, measured against the exact step
  (`training` chapter, D52) — but the number is given to `presk solve --step
  --budget ε` on the command line, and the realization it writes is passed
  to `presk train --realize`.

The `data` chapter defines the file format and the pipeline's semantics.

**The optimizer** (0.2.0, step 3). The name of an `optimizer` block is the
algorithm; `adamw` is the only one in 0.2.0 (`E2010` otherwise). For every
`training run`, the block is lowered to three ordinary PSK-Core functions
over `P = params M`, visible with `presk dump --stage core` and runnable
with `presk run --entry adamw.step`:

<!-- psk: none (the signatures the optimizer block lowers to) -->
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
| `bool[axes]` | tensor of truth values, a mask (0.10.0 step 7b): `causal_mask` gives one and `softmax(…, where = m)` reads one; `bool` is a contextual word, and without axes the hyperparameter scalar below |
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
  selectors, which drop the axis they select along. A selector with `..` is
  a **slice**, `p[pair: 0..1]` or `t[0..2]` on the first axis, which keeps
  its axis at the length `hi − lo` (0.17.0 step 6; the reference's rule
  E-Slice). The bounds are dimensions — literals, constants, dimensions of
  the file or of the function, and their sums and differences — and
  `presk check` proves `0 ≤ lo ≤ hi ≤` the axis's length for every value of
  them (`E2014` with a counterexample otherwise, `E1007` for a bound that is
  not a dimension). Selectors of both kinds mix in one index, left to
  right, on real and on index tensors: `w[seq: 1..T + 1]` is the targets of
  a window of `T + 1` tokens (`examples/spec/concat/`).
- **Ascription** `(expr : type)`.
- **Tuples** `(a, b)`, `()`.
- **Dimension arguments** on a name: `Linear{D, D}`.
- Bare names such as `auto`, `int8`, `sm_90`, `xavier_uniform`, `zeros` are
  ordinary identifiers; their meaning is a matter of typing.

## Elaboration: what is resolved and checked today

After parsing, `presk check` elaborates the file and its imports. This is
the first half of the static semantics; function bodies are typed after it
("Typing", below).

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
cannot decide is an error rather than a silent pass. The assertions of a
file are decided last, after its constants, dimensions and types, which do
not see them: an assertion adds no fact. (The reference reads a symbolic
assertion as a constraint on the file's dimensions; the compiler refuses
one that does not hold for every value, `E1004`.)

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
for `MiniGPT`, `blocks.attn.wq.w : ℝ[12, 768, 768]`. A module that includes
itself, directly or through other modules, has no parameter tree and is
refused (`E2016`, since 0.10.0 step 8), as a function that calls itself is
(below).

**Configuration blocks** and `model` declarations are recorded but their
contents are not checked yet.

## Typing: what is checked in function bodies

Every function is checked once, generically: its type parameters, the file's
`dim`s and its `...` prefix are fixed unknowns. At a call, the callee's
signature is instantiated with fresh unknowns that unification then solves
from the arguments. Once every call is resolved, a function that calls
itself, directly or through other functions, is refused (`E2016`, the
reference's requirement "No recursion"; iteration is `stack{N}(M)`): until
0.10.0 step 8 such a program passed the checker and overflowed the stack of
the analyses that follow calls.

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
with `γ : ℝ[D]`, `s * w` with `w : ℝ[seq: T, seq2: T]`) and, if that
fails because of names, by name (every axis of the smaller operand named and
present in the larger). The result is the larger shape; names are kept. There
is no length-1 broadcasting.

**Operators.** `·` contracts the last axis of the left operand with the first
of the right; `^ n` takes a natural exponent; `- * + /` between numbers of
the same hyperparameter type (`nat`, `f64`, …) stay in that type, except
that `nat - nat` and `nat / nat` are `int`, the quotient the integer
quotient (`real(7 / 2)` is `3`); an `idx`
tensor never mixes with reals (`real(t)` injects it), and a mask `bool[S]`
enters no arithmetic. Comparisons give `bool`.

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
`mean(axis a, x)`, `sum(axis a, x)` (remove axis `a`); `softmax(axis a, x)`,
and `softmax(axis a, x, where = m)` with a mask `m : bool[S2]` that
broadcasts into `x : ℝ[S]` without widening it (S ⊔ S2 = S): the softmax
over the positions `m` allows, 0 where it refuses (0.10.0 step 7b);
`real(n)` (`nat`/`f64` → `ℝ[]`) and `real(t)` (`idx[N][S] → ℝ[S]`);
`embed(w : ℝ[V, D], t : idx[V][S]) → ℝ[S, D]`, whose new axis takes the name
of the table's second axis when it has one (`param tok : ℝ[vocab: V, emb: E]`
gives `ℝ[S, emb: E]`) and is unnamed otherwise; `one_hot(t : idx[V][S]) →
ℝ[S, V]`, the indicator vectors whose contraction with `w` `embed` denotes
(its new axis is always unnamed, since the bound is a dimension, not a named
axis, and is named by the signature it is passed to); `cross_entropy(logits :
ℝ[S, V], y : idx[V][S]) → ℝ[S]`; `argmax(x : ℝ[S, V]) → idx[V][S]` and
`sample(key : key, logits : ℝ[S, V]) → idx[V][S]`, which draws from its key
(0.10.0 step 7b; "Running programs" below);
`merge(axis a, axis b, x)`, which joins two axes into one of length `|a|·|b|`,
row-major (`(i, j) ↦ i·|b| + j`): the two must already be adjacent and in that
order, because joining distant axes would also transpose and `merge` moves no
data, and at least one length must be constant, since the product of two
symbolic dimensions leaves the decidable fragment (`E1007`). The joined axis
is unnamed, and is named by whatever signature it is passed to.
`split(axis a, n1: L1, n2: L2, x)`, its inverse: one axis read as two,
row-major (`k ↦ (k / L2, k % L2)`), with `L1 · L2 = |a|` (`E2006` otherwise)
and at least one of them constant (`E1007`). Unlike `merge`, **a split names
both halves**, because they are new axes the program has to refer to and only
the author knows what they mean; the consequence is that two splits chain
where two merges cannot, since a merge leaves nothing for a second merge to
select. Both are pure retags: they move no data and launch no kernel.
`split_heads` (`ℝ[…, seq, model] → ℝ[…, heads: H, seq, hd: HD]`) and
`merge_heads`, which read the constants `H` and `HD` of the file and, unlike
`merge` and `split`, also permute — which is why they are not sugar for the
pair, and why a head layout with a shared key group is written with `split`
and not with them (`coverage` chapter);
`positions(seq: T) → idx[T][seq: T]`; `causal_mask(seq: T) → bool[seq: T, seq2: T]`,
true where `seq2 ≤ seq` (it was a real tensor of 0 and `−∞` added to the
scores until 0.10.0 step 7b, and no tensor holds `−∞` since);
`matmul`; `grad`; `dropout(rate, key, x)` and `split_key(key) → (key, key)`,
below;
`max(x, y)`, `min(x, y)` and `clamp(x, lo, hi)` (0.17.0 step 6), of real
tensors element by element, broadcast as `+` broadcasts: at a tie `max` and
`min` give their second operand and its derivative, and `clamp` is
`min(max(x, lo), hi)`, so a hyperparameter bound is injected first,
`clamp(s, (0 : ℝ), (1 : ℝ))`;
`concat(axis a, x, y)`, which joins two real tensors along an axis both
carry by name, their shapes equal but for its length (`E2008` for an
operand without the axis: there is no implicit axis of length one, and the
slice `e[a: i..i+1]` gives one; `E2003` for shapes that differ otherwise).
Not implemented: `vjp`, `jvp` (error `E2012`). Initializers are not builtins but
the vocabulary of the `init =` clause: see "Initializers" above.

**Configuration blocks** are checked against schemas: `optimizer` fields
`lr betas weight_decay grad_clip schedule eps`; `training run` fields `model
objective optimizer data epochs parallel precision` plus `checkpoint every N
steps to "path"` (`model`, `objective`, `optimizer` required; `objective`
must be a top-level function whose first parameter is `params <model>`;
`optimizer` must name an `optimizer` block); `inference serve` fields `model
weights precision target batching cache generate` (`model` required;
`generate` names a top-level function of the file whose signature is a
generation function's, since 0.10.0 step 8; the block is resolved with the
file's generation function, the one `generate` names or the first that
fits, `serve` chapter, `W0003` without one). Value forms:
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
and `sample` are `Disc` in everything, and so is `one_hot`, whose only
argument is an index; `embed` and `cross_entropy` are `C∞`
in the table/logits and `Disc` in the indices, and `softmax` in its scores
and its mask; `real(t)` of an index tensor
is `Disc`; `relu`, `abs`, `max`, `min` and `clamp` are `C⁰⁺`; everything else (`+ · ein exp tanh
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
   ,-[ examples/tests/diag/reg_argmax.psk:6:3 ]
 4 |   mean(real(argmax(logits)))
   |             `------ `argmax` makes the result discontinuous in `l`
 6 |   grad(λ l. pick(l))(logits)
   |   `------------------------- `grad` here, with respect to `l`
   | Help: use `softmax` for a smooth surrogate, or an explicit straight-through relaxation
```

**Dropout takes a key.** `dropout(rate, key, x)` multiplies `x` by an
inverted-dropout mask — `0` with probability `rate`, `1/(1 − rate)`
otherwise — drawn from `key` *and nothing else*. The key is in the
signature, not in a hidden stream, and that is a correctness requirement
rather than a style: differentiation may recompute the forward pass (`--remat
recompute`, and a `.vjp` the split into `.fwd` and `.bwd` leaves whole), so
a mask drawn from a stream would differ between the forward and the backward
and the gradient would be silently wrong. Being a function of the key, the
mask recomputes identically, and `dropout` needs no differentiation rule of
its own — it lowers to `x · dropout_mask(rate, key, x)`, and the adjoint of
a product is already known; a gradient that keeps its residuals keeps the
mask. `split_key(key)` gives two independent keys, so a function can drop
more than once:

<!-- psk: examples/snippets/language/split_key.psk#block -->
```text
fn block(x : ℝ[batch: B, model: D], key : key) -> ℝ[batch: B, model: D] uses Rand =
  let (k1, k2) = split_key(key)
  let h = dropout(0.1, k1, attn(x))
  dropout(0.1, k2, ff(h))
```

Outside a training run the mask is all ones, which is the expectation of the
random function `dropout` denotes: `presk run`, `presk serve` and the export
evaluate the model with every mask at its expectation — the usual inference
rule, which is the model's mean only where its output is affine in each mask —
and only `presk train` drops, and the exported `train.py`, with masks
PyTorch draws (D54). The mode is part of what an entry means (the
reference's requirement 3.30, question 78): since 0.16.0 (D160) an entry
compiled outside a training run is its inference reading, a program in which
`dropout(rate, key, x)` is `x`, with no `dropout_mask` and no product by it,
so that no realization types a mask and no solver offers one; until then the
program was the same in both modes and the executors filled the mask with
ones at launch. A training run lowers the program as written. `presk dump
--mode training` prints that one, the default the inference one. An objective may
take a per-step key between `θ` and the batch (`training` chapter).

**A key inside `stack{L}`** needs one idiom, and it is worth stating because
the obvious thing is wrong. A stacked module is a fold, and a fold hands
every iteration the same extra arguments, so a key passed *as an argument* is
one key for all `L` blocks: every layer drops the same positions, silently.
Carry it in the accumulator instead, which is the value the fold threads from
one iteration to the next:

<!-- psk: examples/snippets/language/stack_key.psk#acc -->
```text
type Acc = (ℝ[batch: B, model: D], key)

module Block {
  fn apply(s : Acc, mask : Mask) -> Acc uses Rand =
    let (x, k)   = s
    let (k1, k2) = split_key(k)
    (x + dropout(0.2, k1, attn(x, mask)), k2)
}
```

The caller passes `(x, key)` and projects the result. Nothing new is needed
for this, it differentiates through the ordinary fold adjoint, and
`examples/corpus/nlp/05_gpt.psk` uses it (D37).

**Effects.** The only effect is `Rand`. A function *uses* `Rand`
if it calls `sample`, `dropout`, `split_key` or an initializer, or calls a
function that does. Each of them draws from a key it is given and from
nothing else (`sample` since 0.10.0 step 7b). The
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

<!-- psk: none (PSK-Core, as presk dump --stage core prints it) -->
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
  there (`E5002`). With `L = 0` the initializer stacks no tree: every leaf
  of `M` gets a leading axis of length 0, built from the stacked type, and
  the fold of count 0 gives its initial value; a gradient's scan over it
  stacks the accumulators of no iteration the same way (since 0.10.0 step
  8, on every executor and in the C export; the initializer and the kernel
  backend's host panicked on a stack of none before).
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
  (`mean axis model`, `softmax axis seq2`, `index seq: -1`, `slice pair:
  0..1`, `concat pair`), because runtime tensors carry the axis names of
  their types, and a slice's bounds are dimensions, given values where the
  function is called; `clamp` lowers to a `max` and a `min`; `ein` records the relabeled
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

Excerpt of `mlp.psk` after differentiation, as `presk dump --stage
core-after-ad` prints it (below, the dead code it has lost since 0.10.0
step 3):

<!-- psk: none (PSK-Core after differentiation, as presk dump --stage core-after-ad prints it) -->
```text
fn Linear.apply.vjp{In Out}(θ : params Linear{In, Out}, x : ℝ[..., In], ct : ℝ[..., Out])
    -> (params Linear{In, Out}, ℝ[..., In]) {
  w : ℝ[In, Out] = θ.w
  b : ℝ[Out] = θ.b
  %8 : ℝ[Out] = sum_to ct, b               -- adjoint of the broadcast add
  %9 : params Linear{In, Out} = zeros_like θ
  %10 : params Linear{In, Out} = set_field b %9, %8
  %11 : ℝ[..., In] = dot_vjp_l ct, w       -- ∂x
  %12 : ℝ[In, Out] = dot_vjp_r x, ct       -- ∂w
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
  `heaviside`, `sign`. Since 0.17.0 step 6, `max x1, x2` gives `x1` the
  cotangent times `heaviside(x1 − x2)` and `x2` the rest, so a tie goes to
  the second operand (`min` likewise with `heaviside(x2 − x1)`); a `concat`'s
  adjoint is two `slice`s of the cotangent, and a slice's is the cotangent
  placed into zeros, `slice_vjp` (`examples/tests/data/concat_grad.psk` checks them
  against finite differences at 512 bits).
- **No adjoint** flows into hyperparameters (`nat`, `f64`), indices, keys, or
  through `argmax`/`sample`; the regularity analysis has already rejected
  `grad` where that would be a mistake.
- **Lambdas applied in a differentiated body** (`let f = λ a. prod(a, y)`,
  then `f(x)`) are differentiated through the function they call, their
  captured values included (fixed in 0.10.0 step 3,
  `examples/tests/data/closure_grad.psk`).
- **Not supported** (`E6001`/`E6002`): second-order differentiation
  (`grad` inside a differentiated function, derivatives of the adjoint
  primitives), `grad` of a function value that is not a lambda of the same
  body, applying in a differentiated function a function value that is not
  a lambda of its body, `vjp`/`jvp` as source-level operators.

The interpreter (step 7) executes the adjoint primitives; until then the
transformation is validated by `verify` and by the snapshot of `mlp.psk`.

### Dead code after differentiation

Since 0.10.0 step 3 the functions a differentiation generates lose what
nothing reads before anything runs them (`presk_core::ad::prune`; every
command that differentiates gets it). A `.vjp` copies its function's
forward body before the reverse sweep, and a `.grad` reads one of the
cotangents a `.vjp` returns; much of that is never read.

- **Forward values the sweep does not need**: the primal value of the
  function `grad` differentiates, which `grad` discards (`mse_grad`'s call
  of `mse` in `mse_grad.lambda0.vjp`), and a callee's value that only the
  callee's own `.vjp` computes again (`MLP.apply.vjp`'s second product).
- **Cotangents no caller reads**: a `.vjp` returns the cotangents its
  callers read, in their order, and no others. `mse_grad.lambda0.grad`
  reads `θ`'s, so `mse_grad.lambda0.vjp`, `mse.vjp` and `MLP.apply.vjp`
  return `(params MLP{D, 4*D})`, and `x`'s and `y`'s are not computed. A
  backward fold keeps the accumulators someone reads (not the cotangent of
  the mask a stacked block receives). A function read outside the program
  keeps its result whole: a `.grad`, and the functions `presk train` calls.
- **Values read for their shape**: `sum_to x, t` and `broadcast_like x, t`
  between equal types are `x` (the excerpt's `dot_vjp_l ct, w` read `sum_to
  ct, %4`, and the forward `%4 = dot x, w` and `%6 = add %4, b` went with
  it); and a primitive that reads an operand for its shape only (the
  template of `sum_to`, `mean_vjp`, `dropout_mask`, `zeros_like`,
  `scatter_index`, `embed_vjp`, `reshape_like`) reads another value of that
  type the function computes anyway, when there is one whose type names
  every axis (in the course GPT's `Block.apply.vjp`, `dropout_mask` read
  the feed-forward's output for its shape only; it reads the second
  normalization's, which the feed-forward's `.vjp` reads anyway, and the
  forward feed-forward goes).

What stays is computed as before, bit for bit, in every realization. Every
draw is a function of its key (`sample`'s since 0.10.0 step 7b), so a draw
nothing reads goes like any other `let`. Functions keep their names and bindings their numbers, so a
nest (`Linear.apply.vjp/%11`) keeps its name.

**Fusion groups, under `bf16` or a site typed apart.** A fusion group
(`realize` chapter) computes every value it holds in its root site's
arithmetic type and stores only the root. So a value inside a group stays
in `f32` under `bf16`, where a value outside every group reaches memory in
`bf16`; and a value whose site a realization types apart from the rest of
its function (a `storage` entry) takes, inside a group, its group's type.
Under such a realization removing a reader that kept a value out of a group
would change a result, and the pass then keeps every group of what the
result reads: it holds what would change one, an identity left in place, a
dead `let` kept as the reader it was. On the course GPT's gradient that is
68 of the 2 220 launches. Under a uniform `f32` or `f64` realization, or
one by function, every value of a function is computed and stored in one
type, a group changes no bit, and since 0.10.0 step 4b nothing is held
(`Realization::groups_visible`, D94). A command decides by the
realizations it runs: `presk run`, `export`, `train`, `plan`, `autotune`,
`reprice` and `dump` by their `--realize` or `--plan`; `presk bench` holds
unless it times its plan alone (`--plan-only`, `--calibrate`), since its
rows include `bf16`; `presk solve --step` holds, its menu being `bf16` too;
`presk calibrate --step` holds when one of its realizations would, or when
it writes an envelope (`--out`), which `solve --step` reads. A plan made
under one names nests the other's program lacks, and is refused
(`E7002`).

What it removes, counted by the C export's recorder (`presk export --target
c`, `f32`):

| Gradient | Launches before | After | Of them dead before, after |
|---|---|---|---|
| `mse_grad` (`mlp.psk`, `D = 4`, `B = 2`) | 46 | 28 | 15, 2 |
| `loss_grad` (`examples/tests/plan/minigpt_grad.psk`, the tiny constants) | 767 | 583 | 125, 4 |
| `loss_grad` (`examples/experiments/gpt_grad.psk`, the course GPT) | 2 220 | 1 616 | 386, 12 |

More goes than the recorder counted dead, since it counts a read for a
shape as a read. What it still counts dead is what the pass holds for a
fusion group (the counts are `bf16`'s, holding), and in `mse_grad` the
first layer's `∂x`: one
`Linear.apply.vjp` serves both layers, and the second layer's caller reads
its `∂x`. `presk-opt` (`--opt`) removes
the dead code it finds too, without regard to fusion groups. (The counts
are those of `--remat recompute`; the next section's are storing.)

### Residuals: a gradient keeps its forward pass (`--remat`)

A `.vjp` copies its function's forward body, so a gradient computes each
callee's forward pass twice: once for the value its caller reads, and again
inside the callee's `.vjp`, at every level of calls. Since 0.10.0 step 4 a
gradient keeps what its forward pass computed instead (`--remat store`, the
default of `presk run`, `export`, `train`, `bench`, `plan`, `autotune`,
`reprice` and `dump`; `--remat recompute` keeps the copies). After the dead code has
gone (`presk_core::ad::store`), each `.vjp` is split in two:

- **`f.fwd`** runs `f`'s body once and returns its value and its
  **residuals**, the values its sweep reads that `f` computes: the inputs of
  the primitives' rules, and each callee's residuals, as it calls each
  callee's `.fwd`. Its bindings are `f`'s, numbered as in `f`, so a site of
  `f.fwd` is `f`'s site of the same number: a realization or a plan that
  states a type or a layout for `f/%5` states it for `f.fwd/%5`, unless it
  names `f.fwd/%5` itself.
- **`f.bwd`** takes the parameters of `f` its sweep reads, the residuals, and
  the cotangent, and runs the sweep, calling each callee's `.bwd` on that
  callee's residuals. Its bindings are the `.vjp`'s, numbered as in the
  `.vjp`, so the sweep's nests keep their numbers under another name
  (`Linear.apply.vjp/%11`, the `dot_vjp_l` of a linear layer, is
  `Linear.apply.bwd/%11`).
- **`f.vjp`** is `f.bwd` after `f.fwd`, and `.grad` still calls it.
  `presk train`'s `objective.value_vjp` is `f.fwd`'s value and `f.bwd`'s
  cotangents: the loss it logs is `f.fwd`'s, computed as `f` computes it.

<!-- psk: none (PSK-Core after differentiation, as presk dump --stage core-after-ad prints it) -->
```text
fn MLP.apply.fwd{D hidden}(θ : params MLP{D, hidden}, x : ℝ[..., D])
    -> (ℝ[..., D], (ℝ[..., hidden], ℝ[..., hidden])) {
  l1 : params Linear{D, hidden} = θ.l1
  %3 : ℝ[..., hidden] = call Linear.apply{In = D, Out = hidden}(l1, x)
  %4 : ℝ[..., hidden] = gelu %3
  l2 : params Linear{hidden, D} = θ.l2
  %6 : ℝ[..., D] = call Linear.apply{In = hidden, Out = D}(l2, %4)
  %7 : (ℝ[..., hidden], ℝ[..., hidden]) = (%3, %4)      -- gelu's input, l2's
  %8 : (ℝ[..., D], (ℝ[..., hidden], ℝ[..., hidden])) = (%6, %7)
  return %8
}
```

`Linear.apply` needs no `.fwd`: its sweep reads only its parameters (`x`
and `w`), which its caller passes to `Linear.apply.bwd`. Projections,
tuples and key splits are computed again in `.bwd` rather than kept, and
so is a value the sweep's own fusion groups hold inside.

**A fold stacks its layers' residuals.** A stacked module's forward fold
carries, in its accumulator, the residuals of the layer before (zeros
before the first), and scans: the stack of the accumulators before each
iteration then holds, at slot `i + 1`, layer `i`'s residuals, and the final
accumulator the last layer's. The backward fold carries the residuals of
the layer it differentiates and hands the next its own from the stack. A
key in the accumulator is stacked as the sequence of the layers' keys, as
a tensor is stacked as its slices: until 0.10.0 step 4 a stack kept the
first key for every layer, and the backward fold of a stacked module that
drops gave every layer layer 0's key, a wrong gradient under training
(`examples/tests/data/stack_dropout_grad.psk`).

**The same bits.** Every gradient is bit-identical under `store` and
`recompute`, in `f64`, `f32` and `bf16`. A realization that can see a
fusion group (the previous section: `bf16`, or a site typed apart) is where
it takes care: a group computes its members in its root's type, `f32`
under `bf16`, so a residual must be computed in the same groups as the
`.vjp`'s copy computed it, and `f.fwd`'s value in `f`'s. Where the two
partitions differ, `f.fwd` computes the residual apart, a copy of the
`.vjp`'s computation, beside `f`'s own: a normalization's intermediate the
sweep reads, and, in the course GPT, attention's scores and the
feed-forward after the second normalization, which the launches the
dead-code pass holds made the `.vjp` round where the block does not. Under
any other realization (since 0.10.0 step 4b) a group changes no bit, and
`f.fwd` shares every residual with `f`'s computation, reading it where
`f`'s groups kept it inside: nothing is computed apart. A `.vjp` whose
split cannot be made exact keeps recomputing: one whose fold's body is
generic in its leading axes (`...`, `examples/tests/data/stack_prefix_grad.psk`'s
tower), since no zeros have the first slot's shape, and a function that
reaches itself.

Launches, counted by the C export's recorder, keeping the groups (what
`bf16` runs) and not (what `f32` and `f64` run):

| Gradient | `recompute`, `bf16` | `store`, `bf16` | `recompute`, `f32` | `store`, `f32` |
|---|---|---|---|---|
| `mse_grad` (`mlp.psk`, `D = 4`, `B = 2`) | 28 | 24 | 26 | 23 |
| `loss_grad` (`examples/tests/plan/minigpt_grad.psk`, the tiny constants) | 577 | 412 | 563 | 385 |
| `loss_grad` (`examples/experiments/gpt_grad.psk`, the course GPT) | 1 598 | 1 140 | 1 548 | 1 012 |

What a realization states by function names the functions of the program
as it is: `presk solve --step` names `.fwd` and `.bwd` functions under
`store`, and a realization written for `.vjp`s types nothing of a stored
gradient's sweep, which `W0004` says, naming each such `.vjp` whose
`.fwd` and `.bwd` the realization does not name (a warning, since a
realization for a training step also runs forward entries; since 0.10.0
step 4b). A plan made under one policy names nests the other does
not have (`E7002`).

## Reading a program as mathematics

`presk dump --stage elab --format unicode` (or `latex`) prints what a
program denotes, as the language reference's chapter 3 writes it (its
section 3.10, "Programs as equations"; 0.13.0 step 4, D137): each function
is the equations of its `let`s, then its value `⟦f⟧(x₁, …)`. Nothing is
rounded and no order of summation is chosen: it is the mathematics the
realizations are measured against. For `examples/tutorial/03_attention.psk`:

```text
scores, of q : Q,  k : Q
    ⟦scores⟧(q, k)[batch, heads, seq, seq2] = (∑_{hd} q[batch, heads, seq, hd] k[batch, heads, seq2, hd]) / √HD

attend, of q : Q,  k : Q,  v : Q
    s = ⟦scores⟧(q, k)
    a = softmax axis seq2 s, (causal_mask seq: T)
    ⟦attend⟧(q, k, v)[batch, heads, seq, hd] = ∑_{seq2} a[batch, heads, seq, seq2] v[batch, heads, seq2, hd]
```

An operation keeps its name (`softmax axis seq2`, `dot x, w`), and is never
written as what it lowers to. An `ein` is written as its sum, at an index
named by the axes of its result: the second operand's `seq` of the
attention became `seq2` by the relabeling of the reference's definition
2.39, and the sum shows it. A module's function is printed once, its
parameters by their names (`w` for `θ.w`), a sub-module's call as
`⟦proj.apply⟧(x)`, and a call through `stack{L}` as its recurrence, `y₀ =
x`, `y_(ℓ+1) = ⟦Block.apply⟧(θ.blocks[ℓ], y_ℓ)` for `0 ≤ ℓ < L`. A `let`
that rebinds a name is primed: `let x = x + …` is `x' = x + …`.

`--stage core-after-ad` does the same for PSK-Core after
differentiation, one equation per binding, and says what each binding of a
generated function was made for, read off the provenance table (D134). The
reverse sweep of `Linear.apply`, from `examples/tutorial/04_grad.psk`:

```text
Linear.apply.bwd, of θ : params Linear{In, Out},  x : ℝ[…, In],  ct : ℝ[…, Out]
    b = θ.b                               -- primal b, kept
    %8 = sum_to ct, b                     -- adjoint of %5 (add)
    %9 = zeros_like θ                     -- adjoint of b (field)
    %10 = set_field b %9, %8              -- adjoint of b (field)
    %12 = dot_vjp_r x, ct                 -- adjoint of %3 (dot)
    …
```

`--format latex` writes the same as a fragment in the reference's macros,
which `--standalone` makes a document; `make math-check` compiles those of
the tutorial, the examples and the course GPT. The pipeline view of
`presk explain --format html` shows the same formulas on its cards
(`explain` chapter).

## Running programs: the reference interpreter

`presk run` evaluates PSK-Core after automatic differentiation on an
interpreter over arbitrary-precision reals (MPFR through `rug`, `--bits`,
default 256). It is the executable approximation of the exact denotation: the
error of one operation is about 2⁻²⁵⁶ and every operation is correctly
rounded, so results are deterministic across machines and serve as the
reference for everything else — up to the last bit of the machine's `ln`,
`sqrt` and `cos` in a value drawn from a key (a normal draw is computed in
binary64 with them, then injected exactly) (golden tests, the coming PyTorch conformance
test). Its own distance to the denotation is measured, not bounded: about
3·10⁻³⁸ relative at 128 bits and 4–8·10⁻⁷⁷ at 256 on the course GPT, five to
ten units of roundoff at `--bits` (`make delta-p`; the `precision` chapter,
"The reference's own gap"; D98).

- **Values** are tensors with *named axes* over MPFR reals, index tensors
  (`i64` with their bound; a mask is indices below 2, true 1, which `presk
  run` prints by the entry's signature as `bool[…]` of `true` and `false`,
  in its text and its `--json`, since 0.10.0 step 8; `idx[2][…]` before),
  hyperparameters (`nat`, `f64`), keys, tuples,
  parameter records and function values. After every binding the tensor's axis
  names are set from the binding's type, so primitives select axes by name
  (`mean axis model`, `softmax axis seq2`, `index seq: -1`) exactly as the
  types say.
- **Memory** (0.10.1, D105). An exact real at 128 bits is 64 bytes, and a
  model's parameters are most of what a run holds, so the interpreter
  holds each value once and no longer than it is read. A tensor's elements
  are shared by every value that reads them and copied only when one is
  written, so a field of a parameter record, a call's argument, a stacked
  layer a fold reads and the values that cross to the other executors
  cost no copy; a value is freed after the binding that reads it last,
  and a call is handed the values its caller reads there for the last
  time; zeros are one buffer per length, and adding them gives the other
  operand itself where MPFR gives it. Every result is the same, bit for
  bit, as before (U1): `presk run --bits 128` holds each parameter once
  where 0.10.0 held it twice, and `presk solve` holds the parameters and
  their cotangent, where it kept every partial cotangent of its sweep.
- **Dimensions** are runtime values: a call instantiates the callee's
  dimensions from `dim_args`; the entry's dimensions come from `--dim`;
  initializers (`zeros`, `positions`, `causal_mask`) materialize shapes from
  their types under that environment.
- **Broadcasting** mirrors the type checker: the smaller operand aligns with
  the larger from the trailing axis when names agree, else by name; the
  adjoint `sum_to` reduces along the same alignment.
- **Exact functions**: `gelu` is `x·Φ(x)` with `Φ` from `erf`; `softmax`
  subtracts the row maximum before `exp` (a value-preserving rewrite, since
  the semantics is exact), and with a mask takes its maximum and its sum over
  the positions the mask allows and gives 0 elsewhere, a row the mask
  empties being an error of the run; over an axis of length 0 it is defined,
  and empty; `cross_entropy` is `−log softmax(logits)[y]` per position.
- **Undefined at the dimensions given** (0.10.0 step 7b): before anything
  is computed, an entry whose body (or a function it calls, a closure it
  applies, a fold of count at least one) holds an operation undefined at
  the dimensions given is refused, `E7003`: a `mean` over no element, a
  selection outside its axis, `embed`, `cross_entropy`, `argmax` or `sample`
  with no index below `V`, or their derivatives; so is an entry a type of
  whose parameters has no value (indices below 0). The check reads types,
  not values, on every executor; a division by zero stays an error of the
  run.
- **Dimension arguments** of a call or a fold are evaluated in the caller;
  one the checker left unconstrained (a file-level `dim` that the callee's
  types do not mention) is left unbound, and only a use of it fails.
- **Randomness** is reproducible: `--seed` derives a master ChaCha20 key,
  split into one key per entry parameter; `normal(std)`, `xavier_uniform`,
  `dropout` and `sample` draw from their key (the reference's definitions
  "The generator" and "Draws"; `sample` drew from a stream seeded by
  `--seed` until 0.10.0 step 7b). `sample(key, logits)` takes, at the
  position of rank `p`, the least index whose cumulative probability exceeds
  the key's uniform draw `p`, the draw a dyadic number of 53 bits, the
  probabilities and their sums at the interpreter's precision; the native
  backend sums them in `f64`, so a token can differ only where a draw falls
  within that rounding of a cumulative sum. A run is deterministic across
  machines up to the last bit of `ln`, `sqrt` and `cos` in a drawn value
  (`normal`).
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
`torch.func.grad(lambda θ: e)(θ)`; `uses Rand` → the keyword
`training=False` (the switch of `dropout`), passed on to every `uses Rand`
callee: left out, the function is its expectation. A key stays a parameter,
as in the program, and is read by nothing: `dropout`'s masks and
`sample`'s tokens are drawn by PyTorch's own generator, not from the key,
and differ from the compiler's (the keyword `generator=` went with 0.10.0
step 7b).
Symbolic dimensions of a function are bound from the arguments' shapes
(`T = ids.shape[1]`) where a value is needed (`torch.arange(T)`, the causal
mask, shape assertions).

Expressions: `·` → `@`; `^n` → `** n`; `ein[…](a * b)` → `torch.einsum` with
one letter per axis name (after the relabeling of the type checker);
`mean(axis a, x)` → `x.mean(dim=k)` with `k` from the axis position, counted
from the end when the type has an open prefix; `softmax` likewise; `embed` →
indexing; `positions(seq: T)` → `torch.arange(T)`; `causal_mask` →
`torch.ones((T, T), dtype=torch.bool).tril()`; `softmax(axis a, x, where = m)`
→ `torch.softmax(x.masked_fill(~m, float("-inf")), dim=k)`, the mask aligned
as Presk aligns it; `split_heads`/`merge_heads` → two small
helpers emitted in every file; `cross_entropy` → `F.cross_entropy(…,
reduction="none")` per position; `argmax`, `sample` (`torch.multinomial` of
the softmax, PyTorch's draws);
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

## What the specification left open

`docs/spec-questions.md` in the repository lists every point where the
examples or the specification left this compiler a choice (the two `param`
forms, the grammar of configuration blocks, the layout rule, the `name: e`
/ `name : ℝ` distinction in calls, the contextual keywords, the rule that
`-` never continues a line, and seventy more), each with the answer
implemented. The language reference has settled most of them: its chapter
8, "Open questions", is the list of record, and the file's status column is
that chapter's table (closed, closed otherwise than the compiler, restated,
open, or outside the language), marked at 0.10.0 step 8.
