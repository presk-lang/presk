# Tutorial: from a first module to a PyTorch export

This chapter teaches `presk` by doing, and doubles as the end-to-end test of
the tool: every command below is run by `make tutorial`, and every output
shown is what version 0.1.0 prints. The programs written here are in the
`tutorial/` directory of the repository; the reference programs of the
specification are in `tests/examples/`.

What you will do, in order:

1. build the tool and check a program;
2. write a module with parameters and run it on the reference interpreter;
3. learn how shapes are typed, and read the errors when they are not;
4. use constants, static assertions and bounded dimensions;
5. take a gradient, and see why some functions cannot be differentiated;
6. declare randomness as an effect;
7. compose modules with imports and stacks;
8. look at the intermediate representations;
9. export to PyTorch and run the generated conformance test;
10. run the reference examples;
11. train a model, on the interpreter and in PyTorch;
12. run the whole test suite.

Commands are run from the root of the repository. Presk source files use
Unicode symbols (`ℝ`, `λ`, `·`, `θ`); each has an ASCII synonym (`Real`,
`\`, `@`, and any identifier for `θ`), see the end of step 1. The
compiler prints warnings and errors on standard error, results on standard
output; `NO_COLOR=1` turns colors off.

## 0. Build and first check

Requirements: a Rust toolchain (rustc ≥ 1.85), `make`, `m4` and a C
compiler (the first build compiles GMP and MPFR, a few minutes). See the
`install` chapter.

```sh
make build
alias presk=target/debug/presk        # or: cargo run -q -p presk --
presk --help
```

```text
Presk compiler (0.2.0: front-end, reference interpreter, training, PyTorch export)

Usage: presk <COMMAND>

Commands:
  check   Type-check a program: shapes, regularity classes, effects
  run     Evaluate an entry point on the reference interpreter
  export  Export a program to another framework
  dump    Dump an intermediate representation (development aid)
  train   Run the `training run` of a file on the reference interpreter
  data    Produce and inspect `.psk.d` data files
```

Check the smallest reference program, a two-layer perceptron:

```sh
presk check tests/examples/mlp/mlp.psk
```

```text
[W0001] Warning: implicit dimension `B`
    ,-[ tests/examples/mlp/mlp.psk:26:65 ]
    |
 26 | fn mse{D : dim}(net : MLP{D, 4*D}, θ : params net, x : ℝ[batch: B, model: D],
    |                                                                 |
    |                                                                 `-- not declared; taken as a dimension of this function
    |
    | Help: declare it with `dim B` at file level or as a type parameter `{B : dim}`
----'
```

The program is correct (exit code 0): the checker only warns that `B` is
used as a dimension without being declared, and says how to declare it. Read
a diagnostic from its code (`W` warning, `E` error, then a range per phase:
`E0` syntax, `E1` elaboration, `E2` typing, `E3` regularity, `E4` effects,
`E7` run time, `E8` export), then its labels, then its help line. The
`diagnostics` chapter lists them all.

## 1. A first module: parameters and a function

A Presk program describes a mathematical function over tensors of real
numbers. There are no machine number types: `ℝ` is the real line, and how it
is realized (float32, bfloat16…) is a compilation choice made later, under an
error budget. Shapes are part of the types, with named or symbolic axes.

`tutorial/01_linear.psk`:

```text
-- 01_linear.psk — a linear layer: parameters, initializers, one function.
module Linear{In Out : dim} {
  param w : ℝ[In, Out]  init = xavier_uniform
  param b : ℝ[Out]      init = zeros

  fn apply(x : ℝ[..., In]) -> ℝ[..., Out] =
    x · w + b
}
```

- A **module** is a parameterized definition; `{In Out : dim}` are its
  dimension parameters, fixed when the module is instantiated
  (`Linear{3, 2}`).
- A **`param`** is a learnable tensor with its shape and, after `init =`,
  its initializer: how it is filled before training. `xavier_uniform` and
  `zeros` are not keywords of the language nor functions you can call; they
  are two of the four names this clause accepts, with `ones` and
  `normal(std = 0.02)` (a parameter without `init =` starts at zero).
  `xavier_uniform` draws uniformly in `[−a, a]` with `a = √(6 / (In +
  Out))`, the classical choice for a linear layer; the exact definitions
  are in the `language` chapter, section "Initializers". The parameters of
  a module form its *parameter tree*, written `θ` below.
- **`fn apply`** is the module's function. `ℝ[..., In]` is a tensor whose
  last axis has length `In` and whose leading axes (`...`, the *prefix*) are
  whatever the caller passes: a batch, a batch and a sequence…
- `·` contracts the last axis of `x` with the first axis of `w`; `+ b`
  broadcasts `b` over the leading axes.

Check it:

```sh
presk check tutorial/01_linear.psk        # silent: no diagnostic, exit code 0
```

Now run it. The interpreter needs an *entry point*: which function, and for a
module function, with which dimension arguments.

```sh
presk run tutorial/01_linear.psk
```

```text
presk run: --entry is required; functions: Linear.apply
```

```sh
presk run tutorial/01_linear.psk --entry 'Linear{3, 2}.apply' --dim batch=4 --verbose
```

```text
-- entry Linear{3, 2}.apply with In = 3, Out = 2 at 256 bits, seed 0
θ : params = {
  w = [3, 2] [[-0.983257327109, 1.01328364301], [0.616220484208, -0.0351536766703], [0.541220283065, -0.348589323961]]
  b = [2] [0, 0]
}
x : ℝ[batch: 4, 3] = [batch: 4, 3] [[-1.32784076889, -0.388929350549, -0.313462770017], [0.0468872897054, 0.313691170614, 1.27964233377], [0.622224852609, -0.320194124359, -0.307660772078], [2.30440571969, -0.65592691234, 0.278896360269]]
-- result
ℝ[batch: 4, 2] = [batch: 4, 2] [[0.896290523413, -1.22253725991], [0.839769040003, -0.409586930304], [-0.975629573942, 0.748993526688], [-2.519075041, 2.26085457153]]
```

What happened:

- `Linear{3, 2}` fixes `In = 3`, `Out = 2`. The parameter tree `θ` was
  produced by the module's initializers: `w` is uniform in `[−a, a]` with
  `a = √(6/5) ≈ 1.095` (`xavier_uniform` with `In + Out = 5`), `b` is zero.
  Each parameter draws from its own random key, derived from `--seed`.
- `x : ℝ[..., 3]` has an open prefix; `--dim batch=4` gives it one axis,
  named `batch`, of length 4. Without it the interpreter cannot build `x`:

  ```text
  [E7001] Error: cannot synthesize `x`: cannot materialize a shape with an open prefix `...` (give its axes to `presk run` with extra `--dim` names)
  ```

- Inputs are synthesized from the types: real tensors from a standard normal
  distribution, index tensors uniformly, hyperparameters from their default.
  Everything derives from `--seed` (default 0): the same command prints the
  same digits on every machine, because the arithmetic is MPFR at `--bits`
  (default 256) with correct rounding.
- `--verbose` prints the inputs; `--digits` sets the printed precision;
  `--json` gives the same data as JSON (this is what feeds the PyTorch
  conformance test of step 8).

```sh
presk run tutorial/01_linear.psk --entry 'Linear{3, 2}.apply' --dim batch=4 --seed 1 --bits 64 --digits 6
```

**ASCII spelling.** The same module can be written without Unicode; the two
spellings are the same program, and diagnostics print the Unicode form:

```text
module Linear{In Out : dim} {
  param w : Real[In, Out]  init = xavier_uniform
  param b : Real[Out]      init = zeros
  fn apply(x : Real[..., In]) -> Real[..., Out] =
    x @ w + b
}
```

## 2. Shapes and named axes

`tutorial/02_norm.psk` normalizes the last axis of its input:

```text
-- 02_norm.psk — named axes, reductions, broadcasting, hyperparameters.
module RMSNorm{D : dim} {
  param γ : ℝ[D]  init = ones

  fn apply(x : ℝ[..., model: D], eps : f64 = 1e-6) -> ℝ[..., model: D] =
    let ms = mean(axis model, x^2)        -- ℝ[...]: the `model` axis is reduced
    x / sqrt(ms + real(eps)) * γ          -- ms broadcasts back, γ aligns on `model`
}
```

- An axis can be **named**: `model: D`. Reductions name the axis they act
  on, `mean(axis model, x^2)`, so the code says what it means and the checker
  can verify it; the result `ms` has the prefix only, `ℝ[...]`.
- `let` binds a name for the rest of the body; a `let` ends at the end of
  its line and the body follows (there is no `in`).
- **Broadcasting** aligns the shorter operand by trailing position or, when
  its axes are all named, by name: `ms : ℝ[...]` against `x : ℝ[..., model:
  D]` aligns the prefixes; `γ : ℝ[D]` aligns on the last axis.
- `eps : f64 = 1e-6` is a **hyperparameter**: a machine number, not a real,
  and `real(eps)` is its explicit injection into `ℝ`. Hyperparameters are
  the only place machine types appear, and they are exact constants of the
  model, not numerical guards.

`--summary` prints what the checker inferred, for each function and each
`let`:

```sh
presk check tutorial/02_norm.psk --summary
```

```text
02_norm.psk RMSNorm.apply : ℝ[..., model: D]
    let ms : ℝ[...]
-- regularity and effects --
RMSNorm.apply : [θ C∞, x C∞, eps C∞] uses none
RMSNorm.init : [key C∞] uses Rand
```

The second part is explained in steps 5 and 6. Run it:

```sh
presk run tutorial/02_norm.psk --entry 'RMSNorm{4}.apply' --dim batch=2 --digits 6
```

```text
ℝ[batch: 2, model: 4] = [batch: 2, model: 4] [[-1.8709, -0.547993, -0.441662, 0.0660632], [0.420543, 1.71553, 0.834173, -0.429261]]
```

### When shapes do not match

The checker's job is to reject programs whose shapes are wrong, for *all*
values of the symbolic dimensions, and to say why in the terms of the
source. Try these variants in a scratch file.

Two axes of the same length but different names are different axes:

```text
dim B
fn add(a : ℝ[batch: B, model: 4], b : ℝ[time: B, model: 4]) -> ℝ[batch: B, model: 4] =
  a + b
```

```text
[E2003] Error: cannot broadcast `ℝ[batch: B, model: 4]` with `ℝ[time: B, model: 4]`
   ,-[ bad_axis.psk:3:3 ]
   |
 3 |   a + b
   |   ^^|^^
   |     `---- no alignment by position or by name
   |
   | Note: axes align from the last one, or by name when every axis of the smaller operand is named
---'
```

A wrong return type, in `Linear` (`-> ℝ[..., In]` instead of `Out`). The
dimensions are symbolic, so the checker gives a counterexample:

```text
[E2006] Error: dimension mismatch: `In` vs `Out`
   ,-[ bad_ret.psk:5:5 ]
   |
 5 |     x · w + b
   |     ^^^^|^^^^
   |         `------ differ for In = 1, Out = 0
---'
[E2011] Error: `apply` returns `ℝ[..., Out]` but is declared `-> ℝ[..., In]`
   ,-[ bad_ret.psk:5:5 ]
   |
 4 | ,->   fn apply(x : ℝ[..., In]) -> ℝ[..., In] =
 5 | |->     x · w + b
   | |       ^^^^|^^^^
   | |           `------- this body has type `ℝ[..., Out]`
   | |
   | `------------------- declared `ℝ[..., In]`
---'
```

A name that is not in scope (`x · w + c`):

```text
[E2001] Error: unresolved name `c`
   ,-[ bad_unbound.psk:5:13 ]
   |
 5 |     x · w + c
   |             |
   |             `-- not a variable, parameter, module member or constant in scope
---'
```

A forgotten `=` before a function body:

```text
[E0010] Error: expected `=`, found identifier `x`
   ,-[ bad_syntax.psk:4:5 ]
   |
 4 |     x · w
   |     |
   |     `-- expected `=` here
---'
```

## 3. Constants, assertions and bounded dimensions

`tutorial/03_attention.psk` computes causal attention scores:

```text
-- 03_attention.psk — constants, static assertions, bounded dimensions, ein.
const D  : nat = 8          -- model dimension
const H  : nat = 2          -- heads
const HD : nat = D / H      -- must be exact: checked at elaboration
static assert D % H == 0

dim B
dim T <= 1024

type Q = ℝ[batch: B, heads: H, seq: T, hd: HD]

fn scores(q : Q, k : Q) -> ℝ[batch: B, heads: H, seq: T, seq2: T] =
  ein[batch, heads, seq, seq2](q * k) / sqrt(HD : ℝ)

fn attend(q : Q, k : Q, v : Q) -> Q =
  let s = scores(q, k)
  let a = softmax(axis seq2, s + causal_mask(seq: T))
  ein[batch, heads, seq, hd](a * v)
```

- **`const`** values are evaluated exactly at compile time, in ℚ: `D / H`
  must be a natural number. `nat` includes 0.
- **`static assert`** states a property of the constants and dimensions;
  it is decided for *all* values of the symbolic dimensions by a small
  decision procedure for linear arithmetic with divisibility (Presburger
  arithmetic), the same one that decides shape equalities.
- **`dim B`** declares a symbolic dimension of the file; **`dim T <= 1024`**
  bounds it.
- **`type`** names a shape. **`ein[out](a * b)`** is a typed contraction:
  the output axes are named, the others are summed. `q` and `k` both have a
  `seq` axis; the checker relabels the right operand's `seq` to the output's
  `seq2`, of equal length, so `scores` is the `seq × seq2` matrix per head.
- `sqrt(HD : ℝ)` injects the constant into `ℝ`; `causal_mask(seq: T)` is 0
  on `j ≤ i` and −∞ elsewhere.

```sh
presk check tutorial/03_attention.psk --summary
```

```text
03_attention.psk scores : ℝ[batch: B, heads: 2, seq: T, seq2: T]
03_attention.psk attend : ℝ[batch: B, heads: 2, seq: T, hd: 4]
    let s : ℝ[batch: B, heads: 2, seq: T, seq2: T]
    let a : ℝ[batch: B, heads: 2, seq: T, seq2: T]
-- regularity and effects --
scores : [q C∞, k C∞] uses none
attend : [q C∞, k C∞, v C∞] uses none
```

Constants can be overridden from the command line; the assertions are
re-checked. With three heads, `HD = 8 / 3` is not a natural number and the
assertion fails (the first and last of the errors printed):

```sh
presk check tutorial/03_attention.psk --const H=3
```

```text
[E1005] Error: `HD` is not a natural number
   ,-[ tutorial/03_attention.psk:4:18 ]
   |
 4 | const HD : nat = D / H      -- must be exact: checked at elaboration
   |                  ^^|^^
   |                    `---- evaluates to 8/3
   |
   | Note: constants are evaluated exactly: a division must be exact and the result non-negative
---'
…
[E1004] Error: static assertion fails: D % H == 0
   ,-[ tutorial/03_attention.psk:5:1 ]
   |
 5 | static assert D % H == 0
   | ^^^^^^^^^^^^|^^^^^^^^^^^
   |             `------------- `D % H == 0` is false with the constants of this file
---'
```

Running needs the file dimensions `B` and `T`; the bound on `T` is enforced:

```sh
presk run tutorial/03_attention.psk --entry attend --dim B=1 --dim T=3 --digits 5
presk run tutorial/03_attention.psk --entry attend --dim B=1 --dim T=2000
```

```text
ℝ[batch: 1, heads: 2, seq: 3, hd: 4] = [batch: 1, heads: 2, seq: 3, hd: 4] [[[[-0.26306, -1.2272, -0.52286, -0.6502], [-0.45598, -1.0327, -0.21638, -0.55957], …
[E7001] Error: `dim` bound violated: `T <= 1024` is false for T = 2000
```

## 4. Differentiation and regularity classes

`tutorial/04_grad.psk` defines a loss over the linear layer and its gradient:

```text
-- 04_grad.psk — a loss, its gradient, and regularity classes.
import "01_linear.psk" as lin

fn loss{D B : dim}(net : lin.Linear{D, 1}, θ : params net,
                   x : ℝ[batch: B, D], y : ℝ[batch: B, 1]) -> ℝ[] =
  let ŷ = net.apply(θ, x)
  mean((ŷ - y)^2)

fn loss_grad{D B : dim}(net : lin.Linear{D, 1}, θ : params net,
                        x : ℝ[batch: B, D], y : ℝ[batch: B, 1]) -> params net =
  grad(λ θ. loss(net, θ, x, y))(θ)
```

- **`import "…" as lin`** loads another file, relative to this one; its
  declarations are reached with `lin.`.
- **`net : lin.Linear{D, 1}`** is a module-typed parameter: a compile-time
  argument, the module to use. **`θ : params net`** is its parameter tree.
  Inside a module, `apply` sees its own parameters; outside, they are passed
  explicitly as the first argument: `net.apply(θ, x)`.
- Declaring `{D B : dim}` avoids the `W0001` warning of step 0.
- **`grad(f)`** is the gradient of a scalar function; `λ θ. …` is an
  anonymous function that captures `x` and `y`. The gradient of a function of
  a parameter tree is a parameter tree of the same shape.

```sh
presk check tutorial/04_grad.psk --summary
```

```text
01_linear.psk Linear.apply : ℝ[..., Out]
04_grad.psk loss : ℝ[]
    let ŷ : ℝ[batch: B, 1]
04_grad.psk loss_grad : params Linear{D, 1}
-- regularity and effects --
Linear.apply : [θ C∞, x C∞] uses none
loss : [θ C∞, x C∞, y C∞] uses none
loss_grad : [θ C∞, x C∞, y C∞] uses none
Linear.init : [key C∞] uses Rand
loss_grad.lambda0 : [x C∞, y C∞, θ C∞] uses none
```

The **regularity class** of a function, per argument, says how smooth it is:
`C∞` (smooth), `C⁰⁺` (continuous, differentiable almost everywhere, like
`relu` or `abs`), `Disc` (discontinuous, like `argmax` or an index). `grad`
requires `C⁰⁺` or better in the differentiated argument, and the checker
proves it: `loss` is `C∞` in `θ`, so `loss_grad` is accepted.

```sh
presk run tutorial/04_grad.psk --entry loss --dim D=3 --dim B=2 --digits 6
presk run tutorial/04_grad.psk --entry loss_grad --dim D=3 --dim B=2 --digits 6
```

```text
ℝ[] = 3.47881
params = {
  w = [3, 1] [[-1.30262], [0.342146], [2.7531]]
  b = [1] [3.47884]
}
```

(The gradient of a mean square error with respect to a zero bias is twice
the mean residual, which is why `b`'s gradient is close to the loss itself
here.) The gradient is computed by reverse-mode automatic differentiation on
the intermediate representation, and validated in the test suite against
finite differences at 512 bits.

### A function that cannot be differentiated

Replace the loss by the mean of an `argmax`:

```text
fn hits{D : dim}(net : lin.Linear{D, 8}, θ : params net, x : ℝ[batch: B, D]) -> ℝ[] =
  mean(real(argmax(net.apply(θ, x))))
fn hits_grad{D : dim}(net : lin.Linear{D, 8}, θ : params net, x : ℝ[batch: B, D]) -> params net =
  grad(λ θ. hits(net, θ, x))(θ)
```

```text
[E3001] Error: cannot differentiate through `argmax`: the function is discontinuous (Disc) in `θ`
   ,-[ bad_argmax.psk:6:3 ]
   |
 4 |   mean(real(argmax(net.apply(θ, x))))
   |             ^^^^^^^^^^^|^^^^^^^^^^^
   |                        `------------- `argmax` makes the result discontinuous in `θ`
   |
 6 |   grad(λ θ. hits(net, θ, x))(θ)
   |   ^^^^^^^^^^^^^|^^^^^^^^^^^^
   |                `-------------- `grad` here, with respect to `θ`
   |
   | Help: use `softmax` for a smooth surrogate, or an explicit straight-through relaxation
   |
   | Note: in `hits_grad`; `grad` requires the class C⁰⁺ or better
---'
```

The error names the primitive responsible, even through a call, and the
`grad` that needs it; a framework would silently return a zero gradient.

## 5. Randomness is an effect

`tutorial/05_sample.psk`:

```text
-- 05_sample.psk — randomness is an effect.
dim B

fn pick(logits : ℝ[batch: B, vocab: 8]) -> idx[8][batch: B] uses Rand =
  sample(logits)
```

- **`idx[8][batch: B]`** is a tensor of *indices* in `[0, 8)`, not of reals:
  tokens, labels and sampled classes have this type, and mixing them with
  reals without `real(…)` is an error (`E2009`).
- **`sample`** draws from the softmax of the logits: it *uses* the `Rand`
  effect, which the function must declare with `uses Rand`. Without the
  clause:

```text
[E4001] Error: `pick` uses the effect `Rand` without declaring it
   ,-[ bad_rand.psk:3:3 ]
   |
 2 | ,-> fn pick(logits : ℝ[batch: B, vocab: 8]) -> idx[8][batch: B] =
 3 | |->   sample(logits)
   | |     ^^^^^^^|^^^^^^
   | |            `--------- this uses `Rand`
   | |
   | `---------------------- declared without `uses`
   |
   |     Help: add `uses Rand` to the signature of `pick`
---'
```

Effects are inferred and checked against the declaration, transitively
through calls. Initializers are implicitly `uses Rand` (the `Linear.init`
line of the summaries above). The interpreter draws from a stream seeded by
`--seed`:

```sh
presk run tutorial/05_sample.psk --entry pick --dim B=5
presk run tutorial/05_sample.psk --entry pick --dim B=5 --seed 3
```

```text
idx[8][batch: 5] = [batch: 5] [1, 5, 0, 7, 7]
idx[8][batch: 5] = [batch: 5] [4, 5, 5, 2, 6]
```

## 6. Composition: imports, sub-modules, stacks

`tutorial/06_tower.psk` assembles the previous modules:

```text
-- 06_tower.psk — composition: imports, sub-modules, stacked blocks.
import "01_linear.psk" as lin
import "02_norm.psk"   as norm

const D : nat = 4
const L : nat = 3

module Block {
  use norm : norm.RMSNorm{D}
  use proj : lin.Linear{D, D}

  fn apply(x : ℝ[..., model: D]) -> ℝ[..., model: D] =
    x + proj.apply(gelu(norm.apply(x)))       -- residual connection
}

module Tower {
  use blocks : stack{L}(Block)                 -- L copies of Block, parameters stacked
  use final  : norm.RMSNorm{D}

  fn apply(x : ℝ[..., model: D]) -> ℝ[..., model: D] =
    final.apply(blocks.apply(x))
}
```

- **`use name : Module{args}`** declares a sub-module; its parameters
  become a subtree `θ.name` of the parameter tree. Inside the module,
  `proj.apply(x)` passes the subtree implicitly.
- **`stack{L}(Block)`** is `L` copies of `Block` with their parameters
  stacked along a new leading axis; `blocks.apply(x)` applies them in
  sequence.

```sh
presk check tutorial/06_tower.psk --summary
presk dump tutorial/06_tower.psk --stage elab
```

The elaboration dump shows the resolved program, with the instances and
the layout of each parameter tree:

```text
== tutorial/06_tower.psk (main)
import lin = tutorial/01_linear.psk
import norm = tutorial/02_norm.psk
const D = 4
const L = 3
module Block {
  use norm : RMSNorm{4}
  use proj : Linear{4, 4}
  fn apply(x : ℝ[..., model: 4]) -> ℝ[..., model: 4]
}
module Tower {
  use blocks : stack{3}(Block)
  use final : RMSNorm{4}
  fn apply(x : ℝ[..., model: 4]) -> ℝ[..., model: 4]
}
== instances
RMSNorm{4}:
  γ : ℝ[4]
Linear{4, 4}:
  w : ℝ[4, 4]
  b : ℝ[4]
Block:
  norm.γ : ℝ[4]
  proj.w : ℝ[4, 4]
  proj.b : ℝ[4]
```

```sh
presk run tutorial/06_tower.psk --entry Tower.apply --dim batch=2 --digits 6
presk run tutorial/06_tower.psk --entry Tower.apply --dim batch=2 --digits 6 --const L=1
```

```text
ℝ[batch: 2, model: 4] = [batch: 2, model: 4] [[-1.77145, -0.789038, -0.331211, -0.360083], [-0.136955, 1.71047, 0.446273, 0.925406]]
ℝ[batch: 2, model: 4] = [batch: 2, model: 4] [[-1.87905, -0.523951, -0.435617, 0.0699295], [1.03433, 1.08596, 1.03947, -0.818749]]
```

`Tower.apply` has no dimension parameters, so `Tower.apply` is the whole
entry; `--dim batch=2` fills the prefix as before, and `--const L=1` rebuilds
the program with one block.

## 7. Under the hood

`presk dump --stage STAGE` prints the program after each phase; the format is
a development aid and is not stable.

| Stage | What it shows |
|---|---|
| `tokens` | the token stream (`Kw(Module) @ 77..83`, `Ident("Linear") @ 84..90`, …) |
| `ast` | the syntax tree |
| `elab` | the elaborated program: imports, constants, instances, signatures (step 6) |
| `core` | PSK-Core, the intermediate representation: one typed instruction per line, explicit parameter trees, `stack` unrolled |
| `core-after-ad` | PSK-Core after automatic differentiation: `grad` replaced by generated `.vjp` and `.grad` functions |

For instance, `loss` of step 4 in PSK-Core, then the functions that
differentiation adds:

```sh
presk dump tutorial/04_grad.psk --stage core | sed -n '/^fn loss{/,/^}/p'
presk dump tutorial/04_grad.psk --stage core-after-ad | grep '^fn'
```

```text
fn loss{D B}(θ : params Linear{D, 1}, x : ℝ[batch: B, D], y : ℝ[batch: B, 1]) -> ℝ[] {
  ŷ : ℝ[batch: B, 1] = call Linear.apply{In = D, Out = 1}(θ, x)
  %4 : ℝ[batch: B, 1] = sub ŷ, y
  %5 : ℝ[batch: B, 1] = pow 2 %4
  %6 : ℝ[] = mean %5
  return %6
}
```

```text
fn Linear.apply{In Out}(θ : params Linear{In, Out}, x : ℝ[..., In]) -> ℝ[..., Out] {
fn loss{D B}(θ : params Linear{D, 1}, x : ℝ[batch: B, D], y : ℝ[batch: B, 1]) -> ℝ[] {
fn loss_grad{D B}(θ : params Linear{D, 1}, x : ℝ[batch: B, D], y : ℝ[batch: B, 1]) -> params Linear{D, 1} {
fn Linear.init{In Out}(key : key) -> params Linear{In, Out} uses Rand {
fn loss_grad.lambda0{D B}(x : ℝ[batch: B, D], y : ℝ[batch: B, 1], θ : params Linear{D, 1}) -> ℝ[] {
fn loss_grad.lambda0.vjp{D B}(…, ct : ℝ[]) -> (ℝ[batch: B, D], ℝ[batch: B, 1], params Linear{D, 1}) {
fn loss.vjp{D B}(θ : params Linear{D, 1}, x : ℝ[batch: B, D], y : ℝ[batch: B, 1], ct : ℝ[]) -> (params Linear{D, 1}, ℝ[batch: B, D], ℝ[batch: B, 1]) {
fn Linear.apply.vjp{In Out}(θ : params Linear{In, Out}, x : ℝ[..., In], ct : ℝ[..., Out]) -> (params Linear{In, Out}, ℝ[..., In]) {
fn loss_grad.lambda0.grad{D B}(x : ℝ[batch: B, D], y : ℝ[batch: B, 1], θ : params Linear{D, 1}) -> params Linear{D, 1} {
```

Each `f.vjp` takes a cotangent `ct` and returns the adjoints of all the
arguments; `f.grad` calls it with `ct = 1` and projects on `θ`. The
`language` chapter describes the representation and the differentiation
rules.

## 8. Export to PyTorch and check the export

`presk export` writes a readable PyTorch program whose structure mirrors the
source, and, with `--conformance-entry`, a test that compares it with the
interpreter on the same inputs.

```sh
presk export tutorial/06_tower.psk --target pytorch --mode readable --shapes=check \
    --conformance-entry Tower.apply --dim batch=2 -o out/tutorial_tower
```

```text
presk export: wrote out/tutorial_tower/_01_linear.py, out/tutorial_tower/_02_norm.py, out/tutorial_tower/_06_tower.py, out/tutorial_tower/test_conformance.py, out/tutorial_tower/conformance/inputs.json, out/tutorial_tower/conformance/expected.json
```

One Python file per Presk file (the stem made a Python identifier, hence
the leading `_` for names starting with a digit), importing each other as
the sources do. In `_06_tower.py`:

```python
class Tower(nn.Module):
    """Generated from `module Tower` (06_tower.psk:16). Do not edit by hand."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
        self.final = RMSNorm(4)

    def forward(self, x):
        """Generated from `fn apply` (06_tower.psk:20)."""
        # x : ℝ[..., model: 4]
        _stack0 = x
        for block in self.blocks:  # stack{3}
            _stack0 = block(_stack0)
        return self.final(_stack0)
```

`module` → `nn.Module`, `param` → `nn.Parameter` with its initializer, `use`
→ attribute, `stack{L}` → `nn.ModuleList` and a loop (`--stack=unroll` writes
the three calls), `fn apply` → `forward`, types → comments, and with
`--shapes=check`, `assert x.shape == (…)` after each binding with a closed
shape. A gradient exports to `torch.func.grad`, as `loss_grad` of step 4
shows:

```sh
presk export tutorial/04_grad.psk --target pytorch --mode readable \
    --conformance-entry loss_grad --dim D=3 --dim B=2 -o out/tutorial_grad
```

```python
def loss_grad(net, θ, x, y):
    """Generated from `fn loss_grad` (04_grad.psk:9)."""
    # net : Linear{D, 1}  (a module, passed as an nn.Module)
    # θ : params Linear{D, 1}
    # x : ℝ[batch: B, D]
    B = x.shape[0]
    D = x.shape[1]
    # y : ℝ[batch: B, 1]
    return torch.func.grad((lambda θ: loss(net, θ, x, y)))(θ)
```

### The conformance test

`conformance/inputs.json` holds the inputs the interpreter synthesized
(including `θ`), `conformance/expected.json` its result at 256 bits, and
`test_conformance.py` builds the module (`Tower()`, or `Linear(3, 1)` for
the function entry), loads `θ` into its parameters, runs the forward pass in
`float64` and asserts `torch.allclose(rtol=1e-9, atol=1e-12)`. This is the
"exact versus almost" experiment of the specification: the tolerance
measures the distance between the exact semantics and double precision.

PyTorch is not a dependency of the compiler; it is needed only here. Once:

```sh
make venv            # .venv with PyTorch, NumPy and pytest, via uv
```

Then:

```sh
(cd out/tutorial_tower && ../../.venv/bin/python -m pytest -q test_conformance.py)
(cd out/tutorial_grad  && ../../.venv/bin/python -m pytest -q test_conformance.py)
```

```text
1 passed in 0.64s
1 passed in 1.01s
```

## 9. The reference examples

The programs of the specification, copied under `tests/examples/`, are the
acceptance tests of the compiler; `minigpt.psk` is a complete decoder-only
transformer with its training and serving configurations.

```sh
presk check tests/examples/minigpt/minigpt.psk --summary
```

The summary shows the attention shapes with the relabeled `seq2` axis,
the `Disc` class of every function in its token argument (`ids`), and the
`Rand` effect of the initializers:

```text
minigpt.psk SelfAttention.apply : ℝ[batch: B, seq: T, 768]
    let q : ℝ[batch: B, heads: 12, seq: T, hd: 64]
    let s : ℝ[batch: B, heads: 12, seq: T, seq2: T]
    let a : ℝ[batch: B, heads: 12, seq: T, seq2: T]
    let o : ℝ[batch: B, heads: 12, seq: T, hd: 64]
minigpt.psk MiniGPT.apply : ℝ[batch: B, seq: T, 50257]
-- regularity and effects --
Embedding.apply : [θ C∞, ids Disc] uses none
MiniGPT.apply : [θ C∞, ids Disc] uses none
MiniGPT.init : [key C∞] uses Rand
```

The interpreter runs the model at a size it can afford by overriding its
constants (vocabulary 64, width 16, 2 heads, 2 blocks), and evaluates the
training loss of `train_minigpt.psk`, whose constants live in the imported
file:

```sh
presk run tests/examples/minigpt/minigpt.psk --entry MiniGPT.apply \
     --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8
presk run tests/examples/minigpt/train_minigpt.psk --entry loss \
     --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8
```

```text
ℝ[batch: 2, seq: 8, 64] = [batch: 2, seq: 8, 64] [[[0, 0, 0, …
ℝ[] = 4.15888308336
```

(The logits are zero because `head` is initialized to zeros; the loss is
`ln 64`.) A `training run` block is checked against its schema and its data,
and executed by `presk train` (the `training` chapter):

```sh
presk run tests/examples/minigpt/train_minigpt.psk
```

```text
presk run: this file has a `training run`: use `presk train` to execute it, or --entry to evaluate one function
```

The MiniGPT export (`make export-examples`) is the most
demanding one: `T` becomes a constructor argument because a parameter table
depends on it (`MiniGPT(T)`), the tuple `batch` of `loss` is unpacked, and
the four conformance tests of `make conformance` cover the forward pass and
the loss.

## 11. Train a model

A `training run` block turns a model, an objective and data into a loop that
`presk train` executes and `presk export --train` turns into a PyTorch
script. `tutorial/07_train.psk` teaches the tower of step 6 the function
`y = 2·x + 1`:

```text
-- 07_train.psk — a training run: the tower of 06 learns y = 2·x + 1.
import "06_tower.psk" as t
dim B

fn loss(θ : params t.Tower, batch : (ℝ[batch: B, model: 4], ℝ[batch: B, model: 4])) -> ℝ[] =
  let (x, y) = batch
  mean((t.Tower.apply(θ, x) - y)^2)

optimizer adamw {
  lr           : f64 = 0.01
  weight_decay : f64 = 0.0
  schedule     : cosine(warmup = 20)
}

training run {
  model     = t.Tower
  objective = loss
  optimizer = adamw
  data      = stream "data/pairs.psk.d" |> shuffle(seed = 3) |> batch(B = 8)
  epochs    : nat = 10
  checkpoint every 50 steps to "ckpt/"
}
```

- The **objective** takes the model's parameter tree, then the **batch**: a
  tuple of tensors, each with an axis named `batch`. The **sample type** is
  the batch type without that axis: `(ℝ[model: 4], ℝ[model: 4])`.
- The **optimizer** block's name is the algorithm (`adamw` is the one of
  0.2.0); its fields are hyperparameters, machine numbers as always.
- **`data`** names a `.psk.d` file relative to this one, shuffled with its own
  seed and cut into batches of 8, which fixes `B`.

The data file holds 128 samples written as text (`tutorial/data/pairs.txt`,
eight numbers per line: `x` then `y`) and packed with the type of one sample:

```sh
presk data pack --type '(ℝ[model: 4], ℝ[model: 4])' tutorial/data/pairs.txt -o tutorial/data/pairs.psk.d
presk check tutorial/07_train.psk --summary
```

```text
-- training --
training run: model Tower, objective loss, optimizer adamw
    batch   : (ℝ[batch: B, model: 4], ℝ[batch: B, model: 4])
    sample  : (ℝ[model: 4], ℝ[model: 4])
    data    : data/pairs.psk.d (shuffle 3, batch 8) — file holds (ℝ[model: 4], ℝ[model: 4])
    epochs  : 10
    ckpt    : every 50 steps to ckpt/
    dims    : B = 8
```

The checker opened the file, matched its sample type against the objective's
and bound `B`. Train on the interpreter (`--ckpt` redirects the checkpoints
the block asks for):

```sh
presk train tutorial/07_train.psk --steps 100 --log-every 20 --ckpt out/tutorial_ckpt
```

```text
step 20  epoch 2  loss 1.472339  lr 1.000e-2  |g| 2.0695e0
step 40  epoch 3  loss 0.843784  lr 8.536e-3  |g| 8.2585e-1
step 60  epoch 4  loss 0.521466  lr 5.000e-3  |g| 6.9388e-1
step 80  epoch 5  loss 0.432558  lr 1.464e-3  |g| 5.8567e-1
step 100  epoch 7  loss 0.320685  lr 0.000e0  |g| 4.6561e-1
presk train: 100 steps of Tower, final loss 0.320685, weights in out/tutorial_ckpt/final.psk.d
```

Each step evaluates the loss and its gradient (the `vjp` generated by
differentiation, through the fold of the stacked blocks) and applies the
AdamW step, itself a PSK-Core function; the learning rate follows the cosine
schedule from its warm-up to 0 at step 100 (`--steps` sets the horizon). The
weights are a one-sample `.psk.d` of type `params Tower`. Everything runs at
256 bits: a reference, not a fast trainer. For speed, export the run:

```sh
presk export tutorial/07_train.psk --target pytorch --mode readable \
    --conformance-entry train:10 -o out/tutorial_train
(cd out/tutorial_train && ../../.venv/bin/python -m pytest -q -s test_conformance.py)
(cd out/tutorial_train && ../../.venv/bin/python train.py --steps 100 --log-every 25 --ckpt ckpt)
```

```text
K-step conformance: 10 steps, worst relative gap 1.386e-14
1 passed in 1.29s
step 25/100  epoch 2  loss 1.111248  lr 9.904e-03
step 50/100  epoch 4  loss 0.727428  lr 6.913e-03
step 75/100  epoch 5  loss 0.968215  lr 2.222e-03
step 100/100  epoch 7  loss 0.833188  lr 0.000e+00
train.py: 100 steps in 0.1 s, final loss 0.833188, weights in …/out/tutorial_train/ckpt/final.pt
```

`train.py` is the run as a PyTorch loop over the exported objective and
model, with a reader of `.psk.d` files; the conformance test replays ten
interpreter steps on the same batches in float64 and finds the parameters
equal to fourteen digits. `train.py` itself shuffles with PyTorch, so its
loss curve differs from the interpreter's in the batches, not in the
arithmetic. The `training` chapter has the loop, the flags and the two
reference runs of the compiler's own tests.

## 12. Run it natively, then let the compiler rewrite

Everything so far ran on the reference interpreter at 256 bits. Since 0.3.0
the same programs run *natively*: every tensor primitive of PSK-Core is
lowered to a loop-nest kernel at the shapes of the run (PSK-Kernel v0),
compiled to machine code by Cranelift on first use, in `f64` or `f32`. The
interpreter stays the reference; the native backend is measured against it.

```sh
presk bench tutorial/07_train.psk --entry loss --dim B=8
```

```text
backend               median ms      best ms   speedup
interp 64 bits            0.976        0.852      1.0x
kernel f64                0.318        0.314      3.1x
native f64                0.163        0.162      6.0x
native f32                0.142        0.142      6.8x
```

Four executors of one program: the reference (here at 64 bits, its cheapest
setting), the kernels interpreted, the kernels compiled in `f64` and in
`f32`. Timings are one run's, and vary by a few tens of percent from run to
run at these sizes. The speedups are modest — a batch of eight through a
model of width four is dominated by dispatch, not arithmetic; the `native`
chapter has the tiny MiniGPT at 27×. Train the tower of step 11 natively:

```sh
presk train tutorial/07_train.psk --steps 100 --log-every 20 --backend native --ckpt out/tutorial_ckpt_native
```

```text
step 20  epoch 2  loss 1.472339  lr 1.000e-2  |g| 2.0695e0
step 40  epoch 3  loss 0.843784  lr 8.536e-3  |g| 8.2585e-1
step 60  epoch 4  loss 0.521466  lr 5.000e-3  |g| 6.9388e-1
step 80  epoch 5  loss 0.432558  lr 1.464e-3  |g| 5.8567e-1
step 100  epoch 7  loss 0.320685  lr 0.000e0  |g| 4.6561e-1
presk train: 100 steps of Tower, final loss 0.320685, weights in out/tutorial_ckpt_native/final.psk.d
```

Compare with step 11: the same log, digit for digit, at the six digits
printed. The only difference between the two runs is the arithmetic —
correctly rounded 256-bit reals against `f64` — and here it is invisible
(`--realize f32` prints the same six digits too). The `experiments` chapter
measures what is invisible here: after twenty AdamW steps of the tiny
MiniGPT the `f64` realization is 7e-16 from the reference, the `f32` one
4e-7. The reference is what makes those numbers *measurements*.

Now the other half of the compiler: rewriting. `--opt` rewrites the
PSK-Core program before running it, by equality saturation over rules that
are exact over ℝ — algebra, and the identities that clean the adjoints
automatic differentiation produces (`optimization` chapter). On the
gradient of step 4:

```sh
presk dump tutorial/04_grad.psk --stage core-opt >/dev/null
presk run tutorial/04_grad.psk --entry loss_grad --dim D=3 --dim B=2 --digits 6 --opt
```

```text
presk-opt: 5 of 9 functions rewritten, 59 → 45 lets, cost 97.51 → 68.78 (operation counts)
  loss_grad: 3 → 2 lets, cost 1.01 → 1.01 (saturated, 1 iterations)
  Linear.init: 6 → 5 lets, cost 2.03 → 2.03 (saturated, 1 iterations)
  loss_grad.lambda0.vjp: 6 → 5 lets, cost 3.04 → 3.04 (saturated, 1 iterations)
  loss.vjp: 17 → 13 lets, cost 37.43 → 29.73 (saturated, 7 iterations)
  Linear.apply.vjp: 14 → 7 lets, cost 38.07 → 17.04 (saturated, 6 iterations)
params = {
  w = [3, 1] [[-1.30262], [0.342146], [2.7531]]
  b = [1] [3.47884]
}
```

The adjoint of `Linear.apply` halves: the forward pass that differentiation
recomputed is dead and vanishes, a `sum_to` between equal types is the
identity, and the two `set_field`s of zeros that it summed become one record
`{w = …, b = …}`. The gradient is the one of step 4 to the last digit — the
rules are exact, and the interpreter is the oracle that checks it (every
golden output of the test suite is reproduced under `--opt`).

Which of two equivalent programs is cheaper is the *cost model's* decision.
Without more information it counts operations; with a machine profile it
predicts time. `presk bench --calibrate` times every kernel the entry
launches natively, computes each one's work and traffic from its loop nest,
and fits the roofline `λ + max(W/π, Q/β)` to the measurements:

```sh
presk bench tutorial/07_train.psk --entry loss --dim B=8 --repeat 7 --calibrate out/tutorial_profile.json
```

```text
kernel gelu (%0 : f64[batch: 8, model: 4]) …          384          512        0.71        0.56
kernel dot (%0 : f64[batch: 8, 4], %1 : f64…          256          640        0.42        0.40
kernel add (%0 : f64[batch: 8, 4], %1 : f64…           32          544        0.21        0.17
kernel add (%0 : f64[batch: 8, model: 4], %…           32          768        0.17        0.20
kernel sub (%0 : f64[batch: 8, model: 4], %…           32          768        0.21        0.20
kernel mean (%0 : f64[batch: 8, model: 4]) …           33          264        0.21        0.14
-- 12 kernels timed (0 kept from out/tutorial_profile.json), 7 calls, f64
-- roofline: π = 8.411e8 ops/s, β = 8.119e9 bytes/s, λ = 0.10 µs per launch
-- Spearman with the measured times: operation counts 0.336, work 0.488, roofline 0.511
```

(The last six of twelve lines.) Twelve kernels of a few dozen operations
each make a thin sample: the constants are those of tiny loop nests, they
move from one run to the next, and the rank correlation (0.51 here) is far
below the 0.95 the ninety kernels of the tiny MiniGPT give (`optimization`
chapter). The shape of the result is the point: the model's quality is a
*number*, measured against the machine, and even on this sample it improves
over counting operations (0.34). `--opt --profile
out/tutorial_profile.json` then makes that profile decide the extraction:

```sh
presk bench tutorial/07_train.psk --entry loss --dim B=8 --opt --profile out/tutorial_profile.json
```

```text
presk-opt: 5 of 12 functions rewritten, 139 → 131 lets, cost 687.5 µs → 687.3 µs (roofline, profile macos aarch64 f64)
```

On this program the profile finds almost nothing to save: the objective and
the tower are already what the rules can make them, and the costs are now in
the model's microseconds. The `optimization` chapter reports what the
profile does on the MiniGPT entries — the same, within noise — and why: the
rewrites of this version remove cheap kernels; the decisions that move time
(the order of the attention contractions, fusing elementwise chains) are the
next rules to write, and the model is now there to arbitrate them.

## 13. Testing everything

| Command | What it verifies |
|---|---|
| `make test` | the Rust test suite: unit tests, snapshot tests of every stage on the examples and of every diagnostic in `tests/diag/`, property tests of the shape decision procedure, gradients against finite differences at 512 bits (`mse`, two stacked modules), golden outputs of `presk run` in `tests/run/`, the `.psk.d` container, the optimizer's schedule and step, the training loop (a regression converges, the tiny MiniGPT's 20-step log), and that the generated Python byte-compiles |
| `make check-examples` | `presk check` on the five reference programs: no error, only `W0001` |
| `make run-examples` | the reference `presk run` commands |
| `make export-examples` | the exports with conformance data, under `out/`: three models, the training loss, one optimizer step, twenty training steps, the ONNX graph |
| `make conformance` | the exports, then their `test_conformance.py` in `.venv` (after `make venv`): nine tests |
| `make train-examples` | `presk train` on a linear regression and on the tiny MiniGPT, on the reference and natively in `f64` and `f32` |
| `make serve-examples` | `presk serve` on the tiny MiniGPT's checkpoint: greedy tokens identical on every backend, `--serve` export (`make serve-pt` runs the exported `serve.py`) |
| `make opt-examples` | the e-graph optimizer's report on the examples; the same outputs with and without `--opt`, on the reference and natively |
| `make calibrate` | `presk bench --calibrate` on the tiny MiniGPT's entries, then the entries timed with the profile |
| `make bench` | `presk bench` on the tiny MiniGPT: forward, loss, optimizer step, on every backend |
| `make tutorial` | every command of this chapter, in order, including its conformance tests |
| `cargo insta review` | after a change in the compiler's output: accept or reject each changed snapshot |

Exit codes, for scripts: 0 success, 1 diagnostics with an error, 2 usage or
not implemented, 3 valid but not executable in this version.

## Where to go next

- The `experiments` chapter for what these tools show about the approach
  (`make experiment`): a silent broadcasting bug of plain PyTorch that the
  types rule out, and the gap of float32 and float64 training to the exact
  reference — and presk's own native realizations next to PyTorch's.
- The `native` chapter for the runtime seam, PSK-Kernel v0 and the
  Cranelift backend with their conformance suites; the `optimization`
  chapter for the rules, the oracle and the calibrated cost model; the
  `serve` chapter for `inference serve` and `presk serve`.
- The `cli` chapter for every flag, the `language` chapter for the
  constructs accepted and how they are typed, `diagnostics` for the codes.
- `docs/spec-questions.md` for the points where this implementation had to
  choose what the specification leaves open.
- The specification itself, in the `presk-specs` repository, for the
  design: exact semantics, error budgets, the phases after 0.3.0.
