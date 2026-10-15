# Tutorial: from a first module to a PyTorch export

This chapter teaches `presk` by doing, and doubles as the end-to-end test of
the tool: every command below is run by `make tutorial`, and every output
shown is what version 0.12.0 prints (timings vary from one run and one machine
to the next; these are from an aarch64 Linux machine with twenty cores). The
programs written here are in the
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
12. run it natively, and let the compiler rewrite it;
13. run the same tool on a model that is not a toy;
14. run the whole test suite;
15. give an entry an error budget, and let the compiler choose the types;
16. give the training step a budget, and train under what it chooses;
17. give the attention kernels a layout, chosen statically, and measure the choice;
18. export the attention as a C program, check its bits against the native
    backend, and write its loop nests for a polyhedral tool.

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
Presk compiler (0.12.0: front-end, reference interpreter, native backend, e-graph optimizer, training, serving, precision budgets, layout plans, PyTorch and C exports, a compilation as a document)

Usage: presk <COMMAND>

Commands:
  check      Type-check a program: shapes, regularity classes, effects
  run        Evaluate an entry point on the reference interpreter or natively
  export     Export a program to another framework (PyTorch), or an entry as a C program
  dump       Dump an intermediate representation (development aid)
  train      Run the `training run` of a file, on the reference interpreter or natively
  serve      Generate tokens with the file's `inference serve`: its model, weights and generation function
  plan       Choose a realization plan statically: every nest's layout space costed by the profile's layout model
  explain    One compilation as a document: every stage's representation and the reasons for its decisions, in DIR/explain.md (plan 0.11.0)
  autotune   Measure every point of every nest's layout space on the entry's own buffers, asserting each computes point 0's bits
  reprice    Price the points of a recorded day again under another profile, their times copied: nothing is measured
  rank       Rank accuracy of the layout cost model on two days of measurement, the statements of plan 0.7.0, and plan 0.8.0's Q
  refit      Fit a profile's layout cost model again on its samples, in another form: nothing is measured
  bench      Time an entry on every backend (the reference at --bits, the kernel interpreter, native f64, f32 and bf16)
  calibrate  Measure the magnitude envelope of an entry, or one training step under each realization (--step)
  solve      Solve for a realization under a declared error budget
  quantize   Quantise parameters as a realization's `quant` map says, and measure the gap in two parts: the weights' rounding and the arithmetic
  data       Produce and inspect `.psk.d` data files
  help       Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
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
[W0001] Warning: implicit dimension `B`
    ,-[ tests/examples/mlp/mlp.psk:34:35 ]
    |
 34 |                      x : ℝ[batch: B, model: D], y : ℝ[batch: B, model: D])
    |                                   |
    |                                   `-- not declared; taken as a dimension of this function
    |
    | Help: declare it with `dim B` at file level or as a type parameter `{B : dim}`
----'
```

The program is correct (exit code 0): the checker only warns, once for each
function that uses `B` as a dimension without declaring it (`mse` and
`mse_grad`), and says how to declare it. Read
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
  let a = softmax(axis seq2, s, where = causal_mask(seq: T))
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
- `sqrt(HD : ℝ)` injects the constant into `ℝ`; `causal_mask(seq: T)` is a
  mask, a tensor of truth values `bool[seq: T, seq2: T]`, true where `j ≤ i`,
  and `softmax(…, where = mask)` normalizes over the positions it allows,
  giving 0 at the others (0.10.0 step 7b: it was a real tensor of 0 and −∞
  added to the scores).

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
-- 05_sample.psk — randomness is an effect, and a draw a function of its key.
dim B

fn pick(key : key, logits : ℝ[batch: B, vocab: 8]) -> idx[8][batch: B] uses Rand =
  sample(key, logits)
```

- **`idx[8][batch: B]`** is a tensor of *indices* in `[0, 8)`, not of reals:
  tokens, labels and sampled classes have this type, and mixing them with
  reals without `real(…)` is an error (`E2009`).
- **`sample(key, logits)`** draws from the softmax of the logits, a token
  per position from the key's uniform draw of that position: a function of
  its key, as `dropout` is. It *uses* the `Rand` effect, which the function
  must declare with `uses Rand`. Without the clause:

```text
[E4001] Error: `pick` uses the effect `Rand` without declaring it
   ,-[ bad_rand.psk:3:3 ]
   |
 2 | ,-> fn pick(key : key, logits : ℝ[batch: B, vocab: 8]) -> idx[8][batch: B] =
 3 | |->   sample(key, logits)
   | |     ^^^^^^^^^|^^^^^^^^^
   | |              `------------ this uses `Rand`
   | |
   | `--------------------------- declared without `uses`
   |
   |     Help: add `uses Rand` to the signature of `pick`
---'
```

Effects are inferred and checked against the declaration, transitively
through calls. Initializers are implicitly `uses Rand` (the `Linear.init`
line of the summaries above). `presk run` synthesizes the key as it
synthesizes every argument, from `--seed`, and `sample` draws from it (until
0.10.0 step 7b it took no key and drew from a stream seeded by `--seed`):

```sh
presk run tutorial/05_sample.psk --entry pick --dim B=5
presk run tutorial/05_sample.psk --entry pick --dim B=5 --seed 3
```

```text
idx[8][batch: 5] = [batch: 5] [3, 3, 5, 1, 7]
idx[8][batch: 5] = [batch: 5] [4, 5, 3, 3, 4]
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
| `core` | PSK-Core, the intermediate representation: one typed instruction per line, explicit parameter trees, `stack{L}` as a fold (`--unroll-stack` for `L` calls) |
| `core-after-ad` | PSK-Core after automatic differentiation: `grad` replaced by generated `.vjp` and `.grad` functions |
| `core-opt`, `sched`, `kernel`, `sensitivity` | after the e-graph optimizer (step 12), the fusion groups, the kernels an entry launches, and where a precision budget should go (the `cli` chapter) |

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
fn loss_grad.lambda0.vjp{D B}(…, ct : ℝ[]) -> (params Linear{D, 1}) {
fn loss.vjp{D B}(θ : params Linear{D, 1}, x : ℝ[batch: B, D], y : ℝ[batch: B, 1], ct : ℝ[]) -> (params Linear{D, 1}) {
fn Linear.apply.vjp{In Out}(θ : params Linear{In, Out}, x : ℝ[..., In], ct : ℝ[..., Out]) -> (params Linear{In, Out}) {
fn loss_grad.lambda0.grad{D B}(x : ℝ[batch: B, D], y : ℝ[batch: B, 1], θ : params Linear{D, 1}) -> params Linear{D, 1} {
fn Linear.apply.bwd{In Out}(θ : params Linear{In, Out}, x : ℝ[..., In], ct : ℝ[..., Out]) -> (params Linear{In, Out}) {
fn loss.bwd{D B}(θ : params Linear{D, 1}, x : ℝ[batch: B, D], residuals : (ℝ[batch: B, 1], ℝ[batch: B, 1]), ct : ℝ[]) -> (params Linear{D, 1}) {
fn loss.fwd{D B}(θ : params Linear{D, 1}, x : ℝ[batch: B, D], y : ℝ[batch: B, 1]) -> ((ℝ[batch: B, 1], ℝ[batch: B, 1])) {
fn loss_grad.lambda0.bwd{D B}(…, ct : ℝ[]) -> (params Linear{D, 1}) {
fn loss_grad.lambda0.fwd{D B}(x : ℝ[batch: B, D], y : ℝ[batch: B, 1], θ : params Linear{D, 1}) -> (((ℝ[batch: B, 1], ℝ[batch: B, 1]))) {
```

Each `f.vjp` takes a cotangent `ct` and returns the adjoints of the
arguments its callers read; `f.grad` calls it with `ct = 1` and projects on
`θ`, so here each returns `θ`'s alone: the adjoints of the data `x` and `y`
are not computed, and the forward values nothing reads are not recomputed
(the `language` chapter, "Dead code after differentiation"). A `.vjp` is
its `.bwd` after its `.fwd`: `loss.fwd` runs the loss's forward pass once
and returns what its reverse sweep reads (here `ŷ` and `ŷ − y`; its value,
which no one reads, it does not compute), and `loss.bwd` runs the sweep on
them, calling `Linear.apply.bwd`, whose sweep reads only its parameters and
so needs no `.fwd` (the `language` chapter, "Residuals"; `--remat
recompute` keeps the `.vjp`s that compute their forward pass again). The
`language` chapter describes the representation and the differentiation
rules.

### Every stage at once: `presk explain`

`presk explain` compiles one entry and writes every stage of it as one
document, each section with the command that prints it (0.11.0, the
`explain` chapter):

```sh
presk explain tutorial/04_grad.psk --entry loss_grad --dim D=3 --dim B=2 --out out/tutorial_explain
grep '^## ' out/tutorial_explain/explain.md
```

```text
presk explain: wrote out/tutorial_explain/explain.md (14 sections, 1 listing under listings/, 1 artifact)
## The parameters
## 0 Summary
## 1 The source
## 2 Elaboration
## 3 PSK-Core
## 4 Differentiation
## 5 The e-graph optimizer
## 6 Precision: from a budget to a type per site
## 7 Fusion
## 8 The machine and the layout plan
## 9 The kernels and the host
## 10 Targets
## 11 Measured
## 12 Sites through every stage
## Appendix: the commands
```

Its §3 is the PSK-Core of the functions `loss_grad` reaches, its §4 the
derivatives above, and its §9 the ten kernels the entry launches, each
under the site that launches it. With no budget and no profile, every site
is in `f64` and every nest at its default layout: §6 bounds the error of
that realization (`cond`, 2.4·10⁻¹² against a measured 4.4·10⁻¹⁶) and §8
says nothing is planned. Steps 15 and 17 add a budget (`--budget`) and a
machine (`--profile`); `make explain` writes the document of step 15's
`loss` under both, the manual's chapter `explain_loss`, and `make
explain-pdf` its PDF.

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
.                                                                        [100%]
1 passed in 0.50s
.                                                                        [100%]
1 passed in 1.00s
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
presk train: 5.1 ms a step, median of 99 (the first excluded)
presk train: 100 steps of Tower, final loss 0.320685, weights in out/tutorial_ckpt/final.psk.d
```

Each step evaluates the loss and its gradient in one call (the `value_vjp`
generated by differentiation, which returns the loss and the cotangents,
through the fold of the stacked blocks) and applies the
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
K-step conformance: 10 steps, worst relative gap 1.333e-14
.
1 passed in 0.93s
train.py: torch 2.14.0+cpu on cpu, 20 threads, float32
step 25/100  epoch 2  loss 1.111248  lr 9.904e-03
step 50/100  epoch 4  loss 0.727428  lr 6.913e-03
step 75/100  epoch 5  loss 0.968215  lr 2.222e-03
step 100/100  epoch 7  loss 0.833188  lr 0.000e+00
train.py: 6.1 ms a step, median of 99 (the first excluded)
train.py: 100 steps in 0.6 s, final loss 0.833188, weights in …/out/tutorial_train/ckpt/final.pt
```

`train.py` is the run as a PyTorch loop over the exported objective and
model, with a reader of `.psk.d` files; the conformance test replays ten
interpreter steps on the same batches in float64 and finds the parameters
equal to fourteen digits. `train.py` itself shuffles with PyTorch, so its
loss curve differs from the interpreter's in the batches, not in the
arithmetic. Both loops report their median step time with the first step
left out; at this size the 256-bit interpreter and PyTorch are level, both
dominated by dispatch. The `training` chapter has the loop, the flags and the two
reference runs of the compiler's own tests.

## 12. Run it natively, then let the compiler rewrite

Everything so far ran on the reference interpreter at 256 bits. Since 0.3.0
the same programs run *natively*: every tensor primitive of PSK-Core — since
0.4.0, every chain of elementwise ones — is lowered to a loop-nest kernel at
the shapes of the run (PSK-Kernel v0), compiled to machine code by Cranelift
on first use, in `f64`, `f32` or `bf16`. The interpreter stays the reference;
the native backend is measured against it.

```sh
presk bench tutorial/07_train.psk --entry loss --dim B=8
```

```text
backend               median ms      best ms   speedup
interp 64 bits            0.306        0.293      1.0x
kernel f64                0.123        0.115      2.5x
native f64                0.075        0.071      4.1x
native f64 +lanes         0.071        0.070      4.3x
native f32                0.079        0.072      3.9x
native f32 +lanes         0.073        0.071      4.2x
native bf16               0.075        0.072      4.1x
```

Four executors of one program: the reference (here at 64 bits, its cheapest
setting), the kernels interpreted, and the kernels compiled, in `f64`, `f32`
and `bf16`, the first two also in vector lanes, which change no bit (`native`
chapter). Timings are one run's, and vary by a few tens of percent from run
to run at these sizes. The speedups are modest, and the narrower types and
the lanes buy nothing — a batch of eight through a model of width four is
dominated by dispatch, not arithmetic; the `native` chapter has larger
programs. Train the tower of step 11 natively:

```sh
presk train tutorial/07_train.psk --steps 100 --log-every 20 --backend native --ckpt out/tutorial_ckpt_native
```

```text
step 20  epoch 2  loss 1.472339  lr 1.000e-2  |g| 2.0695e0
step 40  epoch 3  loss 0.843784  lr 8.536e-3  |g| 8.2585e-1
step 60  epoch 4  loss 0.521466  lr 5.000e-3  |g| 6.9388e-1
step 80  epoch 5  loss 0.432558  lr 1.464e-3  |g| 5.8567e-1
step 100  epoch 7  loss 0.320685  lr 0.000e0  |g| 4.6561e-1
presk train: 184 kernel launches a step, 332 without fusion (45% removed)
presk train: 0.6 ms a step, median of 99 (the first excluded)
presk train: 100 steps of Tower, final loss 0.320685, weights in out/tutorial_ckpt_native/final.psk.d
```

Compare with step 11: the same log, digit for digit, at the six digits
printed; the lines before the last count a step's kernel launches, with and
without the fusion of elementwise chains (`realize` chapter), and time
the step. The only
difference between the two runs is the arithmetic —
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
presk-opt: 4 of 14 functions rewritten, 62 → 55 lets, cost 48.95 → 45.04 (operation counts)
  loss_grad: 3 → 2 lets, cost 1.01 → 1.01 (saturated, 1 iterations)
  Linear.init: 6 → 5 lets, cost 2.03 → 2.03 (saturated, 1 iterations)
  Linear.apply.bwd: 9 → 5 lets, cost 12.04 → 9.03 (saturated, 6 iterations)
  loss.bwd: 11 → 10 lets, cost 5.94 → 5.04 (saturated, 4 iterations)
params = {
  w = [3, 1] [[-1.30262], [0.342146], [2.7531]]
  b = [1] [3.47884]
}
```

Differentiation has already removed its own dead code (the forward pass
`Linear.apply.vjp` recomputed, and a `sum_to` between equal types, the
identity), and keeps its forward pass's values rather than computing them
again; the rules take the adjoint of `Linear.apply`, its `.bwd`, from 9
`let`s to 5, the two `set_field`s of zeros that it summed becoming one
record `{w = …, b = …}`, and `loss.bwd`'s `pow 1` goes. The gradient is the one of step 4
to the last digit — the rules are exact, and the interpreter is the oracle
that checks it (every golden output of the test suite is reproduced under
`--opt`).

Which of two equivalent programs is cheaper is the *cost model's* decision.
Without more information it counts operations; with a machine profile it
predicts time. `presk bench --calibrate` times every kernel the entry
launches natively, computes each one's work and traffic from its loop nest,
and fits the roofline `λ + max(W/π, Q/β)` to the measurements:

```sh
presk bench tutorial/07_train.psk --entry loss --dim B=8 --repeat 7 --calibrate out/tutorial_profile.json
```

```text
kernel                                               work        bytes measured µs predicted µs
kernel pow 2 (%0 : f64[batch: 8, model: 4])…           32          512        0.22        0.23
kernel mean axis model (%0 : f64[batch: 8, …           40          320        0.26        0.22
kernel fused 2 steps (%0 : f64[batch: 8], %…           72          136        0.19        0.26
kernel fused 2 steps (%0 : f64[batch: 8, mo…           64          608        0.26        0.25
kernel gelu (%0 : f64[batch: 8, model: 4]) …          384          512        0.75        0.63
kernel dot (%0 : f64[batch: 8, 4], %1 : f64…          256          640        0.48        0.48
kernel add (%0 : f64[batch: 8, 4], %1 : f64…           32          544        0.26        0.23
kernel add (%0 : f64[batch: 8, model: 4], %…           32          768        0.24        0.25
kernel fused 2 steps (%0 : f64[batch: 8, mo…           96          768        0.27        0.29
kernel mean (%0 : f64[batch: 8, model: 4]) …           33          264        0.29        0.21
-- 10 kernels timed (0 kept from out/tutorial_profile.json), 7 calls, f64
-- roofline: π = 8.321e8 ops/s, β = 9.162e9 bytes/s, λ = 0.17 µs per launch
-- Spearman with the measured times: operation counts 0.747, work 0.602, roofline 0.350
-- profile written to out/tutorial_profile.json
```

Ten kernels of a few dozen operations each, some of them fused chains, make
a thin sample: the constants are those of tiny loop nests and move from one
run to the next, and so do the rank correlations. In this run the fitted
roofline (0.35) ranks the kernels *worse* than counting operations (0.75):
at a quarter of a microsecond each, these kernels differ by less than the
timer's noise, and no model can be ranked on them. On the ninety kernels of
the tiny MiniGPT the roofline reaches 0.95 against 0.15 for counts
(`optimization` chapter). The shape of the result is the point: the model's
quality is a *number*, measured against the machine and printed whether it
flatters the model or not. `--opt --profile out/tutorial_profile.json` then
makes that profile decide the extraction:

```sh
presk bench tutorial/07_train.psk --entry loss --dim B=8 --opt --profile out/tutorial_profile.json
```

```text
presk-opt: 5 of 12 functions rewritten, 139 → 131 lets, cost 697.8 µs → 697.7 µs (roofline, profile linux aarch64 f64)
```

On this program the profile finds almost nothing to save: the objective and
the tower are already what the rules can make them, and the costs are now in
the model's microseconds. On the corpus GPT the one rule that turns on sizes
— moving attention's scale onto its smaller operand — is worth 12 % a step,
and the profile tells the two forms apart where operation counts price them
the same (`optimization` chapter).

## 13. A model that is not a toy

Everything so far was small enough to read. This step runs the same tool on
the corpus of an actual course — the NLP course at Mines Paris, whose ladder
of models is in `corpus/nlp/` and whose data is the text of the French civil
code. `make corpus` runs all of it; what follows is the shape of it.

**Text becomes a dataset** in three steps, each of which says what it did:

```sh
presk data chars corpus/nlp/data/civil.md -o corpus/nlp/data/civil_vocab.json
presk data encode corpus/nlp/data/civil.md --vocab corpus/nlp/data/civil_vocab.json -o corpus/nlp/data/civil_tokens.txt
presk data lm-windows --vocab 92 --seq 256 corpus/nlp/data/civil_tokens.txt -o corpus/nlp/data/civil_T256.psk.d
```

```text
presk data: wrote corpus/nlp/data/civil_vocab.json (92 characters over 1182121 of text)
presk data: wrote corpus/nlp/data/civil_tokens.txt (1182121 tokens over a vocabulary of 92)
presk data: wrote corpus/nlp/data/civil_T256.psk.d (4617 windows of type (idx[92][seq: 256], idx[92][seq: 256]) from 1182121 tokens)
```

Characters, not bytes, so an accent is one token; the vocabulary is a JSON
array, so a served model can carry it.

**The first rung is a model with nothing to learn.** The counting bigram is
the maximum-likelihood estimate in closed form, and in Presk it is one
contraction of two indicator tensors — the decomposition the specification
ascribes to `embed`, now writable because `one_hot` exists:

```text
fn counts(ox : ℝ[batch: N, prev: V], oy : ℝ[batch: N, next: V]) -> ℝ[prev: V, next: V] =
  ein[prev, next](ox * oy)
```

`one_hot` leaves its new axis unnamed, as `embed` does; passing the result to
these parameters is what names it, and what lets `ein` tell the two apart.
Evaluate it on all 67 652 character pairs of the civil code's vocabulary —
`--data` fills the entry's last argument from a file instead of synthesizing
it:

```sh
presk check corpus/nlp/01_bigram_count.psk --summary
presk run corpus/nlp/01_bigram_count.psk --entry loss \
    --data corpus/nlp/data/bigrams.psk.d --backend native --digits 10
```

```text
01_bigram_count.psk counts : ℝ[prev: 41, next: 41]
01_bigram_count.psk probs : ℝ[prev: 41, next: 41]
    let c : ℝ[prev: 41, next: 41]
01_bigram_count.psk loss : ℝ[]
    let x : idx[41][batch: N]
    let y : idx[41][batch: N]
    let p : ℝ[prev: 41, next: 41]
ℝ[] = 2.343771504
```

The reference interpreter gives the same ten digits, and so does NumPy. That
number is the point of the next rung: `02_bigram_neural.psk` is the *same
function of the same parameters*, fitted by AdamW instead of solved, and
`make corpus` shows its batch losses closing on 2.3438 from above. Gradient
descent recovering a closed form is the course's own sanity check, and it is
a good one for a compiler too.

`03_mlp.psk` is the first rung that needed something new. Three characters of
context are embedded and then have to become *one* vector, which is
`merge(axis ctx, axis emb, ·)`: two adjacent axes joined, row-major, without
moving any data. It trains to 1.73 against the bigram's 2.34 — more context
predicts better, which is why the rung exists.

**The last rung is the course's GPT**, width 384, six heads, six blocks,
context 256, on the civil code:

```sh
presk check corpus/nlp/05_gpt.psk --summary
```

```text
training run: model GPT, objective loss, optimizer adamw
    batch   : (idx[92][batch: B, seq: T], idx[92][batch: B, seq: T])
    sample  : (idx[92][seq: T], idx[92][seq: T])
    data    : data/civil_T256.psk.d (shuffle 7, batch 16) — file holds (idx[92][seq: 256], idx[92][seq: 256])
    epochs  : 1
    ckpt    : every 500 steps to ckpt/
    dims    : B = 16, T = 256
```

The checker opened the data file, matched its sample type against the
objective's and bound both `B` and `T` from it. The model is real: it trains,
and one step of it cost 39.8 s against PyTorch's 0.405 s when 0.4.0 measured
both on the development machine (30.8 s in `f32` on the machine of this
chapter at 0.6.0, where PyTorch has not been measured). The
`experiments` chapter says what that ratio is made of and what it does and
does not say about the ideas this tool is built on; the short version is that
every optimization of versions 0.4.0 to 0.6.0 is exact, the losses never
moved, and most of the remaining distance is a matrix coprocessor presk does
not reach yet.

## 14. Testing everything

| Command | What it verifies |
|---|---|
| `make test` | the Rust test suite: unit tests, snapshot tests of every stage on the examples and of every diagnostic in `tests/diag/`, property tests of the shape decision procedure, gradients against finite differences at 512 bits (`mse`, two stacked modules), golden outputs of `presk run` in `tests/run/`, the `.psk.d` container, the optimizer's schedule and step, the training loop (a regression converges, the tiny MiniGPT's 20-step log), that the generated Python byte-compiles, and the C export's programs, built and compared bit for bit with the native backend (step 18) |
| `make check-examples` | `presk check` on the five reference programs: no error, only `W0001` |
| `make run-examples` | the reference `presk run` commands |
| `make export-examples` | the exports with conformance data, under `out/`: three models, the training loss, one optimizer step, twenty training steps, the ONNX graph |
| `make conformance` | the exports, then their `test_conformance.py` in `.venv` (after `make venv`): nine tests |
| `make train-examples` | `presk train` on a linear regression and on the tiny MiniGPT, on the reference and natively in `f64` and `f32` |
| `make serve-examples` | `presk serve` on the tiny MiniGPT's checkpoint: greedy tokens identical on every backend, `--serve` export (`make serve-pt` runs the exported `serve.py`) |
| `make opt-examples` | the e-graph optimizer's report on the examples; the same outputs with and without `--opt`, on the reference and natively |
| `make calibrate` | `presk bench --calibrate` on the tiny MiniGPT's entries, then the entries timed with the profile |
| `make corpus`, `make corpus-gpt` | the NLP-course ladder on the French civil code, and the course GPT |
| `make sched` | the fusion groups of the corpus GPT, the launches they remove, and a 1024³ matmul |
| `make coverage` | `presk check` on the MLPerf transcriptions, and BERT-large and the Llama decoder at tiny constants on the reference and natively (`coverage` chapter) |
| `make experiment-gpt` | the course GPT trained by `presk train` and by the exported `train.py` |
| `make bench` | `presk bench` on the tiny MiniGPT: forward, loss, optimizer step, on every backend |
| `make gpt-trained`, `make precision` | the corpus GPT trained for 600 steps; its envelope, bound, sensitivities and solved realization (`precision` chapter) |
| `make curve` | the ε-to-quality curve: seven budgets, each solved and trained from scratch (about half an hour since 0.10.0) |
| `make quant-table`, `make step-table`, `make j-step`, `make step-train`, `make foil` | the experiments of 0.6.0: integer weights, the training step's budget, statement J′ at course scale, the trajectory, and MLPerf's rule beside a budget (`precision` and `experiments` chapters) |
| `make margins` | ε from the margins: the exact model's margins on the held-out windows, the largest budget on the logits that changes at most a fraction δ of the decisions, and every realization's flips against it (`experiments` chapter §5, D118; five minutes, needs `make curve`) |
| `make bert` | BERT-large inference at MLPerf's constants (about five hours; `L=2 make bert` for a quick version) |
| `make llama` | the Llama 2 decoder at the 7B configuration's width, one block: its exact logits, the gaps of seven realizations, the split of the whole quantisation scheme, the time (`coverage` chapter, D104, D106; about thirteen hours, up to 64 GB, alone); `STAGES=solve` adds the budget's solve (14 h, 72 GB) |
| `make plan-check` | a plan stating the default layout lowers every example entry to the same kernels, byte for byte |
| `make calibrate-ra4`, `make ra4-small` | the layout cost model calibrated on a GEMM, pinned; the RA4 sweep's daily subset on this chapter's attention, two runs ranked (step 17) |
| `make ra4`, `make ra4-check`, `make ra4-20`, `make ra4-day2` | the RA4 measurement of 0.7.0: the course GPT's attention, the GEMM and a BERT-shaped sample, every point checked on the kernel interpreter, the 20-core sweep, the second run and the report (`experiments` chapter §7; a night) |
| `make ra4-dev`, `make ra4-dev-frozen`, `make ra4-ablation` | 0.8.0's development set: 0.7.0's measurements priced again under a profile, nothing measured; the frozen model's report on it, byte for byte; every form of the model on it (`cli` chapter, `presk reprice`; minutes) |
| `make calibrate-ra4-small`, `make ra4-f2*`, `make ra4-f1*` | 0.8.0's fresh sweeps, each a dry run, two days, a check of every point on the kernel interpreter and a report: the A725 cores calibrated and 0.7.0's shapes swept on them (F2), new shapes on the X925 cores (F1) (`experiments` chapter §8; hours each) |
| `make c-conformance`, `make c-speed` | the C export on the example entries and the course GPT, under plans, on threads, with integer weights, built with GCC, Clang and for x86-64, bit for bit against the native backend; `presk serve`'s tokens from the exported loop; the SCoP mode's nests against Polly; the exported course GPT timed against the native backend (`export_c` chapter; minutes, installs zig and LLVM's release) |
| `make pytorch-baseline` | the course GPT's forward pass and training step timed on the native backend, the C export and PyTorch, under the default layout and under plans, with the distances to the native `f64` logits and where the C program's time goes (`experiments` chapter §3; twenty minutes, a quiet machine, needs `.venv`; `CFLAGS` builds the C programs with other flags) |
| `make pytorch-decoder` | one block of the Llama 2 decoder at the 7B width, its forward pass at 384 tokens timed on the native backend, under the default layouts and the frozen profile's plans, in `f32` and with `int8` weights, and in PyTorch eager, on the ten X925 cores; then the native kernels alone (D107; alone, about three and a half hours) |
| `make tutorial` | every command of this chapter, in order, including its conformance tests |
| `cargo insta review` | after a change in the compiler's output: accept or reject each changed snapshot |

Exit codes, for scripts: 0 success, 1 diagnostics with an error or a
run-time error, 2 usage or not implemented (the `cli` chapter).

## 15. A budget, and what it buys

Every program so far has been run in one machine type, chosen by a flag.
Version 0.5.0 replaces that with a **budget**: you say how far the execution
may stray from the mathematics, and the compiler chooses the types.

Three commands. First, measure how large the program's values actually get,
on the exact interpreter — a bound needs a scale, and static intervals cannot
supply one for a network:

```sh
presk calibrate --precision tutorial/07_train.psk --entry loss --dim B=4 \
    --bits 128 --out out/tut_env.json
```

```text
presk calibrate: 30 sites over 1 samples of `loss` at 128 bits
  loss/%6                      7.4759e0
  RMSNorm.apply/%3             4.0212e0
  loss/%5                      2.7342e0
  loss/%3                      2.2073e0
  loss/%7                      2.1038e0
presk calibrate: 3 sums and contractions, worst condition numbers Σ|xᵢwᵢ| / |Σxᵢwᵢ|:
  Linear.apply/%3              1.1620e0
  RMSNorm.apply/%4             1.0000e0
  loss/%7                      1.0000e0
presk calibrate: wrote out/tut_env.json
```

It also records, at every sum and contraction, what the terms add up to
without cancelling; divided by the sum itself, that is the sum's condition
number, and here nothing cancels (the worst is 1.16). Then ask what a given
realization costs:

```sh
presk check --precision out/tut_env.json tutorial/07_train.psk --realize f32
```

```text
presk check --precision: `loss` under f32
  magnitude   2.1038e0
  bound    cond      9.4199e-5  (4.48e-5 relative)
  envelope    out/tut_env.json over 1 samples, 30 sites, initial weights
```

That bound is *sound*, and since 0.6.0 it uses each sum's measured condition
number (`--model cond`, the default). On a program this small it is worth
reading: 4.5·10⁻⁵ of the loss in `f32`. On a network it is not — ten orders
of magnitude above the error actually committed on the corpus GPT, as the
`precision` chapter shows at length. Which is why the third command does not
rely on it:

```sh
presk solve tutorial/07_train.psk --precision out/tut_env.json \
    --budget 1e-3 --out out/tut_solved.json --sites 0
```

```text
presk solve: `loss` under budget 1.00e-3 relative (2.104e-3 absolute, |output| 2.104e0)
  sites       11 narrowable, 11 assigned away from f64
  parameters  4 offered, 4 stored away from f64 (3 f32, 1 bf16): 2.8000e2 bytes at rest against 6.0800e2 in f64
  cost        7.6600e2 against 1.5840e3 all-f64 (2.07x), relaxation floor 3.9600e2
  proposal    7.6600e2 before measurement widened anything (time when --profile is given, bytes otherwise)
  estimate    3.480e-6 (sensitivity-weighted, first order: it ranks, it does not guarantee)
  measured    1.699e-7 against the exact reference, after 0 widening round(s) — inside the budget
  optimal     not searched (the space is too large); the relaxation floor above is the lower bound
presk solve: wrote out/tut_solved.json — run it with `--realize @out/tut_solved.json`
```

Eleven sites, out of the envelope's thirty: the values the program
**stores**, each a computation's result that is not fused into another,
the results of the functions among them (since 0.12.0). A type means
something only there. A parameter's field, a call's result or a member of
a fused group carries a value that another site computes, or is computed in
its group's root's type, so the solver does not offer it (since 0.11.0;
until then it offered all twenty-two and counted them, D112). And the four
parameters, by their paths: a parameter is stored once in the type the
realization gives it (`params`), and every nest reads it from there, so
its type is a choice too (since 0.12.0). With eleven sites and four
parameters the space is too large for the exhaustive search that found the
greedy proposal optimal on 0.11.0's seven.

Read the `estimate` and `measured` lines together, because they are the
point. The **estimate** ranks: it says where the budget should go, from one
reverse sweep of the derivative the compiler already computes. The
**measurement** decides: the candidate is run and compared against the exact
denotation, and 3.663e-8 is inside the 2.104e-3 the budget allows. On this
program the bound would have accepted it too; on a network the bound is too
loose to accept anything, which is why a budget is discharged by
measurement.

The file it writes is a realization, and `--realize @FILE` runs it:

```sh
presk run tutorial/07_train.psk --entry loss --dim B=4 \
    --backend native --realize @out/tut_solved.json --digits 10
```

```text
ℝ[] = 2.103758097
```

Two honest notes to end on. Every stored value got a narrower type and the
program moves half as many bytes — but on these shapes that does **not** make
it faster, because the kernels are bound by arithmetic rather than by memory
(the `experiments` chapter's statement E, which fails; step 16 says what
changed that). And the budget is about the distance to the *mathematics*,
not the quality of the model: a degradation beyond it, on the inputs it was
measured on, is the compiler's, and below it is the model's own sensitivity —
both by measurement, not by proof.

## 16. A budget on the training step

Step 15 put a budget on an entry, `loss`. A training step is mostly *other*
functions — the reverse sweep and the optimizer — and a budget on the loss
gives the compiler no licence over them; on the course's GPT that is exactly
why a solved realization bought 1.13× where all-`f32` bought 1.87×. So since
0.6.0 a budget in a `training run` binds the step itself: **one optimizer
step's parameter update**, measured against the exact step from the same
state,

```text
‖θ̂′ − θ′‖₂ / ‖θ′ − θ‖₂ ≤ ε
```

where `θ′` is the objective's `vjp` followed by `adamw.step` over the reals,
and `θ̂′` the same step under a realization. First, measure one step. The
state is the tower trained in step 11, followed by twenty steps of its own
pipeline in `f64` — the schedule's warm-up, so that AdamW's moments are not
the zeros of initialisation and the measured step is at the peak learning
rate — and the batch is eight samples, the run's own size:

```sh
presk calibrate --step tutorial/07_train.psk --weights out/tutorial_ckpt/final.psk.d \
    --warmup 20 --samples 8 --realize f64 --realize f32 --realize bf16 --realize recipe \
    --out out/tut_step.json
```

```text
presk calibrate: step 21 of 160 from out/tutorial_ckpt/final.psk.d after 20 warm-up step(s) in f64, on samples 0..8 of tutorial/data/pairs.psk.d (noise: the next 8)
presk calibrate: 245 sites in 18 functions of one exact step at 256 bits (0.0 s)
  |θ′ − θ|    3.7830e-2   |g| 5.6012e-1   |θ| 5.2636e0
  noise       7.1633e-1   ‖g_B − g_B′‖₂ / ‖g_B‖₂ between two batches of 8 (f64): what a rounding error is to be compared with
realization                    update gap storage floor     grad gap    predicted B
f64                             7.577e-15     7.565e-15    3.600e-16       6.7632e4
f32                              6.963e-6      3.826e-6     2.737e-7       3.3816e4
bf16                              1.000e0      2.111e-1     2.034e-2       1.6908e4
recipe                           8.659e-3      3.826e-6     2.386e-2       2.0496e4
presk calibrate: wrote out/tut_step.json
```

One row per realization, each run natively from the state rounded to its
type and compared with the exact step at 256 bits:

- The **update gap** is what the budget binds.
- The **storage floor** is the exact `θ′` merely rounded to the type the
  optimizer keeps it in. `θ` is some 140 times the step here, so that
  rounding alone costs 3.8·10⁻⁶ of the update in `f32`, and no `f32`
  realization can go below it; `f64`'s whole gap *is* its floor.
- The **gradient gap** says where the error enters. In `f32` the update is 25
  times further from the exact one than the gradient: AdamW divides by
  `√v`, which amplifies the gradient's error, and that is why the budget binds
  the update rather than the gradient (545 times on the corpus GPT, D93;
  340 in D52, whose training used a gradient defect 3f made wrong).
- **`bf16` loses the step.** An update gap of exactly one means `θ̂′` is as
  far from `θ′` as `θ` is: a learning-rate-sized change is below `bf16`'s
  resolution, so the parameters come back unchanged. That is why practice
  keeps master weights in `f32`, and the **recipe** row is mixed precision as
  it is practised — `bf16` everywhere but the optimizer and its weights —
  8.7·10⁻³ from the exact step, a number no framework can produce. Against
  the sampling noise between two batches, 0.72, it is small, which is the
  recipe's justification.

Then solve the step under a budget of 10⁻³:

```sh
presk solve --step tutorial/07_train.psk --precision out/tut_step.json \
    --budget 1e-3 --out out/tut_step_solved.json
```

```text
presk solve: step 21 of 160 from out/tutorial_ckpt/final.psk.d after 20 warm-up step(s) in f64, on samples 0..8 of tutorial/data/pairs.psk.d (noise: the next 8)
presk solve: the step under an update budget of 1.00e-3 (‖θ̂′−θ′‖₂/‖θ′−θ‖₂), 18 functions, 38 native steps measured
function                        sites     f64 cost     gap f32    gap bf16   chosen
Block.apply                         6      1.216e3     5.15e-8     2.67e-3      f32
Block.apply.bwd                    22      5.376e3     8.76e-9     6.87e-4      f32
Block.apply.fwd                     9      5.056e3     1.37e-8     4.13e-4     bf16
Block.apply.vjp                     3      3.136e3    7.58e-15    7.58e-15     bf16
Linear.apply                        4      6.720e2     2.34e-8     4.37e-3      f32
Linear.apply.bwd                   11      1.792e3     1.28e-8     6.61e-4      f32
RMSNorm.apply                       8      1.000e3     6.96e-8     4.19e-3      f32
RMSNorm.apply.bwd                  25      4.272e3     3.98e-8     2.40e-3      f32
RMSNorm.apply.fwd                  11      2.664e3     3.70e-8     2.75e-3      f32
RMSNorm.apply.vjp                   3      1.696e3    7.58e-15    7.58e-15     bf16
Tower.apply                         4      1.120e3    7.58e-15    7.58e-15     bf16
Tower.apply.fold_bwd                7      3.136e3    7.58e-15    7.58e-15     bf16
Tower.apply.vjp                    19      1.021e4     2.83e-9     1.78e-4     bf16
adamw.lr                           29      2.320e2     5.46e-8     1.10e-3      f32
adamw.step                         37      1.412e4     8.73e-6      1.06e0      f32
loss.bwd                           17      7.208e3     1.61e-8     5.91e-4     bf16
loss.fwd                            9      2.576e3     2.08e-8     2.38e-3      f32
loss.vjp                            3      2.152e3    7.58e-15    7.58e-15     bf16
  chosen      0 f64, 10 f32, 8 bf16
  cost        2.5388e4 against 6.7632e4 all-f64 (2.66x), bytes
  estimate    7.422e-4 (√Σ of the chosen functions' isolated gaps: it proposes, it does not guarantee)
  measured    8.537e-4 update gap (storage floor 3.803e-6), 2.723e-3 gradient gap, after 1 widening round(s) — inside the budget
presk solve: wrote out/tut_step_solved.json — train with it with `presk train --realize @out/tut_step_solved.json`
```

Every function of the step is run alone in `f32` and in `bf16`, everything
else in `f64`, and measured against the one exact step; the functions are
then narrowed greedily by bytes saved per unit of gap, and the combination is
measured, not predicted. Here it keeps the update inside the budget with
2.66 times fewer bytes, after widening once what its estimate put too close.
The functions are those of a gradient that keeps its forward pass's values
(the `language` chapter, "Residuals"): `Block.apply.fwd` runs a block once
and `Block.apply.bwd` its reverse sweep, and a `.vjp` only calls the two,
which is why five functions show no gap at all; the tower itself, generic in
its leading axes (`...`), keeps its recomputing `Tower.apply.vjp`. With
`--profile`, the objective is predicted seconds
instead of bytes, and on the corpus GPT that objective refuses `bf16`
everywhere: its conversions cost more than its traffic saves. The file is a
realization whose `functions` map gives each function its type, and `presk
train` takes it. A per-step budget does not bound a whole run, so measure
where the run ends, against step 12's `f64` run and against a change of
seed:

```sh
presk train tutorial/07_train.psk --steps 100 --log-every 20 --backend native \
    --realize @out/tut_step_solved.json --ckpt out/tutorial_ckpt_step
presk data diff out/tutorial_ckpt_native/final.psk.d out/tutorial_ckpt_step/final.psk.d
presk train tutorial/07_train.psk --steps 100 --log-every 100 --backend native \
    --seed 1 --ckpt out/tutorial_ckpt_seed1
presk data diff out/tutorial_ckpt_native/final.psk.d out/tutorial_ckpt_seed1/final.psk.d
```

```text
step 20  epoch 2  loss 1.472400  lr 1.000e-2  |g| 2.0692e0
step 40  epoch 3  loss 0.843814  lr 8.536e-3  |g| 8.2627e-1
step 60  epoch 4  loss 0.521474  lr 5.000e-3  |g| 6.9435e-1
step 80  epoch 5  loss 0.432605  lr 1.464e-3  |g| 5.8487e-1
step 100  epoch 7  loss 0.320642  lr 0.000e0  |g| 4.6503e-1
presk train: 184 kernel launches a step, 332 without fusion (45% removed)
presk train: 0.6 ms a step, median of 99 (the first excluded)
presk train: 100 steps of Tower, final loss 0.320642, weights in out/tutorial_ckpt_step/final.psk.d
presk data: ‖a − b‖₂ 1.210527e-3   ‖a‖₂ 5.344120e0   relative 2.265156e-4   over 76 reals
step 100  epoch 7  loss 0.296131  lr 0.000e0  |g| 4.6998e-1
presk train: 184 kernel launches a step, 332 without fusion (45% removed)
presk train: 0.6 ms a step, median of 99 (the first excluded)
presk train: 100 steps of Tower, final loss 0.296131, weights in out/tutorial_ckpt_seed1/final.psk.d
presk data: ‖a − b‖₂ 4.842373e0   ‖a‖₂ 5.344120e0   relative 9.061123e-1   over 76 reals
```

The log moves in the fifth digit against step 12's, which is the three
`bf16` functions that compute showing. The weights end 2.3·10⁻⁴ from the
`f64` run's, where
another seed ends 0.91 away: four thousand times closer than the ordinary
variation of training, and over a hundred steps nothing accumulated into a
different model.

Two honest notes again. On a tower of width four none of this is faster:
every kernel is dispatch-bound, so time is not what the tower can show. At
the course's scale the same three commands give the `experiments` chapter's
statement J′: the corpus GPT's step, solved under 10⁻³, runs 1.99× faster
than all-`f64` and stays inside the budget by measurement. And the budget is
stated at a state, here the peak learning rate, because the floor grows as
the rate falls: at the end of a cosine schedule every realization, `f64`
included, would fail a relative budget on the update (question 61).


## 17. A plan for attention

Every step so far has taken the kernels as the compiler lowers them by
default. Since 0.7.0 how a loop nest is laid out is a decision of its own,
and the attention of step 3 has two nests where it matters: the scores `ein`
(`q · k`, summed over `hd`) and the values `ein` (`a · v`, summed over
`seq2`). Each can be lowered many ways: a **register tile** of `m_i × n_j`
accumulators, a **block** over the summed axis, an **order** of the block
loops, vector **lanes**, a **split** over threads at a loop depth, and, on the
scores, a **pack** that copies `k` to a panel the lanes can read. Every one of
these points computes the same bits, since each output element adds its
terms in the same order whatever the layout (`native` chapter,
"Contractions"). Only the time differs. Whether a static cost model can
choose among them as well as measuring them all is the research programme's
hypothesis RA4, and the four commands of this step ask it on the tutorial's
attention.

A model that reads a layout needs constants fitted on this machine. It is
calibrated on a matrix product, a program the attention never enters, so
that what it later says about attention is a prediction and not a fit:

```sh
presk bench experiments/gemm.psk --entry gemm --dim N=256 --realize f32 \
    --calibrate out/tut_layout.json --space calib --repeat 5
presk bench experiments/gemm.psk --entry gemm --dim N=512 --realize f32 \
    --calibrate out/tut_layout.json --space calib --repeat 5
```

```text
-- gemm/%2 on cpu0 (f32): L1d 64 KiB, L2 512 KiB, L3 8192 KiB shared by 10, 20 workers, 32 vector registers
-- costed under the rules lines,overhead
layout                                                           group               measured µs predicted µs
…
-- 300 layouts timed (0 ill-formed at these shapes), 300 samples kept from out/tut_layout.json, 5 calls each
-- fit (p-norm, q = 1.00, load term, rules lines,overhead): λ = 0.88 µs, spawn = 0.00 µs, β_L2 = 4.151e11, β_L3 = 8.879e11, β_mem = 1.556e12 bytes/s
--      π[f32+lanes1+w1] = 1.331e10 ops/s
--      π[f32+lanes1+w10] = 6.604e9 ops/s
--      π[f32+lanes1+w20] = 7.123e9 ops/s
--      π[f32+lanes4+w1] = 1.655e11 ops/s
--      π[f32+lanes4+w10] = 1.608e11 ops/s
--      π[f32+lanes4+w20] = 2.837e17 ops/s
--      ρ[f32+lanes1+w1] = 6.220e10 loads/s
--      ρ[f32+lanes1+w10] = 7.789e9 loads/s
--      ρ[f32+lanes1+w20] = 6.260e9 loads/s
--      ρ[f32+lanes4+w1] = 5.534e9 loads/s
--      ρ[f32+lanes4+w10] = 1.678e9 loads/s
--      ρ[f32+lanes4+w20] = 1.439e9 loads/s
-- uncertainty u = 0.101 (median relative residual over 600 samples), Spearman 0.985
-- profile written to out/tut_layout.json
```

Each command times the product under 300 layouts — five register tiles,
`[1, 4]` to `[4, 16]`; the orders `IJ` and `JI`, every order of `IJK` with a
block of 128, and two with a block of 32; one lane or four; one worker, half
the machine or all of it — and the second keeps the first's samples before
fitting again. The first line is the machine as the model reads it, for the
first CPU the process may run on: nothing is pinned here, so that is `cpu0`,
on this machine one of the ten small cores, with a quarter of the L2 of the
ten large ones.

The model is 0.8.0's (`optimization` chapter). It reads five terms of one
worker's share of a kernel: its work over a rate `π`, its register loads
and stores over a rate `ρ`, and the bytes it brings from L2, L3 and memory
over three bandwidths `β`. They combine in a p-norm whose exponent `q` is
fitted, here 1.00, the sum. `π` and `ρ` are fitted per group: a type, a
lane width and a worker count. That is the form 0.8.0 froze before testing
it, and `--space calib`'s default: the p-norm, footprints counted in whole
cache lines (`-- costed under the rules lines,overhead`), and one operation
of work per loop iteration; `--combine` and `--rules` choose another. `u` is the
model's own uncertainty, the median relative error of the fit: 10 %. One
rate came out near 10¹⁷, `π` in four lanes on twenty workers: the fit is
better without that term than with any rate, so it makes it vanish; in
four lanes on many workers the work never shows apart from the loads. And
starting the workers costs nothing the fit can see, `spawn = 0.00 µs`:
since 0.10.0 the native backend's workers are a pool that waits spinning
between launches (`native` chapter). `make calibrate-ra4` does this properly, at four sizes and two
thin products in two types, pinned to like cores with `taskset`.

Now choose a layout for every nest of the entry, statically:

```sh
presk plan tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --realize f32 \
    --profile out/tut_layout.json --out out/tut_plan.json
```

```text
nest                         operands                        space     predicted          undecided  layout
scores/%2                    [16, 2, 256, 4] [16, 2, 256, 4]   413 points       93.72 µs    3 within 1 + u  tiled, tile [4, 16], kblock none, order JI, lanes 4, split (0, 20), pack
scores/%5                    []                           1 points        0.88 µs    1 within 1 + u  lanes 1, split none
scores/%6                    [16, 2, 256, 256] []         7 points      114.08 µs    1 within 1 + u  lanes 1, split (2, 20)
attend/%5                    [16, 2, 256, 256] [256, 256]    14 points      610.03 µs    1 within 1 + u  lanes 1, split (2, 20), passes 2
attend/%6                    [16, 2, 256, 256] [16, 2, 256, 4]   577 points      133.24 µs   14 within 1 + u  tiled, tile [4, 4], kblock none, order JI, lanes 4, split (0, 20), no pack
-- 5 nests, 1012 points costed in 0.046 s, the entry captured in 0.007 s; u = 0.101
-- wall time 0.054 s
-- plan written to out/tut_plan.json
```

The entry's launches are captured from its shapes: the host program runs on
zero inputs and records each nest's launch without running a kernel. Then
every nest is lowered under every point of its space, thread splits
included, and each point is costed. Nothing is compiled or timed: 1 012
points in a twentieth of a second. The nests are named as `presk dump
--stage core` names their bindings: `scores/%2` and `attend/%6` are the two
`ein`s, `attend/%5` the softmax, which reads the causal mask, its second
operand, and takes two passes instead of three.
The model picks the largest register tiles the nests allow, `[4, 16]` with
the pack on the scores and `[4, 4]` on the values, whose `J` is `hd = 4`,
split over all twenty workers. The `undecided` column counts the points the
model predicts within `1 + u` of its best: on the two `ein`s, 3 and 14
points it cannot tell from its choice. The plan is a JSON file (`realize`
chapter), and `--plan` runs it:

```sh
presk run tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --backend native \
    --realize f32 > out/tut_attend.txt
presk run tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --backend native \
    --plan @out/tut_plan.json > out/tut_attend_plan.txt
cmp out/tut_attend.txt out/tut_attend_plan.txt && echo "same output"
```

```text
same output
```

The results are printed to twelve digits, more than an `f32` needs, so the
same text is the same bits. That is why the layout lives in a plan beside the
realization and not inside it: a realization owes a measured gap to the exact
reference, and a layout owes none. (`presk bench … --plan @out/tut_plan.json`
adds a row that times the entry under the plan.)

Was the choice right? Measure every point, twice:

```sh
presk autotune tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --realize f32 \
    --space small --profile out/tut_layout.json --interp --out out/tut_tune/day1
presk autotune tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 --realize f32 \
    --space small --profile out/tut_layout.json --out out/tut_tune/day2
```

```text
scores/%2                    [16, 2, 256, 4] [16, 2, 256, 4]    76 of    76 points  out/tut_tune/day1/scores__2.jsonl
scores/%5                    []                           1 of     1 points  out/tut_tune/day1/scores__5.jsonl
scores/%6                    [16, 2, 256, 256] []         3 of     3 points  out/tut_tune/day1/scores__6.jsonl
attend/%5                    [16, 2, 256, 256] [256, 256]     6 of     6 points  out/tut_tune/day1/attend__5.jsonl
attend/%6                    [16, 2, 256, 256] [16, 2, 256, 4]     4 of     4 points  out/tut_tune/day1/attend__6.jsonl
-- 90 points of 5 nests measured in 2.8 s, costed in 0.009 s; every output bit-identical to point 0, natively and on the kernel interpreter: yes
```

`--space small` is a daily subset of each space. Every point is compiled and
timed seven times on the inputs captured from the entry, one JSON line per
point with the times and the model's prediction, and its output is compared
bit for bit with point 0, the default layout on one worker without lanes;
with `--interp` also on the kernel interpreter. Then rank the model against
the two days:

```sh
presk rank out/tut_tune/day1 out/tut_tune/day2 --entry attend --out out/tut_rank.md \
    | sed -n '/^## Statements/,/^| tile/p'
```

```text
## Statements (plan 0.7.0 §3.7; Q, plan 0.8.0 §3.6)

| | statement | measured | verdict |
|---|---|---|---|
| L | every point bit-identical to point 0, natively and on the kernel interpreter | 95 of 95 points of day 1 checked on the interpreter; 0 outputs differing, natively or on the interpreter, in any sweep | **holds** |
| M | more than 90 % of the distinguished close pairs ranked right, each `ein` of `attend` at B=16,T=256, f32 | `attend/%6` 66.7 % of 3; `scores/%2` 77.5 % of 178 | inconclusive: fewer than 10³ distinguished pairs |
| N | choosing statically at least 10³× cheaper than the exhaustive sweep of the same points, `attend` | B=16,T=256 f32 254× (74× with the entry run); B=16,T=256 f32 238× (69× with the entry run) | **fails** |
| O | the model's pick within 1.25× of the best measured, every contraction nest of `attend` at B=16,T=256, f32 | `attend/%6` 1.00; `scores/%2` 1.01 | **holds** |
| P | the best point of the GEMM space at least 3× the 0.6.0 default, `gemm` at N=1024, f32 | — | not measured |
| Q | measuring only the points predicted within 1 + u of the model's best finds one within 1.05× of the best measured, at no more than 10 % of the exhaustive sweep's time (question 72: the static side and the candidates' measured time), every contraction nest of every configuration swept whole | 2 contraction nests: the best candidate within 1.01× of the best measured at worst (`scores/%2` B=16,T=256 f32); the static side and the candidates' measured time at most 19.2 % of the nest's exhaustive sweep (`attend/%6` B=16,T=256 f32); their share of its timed launches at most 5.30 % (`attend/%6` B=16,T=256 f32, D71's count) | **fails** |

## Statement Q, per contraction nest

*Candidates*: the points predicted within 1 + u of the model's best. *Regret*: the best candidate over the best point measured, the mean of the two days' medians. *Cost* (question 72): statement N's static side, the whole configuration's, charged to each nest, and the candidates' measured time, over the exhaustive sweep of the nest's points. A point's measured time is its launches, the warm-up and the timed ones, at its median, and its time outside them (compiling it, comparing its bits), which a sweep records only as a whole: the sweep's time less its points' launches, over its points, 0.6.0's included (the dry run's estimate, D72). *Kernel share* (D71's count): the candidates' timed launches over the nest's. Each day apart, day one first.

| nest | config | type | points | candidates | regret | cost | static, candidates, exhaustive (s) | kernel share | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `attend/%6` | B=16,T=256 | f32 | 4 | 1 | 1.00 | 19.2 % / 19.0 % | 0.01, 0.02, 0.2 / 0.01, 0.02, 0.1 | 5.26 % / 5.30 % | **fails** |
| `scores/%2` | B=16,T=256 | f32 | 76 | 2 | 1.01 | 2.4 % / 2.3 % | 0.01, 0.03, 1.7 / 0.01, 0.03, 1.6 | 0.52 % / 0.51 % | **holds** |

## Per decision class

| class | close pairs | distinguished | right | right, decided | undecided | tied |
|---|---|---|---|---|---|---|
| lanes | 24 | 24 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| order | 36 | 7 | 42.9 % | — | 100.0 % | 0.0 % |
| pack | 24 | 13 | 23.1 % | — | 100.0 % | 0.0 % |
| passes | 3 | 3 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| split | 87 | 79 | 69.6 % | 69.6 % | 0.0 % | 0.0 % |
| tile | 112 | 67 | 91.0 % | 94.6 % | 44.8 % | 0.0 % |
```

The report's unit is a **close pair**: two points of a nest that differ in one
decision. A pair is *distinguished* when its two median times differ by more
than twice the larger median absolute deviation, on both days, with the same
sign; the model orders it *right* when its predictions order it as the
measurement does, and a pair it predicts equal is *tied*, which is not right.
The per-class table is where to read it. The model is right on every pair
of lanes and of softmax passes, which its terms read (a rate per lane
width, the loads of a pass), and on 91 % of the register tiles, which its
load term reads: 0.7.0's model, with no such term, ordered 39 % of them
here and tied 41 %. It orders 70 % of the thread splits, nearly all of them
the scores', by rates per worker count fitted on the product alone; and
three of the seven loop orders and three of the thirteen packs the
measurement distinguishes, all pairs it predicts within `1 + u` of each
other.

The statements are 0.7.0's five and 0.8.0's `Q`. `L` holds, every point
exact on both executors. `M` asks for more than 90 % of at least a thousand
pairs and is inconclusive at this size. `N` fails because a point takes
milliseconds to time, so measuring them all costs only about 250 times
more than costing them. `O` holds: the model's pick among the scores'
points is 1.01 times slower than the best one measured, within 1.25. `P` is about a
matrix product this sweep does not include. `Q` asks whether measuring only
the model's candidates, the points within `1 + u` of its best, finds a
point within 1.05× of the best at a tenth of the sweep's cost. It holds on
the scores, whose two candidates hold a point within 1.01× of the best at
2.4 % of the sweep, and fails on the values `ein`, which has four points in
all, so that choosing statically and measuring its one candidate costs a
fifth of sweeping it. The table after the statements gives each
nest's candidates, regret and cost.

These sizes test the tools, not the hypothesis: at `hd = 4` the scores sum
four terms. The measurements that decide RA4 are on the course GPT's
attention, every point of every space, pinned and measured twice (the
`experiments` chapter's §7 and §8). 0.7.0's model failed there, since it
counted no load from the first-level cache into registers. 0.8.0's model
was designed on those measurements, frozen, then tested on two fresh sweeps.
It ranks 92.7 % and 91.7 % of the two `ein`s' pairs right on the small
cores at 0.7.0's shapes, but 85.4–88.8 % at BERT-large's attention and at a
head of 128 on the large cores, so `M` fails. `O` and `Q` hold there, and
`N` fails at batch one. For numbers you mean to keep, pin the process to
like cores, as `make ra4-small` does: the machine description, the worker
count and the noise all depend on it.

## 18. A C program

Everything so far has run inside `presk`. Since 0.9.0 an entry can leave it
as a C program, in the style of llama2.c: a few files a C compiler builds on
any machine, which run without the compiler and compute what `presk run
--backend native` computes, bit for bit. Export step 3's attention:

```sh
presk export tutorial/03_attention.psk --target c --entry attend --dim B=1 --dim T=3 -o out/tut_c
```

```text
presk export: `attend` at B = 1, T = 3 in f64: 6 launches (0 dead, on one thread) of 6 kernels, 1 function besides the entry's and 0 loops, an arena of 1280 bytes (1600 released nothing), into out/tut_c; `make -C out/tut_c check` builds it and compares it with the native backend
```

The export runs the entry once on the native backend, at step 3's
dimensions and on the inputs `presk run` would synthesize, and records
what its host does: every kernel it launches, in order, with the buffers
each reads and writes. The program is that recording in C. `kernels.c`
holds the six kernels, each the native backend's, printed as one C
function under its PSK-Kernel text, which the compiler inlines wherever
the host launches it (`PRESK_KERNEL`, a static inline function, always
inlined):

```c
// kernel ein[batch, heads, seq, seq2] (%0 : f64[batch: 1, heads: 2, seq: 3, hd: 4], %1 : f64[batch: 1, heads: 2, seq: 3, hd: 4]) -> %2 : f64[batch: 1, heads: 2, seq: 3, seq2: 3] {
// …
PRESK_KERNEL void k0(const double *restrict in0, const double *restrict in1, double *restrict out, int64_t lo, int64_t hi) {
    int64_t i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0;
    double s0 = 0.0, s1 = 0.0;
    for (i0 = lo; i0 < hi; i0++) {
        for (i1 = 0; i1 < 2; i1++) {
            for (i2 = 0; i2 < 3; i2++) {
                for (i3 = 0; i3 < 3; i3++) {
                    s0 = 0.0;
                    for (i4 = 0; i4 < 4; i4++) {
                        s1 = in0[i0 * 24 + i1 * 12 + i2 * 4 + i4];
                        s0 = s0 + s1 * in1[i0 * 24 + i1 * 12 + i3 * 4 + i4];
                    }
                    out[i0 * 18 + i1 * 9 + i2 * 3 + i3] = s0;
                }
            }
        }
    }
}
```

`model.c` holds the host: a C function per Presk function, here `scores`
(the "1 function besides the entry's") and the entry, which calls it and
launches the mask, the softmax and the values `ein`, each buffer named
after its `let` and taken from one arena whose size the export computed;
the mask is a buffer of indices, `1` where a position is allowed:

```c
void presk_attend(presk_arena *arena, const presk_attend_in *in, presk_attend_out *out)
{
    const double *q = in->q;
    const double *k = in->k;
    const double *v = in->v;
    double *result = presk_alloc(arena, 24 * sizeof(double));
    const size_t mark = presk_mark(arena);
    /* attend/%3 (s) */
    double *s = presk_alloc(arena, 18 * sizeof(double));
    presk_scores(arena, q, k, s);
    /* attend/%4 */
    int64_t *t4 = presk_alloc(arena, 9 * sizeof(int64_t));
    k3(t4, 0, 3);
    /* attend/%5 (a) */
    double *a = presk_alloc(arena, 18 * sizeof(double));
    k4(s, t4, a, 0, 1);
    /* attend/%6 */
    k5(a, v, result, 0, 1);
    out->result = result;
    presk_release(arena, mark);
}
```

Beside them: `main.c`, which reads the inputs from a `.psk.d` file, calls
`presk_attend` and writes the outputs; the runtime; `presk_libm.c`, the
`exp` the softmax calls, musl's as the native backend's `libm` crate ports
it; a `Makefile`; a `README.md` that says all of this for this program; and
`conformance/`, the inputs, the native backend's outputs and the exact
reference's. Build it and compare:

```sh
make -C out/tut_c check PRESK=$PWD/target/debug/presk
```

```text
cc -std=c11 -ffp-contract=off -O2 -Wall -Wextra  -o model main.c model.c presk_rt.c presk_libm.c -lm
sh conformance/check.sh
C1 holds: the outputs are the native backend's, bit for bit
against the exact reference:
presk data: ‖a − b‖₂ 3.477764e-16   ‖a‖₂ 4.687967e0   relative 7.418490e-17   over 24 reals
```

(`make`'s lines entering and leaving the directory are left out here.) The
script runs the program on the inputs and compares its outputs with the
native backend's, byte for byte, with `cmp`; `PRESK` names the compiler, so
that it also prints the distance to the exact reference. Two flags make the
bits the same: `-std=c11 -ffp-contract=off` forbids the compiler to fuse a
product and a sum into one rounding, which the native backend never does
(GCC does by default, in its GNU modes), and the program calls its own
`exp`, not the C library's: glibc's differs from the native backend's in the
last bit on one to ten inputs in a hundred, depending on the function. Run
it by hand:

```sh
out/tut_c/model out/tut_c/conformance/inputs.psk.d out/tut_c/out.psk.d
presk data info out/tut_c/out.psk.d
```

```text
type    ℝ[1, 2, 3, 4]
count   1
version 1
sample  192 bytes, 1 leaves
[0] [[[[-0.2630606226931485, -1.227189555438751, -0.522860365685521, -0.6501994362402844], [-0.455980730477033, -1.0326643349418838, -0.21637551683825543, -0.5595661867201479], [-0.7999946376879477, -0.6410120274962912, 0.3293447034493807, -0.18354497498929842]], [[-1.1081571575929285, -1.3949974763568258, -1.1328024940237533, 0.420503653957042], [-1.5352932984844576, -0.9314255802951602, -1.7831217000720225, 0.3615592894447046], [-1.6226084683183153, -0.6828903979005011, -1.547601863846864, 0.4638027209419699]]]]
```

Step 3's result, to its five digits, in `f64`. The program is the native
backend's run, so it follows what that run followed. At step 17's size and
under its plan:

```sh
presk export tutorial/03_attention.psk --target c --entry attend --dim B=16 --dim T=256 \
    --plan @out/tut_plan.json -o out/tut_c16
make -s -C out/tut_c16 check
OMP_NUM_THREADS=1 make -s -C out/tut_c16 check
```

```text
presk export: `attend` at B = 16, T = 256 in f32 under the plan out/tut_plan.json: 7 launches (0 dead, 4 of them divided among up to 20 threads) of 7 kernels, 1 function besides the entry's and 0 loops, an arena of 17825792 bytes (26345600 released nothing), into out/tut_c16; `make -C out/tut_c16 check` builds it and compares it with the native backend
C1 holds: the outputs are the native backend's, bit for bit
C1 holds: the outputs are the native backend's, bit for bit
```

The kernels are now the plan's, in `f32`: the register tiles, the lanes,
the pack of `k`. Each launch the plan splits is an OpenMP region whose
threads each run the kernel on their piece of the divided loop, the pieces
the native backend computes. Each `ein` is split at its batch loop, of 16
iterations, among the plan's 20 workers, so it runs on 16 of them; the
values `ein`:

```c
#ifdef _OPENMP
#pragma omp parallel num_threads(presk_team(16))
    {
        int64_t t = omp_get_thread_num(), w = omp_get_num_threads();
        k6(a, v, result, 16 * t / w, 16 * (t + 1) / w);
    }
#else
    k6(a, v, result, 0, 16);
#endif
```

Twenty is the plan's number, the workers of step 17's calibration, which
ran unpinned on this machine's twenty cores; on another machine the plan,
and so the program, says what its calibration saw. The team is at most the
threads OpenMP may use, so `OMP_NUM_THREADS=1` runs every region on one
thread, and the second check shows the same bits: each element is computed
by one thread, by the same loops in the same order, whatever the team.
Built without OpenMP (`make OPENMP=`) the regions are plain calls, again the
same bits. `-r` times the entry:

```sh
out/tut_c16/model -r 11 out/tut_c16/conformance/inputs.psk.d out/tut_c16/out.psk.d
OMP_NUM_THREADS=1 out/tut_c16/model -r 11 out/tut_c16/conformance/inputs.psk.d out/tut_c16/out.psk.d
```

```text
presk_attend: 11 calls, median 2.455 ms, best 2.048 ms
presk_attend: 11 calls, median 8.057 ms, best 8.003 ms
```

The `export_c` chapter compares the program's speed with the native
backend's, on the course GPT (statement C4).

A polyhedral compiler would transform the same loop nests that step 17's
plan chooses layouts for, and comparing the two is a question for after
0.9.0. `--mode scop` prepares it: it writes each nest of the entry on its
own, at point zero (its default layout, one lane, one worker), in the form
such tools read:

```sh
presk export tutorial/03_attention.psk --target c --mode scop --entry attend --dim B=16 --dim T=256 \
    --realize f32 -o out/tut_scop
make -s -C out/tut_scop check
```

```text
presk export: `attend` at B = 16, T = 256 in f32: 5 nests at 5 shapes at point zero, a C function each, 5 of them static-control parts and 0 not (`nests.json` says why), into out/tut_scop; `make -C out/tut_scop check` builds them and compares each with the native backend
the 5 nests compute the native backend's bits
```

The five nests are step 17's. Each is a file of its own, `nests/attend_5.c`
for the masked softmax:

```c
void attend_5(const float in0[restrict 16][2][256][256], const int64_t in1[restrict 256][256], float out[restrict 16][2][256][256]) {
    int64_t i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
#pragma scop
    for (i0 = 0; i0 < 16; i0++) {
        for (i1 = 0; i1 < 2; i1++) {
            for (i2 = 0; i2 < 256; i2++) {
                s2 = 1.0f;
                s0 = 0.0f;
                for (i3 = 0; i3 < 256; i3++) {
                    s0 = (float)in1[i2][i3] > 0.5f ? (s2 > 0.5f ? in0[i0][i1][i2][i3] : in0[i0][i1][i2][i3] > s0 ? in0[i0][i1][i2][i3] : s0) : s0;
                    s2 = (float)in1[i2][i3] > 0.5f ? 0.0f : s2;
                }
                s1 = 0.0f;
                for (i3 = 0; i3 < 256; i3++) {
                    s1 = s1 + ((float)in1[i2][i3] > 0.5f ? presk_expf(in0[i0][i1][i2][i3] - s0) : 0.0f);
                }
                for (i3 = 0; i3 < 256; i3++) {
                    out[i0][i1][i2][i3] = (float)in1[i2][i3] > 0.5f ? presk_expf(in0[i0][i1][i2][i3] - s0) / s1 : 0.0f;
                }
            }
        }
    }
#pragma endscop
}
```

Arrays of literal dimensions, loops to literal bounds, every access affine
in the loop variables, and the running maximum and the mask selections
rather than branches (`s2` says the row's first allowed position is still
to come): a static-control part, the region between the pragmas, which pet,
PPCG and PLUTO read. The check runs each nest on inputs its driver draws
and compares it with the native backend's point zero, bit for bit; `make
-C out/tut_scop time` times each nest. `make llvm` installs LLVM's release,
whose clang links Polly, and `make -C out/tut_scop polly
POLLY=$PWD/.llvm/bin/clang` shows each nest as Polly sees it, one SCoP each
here (`make tutorial` leaves this out: the release is 1.8 GB to download).
A nest that gathers (an embedding's `table[ids[b][t]][d]`) is not static
control, and the README says so.

The exported programs of the test set, under plans and with integer
weights, with Clang and for x86-64, and the generation loop of `presk serve`
exported with `--serve`, are checked by `make test` and `make
c-conformance` (step 14); the `export_c` chapter says why the bits are the
native backend's and what is checked.

## Where to go next

- The `experiments` chapter for what these tools show about the approach
  (`make experiment`): a silent broadcasting bug of plain PyTorch that the
  types rule out, the gap of float32 and float64 training to the exact
  reference with presk's own realizations next to PyTorch's, and the
  falsifiable statements of 0.4.0 to 0.10.0, each answered by
  measurement.
- The `precision` chapter for everything behind steps 15 and 16: the
  envelope, the bound and the condition number, sensitivities, the solver,
  `int8` and `int4` weights (`presk quantize`), and the budget on the
  training step at the course's scale; the `coverage` chapter for BERT-large
  run at MLPerf's constants.
- The `native` chapter for the runtime seam, PSK-Kernel v0 and the
  Cranelift backend with their conformance suites, and every layout of a
  contraction; the `realize` chapter for fusion groups and the plan file of
  step 17; the `optimization` chapter for the rules, the oracle, the
  calibrated roofline and the layout cost model, what its fresh test found; the
  `serve` chapter for `inference serve` and `presk serve`; the `export_c`
  chapter for the C programs of step 18, sampling and integer weights
  included, and the SCoP mode.
- The `cli` chapter for every flag, the `language` chapter for the
  constructs accepted and how they are typed, `diagnostics` for the codes.
- `docs/spec-questions.md` for the points where this implementation had to
  choose what the specification leaves open.
- The specification itself, in the `presk-specs` repository, for the
  design: exact semantics, error budgets, and the research programme beyond
  this compiler.
