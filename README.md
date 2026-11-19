# Presk

Presk is a language for neural networks whose programs have an exact
mathematical meaning, and a compiler that runs them under an error budget
you declare.

A Presk program denotes a function over tensors of real numbers. It has no
machine number type: no `float32`, no `bfloat16`. Shapes, with named and
symbolic axes, are part of the types and are checked before anything runs.
Training and inference are derived from the same program. Precision is a
compilation choice: you say how far a result may stray from the
mathematics, and the compiler chooses, value by value, the machine types
that stay within that budget, then checks its choice by measurement against
an exact reference.

- The manual: [docs.presk-lang.org](https://docs.presk-lang.org), starting
  with its [tutorial](https://docs.presk-lang.org/presk_manual/tutorial/index.html)
- The site: [presk-lang.org](https://presk-lang.org)
- The crate: [`presk`](https://crates.io/crates/presk) on crates.io

## Install

```sh
cargo install presk
presk --help
```

You need a Rust toolchain (1.85 or later), `m4` and a C compiler: the exact
interpreter uses GMP and MPFR, which are built from source the first time
(a few minutes). Programs exported to C need only a C11 compiler. PyTorch is
needed only to run the tests that a PyTorch export generates.

## A first program

Save this as `linear.psk` (it is the tutorial's first program,
`examples/tutorial/01_linear.psk`):

```text
module Linear{In Out : dim} {
  param w : ℝ[In, Out]  init = xavier_uniform
  param b : ℝ[Out]      init = zeros

  fn apply(x : ℝ[..., In]) -> ℝ[..., Out] =
    x · w + b
}
```

`Linear` is a module with two dimension parameters, `In` and `Out`, two
parameters with their initializers, and one function. `ℝ[..., In]` is a
tensor of reals whose last axis has length `In`, after any number of
leading axes. Every Unicode symbol has an ASCII spelling: `Real` for `ℝ`,
`@` for `·`.

Check it, then evaluate `apply` with `In = 3`, `Out = 2` and a batch of four
inputs:

```sh
presk check linear.psk
presk run linear.psk --entry 'Linear{3, 2}.apply' --dim batch=4
```

```text
ℝ[batch: 4, 2] = [batch: 4, 2] [[0.896290523413, -1.22253725991], [0.839769040003, -0.409586930304], [-0.975629573942, 0.748993526688], [-2.519075041, 2.26085457153]]
```

`presk check` prints nothing: the program is well typed. `presk run`
evaluates on the reference interpreter. The parameters come from their
initializers and the input from its type, all from a fixed seed (`--seed`),
and the arithmetic is correctly rounded at 256 bits (`--bits`), so every
machine prints the same digits. `--verbose` also prints the parameters and
the input.

## A budget instead of a type

From a checkout of this repository, on the tutorial's training program:

```sh
# How large the values of `loss` get, measured on the exact interpreter
presk calibrate --precision examples/tutorial/07_train.psk --entry loss --dim B=4 --bits 128 --out env.json
# A machine type for every stored value, such that `loss` stays within 10⁻³ of its exact value
presk solve examples/tutorial/07_train.psk --precision env.json --budget 1e-3 --out solved.json
# Run that choice natively, then the exact value
presk run examples/tutorial/07_train.psk --entry loss --dim B=4 --backend native --realize @solved.json --digits 10
presk run examples/tutorial/07_train.psk --entry loss --dim B=4 --digits 10
```

The two runs print `2.10375828` and `2.103758267`. In the types the solver
chose, the native run is within 1.3·10⁻⁸ of the exact value, where the
budget allowed 2.1·10⁻³; the solver measured that gap before writing its
file. Step 15 of the tutorial explains each line of its report.

## What the compiler does

| Command | What it does |
|---|---|
| `presk check` | Type-checks a program: the shape of every value, the regularity of every function (whether it may be differentiated), its effects |
| `presk run` | Evaluates an entry exactly, on the reference interpreter, or natively (`--backend native`) in the types you give (`--realize f32`, `--realize @solved.json`) |
| `presk train`, `presk serve` | Runs a program's training loop, or its generation loop, on the reference interpreter or natively |
| `presk calibrate`, `presk solve` | Measures a program's values, then turns an error budget into a machine type per value, checked by measurement |
| `presk explain` | Writes one compilation as a document: every stage and every decision with its reason, as Markdown or as an HTML page |
| `presk export` | Exports a program to PyTorch, or an entry as a C program that computes the native backend's bits |
| `presk bench`, `presk plan`, `presk memory` | Times an entry on every backend; chooses each loop nest's layout from a cost model; counts what a call needs in memory |

`presk help COMMAND` lists a command's options, and the manual's
[`cli`](https://docs.presk-lang.org/presk_manual/cli/index.html) chapter
describes them all.

## Where it stands

This is version 0.18.2. The compiler runs on CPUs, with a native backend
that has its own code generator and a second one through `clang`; work on
GPUs has not started. Beside `f64`, `f32` and `bf16`, binary16 (`f16`) and
`e8m15` (`f32`'s exponent with 16 significant bits) are stored in their
format and computed in `f32` by both native backends and the exported C,
with the kernel interpreter's bits, and a value that overflows either
stops the run with an error; on the course GPT they save little time
against `f32` yet. Other number formats (8-bit floats, integers
with a scale, any format a machine may have) run only on the kernel
interpreter (`--backend kernel`), which computes them bit for bit; the
native backends refuse them for now. Serving
recomputes the whole context for every new token. `CHANGELOG.md` lists what
each version brought, and the manual's
[`status`](https://docs.presk-lang.org/presk_manual/status/index.html)
chapter what is implemented.

## The repository

- `crates/`: the compiler, a Rust workspace; the `presk` command is the
  package `presk`, in `crates/presk-cli`.
- `examples/`: every Presk program: the language's reference programs
  (`examples/spec/`), the tutorial's (`examples/tutorial/`), models from an
  NLP course, and transcriptions of MLPerf's architectures, BERT-large and
  Llama among them. `examples/README.md` indexes them.
- `experiments/`: the scripts behind the measurements that the manual and
  `CHANGELOG.md` report.

To build and test from source, run `make build` and `make test`.
`CONTRIBUTING.md` says more, and how to report an issue.

## License and citation

Presk is under the Apache License 2.0 (`LICENSE`, `NOTICE`). To cite it,
use `CITATION.cff`.
