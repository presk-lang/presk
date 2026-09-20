# Presk compiler

Compiler for Presk, a domain-specific language for describing, training and
running neural networks with exact real semantics. The language is specified
in the `presk-specs` repository; its `examples/*.psk` are the reference
programs (copied under `tests/examples/`).

**Version 0.4.0**: the front-end (`presk check`), an arbitrary-precision
reference interpreter with automatic differentiation (`presk run`), a native
backend — PSK-Kernel v0 compiled by Cranelift, with fusion groups, threads
over the parallel axes and blocked contractions, `--backend native --realize
f64|f32` — training (`presk train`) from `.psk.d` data files (`presk data`)
and serving (`presk serve`) on either, an e-graph optimizer with rules exact
over ℝ and a calibrated cost model (`--opt`, `presk bench --calibrate`), and
a PyTorch export of models, training and serving with conformance tests
(`presk export`). A corpus of real models (`corpus/nlp/`, the ladder of an
NLP course on the French civil code) and a coverage table of the MLPerf
architectures come with it. Every optimization it adds is *exact*: the
training losses are identical to every printed digit, and one step of a
course-scale GPT went from 289 s to 39.8 s without changing a number. See `CHANGELOG.md` and the user manual (`make doc`); its
tutorial chapter (programs in `tutorial/`, commands in `make tutorial`) is the
place to start.

```sh
make build                                   # cargo build --workspace (first build compiles GMP/MPFR, a few minutes)
make test                                    # cargo test --workspace
make check-examples                          # presk check on the five example programs
make run-examples                            # the reference presk run commands
make train-examples                          # presk train on a regression and the tiny MiniGPT
make serve-examples                          # presk serve on the tiny MiniGPT's checkpoint: greedy tokens identical on every backend
make opt-examples                            # the e-graph optimizer: report, and the same outputs with and without --opt
make calibrate                               # time the native kernels of the tiny MiniGPT, fit the roofline profile (out/profile.json)
make experiment                              # exact versus almost: float32/float64 training against the exact reference
make venv && make conformance                # PyTorch/ONNX Runtime conformance of the exports
make doc                                     # the user manual (rustdoc)
```

## The `presk` command

```sh
presk check FILE.psk [--const NAME=N]... [--summary] [--json]
presk run FILE.psk --entry ENTRY [--dim NAME=N]... [--const NAME=N]... [--seed S] [--bits 256] [--digits 12] [--json] [--verbose] [--backend interp|kernel|native] [--realize f64|f32] [--opt [--profile FILE.json]]
presk train FILE.psk [--const NAME=N]... [--data FILE.psk.d] [--steps N] [--epochs N] [--seed S] [--bits 256] [--log-every N] [--ckpt DIR] [--json] [--backend ...] [--realize ...] [--opt]
presk export FILE.psk --target pytorch --mode readable|graph -o DIR [--shapes=check] [--stack=loop|unroll] [--train] [--serve] [--conformance-entry ENTRY|adamw.step|train:K ...]
presk serve FILE.psk --prompt "t1 t2 ..." [--weights FILE.psk.d] [--tokens N] [--temperature x] [--entry FN] [--dim NAME=N]... [--backend interp|kernel|native] [--realize f64|f32] [--json] [--opt]
presk bench FILE.psk --entry ENTRY [--dim NAME=N]... [--const NAME=N]... [--bits 64] [--repeat 5] [--opt [--profile FILE.json]] [--calibrate FILE.json [--realize f64|f32]]
presk data pack|lm-windows|info|export ...
presk dump FILE.psk --stage tokens|ast|elab|core|core-after-ad|core-opt|kernel [--unroll-stack] [--const NAME=N]...
```

Examples:

```sh
presk check tests/examples/minigpt/minigpt.psk --summary
presk run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2
presk train tests/examples/minigpt/train_minigpt.psk \
     --const V=64 --const D=16 --const H=2 --const L=2 --steps 20 --bits 64 --ckpt out/ckpt
presk export tests/examples/minigpt/train_minigpt.psk --target pytorch --mode readable --train \
     --conformance-entry train:20 --const V=64 --const D=16 --const H=2 --const L=2 -o out/train_pt
presk export tests/examples/minigpt/minigpt.psk --target pytorch --mode graph \
     --conformance-entry MiniGPT.apply --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8 -o out/graph_pt
```

Exit codes: 0 success, 1 diagnostics with errors or a run-time error, 2 usage
or internal error.

## Requirements

A Rust toolchain (edition 2024, rustc ≥ 1.85), `make`, and for the
interpreter `m4` and a C compiler (the `rug` crate builds GMP and MPFR from
source once; Homebrew's `mpfr` can be used instead through the
`use-system-libs` feature of `gmp-mpfr-sys`). PyTorch, ONNX and ONNX Runtime
are not needed to build or test the compiler; `make venv` installs them in
`.venv` to run the generated conformance tests and training scripts.

Layout, commands and conventions for contributors: `CONTRIBUTING.md`.
