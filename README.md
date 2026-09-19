# Presk compiler

Compiler for Presk, a domain-specific language for describing, training and
running neural networks with exact real semantics. The language is specified
in the `presk-specs` repository; its `examples/*.psk` are the reference
programs (copied under `tests/examples/`).

**Version 0.1.0**: the front-end (`presk check`), an arbitrary-precision
reference interpreter with automatic differentiation (`presk run`) and a
readable PyTorch export (`presk export`). See `CHANGELOG.md` and the user
manual (`make doc`); its tutorial chapter (programs in `tutorial/`,
commands in `make tutorial`) is the place to start.

```sh
make build                                   # cargo build --workspace (first build compiles GMP/MPFR, a few minutes)
make test                                    # cargo test --workspace
make check-examples                          # presk check on the five example programs
make run-examples                            # the reference presk run commands
make doc                                     # the user manual (rustdoc)
```

## The `presk` command

```sh
presk check FILE.psk [--const NAME=N]... [--summary] [--json]
presk run FILE.psk --entry ENTRY [--dim NAME=N]... [--const NAME=N]... [--seed S] [--bits 256] [--digits 12] [--json] [--verbose]
presk export FILE.psk --target pytorch --mode readable -o DIR [--shapes=check] [--stack=loop|unroll] [--conformance-entry ENTRY ...]
presk dump FILE.psk --stage tokens|ast|elab|core|core-after-ad
```

Examples:

```sh
presk check tests/examples/minigpt/minigpt.psk --summary
presk run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2
presk run tests/examples/minigpt/minigpt.psk --entry MiniGPT.apply \
     --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8
presk export tests/examples/minigpt/minigpt.psk --target pytorch --mode readable --shapes=check \
     --conformance-entry MiniGPT.apply --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8 \
     -o out/minigpt_pt
```

Exit codes: 0 success, 1 diagnostics with errors, 2 usage or internal error,
3 not executable in this version (`presk run` on a `training run`).

## Requirements

A Rust toolchain (edition 2024, rustc ≥ 1.85), `make`, and for the
interpreter `m4` and a C compiler (the `rug` crate builds GMP and MPFR from
source once; Homebrew's `mpfr` can be used instead through the
`use-system-libs` feature of `gmp-mpfr-sys`). PyTorch is not needed to build
or test the compiler; it is needed to run the generated conformance tests.

Layout, commands and conventions for contributors: `CONTRIBUTING.md`.
