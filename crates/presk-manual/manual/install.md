# Building and running

## Requirements

- A Rust toolchain, `rustc` ≥ 1.85 (edition 2024). Developed with 1.92.
- `make` (the `Makefile` is a thin layer over `cargo`; every recipe can be
  run as the `cargo` command it wraps).
- `m4` and a C compiler: the interpreter uses MPFR through the `rug` crate,
  which `gmp-mpfr-sys` builds from bundled sources the first time (a few
  minutes, cached under `target/`). With Homebrew's `mpfr` installed, the
  `use-system-libs` feature of `gmp-mpfr-sys` links it instead.
- PyTorch is *not* needed to build or test the compiler; only to run the
  conformance tests that `presk export` generates.

## Build

```sh
git clone https://github.com/presk-lang/presk.git
cd presk
make build                       # cargo build --workspace
```

The binary is `target/debug/presk`. During development it is simplest to run
it through cargo:

```sh
cargo run -q -p presk -- check tests/examples/mlp/mlp.psk
cargo run -q -p presk -- run tests/examples/mlp/mlp.psk --entry mse --dim D=4 --dim B=2
cargo run -q -p presk -- dump --stage core tests/examples/mlp/mlp.psk
```

Add `--release` for an optimized build (`target/release/presk`).

## Tests

```sh
make test                        # cargo test --workspace
cargo insta review               # accept or reject changed snapshots
make check-examples              # presk check on the five example programs
make run-examples                # the reference presk run commands
make export-examples             # export the example files to out/ (mlp, layernorm, minigpt, train_minigpt)
make venv                        # once: .venv with PyTorch, NumPy and pytest (needs uv)
make conformance                 # export, then run every generated test_conformance.py in .venv
make tutorial                    # every command of the tutorial chapter (ends with two conformance tests)
make train-examples              # presk train on the regression and the tiny MiniGPT
make experiment                  # exact versus almost (manual chapter `experiments`)
make bench                       # every backend on the tiny MiniGPT (manual chapter `native`)
```

The conformance tests are the only thing that needs PyTorch; `.venv` is
ignored by git and created with [uv](https://docs.astral.sh/uv/).

Snapshot tests (crate `insta`) record the tokens, ASTs and rendered
diagnostics of reference inputs under `crates/*/tests/snapshots/`; a change
in output shows up as a failing test with a diff, to be reviewed with
`cargo insta review`.

## Documentation

```sh
make doc                         # cargo doc -p presk-manual --open
cargo doc --workspace --no-deps  # API documentation of the crates as well
```

## Layout of the repository

```text
crates/presk-syntax          source map, diagnostics, lexer, AST, parser
crates/presk-types           dimensions, Presburger solver, elaboration, typing   (step 2–3)
crates/presk-core            PSK-Core IR, lowering, regularity, effects, AD       (step 4–6)
crates/presk-interp          reference interpreter                                (step 7)
crates/presk-export-pytorch  readable PyTorch export                              (step 7b)
crates/presk-data            .psk.d data files and the training pipeline           (0.2.0 step 1)
crates/presk-rt              the runtime seam: values, the Backend trait, InterpBackend (0.3.0 step 1)
crates/presk-kernel          PSK-Kernel v0: kernels, their interpreter, the host program   (0.3.0 step 2)
crates/presk-cranelift       the Cranelift JIT over the kernels, --backend native          (0.3.0 step 3)
crates/presk-train           the training loop over a Backend, presk train           (0.2.0 step 5)
crates/presk-serve           the generation loop over a Backend, presk serve         (0.3.0 step 5)
crates/presk-opt             e-graph rewriting of PSK-Core, cost models, profiles    (0.3.0 steps 6–7)
crates/presk-sched           PSK-Sched v0: fusion groups over PSK-Core bodies       (0.4.0 step 4)
crates/presk-precision       realizations, error bounds, sensitivity, the solver    (0.5.0 steps 2–6)
crates/presk-cli             the `presk` binary
crates/presk-manual          this manual
tests/examples               the example programs of presk-specs (do not edit here)
tutorial                     the programs of the tutorial chapter
tests/diag                   one erroneous program per diagnostic, with snapshots
tests/data                   a token stream for the .psk.d tests
tests/train                  a linear regression trained by presk train
experiments                  the script of the `experiments` chapter
```

The example programs are refreshed from the specification repository with
`make sync-examples` (variable `PRESK_SPECS_DIR`, default `../presk-specs`).
