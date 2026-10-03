# Building and running

## Requirements

- A Rust toolchain, `rustc` ≥ 1.85 (edition 2024). Developed with 1.92 and 1.98.
- `make` (the `Makefile` is a thin layer over `cargo`; every recipe can be
  run as the `cargo` command it wraps).
- `m4` and a C compiler: the interpreter uses MPFR through the `rug` crate,
  which `gmp-mpfr-sys` builds from bundled sources the first time (a few
  minutes, cached under `target/`). With Homebrew's `mpfr` installed, the
  `use-system-libs` feature of `gmp-mpfr-sys` links it instead. Since 0.9.0
  `make test` also compiles the C export's kernels and programs with it
  (`cc`, or the compiler `CC` names), with OpenMP when it has it (GCC's
  `-fopenmp`), and so does step 18 of `make tutorial`. An exported program
  needs only a C11 compiler (`export_c` chapter).
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
make tutorial                    # every command of the tutorial chapter, steps 0 to 18 (needs .venv and a C compiler)
make train-examples              # presk train on the regression and the tiny MiniGPT
make experiment                  # exact versus almost (manual chapter `experiments`)
make bench                       # every backend on the tiny MiniGPT (manual chapter `native`)
make precision curve quant-table step-table j-step step-train foil bert
                                 # the experiments of 0.5.0 and 0.6.0 (chapters `precision`, `experiments`; minutes to hours)
make plan-check                  # a plan stating the default layout gives the same kernels (0.7.0)
make calibrate-ra4 ra4-small     # the layout cost model's calibration; the RA4 sweep's daily subset (a minute)
make ra4 ra4-check ra4-20 ra4-day2
                                 # the RA4 measurement of 0.7.0 (chapter `experiments` §7; a night)
make ra4-dev ra4-dev-frozen ra4-ablation
                                 # 0.8.0's development set: 0.7.0's days priced again, nothing measured (minutes)
make calibrate-ra4-small ra4-f2-dry ra4-f2 ra4-f2-check ra4-f2-day2
make ra4-f1-dry ra4-f1 ra4-f1-check ra4-f1-day2
                                 # 0.8.0's fresh sweeps F2 and F1 (chapter `experiments` §8; hours each)
make c-conformance               # the C export on the test set, the course GPT and the trained checkpoints, Clang and x86-64, Polly (0.9.0; minutes, installs .venv-zig and .llvm)
make c-speed                     # statement C4: the exported course GPT against the native backend (a quiet machine)
make pytorch-baseline            # the course GPT's forward pass and training step: native, C and PyTorch (D87; 0.10.0's T2 and T4; a quiet machine, about twenty minutes, needs .venv)
make ra4-t3-dry ra4-t3 ra4-t3-check ra4-t3-day2
                                 # 0.10.0's T3: the frozen model on the derivatives of `dot` (chapter `experiments` §8; hours each)
make host-profile                # the native host by phase on the course GPT (0.10.0 step 5; a quiet machine)
make delta-p                     # the reference's own gap at 128 to 1 024 bits on the course GPT (D98; about half an hour, 9 GB)
make compile-times               # the time presk check, calibrate, solve and the exact reference take per model (D101; a quiet machine, hours)
```

The conformance tests are the only thing that needs PyTorch; `.venv` is
ignored by git and created with [uv](https://docs.astral.sh/uv/). Without
`uv`, `python3 -m venv .venv` followed by `.venv/bin/pip install` of the
packages the `venv` target names does the same.

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
crates/presk-export-c        the C export: the kernel printer, the host, the runtime, the math functions, the SCoP mode (0.9.0)
crates/presk-data            .psk.d data files and the training pipeline           (0.2.0 step 1)
crates/presk-rt              the runtime seam: values, the Backend trait, InterpBackend, Layout and Plan (0.3.0 step 1, 0.7.0)
crates/presk-kernel          PSK-Kernel v0: kernels, their interpreter, the host program, the contraction builder, the layout cost model, the host's recorder   (0.3.0 step 2, 0.7.0, 0.8.0, 0.9.0)
crates/presk-cranelift       the Cranelift JIT over the kernels, --backend native          (0.3.0 step 3)
crates/presk-train           the training loop over a Backend, presk train           (0.2.0 step 5)
crates/presk-serve           the generation loop over a Backend, presk serve         (0.3.0 step 5)
crates/presk-opt             e-graph rewriting of PSK-Core, cost models, profiles, the layout model's fit    (0.3.0 steps 6–7, 0.7.0, 0.8.0)
crates/presk-realize         PSK-Realize v0: fusion groups over PSK-Core bodies     (0.4.0 step 4)
crates/presk-precision       envelope, error bounds, sensitivity, the solver         (0.5.0 steps 2–6, 0.6.0)
crates/presk-cli             the `presk` binary
crates/presk-manual          this manual
tests/examples               the example programs of presk-specs (do not edit here)
tutorial                     the programs of the tutorial chapter
tests/diag                   one erroneous program per diagnostic, with snapshots
tests/data                   a token stream for the .psk.d tests
tests/train                  regressions trained by presk train
tests/quant                  a program of odd widths for the int4 kernels
tests/plan                   realization plans, and the entries make plan-check compares
tests/export_c               programs and a plan the C export's tests need beyond the test set
corpus/nlp                   the corpus of an NLP course (manual chapter `corpus`)
coverage                     the MLPerf transcriptions (manual chapter `coverage`)
experiments                  the scripts behind make experiment, curve, quant-table, step-table, j-step, step-train, foil, bert, ra4*, c-speed, pytorch-baseline
```

The example programs are refreshed from the specification repository with
`make sync-examples` (variable `PRESK_SPECS_DIR`, default `../presk-specs`).
