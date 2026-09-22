# Presk compiler

Compiler for Presk, a domain-specific language for describing, training and
running neural networks with exact real semantics. The language is specified
in the `presk-specs` repository; its `examples/*.psk` are the reference
programs (copied under `tests/examples/`).

**Version 0.6.0**: the front-end (`presk check`); an arbitrary-precision
reference interpreter with automatic differentiation (`presk run`), the oracle
every other executor is measured against; a native backend — PSK-Kernel v0
compiled by Cranelift, with fusion groups, threads, blocked contractions and
vector lanes, all exact — with training (`presk train`) from `.psk.d` data
files (`presk data`) and serving (`presk serve`) on any backend; an e-graph
optimizer with rules exact over ℝ and a calibrated cost model (`--opt`,
`presk bench --calibrate`); precision as a resource — a realization assigns a
machine type per site or per function (`--realize f64|f32|bf16|@FILE`),
weights may be stored as `int8` or `int4` with per-channel scales, and a
solver turns a declared error budget into a realization and verifies it by
measurement against the exact denotation, on an entry or on a training step
(`presk calibrate`, `presk solve`, `presk quantize`); and a PyTorch export of
models, training and serving with conformance tests (`presk export`). A
corpus of real models (`corpus/nlp/`, the ladder of an NLP course on the
French civil code) and transcriptions of the MLPerf architectures
(`coverage/`, BERT-large running at MLPerf's constants) come with it. Every
optimization is *exact* — one step of a course-scale GPT went from 289 s to
39.8 s in 0.4.0 without changing a number — and every narrowing is measured: a
training step solved under a 10⁻³ budget on its update runs 1.99× faster than
all-`f64`, inside the budget. See `CHANGELOG.md` and the user manual (`make
doc`); its tutorial chapter (programs in `tutorial/`, commands in `make
tutorial`) is the place to start.

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
make tutorial                                # every command of the tutorial chapter (needs the venv)
make corpus && make corpus-gpt               # the NLP-course ladder on the Code civil, and the course GPT
make coverage                                # the MLPerf transcriptions
make gpt-trained && make precision           # a trained corpus GPT; envelope, bound, sensitivity, solve on it
make curve                                   # the ε-to-quality curve (about two hours)
make quant-table step-table j-step foil      # the experiments of 0.6.0 (manual chapters precision and experiments)
make bert                                    # BERT-large inference at MLPerf's constants (about five hours; L=2 for minutes)
make doc                                     # the user manual (rustdoc)
```

## The `presk` command

```sh
presk check FILE.psk [--const NAME=N]... [--summary] [--json]
presk check FILE.psk --precision ENV.json [--realize R] [--model cond|normwise|k|rss|all] [--measure] [--sites N]
presk run FILE.psk --entry ENTRY [--dim NAME=N]... [--const NAME=N]... [--seed S] [--bits 256] [--digits 12] [--json] [--verbose]
    [--backend interp|kernel|native] [--realize f64|f32|bf16|@R.json] [--threads N] [--opt [--profile FILE.json]]
    [--weights ckpt/final.psk.d] [--data FILE.psk.d [--samples N] [--from I]] [--unroll-stack]
presk train FILE.psk [--const NAME=N]... [--data FILE.psk.d] [--steps N] [--epochs N] [--seed S] [--bits 256] [--log-every N]
    [--ckpt DIR] [--json] [--backend ...] [--realize ...] [--threads N] [--opt [--profile FILE.json]]
presk serve FILE.psk --prompt "t1 t2 ..." [--weights FILE.psk.d] [--tokens N] [--temperature x] [--seed S] [--entry FN]
    [--dim NAME=N]... [--const NAME=N]... [--bits 256] [--backend ...] [--realize ...] [--json] [--opt [--profile FILE.json]]
presk export FILE.psk --target pytorch --mode readable|graph -o DIR [--shapes=check] [--stack=loop|unroll] [--train] [--serve]
    [--conformance-entry ENTRY|adamw.step|train:K ...] [--dim NAME=N]... [--const NAME=N]... [--seed S] [--opt [--profile FILE.json]]
presk bench FILE.psk --entry ENTRY [--dim NAME=N]... [--const NAME=N]... [--bits 64] [--repeat 5] [--seed S] [--threads N]
    [--realize R] [--native-only] [--opt [--profile FILE.json]] [--calibrate FILE.json]
presk calibrate --precision FILE.psk --entry ENTRY [--dim NAME=N]... [--const NAME=N]... [--data FILE.psk.d] [--samples N] [--from K]
    [--weights ckpt/final.psk.d] [--bits 256] [--seed S] [--out ENV.json] [--compare OTHER.json]
presk calibrate --step FILE.psk --weights ckpt/final.psk.d [--warmup 10] [--steps N] [--data FILE.psk.d] [--samples 2] [--from 0]
    [--realize f64|f32|bf16|recipe|@R.json]... [--profile FILE.json] [--reference exact|f64] [--bits 256] [--out ENV.json]
presk solve FILE.psk --precision ENV.json [--budget 1e-3] [--weights ckpt/final.psk.d] [--profile FILE.json]
    [--quant [--scheme absmax|pow2]] [--no-verify] [--bits 128] [--seed S] [--out R.json] [--sites N]
presk solve --step FILE.psk --precision ENV.json [--budget 1e-3] [--profile FILE.json] [--no-verify] [--bits 128] [--out R.json]
presk quantize FILE.psk --entry ENTRY [--realize @R.json] [--weights ckpt/final.psk.d] [--data FILE.psk.d [--samples N] [--from I]]
    [--dim NAME=N]... [--const NAME=N]... [--backend interp|kernel|native] [--bits 128] [--seed S] [--list]
presk data pack|chars|encode|lm-windows|info|diff|export ...
presk dump FILE.psk --stage tokens|ast|elab|core|core-after-ad|core-opt|sched|kernel|sensitivity [--entry ENTRY] [--dim NAME=N]...
    [--const NAME=N]... [--realize R] [--profile FILE.json] [--weights ckpt/final.psk.d] [--sites N] [--unroll-stack]
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
presk calibrate --step tutorial/07_train.psk --weights out/tutorial_ckpt/final.psk.d --warmup 20 --samples 8 --out out/tut_step.json
presk solve --step tutorial/07_train.psk --precision out/tut_step.json --budget 1e-3 --out out/tut_step_solved.json
presk train tutorial/07_train.psk --backend native --realize @out/tut_step_solved.json
```

Exit codes: 0 success, 1 diagnostics with errors or a run-time error, 2 usage
error or not implemented.

## Requirements

A Rust toolchain (edition 2024, rustc ≥ 1.85), `make`, and for the
interpreter `m4` and a C compiler (the `rug` crate builds GMP and MPFR from
source once; Homebrew's `mpfr` can be used instead through the
`use-system-libs` feature of `gmp-mpfr-sys`). PyTorch, ONNX and ONNX Runtime
are not needed to build or test the compiler; `make venv` installs them in
`.venv` with `uv` to run the generated conformance tests and training scripts
(without `uv`: `python3 -m venv .venv`, then `pip install` the packages the
target names).

Layout, commands and conventions for contributors: `CONTRIBUTING.md`.
