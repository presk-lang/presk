# The `presk` command line

```text
presk <COMMAND>

Commands:
  check      Type-check a program: shapes, regularity classes, effects
  run        Evaluate an entry point on the reference interpreter or natively
  export     Export a program to another framework
  dump       Dump an intermediate representation (development aid)
  train      Run the `training run` of a file, on the reference interpreter or natively
  serve      Generate tokens with the file's `inference serve`: its model, weights and generation function
  plan       Choose a realization plan statically: every nest's layout space costed by the profile's layout model
  autotune   Measure every point of every nest's layout space on the entry's own buffers, asserting each computes point 0's bits
  rank       Rank accuracy of the layout cost model on two days of measurement, and the statements of plan 0.7.0
  bench      Time an entry on every backend (the reference at --bits, the kernel interpreter, native f64, f32 and bf16)
  calibrate  Measure the magnitude envelope of an entry, or one training step under each realization (--step)
  solve      Solve for a realization under a declared error budget
  quantize   Quantise parameters as a realization's `quant` map says, and measure the gap in two parts: the weights' rounding and the arithmetic
  data       Produce and inspect `.psk.d` data files
  help       Print this message or the help of the given subcommand(s)
```

`presk --help` and `presk <command> --help` print the full option lists;
`presk --version` prints the version.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success (warnings may have been printed) |
| 1 | diagnostics with at least one error were reported |
| 2 | not implemented in this version, or usage error (unreadable file, bad flag) |
| 3 | reserved (0.1.0 used it for `presk run` on a `training run`; that case is now a usage error pointing to `presk train`) |

Diagnostics go to standard error; dumps and results go to standard output.
Colors are used when standard error is a terminal and `NO_COLOR` is unset.

## `presk check FILE.psk`

Reads one source file and reports its diagnostics.

**Implemented:** the whole static check of 0.1.0:

1. lexical and syntactic analysis;
2. *elaboration*: `import`s loaded relative to the importing file, `const`s
   evaluated exactly, `dim` bounds recorded, type aliases resolved, modules
   and their instances built with the shapes of their parameters, function
   signatures converted to shape-ready types, `static assert`s discharged;
3. *shape typing* of every function body: named axes, `...` prefixes,
   broadcasting, `idx` tensors, `params` trees, `ein`, the four call argument
   forms, module member calls, `grad`, the builtin table;
4. the configuration blocks (`optimizer`, `training run`, `inference serve`)
   against their schemas;
5. lowering to PSK-Core (verified), then the *regularity classes* of every
   function with respect to each parameter and the legality of every `grad`,
   and the *effects* each function uses against its `uses` clause.

Exit code 0 when the file has no error (warnings may be printed), 1
otherwise. The five example programs pass with only `W0001` warnings
(`make check-examples`).

| Flag | Meaning | Status |
|---|---|---|
| `--const NAME=N` | override a `const` of the main file before evaluation (repeatable); `static assert`s are re-checked with the new value | implemented |
| `--summary` | print, for every function of the file and its imports, the inferred type of its body and of each `let` binding, then (after `-- regularity and effects --`) the class of its result with respect to each parameter and the effects it uses, then (after `-- training --`, when the file has a `training run`) the resolved run: model, objective, batch and sample types, pipeline, epochs, checkpoint, and the dimensions bound by the pipeline and the data file | implemented |
| `--json` | machine-readable output on stdout: `{"file", "ok", "diagnostics": [{code, severity, message, labels: [{file, line, col, start, end, text}], notes, help}]}`; nothing is rendered on stderr; `--summary` is ignored | implemented |

Example of `--summary` on `tests/examples/minigpt/minigpt.psk` (excerpt):

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

`presk check --const H=7 tests/examples/minigpt/minigpt.psk` reports that
`HD = D / H` is not a natural number and that `static assert D % H == 0`
fails.

## `presk dump FILE.psk --stage STAGE`

Prints an intermediate representation of the file; a development aid whose
output format is not stable.

| Stage | Output | Status |
|---|---|---|
| `tokens` | one token per line, `kind @ start..end` (byte offsets) | implemented |
| `ast` | the abstract syntax tree (Rust `Debug` format, spans as `start..end`) | implemented |
| `elab` | the elaborated program: per file, imports, constants with their values, dimensions with bounds, type aliases, modules with the shapes of their parameters and sub-modules, function signatures, discharged assertions; then the table of module instances with every parameter's shape (stacked axes included) | implemented |
| `core` | the PSK-Core intermediate representation after lowering (every function in A-normal form with its dimensions, parameters and types; `init` functions; lifted lambdas), verified before printing | implemented |
| `core-after-ad` | PSK-Core after automatic differentiation: every `grad` rewritten, with the generated `.vjp` and `.grad` functions; verified before printing | implemented |
| `kernel` | the PSK-Kernel v0 kernels launched by an entry, specialized on its shapes: needs `--entry` and `--dim`/`--const` as `presk run`, plus `--realize` or `--plan` (`native` chapter). The kernel interpreter lowers without lanes unless a plan states them, so a dump without a plan, or with one that states none, prints the scalar kernels | implemented (0.3.0 step 2; `--plan` 0.7.0 step 1) |
| `sensitivity` | `‖∂L/∂v‖₁` at every site of a scalar entry and summed per module, from one reverse sweep: needs `--entry`, `--dim`/`--const`, and `--weights` to be worth reading, since a zero-initialized head makes every upstream gradient zero; `--sites N` lists more (`precision` chapter) | implemented (0.5.0 step 5) |
| `realize` (`sched`) | the fusion groups of every function (`realize` chapter): one line per group with its steps, its inputs and the launches it removes; with `--plan @FILE`, then the plan: its realization, and each nest it gives a layout, with the nest's primitive and the decisions stated, after checking that every nest named exists (`E7002`) | implemented (0.4.0 step 4; `--plan` 0.7.0 step 1) |
| `core-opt` | PSK-Core after AD and the e-graph optimizer (`optimization` chapter); the report (functions rewritten, `let`s and cost before and after, stop reason) goes to standard error; `--profile FILE.json` makes the roofline of that profile the extraction cost, `--dim` binds the dimensions it evaluates | implemented (0.3.0 steps 6–7) |

`--unroll-stack` lowers `stack{L}` as `L` calls, as on `presk run`; `kernel`
and `sensitivity` take `--entry`, `--dim`, `--const` and `--realize` as `presk
run` does, `kernel` and `realize` also `--plan`, `sensitivity` also
`--weights` and `--sites`, and `core-opt` takes `--profile`.

Diagnostics (if any) are reported on standard error; the exit code is 1 when
there are errors, 0 otherwise.

## `presk run FILE.psk --entry ENTRY …`

Evaluates one function, on the reference interpreter unless `--backend`
says otherwise: the whole pipeline (`check`, lowering, automatic
differentiation) runs first, then the entry point is called on synthesized
inputs and its result printed.

```sh
presk run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2
presk run tests/examples/layernorm/layernorm.psk --entry 'LayerNorm{8}.apply' --dim batch=3
presk run tests/examples/minigpt/minigpt.psk --entry MiniGPT.apply \
     --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8
presk run tests/examples/minigpt/train_minigpt.psk --entry loss \
     --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8
```

| Flag | Meaning |
|---|---|
| `--entry ENTRY` | a top-level function (`mse`), a module function (`MiniGPT.apply`), a module function with the module's dimension arguments (`LayerNorm{8}.apply`), or an optimizer function of a file with a `training run` (`adamw.step`, `adamw.lr`, `adamw.init`; their inputs are synthesized with a non-negative second moment, `t = 3`, `total = 10`). Required: without it, a file with a `training run` is directed to `presk train` (exit 2), and any other file lists the candidate entries. |
| `--dim NAME=N` | binds a dimension of the entry (its type parameters, implicit dimensions, the file's `dim`s). A name that is not a dimension of the entry gives, in order, the axes of the open prefix `...` of its parameters: `--dim batch=3` makes `x : ℝ[..., model: D]` an `ℝ[batch: 3, model: 8]`. Every dimension must be bound, and the file's bounds (`dim T <= 1024`) are enforced (`E7001`). |
| `--const NAME=N` | overrides a `const` of the file *or of its imports* (`minigpt.psk`'s `V`, `D`, `H`, `L` from `train_minigpt.psk`); `static assert`s are re-checked. This is how the MiniGPT examples run at a size the interpreter can afford. |
| `--seed S` | seed of every random choice (default 0): synthesized inputs, initializers, `sample`. Same seed, same output, on every machine. |
| `--bits B` | MPFR precision of every intermediate, in bits (default 256). |
| `--digits D` | significant digits when printing (default 12). |
| `--json` | machine-readable output: entry, seed, bits, dimensions, inputs and result, reals as decimal strings at `--digits`. |
| `--verbose` | also print the synthesized inputs. |
| `--unroll-stack` | lower `stack{L}` as `L` calls (0.1.0's form) instead of a `fold`; a debugging aid, also on `presk dump`. |
| `--backend interp\|kernel\|native` | the executor behind the runtime seam (0.3.0): the reference interpreter, PSK-Kernel v0 interpreted in the `--realize` machine type, or the same kernels compiled by Cranelift (`native` chapter); `presk train` takes the same flag. |
| `--realize f64\|f32\|bf16\|@FILE` | the realization (0.3.0, an object since 0.5.0 step 2): a uniform machine type, or `@FILE` for a solved one carrying per-site storage and accumulator types and a declared reduction order (`native` chapter), since 0.6.0 step 5a a `quant` map of parameters stored as `int8`/`int4` (`precision` chapter), and since 0.6.0 step 7b a `functions` map giving every site of a function one type (a site named in the per-site map overrides it). The interpreter ignores the machine types, computing at `--bits`, but **not** the `quant` map: every backend runs on the dequantised parameters. |
| `--plan @FILE` | a realization plan (`realize` chapter, "The realization plan"; 0.7.0 step 1): a realization and a layout per nest. `--realize R` is the plan with realization `R` and the default layout, so the two flags are exclusive. The plan is checked against the program before anything runs, and each nest's layout against its shapes and its kernel when it is launched; an ill-formed layout is refused (`E7002`), never ignored. Every decision of the reference is lowered since 0.7.0 step 3. The reference interpreter ignores the layout. Also on `presk train`, `presk bench` and `presk dump`. |
| `--opt` | rewrite the program with the e-graph optimizer first (`optimization` chapter); one summary line on standard error, the whole report with `--verbose`. Also on `presk train`, `presk serve`, `presk bench` and `presk export` (its conformance data). |
| `--profile FILE.json` | with `--opt`: extract by the calibrated roofline of this machine profile (`presk bench --calibrate`) instead of operation counts; the run's `--dim`s are the dimensions it evaluates. Also on `train`, `serve`, `bench`, `export`, `dump --stage core-opt`. |
| `--threads N` | workers a parallel kernel may use on the native backend (`native` chapter); the default is the machine's cores, `1` keeps every kernel on one thread. Also on `presk train` and `presk bench`. |
| `--data FILE.psk.d` | take the entry's **last** argument from a data file instead of synthesizing it, and bind the `batch` axis of that argument to the number of samples read. An objective is then evaluated on its own dataset, and a plain function on a corpus; `--samples N` reads only the first `N`, from sample `--from I` (default 0). The other arguments are synthesized from `--seed` as usual. |
| `--weights ckpt/final.psk.d` | take the entry's **first** argument, its parameters, from a checkpoint (0.6.0 step 7): with `--data`, a trained model's loss on held-out text is one command. The checkpoint must be of the dimensions the entry is run at (`E9005` otherwise). |

**Inputs** are synthesized from the parameter types: real tensors from a
standard normal, index tensors uniformly in `[0, N)`, `params M` by running
the module's initializer (`init =` clauses: `zeros`, `ones`, `xavier_uniform`,
`normal(std = …)`), hyperparameters from their source default (`eps : f64 =
1e-5`) or `1`. Each parameter draws from its own key derived from `--seed`.

**Output**: the result with its type, e.g. `ℝ[batch: 2, model: 4] = [batch: 2,
model: 4] [[…], […]]`; parameter trees as nested `{ name = … }` records; index
tensors as integers.

**Exit codes**: 0 success; 1 a diagnostic (including a runtime error `E7001`,
labelled with the source of the failing operation); 2 usage or internal
error, including a `training run` file without `--entry`.

**Randomness**: `sample` draws from a stream seeded by `--seed` (the `Rand`
effect has no explicit key argument in 0.1.0; see `docs/spec-questions.md`).

## `presk export FILE.psk --target pytorch --mode readable -o DIR`

Writes a readable PyTorch program, one `.py` per Presk file (the file and its
imports), whose structure mirrors the source: one `module` → one `nn.Module`
with the parameters and sub-modules of the source under the same names,
`fn apply` → `forward`, other functions → methods or plain functions, `let`
→ assignment with the Presk type as a trailing comment. Details in the
`language` chapter ("PyTorch export").

```sh
presk export --target pytorch --mode readable tests/examples/mlp/mlp.psk -o out/mlp_pt
presk export --target pytorch --mode readable --shapes=check \
     --conformance-entry MiniGPT.apply --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8 \
     tests/examples/minigpt/minigpt.psk -o out/minigpt_pt
python3 -m py_compile out/minigpt_pt/*.py    # syntax; `pytest out/minigpt_pt` where PyTorch is installed
```

| Flag | Meaning |
|---|---|
| `--target pytorch` | the only target. |
| `--mode readable` | one `nn.Module` per module (default). |
| `--mode graph` | the specification's tool-level mode, in its light form: with `--conformance-entry Module.apply`, also writes `export.py`, a driver that builds the model with the conformance parameters and saves `model.pt2` (`torch.export`) and `model.onnx` (`torch.onnx.export`, dynamo), in float32; the generated test gains `test_onnx`, which runs `model.onnx` under ONNX Runtime against the interpreter (`rtol 1e-4`). The Presk-native StableHLO path waits for the precision solver. `--precision=solved`: not implemented. |
| `--train` | also write `pskd.py`, `train.py` and a copy of the data file: the file's `training run` as a PyTorch loop (`training` chapter); a keyed objective is called with `training=True`, so its `dropout` drops (D54). |
| `--serve` | also write `pskd.py` and `serve.py`: the file's `inference serve` as a PyTorch generation loop reading `final.pt` or a `.psk.d` checkpoint (`serve` chapter). |
| `-o DIR` | output directory (created); the generated files import each other by file stem, made a Python identifier (`01_linear.psk` → `_01_linear.py`). |
| `--shapes=check` | insert `assert x.shape == (…)` after every parameter and `let` with a closed shape. |
| `--stack=loop\|unroll` | `stack{L}(M)` as a `for` over an `nn.ModuleList` (default) or as `L` explicit calls. |
| `--conformance-entry ENTRY` | also run the reference interpreter on `ENTRY` (with `--dim`, `--const`, `--seed` as for `presk run`, at 256 bits) and write `conformance/inputs.json`, `conformance/expected.json` and `test_conformance.py`; `adamw.step` compares one optimizer step with `torch.optim.AdamW`; `train:K` runs `K` steps of the training run on the interpreter (implies `--train`) and generates a test that replays the same batches in PyTorch and compares the parameters (`training` chapter); the same `K` steps also run natively in `f64` and `f32`, written to `conformance/native_{f64,f32}.json` for the `experiments` chapter. |
| `--opt`, `--profile FILE.json` | rewrite the program with the e-graph optimizer before computing the conformance data, extracting by the profile when given; the exported code is the source's. |

**The conformance test** loads the interpreter's inputs and expected result,
builds the model or function (`MiniGPT(8)`, `MLP(4, 16)` — the dimension
values of the run), loads the parameter tree into the PyTorch parameters
(splitting stacked sub-modules over the `nn.ModuleList`), runs the forward
pass in `float64` and checks `torch.allclose(rtol=1e-9, atol=1e-12)`. It is
the "exact versus almost" experiment of the specification: the tolerance
measures the gap between the exact semantics and double precision. It runs
where PyTorch is installed: `make venv` creates `.venv` with PyTorch, and
`make conformance` exports the examples and runs their tests
(`mse_grad`, `LayerNorm{8}.apply`, `MiniGPT.apply`, the training `loss` and
`adamw.step` against `torch.optim.AdamW` with `clip_grad_norm_` and the
schedule, `train:20` and the ONNX graph of MiniGPT pass). PyTorch, ONNX and
ONNX Runtime are not dependencies of the compiler.

**Exit codes**: 0 written; 1 diagnostics (a construct with no PyTorch form is
`E8001`); 2 usage.

## `presk train FILE.psk …`

Executes the file's `training run` on the backend of `--backend`/`--realize`
(the reference interpreter by default): see the `training` chapter for the
loop, the flags (`--const`, `--data`, `--steps`, `--epochs`, `--seed`,
`--bits`, `--log-every`, `--ckpt`, `--json`, `--threads`, `--opt`,
`--profile`) and two runs. `--realize @R.json` takes a realization solved for
the training step (`presk solve --step`), whose `functions` map gives each
function of the step its type; a realization that quantises weights is
refused. `--plan @P.json` takes a realization plan instead, as `presk run`
does (0.7.0 step 1). Besides the log, standard error gets the kernel launches a step
(native backend) and the median step time with the first step excluded;
`--json` carries `ms` per step and `step_ms`.

```sh
presk train tests/train/mlp_regression.psk --steps 200 --ckpt out/reg_ckpt
presk train tests/examples/minigpt/train_minigpt.psk \
     --const V=64 --const D=16 --const H=2 --const L=2 --steps 20 --bits 64 --ckpt out/ckpt
```

**Exit codes**: 0; 1 a diagnostic or a run-time error; 2 no `training run`
or an unreadable data file.

## `presk serve FILE.psk --prompt "t₁ t₂ …" …`

Generates tokens with the file's `inference serve`: its model, its weights
(`--weights FILE.psk.d` overrides the block's) and its generation function
(`--entry FN` overrides the one found by signature), on the backend of
`--backend`/`--realize`; `--tokens N` (16), `--temperature x` (0 is greedy;
default the function's own), `--dim NAME=N` for a dimension of the parameters
not fixed by the prompt (`T` of a positional embedding), `--seed`, `--bits`,
`--json`, `--opt`, `--profile`. A realization with a `quant` map serves its
parameters as integers. The `serve` chapter has the loop and the rules.

```sh
presk serve tests/examples/minigpt/serve_minigpt.psk --const V=64 --const D=16 --const H=2 --const L=2 \
     --weights out/ckpt/final.psk.d --prompt "1 2 3 4 5 6 7 8" --tokens 8 --temperature 0 --backend native
```

**Exit codes**: 0; 1 a diagnostic or a run-time error; 2 no `inference
serve`, no weights file, or a malformed `--prompt`/`--dim`.

## `presk calibrate --precision FILE.psk --entry ENTRY …`

```sh
presk calibrate --precision FILE.psk --entry ENTRY [--data FILE.psk.d]
    [--samples N] [--from K] [--dim NAME=N] [--const NAME=N]
    [--weights ckpt/final.psk.d] [--bits 256] [--seed 0] [--out ENV.json]
    [--compare OTHER.json]
```

Measures the **magnitude envelope**: the largest `|x|` at each site, on the
*reference* interpreter, since the bound it feeds is a statement about the
denotation and must not be read off a rounded run. `--samples N` folds in a
batch of `N` and `--from K` skips the first `K`, so two disjoint ranges give
two independent measurements of the same program; `--compare` prints how far
two envelopes disagree. `--precision` is required and is the only mode — the
roofline calibration is `presk bench --calibrate`, a different measurement of
a different thing. See the `precision` chapter.

At every sum and contraction it also records what the terms add up to
without cancelling — `Σ|xᵢwᵢ|`, `√Σ(xᵢwᵢ)²` and each operand's normwise
factor — and prints the five worst condition numbers `Σ|xᵢwᵢ| / |Σxᵢwᵢ|`
(0.6.0 step 6). `--weights` reads the entry's first argument from a
checkpoint, which is refused if it is not exactly the layout's size (D48);
the envelope records the weights and the data slice, so that a gap measured
beside the bound, and a realization solved from it, are on the same inputs.

### `presk calibrate --step`: one training step

```sh
presk calibrate --step FILE.psk --weights ckpt/final.psk.d [--warmup 10] [--steps N]
    [--data FILE.psk.d] [--samples 2] [--from 0] [--const NAME=N]
    [--realize f64|f32|bf16|recipe|@R.json]… [--profile FILE.json]
    [--reference exact|f64] [--bits 256] [--seed 0] [--out ENV.json]
```

Measures one **optimizer step** of the file's `training run` instead of an
entry (0.6.0 step 7b, D52): the thing a budget in a training run binds. The
state is the checkpoint's `θ` followed by `--warmup` steps of the run's own
pipeline natively in `f64`, so that AdamW's second moment is not the zero of
initialisation; the measured step is number `warmup + 1` of `--steps` (the
run's horizon by default), on `--samples` samples of `--data` (the run's
file by default) from `--from`. The envelope is recorded on the **exact**
step — the objective's `vjp` and `adamw.step` on the reference interpreter —
and so covers the reverse sweep's and the optimizer's sites; `entry` is
`"step"` and a `step` record says how to rebuild the state.

It prints the step's size, the **sampling noise** `‖g_B − g_B′‖₂ / ‖g_B‖₂`
between the batch and the next `--samples` (the scale a rounding error is to
be compared with), and for each `--realize` a row: the **update gap**
`‖θ̂′ − θ′‖₂ / ‖θ′ − θ‖₂`, the **storage floor** (the same with `θ′` exact and
only rounded to the type the optimizer stores it in), the **gradient gap**
`‖ĝ − g‖₂ / ‖g‖₂` and the predicted cost of the step (seconds with
`--profile`, bytes otherwise). Each realization runs natively from the state
rounded to its default type and is compared with the exact step from the
same rounded state. `recipe` is mixed precision as practitioners use it:
every function but the optimizer in `bf16`, the optimizer and the master
weights in `f32`. `--reference f64` compares against a native `f64` step
instead of the exact one, for a model too large for the reference
interpreter, and writes no envelope. See the `precision` chapter, "A budget
on the training step".

**Exit codes**: 0; 1 a diagnostic, a run-time error or a checkpoint that
does not load (of other dimensions: `E9005`); 2 no `training run`, no
`--weights`, an unknown `--reference`, or `--reference f64` with `--out`;
`--steps`, `--realize`, `--profile` or `--reference` without `--step` is
refused with 2 rather than ignored.

## `presk check FILE.psk --precision ENVELOPE.json`

```sh
presk check --precision ENV.json FILE.psk [--realize f64|f32|bf16|@R.json]
    [--model cond|normwise|k|rss|all] [--measure] [--sites N]
```

Propagates the rounding error of the entry the envelope was measured on,
under a realization, and prints the bound on its output. `--model` chooses
the rule for sums and contractions: `cond`, the condition number (the
default); `normwise`, sound without assumption; `k`, 0.5.0's worst case; `rss`,
the root-sum-square, printed as an **estimate**; `all` prints the four.
`--measure` runs the realization natively on the envelope's own inputs and
weights and prints the measured distance to the exact reference, with each
row's ratio to it. `--sites N` lists the `N` widest sites with their condition
numbers. See the `precision` chapter, "Does the condition number rescue the
bound?".

## `presk solve FILE.psk --precision ENVELOPE.json --budget ε`

```sh
presk solve FILE.psk --precision ENV.json [--budget 1e-3] [--weights ckpt/final.psk.d]
    [--profile FILE.json] [--quant [--scheme absmax|pow2]] [--no-verify] [--bits 128]
    [--seed 0] [--out R.json] [--sites N]
```

Chooses a machine type per site under an error budget, minimising the
predicted time of the roofline fitted per machine type when given a
`--profile` (0.6.0 step 4; `precision` chapter, "Time, not traffic") and the
traffic otherwise, and writes a realization `--realize @R.json` can run. It **verifies by
measurement** against the exact reference rather than by the propagated
bound, which is vacuous on a real model; `--no-verify` gives the proposal
alone. `--weights` matters: a sensitivity ranking at initialization is
vacuous. The inputs are the envelope's: when it was measured on a data slice,
the sweep and every measurement run on that slice. `--quant` also offers every matrix parameter as `int8` or `int4`
(`--scheme absmax|pow2`), priced by the parameters' gradient and accepted by
measurement, and prints the three first-order readings per parameter with
`--sites`. A `--weights` file of other dimensions is an error, not a fallback
to the initial parameters. See the `precision` chapter.

### `presk solve --step`: a budget on the update

```sh
presk solve --step FILE.psk --precision ENV.json [--budget 1e-3] [--profile FILE.json]
    [--no-verify] [--bits 128] [--out R.json]
```

With an envelope written by `presk calibrate --step`, solves the whole
training step under a budget on its **update** (0.6.0 step 7b, D52): the
state is rebuilt from the envelope's `step` record, every function of the
step (`Linear.apply.vjp`, `adamw.step`, the forward functions the reverse
sweep recomputes) is run natively alone in `f32` and in `bf16`, everything
else in `f64`, and its update gap measured against the one exact step; the
functions are then narrowed greedily by predicted time saved per unit of
isolated gap while the gaps' root-sum-square fits the budget, the
combination is measured, and while it is outside the function with the
largest isolated gap is widened one step of the menu. It prints a row per
function (sites, cost in `f64`, isolated gaps, chosen type), the predicted
cost against all-`f64`, and the combination's measured update gap, storage
floor and gradient gap. `--out` writes a realization whose `functions` map
gives each narrowed function its type; `presk train --realize @R.json` trains
with it.

**Exit codes**: 0, whether or not the measured gap is inside the budget (the
report says which); 1 a run-time error; 2 an envelope of an entry rather than
of a step.

## `presk quantize FILE.psk --entry ENTRY --realize @R.json …`

```sh
presk quantize FILE.psk --entry ENTRY [--dim N=n] [--const N=n] [--realize @R.json]
    [--weights ckpt/final.psk.d] [--data FILE.psk.d [--samples N] [--from I]]
    [--backend interp|kernel|native] [--bits 128] [--seed 0] [--list]
```

Quantises the parameters the realization's `quant` map names and measures
what it costs, in two parts: the **representation** error (the exact
denotation on the dequantised weights against the exact denotation on the
true ones) and the **arithmetic** error (the `--backend` run of the
realization on the dequantised weights against the former). The entry's
first argument must be the parameter tree. A realization without a `quant`
map is measured too, its representation part zero, which is how the float
rows of `make quant-table` are made; `--list` lists the parameters, with
their paths and shapes, so a map can be written. `--data` takes the entry's
last argument from a data file as `presk run --data` does — the loss a model
was trained for, rather than its loss on random tokens. Per
parameter it prints the scheme, the axis, the number of channels, the largest
scale, the largest `|w − ŵ|` beside the largest `s/2` it must not exceed, and
the bytes stored. Without `--weights` it quantises the initial parameters,
which says little: a zero-initialized head makes the loss `ln V` whatever the
other weights are. See the `precision` chapter.

## `presk bench FILE.psk --entry ENTRY …`

Times one entry on every backend (the reference at `--bits`, the kernel
interpreter, native `f64`, `f32` and `bf16`, the first two also with vector
lanes) with `--dim`/`--const`/`--seed` as `presk run`, and prints the median
of `--repeat` calls with the speedup over the first row (`native` chapter);
`--opt [--profile FILE.json]` times the rewritten program, `--threads N` as
for `presk run`. `--native-only` drops the reference and kernel-interpreter
rows: the exact interpreter at a thousand cubed is what kept a large
benchmark from finishing at all (0.5.0 step 7). A `--realize` that is not one of the uniform rows gets a row of its
own, and a quantised one runs on its integer weights. `--plan @P.json` (0.7.0
step 1) gives its realization and adds a row `native … plan`, the same
realization under the plan's layout, so that the two rows differ by the layout
alone; with `--calibrate`, the kernels are timed under the plan's layout. The
inputs are rounded
to each row's machine type, and weights quantised, **once, outside the
timing** — since 0.6.0 step 5b; before it, every native row also paid a
conversion of its inputs from the reference's exact reals on every call,
which on a 1024³ matmul was most of the time measured (D46).

`--calibrate FILE.json` instead times every kernel the entry launches on the
native backend in the `--realize` machine type (`--repeat` calls, at least
three; the first launch of a kernel is dropped, the median of the others
kept), computes each kernel's work and traffic from its IR, fits the
roofline `λ + max(W/π, Q/β)` and writes the profile; an existing profile of
the same realization keeps its other kernels and is refitted with the new
ones. The table printed has one line per kernel (work, bytes, measured and
predicted microseconds), then the constants and the rank correlations of
three predictors with the measured times (`optimization` chapter).

```sh
presk bench tests/examples/minigpt/train_minigpt.psk --entry loss --const V=64 --const D=16 --const H=2 --const L=2 \
     --dim B=2 --dim T=8 --calibrate out/profile.json
```

`--calibrate FILE.json --space calib` (0.7.0 step 4) calibrates the **layout
cost model** instead (`optimization` chapter, "The layout cost model"). The
entry must have one contraction, a GEMM (`experiments/gemm.psk`, `--dim
N=…`). It is timed natively under 96 layouts that span the model's terms:
the register tiles `[1, 8]` and `[4, 16]`; no k-block in the orders `IJ` and
`JI`, and a k-block of 128 in every order of `IJK`; one lane and the type's
width; one worker, half the machine's and all of them. Each kernel's
planned cost (one worker's work, its traffic from L2, L3 and memory, its
register pressure) is computed on the machine's description, read from the
operating system for the first CPU the process may run on. The model is
fitted on every sample the file holds, this run's and earlier ones, so
several sizes and types add up. The table has one line per layout (group,
measured and predicted microseconds), then the constants, the uncertainty
`u` and Spearman. The file keeps its roofline, and a roofline calibration
keeps the layout model. A layout ill-formed at the size (a k-block that does
not divide `N`) is skipped and counted. `make calibrate-ra4` runs it at 256,
512 and 1024 in `f32` and `f64`, pinned to the cores of `RA4_CORES`.

```sh
taskset -c 5-9,15-19 presk bench experiments/gemm.psk --entry gemm --dim N=512 --realize f32 \
     --calibrate out/profile.json --space calib --repeat 5
```

## `presk plan FILE.psk --entry ENTRY --profile P.json …`

```text
presk plan FILE.psk --entry ENTRY [--dim N=v …] [--const N=v …] [--realize R]
           --profile P.json [--nest NEST] [--space full|small] [--cost] [--seed S] [--out PLAN.json]
```

Chooses a layout for every nest of an entry **statically**, from the layout
cost model of a calibrated profile (`presk bench --calibrate P.json --space
calib`; `optimization` chapter), and writes a realization plan that `presk
run --plan @PLAN.json` reads (0.7.0 step 5). The entry is run once natively,
under its default layout, on inputs synthesized from `--seed` as `presk run`
synthesizes them, to capture each nest's first launch at every shape — its
operands, its types, its input buffers; each captured nest is then
lowered under every layout of its space (the reference's `L_r(nest)`, times
every thread split the machine allows, each checked against the kernel as
condition 8 asks) and costed. No kernel is compiled or timed. The table has
one line per nest: its operands, the size of its space, the best prediction,
how many points lie within `1 + u` of it — the ones the model cannot tell
from the best, `u` being its median relative residual — and the layout
chosen; `--cost` also prints every point's prediction, and the last lines
give the time spent costing and the wall time. `--space small` narrows every
space to the daily subset of `presk autotune`. The plan names the profile it
was chosen against (question 64: a plan is not portable). Each nest gets the
best prediction, a tie broken by the order of the space, at the first shape
it is launched at. On the course GPT's attention at batch 16 that pick is
2.2–3.0 times slower than the best point measured, because the model cannot
see a register tile (`optimization` chapter, "What the layout model cannot
see"): the plan is the model's choice, not the best one.

```sh
presk plan corpus/nlp/05_gpt.psk --entry SelfAttention.apply --dim B=16 --dim T=256 --realize f32 \
     --profile out/profile.json --out out/attn_plan.json
```

At these shapes the entry has twelve nests and 4 857 points, costed in about a
second on falicon.

## `presk autotune FILE.psk --entry ENTRY --out DIR …`

```text
presk autotune FILE.psk --entry ENTRY [--dim N=v …] [--const N=v …] [--realize R]
               [--space full|small] [--repeat 7] [--profile P.json] [--nest NEST] [--interp]
               [--top N --random N] [--seed S] [--shard K/N] --out DIR
```

The exhaustive side of the RA4 test (plan 0.7.0 §3.6). Captures the entry's
launches as `presk plan` does, then, nest by nest and point by point, lowers
the nest under the layout, compiles it natively, launches it once to warm up
and `--repeat` times more **on the captured buffers**, not through the entry,
and records the median time and the median absolute deviation (MAD). Point 0
— the default layout, one lane, one worker, no pack — comes first, and every
other point's output is compared with its output **bit for bit** (statement
L); with `--interp`, every point also runs on the kernel interpreter and is
compared with point 0's output there (slow at large shapes: most of the
time of a sweep at the tutorial's sizes). A point whose bits differ is
reported and the command exits with 1. `--space small` is the daily subset:
the tiles `{1, 4} × {8, 16}`, no k-block or 128, the orders `IJ`, `JI`,
`KIJ`, `IKJ`, `IJK`, the splits none, `(0, p)` and `(1, p)` with `p` all
the machine's workers; `full` adds the other tiles, k-blocks and orders, the
split at depth 2 and half the workers. With `--profile`, each point also
records the model's prediction.

Beside each nest's space, 0.6.0's point is measured and checked too: the
default layout with its split left to the executor, as a run without a plan
has it (0.7.0 step 6), recorded with `"baseline": true` under the key
`0.6.0-default`. It is not a point of the space, and `presk rank` compares
the best point with it. `--top N --random M` (with `--profile`) takes a
sample instead of the whole space: point 0, the model's `N` best points and
`M` others, drawn by a hash of each layout's key under `--seed`, so that two
runs with one profile draw the same sample. `--repeat 0` runs each point
once, untimed, for its bits alone, and `--shard K/N` keeps the points whose
place in the selection is `K` modulo `N`. Point 0, the reference, runs in
every shard, and shard 0 records it and 0.6.0's point. Together, with
`--interp`, they split the check of statement L over processes
(`experiments/ra4_check.sh`, `make ra4-check`). Such a process should state
`PRESK_THREADS` as the timed sweep's worker count, so that it enumerates the
same splits.

`DIR` is one day of measurement: one file per nest, `<nest>.jsonl` with `/`
and `%` replaced by `_`, a JSON line per point (`entry`, `config` — the
constants then the dims, `B=16,T=256` —, `nest`, `op`, the primitive, `ein`,
`dot`, `softmax`, … or `fused`, `shape`, `dtype`, `layout`, `key`,
`baseline`, `timed`, the `seconds` of every timed launch, `median`, `mad`,
`predicted`, `identical`, `interp_checked`, and per kernel — a pack's copy
first — the model's `group`, its inputs `cost`, the `terms` of its
prediction, `λ`, spawn and the four bounds, and the `bound` that set it), and
`sweep.json` (the entry, the configuration, the dims, the space, the sample
and the shard if any, the number of points and of baselines, the time spent
measuring, costing, capturing and checking
on the interpreter, the wall time, the machine description, `u`, and
`not_identical`). Pin the process to a set of like cores (`taskset -c`):
the machine description is read for the first CPU it may run on.

## `presk rank DAY1 DAY2 --out FILE.md`

```text
presk rank DAY1 DAY2 --out FILE.md [--check DIR] [--entry E] [--at CONFIG] [--type T]
           [--gemm E] [--gemm-at CONFIG]
```

Compares the layout cost model with two days of `presk autotune`, by the
rule of plan 0.7.0 §3.6 (question 65). A **close pair** is two points of one
nest, at one shape and type, whose layouts differ in exactly one decision;
it is **distinguished** when on both days the medians differ by more than
twice the larger MAD, with the same sign; pairs not distinguished are left
out of the denominator. The model orders a pair **right** when its
predictions order it as the measurement does. A pair it predicts equal is
**tied**: not ranked, so not right (question 70), and counted apart. It
**decides** a pair when the predicted ratio lies outside `1 ± u`. Per nest, shape and type the report
gives the points both days measured and the model predicted, the close
pairs, the distinguished ones, the rank accuracy on them and on the decided
ones, the undecided and the tied fractions, Spearman between the predictions and
the mean of the two days' medians over the whole space, the top-1 regret
(the measured time of the model's pick over the best measured; the pick is
`presk plan`'s, a tie broken by the space's order) and the top-k
regret (the best measured among the `k` points within `1 + u` of the
model's best), and the best point against 0.6.0's. The **statements** of
plan 0.7.0 §3.7 follow, each measured and decided by the tool, not by hand:

- **L** holds when every point of day one was checked on the interpreter,
  in the days or in the check run apart that `--check` reads, and no output
  differed anywhere.
- **M** holds when each `ein` of `--entry` at `--at` in `--type`
  (`SelfAttention.apply`, `B=16,T=256`, `f32`) orders more than 90 % of at
  least 10³ distinguished pairs right. With fewer pairs it is inconclusive
  (plan §7).
- **N** holds when every configuration of that entry swept whole is chosen
  statically at least 10³× faster than it was swept, on each day.
- **O** holds when the model's pick is within 1.25× of the best measured on
  every contraction nest (`ein` or `dot`) at `--at`.
- **P** holds when the best point of `--gemm` at `--gemm-at` (`gemm`,
  `N=1024`) is at least 3× faster than 0.6.0's point.

Then come the counts per decision class (`tile`, `kblock`, `order`,
`lanes`, `split`, `pack`, `passes`, `form`), per nest and class, and the
**decision cost** per sweep: the exhaustive sweep's time (all of `presk
autotune` but the interpreter's check) against the static choice's (the
capture and the costing of the same points, which is what `presk plan`
does), a sampled sweep marked. A day's directory may hold one sweep or one
per configuration in subdirectories, as `experiments/ra4.sh` writes them.
Given one directory twice, `presk rank` applies the rule to one sweep and
says so. The report is also printed.

`make ra4-small` runs the whole test at the daily size: `experiments/ra4.sh`
sweeps the tutorial's attention (`tutorial/03_attention.psk`, entry
`attend`) at `B` = 1 and 16, `T` = 256, in `f32` and `f64`, twice — the first
day checking every point on the kernel interpreter too — pinned to the cores
of `RA4_CORES`, and ranks the two days into `out/ra4-small.md`, in about a
minute on falicon.

The measurement of 0.7.0 step 6 is four targets, run in this order on a
quiet machine:

- `make ra4` is day one, pinned: the course GPT's attention at the
  constants of the plan (`B` = 1 and 16, `T` = 256, `f32` and `f64`, the
  whole space), the GEMM at 1024 in `f32`, and a BERT-large-shaped attention
  (`D` = 1024, `H` = 16, `T` = 384, `B` = 1) on a sample of 40 points per
  nest.
- `make ra4-20` calibrates on all twenty cores (`out/profile-20.json`) and
  sweeps the attention there once, ranked alone into `out/ra4-20.md`.
- `make ra4-check` runs every point of day one again, untimed, natively and
  on the kernel interpreter, in shards on every core. The interpreter does
  about 40 M multiply-adds a second, 30 hours of it for the attention at
  `B` = 16 alone.
- `make ra4-day2` repeats day one and ranks the two days, with the check,
  into `out/ra4.md`.

```sh
taskset -c 5-9,15-19 presk autotune tutorial/03_attention.psk --entry attend --dim B=16 --dim T=256 \
     --space small --profile out/profile.json --interp --out out/ra4-small/day1
presk rank out/ra4-small/day1 out/ra4-small/day2 --out out/ra4-small.md
```

## `presk data chars | encode | pack | lm-windows | info | diff | export`

Produces and inspects `.psk.d` files (the `data` chapter has the format and
the rules):

```text
presk data chars INPUT.txt -o VOCAB.json                  # the distinct characters of a text, sorted
presk data encode INPUT.txt --vocab VOCAB.json -o TOKENS.txt
presk data pack --type TYPE INPUT.txt -o OUT.psk.d        # one sample per line, values in leaf order
presk data lm-windows --vocab V --seq T [--stride S] TOKENS.txt -o OUT.psk.d
presk data info FILE.psk.d [--samples N]                  # header and first samples (default 3)
presk data diff A.psk.d B.psk.d                           # ‖a − b‖₂ over the reals, and relative to ‖a‖₂
presk data export FILE.psk.d -o OUT.json                  # samples as nested JSON lists
```

`diff` compares two files of the same type value for value; two checkpoints
(`params M`) are compared without the program, a parameter tree being only
reals in file order — how far two training runs ended apart (0.6.0 step 7b).

**Exit codes**: 0 written or printed; 1 a data error (`E9xxx`, see the
`diagnostics` chapter); 2 an unreadable input file.
