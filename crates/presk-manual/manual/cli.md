# The `presk` command line

```text
presk <COMMAND>

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
output format is not stable. From PSK-Core on, the program is the inference
mode's, every command's but a training run's, in which `dropout` is the
identity and no `dropout_mask` remains; `--mode training` prints the
program as written, as `presk train` compiles it (0.16.0, D160).

| Stage | Output | Status |
|---|---|---|
| `tokens` | one token per line, `kind @ start..end` (byte offsets) | implemented |
| `ast` | the abstract syntax tree (Rust `Debug` format, spans as `start..end`) | implemented |
| `elab` | the elaborated program: per file, imports, constants with their values, dimensions with bounds, type aliases, modules with the shapes of their parameters and sub-modules, function signatures, discharged assertions; then the table of module instances with every parameter's shape (stacked axes included) | implemented |
| `core` | the PSK-Core intermediate representation after lowering (every function in A-normal form with its dimensions, parameters and types; `init` functions; lifted lambdas), verified before printing | implemented |
| `core-after-ad` | PSK-Core after automatic differentiation: every `grad` rewritten, with the generated `.vjp` and `.grad` functions, without their dead code (since 0.10.0 step 3: a `.vjp` returns the cotangents its callers read; `language` chapter, "Dead code after differentiation"), and since step 4 each `.vjp` split into `.fwd` and `.bwd` unless `--remat recompute` ("Residuals"); verified before printing | implemented |
| `kernel` | the PSK-Kernel v0 kernels launched by an entry, specialized on its shapes: needs `--entry` and `--dim`/`--const` as `presk run`, plus `--realize` or `--plan` (`native` chapter). Since 0.11.0 (D111) the kernels are lowered at the entry's shapes and none runs, so a model's listing takes the time of its host (the course GPT's forward pass at `B = 16`, `T = 256` in 0.18 s, where the kernel interpreter ran it for minutes); the listing is the one the run gave, at the kernel interpreter's defaults: without lanes unless a plan states them, so a dump without a plan, or with one that states none, prints the scalar kernels | implemented (0.3.0 step 2; `--plan` 0.7.0 step 1; from the shapes 0.11.0) |
| `sensitivity` | `‖∂L/∂v‖₁` at every site of a scalar entry and summed per module, from one reverse sweep: needs `--entry`, `--dim`/`--const`, and `--weights` to be worth reading, since a zero-initialized head makes every upstream gradient zero; `--sites N` lists more (`precision` chapter) | implemented (0.5.0 step 5) |
| `realize` (`sched`) | the fusion groups of every function (`realize` chapter): one line per group with its steps, its inputs and the launches it removes; with `--plan @FILE`, then the plan: its realization, and each nest it gives a layout, with the nest's primitive and the decisions stated, after checking that every nest named exists (`E7002`) | implemented (0.4.0 step 4; `--plan` 0.7.0 step 1) |
| `core-opt` | PSK-Core after AD and the e-graph optimizer (`optimization` chapter); the report (functions rewritten, `let`s and cost before and after, stop reason) goes to standard error; `--profile FILE.json` makes the roofline of that profile the extraction cost, `--dim` binds the dimensions it evaluates | implemented (0.3.0 steps 6–7) |
| `network` | the network diagram of an entry, as `presk explain` draws it (`explain` chapter, "The network diagram"): needs `--entry` and `--dim`/`--const` as `presk run`, takes `--realize` or `--plan`, `--profile` and `--weights`. `text`, the default, is its outline: every box's graph, each node and arrow with the element of the provenance table it names and its numbers, and the default opening's positions; `--format json` the `network` object of `explain.json`; `--format svg` and `--format tikz` one opening, `--level N` (0 the entry alone, 1 its first box opened, the default, then at each level a stack, else the first box, opened), `--level all` (every box opened, the exploded view, 0.14.1) or `--box NAME` (the box called `NAME` opened by itself), the TikZ a `tikzpicture`, a document with `--standalone` (LuaLaTeX); `--memory` colours each box by its share of the peak of live bytes, as the page's memory overlay does (`presk memory`) | implemented (0.14.0 step 3; `--level all` 0.14.1; `--memory` 0.15.0 step 4) |

`--unroll-stack` lowers `stack{L}` as `L` calls, as on `presk run`; every
stage but `tokens` and `ast` binds `--const` (`elab` and `core` ignored it
until 0.11.0 step 7); `kernel` and `sensitivity` take `--entry`, `--dim`,
`--const` and `--realize` as `presk run` does, `kernel` and `realize` also
`--plan`, `sensitivity` also `--weights` and `--sites`, and `core-opt`
takes `--profile`.

`--format latex|unicode` (0.13.0 step 4, D137) reads `--stage elab` or
`--stage core-after-ad` as mathematics, in the notation of the language
reference's chapter 3 (its section 3.10, "Programs as equations"; the
`language` chapter, "Reading a program as mathematics"); `text`, the
default, is the listing above. `elab` prints each function of the file and
its imports as the equations of its `let`s and its value, a module's once:
an operation by its name, `ein` as its sum at an index named by its axes, a
call through a stack as its recurrence. `core-after-ad` prints every
function of PSK-Core after differentiation, one equation per binding, the
bindings of a generated function with what they were made for (an adjoint
by the rule of its primitive, a primal's copy). `latex` is a fragment that
compiles with the reference's preamble (its macros `\code` and `\den`),
`--standalone` a document that compiles alone; `unicode` is the same for a
terminal. `--module M` and `--function F` select. A form the printer does
not write is printed as its source text and named on standard error; no
program of the repository has one. Unlike the other stages' listings, this
output is a product: its form is snapshot-tested (`presk-print-math`).

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
| `--remat store\|recompute` | what a gradient does with its forward pass's values (0.10.0 step 4, `language` chapter, "Residuals"): `store` (the default) keeps them, each `.vjp` its `.bwd` after its `.fwd`; `recompute` computes them again inside each callee's `.vjp`, as before. The same bits either way, fewer launches storing, more memory. `presk export`, `train`, `bench`, `plan`, `autotune`, `reprice`, `dump`, `calibrate` and `solve` take it too; every other command that differentiates stores. Under a realization that can see a fusion group (`bf16`, or a site typed apart from its function) a gradient keeps its groups as the recomputing `.vjp` had them, and not otherwise (0.10.0 step 4b, `language` chapter, "Dead code after differentiation"); a realization naming a stored `.vjp` is warned of (`W0004`). |
| `--seed S` | seed of every random choice (default 0): synthesized inputs, initializers, and the keys `sample` and `dropout` draw from when they are inputs. Same seed, same output, on every machine. |
| `--bits B` | MPFR precision of every intermediate, in bits (default 256). |
| `--digits D` | significant digits when printing (default 12). |
| `--json` | machine-readable output: entry, seed, bits, dimensions, inputs and result, reals as decimal strings at `--digits`, a value of type `bool[…]` as its axes and JSON booleans (since 0.10.0 step 8; its text output prints `bool[…]` of `true` and `false` too, by the entry's signature, where a mask is indices below 2 at run time). |
| `--verbose` | also print the synthesized inputs. |
| `--unroll-stack` | lower `stack{L}` as `L` calls (0.1.0's form) instead of a `fold`; a debugging aid, also on `presk dump`. |
| `--backend interp\|kernel\|native` | the executor behind the runtime seam (0.3.0): the reference interpreter, PSK-Kernel v0 interpreted in the `--realize` machine type, or the same kernels compiled by Cranelift (`native` chapter); `presk train` takes the same flag. |
| `--realize f64\|f32\|bf16\|@FILE` | the realization (0.3.0, an object since 0.5.0 step 2): a uniform machine type, or `@FILE` for a solved one carrying per-site storage and accumulator types and a declared reduction order (`native` chapter), since 0.6.0 step 5a a `quant` map of parameters stored as `int8`/`int4` (`precision` chapter), since 0.6.0 step 7b a `functions` map giving every site of a function one type (a site named in the per-site map overrides it), and since 0.12.0 step 1 a `params` map giving a parameter its own type at rest, `f32` or `bf16` (`precision` chapter; run by every executor and the C export since step 2). The interpreter ignores the machine types, computing at `--bits`, but **not** the `quant` map: every backend runs on the dequantised parameters. |
| `--plan @FILE` | a realization plan (`realize` chapter, "The realization plan"; 0.7.0 step 1): a realization and a layout per nest. `--realize R` is the plan with realization `R` and the default layout, so the two flags are exclusive. The plan is checked against the program before anything runs, and each nest's layout against its shapes and its kernel when it is launched; an ill-formed layout is refused (`E7002`), never ignored. Every decision of the reference is lowered since 0.7.0 step 3. The reference interpreter ignores the layout. Also on `presk train`, `presk bench` and `presk dump`. |
| `--opt` | rewrite the program with the e-graph optimizer first (`optimization` chapter); one summary line on standard error, the whole report with `--verbose`. Also on `presk train`, `presk serve`, `presk bench` and `presk export` (its conformance data). The optimizer writes each function it optimizes back with new numbers, even where no rule changed it, so a realization that types sites (`storage`, `accum`) or a plan's layouts name the program before it: under `--opt` they are refused, `E7002`, each site named with what the optimizer made of it (since 0.13.0 step 1, D134; until then they applied to nothing they named, D133). A uniform realization runs. |
| `--profile FILE.json` | with `--opt`: extract by the calibrated roofline of this machine profile (`presk bench --calibrate`) instead of operation counts; the run's `--dim`s are the dimensions it evaluates. Also on `train`, `serve`, `bench`, `export`, `dump --stage core-opt`. |
| `--threads N` | workers a parallel kernel may use on the native backend (`native` chapter); the default is the machine's cores, `1` keeps every kernel on one thread. Also on `presk train` and `presk bench`. |
| `--data FILE.psk.d` | take the entry's **last** argument from a data file instead of synthesizing it, and bind the `batch` axis of that argument to the number of samples read. An objective is then evaluated on its own dataset, and a plain function on a corpus; `--samples N` reads only the first `N`, from sample `--from I` (default 0). The other arguments are synthesized from `--seed` as usual. |
| `--weights ckpt/final.psk.d` | take the entry's **first** argument, its parameters, from a checkpoint (0.6.0 step 7): with `--data`, a trained model's loss on held-out text is one command. The checkpoint must be of the dimensions the entry is run at (`E9005` otherwise). |

**Inputs** are synthesized from the parameter types: real tensors from a
standard normal, index tensors uniformly in `[0, N)`, `params M` by running
the module's initializer (`init =` clauses: `zeros`, `ones`, `xavier_uniform`,
`normal(std = …)`), hyperparameters from their source default (`eps : f64 =
1e-5`) or `1`, a key from `--seed`, and a mask `bool[…]` with every position
allowed (0.10.0 step 7b). Each parameter draws from its own key derived from
`--seed`.

**Output**: the result with its type, e.g. `ℝ[batch: 2, model: 4] = [batch: 2,
model: 4] [[…], […]]`; parameter trees as nested `{ name = … }` records; index
tensors as integers.

**Exit codes**: 0 success; 1 a diagnostic (including a runtime error `E7001`,
labelled with the source of the failing operation, and an entry refused at
the dimensions given, `E7003`); 2 usage or internal error, including a
`training run` file without `--entry`.

**Undefined at the dimensions given** (0.10.0 step 7b): once the `--dim`
values are bound and checked against the file's bounds, an entry whose body
(or a function it calls, a closure it applies, a fold of count at least
one) holds an operation undefined at them is refused before anything runs
(`E7003`, `diagnostics` chapter): `mean` over no element, a selection
outside its axis, `embed`, `cross_entropy`, `argmax`, `sample` with no
index below `V`, their derivatives. Every command that runs an entry
checks it, `train` and `serve` at each call, on every backend.

**Randomness**: every draw is a function of a key. A key argument is
synthesized from `--seed` like any input; `sample(key, logits)` draws from
its key (it drew from a stream seeded by `--seed` until 0.10.0 step 7b).

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
| `--target pytorch` | the export this section describes. |
| `--target c` | a C program of one entry: the next section. |
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
`E8001`); 2 usage, including an option of `--target c`.

## `presk export FILE.psk --target c --entry E … -o DIR`

Writes one entry, at bound dimensions, as a C program that computes the bits
`presk run --backend native` computes with the same realization, dimensions,
seed and weights (plan 0.9.0; statement C1). Its kernels are the native
backend's, printed as C; its host is a recording of the native backend's own
run of the entry, every launch in order, structured as C functions: one per
distinct recording of a Presk function, each `stack` fold a loop over its
layers, buffers named after the source, each function releasing what it
allocated when it returns. Under a plan the kernels are the plan's, and a
launch the native backend divides among threads is an OpenMP region (0.9.0
step 5). An entry that samples draws from its key argument, as the native
backend does (0.10.0 step 7b), and `--serve` exports a generation function
with `presk serve`'s loop (step 6). A realization that quantizes weights (`int8`, `int4`) is
exported too: the program reads their reals and quantizes them as it reads
them, as `presk run` does (step 7). The `export_c` chapter says how it is
made, why its bits are the native backend's and what is checked; tutorial
step 18 exports the tutorial's attention.

```sh
presk export tests/examples/mlp/mlp.psk --target c --entry mse_grad --dim D=4 --dim B=2 -o out/mse_c
make -C out/mse_c check      # builds it, runs it on conformance/, compares with the native backend
presk export corpus/nlp/05_gpt.psk --target c --entry GPT.apply --dim B=1 --dim T=256 \
     --realize bf16 --weights out/course_probe_f32/final.psk.d --no-exact -o out/gpt_c
presk plan corpus/nlp/05_gpt.psk --entry GPT.apply --dim B=16 --dim T=256 \
     --profile ~/presk-data/ra4-model-0.8.0/profile.json --out out/gpt16.json
presk export corpus/nlp/05_gpt.psk --target c --entry GPT.apply --dim B=16 --dim T=256 \
     --plan @out/gpt16.json --weights out/course_probe_f32/final.psk.d --no-exact -o out/gpt16_c
OMP_NUM_THREADS=1 make -C out/gpt16_c check    # the same bits on one thread
presk export tests/examples/minigpt/serve_minigpt.psk --target c --serve \
     --const V=64 --const D=16 --const H=2 --const L=2 --dim T=8 --weights out/ckpt/final.psk.d -o out/serve_c
make -C out/serve_c          # model and generate
out/serve_c/generate --weights out/ckpt/final.psk.d --prompt "1 2 3 4 5 6 7 8" --tokens 8 --json
```

| Flag | Meaning |
|---|---|
| `--entry E` | the entry, as `presk run --entry` names it (`mse_grad`, `LayerNorm{8}.apply`, `adamw.step`, `next_token`); required, but with `--serve`, where it names the generation function as for `presk serve`. |
| `--serve` | the generation function of the file's `inference serve` (or `--entry`), at one sequence (`B = 1`, which `--dim` may not change) and the window `--dim T=N` its model's parameters fix, with its greedy copy and `generate`, `presk serve`'s loop (step 6; below). A model whose parameters do not fix the context's length is refused (`E8009`). |
| `--dim NAME=N`, `--const NAME=N` | as for `presk run`; every dimension must be bound (`E8003`). |
| `--remat store\|recompute` | as for `presk run` (0.10.0 step 4): a gradient entry is exported with its forward pass stored, `f.bwd` after `f.fwd`, by default; `recompute` exports the recomputing `.vjp`s. The C program follows the native backend's run either way, and C1 holds for both. |
| `--realize f64\|f32\|bf16\|@FILE` | the realization, `f64` by default; a solved one (`@FILE`) too, its buffers converted at the kernels' edges as the native backend converts them; one with a `quant` map (`int8`, `int4`; `presk quantize`) too since step 7, below; one with a `params` map since 0.12.0, each such weight in its own type in the arena and widened by the kernels' loads. |
| `--plan @P.json` | a realization plan (`presk plan --out`), in place of `--realize`: its realization and a layout per nest, checked as `presk run --plan` checks them (`E7002`); the native run is the plan's, so the program's kernels are the plan's tiles, orders, lanes and packs, and its splits are the program's threads (0.9.0 step 5). |
| `--threads N` | the native executor's workers, as for `presk run` (default: the cores). A plan's split must state `N` or `N / 2` workers (condition 5), and a kernel the plan leaves to the executor is divided among `N` when its work reaches the executor's threshold: the exported program divides it the same way. |
| `--weights CKPT.psk.d` | trained parameters for the entry's first argument, a `params M` checkpoint of the dimensions the entry is exported at, as `presk run --weights` reads it. The program reads it where it is: the conformance script names its absolute path. Without it, the parameters are the ones `presk run` synthesizes, written to `conformance/weights.psk.d`. |
| `--seed S` | the seed of the synthesized inputs and parameters, `presk run`'s, and of the keys `sample` draws from: `main -s` and `generate --seed` default to it. |
| `--no-exact` | leave the exact reference out of `conformance/`: at a model's full size the reference interpreter takes hours (the course GPT at `T = 256`). |
| `--straight-line` | write the host as one function performing every operation in order, releasing nothing (0.9.0 step 3): the structured host's oracle, which computes the same bits in an arena as large as every buffer together. |
| `-o DIR` | the directory, created; files already there are replaced. |

The PyTorch export's options (`--mode readable` and `graph`, `--shapes`,
`--stack`, `--conformance-entry`, `--train`, `--opt`, `--profile`) are
refused with this target, and this target's with `--target pytorch` (exit
code 2); `--mode scop` is this target's (below); `--serve` is
both targets' (`serve.py` for PyTorch, `generate` here);
`--plan` with `--realize` too. The options are checked first, then the
program (its diagnostics, exit code 1), then the plan, then the entry.
`--plan` and `--threads` were refused with `E8002` until 0.9.0 step 5.

**What is fixed at export.** Dimensions and constants, the realization, and
the entry's natural, integer and boolean arguments (`t` and `total` of
`adamw.step`), which may shape the program: they take the values `presk run`
synthesizes, and the README lists them. An `f64` argument (`eps` of a norm,
the temperature of `next_token`) is not fixed since step 6: it is a `double`
of the entry's `_in` struct, which `main` sets to the value `presk run`
synthesizes and a caller sets as it likes. A key argument is dropped: the
program holds no key, and a key may only be split or given to a dropout mask
outside training, which reads none (`E8006` otherwise). An entry that calls
an initializer (`E8004`) or returns a scalar (`E8008`) is refused. An entry
that samples was refused until step 6 (`E8005`): `sample` takes no key.

**Integer weights** (step 7). Under a realization with a `quant` map, the
leaves it names are integers and a scale per channel in the program, as in
the native backend, and no file holds them: a checkpoint holds reals, and
`main` (and `generate`) reads a quantized leaf's reals and quantizes them
as it reads them (`presk_quantize`), as `presk run --weights` does, the
scheme being the realization's and the scales computed from the reals. So
the weights file is the same with and without the `quant` map:
`conformance/weights.psk.d` holds the parameters before quantization, and
`--weights` names a checkpoint of reals. A leaf of integer weights is a
`presk_q8` or `presk_q4` of the `_in` struct, its integers and its scales
(`presk_rt.h`); a program embedding the entry quantizes its weights with
`presk_quantize`. An entry whose result holds integer weights, or which
reshapes them (`merge`, `split`), is refused (`E8007`), where every integer
weight was until step 7.

**The directory.**

| file | holds |
|---|---|
| `model.h`, `model.c` | the entry, `presk_<entry>(arena, in, out)` (the entry's name made a C identifier: `LayerNorm{8}.apply` is `presk_LayerNorm_8_apply`, `θ` is `theta`), its inputs and outputs as structs of typed pointers, one per leaf, a leaf of integer weights a `presk_q8` or `presk_q4` of its integers and scales, a `double` per `f64` argument, a `presk_key` per key argument that `sample` draws from (0.10.0 step 7b; a key nothing draws from is no field), and the arena's size; with `--serve`, the greedy copy `presk_<entry>_greedy(arena, in, out)` beside it, on the same structs; in `model.c` the `static` functions they call, callees first (below) |
| `kernels.c` | its kernels, one `static` function each under its PSK-Kernel text, deduplicated; `model.c` includes it |
| `main.c` | reads the weights and the inputs (quantizing the leaves the realization quantizes as it reads them), calls the entry, writes the outputs: `./model [-r R] [-s SEED] [WEIGHTS.psk.d] [INPUTS.psk.d] OUT.psk.d`; with `-r R` it calls the entry `R` times, the arena released between calls, and prints the median and the best wall time on standard error; `-s SEED` (an entry that samples) gives each key argument the key `presk run --seed` synthesizes for it, the export's `--seed` by default |
| `generate.c` | with `--serve`: `presk serve`'s loop over the entry and its greedy copy (below) |
| `presk_rt.h`, `presk_rt.c`, `presk_libm.c` | the runtime and musl 1.1.22's transcendental functions (`export_c` chapter) |
| `Makefile` | `make` builds `model` (and `generate`) with `-std=c11 -ffp-contract=off` (`STRICT`, never to change), `-O2 -Wall -Wextra`, and `OPENMP`: `-fopenmp` when a launch is divided among threads, empty otherwise; `make OPENMP=` builds without OpenMP, on one thread, with the same bits; `make check` runs the conformance script |
| `README.md` | what was exported: entry, dimensions, constants, realization, seed, the scalars fixed, the files' types, the launches, the functions and loops, how each fold was emitted, the arena against releasing nothing, the commit of the program's repository |
| `conformance/` | `weights.psk.d` (the synthesized parameters, as `presk train` writes a checkpoint: `params M`; reals, before any quantization), `inputs.psk.d` (the other tensor arguments), `native.psk.d` and `exact.psk.d` (the native backend's and the exact reference's outputs, reals as `f64`), `check.sh` |

The inputs and outputs are `.psk.d` files of closed types, a tuple of the
leaves with literal dimensions (`(ℝ[2, 4], ℝ[2, 4])`), so `presk data diff`
and `presk data export` read them without the program. `check.sh` builds the
program, runs it, and compares its outputs with `native.psk.d` byte for byte
(`cmp`); with `presk` on the `PATH` (or in `$PRESK`) it also prints the
distance to `exact.psk.d`.

**The host in `model.c`** (0.9.0 step 4). One `static` function per distinct
recording of a Presk function, callees first, then the entry's:

- **Names.** `presk_Block_apply`; with the dimensions when a function is
  called at several (`presk_Linear_apply_In16_Out64`); numbered when two
  calls at the same dimensions record different operations (an argument of
  another type under a mixed realization), the comment above each naming the
  other. Parameters are the Presk parameter and the leaf's path
  (`theta_attn_wq_w`); buffers the `let`'s source name (`q`, `h0`) or
  `t<let>`; a comment above the operations names their site,
  `"<function>/%<let>"`.
- **Calls.** A function writes its result into buffers its caller allocated
  and passed (`out`, `out_0_w`, …), and releases everything it allocated when
  it returns. A constant argument (`eps`) is not passed: it is in the
  operations that read it.
- **Folds.** A `stack` fold is a loop, `for (int64_t l = 0; l < L; l++)`,
  calling the block's function with each stacked parameter sliced as `w + l
  * chunk`, the accumulator alternating between two buffers (`x_buf[l %
  2]`). A gradient's backward fold runs from the last layer down, its body
  taking the index `i`. An iteration whose recording differs from the next
  one's runs apart, as a call of its own function, and the README says so.
- **The arena** is the program's peak: the inputs, the outputs, and the
  deepest chain of calls. `presk export` prints it beside what releasing
  nothing would take: `mse_grad`, 9 600 bytes against 14 848; the course GPT
  at `T = 256` in `bf16`, 29 MB against 88.

**Threads** (0.9.0 step 5). A launch is divided among the workers the native
backend divides it among: a plan's split among its workers, at the loop its
depth names; a kernel left to the executor's default, when its work reaches
the executor's threshold, over its outermost parallel loop among `--threads`
workers; everything else on one. Such a launch is one OpenMP region,

```c
#ifdef _OPENMP
#pragma omp parallel num_threads(presk_team(10))
    {
        int64_t t = omp_get_thread_num(), w = omp_get_num_threads();
        k7(x, mu, t4, 256 * t / w, 256 * (t + 1) / w);
    }
#else
    k7(x, mu, t4, 0, 256);
#endif
```

each thread calling the kernel on its piece of the divided loop, the pieces
the native backend computes. The team is the launch's workers, at most the
threads OpenMP would use (`presk_team`): `OMP_NUM_THREADS` lowers it and
never raises it, and `OMP_NUM_THREADS=1` runs the program on one thread. (A
`num_threads` clause alone would override `OMP_NUM_THREADS`.) The bits do not
depend on the team, nor on OpenMP being there: every element of a divided
loop is computed by one thread, by the same loops in the same order. The
split belongs to the launch, not to the kernel's text, so two calls of one
function whose launches are divided differently are two functions. At the
test set's sizes nothing is divided, since `presk plan` states no split and
no kernel reaches the threshold; the course GPT at `B = 16`, `T = 256` under
its plan divides 201 of its 228 launches among ten threads.

**Sampling and generation** (0.9.0 step 6; 0.10.0 step 7b). `sample(key,
logits)` draws from its key, an argument of the entry the program holds as
a `presk_key` (`presk_rt.h`: 32 bytes; the key a seed names, a key's split
and its child; the uniform draws of ChaCha20 as `rand_chacha`'s), which
`main` derives from `-s SEED` as `presk run --seed` does and a caller sets
as it likes. The draw is the native backend's, in `f64`, with `presk_exp`.
A key the program does not hold (split, or stacked) given to `sample` is
refused (`E8006`). With `--serve`, `generate` runs `presk serve`'s loop,
token `i` given the child `i` of the key `--seed` names, and its output is
`presk serve --backend native`'s, byte for byte, `--json` included
(statement C5):

```sh
out/serve_c/generate --weights W.psk.d --prompt "t1 t2 …" [--tokens N] [--temperature X] [--seed S] [--json]
```

The context is the last `T` tokens (the window the export fixed); each step
computes a token, appends it and moves the window on. At temperature 0 a step
calls the greedy copy (every `sample` an `argmax`) with temperature 1, above
it the entry, from the token's key. The defaults are
`presk serve`'s: 16 tokens, the function's default temperature, and the seed
the program was exported with. A prompt shorter than the window, a token
outside the vocabulary and a negative temperature are refused as `presk
serve` refuses them.

**Exit codes**: 0 written; 1 diagnostics, a refusal (`E8003`, `E8004`,
`E8006`–`E8009`), a plan the program refuses (`E7002`) or an error of the run
(`E7001`, `E9xxx` for a checkpoint); 2 usage (`--serve` with `B` other than
1 too), a plan file that cannot be read, an internal error.

## `presk export FILE.psk --target c --mode scop --entry E … -o DIR`

Writes each nest of an entry at **point zero**, its default layout on one
lane and one worker, no pack (0.6.0's kernel, the one every point of RA4's
layout space computes the bits of), as a standalone C function between
`#pragma scop` and `#pragma endscop`: the input a polyhedral tool reads
(pet, PPCG, PLUTO, Polly), with a driver that times each nest and checks it
against the native backend (plan 0.9.0 §3.7, step 8). The nests are the ones
`presk plan` ranks, captured the same way from the entry's shapes, running
no kernel: each fusion group and tensor primitive of a function
(`<function>/%<let>`), one C function per shape and type it is launched at.
The kernels the host launches itself (positions, causal masks, one-hot rows,
fills) are not nests, no plan stating their layout, and are left out. There
is no host: the nests are what a polyhedral tool would transform, and the
comparison of such a tool with RA4's ranking comes after 0.9.0.

```sh
presk export tests/plan/minigpt_grad.psk --target c --mode scop --entry loss_grad \
     --const V=64 --const D=16 --const H=2 --const L=2 --dim B=2 --dim T=8 --realize f32 -o out/loss_grad_scop
make -C out/loss_grad_scop check     # every nest built on its own, run once, compared with the native backend
make -C out/loss_grad_scop time      # each nest called eleven times: median and best, in µs
make -C out/loss_grad_scop -j polly POLLY=.llvm/bin/clang    # Polly's SCoPs, nests/NAME.polly (make llvm)
```

| Flag | Meaning |
|---|---|
| `--mode scop` | this mode; `--target c` only. |
| `--entry E` | the entry whose nests are written; required. |
| `--dim`, `--const` | as for `presk run`. |
| `--realize f64\|f32\|bf16\|@FILE` | the realization, `f64` by default; a mixed one (`@FILE`) gives each nest its site's type. A realization that quantizes weights is refused: a nest reading integer weights loads `f32(q)·s` from packed integers and their scales, which the arrays of this mode do not print. |
| `--seed S` | the seed of the shapes' capture and of the driver's draws. |
| `-o DIR` | the directory, created. |

`--plan` (every nest is at point zero, before any plan), `--threads` (one
worker, no OpenMP), `--straight-line` and `--serve` (no host), `--weights`
(the shapes do not depend on them, and the driver draws its inputs) and
`--no-exact` (the nests are compared with the native backend) are refused,
as are the PyTorch export's options (exit code 2, nothing written).

**The form.** Each nest is `nests/NAME.c`, `NAME` the nest's name made a C
identifier (`Linear.apply/%3` is `Linear_apply_3`, `Linear_apply_3_2` at its
second shape), its PSK-Kernel text and what it is in a comment:

```c
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
void loss_vjp_6(const float in0[restrict 2][8][64], const int64_t in1[restrict 2][8], float out[restrict 2][8]) {
    int64_t i0 = 0, i1 = 0, i2 = 0;
    float s0 = 0.0f, s1 = 0.0f;
#pragma scop
    for (i0 = 0; i0 < 2; i0++) {
        for (i1 = 0; i1 < 8; i1++) {
            s0 = in0[i0][i1][0];
            for (i2 = 0; i2 < 64; i2++) {
                s0 = in0[i0][i1][i2] > s0 ? in0[i0][i1][i2] : s0;
            }
            s1 = 0.0f;
            for (i2 = 0; i2 < 64; i2++) {
                s1 = s1 + presk_expf(in0[i0][i1][i2] - s0);
            }
            out[i0][i1] = presk_logf(s1) + s0 - in0[i0][i1][in1[i0][i1]];
        }
    }
#pragma endscop
}
#pragma GCC diagnostic pop
```

- **Arrays of literal dimensions**, one index per axis, as PolyBench/C writes
  its kernels (a buffer of rank 0 is `[1]`); the export's other programs
  address a buffer at its linear offset. Reals are `double`, `float` or
  `uint16_t` (`bf16`, computed in `float`), indices `int64_t`. The output
  starts at zeros, as the native backend allocates it.
- **Every loop** runs from 0 to a literal bound; no range, no OpenMP.
- **A branch on a computed value that only assigns scalars is a selection**
  of each value: a softmax's running maximum (above), an argmax, whose two
  values share one condition computed first (`c0`). The same bits, and
  control that depends on no data.
- **The transcendental functions** are declared `const` in each file that
  calls them (musl's functions read and write no memory), and `presk_rt.h`'s
  conversions are `const` (`PRESK_CONST`): a tool may take a call for an
  operation of its statement.
- **`#pragma scop` warns under `-Wall`** on GCC and Clang, which do not know
  it: each file ignores `-Wunknown-pragmas` around its function, so it builds
  without a warning, and any other unknown pragma still warns.

**Static control.** A nest is a static-control part when no loop variable is
assigned and every access indexes each axis by an affine function of the
loop variables (every loop runs to a literal bound already). The nests that
are not are the gathers and the scatters, which index an axis by a value
loaded from an index array: an embedding (`table[ids[b][t]][d]`), a
cross-entropy's label (`logits[b][t][y[b][t]]`, above), an embedding's
gradient (`out[ids[b][t]][d] += …`). On the test set, 366 of the 383 nests of
each type; Polly takes each of them as one SCoP, the whole function, with
the nest's loops, and none of the 17 others unless allowed non-affine
accesses. `nests.json` and the README list every nest, its loops (those of
two iterations or more) and whether it is static control, and why not.

**The directory.**

| file | holds |
|---|---|
| `nests/NAME.c` | one nest each, compiled on its own (`#include "../presk_rt.h"`) |
| `nests.h` | their declarations |
| `driver.c` | `./scop [-r R] [-s SEED] [NEST…]`: each nest (or those named, by C or Presk name) called `R` times (5 by default) on inputs it draws, the median and the best wall time in µs; `./scop -c conformance/native.bin`: each called once, its output compared with the native backend's byte for byte |
| `presk_rt.h`, `presk_libm.c` | the conversions and musl 1.1.22's transcendental functions |
| `Makefile` | `make` builds each nest's object with `-std=c11 -ffp-contract=off` (`STRICT`) `-O2 -Wall -Wextra` and the driver `scop`; `make check`, `make time`; `make polly POLLY=CLANG`, Polly's view of each nest in `nests/NAME.polly` |
| `nests.json` | per nest: its C name, Presk nest, kernel, operands, type, launches, loops, whether it is static control and why not |
| `README.md` | the above, and a table of the nests |
| `conformance/native.bin` | the native backend's output of each nest at point zero on the driver's inputs, in order, little-endian |

**The driver's inputs** are drawn with `splitmix64`, seeded per nest by the
seed and the nest's place, the same numbers in presk and in C: a real
uniformly in [-1, 1), rounded to the nest's type as presk rounds it; an index
below the least length of an axis it indexes, or of a loop it is compared
with (a cross-entropy's gradient compares its labels with the classes), so
that every gather and scatter stays in its array; 0 when it does neither.

**Polly.** `make polly POLLY=CLANG` compiles each nest with a clang that links
Polly (LLVM's releases do; `make llvm` installs LLVM 23.1.2's in presk's
`.llvm`), with `POLLYFLAGS`: Polly at its early position, before LLVM's loop
passes (at its default position `-O2` has rewritten the outer loops of most
nests, whose bounds Polly then cannot derive, and unrolled the short ones);
a whole function allowed as a SCoP, and a small one kept
(`-polly-detect-full-functions`, `-polly-process-unprofitable`); no
rescheduling (`-polly-optimizer=none`), so that the AST is the nest's own
schedule; no code, only the analysis (`-polly-code-generation=ast`: Polly
23.1.2 crashes generating the code of a SCoP of no loop); `-fno-math-errno`,
under which `sqrt` is the operation and not a call that may write `errno`.

**Exit codes**: 0 written; 1 diagnostics or an error of the capture
(`E7001`); 2 usage (the options above), an internal error.

## `presk train FILE.psk …`

Executes the file's `training run` on the backend of `--backend`/`--realize`
(the reference interpreter by default): see the `training` chapter for the
loop, the flags (`--const`, `--data`, `--steps`, `--epochs`, `--seed`,
`--bits`, `--log-every`, `--ckpt`, `--json`, `--threads`, `--opt`,
`--profile`) and two runs. `--realize @R.json` takes a realization solved for
the training step (`presk solve --step`), whose `functions` map gives each
function of the step its type; a realization that quantises weights is
refused. `--plan @P.json` takes a realization plan instead, as `presk run`
does (0.7.0 step 1). The loop calls the objective's `value_vjp` once a step
(0.10.0 step 3), and what a realization or a plan states of the
objective's `.vjp` and of the objective's own sites it states of that
function (`training` chapter). `--remat store|recompute` as `presk run`: stored
(the default), the step is `f.bwd` after `f.fwd`, one forward pass. Besides the log, standard error gets the kernel launches a step
(native backend) and the median step time with the first step excluded;
`--json` carries `ms` per step and `step_ms`. `--host-profile` (0.10.0 step
5), with `--backend native` only (exit code 2 otherwise), times the native
host by phase in every step, as `presk bench --host-profile` does a call,
and prints the table of a step on standard error, the mean of every step
but the first, with its page faults; only the clock is read, and the run's weights are the same
bytes with it and without.

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
(`--entry FN` overrides the one the block's `generate` field names or, without it, the one found by signature), on the backend of
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
    [--compare OTHER.json] [--remat store|recompute]
```

Measures the **magnitude envelope**: the largest `|x|` at each site, on the
*reference* interpreter, since the bound it feeds is a statement about the
denotation and must not be read off a rounded run. With `--data`,
`--samples N` folds in a batch of `N` and `--from K` skips the first `K`, so
two disjoint ranges give two independent measurements of the same program;
without `--data` the entry runs on the one input `--seed` synthesizes, and
`--samples` is refused (exit 2) since 0.11.0, where it was recorded and
printed as `N` samples (D108, finding 2): bind a batch axis with `--dim` to
measure several rows. `--compare` prints how far two envelopes disagree. `--precision` is required and is the only mode — the
roofline calibration is `presk bench --calibrate`, a different measurement of
a different thing. See the `precision` chapter.

At every sum and contraction it also records what the terms add up to
without cancelling — `Σ|xᵢwᵢ|`, `√Σ(xᵢwᵢ)²` and each operand's normwise
factor — and prints the five worst condition numbers `Σ|xᵢwᵢ| / |Σxᵢwᵢ|`
(0.6.0 step 6). `--weights` reads the entry's first argument from a
checkpoint, which is refused if it is not exactly the layout's size (D48);
the envelope records the weights and the data slice, so that a gap measured
beside the bound, and a realization solved from it, are on the same inputs.
With `--data` and more than one sample it also runs the entry on the first
sample alone, for its element counts only, and records them as
`per_sample` (since 0.16.0, D158): `presk solve` prices the activations at
them, the same at any `--samples`.

### `presk calibrate --step`: one training step

```sh
presk calibrate --step FILE.psk --weights ckpt/final.psk.d [--warmup 10] [--steps N]
    [--data FILE.psk.d] [--samples 2] [--from 0] [--const NAME=N]
    [--realize f64|f32|bf16|recipe|@R.json]… [--profile FILE.json]
    [--reference exact|f64] [--bits 256] [--seed 0] [--out ENV.json]
    [--remat store|recompute]
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
on the training step". `--remat` as `presk run` (since D93): the step's
functions are the `.fwd`s and `.bwd`s storing, the `.vjp`s recomputing, and
the envelope records which, since the sites it measures are theirs.

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
`--measure` runs the realization on the kernel interpreter, whose bits
are the native backend's (statement U5), on the envelope's own inputs and
weights, and prints the measured distance to the exact reference, with
each row's ratio to it (until 0.11.0 step 7 the line said "native"); an element whose difference is not finite (a value that
overflowed or became NaN where the reference is finite) makes it `inf`, the
reference's gap, as for `presk solve`'s and `presk quantize`'s measurements
(since 0.10.0 step 8; such an element was left out before, so a realization
that overflowed was measured on the others, and `presk solve` could accept
it). `--sites N` lists the `N` widest sites with their condition
numbers. The bound is at the envelope's dimensions, and on a module entry
(`LayerNorm{8}.apply`) at the module's own arguments too: until 0.11.0 those
were left unbound, and a sum over them was priced at a default length of
1 024 terms, 93× the bound on `LayerNorm{8}.apply` (D108, finding 1). See
the `precision` chapter, "Does the condition number rescue the bound?".

## `presk solve FILE.psk --precision ENVELOPE.json --budget ε`

```sh
presk solve FILE.psk --precision ENV.json [--budget 1e-3 | --budget-abs EPS] [--weights ckpt/final.psk.d]
    [--profile FILE.json] [--quant [--scheme absmax|pow2]] [--no-verify] [--bits 128]
    [--rounding unit|measured] [--narrow off|measured|heldout] [--heldout M]
    [--estimate sum|rss] [--margin 2.5] [--min-gain 1e-3]
    [--seed 0] [--out R.json] [--sites N] [--remat store|recompute]
```

Chooses a machine type per site under an error budget, minimising the
predicted time of the roofline fitted per machine type when given a
`--profile` (0.6.0 step 4; `precision` chapter, "Time, not traffic") and the
traffic otherwise, and writes a realization `--realize @R.json` can run. It **verifies by
measurement** against the exact reference rather than by the propagated
bound, which is vacuous on a real model; `--no-verify` gives the proposal
alone. It offers the values the program stores, each a computation's
result not fused into another, and nothing else: a parameter's field, a
tuple's projection, a call's or a fold's result carry a value another site
computes, and a fused member is computed in its group's root's type, so a
type stated for them changes no bit (since 0.11.0, D112: until then the
solver offered, counted and priced them). Such a value's rounding is
charged where it happens, a member's to its root. Since 0.12.0 (D123) it
also offers every function's result, which a sweep now ranks, and every
parameter, decided by its path and written in the realization's `params`
(`f32` or `bf16`): its term is its rounding at the narrowest of its own type
and the types of the nests that read it, and the fields that read it carry
none. Under a `--profile` every cost it prints is a time in
seconds, the exhaustive search's too (until 0.11.0 step 3 its line called
that time bytes and printed the greedy proposal as 0.000x it, D113).
`--weights` matters: a sensitivity ranking at
initialization is vacuous. The envelope's entry returns a real tensor: a
loss, its sites ranked by its sensitivity `∂L/∂v` as `presk dump --stage
sensitivity` ranks them; or, since 0.12.0, any tensor (logits), under a
budget on its largest elementwise gap, relative to its largest magnitude
(`--budget`) or absolute (`--budget-abs EPS`), the sites ranked by one
reverse sweep seeded with ones over the output, read per output element,
the measurement deciding. Another entry is refused (exit 2) before anything
is computed (D101; until then it failed in the reverse sweep's first
`.vjp`, after its forward half). The inputs are the envelope's: when it was measured on a data slice,
the sweep and every measurement run on that slice. `--quant` also offers every matrix parameter as `int8` or `int4`
(`--scheme absmax|pow2`), priced by the parameters' gradient and accepted by
measurement, and prints the three first-order readings per parameter with
`--sites`. A `--weights` file of other dimensions is an error, not a fallback
to the initial parameters. The traffic is one sample's: the activations at
the elements the envelope records for one sample (`per_sample`, since
0.16.0, D158), the parameters once, so that the cost is the same whatever
the number of samples the envelope was calibrated on. Two options of D155:
`--rounding measured`, the default since 0.16.0 (D157, D158), takes the
estimate's rounding step at each site from the envelope's measured
`max |fl_d(x) − x|` instead of `u(d)·max|x|`, so that a value exact in a
type costs nothing there, an envelope written before D155 keeping the unit
term, and `--rounding unit` is the solver of 0.12.0–0.15.0;
`--narrow measured`, off by default, adds a step after a
passing measurement: each decision not at its narrowest is narrowed one
step, the largest saving first, and kept while the measured gap stays inside
the budget divided by `--margin` (inside the budget itself until 0.16.0,
`--margin 1`), in up to three passes, and the output says how many it kept and
how many measurements the solve made. Three more since 0.16.0 (D161), the
ways to spend the budget the `precision` chapter describes: `--estimate
sum|rss`, the proposal's estimate as the sum of the sites' terms (the
default) or their root-sum-square, aimed at the budget divided by `--margin`;
`--margin K` (2.5 by default), the factor kept between the budget and what is
spent in sample, which `--estimate rss` and the narrowings read and the
default solve does not; and `--narrow heldout [--heldout M]`, the narrowing
stopped early on the `M` samples after the envelope's in its data file (as
many as its own by default), refused (exit 2) for an envelope not calibrated
on a slice of a data file. See the `precision` chapter. Since 0.16.1 (D169)
two rules keep a value at `f64`, unoffered: a value of one element (rule
1, no option), and a value whose first narrowing, `f64` to `f32`, saves
less than `--min-gain θ` of the entry's all-`f64` traffic (rule 2, 10⁻³ by
default, 0 to turn it off, in bytes even under a `--profile`; a θ outside
[0, 1) is refused, exit 2). The line `rules` names those values, with
rule 2's gains beside θ·B and what the two forgo at most; the line `sites`
counts the values offered (the `precision` chapter, "Two rules").

### `presk solve --step`: a budget on the update

```sh
presk solve --step FILE.psk --precision ENV.json [--budget 1e-3] [--profile FILE.json]
    [--no-verify] [--bits 128] [--out R.json] [--remat store|recompute]
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
with it. `--remat` must be the policy the envelope was calibrated under (an
envelope written before D93 does not record it, and its sites' names say
it), and so should `presk train`'s: a
realization of the recomputing step names `.vjp`s, which a stored gradient
only calls.

**Exit codes**: 0, whether or not the measured gap is inside the budget (the
report says which); 1 a run-time error; 2 an envelope of an entry rather than
of a step, or of a step calibrated under the other `--remat`.

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
alone; with `--calibrate`, the kernels are timed under the plan's layout.
`--plan-only` (0.9.0 step 5) times that row alone, not the uniform
realizations' (at the course GPT's `B = 16` on one thread they take most of
half an hour), and `--weights CKPT.psk.d` replaces the entry's first argument
by trained parameters, as for `presk run`: how `make c-speed` times the
native side of an exported program. `--warmup N` (after 0.9.0, D87) makes
`N` calls per row before the timed ones and times none of them: the native
backend compiles its kernels on its first call, which the median of
`--repeat` calls otherwise counts (`--warmup 1 --repeat 7` is how `make
pytorch-baseline` takes a median of seven). The inputs are rounded
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
three predictors with the measured times (`optimization` chapter). A line
`-- a call:` (after 0.9.0, D87) gives the launches of one call and the
kernels' share of its time, each kernel's median launch times its launches:
the call's own time, from `presk bench` without `--calibrate`, less this is
the host's (its buffers, copies and views), provided both run the same
kernels. They do under a plan that states every nest's lanes: `--calibrate`
runs a nest no plan states on one lane, where the native backend of `presk
run`, `train` and `bench` gives it its type's natural width.

```sh
presk bench tests/examples/minigpt/train_minigpt.psk --entry loss --const V=64 --const D=16 --const H=2 --const L=2 \
     --dim B=2 --dim T=8 --calibrate out/profile.json
```

`--host-profile` (0.10.0 step 5, D95) times the native backend's host by
phase, under `--plan` (or the realization's default layout without one),
where `--calibrate` gives only the kernels' share. The host reads the clock
around each region of its work and the executor around what of a launch is
not the kernel's code; regions do not nest, so each phase's time is its
own, and what no region covers, the interpreter of the host program, is the
call's time less every phase. Only the clock is read: every buffer, kernel
and launch is the one a call without the flag makes. `--repeat` calls are
timed so and as many not, alternated, after `--warmup` calls (at least one,
since the first compiles); each is timed as `presk bench` times a call, the
copy of its arguments and the drop of its result included. Printed: both
medians and the timers' cost (their difference); whether a call with the
timers and one without give the same bits (exit code 1 if not); the
process's minor page faults a call, pages touched for the first time (read
from `/proc/self/stat` before and after a call, on Linux); and a
table, each phase's mean a call, its share, the megabytes it moved and its
regions a call, in six groups. Since step 6 (D96) a value's elements are
shared by its copies, views and stacks, so the phases of groups 3 and 4
count the bytes they still copy (none on a machine buffer), and `free` the
bytes a drop releases:

| group | phases |
|---|---|
| `kernels` | the kernels' own code: each launch timed around the executor, as `--calibrate` times it, less what follows under `threads` |
| `find` (1) | `site`: a `let`'s site named and its type found; `groups`: a function's fusion groups cloned at its call; `layout`: a nest's kind and its layout in the plan; `key`: the cache key formatted and the kernel looked up; `lower`: lowering and compiling (first launches; none after the warm-up) |
| `buffers` (2) | `alloc`: output buffers allocated; `convert`: inputs converted to a kernel's type; `free`: a function's values freed when it returns, and the result |
| `views` (3) | `slice`: a stacked tree's layer sliced, a scan's results stacked, a slice set or scattered (views and parts since step 6); `axes`: merged, split or reshaped axes, copied |
| `copies` (4) | `entry`: the entry's arguments typed (a leaf in another type converted); `operands`: a launch's operands read out of the environment; `as-tensor`: the same taken again as tensors; `args`: a call's arguments, a closure's captures; `tuples`: tuples, records, projections, fields, a tree's leaves around a primitive mapped over them; `fold`: a fold's arguments and a scan's accumulators. Copies until step 6 |
| `threads` (5) | `dispatch`: the executor's dispatch (addresses, the split's size); `start-join`: a divided launch's time less its longest piece's, its workers' start (their wake, since step 6's pool) and join |
| `left` (6) | `dropout`: masks drawn on the host; `host-prims`: initializers, `sample`, `zeros_like`, `split_key`; `interpreter`: the rest, by difference |

A last line gives the launches a call, those divided and their threads, and
the kernels as `--calibrate` counts them (code, dispatch and threads) against
the host. `make host-profile` runs it on the course GPT (`experiments`
chapter).

```sh
presk bench corpus/nlp/05_gpt.psk --entry GPT.apply --dim B=16 --dim T=256 \
     --plan @plans/forward.json --weights out/course_probe_f32/final.psk.d \
     --threads 10 --warmup 1 --repeat 7 --host-profile
```

`--calibrate FILE.json --space calib` (0.7.0 step 4) calibrates the **layout
cost model** instead (`optimization` chapter, "The layout cost model"). The
entry must have one contraction: a GEMM (`experiments/gemm.psk`, `--dim
N=…`), or a product of any shape (`experiments/matmul.psk`, `--dim I=… --dim
K=… --dim J=…`). Since 0.8.0 step 3 (plan 0.8.0 §3.3) it is timed natively
under 300 layouts that span the model's terms:
- five register tiles, `[1, 4]`, `[1, 8]`, `[2, 8]`, `[2, 16]` and
  `[4, 16]` (0.7.0 had the second and the last);
- no k-block in the orders `IJ` and `JI`, a k-block of 128 in every order of
  `IJK`, and of 32 in `IJK` and `KIJ`;
- one lane and the type's width;
- one worker, half the machine's and all of them; `--min-workers N` (0.8.0
  step 4) leaves out the layouts stated on fewer than `N`.

A layout a shape cannot take (a loop of one iteration split) is skipped and
counted. Each kernel's
planned cost is computed on the machine's description, read from the
operating system for the first CPU the process may run on: one worker's
work, its register loads and stores, its traffic from L2, L3 and memory,
and its register pressure. The model is fitted on every sample the file
holds, this run's and earlier ones, so several sizes and types add up. The
exception is samples recorded before 0.8.0 step 2, which have no load count
and are dropped, with a message. `--load-term` (0.8.0 step 2) says whether
the model has the load term and how its rate `ρ` is fitted: `group`, the
default, one per group as `π`; `machine`, one for every group; `none`, 0.7.0's
model without it. `--combine` (0.8.0 step 3) says how the terms combine:
`max`, 0.7.0's form; `sum`, the in-core term plus the memory term; or
`pnorm`, the default, whose exponent `q` is fitted too. `--rules` (0.8.0 step
4, D70) names the rules of the planned cost the samples are costed under:
`none`, or among `lines`, `shared` and `overhead` (`optimization` chapter).
Without it a calibration costs under the rules of the profile's samples,
and a new profile under `lines,overhead`; it refuses rules other than those
of samples it would keep. The defaults are the frozen model's form (D71),
since 0.8.0 step 8 (D75); 0.7.0's is `--load-term none --combine max
--rules none`. The table has one line per layout (group, measured and
predicted microseconds), then the constants (`ρ` per group when the model
has it), the uncertainty `u` and Spearman. The file keeps its roofline, and a roofline calibration
keeps the layout model. A layout ill-formed at the size (a k-block that does
not divide `N`) is skipped and counted. `make calibrate-ra4` runs it pinned
to the cores of `RA4_CORES`, in `f32` and `f64`, on the GEMM at 256, 512,
1024 and 2048 (`CALIB_SIZES`) and on the thin products `16 × 4096 × 4096` and
`4096 × 4096 × 16` (`CALIB_THIN`), meant to be bound by the traffic from
memory (on falicon they are not, D67). At 2048 (`CALIB_NO_ONE_WORKER`) it
passes `--min-workers 2`: the one-worker layouts took three quarters of that
size's time. Then it prints `presk refit --compare` and how long the
measurement took. The form fitted into the profile is `RA4_LOAD_TERM`,
`RA4_COMBINE` and `RA4_RULES`, by default the frozen model's since 0.8.0
step 4: `group`, `pnorm`, and `lines,overhead` (D70–D71). A calibration is
one run: an existing `RA4_PROFILE` is first
moved to `*-prev.json`, so that none of its samples stays in the fit.

```sh
taskset -c 5-9,15-19 presk bench experiments/gemm.psk --entry gemm --dim N=512 --realize f32 \
     --calibrate out/profile.json --space calib --repeat 5
```

## `presk refit P.json [--load-term L] [--combine C] [--out Q.json] [--compare] [--recost FILE.psk [--entry E] [--rules R]]`

Fits a profile's layout cost model again on the samples it holds, in another
form, **without measuring anything** (0.8.0 step 3). Every form of plan
0.8.0 §3.2 is fitted on the same calibration, and step 4's ablation re-prices
the development set under each. `--load-term` and `--combine` are those of
`presk bench --calibrate --space calib`, with the same defaults, the frozen
model's form. The refitted profile goes to `--out`,
or back into `P.json`, whose roofline, machine description and samples are
kept. Samples recorded before 0.8.0 step 2 have no load count and are left
out, with a message.

`--compare` fits nine forms instead and writes nothing: the three load terms
(`none`, `group`, `machine`) under each combination (`max`, `sum`, `pnorm`).
It prints one line per form: `u`, Spearman, `λ`, the spawn cost, the three
`β`, the p-norm's `q`, and how long the fit took. A p-norm's `q` is searched
from 1 to 100, two grids of ten fits run on as many threads, each fitting the
rates again, since a coordinate at a time cannot follow the trade-off between
`q` and every rate.

`--recost FILE.psk` (0.8.0 step 4) costs every sample again before the fit,
its time kept, so that a rule of the planned cost is tried on a calibration
without measuring it again:

- Each sample's layout is read back from its key and lowered at the product
  its signature names, on the entry `--entry` of the program (`matmul` by
  default, dimensions `I`, `K` and `J`; `experiments/matmul.psk`). The
  entry is captured once per product, its shapes only.
- The kernel is costed on the profile's own machine description, not the
  machine the command runs on, under `--rules`: `none`, or among `lines`,
  `shared` and `overhead`, comma-separated (`optimization` chapter), and by
  default the rules the profile's samples were costed under. `--rules` needs
  `--recost`.
- The kernel lowered must have the sample's signature and group, or the
  command fails.

It prints how many costs came back identical and which terms moved. The
fit records the rules, and every prediction made from the profile is then
costed under them (`presk plan`, `autotune`, `reprice`). A description
recorded before 0.8.0 step 4 has no line size: under `lines` it is read
from the operating system for the CPU the description names, and the
refitted profile records it. The definitive calibration comes back byte for
byte under its own rules, `none`; re-costed under `lines,overhead` and
refitted with the defaults, it gives the frozen model's profile byte for
byte (0.8.0 step 8, D75).

```sh
presk refit out/profile.json --compare
presk refit out/profile.json --load-term group --combine sum --out out/profile-sum.json
presk refit ~/presk-data/ra4-0.8.0-calibration/out/profile.json --recost experiments/matmul.psk \
     --rules lines,overhead --out out/profile-rules.json
```

## `presk plan FILE.psk --entry ENTRY --profile P.json …`

```text
presk plan FILE.psk --entry ENTRY [--dim N=v …] [--const N=v …] [--realize R] [--mode inference|training]
           --profile P.json [--nest NEST …] [--space full|small] [--cost] [--seed S] [--out PLAN.json]
```

Chooses a layout for every nest of an entry **statically**, from the layout
cost model of a calibrated profile (`presk bench --calibrate P.json --space
calib`; `optimization` chapter), and writes a realization plan that `presk
run --plan @PLAN.json` reads (0.7.0 step 5). The entry's launches are
captured from its **shapes** (0.8.0 step 1, D65). The host program runs
under the default layout on zero inputs: tensors are zeros of their axes,
and parameter trees are built by their initializers, with no kernel
compiled or run and nothing computed in exact arithmetic. It records each
nest's first launch at every shape, with its operands and types. Each
captured nest is then lowered under every layout of its space (the reference's `L_r(nest)`, times
every thread split the machine allows, each checked against the kernel as
condition 8 asks) and costed. No kernel is compiled or timed. The table has
one line per nest: its operands, the size of its space, the best prediction,
how many points lie within `1 + u` of it — the ones the model cannot tell
from the best, `u` being its median relative residual — and the layout
chosen; `--cost` also prints every point's prediction, and the last lines
give the time spent costing and the wall time. `--space small` narrows every
space to the daily subset of `presk autotune`. `--nest` restricts the plan
to the nests it names, `<function>/%<let>`, the option repeated or the names
separated by commas (several since 0.10.0 step 2); a name the entry does not
launch is refused, exit code 2. The plan names the profile it
was chosen against (question 64: a plan is not portable). A plan states one
layout per nest. A nest launched at one shape gets the best prediction, a tie
broken by the order of the space. A nest launched at several (a module
function at two sizes, `Linear.apply` at D→4D and 4D→D) gets the layout the
model prices at every one of them whose predicted time, each shape's
weighed by the times the entry launches it, is least; a line `= NEST at N
shapes …` gives it, with its time and the time at each shape's own best,
when the shapes do not all choose it. Until 0.9.0 (D82) the first shape
decided, and the MLP's plan stated a tile that did not divide its second
shape, which `presk run` refused (condition 3). A nest whose type the
profile was not calibrated in is left to the default layout and named on a
last line; when no nest can be priced (`--realize bf16` under a profile
calibrated in `f32` and `f64`), `presk plan` refuses, exit code 2, and writes
nothing, where it wrote an empty plan. The plan names the nests of the
program its realization makes: since 0.10.0 step 4b a gradient keeps its
fusion groups under `bf16` and not under `f32` or `f64` (`language`
chapter, "Dead code after differentiation"), so a gradient's plan made
under one of them is refused under the other (`E7002`, a nest the program
lacks or computes inside a group). The plan is also of one mode's program
(0.16.0, D164): by default the inference reading's, which `presk run`,
`bench`, `serve` and the C export compile, with no `dropout_mask` (D160).
A plan for `presk train --plan` is planned with `--mode training`, on the
program as written. A gradient with masks numbers its nests otherwise, so
the inference plan of a gradient that drops names nests a training run
fuses into others, and `presk train` refuses it (`E7002`). Since 0.10.0 step 7 (D97) a product
whose value one `add` alone reads, in the product's type, at every shape it
is launched at, takes that `add` as its epilogue (`"epilogue": true`, the
`realize` chapter's "The bias as an epilogue"), over the layout chosen
without it, wherever that layout lowers with it; a line `-- N products take
the add that alone reads them as their epilogue: …` names them. The model
prices a nest alone and has no term for it, so no prediction moves: the
epilogue saves the `add`'s launch and its pass over the output, and adds no
arithmetic. On the course GPT that is `Linear.apply/%3`, a linear layer's
product and its bias. The plan is the model's choice, not the best one
measured. Under 0.7.0's model that pick was 2.2–3.0 times slower than the
best point on the course GPT's attention at batch 16, because the model
could not see a register tile (D62). Under 0.8.0's frozen model it is within
1.05× of the best on every contraction nest statement O names, on both
fresh sweeps (D74; `optimization` chapter, "What the fresh sweeps
found").

```sh
presk plan corpus/nlp/05_gpt.psk --entry SelfAttention.apply --dim B=16 --dim T=256 --realize f32 \
     --profile out/profile.json --out out/attn_plan.json
```

At these shapes the entry has twelve nests and 4 857 points, costed in
0.36 s in `f32` on falicon's X925 cores after a capture of 0.18 s. In 0.7.0
it was 0.94 s after a capture of 0.57 s that ran the entry. Since 0.8.0
step 1 a layout is lowered once for all its thread splits, which change
only the kernel's `threads`, and the traffic into the three cache levels
is computed in one walk that shares the footprints. Every prediction is
the one 0.7.0 made (`make ra4-dev`, D64).

## `presk explain FILE.psk --entry ENTRY … --out DIR`

```text
presk explain FILE.psk --entry ENTRY [--dim N=v …] [--const N=v …]
    [--budget ε | --realize R | --plan @P.json] [--profile P.json] [--threads N]
    [--remat store|recompute] [--opt] [--weights W] [--seed S] [--bits B]
    [--envelope ENV.json] [--no-exact] [--target native,c,pytorch] [--measure]
    [--site SITE …] [--format md|json|html] --out DIR
```

Compiles the entry under one compilation's parameters and writes
`DIR/explain.md`: the source, the elaboration, PSK-Core, differentiation,
the envelope, the sensitivities, the solve or the realization given, the
bound beside the measured gap, fusion, the machine and the layout plan, the
kernels and the host, and the targets, each section ending with the
commands that print what it shows (0.11.0 step 3, D113). The long listings
go to `DIR/listings/`, the envelope, the realization, the plan and the
exported programs to `DIR/artifacts/`. The flags are the commands' own;
`--budget`, `--realize` and `--plan` exclude one another, and `--opt` is
refused with a profile or a plan (exit 2). `--measure` adds what one run
measures on this machine (the native output against the exact reference,
a call, each kernel beside its prediction, the host by phase); `--site`
names the sites traced through every stage (0.11.0 step 5). Without
`--measure` the document is deterministic, and every line the commands it
quotes print is in it verbatim (X3). `--format json` also writes
`DIR/explain.json`, the document's facts and the provenance table as values
(0.13.0 step 2, D135), and `--format html` also `DIR/explain.html`, the
pipeline view (step 3, D136), with the formulas of the printer of the
denotation (step 4, D137). `--opt` is refused with `--budget`, and with a
realization that types sites (`E7002`, 0.13.0 step 1). See the `explain`
chapter.

## `presk memory FILE.psk --entry ENTRY …`

```text
presk memory FILE.psk --entry ENTRY [--dim N=v …] [--const N=v …]
    [--realize R | --plan @P.json] [--remat store|recompute] [--depth K] [--seed S] [--symbolic]
presk memory FILE.psk --step [--dim N=v …] [--const N=v …] [--realize R | --plan @P.json] [--seed S]
```

What one call of the entry needs in memory, counted from the program at
its dimensions and its realization's types, never a process's resident
size (plan 0.15.0 step 1, D148). The entry is recorded as `presk export
--target c` records it, on an executor that runs no kernel, so it takes
the time of its host; each buffer the call makes is live from its first
write to its last read. It prints:
- the **parameters**, the entry's first argument when it is a parameter
  tree, each leaf at its stored type, integer weights with their `f32`
  scales;
- the **inputs**, every argument's buffers, which the caller holds;
- the **peak** of the live bytes, the operation where it is first reached,
  its site and its call;
- the **outputs**, and what is **needed**, the inputs and the peak;
- the **C arena**, the arena of the program `presk export --target c`
  writes, as the export computes it: every function's buffers held until
  it returns, so it is at least the peak (1.3 to 2.5 times it on the
  course GPT);
- the bytes the buffers take **placed** at offsets by their lives, two
  live at once never overlapping, the largest first: the hidden `presk
  export --target c --straight-line --placed` writes that program, and
  `make c-conformance` checks that it computes the native backend's bits,
  which is what shows the lifetimes sound (M1);
- each **call**, as the network diagram's boxes are, the iterations of a
  fold taken together (`blocks ×6`): its share of the peak, what it made
  that is live there, and its own peak, the most what it made weighs while
  it runs; `--depth K` shows the calls `K` levels deep (2 by default).

`--realize`, `--plan` and `--remat` are `presk export`'s, so the arena
beside the peak is the one the export prints. An entry whose result a C
program cannot write (a key, a host scalar) is counted all the same, with
no C arena. `presk explain` prints the same lines in its section 13.

**`--symbolic`: the polynomials** (plan 0.15.0 step 3, D150). The
inputs, the outputs and the peak as polynomials in the entry's
dimensions, the `--dim` names, the `const`s as their numbers. Every axis
of a buffer is a linear form in the dimensions (chapter 2), so a
buffer's bytes are a polynomial:
- each buffer's bytes are interpolated, exactly, in rationals, from
  recordings of the call on a grid of three values per dimension
  (multiples of the given value, or of its divisors, that the entry
  records at), the others at the given ones;
- the operations must be the same at every point of the grid. A
  dimension that counts a fold's iterations is refused;
- the polynomials are checked at one point off the grid, so a dimension
  in more than two axes of a buffer is refused too;
- the peak is the largest of the operations' live bytes that no other's
  dominates for dimensions of at least 1, all of them printed when there
  are several. On the course GPT's forward pass there are three: the
  feed-forward's 4 608·B·T + 8·T², and two of the attention's, which
  grow as B·T².

The printer of the denotation writes them (`presk-print-math`), in
Unicode here and in LaTeX in `explain.json`.

**`--step`: a training step** (plan 0.15.0 step 2, D149). No entry is a
training step. `presk train` makes one from three calls on one backend:
- the objective's `.value_vjp` on the parameters, a key, the batch and
  the seed;
- `adamw.lr`;
- `adamw.step` on the parameters, AdamW's two moments and the gradient.

The loop holds the parameters and the moments across them. `--step`
prepares the file's training run as `presk train` does, at `--dim`'s
batch, and records the three calls, in training mode, under `--remat
store` and `recompute`, side by side. It prints:
- the parameters, the moments (twice the gradient), the gradient (the
  parameters' shapes in the step's type), the batch, key and seed;
- what is **kept for the backward**: what the gradient's forward sweep
  made that is still live where its reverse sweep starts, at the first
  operation that reads the seed cotangent. That is the term `--remat`
  moves;
- each call's peak, and with what the loop holds beside it (its phase);
- the step's peak, the largest phase, and its terms, which sum to it.

The compiler's AdamW is not in place: at its last launch the new
parameters and moments are live beside the old ones and the gradient,
seven times the parameters' bytes in `f64`.

```sh
presk memory corpus/nlp/05_gpt.psk --entry GPT.apply --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16 --depth 3
presk memory experiments/gpt_grad.psk --entry loss_grad --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16 --remat recompute
presk memory corpus/nlp/05_gpt.psk --step --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16
presk memory corpus/nlp/05_gpt.psk --entry GPT.apply --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16 --symbolic
```

## `presk autotune FILE.psk --entry ENTRY --out DIR …`

```text
presk autotune FILE.psk --entry ENTRY [--dim N=v …] [--const N=v …] [--realize R]
               [--space full|small] [--repeat 7] [--profile P.json] [--nest NEST …] [--interp]
               [--top N --random N] [--seed S] [--shard K/N] --out DIR
```

The exhaustive side of the RA4 test (plan 0.7.0 §3.6). Runs the entry once
natively, under its default layout, on inputs synthesized from `--seed` as
`presk run` synthesizes them, to capture each nest's first launch at every
shape with its input buffers, the values a point is timed on. Then, nest by
nest and point by point, it lowers
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
records the model's prediction. `--nest`, as `presk plan`'s, restricts the
sweep to the nests it names, every shape each is launched at: since 0.10.0
step 2 it takes several, so that one sweep holds a gradient's derivatives
of `dot` (T3, below).

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
and `%` replaced by `_`, holding every shape the nest is launched at (until
0.10.0 step 2 a nest at several shapes kept only its last shape's lines, a
case no sweep until then had: every archived sweep holds all its points),
a JSON line per point (`entry`, `config` — the
constants then the dims, `B=16,T=256` —, `nest`, `op`, the primitive, `ein`,
`dot`, `softmax`, … or `fused`, `shape`, `dtype`, `layout`, `key`,
`baseline`, `timed`, the `seconds` of every timed launch, `median`, `mad`,
`predicted`, `identical`, `interp_checked`, and per kernel — a pack's copy
first — the model's `group`, its inputs `cost` (since 0.8.0 step 2 with
`loads` and `spill`), the `terms` of its prediction, `λ`, spawn, the work,
the loads when the model has the term, and the three traffic bounds, and the
`bound` that set it), and
`sweep.json` (the entry, the configuration, the dims, the nests `--nest`
named since 0.10.0 step 2, `null` for every nest, the space, the sample
and the shard if any, the number of points and of baselines, the time spent
measuring, costing, capturing and checking
on the interpreter, the wall time, the machine description, `u`, and
`not_identical`). Since 0.8.0 step 1 it also records
`static_capture_seconds`: how long `presk plan`'s capture of shapes takes on
the same entry, timed after the wall time and not part of it. That capture
and the costing are statement N's static side (D65). Pin the process to a set of like cores (`taskset -c`):
the machine description is read for the first CPU it may run on.

## `presk reprice FILE.psk --entry ENTRY --day DAY --profile P.json --out DIR`

```text
presk reprice FILE.psk --entry ENTRY --day DAY --profile P.json --out DIR [--seed S]
```

Prices the points of a recorded day again under another profile, **without
measuring anything** (0.8.0 step 1): a candidate model is judged on
measurements already taken in the time it takes to cost them. For each sweep
of `ENTRY` under `DAY` — a directory holding the `sweep.json` of a `presk
autotune`; the day's sweeps of other entries are left — the entry's shapes
are captured once, as `presk plan` captures them, at the constants, dims and
realization the sweep records, and every
recorded point is lowered again from its `layout`, against the worker count
the sweep ran with (the `machine` of `sweep.json`), and costed under the
profile's layout model, on the profile's machine description when it has one
and the sweep's otherwise. Its line is written to the same place under `DIR`
with `predicted` and the per-kernel records (`cost`, `group`, `terms`,
`bound`) replaced; every other field — the `seconds`, the `median`, the
`mad`, `identical` — is copied as written, character for character.
`sweep.json` is copied with `u` replaced by the profile's and a `repriced`
record added (the profile, its `u`, the points, the time the capture and the
costing took), and the day's `day.txt` as it is; `presk plan`'s files are not
copied. A point whose nest the entry does not launch, or whose layout no
longer lowers, stops the command, and so does a `DIR` that is `DAY` itself.
`presk rank` then ranks the re-priced days as it ranks measured ones.

Under the profile a day was priced with, every line comes back as it was.
`make ra4-dev` relies on it for 0.8.0's **development set**: 0.7.0's
measurements, archived read-only in `~/presk-data/ra4-0.7.0` (the
Makefile's `RA4_DEV`, D63), re-priced under `RA4_DEV_PROFILE` and
`RA4_DEV_PROFILE_20` (0.7.0's own by default) into `out/ra4-dev/out/…` and
ranked from `out/ra4-dev`, so that each report quotes its days as 0.7.0's
did: `out/ra4-dev/ra4.md` and `ra4-20.md`, in about half a minute. Under
0.7.0's profiles it also checks that each of the 52 142 re-priced lines is
the archived one and each report the one `docs/` holds, byte for byte, and
fails otherwise. An empty `RA4_DEV_PROFILE_20` leaves the twenty-core day
out, and the check covers the ten-core days alone (0.8.0 step 4, D68):
`make ra4-dev RA4_DEV_PROFILE=P RA4_DEV_PROFILE_20=`. Under the frozen
model's profile (`RA4_FROZEN`, archived read-only in
`~/presk-data/ra4-model-0.8.0`, D71) it checks that the report is
`docs/ra4-dev-model-0.8.0.md`, byte for byte: the check a defect fixed after
the freeze must pass (plan 0.8.0 §3.4). `make ra4-dev-frozen` runs it.

`make ra4-ablation` (0.8.0 step 4) prices the ten-core days under every form
of the model, three load terms under three combinations. Each form is
refitted by `presk refit` on the definitive calibration (`RA4_CALIBRATION`,
archived read-only, D68). 0.7.0's own profile is priced beside them. It
writes `out/ra4-ablation/ablation.md`, which tabulates statement M's nests,
O, the per-class table and the hybrid chooser of statement Q, in about
3½ minutes. `FORMS` restricts it, for example `FORMS=group-pnorm`. D69
records the result. A form `L-C+R` names rules of the planned cost too
(0.8.0 step 4, 4c): the calibration is re-costed under them first (`presk
refit --recost experiments/matmul.psk --rules R`), for example
`FORMS="group-pnorm group-pnorm+lines group-pnorm+lines,overhead"`.

```sh
presk reprice corpus/nlp/05_gpt.psk --entry SelfAttention.apply --day ~/presk-data/ra4-0.7.0/out/ra4/day1 \
     --profile out/profile.json --out out/dev/day1
```

## `presk rank DAY1 DAY2 --out FILE.md`

```text
presk rank DAY1 DAY2 --out FILE.md [--check DIR] [--frozen TAG] [--entry E] [--at CONFIG] [--type T]
           [--gemm E] [--gemm-at CONFIG] [--statement-q]
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
plan 0.7.0 §3.7 follow, then plan 0.8.0's Q, each measured and decided by
the tool, not by hand:

- **L** holds when every point of day one was checked on the interpreter,
  in the days or in the check run apart that `--check` reads, and no output
  differed anywhere.
- **M** holds when each `ein` of `--entry` at `--at` in `--type`
  (`SelfAttention.apply`, `B=16,T=256`, `f32`) orders more than 90 % of at
  least 10³ distinguished pairs right. With fewer pairs it is inconclusive
  (plan §7). So it is when that configuration was swept on a sample (plan
  0.8.0 §3.5, D72), whatever its pairs say. M's and O's cells are then
  marked `(sample)`, since O's best point is the sample's.
- **N** holds when every configuration of that entry swept whole is chosen
  statically at least 10³× faster than it was swept, on each day. The static
  side is the capture of shapes plus the costing, when the sweep records it
  (0.8.0 step 1, D65). The ratio with the capture that runs the entry, which
  0.7.0 measured, is printed beside it.
- **O** holds when the model's pick is within 1.25× of the best measured on
  every contraction nest (`ein` or `dot`, and since 0.10.0 step 2 the
  derivatives of `dot`, `dot_vjp_l` and `dot_vjp_r`) at `--at`. M stays the
  `ein`s'.
- **P** holds when the best point of `--gemm` at `--gemm-at` (`gemm`,
  `N=1024`) is at least 3× faster than 0.6.0's point.
- **Q**, plan 0.8.0's (§3.6), holds when on every contraction nest of every
  configuration swept whole, measuring only the *candidates*, the points
  predicted within `1 + u` of the model's best, finds a point within 1.05×
  of the best measured (the top-k regret), at no more than 10 % of the
  nest's exhaustive sweep on each day. The cost is question 72's: statement
  N's static side, the whole configuration's, charged to each nest, plus the
  candidates' measured time (0.8.0 step 7, D74). A point's measured time is
  its launches, the warm-up and the timed ones, at its median, plus its time
  outside them, compiling it and comparing its bits. A sweep records that
  time only as a whole, so it is estimated as the sweep's time less its
  points' launches, divided by its points, as `experiments/ra4_dry.py`
  estimates it (D72). Beside it is D71's count: the candidates' share of the
  nest's timed launches. A table per contraction nest follows the
  statements. Q is decided on days that record the capture of shapes (0.8.0
  step 1, D65). For earlier days, 0.7.0's, it is decided only with
  `--statement-q`, which costs their static side with the capture that runs
  the entry, and their reports are otherwise 0.7.0's. Its contraction nests
  are O's.

A nest the day holds at several shapes in one configuration and type, as a
gradient's derivatives of a linear layer's product, is named with its
operands in the statements and the tables that have no operands column
(`` `Linear.apply.vjp/%11` at [16, 256, 384] [1536, 384] ``; 0.10.0 step 2;
`Linear.apply.bwd/%11` since step 4).
A nest at one shape is named as before, so every earlier report comes back
byte for byte.

Then come the counts per decision class (`tile`, `kblock`, `order`,
`lanes`, `split`, `pack`, `passes`, `form`), per nest and class, and the
**decision cost** per sweep: the exhaustive sweep's time (all of `presk
autotune` but the interpreter's check) against the static choice's (the
capture and the costing of the same points, which is what `presk plan`
does), a sampled sweep marked. With a sweep recorded since 0.8.0 step 1 the
capture is of shapes, and two more columns give the static side and the
ratio with the capture that runs the entry. A sweep recorded before is costed
with that capture, and its table is 0.7.0's. A day's directory may hold one sweep or one
per configuration in subdirectories, as `experiments/ra4.sh` writes them.
Given one directory twice, `presk rank` applies the rule to one sweep and
says so. The report is also printed.

`--frozen TAG` is plan 0.8.0's freeze (§3.4): before reading anything it
refuses, with exit code 1, a day whose `day.txt` names a commit that does
not descend from `TAG` in the repository `presk rank` runs in, or a commit
marked `-dirty` (`experiments/ra4.sh` marks one when the working tree
differed from it), or no commit. Every run into a day adds its commit to
`day.txt`, and every one is checked.

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

0.8.0's fresh sweep **F2** (plan 0.8.0 §3.5, step 5, D72) measures 0.7.0's
shapes again on the ten A725 cores (`RA4_CORES_SMALL`, `0-4,10-14`), under
the model frozen at the tag `ra4-model-0.8.0`. It uses these targets, in
this order, on a quiet machine, with nothing running on the X925 cores
either, since they share the L3:

- `make calibrate-ra4-small` is `make calibrate-ra4` pinned to the A725
  cores, with the same space, the same exception at 2048 and the frozen
  form and rules. It writes a profile of its own, `RA4_PROFILE_SMALL`
  (`out/profile-a725.json`), and refuses to be pointed at `out/profile.json`.
- `make ra4-f2-dry` sizes a day before the first night:
  `experiments/ra4_dry.sh`.
  1. It prices every configuration with `presk plan --cost`.
  2. It sweeps the nest the model predicts longest, pinned and timed as a
     night times it.
  3. It scales every point's prediction by that nest's measured time over
     its predicted, adds its time a point outside the kernels, and writes the
     estimate to `out/ra4-f2/dry/dry.md`. A configuration too slow to sweep
     whole is swept on a sample, and its M is reported inconclusive.
- `make ra4-f2` is day one: the course GPT's attention at `D` = 384, `H` =
  6, `T` = 256, `B` = 1 and 16, `f32` and `f64`, the whole space, priced by
  the A725 profile, into `out/ra4-f2/day1`.
- `make ra4-f2-check` is statement L on the kernel interpreter, sharded over
  every core like `ra4-check`, never beside a timed run.
- `make ra4-f2-day2` is day two, on another calendar day unless
  `RA4_SAME_DATE=1` (below), into `out/ra4-f2/day2`. It ends with `make ra4-f2-rank`: `presk rank --frozen
  ra4-model-0.8.0` of the two days, with the check, into `out/ra4-f2.md`.

Before measuring, `experiments/ra4_fresh.sh` refuses a day that would not
count. It refuses a day directory that exists, since a day is one run. It
refuses a working tree that differs from its commit, and a commit that does
not descend from the tag, which `presk rank --frozen` would refuse after the
night. And it refuses day two on the calendar date day one started on,
unless `RA4_SAME_DATE=1` accepts it: F2's day two ran on day one's date, by
the user's decision (D72).
Every target refuses to start beside another `presk` process.
`RA4_TRIAL=1` tries the targets on a tiny configuration, with other
`RA4_F2_DIR`, `RA4_PROFILE_SMALL`, `RA4_F2_RUN` and `RA4_F2_SHAPES`. In a
trial, none of these checks applies but the first, and the ranking is made
without `--frozen`.

```sh
make ra4-f2-day2 RA4_TRIAL=1 RA4_F2_DIR=out/ra4-f2-trial RA4_PROFILE_SMALL=out/ra4-f2-trial/profile-a725.json \
     RA4_F2_SHAPES="B=1,T=256:2 B=16,T=256:8" RA4_F2_RUN='FILE=tutorial/03_attention.psk ENTRY=attend TYPES="f32 f64" SPACE=small'
```

The fresh sweep **F1** (plan 0.8.0 §3.5, step 6, D73) measures new shapes
on the ten X925 cores (`RA4_CORES`). It is priced by the frozen model's
profile itself: `RA4_F1_PROFILE`, by default the archived
`~/presk-data/ra4-model-0.8.0/profile.json`, since the X925 is not
calibrated again. It sweeps the course GPT's attention in `f32`, over the
whole space, at two configurations, both in one day. Each is written
`constants:shape`:

- `RA4_F1A` is F1a, BERT-large's attention: `D=1024,H=16:B=1,T=384`.
- `RA4_F1B` is F1b, a head of 128: `D=1024,H=8:B=2,T=512`.

The targets are F2's, in the same order:

- `make ra4-f1-dry` runs `experiments/ra4_dry.sh` once per configuration,
  each timing its own longest-predicted nest, into `out/ra4-f1/dry/f1a` and
  `f1b`. `ra4_dry.py day` then adds the two into a day in
  `out/ra4-f1/dry/dry.md`. A configuration is swept whole while the day fits
  in `RA4_F1_BUDGET` hours (8), the quickest first. The others are marked to
  be swept on a sample.
- `make ra4-f1` and `make ra4-f1-day2` are the two days, into
  `out/ra4-f1/day1` and `day2`, with the checks before a fresh day
  (`RA4_SAME_DATE=1` accepts day two on day one's date).
- `make ra4-f1-check` is statement L, with 80 shards of F1b
  (`RA4_F1B_SHARDS`) and then 20 of F1a (`RA4_F1A_SHARDS`), over every core.
- `make ra4-f1-rank`, which ends day two, ranks the two days twice with
  `presk rank --frozen`. `presk rank` decides M and O at one configuration
  (`--at`), so the first report is at F1a, in `out/ra4-f1a.md`, and the
  second at F1b, in `out/ra4-f1b.md`. L and N are the same in both.

A configuration too slow to sweep whole is swept on a sample:
`RA4_F1A_SAMPLE` or `RA4_F1B_SAMPLE`, passed to `presk autotune`
(`--top 8 --random 32`, say). It is set by a commit before day one, so that
both days and the check take the same points, and `presk rank` leaves that
configuration's M inconclusive. A trial on a tiny configuration:

```sh
make ra4-f1-day2 RA4_TRIAL=1 RA4_F1_DIR=out/ra4-f1-trial RA4_F1A=D=32,H=2:B=1,T=16 RA4_F1B=D=32,H=4:B=2,T=16 \
     RA4_F1_RUN='FILE=corpus/nlp/05_gpt.psk ENTRY=SelfAttention.apply TYPES=f32 SPACE=small'
```

Both sweeps were measured on 2026-09-25, each day's two runs on one date,
and archived read-only in `~/presk-data/ra4-f2-0.8.0` and
`~/presk-data/ra4-f1-0.8.0` (D72, D73). 0.8.0 step 7 ranked them from the
presk tree, over copies of the archived days checked against their
`SHA256SUMS`, with every statement (D74; `experiments` chapter §8):

```sh
presk rank out/ra4-f2/day1 out/ra4-f2/day2 --frozen ra4-model-0.8.0 --check out/ra4-f2/check \
     --out docs/ra4-f2-2026-09-25.md
presk rank out/ra4-f1/day1 out/ra4-f1/day2 --frozen ra4-model-0.8.0 --check out/ra4-f1/check \
     --at D=1024,H=16,B=1,T=384 --out docs/ra4-f1a-2026-09-25.md
presk rank out/ra4-f1/day1 out/ra4-f1/day2 --frozen ra4-model-0.8.0 --check out/ra4-f1/check \
     --at D=1024,H=8,B=2,T=512 --out docs/ra4-f1b-2026-09-25.md
```

**T3** (plan 0.10.0 §3.5, step 2, D90) is RA4's third fresh test: the
derivatives of `dot`, contraction nests since 0.10.0 step 1, which the
frozen model has never seen. It sweeps the course GPT's gradient
(`experiments/gpt_grad.psk`, entry `loss_grad`) at `B = 16`, `T = 256`, in
`f32`, over the whole space, on the ten X925 cores, priced by the frozen
model's profile itself (`RA4_T3_PROFILE`, by default `RA4_FROZEN`), neither
changed nor refitted. A day is one sweep of the four nests `RA4_T3_NESTS`
names: `Linear.apply.vjp/%11` (`dot_vjp_l`) and `%12` (`dot_vjp_r`), each at
the three shapes of the block's products, and the head's `GPT.apply.vjp/%28`
and `%29`, eight nests and shapes, 11 028 points (the days of D90, archived,
name them so; since 0.10.0 step 4 the sweeps are the `.bwd`s, `RA4_T3_NESTS`
names `Linear.apply.bwd/%11`, …, and re-pricing the archived days takes
`--remat recompute`). The targets are F2's:

- `make ra4-t3-dry` prices the four nests and sweeps the one the model
  predicts longest, at its three shapes, into `out/ra4-t3/dry`, then sizes a
  day into `out/ra4-t3/dry/day.md` against `RA4_T3_BUDGET` hours (8).
- `make ra4-t3` and `make ra4-t3-day2` are the two days, into
  `out/ra4-t3/day1` and `day2`, with the checks before a fresh day.
- `make ra4-t3-check` is statement L on the kernel interpreter, in
  `RA4_T3_SHARDS` shards (80) over every core, never beside a timed run.
- `make ra4-t3-rank`, which ends day two, is `presk rank --frozen
  ra4-model-0.8.0 --entry loss_grad --at B=16,T=256 --type f32` into
  `out/ra4-t3.md`: O and Q on the eight derivative nests, L with the check,
  N on the configuration. M is the `ein`s', and the sweep holds none.

A day too long for the budget would be swept on a sample, `RA4_T3_SAMPLE`,
which leaves Q undecided. A trial on a tiny configuration:

```sh
make ra4-t3-day2 RA4_TRIAL=1 RA4_T3_DIR=out/ra4-t3-trial RA4_T3_CONSTS="D=32 H=2" RA4_T3_SHAPE=B=2,T=16 \
     RA4_T3_SPACE=small
```

T3 was measured on 2026-09-29 in this order: the dry run (1 h 50 min, a day
estimated at 2 h 51 min, the space whole), day one (3 h 05 min), the check
(80 shards, 7 h 06 min), and day two on day one's date (`RA4_SAME_DATE=1`,
3 h 05 min), then ranked. Each piece is archived read-only in
`~/presk-data/ra4-t3-0.10.0` (D90). The report, `out/ra4-t3.md`, is kept as
`docs/ra4-t3-2026-09-29.md`: O holds, and Q fails on the head's two nests
(`experiments` chapter, §8's addendum).

```sh
presk rank out/ra4-t3/day1 out/ra4-t3/day2 --frozen ra4-model-0.8.0 --check out/ra4-t3/check \
     --entry loss_grad --at B=16,T=256 --type f32 --out docs/ra4-t3-2026-09-29.md
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
