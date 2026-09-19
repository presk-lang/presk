# The `presk` command line

```text
presk <COMMAND>

Commands:
  check   Type-check a program: shapes, regularity classes, effects
  run     Evaluate an entry point on the reference interpreter
  export  Export a program to another framework
  dump    Dump an intermediate representation (development aid)
```

`presk --help` and `presk <command> --help` print the full option lists.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success (warnings may have been printed) |
| 1 | diagnostics with at least one error were reported |
| 2 | not implemented in this version, or usage error (unreadable file, bad flag) |
| 3 | the program is valid but not executable in this version (reserved for `run` on a `training run`) |

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
| `--summary` | print, for every function of the file and its imports, the inferred type of its body and of each `let` binding, then (after `-- regularity and effects --`) the class of its result with respect to each parameter and the effects it uses | implemented |
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

Diagnostics (if any) are reported on standard error; the exit code is 1 when
there are errors, 0 otherwise.

## `presk run FILE.psk --entry ENTRY …`

Evaluates one function on the reference interpreter: the whole pipeline
(`check`, lowering, automatic differentiation) runs first, then the entry
point is called on synthesized inputs and its result printed.

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
| `--entry ENTRY` | a top-level function (`mse`), a module function (`MiniGPT.apply`), or a module function with the module's dimension arguments (`LayerNorm{8}.apply`). Required, except that a file with a `training run` and no `--entry` exits with 3 ("not executable in 0.1.0"); a file without one lists the candidate entries. |
| `--dim NAME=N` | binds a dimension of the entry (its type parameters, implicit dimensions, the file's `dim`s). A name that is not a dimension of the entry gives, in order, the axes of the open prefix `...` of its parameters: `--dim batch=3` makes `x : ℝ[..., model: D]` an `ℝ[batch: 3, model: 8]`. Every dimension must be bound, and the file's bounds (`dim T <= 1024`) are enforced (`E7001`). |
| `--const NAME=N` | overrides a `const` of the file *or of its imports* (`minigpt.psk`'s `V`, `D`, `H`, `L` from `train_minigpt.psk`); `static assert`s are re-checked. This is how the MiniGPT examples run at a size the interpreter can afford. |
| `--seed S` | seed of every random choice (default 0): synthesized inputs, initializers, `sample`. Same seed, same output, on every machine. |
| `--bits B` | MPFR precision of every intermediate, in bits (default 256). |
| `--digits D` | significant digits when printing (default 12). |
| `--json` | machine-readable output: entry, seed, bits, dimensions, inputs and result, reals as decimal strings at `--digits`. |
| `--verbose` | also print the synthesized inputs. |

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
error; 3 `training run` without `--entry`.

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
| `--mode readable` | the only mode implemented (`--mode graph`, `--precision=solved` of the specification: not in 0.1.0). |
| `-o DIR` | output directory (created); the generated files import each other by file stem, made a Python identifier (`01_linear.psk` → `_01_linear.py`). |
| `--shapes=check` | insert `assert x.shape == (…)` after every parameter and `let` with a closed shape. |
| `--stack=loop\|unroll` | `stack{L}(M)` as a `for` over an `nn.ModuleList` (default) or as `L` explicit calls. |
| `--conformance-entry ENTRY` | also run the reference interpreter on `ENTRY` (with `--dim`, `--const`, `--seed` as for `presk run`, at 256 bits) and write `conformance/inputs.json`, `conformance/expected.json` and `test_conformance.py`. |

**The conformance test** loads the interpreter's inputs and expected result,
builds the model or function (`MiniGPT(8)`, `MLP(4, 16)` — the dimension
values of the run), loads the parameter tree into the PyTorch parameters
(splitting stacked sub-modules over the `nn.ModuleList`), runs the forward
pass in `float64` and checks `torch.allclose(rtol=1e-9, atol=1e-12)`. It is
the "exact versus almost" experiment of the specification: the tolerance
measures the gap between the exact semantics and double precision. It runs
where PyTorch is installed: `make venv` creates `.venv` with PyTorch, and
`make conformance` exports the examples and runs their tests
(`mse_grad`, `LayerNorm{8}.apply`, `MiniGPT.apply` and the training `loss` pass). PyTorch is not a
dependency of the compiler.

**Exit codes**: 0 written; 1 diagnostics (a construct with no PyTorch form is
`E8001`); 2 usage.
