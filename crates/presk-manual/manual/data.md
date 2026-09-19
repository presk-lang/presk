# Data files: `.psk.d`

Training data, checkpoints and weights live in `.psk.d` files: **a sequence
of samples of one Presk type**. The format is defined by this project (the
specification names the extension without defining it; question 28 of
`docs/spec-questions.md`) and read and written by the crate `presk-data`
and the `presk data` command.

## The container

```text
magic      "PSKD"                      4 bytes
version    u16 = 1                     little-endian
type_len   u32, then the type, UTF-8   the sample type in Presk syntax
count      u64                         number of samples
data                                   the samples, back to back
```

A sample is the concatenation of the leaves of its type, in order: for a
tuple, its elements left to right; for a parameter tree, its fields in
declaration order. Each leaf is a tensor in row-major order:

| Leaf type | Element on disk |
|---|---|
| `idx[N][…]` | the smallest of `u8`, `u16`, `u32`, `u64` that holds `N − 1`, little-endian |
| `ℝ[…]` | `f64`, little-endian |

Two rules follow from the language's semantics:

- **Reals are injected exactly.** An `f64` is the dyadic rational it encodes;
  a `.psk.d` denotes exactly the numbers it holds, and the interpreter reads
  them into ℝ at its working precision without further rounding (the same
  rule as for hyperparameters). Weights written by the interpreter at 256
  bits are rounded to `f64` on the way out (question 31).
- **Indices are bounded.** A value at or above the bound of its `idx[N]` is
  refused when writing (`E9004`) and when packing text.

**Sample types** are closed data types: `ℝ[…]`, `idx[N][…]` and tuples of
them, with every dimension a literal (`(idx[64][seq: 8], idx[64][seq: 8])`,
`ℝ[model: 4]`, `ℝ[]`). Axis names are kept for documentation; readers check
lengths and bounds, not names. A `params M` type is also a sample type,
which is how **checkpoints and weights** are stored: one sample of type
`params MiniGPT`. Its leaf layout is not derivable from the type text alone
(it needs the module's declarations), so such a file is opened from the
program (`presk train`, later `inference serve`); `presk data info` prints
its header and says so.

## `presk data`

```text
presk data pack --type TYPE INPUT.txt -o OUT.psk.d
presk data lm-windows --vocab V --seq T [--stride S] TOKENS.txt -o OUT.psk.d
presk data info FILE.psk.d [--samples N]
presk data export FILE.psk.d -o OUT.json
```

- **`pack`** reads one sample per line, values separated by whitespace in
  leaf order (indices as integers, reals as decimals), and writes them with
  the given type. Empty lines and lines starting with `--` are skipped.
- **`lm-windows`** turns a whitespace-separated stream of tokens into
  language-modelling samples: window `k` is `(tokens[k·S .. k·S + T],
  tokens[k·S + 1 .. k·S + T + 1])`, of type `(idx[V][seq: T], idx[V][seq:
  T])`. The stride defaults to `T` (non-overlapping windows); `--stride 1`
  gives every window. The shift between input and target is done here, once,
  rather than by the language's `stream` (question 29).
- **`info`** prints the header and the first samples.
- **`export`** writes the samples as JSON (tensors as nested lists, tuples as
  lists, trees as objects), for the PyTorch side and for tests.

```sh
presk data lm-windows --vocab 64 --seq 8 tests/data/tokens.txt -o out/tokens.psk.d
presk data info out/tokens.psk.d --samples 2
```

```text
presk data: wrote out/tokens.psk.d (24 windows of type (idx[64][seq: 8], idx[64][seq: 8]) from 200 tokens)
type    (idx[64][seq: 8], idx[64][seq: 8])
count   24
version 1
sample  16 bytes, 2 leaves
[0] ([41, 19, 50, 6, 9, 12, 46, 7], [19, 50, 6, 9, 12, 46, 7, 27])
[1] ([27, 4, 11, 55, 53, 8, 30, 11], [4, 11, 55, 53, 8, 30, 11, 54])
```

## The pipeline

`data = stream "path" |> shuffle(seed = s) |> batch(B = n) [|> take(n = k)]` of
a `training run` is evaluated over the samples of the file:

- `stream "path"`: the samples in file order; the path is relative to the
  `.psk` file.
- `shuffle(seed = s)`: a permutation of the samples, redrawn every epoch
  from a ChaCha20 key derived from `s` and the epoch number; the same on
  every machine.
- `batch(B = n)`: consecutive groups of `n` samples, stacked under a new
  leading axis named `batch`; an incomplete last group is dropped. This
  binds the dimension `B` of the objective.
- `take(n = k)`: only the first `k` batches of each epoch.

The crate implements the pipeline over sample indices (`Pipeline::batches`)
so that the interpreter, the training loop and the export agree on which
samples form which batch. `presk check` types the pipeline against the
objective (the `language` chapter, "Configuration blocks"): the file's
sample type must be the objective's batch type without its `batch` axis,
and `batch(B = n)` binds `B`. Running the pipeline is step 5 of the plan.
`take` is written `take(n = N)`.

## Errors (`E9xxx`)

| Code | Cause |
|---|---|
| `E9001` | the type is not a closed data type: a symbolic or missing dimension, an open prefix `...`, `idx[0]`, a named type, or `params M` opened without the program |
| `E9002` | not a `.psk.d` file: bad magic, unsupported version, unreadable path |
| `E9003` | truncated file, or a sample index beyond the count; I/O errors |
| `E9004` | a value out of its bound, or a text value that is not an index or a real, or too few or too many values on a line |
| `E9005` | a sample whose structure does not match the layout (leaf count, element count, reals for indices) |

Not in 0.2.0 step 1: compression, several sample types per file, an index for
variable-size samples, MPFR-precision leaves.
