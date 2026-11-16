# Data files: `.psk.d`

Training data, checkpoints and weights live in `.psk.d` files: **a sequence
of samples of one Presk type**. The format is defined by this project (the
specification names the extension without defining it; question 28 of
`docs/spec-questions.md`) and read and written by the crate `presk-data`
and the `presk data` command.

## The container

```text
magic      "PSKD"                      4 bytes
version    u16 = 1, 2 or 3             little-endian
type_len   u32, then the type, UTF-8   the sample type in Presk syntax
count      u64                         number of samples
leaves     u32, versions 2 and 3       the leaves of the type
encodings  u8 a leaf, versions 2, 3    0 f64 (and every idx), 1 f32, 2 bf16;
                                       3 a format by its parameters (version 3),
                                       followed by six bytes
data                                   the samples, back to back
```

**Version 2** (0.17.0 step 7, D180) declares each real leaf's encoding in
the header and stores the leaf in it: `f64`, `f32`, or `bf16` as the upper
sixteen bits of an `f32`. Every `f32` and every `bf16` is a number of
`f64`, so a leaf is injected exactly whatever its encoding, as an `f64`
leaf is. A checkpoint published in `bf16` is thus stored at its own size:
Llama 3.2 1B is 2.5 GB where `f64` would take 10 GB. A writer writes
version 1 unless a leaf is narrower, so a file of `f64` leaves is the same
bytes as before; every reader takes both versions (`presk-data`, the C
export's `presk_rt.c`, the PyTorch export's `pskd.py`, the experiments'
scripts). `presk_data::write_encoded` writes one, refusing a number that
the leaf's encoding does not hold (`E9004`), and `presk data info` lists
the encodings. A version 1 file of the same leaves is read to the same
bits, whatever the version.

**Version 3** (0.18.0 step 1, D204) stores a leaf in any float format,
given by its parameters (the `precision` chapter, "Formats"): encoding 3,
followed in the header by the format's six bytes,

```text
p      u8    bits of significand
emin   i16   little-endian
emax   i16   little-endian
flags  u8    bit 0 subnormals; bits 1–2 the special values (0 IEEE, 1 NaN,
             2 none); bits 3–4 the overflow (0 an infinity, 1 NaN, 2
             saturation); bit 5 an underflow to +0
```

and each element of the leaf as its code in the fewest whole bytes that
hold the format's bits, little-endian: a byte for an 8-bit or a 4-bit
format, two bytes for binary16, three for `e8m15`. `f64`, `f32` and `bf16`
keep their codes, so a file is version 3 only when a leaf is in another
format; `write_encoded` writes one by its parameters, and `presk data
info` names it (`float(p=4,emin=-6,emax=8,specials=none,underflow=+0)`).
Every number of a format is a number of `f64`, so the leaf is read
exactly. A header that describes no format, or encoding 3 in a version 2
file, is refused (`E9002`). The C export's and the PyTorch export's
readers take versions 1 and 2 and refuse version 3 (`E9002`): no program
they run computes in another format before version 0.18.2.

A sample is the concatenation of the leaves of its type, in order: for a
tuple, its elements left to right; for a parameter tree, its fields in
declaration order. Each leaf is a tensor in row-major order:

| Leaf type | Element on disk |
|---|---|
| `idx[N][…]` | the smallest of `u8`, `u16`, `u32`, `u64` that holds `N − 1`, little-endian |
| `ℝ[…]` | `f64`, little-endian; in versions 2 and 3, the leaf's encoding |

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
`ℝ[model: 4]`, `ℝ[]`); a mask is stored as indices below 2, false 0 and
true 1, and a type that writes it `bool[…]` reads as `idx[2][…]` (0.10.0
step 7b, D99). Axis names are kept for documentation; readers check
lengths and bounds, not names. A `params M` type is also a sample type,
which is how **checkpoints and weights** are stored: one sample of type
`params MiniGPT`. Its leaf layout is not derivable from the type text alone
(it needs the module's declarations), so such a file is opened from the
program (`presk train`, later `inference serve`); `presk data info` prints
its header and says so.

## `presk data`

```text
presk data chars INPUT.txt -o VOCAB.json
presk data encode INPUT.txt --vocab VOCAB.json -o TOKENS.txt
presk data pack --type TYPE INPUT.txt -o OUT.psk.d
presk data lm-windows --vocab V --seq T [--stride S] TOKENS.txt -o OUT.psk.d
presk data info FILE.psk.d [--samples N]
presk data diff A.psk.d B.psk.d
presk data export FILE.psk.d -o OUT.json
```

- **`chars`** builds a character vocabulary from a UTF-8 text: every distinct
  character, sorted, one token each, written as a JSON array in which index
  `i` denotes token `i`. Characters, not bytes: an accent is one token.
- **`encode`** turns a text into a whitespace-separated stream of token
  indices against such a vocabulary, which `lm-windows` then cuts into
  windows. A character the vocabulary does not hold is an error naming it and
  its code point. The vocabulary is a `String` array so that the same format
  can later carry sub-word tokens; encoding one needs longest-match matching,
  which this version refuses rather than guesses (`presk data bpe` is
  planned, and the specification has not said whether a vocabulary belongs to
  a program or to its data — question 47).
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
- **`diff`** compares two files of the same type value for value, over their
  real leaves: `‖a − b‖₂`, and relative to `‖a‖₂` (0.6.0 step 7b). Two
  checkpoints compared this way say how far two training runs ended apart,
  without the program (tutorial step 16).
- **`export`** writes the samples as JSON (tensors as nested lists, tuples as
  lists, trees as objects), for the PyTorch side and for tests.

Text to a trainable dataset is the three-step chain, here on the 1.18 MB of
Code civil the NLP corpus uses (`corpus` directory):

```sh
presk data chars examples/corpus/nlp/data/civil.md -o examples/corpus/nlp/data/civil_vocab.json
presk data encode examples/corpus/nlp/data/civil.md --vocab examples/corpus/nlp/data/civil_vocab.json -o examples/corpus/nlp/data/civil_tokens.txt
presk data lm-windows --vocab 92 --seq 256 examples/corpus/nlp/data/civil_tokens.txt -o examples/corpus/nlp/data/civil_T256.psk.d
```

```text
presk data: wrote examples/corpus/nlp/data/civil_vocab.json (92 characters over 1182121 of text)
presk data: wrote examples/corpus/nlp/data/civil_tokens.txt (1182121 tokens over a vocabulary of 92)
presk data: wrote examples/corpus/nlp/data/civil_T256.psk.d (4617 windows of type (idx[92][seq: 256], idx[92][seq: 256]) from 1182121 tokens)
```

At the default stride the windows do not overlap, so a million tokens make a
2.4 MB file; `--stride 1` would make it 256 times larger, which is why the
stride is a choice and not a default of 1. Training never holds the file: the
reader seeks to each sample as the batch asks for it, so the resident memory
of `presk train` is that of the model and its batch, not of the dataset —
measured identical at 3 and at 12 steps on the corpus GPT.

```sh
presk data lm-windows --vocab 64 --seq 8 examples/tests/data/tokens.txt -o out/tokens.psk.d
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
