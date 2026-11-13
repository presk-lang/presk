# Memory: what a program needs, counted from the program

Version 0.15.0 (plan `presk-specs/implementation/presk-plan-0.15.0.md`,
D147–D152) counts what a program needs in memory from the program at its
dimensions and its realization's types. It never measures a process's
resident size. The counts are given four ways:
- `presk memory` prints them for one call of an entry;
- `presk memory --step` for a training step;
- `presk memory --symbolic` as polynomials in the dimensions;
- the page draws them on the Network tab, and `presk explain` puts them in
  its section 13 (the `explain` chapter).

## What is counted

The entry is recorded as the C export records it (`export_c`): the native
host's operations, in order, on buffers that have an identity. The
recording runs on an executor that computes nothing, so it costs the
host's time alone.

- **A buffer's life** runs from its first write to its last read, both
  included, so that an operation's inputs and its output are live
  together. A view reads its buffer.
- **The inputs** (the parameters among them) are held by the caller and
  counted apart. The result's buffers live to the end.
- **Bytes** are a buffer's elements times its stored type's bytes.
  Integer weights count their integers (two `int4`s a byte) and their
  `f32` scales. Nothing is rounded.
- **The peak** is the largest live set, at the first operation that
  reaches it. **What is needed** is the inputs and the peak.
- **Each call** is counted by its path, as the network diagram's boxes are
  (a fold's iterations together, `blocks ×6`):
  - its *share* is what it made that is live at the peak; the shares and
    the entry's own sum to the peak;
  - its *own peak* is the most what it made weighs while it runs.

This is what any executor of that launch sequence holds if it frees each
buffer after its last read and keeps buffers whole. It leaves out:
- the allocator's reserves and the process's own footprint;
- the exact reference's memory, a different executor;
- a KV cache: the serve loop has none.

## The C program's arena beside it

The program `presk export --target c` writes has an arena of frames
(D80): each function holds what it allocates until it returns. Its size,
which the export prints, is at least the peak.
- `presk memory` prints it beside the peak, computed by the export's own
  code on the same recording, not recomputed.
- On the course GPT at `D = 64` the arena is 1.3 to 2.2 times the peak
  (2.5 until 0.16.0, whose inference program has no dropout mask, D160);
  on the decoder block at the 7B width, 3.06 times.
- Placing each buffer at an offset by its life (the largest first, each at
  the lowest offset where it fits beside those live with it) takes the
  peak, rounded to 64 bytes. The hidden `presk export --target c
  --straight-line --placed` writes that program.
- `make c-conformance` checks, on every program it exports (285), that
  the placed program computes the native backend's bits and that `presk
  memory`'s arena is the export's. A buffer reused while still read would
  change the bits, so this is the check that the lifetimes are sound
  (statement M1). A mutation that lets a kernel's output share memory with
  its own input makes C1 fail on 13 of 14 entries.

A C program that placed its buffers this way would need about the peak.
That is a later version's: the gap measured here is its argument (the
user, 2026-10-18).

## `presk memory`

```sh
presk memory examples/corpus/nlp/05_gpt.psk --entry GPT.apply --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=16 --depth 3
```

```text
presk memory: `GPT.apply` at B = 2, T = 16 in f64: 86 operations, 79 launches
  parameters  903168 bytes: 112896 elements, 112896 in f64
  inputs      903424 bytes, the parameters among them
  peak        149504 bytes live, at operation 38 of 86, `Linear.apply/%5` in blocks.ff.up, iteration 0
  outputs     23552 bytes
  needs       1052928 bytes: the inputs and the peak
  C arena     1232256 bytes, 328832 beside the inputs, 2.20 times the peak: the frames of `presk export --target c`, each function's buffers held until it returns (2137792 released nothing)
  placed      149504 bytes beside the inputs: every buffer at an offset by its life, rounded to 64 (`--straight-line --placed`)
```

The peak of the course GPT's forward pass at `T = 16` is at the
feed-forward's up projection. Four buffers are live there:
- two of `B·T·4D` reals: the product, and the bias added to it;
- the residual stream;
- the causal mask.

Its calls' table shows `blocks ×2` holding 147 456 of the 149 504 bytes,
its `ff.up` 131 072. The flags are `presk export`'s (`--realize`,
`--plan`, `--remat`); the `cli` chapter has them all.

## A training step: `--step`

No entry is a training step. `presk train` makes one from three calls on
one backend:
- the objective's `.value_vjp`;
- `adamw.lr`;
- `adamw.step`.

The loop holds the parameters and AdamW's two moments across them. `presk
memory FILE --step` records those calls as `presk train` prepares them, at
`--dim`'s batch, under `--remat store` and `recompute`, side by side. It
prints:
- the parameters, the moments, the gradient, the batch;
- what is **kept for the backward**: what the forward sweep made that is
  still live where the reverse sweep starts. That is the term `--remat`
  moves;
- each call's phase, what it and the loop hold at its peak;
- the step's peak, whose terms sum to it (M2).

The course GPT at its course constants (`D = 384`, `H = 6`, `L = 6`), at
`B = 16` and `T = 256`, in `f64`:

| | `store` | `recompute` |
|---|---|---|
| the parameters | 86 532 096 | 86 532 096 |
| the moments | 173 064 192 | 173 064 192 |
| the gradient | 86 532 096 | 86 532 096 |
| kept for the backward | 3 238 627 392 | 3 047 424 |
| the gradient's phase, the step's peak | 3 762 854 984 | 920 254 480 |
| `adamw.step`'s phase | 605 724 712 | 605 724 712 |

Storing the forward pass's values for the backward costs 3.2 GB of the
3.8 GB the step needs. Recomputing them brings the step down to 0.92 GB,
4.1 times less. At `D = 64` and `T = 16` the same saving is 7 %.

The compiler's AdamW is not in place: at its last launch the new
parameters and moments are live beside the old ones and the gradient,
seven times the parameters in `f64`.

## As polynomials: `--symbolic`

Every axis of a buffer is a linear form in the dimensions (the `const`s
are numbers), so a buffer's bytes are a polynomial. `presk memory
--symbolic` interpolates each buffer's exactly, from recordings on a grid
of three values per dimension. The operations must be the same at every
point, so a dimension that counts a fold's iterations is refused. The
polynomials are checked at a point off the grid. From them follow:
- the inputs and the outputs;
- the live bytes at each operation;
- the peak, the largest of the live sets that no other dominates for
  dimensions of at least 1.

The printer of the denotation writes them.

```text
presk memory --symbolic: `GPT.apply` in f64, in B and T: each buffer's bytes interpolated on B ∈ {16, 32, 48}, T ∈ {256, 512, 768}, checked at B = 64, T = 1024
  inputs      8 · B · T + 3072 · T + 85745664
  outputs     736 · B · T
  peak        the largest of 3, none of which dominates another:
                48 · B · T² + 12288 · B · T + 8 · T², at operation 23 (`SelfAttention.apply/%13`)
                144 · B · T² + 6144 · B · T + 8 · T², at operation 29 (`SelfAttention.apply/%21 (a)`)
                27648 · B · T + 8 · T², at operation 43 (`Linear.apply/%5`)
```

At the course constants the attention's probabilities, `144·B·T²`, hold
the peak from `T = 150`, which recordings at `T = 149` and `150` confirm. At `D = 64` the feed-forward's `4 608·B·T +
8·T²` holds it until `T = 75`. M3 checks the polynomials against
recordings at three more assignments on 14 programs.

## On the page

The Network tab's **memory** button colours each box by its share of the
peak, each share in per cent on its header. A panel has:
- a slider for each dimension;
- a stored type for each parameter group;
- the numbers and the peak's formula;
- the live bytes over the operations as a curve.

The page counts these again from the polynomials, by the compiler's
arithmetic, and `#bench` checks it against the compiler's numbers (M4).
`presk dump --stage network --memory` colours the SVG and the TikZ the
same way. The `explain` chapter has the details.

## The decoder block at the 7B width

The Llama 2 decoder block at the 7B configuration's width (X5's
parameters, D106's realization), `B = 1`, `T = 384`:

| | bytes |
|---|---|
| the parameters (in `f32`) | 1 864 417 280 |
| the peak of live bytes, at the SwiGLU | 57 016 320 |
| what is needed | 1 921 439 744 |
| the C arena beside the inputs | 174 591 040 (3.06 times the peak) |

Its peak is the largest of four polynomials:
- the SwiGLU's `148 480·T·B`;
- three of the attention's, two of which grow as `T²·B`.

`presk memory --symbolic` takes about 30 s and 9 GB on it.
