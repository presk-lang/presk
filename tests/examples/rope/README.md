# `rope/` — Rotary position embeddings

The position encoding of the Llama models, written in the language: each
pair of channels of a query or key head is rotated by an angle proportional
to the position, so that the product of a query at `t` and a key at `t'`
depends on `t − t'` alone (Su et al., RoFormer, 2021).

## What it illustrates

- The **slice selector** `p[pair: 0..1]` (§3.2): half-open, it keeps the axis
  `pair` at length 1, where the selector `p[pair: 0]` drops it. The two halves
  of a pair are thus tensors of the same rank as the whole.
- **`concat(axis pair, a, b)`** puts the two rotated halves back along that
  axis, of length 1 + 1. Both operands carry the axis by name; there is no
  implicit unit axis.
- The angles as an expression: `real(positions(…))`, `exp`, `log`, and an
  outer product written as an `ein` that sums nothing.
- **Llama 3's frequency scaling** (`rope_scaling` of type `llama3`) with
  `clamp`, so that Llama 3.1 and 3.2 need no table of frequencies loaded with
  their weights.
- Hugging Face's pairing (`r` with `r + HD/2`, its `rotate_half`). Meta's
  first code paired `2r` with `2r + 1`; that is one `split` apart.
- An attention whose heads lie before the positions, so that `rope` sees the
  axis `seq` just before `hd`; the projections are `ein`s and nothing is
  transposed.

The denotation of a slice and of a `concat`, their derivatives (a slice's
adjoint places the cotangent into zeros, a `concat`'s cuts it into two
slices) and the convention of `clamp`'s derivative at its bounds are the
reference's chapters 2, 3 and 5.

## Files

- [`rope.psk`](rope.psk) — `frequencies`, `llama3_scaling`, `angles`, `rope`,
  the module `Attention`, and the entries `attention2` (base 10 000) and
  `attention3` (Llama 3.2's base 500 000, scaled).

## Commands

```sh
presk check rope.psk
presk run rope.psk --entry attention2 --dim B=2 --dim T=8
presk run rope.psk --entry attention3 --dim B=2 --dim T=8
```
