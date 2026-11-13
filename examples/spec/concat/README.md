# `concat/` — Slices and concatenation along an axis

The two constructs that build a tensor along an axis out of computed parts,
and the three elementwise functions that cut one: a slice keeps its axis, a
`concat` joins two tensors along it, and `min`, `max` and `clamp` compare
element by element.

## What it illustrates

- **Slices of an index tensor** (§3.2, §3.5): the inputs and the targets of a
  next-token loss are the first `T` and the last `T` tokens of a window of
  `T + 1`. The bounds of a slice are dimensions, and the facts must place
  them inside the axis for every value of `T`; the checker decides it.
- **`concat(axis out, x, y)`** on the weights of three projections, whose
  output axis has the length `D + D + D`, and three slices of the product:
  `[x·Wq | x·Wk | x·Wv] = x · [Wq | Wk | Wv]`, an identity the compiler may
  rewrite by.
- **`min`, `max` and `clamp`**, of class `C⁰⁺` like `relu`: `hardtanh` is
  `clamp(x, −1, 1)`, a leaky rectifier is `max(x, 0.01·x)`. They broadcast as
  the arithmetic operators do.
- **A gradient through all of them**: `concat`'s adjoint is two slices of the
  cotangent, a slice's places it into zeros, and at a tie `max` and `min`
  pass the cotangent to their second operand, which makes `max(x, 0)`'s
  derivative `relu`'s (the reference's chapter 3).

## Files

- [`concat.psk`](concat.psk) — the window's slices, the module `QKV`,
  `hardtanh`, `leaky`, the module `Toy`, its `loss` and `loss_grad`.

## Commands

```sh
presk check concat.psk
presk run concat.psk --entry loss_grad --dim B=2 --dim T=8
```
