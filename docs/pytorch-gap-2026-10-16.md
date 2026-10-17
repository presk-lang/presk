> PyTorch's own gap to the exact model, and its flips (presk D130). The prompt was presk-project `docs/prompt-paper-results-2026-10-16.md`, run 3, at the user's request through the paper session. Version 0.12.0 (D128's binary), PyTorch 2.14.0+cpu, on falicon, 2026-10-16; nothing is timed. The tables come from `experiments/pytorch_gap.sh` (`make pytorch-gap`): `out/pytorch-gap/margins.md` and `gaps.md`. The outputs are in presk-project `data/pytorch-gap-0.12.0.tar.xz`.

# PyTorch's own gap to the exact model

The paper's claim is that a compiler which keeps the exact denotation can say how far a run is from the function it was written for. This run asks that question of the competitor's run: the same model, exported to PyTorch, run by PyTorch eager in `float32`, against the exact logits.

## What ran

**The model and the windows** are `make margins`' (D118, D125):
- the course GPT at D = 64, H = 2, L = 2, on `make curve`'s all-`f64` weights, `out/curve_train_f64/final.psk.d`;
- the 461 held-out windows of 256 tokens of the Code civil, 118 016 positions.

**The exact logits** were taken again by `make margins` (`experiments/margins.sh`, at 128 bits): the D = 64 outputs had been removed from `out/`.
- The run's `EXTRA` added two realizations: D124's curve realization at 10⁻³ (from presk-project `data/epsilon-curve-0.12.0.tar.xz`), and V6's realization solved on the logits (`out/v6-0.12.0/solved.json`).
- Its figures are D125's and V6's to every digit printed. D124's realization at 10⁻³ is 2.516·10⁻⁴ from the exact logits, with no flip and a bound of 51; V6's is 7.525·10⁻⁵, with no flip and a bound of 18.

**PyTorch's side.**
- *The module:* `presk export corpus/nlp/05_gpt.psk --target pytorch --mode readable --train` at the same constants (`_05_gpt.py`, `pskd.py`).
- *The run:* in eval mode under `torch.no_grad()`, on presk's checkpoint read by `pskd.read_params`, on the same 461 windows. `experiments/pytorch_gap.py` takes them chunk by chunk, as `make margins` took the exact logits.
- *What it records:* at every position, the largest gap of PyTorch's logits to the exact ones, PyTorch's argmax (the reference's: the greatest index among equal maxima), and the sum of the gaps. These are the files `make margins` writes for a realization of presk, so `experiments/margins.py report` counts PyTorch's flips and bound as it counts every realization's.
- *The three rows:*
  - `float32` on ten X925 cores pinned (5–9, 15–19), the baseline's configuration (D100);
  - `float32` on one core (cpu 15);
  - `float64` on ten, the control.

## The gaps and the flips

ε\*(10⁻³) from the exact margins is **4.67·10⁻⁴** (Clopper–Pearson, α = 0.05). The exact model's accuracy is 28 612 of 118 016, 0.242442.

| run | largest gap on the logits | relative | mean | flips | bound #{m < 2ε} | inside ε\*(10⁻³)? | accuracy |
|---|---|---|---|---|---|---|---|
| **PyTorch `float32`, ten threads** | **2.948·10⁻⁶** | 5.66·10⁻⁷ | 2.68·10⁻⁷ | **0** | 0 | yes, 158× inside | 0.242442 |
| PyTorch `float32`, one thread | 2.948·10⁻⁶ | 5.66·10⁻⁷ | 2.68·10⁻⁷ | 0 | 0 | yes | 0.242442 |
| PyTorch `float64`, ten threads | 5.878·10⁻¹⁵ | 1.13·10⁻¹⁵ | 5.08·10⁻¹⁶ | 0 | 0 | yes | 0.242442 |
| presk, all-`f32`, native | 3.115·10⁻⁶ | 5.98·10⁻⁷ | 2.96·10⁻⁷ | 0 | 0 | yes | 0.242442 |
| presk, all-`f64`, native | 5.894·10⁻¹⁵ | 1.13·10⁻¹⁵ | 5.57·10⁻¹⁶ | 0 | 0 | yes | 0.242442 |
| presk, solved at 10⁻³ on the loss (D124) | 2.516·10⁻⁴ | 4.83·10⁻⁵ | 1.14·10⁻⁵ | 0 | 51 | yes | 0.242442 |
| presk, solved at 4.67·10⁻⁴ on the logits (V6) | 7.525·10⁻⁵ | 1.44·10⁻⁵ | 3.03·10⁻⁶ | 0 | 18 | yes | 0.242442 |
| presk, all-`bf16`, native | 5.033·10⁻² | 9.66·10⁻³ | 6.07·10⁻³ | 480 | 13 661 | no | 0.242425 |

The distribution of the per-position largest gap (`gaps.md`):

| run | median | 90 % | 99 % | 99.9 % | largest |
|---|---|---|---|---|---|
| PyTorch `float32` | 1.08·10⁻⁶ | 1.54·10⁻⁶ | 1.99·10⁻⁶ | 2.36·10⁻⁶ | 2.95·10⁻⁶ |
| presk all-`f32` | 1.14·10⁻⁶ | 1.64·10⁻⁶ | 2.14·10⁻⁶ | 2.54·10⁻⁶ | 3.12·10⁻⁶ |
| presk solved on the logits | 5.69·10⁻⁶ | 8.66·10⁻⁶ | 1.35·10⁻⁵ | 2.44·10⁻⁵ | 7.53·10⁻⁵ |
| PyTorch `float64` | 2.04·10⁻¹⁵ | 2.92·10⁻¹⁵ | 3.78·10⁻¹⁵ | 4.46·10⁻¹⁵ | 5.88·10⁻¹⁵ |

**What it says.**
- **PyTorch's `float32` run is 2.95·10⁻⁶ from the function it was written for**, at worst over the 118 016 positions and their 92 logits each.
  - It flips no decision of the exact model.
  - It lies 158 times inside the budget ε\*(10⁻³) that the margins license.
  - It is as close as the compiler's own all-`f32` realization (3.12·10⁻⁶) and closer than the realizations the solver narrows further for traffic, which spend their budget.
- **The thread count changes nothing:** one thread and ten give the same gaps and the same decisions, bit for bit.
- **The `float64` control** is 5.9·10⁻¹⁵ from the exact logits, as presk's native `f64` is (5.9·10⁻¹⁵). The export computes the function the source denotes, and the `float32` gap is a rounding gap, not a difference of programs.

**What differs.**
- *PyTorch:* its products go through OpenBLAS (`BLAS_INFO=open` in this build), and its `einsum` products, softmax, GELU and layer norm through ATen's own kernels, vectorized for SVE128, each in its library's reduction order.
- *The compiler:* it sums in the order its realization declares (sequential here).

This run does not attribute the 6 % between 2.95·10⁻⁶ and 3.12·10⁻⁶ to any one of them.

**What PyTorch can and cannot produce.**
- *What it can:* its own `float64` run, set against its `float32` one, would print 2.95·10⁻⁶ too, to within 6·10⁻¹⁵, since its `float64` lies that close to the exact logits.
- *What it cannot:* know that its `float64` lies that close. The 5.9·10⁻¹⁵ is itself a measurement against the exact reference, which only this compiler keeps.

So the claim for the paper is this. "PyTorch's float32 run is 2.9·10⁻⁶ from the function it was written for" is a number PyTorch can approximate, but cannot certify, without an exact reference. It is not a number PyTorch cannot approach at all.

## To reproduce

```sh
EXTRA="d124-1e-3=@CURVE/curve_1e-3.json logits-4.67e-4=@out/v6-0.12.0/solved.json" make margins
make pytorch-gap      # experiments/pytorch_gap.sh; out/pytorch-gap/margins.md, gaps.md
```

`CURVE` is the directory holding D124's realizations, from presk-project `data/epsilon-curve-0.12.0.tar.xz`. The script's `CONSTS=`, `W=` and `MARGINS=` take another model: the course constants, with run 2's exact logits (D129).
