> The curve's envelope and the envelope-stability figure on the trained weights (presk D132). The prompt was presk-project `docs/prompt-envelope-state-2026-10-16.md`, at the user's request, after the user's observation on `make explain-pdf` that an envelope measured at initialization says nothing of the trained model. Version 0.12.0 (presk `main` `cce1379`), on falicon, 2026-10-16; nothing is timed. The tables come from `experiments/curve_trained.sh` (`out/curve-trained.md`), `experiments/curve_course_trained.sh` (`out/curve-course-trained.md`), `make margins` and `experiments/margins_course.sh` with the new realizations added, and the stability calibrations under `out/envelope-state/`. The outputs are in presk-project `data/envelope-state-0.12.0.tar.xz`, and the quality axis's, added after, in `data/envelope-state-quality-0.12.0.tar.xz`.

# The curve's envelope and the stability figure, on the trained weights

**What the old figures rested on.**
- *The curve's solves* (`curve.sh`, D124 at D = 64; `curve_course.sh`, D129 at course scale) calibrate their envelope with no `--weights` and no `--data`: the initialization's magnitudes on one batch of random tokens. They then solve every budget on the trained weights.
- *A solve runs on its envelope's inputs* (`presk-driver/src/precision.rs`, "the envelope's own inputs"). Those solves' sensitivities and measurements were therefore on random tokens too.
- *`make precision`'s three stability envelopes* (D35's 1.241× and 1.128×) are the initialization's.

**What this run does.**
- It solves the curve again at both scales, on an envelope of the trained weights on two held-out windows of text (`--weights W --data out/civil_val_T256.psk.d --samples 2 --from 0`, windows 0 and 1).
- It measures the new realizations on the exact logits that exist.
- It takes the stability calibrations again on the trained weights.
- Everything else is the old scripts'. The new scripts are copies (`curve_trained.sh`, `curve_course_trained.sh`), whose outputs go under new names.

## 1. The two envelopes at D = 64

The trained envelope (`out/curve-trained_env.json`) against the initialization's, which `curve.sh` takes and which was taken again here (`out/envelope-state/curve_env_init.json`):
- **The loss:** 4.522 at initialization on random tokens (ln 92: the zero-initialized head predicts uniformly), 2.788 for the trained model on the two windows. **Every budget is relative to it, so every absolute budget is 1.6 times smaller under the trained envelope.**
- **The magnitudes of the other sites:** they move by factors of 0.50 to 4.6, with a median of 1.04. The largest moves are the layer norms' variances (×4.6, their square roots ×2.15) and squares (×3.2), and one site of `GPT.apply`, `%19` (×1.8).
- **Five sites are zero at initialization and not trained**: the logits (`GPT.apply/%24`, 5.23 trained), `GPT.apply/%23`, `LayerNorm.apply/%14`, `Linear.apply/%4` and `loss/%5`, the values the zero-initialized head and its bias reach. At initialization the head is zero, so they have no magnitude there. A site without a magnitude is not narrowed, which is why D124's curve kept the logits in `f64` at every budget.

## 2. The curve at D = 64

| budget | D124: narrowed | D124: vs all-`f64` | D124: measured gap | **D132: narrowed** (`bf16`) | **D132: parameters** (`bf16`) | **D132: vs all-`f64`** | **D132: measured gap** | widening rounds (both) |
|---|---|---|---|---|---|---|---|---|
| 10⁻¹ | 32 | 2.78× | 2.401·10⁻⁴ | 33 (24) | 21 (21) | **3.17×** | 3.570·10⁻⁵ | 0 |
| 10⁻² | 32 | 2.29× | 1.308·10⁻⁵ | 33 (8) | 21 (12) | **2.44×** | 1.651·10⁻⁵ | 0 |
| 10⁻³ | 32 | 2.00× | 2.114·10⁻⁶ | 33 (4) | 21 (11) | **2.12×** | 1.603·10⁻⁵ | 0 |
| 10⁻⁴ | 30 | 1.97× | 1.160·10⁻⁶ | 32 (2) | 21 (9) | **1.97×** | 6.565·10⁻⁷ | 0 |
| 10⁻⁵ | 25 | 1.58× | 1.637·10⁻⁶ | 28 (1) | 21 (2) | **1.92×** | 1.796·10⁻⁷ | 0 |
| 10⁻⁶ | 13 | 1.35× | 4.542·10⁻⁹ | 19 (0) | 19 (1) | **1.52×** | 4.418·10⁻⁹ | 0 |
| 10⁻⁷ | 6 | 1.08× | 3.729·10⁻⁹ | 8 (0) | 10 (1) | **1.15×** | 1.738·10⁻⁹ | 0 |

- *What the columns measure:* D124's are its record's (`make curve` at 0.12.0, `curve-D124.md`). D132's measured gap is on the loss over the two windows, D124's on random tokens; neither is a gap on held-out text.
- *Every solve is inside its budget, and none widened.* D132 measured nothing that needed a widening round.
- *The bound (`f64`, `cond`) under the new envelope* is 1.60·10⁻⁶, against 2.58·10⁻¹³ under the initialization's.

**The quality axis** (the addendum, `QUALITY=1 experiments/curve_trained.sh`). Each realization, less its `params` as `curve.sh` does, trains the model from scratch: 600 steps on 90 % of the Code civil, into `out/curve-trained_train_<b>`, evaluated in `f64` on the held-out 10 %. All-`f64` is `make curve`'s own run (`out/curve_train_f64`, the same initialization, seed and batch order), 2.7736101.

| budget | D124: validation loss | D124: vs all-`f64` | **D132: validation loss** | **D132: vs all-`f64`** |
|---|---|---|---|---|
| 10⁻¹ | 2.7736617 | +5.16·10⁻⁵ | 2.7737382 | **+1.28·10⁻⁴** |
| 10⁻² | 2.7736090 | −1.10·10⁻⁶ | 2.7736033 | −6.80·10⁻⁶ |
| 10⁻³ | 2.7736114 | +1.30·10⁻⁶ | 2.7736036 | −6.50·10⁻⁶ |
| 10⁻⁴ | 2.7736114 | +1.30·10⁻⁶ | 2.7736114 | +1.30·10⁻⁶ |
| 10⁻⁵ | 2.7736114 | +1.30·10⁻⁶ | 2.7736114 | +1.30·10⁻⁶ |
| 10⁻⁶ | 2.7736101 | 0 | 2.7736101 | 0 |
| 10⁻⁷ | 2.7736101 | 0 | 2.7736101 | 0 |

**The quality axis stays flat.**
- *The spread:* every trained-envelope realization trains to within 1.3·10⁻⁴ of all-`f64`. The loosest, 10⁻¹, with every matrix in `bf16`, is 1.28·10⁻⁴ above it, against D124's largest, 5.2·10⁻⁵.
- *The yardstick:* the seeds' spread, 0.092 (D124: all-`f64` at seeds 0, 1, 2), is 700 times the largest difference.
- *The two tightest budgets* train to all-`f64`'s loss to the eighth digit, as under the initialization's envelope.
- *The full table*, in the format of `curve-D124.md`, is `curve-D132.md` (presk-project `data/envelope-state-quality-0.12.0.tar.xz`).

**The realizations moved.**
- *Under the trained envelope the solver narrows more at every budget but 10⁻⁴:* 3.17× against 2.78× at 10⁻¹, 2.12× against 2.00× at 10⁻³, 1.92× against 1.58× at 10⁻⁵.
- *At 10⁻³, six decisions differ:*
  - the logits from `f64` to `f32`;
  - two of the attention's stored values (`SelfAttention.apply/%16` and `%20`) from `f32` to `bf16`;
  - three norm parameters (`blocks.ln1.γ`, `blocks.ln2.γ`, `lnf.β`) from `f32` to `bf16`.
- *At 10⁻¹ eighteen decisions differ,* every matrix among them, to `bf16`.

## 3. The margins at D = 64

**The setup.** The new realizations are measured on `make margins`' model and windows, from the exact logits taken that day (D130), beside D124's realizations from the same run. ε\*(10⁻³) = 4.67·10⁻⁴.

| budget | D124: gap on the logits | D124: flips | D124: certified at 10⁻³ | **D132: gap on the logits** | **D132: flips** | **D132: bound** | **D132: certified at 10⁻³** |
|---|---|---|---|---|---|---|---|
| 10⁻¹ | 3.553·10⁻² | 76 | no | 4.126·10⁻² | 518 | 9 298 | no |
| 10⁻² | 1.452·10⁻² | 38 | no | 1.704·10⁻² | 65 | 3 706 | no |
| **10⁻³** | **2.516·10⁻⁴** | **0** | **yes** | **7.785·10⁻³** | **30** | 1 692 | **no** |
| 10⁻⁴ | 7.486·10⁻⁵ | 0 | yes | 2.746·10⁻⁴ | 0 | 59 | yes |
| 10⁻⁵ | 7.608·10⁻⁵ | 0 | yes | 7.521·10⁻⁵ | 0 | 18 | yes |
| 10⁻⁶ | 8.897·10⁻⁷ | 0 | yes | 1.083·10⁻⁶ | 0 | 0 | yes |
| 10⁻⁷ | 5.944·10⁻⁸ | 0 | yes | 7.468·10⁻⁷ | 0 | 0 | yes |

**The trained envelope's realization at 10⁻³ is not certified.**
- *Its gap on the logits:* 7.8·10⁻³, 31 times D124's, flipping 30 of the 118 016 decisions.
- *Its gap on the loss:* inside its budget, 1.6·10⁻⁵ against 2.8·10⁻³.

**The budget on the loss does not bound the logits.** The loss is a mean over 512 positions, and the budget constrains that mean, not each position's logits; the budget on the logits does (V6, D123). D124's realization at 10⁻³ happened to lie close to the logits (2.5·10⁻⁴). This one, with two attention values and three norm parameters in `bf16`, does not.

**Statement S** concerns the certified realizations. It holds under both envelopes:
- *D132:* four of seven certified at 10⁻³ (10⁻⁴ to 10⁻⁷), no flip;
- *D124:* five of seven (10⁻³ to 10⁻⁷), no flip.

**The report's rows named "solved at 10⁻ᵏ" are neither D124's nor D132's.** `make margins` reads `out/curve_<b>.json` by default.
- *What those files are:* the realizations `make curve` wrote at the tag `v0.10.0` on 2026-10-03, between 08:10 and 08:16 (D102). They hold 65 stored sites, the 0.10.0 solver's, before D112 left the inert ones out, and no `params` map. D117's and D124's curves wrote theirs elsewhere (D124's are kept in presk-project `data/epsilon-curve-0.12.0.tar.xz`, here `d124-<b>`).
- *Against D118:* measured with 0.12.0's binary, they give D118's figures to every digit at 10⁻², 10⁻³, 10⁻⁴ and 10⁻⁷ (at 10⁻³: 3.345·10⁻⁴, one flip), where D112's inert sites changed no bit and D117's solves chose the same effective types. They give other figures at 10⁻¹, 10⁻⁵ and 10⁻⁶.
- *What to read instead:* none of these rows is of record. Read the `d124-<b>` rows (D125's to every digit), the `d132-<b>` rows, the uniform types' and the foil's rows (D125's), and V6's `logits-4.67e-4`.

`make margins`' model is `make curve`'s all-`f64` run (`out/curve_train_f64`), and the solves' weights are `make gpt-trained`'s (`out/gpt_trained`), as in D118, D125 and D130: two trainings of the same model.

## 4. The course scale

**The envelope.** On `out/course_train_f64` and windows 0 and 1, from 18:20, 25 min. It compares with D129's initialization envelope (`out/curve-course_env.json`):
- *The loss:* 4.522 at initialization against **2.248** trained, so every absolute budget is **2.0 times smaller**.
- *The other magnitudes:* they move by 0.50 to 11.4, with a median of 1.24; the most moved is a layer norm's variance (`LayerNorm.apply/%5`, ×11.4).
- *Zero at initialization:* the same five sites as at D = 64.
- *The bound (`f64`, `cond`):* 9.78·10¹⁴ under the trained envelope, against 2.58·10⁻¹³ under the initialization's. The bound is vacuous on the trained model, as D39 found on a transformer.

**The curve.** Seven solves side by side, 18:45 to 19:34:

| budget | D129: narrowed | D129: vs all-`f64` | D129: measured gap | **D132: narrowed** (`bf16`) | **D132: parameters** (`bf16`) | **D132: vs all-`f64`** | **D132: measured gap** | widening rounds (both) |
|---|---|---|---|---|---|---|---|---|
| 10⁻¹ | 32 | 2.19× | 2.772·10⁻³ | 33 (8) | 21 (16) | **2.68×** | 6.862·10⁻⁶ | 0 |
| 10⁻² | 32 | 2.04× | 1.221·10⁻⁴ | 30 (7) | 21 (11) | **2.05×** | 9.112·10⁻⁶ | 0 |
| 10⁻³ | 30 | 1.98× | 1.021·10⁻⁶ | 30 (3) | 21 (11) | **1.98×** | 2.198·10⁻⁶ | 0 |
| 10⁻⁴ | 27 | 1.94× | 2.794·10⁻⁶ | 29 (1) | 21 (5) | **1.96×** | 2.907·10⁻⁷ | 0 |
| 10⁻⁵ | 12 | 1.51× | 7.289·10⁻⁹ | 26 (1) | 21 (1) | **1.81×** | 5.224·10⁻⁸ | 0 |
| 10⁻⁶ | 9 | 1.08× | 9.338·10⁻⁹ | 8 (0) | 11 (1) | **1.27×** | 1.338·10⁻¹⁰ | 0 |
| 10⁻⁷ | 4 | 1.02× | 2.318·10⁻¹⁰ | 6 (0) | 11 (1) | **1.03×** | 5.322·10⁻¹⁰ | 0 |

- *Every solve is inside its budget, and none widened.*
- *At 10⁻³ the traffic is the same, 1.98×, but seven decisions differ:*
  - the logits from `f64` to `f32`;
  - two attention values (`SelfAttention.apply/%15`, `%16`) to `bf16`;
  - three norm parameters (`blocks.ln2.γ`, `lnf.β`, `lnf.γ`) to `bf16`;
  - and one the other way, a layer norm's variance (`LayerNorm.apply/%5`) from `f32` to `f64`.
- *At 10⁻⁵, 23 decisions differ,* nearly all from `f64` to `f32`, the parameters among them.

**The margins at course scale.** D129's exact logits and windows; ε\*(10⁻³) = 4.96·10⁻⁴; D129's rows from the same report:

| budget | D129: gap on the logits | D129: flips | D129: certified at 10⁻³ | **D132: gap on the logits** | **D132: flips** | **D132: bound** | **D132: certified at 10⁻³** |
|---|---|---|---|---|---|---|---|
| 10⁻¹ | 0.1055 | 87 | no | 4.988·10⁻² | 35 | 12 082 | no |
| 10⁻² | 6.079·10⁻³ | 3 | no | 3.169·10⁻² | 10 | 7 541 | no |
| **10⁻³** | **2.819·10⁻⁴** | **0** | **yes** | **6.523·10⁻³** | **29** | 1 482 | **no** |
| 10⁻⁴ | 2.062·10⁻⁴ | 0 | yes | 1.991·10⁻⁴ | 0 | 35 | yes |
| 10⁻⁵ | 6.463·10⁻⁶ | 0 | yes | 2.050·10⁻⁴ | 0 | 36 | yes |
| 10⁻⁶ | 6.530·10⁻⁶ | 0 | yes | 1.887·10⁻⁷ | 0 | 0 | yes |
| 10⁻⁷ | 3.911·10⁻⁹ | 0 | yes | 7.627·10⁻⁸ | 0 | 0 | yes |

The same as at D = 64:
- *At 10⁻³:* the trained envelope's realization is 6.5·10⁻³ from the exact logits and flips 29 decisions, so ε\* does not certify it. Its loss is within 9.8·10⁻⁷ of the exact one, relatively, on two windows.
- *From 10⁻⁴ down:* its realizations are certified and flip nothing, so **statement S holds**.
- *The accuracy:* every certified realization keeps the exact model's 0.303722; the 10⁻³ realization's 29 flips leave it at 0.303730, one more correct position.

## 5. The envelope-stability figure

D35's three calibrations of `make precision`, eight windows of the corpus (`corpus/nlp/data/civil_T256.psk.d`) from window 0, from window 8 (disjoint) and from window 64 (far apart). The worst ratio between the envelopes over the sites compared:

| model | weights | 0–7 against 8–15 | 0–7 against 64–71 | sites compared | zero in both |
|---|---|---|---|---|---|
| D = 32 (D35's) | initialization | 1.241× | 1.128× | 76 | 24 |
| D = 64 | initialization | 1.191× | 1.286× | 76 | 24 |
| **D = 64** | **`gpt_trained`** | **1.251×** | **1.102×** | **81** | **19** |

- *The baseline:* at 0.12.0, D35's figure comes back to the digit.
- *The trained model's envelope* agrees across slices to within 1.25×: the same factor as the initialization's.
- *The sites:* five more sites are compared on the trained weights, those zero at initialization (§1).

## The answer

**The curve's realizations are not the same under the trained model's envelope.**
- *Why they move:* the envelope at initialization differs from the trained model's. The loss is 1.6 times larger at D = 64 and 2.0 times at course scale, which scales every absolute budget; site magnitudes move by up to 4.6 and 11.4 times; and five sites, the logits among them, are zero at initialization.
- *Traffic, D = 64:* the solver narrows more at every budget but 10⁻⁴: 3.17× against 2.78× at 10⁻¹, 2.12× against 2.00× at 10⁻³, 1.15× against 1.08× at 10⁻⁷.
- *Traffic, course scale:* 2.68× against 2.19× at 10⁻¹, the same 1.98× at 10⁻³, 1.81× against 1.51× at 10⁻⁵.
- *No widening:* no proposal needed a widening round, at either scale, under either envelope, and every solve is inside its budget on the loss.
- *But at 10⁻³ the realizations differ in kind.* The trained envelope puts two attention values and three norm parameters in `bf16`, and that realization is far from the exact logits: 7.8·10⁻³ at D = 64 and 6.5·10⁻³ at course scale, 30 and 29 flips, outside ε\*(10⁻³). The initialization's realizations at 10⁻³ were inside it, with no flip.
- *What that says:* a budget on the loss, a mean over positions, does not bound the logits. That the old 10⁻³ realizations flipped nothing was a property of those realizations, not of the budget. The budget on the logits does bound them, and its solves (V6 at D = 64, D129 at course scale) are on the trained model's envelope already.
- *Statement S holds under both envelopes and at both scales:* every realization certified at 10⁻³ (10⁻⁴ and below under the trained envelope) flips nothing.
- *The envelope-stability factor on trained weights* is **1.25× and 1.10×**, against D35's 1.24× and 1.13× at initialization (1.19× and 1.29× at initialization at D = 64): the magnitudes are as stable across slices of text on the trained model as on the initialization.

## To reproduce

```sh
experiments/curve_trained.sh                          # D = 64, minutes
experiments/curve_course_trained.sh                   # course scale: the envelope, then seven solves side by side
EXTRA="d124-1e-3=@… d132-1e-3=@out/curve-trained_1e-3.json …" make margins
EXTRA="ct-1e-3=@out/curve-course-trained_1e-3.json …" CHUNK=32 experiments/margins_course.sh   # the exact chunks regrouped by 32 (D129)
presk calibrate --precision corpus/nlp/05_gpt.psk --entry loss --const D=64 --const H=2 --const L=2 --dim T=256 \
    --data corpus/nlp/data/civil_T256.psk.d --bits 128 --samples 8 --from 0|8|64 --weights out/gpt_trained/final.psk.d \
    [--compare A.json] --out …
```
