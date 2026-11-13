> The course GPT trained at its course constants and taken through the margins' whole pipeline (presk D129). It was run at the user's request. Version 0.12.0 (D128's binary), on falicon, 2026-10-16, from 08:41 to 17:20. Nothing is timed but the training's step, reported as a figure of the run, not of record. The tables come from `experiments/margins_course.sh` (`out/margins-course/margins.md`), `experiments/curve_course.sh` (`out/curve-course.md`), `experiments/val_loss.sh` and `experiments/pytorch_gap.sh` at the course constants (`out/pytorch-gap-course/`). The outputs are in presk-project `data/margins-course-0.12.0.tar.xz`.

# The course GPT at course scale, trained, through the margins

The largest limitation of the results so far was "two models at their initialization, no accuracy". At course scale nothing was trained: `out/course_probe_f64` is one step from initialization, used for identity and timing (D93). The margins' pipeline (ε from the exact model's margins, the flips, the budget on the logits; D118, D123, D125) had run only at D = 64. This run trains the course GPT at its course constants and takes it through the whole pipeline.

## How

**0. The training.**
- *What:* the course GPT at its course constants (D = 384, H = 6, L = 6, 10.8 M parameters), trained as `experiments/curve.sh` trains the D = 64 model.
  - 600 steps natively in `f64`, on 90 % of the Code civil (`out/civil_train_T256.psk.d`).
  - The batch of the file's training run, B = 16 windows of T = 256.
  - Every core, `--remat store`, into `out/course_train_f64`.
- *The command:*

  ```sh
  presk train corpus/nlp/05_gpt.psk --const D=384 --const H=6 --const L=6 \
      --data out/civil_train_T256.psk.d --steps 600 --epochs 10 --log-every 10 \
      --backend native --realize f64 --remat store --ckpt out/course_train_f64
  ```

- *When:* 08:41 to 12:44, 4 h 03 min. A median of 24 453 ms a step over 599 steps, the first excluded; D127's J′ found 24.35 s for all-`f64`.
- *The loss:* 4.47 at step 10, 2.2515 on the last batch.
- *The binary:* the `widening` branch's (D128), whose change does not touch training.

**1. The curve at course scale** (`experiments/curve_course.sh`, a copy of `curve.sh`'s QUALITY=0 path, the course dims and the trained weights):
- *The envelope:* `loss` on the seed's synthesized tokens at B = 2, T = 256, 128 bits, in 24.7 min.
- *The solves:* the seven budgets 10⁻¹…10⁻⁷ with the trained weights (sensitivities and measurement on them), 43 to 69 min each, side by side.

**2. The exact logits** (`experiments/margins_course.sh`, a copy of `margins.sh` with the course dims, the trained weights, the curve above, and `out/margins-course`):
- *The run:* `logits` of `experiments/margins.psk` at 128 bits on the 461 held-out windows of `out/civil_val_T256.psk.d`, 118 016 positions.
  - 116 chunks of 4 windows, twelve at a time, from 13:05 to 15:33.
  - 180 s a window on an X925 core (D101: 178.6 s); 1.6 to 3 GB a run.
- *What is kept:* every chunk's JSON at 30 digits.
  - It is kept without the parameter tree θ that `presk run --json` prints beside the result, 623 MB a chunk and `out/course_train_f64`'s already.
  - What remains is 373 MB: the logits, the tokens, the key and the header.
- *For the native stage:* `experiments/margins_rechunk.py` regroups the chunks by 32 windows, byte for byte in the same order, because each native `presk run --json` prints θ too.
- *Then the margins, ε\*, every realization's flips and decisions,* as `margins.sh` does them.

**3. The budget on the logits**, by V6's protocol (D124):
- *The envelope:* `presk calibrate --precision experiments/margins.psk --entry logits` on the first 4 held-out windows, at 128 bits, with the trained weights; 50 min 48 s, 1.4 GB.
- *The solve:* `presk solve --budget-abs 4.96e-4`, ε\*(10⁻³) rounded down; 1 h 36 min, 3.1 GB.
- *The measurement:* the realization through stage 2's rows (`EXTRA`).

**4. The validation loss** of every realization on all 461 windows:
- *The realizations:* natively (`experiments/val_loss.sh`, `loss` on every window at once, as `curve.sh`'s quality axis reads it).
- *The exact model's:* from the exact logits (`experiments/margins_loss.py`).

**5. PyTorch's rows at course scale** (run 3's script, D130): `CONSTS=` the course constants, `W=out/course_train_f64/final.psk.d`, `MARGINS=out/margins-course`.

## The trained model

| | value |
|---|---|
| training loss, step 600 (one batch) | 2.2515 |
| **validation loss, the exact model** (461 windows, 118 016 positions) | **2.330740** (2.330739966977) |
| validation loss, native `f64` | 2.33073996698 |
| **accuracy of the exact model** (next character, argmax) | **0.303722** (35 844 of 118 016) |
| largest \|logit\| | 10.856 |

At D = 64 the model of `make margins` reaches 0.242442 (D118). The course constants, trained the same 600 steps, reach 0.303722, and a validation loss of 2.3307.

## ε\* and the margins

**The margins:** smallest 1.83·10⁻⁵, median 0.424, largest 5.20; no ties.

| q | margin |
|---|---|
| 10⁻⁴ | 1.415·10⁻⁴ |
| 10⁻³ | 1.177·10⁻³ |
| 10⁻² | 1.070·10⁻² |
| 10⁻¹ | 9.734·10⁻² |
| 0.5 | 0.4239 |

| δ | k corrected (Clopper–Pearson, α = 0.05) | **ε\* corrected** | ε\* uncorrected |
|---|---|---|---|
| 10⁻² | 1 123 | 5.037·10⁻³ | 5.348·10⁻³ |
| **10⁻³** | 99 | **4.960·10⁻⁴** | 5.883·10⁻⁴ |
| 10⁻⁴ | 5 | 3.098·10⁻⁵ | 7.074·10⁻⁵ |

At D = 64, ε\*(10⁻³) is 4.67·10⁻⁴ (D118). Out of sample, ε\* from the even windows (4.63·10⁻⁴ at 10⁻³) puts 7.30·10⁻⁴ of the odd windows' margins below 2ε\*, and the other way round 8.46·10⁻⁴: both inside δ.

## Every realization: gap, flips, validation loss

| realization | gap on the logits | flips | bound #{m < 2ε} | certified at 10⁻³ | validation loss | vs exact | accuracy |
|---|---|---|---|---|---|---|---|
| all-`f64` | 2.16·10⁻¹⁴ | 0 | 0 | yes | 2.33073996698 | +3·10⁻¹² | 0.303722 |
| all-`f32` | 1.755·10⁻⁵ | 0 | 2 | yes | 2.33072781563 | −1.2·10⁻⁵ | 0.303722 |
| all-`bf16` | 0.2097 | 662 | 58 508 | no | 2.328125 | −2.6·10⁻³ | 0.303832 |
| `int8`, `f32` elsewhere | 0.2220 | 149 | 61 106 | no | 2.33076953888 | +3.0·10⁻⁵ | 0.303789 |
| `int8` pow2 | 0.1840 | 234 | 51 766 | no | 2.33073401451 | −6·10⁻⁶ | 0.303798 |
| `int4`, `f32` elsewhere | 2.480 | 3 078 | 117 895 | no | 2.33253717422 | +1.8·10⁻³ | 0.303510 |
| `int4` pow2 | 4.952 | 4 918 | 118 016 | no | 2.33444333076 | +3.7·10⁻³ | 0.302976 |
| solved at 10⁻¹ on the loss | 0.1055 | 87 | 27 300 | no | 2.33072137833 | −1.9·10⁻⁵ | 0.303823 |
| solved at 10⁻² | 6.079·10⁻³ | 3 | 1 364 | no | 2.33072185516 | −1.8·10⁻⁵ | 0.303730 |
| **solved at 10⁻³** | **2.819·10⁻⁴** | **0** | 52 | **yes** | 2.33072757721 | −1.2·10⁻⁵ | 0.303722 |
| solved at 10⁻⁴ | 2.062·10⁻⁴ | 0 | 37 | yes | 2.33072757721 | −1.2·10⁻⁵ | 0.303722 |
| solved at 10⁻⁵ | 6.463·10⁻⁶ | 0 | 0 | yes | 2.33073996726 | +3·10⁻¹⁰ | 0.303722 |
| solved at 10⁻⁶ | 6.530·10⁻⁶ | 0 | 0 | yes | 2.33073996732 | +3·10⁻¹⁰ | 0.303722 |
| solved at 10⁻⁷ | 3.911·10⁻⁹ | 0 | 0 | yes | 2.33073996698 | +3·10⁻¹² | 0.303722 |
| **solved at ε\* = 4.96·10⁻⁴ on the logits** | **2.093·10⁻⁴** | **0** | 37 | **yes** | 2.33073978203 | −1.8·10⁻⁷ | 0.303722 |
| PyTorch `float32`, ten threads (D130's script) | **6.42·10⁻⁶** | 0 | 0 | yes | — | — | 0.303722 |
| PyTorch `float32`, one thread | 6.42·10⁻⁶, bit for bit | 0 | 0 | yes | — | — | 0.303722 |
| PyTorch `float64` | 1.63·10⁻¹⁴ | 0 | 0 | yes | — | — | 0.303722 |

"vs exact" is a realization's validation loss minus the exact model's (2.330739966977). "Accuracy" is a fraction of the 118 016 positions.

**Every flip lies at a position whose margin is at most twice that position's gap** (the check of D118), and every realization's flips stay at or under its bound.

**For every realization ε\* certifies, the validation loss agrees with the exact model's to within 2·10⁻⁵, the fifth digit.**
- *What dominates it:* the loss's own reduction. The mean over 118 016 positions is taken in the realization's type, so all-`f32`'s −1.2·10⁻⁵ is mostly that sum's `f32` rounding, and `bf16`'s 2.328125 is a `bf16` number.
- *The accuracy:* the realizations inside ε\* match the exact model's 0.303722 at every position, since they flip nothing.

## The curve at course scale

| budget | stored values narrowed (`bf16`) | parameters typed (`bf16`) | traffic vs all-`f64` | estimate | measured gap (loss, B = 2) | widening rounds |
|---|---|---|---|---|---|---|
| 10⁻¹ | 32 of 32 (9) | 21 (11) | 2.19× | 4.40·10⁻¹ | 2.772·10⁻³ | 0 |
| 10⁻² | 32 (4) | 21 (10) | 2.04× | 4.27·10⁻² | 1.221·10⁻⁴ | 0 |
| **10⁻³** | 30 (1) | 21 (8) | **1.98×** | 4.39·10⁻³ | 1.021·10⁻⁶ | 0 |
| 10⁻⁴ | 27 (1) | 21 (1) | 1.94× | 2.31·10⁻⁴ | 2.794·10⁻⁶ | 0 |
| 10⁻⁵ | 12 (0) | 13 (1) | 1.51× | 4.51·10⁻⁵ | 7.289·10⁻⁹ | 0 |
| 10⁻⁶ | 9 (0) | 11 (1) | 1.08× | 4.45·10⁻⁶ | 9.338·10⁻⁹ | 0 |
| 10⁻⁷ | 4 (0) | 9 (1) | 1.02× | 3.35·10⁻⁷ | 2.318·10⁻¹⁰ | 0 |

Every solve is inside its budget, and none widened. The budget is relative to the loss on synthesized tokens, 8.83.

**The solve on the logits at ε\* = 4.96·10⁻⁴:**
- *Types:* 28 of 31 stored values in `f32` but the layer norm's ε in `bf16`; all 21 parameters typed, 20 in `f32` and `blocks.attn.wk.b` in `bf16`.
- *Traffic:* **1.97×** less than all-`f64`, the weights counted.
- *Gap:* estimate 4.23·10⁻⁴ for a mean element, measured 4.34·10⁻⁵ on the 4 envelope windows, no widening.
- *On all 461 windows:* 2.09·10⁻⁴, 4.8 times the envelope's measurement and still 2.4 times inside ε\*. At D = 64 the same ratio was 2.3 (V6: 3.24·10⁻⁵ on the envelope, 7.53·10⁻⁵ on all).

## Statement S at course scale

- **Does ε\*(10⁻³) still license a realization narrower than `f32`?** Yes.
  - The curve's realization at 10⁻³ is certified at 10⁻³: 2.82·10⁻⁴ ≤ 4.96·10⁻⁴.
  - It stores 8 of the 21 parameters in `bf16` (the attention and feed-forward biases and the norms' β) and the layer norm's ε in `bf16`, every other value in `f32`, at 1.98× less traffic than all-`f64`.
  - It flips none of the 118 016 decisions.
  - So does the curve's realization at 10⁻⁴ (2.06·10⁻⁴).
- **Does the realization solved on the logits flip nothing?** Yes: 0 flips, with a bound of 37, certified at 10⁻² and 10⁻³.
- **The statement, as `margins.py` checks it:**
  - every curve realization certified at 10⁻³ (10⁻³ to 10⁻⁷) flips at most δn = 118, and in fact none;
  - every realization's flips stay at or under its bound.
  - **S holds at course scale.**
- **Beyond the budget:** the realizations ε\* does not certify flip decisions. At 10⁻² the curve's flips 3 of 118 016; at 10⁻¹, 87; all-`bf16`, 662; the foil's `int8`, 149.

## PyTorch at course scale

- **PyTorch's `float32` run of the trained model is 6.42·10⁻⁶ from the exact logits:** 77 times inside ε\*(10⁻³), no flip, the same on one thread and ten.
- **The compiler's all-`f32` is 1.755·10⁻⁵, 2.7 times further.** At D = 64 the two were within 6 % (D130).
- **What differs:** PyTorch's products go through OpenBLAS and the rest through ATen's kernels, in their libraries' orders; the compiler sums in its declared sequential order, and the sums are longer at these constants (the feed-forward's down projection adds 1 536 terms). This run does not attribute the factor to any one of them.
- **The control:** PyTorch's `float64` run is 1.6·10⁻¹⁴ from the exact logits, as presk's native `f64` is (2.2·10⁻¹⁴).

## To reproduce

```sh
presk train … --ckpt out/course_train_f64            # the command above, 4 h
experiments/curve_course.sh                          # the envelope, then the seven solves (JOBS=7)
EXACT_ONLY=1 CHUNK=4 JOBS=12 experiments/margins_course.sh       # the exact logits, 2 h 30
mv out/margins-course/exact out/margins-course/exact4
python3 experiments/margins_rechunk.py out/margins-course/exact4 out/margins-course/exact \
    --from-chunk 4 --to-chunk 32 --windows 461
presk calibrate --precision experiments/margins.psk --entry logits --const D=384 --const H=6 --const L=6 \
    --dim T=256 --weights out/course_train_f64/final.psk.d --data out/civil_val_T256.psk.d --samples 4 \
    --bits 128 --out out/margins-course/logits/env_logits.json
presk solve experiments/margins.psk --precision out/margins-course/logits/env_logits.json \
    --budget-abs 4.96e-4 --weights out/course_train_f64/final.psk.d --sites 60 \
    --out out/margins-course/logits/solved.json
EXTRA="logits-4.96e-4=@out/margins-course/logits/solved.json" CHUNK=32 JOBS=8 experiments/margins_course.sh
experiments/val_loss.sh f64=f64 f32=f32 … curve-1e-3=@out/curve-course_1e-3.json …
python3 experiments/margins_loss.py out/margins-course/exact out/margins-course/val.json
CONSTS="--const D=384 --const H=6 --const L=6" W=out/course_train_f64/final.psk.d \
    MARGINS=out/margins-course OUT=out/pytorch-gap-course \
    PRESK_ROWS="f64 f32 bf16 curve-1e-3 curve-1e-4 logits-4.96e-4" experiments/pytorch_gap.sh
```

Each long stage ran under `systemd-run --user --scope -p MemoryMax=…`; no stage came near its cap.
