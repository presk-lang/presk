> The study proposed in presk-project `docs/prompt-second-corpus-2026-10-22.md`, run on 2026-10-22 from 14:18 to 17:42 at the user's go, relayed by the paper session. Nothing was calibrated, solved, trained or timed. Version 0.16.0. The binary is D163's, `out/bin/presk-22dd345` (`presk 0.16.0-dev`), so every row on the Code pénal comes from the binary that wrote its row on the Code civil. The script is `experiments/second_corpus.sh` (`make second-corpus`), with `experiments/second_corpus.py` (standard library only). The outputs are in `out/second-corpus/`, `out/margins-penal/` and `out/margins-course-penal/`, archived in presk-project `data/second-corpus-0.16.0.tar.xz`. The decision is D168.

# The certificate on a second text: the Code pénal through the Code civil's models

Every precision figure of record is measured on one text, the Code civil:
- the envelopes and the solves;
- the gaps on its 461 held-out windows;
- ε\*(10⁻³), from the exact model's margins there;
- the certification.

This study keeps the models, the realizations and ε\* as they are and changes the text to the Code pénal: the same container, the same Markdown form, the same vocabulary but for 32 characters. Two statements were fixed before the measurement:
- **Y1 (the certificate carries).** Every realization certified at 10⁻³ on the Code civil's 461 windows has, on the Code pénal's 461, a gap at most the Code civil's ε\*(10⁻³), and it changes no decision of the exact model.
- **Y2 (the gap carries within 2×).** Every realization's largest gap on the Code pénal is within [0.5×, 2×] of its largest gap on the Code civil.

**In one paragraph.**
- **At D = 64 both hold.** The twelve certified realizations are all inside ε\* = 4.67·10⁻⁴ on the Code pénal and flip nothing. V6 is at 1.016·10⁻⁴, against 1.431·10⁻⁴ on the Code civil. The 22 ratios run from 0.54× to 1.31×, and the median window moves by 5 % at most.
- **At the course constants both fail, on one stretch of text.** Window 261 runs from the prediction after "par le J" ("…est faite par le J|ournal officiel de la République française") to the window's end. Eighteen of the 22 course realizations have their largest gap there.
  - Four of the twelve certified realizations go above ε\* = 4.96·10⁻⁴ there.
  - Among them is D163 §5's, the logits solved at ε\*, at **9.41·10⁻⁴, 1.90 ε\***. On the Code civil it was 4.23·10⁻⁴.
  - Seven of the 22 realizations fall outside 2×, the highest at 2.82×.
- **The decision that carries it is LayerNorm's ε in `bf16`.**
  - Run on that window's chunk with ε back to `f32`, the course curve's 10⁻⁴ realization falls from 5.07·10⁻⁴ to 2.0·10⁻⁵ there.
  - D163 §5's realization falls to 4.6·10⁻⁴. With the queries' bias also back to `f32` (step 0b's decision), it falls to 1.4·10⁻⁵.
  - The five certified realizations that keep ε in `bf16` move 1.70× to 2.73×. The five that keep it in `f32` move 0.78× to 1.26×, but for one 370 times inside ε\*.
- **The same four realizations flip one or two decisions** at two near-ties of the exact model, margins 1.07·10⁻⁵ and 2.6·10⁻⁶, smaller than any on the Code civil (1.8·10⁻⁵). Every flip has a margin below twice the gap at its position, as the bound says. One or two flips in 118 016 are inside the certificate's δ = 10⁻³ by two orders, but Y1 asked for none.
- **The Code pénal's margins.** At D = 64 the Code civil's ε\* holds its δ there: 80 margins below 2ε\* (6.8·10⁻⁴). At course scale it does not quite: 125 (1.06·10⁻³), and the Code pénal's own ε\*(10⁻³) is 0.80 of the Code civil's.
- **The exact model is as good on both texts.** The loss is 2.747 against 2.774 at D = 64 and 2.313 against 2.331 at course scale, and the accuracy is within 0.6 point. The domain shift is small. What moved is one stretch of text that the solve's envelope never saw.

## The data

- **The text.** `penal.md` from the course's container, the one that holds `civil.md.zip` (`…/dila-legi-codes/penal.md.zip`, fetched 2026-10-22). It is 905 254 bytes and 876 212 characters, `title: Code pénal`, `date: 2024-01-15`. Its sha256 is `be64baff…243d`, and the script refuses another. The container's `civil.md` is byte for byte the course's.
- **The vocabulary.** The Code civil's, `civil_vocab.json` (92 characters, CR included). The Code pénal has 94 distinct characters. Six are not in the Code civil's vocabulary, and their 32 occurrences are deleted (`out/second-corpus/deleted.tsv`):

  | character | occurrences | where |
  |---|---|---|
  | `/` | 19 | `561/2006`, `km/h` |
  | `k` | 7 | `km/h` |
  | `“` `”` | 2 and 2 | a quoted amendment |
  | `[` `]` | 1 and 1 | `[Dispositions déclarées non conformes …]` |

  Without the deletion `presk data encode` refuses the text (`E9004`, the character `/`). The 61 CRLF are kept, as the Code civil's 39 are.
- **The windows.** `presk data lm-windows --vocab 92 --seq 256` on the 876 180 tokens gives 3 422 windows of 256 (`corpus/nlp/data/penal_T256.psk.d`). The judge set is the first 461 (`out/penal_461_T256.psk.d`), as many as the Code civil's held-out windows, 118 016 positions.
  - It is the same stream cut at 461·256 + 1 tokens. Its 461 samples are those of the whole file, checked.
  - It covers the first 13.5 % of the code, inside its Livre Ier (*Dispositions générales*), up to article 132-41-1.
  - Three of the 32 deletions fall in it.

## The protocol

- **D = 64.** `experiments/margins.sh` on `out/curve_train_f64` with the judge set, into `out/margins-penal`, 14:19 to 14:25. The steps:
  - the exact logits at 128 bits;
  - every realization of `out/margins-0.16.0` (D163 §4) run natively: the seven uniform and integer rows, v0.10.0's curve (`curve-<b>`), the trained-envelope curve (`d016-<b>`) and V6 (`logits-4.67e-4`).
- **The course constants.** `experiments/margins_course.sh` on `out/course_train_f64`, into `out/margins-course-penal`. The steps:
  - the exact logits D129's way, 116 chunks of 4 windows twelve at a time (14:25 to 16:46, 2 h 21 min), regrouped by 32 (`margins_rechunk.py`);
  - every realization of `out/margins-course-0.16.0` (D163 §5) run natively (16:46 to 17:40): the uniform and integer rows, D129's curve solved at 0.16.0 (`curve-<b>`), the trained-envelope curve (`ct-<b>`) and the logits at ε\* (`logits-4.96e-4`).
- **The Code civil's rows are those of record.** They are `out/margins-0.16.0` and `out/margins-course-0.16.0`, read and not run again. The realization files are the archived ones (presk-project `data/rows-0.16.0.tar.xz`), checked byte for byte.
- **The exact loss** is `experiments/margins_loss.py`'s, from the exact logits. On the Code civil at course scale it gives D129's 2.33074.
- **One chunk under step 0b's variants.** Windows 256–287 of the Code pénal under the variants of two realizations, D163 §5's and the course curve's 10⁻⁴ (`experiments/off_sample_tail.py variants`). The variants put each `bf16` decision back to `f32`, all of them, and D129's types. Native runs only, 17:41 to 17:42.

## The two texts under the exact model

| | D = 64: Code civil | D = 64: Code pénal | course: Code civil | course: Code pénal |
|---|---|---|---|---|
| exact loss (nats) | 2.7736 | 2.7474 | 2.3307 | 2.3130 |
| exact accuracy | 0.24244 | 0.24858 | 0.30372 | 0.30038 |
| largest \|logit\| | 5.210 | 5.210 | 10.856 | 11.047 |
| smallest margin | 1.19·10⁻⁵ | 1.86·10⁻⁵ | 1.83·10⁻⁵ | **2.59·10⁻⁶** |
| median margin | 0.458 | 0.474 | 0.424 | 0.420 |
| ε\*(10⁻³), its own, corrected | 4.670·10⁻⁴ | 5.758·10⁻⁴ (1.23×) | 4.960·10⁻⁴ | 3.990·10⁻⁴ (0.80×) |
| margins below twice the Code civil's ε\* | 99 (8.4·10⁻⁴) | 80 (6.8·10⁻⁴) | 99 (8.4·10⁻⁴) | 125 (**1.06·10⁻³**) |

Statement S (D118) holds on the Code pénal at both scales, at its own ε\*: every realization within it flips at most δn, and every row flips at most its bound. The ε\* of one half of its windows holds on the other half at every δ at D = 64. At course scale it holds at every δ but 10⁻² from the odd windows, where the other half has a fraction of 1.008·10⁻².

## D = 64

ε\* is the Code civil's, 4.67·10⁻⁴. '≤ ε\*' is the gap against it on each text. The bound is #{m < 2·gap} on the Code pénal.

| realization | gap, Code civil | gap, Code pénal | ratio | flips, civil / pénal | bound, pénal | ≤ ε\*: civil / pénal |
|---|---|---|---|---|---|---|
| f64 | 5.894e-15 | 6.021e-15 | 1.02× | 0 / 0 | 0 | yes / yes |
| f32 | 3.115e-06 | 3.224e-06 | 1.04× | 0 / 0 | 0 | yes / yes |
| bf16 | 5.033e-02 | 4.878e-02 | 0.97× | 480 / 419 | 12109 | no / no |
| int8 | 7.808e-02 | 4.182e-02 | 0.54× | 161 / 124 | 8617 | no / no |
| int8-pow2 | 1.149e-01 | 1.228e-01 | 1.07× | 168 / 157 | 43186 | no / no |
| int4 | 1.240e+00 | 8.976e-01 | 0.72× | 2968 / 2603 | 118016 | no / no |
| int4-pow2 | 2.083e+00 | 1.571e+00 | 0.75× | 3488 / 3068 | 118016 | no / no |
| curve-1e-1 | 3.040e-02 | 2.534e-02 | 0.83× | 80 / 71 | 4842 | no / no |
| curve-1e-2 | 1.422e-02 | 1.856e-02 | 1.31× | 36 / 38 | 3526 | no / no |
| curve-1e-3 | 3.345e-04 | 2.753e-04 | 0.82× | 1 / 0 | 39 | yes / yes |
| curve-1e-4 | 7.509e-05 | 4.972e-05 | 0.66× | 0 / 0 | 5 | yes / yes |
| curve-1e-5 | 1.607e-06 | 1.047e-06 | 0.65× | 0 / 0 | 0 | yes / yes |
| curve-1e-6 | 1.023e-06 | 9.147e-07 | 0.89× | 0 / 0 | 0 | yes / yes |
| curve-1e-7 | 6.712e-08 | 4.944e-08 | 0.74× | 0 / 0 | 0 | yes / yes |
| **logits-4.67e-4** (V6) | 1.431e-04 | 1.016e-04 | 0.71× | 0 / 0 | 11 | yes / yes |
| d016-1e-1 | 4.126e-02 | 4.310e-02 | 1.04× | 518 / 435 | 9012 | no / no |
| d016-1e-2 | 2.181e-02 | 2.425e-02 | 1.11× | 67 / 46 | 4622 | no / no |
| d016-1e-3 (H2's) | 2.551e-03 | 2.176e-03 | 0.85× | 9 / 9 | 371 | no / no |
| d016-1e-4 | 3.467e-04 | 3.563e-04 | 1.03× | 0 / 0 | 60 | yes / yes |
| d016-1e-5 | 1.623e-04 | 1.190e-04 | 0.73× | 0 / 0 | 15 | yes / yes |
| d016-1e-6 | 7.644e-05 | 4.975e-05 | 0.65× | 0 / 0 | 5 | yes / yes |
| d016-1e-7 | 6.207e-07 | 7.797e-07 | 1.26× | 0 / 0 | 0 | yes / yes |

- **Y1 holds.** The twelve certified realizations (f64, f32, `curve-1e-3` to `-1e-7`, V6, `d016-1e-4` to `-1e-7`) are inside ε\* on the Code pénal and flip nothing.
- **Y2 holds.** The ratios run from 0.54× (int8) to 1.31× (`curve-1e-2`), median 0.85×.
- **The bulk does not move.** Every realization's median window is within 5 % of its median window on the Code civil (0.95× to 1.02×). What moves is the worst window, lower on the Code pénal for 14 of the 22. The Code civil's held-out text ends with the provisions for Mayotte, which hold V6's worst windows.
- **The same kind of position.** V6's ten worst windows on the Code civil are all a prediction right after a capital letter, seven of them the "M" of Mayotte. On the Code pénal it is nine of ten: "le titre V|I", "N|'", "T|o", "H|o". None follows an opening parenthesis.

## The course constants

ε\* is the Code civil's, 4.9605·10⁻⁴.

| realization | gap, Code civil | gap, Code pénal | ratio | flips, civil / pénal | bound, pénal | ≤ ε\*: civil / pénal |
|---|---|---|---|---|---|---|
| f64 | 2.164e-14 | 3.273e-14 | 1.51× | 0 / 0 | 0 | yes / yes |
| f32 | 1.755e-05 | 1.447e-05 | 0.82× | 0 / 0 | 4 | yes / yes |
| bf16 | 2.097e-01 | 2.965e-01 | 1.41× | 662 / 724 | 73291 | no / no |
| int8 | 2.220e-01 | 2.982e-01 | 1.34× | 149 / 188 | 73517 | no / no |
| int8-pow2 | 1.840e-01 | 2.637e-01 | 1.43× | 234 / 273 | 68817 | no / no |
| int4 | 2.480e+00 | 3.194e+00 | 1.29× | 3078 / 2983 | 118016 | no / no |
| int4-pow2 | 4.952e+00 | 3.983e+00 | 0.80× | 4918 / 5224 | 118016 | no / no |
| curve-1e-1 | 9.710e-02 | 1.049e-01 | 1.08× | 89 / 88 | 28315 | no / no |
| curve-1e-2 | 3.169e-02 | 8.931e-02 | **2.82×** | 10 / 14 | 23701 | no / no |
| curve-1e-3 (D129's curve row) | 2.629e-04 | 4.480e-04 | 1.70× | 0 / 0 | 109 | yes / yes |
| curve-1e-4 | 2.043e-04 | **5.072e-04** | **2.48×** | 0 / **1** | 126 | yes / **no** |
| curve-1e-5 | 5.074e-06 | 6.375e-06 | 1.26× | 0 / 0 | 2 | yes / yes |
| curve-1e-6 | 5.968e-06 | 5.636e-06 | 0.94× | 0 / 0 | 2 | yes / yes |
| curve-1e-7 | 3.911e-09 | 3.731e-09 | 0.95× | 0 / 0 | 0 | yes / yes |
| **logits-4.96e-4** (D163 §5) | 4.228e-04 | **9.410e-04** | **2.23×** | 0 / **1** | 228 | yes / **no** |
| ct-1e-1 | 9.788e-02 | 8.591e-02 | 0.88× | 47 / 69 | 22715 | no / no |
| ct-1e-2 | 1.498e-02 | 1.924e-02 | 1.28× | 29 / 31 | 4729 | no / no |
| ct-1e-3 | 6.518e-03 | 1.374e-02 | **2.11×** | 29 / 35 | 3433 | no / no |
| ct-1e-4 | 2.627e-04 | **7.173e-04** | **2.73×** | 0 / **2** | 169 | yes / **no** |
| ct-1e-5 | 2.059e-04 | **5.070e-04** | **2.46×** | 0 / **1** | 126 | yes / **no** |
| ct-1e-6 | 1.093e-06 | 8.482e-07 | 0.78× | 0 / 0 | 0 | yes / yes |
| ct-1e-7 | 6.263e-07 | 1.340e-06 | **2.14×** | 0 / 0 | 1 | yes / yes |

- **Y1 fails.** Four of the twelve certified realizations are above ε\* on the Code pénal: `logits-4.96e-4` (1.90 ε\*), `ct-1e-4` (1.45 ε\*), `curve-1e-4` and `ct-1e-5` (1.02 ε\*). The same four flip one or two decisions.
- **Y2 fails.** Seven of the 22 are above 2×, from 2.11× to 2.82×. Most ratios are above 1 (median 1.41×), where at D = 64 most were below.
- **The bulk does not move here either.** The median window is within 17 % of the Code civil's for every realization (0.83× to 1.01×).

## What moved: one stretch of window 261

**Every course realization but four has its largest gap in window 261,** f64 and f32 included. Three of the four are `curve-1e-5`, `-1e-6` and `-1e-7`, whose gaps are below 10⁻⁵. The fourth is `ct-1e-1`, whose worst is window 269.
- The position is 176, the prediction after "par le J" in "La diffusion de la décision est faite par le J|ournal officiel de la République française, par une ou plusieurs autres publications…" (article 131-35).
- The exact model's choice there is `*`, with a margin of 0.44. No tie is near.
- D163 §5's realization has 15 positions above 10⁻⁴ on the Code pénal. Twelve are in window 261, from position 176 to 248, that is the rest of the window from that "J" on.
- Its second-worst window is 57, at 1.63·10⁻⁴, inside ε\* by 3×.
- At D = 64 window 261 is ordinary: it ranks 113th of 461 for V6.

**The same kind of position as the Code civil's tail.** On the Code civil, D163 §5's second-worst position is in window 385, the prediction after a capital J ("Mamoudzou " J|uge"), where the exact model also chooses `*`. Step 0's enumerations after an opening parenthesis are the Code civil's tail: four of the ten worst windows. They are not the Code pénal's, with one of ten ("règlement (|CE)", window 23). The Code pénal's judge set has one opening parenthesis in 118 016 characters, where the Code civil's held-out windows have 11, seven of them in "(Abrogé)".

**The decision that carries it.** Windows 256–287 were run under step 0b's variants of two realizations (`out/second-corpus/w261/report.md`). Here is the gap at window 261, position 176, beside the largest elsewhere in the chunk:

| variant | at (261, 176) | the chunk's other windows |
|---|---|---|
| `curve-1e-4` as it is (ε, `wk.b`, the head width and its root in `bf16`) | 5.072·10⁻⁴ | 6.1·10⁻⁵ |
| `curve-1e-4`, ε back to `f32` | **2.030·10⁻⁵** | 3.3·10⁻⁶ |
| `curve-1e-4`, `wk.b`, the head width or its root back to `f32` | 5.072·10⁻⁴ | 6.1·10⁻⁵ |
| `logits-4.96e-4` as it is (ε, `wq.b`, `wk.b`, `ln2.β`, the head width and its root in `bf16`) | 9.410·10⁻⁴ | 8.3·10⁻⁵ |
| `logits-4.96e-4`, ε back to `f32` | 4.608·10⁻⁴ | 2.5·10⁻⁵ |
| `logits-4.96e-4`, `wq.b` back to `f32` | 5.008·10⁻⁴ | 5.7·10⁻⁵ |
| `logits-4.96e-4`, every `bf16` decision back to `f32` | **1.447·10⁻⁵** | 9.1·10⁻⁶ |
| D129's realization (ε and `wk.b` in `bf16`) | 5.028·10⁻⁴ | 6.1·10⁻⁵ |

- **LayerNorm's ε in `bf16` carries the position.** ε = 10⁻⁵ is 1.00136·10⁻⁵ in `bf16`. Its effect on a norm's output grows as the variance of the norm's input shrinks, since the norm divides by √(σ² + ε). The position is the worst for f64 and f32 too, so it is ill-conditioned for every type, and ε in `bf16` amplifies it 25 times on the curve's realization. This run does not say which of the thirteen norms it is.
- **On D163 §5's realization ε and the queries' bias add up:** each alone back to `f32` halves the gap there, and with all six `bf16` decisions back to `f32` it is all-`f32`'s gap to the digit, 1.447·10⁻⁵. Step 0b found that the queries' bias carries that realization's tail on the Code civil.
- **D129's realization of 0.12.0 would also fail here,** at 5.03·10⁻⁴, 1.01 ε\*.
- **Across the table.** The certified realizations that keep ε in `bf16` move 1.70× to 2.73× (`curve-1e-3`, `curve-1e-4`, `logits-4.96e-4`, `ct-1e-4`, `ct-1e-5`). Those that keep it in `f32` move 0.78× to 1.26× (`curve-1e-5` to `-1e-7`, `ct-1e-6`), but for `ct-1e-7`, which moves 2.14× at 1.3·10⁻⁶. Of the realizations outside ε\*, the two that fail Y2 keep ε in `bf16` (`curve-1e-2`, `ct-1e-3`). The others, whose gaps are other decisions' (`curve-1e-1`, `ct-1e-1`, `ct-1e-2`), move 0.88× to 1.28×.
- **Does D163 §5's realization move most?** Not by its ratio. It is fifth, at 2.23×, after `curve-1e-2`, `ct-1e-4`, `curve-1e-4` and `ct-1e-5`. It moves most against ε\*, 1.90 times it, where it was the realization closest to ε\* on the Code civil (1.17× inside).

**The flips are near-ties, not window 261.** Every flip on the Code pénal of a realization certified on the Code civil is at one of the two smallest margins of the exact model:
- position (454, 127), margin 1.07·10⁻⁵, "Lorsque la|": the exact `n` against a space, flipped by gaps of 1.0 to 1.3·10⁻⁵ there;
- position (14, 233), margin 2.6·10⁻⁶ (`ct-1e-4` only), against 1.8·10⁻⁵ for the Code civil's smallest.

These gaps are typical: D163 §5's median position is at 1.08·10⁻⁵ on the Code pénal. The bound counts both positions, and each realization flips at most its bound.

## What it says

- **The certificate is a statement about the text it was measured on.** At D = 64 it carries to a second text of the same kind: every certified realization stays inside ε\*, and the gaps move less than within the Code civil.
- **At the course constants it does not carry.** One stretch of the Code pénal, one window of 461, takes the realization solved at ε\* to 1.9 ε\*. The stretch is ill-conditioned for every type, and LayerNorm's ε in `bf16` amplifies it. Near-ties of the exact model, smaller than any on the Code civil, flip one decision in 118 016, inside δ but not none.
- **No new type, solve or calibration was tried.** If Y1 or Y2 failed, the user decides what comes next (the prompt's §6). One remedy is not run here: an envelope that holds the Code pénal's window 261 would see this position.

## How to run it again

```sh
make second-corpus                                      # every stage below, in order
experiments/second_corpus.sh data d64                   # the text, its windows, D = 64 (six minutes)
experiments/second_corpus.sh exact course window report # the course constants (three hours and a quarter)
```

The stages reuse the outputs that exist. The report stage writes `out/second-corpus/d64.md` and `course.md`, with each realization's windows (`windows-d64-*.tsv`, `windows-course-*.tsv`) and each scale's ten worst windows decoded. The window stage writes `out/second-corpus/w261/report.md`.
