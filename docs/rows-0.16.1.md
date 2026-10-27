# The rows of record at 0.16.1 (plan 0.16.1, step 2)

The rows that the two rules of D169 touch were solved again, and not timed.
The binary is the rules' tree (`out/bin/presk-0.16.1-rules`, presk !137).
The envelopes of record are 0.16.0's (D163), reused: the rules change the
solve, not the envelope. `make precision` calibrates its own, as at every
version. The rows ran side by side from 18:47 to 21:52 on 2026-10-22 by
`out/step2-0.16.1/phaseA.sh`, D163's phase A less what the rules do not
touch.
- Every output a script writes at a fixed path was first moved to
  `out/pre-0.16.1/`. These are 0.16.0's, archived in presk-project
  `data/rows-0.16.0.tar.xz`.
- The exact logits, the uniform rows, the integer rows and v0.10.0's curve
  are linked from the directories of record. `make margins` and
  `margins_course.sh` therefore ran only the new realizations, on the Code
  civil's 461 held-out windows and on the Code pénal's first 461 (D168's
  judge set).

The figures are read by `experiments/rows_0161.py`
(`out/step2-0.16.1/rows.md`) and by D168's `second_corpus.py report`
(`out/step2-0.16.1/penal-{d64,course}.md`). ε\* is the Code civil's at
δ = 10⁻³, corrected: 4.670·10⁻⁴ at D = 64 and 4.961·10⁻⁴ at the course
constants. A gap is the largest elementwise gap of the native logits to the
exact ones over a text's 461 windows. *Certified* means the gap on the Code
civil is at most ε\*.

**The two rules, in one line each** (D169):
- *Rule 1.* A value of one element is not offered.
- *Rule 2.* A value is offered only when its first narrowing, `f64` to
  `f32`, saves at least 10⁻³ of the entry's all-`f64` traffic.

A value not offered keeps `f64`, and the `f32` nests that read it round it
at their edge.

## 0. A correction to the plan's facts

The plan's §2 counted 16 values left out on the course logits (and on V6),
from 0.16.0's solve listings. The listing's column of elements is the
envelope's: four windows, for an activation. The cost and the rules read one
window's (`per_sample`, D157). Read per window, three more stored values
fall under θ·B on the course logits:
- LayerNorm's two other statistics (`LayerNorm.apply/%5`, `%10`), 256
  elements each, a first step of 1 024 bytes;
- the logits themselves (`GPT.apply/%24`), 23 552 elements, a first step
  of 9.42·10⁴ bytes, 0.82 θ·B.

So 19 values are left out there, and 18 on V6. The rules are the plan's,
unchanged; only the forecast of what they remove was short. The listing
keeps the envelope's counts, as at 0.16.0, so that it reads the same.

## 1. What the rules leave out (Z1)

On every row, no value of one element is offered. Every value rule 2
leaves at `f64` is named in the solve's line `rules`, with its first
step's gain, which is below θ·B.

| row | rule 1 | rule 2 | θ·B (bytes) | largest gain left out / θ·B | forgone at most (their saving in `bf16`) | of the traffic | what 0.16.0's types saved on them |
|---|---|---|---|---|---|---|---|
| V6 | 3 | 15 | 7 512 | 0.27 | 1.54·10⁴ | 0.40 % | 1.03·10⁴ |
| the course logits | 3 | 16 | 1.150·10⁵ | 0.82 | 3.30·10⁵ | 0.57 % | 2.32·10⁵ |
| the trained curve at D = 64 | 4 | 16 | 7 514 | 0.27 | 1.69·10⁴ | 0.29–0.75 % | 0.74–1.59·10⁴ |
| the trained course curve | 4 | 17 | 1.150·10⁵ | 0.82 | 3.32·10⁵ | 0.29–0.96 % | 1.28–2.77·10⁵ |
| D129's course curve | 4 | 18 | 1.424·10⁵ | 0.99 | 6.21·10⁵ | 0.44–0.95 % | 1.10–4.76·10⁵ |
| `make precision` | 4 | 16 | 7 514 | 0.27 | 1.69·10⁴ | 0.45 % | — |

The fourth value of one element in the loss's rows is the loss's mean
(`loss/%7`). **Z1 holds.** The prompt expected under 1 % of the traffic, and
the largest share forgone is 0.96 %, at the trained course curve's 10⁻¹.

## 2. The curve at D = 64 on the trained envelope (Figure 4, H2): D163 → D170

`experiments/curve_trained.sh`, then its quality axis (`QUALITY=1`).

| budget | 0.16.0: narrowed (bf16), weights (bf16) | traffic | gap | 0.16.1: narrowed (bf16), weights (bf16) | traffic | gap |
|---|---|---|---|---|---|---|
| 10⁻¹ | 27 (21), 21 (21) | 3.35× | 6.261·10⁻³ | 19 (15), 9 (9) | 3.33× | 1.778·10⁻⁴ |
| 10⁻² | 27 (7), 21 (17) | 2.52× | 1.448·10⁻⁵ | 19 (4), 9 (5) | 2.51× | 6.487·10⁻⁵ |
| 10⁻³ | 27 (3), 21 (12) | 2.04× | 4.948·10⁻⁶ | 19 (0), 9 (1) | **2.03×** | 3.369·10⁻⁷ |
| 10⁻⁴ | 26 (3), 21 (8) | 2.00× | 7.740·10⁻⁷ | 19 (0), 9 (0) | 1.99× | 1.242·10⁻⁸ |
| 10⁻⁵ | 24 (2), 21 (3) | 1.97× | 1.796·10⁻⁷ | 18 (0), 9 (0) | 1.96× | 1.649·10⁻⁸ |
| 10⁻⁶ | 21 (2), 21 (1) | 1.68× | 2.972·10⁻⁷ | 15 (0), 9 (0) | 1.67× | 2.275·10⁻⁸ |
| 10⁻⁷ | 8 (1), 21 (1) | 1.31× | 1.855·10⁻⁹ | 5 (0), 9 (0) | 1.30× | 1.576·10⁻⁹ |

- **Every point is inside its budget, with no widening round.** H2 holds:
  at 10⁻³ the gap is 3.37·10⁻⁷ against 2.79·10⁻³.
- **Traffic.** At every budget it is within 0.02× of 0.16.0's ratio. That
  is the 0.3–0.75 % the rules forgo.
- **The curve's bottom.** From 10⁻⁴ down no value is in `bf16`. 0.16.0's
  non-monotone 10⁻⁵ to 10⁻⁷ came from the head width and ε in `bf16`, and
  is gone: the traffic falls monotonically, from 3.33× to 1.30×. The last
  point is `f32` but for the values the rules keep at `f64`. It is not
  all-`f32`.
- **The gaps on the envelope** are smaller at every budget but 10⁻², by up
  to 62× (10⁻⁴). The measured gap is not monotone below 10⁻³: 1.24·10⁻⁸,
  1.65·10⁻⁸, 2.28·10⁻⁸, 1.58·10⁻⁹.
- **The quality axis.** The validation loss is within 1.71·10⁻⁴ of
  all-`f64`'s 2.7736101 at 10⁻¹ (0.16.0: 1.28·10⁻⁴). It is within
  1.1·10⁻⁶ at 10⁻². From 10⁻³ to 10⁻⁷ it equals all-`f64`'s to the seven
  digits printed (0.16.0: within 6.5·10⁻⁶).

On the two texts (`make margins` and D168's protocol):

| budget | 0.16.0: Code civil gap, flips | cert. | Code pénal gap, flips | ratio | 0.16.1: Code civil gap, flips | cert. | Code pénal gap, flips | ratio |
|---|---|---|---|---|---|---|---|---|
| 10⁻¹ | 4.126·10⁻², 518 | no | 4.310·10⁻², 435 | 1.04× | 4.028·10⁻², 475 | no | 3.419·10⁻², 410 | 0.85× |
| 10⁻² | 2.181·10⁻², 67 | no | 2.425·10⁻², 46 | 1.11× | 2.328·10⁻², 78 | no | 2.725·10⁻², 52 | 1.17× |
| 10⁻³ | 2.551·10⁻³, 9 | no | 2.176·10⁻³, 9 | 0.85× | 1.169·10⁻³, 0 | no | 1.216·10⁻³, 2 | 1.04× |
| 10⁻⁴ | 3.467·10⁻⁴, 0 | yes | 3.563·10⁻⁴, 0 | 1.03× | 3.090·10⁻⁶, 0 | yes | 3.005·10⁻⁶, 0 | 0.97× |
| 10⁻⁵ | 1.623·10⁻⁴, 0 | yes | 1.190·10⁻⁴, 0 | 0.73× | 2.883·10⁻⁶, 0 | yes | 3.081·10⁻⁶, 0 | 1.07× |
| 10⁻⁶ | 7.644·10⁻⁵, 0 | yes | 4.975·10⁻⁵, 0 | 0.65× | 2.948·10⁻⁶, 0 | yes | 2.847·10⁻⁶, 0 | 0.97× |
| 10⁻⁷ | 6.207·10⁻⁷, 0 | yes | 7.797·10⁻⁷, 0 | 1.26× | 7.339·10⁻⁷, 0 | yes | 7.817·10⁻⁷, 0 | 1.07× |

From 10⁻⁴ to 10⁻⁶ the gap on the 461 windows falls 26 to 112 times, to
all-`f32`'s 3.1·10⁻⁶, and no bound is left (#{m < 2 gap} = 0). At 10⁻³ it is
1.169·10⁻³ (2.50 ε\*), not certified, with no flip and a bound of 243 on the
Code civil, where 0.16.0's was 2.551·10⁻³ (5.5 ε\*) with 9 flips and a bound
of 525. On the Code pénal it is 1.216·10⁻³ with 2 flips and a bound of 205
(0.16.0: 2.176·10⁻³, 9 flips, 371).

## 3. The course curve on the trained envelope (Figure 5) → D170

`experiments/curve_course_trained.sh`, `JOBS=7`.

| budget | 0.16.0: narrowed (bf16), weights (bf16) | traffic | gap | 0.16.1: narrowed (bf16), weights (bf16) | traffic | gap |
|---|---|---|---|---|---|---|
| 10⁻¹ | 25 (6), 21 (16) | 3.34× | 1.049·10⁻⁴ | 18 (3), 9 (6) | 3.31× | 1.055·10⁻⁴ |
| 10⁻² | 25 (3), 21 (12) | 2.05× | 4.955·10⁻⁶ | 17 (0), 9 (1) | 2.04× | 9.278·10⁻⁶ |
| 10⁻³ | 25 (3), 21 (11) | 1.99× | 2.198·10⁻⁶ | 17 (0), 9 (0) | 1.98× | 4.976·10⁻⁹ |
| 10⁻⁴ | 24 (3), 21 (6) | 1.99× | 2.907·10⁻⁷ | 17 (0), 9 (0) | 1.98× | 4.976·10⁻⁹ |
| 10⁻⁵ | 22 (3), 21 (1) | 1.96× | 1.862·10⁻⁷ | 15 (0), 9 (0) | 1.95× | 1.049·10⁻⁸ |
| 10⁻⁶ | 3 (2), 17 (1) | 1.51× | 8.157·10⁻¹⁰ | 0 (0), 5 (0) | 1.51× | 3.830·10⁻¹⁰ |
| 10⁻⁷ | 5 (2), 12 (1) | 1.00× | 3.144·10⁻¹⁰ | 1 (0), 0 (0) | 1.00× | 7.376·10⁻¹¹ |

| budget | 0.16.0: Code civil gap, flips | cert. | Code pénal gap, flips | ratio | 0.16.1: Code civil gap, flips | cert. | Code pénal gap, flips | ratio |
|---|---|---|---|---|---|---|---|---|
| 10⁻¹ | 9.788·10⁻², 47 | no | 8.591·10⁻², 69 | 0.88× | 1.031·10⁻¹, 59 | no | 9.100·10⁻², 71 | 0.88× |
| 10⁻² | 1.498·10⁻², 29 | no | 1.924·10⁻², 31 | 1.28× | 1.547·10⁻², 12 | no | 1.494·10⁻², 11 | 0.97× |
| 10⁻³ | 6.518·10⁻³, 29 | no | 1.374·10⁻², 35 | 2.11× | 8.695·10⁻⁶, 0 | **yes** | 2.238·10⁻⁵, 0 | 2.57× |
| 10⁻⁴ | 2.627·10⁻⁴, 0 | yes | 7.173·10⁻⁴, 2 | 2.73× | 8.695·10⁻⁶, 0 | yes | 2.238·10⁻⁵, 0 | 2.57× |
| 10⁻⁵ | 2.059·10⁻⁴, 0 | yes | 5.070·10⁻⁴, 1 | 2.46× | 1.292·10⁻⁵, 0 | yes | 1.780·10⁻⁵, 0 | 1.38× |
| 10⁻⁶ | 1.093·10⁻⁶, 0 | yes | 8.482·10⁻⁷, 0 | 0.78× | 5.845·10⁻⁷, 0 | yes | 3.817·10⁻⁷, 0 | 0.65× |
| 10⁻⁷ | 6.263·10⁻⁷, 0 | yes | 1.340·10⁻⁶, 0 | 2.14× | 1.161·10⁻⁷, 0 | yes | 1.541·10⁻⁷, 0 | 1.33× |

At 10⁻³ the course curve is now certified: it measures 8.70·10⁻⁶ with no
flip, where 0.16.0's flipped 29 decisions at 6.52·10⁻³ (13 ε\*). The
budgets 10⁻³ and 10⁻⁴ give the same realization, byte for byte, and its
gaps on both texts are the logits' of §6 to the digit.

## 4. The course curve on the initialization's envelope (D129) → D170

`experiments/curve_course.sh`, `JOBS=7`.

| budget | 0.16.0: narrowed (bf16), weights (bf16) | traffic | gap | 0.16.1: narrowed (bf16), weights (bf16) | traffic | gap |
|---|---|---|---|---|---|---|
| 10⁻¹ | 26 (10), 21 (11) | 2.20× | 2.118·10⁻² | 18 (4), 7 (0) | 2.18× | 2.169·10⁻³ |
| 10⁻² | 26 (3), 21 (11) | 2.00× | 3.386·10⁻⁴ | 18 (0), 7 (0) | 1.99× | 7.804·10⁻⁸ |
| 10⁻³ | 25 (3), 21 (9) | 1.98× | 4.203·10⁻⁵ | 17 (0), 7 (0) | 1.97× | 4.258·10⁻⁸ |
| 10⁻⁴ | 22 (3), 21 (1) | 1.96× | 1.840·10⁻⁶ | 16 (0), 7 (0) | 1.95× | 2.979·10⁻⁹ |
| 10⁻⁵ | 14 (2), 15 (1) | 1.71× | 2.794·10⁻⁶ | 9 (0), 6 (0) | 1.71× | 4.263·10⁻⁸ |
| 10⁻⁶ | 9 (2), 11 (1) | 1.07× | 2.794·10⁻⁶ | 3 (0), 0 (0) | 1.07× | 4.380·10⁻¹⁰ |
| 10⁻⁷ | 4 (2), 9 (1) | 1.00× | 2.524·10⁻⁸ | 0 (0), 0 (0) | 1.00× | 1.066·10⁻¹⁴ |

This envelope has no data file: one batch of two random sequences, priced
at that batch. So B is larger (1.424·10⁸ bytes, θ·B = 1.424·10⁵), the
weights weigh less against it, and `tok` and `head` fall under θ·B as well
(a first step of 1.413·10⁵ bytes, 0.99 θ·B). Seven weights are offered, not
nine.

| budget | 0.16.0: Code civil gap, flips | cert. | Code pénal gap, flips | ratio | 0.16.1: Code civil gap, flips | cert. | Code pénal gap, flips | ratio |
|---|---|---|---|---|---|---|---|---|
| 10⁻¹ | 9.710·10⁻², 89 | no | 1.049·10⁻¹, 88 | 1.08× | 6.890·10⁻², 93 | no | 9.765·10⁻², 83 | 1.42× |
| 10⁻² | 3.169·10⁻², 10 | no | 8.931·10⁻², 14 | 2.82× | 8.859·10⁻⁶, 0 | **yes** | 2.212·10⁻⁵, 0 | 2.50× |
| 10⁻³ | 2.629·10⁻⁴, 0 | yes | 4.480·10⁻⁴, 0 | 1.70× | 8.676·10⁻⁶, 0 | yes | 2.241·10⁻⁵, 0 | 2.58× |
| 10⁻⁴ | 2.043·10⁻⁴, 0 | yes | 5.072·10⁻⁴, 1 | 2.48× | 1.230·10⁻⁵, 0 | yes | 1.953·10⁻⁵, 0 | 1.59× |
| 10⁻⁵ | 5.074·10⁻⁶, 0 | yes | 6.375·10⁻⁶, 0 | 1.26× | 5.043·10⁻⁶, 0 | yes | 6.300·10⁻⁶, 0 | 1.25× |
| 10⁻⁶ | 5.968·10⁻⁶, 0 | yes | 5.636·10⁻⁶, 0 | 0.94× | 5.841·10⁻⁶, 0 | yes | 5.519·10⁻⁶, 0 | 0.94× |
| 10⁻⁷ | 3.911·10⁻⁹, 0 | yes | 3.731·10⁻⁹, 0 | 0.95× | 2.164·10⁻¹⁴, 0 | yes | 3.273·10⁻¹⁴, 0 | 1.51× |

## 5. V6 and `make margins` at D = 64 (H5) → D170

V6 is the logits solved at ε\*(10⁻³) = 4.67·10⁻⁴ on 0.16.0's envelope of
four held-out windows (`out/v6-0.16.1/`).

| | 0.16.0 (D163) | 0.16.1 |
|---|---|---|
| stored values | 22 of 25 (20 `f32`; ε and the head width in `bf16`) | 18 of 25 (18 `f32`), 6 not offered |
| weights | 21 (16 `f32`, 5 `bf16`) | 9 of 21 (9 `f32`), 12 not offered |
| traffic | 1.97× | **1.96×** |
| estimate | 4.620·10⁻⁴ | 5.447·10⁻⁵ |
| measured on the envelope's four windows | 7.372·10⁻⁵ | 2.138·10⁻⁶ |
| on the Code civil's 461 | 1.431·10⁻⁴, 0 flips, bound 32 | **2.883·10⁻⁶**, 0 flips, bound 0 |
| on the Code pénal's 461 | 1.016·10⁻⁴, 0 flips, bound 11 | 3.081·10⁻⁶, 0 flips, bound 0 |
| certified at 10⁻³ | yes (0.31 of ε\*) | yes (0.006 of ε\*) |

The rules leave out 18 values. Sixteen of them 0.16.0 had narrowed, and
the realization is 0.16.0's with those 16 moved to `f64`; no other decision
moved. The other two, LayerNorm's `%5` and `%10`, were among the three
stored values 0.16.0's solve already left at `f64`. The third,
`LayerNorm.apply/%4` (`pow 2 %3`, the squared deviations whose mean is the
variance), is offered and stays at `f64`, as at 0.16.0. Without those values
nothing narrower is cheap enough: every other value is `f32`, and any
`bf16` left on the menu costs more than the budget has. The solver spends
an eighth of it (5.4·10⁻⁵ of 4.67·10⁻⁴). The gap on the 461 windows is
all-`f32`'s, 3.1·10⁻⁶.

## 6. The course GPT through the margins (H5 at course scale) → D170

The logits solved at ε\*(10⁻³) = 4.96·10⁻⁴ on 0.16.0's envelope
(`out/margins-course-0.16.1/logits/`): 1 h 48 min and 2.9 GB, as at 0.16.0.

| | 0.16.0 (D163) | 0.16.1 |
|---|---|---|
| stored values | 22 of 25 (19 `f32`, 3 `bf16`) | 17 of 25 (17 `f32`), 7 not offered |
| weights | 21 (18 `f32`, 3 `bf16`) | 9 of 21 (9 `f32`), 12 not offered |
| traffic | 1.99× | **1.98×** |
| estimate | 4.067·10⁻⁴ | 2.583·10⁻⁴ |
| measured on the envelope's four windows | 5.548·10⁻⁵ | 3.017·10⁻⁶ |
| on the Code civil's 461 | 4.228·10⁻⁴ (7.6×), 0 flips, bound 82 | **8.695·10⁻⁶** (2.9×), 0 flips, bound 0 |
| on the Code pénal's 461 | 9.410·10⁻⁴ (1.90 ε\*), 1 flip, bound 228 | **2.238·10⁻⁵** (0.045 ε\*), 0 flips, bound 4 |
| certified at 10⁻³ | yes (0.85 of ε\*) | yes (0.018 of ε\*) |

No value is in `bf16`. The realization is `f32` but for the 19 values the
rules keep at `f64` and one value offered that the solver leaves there,
`LayerNorm.apply/%4` (`pow 2 %3`, the squared deviations), at `f64` in
0.16.0's too. Its
gap on the Code civil is half all-`f32`'s (1.755·10⁻⁵). On the Code pénal
it is 1.55× all-`f32`'s (1.447·10⁻⁵), in window 261, D168's window.

**What carries window 261** (`out/step2-0.16.1/w261/`). D168's chunk
(windows 256 to 287) was run natively under the realization. It was run
again with each value the rules keep at `f64` and 0.16.0 had narrowed put
back to `f32`, alone, and with all of them together:

| at `f32` | gap at window 261, position 176 | the rest of window 261 | the chunk's other windows |
|---|---|---|---|
| none (0.16.1's realization) | 2.238·10⁻⁵ | 9.288·10⁻⁶ | 6.881·10⁻⁶ |
| any one of the twelve vector parameters, the head width or its root | 2.238·10⁻⁵ | 9.288·10⁻⁶ | 6.881·10⁻⁶ |
| ε (`LayerNorm.apply/%8`) | 2.209·10⁻⁵ | 9.244·10⁻⁶ | 6.881·10⁻⁶ |
| the logits (`GPT.apply/%24`) | 2.258·10⁻⁵ | 1.014·10⁻⁵ | 9.356·10⁻⁶ |
| LayerNorm's mean (`LayerNorm.apply/%2`) | 1.739·10⁻⁵ | 6.764·10⁻⁶ | 4.164·10⁻⁶ |
| all seventeen | 1.447·10⁻⁵ | 8.234·10⁻⁶ | 9.145·10⁻⁶ |

A vector parameter's type changes no bit here. Every nest that reads one is
`f32`, and it rounds the value at its edge whatever the value's type at
rest. Rule 2 on parameters therefore costs traffic and changes nothing
else. What moves the worst position are LayerNorm's statistics: stored in
`f64` and read by `f32` nests, they change how the window's
ill-conditioned position is rounded (D168: ill-conditioned in every type).
All seventeen together give all-`f32`'s 1.447·10⁻⁵ to the digit. The gap is
`f32`'s rounding, 1.5× above or 2× below all-`f32`'s, depending on which
statistics are stored wide; it is no longer `bf16`'s.

## 7. §2's realization and Figure 2 (`make precision`) → D170

`make precision` (`out/paper-0.16.1/precision.txt`, `make -o build`),
then `presk explain` on the loss at 10⁻³ (`out/paper-0.16.1/explain-gpt/`).

| | 0.16.0 (D163) | 0.16.1 |
|---|---|---|
| stored values | 27 of 27 (23 `f32`, 4 `bf16`) | 19 of 27 (18 `f32`, 1 `bf16`), 8 not offered |
| weights | 21 (10 `f32`, 11 `bf16`) | 9 of 21 (9 `f32`), 12 not offered |
| traffic | 3.7210·10⁶ against 7.5141·10⁶, 2.02× | 3.7356·10⁶ against 7.5141·10⁶, **2.01×** |
| estimate | 2.446·10⁻³, 87 % of 2.805·10⁻³ | 2.018·10⁻³, 72 % |
| measured | 1.249·10⁻⁶ | 3.547·10⁻⁶ |

The one stored value in `bf16` is `SelfAttention.apply/%9`, as at 0.16.0.
The three others there (ε, the head width and its root) are rule 1's. The
explain document gives the 20 values left out their rule as their reason,
and its realization line reads "19 stored values away from `f64`: 1 in
`bf16`, 18 in `f32`; …; 9 parameters stored in their own types". The
measured gap is larger than 0.16.0's, still 790 times inside the budget.
The budget the rules free goes elsewhere, and the measurement sees a
different realization.

## 8. The quant and step tables (H4, Figure 6)

Byte-identical to `v0.16.0`'s, and so are the step's realizations of
record (D169, Z0). The training step's solver is outside the rules.

## 9. The certificate rows on both texts (Z2), and D168's table redone (Z4)

**Z2.** V6 meets the four conditions:
1. it is inside its budget on its envelope, at 2.14·10⁻⁶;
2. it is certified on the Code civil, at 2.88·10⁻⁶ with no flip;
3. it is inside ε\* with no flip on the Code pénal, at 3.08·10⁻⁶;
4. out/in, the realization held fixed against N (`off_sample_fixed.py`,
   `out/step2-0.16.1/offsample/`), is at most 1.12× on the Code civil and
   1.24× on the Code pénal from 16 windows on. The ratio of the two texts'
   gaps is 1.07×.

The course logits meet (i), (ii) and (iii):
1. 3.02·10⁻⁶;
2. 8.70·10⁻⁶, no flip;
3. 2.24·10⁻⁵, no flip.

They fail (iv) on both of its parts:
- **out/in on the Code civil:** 1.92× at 16 windows, 1.84× at 32 and 64,
  1.08× from 128.
- **out/in on the Code pénal:** 7.28× at 16 and 3.08× at 32, window 261
  against the first windows.
- **the ratio of the two texts' gaps:** 2.57×.

**Z2 fails, on (iv), for the course logits.** For comparison, all-`f32`'s
own out/in on the Code civil at the course constants is 1.80× from 16
windows and 1.74× at 64, and its gap 1.755·10⁻⁵. The 1.5× of (iv) is D129's
level for a realization with `bf16` in it. A realization near `f32` has the
spread of `f32`'s rounding across windows, at a gap 57 times smaller than
ε\*.

**D168's table with 0.16.1's realizations** (`out/step2-0.16.1/penal-*.md`):

| | 0.16.0 (D168) | 0.16.1 |
|---|---|---|
| D = 64, Y1 | holds (12 of 12) | **holds** (12 of 12) |
| D = 64, Y2 | holds (0.54×–1.31×, median 0.85×) | **holds** (0.54×–1.31×, median 0.97×) |
| course, Y1 | fails (4 of 12 above ε\*, 1–2 flips each) | **holds** (14 of 14 certified inside ε\*, no flip) |
| course, Y2 | fails (7 of 22 above 2×, up to 2.82×) | **fails** (5 of 22 above 2×, up to 2.58×) |

At course scale the five rows that fail Y2 are the logits, the trained
curve's 10⁻³ and 10⁻⁴, and D129's 10⁻² and 10⁻³. All five measure
2.2·10⁻⁵ on the Code pénal, at window 261, against 8.7·10⁻⁶ on the Code
civil. Every realization's worst window on the Code pénal at course scale
is 261, all-`f64`'s included.
- The solved realizations' median window is 2.0·10⁻⁶ on the Code pénal, a
  quarter of all-`f32`'s 8.5·10⁻⁶.
- Their worst window is 11 times their median, where all-`f32`'s is 1.7
  times.

## 10. What stands

- **The rules do what they say.** They remove from every row the values
  without gain (Z1), for 0.3–1 % of the traffic. The ratios of record move
  by at most 0.03×: the course logits 1.98×, V6 1.96×, the curve at 10⁻³
  2.03×, `make precision` 2.01×.
- **The tail of `bf16` decisions is gone.** It carried the certificate's
  gaps at both scales (steps 0 to 0d, D168).
  - At D = 64 V6 measures all-`f32`'s gap.
  - At the course constants the logits measure half all-`f32`'s on the
    Code civil, 49 times smaller than 0.16.0's. On the Code pénal they
    measure 2.2·10⁻⁵ where 0.16.0's measured 1.90 ε\*.
- **Y1 now holds on the Code pénal at both scales.** Y2 holds at D = 64.
- **What fails is relative, at `f32`'s level.** At course scale Y2 and
  Z2 (iv) fail. In absolute terms every certified realization is at least
  22 times inside ε\* on both texts, and flips nothing.
- **The rows not solved again** (the decoder, BERT's two blocks, the timed
  rows, J′) stay 0.16.0's (the plan's §6).
