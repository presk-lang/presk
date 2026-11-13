# The rows of record at 0.16.0 (plan 0.16.0, step 5)

Every row of the plan's §5 is solved, timed or checked again by its whole
protocol and its script of record, on the tree of steps 1 to 4 (presk
`22dd345`, the binary `out/bin/presk-22dd345`). Each section gives the old
value beside the new and the decision numbers of both. The
changes that move a row are the plan's letters:
- **R**, the measured rounding term as the default (D155, D158). A value
  exact in a type costs nothing there.
- **C**, the traffic of one sample (D158): the activations at one window's
  elements and the weights once, so the weights' share of the bytes grows
  and the ratios move.
- **E**, the kernel interpreter on the native backend's bits (D159). The
  solver's measurement is the native gap.
- **M**, no `dropout_mask` outside a training run (D160). The masks' stored
  values leave the menu and the traffic.

The Llama 2 decoder is its own report, `docs/compile-times-2026-10-19-llama.md`
(D162). This file holds the rows that are solved and not timed (D163), run
side by side from 19:48 to 23:15 on 2026-10-20 by
`out/step5-0.16.0/phaseA.sh`. Every output a script writes at a fixed path
was first moved to `out/pre-0.16.0/`. The rows no 0.16.0 change moves
(all-`f64`, all-`f32`, all-`bf16`, the integer weights, v0.10.0's curve) and
the exact logits are linked from `out/margins` and `out/margins-course`, so
`make margins` ran only the new realizations.

## 1. The curve at D = 64 on the trained envelope (statement D): D132 → D163

`experiments/curve_trained.sh`, then its quality axis (`QUALITY=1`, D132's
addendum: every realization trains the model again, 600 steps). The
outputs are `out/curve-trained_*` and `out/curve-trained.md`.

| budget | D132: narrowed (bf16), weights (bf16) | traffic | gap | 0.16.0: narrowed (bf16), weights (bf16) | traffic | gap |
|---|---|---|---|---|---|---|
| 10⁻¹ | 33 (24), 21 (21) | 3.17× | 3.570·10⁻⁵ | 27 (21), 21 (21) | 3.35× | 6.261·10⁻³ |
| 10⁻² | 33 (8), 21 (12) | 2.44× | 1.651·10⁻⁵ | 27 (7), 21 (17) | 2.52× | 1.448·10⁻⁵ |
| 10⁻³ | 33 (4), 21 (11) | 2.12× | 1.603·10⁻⁵ | 27 (3), 21 (12) | **2.04×** | 4.948·10⁻⁶ |
| 10⁻⁴ | 32 (2), 21 (9) | 1.97× | 6.565·10⁻⁷ | 26 (3), 21 (8) | 2.00× | 7.740·10⁻⁷ |
| 10⁻⁵ | 28 (1), 21 (2) | 1.92× | 1.796·10⁻⁷ | 24 (2), 21 (3) | 1.97× | 1.796·10⁻⁷ |
| 10⁻⁶ | 19 (0), 19 (1) | 1.52× | 4.418·10⁻⁹ | 21 (2), 21 (1) | 1.68× | 2.972·10⁻⁷ |
| 10⁻⁷ | 8 (0), 10 (1) | 1.15× | 1.738·10⁻⁹ | 8 (1), 21 (1) | 1.31× | 1.855·10⁻⁹ |

- **Every point is still inside its budget, with no widening round.** D
  holds: at 10⁻³ the gap is 4.95·10⁻⁶ against 2.79·10⁻³.
- **Six stored values fewer are offered (33 → 27):** the four dropout masks
  and two products by a mask (M). Every other stored value keeps its name.
- **The traffic ratios move with C.** The activations are priced at one
  window's elements and the weights once, so the weights' share of the
  bytes grows.
- **The bottom of the curve is no longer monotone, because of R.** Two
  stored values are exact or nearly exact in `bf16`:
  - `SelfAttention.apply/%15`, the head width 32 as a real. It is exact,
    so the measured term charges it nothing, and it stays in `bf16` at
    every budget.
  - LayerNorm's ε, 10⁻⁵ (`LayerNorm.apply/%8`). Its rounding in `bf16` is
    1.4·10⁻⁸, so it stays there down to 10⁻⁶ and leaves at 10⁻⁷, where
    D132 had it leave at 10⁻⁶.

  The gap is 1.80·10⁻⁷ at 10⁻⁵, 2.97·10⁻⁷ at 10⁻⁶ and 1.86·10⁻⁹ at 10⁻⁷.
- **The bound** on all-`f64` (`cond`) is 1.5947·10⁻⁶ (D132: 1.6011·10⁻⁶).
  The masks' terms are gone.
- **The quality axis is as flat as D132's.** The validation loss is within
  1.28·10⁻⁴ of all-`f64`'s 2.7736101 at 10⁻¹, the same figure, and within
  6.5·10⁻⁶ everywhere else (+4.0·10⁻⁶ at 10⁻², −6.5·10⁻⁶ at 10⁻³ and 10⁻⁴,
  +1.3·10⁻⁶ at 10⁻⁵ and 10⁻⁶, 0 at 10⁻⁷). All-`f64`'s run is the same.

## 2. The curve at the course constants on the trained envelope (D132) → D163

`experiments/curve_course_trained.sh` (`JOBS=7`), `out/curve-course-trained*`.

| budget | D132: narrowed (bf16), weights (bf16) | traffic | gap | 0.16.0 | traffic | gap |
|---|---|---|---|---|---|---|
| 10⁻¹ | 33 (8), 21 (16) | 2.68× | 6.862·10⁻⁶ | 25 (6), 21 (16) | 3.34× | 1.049·10⁻⁴ |
| 10⁻² | 30 (7), 21 (11) | 2.05× | 9.112·10⁻⁶ | 25 (3), 21 (12) | 2.05× | 4.955·10⁻⁶ |
| 10⁻³ | 30 (3), 21 (11) | 1.98× | 2.198·10⁻⁶ | 25 (3), 21 (11) | **1.99×** | 2.198·10⁻⁶ |
| 10⁻⁴ | 29 (1), 21 (5) | 1.96× | 2.907·10⁻⁷ | 24 (3), 21 (6) | 1.99× | 2.907·10⁻⁷ |
| 10⁻⁵ | 26 (1), 21 (1) | 1.81× | 5.224·10⁻⁸ | 22 (3), 21 (1) | 1.96× | 1.862·10⁻⁷ |
| 10⁻⁶ | 8 (0), 11 (1) | 1.27× | 1.338·10⁻¹⁰ | 3 (2), 17 (1) | 1.51× | 8.157·10⁻¹⁰ |
| 10⁻⁷ | 6 (0), 11 (1) | 1.03× | 5.322·10⁻¹⁰ | 5 (2), 12 (1) | 1.00× | 3.144·10⁻¹⁰ |

Every point is inside its budget, with no widening round. At 10⁻³ and
10⁻⁴ the measured gap is D132's to four digits: the realization computes
the same values, the masks aside. The all-`f64` bound is 9.7665·10¹⁴
(9.7793·10¹⁴).

## 3. The curve at the course constants on the initialization's envelope (D129) → D163

`experiments/curve_course.sh` (`JOBS=7`), `out/curve-course*`. The
calibration is on random tokens at the initialization, the solves on the
trained weights.

| budget | D129: narrowed | traffic | gap | 0.16.0: narrowed | traffic | gap |
|---|---|---|---|---|---|---|
| 10⁻¹ | 32 | 2.19× | 2.772·10⁻³ | 26 | 2.20× | 2.118·10⁻² |
| 10⁻² | 32 | 2.04× | 1.221·10⁻⁴ | 26 | 2.00× | 3.386·10⁻⁴ |
| 10⁻³ | 30 | 1.98× | 1.021·10⁻⁶ | 25 | **1.98×** | 4.203·10⁻⁵ |
| 10⁻⁴ | 27 | 1.94× | 2.794·10⁻⁶ | 22 | 1.96× | 1.840·10⁻⁶ |
| 10⁻⁵ | 12 | 1.51× | 7.289·10⁻⁹ | 14 | 1.71× | 2.794·10⁻⁶ |
| 10⁻⁶ | 9 | 1.08× | 9.338·10⁻⁹ | 9 | 1.07× | 2.794·10⁻⁶ |
| 10⁻⁷ | 4 | 1.02× | 2.318·10⁻¹⁰ | 4 | 1.00× | 2.524·10⁻⁸ |

Every point is inside its budget. The bound is 2.5754·10⁻¹³, unchanged.

## 4. V6 and `make margins` at D = 64 (D124, D125, D132; S) → D163

**V6** is the logits solved at ε\*(10⁻³) = 4.67·10⁻⁴ on four held-out
windows (`out/v6-0.16.0/`):

| | D124 | 0.16.0 |
|---|---|---|
| stored values | 29 of 31 (28 `f32`, ε in `bf16`) | 22 of 25 (20 `f32`; ε and the head width in `bf16`) |
| weights | 21 (20 `f32`, `wk.b` in `bf16`) | 21 (16 `f32`, 5 `bf16`) |
| traffic | 1.97× | **1.97×** |
| estimate | 4.144·10⁻⁴ | 4.620·10⁻⁴ |
| measured on the envelope's four windows | 3.242·10⁻⁵ | 7.372·10⁻⁵ |
| on the 461 windows | 7.525·10⁻⁵, 0 flips, bound 18 | **1.431·10⁻⁴**, 0 flips, bound 32 |
| certified at 10⁻³ | yes | yes (0.31 of ε\*) |

**`make margins`** (`out/margins-0.16.0/margins.md`) ran on the curve of §1
(rows `d016-<b>`, in place of D132's `d132-<b>`) and on V6:

| row | D132 / D124: gap on the logits, flips, bound | 0.16.0 |
|---|---|---|
| 10⁻¹ | 4.126·10⁻², 518, 9 298 | 4.126·10⁻², 518, 9 298 |
| 10⁻² | 1.704·10⁻², 65, 3 706 | 2.181·10⁻², 67, 4 729 |
| 10⁻³ (D's) | 7.785·10⁻³ (17 ε\*), 30, 1 692 | **2.551·10⁻³ (5.5 ε\*), 9, 525** |
| 10⁻⁴ | 2.746·10⁻⁴, 0, 59 | 3.467·10⁻⁴, 0, 74 |
| 10⁻⁵ | 7.521·10⁻⁵, 0, 18 | 1.623·10⁻⁴, 0, 35 |
| 10⁻⁶ | 1.083·10⁻⁶, 0, 0 | 7.644·10⁻⁵, 0, 18 |
| 10⁻⁷ | 7.468·10⁻⁷, 0, 0 | 6.207·10⁻⁷, 0, 0 |

- **S holds as before.** Four of the curve's seven realizations, 10⁻⁴ and
  below, are certified at 10⁻³ and change no decision. Every one of the
  fifteen rows flips at most its bound.
- **D's realization is three times closer to the logits than D132's.**
  It is 5.5 ε\* and 9 flips, where D132's was 17 ε\* and 30. D132's stored
  the attention's dropout mask in `bf16`, which is gone (M). It keeps ε,
  the head width 32 and its square root in `bf16`.

## 5. The course GPT through the margins (D129; S at course scale) → D163

The logits are solved at ε\*(10⁻³) = 4.96·10⁻⁴ on four held-out windows
(`out/margins-course-0.16.0/logits/`). The calibration took 1 h 00 min
and 1.3 GB, the solve 1 h 49 min and 2.9 GB (D129: 51 min, 1 h 36 min and
3.1 GB). Then `margins_course.sh` ran the new realizations:
- the logits' solve;
- the course curve of §2 (`ct-<b>`);
- the course curve of §3 (`solved at <b>`).

| row | D129 / D132 | 0.16.0 |
|---|---|---|
| the logits' solve | 28 of 31, 21 weights (1 `bf16`), 1.97×; 4.344·10⁻⁵ in sample, 2.093·10⁻⁴ on the 461 (4.8×), 0 flips, bound 37 | 22 of 25, 21 weights (3 `bf16`), **1.99×**; 5.548·10⁻⁵ in sample, **4.228·10⁻⁴** on the 461 (7.6×), 0 flips, bound 82 |
| §2's 10⁻³ (`ct-1e-3`) | 6.523·10⁻³ (13 ε\*), 29 flips, bound 1 482 | 6.518·10⁻³ (13 ε\*), 29 flips, bound 1 481 |
| §2's 10⁻² | 3.169·10⁻², 10 flips | 1.498·10⁻², 29 flips |
| §2's 10⁻⁴ … 10⁻⁷ | certified, 0 flips | certified, 0 flips |
| §3's 10⁻³ (D129's curve row) | 2.819·10⁻⁴, 0 flips, bound 52, certified | 2.629·10⁻⁴, 0 flips, bound 47, certified |

S holds at course scale. Every certified realization changes no decision,
and every row flips at most its bound. The logits' solve is still inside
ε\* on the 461 windows, by 1.17× where D129's was inside by 2.4×. It moved
7.6× out of sample, where D129's moved 4.8×. Its three `bf16` stored values
are ε, the head width 64 and its square root 8 (`SelfAttention.apply/%15`
and `%16`, both exact in `bf16`), and its three `bf16` parameters the
biases of the queries and of the keys and the second norm's β.
*Corrected 2026-10-22: this paragraph said "the head width 32", D = 64's.
Step 0b (`off-sample-course-step0b-2026-10-22.md`) finds that the queries'
bias in `bf16` carries the 7.6×.*

## 6. `make precision`'s realization (D124) → D163

`make precision` (its `precision.txt`). Its `build`
prerequisite was skipped (`make -o build`, the binary given by `PRESK`).
Then `presk explain` on the loss at 10⁻³ (its `explain-gpt/`).

| | D124 | 0.16.0 |
|---|---|---|
| sites of the loss (`calibrate`) | 100 | 85 |
| stored values narrowed | 33 of 33 (29 `f32`, 4 `bf16`) | 27 of 27 (23 `f32`, 4 `bf16`) |
| weights | 21 (10 `f32`, 11 `bf16`) | the same |
| traffic | 9.0293·10⁶ against 1.9114·10⁷, 2.12× | 3.7210·10⁶ against 7.5141·10⁶, **2.02×** |
| estimate | 2.718·10⁻³, 97 % of 2.805·10⁻³ | 2.446·10⁻³, **87 %** |
| measured | 3.872·10⁻⁶ | **1.249·10⁻⁶** |
| bound `cond` | 3.0232·10⁶ | 3.0209·10⁶ |
| the estimate's largest share (explain §6.3) | the attention's dropout mask in `bf16`, 55.1 % | `SelfAttention.apply/%9` (`split_heads`) in `bf16`, **74.4 %** |

Under one type everywhere, the four rules' bounds move in their third or
fourth digit (the masks' terms gone). The measured gaps are the same.

## 7. The quant table (K) and the step table

Both are byte-identical to D124's:
- the quant table equals D124's (its archive's `runs/new/quant_table.md`)
  (statement K holds, int8 2.73·10⁻⁵);
- `step_table.md` equals the one in `out/pre-0.16.0/`.

The step's realizations `out/step_1e-3.json`, `step_1e-2.json` and
`step_recipe.json` are the same bytes too, and the step's calibration
differs only by its time. R2 (iii) holds on the version's tree: a training
step keeps its masks, and its solve is unchanged. So **J′ stands**,
and T2's C step runs the same realization.

## 8. D128: the measurement overruling the estimate (`make widening`) → D163

`OUT=out/widening-0.16.0 experiments/widening.sh`. The six programs have no
`dropout`. The measured term (R) lowers the estimates, so some proposals
move:

| program, budget | D128: estimate, measured, rounds, final, traffic | 0.16.0 |
|---|---|---|
| `attn_raw`, 10⁻³ | 6.32·10⁻⁶, 3.72·10⁻¹, 4 rounds to all-`f64`, 1.44·10⁻¹⁵, 1.00× | 3.89·10⁻⁶, 3.72·10⁻¹, **2 rounds** (`%5`, `%8` to `f64`), 2.87·10⁻⁸, 1.00× |
| `attn_softmax`, 10⁻³ | 5.53·10⁻⁶, 4.63·10⁻⁷, 0 rounds, 2.00× | 3.53·10⁻⁶, 4.63·10⁻⁷, 0 rounds, 2.00× |
| `attn_softmax`, 10⁻¹ | 1.46·10⁻³, 9.79·10⁻⁴, 0 rounds, 2.01× | 3.19·10⁻², 8.33·10⁻², **2 rounds** (`%6`, `%5` `bf16`→`f32`), 9.79·10⁻⁴, 2.01× |
| `std_two_pass`, 10⁻³ | 4.84·10⁻⁵, 4.25·10⁻², 5 rounds (μ `bf16`→`f32` last), 4.29·10⁻⁸, 1.00×; μ alone 6.11·10⁻⁶, 2.00× | 2.50·10⁻⁵, the rest the same |
| `std_one_pass`, 10⁻¹ / 3·10⁻² / 10⁻³ | 3 / 0 / 0 rounds | **2 / 2 / 0** rounds; at 3·10⁻² it ends at 6.04·10⁻⁶ and 1.32× (D128: 3.13·10⁻³, 1.96×, no round) |
| `clock`, 10⁻⁵ | 7.32·10⁻⁵, 1.58·10⁻², 1 round, 6.10·10⁻⁶, 2.00× | 1.83·10⁻⁵, the rest the same |
| `noisy`, 10⁻⁵ | 1.15·10⁻⁴, 3.80·10⁻⁴, 0 rounds | 6.24·10⁻⁵, 3.80·10⁻⁴, 0 rounds |

D128's findings hold at its budgets. Wherever
D128 found a refusal, the measurement still refuses the estimate's
proposal. The estimates are lower under R, and at the two largest budgets
the proposals narrow further. The measurement then refuses two that D128
accepted:
- `attn_softmax` at 10⁻¹, two rounds back to D128's realization;
- `std_one_pass` at 3·10⁻², two rounds, ending at 1.32×.

`attn_raw` widens two values to `f64` and stops at 2.87·10⁻⁸, where D128
widened four, to all-`f64`. The culprit's cost check holds on every final
realization.

## 9. D130: PyTorch's own gap at D = 64 (`make pytorch-gap`) → D163

`out/pytorch-gap-0.16.0/gaps.md`. PyTorch's rows (`float32` on ten threads
and on one, `float64`) and the compiler's uniform rows are the same digits.
The solved rows are 0.16.0's: V6 at 1.431·10⁻⁴ (D130: 7.525·10⁻⁵), and the
curve's 10⁻³, now §1's trained-envelope row, at 2.551·10⁻³. D130 had
`make curve`'s row from the initialization's envelope, 2.516·10⁻⁴.

## 10. What stands

- **The decoder against PyTorch (D107).** The decoder's PSK-Core, its
  optimized PSK-Core and its kernels at T = 384 are byte-identical between
  `454bb9f` and `22dd345` (D162).
- **J′ and `presk train`'s native step.** The step's realizations are
  byte-identical (§7, R2 (iii)).
- **The integer and uniform rows of `make margins`,** linked: no 0.16.0
  change moves a forward pass in `f64`, `f32` or `bf16` (D160 (ii)).
- **The Gappa proofs (D108, D131).** Each is a program of its own in
  uniform `f32`: the examples' `LayerNorm{D}.apply` and a `dot` of
  [8, K]·[K, 8]. Neither has a `dropout`, and the layer norm's only
  function is `sqrt`, which IEEE rounds correctly and the `libm` change of
  D159 does not touch. The nests' types are those of record, so the
  proofs stand.

## 11. D154: the precision-tuning tools on the exported C (`make precision-tuning`) → D165

`experiments/tuning/tuning.sh out/precision-tuning-0.16.0` (phase B, alone,
2026-10-21 08:24–08:46). The logits at D = 64. presk's space is now 25
stored values and 21 parameters (D154: 31 and 21), the four masks and two
products gone.

| | D154 (0.15.0) | 0.16.0 |
|---|---|---|
| the solver at ε\* | 29 of 31, 1.97×, 7.5·10⁻⁵ on the 461 | 22 of 25, 1.96×, 1.43·10⁻⁴, 0 flips |
| the searches (CRAFT's, Precimonious's) on stored values and parameters | 2.17×, 70 and 2 858 trials, 3.687·10⁻⁴, 0 flips, certified | **2.00×**, 64 and 1 972 trials, the same 3.687·10⁻⁴, 0 flips, certified |
| the searches on stored values alone | 2.11×, 43 and 716 trials | 1.93×, 37 and 246 trials |
| decisions where the solver differs from the searches | 16 (unit/off) | 10 (unit/off), 8 (measured/off) |
| at 10⁻⁶: the searches on the 461 | 1.41 to 1.69·10⁻⁶, over the budget | 1.41 to 1.72·10⁻⁶, over |
| CRAFT on the C program, 4.67·10⁻⁴: `bin_simple`, `ddebug` | 0.58×, 1.29× the `f64` export's time | 0.56×, 1.28× |

The searches find the same realization's gap as D154, 3.687·10⁻⁴. They
now price it at 2.00× and not 2.17×: D154's 2.17× counted the masks in
`bf16`. The solver is 2 % short of them, where it was 10 % short. The
tools' C programs and CRAFT's timings change in their second digit.

## 12. D155: the measured rounding term and the narrowing (`make solver-rounding`) → D165

`experiments/solver_rounding.sh` (phase B, 08:46–09:05). The narrowing ran
with `--margin 1` (`NARROW_MARGIN=1`, D161), which is D155's rule. Its
report was regenerated once its parser read D161's narrowing line.

| budget, form | D155: stored values, traffic, gap on the 461 | 0.16.0 |
|---|---|---|
| 4.67·10⁻⁴, unit / off | 29 of 31, 1.97×, 7.5·10⁻⁵ | 22 of 25, 1.96×, 1.64·10⁻⁴ |
| 4.67·10⁻⁴, measured / off | 29 of 31, **2.14×**, 1.6·10⁻⁴ | 22 of 25, **1.96×**, 1.43·10⁻⁴ |
| 4.67·10⁻⁴, measured / narrow | 31 of 31, 2.17×, the searches' realization, 79 measurements | 25 of 25, **2.00×**, the searches' realization, 74 measurements |
| 10⁻⁶, unit / off | 1.27×, 8.7·10⁻⁷ | 1.26×, 8.84·10⁻⁷ |
| 10⁻⁶, measured / off | 1.40×, 8.9·10⁻⁷ | 1.31×, 8.87·10⁻⁷ |
| 10⁻⁶, measured / narrow | 2.07×, over the budget on the 461 | 1.91×, 1.50·10⁻⁶, over |

**The measured term buys nothing on V6 at 0.16.0.** At ε\* it gives 1.96×,
the same as `unit`. Its gain at 0.15.0, 1.97 → 2.14×, was the masks: ones,
exact in `bf16`, which the measured term charged nothing and `unit`
charged s·u·|v|. With the masks gone, the values left exact in a type are
the head width 32 and the decoder's gains (D162). At 10⁻⁶ it still gains
a little, 1.31× against 1.26×. The narrowing still ends at the searches'
realization at ε\*, and still fails out of sample at 10⁻⁶.

## 13. D156: the certificate off-sample against N (`make off-sample`) → D165

`experiments/off_sample.sh` (phase B, 09:06–10:37), N = 1 to 32 held-out
windows, judged on the others.

| | D156 (the solver of record, `unit / off`) | 0.16.0 (the solver of record, `measured / off`) |
|---|---|---|
| out/in at ε\*, N = 4 | 2.30× | **1.94×** |
| out/in at ε\*, N = 32 | 1.92× | **1.15×** |
| out/in at 10⁻⁶, N = 32 | 1.53× | 1.52× |
| the range of out/in | 1.5× to 3.8× | 1.15× to 3.62× |
| the same realization | from 4 windows to 32 | **from 1 window to 32, at both budgets** |
| the solver's measurement against the native gap | at most 14.9 % apart; two solves inside by the solver and outside natively | **0.0 % apart; none** |
| the narrowing at 10⁻⁶ | fails out of sample from every N | still fails (1.44 to 1.50·10⁻⁶) |
| the largest gap over the 461 | window 402 | window 402 (`measured / off`) |

- **R3 holds at this scale.** The solver's measurement is the native gap
  to every digit, so the executor premise of D156's 15 % is gone (D159).
- **The default's realization depends on no N.** It does not change from
  one window to 32.
- **Its out-of-sample ratio is smaller than D156's.** At 32 windows the
  gap out of sample is 1.15× the gap in sample.

## 14. The compiler's own times (`make compile-times`) → D166

`experiments/compile_times.sh`, alone, one thread unpinned, one process per
command (D101's protocol). The course GPT's rows and MiniGPT's ran on
2026-10-20 23:23 – 2026-10-21 00:41, BERT-large's two blocks 00:41–05:17
(phase B, presk `22dd345`). The reports are `docs/compile-times-2026-10-20.md`
and `docs/compile-times-2026-10-20-bert.md`, the decoder's
`docs/compile-times-2026-10-19-llama.md` (D162).

| model | before: check, solve, calibrate, reference (s) | 0.16.0 | peaks (GB) |
|---|---|---|---|
| the course GPT, course constants | D101: 0.007, 2 702.4, 1 497.7, 178.6 | 0.005, **2 647.6**, **1 732.5**, **186.2** | 2.4 (solve) |
| the course GPT, D = 64, trained weights | D101: 0.006, 60.5, 28.8, 3.78 | 0.005, 56.4, 32.8, 3.64 | 0.2 |
| MiniGPT | D101: 0.004, 0.160, 0.049, 0.022 | 0.005, 0.142, 0.060, 0.019 | — |
| BERT-large, two blocks | D124: 0.006, 10 112.6, 4 821.6, 557.2 | 0.004, **10 204.2**, **5 742.5**, **595.1** | 4.8, 10.6, 4.0 (D124: 5.0, 11.1, 4.2) |

- **The solves are within 7 % of their records,** the course GPT's 2 %
  faster and BERT's 1 % slower.
- **The calibrations are 14 to 19 % slower.** Two causes were measured
  alone on the D = 64 row, three processes per binary (2026-10-21,
  `out/step5-0.16.0/calib-time/`):

  | binary | two samples (s) | one sample (s) | what it adds |
  |---|---|---|---|
  | 0.13.0 | 28.7 | 14.4 | — |
  | `454bb9f` (0.15.0, D154–D156) | 30.3 | 15.2 | D155's rounding table: +5 % |
  | `22dd345` (0.16.0) | 32.9 | 15.1 | D158's per-sample pass, only with several samples: +9 % |

  Every calibration of the compile times is on two samples, BERT's too, so both
  apply to every row.
- **The references are 4 to 7 % slower.** Each is one process, and the
  cause was not separated.
- **BERT's solve moves as the decoder's does (D162).** It keeps 28 of 28
  stored values and 23 of 23 parameters, but:
  - its three LayerNorm gains `γ`, ones at initialization, are now in
    `bf16`, as is the query after `split_heads` (`SelfAttention.apply/%5`);
  - 13 parameters are in `bf16` (D124: 10), and the traffic is
    2.9418·10⁸ against 5.9005·10⁸ (D124: 3.6105·10⁸ against 7.2219·10⁸),
    **2.01×** (2.00×);
  - the estimate is 4.634·10⁻³ (1.895·10⁻³), and the measured gap
    **9.981·10⁻⁶** (9.208·10⁻⁷) against 6.254·10⁻³, with no widening
    round.

  The exact loss is the same, 6.75736064365.
- **The course GPT's compile-times solve** is on the initialization's envelope
  (D101's command). It puts all 26 stored values and 21 parameters in
  `bf16`, 4.00×. Its time is the one reported.

## 15. V5: the decoder block's speed under its realizations (D124) → D166

`experiments/pytorch_decoder.sh` with `ROWS=realizations`, the frozen
profile's plans, ten X925 cores pinned. There were three rounds at batch 1
and one at batch 20 (06:15–08:22), then five more processes a row at
batch 1 for all-`f32` and 0.16.0's realization (10:38–11:00,
`out/v5-0.16.0/b1-extra`). The four realizations are all-`f64`,
all-`f32`, D124's solved (0.12.0) and D162's (0.16.0).

| realization | B = 1, rounds 1 / 2 / 3 (ms) | ÷ all-`f32` | B = 1, five more processes (ms) | B = 20 (ms) | ÷ all-`f32` |
|---|---|---|---|---|---|
| all-`f64` | 8 762.4 / 8 703.9 / 9 039.5 | 2.20 / 2.23 / 2.32 | | 107 839.8 | 1.54 |
| all-`f32` | 3 985.0 / 3 895.0 / 3 892.0 | 1 | 3 850.4, 3 851.1, 3 846.4, 3 874.4, 3 863.7 | 69 947.0 | 1 |
| D124's solved | 2 795.4 / 2 890.1 / 3 109.3 | 0.70 / 0.74 / 0.80 | | 46 073.6 | 0.66 |
| **D162's solved** | **2 412.2 / 2 990.4 / 2 518.8** | **0.61 / 0.77 / 0.65** | 2 856.7, 2 644.1, 2 772.1, 3 064.1, 2 983.7 | **47 108.2** | **0.67** |

- **D162's realization beats all-`f32` in every process.** Over its eight
  processes at batch 1, the median is 2 814.4 ms and the fastest 2 403.8
  ms. All-`f32`'s eight give a median of 3 869.1 ms and a fastest of
  3 839.4 ms. So the realization runs at 0.73× all-`f32` by the medians
  and 0.63× by the fastest.
- **This differs from D124 in one place.** D124 had all-`f32` at 2 438.9 ms
  in its first round, a process in the fast mode, and its realization
  "as fast as all-`f32`". No all-`f32` process ran in that mode here.
- **The two solved realizations are within each other's spread.** D162's
  three gains in `bf16` move 24 KB, and they convert 18.9 MB a call at the
  kernels' edges, as D124's does.

## 16. T2 and T4: the course GPT against PyTorch (D127's protocol) → D166

Five rounds of `experiments/pytorch_baseline.sh`, alone, on presk
`d318d6f` (D164), 2026-10-21 11:44–13:35 (`out/retime-0.16.0/`, combined by
`experiments/retime_processes.py` into `docs/retime-2026-10-21.md` and
`docs/retime-2026-10-21-table4.md`, the latter in D127's
`retime-2026-10-15-table4.md` format). The first five rounds, on `22dd345`,
stopped at the native step under the plan, the defect D164 fixed
(`out/retime-0.16.0-22dd345-failed`). PyTorch's rows are timed again in
every round. The ten X925 cores are pinned; each value is the median of
five processes.

| row (ms) | D127 | 0.16.0 |
|---|---|---|
| PyTorch's step (`train.py`, eager) | 1 026.9 | 1 018.2 |
| the C step under the plan | 1 003.5 | **951.1** |
| the C step, `GOMP_SPINCOUNT=8M` | 846.9 | **803.1** |
| the C forward pass under the plan, spinning | 226.0 | **207.0** |
| the native forward pass under the plan | 343.4 | 338.3 |
| the native step, `loss_grad` + `adamw.step` under the plan | 1 354.3 | 1 340.8 |
| `presk train`'s native step under the plan | 1 442.4 | 1 442.5 |

| statement | D127 | 0.16.0 |
|---|---|---|
| T2: the C step ÷ PyTorch's, at most 2× | 0.98× | **0.93×** (fastest 0.93×) |
| T4: the native forward pass ÷ the C program's, at most 1.3× | 1.52× | **1.63×** (1.65×) |
| T4: the native step ÷ the C program's, like for like | 1.60× | **1.67×** (1.66×) |

- **The C program got faster, by 5 % on the step and 8 to 11 % on the
  forward pass (spinning and not), and nothing else moved.** That is D160: an inference entry has no
  mask, so the C export neither fills one with ones nor multiplies by
  it. The step's gradient is `loss_grad`, an entry outside a training
  run, so it is the inference reading too.
- **The native rows move within 2 %.** `presk train`'s step, which draws
  its masks, is D127's to 0.1 ms.
- **T2 holds by more, and T4 fails by more.** Both move because the C
  side got faster.
- **J′ stands**, since the step's realizations are byte-identical
  (D163 §7).

## 17. D153's ratios, from the new C step → D166

PyTorch's compiled and mixed-precision rows stand (D153, PyTorch's own
programs). The ratios that read the compiler's row are recomputed from §16's
C step, 951.1 ms, and C forward pass, 243.0 ms (the plan, OpenMP's default):

| PyTorch | step (ms, D153) | the C step ÷ it: D153 | 0.16.0 |
|---|---|---|---|
| eager, `train.py` | 1 022.0 | 0.98 | **0.93** |
| `torch.compile` | 558.2 | 1.80 | **1.70** |
| AMP `bfloat16` | 827.3 | 1.21 | 1.15 |
| `torch.compile` + AMP | 466.2 | 2.15 | 2.04 |
| eager, no dropout | 685.0 | 1.46 | **1.39** |
| `torch.compile`, no dropout | 483.8 | 2.07 | **1.97** |

The C forward pass under the plan is 1.99× the compiled one (122.4 ms),
where D153 had 2.24× (274.2 ms).

The like-for-like rows mean more now. D153's C step multiplied by masks of
ones; 0.16.0's draws none and multiplies by none. So "the C step against
PyTorch without dropout" compares the same arithmetic: **1.39× eager and
1.97× compiled.**

## 18. J: a budget on the loss, timed per training step (D50) → D166

J is plan 0.6.0's statement, written for 0.5.0 (D50). The loss's
realization is solved at 10⁻³ with the time objective and applied to a
training step, which must then be at least 1.3× faster than all-`f64`. It
had no script of record: D50 timed it by hand, from the difference of a
one-step and a four-step run, which is `j_step.sh`'s method. Its 1.06× at
D = 64 was apremont's, diluted by the conversion D46 found, and cannot be
measured again. Its 1.13× at course scale was falicon's.

At 0.16.0 the method is `out/step5-0.16.0/h6.sh`, alone, five rounds
(2026-10-21 14:07–14:47, presk `d318d6f`):
- **The realization.** `loss` at D = 64 solved at 10⁻³ relative with the
  frozen per-dtype profile of the corpus GPT (`out/gpt_prof.json`, D93),
  on `make precision`'s envelope (D163). It narrows 27 of 27 stored values
  (3 `bf16`) and its 21 parameters (12 `bf16`), with a predicted time of
  1.01× and a measured gap of 1.269·10⁻⁵.
- **How it trains.** `presk train` refuses a realization that types the
  parameters (`E7001`: a training run keeps its master weights in the
  default type), so the step trains under the realization less its
  `params`, as `curve.sh`'s quality axis does (D124). Phase C's first try
  did not do so and is kept as `out/h6-0.16.0-params-refused`.

| per training step (s), the median of five processes | all-`f64` | the solved realization | ÷ all-`f64` | all-`f32` | ÷ all-`f64` |
|---|---|---|---|---|---|
| course constants (D50: 56.9, 50.2, 30.4; 1.13×) | 24.04 | 23.56 | **1.02×** (fastest 1.01×) | 13.76 | 1.75× |
| D = 64 (D50, apremont: 1.06×) | 0.340 | 0.460 | **0.74×** (0.75×) | 0.303 | 1.12× |

**J fails at both scales, for D50's reason.** The budget binds the loss's
forward pass, so the realization names only those sites. A training step
is mostly the reverse sweep and the optimizer, which the realization does
not mention and which run at its default, `f64`. All-`f32` shows that the
gain is there to be had: 1.75× at course scale.

At D = 64 the realization is slower than all-`f64`, by 35 % a step. The
cause was not measured. The likely one is the conversions where the
narrowed forward sites meet a gradient in `f64`, which at this size would
cost more than the narrowing saves.

## Found on the way: `presk plan` in the inference mode (D164)

T2 and T4's first five rounds (phase B, 2026-10-21 05:17–06:14) stopped at
the native training step under the plan. `presk train` refused it with
`E7002`: the plan named `Block.apply.bwd/%50`, a `let` that `presk train`'s
program computes inside a fusion group.

Since D160, `presk plan` lowers the inference reading. A gradient that drops
then numbers its nests otherwise than the training run's program does, and
`presk plan` had no way to plan that program. D164 adds `presk plan --mode
training`, and T2 and T4 were run again, five rounds, on the fixed binary
(§16).
