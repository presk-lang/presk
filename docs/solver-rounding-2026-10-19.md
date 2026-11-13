> The estimate's rounding term measured on the envelope, and a narrowing step decided by measurement (presk D155). The prompt was presk-project `docs/prompt-solver-rounding-2026-10-19.md`, written on the user's word ("allons jusqu'au bout de ce point avec le compilateur"). Version 0.15.0 with D155's code (presk `ecd8357`) and script (`7eab040`), on falicon, 2026-10-19 from 11:11 to 11:31, nothing timed of record. The tables come from `experiments/solver_rounding.sh` (`make solver-rounding`): `out/solver-rounding-0.15.0/report.md`, `margins.md`, `calibrate.txt`, `solve/`, `bounds.txt`. The outputs are in presk-project `data/solver-rounding-0.15.0.tar.xz`.

# The estimate's rounding term measured, and a narrowing step

D154 found that on the solver's own space the tuning tools' searches find a realization of the logits at ε\*(10⁻³) with **2.17×** less traffic than all-`f64`, against the solver's **1.97×**, holding on the 461 held-out windows. 84 % of the bytes they save are four tensors of ones, this entry's dropout masks, put in `bf16`. They are exact there, but the solver's first-order estimate `s · u(bf16) · |v|` charges them more than the whole budget. Two things kept the solver from those bytes:
- a term that charges a rounding the values do not take;
- the lack of any step that narrows after a passing measurement.

This run adds a flag for each and measures both on D154's program, at D154's two budgets, against the same held-out windows.

**In one paragraph.**
- **The measured term alone does not reach 2.17×.** With it the solver reaches **2.14×**, in one sweep and one measurement as before. It holds on the 461 windows: 1.64·10⁻⁴ from the exact logits, no flip, certified at 10⁻³.
- **The narrowing step reaches it.** With narrowing the solver finds **the searches' realization exactly** (2.17×, 3.69·10⁻⁴, no flip, bound 80), in 79 measurements with the measured term and 87 with the unit one, where D154's searches took 70 and 2 858 trials.
- **At 10⁻⁶ the narrowing overfits, as the searches did.** Both narrowing forms pass their four windows at 9.9·10⁻⁷ and fail the 461 at 1.61·10⁻⁶ (2.07×). The measured term alone holds there: 1.40× against the unit term's 1.27×, at 8.9·10⁻⁷.
- **So the measured term is safe to take; the narrowing step is safe only behind a held-out check.**
- **The masks cost more than any term can recover.** They are 15 % of the bytes of the solver's realization of record. Halving them in `bf16` gives what the measured term gains, and not storing them would take the realization of record from 1.97× to 2.33×. The compiler does not know statically that they are ones: that is a question for the specification and for the user.

## The two flags

Both are off by default. Every row of record was solved without them, and with them off the solver is the solver of record bit for bit: the solves below reproduce V6's realization to every type.

**`presk solve --rounding measured`** (default `unit`). The envelope now records, per site and per storage type, the largest rounding error of the sampled values, `round[d][site] = max |fl_d(x) − x|`. It is computed where the magnitudes are observed, from the interpreter's 128-bit values rounded as the runtime rounds them: to the nearest `f64`, that to the nearest `f32`, and that to the nearest `bf16`, ties to even on the `f32`'s bits. Under the flag the term of a site in type `d` is `s · round[d][site]`, where it was `s · u(d) · max|x|`; a parameter's comes from its own values at rest.
- *What it is.* The estimate's rounding step measured on the sample, as the magnitude already is. It is never more than `u(d) · max|x|`, and it is 0 for a value exact in the type. It is as conditional on the envelope as the magnitude: a site whose sampled values all sit low in their binade rounds less on this sample than another input might make it.
- *What it is not.* A bound. The estimate never was one: it ranks, and the measurement discharges the budget. The bound `cond` is unchanged.
- *The details.*
  - An envelope written before D155 has no rounding and keeps the unit term.
  - A value that overflows a type is recorded as the largest double, since JSON has no infinity.
  - `presk calibrate --precision` prints the sites where the term departs most from `u · max|x|`, and so does `presk explain`'s envelope section (X3).

**`presk solve --narrow measured`** (default `off`). This is a step 3′ after the solver's measurement passes, the widening included:
- *The order.* The decisions not at the menu's narrowest, by the bytes one step saves, the largest first.
- *Each trial.* The decision is narrowed one step and the realization measured as step 2 measures it: the kernel backend on the envelope's windows against the exact outputs, already computed. The step is kept if the gap stays inside the budget, and undone if not.
- *The passes.* A second pass runs while the first changed something, three at most.
- *What it reports.* The moves (`Move::Narrowed`, which `presk explain`'s reasons name), the decisions kept narrowed, the passes, and every measurement the solve made.

This is the solver taking the tools' step. It decides a narrowing by the measurement it already has, where before only the proposal decided one. Each trial costs one run of the entry on the envelope's windows: about 6 s here, four solves running at once, and the whole block at the decoder's width. It risks the overfitting D154 saw at 10⁻⁶, which is why every result below is judged on the 461 windows.

## The sites whose term changes by more than 10×

The envelope's six sites where the measured term is below a tenth of `u · max|x|` in `f32` or `bf16`. No parameter is among them.

| site | elements | max\|x\| | `f32` | ÷ `u·max|x|` | `bf16` | ÷ `u·max|x|` | what they are |
|---|---|---|---|---|---|---|---|
| `SelfAttention.apply/%20` | 524 288 | 1 | 0 | 0 | 0 | 0 | the attention's dropout mask: ones |
| `Block.apply/%19`, `%27` | 65 536 each | 1 | 0 | 0 | 0 | 0 | the residual dropout masks: ones |
| `GPT.apply/%14` | 65 536 | 1 | 0 | 0 | 0 | 0 | the embedding's dropout mask: ones |
| `SelfAttention.apply/%15` | 1 | 32 | 0 | 0 | 0 | 0 | the head dimension, 32 |
| `SelfAttention.apply/%16` | 1 | 5.657 | 9.7·10⁻⁸ | 0.29 | 6.0·10⁻⁴ | 0.027 | √32 |

Every other site's term, 73 of them, is within a factor of 4 of the unit term: the dropout rate 0.2 at exactly a quarter, 61 within a factor of 2, since a site's largest value sits near the top of its binade.

## The solves

V6's protocol (D124): the envelope of `logits` on the first 4 held-out windows at 128 bits, `presk solve --budget-abs` with `make margins`' weights. Traffic is the solver's own cost (`cost_at`, as D154 reported it). The wall times are notes, four solves running at a time.

| budget | `--rounding` | `--narrow` | `f64` / `f32` / `bf16` of 52 | traffic vs all-`f64` | estimate | measured on the envelope's windows | narrowed (passes) | measurements | wall |
|---|---|---|---|---|---|---|---|---|---|
| 4.67·10⁻⁴ | unit | off | 2 / 48 / 2 | 1.97× | 4.14·10⁻⁴ | 3.24·10⁻⁵ | — | 1 | 117 s |
| 4.67·10⁻⁴ | **measured** | off | 2 / 40 / 10 | **2.14×** | 4.53·10⁻⁴ | 9.58·10⁻⁵ | — | **1** | 117 s |
| 4.67·10⁻⁴ | unit | **measured** | 0 / 36 / 16 | **2.17×** | 2.01·10⁻² | 1.35·10⁻⁴ | 16 (2) | 87 | 534 s |
| 4.67·10⁻⁴ | **measured** | **measured** | 0 / 36 / 16 | **2.17×** | 2.76·10⁻³ | 1.35·10⁻⁴ | 8 (2) | 79 | 494 s |
| 10⁻⁶ | unit | off | 33 / 18 / 1 | 1.27× | 9.76·10⁻⁷ | 3.49·10⁻⁷ | — | 1 | 118 s |
| 10⁻⁶ | **measured** | off | 32 / 14 / 6 | **1.40×** | 9.92·10⁻⁷ | 3.47·10⁻⁷ | — | 1 | 117 s |
| 10⁻⁶ | unit | **measured** | 6 / 40 / 6 | 2.07× | 1.70·10⁻² | 9.93·10⁻⁷ | 32 (3) | 145 | 819 s |
| 10⁻⁶ | **measured** | **measured** | 6 / 40 / 6 | 2.07× | 1.20·10⁻³ | 9.93·10⁻⁷ | 26 (2) | 93 | 565 s |

- **The unit form is the solver of record.** At 4.67·10⁻⁴ it is V6's realization to every type, as D154 found it.
- **The measured term without narrowing, at 4.67·10⁻⁴,** puts the four masks and 32 in `bf16` at no cost, and three biases with them. It stops 8 decisions short of the searches' realization:
  - the layer norm's `%4` and `%10`, kept in `f64`;
  - √32, kept in `f32`;
  - five biases and shifts, kept in `f32` (`wv.b`, `wo.b`, `up.b`, `ln1.β`, `lnf.β`).

  Their measured term is the unit term within a factor of 4: they round as the unit term says, and the estimate, 4.1 to 4.5·10⁻⁴ against measurements of 3 to 10·10⁻⁵, has no room for them. Only a measurement shows that they fit.
- **With narrowing, both terms end at the searches' realization, decision for decision** (`search/*-params-4.67e-4/final.json` of D154). The measured term gets there in 79 measurements against the unit term's 87, since it has fewer decisions left to narrow. After narrowing, the estimate is no longer what decides: the unit term's is 43 times the budget. The measurement is.
- **At 10⁻⁶ the narrowing spends the budget to 0.7 % on the envelope's windows** (9.93·10⁻⁷ against 10⁻⁶), as D154's searches spent theirs on their two.

## Judged on the 461 held-out windows

`margins.md`, 118 016 positions against the exact logits at 128 bits (D118's), beside D154's searches over the 52 decisions. The bound `cond` is `presk check --precision`'s, absolute, against logits of magnitude 5.2.

| realization | gap on the logits | flips | bound #{m < 2ε} | cert. 10⁻² | cert. 10⁻³ | cert. 10⁻⁴ | traffic vs all-`f64` | bound `cond` |
|---|---|---|---|---|---|---|---|---|
| all-`f32` | 3.12·10⁻⁶ | 0 | 0 | yes | yes | yes | 2.00× | 3.1·10³ |
| the solver at 4.67·10⁻⁴, unit / off (of record) | 7.53·10⁻⁵ | 0 | 18 | yes | yes | no | 1.97× | 6.6·10⁶ |
| **the solver at 4.67·10⁻⁴, measured / off** | **1.64·10⁻⁴** | **0** | 36 | yes | yes | no | **2.14×** | 7.1·10⁶ |
| the solver at 4.67·10⁻⁴, unit / narrow | 3.69·10⁻⁴ | 0 | 80 | yes | yes | no | 2.17× | 7.2·10⁶ |
| **the solver at 4.67·10⁻⁴, measured / narrow** | **3.69·10⁻⁴** | **0** | 80 | yes | yes | no | **2.17×** | 7.2·10⁶ |
| D154's searches at 4.67·10⁻⁴ (both) | 3.69·10⁻⁴ | 0 | 80 | yes | yes | no | 2.17× | 7.2·10⁶ |
| the solver at 10⁻⁶, unit / off (of record) | 8.66·10⁻⁷ | 0 | 0 | yes | yes | yes | 1.27× | 6.6·10⁶ |
| **the solver at 10⁻⁶, measured / off** | **8.87·10⁻⁷** | 0 | 0 | yes | yes | yes | **1.40×** | 7.1·10⁶ |
| the solver at 10⁻⁶, unit / narrow | **1.61·10⁻⁶: fails** | 0 | 0 | yes | yes | yes | 2.07× | 7.1·10⁶ |
| the solver at 10⁻⁶, measured / narrow | **1.61·10⁻⁶: fails** | 0 | 0 | yes | yes | yes | 2.07× | 7.1·10⁶ |
| D154's CRAFT search at 10⁻⁶ | 1.41·10⁻⁶: fails | 0 | 0 | yes | yes | yes | 1.53× | 7.3·10³ |
| D154's Precimonious search at 10⁻⁶ | 1.69·10⁻⁶: fails | 0 | 0 | yes | yes | yes | 1.89× | 7.1·10⁶ |

Every row's accuracy is the exact model's, 0.242442.

**The two sentences the run is for.**
1. **The measured term does not reach the searches' 2.17× in one sweep and one measurement: it reaches 2.14×,** and holds on the 461 windows (1.64·10⁻⁴, no flip, certified at 10⁻³). With the narrowing step it reaches 2.17× in 79 measurements, and the realization it finds is the searches', which holds there too.
2. **The narrowing step does not escape the overfit at 10⁻⁶.** Both its forms pass the envelope's four windows (9.93·10⁻⁷) and fail the 461 (1.61·10⁻⁶), as D154's searches failed. Measuring on four windows of text instead of two of random tokens does not prevent it. The measured term alone holds at 10⁻⁶ (1.40×, 8.87·10⁻⁷).

## The dropout masks

The four masks are this entry's `dropout_mask` sites: `SelfAttention.apply/%20`, `Block.apply/%19`, `%27` and `GPT.apply/%14`. Together they are 720 896 of the 4 504 579 elements the stored values hold.

| realization | the masks' type | their bytes | share of its traffic | its traffic without them, vs all-`f64` |
|---|---|---|---|---|
| the solver at 4.67·10⁻⁴, unit / off (of record) | `f32` | 2 883 584 | 15.3 % | 2.33× |
| the solver at 4.67·10⁻⁴, measured / off | `bf16` | 1 441 792 | 8.3 % | 2.33× |
| the solver at 4.67·10⁻⁴, narrow (either) and D154's searches | `bf16` | 1 441 792 | 8.4 % | 2.37× |
| the solver at 10⁻⁶, unit / off (of record) | `f32`, `f64` | 3 145 728 | 10.8 % | 1.42× |
| the solver at 10⁻⁶, measured / off | `bf16` | 1 441 792 | 5.5 % | 1.48× |

**What the compiler knows about them.** The lowering of `dropout(rate, key, x)` emits a `dropout_mask` and a product whatever the run (`presk-core`'s `lower/expr.rs`). The executors fill the mask with ones at launch when the run does not train (`presk-kernel`'s host, `presk-interp`'s primitive): "outside training every element is kept, `x · 1 = x`: the mask is the host's buffer of ones, drawn nowhere". So the mask is a stored value by construction, not by a missed rewrite.
- *Why no rewrite takes it.* The program is the same in both modes. The mode is the run's (`training: bool` on the interpreter and the kernel backend), not the program's, and no rewrite of PSK-Core can know that `dropout_mask` is ones.
- *What the C export shows.* It records the host from a run that does not train, so it shows the fills (`presk_fill(t19, PRESK_F64, 1.0, 32768)`).
- *What dropping them needs.* An entry evaluated outside a training run (`run`, `serve`, an export, the solver's sweep and measurements) would have to be specialized as such: `dropout_mask` folded to ones and the product to its operand. The solver would then not offer the masks at all, and the realization of record would move 15 % fewer bytes.

**A question for the user, and spec question 78** (`docs/spec-questions.md`): whether an entry's mode is part of its meaning, so that `dropout` outside a training run is the identity as a rewrite of the language and not an executor's shortcut. Nothing changes in the optimizer in this run.

## The question of the default

The flags are off, so the released compiler solves every row of record as it was solved. Turning `--rounding measured` on by default would change every solved realization of record. Each would then need solving again by its whole protocol (presk-project memory "rerun the whole protocol"):

| row of record | what it reruns | hours |
|---|---|---|
| the curve at D = 64 (D124, D132's on the trained envelope) | `make curve`: seven solves and their training runs | 0.4 |
| V6, the logits at D = 64 (D124) | one solve and `make margins`' rows | 0.1 |
| D132's curve at course scale | the envelope and seven solves | 1.2 |
| D129, the logits at course scale | the envelope (51 min) and the solve (1 h 36 min); the exact logits kept | 2.5 |
| BERT-large, two blocks (D101, D117) | the envelope, the solve, the reference | 4.5 |
| the Llama 2 decoder at the 7B width (D106, D117) | D106's whole protocol | 24 |

Turning `--narrow measured` on adds at each solve one native run of the entry per decision and pass on the envelope's inputs:
- here 79–145 runs, 6–12 min;
- at the decoder's width, minutes a run over 56 decisions, hours a pass.

It also needs the held-out check this run makes before its answer can be trusted. **The recommendation:**
- **`--rounding measured`** is safe to make the default at the next version that solves the rows again: a term never larger than the unit one, from the same sample, its realizations holding at both budgets here;
- **`--narrow measured`** should stay a flag, named wherever it is used, with a held-out check beside it.

The decision is the user's.

## What it does not say

- **One program, one entry, one machine.** D = 64, two budgets (one from the margins, one chosen to exercise the searches).
- **The measured term is on the sample.** It is computed on the envelope's four windows, as the magnitude is. A site whose values sit low in their binade there may round more on other inputs.
- **The narrowing measures on the envelope's windows.** Its overfit at 10⁻⁶ is that of measuring and narrowing on the same four windows, not a property of the step's order.
- **The masks were not removed.** Their cost is computed from the realizations, not measured on a specialized program.
- **Nothing timed.** The wall times are notes, four solves running at once.

## Files

- **The code** (presk `ecd8357`):
  - the interpreter's `roundings` (`presk-interp`'s `eval.rs`);
  - the envelope's `rounding` (`presk-precision`'s `envelope.rs`);
  - the solver's `RoundingTerm`, `Narrow`, `Options`, `solve_with` and `Move::Narrowed` (`solve.rs`);
  - the driver's options and the parameters' rounding (`presk-driver`);
  - the CLI's `--rounding` and `--narrow`, and calibrate's listing;
  - `presk explain`'s envelope section and narrowing reasons.
- **The tests:**
  - `presk-precision`'s `tests/rounding.rs` and the envelope's unit test;
  - `presk-driver`'s `the_measured_rounding_is_at_most_the_unit_term`;
  - the explain documents and the manual's generated chapter `explain_loss`, which gain the rounding listing and nothing else.
- **`experiments/solver_rounding.{sh,py}`** (`make solver-rounding`).
- **`out/solver-rounding-0.15.0/`:**
  - `env_logits.json` and `calibrate.txt`;
  - `solve/` (each solve's text, realization and time);
  - `space.txt`;
  - `margins/` (the rows, with `make margins`' and D154's linked);
  - `margins.md`, `bounds.txt`, `report.md`.
- **Archived** as presk-project `data/solver-rounding-0.15.0.tar.xz`, without the native rows of the 461 windows, which the script writes again. Their checksums are in `EXCLUDED.sha256`, the linked rows in `LINKS.txt`.
