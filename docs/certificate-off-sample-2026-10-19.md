> The certificate off-sample against the size of the calibration sample (presk D156). The prompt was presk-project `docs/prompt-certificate-off-sample-2026-10-19.md`, item 7 of the paper's improvements, written by the paper session on the user's word ("va pour 7"). Version 0.15.0 at presk `c162dbf`, nothing in the compiler changed; on falicon, 2026-10-19 from 12:22 to 13:58, nothing timed of record. The tables come from `experiments/off_sample.sh` (`make off-sample`): `out/off-sample/report.md`, `report.json`, `env/`, `solve/`, `realizations.tsv`, `margins/`. The outputs are in presk-project `data/certificate-off-sample-0.15.0.tar.xz`.

# The certificate off-sample against the size of the calibration sample

Every claim the solver measures is a claim on a sample. The envelope's magnitudes, the estimate, the measured gap and the realization are all taken on the few held-out windows the envelope was calibrated on: the first 4 at D = 64 for V6 and D154–D155, 4 at course scale for D129. The paper then judges the realization on the 461 held-out windows and quotes the ratio: 2.3× at D = 64 (V6) and 4.8× at course scale (D129).

This run asks how that ratio depends on the size of the sample. It also asks whether D155's narrowing by measurement, which overfit four windows at 10⁻⁶, becomes safe once the sample is larger.
- *The method.* The envelope of the logits at D = 64 is calibrated on the first N = 1, 2, 4, 8, 16, 32 held-out windows, and on all 461. The solver runs on each envelope at ε\*(10⁻³) = 4.67·10⁻⁴ and at 10⁻⁶, in three forms.
- *The judgment.* Every realization is judged on the 461 windows, its gap split in two: the N windows it was calibrated on (*in*) and the 461 − N others (*out*). The flips, the bound #{m < 2ε} and the certification are taken on the out windows only.

**In one paragraph.**
- **The out-of-sample ratio falls slowly and in steps.** For the solver of record it is 2.3× at the 4 windows of record and still 1.9× at ε\* and 1.5× at 10⁻⁶ at 32 windows; it is never within 1.5× in this run.
  - It is one window's doing. The realization of record's largest gap over the 461 windows is in window 402, and no other window comes within 10 % of it.
  - Held fixed, that realization's in-sample maximum comes within 1.15× of the others' only at 256 windows.
- **The realization of record does not depend on the sample from 4 windows on.** At ε\* all 52 decisions are V6's from N = 4 to 32; at 10⁻⁶ they are V6's at N = 1, 4, 8 and 16, six differ at N = 2, and only √32 (one element) flips at 32. The sample sets the gap the solver measures, not the realization it picks.
- **The narrowing step does not become safe with the sample.**
  - At 10⁻⁶ it fails out of sample at every N from 1 to 32: 1.44 to 1.71·10⁻⁶ against the budget 10⁻⁶.
  - It narrows less as the sample grows (2.12× at one window, 1.89× at 16 and 32), but the excess only falls from 71 % to 44 %.
  - Measuring and narrowing on the same windows overfits at every sample this run can afford, so `--narrow measured` needs a held-out check whatever N.
- **The factor an in-sample gap promises is up to 2.7× at four windows on this model**, 3.8× at one. This is below D129's 4.8× at course scale, so it does not carry from one model to another.
- **No realization inside ε\* out of sample flipped a decision on the windows it was not calibrated on.** The one realization outside ε\* out of sample, the narrowing from a single window, flipped one of 117 760 decisions. At 10⁻⁶, every realization, including the narrowings that fail the budget, flips nothing.

## The envelopes

**The checks.**
- `env/4.json`, on windows 0–3, is D155's envelope bit for bit: sites, minima, rounding, element counts, ops and terms. D155's was V6's to every type.
- The population is calibrated in 15 chunks of 32 windows (13 in the last) and folded together (`off_sample.py merge`).
  - *The fold.* An envelope's figures are maxima (magnitudes, rounding errors, sums of terms), minima, and the first op seen, so the fold is exact. One calibration runs its windows as one batch.
  - *The check.* The fold of windows 0–15 and 16–31 is the envelope of windows 0–31 bit for bit, in every figure but the element counts.
  - *The element counts.* They are left out of the fold. Most count the batch's elements, a few count the model's (the mask, √32), and an envelope does not say which. The population is an envelope to compare against, not to solve on.

**Agreement.** Each envelope against the next (N against 2N) and against the population: how many of the 98 sites' magnitudes `max|x|` grow, and the largest ratio.

| N | calibration (note) | vs 2N: sites growing | largest ratio (site) | vs the 461: sites growing | largest ratio (site) | `f32` / `bf16` rounding vs the 461, largest ratio |
|---|---|---|---|---|---|---|
| 1 | 15 s, 62 MB | 33 | 1.167 (`Block.apply/%18`) | 44 | 1.505 (`LayerNorm.apply/%2`) | 2.00 / 2.00 |
| 2 | 30 s, 105 MB | 13 | 1.065 (`Block.apply/%21`) | 41 | 1.505 (`LayerNorm.apply/%2`) | 2.00 / 2.00 |
| 4 | 61 s, 191 MB | 17 | 1.059 (`Block.apply/%3`) | 41 | 1.502 (`LayerNorm.apply/%2`) | 2.00 / 2.00 |
| 8 | 122 s, 363 MB | 13 | 1.077 (`LayerNorm.apply/%2`) | 41 | 1.502 (`LayerNorm.apply/%2`) | 2.00 / 2.00 |
| 16 | 247 s, 707 MB | 17 | 1.187 (`LayerNorm.apply/%2`) | 41 | 1.394 (`LayerNorm.apply/%2`) | 2.00 / 2.00 |
| 32 | 494 s, 1.4 GB | (the 461) | | 33 | 1.175 (`LayerNorm.apply/%2`) | 1.97 / 1.62 |

- **The magnitudes stop moving early and never quite stop.** From one window to the next size they grow by at most 19 %.
  - *Against the population.* 41 to 44 of the 98 sites sit below its magnitudes at every N up to 16, and 33 still do at 32.
  - *The largest gap.* It is the first layer norm's mean, `LayerNorm.apply/%2`, 1.50× below the population at 4 windows and 1.17× at 32. The others are within 1.23× at 4 windows and 1.09× at 32.
- **D155's measured rounding term moves more than the magnitudes.**
  - The largest rounding error of a site's values approaches `u · max|x|` only when the sample holds a value that rounds badly.
  - Some sites' error in `f32` or `bf16` doubles between any N up to 32 and the population (`LayerNorm.apply/%2`, 2.0× in `bf16`).
  - That is the conditionality D155 stated, measured. The measured form still holds out of sample at every N (below).
- **The population costs little in wall time.** One process would take 15 s a window, 2.1 hours for the 461. In 15 chunks, 12 at once, it took about 16 minutes. It was calibrated: the prompt's "skip above two hours" was about wall time.

## The solves

Each solve is `presk solve --budget-abs B --rounding R --narrow M` on the envelope of the first N windows. 36 solves found 15 distinct realizations, and each is run once on the 461 windows. Six of them are D155's; their rows are linked from `make solver-rounding`.
- *Traffic.* It is the solver's `cost_at` without a profile, Σ elements × bytes, priced at the 4-window envelope's elements, the convention of V6 and D154–D155. This lets the rows compare across N.
- *Why a fixed convention.* An envelope counts N windows' elements at each stored value and one model's at each parameter, so the solver's own `cost` line weighs a parameter 4/N times as much as at N = 4. Its own ratio differs by at most 0.04× from the fixed one (`report.md`).
- *The wall times* are notes, with 12 solves at once: 31 s to 17 min without narrowing, up to 76 min with it at N = 32.

| budget | form | N = 1 | N = 2 | N = 4 | N = 8 | N = 16 | N = 32 |
|---|---|---|---|---|---|---|---|
| 4.67·10⁻⁴ | unit/off (of record) | 3/44/5, 1.97× | the same | **V6**: 2/48/2, 1.97× | V6 | V6 | V6 |
| 4.67·10⁻⁴ | measured/off | 3/38/11, 2.14× | the same | D155's: 2/40/10, 2.14× | the same | the same | the same |
| 4.67·10⁻⁴ | measured/narrow | 0/35/17, 2.18× (77 meas.) | D154–D155's: 0/36/16, 2.17× (78) | the same (79) | the same (79) | the same (79) | the same (79) |
| 10⁻⁶ | unit/off (of record) | V6 | 37/14/1, 1.28× | **V6**: 33/18/1, 1.27× | V6 | V6 | 34/17/1, 1.27× |
| 10⁻⁶ | measured/off | D155's: 32/14/6, 1.40× | the same | the same | the same | the same | the same |
| 10⁻⁶ | measured/narrow | 1/45/6, 2.12× (139) | 2/44/6, 2.09× (93) | D155's: 6/40/6, 2.07× (93) | 4/42/6, 1.97× (139) | 3/43/6, 1.89× (93) | the same (93) |

Each cell gives `f64`/`f32`/`bf16` of the 52 decisions (31 stored values, 21 parameters) and the traffic, with the measurements of a narrowing solve in brackets. Without narrowing a solve measures once; no solve widened anything.

**The decisions that change with N** (`report.md`, "The realization against N"), against N = 4:
- *Unit/off at ε\*, N = 1 and 2.* Four decisions differ, at the same 1.97×: the second layer norm's mean `LayerNorm.apply/%5` goes from `f32` to `f64`, and three biases (`wq.b`, `ff.down.b`, `ln2.β`) from `f32` to `bf16`.
  - Narrowing a parameter is what a cost weighing the parameters 4× (N = 1) or 2× (N = 2) as much favours. The sensitivities, taken on the same N windows, change as well, and this run does not separate the two.
- *Unit/off at 10⁻⁶.*
  - At N = 2, six decisions differ.
  - At N = 32, only √32 (`SelfAttention.apply/%16`, one element, eight bytes) goes from `f32` to `f64`.
- *Measured/off.* At ε\* two decisions differ at N = 1 and 2: `LayerNorm.apply/%5` to `f64`, `ff.up.b` to `bf16`. At 10⁻⁶ none differs at any N.
- *Measured/narrow.* At ε\* one decision differs at N = 1 (`ff.down.w` to `bf16`). At 10⁻⁶ five or six differ at every N, and the realization narrows less as N grows.

## In sample and out of sample

Each cell gives the traffic, the gap on the N calibration windows (in), the gap on the 461 − N others (out), out/in, and whether out is inside the budget.
- *Both gaps are native.* Both are the largest elementwise gap of the native logits to the exact ones, the same instrument on both sides.
- *The solver's own measurement* is on the kernel interpreter and differs from the native in by up to 15 %. See "The two executors" below.

**Budget ε\*(10⁻³) = 4.67·10⁻⁴**

| N | unit/off (of record) | measured/off | measured/narrow |
|---|---|---|---|
| 1 | 1.97×, 9.57·10⁻⁵, 1.64·10⁻⁴, 1.72×, holds | 2.14×, 7.37·10⁻⁵, 1.43·10⁻⁴, 1.94×, holds | 2.18×, 3.97·10⁻⁴, 8.55·10⁻⁴, 2.15×, **fails** |
| 2 | 1.97×, 9.57·10⁻⁵, 1.64·10⁻⁴, 1.72×, holds | 2.14×, 7.37·10⁻⁵, 1.43·10⁻⁴, 1.94×, holds | 2.17×, 1.35·10⁻⁴, 3.69·10⁻⁴, 2.72×, holds |
| 4 | 1.97×, 3.27·10⁻⁵, 7.52·10⁻⁵, 2.30×, holds | 2.14×, 9.57·10⁻⁵, 1.64·10⁻⁴, 1.71×, holds | 2.17×, 1.35·10⁻⁴, 3.69·10⁻⁴, 2.72×, holds |
| 8 | 1.97×, 3.27·10⁻⁵, 7.52·10⁻⁵, 2.30×, holds | 2.14×, 9.57·10⁻⁵, 1.64·10⁻⁴, 1.71×, holds | 2.17×, 1.45·10⁻⁴, 3.69·10⁻⁴, 2.55×, holds |
| 16 | 1.97×, 3.91·10⁻⁵, 7.52·10⁻⁵, 1.92×, holds | 2.14×, 1.48·10⁻⁴, 1.64·10⁻⁴, 1.11×, holds | 2.17×, 1.56·10⁻⁴, 3.69·10⁻⁴, 2.36×, holds |
| 32 | 1.97×, 3.91·10⁻⁵, 7.52·10⁻⁵, 1.92×, holds | 2.14×, 1.48·10⁻⁴, 1.64·10⁻⁴, 1.11×, holds | 2.17×, 2.47·10⁻⁴, 3.69·10⁻⁴, 1.49×, holds |

**Budget 10⁻⁶**

| N | unit/off (of record) | measured/off | measured/narrow |
|---|---|---|---|
| 1 | 1.27×, 2.30·10⁻⁷, 8.66·10⁻⁷, 3.76×, holds | 1.40×, 2.45·10⁻⁷, 8.87·10⁻⁷, 3.62×, holds | 2.12×, 8.46·10⁻⁷, 1.64·10⁻⁶, 1.94×, **fails** |
| 2 | 1.28×, 3.44·10⁻⁷, 8.85·10⁻⁷, 2.57×, holds | 1.40×, 3.47·10⁻⁷, 8.87·10⁻⁷, 2.56×, holds | 2.09×, 9.37·10⁻⁷, 1.71·10⁻⁶, 1.82×, **fails** |
| 4 | 1.27×, 3.49·10⁻⁷, 8.66·10⁻⁷, 2.48×, holds | 1.40×, 3.47·10⁻⁷, 8.87·10⁻⁷, 2.56×, holds | 2.07×, 1.02·10⁻⁶, 1.61·10⁻⁶, 1.57×, **fails** |
| 8 | 1.27×, 4.98·10⁻⁷, 8.66·10⁻⁷, 1.74×, holds | 1.40×, 4.65·10⁻⁷, 8.87·10⁻⁷, 1.91×, holds | 1.97×, 1.01·10⁻⁶, 1.53·10⁻⁶, 1.51×, **fails** |
| 16 | 1.27×, 5.65·10⁻⁷, 8.66·10⁻⁷, 1.53×, holds | 1.40×, 5.83·10⁻⁷, 8.87·10⁻⁷, 1.52×, holds | 1.89×, 8.02·10⁻⁷, 1.44·10⁻⁶, 1.79×, **fails** |
| 32 | 1.27×, 5.65·10⁻⁷, 8.66·10⁻⁷, 1.53×, holds | 1.40×, 5.83·10⁻⁷, 8.87·10⁻⁷, 1.52×, holds | 1.89×, 8.02·10⁻⁷, 1.44·10⁻⁶, 1.79×, **fails** |

**A fixed realization against N.** The realizations solved on the 4-window envelope, held fixed. For each, the gap over the first N windows and the ratio of the others' gap to it. This separates how fast a sample's maximum approaches the population's from what the solver does with the sample.

| budget | form | N = 4 | 8 | 16 | 32 | 64 | 128 | 256 | the 461: largest, its window |
|---|---|---|---|---|---|---|---|---|---|
| 4.67·10⁻⁴ | unit/off (V6) | 2.30× | 2.30× | 1.92× | 1.92× | 1.92× | 1.72× | 1.15× | 7.52·10⁻⁵, window 402, none other within 10 % |
| 4.67·10⁻⁴ | measured/off | 1.71× | 1.71× | 1.11× | 1.11× | 1.11× | 1.11× | 1.11× | 1.64·10⁻⁴, window 336, 2 others within 10 % |
| 4.67·10⁻⁴ | measured/narrow | 2.72× | 2.55× | 2.36× | 1.49× | 1.38× | 1.11× | 1.11× | 3.69·10⁻⁴, window 368, 1 other |
| 10⁻⁶ | unit/off (V6) | 2.48× | 1.74× | 1.53× | 1.53× | 1.49× | 0.76× | 0.69× | 8.66·10⁻⁷, window 121, none other |
| 10⁻⁶ | measured/off | 2.56× | 1.91× | 1.52× | 1.52× | 1.52× | 0.73× | 0.67× | 8.87·10⁻⁷, window 121, none other |
| 10⁻⁶ | measured/narrow | 1.57× | 1.54× | 1.54× | 1.33× | 1.01× | 1.01× | 1.01× | 1.61·10⁻⁶, window 426, 3 others |

Out/in falls below 1 once the sample holds the population's worst window: the others' maximum is then below the sample's.

**The flips and the certificate, on the out windows only.**
- *What is counted.* The flips are `decisions`' against the exact argmax. The bound is #{m < 2ε} at the out-of-sample gap. Certification is the out-of-sample gap at most ε\*(δ) of the out windows' own margins, Clopper–Pearson at α = 0.05. That ε\*(10⁻³) is 4.670·10⁻⁴ for N ≤ 16 and 4.724·10⁻⁴ at N = 32.
- *At ε\*.* The realizations of the 17 solves inside ε\* out of sample flip **no decision** on the 109 824 to 117 760 positions they were not calibrated on. Their bounds are 17 to 80, all certified at 10⁻² and 10⁻³, none at 10⁻⁴. D118's certificate promised at most 118 flips of 118 016 at 95 %.
- *The one outside ε\*.* The narrowing from one window measures 8.55·10⁻⁴ out of sample. It flips **one** decision of 117 760, with a bound of 171, and is not certified at 10⁻³. One flip is still within δn = 117.8.
- *At 10⁻⁶.* Every realization flips nothing out of sample, and #{m < 2ε} is 0. That includes the six narrowings that fail the budget at 1.4–1.7·10⁻⁶, since no margin is below 3.4·10⁻⁶. The certificate's deterministic part, no flip where the margin is at least twice the gap, holds where the budget does not.
- *In sample.* No realization flips any decision on its calibration windows.

**The two executors.** The solver measures a realization on the kernel interpreter, whose `exp`, `log` and `tanh` are the system's C library's. The 461-window rows are native, with the `libm` crate's (musl's) functions (D76: the kernel interpreter keeps Rust's standard functions, the native backend and the C export ship musl's).
- *Small apart.* On this model the two differ in the last bits everywhere, `f64` included (up to 3.6·10⁻¹⁵ on the `f64` logits).
- *Large apart.* Under a realization with `bf16` sites a last-bit difference can cross a `bf16` rounding. On the same windows the solver's measured gap and the native one then differ by up to 15 % (the narrowing at 10⁻⁶ from one window: 7.20·10⁻⁷ against 8.46·10⁻⁷).
- *Inside, then outside.* Two narrowings are inside the budget by the solver's measurement and outside it natively on their own windows: N = 4 at 9.93·10⁻⁷ against 1.02·10⁻⁶, and N = 8 at 9.94·10⁻⁷ against 1.01·10⁻⁶. A narrowing that stops within 1 % of the budget leaves no room for the executor's last bits.
- *Not a defect here.* This is D76's documented choice, and nothing changes. The out/in ratios above use the native gap on both sides.

## The sentences

**(a) How the out-of-sample ratio falls with N, for the solver of record.**
- *At ε\*.* 1.72× at one and two windows (another realization, at the same 1.97×), 2.30× at 4 and 8 (V6's, the paper's 2.3×), and 1.92× at 16 and 32.
- *At 10⁻⁶.* 3.76×, 2.57×, 2.48×, 1.74×, 1.53× and 1.53×.
- *Not within 1.5× in this run.* It never comes within 1.5× up to 32 windows; at 10⁻⁶ it comes within 1.53×.
- *Held fixed.* V6's realization at ε\* comes within 1.5× only at 256 windows (1.15×), and within 1.1× only once the sample holds window 402. At 10⁻⁶ it comes within 1.5× at 64 windows and below 1 from 128, once the sample holds window 121.
- *The ratio is extreme-value statistics, not averaging.* The gap over many windows is one window's gap, and the ratio moves only when the sample takes in a window above all those before it. A sample of consecutive windows reaches the worst one only by covering the text up to it.

**(b) Whether the narrowing step becomes safe with the sample.**
- *At 10⁻⁶, no.* `--rounding measured --narrow measured` fails out of sample from every N.

  | N | traffic | out | over the budget |
  |---|---|---|---|
  | 1 | 2.12× | 1.64·10⁻⁶ | 64 % |
  | 2 | 2.09× | 1.71·10⁻⁶ | 71 % |
  | 4 | 2.07× | 1.61·10⁻⁶ | 61 % |
  | 8 | 1.97× | 1.53·10⁻⁶ | 53 % |
  | 16 and 32 | 1.89× (one realization) | 1.44·10⁻⁶ | 44 % |

- *As predicted, it narrows less with N.* The in-sample maximum grows and the traffic the step keeps falls from 2.12× to 1.89×. The excess falls, but at 32 windows it is still 44 %, against the measured/off form's 1.40× that holds at every N.
- *So the flag needs the held-out check whatever N.* Measuring and narrowing on the same windows overfits at every sample this run can afford, which is D155's recommendation, now measured.
- *At ε\*, it holds from two windows.* The narrowing holds from N = 2 at 2.17× (out/in 1.49× to 2.72×), D154's searches' realization. From one window it fails: 8.55·10⁻⁴, 1 flip.

**(c) Whether the solver of record's realization changes with N.**
- *At ε\*.* From N = 4 to 32 all 52 decisions are V6's, and at N = 1 and 2 four decisions differ at the same traffic.
- *At 10⁻⁶.* The decisions are V6's at N = 1, 4, 8 and 16. At N = 2 six differ; at N = 32 one does, √32, eight bytes.
- *The measured term.* Its realization is the same at all six N at 10⁻⁶, and from N = 4 at ε\*.
- *So from four windows on, the realization is a property of the model, not of the sample.* What the sample changes is the gap the solver measures and the figure it would quote.
- *Below four windows, more than the sample changes.* The decisions that differ there are mostly parameters, and the solver's own cost weighs the parameters (one model's bytes) 2× to 4× more against the activations (N windows' bytes) than at N = 4. The sensitivities, taken on the same windows, change too, and this run does not separate the two.

**(d) The in-sample measurement as a predictor.** The largest out/in ratio per form, over N and both budgets:

| form | over N = 1 to 32 | over N ≥ 4 | at N = 4 (the protocol of record) |
|---|---|---|---|
| unit/off (of record) | 3.76× (10⁻⁶, N = 1) | 2.48× (10⁻⁶, N = 4) | 2.30× (ε\*), 2.48× (10⁻⁶) |
| measured/off | 3.62× (10⁻⁶, N = 1) | 2.56× (10⁻⁶, N = 4) | 1.71× (ε\*), 2.56× (10⁻⁶) |
| measured/narrow | 2.72× (ε\*, N = 2 and 4) | 2.72× (ε\*, N = 4) | 2.72× (ε\*), 1.57× (10⁻⁶) |

- *What a gap on the envelope's four windows promises.* For this model and this text, elsewhere it is at most 2.5× that for the solver of record, and at most 2.7× in any form. From one window it can be 3.8×, from 32 windows still 1.9×.
- *The factor does not carry across models.* D129's 4.8× at course scale is above every ratio here.

## For the paper

**§4, beside "4.8× out of sample".**

> How the ratio depends on the sample was measured at D = 64 (presk D156). Solved on the first N = 1 to 32 held-out windows and judged on the 461 − N others, the solver of record's realization is 1.5× to 3.8× further from the exact logits out of sample than in. It is 2.3× at the four windows of record, and at 32 windows it is still 1.9× at ε\* and 1.5× at 10⁻⁶. The realization itself is the same, decision for decision, from 4 to 32 windows: the sample sets the gap the solver measures, not the realization it picks. The out-of-sample gap is one window's. The realization of record's largest gap over the 461 windows is in window 402, no other comes within 10 % of it, and a sample's maximum approaches it only when the sample reaches that window. No realization inside ε\* out of sample changed a decision on the windows it was not calibrated on.

**Threats.**

> A measured gap is a claim on the envelope's windows. At D = 64 the same realization's gap on the other held-out windows is up to 2.5× larger from four windows (3.8× from one, 1.9× from 32), and 4.8× at course scale. Narrowing decided by that measurement (`presk solve --narrow measured`, off by default) fails the budget out of sample at 10⁻⁶ from every sample size up to 32 windows, while flipping no decision. It is safe only behind a held-out check, whatever the sample's size.

## What it does not say

- **One program, one entry, one scale.** It covers the logits of the course GPT at D = 64 on one trained model, on the Code civil's 461 held-out windows. The ratio at course scale (D129's 4.8×) is not re-measured here.
- **The windows are not independent draws.** They are consecutive 256-token windows of one text, and the first N are its opening. The out-of-sample gap is the rest of the same text, and a sample's maximum reaches the worst window only by covering the text up to it. The bootstrap's caveat of D118 and §4 applies. A random draw of N windows would have other ratios.
- **The sample sizes stop at 32.** The population, 461, was calibrated for the envelopes' agreement, and no solve was run on it.
- **Two budgets, three forms.** `--rounding unit --narrow measured` was not run; D155 found it reaches the same realizations as `measured/narrow` at N = 4.
- **Nothing timed.** The calibration and solve times are notes, 12 jobs at once on falicon's 20 cores. Nothing changed in the compiler, and the defaults stay as the user decided on D155.

## Files

- `experiments/off_sample.sh` (`make off-sample`) runs the four phases. It reuses the outputs present and needs `make margins` and `make solver-rounding`.
- `experiments/off_sample.py` does the fold, the comparison, the deduplication of realizations and the report, with the standard library only. It imports `margins.py`'s Clopper–Pearson and `tuning/search.py`'s space reader.
- `out/off-sample/`:
  - `env/` holds each N's envelope, the 15 chunks, the fold `all.json` and the two checks;
  - `solve/` holds the 36 solves, their listings and times;
  - `realizations.tsv` maps each solve to its realization;
  - `margins/` holds the 461-window rows by `margins.sh`, `make margins`' rows and D155's linked;
  - `report.md` and `report.json` are the report.
- presk-project `data/certificate-off-sample-0.15.0.tar.xz`.
  - *Left out and checksummed* (`EXCLUDED.sha256`): the native rows of the nine new realizations, which the script takes again in minutes.
  - *Listed* (`LINKS.txt`): the rows of `make margins` and D155 that it links, the exact logits among them.
