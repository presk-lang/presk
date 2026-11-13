> A precision-tuning tool on the exported C (presk D154). It was run at the user's request. Version 0.15.0 (presk `9e8d377`, `main` after !118, with the scripts at `a8b93a3`), on falicon, 2026-10-19 from 09:33 to 10:04: the solver 7 min, CRAFT 8 min, the searches on presk's space 11 min, the 461 windows 2 min. Nothing here is a timing of record. The tables come from `experiments/tuning/tuning.sh` (`make precision-tuning`): `out/precision-tuning-0.15.0/report.md` and `margins.md`, `craft/*/`, `search/*/`, `solver/`, `adapt/`. The outputs are in presk-project `data/precision-tuning-0.15.0.tar.xz`.

# A precision-tuning tool on the exported C

The solver stands beside the precision-tuning tools. Precimonious, HiFPTuner, ADAPT and CRAFT/FloatSmith check a configuration against a run. FPTuner and Daisy decide from a bound. What separates the solver from them rests on three things:

- the reference: the exact function of the whole program;
- the criterion: a budget on the output, discharged by measurement;
- the space: the program's stored values and its parameters' storage types.

That is argued, not measured. This run measures it on the one form of the program a C tool can take, the C program the compiler exports, under the criterion of record. Then it gives the tools' two searches the compiler's own space, and judges everything found on the 461 held-out windows.

**In one paragraph.**

*On the C program.* CRAFT's variable mode runs on the export, with a script that retypes its declarations. At ε\*(10⁻³) its default search passes the whole program in `float` on its first trial (2.5·10⁻⁶ on the conformance inputs). It keeps that program though it runs at 0.58× the `f64` program's speed (30.4 against 17.5 ms): the weights stay `double`, inputs the tool does not own, so every product mixes the two types. Its `ddebug` search, which is Precimonious's and whose objective is the runtime, keeps 21 of the 63 variables in `double`, mostly the products' register tiles, and runs 1.29× faster than `f64` after 227 trials. Neither can use `bf16` or touch a parameter. Written as realizations, the two answers are 1.95× and 1.81× less traffic than all-`f64`, against the solver's 1.97×. As programs, they hold on the 461 windows (3.3·10⁻⁶ and 2.9·10⁻⁶).

*On presk's space,* **the same two searches find a narrower realization than the solver's under the same criterion: 2.17× less traffic than all-`f64`, against 1.97×.**
- *What differs.* 16 of the 52 decisions:
  - the eval-mode dropout masks, four tensors of ones, and the scalars 32 and √32, all six in `bf16`;
  - eight biases and shifts in `bf16`;
  - the layer norm's two values the solver keeps in `f64`, in `f32`.
- *Where the bytes come from.* 84 % of the 1.71·10⁶ bytes saved are the four tensors of ones, exact in `bf16`. The solver's first-order estimate charges them as if they rounded.
- *Out of sample.* The realization holds on the 461 windows: 3.69·10⁻⁴ from the exact logits, no flip, certified at 10⁻² and 10⁻³ (the solver's: 7.5·10⁻⁵, bound 18 against 80).
- *The cost.* 70 trials for CRAFT's search, 2 858 for Precimonious's. Each is a native run on two windows against the exact outputs the compiler computed once, where the solver makes one sweep and one measurement.

*At a budget of 10⁻⁶,* below all-`f32`, the searches overfit their two windows. Their realizations pass on them and fail on the 461 (1.4–1.7·10⁻⁶), where the solver's holds (8.7·10⁻⁷).

## What ran

**The program** is `presk export experiments/margins.psk --entry logits --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=256 --weights out/curve_train_f64/final.psk.d --realize f64 --target c`. It is `make margins`' model, the course GPT at D = 64 on `make curve`'s all-`f64` weights, and computes its logits on two windows of 256 tokens.
- *Its C* is 4 980 lines, in two files:
  - `model.c` holds the entry's function and eight others, one per recording of a Presk function;
  - `kernels.c` holds 26 kernels, each `static inline` and shared by every call.

  Every buffer is a `double *` allocated from one arena (`presk_alloc(arena, N * sizeof(double))`), under a comment naming its site. The program makes 82 launches. C1 holds: its outputs are the native backend's bit for bit.
- *Its conformance inputs* are the tokens `presk run --seed 0` synthesizes: two windows of random tokens, not text. The export ships the exact reference's outputs on them, rounded to `f64` (`conformance/exact.psk.d`).

**The criterion** is that of record: the largest elementwise gap of the logits to the exact reference's, at most **ε\*(10⁻³) = 4.67·10⁻⁴**, on the conformance inputs, against `exact.psk.d` (`experiments/tuning/criterion.py`).
- *Absolute, not relative.* The prompt says "relative to their magnitude". But ε\*(δ) is a number on the logits, the margin argument's, and V6 solved under it as `--budget-abs` (D124). So the criterion here is absolute, the solver's own. The relative figure, the gap over the largest |exact logit| (5.16 on these inputs, 5.21 on the 461 windows), is in tier A's table and in `margins.md`.
- *Against the `f64` run too.* The gap to the export's `native.psk.d` is reported beside the gap to the exact outputs. At these budgets the two references agree: the `f64` run is 4.4·10⁻¹⁵ from the exact logits on these inputs.
- *A second budget, 10⁻⁶,* is not in the prompt. At 4.67·10⁻⁴ every all-`float` program passes on the first trial (2.5·10⁻⁶), so a search over `double` and `float` has nothing to search. At 10⁻⁶, below every all-`f32`'s gap, the searches have to split. It is not a budget from the margins.

**The solver's row** is V6's protocol at 0.15.0 (D124): the envelope of `logits` on the first 4 held-out windows at 128 bits, then `presk solve --budget-abs` with the weights, once per budget (1 min 56 s, 377 MB).
- *At 4.67·10⁻⁴* its realization is V6's (0.12.0) to every type, the envelope's path aside.
- *Its space,* every decision listed with its elements, is a solve at 10³ unverified: **31 stored values and 21 parameters**.
- *Every cost here is the solver's own:* `cost_at` in `presk-precision`'s `solve.rs`, without a profile, which sums the elements times the bytes of each decision's type. It gives back the solver's printed totals: 3.7063·10⁷ for all-`f64`, 1.8797·10⁷ for V6's realization, 9.2657·10⁶ for all-`bf16`.

**Tier A, CRAFT on the C as it is.** CRAFT (crafthpc/craft at `03f2514`, 2020-06-09) is the first of the prompt's tools in its order. It ran, so the order stopped there.
- *The driver* is CRAFT's own variable mode, `craft start -V`. It runs on Ruby 3.1 from conda-forge; Ruby 3.2 removed the `File.exists?` it calls. It searches a set of named variables, and calls two scripts its user writes: a builder for a configuration, and a test of the built program.
- *The variables* are what `experiments/tuning/retype.py` lists from the export:
  - the 39 buffers `model.c` declares, its output among them, each named by its function and declaration (`presk_LayerNorm_apply:mu`) and labelled by its site;
  - the 24 `double` locals of the kernels, their accumulators (`s0`, the register tile `v0`…`v3`) and the values they hold in registers.

  Pointers tied by an assignment are one variable, as TypeForge's type-dependence sets make them. The fold's state `s_0` points to `GPT.apply/%15`'s buffer before the loop and to the loop's two buffers after it.
- *The builder* is `retype.py apply`, the prompt's stand-in for TypeForge. It retypes as follows:
  - a declaration's `double` becomes `float`, its `sizeof` with it;
  - each function, host or kernel, is specialized by the types of the pointers its calls pass, as a C++ template would be. A kernel called with a `float *` where another call passes a `double *` is copied under a new name, that parameter retyped and its body unchanged. A kernel's local takes the type the configuration gives it in every copy;
  - a `presk_fill` of a `float` buffer writes `PRESK_F32`;
  - an output in `float` is declared so in `model.h` and written by `main.c` with `PRESK_F32`.

  Then `make`, without OpenMP: one thread, the same bits.
- *What the builder does not retype:*
  - the parameters, `theta_*`, which `main.c` reads from the weights file as `f64`;
  - the indices;
  - the literals: `64.0` stays `double`, so a `float` combined with a literal is computed in `double`, as C's conversions say;
  - the transcendental functions: `presk_exp`, `presk_erf` and `sqrt` take a widened `float`;
  - the arena's size, which stays an upper bound.

  There is no `bf16`: plain C has none.
- *The test* runs the program on the conformance inputs, pinned to one X925 core, for five calls (`-r 5`). The median of those calls is CRAFT's runtime. Then the test applies the criterion.
- *CRAFT's two strategies:*
  - `bin_simple`, its default: the whole program first, then a failed node split into files, functions and variables, halved above nine children;
  - `ddebug`, its delta-debugging strategy "as described in Rubio-González et al. [SC'13]", that is, Precimonious's search.

  In variable mode both return the fastest passing configuration they tested (`build_fastest_config`), not the union of what passed. So speed decides their answer.

**ADAPT** (LLNL/adapt-fp at `09119e1`, 2022-09-26) was looked at, not run on the export; see below. **FloatSmith and Precimonious** were not run: the prompt orders them only if CRAFT's variable mode had not run. Their searches are here otherwise: CRAFT's `ddebug` is Precimonious's on the C, and tier B reimplements both on presk's space.

**Tier B, the two searches on presk's space** (`experiments/tuning/search.py`). These are not the tools: their search strategies are reimplemented as a driver over presk's realizations.
- *A trial* is `presk run --backend native --realize @R` on the conformance inputs, under the criterion.
- *The space* is the solver's 31 stored values, each `f64`, `f32` or `bf16`, and again with the 21 parameters, which the solver also types.
- *The cost* is the solver's traffic, where Precimonious's is a measured runtime. So the searches and the solver minimize the same objective.
- *Precimonious's search* follows the order and the stopping rule of its implementation (`scripts/dd2.py` of corvette-berkeley/precimonious, `5ac56e1`). It goes level by level of the type vectors, `f64` → `f32` and then `f32` → `bf16`, and first lowers every variable one level. If that fails, delta debugging searches the change set, the variables kept at the higher type:
  - the set is split into `div` parts;
  - each part, and above two parts each complement, is kept high with the rest lowered;
  - the passing one of least cost is taken and searched again from two parts;
  - `div` is doubled when none passes, and the set is kept high when `div` reaches its size;
  - a variable left at the higher type is fixed there.

  A configuration counts only if it passes and costs less than the original.
- *CRAFT's binary search* follows `bin_simple`'s rule (`scripts/craft_strategies.rb`):
  - the whole program is lowered first;
  - a failed node is split into its children: the stored values and the parameters, then the stored values by Presk function and the parameters by their tree, then each one;
  - a node with one child is replaced by it, and more than nine children are merged into two halves;
  - each configuration lowers its node alone, the rest at the level's start.

  A level's result is the union of the passing nodes (`build_final_config`), tested again; if the union fails, the cheapest passing configuration. There are two levels, `f64` → `f32`, then `f32` → `bf16` over what the first lowered.
- *Trials* are every run the search makes, as the tools count them. A realization run twice is measured once (the native backend is deterministic), and both counts are given.

**The compiler judges.** Every assignment found is run on the 461 held-out windows of `make margins` (118 016 positions), against their exact logits at 128 bits.
- *As a realization:* tier B's, the solver's, and tier A's written as one. For tier A, `retype.py realization` puts a site in `f32` when every buffer holding its values is `float`, and leaves the parameters in `f64`.
  - Each runs natively by `experiments/margins.sh`.
  - `margins.py report` gives the gap on the logits, the flips, the bound #{m < 2ε} and the certification at 10⁻², 10⁻³ and 10⁻⁴.
  - Beside them, the solver's traffic and the bound `cond` (`presk check --precision` on the envelope).
- *As the program the tool built:* each of CRAFT's final C programs as it stands, on the windows two by two (`experiments/tuning/windows.py`).
  - Its logits are set against the exact ones as `margins.py` sets a native run's, and its decisions are the argmax of its logits.
  - The control: the export's own C program, judged in the same way, is `make margins`' native `f64` row bit for bit.

## The solver

| budget | stored values away from `f64` | parameters away from `f64` | `f64` / `f32` / `bf16` of 52 | traffic vs all-`f64` | estimate | measured, the envelope's 4 windows | widening rounds | gap, the conformance inputs |
|---|---|---|---|---|---|---|---|---|
| 4.67·10⁻⁴ | 29 of 31 | 21 of 21 | 2 / 48 / 2 | 1.97× | 4.14·10⁻⁴ | 3.24·10⁻⁵ | 0 | 5.63·10⁻⁵ |
| 10⁻⁶ | 9 of 31 | 10 of 21 | 33 / 18 / 1 | 1.27× | 9.76·10⁻⁷ | 3.49·10⁻⁷ | 0 | 5.26·10⁻⁷ |

At 4.67·10⁻⁴, the two stored values left in `f64` are the layer norm's `LayerNorm.apply/%4` and `%10` (its centred squares and its standard deviation). The `bf16` ones are the layer norm's ε (`LayerNorm.apply/%8`) and `blocks.attn.wk.b`, a key bias with a sensitivity of 10⁻³⁹.

## Tier A: CRAFT's variable mode on the C program

| budget | strategy | configurations tested | passed / failed | wall (note) | buffers in `float`, of 39 | locals in `float`, of 24 | gap, conformance inputs (relative) | against the `f64` run | ms a call (note) | ÷ the `f64` program |
|---|---|---|---|---|---|---|---|---|---|---|
| 4.67·10⁻⁴ | `bin_simple` | 1 | 1 / 0 | 3.1 s | 39 | 24 | 2.51·10⁻⁶ (4.9·10⁻⁷) | 2.51·10⁻⁶ | 30.4 | **0.58×** |
| 4.67·10⁻⁴ | `ddebug` | 227 | 227 / 0 | 239 s | 34 | 8 | 2.15·10⁻⁶ (4.2·10⁻⁷) | 2.15·10⁻⁶ | 13.5 | 1.29× |
| 10⁻⁶ | `bin_simple` | 12 | 9 / 3 | 17 s | 15 | 0 | 2.10·10⁻⁷ (4.1·10⁻⁸) | 2.10·10⁻⁷ | 16.7 | 1.05× |
| 10⁻⁶ | `ddebug` | 211 | 128 / 83 | 223 s | 37 | 8 | 7.15·10⁻⁷ (1.4·10⁻⁷) | 7.15·10⁻⁷ | 13.5 | 1.29× |

The `f64` program takes 17.5 ms a call on one X925 core (the median of 21 calls). Presk's own C export of the solver's realization takes 11.4 ms, and of all-`f32` 11.2 ms. The configurations tested exclude CRAFT's baseline run and its test of the final configuration.

- **At ε\*(10⁻³), `bin_simple` stops at its first trial:** the whole program in `float` passes. That is also the only passing configuration it tested, so it is its answer, and CRAFT reports "No speedup found". The program is slower than the `f64` one:
  - the parameters stay `double`, so every product multiplies a `float` by a `double`, in `double`, and rounds the sum back to `float`;
  - the literals are `double`, so `x / 64.0` widens too.
- **`ddebug` searches for speed,** since every configuration passes. It keeps in `double`:
  - the three products' register tiles (`k10`, `k21`, `k24`: `v0`…`v3`, and `k10`'s `s0`);
  - the softmax's maximum and sum (`k16:s0`, `s1`) and the mean's sum (`k6:s0`);
  - the embedding's buffers (`GPT.apply/%5`, `%8`, `%9`, `%14`) and the fold's state.

  Everything else is `float`. The search depends on the timings. An exploratory run of the same search the same morning tested 206 configurations and kept 22 variables in `double`.
- **At 10⁻⁶, `bin_simple`:**
  - the whole program fails (2.5·10⁻⁶), and so does `model.c`'s 39 buffers (2.3·10⁻⁶);
  - `kernels.c`'s 24 locals in `float` pass (10⁻⁶), at 0.24× the speed;
  - each host function then passes alone;
  - the output alone fails (2.1·10⁻⁶): the head's kernel `k25` accumulates its product in its output buffer, so retyping that buffer retypes the accumulation.

  The fastest passing configuration is `SelfAttention.apply`'s 15 buffers (1.05×).
- **At 10⁻⁶, `ddebug`** keeps the products' tiles, the softmax's sum, the mean's sum, the last layer norm's output and the output in `double`: 18 variables. It runs 1.29× the `f64` program.

**ADAPT does not run on the export.** Its own example (`sum2pi_x`) builds and runs on falicon. It needs CoDiPack v2.0.2 (2022-06-14) and C++17: ADAPT's implementation uses CoDiPack 2's names (`Tape`, `Identifier`), which v1.9.3 lacks, and CoDiPack's master of 2026 no longer compiles with it. On the export, three things stop it, each a fact of the run:
- *ADAPT is C++, the export C11.* Compiled as C++17, `model.c` gives 248 errors: `restrict`, which C++ lacks, and 41 `presk_alloc` results that C++ will not convert from `void *` without a cast. `main.c` gives 24 errors and `presk_rt.c` 3.
- *The arena holds bytes.* The sizes are written `N * sizeof(double)`, so an AD type could be allocated by changing `sizeof`, and the arena's total (`PRESK_LOGITS_ARENA`, 17 010 816 bytes) is a literal for 8-byte values. But CoDiPack's active real is a C++ object, a value and a tape index, and a bump allocator from C constructs none.
- *The instrumentation is the user's.* ADAPT records an error contribution at each `AD_INTERMEDIATE(x, "name")` the source places after an assignment: one in each of the 26 kernels' stores at the least, the analysis's choice of what a variable is.

  That is a port of the program to C++, not the tool on the export. The prompt says to look and not to change the export.

## Tier B: the tools' searches on presk's space

| budget | search | space | trials | distinct realizations | `f64` / `f32` / `bf16` of 52 | traffic vs all-`f64` | gap, conformance inputs |
|---|---|---|---|---|---|---|---|
| 4.67·10⁻⁴ | Precimonious's | 31 stored values | 716 | 460 | 21 / 24 / 7 | 2.11× | 3.57·10⁻⁴ |
| 4.67·10⁻⁴ | CRAFT's | 31 stored values | 43 | 42 | 21 / 24 / 7 | 2.11× | 3.57·10⁻⁴ |
| 4.67·10⁻⁴ | Precimonious's | the 31 and the 21 parameters | 2 858 | 1 690 | 0 / 36 / 16 | **2.17×** | 2.76·10⁻⁴ |
| 4.67·10⁻⁴ | CRAFT's | the 31 and the 21 parameters | 70 | 69 | 0 / 36 / 16 | **2.17×** | 2.76·10⁻⁴ |
| 10⁻⁶ | Precimonious's | 31 stored values | 816 | 427 | 25 / 22 / 5 | 1.98× | 9.22·10⁻⁷ |
| 10⁻⁶ | CRAFT's | 31 stored values | 31 | 31 | 40 / 10 / 2 | 1.53× | 7.29·10⁻⁷ |
| 10⁻⁶ | Precimonious's | the 31 and the 21 parameters | 1 596 | 958 | 3 / 43 / 6 | 1.89× | 9.80·10⁻⁷ |
| 10⁻⁶ | CRAFT's | the 31 and the 21 parameters | 33 | 33 | 40 / 10 / 2 | 1.53× | 7.29·10⁻⁷ |

The two strategies find **the same realization** at ε\*(10⁻³), in either space. CRAFT's takes 43 and 70 trials, Precimonious's 716 and 2 858. With the stored values alone (the parameters in `f64`), the realization is already 2.11×, above the solver's 1.97× with the parameters typed.

**Against the solver at ε\*(10⁻³)** (`search/*-params-4.67e-4/final.json` against `solver/solved-4.67e-4.json`), 16 decisions differ:

| decision | solver | searches | elements | bytes saved |
|---|---|---|---|---|
| `SelfAttention.apply/%20`, the attention's dropout mask (ones) | `f32` | `bf16` | 524 288 | 1 048 576 |
| `Block.apply/%19`, `%27`, two residual dropout masks (ones) | `f32` | `bf16` | 2 × 65 536 | 262 144 |
| `GPT.apply/%14`, the embedding's dropout mask (ones) | `f32` | `bf16` | 65 536 | 131 072 |
| `SelfAttention.apply/%15`, `%16`: 32 and √32 | `f32` | `bf16` | 2 × 1 | 4 |
| `LayerNorm.apply/%4`, the centred squares | `f64` | `f32` | 65 536 | 262 144 |
| `LayerNorm.apply/%10`, the standard deviation | `f64` | `f32` | 1 024 | 4 096 |
| eight biases and shifts: `wq.b`, `wv.b`, `wo.b`, `up.b`, `down.b`, `ln1.β`, `ln2.β`, `lnf.β` | `f32` | `bf16` | 1 344 | 2 688 |
| **total** | | | | **1 710 724** |

- **The four tensors of ones are 84 % of the bytes.** They are this entry's dropout masks: the key is dropped, so the masks are constant ones. 1.0 and 32.0 are exact in `bf16`, and √32 is 1.1·10⁻⁴ off. The solver's estimate charges a site `s · u(bf16) · |v|`: 1.7 to 7.6·10⁻³ for these masks (`u(bf16)` = 2⁻⁸), each more than the whole budget. So it cannot see that rounding these values changes no bit. That is a finding for later about the solver's estimate, not a change here.
- **The layer norm's two values** pass in `f32` where the estimate kept them in `f64`. The estimate is conservative there: 4.1·10⁻⁴ estimated for 3.2·10⁻⁵ measured, D124.
- **At 10⁻⁶** the two strategies differ, and the larger space gives Precimonious's search a wider answer (1.89× against 1.98×). Delta debugging finds a local minimum, and the order of the variables decides which.

## Judged by the compiler on the 461 held-out windows

`out/precision-tuning-0.15.0/margins.md`, 118 016 positions against the exact logits at 128 bits (D118's). The traffic is the solver's cost. `cond` is `presk check --precision`'s bound on the envelope, absolute, against logits of magnitude 5.2.

| realization | gap on the logits | flips | bound #{m < 2ε} | cert. 10⁻² | cert. 10⁻³ | cert. 10⁻⁴ | traffic vs all-`f64` | bound `cond` |
|---|---|---|---|---|---|---|---|---|
| all-`f64` | 5.89·10⁻¹⁵ | 0 | 0 | yes | yes | yes | 1.00× | 5.7·10⁻⁶ |
| all-`f32` | 3.12·10⁻⁶ | 0 | 0 | yes | yes | yes | 2.00× | 3.1·10³ |
| all-`bf16` | 5.03·10⁻² | 480 | 13 661 | no | no | no | 4.00× | 1.2·10⁷ |
| **the solver at 4.67·10⁻⁴** | **7.53·10⁻⁵** | **0** | 18 | yes | yes | no | **1.97×** | 6.6·10⁶ |
| **tier B at 4.67·10⁻⁴, with the parameters** (both searches) | **3.69·10⁻⁴** | **0** | 80 | yes | yes | no | **2.17×** | 7.2·10⁶ |
| tier B at 4.67·10⁻⁴, the stored values (both searches) | 3.35·10⁻⁴ | 1 | 70 | yes | yes | no | 2.11× | 5.4·10⁵ |
| CRAFT `bin_simple` at 4.67·10⁻⁴, its C program | 3.26·10⁻⁶ | 0 | 0 | yes | yes | yes | — | — |
| CRAFT `bin_simple` at 4.67·10⁻⁴, as a realization | 3.12·10⁻⁶ | 0 | 0 | yes | yes | yes | 1.95× | 3.0·10³ |
| CRAFT `ddebug` at 4.67·10⁻⁴, its C program | 2.95·10⁻⁶ | 0 | 0 | yes | yes | yes | — | — |
| CRAFT `ddebug` at 4.67·10⁻⁴, as a realization | 3.13·10⁻⁶ | 0 | 0 | yes | yes | yes | 1.81× | 2.9·10³ |
| **the solver at 10⁻⁶** | **8.66·10⁻⁷** | 0 | 0 | yes | yes | yes | 1.27× | 6.6·10⁶ |
| tier B at 10⁻⁶, Precimonious's, the stored values | 1.60·10⁻⁶ | 0 | 0 | yes | yes | yes | 1.98× | 4.4·10⁵ |
| tier B at 10⁻⁶, Precimonious's, with the parameters | 1.69·10⁻⁶ | 0 | 0 | yes | yes | yes | 1.89× | 7.1·10⁶ |
| tier B at 10⁻⁶, CRAFT's (either space) | 1.41·10⁻⁶ | 0 | 0 | yes | yes | yes | 1.53× | 7.3·10³ |
| CRAFT `bin_simple` at 10⁻⁶, its C program | 3.44·10⁻⁷ | 0 | 0 | yes | yes | yes | — | — |
| CRAFT `bin_simple` at 10⁻⁶, as a realization | 1.41·10⁻⁶ | 0 | 0 | yes | yes | yes | 1.47× | 1.8·10³ |
| CRAFT `ddebug` at 10⁻⁶, its C program | 9.65·10⁻⁷ | 0 | 0 | yes | yes | yes | — | — |
| CRAFT `ddebug` at 10⁻⁶, as a realization | 1.65·10⁻⁶ | 0 | 0 | yes | yes | yes | 1.88× | 2.9·10³ |
| the export's own C program (the control) | 5.89·10⁻¹⁵ | 0 | 0 | yes | yes | yes | — | — |

Every row's accuracy is the exact model's, 0.242442, but all-`bf16`'s (0.242425).

- **At ε\*(10⁻³), the narrower realization holds out of sample.** The searches' realization with the parameters is 3.69·10⁻⁴ from the exact logits over the 461 windows, inside the budget. It flips nothing and certifies at 10⁻² and 10⁻³, as the solver's does. It spends 79 % of the budget where the solver's spends 16 %, and its bound #{m < 2ε} is 80 positions against 18. Without the parameters it is 3.35·10⁻⁴ with one flip, which certification at 10⁻³ allows (at most 118).
- **At 10⁻⁶, everything the searches found on the two windows fails on the 461:** 1.41 to 1.69·10⁻⁶ against a budget of 10⁻⁶. The solver's realization holds (8.66·10⁻⁷). The searches tune to the windows they are tested on, and spend the budget to within 2–27 % on them. The solver stops with room left: its estimate, 9.8·10⁻⁷, is 2.8 times its measurement.
- **CRAFT's C programs are not their realizations.** As programs, all four hold at their budget, `ddebug`'s at 10⁻⁶ by 3.5 %. As realizations, the two at 10⁻⁶ fail (1.41·10⁻⁶ and 1.65·10⁻⁶): presk computes a site in its own type, where the C programs accumulate their `float` buffers in `double`. `ddebug` kept the products' tiles in `double` for speed, and that is what kept its program inside the budget.
- **`cond` is vacuous for every realization but all-`f64`:** 1.8·10³ to 1.2·10⁷ against logits of 5.2, the tools' included. It is the bound the compiler can state for any realization; the tools state none. The certificate that holds is the margins'.

## What the tool cannot state

Each item is a fact of this run.

1. **A bound.** CRAFT reports a pass or a fail on the two windows it ran, and a runtime. The compiler states `cond` for any realization, the tools' included, conditional on the envelope (vacuous here at 10³–10⁷). It also states the margins' certificate, which holds on the 461 windows for every row at ε\*(10⁻³) but all-`bf16`.
2. **An exact reference.** CRAFT's test compares what its user's script compares. CRAFT's own sample driver checks a line of output. Here the test was given `conformance/exact.psk.d`, the exact reference's outputs on its inputs, which only the compiler produces. At these budgets the `f64` run would have served (4.4·10⁻¹⁵ away), and the tools' gaps against it are the same to three digits. The 461-window judgment needs the exact logits at 128 bits, which `make margins` takes in four minutes on eight cores.
3. **A budget on a metric.** The criterion is a gap on two windows. That this gap means at most δ of the decisions flipped, and that it holds on held-out data, comes from the margins on 461 windows. At 10⁻⁶ the tools' passes on two windows did not hold on them.
4. **A parameter's storage type.** The 21 parameters are inputs of the C program (`theta_*`), which `main.c` reads from the weights file as `f64`. The tool's realizations leave them in `f64`, and every product mixes types: the all-`float` program is 0.58× the `f64` one. The solver stores 20 of them in `f32` and one in `bf16` (5.13·10⁵ bytes at rest against 1.03·10⁶). The searches, given them, put 9 in `bf16`.
5. **`bf16`.** Plain C has no such type; the tool's space is `double` and `float`. Of the 16 decisions where the searches go beyond the solver, 14 are `bf16` (1.44·10⁶ of the 1.71·10⁶ bytes); the other two are the layer norm's values in `f32`.
6. **The compiler's space against the C's declarations.**
   - *What the C has.* 63 variables, against presk's 52 decisions:
     - its 39 buffers hold all 31 stored values, so no stored value is out of the script's reach in this export;
     - three stored values are held by several declarations: `Linear.apply/%5` by 6, `Linear.apply/%3` and `LayerNorm.apply/%15` by 3. So the C can type one call's result apart from another's, and at 10⁻⁶ CRAFT did, for `Linear.apply/%5` and `LayerNorm.apply/%15`;
     - one declaration holds two stored values, the fold's state (`GPT.apply/%15` and `Block.apply/%29`);
     - its 24 kernel locals have no site. They are the accumulators and register values presk computes in the site's type, and the C can accumulate a `float` buffer in `double`.
   - *Where a declaration is more than a buffer.* The head's kernel `k25` accumulates in its output buffer, so the output's declaration is also an accumulator: at 10⁻⁶ the output alone in `float` fails (2.1·10⁻⁶). The fused values that never reach memory, such as the layer norm's `(x − μ)/σ`, are neither presk decisions nor C declarations. A kernel's epilogue, the bias of a product, is a separate kernel in this export (`k11`, `k22`).
   - *What presk has.* 21 parameters and `bf16` on every decision; the C has neither.

## What it says

**For §7.** The comparison §7 argues can be measured, and it shows three things.
- *On the C, the tool works but cannot express the solver's realization.* It found programs in `double`/`float` that pass the criterion and hold on the 461 windows. But the realization the solver gives has `bf16` and typed parameters, and the tool's best has neither: as a realization it is 1.81–1.95× against 1.97×. Its objective is the C program's speed, and its answer is the fastest program, at most 1.29× the `f64` one, where presk's own C of the solver's realization is 1.54×.
- *It needs what the compiler supplies.* To run at all, the tool needed three things: the exact outputs, which only the compiler ships; a retyping script, the stand-in for TypeForge (a ROSE source-to-source tool not on this machine); and a statement of which variables exist, which here came from the compiler's comments.
- *Search beats the solver's proposal on the compiler's own space.* Given that space and the exact reference, the tools' searches beat the solver's one-sweep proposal by 9 % of its traffic at ε\*(10⁻³), out of sample too. The cost is 43–2 858 trials and a budget spent to 79 %.

  It cannot be claimed that the solver's realization is the narrowest that passes. What can be claimed is that the solver's realization comes from one sweep and one measurement, with an estimate and a bound, and that a search over the same space can narrow it further. Such a search needs the exact reference to test against, and held-out windows to tell an answer that holds (ε\*(10⁻³)) from one that overfits (10⁻⁶).

**For §4's claim about the space.** "The program's stored values and the parameters' storage types, not a source's variable declarations": measured, the two spaces are not the same space.
- *The C has more variables.* 63 against 52: the kernels' locals, and one stored value split over up to six declarations.
- *It ties two stored values into one* (the fold's state).
- *It cannot express two things the solver's decisions use:* `bf16`, and the parameters.

What the searches gained over the solver comes from exactly those two.

## What it does not say

- **One program at D = 64 and one machine.** Two budgets, one from the margins and one chosen to exercise the searches.
- **The tool at its defaults, with one stand-in.** CRAFT's strategies were run as they stand, their final configuration the fastest. The retyping is the prompt's stand-in for TypeForge: it specializes functions as templates would, which TypeForge's type-dependence sets do not, and so gives the tool a finer space than TypeForge's. No FloatSmith, no emulation, no Precimonious binary; HiFPTuner not looked at.
- **Tier B is not the tools.** It is their search strategies, reimplemented on presk's realizations, under the solver's cost and not a runtime.
- **Timing-dependent searches.** The C programs' times are notes on one core, not of record. `ddebug`'s answer depends on them: an exploratory run tested 206 configurations against this run's 227, and kept 22 variables in `double` against 21.
- **Inputs that are not text.** The conformance inputs are two windows of random tokens, the export's. The solver measured on four windows of text, and the searches tuned to the two random ones. The 461 windows are text.

## Files

- `experiments/tuning/` (`make precision-tuning`): no script of record changed.
  - `tuning.sh`, the run;
  - `setup.sh`, the tools fetched into `.tuning`;
  - `retype.py`, the variables, the rewrite and the realization of an assignment;
  - `criterion.py`, the test;
  - `craft.sh`, one CRAFT search with its builder and test;
  - `search.py`, tier B;
  - `windows.py`, a C program on the 461 windows;
  - `report.py`, the tables.
- `out/precision-tuning-0.15.0/`:
  - `report.md`, `margins.md` and `bounds.txt`, the tables;
  - `c-f64/`, the export;
  - `solver/`, the envelope, the solves and the space;
  - `craft/<strategy>-<budget>/`, CRAFT's search directory: its configurations passed and failed, `craft.out`, `trials.tsv` with every trial's assignment in `trials/`, and `final/`, the final program;
  - `search/<strategy>-<space>-<budget>/`, each search's `trials.tsv` (every realization tried, with its gap and cost), `final.json` and `summary.json`;
  - `margins/`, the rows of the 461 windows, `make margins`' rows linked;
  - `adapt/`, ADAPT's example and the export compiled as C++;
  - `c-times.txt`, the notes;
  - `tools.txt` and `machine.txt`.
- Archived as presk-project `data/precision-tuning-0.15.0.tar.xz`, without what the scripts write again: the native rows of the 461 windows, the input windows, the C binaries. Their checksums are in `EXCLUDED.sha256`. The tools' sources are not archived; `tools.txt` names their commits.
