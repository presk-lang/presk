# Experiments: exact versus almost

What the tool can show about the approach, with numbers, and what it cannot.
§1 and §2 are the first two experiments: `make experiment` runs both (after
`make venv`), with the script `experiments/exact_vs_almost.py`, which takes
any output directory of `presk export … --conformance-entry train:K`. §3 is
0.4.0's decisive experiment against PyTorch (`make experiment-gpt`), §4 the
statements of 0.5.0, §5 MLPerf's acceptance rule set beside a budget on one
model (`make foil`), §6 the statements of 0.6.0, §7 those of 0.7.0, the
first test of the research programme's hypothesis RA4 (`make ra4`), §8
those of 0.8.0, its second, with a model frozen before two fresh sweeps
(`make ra4-f2`, `make ra4-f1`), and §9 those of 0.9.0, a C export
(`make c-conformance`, `make c-speed`). Each
version named its statements before it began, with what would count as
failure, and each section reports them as measured.

## 1. Shapes as types

The RMSNorm of the tutorial divides `x : ℝ[..., model: D]` by `ms : ℝ[...]`,
the per-row mean of squares. In Presk the two align by axis name, and the
checker proves it for every `B` and `D`. In plain PyTorch the natural line
`x / torch.sqrt(ms)` broadcasts from the trailing axis: it divides column `j`
by the norm of row `j`. When the batch happens to equal the model width, it
runs without any error.

```text
plain PyTorch, batch == model width: `x / sqrt(ms)` runs without error, max |wrong - right| = 0.773
Presk: `x / sqrt(ms)` with x : ℝ[..., model: D], ms : ℝ[...] aligns by name; the export inserts unsqueeze(-1)
```

The export had to learn the same lesson: it inserts `unsqueeze(-1)` exactly
where Presk's alignment (by leading prefix or by name) and PyTorch's (from the
trailing axis) differ. The `E2003` and `E2006` diagnostics of the tutorial,
with their symbolic counterexamples, are the compile-time side of the same
point: a shape error is a type error, found before the first batch.

## 2. The realization gap

`presk export --conformance-entry train:K` records the initial parameters,
the sample indices of every batch and the parameters after `K` steps of the
reference interpreter at 256 bits. The script replays those `K` steps in
PyTorch, on the same batches, with the same optimizer and schedule, in
float32 and in float64, and measures the largest absolute gap to the exact
parameters. Since 0.3.0 the export also runs the same steps on presk's own
native backend (`--backend native`, `native` chapter) in `f64` and `f32`,
and the script measures those too: four realizations of one program against
one reference.

| Run | Steps | PyTorch float32 | PyTorch float64 | presk `f32` | presk `f64` | Parameter scale |
|---|---|---|---|---|---|---|
| Tutorial tower (`tutorial/07_train.psk`, AdamW, cosine warm-up) | 10 | 3.2e-7 | 4.4e-16 | 2.0e-7 | 4.4e-16 | 1.0 |
| Tiny MiniGPT (`train_minigpt.psk`, weight decay 0.1, clipping 1.0, cosine warm-up) | 20 | 9.6e-7 | 8.9e-16 | 3.7e-7 | 6.7e-16 | 1.0 |

Read against the machine epsilons (6e-8 for float32, 1.1e-16 for float64),
all four arithmetics lose a handful of units in the last place over ten to
twenty AdamW steps on these small problems: float64 is exact to fifteen
digits, float32 to six. Presk's kernels are two to three times closer than
PyTorch's in float32, because they follow the reference operation for
operation (no fused or reassociated kernel); this is a property of naive
kernels that the e-graph pass will have to preserve or account for. Two
remarks on the measurement itself:

- The gap is *absolute*, with the model's scale given for context. An
  elementwise relative gap is meaningless here: some parameters are kept near
  zero by both arithmetics (the zero-initialized head of MiniGPT in its first
  steps), and dividing by them turns rounding noise into enormous ratios.
- What is measured is the arithmetic alone. Shuffling, initialization and the
  optimizer's definition are identical on both sides, because the export
  carries the interpreter's batches and parameters and the optimizer is a
  function of the language tested against PyTorch's (`adamw.step`, rtol
  1e-9). In an ordinary PyTorch workflow there is no reference to measure
  against; the "right" answer is whatever float32 produced.

These are the first two points of the specification's "exact versus almost"
experiment, and of its ε-to-quality curve: the error of a realization is a
number attached to a program, measured, not argued. Longer runs, larger
models, bfloat16 and mixed precision are the obvious next points; they need
the PyTorch side only (`train.py --dtype`), the reference being fixed.

## 3. The decisive experiment: against PyTorch

Version 0.4.0 set three falsifiable statements and trained the course GPT to
answer them (`D = 384`, 6 heads, 6 blocks, context 256, batch 16, the
character-level Code civil, `f32`, twelve cores, `presk train` against the
`train.py` the export writes). The answers:

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| A | presk trains within 3× of PyTorch CPU per step | ≤ 3× | **94×** (82× with `--opt`) | fails |
| B | `--opt --profile` is worth at least 1.3× | ≥ 1.30× | **1.14×** | fails |
| C | the profile predicted that speedup within 30 % | ≤ 30 % | **5 %** | holds |

One training step is about 266 GFLOP. PyTorch did it in 0.405 s, presk in
37.9 s, 33.3 s with the optimizer: 656 GFLOP/s against 7.0. Those are the
numbers of that machine and that version; the re-measurement below is this
machine's.

### The same experiment on falicon, 23 September 2026

The numbers above are 0.4.0's, on the machine it was written on. This is the
same experiment on the machine the compiler now runs on (falicon: aarch64,
twenty cores — ten Cortex-X925 and ten Cortex-A725 at 2.8 GHz — SVE at
128 bits, no matrix coprocessor reachable by either side; PyTorch 2.14, a
CPU-only wheel, so the machine's GB10 GPU takes no part). Same model, same
data file, same batches, same six steps, `f32` and twenty threads on both
sides, each side reporting the median of its five timed steps with the first
excluded, three repetitions of each row:

| side | the three runs (ms) | median (s) | vs PyTorch | GFLOP/s |
|---|---|---|---|---|
| PyTorch 2.14 `float32` | 1332, 1338, 1342 | **1.34** | 1.0× | 199 |
| presk `f32` | 29469, 29960, 31440 | **29.96** | 22.4× | 8.9 |
| presk, step solved at 10⁻³ (J′) | 30075, 30611, 30995 | **30.61** | 22.9× | 8.7 |

**The factor is 22×, not 94×, and fourteen of it is one kernel.** On this
machine a 1024³ `f32` product runs at **710 GFLOP/s** under PyTorch and at
**49.6 GFLOP/s** under presk's blocked kernel with lanes (`presk bench
out/gemm.psk --entry gemm --realize f32`), a factor of 14.3 on the
contraction alone. PyTorch's side of that is not a matrix unit: this wheel
carries its own OpenBLAS 0.3.34 (built `DYNAMIC_ARCH USE_OPENMP`, selecting
its `neoversev2` kernels) and oneDNN 3.12 over Arm Compute Library 53.2, and
turning oneDNN off leaves 680 GFLOP/s, so the two paths are within 4 % of
each other here. The cores report `sve sve2 i8mm bf16` and **no `sme`**, so
710 GFLOP/s is about four fifths of what 128-bit SVE fused multiply-add
allows on twenty cores at 2.8 GHz, and presk's 49.6 is about a twentieth of
it. On the machine of 0.4.0 the same product ran at 1309–1454 GFLOP/s under
PyTorch, through a matrix coprocessor its platform BLAS could reach, and at
47–56 under presk.

So neither side moved for a reason inside the compiler. Presk's matmul rate
is where it was (47–56 → 49.6) and PyTorch's halved for want of that
coprocessor (≈1400 → 710); that accounts for a factor of 14 of the 22, and
the remaining 1.6 is everything that is not a contraction. Each side's
training step runs below its own matmul rate — presk's at 18 % of it, 8.9
against 49.6, PyTorch's at 28 %, 199 against 710 — where on the earlier
machine the same two figures were 14 % and 45 %. Presk has gained a little
on the rest of the step and PyTorch has lost a good deal of it, and that,
with the missing coprocessor, is what turned 94 into 22. The honest reading
of §3 is unchanged in kind and smaller in size: a mature stack is still an
order of magnitude ahead of these kernels, and the reason is the
instruction set they reach, not the discipline.

The third row is the realization the solver chose for one training step
under a 10⁻³ budget on its update, statement J′ of §6. Against PyTorch it
buys nothing, and that is not news: J′'s 1.99× is against all-`f64`, and
0.6.0 already reported the solved step level with all-`f32`. Here it is
30.61 s against 29.96, inside the spread of the three `f32` runs
(29.47 to 31.44 s). A budget bought time against the exact end of the menu,
not against the type a practitioner would have written.

Two things make this comparison fairer than 0.4.0's. Both sides now **drop**:
`make experiment-gpt` had been broken since 0.5.0, and the exported loop ran
its dropout at the expectation while `presk train` dropped, so PyTorch was
doing less work (D54). The masks differ — PyTorch draws its own, presk
derives its from the step's key — so the **losses are comparable to about
four digits and no further** (4.507609 against 4.507878 after six steps),
while the timings are of the same computation. And both sides are timed per
step, the first step excluded, rather than by total wall time, so neither
pays for the other's warm-up (the native backend compiles its kernels on the
first step, PyTorch allocates on its).

**Where the factor of ninety-four is.** It is not distributed through the
compiler; most of it is one thing. PyTorch's CPU matmul reaches Apple's
matrix coprocessor through the platform BLAS and runs a 1024³ `f32` product
at 1309–1454 GFLOP/s — a rate that barely moves with the thread count,
because it is not thread parallelism. Presk's own blocked kernel, scalar
loops over twelve threads, runs the same product at 47–56 GFLOP/s. That is
a factor of about twenty-six, and it is an *instruction-set access* gap, not
a compiler-cleverness gap.

The remaining factor of three is presk's, and it is visible in the same
numbers: PyTorch's training step runs at 45 % of its own matmul rate, presk's
at 14 % of its own. The difference is everything that is not a contraction —
softmax, normalization, the elementwise chains reductions keep splitting, and
2274 kernel launches a step.

**What that does and does not say about the thesis.** The claim under test is
that exact semantics plus rewriting plus static cost models buy speed. This
version bought 7.3× (289 s a step to 39.8 s) and *every* transformation it
shipped is exact: fusion, threads, blocked contractions and the scalar
rewrite all leave the training losses identical to every digit printed. The
discipline held under a sevenfold speedup, which is the part worth keeping.

What it did not do is close the gap to a mature stack, and the reason is
recorded rather than excused. The kernels this compiler emits are scalar
because the two ways to go faster both reassociate a sum — vectorizing the
contraction's inner loop, and calling a BLAS whose summation order is not
ours — and a reassociated sum is a different realization in this system,
owing a declared order and a measured gap (decisions D31 and D32). That is a
real cost of the position, and the honest form of the result is: the
discipline is affordable at 7.3×, and the next order of magnitude requires
admitting a new realization and measuring it, not writing a faster loop.
(0.5.0 then found a third way that does not reassociate — vector lanes across
the output columns of the blocked form, D42 — and 0.6.0 emits it; §6 says
what it bought.)

Matching a tuned BLAS is a target this project will have to take on, for the
CPU's vector units and matrix coprocessor and for the GPU, with its own
kernels rather than a dependency; `docs/kernels-roadmap.md` records what that
needs and why it is a research question here — every one of those paths
reassociates a sum, so each is a realization owing a declared order and a
measured gap.

Statement A, re-posed on falicon, still fails: 22× against the 3× it asked
for. Statement C is the one that held, and it is the smallest of the three: asked
to predict its own speedup, the calibrated model said 1.087× and measurement
said 1.14×. The model is right about what it is asked; it is asked very
little, because the rule set offers few decisions that turn on size
(`optimization` chapter).

## 4. Precision as a resource: the four statements of 0.5.0

Version 0.5.0 named four falsifiable statements before it began
(`presk-specs/implementation/presk-plan-0.5.0.md` §3.7), in the form 0.4.0
used. One holds, two fail, and one cannot be asked of anything that exists.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| D | the solved realization's **measured** gap is inside a 1e-3 budget | ≤ 4.52e-3 | **1.05e-7** | holds |
| E | the solved realization is faster per step than all-`f64` | ≥ 1.5× | **0.96×** | fails |
| F | the propagated bound is within 10³ of the measured gap on half the entries | ≥ 11 of 21 | **10 of 21**, median 1.8e3 | fails |
| G | the vectorized GEMM's gap is inside its declared order's bound | — | — | not askable |

**D holds, and it is the one that matters most.** A budget declared in the
program becomes an assignment whose distance from the exact denotation is
*measured*, not asserted, and the measurement is inside the budget by four
orders of magnitude. No other compiler can run this check, because no other
compiler keeps the exact denotation to measure against.

**E fails, and the reason is the version's own finding turned against it.**
The solver minimises **traffic**, because traffic is what a narrower type
reduces. At these shapes the kernels are not traffic-bound, so narrowing buys
nothing and the conversions cost: the solved realization runs at 5.63 ms
against all-`f64`'s 5.39, and uniform `bf16` at 6.05. Step 2 measured exactly
this on a single kernel and D41 wrote it down as the reason the objective is
traffic and not time. The consequence was still not drawn: **optimising the
right proxy in the wrong regime is not optimising**. Closing E needs two
things, both already named — the roofline's `W` per site so the objective can
be time, and vector lanes so that arithmetic width buys time at all.

**F fails narrowly and informatively.** Ten of twenty-one entries are within
10³ and the median ratio is 1.8e3; on a normaliser the bound is within a
factor of 27, on the corpus GPT it is 10²⁰. The threshold was set generously
on purpose, because the interesting outcome was the number and not the
verdict. The number says the bound is worth having on a kernel and worthless
on a network, and locates why: a contraction of `k` terms multiplies error by
`k` where committed errors grow like `√k`.

**G cannot be asked.** It is about a vectorized GEMM, and there is none. The
step that would have built it found instead that vectorizing across output
*columns* is bit-exact (D42), so the kernel G describes should be built under
a sequential order and would have no realization gap for G to measure. The
statement needs restating before the next version can answer it — against the
exact vector kernel, with the margin printed beside the verdict rather than a
bare pass. It became 0.6.0's statement H (§6, D43).

**What the four together say.** The claim that precision can be *declared*
and *verified* is demonstrated: D holds by measurement. The claim that it
**pays** is not: E fails, and it fails for a reason this version measured
twice before believing it. That is the same shape as 0.4.0, where the
discipline was shown affordable and the speed was not there yet, and it puts
the same item at the top of the next version's list: make a narrower type buy
time, not only bytes.


## 5. Two acceptance rules, one model

MLPerf's closed division accepts a realization by its task accuracy over the
whole validation set, within 99 % or 99.9 % of an FP32 reference's; a budget
accepts it by its measured distance to the exact denotation, on the
envelope's inputs. The `coverage` chapter has the argument and D53 the
decision (0.6.0 step 8). `make foil` applies both rules to the same eight
realizations of the corpus GPT at `D = 64`. The weights are the all-`f64` run
of the `precision` chapter's curve, trained for 600 steps on 90 % of the Code
civil, so the other tenth — 461 windows, 118 016 next-character predictions —
is held out.

- **MLPerf's side** is next-character accuracy over the held-out windows (a
  new entry of `corpus/nlp/05_gpt.psk`, `accuracy`), with the loss beside it
  because perplexity on a held-out corpus is the roadmap's declared NLP
  measure. The model runs in the realization and the metric in `f64`, as
  MLPerf's harness scores a model from outside it: in `bf16`, the mean over
  118 016 positions would round the metric itself to eight bits.
- **The budget's side** is `presk quantize` on two held-out windows: the gap
  to the exact loss, split into the weights' rounding and the arithmetic.

| realization | gap to the exact loss | representation | arithmetic | check (s) | accuracy | vs `f32` | 99 % | 99.9 % | loss vs `f32` | pass (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `f64` | 8.81·10⁻¹⁶ | 0 | 8.81·10⁻¹⁶ | 0.11 | 0.157733 | 0 | yes | yes | 0 | 11.1 |
| `f32` | 2.24·10⁻⁷ | 0 | 2.24·10⁻⁷ | 0.09 | 0.157733 | — | yes | yes | — | 4.6 |
| `bf16` | 2.96·10⁻³ | 0 | 2.96·10⁻³ | 0.12 | 0.157724 | −5.4·10⁻⁵ | yes | yes | +6.6·10⁻⁵ | 3.7 |
| `int8`, `f32` elsewhere | 1.46·10⁻⁴ | 1.46·10⁻⁴ | 4.92·10⁻⁸ | 0.08 | 0.157733 | 0 | yes | yes | +1.5·10⁻⁶ | 4.4 |
| `int8` `pow2` | 1.14·10⁻⁴ | 1.14·10⁻⁴ | 4.09·10⁻⁷ | 0.08 | 0.157724 | −5.4·10⁻⁵ | yes | yes | −4.0·10⁻⁶ | 4.9 |
| `int4`, `f32` elsewhere | 1.02·10⁻³ | 1.02·10⁻³ | 1.72·10⁻⁸ | 0.08 | 0.157674 | −3.8·10⁻⁴ | yes | yes | −9.0·10⁻⁵ | 4.8 |
| `int4` `pow2` | 1.56·10⁻³ | 1.56·10⁻³ | 2.92·10⁻⁷ | 0.10 | 0.157724 | −5.4·10⁻⁵ | yes | yes | +9.1·10⁻⁴ | 4.5 |
| solved at 10⁻³ (statement D's) | 2.41·10⁻⁷ | 0 | 2.41·10⁻⁷ | 0.09 | 0.157733 | 0 | yes | yes | +5.3·10⁻⁹ | 8.8 |

The exact loss of the two windows, 3.0245777, takes 7.5 s at 128 bits, once.
A check is one native run of those windows in the realization; a pass is one
run of the accuracy entry over the 461 held-out windows. Both are wall times
of one `presk run`, start-up included. One prediction is 5.4·10⁻⁵ of `f32`'s
18 615 correct ones.

- **The accuracy rule admits all eight**, at 99.9 % as at 99 %. The largest
  change is seven predictions in 118 016. The budget separates the same
  realizations over twelve orders of magnitude, four of them between `f32` and
  `bf16`, from two windows.
- **The accuracy is not monotone in the gap.** `int4 pow2` is further from
  the exact loss than `int4` and loses one prediction where `int4` loses seven.
  A thresholded metric reads the computation coarsely and with noise, which
  is what it is designed to do: it asks whether the task survived, not how
  far the arithmetic went.
- **Attribution.** Every quantised gap is the weights' and not the kernels':
  the representation part is 10² to 10⁵ times the arithmetic part. The
  accuracy rule has no way to say so, because it has a reference score and no
  reference computation.
- **Cost.** Per realization a check takes about 0.1 s and a pass about 4 s;
  the exact reference, paid once per checkpoint and sample, costs about two
  passes. On BERT-large the same accounting is in hours (`coverage` chapter).
- **Statement K, on other windows.** On this checkpoint and these two held-out
  windows `int8` weights are 1.46·10⁻⁴ from the exact loss, where K measured
  3.23·10⁻⁶ on `make gpt-trained`'s weights and two training windows. That is
  forty-five times more, and still seventy times inside 10⁻². The error of a
  quantised loss is a signed sum of many terms (D49), and on K's windows it
  cancelled well: K holds, and its margin of three thousand was one
  favourable measurement. The same holds for `int4`, 1.02·10⁻³ here against
  8.32·10⁻⁵ there.
- **`bf16` on a loss entry.** Most of its 2.96·10⁻³ is the output's own
  rounding: the exact loss is 2.2·10⁻³ from the nearest `bf16` number. A budget
  binds what the entry returns, and this entry returns one number stored in
  eight bits. With the metric in `f64`, the `bf16` model's loss is 6.6·10⁻⁵
  from `f32`'s.

## 6. Make it pay: the statements of 0.6.0

Version 0.6.0 named four falsifiable statements before it began
(`presk-specs/implementation/presk-plan-0.6.0.md` §3.6), in the form the two
previous versions used, and added a fifth, J′, once J's failure had shown
what J was really asking. Three hold and two fail.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| H | the vector kernel is **bit-identical** to the scalar blocked one, in `f32` and `f64` | every bit | every bit, on both executors, integer weights included | holds |
| I | a 1024³ `f32` matmul at 3× the scalar figure of 0.4.0 (47–56 GFLOP/s) | ≥ 3× | lanes **1.15–1.25×** over scalar (before D46); on falicon after it 41–46 GFLOP/s, below 1× | fails, retired |
| J | the solved realization at a 10⁻³ budget faster per training step than all-`f64`, on the corpus GPT | ≥ 1.3× | **1.06×** at `D = 64`, **1.13×** at course scale | fails |
| J′ | the same, with the budget binding the training step's **update** and the solver free over the whole step, at course scale | ≥ 1.3×, inside 10⁻³ | **1.99×**, 2.02× re-measured; 2.2·10⁻⁴ | holds |
| K | `int8` weights, per channel, inside a 10⁻² budget on the corpus GPT's loss, with trained weights | ≤ 10⁻² | **3.23·10⁻⁶**; 1.46·10⁻⁴ on held-out windows | holds |

**H holds, and it is the claim the design supports.** 0.5.0's G asked for a
vector kernel's measured gap to fall inside its declared order's bound. D42
showed that lanes across output columns do not reassociate anything, so the
claim became an equality rather than a tolerance (D43). Tests assert it at
two and four lanes in `f32` and `f64`, on the kernel interpreter and in the
emitted code, and the integer-weight kernels are asserted bit-identical to
the dequantised path the same way (D45, D47). Since lanes change no bit,
they are on by default. What they buy depends on the kernel and the batch:
on BERT-large they make `int8` weights 1.5× faster, and scalar `f32` code is
15 % faster than `f32` in lanes at batch 20.

**I fails, and it is retired rather than pursued.** On the development
machine the lanes gave 1.25× in `f64` and 1.15× in `f32`, measured before
D46 took the conversion of `bench`'s inputs out of the timing. On falicon
after it, lanes give 1.1–1.2× in `f64` and nothing measurable in `f32`, and
`bf16` is the fastest row: at 1024³ the kernel waits for memory (`native`
chapter). The blocked form already ran eight independent accumulator
chains, so the parallelism lanes add was mostly there. The factor I asked
for is an instruction set's — the matrix coprocessor PyTorch reaches through
the platform BLAS (§3) — which `docs/kernels-roadmap.md` puts after this
version, behind the realization contract, since it reassociates.

**J fails, and J′ says why.** Solved at `D = 64` with the time objective,
the realization was 1.06× at `D = 64` (with D46's dilution, and inside noise
several per cent wide) and 1.13× at course scale, where all-`f32` was 1.87×.
The budget constrained the forward loss, so the solver had no licence over
the reverse sweep and the optimizer, which are most of a step (D50, question
60). J′ gives the budget the step itself: one optimizer step's parameter
update against the exact step from the same state (D52).

**J′ holds.** The step solved at 10⁻³ is `f32` in every function. Per step at
course scale, from the difference of a one- and a four-step run: all-`f64`
59.7 s, solved 30.0 s, all-`f32` 31.0 s, the mixed-precision recipe 34.6 s
(D52), and 61.1, 30.3, 30.8 and 35.2 s re-measured at the end of the
version: **1.99×**, then 2.02×. It is inside its budget by measurement, 2.2·10⁻⁴
against the exact step at `D = 64` and 6.7·10⁻⁵ against a native `f64` step at
course scale. Two limits, stated in D53. The solved step is level with
all-`f32`: the budget licenses leaving `f64` and the time objective refuses
`bf16`, which the profile predicts slower, so J′ is faster than `f64` and not
than FP32, MLPerf's reference. And at course scale the discharge is against
`f64`, because the exact step is not affordable there (question 62).

**K holds, and its margin was one favourable measurement.** On `make
gpt-trained`'s weights and two training windows, every matrix in `int8` and
`f32` elsewhere is 3.23·10⁻⁶ from the exact loss, three thousand times inside
10⁻². On D50's checkpoint and two held-out windows it is 1.46·10⁻⁴, seventy
times inside (§5): the error of a quantised loss is a signed sum of many
terms (D49), and on K's windows it cancelled well. K is a claim about a
loss, a scalar averaged over 512 positions. On BERT-large's logits, at its
initialization, the same scheme is 5.7·10⁻², outside 10⁻² (`coverage`
chapter). There `int8` weights are also 2.2× faster than `f32`, the first
type narrower than `f32` to buy time on a model, because that kernel reads
the whole weight matrix again for every row of the result, and 4 MB of
`int8` stay in cache where 16 MB of `f32` do not.

**What the five together say.** The version's purpose was to make a
narrower type buy time inside a declared budget. It does, twice, each time
for a reason the plan did not expect. Not through vector lanes, which are
exact and free and buy little. Through the budget's scope: once the budget
binds the whole training step, the solver may leave `f64` wherever the
measurement allows, and a step runs 1.99× faster. And through memory: `int8`
weights run 2.2× faster on BERT-large, where the kernel streams its weights.
Against FP32 — MLPerf's reference, and every practitioner's — neither is yet
both inside a tight budget and faster: the solved step is level with `f32`,
and `int8` on BERT-large's logits is inside 10⁻¹, not 10⁻². The part of the
speed question that is a compiler question at the kernel level has an
answer — lanes are exact and buy little — and what is left is an instruction
set.

## 7. Ranking, not speed: the statements of 0.7.0

The research programme's central hypothesis, RA4, is that on regular tensor
kernels a **static cost model** orders more than 90 % of the pairs of
variants that autotuning distinguishes, at a decision cost lower by orders
of magnitude. No version had tested it: until 0.7.0 there was no space of
variants to rank. 0.7.0 built one — a layout space per loop nest (register
tile, k-block, loop order, lanes, pack, a thread split at a depth, softmax
passes), every point exact by construction (`realize` and `native`
chapters) — and a model that reads a layout, calibrated on a GEMM the test
never enters (`optimization` chapter). Five statements were fixed before the
version began (`presk-specs/implementation/presk-plan-0.7.0.md` §3.7). Two
hold and three fail.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| L | every point of every nest's space **bit-identical** to the scalar sequential kernel, natively and on the kernel interpreter | every bit | 17 879 of 17 879 points of day one checked on the interpreter; no output differing in any sweep, on either executor | holds |
| M | on each attention `ein` at `B = 16`, `f32`, the static model orders the distinguished close pairs right | > 90 % | scores **47.9 %** of 12 925 pairs, values **43.9 %** of 14 868 | fails |
| N | choosing statically costs less wall time than the exhaustive sweep of the same points | ≥ 10³× | **582–717×** at `B = 16`, **75–89×** at `B = 1` | fails |
| O | the model's pick within reach of the best measured point, every contraction of the entry at `B = 16`, `f32` | ≤ 1.25× | `Linear` **2.87**, scores **2.18**, values **3.01** | fails |
| P | the best measured point of the GEMM space at 1024³ `f32` against 0.6.0's point | ≥ 3× | **5.49×**, 460 GFLOP/s against 84 | holds |

**How it was measured** (D62; `docs/ra4-2026-09-24.md` is the report, written
by `presk rank` and not by hand). One night on falicon, 23–24 September 2026.
`make ra4` swept the course GPT's `SelfAttention.apply` at `D = 384`, `T = 256`,
`B` = 1 and 16, in `f32` and `f64` — every point of every nest's space, 3 305
points a type at `B = 1` and 4 857 at `B = 16` — then the GEMM at 1024³ in `f32`
(1 333 points) and a BERT-large-shaped attention on a sample of 41 points per
contraction, pinned to the ten Cortex-X925 cores, each point timed seven
times on the entry's own captured inputs. `make ra4-20` swept the attention
once more on all twenty cores with its own calibration; `make ra4-check` ran
every point of day one again on the kernel interpreter, in 130 shards (about
45 CPU-hours); and `make ra4-day2` repeated day one four hours later, in the
same night rather than on another day as the plan asks. The two runs agree
within 0.1 % in each configuration's sweep time, and 137 307 of their 160 509
close pairs — two points of a nest differing in one decision — are
*distinguished*: their median times differ by more than twice the larger
median absolute deviation on both runs, with the same sign (question 65).

**L holds.** No point of any sweep computed a different bit from the scalar
sequential kernel, natively or on the kernel interpreter: every register
tile, k-block, loop order, lane width, pack and thread split of the space.
Exactness by construction survives a layout space, which is what makes the
other statements about time alone.

**M fails, under any reading.** A pair the model predicts equal is *tied* and
counted as not right (question 70), and 37.4 % and 46.1 % of the two `ein`s'
distinguished pairs are ties. Leaving the ties out, the model orders 76.5 %
and 81.4 % of the rest right; on the pairs it *decides* (a predicted ratio
outside `1 ± u`), 79.0 % and 82.1 %. No contraction nest reaches 90 % on any
reading, at any shape or type; the closest is 88.1 %, on the decided pairs
of the `f64` values `ein` and of the `f64` `Linear` dot. By decision, over
the whole measurement:

| Decision | Distinguished pairs | Right | Tied |
|---|---|---|---|
| thread split | 39 144 | 96.7 % | 1.3 % |
| lanes | 7 289 | 93.2 % | 6.8 % |
| softmax passes | 27 | 100 % | 0 % |
| register tile | 69 136 | 27.7 % | 46.7 % |
| loop order | 17 565 | 30.0 % | 61.8 % |
| k-block | 2 998 | 0.6 % | 97.0 % |
| pack | 1 148 | 57.8 % | 0 % |

The model is right on the decisions its terms read — a rate per lane width
and per worker count, the cost of starting workers — and blind to the rest.
Every point of both `ein`s at `B = 16` in `f32` is predicted work-bound: a load
weighs nothing in its work count, and its traffic is counted from L2, where
every register tile of a form has the same footprint. So the values `ein`'s
tiles `[1, 4]`, `[1, 16]` and `[2, 8]` are all predicted 2.27 ms and measure
7.34, 3.39 and 4.66 ms. What a register tile saves is loads from L1 into
registers per multiply-add, and no term counts them (`optimization` chapter,
"What 0.7.0's layout model could not see"). That is the first risk the plan named,
at the one level the model does not have.

**N fails, and grows with the kernel.** The static side is `presk plan`'s:
the entry run once to capture its launches, then every point lowered and
costed, about 0.2 ms a point, 1.5 s for the whole entry at `B = 16`. The
exhaustive side is `presk autotune` without the interpreter's check: eight
launches a point of a few milliseconds each, 1 048 s at `B = 16` in `f32`.
The ratio is 582–717 at `B = 16` and 75–89 at `B = 1`, where the kernels are
short. On the GEMM at 1024³, whose kernels are long, it is 1 460–1 538.

**O fails.** The model's pick is `presk plan`'s: its best prediction, a tie
broken by the space's order. On the attention's three contractions at `B = 16`
in `f32` it is 2.87, 2.18 and 3.01 times slower than the best point measured.
On the GEMM, 21 points tie at its best prediction; the first, the tile
`[1, 4]`, runs in 28.6 ms, slower than 0.6.0's own point.

**P holds.** The best measured point of the GEMM space — the tile `[4, 16]`
in four lanes, order `JI`, split over ten workers at depth 1 — runs 1024³ in
4.67 ms, 460 GFLOP/s, against 84 GFLOP/s for 0.6.0's point in the same sweep:
5.49×. It is 0.6.0's retired statement I re-posed, its factor of three
found in the layout rather than in the lanes, and it is within 1.6× of the 710
GFLOP/s PyTorch reaches on twenty cores (§3). P is a by-product: it is about
the best point, which measuring found, not about the model's pick.

**Elsewhere, the same picture.** On all twenty cores: M 48.4 % and 45.8 %, O
4.13, 2.31 and 2.80, N 91–762×, the split right on 93.6 % of its pairs —
the unlike cores did not make the thread term unlearnable
(`docs/ra4-20cores-2026-09-24.md`). The BERT-shaped sample gives 13–17
distinguished close pairs per contraction, too few to rank, and top-1 regrets
of 1.90–4.96.

**What does hold.** The model is a good **pruner**, which RA4 does not claim:
on every contraction nest at the attention's shapes, the points it predicts
within `1 + u` of its best contain the best measured point or one within
1.08× of it; at `B = 16` in `f32` and on the GEMM, the best itself, among
36–201 candidates of 1 333–1 717 points. Measuring its candidates alone
finds the best point at 2.4–15 % of the sweep's cost. And **the space pays**:
its best point beats 0.6.0's by 9.2× and 16.5× on the two `ein`s at `B = 16`,
by 47× and 77× at `B = 1`, where 0.6.0 ran them on one worker with the tile
`(1, 1)`, by 2.6× on the `Linear` dot and by 5.5× on the GEMM.

**Two corrections made before the statements were read.** The first reading
of the two runs counted a tie as right, since Rust's `0.0.signum()` is `1.0`,
and took the first *key* among tied predictions as the pick. It gave M 73.9 %
and 74.2 % and O 1.73, 1.25 and 1.41. A model that predicts two variants
equal has not ordered them, and the pick is the one `presk plan` makes, so
both were corrected and the report regenerated from the same measurements;
every verdict is the same under both readings. Counting a tie as half right,
as concordance indices do, would give M about 66 % and 67 %, still a failure.

**What the five together say.** Exactness costs nothing here: every one of
the 17 879 points measured is bit-identical, and the best point of each
contraction is 2.6 to 77 times faster than what 0.6.0 emitted. The static model does not choose
among them as RA4 says, and the reason is named rather than guessed: it has
no term for the loads a register tile saves. The model ranks what it can see,
and it narrows the search to a few per cent of the space that holds the best
point. Whether a load term makes it rank is not something these measurements
can answer, since a model designed after seeing where this one failed would
be fitted to its test; the term is recorded in `docs/kernels-roadmap.md`, for
a later version and a fresh sweep. 0.7.0 closes on these results, and 0.8.0
took them up (§8).

## 8. Ranking, re-tested: the statements of 0.8.0

0.7.0's model failed RA4 for a named reason: it weighed no load from the
first-level cache into registers, and under a maximum its traffic terms
never bound, so it tied register tiles, loop orders and k-blocks (§7).
0.8.0 changed the whole model (`optimization` chapter): a term for one
worker's register loads and stores, `N_ld/ρ_g`; a p-norm of the five terms
in place of the maximum, its exponent fitted (`q` = 1.06); a calibration
widened to five tiles, sizes to 2048 and two thin products; and two rules
of the planned cost, footprints in whole cache lines and one operation of
work per loop iteration. It was designed on 0.7.0's measurements alone,
which became the **development set** and decide nothing, and **frozen**
at the tag `ra4-model-0.8.0` on 2026-09-25 before any new measurement.
`presk rank --frozen` refuses a day measured on a commit that does not
descend from it. Then two fresh sweeps tested it. Six statements were fixed
before the version began (`presk-specs/implementation/presk-plan-0.8.0.md`
§3.6): L, M, N and O are 0.7.0's, with the same thresholds; Q is new; D
reports the development figures. Three hold and two fail.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| L | every point of both fresh sweeps **bit-identical** to the scalar sequential kernel, natively and on the kernel interpreter | every bit | 9 286 of 9 286 points of F1's day one and 16 368 of 16 368 of F2's checked on the interpreter; no output differing on any day or shard | holds |
| M | on each attention `ein` of F1a, F1b and F2 at `B = 16`, `f32`, the frozen model orders the distinguished close pairs right | > 90 % | F2 **92.7 %** and **91.7 %**; F1a **87.4 %** and **88.5 %**; F1b **85.4 %** and **88.8 %** | fails |
| N | choosing statically costs less wall time than the exhaustive sweep, each fresh configuration, each day | ≥ 10³× | F1a 1 013–1 016×, F1b 2 880–2 933×, F2 at `B = 16` 2 168–2 734×, F2 at `B = 1` **218–252×** | fails |
| O | the model's pick within reach of the best measured point, every contraction nest of the configurations M names | ≤ 1.25× | at most **1.05** | holds |
| Q | measuring only the points predicted within `1 + u` of the model's best finds a point within reach of the best, at a fraction of the sweep's time, every contraction nest of every fresh configuration | ≤ 1.05×, ≤ 10 % | 18 nests: the best candidate within **1.03×**, at most **1.8 %** of the sweep | holds |
| D | the same, on the development set under the frozen model | reported | M 89.3 % and 89.6 %; O 1.07, 1.00, 1.01; Q on 12 of 13 nests | — |

**How it was measured** (D72–D74; the reports are
`docs/ra4-f2-2026-09-25.md`, `docs/ra4-f1a-2026-09-25.md` and
`docs/ra4-f1b-2026-09-25.md`, written by `presk rank`).
- **F2**, 0.7.0's shapes on the other cores: the course GPT's
  `SelfAttention.apply` at `D = 384`, `H = 6`, `T = 256`, `B` = 1 and 16,
  `f32` and `f64`, every point of every space, 16 324 points a day, pinned
  to the ten Cortex-A725 cores. Those cores have a quarter of the X925's L2
  and their own calibration, made in the frozen form and space after the
  freeze: `u` = 0.168, where the X925's is 0.132. It tests whether the model
  carries over to another microarchitecture.
- **F1**, new shapes on the ten X925 cores, priced by the frozen profile
  itself: BERT-large's attention (F1a: `D = 1024`, `H = 16`, `T = 384`,
  `B = 1`), which 0.7.0 only sampled, and a head dimension of 128, which the
  development set never had (F1b: `D = 1024`, `H = 8`, `T = 512`, `B = 2`),
  both in `f32`, every point, 9 264 points a day. It tests whether the model
  carries over to other shapes.
- **Twice each**, on 2026-09-25: F2 from 12:07 and from 16:45, F1 from 20:02
  and from 22:35, the other cores idle. The plan asked for two dates, and
  the user decided on one, hours apart (D72): a date is only a proxy for
  independent conditions, and a run at midnight would have been little more
  independent. The days' sweep times agree within 1.3 %.
- **Before each**, a dry run of the longest nest sized the day (1 h 41 min
  estimated for F2, 1 h 45 min measured; 52 min and 48 min for F1), so both
  were swept whole and no M rests on a sample.
- **Statement L's check** ran apart on the kernel interpreter: 88 shards for
  F2 (1 h 58 min) and 100 for F1 (1 h 44 min), on every core.
- **Every day, check and calibration is archived** read-only with its
  checksums, since none can be taken again, and ranked in step 7 from
  copies checked against those checksums.

**L holds.** No point of either sweep, 25 588 a day and 0.6.0's beside
them, computed a different bit from the scalar sequential kernel, natively
or on the kernel interpreter, on the A725 cores as on the X925.

**M fails: the model carried over to another core, not to new shapes.** On
F2 it ranks 92.7 % and 91.7 % of the two `ein`s' distinguished pairs right,
above RA4's 90 % and above its own development figure (89.3 % and 89.6 %),
where 0.7.0's model ranked 47.9 % and 43.9 % of the same shapes' pairs. On
F1 it ranks 85.4–88.8 %, 0.8–3.9 points below the development figure. More
than 90 % would have taken 200 and 134 more right pairs at F1a, 1 064 and
183 at F1b. By decision, over every nest:

| Decision | F1 | F2 | development set | 0.7.0's model (§7) |
|---|---|---|---|---|
| register tile | 90.7 % | 92.4 % | 89.1 % | 27.7 % |
| thread split | 99.1 % | 98.1 % | 97.7 % | 96.7 % |
| lanes | 100 % | 98.7 % | 100 % | 93.2 % |
| loop order | 60.6 % | 72.6 % | 67.0 % | 30.0 % |
| k-block | 64.3 % | 53.4 % | 65.9 % | 0.6 % |
| pack | 47.9 % | 62.9 % | 57.8 % | 57.8 % |
| every pair | 87.6 % of 76 830 | 90.4 % of 134 022 | 88.5 % of 137 307 | 50.8 % |

Ties are gone: 0.1 % of the pairs, where 0.7.0's model tied 34 %. The
pairs the model decides, predicted apart by more than `u`, are 99.2 % right
on F1 and 98.8 % on F2. **What M is decided by** is how the model orders
the pairs it predicts within `1 + u` of each other: 97–98 % of the misses on
F1, and 99 % on F2, are such pairs. It orders 69–73 % of them right on F1
and 78–81 % on F2, and they are 39–46 % of F1's distinguished pairs. They
are tiles (about 10 % wrong, on the largest counts), loop orders (47–71 %
right in M's nests), the pack and k-blocks, whose pairs almost all lie
within `u`.

**One decided failure, which names a term.** At F1b's scores `ein` the model
decides 121 loop-order pairs and gets 67 wrong. All 67 are one-row tiles,
`[1, 8]` and `[1, 16]`, packed, in four lanes; every decided order pair of
a two- or four-row tile is right. The order with `I` outside `J` reads the
whole packed operand once a row of the output, where the other keeps a
panel of it. The model predicts it 15–23 % slower, for its L2 term, and it
measures 1–16 % faster. With `q` = 1.06 the p-norm is nearly a sum, which
adds the L2 time to the load time, and a one-row kernel does not pay it on
top of its loads. The packed operand of one head is 256 KiB at F1b and
96 KiB at F1a, against 64 KiB of L1. At 0.7.0's shapes it is 64 KiB, and on
F2 and the development set every decided order pair at M's configuration is
right.

**N fails at batch one on the A725 cores**, as the plan foresaw. The static
side is now a capture of the entry's shapes, which runs no kernel (D65),
and the costing of every point: 0.45–0.85 s on the X925's configurations,
0.96–1.55 s on the A725's at `B = 16`. At `B = 1` on the A725 a sweep takes 143–170 s and
the static side 0.57–0.78 s, of which the capture is 0.03–0.05 s: costing
3 305 points at 0.16–0.22 ms each is the rest. Clearing 10³ there would take
costing four to five times faster. At F1a, `B = 1` on the X925, the ratio
clears 10³ by 1–2 %. With the capture that runs the entry, as 0.7.0 counted
it, the ratios would be 161–173× on F2 at `B = 1`.

**O holds.** The model's pick, `presk plan`'s, is within 1.00–1.03× of the
best point measured on F1a's three contractions, 1.00–1.05× on F1b's and
1.00–1.01× on F2's at `B = 16` in `f32`, where 0.7.0's was 2.18–3.01× off.
Elsewhere the worst top-1 regret on a contraction nest is 1.11, on F2's
values `ein` in `f64`. The space pays on the new shapes as on the old: the
best point beats 0.6.0's kernel by 27–132× on F1's three contractions, and
by 3.2–13.7× on F2's at `B = 16` in `f32`.

**Q holds.** Q is not RA4's claim. It tests the chooser 0.7.0's data
suggested: prune statically, then measure the candidates, the points
predicted within `1 + u` of the model's best. Its cost is question 72's:
the whole configuration's static side, charged to each nest, plus the
candidates' measured time, over the nest's exhaustive sweep, on each day
(D74).

| configuration | candidates | best candidate / best | cost (question 72) | candidates' share of the timed launches |
|---|---|---|---|---|
| F1a, `B = 1`, `T = 384` | 12–18 | 1.00 | 0.3–0.6 % | 0.10–0.16 % |
| F1b, `B = 2`, `T = 512` | 11–13 | 1.00 | 0.1–0.2 % | 0.05–0.06 % |
| F2, `B = 1`, `f32` | 5–15 | 1.00 | 1.2–1.5 % | 0.05–0.16 % |
| F2, `B = 1`, `f64` | 13–19 | 1.00–1.01 | 1.5–1.8 % | 0.21–0.31 % |
| F2, `B = 16`, `f32` | 13–19 | 1.00 | 0.2 % | 0.10–0.12 % |
| F2, `B = 16`, `f64` | 19–28 | 1.00–1.03 | 0.4–0.6 % | 0.20–0.44 % |

Where the sweep is short, the static side is most of the cost: the dearest
nest, the scores `ein` at `B = 1` in `f64` on the A725, spends 0.77 s
choosing and 0.15 s measuring its candidates, against 50.4 s swept.

**D, the development figures, decided nothing.** They foresaw F1's M within
four points and understated F2's. They showed Q's one miss, the scores
`ein` at `B = 1` in `f32` at 1.08×, which the fresh sweeps do not repeat.
The development report under the frozen profile is
`docs/ra4-dev-model-0.8.0.md`, which `make ra4-dev-frozen` regenerates byte
for byte.

**What the six together say.** The load term and the combination did what
0.7.0's failure asked of them: the model now reads register tiles, ties
almost nothing, and is right on nearly every pair it decides, on a
machine it was not designed on. RA4's 90 % is met on the other cores and
missed on new shapes, by the pairs the model predicts within its own
uncertainty. Its pick is within 1.05× of the best, and measuring its
candidates finds the best at under 2 % of a sweep's cost. The static choice
is cheap enough at batch 16, and costing a point is what keeps it from 10³ at
batch one. What the failures point to (the L2 term beside the loads of a
one-row tile, the order of pairs within `u`, the cost of costing a point) is
recorded in `docs/kernels-roadmap.md`, for a model to be frozen and tested
again on a sweep it was not designed on.

## 9. A C program: the statements of 0.9.0

0.9.0 is not a test of a hypothesis of the research programme. It gives an
entry a way out of the compiler, as a C program that runs without it
(`export_c` chapter), and its five statements, fixed before the version
began (`presk-specs/implementation/presk-plan-0.9.0.md` §3.8), say what the
program owes: the native backend's bits, the exact reference's band, a
clean build on the compilers that matter, and `presk serve`'s tokens; its
speed is reported and decides nothing. The test set is the eleven entries
`make plan-check` compares (an MLP's loss and its gradient, a LayerNorm
and an RMSNorm, MiniGPT's forward pass, its training loss, that loss's
gradient through attention, an optimizer step and `next_token`, and the
course GPT's attention and forward pass, all at small constants) and the
course GPT at its course constants. Four hold; the fifth
decides nothing.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| C1 | every exported entry of the test set, in `f64`, `f32` and `bf16`, under the default layout and under a plan from `presk plan`, computes outputs **bit-identical** to `presk run --backend native` with the same realization, plan, dimensions, seed and weights | every bit | every entry in the three types, on synthesized parameters and with their zeros drawn, under a mixed realization, under `presk plan`'s plans, both hosts, with OpenMP and without, on one thread and every thread; the course GPT at `T = 256` with its trained checkpoint, at `B = 16` under its plans (201 of 228 launches divided among ten threads); 42 programs with integer weights; every nest in the SCoP form: no bit differs | holds |
| C2 | the same outputs lie within the realization's conformance band of the exact reference, as the native backend's do | no output outside the band where the native backend's is inside | the outputs are the native backend's, so outside the band on exactly its elements: 34, 121 and 99 of `loss_grad`'s 8 768 in `f64`, `f32` and `bf16`, and a few cancelling logits of the forward passes (question 75) | holds |
| C3 | the exported C compiles without warnings (`-Wall -Wextra -std=c11`) with GCC and Clang on aarch64, and cross-compiles for x86-64 | no warning or error | GCC 13.3 with and without OpenMP; Clang 21.1 (`zig cc`), linked, run and bit-identical, `generate` giving `presk serve`'s tokens; x86-64 (`x86_64-linux-musl`), linked; the SCoP mode's files on Clang 23.1.2 too: none warns | holds |
| C4 | the exported program's wall time against the native backend's, one thread and every thread, on the course GPT's forward at `B = 16` | reported | every thread: native 1 094 ms, C 290 ms; one thread: 3 132 and 1 700 ms (`f32`, `T = 256`, under the frozen profile's plan) | — |
| C5 | the tokens the exported `next_token` generates over a sliding window are those `presk serve` generates, greedy and sampled with the same seed | every token | `generate`'s `--json` document is `presk serve --backend native`'s byte for byte, greedy and at three temperatures and four seeds, in the three types, through both hosts, on the example with its zeros drawn and on the trained checkpoint, with integer weights too | holds |

**How it was measured** (D77–D85). `make test` exports, builds and runs
every program through the binary and compares its output file with the
native backend's with `cmp`; `make c-conformance` adds the course GPT, the
trained checkpoints, the plans, Clang and x86-64 through zig from PyPI, and
Polly from LLVM's release; `make c-speed` measures C4. Each check was
checked by mutations of the program that each change its outputs: a
dropped launch, a slice off by a layer, a loop bound off by one, an early
release, a gap between threads' pieces, a skipped draw, a scale one channel
off, and others (`export_c` chapter, "What is checked").

**What makes C1 hold.** Two things the plan named, and one it did not
know. The program is built without floating-point contraction (`-std=c11
-ffp-contract=off`; GCC contracts in its default GNU mode, and 14 of 246
primitive runs then move). It ships the transcendental functions the
native backend calls through the `libm` crate, which ports musl as it was
before 1.1.23: current musl's `exp` and `log` differ from it on 8–10 % of
inputs, glibc's on 1.2–9.8 %, and musl 1.1.22's on none of 10⁶ inputs a
function. And the native backend's own `sample` called the platform's
`exp`, not the crate's, which step 6 changed: 540 generations of 32 tokens
did not move.

**C2 became a check of a copy.** The band is per element, and the native
backend's outputs cross it where large terms cancel, in gradients and in
forward passes, where the kernel interpreter, another conforming
executor, crosses it on other elements. So C2 was judged against the
native backend's own gap from step 0 on, and question 75 asks how
conformance should judge an output that cancels.

**C4, reported.** The C program is 3.8 times faster on ten threads and 1.8
times on one. On one thread both run the same kernels, so the difference
is the code generator (GCC `-O2` against Cranelift) and the host (the
native host copies each stacked layer's slice, the program slices by
pointer); on ten, the native backend also starts a thread per piece at
each divided launch, where OpenMP keeps a pool. The shares are not
measured apart.

**What the five together say.** A Presk entry, a gradient or a sampling
loop included, can leave the compiler as a small C program that computes
the native backend's bits on any compiler and thread count, so the exact
reference's measurements of the native backend (its gap, a budget's
verdict) carry over to it unchanged. The polyhedral comparison the SCoP
mode prepares, a tool's transformation of the same nests against RA4's
ranking, comes after 0.9.0.

## What these experiments do not show

- **Speed against a mature stack.** §3's comparison, redone on falicon on
  23 September 2026, is 22× a training step, 14 of it the contraction.
  0.7.0 and 0.8.0 did not redo it: the best GEMM point is within 1.6× of
  PyTorch's product (§7), and 0.8.0's model picks a point within 1.05× of
  the best on attention's contractions (§8), but a training step's largest
  contractions, the derivatives of `dot`, are not in the layout space yet
  (`realize` chapter).
- **A static choice as good as measuring.** RA4, the programme's first
  thesis, was tested twice on the attention of one model family, on one
  machine's two kinds of core. 0.7.0's model failed it for a named cause
  (§7). 0.8.0's, frozen before its test, ranks more than 90 % of the
  attention `ein`s' pairs on the other cores and 85–89 % on new shapes, so
  RA4 still fails as stated. Its misses are pairs it predicts within its
  own uncertainty, and its pick and its candidates are within a few per
  cent of the best (§8). Kernels other than attention's and a GEMM,
  machines other than falicon, and the comparison with the MDH and ATF
  autotuners on the same space have not been tried.
- **A budget discharged by a bound.** Every budget here is discharged by
  measurement, on the envelope's inputs (D41, D51); the conditioned bound is
  still ten orders of magnitude above the gap on the trained corpus GPT in
  `f32`. And the reference measured against is the exact denotation only
  where the interpreter reaches: BERT-large's inference, not a course-scale
  training step, where J′ was measured against `f64` (question 62).
- **Quality beyond this scale.** The ε curve's quality axis is flat after
  600 steps of a `D = 64` GPT, within 1.2·10⁻³ against a seed spread of 0.13
  (D50). There is no trained BERT, no SQuAD and no F1 here, so MLPerf's own
  threshold on its own benchmark was not applied; §5 applies the rule to the
  corpus GPT instead.
- **Verified transformations** (the KV cache), **parallelism** beyond threads
  inside a kernel, and **energy**: declared in the language, not implemented.

What the tool delivers is the front half of the argument and a first piece
of the back half: a language where shapes, smoothness, randomness and data
are checked before anything runs; an exact reference that turns numerical
precision into a measurement; a declared budget that buys time — against
`f64`, not yet against `f32` inside a tight budget; and a space of kernel
layouts in which exactness costs nothing and the best points are several
times faster than the defaults. That a **static** cost model can choose among
them as well as measuring does, which is what would make the discipline buy
speed without autotuning, is the part still to show. The first test failed
for a reason it named; the second, on a model frozen before it, came within
a few points of the hypothesis's threshold and met it on one of its two
sweeps, and a chooser that prunes by the model and measures the rest found
a point within 1.03× of the best everywhere, at under 2 % of the cost.
