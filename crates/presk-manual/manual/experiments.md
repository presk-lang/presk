# Experiments: exact versus almost

What the tool can show about the approach, with numbers, and what it cannot.
§1 and §2 are the first two experiments: `make experiment` runs both (after
`make venv`), with the script `experiments/exact_vs_almost.py`, which takes
any output directory of `presk export … --conformance-entry train:K`. §3 is
0.4.0's decisive experiment against PyTorch (`make experiment-gpt`), §4 the
statements of 0.5.0, §5 MLPerf's acceptance rule set beside a budget on one
model (`make foil`), and §6 the statements of 0.6.0. Each version named its
statements before it began, with what would count as failure, and each
section reports them as measured.

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

One training step is about 266 GFLOP. PyTorch does it in 0.405 s, presk in
37.9 s, 33.3 s with the optimizer: 656 GFLOP/s against 7.0.

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

Statement C is the one that held, and it is the smallest of the three: asked
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

## What these experiments do not show

- **Speed against a mature stack, today.** The one comparison with PyTorch is
  §3's, made by 0.4.0 on the development machine: 94× a step, about
  twenty-six of it the matrix coprocessor. Vector lanes, the solved training
  step and integer weights have moved presk's own figures since, and none of
  them was measured against PyTorch again; the comparison is to be redone on
  this machine before the next version is planned. The research claim under
  it, that exact semantics with rewriting and **static** cost models buy
  speed (the research programme's RA4, the first thesis), has not been
  tested: the cost model ranks kernels (a rank correlation of 0.95 on the
  tiny MiniGPT) and prices realizations, and has never been asked to choose
  among schedule variants of one kernel against autotuning.
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
precision into a measurement; and a declared budget that buys time — against
`f64`, not yet against `f32` inside a tight budget. That this discipline buys
speed through static cost models, rather than costing it, is the part still
to show.
