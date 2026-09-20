# Experiments: exact versus almost

Two experiments that the tool can run today, with what they show and what
they do not. `make experiment` runs both (after `make venv`); the script is
`experiments/exact_vs_almost.py` and takes any output directory of
`presk export … --conformance-entry train:K`.

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

## What these experiments do not show

- **Speed, beyond the first step.** Since 0.3.0 the same program trains
  natively (twenty MiniGPT steps in 0.4 s against half a minute on the
  reference; `native` chapter), but the kernels are naive loop nests, far
  behind PyTorch's BLAS on real shapes. The claim that exact semantics plus
  e-graph rewriting and static cost models yield fast kernels is the
  research program of the rest of 0.3.0 and after, untested so far.
- **Error budgets.** `precision = budget(rel_error = 1e-3)` is parsed and
  checked, not solved: the gap of a given dtype can be measured, dtypes
  cannot yet be chosen from a budget.
- **Verified transformations** (the KV cache), **parallelism**, **energy**:
  declared in the language, not implemented.

What the tool delivers today is the front half of the argument: a language
where shapes, smoothness, randomness and data are checked before anything
runs, and an exact reference that turns numerical precision into a
measurement. The back half, that this discipline buys speed rather than
costing it, is what the next version has to start showing.


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
bare pass.

**What the four together say.** The claim that precision can be *declared*
and *verified* is demonstrated: D holds by measurement. The claim that it
**pays** is not: E fails, and it fails for a reason this version measured
twice before believing it. That is the same shape as 0.4.0, where the
discipline was shown affordable and the speed was not there yet, and it puts
the same item at the top of the next version's list: make a narrower type buy
time, not only bytes.
