# Precision as a resource

This chapter is being written as version 0.5.0 is built
(`presk-specs/implementation/presk-plan-0.5.0.md`). Its subject is the third
axis of the research programme: given a declared error budget, which machine
types and which reduction orders should the backends use, and how far from
the exact denotation does that put the execution?

Two pieces exist so far. The **realization** — what the backends are allowed
to do — is in the `native` chapter, because that is where it is spent. The
**magnitude envelope**, below, is what makes a bound computable at all.

## What a bound will and will not promise

The specification is blunt about the objection that has to be met: neural
networks tolerate enormous numerical error, four-bit quantisation works, and
no analytical function relates a per-operation bound to a model's perplexity.
A budget that claimed to guarantee the quality of a model would be useless.

What it does instead is split the total degradation in two. The quality of
the executed model is the quality of the model *over ℝ*, which is an
empirical machine-learning question, plus the effect of the execution being
within ε of that denotation, which is a compiler question. Presk guarantees
the second term and only that one. Three things change in practice: a
degradation beyond ε becomes a compiler bug rather than a mystery, the
ε-to-quality curve becomes measurable because it is one program with one
parameter varying, and precision becomes a resource that can be *allocated*
rather than assigned by folklore.

## The magnitude envelope

A first-order error analysis needs, for every value, a bound on `|v|` as well
as on its distance to the exact value. An absolute error means nothing
without a scale, and a relative one is unbounded wherever a result passes
near zero — which is not a subtlety but the first thing the conformance suite
ran into when `bf16` arrived (`native` chapter).

Static interval arithmetic cannot supply that scale here. Weights are
unbounded and intervals grow with depth until they say nothing. So the
magnitudes are **measured**, on the exact interpreter:

```sh
presk calibrate --precision corpus/nlp/05_gpt.psk --entry loss \
    --const D=32 --const H=2 --const L=2 --dim T=256 \
    --data corpus/nlp/data/civil_T256.psk.d --bits 128 --samples 8 \
    --out out/envelope_a.json
```

```text
presk calibrate: 101 sites over 8 samples of `loss` at 128 bits
  LayerNorm.apply/%4           2.5127e1
  SelfAttention.apply/%13      2.1522e1
  SelfAttention.apply/%15      1.6000e1
```

The file records the magnitudes per **site** — `"<function>/%<let>"`, the
same key a realization assigns a type to — together with the program, the
entry, the sample count, the precision and the day. Every bound computed from
it is conditional on it and is printed with it (decision D35). That is a
weaker claim than a bound usually implies, and saying so in the tool's own
output is the answer rather than a footnote: **presk bounds the distance
between an execution and the denotation over an observed input domain.**

## Is the envelope a property of the program, or of the sample?

The whole analysis rests on the answer, so it is measured rather than
assumed. `make precision` calibrates three disjoint slices of the civil code
and compares them:

| Slice | Against the first | Worst ratio | Where |
|---|---|---|---|
| samples 8–15 | 0–7 | **1.241×** | `LayerNorm.apply/%2` |
| samples 64–71 | 0–7 | **1.128×** | `LayerNorm.apply/%5` |

Of the 101 sites, 77 are compared, 24 are identically zero in both — the
zero-initialized head and what is downstream of it — and **none appears in
only one**, which is what a program with no data-dependent control flow
should give.

So the magnitudes are a property of the program within about a quarter, not
of the sample that happened to be drawn. A bound quoting this envelope is
conditional on a factor of roughly 1.25, not on orders of magnitude, and the
worst disagreement sits where one would expect it: the input of a
normalization, which is the widest thing in a transformer block and the one
most exposed to an unusual batch.

Two honest limits. The measurement is at reduced constants (`D = 32`, two
heads, two blocks) because the exact interpreter is some two orders of
magnitude slower than the native backend and a course-scale reference run is
not affordable; the claim is about the *shape* of the program, which does not
change with width. And the parameters are the synthesized ones of a fresh
initialization, so this says nothing yet about how the envelope moves during
training — which is a question step 8's sweep will have to face.

## From magnitudes to a bound

`presk check --precision` propagates the rounding error of the entry the
envelope was measured on, under a given realization:

```sh
presk check --precision out/envelope_a.json corpus/nlp/05_gpt.psk --realize f32 --sites 6
```

Each value carries a pair: a ceiling on `|v|`, read from the envelope, and a
ceiling on `|executed(v) − exact(v)|`, computed by first-order rules. The two
travel together because neither is usable alone.

Three rules carry most of the analysis. A **quotient propagates in relative
terms**, `d(a/b)/(a/b) = da/a − db/b`, which is why a normaliser costs
nothing — not as a special case for `LayerNorm`, but because that is the
quotient rule written in the right variable. A **residual adds** errors
rather than multiplying them, so depth costs a sum. And a **softmax is
1-Lipschitz in the ∞ norm**, so its error crosses with a constant and not
with the length of the row.

Where the reduction order enters is the point of the realization being an
object. A contraction of `k` terms adds `c(order, k) · u`, with `c = k − 1`
for a sequential or blocked sum and `c ≈ k/n + n` for `n` interleaved
partials. The declared order is an *argument* of the analysis, and a test
asserts what that means: blocking leaves the bound untouched, because it
changes no addition's order, while eight partials change it.

## Is the bound any use?

It is sound. On every conformance entry, in every realization, the propagated
bound exceeds the gap actually measured against the exact reference — which
is checkable exactly here, because the reference supplies the exact value,
and a test asserts it on 24 combinations.

Whether it is *tight* is another question, and the answer on a real model is
no:

| Program | Realization | Measured gap | Bound | Ratio |
|---|---|---|---|---|
| `RMSNorm{8}.apply` | f64 | 2.2e-16 | 6.0e-15 | 27 |
| `LayerNorm{8}.apply` | f32 | 1.4e-7 | 6.9e-6 | 48 |
| `mse` | f32 | 6.0e-8 | 1.0e-4 | 1.8e3 |
| `loss` (tiny MiniGPT) | f64 | 8.9e-16 | 7.7e4 | 8.7e19 |
| `loss` (corpus GPT) | f64 | — | 7.2e20 | — |

Over the 21 comparable cases the median ratio is 1.8·10³ and ten are within
10³. On a normaliser the bound is worth having. On a two-block transformer it
is not: a relative bound of 10²⁰ on a loss of 4.5 says nothing.

**The reason is located, not mysterious.** A contraction of `k` terms
multiplies its input error by `k`, which is worst-case correct — all `k`
roundings *could* align — while the errors actually committed grow like `√k`,
being independent. Four contractions in series therefore carry `k⁴` of
pessimism. What would close it is each sum's **condition number**,
`Σ|xᵢwᵢ| / |Σxᵢwᵢ|`, measured into the envelope beside the magnitudes, which
replaces the worst case with what the program does. That is named future
work, and it is what statement F of the plan will be reported against.

This is the specification's own risk register arriving on schedule: worst-case
bounds are pessimistic by several orders of magnitude with respect to the
degradation observed. It is why the version's primary deliverable is the
*measured* curve of step 8 and not the bound, and why the solver of step 6
cannot simply verify against this number.

## Where a budget should go

A bound is the wrong instrument for *allocating* precision: allocating by a
pessimistic bound spends the budget where the bound is loose rather than
where the error matters. The right instrument is the one this compiler
already has. For a scalar output `L`, the sensitivity of `L` to a
perturbation of a value `v` is `‖∂L/∂v‖₁`, and a reverse sweep computes
exactly that for **every** intermediate on the way to the parameters — then
discards all of them but the parameters'. So no second differentiation and no
perturbation per site is needed; only a record of where each adjoint ended
up.

```sh
presk dump --stage sensitivity --entry loss corpus/nlp/05_gpt.psk     --const D=64 --const H=2 --const L=2 --dim B=2 --dim T=32     --weights out/gpt_tiny/final.psk.d
```

**A ranking taken at initialization is vacuous, and finding that out is part
of the result.** The corpus GPT's output head is zero-initialized, as
Karpathy's is, so no gradient reaches anything upstream of it and every
module's sensitivity is exactly zero. `--weights` starts from trained
parameters instead. The same caveat applies to the magnitude envelope, and
for the same reason.

## Does it reproduce the folklore?

Mixed-precision practice says: keep the output head and the embedding table
wide, the down-projection can be narrow. Measured on the corpus GPT after
training, at `D = 64`, two heads, two blocks:

| Site | What it is | `‖∂L/∂v‖₁` |
|---|---|---|
| `GPT.apply/%23` | the output head's parameters | **5.86** |
| `GPT.apply/%17` | the block stack's parameters | 2.46 |
| `GPT.apply/%22` | the final normalization's output | 1.03 |
| `FeedForward.apply/%4` | the up-projection through `gelu` | 0.65 |
| `FeedForward.apply/%5` | the down-projection's parameters | 0.46 |
| `GPT.apply/%4` | **the token embedding table** | 0.28 |
| `GPT.apply/%6` | the position table | 0.22 |

Half of the folklore is reproduced and half is not, which is the more useful
outcome.

**The output head dominates**, by a factor of twelve over the block stack and
of twenty over anything else. Keeping it wide is not a convention; it is what
the derivative says.

**The embedding table does not.** It sits twentieth of the head, below the
down-projection that practice is happy to narrow. So the practice of keeping
embeddings in high precision is *not* explained by the sensitivity of the
loss to them, and if it is right it must be right for another reason — an
embedding is gathered, so a step touches only the rows a batch mentions, and
the arguments usually given for it are about dynamic range and sparse
optimizer state rather than about `∂L/∂v`. That is a claim this measurement
can make and the folklore cannot.

The two quantities must never be confused, and the code says so in as many
words: a bound guarantees, a sensitivity ranks. The solver of step 6 proposes
with the second and verifies with the first, which is what the specification
means by calling precision assignment an inverse Herbie.

## A budget in, a realization out

```sh
presk solve corpus/nlp/05_gpt.psk --precision out/envelope_a.json \
    --budget 1e-3 --weights out/gpt_tiny/final.psk.d --out out/solved.json
presk run corpus/nlp/05_gpt.psk --entry loss --realize @out/solved.json …
```

The solver chooses one machine type per site to minimise **traffic** subject
to an error constraint, which is a multiple-choice knapsack. It proposes
greedily, in order of bytes saved per unit of sensitivity-weighted error
added; it verifies; and while the verification fails it widens the site whose
estimated contribution is largest, one step of the menu at a time.

**It verifies by measurement, not by the bound** (D41). The bound is vacuous
on a transformer, so verifying against it would accept everything. Running
the candidate and measuring its distance to the exact reference is a stronger
check — the error actually committed rather than an envelope around it — and
it is a check only this compiler can make, because only this compiler keeps
an exact denotation to measure against.

The objective is traffic rather than time, and deliberately: `bf16` is
*slower* than `f32` on a compute-bound kernel (the `native` chapter has the
measurement), so minimising bytes and minimising seconds are different
problems. Pricing time per site needs the roofline's `W` beside its `Q`.

On the corpus GPT at `D = 64`, trained, budget 10⁻³:

| | |
|---|---|
| traffic | 1.29e6 bytes against 3.45e6 all-`f64`, **2.67×** |
| widening rounds | 0 — the proposal was accepted first time |
| measured gap | 1.05e-7 against a budget of 4.5e-3 |
| solved run, natively | 4.521788597, where `f64` gives 4.521788577 |

On a program small enough to search exhaustively, the greedy proposal is
**exactly optimal** at a 10⁻³ budget and 1.057× off at 10⁻⁶. That comparison
is against the optimum *of the same first-order estimate*, which is the only
fair one: the search never runs a measurement.

One rule in the solver is worth stating because getting it wrong was
instructive. **A site with no measured sensitivity is not an insensitive
site**; it is one the run did not reach, and reading the absence as zero
makes it look maximally safe to narrow. With that bug the solver narrowed
everything, measurement rejected it, and a hundred and three widening rounds
recovered a 1.06× saving. Reading the absence as *unknown* and leaving such
sites alone gives 2.67× accepted on the first try.

## The curve

`experiments/curve.sh` sweeps the budget on the corpus GPT. One program, one
parameter varying — which is the whole point. The specification's complaint
about the literature is that every quantisation level is a different pipeline
with different code and different bugs, so a sweep there compares artefacts
rather than precisions. Here the program is identical at every point.

Corpus GPT, `D = 64`, trained, the reference at 128 bits:

| budget | traffic | against all-`f64` | measured gap | inside |
|---|---|---|---|---|
| 1e-1 | 8.63e5 | 4.00× | 1.34e-3 | yes |
| 1e-2 | 8.87e5 | 3.89× | 6.94e-6 | yes |
| 1e-3 | 1.29e6 | 2.67× | 1.05e-7 | yes |
| 1e-4 | 1.52e6 | 2.26× | 8.23e-9 | yes |
| 1e-5 | 1.66e6 | 2.08× | 3.06e-8 | yes |
| 1e-6 | 1.68e6 | 2.06× | 3.06e-8 | yes |
| 1e-7 | 2.05e6 | 1.69× | 2.75e-10 | yes |

**Every point is inside its budget**, by measurement against the exact
reference, and the traffic rises monotonically as the budget tightens: four
times less at a tenth, still 1.7× less at ten to the minus seven. There is a
knee between 10⁻² and 10⁻³, where the cost of another five decimal digits is
a third of the saving.

Two things the table does not hide. The measured gap is **not** monotone —
10⁻⁵ and 10⁻⁶ both sit at 3.06e-8, above 10⁻⁴'s 8.23e-9 — because the greedy
proposal changes shape between budgets and the constraint is on being inside,
not on being tight. And the propagated bound at these constants is 7.15e11
absolute, which is vacuous and is why it is not a column anyone should read:
the measured columns are the result, exactly as the plan said they would have
to be.

What is missing from the curve is the *quality* axis. A validation loss per
point needs a training run per point, and the honest version of that is the
one experiment this version has not been able to afford.

## What comes next

Step 7 writes one vectorized matrix kernel under the realization contract,
which is what makes a narrower type buy time rather than only traffic. Step 8
sweeps the budget and measures the curve — the deliverable that does not
depend on the bound being tight, and the reason it was chosen as the primary
one before any of this was known.
