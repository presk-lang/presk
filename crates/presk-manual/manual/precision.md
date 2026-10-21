# Precision as a resource

This chapter follows the third axis of the research programme through
versions 0.5.0 and 0.6.0 (`presk-specs/implementation/presk-plan-0.5.0.md`
and `presk-plan-0.6.0.md`): given a declared error budget, which machine
types and which reduction orders should the backends use, and how far from
the exact denotation does that put the execution?

The **realization** — what the backends are allowed to do — is in the
`native` chapter, because that is where it is spent. This chapter goes from
the **magnitude envelope**, which makes a bound computable at all, through
the bound and why it is vacuous on a network, sensitivities, the solver that
verifies by measurement, the ε curve, time as the objective and integer
weights, to a budget on a training step. The sections are in the order they
were built, and a later one corrects an earlier one where a measurement did
(D48).

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
  SelfAttention.apply/%13      2.4032e1
  SelfAttention.apply/%15      1.6000e1
```

The file records the magnitudes per **site** — `"<function>/%<let>"`, the
same key a realization assigns a type to — together with the program, the
entry, the sample count, the precision and the day (in UTC). Every bound
computed from it is conditional on it and is printed with it (decision D35).
That is a weaker claim than a bound usually implies, and saying so in the
tool's own output is the answer rather than a footnote: **presk bounds the
distance between an execution and the denotation over an observed input
domain.**

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
initialization, so this says nothing about how the envelope moves during
training. The later sections measure on trained weights instead (`presk
calibrate --weights`, from 0.6.0 step 6 on).

## From magnitudes to a bound

`presk check --precision` propagates the rounding error of the entry the
envelope was measured on, under a given realization:

```sh
presk check --precision out/envelope_a.json corpus/nlp/05_gpt.psk --realize f32 --sites 6
presk check --precision out/gpt_env.json corpus/nlp/05_gpt.psk --realize f32 --model all --measure
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
and a test asserts it on 24 combinations. (A measured gap is the largest
difference over the output's elements, and +∞ when one of them is not
finite, as the language reference defines it: since 0.10.0 step 8, `presk
check --measure`, `presk solve` and `presk quantize` no longer leave such
an element out, which let a realization that overflowed pass on the
others; `tests/data/overflow.psk`.)

Whether it is *tight* is another question, and the answer on a real model is
no:

| Program | Realization | Measured gap | Bound | Ratio |
|---|---|---|---|---|
| `RMSNorm{8}.apply` | f64 | 2.2e-16 | 6.0e-15 | 27 |
| `LayerNorm{8}.apply` | f32 | 1.4e-7 | 6.9e-6 | 48 |
| `mse` | f32 | 6.0e-8 | 1.0e-4 | 1.8e3 |
| `loss` (tiny MiniGPT) | f64 | 8.9e-16 | 7.7e4 | 8.7e19 |
| `loss` (corpus GPT) | f64 | — | 7.2e20 | — |

(0.5.0's table, under its rule for sums, `k`; the corpus GPT's row was
measured before D48, on a `T = 256` checkpoint opened at `T = 32`. On the
trained model at `T = 256` the same rule gives 2.7·10²² against a measured
1.3·10⁻¹⁵ at 0.10.0: the table of "Does the condition number rescue the
bound?" below.) Over the 21 comparable cases the median ratio is 1.8·10³ and
ten are within 10³. On a normaliser the bound is worth having. On a two-block transformer it
is not: a relative bound of 10²⁰ on a loss of 4.5 says nothing.

**The reason is located, not mysterious.** A contraction of `k` terms
multiplies its input error by `k`, which is worst-case correct — all `k`
roundings *could* align — while the errors actually committed grow like `√k`,
being independent. Four contractions in series therefore carry `k⁴` of
pessimism. What would close it is each sum's **condition number**,
`Σ|xᵢwᵢ| / |Σxᵢwᵢ|`, measured into the envelope beside the magnitudes, which
replaces the worst case with what the program does. 0.6.0 step 6 measured
it; the next section is the answer.

**And it is a first-order analysis, not a proof** (D108, `make gappa`,
`docs/gappa-2026-10-09.md`). On the layer norm at `D = 64` in `f32`,
Gappa — given the kernels' operations in their order and the envelope's
magnitudes at every site as hypotheses — proves the distance to the exact
value is at most 3.3·10⁻⁵ on one row, whose native kernels are 2.4·10⁻⁷
from it and whose `cond` bound is 1.3·10⁻⁵; on an envelope of eight rows
it proves 6.9·10⁻⁵, where the kernels are 4.1·10⁻⁷ off and `cond` says
1.4·10⁻⁵:

| D | envelope | measured | `k` | `cond` | proved (Gappa, the envelope's sites) | proved / `cond` |
|---|---|---|---|---|---|---|
| 8 | one row | 6.8·10⁻⁸ | 5.8·10⁻⁶ | 2.4·10⁻⁶ | 2.3·10⁻⁶ | 0.98× |
| 64 | one row | 2.4·10⁻⁷ | 8.8·10⁻⁵ | 1.3·10⁻⁵ | 3.3·10⁻⁵ | 2.5× |
| 8 | eight rows | 1.4·10⁻⁷ | 5.0·10⁻⁶ | 2.8·10⁻⁶ | 9.2·10⁻⁵ | 33× |
| 64 | eight rows | 4.1·10⁻⁷ | 1.0·10⁻⁴ | 1.4·10⁻⁵ | 6.9·10⁻⁵ | 4.8× |

Every bound is above the measurement. On one row's envelope the proof and
the compiler agree (the proof the tighter at `D = 8`); on a sample's the
proof is above the compiler's, by 33× at `D = 8`. The two agree within a
factor of two on every sum and product and part at the square root and
the quotient, whose rules here are in relative terms at the site's
maximum — a value's error taken as proportional to its magnitude, so that
the row with the smallest variance carries the smallest error — where a
proof must hold for a row with the envelope's smallest variance and its
largest errors at once, which a sample's box admits and one row's does
not. So the compiler's bound on a sample is not a theorem of the sample's
envelope: the assumption the table above states for `cond`'s sums is in
every rule's quotients and roots as well, and the measurement supports it
where a proof does not have it. From the inputs' ranges alone Gappa proves 3.2·10³ (nothing then
keeps the variance from 0): the envelope's floor on the variance is what
a proof buys its 3.5·10⁷ from. Gappa's time grows about sixfold per
doubling of the width (4 s at `D = 8`, 10 min at 64), so 384 was not run.

This is the specification's own risk register arriving on schedule: worst-case
bounds are pessimistic by several orders of magnitude with respect to the
degradation observed. It is why the version's primary deliverable is the
*measured* curve of step 8 and not the bound, and why the solver of step 6
cannot simply verify against this number.

## Does the condition number rescue the bound?

No — it buys twenty-eight orders of magnitude and not the last nine (0.6.0
step 6, D51; measured again on the weights of D93).

`presk calibrate` now records, at every sum and contraction, what its terms
add up to without cancelling: `Σ|xᵢwᵢ|`, the numerator of the condition
number; `√Σ(xᵢwᵢ)²`, what independent errors in the terms add up to; and per
operand the factor its error is multiplied by, `Σ|wᵢ|` for `x`. The exact
interpreter gets them by evaluating each contraction again on `|operands|`
and on `operands²`, which is exact because a contraction is linear in each
operand. The propagation then has four rules for a sum, and nothing else
changes:

| `--model` | propagated error | what it is |
|---|---|---|
| `k` | `k · max|w| · max|δx|` | 0.5.0's worst case |
| `normwise` | `max|δx| · Σ|wᵢ|` | sound, no assumption |
| `cond` (default) | `(|δx|/|x|) · Σ|xᵢwᵢ|` | sound if each element's error is proportional to its magnitude |
| `rss` | `(|δx|/|x|) · √Σ(xᵢwᵢ)²`, and `√c` roundings | an **estimate**: the terms' errors independent |

The roundings of the three bounds are Higham's `γ_c · Σ|xᵢwᵢ|`, sound
whatever the rule. The componentwise assumption of `cond` is true of a value
rounded once and only assumed after that, so the soundness test still
asserts `bound ≥ measured` for it — and it holds everywhere.

On the conformance entries, the median ratio of bound to measured gap is
1.8·10³ under `k`, 53 `normwise`, **18 `cond`** and 12 `rss`; the tiny
MiniGPT's loss goes from 8.7·10¹⁹ to 8.8. On the corpus GPT at `D = 64`,
`T = 256`, trained 600 steps, on two windows of the Code civil (the loss
2.805):

| realization | measured | `k` | `normwise` | `cond` | `rss` |
|---|---|---|---|---|---|
| `f64` | 1.3·10⁻¹⁵ | 2.1·10³⁷× | 6.1·10²⁴× | 1.9·10⁹× | 9.7× |
| `f32` | 4.2·10⁻⁷ | 3.5·10³⁷× | 1.0·10²⁵× | 3.2·10⁹× | 17× |
| `bf16` | 7.8·10⁻³ | 9.5·10³⁶× | 3.4·10²⁴× | 7.0·10⁸× | 3.2× |
| solved at 10⁻³ | 1.2·10⁻⁷ | 1.2·10³⁸× | 1.3·10²⁶× | 4.0·10¹¹× | 1.1·10⁴× |

`make precision` prints these rows; since 0.10.0 step 8 `bf16`'s unit
roundoff is 2⁻⁸, where it was 2⁻⁹, every `bf16` rounding counted at half its
size (the language reference's chapter 4): the `bf16` row's bounds
doubled, the solved row's, with its eight `bf16` sites, rose by up to
twice, and the solver at 10⁻³ keeps in `f32` two sites it gave to `bf16`,
for the same measured gap. (On the weights of 0.6.0, trained with
the gradient defect 3f made wrong, D93, the same rows read `k` 10³³, `cond`
10¹⁰ to 10¹², `rss` 3.7 to 10 on the uniform realizations: the better
trained weights widen the worst-case bounds by four orders and leave the
conclusion where it was.) The GPT's sums are well conditioned —
the worst, `LayerNorm`'s mean, is 15 — so cancellation was never what made
the bound vacuous. What does is visible in the site tables: `cond` and `rss`
differ only in their rule for sums, and about eight sums in series each cost
roughly a factor of `k` between them. That is `k` against `√k`, and it lives
in the independence of the roundings, which no worst-case rule can see.

So the budget is **discharged by measurement** (D41) and the bound stays the
weaker artefact, printed as such; the statistical estimate is accurate on a
uniform realization and is what a proposal should use. On a mixed one it is
1.1·10⁴ times high (6.2·10³ before step 8's `bf16` unit roundoff), because a single magnitude per site charges every
attention logit the rounding of the largest one — finer magnitudes, not a
better rule for the sum, would close that. Questions 56 and 58 carry the
recommendation to the specification.

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
wide, the down-projection can be narrow. Measured on the corpus GPT at
`D = 64`, two heads, two blocks, trained for 600 steps (loss 4.52 → 2.76), at
`T = 256` — `make precision` does it:

| Site | What it is | `‖∂L/∂v‖₁` | per element |
|---|---|---|---|
| `GPT.apply/%17` | the block stack's parameters | 566.4 | 5.7·10⁻³ |
| `GPT.apply/%6` | the position table | 214.8 | 1.3·10⁻² |
| `GPT.apply/%4` | **the token embedding table** | **209.9** | **3.6·10⁻²** |
| `FeedForward.apply/%5` | the down-projection's parameters | 76.0 | 2.3·10⁻³ |
| `GPT.apply/%23` | the output head's parameters | 72.0 | 1.2·10⁻² |

**The folklore is reproduced.** Per element the token embedding table is the
most sensitive parameter of the model, the position table and the output
head come next; the down-projection is fifteen times less sensitive than
the embedding, which is exactly the practice of keeping embeddings and head
wide and narrowing the feed-forward matrices. (These are the weights of the
corrected gradient, D93: until 0.10.0 step 4 every training of this model
used a wrong gradient under dropout, defect 3f of D92, and the table read
119.2, 2.0·10⁻² for the embedding and 29.6, 9.0·10⁻⁴ for the
down-projection, the same order.)

This section said the opposite until 0.6.0 step 5d, and the reason is worth
keeping in view. 0.5.0 measured the ranking with a checkpoint trained at
`T = 256` and opened at `T = 32`; because `pos : ℝ[T, D]` makes `T` part of
the parameters, every leaf after the position table was read from shifted
offsets, and the "trained" weights were neither trained nor initial (D48).
On those weights the head came first by a factor of twelve and the embedding
sat below the down-projection, and the section built an argument on it. The
argument was about weights that did not exist. A measurement is only as good
as the claim about what it was measured on, which is the whole of D35 in a
different place.

The inputs are the solver's synthesized tokens, not text: a ranking is
driven by which values the output depends on and through how many
operations, and the magnitudes would move on text where the order should
not — which is a claim, and the one thing here not measured twice.

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
to an error constraint, which is a multiple-choice knapsack. The sites are
the values the program stores: a computation's result that is not fused
into another. The reference's chapter 4 gives a type no meaning anywhere
else, and the executors agree. A field of the parameters, a projection, a
call's or a fold's result carry a value unchanged; a fused member is
computed in its group's root's type. Since 0.11.0 step 4 the solver also
records each move it makes (`Solution::moves`: a narrowing tried, taken or
refused by the budget, a widening after a measurement), from which `presk
explain` gives each decision's type its reason. Until 0.11.0 the solver
offered those too, counted them as narrowed and priced their bytes, though
their types changed no bit (D112). Their rounding is now charged where it
happens: a member's at its root's type, a parameter's at the narrowest type
of the nests that read it. The bound charges a fused member at its root's
type too. It proposes greedily, in order of bytes saved per unit of
sensitivity-weighted error added; it verifies; and while the verification
fails it widens the site whose estimated contribution is largest, one step
of the menu at a time.

**What it offers since 0.12.0** (plan 0.12.0 step 4, D123):
- **The functions' results.** A function's result has its adjoint in its
  `.vjp`'s cotangent argument, and the entry's is the sweep's seed: the
  interpreter now records that argument's 1-norm as it records a site's, so
  a result is ranked and offered like any stored value. On the decoder at
  0.11.0 these were the head, the feed-forward products and the
  attention's output projection, left in `f64` under any budget (D117); at
  0.12.0 they are all `f32` at 10⁻³, and the block runs 3.5 to 4 times
  faster natively (D124).
- **The parameters,** by path, written in the realization's `params` (the
  section above). A parameter's sensitivity is its cotangent's 1-norm, read
  from the sweep's result as it returns and dropped, so never kept whole;
  its magnitude its largest element; its term its rounding at the
  narrowest of its own type and the types of the nests that read it, since
  the value a nest reads is rounded at rest and again at a narrower nest's
  edge. Which nests read it is followed from the entry through fields,
  calls and folds (`presk_precision::solve::param_reads`); the field sites
  that read it carry no term of their own. Its cost is its bytes at rest.
  So typing a parameter `f32` where only `f32` nests read it adds nothing
  to the estimate and saves its bytes. With `--quant` the integer menu
  still runs on top, priced by the whole cotangent, and a parameter it
  quantises leaves `params`. One menu for both kinds of type is a design
  for later (D126, `docs/precision-menu-roadmap.md`): the two are priced
  in different units. An entry whose first argument is not a parameter
  tree has no parameters to offer; until D128 a tensor there was offered
  as the parameter of path ``, which no executor runs in a type of its
  own, and every measurement narrowing it failed.
- **An entry whose result is a tensor** (logits): the budget bounds the
  largest elementwise gap, relative to the output's largest magnitude
  (`--budget`) or absolute (`--budget-abs`). One reverse sweep seeded with
  ones over the output ranks the sites by `‖∂(Σ y)/∂v‖₁`, read per output
  element, so that the estimate is the first-order change of a mean
  element; it ranks, and the measurement of the largest gap decides, the
  solver widening while it exceeds the budget, as since D41. Where the
  output's elements sum to a constant, as a layer norm's do, the seed of
  ones ranks nothing (every sensitivity about 10⁻⁴⁰) and the widening does
  all the work.

**It verifies by measurement, not by the bound** (D41). The bound is vacuous
on a transformer, so verifying against it would accept everything. Running
the candidate and measuring its distance to the exact reference is a stronger
check — the error actually committed rather than an envelope around it — and
it is a check only this compiler can make, because only this compiler keeps
an exact denotation to measure against.

**Where the measurement decides** (D128, `make widening`). The estimate
charges each stored value one rounding at its magnitude, to first order.
Three small programs show what that misses, each beside a form it judges
right (`docs/widening-2026-10-16.md`):
- an unshifted softmax of logits near 84, whose row sum of `exp` passes
  `f32`'s range: the estimate admits `f32` at 6.3·10⁻⁶ against a budget of
  3.7·10⁻⁴, the measurement finds 0.372, and four rounds widen it to
  all-`f64` (the `softmax` primitive, which subtracts the row's maximum,
  is accepted at 2.00×);
- the two-pass standard deviation of rows offset to 1 000, stationary in
  the mean, whose cotangent is therefore zero: the mean goes to `bf16`,
  the gap is 4.3·10⁻² against 1.2·10⁻³, and five rounds end at 1.00×;
- a sum of 4 096 equal terms, whose roundings share their sign: 1.6·10⁻²
  against 4.1·10⁻³, and one round puts the sum in `f64`.

On the first the bounds of `presk check` are below the gap too: they are
first-order, and nothing compares a value's magnitude with its type's
range. The widening goes by the estimate's terms, so the value it
misjudged is the last it widens, and it narrows nothing again: the first
two end at 1.00× where widening the culprit alone holds the budget at
1.33× and 2.00×.

The objective is traffic rather than time, and deliberately: `bf16` is
*slower* than `f32` on a compute-bound kernel (the `native` chapter has the
measurement), so minimising bytes and minimising seconds are different
problems. Pricing time per site needs the roofline's `W` beside its `Q`.

On the corpus GPT at `D = 64`, trained 600 steps (`make gpt-trained`, the
weights of the corrected gradient), budget 10⁻³ (`make precision` at
0.16.0, D163):

| | |
|---|---|
| traffic | 3.72e6 bytes against 7.51e6 all-`f64`, one sample's activations and the weights counted once, **2.02×** |
| stored values | 27, 23 in `f32` and 4 in `bf16`, the functions' results among them; no dropout mask, the loss being compiled outside a training run |
| parameters | 21, 10 in `f32` and 11 in `bf16` (the biases and the norms' β among them): 5.10e5 bytes at rest against 1.03e6 |
| widening rounds | 0 — the proposal was accepted first time |
| measured gap | 1.25e-6 against a budget of 2.8e-3 (estimate 2.45e-3, 87 % of it) |
| solved run, natively | 4.521755695, where `f64` gives 4.521788577 (on the parameters `presk run` synthesizes, as the target runs it) |

(At 0.12.0, D124: 33 stored values with the masks among them, 9.03·10⁶
bytes against 1.91·10⁷ at the envelope's two samples, 2.12×, measured
3.87·10⁻⁶, estimate 2.72·10⁻³. At 0.11.0, the stored values alone and the weights not counted: 29
sites, 25 in `f32` and 4 in `bf16`, 7.68·10⁶ bytes against 1.64·10⁷,
2.14×, measured 1.20·10⁻⁷, the solved run 4.521788597, D117. The bound
`cond` moves from 4.8·10⁴ to 3.0·10⁶, the parameter tree now charged at
`bf16`. At 0.10.0, before D112: 69 sites, 1.17·10⁷ bytes against 2.45·10⁷,
2.09×, the same measured gap and the same native loss; the 40 sites no
longer offered changed no bit. 0.5.0's figures, D41, on weights trained
then, with a smaller envelope: 1.29·10⁶ bytes against 3.45·10⁶, 2.67×, a
measured gap of 1.05·10⁻⁷ against 4.5·10⁻³. The curve at 0.12.0, at 10⁻³:
2.00× with the weights and 2.1·10⁻⁶, D124; at 0.11.0, 2.00× and
2.1·10⁻⁵, D117, as at the tag `v0.10.0`, D102; D93's, 2.08× and
1.89·10⁻⁶.)

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
sites alone gave 2.67× accepted on the first try (0.5.0's figure).

## The curve

`experiments/curve.sh` sweeps the budget on the corpus GPT. One program, one
parameter varying — which is the whole point. The specification's complaint
about the literature is that every quantisation level is a different pipeline
with different code and different bugs, so a sweep there compares artefacts
rather than precisions. Here the program is identical at every point.

Corpus GPT, `D = 64`, two heads, two blocks. The solver columns are measured
on the weights of `make gpt-trained` at `T = 256`, the reference at 128 bits;
the quality column trains the model **from scratch under each realization**,
600 steps on 90 % of the Code civil, and evaluates it in `f64` on the other
10 %. Every run starts from the same initialisation, seed and batch order, so
the only thing that differs between two rows is the realization. The
table is 0.12.0's (D124). 0.16.0 measured the curve again on the trained
model's envelope only, the curve of record since D132 (below):

| budget | traffic, vs all-`f64`, the weights counted | measured gap | inside | validation loss | vs all-`f64` |
|---|---|---|---|---|---|
| all-`f64` | 1.00× | | | 2.77361 | |
| 1e-1 | 2.78× less | 2.40e-4 | yes | 2.77366 | +5.2e-5 |
| 1e-2 | 2.29× less | 1.31e-5 | yes | 2.77361 | −1.1e-6 |
| 1e-3 | 2.00× less | 2.11e-6 | yes | 2.77361 | +1.3e-6 |
| 1e-4 | 1.97× less | 1.16e-6 | yes | 2.77361 | +1.3e-6 |
| 1e-5 | 1.58× less | 1.64e-6 | yes | 2.77361 | +1.3e-6 |
| 1e-6 | 1.35× less | 4.54e-9 | yes | 2.77361 | 0 |
| 1e-7 | 1.08× less | 3.73e-9 | yes | 2.77361 | 0 |

**On the trained model's envelope** (D132, `experiments/curve_trained.sh`,
`docs/envelope-state-2026-10-16.md`). `curve.sh` calibrates its envelope on
the initialization and random tokens, and a solve runs on its envelope's
inputs. Calibrated on the trained weights and two windows of the validation
text instead (windows those weights were trained on: `make gpt-trained`
streams the whole corpus), the solver narrows more at most budgets.
At 0.16.0 (D163) it gives 3.35× at 10⁻¹, 2.04× at 10⁻³ and 1.31× at 10⁻⁷,
every point still inside its budget and none widening (D132 at 0.12.0:
3.17×, 2.12×, 1.15×). The stored values exact or nearly exact in `bf16`
stay there for nothing under the measured term: the head width 32 at
every budget, and the layer norm's ε down to 10⁻⁶. Its realization at
10⁻³ lands 2.6·10⁻³ from the exact logits and flips 9 decisions (D132:
7.8·10⁻³ and 30), where this table's lands 2.5·10⁻⁴ and flips none: a
budget on the loss, a mean, does not bound the logits. The budget on the
logits does (V6), and statement S holds under both envelopes.

**Every point is inside its budget**, by measurement against the exact
reference, and the traffic falls monotonically as the budget loosens. The
measured gap is not monotone in the budget — 10⁻⁵'s assignment measures
1.6·10⁻⁶, above 10⁻⁴'s 1.2·10⁻⁶, though it narrows 25 stored values where
10⁻⁴'s narrows 30 — because the greedy proposal changes shape between
budgets and the constraint is on being inside, not on being tight.

**The quality axis is flat.** Every realization trains to within 5.2·10⁻⁵
of all-`f64`: the most narrowed, 32 stored values at 10⁻¹, 16 of them in
`bf16`, ends 5.2·10⁻⁵ above it, 10⁻² 1.1·10⁻⁶ below it, 10⁻³, 10⁻⁴ and
10⁻⁵ 1.3·10⁻⁶ above, and the two tightest on it to seven digits. The
yardstick is the seed: all-`f64` trained at seeds 0, 1 and 2 reaches
2.7736, 2.7584 and 2.6813, a spread of 0.092, some 1 800 times anything a
budget did. So there is no knee in quality to identify on this model at
this length of training: down to a budget of 10⁻¹ on the forward loss —
every stored value narrowed, half of them to `bf16` — precision costs
nothing a validation loss can see. The knee in *error* is still where it
was, between 10⁻² and 10⁻³ in traffic.

This table is D124's, measured at 0.12.0 on 2026-10-14 (`make curve`, 22
minutes). Since 0.12.0 the solver also offers the functions' results and
the parameters (D123): 32 stored values at the loosest budgets where it
offered 29, and each realization types the parameters too, 21 of them at
10⁻¹ to 10⁻⁵ (11 in `bf16` at 10⁻¹), 10 at the two tightest. So the
traffic counts the weights at rest. A training run trains under the
realization less its `params`: its parameters are the master weights it
updates, kept in the default type, and `presk train` refuses `params`. So
the quality column is of the nests' types, as every curve before 0.12.0
was. D117's table, at 0.11.0 on 2026-10-09, counted the 29 stored values
without the weights: 2.87× to 1.09× less traffic, the measured gaps
5.5·10⁻⁴ to 3.7·10⁻⁹, every validation loss within 1.7·10⁻⁴ of
all-`f64`'s. Since 0.11.0 step 2b the solver offers the values the program
stores (D112), 29 where D102's table at the tag `v0.10.0` counted 65, and
the traffic is theirs. At 10⁻², 10⁻³, 10⁻⁴ and 10⁻⁷ the measured gaps and
the validation losses are D102's: the stored values have the same types,
and the ratio moves only with what is counted. At 10⁻¹, 10⁻⁵ and 10⁻⁶ the
greedy order is another, and so is the assignment, each inside its budget;
at 10⁻¹ D102's was 2.69× less traffic, measured 3.7·10⁻⁴, and trained to
2.0·10⁻⁴ above all-`f64`. D102's table follows D93's of 2026-09-30,
measured again after the gradient defect 3f: until 0.10.0 step 4 every training of this model used
a wrong gradient under dropout (defect 3f, D92), and the curve measured
then (D50) read the same conclusions from wider numbers, every realization
within 1.2·10⁻³ of all-`f64`'s 3.0263 against a seed spread of 0.13. With
the gradient right the runs stay closer together: the step solved at 10⁻³
ends 5.5·10⁻⁶ from the `f64` run's weights where it ended 5.0·10⁻³
("A budget on the training step", below).

Between D93's table and D102's, two things moved the solver's columns.
Since 0.10.0 step 7b the attention reads its mask instead of adding it,
one site fewer: 65 narrowable, where D93's 66. Since step 8 `bf16`'s unit
roundoff is 2⁻⁸, where it was 2⁻⁹, so the greedy proposal prices a `bf16`
site at twice its former error, and the loosest budgets narrow less to
`bf16`: 21 sites at 10⁻¹ where 31, 2.69× less traffic where 2.95×.
Statement D's realization, solved at 10⁻³, became another assignment, 62
sites in `f32` and 3 in `bf16` (at 0.11.0 its 29 stored values, 26 in
`f32` and 3 in `bf16`), measuring 2.1·10⁻⁵ where D93's measured
1.9·10⁻⁶, inside by two orders of magnitude where it was by three. The
training runs use `--remat store`, the default, where D93's used
`recompute`; in `f64` the two compute the same bits, and the all-`f64`
run's weights are D93's byte for byte.

Two limits, stated. The model is the course's at reduced constants, because
seven runs at course scale would take days here; and 600 steps is short of
convergence, which is where a seed matters most. A realization that is free
at 600 steps might not be at 60 000.

This table replaces 0.5.0's, which was measured on a checkpoint opened at
the wrong `T` (D48): its gaps described weights that were neither trained
nor initial.

## A budget from an accuracy target

The curve's quality axis is flat; it does not say which ε an accuracy
target needs. The margins do (D118, `make margins`, the `experiments`
chapter §5):
- **A flip needs a close call.** A realization within ε of every logit
  changes the decision `argmax` only where the exact model's margin, its
  top logit minus the second, is below 2ε. So the accuracy it loses is
  at most the fraction of positions with m < 2ε, which the exact model
  alone gives.
- **ε\*(δ)** is half the δ-quantile of the margins on held-out data,
  corrected for the sample.

On the course GPT's 118 016 held-out positions the exact margins give
ε\*(10⁻³) = 4.7·10⁻⁴ at 95 % confidence. Five of the curve's seven
realizations at 0.11.0 are within it on the logits, statement D's at 3.3·10⁻⁴,
and that one changes one decision where the margins allow 70. Every
realization measured changes at most as many as its margins allow. On
0.12.0's curve the same five are within it, statement D's at 2.5·10⁻⁴
with no decision changed (D125).

So an accuracy target is a budget on the logits, and it covers every
realization within it. Two limits:
- **The budget must be on the logits.** A budget on the loss is not
  enough: the curve's 10⁻² realization is inside its budget and 30 times
  ε\*(10⁻³) on the logits.
- **The solver takes the target since 0.12.0** (D123): `presk solve
  --budget-abs 4.67e-4` on the logits, a budget on their largest
  elementwise gap. It narrows 29 of the 31 stored values and types all 21
  parameters, at 1.97× less traffic than all-`f64`, the weights counted.
  Its largest gap on the 461 held-out windows is 7.5·10⁻⁵, and it changes
  no decision, where statement D's realization, solved on the loss at
  10⁻³, has 3.3·10⁻⁴ and one (D124).

## A rounding term from the values, and narrowing by measurement

Two options of `presk solve` (D155, `make solver-rounding`, the
`experiments` chapter §17). The first is the default since 0.16.0 (D157,
D158), the user's decision after D155 and D156 measured it holding out of
sample at both budgets and every sample size; `--rounding unit` is the
solver of 0.12.0–0.15.0, by which every row of record until 0.16.0 was
solved. The second is off by default.
- **`--rounding measured`.** The envelope records at each site the largest
  rounding error of its sampled values in each type, `max |fl_d(x) − x|`,
  and the estimate charges a site that, where it charged `u(d) · max|x|`.
  The term is never larger. It is 0 for a value exact in the type: the
  constant 32 costs nothing in `bf16`, and so did the four dropout masks
  of the course GPT's logits, tensors of ones outside training, until
  0.16.0 compiled an entry outside training with no mask (D160). It is the
  estimate's rounding step measured on the sample, as conditional on the
  envelope as the magnitude, and no bound; the bound `cond` is unchanged.
  On the logits at ε\*(10⁻³) the solver went from 1.97× to 2.14× in one
  sweep and one measurement at 0.15.0, 1.6·10⁻⁴ from the exact logits on
  the 461 held-out windows with no decision changed. That gain was the
  masks. At 0.16.0, with no mask left, the term gives 1.96× at ε\*, as the
  unit term does, and 1.31× against 1.26× at 10⁻⁶ (D165). It still counts
  where a value is exact: the head width, and the norms' gains of a model
  at its initialization (D162, D166).
- **`--narrow measured`.** After the measurement passes, each decision is
  narrowed one step, the largest saving first, measured, and kept while the
  gap stays inside the budget, in up to three passes. At ε\*(10⁻³) it finds
  the realization the tuning tools' searches found (D154), in 79
  measurements, and it holds on the held-out windows. That was 2.17× at
  0.15.0, and 2.00× in 74 measurements at 0.16.0, the masks no longer
  counted (D165). At 10⁻⁶ it
  narrows until the envelope's four windows are within 0.7 % of the budget,
  and fails on the held-out ones, as the searches did. A narrowing decided
  by measurement needs a check on data it was not measured on, whatever the
  sample's size: calibrated on the first 1 to 32 windows, it fails the
  others at 10⁻⁶ every time (D156, the `experiments` chapter §18).

## Three ways to spend the budget

The solver of record spends a small part of its budget in sample: its
first-order estimate is conservative, and a measurement that passes is not
narrowed further. Narrowing by measurement spends it to the last per cent and
fails out of sample (D155, D156). D156 gives the factor between the two: on
the course GPT's logits, the gap out of sample is at most 2.5 times the gap in
sample from four windows on. Three options of `presk solve` (0.16.0, D161)
keep that factor:
- **`--estimate rss`**: the proposal's estimate is the root-sum-square of the
  sites' terms, the roundings taken as independent (Higham and Mary's
  probabilistic analysis), instead of their sum, and it is aimed at the
  budget divided by `--margin`. The measurement then checks the budget as
  before.
- **`--narrow measured --margin k`**: D155's narrowing, each step kept while
  the measured gap is inside the budget divided by `k` (D155's is `--margin
  1`).
- **`--narrow heldout`**: the same narrowing, then early stopping. The steps
  kept are undone, the last first, until the gap on the samples after the
  envelope's in its data file (as many as the envelope's, or `--heldout M`)
  is inside the budget divided by `k`. Kept while the gap was inside one
  threshold on both parts, a narrowing would be the narrowing measured on
  their union, which D156 measured failing; the held-out samples judge the
  sequence of steps instead.

`--margin` is 2.5 by default; the solver of record does not read it. Each
option was to become a default only if, measured as D155 and D156 were, it
held out of sample in every cell with a margin of 1.5× left and its
realization stayed the same from 4 to 16 windows (the plan's rule, `make
spend`, the `experiments` chapter §19). None did (D161):
- *Every form holds* out of sample on the course GPT's logits at D = 64.
- *Nothing keeps 1.5× at 10⁻⁶,* the default included, which ends at 0.89
  of the budget.
- *No new form keeps its realization* between 4 and 16 windows.
- *The narrowing with k = 2.5* is the one form that saves traffic at both
  budgets: 2.002× against 1.966×, and 1.554× against 1.266×. It holds where
  D155's failed, but at 16 windows it keeps nothing at 10⁻⁶.

The three stay flags.

## The cost per sample

The traffic of a site is its elements times its type's bytes. A parameter
is counted once; an activation's elements, at an envelope calibrated on N
samples of a data file, are N samples' at most sites and the model's at a
few (a mask, a constant). Until 0.16.0 the solver priced the activations at
the envelope's N, so that a parameter weighed 4/N times as much against them
as at four samples, and the decisions moved with N below four (D156). Since
0.16.0 (D158) `presk calibrate --data` also counts each site's elements for
one sample, by running the entry again on the first sample alone, and
writes them as the envelope's `per_sample`; the solver prices the
activations at those counts, the parameters once, the same cost at any N.
`presk solve` says so in a line `per sample`. An envelope without data, or
of one sample, is priced as it is; one of several samples written before
0.16.0 is priced at its batch, and the line says that too. The ratio the
solver prints, all-`f64`'s traffic over the realization's, is therefore one
sample's. Ratios from before 0.16.0 read differently: the weights weigh
more against one sample's activations than against N's, and the masks no
longer count. The course GPT's curve at 10⁻³ prints 2.04× at 0.16.0 where
D132's printed 2.12×, with fewer stored values narrowed and the same
budget met.

## Time, not traffic

0.5.0's solver minimised **traffic**, because traffic is what a narrower type
reduces, and statement E failed at 0.96×: on a compute-bound site narrowing
buys nothing and the conversions cost, so minimising bytes chose assignments
that were *slower*. Since 0.6.0 step 4, `presk solve --profile` prices each
site with the roofline's own prediction, `T̂ = λ + max(W/π, Q/β)`, with the
dtype in `Q` and not in `W`, and `W` taken from the site's element count and
the weight of its primitive, which the envelope now records.

The difference shows in the objective's own numbers. Under traffic the
solver reports 2.67× less; under time it reports 1.19× faster — and the
**relaxation floor rises above the chosen cost**, meaning that narrowing
everything would now be *slower*, which the traffic objective could not
express at all.

Measured on the corpus GPT at `D = 64`:

| | median ms |
|---|---|
| all-`f64` | 5.589 |
| all-`f64`, two lanes | 4.913 |
| `f32`, four lanes | 4.486 |
| the solved realization | **5.287** |

**Statement J fails**: 1.06× against the 1.3× it asked for. But the sign has
changed — 0.5.0's traffic-solved assignment was 0.96×, *slower* than doing
nothing — so pricing time rather than bytes turned a loss into a gain, and
what is left is the size of the gain rather than its direction.

Lanes are now **on by default** on the native backend, because they are
bit-exact: there is no accuracy to trade, no realization to declare and no
gap to measure, and a transformation that is free and exact has no reason to
be opt-in. That does not close J, because it raises both sides — with lanes
everywhere the solved realization is 5.20 ms against all-`f64`'s 5.50, still
1.06×. It made every native run faster for nothing on the development
machine, which was worth having on its own. It is not a rule: on falicon,
BERT-large's forward pass at batch 20 is 15 % slower in lanes in `f32`, and
1.5× faster in lanes with `int8` weights (D53's benchmark, `native`
chapter).

Turning them on exposed a bug worth recording: the lane width was being taken
from the *realization's default*, and a mixed assignment has kernels of
several types, so a two-lane request reached a `bf16` kernel that cannot take
one and the whole run was refused. The width belongs to each kernel's own
type, not to the assignment's default.

What is still missing for J is named: the conversion cost at kernel edges is
not in the model, so the solver does not see what a mixed assignment pays at
the boundaries between types. Run-to-run variation on this model is also
several per cent, which is the same order as the effect being measured, so J
needs a larger model before its number means much.

## Integer weights

Since 0.6.0 step 5a a realization can store parameters as `int8` or `int4`
with a scale per channel (D44). It is a **storage** type, like `bf16` with a
scale: each weight means `f32(q·s)`, dequantised when it is loaded, and every
contraction then runs in `f32` in the usual Blocked form. Only parameters are
quantised, never activations, gradients or optimizer state.

The realization names the parameters by their path in the tree and says how:

```json
{ "default": "f32", "order": { "kind": "sequential" },
  "quant": { "head":           { "bits": 8, "over": -2 },
             "blocks.ff.up.w": { "bits": 4, "over": -2, "scheme": "pow2" } } }
```

`over` is the axis the parameter is **contracted** over, by name or by
position from the end; there is one scale per index of every *other* axis,
so a matrix of a stacked module gets scales per layer and per output column.
That is what makes the scale factor out of the sum. A position rather than a
name is usually needed, because module parameters are written without axis
names (`param w : ℝ[In, Out]`), and `-2` survives the leading axis `stack{L}`
adds. `presk quantize --list` lists the paths and shapes.

The scales are computed from the weights — symmetric absmax, `s = max|w| /
(2^(b−1) − 1)` rounded to `f32`, round to nearest — whenever the weights are
loaded, by `run`, `serve` and `quantize` alike. They are not written into the
realization file, because they belong to a checkpoint and not to a program
(question 57), and they need **no calibration**: a weight-only realization is
not conditional on an envelope. `"scheme": "pow2"` rounds each scale up to a
power of two instead, up to twice as coarse, and in exchange `q·s` is exact.
An all-zero channel gets a zero scale and stores zeros. `presk train` refuses
a quantised realization: it applies to trained weights.

The error rule is `|w − ŵ| ≤ s/2`, absolute, plus one `f32` rounding of `q·s`
under `absmax`. What only an exact reference can do is split the gap of the
*output* in two, and `presk quantize` does: the **representation** part is
the exact denotation on the dequantised weights against the exact denotation
on the true ones, the **arithmetic** part the native run against the former.
`make quant-table` measures every realization that way, on the corpus GPT at
`D = 64` trained for 600 steps (`make gpt-trained`, loss 4.52 → 2.76), on two
windows of the Code civil at `T = 256`, every matrix quantised over `-2`:

| realization | weights (bytes) | representation | arithmetic | total |
|---|---|---|---|---|
| `f64` | 1 026 048 | 0 | 9.67·10⁻¹⁶ | 9.67·10⁻¹⁶ |
| `f32` | 513 024 | 0 | 2.99·10⁻⁷ | 2.99·10⁻⁷ |
| `bf16` | 256 512 | 0 | 1.98·10⁻³ | 1.98·10⁻³ |
| `int8`, `f32` elsewhere | 139 120 | 2.77·10⁻⁵ | 3.73·10⁻⁷ | 2.73·10⁻⁵ |
| `int8` `pow2` | 139 120 | 1.46·10⁻⁵ | 4.48·10⁻⁷ | 1.50·10⁻⁵ |
| `int4`, `f32` elsewhere | 75 888 | 4.74·10⁻⁴ | 5.50·10⁻⁷ | 4.73·10⁻⁴ |
| `int4` `pow2` | 75 888 | 1.91·10⁻⁴ | 5.57·10⁻⁷ | 1.91·10⁻⁴ |

(relative to the exact loss, 2.7555). **Statement K holds**: `int8` stays
inside a 10⁻² budget by a factor of three hundred and sixty. Rounding the
weights costs two orders of magnitude more than `f32` arithmetic at eight
bits and three at four, and the two parts need not add. The comparison with
`bf16` is not like for like — `bf16` rounds every activation as well, and
that is where its error comes from — but it is the one a practitioner
makes: storing the weights in eight bits and computing in `f32` is 72 times
closer to the exact loss than storing everything in sixteen, in half the
bytes. On random tokens instead of text the same `int8` row is 4.79·10⁻⁵,
1.8 times larger: a trained model's loss on inputs it was not trained for
is a different function, and the table is measured on the one it was.
(This table is D93's, on the weights of the corrected gradient. On 0.6.0's,
which defect 3f trained wrong, D92, the `int8` row read 3.23·10⁻⁶, a factor
of three thousand inside, and 7.96·10⁻⁵ on random tokens: K's margin moved
by an order of magnitude with the weights, which is D53's point below.)

K is a statement about a loss on two windows, and 0.6.0 step 8 measured how
far it carries (D53). On the held-out windows of another checkpoint — the
curve's all-`f64` run, scored on the tenth of the Code civil it never saw —
the same `int8` scheme is 3.90·10⁻⁵ from the exact loss, and `int4`
3.99·10⁻⁴: 1.4 and 0.8 times the table's figures, and inside 10⁻². On
0.6.0's weights the held-out figures were forty-five and twelve times the
table's (1.46·10⁻⁴ and 1.02·10⁻³): those windows had been favourable rather
than typical, since the error of a quantised loss is a signed sum of many
terms, as the next section shows, and there it had cancelled well. A loss also averages over 512 positions, where a model's outputs do not:
on BERT-large's SQuAD logits, which is what a budget on an inference entry
binds, the same scheme is 5.7·10⁻² from the exact denotation (`coverage`
chapter).

A 0.5.0 bug is behind the choice of `T = 256`. `pos : ℝ[T, D]` makes `T` part
of the parameters, and a checkpoint trained at `T = 256` used to load at
`T = 32` without a word, every leaf after `pos` read from shifted offsets.
Since step 5d a checkpoint that is not exactly the size of the program's
layout is refused (`E9005`, D48).

## Parameters in their own type

A realization stores every parameter in its default type, `f64` for a
solved one, and an `f32` nest reads each weight converted at its edge: so
typing a weight's binding narrower changed no bit and saved no byte (D109),
and 0.11.0's solver stopped offering it (D112). Version 0.12.0 gives a
parameter a storage type of its own at rest, the floating-point
counterpart of `quant` (plan 0.12.0, D119; the reference's chapter 4,
W18):

```json
{ "default": "f64", "order": { "kind": "sequential" },
  "params": { "blocks.ff.up.w": "bf16", "head": "f32" } }
```

The leaf is converted once to its type when the entry is given it, from
the value given, and every nest reads it from that type. A path is a
parameter's, as `presk quantize --list` prints them; a stacked parameter
has one type for all its layers.

Since 0.12.0 step 1 (D120) the record is parsed and checked where the
parameters are loaded, and refused with `E7001` for a type other than `f32`
or `bf16` (`f64` is the default's to state), for a path `quant` names too
(a parameter is typed or quantised, not both), and for a path that is not
a parameter of the model, naming those there are. `presk train`, and the
steps `calibrate --step` and `solve --step` measure, refuse a realization
that types parameters, as they refuse one that quantises them: the master
weights stay in the default type.

Since 0.12.0 steps 2 and 3 (D121, D122) every executor runs it: `presk
run`, `bench`, `serve` and the measurements of `solve`, `check --precision
--measure` and `quantize` store the parameters in their types once, when
they are loaded, and keep them there.
- **A nest of the parameter's type** reads it as it is, with no copy.
- **A nest of a wider type** (a `bf16` weight read by an `f32` product, an
  `f32` one by an `f64` nest) gets an input marked as narrower, as integer
  weights are marked, and its loads widen each element, exactly. Nothing
  is converted at its edge (`presk dump --stage kernel` shows such an input
  in its own type, `bf16[…]` in an `f32` kernel).
- **A `bf16` nest** reads an `f32` parameter rounded to `bf16` at its edge,
  as it reads every wider operand: its inputs are `bf16`, so it reads what
  it stores. The reference's chapter 4 read such an operand in `f32` until
  D120 found the difference; the user decided on 2026-10-15 to amend the
  reference to what the executors do, which moves no output (its question
  W19, closed).
- **The C export** keeps each such weight in its type in the program's
  arena, read from the `.psk.d` as the weights always are, and its kernels
  widen it as the native ones do, `presk serve`'s loop (`generate.c`)
  included.

Under such a record, each executor gives the bits the same record without
`params` gives on the parameters rounded to their types beforehand, and the
three agree as they always have (`presk-cranelift/tests/params.rs`,
`presk-cli/tests/export_c.rs`, statement V2). With every parameter in a type
no wider than the nests that read it, the host's conversion phase moves no
byte of it (`presk bench --host-profile`'s `buffers.convert`, V3).

`presk solve` chooses these types since step 4 (the solver's section
above). At the Llama 2 decoder's 7B width, under its solved realization
(D124), the 13 parameters are in `f32`, 1.74 GB at rest against 3.47 GB in
`f64`, and no byte of them is converted at an edge. The block's forward
pass runs 3.5 to 4 times faster than under 0.11.0's realization, which
converted 944 MB of weights a call.

## Which weights to quantise

`presk solve --quant` adds the parameters to the menu: after the float
assignment, every matrix — a parameter *declared* of rank two, so a stacked
bias is not one — may be stored as `int8` or `int4`, under the whole budget,
accepted by measurement as the float assignment is (D41). The proposal is
priced by the gradient the reverse sweep already computes: quantising moves
each weight by a known `δ = ŵ − w`, so to first order the loss moves by
`Σ ∂L/∂w · δ`. On the trained corpus GPT (random tokens, the solver's inputs):

| all matrices | bound `Σ|g|·|δ|` | signed `Σ g·δ` | `√Σ (g·δ)²` | measured |
|---|---|---|---|---|
| `int8` | 2.55·10⁻¹ | 3.26·10⁻⁴ | 1.55·10⁻³ | 2.99·10⁻⁴ |
| `int4` | 4.62 | 5.79·10⁻² | 2.79·10⁻² | 4.94·10⁻² |

The signed first-order prediction is within nine per cent of the
measurement in `int8` and seventeen in `int4`; the worst-case bound is 855
times too large (at the tag `v0.10.0`, D103's addendum, the `int8`
measurement on top of D102's float assignment; D93's, on the same weights,
3.22·10⁻⁴, within two per cent and 790 times, the `int4` row D93's; on
0.6.0's weights, D49, within five and nine per cent, and 230 times). The linearisation is right, and
what makes a deterministic bound vacuous is only its refusal to let
independent roundings cancel — the same diagnosis as D39, now with the
number that decides it (D49, question 58). The proposal uses the
root-sum-square, which is within a factor of two, never negative, and cannot
make a set of parameters look free because their signs happen to cancel. At
a 10⁻³ budget the solver quantises all nine matrices in `int8`, the weights
7.7× smaller than in `f64`, estimated 1.55·10⁻³ and measured 2.99·10⁻⁴,
accepted without widening (on 0.6.0's weights five `int8` and four `int4`,
11× smaller, measured 3.11·10⁻³). `--scheme pow2` takes
power-of-two scales.

Since step 5b the weights stay integers all the way to the kernels, and a
contraction loads them as `f32(q)·s` — which is exactly the value the
reference path substitutes, so running on the integers is **bit-identical**
to running on their dequantised `f32` values, on both executors, scalar and
in lanes, in `f32` and in `f64` (a test asserts it). The arithmetic column
above is therefore the integer kernels' as much as the float ones'.

What they buy is bytes, and bytes buy time only where the kernel waits for
memory. Native, lanes on, median ms (since step 5c `int4` too, two to a
byte):

| | `f32` | `int8` | `int4` |
|---|---|---|---|
| one contraction, 64 × 1024 × 4096 (16 MB of `f32` weights) | 52.9 | 10.8 | 10.1 |
| GPT forward, `D = 256` | 27.7 | 25.9 | 27.3 |
| GPT forward, `D = 64` | 1.87 | 1.92 | 2.28 |
| BERT-large forward at MLPerf's constants, per sequence at batch 20 (0.6.0 step 8) | 6 918 | 3 101 | — |

Run to run, the `D = 256` rows move by about ten per cent, so the GPT is
level in both integer types and `int4` pays for its unpacking at `D = 64`.
`int4` buys nothing over `int8` on the large contraction: once the weights
fit in cache at eight bits, halving them again saves traffic the kernel was
no longer waiting for.

BERT-large is where the integers first pay on a whole model: 2.2×, at batch 20
on twenty cores as at batch 1 on one. (Its row is the median per sequence of a
batch of 20, `f32` in scalar code, which is faster there than its lanes at
7 926 ms; `int4` was timed on two of the 24 blocks only, level with `int8` at
5 869 against 5 625 ms.) Its contractions multiply every row by weight
matrices of up to 16 MB in `f32`, and the blocked kernel reads the whole
matrix again for every row, so in `f32` it waits for memory; at eight bits,
4 MB, the matrix fits in one of the machine's two 12 MB last-level caches,
which is D45's account of the 16 MB contraction above. That is this kernel's regime rather than
a property of `int8`: a cache-blocked contraction would be compute-bound there
(`docs/kernels-roadmap.md`).

A weight matrix that fits in cache gains nothing, and the conversion is then
the price; one streamed from memory once per row gains roughly what its bytes
say, a little more when a quarter of them fits a cache level the whole did
not. Before the four-lane load was written, converting lane by lane made the
GPT forward 1.6× *slower* than `f32` — the same lesson `bf16` taught in
0.5.0, that a narrower type is a conversion before it is a saving.

## J at course scale

A realization is keyed by function and `let`, not by shape, so one solved at
`D = 64` applies unchanged to the course's `D = 384`, six blocks, where a
training step is long enough — a minute in `f64` — for run-to-run noise not
to matter. Solved at 10⁻³ with the time objective (a profile fitted per dtype
on the corpus GPT, `presk bench --calibrate`), 62 sites in `f32` and 4 in
`bf16`; per step, the four-step time minus the one-step time over three:

| realization | s / step | vs all-`f64` |
|---|---|---|
| all-`f64` | 56.9 | 1.00× |
| solved at 10⁻³ | 50.2 | **1.13×** |
| all-`f32` | 30.4 | 1.87× |

**Statement J fails at course scale too**, and the last row says why: the
speed is there, and the solved realization does not reach it because it
assigns the sites of the entry it was solved on, the loss. A training step
is mostly the reverse sweep and the optimizer, whose sites it does not
mention and which run at its default, `f64`. The budget constrains the
forward loss and gives the solver no licence to narrow the gradient.
Closing J is a matter of saying what a budget binds in a training run —
question 60 — not of a faster kernel.

## A budget on the training step

Step 7b (D52) gives `precision = budget(rel_error = ε)` in a `training run`
its meaning: it binds **one optimizer step's parameter update**. From a state
`(θ, m, v, t)` and a batch, the exact step `θ′` — the objective's `vjp`, then
`adamw.step`, over the reals — and the step under a realization `θ̂′` must
satisfy

```text
‖θ̂′ − θ′‖₂ / ‖θ′ − θ‖₂ ≤ ε
```

by measurement. The update and not the gradient, because the optimizer's own
sites are part of the step and because AdamW divides by `√v`, which a
gradient budget cannot see; relative to the update and not to `θ`, which is a
thousand times larger; the 2-norm, which gradient clipping already uses. The
state is a trained checkpoint followed by a warm-up natively in `f64`, so that
`v` is not zero, and "the same state" is that state rounded to the
realization's default type: an `f32` run's parameters already live in `f32`.

`presk calibrate --step` measures it, and prints two more numbers beside it.
The **gradient gap** `‖ĝ − g‖₂ / ‖g‖₂` says where the error enters. The
**storage floor** is the update gap of the exact `θ′` rounded to the type the
optimizer stores it in, which no arithmetic can go below. The update is a
small difference of large numbers — `‖θ‖ / ‖θ′ − θ‖` is 1 051 on the corpus
GPT at its peak learning rate — so the floor is `u·‖θ‖ / ‖θ′ − θ‖`, and it
depends on the schedule: on the tiny MiniGPT at step 4 of a 2000-step warm-up
`f64`'s own floor is 2.2·10⁻¹¹ and `f32`'s 1.2·10⁻². A budget relative to the
update is therefore stated at a state, here the step after the warm-up
(question 61).

**The table** (`make step-table`): step 101 of 600 of the corpus GPT at `D =
64` from `make gpt-trained`'s weights and 100 warm-up steps, two windows of
the Code civil, against the exact step at 128 bits. The sampling noise between
two such batches, `‖g_B − g_B′‖₂ / ‖g_B‖₂`, is 1.40.

| realization | update gap | storage floor | gradient gap | predicted time |
|---|---|---|---|---|
| all-`f64` | 6.8·10⁻¹³ | 4.8·10⁻¹⁴ | 9.9·10⁻¹⁶ | 1.00× |
| all-`f32` | 3.0·10⁻⁴ | 2.6·10⁻⁵ | 5.5·10⁻⁷ | 1.01× |
| all-`bf16` | 0.78 | 0.78 | 1.2·10⁻² | 0.49× |
| the recipe (`bf16` but the optimizer) | 4.4·10⁻² | 2.6·10⁻⁵ | 1.3·10⁻² | 0.53× |
| solved at 10⁻³ (16 functions `f32`) | 3.0·10⁻⁴ | 2.6·10⁻⁵ | 5.6·10⁻⁷ | 1.01× |
| solved at 10⁻² (15 `f32`, `adamw.lr` `bf16`) | 1.8·10⁻³ | 2.6·10⁻⁵ | 5.6·10⁻⁷ | 1.01× |

(D93's table, on the weights of the corrected gradient and with the step's
own gradient corrected, both of which defect 3f made wrong until 0.10.0
step 4, D92; D52's, measured on them, is kept there and reads the same
conclusions: 2.2·10⁻⁴ in `f32`, 340, 2.9·10⁻², 16 functions `f32`.)

Four things in it. The `f32` gradient is 5.5·10⁻⁷ from the exact one and the
`f32` update 3.0·10⁻⁴: AdamW multiplies the gradient's relative error by 545,
and a budget on the gradient would read this step as 545 times safer than it
is — question 60's provisional answer, measured wrong. Where it enters is the
solver's per-function table: `SelfAttention.apply.vjp` in `f32` costs
3.1·10⁻⁴ alone, `Linear.apply.vjp` 3.7·10⁻⁵ and `adamw.step` 3.7·10⁻⁵ (most
of it the floor); no other function reaches 2·10⁻⁷. `bf16` master
weights are impossible — the floor alone is 0.78 — which is why the recipe
keeps them in `f32`, and with them in `f32` the **recipe is 4.4·10⁻² from the
exact step**, outside a 10⁻² budget, a hundred and fifty times further than
`f32`: a number no framework can produce, and small against the noise of
1.40 between two batches, which is the recipe's justification. And the
solver, which may choose `f32` or `bf16` per function, takes `f32`
everywhere and `bf16` nowhere, because the profile predicts `bf16` slower
than `f64`: the budget licenses the narrowing, and the time objective refuses
the part that does not pay, as §3.7 expected. At 10⁻² it also puts the
scalar schedule `adamw.lr` in `bf16` for a predicted 5 µs, which is legal and
pointless — the resolution of the time model, not of the budget.

**J′ at course scale** (`make j-step`): the realization solved at `D = 64` is
keyed by function, so it applies to the course's `D = 384`, six blocks. Per
step, the four-step time minus the one-step time over three; the update gap
of one step from a course-scale checkpoint and ten warm-up steps, against a
native `f64` step from the same state (the exact one is not affordable at
this size, and `f64` is within 10⁻¹² of it wherever both were measured):

| realization | s / step | vs all-`f64` | update gap | storage floor | gradient gap |
|---|---|---|---|---|---|
| all-`f64` | 24.4 | 1.00× | — | — | — |
| solved at 10⁻³ | 13.9 | **1.75×** | 6.8·10⁻⁵ | 4.8·10⁻⁵ | 3.9·10⁻⁷ |
| all-`f32` | 14.0 | 1.75× | 4.8·10⁻⁵ | 4.8·10⁻⁵ | 3.9·10⁻⁷ |
| the recipe | 16.6 | 1.47× | 6.5·10⁻³ | 4.8·10⁻⁵ | 4.1·10⁻³ |

**Statement J′ holds**: 1.75× faster per training step than all-`f64`, and
inside its budget by measurement at both scales. (The measurement at the
tag `v0.10.0`, under `--remat store`, D103. D93 measured 1.95×, 44.2 s
against 22.7, on 0.10.0-dev under `--remat recompute`, with the same gaps;
D52 measured 1.99×, 59.7 s against 30.0, on 0.6.0 and on weights defect 3f
trained wrong, and 2.02× at the end of 0.6.0. Every step is faster than
then, all-`f64` the most, so the ratio fell; the policy moves neither:
under `recompute` at the tag all-`f32` is 1.76×. The realization is keyed
by function, and a stored gradient's functions are `.fwd` and `.bwd`, so a
step is solved under the policy it runs under.) The solved step is all-`f32`
by function and runs level with all-`f32`; the recipe is slower than `f32` at
this size, the `bf16` conversions costing more than the traffic saves. J
failed at 1.13× in step 7 for a reason of scope, not of speed, and J′ is the
same measurement with the budget binding what a training step does.

**The trajectory** (`make step-train`). A per-step budget does not bound a
training run, and is not claimed to, so the run is measured: the corpus GPT
trained from scratch for 600 steps at `D = 64` under the solved step, and
under the recipe, exactly as the curve's rows were (same initialisation, seed,
batch order and split of the Code civil), then evaluated in `f64` on the
held-out tenth, against the curve's all-`f64` runs:

| trained under | validation loss | vs all-`f64` | `‖θ − θ_f64‖ / ‖θ_f64‖` |
|---|---|---|---|
| all-`f64`, seed 0 | 2.7736101 | | 0 |
| the solved step, 10⁻³ | 2.7736099 | −2.0·10⁻⁷ | 5.5·10⁻⁶ |
| the recipe | 2.7694 | −4.2·10⁻³ | 1.1·10⁻² |
| all-`f64`, seed 1 | 2.7584 | −1.5·10⁻² | 1.22 |
| all-`f64`, seed 2 | 2.6813 | −9.2·10⁻² | 1.22 |

The run under the solved step ends two hundred thousand times closer to the
`f64` run than a change of seed puts it, and the recipe a hundred times;
both validation losses are inside the spread the seed makes. Over 600
steps, then, nothing accumulated into a different model: the solved step's
distance is about what its per-step gap adds up to without being
amplified, 3·10⁻⁴ of an update a thousandth of `‖θ‖`, over 600 steps. That
is what this length of training shows, and a longer run is the obvious
next measurement, as D50 said of the quality axis. (D93's table; with the
gradient defect 3f made wrong, D52 measured the solved step 5.0·10⁻³ from
the `f64` run, 245 times closer than a seed, and the recipe 3.0·10⁻².)

## The reference's own gap

Every gap of this chapter is measured against the reference interpreter
(`language` chapter, "Running programs: the reference interpreter"): MPFR at
`--bits`, 256 by default, 128 in the calibrations above, every operation
correctly rounded, the sums term by term in rank order. That is a
realization of the denotation too, not the denotation, and the language
reference (chapter 4, "Discharge") names its distance δ_p and takes it as 0
at p ≥ 256 bits — a discharge conditional on δ_p lying below the
resolution the measured gap is read at, which no bound establishes. `make
delta-p` measures it (`experiments/delta_p.sh`, D98): the denotation is not
computable, but the reference at twice the bits stands in for it, since
every rounding of a run shrinks by 2⁻ᵖ when p doubles, so |f_p − f_2p| ≈
δ_p, and the ratio δ_p / δ_2p, which should be near 2ᵖ, checks that it is
so. On the corpus GPT at `D = 64` on `make gpt-trained`'s weights and the
eight windows the calibrations use, each run printed at 330 digits and
compared exactly (`experiments/delta_p.py`), the reference at p against the
reference at 1024 bits:

| p | `loss`, relative to the loss | `GPT.apply`'s logits, ∞-norm relative to max\|y\| | the gradient of `loss`, 2-norm relative to ‖g‖ | δ_p / δ_2p | time at p (loss) |
|---|---|---|---|---|---|
| 128 | 2.9·10⁻³⁸ | 2.5·10⁻³⁸ | 3.0·10⁻³⁸ | 2¹²⁹, 2¹²⁸, 2¹²⁸ | 47 s |
| 256 | 4.0·10⁻⁷⁷ | 8.1·10⁻⁷⁷ | 8.8·10⁻⁷⁷ | 2²⁵⁶, 2²⁵⁶, 2²⁵⁶ | 60 s |
| 512 | 3.7·10⁻¹⁵⁴ | 6.4·10⁻¹⁵⁴ | 7.6·10⁻¹⁵⁴ | | 86 s |
| 1024 | — | — | (1000 bits) | | 133 s |

Five to ten units of roundoff at p bits (2⁻¹²⁸ is 2.9·10⁻³⁹), on an entry
of some 7·10⁸ elementary operations, a few thousand of them on the path to
any one output element: the roundings compensate as a random walk does, and
the sums' conditioning is benign (D51's `cond` rule). At 128 bits δ_p is 21 orders of magnitude below
the resolution of binary64 at which every gap of this chapter is read
(2⁻⁵³ ≈ 1.1·10⁻¹⁶: the reference's value is read as a binary64 number
before the difference is taken) and 22 below the smallest gap ever measured
(the `f64` loss, 1.3·10⁻¹⁵, D93; the `f64` gradient, 9.9·10⁻¹⁶). "δ_p
taken as 0" is a statement with a number behind it, on this model and these
inputs; whether the reference states it so, whether 128 bits suffice for
the calibrations, and whether a cheaper reference arithmetic should replace
the interpreter are the reference's W16, the user's to decide. The cost of
p is in the last column: 256 bits is 1.3× the time of 128 and 1.3× its
memory, 1024 bits 2.8× and 2.7×, the interpreter's time mostly its own
overhead and not the arithmetic. The gradient is measured against 1000
bits, not 1024, and the training step only checked at four digits
(`presk calibrate --step` prints no more and writes no parameter at full
precision); `docs/delta-p-2026-10-02.md` has every number, the commands
and the two defects the measurement found (D98).

## What is left open

0.6.0 set a budget beside MLPerf's accuracy rule on one model, ran BERT-large
at MLPerf's constants, and stated each result of this chapter as what it is
(the `coverage` chapter, the `experiments` chapter §5 and §6, D53). What its
measurements leave for later, and where each is recorded:

- **A reference that scales** (question 62). A budget is discharged against
  the exact denotation. The reference interpreter reaches BERT-large's
  inference at its own constants, two hours for one feature, but not a
  course-scale training step, where J′ was discharged against native `f64`,
  whose own distance to the exact step is measured only where both run.
  Since 0.10.1 it holds each value once (D105), and reaches a budget solved
  on one block of the Llama 2 decoder at the 7B width: the exact loss of
  384 tokens in 1 h 54 min and 29 GB, `presk solve --budget 1e-3` in 14 h
  and 72 GB, 52 of 56 sites narrowed, measured 4.3·10⁻⁶ against 1.1·10⁻²
  (D106, the `coverage` chapter; at 0.11.0, 24 of 25 stored values in the
  same time, measured 2.0·10⁻⁵, D117; at 0.12.0, 31 of 32 and the 13
  parameters in 13 h 32 min, measured 2.6·10⁻⁵, D124; at 0.16.0, the same 31,
  three of the 13 parameters in `bf16`, in 13 h 35 min, measured 2.5·10⁻⁵,
  the native gap, D162). The time is the next limit: the
  reference is one thread on MPFR, and the solve takes 7.4 times it.
- **The statistical model, to propose** (question 58). The root-sum-square
  propagation is within 3.2–17× of the measured gap on uniform realizations
  and 1.1·10⁴× high on a mixed one (`make precision` at 0.10.0; 1.7–17× and
  6.2·10³× before step 8's `bf16` unit roundoff, D93; 3.7–10× and 2.4·10³×
  on 0.6.0's weights, D51), because one magnitude per site is too
  coarse; finer magnitudes, per row or per channel, are named and not built,
  and the solver still proposes with sensitivities (D41, D51).
- **Results beyond the data measured.** Every budget met, every
  certification and the envelope itself hold on the windows measured, on
  one text, the Code civil, and the static bound only for inputs within the
  envelope's magnitudes (presk-project `docs/presk-data-dependence-2026-10-21.md`).
  A second corpus, an exact fallback when serving, and a bound over a region
  of inputs are recorded there and not built.
- **A bound that knows an exact value.** The measured rounding term charges
  nothing to a value exact in its type, but the static bound still charges
  it `u·|v|`. On the decoder, three gains of ones in `bf16` raise `cond`
  129 times (D162).
- **The state a training budget is stated at** (question 61), and the
  program's `precision` clause, which nothing reads yet: the budget is
  `presk solve`'s `--budget`.
- **Quantisation beyond weights**: activations (W8A8, with `i32`
  accumulation and requantisation), group scales for `int4`, and GPTQ or AWQ
  as values of `scheme` (D44).
- **Time inside a tight budget.** The solved training step is level with
  all-`f32`: on this instruction set `bf16` costs more in conversions than it
  saves in traffic wherever the profile looked, and `int8` weights buy time
  only where a kernel streams its weights from memory (D53,
  `docs/kernels-roadmap.md`).
- **Quality at scale.** The quality axis is flat after 600 steps of a small
  model; a longer run and a larger model are the next measurement (D50).
