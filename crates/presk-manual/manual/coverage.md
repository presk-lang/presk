# Coverage: what the shape fragment can say

The specification's roadmap asks for one measurement of the type system:
*coverage of the shape fragment on a corpus of models*. This chapter answers
it with MLCommons' MLPerf suites as the corpus, because they are the models a
systems paper is expected to know and because their architectures are public
and fixed. The first question is only whether the language can *say* each
model, and where exactly it stops. Since 0.6.0 step 8 one of them is also
run: BERT-large, the one MLPerf benchmark within reach, at MLPerf's own
constants, for what that says about the compiler and about acceptance rules.
No MLPerf submission is a goal of this project
(`presk-specs/implementation/presk-plan-0.4.0.md` §3.7, decisions D27 and
D53).

The transcriptions are in `coverage/`; `make coverage` checks them and runs
BERT at tiny constants, `make bert` runs it at MLPerf's.

## The table

| Family | MLPerf benchmarks | Status | What is missing |
|---|---|---|---|
| Transformer encoder | BERT-large (training and inference) | **expressible**, `coverage/bert.psk`; inference **runs** at MLPerf's constants, natively and exactly | — |
| Transformer decoder | GPT-3, GPT-J 6B, Llama 2 70B, Llama 3.1 405B | **expressible** with one substitution, `coverage/llama.psk` | building a tensor along an axis (`concat`) |
| Mixture of experts | Mixtral 8×7B | not expressible | top-*k*; indexing a stacked parameter tree by a runtime value |
| Convolutional vision | ResNet-50, RetinaNet, 3D U-Net, Stable Diffusion | not expressible | convolution, pooling, batch and instance normalization; transposed convolution |
| Detection post-processing | RetinaNet | outside the language | non-maximum suppression is data-dependent control flow |
| Recommendation | DLRM-dcnv2 | not expressible | embedding bags over variable-length index lists |
| Graph | R-GAT | not expressible | reductions segmented by a runtime index |

Two of the seven families are expressible, and they are the transformer
families — which is where the suite itself has been moving.

## What the two transcriptions show

`coverage/bert.psk` is BERT-large as MLPerf fixes it: 24 post-normalized
blocks of width 1024, 16 heads, context 384, the three embedding tables
(token, position, segment) summed and normalized, an additive padding mask
that broadcasts by name across heads and queries, and the two-logit SQuAD
span head. It checks whole, with shapes, regularity classes and effects. An
MLPerf model, in its training and its inference form, is a Presk program
today — and, since 0.6.0 step 8, one that runs: see *BERT-large, run* below.

`coverage/llama.psk` is the decoder the Llama, GPT-J and Mixtral benchmarks
share: no biases, `RMSNorm`, SwiGLU with its own hidden width, a causal mask,
a zero-initialized head. `silu` is `x · σ(x)`, written from `sigmoid`. In
0.4.0 it checked with *two* substitutions, both attributed to the same
missing construct — splitting one axis into two. 0.5.0 step 1 added the
split. One substitution went away and the other did not, and the reason it
did not is the more useful of the two results.

**Grouped-query attention is now transcribed exactly.** Llama 2 70B gives the
keys and values eight heads shared by thirty-two query heads, so the queries
are read as `(groups: 8, per: 4, hd: 128)` and the keys and values as
`(groups: 8, hd: 128)`:

```text
let qg = split(axis qout,  groups: G, rest: P*HD, x · wq)
let q  = split(axis rest,  per: P,    hd: HD,     qg)
let k  = split(axis kvout, groups: G, hd: HD,     x · wk)
let s  = ein[batch, groups, per, seq, seq2](q * k) / sqrt(HD : ℝ)
```

Nothing transposes, and that is the point: `ein` contracts by *name*, so the
group axis of the queries meets the group axis of the keys wherever each sits
in its operand. Two things follow that the 0.4.0 chapter did not anticipate.
The split is not a companion to `split_heads` but its **replacement** for any
real head layout — `split_heads` is wired to `const H` and `const HD` and
also permutes, so it cannot say "eight shared by thirty-two" at all, and the
transcription now uses neither head primitive. And because a merge leaves its
axis unnamed (D28) while a split names both halves, **splits chain and merges
do not**, which is why the output projection reads `wo` under the same three
axes instead of flattening the attention result back.

**Rotary positions are still not expressible, and the split was not what they
were missing.** The pairing they need now checks:

```text
split(axis hd, rot: R, pair: 2, q)               -- ℝ[…, rot: R, pair: 2]
split(axis hd, rot: R, pair: 2, q)[pair: 0]      -- reading one member
```

The rotation reads both members of a pair and writes both back,
`out[…, r, 0] = x₀·cos θ − x₁·sin θ` and `out[…, r, 1] = x₀·sin θ + x₁·cos
θ`. Reading is `x[pair: 0]` and `x[pair: 1]`, which the language has. Writing
them back is building a tensor along an axis out of two computed parts, which
it does not:

```text
concat(a, b)          [E2012] `concat` is not implemented
x[hd: 0..HALF]        [E0010] expected `]`, found `.`
```

So what the decoder family is waiting for is not a shape construct at all but
a way to *make* a tensor from values, and the 0.4.0 chapter attributing both
substitutions to the split was wrong about one of them. That is the finding,
and it moves `concat` from a convenience to the last thing between this
language and an exact Llama.

## BERT-large, run

0.6.0 step 8 runs the transcription the way the one MLPerf benchmark within
reach runs it: inference, at MLPerf's constants — 24 blocks, width 1024, 16
heads, context 384, vocabulary 30 522 — on SQuAD-shaped features built as
MLPerf's harness builds them: `[CLS]`, the question, `[SEP]`, the context,
`[SEP]`, then padding to 384, which the additive mask sets to −10 000
(`experiments/bert_features.py`; the feature measured has 18 question tokens,
320 of context and 43 of padding). The file gained one entry, `squad`, which
takes a feature as one tuple so that a data file can supply it.

What it is not, before any number. The weights are the program's own
initialization, not the fine-tuned checkpoint MLPerf distributes, and the
token ids are random over WordPiece's range. There is no tokenizer, no SQuAD
and no F1 here. So nothing below is about answering questions, or about a
*trained* BERT, whose activations carry outlier features an initialization
does not have. What it measures is the compiler's reach, and what each
acceptance rule costs on the benchmark's own model.

**It runs, exactly and natively.** The reference interpreter computes the
exact logits of one feature at 128 bits in 1 h 57 min and 44 GB: the
denotation of an MLPerf model at its own constants, against which every
realization below is measured. That needed a change to the interpreter (D53). It
cloned a value at every use, a field of the parameter tree was a clone of the
whole tree, and BERT-large's 335M parameters would have been held about eight
times, some 180 GB. Tuples and records are now shared rather than copied, which
changes no output — four blocks run byte-identically before and after, and so
does every golden — and four blocks went from 40 GB to 13.5.

**The gaps**, over the 768 logits, `max|ŷ − y| / max|y|` — what a budget on
an entry binds (spec §6.1):

| realization | gap to the exact logits | inside 10⁻² |
|---|---|---|
| `f64` | 1.5·10⁻¹⁴ | yes |
| `f32` | 1.1·10⁻⁵ | yes |
| `bf16` | 1.1·10⁻¹ | no |
| `int8` weights, `f32` arithmetic | 5.7·10⁻² | no |
| `int4` weights, `f32` arithmetic | 2.0 | no |

The largest exact logit is 0.477, so `int4`'s error is twice the signal: its
logits say nothing. `f32` is the one narrowed realization inside 10⁻², by a
factor of nine hundred.

Split on two of the 24 blocks at full width and context (`presk quantize`: at
24, an exact run on the dequantised weights would cost two more hours per
scheme), relative to the largest exact logit:

| realization | the weights' rounding | the arithmetic | total |
|---|---|---|---|
| `int8` weights | 1.09·10⁻² | 9.6·10⁻⁷ | 1.09·10⁻² |
| `int4` weights | 2.17·10⁻¹ | 8.7·10⁻⁷ | 2.17·10⁻¹ |

On two blocks the weights' rounding is the whole of a quantised gap, and the
kernels add less than 10⁻⁶. The same `int8` scheme that statement K measured at
3.23·10⁻⁶ on the GPT's loss is 5.7·10⁻² here, outside a 10⁻² budget. A loss
averages its errors over 512 positions and ends in one number, where each of
768 logits is an output in its own right; and this model is sixteen times
wider.

**The time**, per sequence of 384 tokens, natively (`presk bench`, the median
of two runs, and for each type the better of its scalar and four-lane code):

| realization | batch 1, one core | batch 20, per sequence |
|---|---|---|
| `f64` | 80.6 s | 13.1 s |
| `f32` | 52.3 s | 6.92 s |
| `bf16` | 66.5 s | 5.28 s |
| `int8` weights, `f32` arithmetic | **23.4 s** | **3.10 s** |

`int8` weights are 2.2× faster than `f32` at both batch sizes: the first time
in this project that a type narrower than `f32` buys time on a model, and it
happens on the one MLPerf benchmark within reach. It happens because at this
width the contraction waits for memory. The blocked kernel reads the whole
weight matrix again for every row of the result, and a 16 MB matrix does not
stay in cache where its 4 MB `int8` copy does. `bf16`, which only halves the
bytes, is faster than `f32` at batch 20 too, and there `f32`'s four-lane code is
slower than its scalar code. That is a property of this kernel, which has no
packed panels or cache blocking, rather than of `int8`
(`docs/kernels-roadmap.md`). `int4` is level with `int8` (on two blocks at batch
20, 5.87 s against 5.62 s): unpacking costs what the halved bytes save. At
batch 1 a whole pass runs on one core, because threads split a kernel's
outermost loop, which is then the batch axis.

**What each rule costs on it.** MLPerf's accuracy rule is a pass over the
10 833 features per realization: at batch 20 on this machine, 9.3 h in `int8`,
20.8 h in `f32` and 39.5 h in `f64`. A budget's check is one native run of
the envelope's inputs per realization, here one feature: 2 to 3 min, most of
it drawing the initial parameters exactly. It is measured against an exact
reference computed once in 1 h 57 min, a tenth of one pass in `f32`.

## What the rest would cost

- **Convolution** is the large one, and it blocks four benchmarks at once.
  It is not another `ein`: a sliding window needs axes with a stride and a
  dilation, which the shape language has no word for, and its output length
  `(N + 2p − d(k−1) − 1)/s + 1` is an integer division whose divisor must
  stay literal for the decision procedure to keep working. It brings two
  adjoints with it, pooling, and normalizations over the batch. Weeks, and a
  genuine extension of the type system rather than a primitive.
- **Segmented reductions** — a sum or a softmax over groups named by a
  runtime index tensor — are what DLRM's embedding bags and R-GAT's
  attention both need. Underneath is the same thing: axes of variable
  length. The `.psk.d` container holds closed rectangular types only, so
  this reaches the data format as well as the language. The specification
  calls it the ragged extension and puts it in its risk register.
- **Top-*k* and a parameter gather.** A mixture of experts routes each token
  to two of eight expert weight matrices chosen at run time. `argmax` gives
  the top one; there is no top-*k*. Worse, selecting an expert means
  indexing a `stack{N}(M)` parameter tree by a *value*, where `slice_params`
  takes a compile-time constant. That the parameter tree is static is an
  assumption the whole lowering rests on, so this is a design question, not
  a primitive.
- **Non-maximum suppression** is not missing a primitive. It is a loop whose
  trip count depends on the values it reads, which is outside what a shape
  language of this kind describes at all. It belongs to the detection
  post-processing that a serving runtime does, not to the model.

## What this measures, and what it does not

It measures reach, not quality: a model that checks is one whose shapes,
regularity classes and effects the compiler proved consistent, which is worth
something and is not the same as being fast or being trained. The two
transcriptions were written from the architectures, not from the reference
code, so they carry no weights. BERT-large now runs, and a tiny instance of it
is in the native conformance suite, checked against the exact reference; it is
still not conformance-tested against MLPerf's reference implementation, which
would need that implementation's weights.

The honest summary is that the language covers the transformer, which is one
family of models described completely. Of the two things 0.4.0 named as next,
the axis split is done and bought more than expected — it replaced the head
primitives rather than complementing them. What is left is `concat`, small
and now the only thing between this language and an exact Llama, and
convolution, which is large and opens another four benchmarks.


## The foil: what a budget offers that an accuracy target does not

0.4.0 scoped MLPerf to three things (D27), and the third was the research
content: not a submission, but a **comparison of acceptance rules**. 0.5.0
argued it; 0.6.0 step 8 measures it (D53).

MLCommons' closed division admits any realization whose *task accuracy* stays
within 99 % — 99.9 % in the high-accuracy variant — of an FP32 reference's,
over the whole validation set: for BERT-large, F1 over the 10 833 features of
SQuAD v1.1, against the reference's 90.874. A budget admits a realization
whose *measured distance to the exact denotation*, on the envelope's inputs,
stays within the ε the program declares (spec §8.3). Both were applied to the
same realizations of one model: the corpus GPT at `D = 64`, trained on 90 % of
the Code civil and scored on the other tenth, 461 windows and 118 016
predictions (`make foil`; the whole table is in the `experiments` chapter,
§5).

| realization | gap to the exact loss | of which the weights' | correct predictions |
|---|---|---|---|
| `f32`, the FP32 reference | 2.2·10⁻⁷ | 0 | 18 615 |
| `bf16` | 3.0·10⁻³ | 0 | 18 614 |
| `int8` weights, `f32` arithmetic | 1.5·10⁻⁴ | 1.5·10⁻⁴ | 18 615 |
| `int4` weights, `f32` arithmetic | 1.0·10⁻³ | 1.0·10⁻³ | 18 608 |
| solved at 10⁻³, statement D's | 2.4·10⁻⁷ | 0 | 18 615 |

The accuracy rule's four properties, and a budget's answer to each, with the
numbers:

- **The constant is negotiated, not derived.** 99 % is a number a working
  group agreed; ε is declared by the program. On this model the accuracy rule
  admits every realization, at 99.9 % as at 99 % — seven predictions in
  118 016 is the largest change — so here it separates nothing. Nor is it
  monotone: `int4 pow2`, further from the exact loss than `int4`, loses one
  prediction where `int4` loses seven. The budget orders the same
  realizations over four orders of magnitude, and `f64` eight more below them.
  Where the accuracy rule does bind, its constant has in practice chosen a
  precision: for BERT, "FP16 was required to meet the highest accuracy
  target" ([NVIDIA, round 0.7](https://developer.nvidia.com/blog/winning-mlperf-inference-0-7-with-full-stack-approach/)),
  and FP8 later met it "where previously FP16 was required"
  ([round 2.1](https://developer.nvidia.com/blog/full-stack-innovation-fuels-highest-mlperf-inference-2-1-results-for-nvidia/)).
- **Checking it costs a dataset pass.** On the GPT a pass takes about 4 s per
  realization; on BERT-large, 9.3 to 20.8 h here. A budget is checked by one native
  run of the envelope's inputs per realization — 0.1 s on the GPT, 2 to 3 min
  on BERT-large, most of it drawing the initial parameters exactly — against
  an exact reference computed once: 7.5 s on the GPT, 1 h 57 min on
  BERT-large. Cheaper per realization and dearer once: the
  reference is the price, and on BERT-large it costs a tenth of one pass in
  `f32`.
- **It cannot attribute a failure.** A realization two points down might be a
  fragile model or a defective kernel, and the rule cannot tell, because it has
  a reference *score* and no reference *computation*. A budget's reference is
  the computation. The split attributes a quantised gap to the weights'
  rounding or to the arithmetic — on the GPT the weights' part is 10² to 10⁵
  times the kernels' — and an isolated measurement attributes a training
  step's gap per function (D52: `SelfAttention.apply.vjp` carries nearly all
  of the `f32` step's). By measurement on the envelope's inputs, which is not
  a proof and says nothing beyond them (question 53).
- **It is per benchmark.** The rule is a statement about one dataset. A
  budget is portable and a realization is not (question 59): what carries to
  another checkpoint or scale is the realization, keyed by function, and its
  gap is measured again there. Statement D's realization, solved on one
  checkpoint, is 2.4·10⁻⁷ from the exact loss on the held-out one, against
  4.48·10⁻⁶ where it was solved; J′'s, solved at `D = 64`, is inside its budget
  at course scale too. The number never transfers; the measurement does.

**What holds, each result as what it is.**

| | measured | what it is | what it is not |
|---|---|---|---|
| D | 4.48·10⁻⁶ against 10⁻³ (D50) | a budget declared on an entry, discharged by measurement against the exact denotation, with 2.12× less traffic | independent evidence of the solver, which widens until the measurement passes; a speedup (J: 1.06×, then 1.13×) |
| K | `int8` weights, 3.23·10⁻⁶ against 10⁻²; 1.46·10⁻⁴ on held-out windows | inside a budget on a *loss*, a scalar averaged over 512 positions | a speed claim (the GPT forward is level with `f32`); a claim about outputs: on BERT-large's logits the same scheme is 5.7·10⁻², outside 10⁻² |
| the split | 3.86·10⁻⁶ of weights, 6.32·10⁻⁷ of arithmetic (K's windows) | a quantised gap attributed to the weights or to the kernels | exclusive to an exact reference: `f64` would approximate it wherever its own gap is known to be small, and knowing that is the exact reference's work |
| J′ | 1.99× per training step against all-`f64`, inside its 10⁻³ update budget (D52) | the budget licensing the move off `f64` by measurement, and the time objective refusing `bf16` where it does not pay | faster than FP32 — all-`f32` runs at 31.0 s a step against the solved step's 30.0 s, inside the run-to-run noise, and FP32 is MLPerf's reference; an inference result |

**On the benchmark's own model.** `int8` weights are the first realization in this project
faster than FP32 on a model — 2.2× on BERT-large at MLPerf's constants — and
they are 5.7·10⁻² from the exact logits, almost all of it the weights'
rounding: inside a budget of 10⁻¹, outside 10⁻². `bf16` is 1.3× faster at
batch 20 and outside both. So this version's aim, a realization both inside a
budget and faster than the reference, is met on the benchmark's own model at
one ε and not at the next. The budget does not say which ε the task needs:
that is the accuracy rule's question, and answering it here would take a
trained BERT, SQuAD and nine to twenty-one hours per realization. What it does
is price each speedup in computational error, say where the error comes from,
and do so from one feature and one exact run, before any dataset pass.

**The honest part of the comparison, and it is most of it.** A budget does not
promise task accuracy and must not pretend to: the specification is explicit
that no analytical function relates a per-operation ε to perplexity. What it
guarantees is the second term of the split — quality(executed) =
quality(denotation) + effect of being within ε — and the second term only.
MLPerf's rule measures the thing people actually care about, which a budget
never will; a budget measures a thing MLPerf cannot see at all, and measures
it before the dataset runs. The GPT's table shows both sides of that: the
accuracy rule is blind to four orders of magnitude of computational error that
the task did not notice, and a budget of 10⁻⁴ would refuse `bf16` and every
quantised realization for errors the task did not notice either.

So this is not a claim that the rule should be replaced. It is a claim that
the two are complementary, and that only one of them is available elsewhere,
because the second needs an exact reference and a compiler built around it.
What the comparison still lacks is the accuracy side on the benchmark itself:
a trained BERT-large and SQuAD, so that the two rules can be read side by side
where MLPerf reads its own. That is the evaluation half of a submission, and
not a goal (D27).
