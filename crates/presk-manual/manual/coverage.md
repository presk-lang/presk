# Coverage: what the shape fragment can say

The specification's roadmap asks for one measurement of the type system:
*coverage of the shape fragment on a corpus of models*. This chapter answers
it with MLCommons' MLPerf suites as the corpus, because they are the models a
systems reader is expected to know and because their architectures are public
and fixed. The first question is only whether the language can *say* each
model, and where exactly it stops. Since 0.6.0 step 8 one of them is also
run: BERT-large, the one MLPerf benchmark within reach, at MLPerf's own
constants, for what that says about the compiler and about acceptance rules.
Since D104 the other runs too: the Llama 2 decoder at the 7B configuration's
width, one of its blocks, on which D106 solves a budget. No MLPerf submission is a goal of this project
(`presk-specs/implementation/presk-plan-0.4.0.md` §3.7, decisions D27, D53
and D104).

The transcriptions are in `examples/coverage/`; `make coverage` checks them and runs
both at tiny constants, `make bert` runs BERT at MLPerf's constants and
`make llama` the decoder at Llama 2 7B's width.

## The table

| Family | MLPerf benchmarks | Status | What is missing |
|---|---|---|---|
| Transformer encoder | BERT-large (training and inference) | **expressible**, `examples/coverage/bert.psk`; inference **runs** at MLPerf's constants, natively and exactly | — |
| Transformer decoder | GPT-3, GPT-J 6B, Llama 2 70B, Llama 3.1 405B | **expressible** exactly, rotary positions included, `examples/coverage/llama.psk` (since 0.17.0 step 6, D179); one block **runs** at Llama 2 7B's width, natively and exactly (D104, with a learned position table then), and a budget is **solved** on it (D106) | — |
| Mixture of experts | Mixtral 8×7B | not expressible | top-*k*; indexing a stacked parameter tree by a runtime value |
| Convolutional vision | ResNet-50, RetinaNet, 3D U-Net, Stable Diffusion | not expressible | convolution, pooling, batch and instance normalization; transposed convolution |
| Detection post-processing | RetinaNet | outside the language | non-maximum suppression is data-dependent control flow |
| Recommendation | DLRM-dcnv2 | not expressible | embedding bags over variable-length index lists |
| Graph | R-GAT | not expressible | reductions segmented by a runtime index |

Two of the seven families are expressible, and they are the transformer
families — which is where the suite itself has been moving.

## What the two transcriptions show

`examples/coverage/bert.psk` is BERT-large as MLPerf fixes it: 24 post-normalized
blocks of width 1024, 16 heads, context 384, the three embedding tables
(token, position, segment) summed and normalized, an additive padding mask
that broadcasts by name across heads and queries, and the two-logit SQuAD
span head. It checks whole, with shapes, regularity classes and effects. An
MLPerf model, in its training and its inference form, is a Presk program
today — and, since 0.6.0 step 8, one that runs: see *BERT-large, run* below.

`examples/coverage/llama.psk` is the decoder the Llama, GPT-J and Mixtral benchmarks
share: no biases, `RMSNorm`, SwiGLU with its own hidden width, a causal mask,
and a head drawn as the other matrices (zero-initialized until D104, see
*The decoder, run*). `silu` is `x · σ(x)`, written from `sigmoid`. In
0.4.0 it checked with *two* substitutions, both attributed to the same
missing construct — splitting one axis into two. 0.5.0 step 1 added the
split. One substitution went away and the other did not, and the reason it
did not is the more useful of the two results. Since 0.17.0 step 6 it checks
with none: the second waited for a slice that keeps its axis and for
`concat` (D179), below.

**Grouped-query attention is now transcribed exactly.** Llama 2 70B gives the
keys and values eight heads shared by thirty-two query heads, so the queries
are read as `(groups: 8, per: 4, hd: 128)` and the keys and values as
`(groups: 8, hd: 128)`:

<!-- psk: none (lines of a body: the attention as 0.5.0 transcribed it, before examples/coverage/llama.psk took RoPE) -->
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

**Rotary positions, since 0.17.0.** Until then they were not expressible,
and the split was not what they were missing. The pairing they need checked
since 0.5.0, `split(axis hd, pair: 2, rot: HD/2, q)`, and so did reading one
member, `q[pair: 0]`. But the rotation reads both members of a pair and
writes both back, `out₀ = x₀·cos θ − x₁·sin θ` and `out₁ = x₁·cos θ + x₀·sin
θ`, and writing them back is building a tensor along an axis out of two
computed parts: `concat` was refused (`E2012`), and `q[pair: 0]` drops its
axis, so there was nothing to concatenate along. The 0.4.0 chapter
attributing both substitutions to the split was wrong about this one, and
the finding moved `concat` from a convenience to the last thing between the
language and an exact Llama. Plan 0.17.0 step 6 brings both constructs, and
the transcription takes RoPE in Hugging Face's pairing (dimension `r` of a
head with `r + HD/2`) and drops its learned table:

<!-- psk: examples/coverage/llama.psk#rope, same tokens -->
```text
fn rope(φ : ℝ[seq: T, rot: HD/2], x : ℝ[..., seq: T, hd: HD]) -> ℝ[..., seq: T, hd: HD] =
  let p  = split(axis hd, pair: 2, rot: HD/2, x)
  let x0 = p[pair: 0..1]                     -- a slice: the axis `pair` kept, at length 1
  let x1 = p[pair: 1..2]
  let c  = cos(φ)                            -- ℝ[seq: T, rot: HD/2], broadcast by name
  let s  = sin(φ)
  merge(axis pair, axis rot, concat(axis pair, x0 * c - x1 * s, x1 * c + x0 * s))
```

The angles are `ein[seq, rot](real(positions(seq: T)) * frequencies())`, an
outer product written as an `ein` that sums nothing. `rope` wants `seq` just
before `hd`, so the queries and keys are projected by `ein` with the heads
before the positions (`ein[batch, groups, per, seq, hd](x * wq3)`), and still
nothing transposes. The specification's example `examples/spec/rope/` is the same
rotation with Llama 3's frequency scaling, written with `clamp`.

## The decoder, run

After the tag `v0.10.0`, the second transcription runs too (D104), by BERT's
protocol: `make llama` (`experiments/llama.sh`). It is `examples/coverage/llama.psk`
with `G = 32`, every query head its own key head, so that the width (4 096),
the heads (32 of 128), SwiGLU's width (11 008) and the vocabulary (32 000)
are Llama 2 7B's published configuration; at its 32 blocks the file counts
6 738 415 616 parameters besides its position table, the 7B's count exactly.
What it is: the Llama 2 decoder at the 7B configuration's width, **one** of
its 32 blocks, a learned position table in place of RoPE (the file before
0.17.0; since then it has RoPE and no table, and `make llama` runs that), on its
initialization and one sequence of 384 random token ids
(`experiments/llama_ids.py`). What it is not: MLPerf's constants (MLPerf's
Llama 2 is the 70B), a trained Llama, text, or a perplexity. For the run the
file gained a scalar entry, `loss`, the mean next-token cross-entropy, and
its head is drawn as the other matrices: it was `zeros`, which makes every
logit zero, so that no gap relative to the largest logit, and no ranking of
sites by their sensitivity, could be read.

**What fits.** At the tag, the exact interpreter held every parameter at
128 bits twice: D53 shared trees, not the tensors in them. One block of this
width with its two tables of 131 M entries is 466 M parameters, 85 GB on 16
tokens; two blocks were killed at 100 GB, drawing their parameters. `presk
solve`, which adds a reverse sweep, was killed at 108 GB on one block and 16
tokens, and `presk quantize`, which holds the parameters a third time,
dequantised, at one block too. So D104's exact logits, gaps and time are on
one block, its split on **no** block (`stack{0}`: the embeddings, the final
norm and the head, 264 M parameters), as BERT's ran on two of its 24, and it
solved no budget. Version 0.10.1 holds each parameter once and frees a value
after its last use (D105, the `language` chapter): on 16 tokens, one block's
exact logits take 29.0 GB, two blocks' 41.1, the solve on one block 71.2 and
the split on one block 61.1. The solve on two blocks still does not fit.

**It runs, exactly and natively.** The exact logits of the sequence, 384 ×
32 000 of them, take 1 h 55 min and 87.6 GB at the tag; their first 16
positions are, to every printed digit, those of the 16-token run, as a
causal model's must be. The gaps, over the 12 288 000 logits (the largest
exact one is 6.88):

| realization | gap to the exact logits | inside 10⁻² |
|---|---|---|
| `f64` | 1.3·10⁻¹⁴ | yes |
| `f32` | 6.2·10⁻⁶ | yes |
| `bf16` | 2.5·10⁻² | no |
| `int8` weights but `wo`, `f32` arithmetic | 4.9·10⁻² | no |
| `int4` weights but `wo`, `f32` arithmetic | 7.1·10⁻¹ | no |
| `int8` weights, every matrix | 5.3·10⁻² | no |
| `int4` weights, every matrix | 7.6·10⁻¹ | no |

D104's native rows leave `wo` in `f32` because of a defect of the native
backend at the tag, fixed in 0.10.1 (D105, the `native` chapter): a
quantised parameter that the program reads through `split` — here `wo`,
read as `(groups, per, hd)` — was computed wrongly, with values that changed
from run to run, where the kernel interpreter was right. D104 measured the
whole of K's scheme, all ten matrices, on the kernel interpreter; since
0.10.1 the native backend computes it, in under two minutes and 36.5 GB, to
the same gaps (5.2730·10⁻² and 0.75646) and within 1.9·10⁻⁵ of the kernel
interpreter's logits, the two backends' `libm`s (D106).

The split (`presk quantize`, relative to the largest exact logit there): on
no block (D104, 6.92), where there is no `wo` to leave out, and since 0.10.1
on one block, the whole scheme (D106, 6.88):

| realization | the weights' rounding | the arithmetic | total |
|---|---|---|---|
| `int8` weights, no block | 1.18·10⁻² | 3.3·10⁻⁶ | 1.18·10⁻² |
| `int4` weights, no block | 2.25·10⁻¹ | 2.9·10⁻⁶ | 2.25·10⁻¹ |
| `int8` weights, every matrix, one block | 5.27·10⁻² | 6.4·10⁻⁶ | 5.27·10⁻² |
| `int4` weights, every matrix, one block | 7.56·10⁻¹ | 6.1·10⁻⁶ | 7.56·10⁻¹ |

As on BERT, the weights' rounding is the whole of a quantised gap, and the
kernels add a few 10⁻⁶. The block's seven matrices multiply the rounding's
effect on the logits by 4.5 in `int8`, to five times BERT's at two blocks.
At one block the split took about four hours a scheme and 63 GB, for a storage
4.00× (`int8`) and 7.99× (`int4`) smaller than `f32`'s.

**The time** per sequence of 384 tokens, on one block (`presk bench`, the
median of two runs, the better of the scalar and four-lane code; the last
row from D106's run, where `f32` took 80.9 s and 9.40 s):

| realization | batch 1, one core | batch 20, per sequence |
|---|---|---|
| `f64` | 122.0 s | 14.5 s |
| `f32` | 83.0 s | 8.48 s |
| `bf16` | 86.8 s | 6.79 s |
| `int8` weights but `wo`, `f32` arithmetic | **52.2 s** | **4.56 s** |
| `int8` weights, every matrix, `f32` arithmetic | **47.0 s** | **4.02 s** |

`int8` weights but `wo` are 1.6× faster than `f32` at batch 1 and 1.9× at
batch 20; on every matrix, within one run, 1.72× and 2.34×, for BERT-large's
2.2×. One block is not the model: its head and token table are 56 % of these
parameters and 4 % of the 7B's.

**Against PyTorch eager** (D107, `make pytorch-decoder`,
`docs/pytorch-decoder-0.10.1.md`). The same block's forward pass by the
PyTorch baseline's protocol (the `experiments` chapter, §3 and §10): the
ten X925 cores pinned, ten threads, the median of seven calls, the native
backend under its default layouts and under `presk plan`'s plan with the
frozen profile, beside PyTorch eager on the module `presk export --target
pytorch` writes, with its own draw of the same distributions:

| side | B = 1, ms a call | ÷ PyTorch | B = 20, ms a call | ÷ PyTorch |
|---|---|---|---|---|
| native `f32`, default layouts | 95 806.6 | 348 | 214 063.7 | 47.1 |
| native `f32`, plan | 3 744.1 | 13.6 | 65 195.9 | 14.4 |
| native, `int8` every matrix, plan | 1 870.6 | 6.80 | 37 602.9 | 8.28 |
| PyTorch eager | 275.1 | 1.00 | 4 541.2 | 1.00 |

The plan is worth 25.6× at batch 1, where the default layouts divide the
batch axis and so run on one core, and 3.3× at batch 20. What is left is
mostly one nest: the attention's output projection, `ein[batch, seq,
model]` over `groups`, `per` and `hd`, takes two thirds of the kernels'
time at 7.5–8.8 GFLOP/s, because the contraction builder tiles an `ein` with one
summed axis only; the tiled products run at about 400 GFLOP/s. The course
GPT's forward pass under its plan was 2.32× PyTorch's (D100).

**A budget, solved** (D106). `presk solve --budget 1e-3` on the next-token
loss of the sequence, at one block, with the envelope of that sequence, as
`make compile-times` times it (`STAGES=solve` of `experiments/llama.sh`):

| command | wall | peak |
|---|---|---|
| `presk calibrate --precision`, one sequence | 8 h 23 min | 32.0 GB |
| `presk solve --budget 1e-3` | **14 h 07 min** | **71.7 GB** |
| the exact reference, `presk run --bits 128` on `loss` | 1 h 54 min | 29.3 GB |

The loss is 11.18. Of 56 sites that can be narrowed, 52 are: 50 to `f32`
and two scalar constants to `bf16` (the head width, exact in it, and the
norm's ε). The traffic is 1.99× less than all-`f64`'s, and the realization
measures **4.3·10⁻⁶** against the exact loss, where the budget allows
1.1·10⁻², with no widening round. BERT-large's two blocks measured 5.1·10⁻⁷
at 2.00× (D101; 1.05·10⁻⁵ at 2.03× at 0.11.0, D117; 9.2·10⁻⁷ at 2.00×,
the weights counted, at 0.12.0, D124; 1.0·10⁻⁵ at 2.01× at 0.16.0, D166,
its three LayerNorm gains in `bf16`, ones at its initialization), the
course GPT 2.1·10⁻⁵ at 2.00× (D102, D117; 2.1·10⁻⁶ at 0.12.0, D124). The solve takes
7.4 times the reference, calibrate 4.4 times. The bound on that realization
is 6.2·10⁵, vacuous: the discharge is the measurement's (the `precision`
chapter).

**At 0.11.0** (D117, `docs/compile-times-2026-10-09-llama.md`) the solver
offers the values the program stores and nothing else (D112), and the same
four commands were run again: the envelope and the exact loss are
D106's, the times within 3.5 % and the peaks the same. Of 25 stored
values, 24 are narrowed, 21 to `f32` and 3 to `bf16`; the traffic, now
the stored values', is 1.96× less than all-`f64`'s, and the realization
measures **2.0·10⁻⁵** against the same 1.1·10⁻², with no widening round.
Three stored values have another type than in D106: the softmax and
RMSNorm's mean, `f64` there, are `f32`, and the head width's square root
is `bf16`; the bound is 1.2·10⁶. The 31 sites D106 counted besides change
no bit when typed: D106's realization, read under 0.11.0's roles, narrows
22 of the 25 at 1.84×. A function's result is not offered, so the head,
the output projection and the feed-forward products stay in `f64`.

**At 0.12.0** (D124, `docs/compile-times-2026-10-14-llama.md`) the solver
also offers the functions' results and the parameters (D123), and the
traffic counts the weights at rest. The same four commands, run again:
- *the same envelope and loss,* D106's, the times within 4 % of D117's and
  the peaks the same, the solve 13 h 32 min and 71.7 GB;
- *32 stored values offered, 31 narrowed,* 28 to `f32` and 3 to `bf16`.
  The seven added are results, all in `f32`: the head, the output
  projection, the feed-forward product, SiLU's product, the norm's
  output, the block's second residual and the loss. RMSNorm's square
  stays `f64`;
- *the 13 parameters in `f32`,* 1.74 GB at rest against 3.47 GB in `f64`;
- *1.99× less traffic,* the weights counted (1.97× on the stored values
  alone), measured **2.6·10⁻⁵** against the same 1.1·10⁻², with no
  widening round. The bound is 1.2·10⁶.

Natively, the block's forward pass under that realization takes 2.4 to
2.9 s at batch 1, as fast as all-`f32`, where under 0.11.0's it took 9.7 s,
its results in `f64` and every weight converted at the kernels' edges
(D124, the `experiments` chapter).

**At 0.16.0** (D162, `docs/compile-times-2026-10-19-llama.md`) the solver
charges each site the rounding it measures on the values (D155, the
default since D158), and it measures on the native backend's bits (D159).
The same four commands, run again:
- *the same envelope and loss,* the times within 1 % of D124's and the
  peaks the same, the solve 13 h 35 min and 71.7 GB;
- *the same 31 of 32 stored values,* each in D124's type;
- *10 parameters in `f32` and 3 in `bf16`.* The three are RMSNorm's gains.
  They are ones at initialization, which `bf16` holds exactly, so the
  measured term charges them nothing;
- *1.99× less traffic,* measured **2.5·10⁻⁵**, the native gap, which
  D124's realization has too (its 2.6·10⁻⁵ was the kernel interpreter's
  `exp`). The bound is 1.5·10⁸: it charges the three gains `bf16`'s unit
  roundoff whatever their values.

**Two blocks** (D106). The exact logits at two blocks take 3 h 36 min and
41.4 GB (the largest is 7.21), and the native gaps move little:

| realization | one block | two blocks |
|---|---|---|
| `f64` | 1.3·10⁻¹⁴ | 1.6·10⁻¹⁴ |
| `f32` | 6.2·10⁻⁶ | 7.8·10⁻⁶ |
| `bf16` | 2.5·10⁻² | 3.4·10⁻² |
| `int8` weights but `wo` | 4.9·10⁻² | 5.6·10⁻² |
| `int4` weights but `wo` | 7.1·10⁻¹ | 9.4·10⁻¹ |
| `int8` weights, every matrix | 5.3·10⁻² | 5.7·10⁻² |
| `int4` weights, every matrix | 7.6·10⁻¹ | 9.8·10⁻¹ |

So the second architecture sits where the first does. `f32` is the one
narrowed type inside 10⁻², `bf16` and `int8` weights are a few 10⁻² away,
`int4` weights' logits say nothing, and the quantised gap is the weights'
rounding. K's scheme is outside 10⁻² on this decoder's logits as on BERT's,
and `int8` weights buy time on both. And a budget of 10⁻³ is solved and
discharged on a model the project did not write, at a published
configuration's width, one block of 32: nearly every site narrowed, the
traffic halved, the measurement three orders inside the budget.

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
would need that implementation's weights. The decoder runs one block at a
time at Llama 2 7B's width, and nothing of it is checked against Meta's code
or weights.

The honest summary is that the language covers the transformer, which is one
family of models described completely. Of the two things 0.4.0 named as next,
the axis split is done and bought more than expected — it replaced the head
primitives rather than complementing them. `concat` and the slice came in
0.17.0 step 6, and the decoder is exact since. What is left is convolution,
which is large and opens another four benchmarks.


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
| `f32`, the FP32 reference | 4.7·10⁻⁸ | 0 | 28 612 |
| `bf16` | 1.7·10⁻³ | 0 | 28 610 |
| `int8` weights, `f32` arithmetic | 3.9·10⁻⁵ | 3.9·10⁻⁵ | 28 618 |
| `int4` weights, `f32` arithmetic | 4.0·10⁻⁴ | 4.0·10⁻⁴ | 28 476 |
| solved at 10⁻³, statement D's | 2.5·10⁻⁷ | 0 | 28 612 |

(D102's measurement, at the tag `v0.10.0`, on a checkpoint trained with the
right gradient; D93's had statement D's realization at 4.7·10⁻⁸ and every
other row the same. D53's,
on one trained with the gradient defect 3f of D92 made wrong, had the FP32
reference right on 18 615 predictions and every realization within seven of
it; the rules' behaviour below is on the new one.)

The accuracy rule's four properties, and a budget's answer to each, with the
numbers:

- **The constant is negotiated, not derived.** 99 % is a number a working
  group agreed; ε is declared by the program. On this model the accuracy rule
  admits every realization at 99 %, and at 99.9 % every one but `int4`
  weights, which lose 136 predictions in 118 016 (0.48 % of the reference's)
  with a gap of 4.0·10⁻⁴ to the exact loss, inside a budget of 10⁻³. Nor is
  it monotone: `int4 pow2`, twice as far from the exact loss as `int4`, gains
  115 predictions where `int4` loses 136. The budget orders the same
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
  checkpoint, is 2.5·10⁻⁷ from the exact loss on the held-out one, against
  2.1·10⁻⁵ where it was solved (D102, at the tag; 4.7·10⁻⁸ and 1.89·10⁻⁶ in
  D93; 2.4·10⁻⁷ and 4.48·10⁻⁶ in D53); J′'s,
  solved at `D = 64`, is inside its budget at course scale too. The number never transfers; the measurement does.

**What holds, each result as what it is.**

| | measured | what it is | what it is not |
|---|---|---|---|
| D | 2.1·10⁻⁵ against 10⁻³ (D102, at the tag; 1.89·10⁻⁶ in D93, 4.48·10⁻⁶ in D50) | a budget declared on an entry, discharged by measurement against the exact denotation, with 2.00× less traffic | independent evidence of the solver, which widens until the measurement passes; a speedup (J: 1.06×, then 1.13×) |
| K | `int8` weights, 2.73·10⁻⁵ against 10⁻²; 3.90·10⁻⁵ on held-out windows (D93; 3.23·10⁻⁶ and 1.46·10⁻⁴ in D53) | inside a budget on a *loss*, a scalar averaged over 512 positions | a speed claim (the GPT forward is level with `f32`); a claim about outputs: on BERT-large's logits the same scheme is 5.7·10⁻², outside 10⁻² |
| the split | 2.77·10⁻⁵ of weights, 3.73·10⁻⁷ of arithmetic (K's windows, D93) | a quantised gap attributed to the weights or to the kernels | exclusive to an exact reference: `f64` would approximate it wherever its own gap is known to be small, and knowing that is the exact reference's work |
| J′ | 1.99× per training step against all-`f64`, inside its 10⁻³ update budget (D52); 1.95× measured again on the corrected gradient (D93); 1.75× at the tag `v0.10.0` (D103) | the budget licensing the move off `f64` by measurement, and the time objective refusing `bf16` where it does not pay | faster than FP32 — all-`f32` runs at 14.0 s a step against the solved step's 13.9 s at the tag (D103; 23.4 and 22.7 in D93, 31.0 and 30.0 in D52), inside the run-to-run noise, and FP32 is MLPerf's reference; an inference result |

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
