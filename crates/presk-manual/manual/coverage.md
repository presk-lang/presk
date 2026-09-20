# Coverage: what the shape fragment can say

The specification's roadmap asks for one measurement of the type system:
*coverage of the shape fragment on a corpus of models*. This chapter answers
it with MLCommons' MLPerf suites as the corpus, because they are the models a
systems paper is expected to know and because their architectures are public
and fixed. Nothing here is run — no MLPerf submission is a goal of this
project (`presk-specs/implementation/presk-plan-0.4.0.md` §3.7, decision
D27). The question is only whether the language can *say* each model, and
where exactly it stops.

The transcriptions are in `coverage/`; `make coverage` checks them.

## The table

| Family | MLPerf benchmarks | Status | What is missing |
|---|---|---|---|
| Transformer encoder | BERT-large (training and inference) | **expressible**, `coverage/bert.psk` | — |
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
today.

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
transcriptions here were written from the architectures, not from the
reference code, so they carry no weights and are not conformance-tested
against anything.

The honest summary is that the language covers the transformer, which is one
family of models described completely. Of the two things 0.4.0 named as next,
the axis split is done and bought more than expected — it replaced the head
primitives rather than complementing them. What is left is `concat`, small
and now the only thing between this language and an exact Llama, and
convolution, which is large and opens another four benchmarks.


## The foil: what a budget offers that an accuracy target does not

0.4.0 scoped MLPerf to three things (D27), and the third was the research
content: not a submission, but a **comparison of acceptance rules**. 0.5.0
has the machinery to make it, so here it is against BERT-large, which
`coverage/bert.psk` expresses whole.

MLCommons' closed division admits any realization whose *task accuracy* stays
within 99 % of an FP32 reference. Four properties follow from that rule, and
none of them is an accident of its thresholds.

- **The constant is negotiated, not derived.** 99 %, or 99.9 % for some
  benchmarks, is a number the working group agreed. Nothing computes it from
  the model.
- **Checking it costs a dataset pass.** The accuracy of a realization is
  known only after running the whole validation set through it.
- **It cannot attribute a failure.** A realization two points down might be a
  fragile model or a defective `fp16` accumulation in one fused kernel. The
  rule gives no way to tell, because there is no reference for the
  *computation*, only for the *task*.
- **It is per benchmark.** The rule says nothing transferable about a kernel,
  because it is a statement about a dataset.

A budget inverts each. The constant is **declared by the program** and the
compiler meets it or refuses. Checking it costs **one run against the exact
denotation**, not a dataset pass, and 0.5.0 does this in milliseconds on the
corpus GPT. A failure is **attributable**: a degradation beyond ε is a
compiler bug, provably, and below it is the model's own sensitivity. And it
is **per kernel**, so the number transfers.

The honest part of the comparison, and it is most of it. A budget does not
promise task accuracy and must not pretend to: the specification is explicit
that no analytical function relates a per-operation ε to perplexity. What it
guarantees is the second term of the split — quality(executed) =
quality(denotation) + effect of being within ε — and it is the second term
only. MLPerf's rule measures the thing people actually care about, which a
budget never will; a budget measures a thing MLPerf cannot see at all, and
measures it before the dataset runs.

So this is not a claim that the rule should be replaced. It is a claim that
the two are complementary and that only one of them is currently available,
because computing the second needs an exact reference and a compiler built
around it. That is the paper 0.4.0 said would need the precision solver, and
the solver now exists. What it still needs before it can be written is the
one thing steps 7 and 8 did not deliver: a realization that is both inside a
budget *and* faster, on a benchmark-scale model.
