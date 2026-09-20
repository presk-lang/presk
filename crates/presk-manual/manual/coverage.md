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
| Transformer decoder | GPT-3, GPT-J 6B, Llama 2 70B, Llama 3.1 405B | **expressible** with two substitutions, `coverage/llama.psk` | splitting one axis into two |
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
a zero-initialized head. `silu` is `x · σ(x)`, written from `sigmoid`. It
checks, with two substitutions marked in the file, and both are the same
missing construct:

- **Rotary positions.** RoPE rotates *pairs* of channels by an angle that
  depends on the position. Pairing them means reading `hd` as `(hd/2, 2)`,
  and the language has `merge`, which joins two axes, but not its inverse.
  The transcription keeps a learned position table in its place.
- **Grouped-query attention.** Llama 2 70B gives the keys and values eight
  heads shared by thirty-two query heads. Expressing the sharing means
  reading `heads` as `(groups, heads-per-group)` so the group axis can
  broadcast — the same split. The transcription uses plain multi-head
  attention.

The three ways one might work around a split are all closed, and say so:

```text
split(axis hd, half: HALF, pair: 2, x)   [E2001] unresolved function `split`
concat(a, b)                             [E2012] `concat` is not implemented
x[hd: 0..HALF]                           [E0010] expected `]`, found `.`
```

So the split is the cheapest thing on this list and the one that buys the
most: it is the declared inverse of `merge`, whose implementation already
exists inside the compiler as the adjoint of `merge` (`reshape_like`), and it
moves the decoder family from "with substitutions" to "exactly". It is the
open half of spec question 45.

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
family of models described completely, and that the next two things worth
having are the axis split, which is small and finishes that family, and
convolution, which is large and opens another four benchmarks.
