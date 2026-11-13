# The corpus: models someone else chose

Until 0.4.0 every program this compiler ran was one the compiler's authors
had written: five examples from the specification, a tutorial, a tiny MiniGPT
at constants small enough to be comfortable. That is a bad way to find out
what a language cannot say and a worse way to find out what a code generator
cannot do, because the programs drift towards what already works.

`examples/corpus/nlp/` is the correction. It is the ladder of models of an NLP course
at Mines Paris, on that course's own data — the text and the vocabulary of
the French civil code — transcribed into Presk without permission to change
the models. Its purpose is to be a *forcing function*, not a demonstration.

## The ladder

| Rung | Model | What it forced |
|---|---|---|
| 1 | `01_bigram_count.psk`, the counting bigram in closed form | `one_hot`, and `presk run --data` so a function can be evaluated on a dataset |
| 2 | `02_bigram_neural.psk`, the same model fitted | nothing — it was expressible, which was worth knowing |
| 3 | `03_mlp.psk`, Bengio's embedding MLP | `merge`, joining two adjacent axes (question 45) |
| 4 | `04_wavenet.psk`, dilated convolutions | not written: convolution is a type-system extension, not a primitive (`coverage` chapter) |
| 5 | `05_gpt.psk`, the course's GPT | character tokenization, fusion, threads, blocked contractions — most of 0.4.0; then dropout at its three sites, which forced the per-layer key idiom (0.5.0 step 1) |

`data/prepare.py` fetches the course's files and derives the datasets; only
the 71 KB word list is committed, everything else is regenerated. `make
corpus` runs rungs 1 to 3 and `make corpus-gpt` the last.

## Where the numbers live

`examples/corpus/nlp/EXPECTED.md`, with the date of every measurement and the
configuration that produced it. That file is the corpus's point: a loss or a
time quoted in this manual can be traced to a command and a day. Two of its
entries are worth repeating here because they are what the corpus was for.

Rung 1 evaluates to **2.343771504** on all 67 652 character pairs, identical
on the reference interpreter and natively, and equal to the value NumPy
computes for the same estimator. Rung 2, which is the same function of the
same parameters fitted by gradient descent instead of solved, closes on it
from above. A compiler that gets those two to agree has shown something about
its differentiation and its arithmetic that no synthetic test does.

Rung 5 is the one that hurt: a training step took 289 s when it was first
written and takes 39.8 s now, and the losses never moved. Every number in the
`realize`, `native` and `optimization` chapters was measured on it.

Since 0.5.0 step 1 it also carries the course's dropout, at the three sites
the course has it: the attention weights, the attention output and the
feed-forward output. That is what forced the accumulator idiom of the
`language` chapter, because the blocks are a fold and a fold would otherwise
have handed all six of them one key. The first step's loss is unchanged to
every digit — the head is zero-initialized, so the logits are zero whatever
is dropped — while the gradient norm now moves with the run's seed, which is
how one can see from the outside that the masks are live. The step costs
45.1 s instead of 39.8 s, about 13 %, and `EXPECTED.md` keeps both figures,
because the 0.4.0 numbers are what this manual's other chapters were
measured against.

## Adding to it

Transcribe the model, do not adapt it. The value of the corpus is that its
authors did not know this compiler existed; a rung rewritten to fit the
language teaches nothing. When a rung will not check, that is the result —
record what is missing and what it would cost, as the `coverage` chapter does
for the MLPerf architectures, and leave the file with the gap marked. Rung 4
is left unwritten for exactly that reason.
