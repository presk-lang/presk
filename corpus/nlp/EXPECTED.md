# The NLP corpus: what each rung does, and the numbers it produced

The programs follow the ladder of the NLP course at Mines Paris
(`mines.paris/nlp`), on the course's own data: the words of the Code civil
(`data/civil_mots.txt`, 7223 words over 40 characters) and the text of the
Code civil (`data/civil.md`, 1.18 MB, 91 characters). `data/prepare.py`
derives the datasets; `make corpus` runs everything below.

The boundary token `.` is index 0, so the vocabulary is `V = 41`.

| Rung | Program | Status |
|---|---|---|
| 1 | `01_bigram_count.psk` — the counting bigram in closed form | done (0.4.0 step 1) |
| 2 | `02_bigram_neural.psk` — the same model by gradient descent | done (0.4.0 step 1) |
| 3 | `03_mlp.psk` — Bengio: a context of three characters | done (0.4.0 step 1) |
| 4 | `04_wavenet.psk` — dilated convolutions | needs a convolution primitive |
| 5 | `05_gpt.psk` — the course GPT on the Code civil | checks at course scale; trains only reduced (0.4.0 steps 4–6) |

## Measured on 2026-09-20 (Apple silicon, `presk` at 0.4.0-dev)

**Rung 1 — the counting bigram.** `loss` is the average negative
log-likelihood of the maximum-likelihood bigram model over all 67652 pairs,
computed as one `ein` over two `one_hot` tensors and read back with `embed`:

```sh
presk run corpus/nlp/01_bigram_count.psk --entry loss \
    --data corpus/nlp/data/bigrams.psk.d --backend native --digits 10
```

| Executor | Loss |
|---|---|
| reference interpreter, 128 bits | 2.343771504 |
| native `f64` | 2.343771504 |
| NumPy reference (`data/prepare.py` docstring) | 2.343772 |

**Rung 2 — the neural bigram.** The same function of the same parameters,
fitted by AdamW (lr 0.5, cosine warm-up 50, 30 epochs of batches of 512,
3960 steps) in 3.6 s on the native backend:

| Step | Batch loss |
|---|---|
| 1000 | 2.484712 |
| 2000 | 2.344189 |
| 3000 | 2.335378 |
| 3960 | 2.323520 |

The batch losses straddle the closed-form 2.343772 of rung 1, which is the
course's own sanity check: gradient descent on the neural bigram recovers the
counting bigram, because they are the same function.

**Rung 3 — the embedding MLP.** Three characters of context, each embedded
into `ℝ[emb: 10]`; the three embeddings are joined into one vector of 30 by
`merge(axis ctx, axis emb, ·)`, then a hidden layer of 200 and the head. 2112
AdamW steps (lr 0.05, cosine warm-up 100, 8 epochs of batches of 256) in 35 s
on the native backend:

| Step | Batch loss |
|---|---|
| 400 | 2.121748 |
| 1200 | 1.894083 |
| 2000 | 1.838011 |
| 2112 | 1.733731 |

Well below the 2.343772 of rungs 1 and 2, which is the point of the rung:
three characters of context predict better than one. The gradient through
`merge` is checked against central finite differences at 512 bits
(`tests/data/merge_grad.psk`), and the PyTorch export of the same model
passes its conformance test.

**Rung 5 — the course GPT.** `make corpus-gpt` turns the Code civil into a
character-level dataset and trains the model:

```text
presk data: wrote corpus/nlp/data/civil_vocab.json (92 characters over 1182121 of text)
presk data: wrote corpus/nlp/data/civil_tokens.txt (1182121 tokens over a vocabulary of 92)
presk data: wrote corpus/nlp/data/civil_T256.psk.d (4617 windows of type (idx[92][seq: 256], idx[92][seq: 256]) from 1182121 tokens)
```

`presk check` passes at the course's own constants — `D = 384`, `H = 6`,
`L = 6`, `T = 256`, the data binding `T` — so the language holds the model.
What does not hold is the speed. On the native backend in `f32`, batches of
16:

| Configuration | Time per step | Peak resident memory |
|---|---|---|
| `D = 64, H = 2, L = 2` | 4.2 s → **0.83 s** | 344 MB (3 steps), 336 MB (12 steps) |
| `D = 384, H = 6, L = 6` (the course's) | 289 s → **39.8 s** | 1.87 GB |

The second figure of each row is after the fusion, threads and blocked
contractions of 0.4.0 steps 4 to 6 (twelve cores, `f32`); the losses are
unchanged to the last digit, since none of the three changes the order of an
arithmetic operation.

Memory is the same after three steps and after twelve: the reader seeks to
each sample as the batch asks for it, so what is resident is the model and
its batch, never the dataset. The streaming the plan asked for at this step
was already in the container's design; only overlapping windows
(`--stride 1`) would need windows generated lazily.

The time is the point. At course scale a step is about 260 GFLOP, so 289 s
is roughly **0.9 GFLOP/s**, the same order the 0.3.0 assessment measured on
the tiny MiniGPT. One epoch is 288 steps, which is 23 hours; the course
trains for thousands of steps. That is the wall steps 4 to 6 of the plan
exist to break — fusion, vectorization, threads and a blocked GEMM — and
these two rows are what they will be measured against, together with the
PyTorch comparison of step 8, which has not been run yet.

**Against PyTorch.** One training step of rung 5 at the course's constants,
batch 16, `f32`, twelve cores: PyTorch 0.405 s, presk 37.9 s (33.3 s with
`--opt --profile`). The `experiments` chapter has the three statements this
version set and which of them the measurement kept.

## What the corpus is for

It is the forcing function of 0.4.0, not a demonstration: rung 5 at course
scale is about 1 TFLOP per training step, and presk sustains of the order of
1 GFLOP/s today. Rungs 1 and 2 run at full size because they are small;
everything above rung 3 is gated on the language and the code generator, as
`presk-specs/implementation/presk-plan-0.4.0.md` sets out.
