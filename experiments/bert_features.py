"""SQuAD-shaped features for the `squad` entry of coverage/bert.psk.

Usage: bert_features.py V T N SEED > features.txt, then
presk data pack --type '(idx[V][seq: T], idx[2][seq: T], ℝ[seq2: T])'.

One line per feature: the token ids, the segment ids and the additive mask,
laid out as MLPerf's harness builds a SQuAD feature (max_seq_length 384):
[CLS] question [SEP] context [SEP], then padding. Segment 0 is the question,
1 the context; the mask is 0 on real tokens and -10000 on padding, the value
the reference implementation adds. The token ids are random over WordPiece's
ordinary range (from 1996): there is no tokenizer here, and a model at its
initialization does not care which words they are (0.6.0 step 8, D53).
"""
import random
import sys

V, T, N, SEED = (int(a) for a in sys.argv[1:5])
CLS, SEP, PAD = 101 % V, 102 % V, 0
LO = 1996 if V > 1996 else 3
rng = random.Random(SEED)
for _ in range(N):
    q = rng.randint(max(1, T // 32), max(1, T // 16))      # question tokens
    real = rng.randint(T * 3 // 4, T - 1)                  # tokens before the padding
    c = max(1, real - q - 3)                               # context tokens
    ids = [CLS] + [rng.randrange(LO, V) for _ in range(q)] + [SEP] \
        + [rng.randrange(LO, V) for _ in range(c)] + [SEP]
    types = [0] * (q + 2) + [1] * (c + 1)
    pad = T - len(ids)
    ids += [PAD] * pad
    types += [0] * pad
    mask = ["0"] * (T - pad) + ["-10000"] * pad
    print(" ".join(map(str, ids + types)) + " " + " ".join(mask))
