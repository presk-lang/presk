"""Token sequences for the `Llama.apply` and `loss` entries of examples/coverage/llama.psk.

Usage: llama_ids.py V T N SEED [--shift] > ids.txt, then
presk data pack --type 'idx[V][seq: T]'
(with --shift: '(idx[V][seq: T], idx[V][seq: T])').

One line per sequence: T token ids drawn uniformly over the vocabulary
[0, V) from a generator seeded with SEED. There is no tokenizer and no text
here: a model at its initialization does not care which tokens they are, as
BERT's features do not (bert_features.py, D53). Each sequence is drawn as
T + 1 ids; the line holds the first T, and with --shift the last T after
them, the next token at each position, which the `loss` entry's
cross-entropy is taken against. So the inputs are the same with and without
--shift (D104).
"""
import random
import sys

V, T, N, SEED = (int(a) for a in sys.argv[1:5])
SHIFT = "--shift" in sys.argv[5:]
rng = random.Random(SEED)
for _ in range(N):
    ids = [rng.randrange(V) for _ in range(T + 1)]
    line = " ".join(map(str, ids[:T]))
    if SHIFT:
        line += " " + " ".join(map(str, ids[1:]))
    print(line)
