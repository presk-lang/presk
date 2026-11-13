#!/usr/bin/env python3
"""The exact model's validation loss from the exact logits make margins keeps
(D129): the mean cross-entropy over every held-out position, read from
`experiments/margins.py exact`'s PREFIX.hilo files (each logit as two
doubles whose sum is it to about 10⁻³² relative) and the windows' targets in
`presk data export`'s JSON.

    margins_loss.py DIR/exact VAL.json [--seq 256]

prints the number of positions and the loss to 12 digits. At each position
the loss is log Σ exp(y_v) − y_target, computed in doubles from the logits'
leading parts with the largest subtracted first (so to about 10⁻¹⁵ of the
loss, far below any realization's gap to it), and the mean is `math.fsum`'s.
This is `loss` of examples/corpus/nlp/05_gpt.psk on the exact model — the mean over
batch and sequence of `cross_entropy` — which the reference itself would
take 178.6 s a window to compute at the course constants (D101).
"""
import argparse
import array
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("exact")
ap.add_argument("val")
ap.add_argument("--seq", type=int, default=256)
ap.add_argument("--vocab", type=int, default=92)
a = ap.parse_args()

chunks = sorted(f[:-5] for f in os.listdir(a.exact) if f.endswith(".hilo"))
samples = json.load(open(a.val))["samples"]
V, T = a.vocab, a.seq
terms = []
window = 0
for c in chunks:
    h = array.array("d")
    with open(os.path.join(a.exact, c + ".hilo"), "rb") as f:
        h.frombytes(f.read())
    rows = len(h) // (2 * V)
    assert rows % T == 0, (c, rows)
    for r in range(rows):
        y = [h[2 * (r * V + v)] + h[2 * (r * V + v) + 1] for v in range(V)]
        _, target = samples[window + r // T]
        t = target[r % T]
        m = max(y)
        terms.append(m + math.log(math.fsum(math.exp(v - m) for v in y)) - y[t])
    window += rows // T
print(f"{len(terms)} positions in {window} windows: exact loss {math.fsum(terms) / len(terms):.12f}")
