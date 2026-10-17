#!/usr/bin/env python3
"""PyTorch's own gap to the exact model, and its flips (D130).

    pytorch_gap.py logits --export DIR --weights W.psk.d --data VAL.psk.d
                          --exact MARGINS/exact --dtype float32 --threads N
                          --name NAME --out OUT

runs the course GPT as `presk export --target pytorch --mode readable
--train` writes it (DIR: `_05_gpt.py`, `pskd.py`), in eval mode under
`torch.no_grad()` on presk's checkpoint W, on the held-out windows of VAL,
chunk by chunk as `make margins` took the exact logits (each chunk of
MARGINS/exact, PREFIX.hilo, its first window the prefix's number), and
writes what `experiments/margins.py native` and `decisions` write for a
realization of presk:

    OUT/native/NAME/PREFIX.tsv   per position: the largest gap of PyTorch's
                                 logits to the exact ones, PyTorch's argmax,
                                 the sum of the gaps
    OUT/decisions/NAME.txt       PyTorch's argmax at every position, one per
                                 line

so that `experiments/margins.py report` counts its flips against the exact
decisions and its bound #{m < 2ε} as it does every realization's. The
argmax is the reference's: the greatest index among equal maxima
(`torch.argmax` returns the first, so it is taken on the reversed row). The
gap at a logit is |(ŷ − hi) − lo|, margins.py's, the exact logit being
hi + lo to about 10⁻³² relative. It prints torch's version, the threads,
the dtype and the time per chunk.
"""
import argparse
import array
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_baseline import LAYOUT  # noqa: E402  the parameter tree of `params GPT`


def logits(a):
    torch.set_num_threads(a.threads)
    sys.path.insert(0, os.path.abspath(a.export))
    from _05_gpt import GPT
    from pskd import PskdDataset, read_params

    dtype = getattr(torch, a.dtype)
    data = PskdDataset(a.data)
    x0, _ = data[0]
    T = x0.shape[0]
    model = GPT(T).to(dtype)
    model.load_state_dict(read_params(a.weights, LAYOUT, model))
    model.eval()
    os.makedirs(os.path.join(a.out, "native", a.name), exist_ok=True)
    os.makedirs(os.path.join(a.out, "decisions"), exist_ok=True)
    chunks = sorted(f[:-5] for f in os.listdir(a.exact) if f.endswith(".hilo"))
    print(f"pytorch_gap: torch {torch.__version__}, {torch.get_num_threads()} threads, {a.dtype}, {len(chunks)} chunks of {a.exact}")
    decisions = []
    total_s = 0.0
    for c in chunks:
        hilo = array.array("d")
        with open(os.path.join(a.exact, c + ".hilo"), "rb") as f:
            hilo.frombytes(f.read())
        V = 92
        rows = len(hilo) // (2 * V)
        start, count = int(c), rows // T
        ids = torch.stack([data[i][0] for i in range(start, start + count)])
        t0 = time.perf_counter()
        with torch.no_grad():
            y = model(ids, None)
        total_s += time.perf_counter() - t0
        y = y.double().numpy().reshape(rows, V)
        ex = np.frombuffer(hilo, dtype=np.float64).reshape(rows, V, 2)
        gap = np.abs((y - ex[:, :, 0]) - ex[:, :, 1])
        top = (V - 1) - np.argmax(y[:, ::-1], axis=1)
        with open(os.path.join(a.out, "native", a.name, c + ".tsv.part"), "w") as out:
            for r in range(rows):
                out.write(f"{float(gap[r].max())!r}\t{int(top[r])}\t{float(gap[r].sum())!r}\n")
        os.replace(os.path.join(a.out, "native", a.name, c + ".tsv.part"), os.path.join(a.out, "native", a.name, c + ".tsv"))
        decisions.extend(int(t) for t in top)
        print(f"  {c}: {count} windows, {rows} positions, largest gap {gap.max():.3e}")
    with open(os.path.join(a.out, "decisions", a.name + ".txt"), "w") as out:
        out.write("".join(f"{d}\n" for d in decisions))
    print(f"pytorch_gap: {len(decisions)} positions, the forward passes {total_s:.2f} s in all")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("logits")
    g.add_argument("--export", required=True)
    g.add_argument("--weights", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--exact", required=True)
    g.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    g.add_argument("--threads", type=int, required=True)
    g.add_argument("--name", required=True)
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    {"logits": logits}[a.cmd](a)


if __name__ == "__main__":
    main()
