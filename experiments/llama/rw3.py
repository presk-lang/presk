#!/usr/bin/env python3
"""rw3.py — statement RW3 of plan 0.17.0 (presk D180), reported: a public
model's error against the exact reference.

    rw3.py DIR

DIR holds, for prompts k = 0, 1, …:
- `exact.{k}.json`, `presk run --backend interp --bits 128 --json`;
- `{realization}.{k}.json`, `presk run --backend native --realize R --json`.

`experiments/llama/llama3.sh` runs them. For each realization and prompt
this prints the largest gap of the logits to the exact ones, relative to
their largest magnitude, and the positions whose argmax changes. It writes
`rw3.json` and `rw3.md`.
"""

import glob
import json
import os
import re
import sys

import numpy as np


def logits(path):
    with open(path) as f:
        r = json.load(f)["result"]

    def flat(x):
        if isinstance(x, list):
            for y in x:
                yield from flat(y)
        else:
            yield float(x)

    return np.fromiter(flat(r["data"]), dtype=np.float64).reshape([ax["len"] for ax in r["axes"]])


def main():
    d = sys.argv[1]
    prompts = sorted(int(m.group(1)) for p in glob.glob(os.path.join(d, "exact.*.json")) if (m := re.search(r"exact\.(\d+)\.json$", p)))
    names = sorted({m.group(1) for p in glob.glob(os.path.join(d, "*.*.json")) if (m := re.search(r"/([a-z0-9_-]+)\.(\d+)\.json$", p)) and m.group(1) != "exact"})
    rows = []
    for name in names:
        for k in prompts:
            path = os.path.join(d, f"{name}.{k}.json")
            if not os.path.exists(path):
                continue
            exact = logits(os.path.join(d, f"exact.{k}.json"))
            got = logits(path)
            gap = float(np.abs(got - exact).max())
            mag = float(np.abs(exact).max())
            flips = [int(t) for t in np.nonzero(got.argmax(-1)[0] != exact.argmax(-1)[0])[0]]
            rows.append({"realization": name, "prompt": k, "positions": int(exact.shape[1]), "gap": gap, "relative": gap / mag, "argmax_changed": flips})
    with open(os.path.join(d, "rw3.json"), "w") as f:
        json.dump(rows, f, indent=2)
    lines = ["| realization | prompt | max \\|logit − exact\\| / max \\|exact\\| | positions whose argmax changes |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['realization']} | {r['prompt']} | {r['relative']:.3e} | {len(r['argmax_changed'])} of {r['positions']} {r['argmax_changed'] if r['argmax_changed'] else ''} |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(d, "rw3.md"), "w") as f:
        f.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
