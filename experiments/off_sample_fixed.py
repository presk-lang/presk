#!/usr/bin/env python3
"""A fixed realization against N (D156's table, from rows that exist): the
standard library only, nothing calibrated, solved or run.

  off_sample_fixed.py MARGINS NAME[:IN] [NAME[:IN] ...] [--eps E] [--sizes 1,2,…]
                      [--draws R] [--seed S] [--label TEXT]

MARGINS is a directory `margins.sh` or `margins_course.sh` wrote: its
`native/<name>/<chunk>.tsv` hold, per position of the held-out windows in
order, the largest elementwise gap of the realization's native logits to the
exact ones (the first column, absolute, `margins.py native`). A window's gap is
the largest over its T = 256 positions. For each N, the gap over the first N
windows (in sample if the realization's envelope is those windows), the gap
over the others, and their ratio, as D156's "A fixed realization against N";
then the windows' gaps by quantile, the worst window, and where the envelope's
own windows (IN, a count from window 0, default 4) rank among them.

Beside the first N windows, R random sets of N windows (seeded): the median
and the 5 % and 95 % quantiles of the same ratio, what the order statistics
of these windows' gaps alone give a sample of N, whatever the solver did with
it; and the fraction of draws whose ratio is at least the first N's.

E is ε*, to say by how much each out-of-sample gap is inside it. The report
is Markdown on stdout; the files read and their sha256 at its end.
"""

import argparse
import hashlib
import os
import random
import sys

T = 256


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(mdir, name, read):
    d = os.path.join(mdir, "native", name)
    gap = []
    for c in sorted(f for f in os.listdir(d) if f.endswith(".tsv")):
        p = os.path.join(d, c)
        read.append(p)
        for line in open(p):
            gap.append(float(line.split("\t")[0]))
    assert len(gap) % T == 0, (name, len(gap))
    return gap, [max(gap[w * T:(w + 1) * T]) for w in range(len(gap) // T)]


def q(xs, p):
    """The p-quantile of xs, the nearest rank."""
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def sci(x):
    return f"{x:.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("margins")
    ap.add_argument("names", nargs="+")
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--sizes", default="1,2,4,8,16,32,64,128,256")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=156)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",")]
    out, read = [], []
    w = out.append
    w(f"# A fixed realization against N{': ' + a.label if a.label else ''}\n")
    w(f"Rows of `{a.margins}`. " + (f"ε\\* = {a.eps:.3e}. " if a.eps else "")
      + f"Random sets: {a.draws} draws of N windows per N, seed {a.seed}.\n")
    for spec in a.names:
        name, _, inn = spec.partition(":")
        inn = int(inn or 4)
        gap, win = rows(a.margins, name, read)
        n = len(win)
        ks = [k for k in sizes if k < n]
        rng = random.Random(a.seed)
        order = sorted(range(n), key=lambda i: -win[i])
        w(f"## `{name}`, its envelope on windows 0 … {inn - 1}\n")
        w(f"{n} windows, {len(gap)} positions. The largest gap {sci(max(win))} (window "
          f"{max(range(n), key=lambda i: win[i])}); over the envelope's {inn} windows {sci(max(win[:inn]))}, "
          f"the {n}'s {max(win) / max(win[:inn]):.2f}× it.\n")
        head = "| N | in: first N | out: the others | out/in | worst out window | random N: median out/in | 5 % | 95 % | P(≥ first N's) |"
        if a.eps:
            head += " ε\\*/out |"
        w(head)
        w("|---" * (head.count("|") - 1) + "|")
        for k in ks:
            g_in, g_out = max(win[:k]), max(win[k:])
            worst = max(range(k, n), key=lambda i: win[i])
            r = g_out / g_in
            rs = []
            idx = list(range(n))
            for _ in range(a.draws):
                s = set(rng.sample(idx, k))
                rs.append(win[next(i for i in order if i not in s)] / max(win[i] for i in s))
            ge = sum(1 for x in rs if x >= r) / len(rs)
            row = (f"| {k} | {sci(g_in)} | {sci(g_out)} | **{r:.2f}×** | {worst} | {q(rs, 0.5):.2f}× | "
                   f"{q(rs, 0.05):.2f}× | {q(rs, 0.95):.2f}× | {ge:.3f} |")
            if a.eps:
                row += f" {a.eps / g_out:.2f}× |"
            w(row)
        w("")
        rank = {i: order.index(i) + 1 for i in range(n)}
        top = win[order[0]]
        w("The windows' gaps (each window's largest over its positions) and the positions':\n")
        w("| | min | median | 0.9 | 0.99 | max |")
        w("|---|---|---|---|---|---|")
        w(f"| per window | {sci(min(win))} | {sci(q(win, 0.5))} | {sci(q(win, 0.9))} | {sci(q(win, 0.99))} | {sci(top)} |")
        w(f"| per position | {sci(min(gap))} | {sci(q(gap, 0.5))} | {sci(q(gap, 0.9))} | {sci(q(gap, 0.99))} | {sci(max(gap))} |")
        w("")
        w(f"- The ten worst windows: " + ", ".join(f"{i} ({sci(win[i])})" for i in order[:10]) + ".")
        w(f"- Within 10 % of the worst: {sum(1 for x in win if x > 0.9 * top)}; within a factor 2: "
          f"{sum(1 for x in win if x > 0.5 * top)}; above the envelope's maximum: "
          f"{sum(1 for x in win if x > max(win[:inn]))} of {n}.")
        w(f"- The envelope's windows rank " + ", ".join(f"{i}: {rank[i]}" for i in range(inn))
          + f" of {n} (1 the worst).")
        if a.eps:
            w(f"- Windows above ε\\*: {sum(1 for x in win if x > a.eps)}; above ε\\*/2: "
              f"{sum(1 for x in win if x > a.eps / 2)}.")
        w("")
    w("## Files read\n")
    for p in read:
        w(f"- `{p}` {sha(p)}")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
