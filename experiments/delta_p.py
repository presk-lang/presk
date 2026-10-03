#!/usr/bin/env python3
"""The table of experiments/delta_p.sh (D98): the reference at p bits against
the reference at the highest precision run, in the form of each budget's gap.

Reads out/delta_p/<entry>_<bits>.json (`presk run --json`, reals as decimal
strings at --digits) and <entry>_<bits>.time, compares exactly (every
string is read as a decimal number at 400 digits, far below what any run
printed) and prints the Markdown report.
"""
import glob
import json
import math
import os
import re
import sys
from decimal import Decimal, getcontext

getcontext().prec = 400
OUT = sys.argv[1] if len(sys.argv) > 1 else "out/delta_p"


def leaves(v):
    """The real leaves of a `presk run --json` value, in order."""
    if isinstance(v, dict):
        if "data" in v and "axes" in v:
            yield from strings(v["data"])
        else:
            for k in v:
                yield from leaves(v[k])
    elif isinstance(v, list):
        for x in v:
            yield from leaves(x)


def strings(d):
    if isinstance(d, list):
        for x in d:
            yield from strings(x)
    else:
        yield Decimal(d)


def load(entry, bits):
    path = f"{OUT}/{entry}_{bits}.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(leaves(json.load(f)["result"]))


def cost(name):
    try:
        s, kb = open(f"{OUT}/{name}.time").read().split()
    except (OSError, ValueError):
        return "—"
    mb = int(kb) / 1024
    mem = f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"
    return f"{float(s):.0f} s, {mem}"


def sci(x):
    """A Decimal as d.d·10^e."""
    if x == 0:
        return "0"
    e = x.adjusted()
    m = x.scaleb(-e)
    return f"{m:.2f}·10^{e}"


def bits_of(entry):
    """The precisions whose run is complete (its .time written)."""
    return sorted(int(re.search(rf"{entry}_(\d+)\.time$", p).group(1)) for p in glob.glob(f"{OUT}/{entry}_*.time"))


def forms(a, b):
    """The gap of a against b in the ∞-norm relative to max|b| and in the
    2-norm relative to ‖b‖, with max|b|."""
    dinf = max(abs(x - y) for x, y in zip(a, b))
    minf = max(abs(y) for y in b)
    d2 = sum((x - y) ** 2 for x, y in zip(a, b)).sqrt()
    n2 = sum(y**2 for y in b).sqrt()
    return dinf / minf, d2 / n2, minf


def table(entry, title, norm):
    bits = bits_of(entry)
    if len(bits) < 2:
        return
    top = bits[-1]
    ref = load(entry, top)
    print(f"### {title}\n")
    print(f"Against the reference at {top} bits ({len(ref)} real leaves, {cost(f'{entry}_{top}')}).\n")
    print(f"| p | δ_p, {norm} | δ_p / δ_2p | log₂ of it | time, peak memory |")
    print("|---|---|---|---|---|")
    deltas = {}
    for p in bits[:-1]:
        v = load(entry, p)
        assert len(v) == len(ref), (entry, p, len(v), len(ref))
        rinf, r2, m = forms(v, ref)
        deltas[p] = r2 if norm.startswith("2-norm") else rinf
    for p in bits[:-1]:
        ratio = deltas[p] / deltas[2 * p] if 2 * p in deltas and deltas[2 * p] != 0 else None
        r = f"{sci(ratio)} | {math.log2(float(ratio)):.0f}" if ratio is not None else "— | —"
        print(f"| {p} | {sci(deltas[p])} | {r} | {cost(f'{entry}_{p}')} |")
    print()
    return deltas


def step():
    rows = []
    for p in bits_of("step"):
        txt = open(f"{OUT}/step_{p}.json").read()
        m = re.search(r"^f64\s+(\S+)\s+(\S+)\s+(\S+)", txt, re.M)
        n = re.search(r"\|θ′ − θ\|\s+(\S+)\s+\|g\|\s+(\S+)", txt)
        if m and n:
            rows.append((p, *m.groups(), *n.groups(), cost(f"step_{p}")))
    if not rows:
        return
    print("### The training step, checked at four digits\n")
    print("`presk calibrate --step --realize f64`, the all-`f64` native step against the exact step at p bits, D52's state (step 101 of 600 from the weights, 100 warm-up steps in `f64`, two windows from 4000): the exact step has no printer at full precision, so the row can only be seen not to move.\n")
    print("| p | update gap | storage floor | gradient gap | ‖θ′ − θ‖₂ | ‖g‖₂ | time, peak memory |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")
    print()


print("## δ_p measured: the reference at p bits against itself at more\n")
loss = table("loss", "The loss, `loss` (∞-norm: |L_p − L_P| / |L_P|)", "|L_p − L_P| / |L_P|")
if loss:
    ref = load("loss", bits_of("loss")[-1])
    print(f"L at the highest precision: {str(ref[0])[:60]}…\n")
logits = table("logits", "The logits, `GPT.apply` in its inference reading (∞-norm relative to max|y|)", "max|Δy| / max|y|")
grad = table("grad", "The gradient of `loss` at the weights (2-norm relative to ‖g‖₂)", "2-norm, ‖g_p − g_P‖₂ / ‖g_P‖₂")
if grad:
    bits = bits_of("grad")
    ref = load("grad", bits[-1])
    print("The same in the ∞-norm, max|Δg| / max|g|: " + ", ".join(f"{p}: {sci(forms(load('grad', p), ref)[0])}" for p in bits[:-1]) + ".\n")
step()
