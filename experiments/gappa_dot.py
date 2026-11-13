#!/usr/bin/env python3
"""One element of a contraction's kernel as a Gappa statement (D131).

Writes, for `dot` of examples/experiments/gappa_dot.psk, `[M, K] · [K, N]`, the Gappa
file whose goal is the distance between the first output element the
native `f32` (or `f64`, `bf16`) kernel computes and the exact denotation,
with one rounding per operation the kernel performs, in its order. That is
the order `presk dump --stage kernel` prints, which experiments/gappa_dot.sh
keeps beside the file; this script mirrors it and does not read it.

    sequential        s = 0; s = s + x[k] * w[k] for k in 0..K
                      (the default's register-blocked kernel sums each
                      output element in the same order, D32)
    interleaved(n)    s_t = 0 for t in 0..n;
                      s_t = s_t + x[k] * w[k] for k = c·n + t, c in 0..K/n;
                      y = ((s_0 + s_1) + s_2) + … + s_{n-1}

Every product is rounded, then every sum. The exact value is the same tree
over the reals: a sum's parenthesization does not change its value, and
Gappa pairs each rounded node with its exact twin, so the real tree mirrors
the kernel's. The operands arrive rounded once into the realization's type,
as every backend rounds a value crossing the seam (`x = a + 0` is then
exact and is not written).

The hypotheses are read from the envelope `presk calibrate --precision`
wrote for the entry (ENV.json), at the two operands' sites `x` and `w`:
every element of the row of x and the column of w is bounded by that site's
largest magnitude. With `--hypotheses sites`, the output's largest magnitude
is added as `|y| <=`. The envelope records no partial sum, so nothing bounds
one but what Gappa derives from the operands. The sites are found by their
operation in the envelope's `ops` map: two `add`s, then the `dot`.

With `--prove BOUND` the goal is `|Y - y| <= BOUND` instead of a query
(`in ?`), so that the file is a proof Gappa checks (exit 0) and not a
question; experiments/gappa_dot.sh runs the query first and writes the proof
with the bound found.

One hint is given, the same as D108's: every accumulator's first addition,
`0 + P`, is exact, and Gappa would charge it a rounding otherwise.
"""
import argparse
import json
import sys

FORMATS = {
    # Gappa's rounding operators: IEEE binary32 and binary64 to nearest,
    # ties to even; bf16 as 8 bits of significand and binary32's exponent
    # range (the format the native backend rounds to, D29).
    "f32": "float<ieee_32,ne>",
    "f64": "float<ieee_64,ne>",
    "bf16": "float<8,-133,ne>",
}


def sites_of(env):
    """The envelope's x, w and y sites of `dot`, by their operations."""
    ops = env.get("ops", {})
    mags = env["sites"]
    prefix = "dot/%"
    lets = sorted((int(k[len(prefix):]), k) for k in mags if k.startswith(prefix))
    got = [(ops.get(k), k) for _, k in lets]
    adds = [k for op, k in got if op == "add"]
    dots = [k for op, k in got if op == "dot"]
    if len(adds) != 2 or len(dots) != 1:
        sys.exit(f"gappa_dot.py: the envelope's sites of dot are {got}, not two adds and a dot (examples/experiments/gappa_dot.psk)")
    return mags[adds[0]], mags[adds[1]], mags[dots[0]]


def ceil(x):
    """A decimal literal Gappa reads exactly, widened by one part in 10¹²
    so that a hypothesis is not tighter than the measurement it stands for
    (the envelope's values are f64 numbers printed shortest round-trip)."""
    return repr(x * (1 + 1e-12)) if x > 0 else "0"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("env", help="the envelope `presk calibrate --precision` wrote (ENV.json)")
    ap.add_argument("--width", type=int, required=True, help="K, the contracted length")
    ap.add_argument("--order", default="sequential", help="sequential, or interleaved(n)")
    ap.add_argument("--realize", default="f32", choices=sorted(FORMATS), help="the realization (default f32)")
    ap.add_argument("--hypotheses", default="operands", choices=["operands", "sites"], help="the operands' magnitudes (default), and the output's too")
    ap.add_argument("--prove", default=None, help="write the goal as `|Y - y| <= BOUND` (a proof) instead of a query")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    K = a.width
    if a.order == "sequential":
        n = 1
    elif a.order.startswith("interleaved(") and a.order.endswith(")"):
        n = int(a.order[len("interleaved("):-1])
        if n < 2 or K % n:
            sys.exit(f"gappa_dot.py: interleaved({n}) needs n ≥ 2 dividing K = {K} (the kernel falls back to sequential otherwise)")
    else:
        sys.exit(f"gappa_dot.py: order {a.order}: sequential or interleaved(n)")
    xm, wm, ym = sites_of(json.load(open(a.env)))
    fmt = FORMATS[a.realize]

    L = []
    w = L.append
    w(f"# {a.env}: dot, [M, {K}] · [{K}, N], element (0, 0), {a.realize}, {a.order}, hypotheses: {a.hypotheses}")
    w("# Written by experiments/gappa_dot.py (D131); the kernel's order is the")
    w("# one `presk dump --stage kernel` prints, mirrored in the real tree.")
    w(f"@rnd = {fmt};")
    w("")
    w("# The operands, rounded once into the realization's type at the seam.")
    for k in range(K):
        w(f"X{k} = rnd(x{k});")
        w(f"W{k} = rnd(w{k});")
    w("")
    w("# The products, each rounded.")
    for k in range(K):
        w(f"p{k} = x{k} * w{k};")
        w(f"P{k} = rnd(X{k} * W{k});")
    w("")
    if n == 1:
        w(f"# sequential: s = 0; s = s + p[k], k in 0..{K}")
        te, tr = "0", "0"
        for k in range(K):
            te = f"({te} + p{k})"
            tr = f"rnd({tr} + P{k})"
        w(f"y = {te};")
        w(f"Y = {tr};")
        firsts = ["P0"]
    else:
        w(f"# interleaved({n}): s_t = 0; s_t = s_t + p[c·{n} + t], c in 0..{K // n}; then the partials left to right")
        for t in range(n):
            te, tr = "0", "0"
            for c in range(K // n):
                k = c * n + t
                te = f"({te} + p{k})"
                tr = f"rnd({tr} + P{k})"
            w(f"s{t} = {te};")
            w(f"S{t} = {tr};")
        te, tr = "s0", "S0"
        for t in range(1, n):
            te = f"({te} + s{t})"
            tr = f"rnd({tr} + S{t})"
        w(f"y = {te};")
        w(f"Y = {tr};")
        firsts = [f"P{t}" for t in range(n)]
    w("")

    hyp = [f"# the operands: every element of x's row and w's column at the largest magnitude the envelope records at its site, {xm!r} and {wm!r}"]
    for k in range(K):
        hyp.append(f"|x{k}| <= {ceil(xm)}")
    for k in range(K):
        hyp.append(f"|w{k}| <= {ceil(wm)}")
    if a.hypotheses == "sites":
        hyp.append(f"# the output, at the magnitude the envelope records for it")
        hyp.append(f"|y| <= {ceil(ym)}")
    goal = f"|Y - y| <= {a.prove}" if a.prove else "|Y - y| in ?"

    w("{")
    conj = [h for h in hyp if not h.startswith("#")]
    for h in hyp:
        if h.startswith("#"):
            w("  " + h)
        else:
            w("  " + h + (" /\\" if h is not conj[-1] else ""))
    w("  ->")
    w("  " + goal)
    w("}")
    w("")
    w("# Hints. Every accumulator's first addition, 0 + P, is exact: Gappa would")
    w("# charge it a rounding otherwise.")
    for p in firsts:
        w(f"0 + {p} -> {p};")
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"gappa_dot.py: wrote {a.out} (K = {K}, {a.realize}, {a.order}, hypotheses {a.hypotheses})")


if __name__ == "__main__":
    main()
