#!/usr/bin/env python3
"""The layer norm's kernels as a Gappa statement (D108).

Writes, for `LayerNorm{D}.apply` on one row of `D` elements, the Gappa file
whose goal is the distance between the value the native `f32` (or `f64`,
`bf16`) kernels compute and the exact denotation, with one rounding per
operation the kernels perform, in their order — the order `presk dump
--stage kernel` prints for the entry (experiments/gappa.sh keeps the dump
beside the file; this script mirrors it and does not read it):

    mean      s = 0; s = s + x[i] for i in 0..D; mu = s / D
    fused 2   q[i] = (x[i] - mu) * (x[i] - mu)
    mean      s = 0; s = s + q[i] for i in 0..D; v = s / D
    fused 2   r = sqrt(v + eps)                    eps the f32 of the f64 literal
    fused 4   y[i] = (((x[i] - mu) / r) * g[i]) + b[i]

The exact value is the same tree over the reals (a sum's parenthesization
does not change its value, and Gappa pairs each rounded node with its exact
twin), with the literal `1e-5` denoting the binary64 number it denotes in
the reference (the language reference, "What a literal denotes"). The
inputs arrive rounded once into the realization's type, as every backend
rounds a value crossing the seam and as `presk check --precision` counts
them.

The hypotheses are read from the envelope `presk calibrate --precision`
wrote for the entry (ENV.json): with `--hypotheses sites` (the default),
the magnitude the envelope records at every site of the body — `|mu|`,
`|x - mu|`, the variance's and the square root's floor and ceiling, `|x̂|`,
`|y|` — as the compiler's bound is conditional on them, so that the two
bounds rest on the same conditions; with `--hypotheses input`, the inputs'
ranges alone, what a user of Gappa without the envelope would write. The
input's range is not a site of the envelope (the entry's arguments are not
measured; `check` takes their magnitude as 1), so it is derived from the
sites: `|x| <= max|x - mu| + max|mu|`.

The sites are found by their operation in the envelope's `ops` map, in
the order of the lets of `LayerNorm.apply` (mean, sub, pow 2, mean, sub,
real, add, sqrt, div, field, mul, field, add), and the script stops if the
body is not that one.

With `--prove BOUND` the goal is `|Y_0 - y_0| <= BOUND` instead of a query
(`in ?`), so that the file is a proof Gappa checks (exit 0) and not a
question; experiments/gappa.sh runs the query first and writes the proof
with the bound found. `--all-rows` emits the goal for every element of the
row (the hypotheses being the same for every `x_i`, the bound for `y_0` is
the bound for every `y_j` by renaming; at D = 8 the file checks that).

Three kinds of hint are given to Gappa, each where it is needed and with
its reason in the file: `0 + X -> X` (the sum's first addition is exact),
the division by `D` restated as the error of the sum divided by `D` plus
the quotient's own rounding (Gappa does not pair `rnd(a / D)` with `a / D`
by itself; by `D` a power of two the quotient is exact and Gappa still
charges it a rounding, which is sound), and the quotient by the square
root in absolute terms, `(D - d) / R + d (r - R) / (R r)`, since `x - mu`
has no floor and Gappa's rule for a quotient needs the numerator's
relative error. No bisection. Gappa warns that the hint's denominators
are assumed non-zero when it checks the rewriting; the goal's proof has
`R` and `r` above the square root of `eps` whatever the hypotheses.
"""
import argparse
import json
import math
import sys

FORMATS = {
    # Gappa's rounding operators: IEEE binary32 and binary64 to nearest,
    # ties to even; bf16 as 8 bits of significand and binary32's exponent
    # range (the format the native backend rounds to, D29).
    "f32": "float<ieee_32,ne>",
    "f64": "float<ieee_64,ne>",
    "bf16": "float<8,-133,ne>",
}

# The lets of `LayerNorm.apply` in order, by the operation the envelope's
# `ops` map records, and the name this script gives each (None: unnamed,
# a field read or a duplicate of an earlier value).
BODY = [
    ("mean axis model", "mu"),
    ("sub", "d"),       # x - mu, the variance's operand
    ("pow 2", "q"),
    ("mean axis model", "v"),
    ("sub", "d2"),      # x - mu again, the quotient's operand (the same value)
    ("real", "eps"),
    ("add", "ve"),      # v + eps
    ("sqrt", "r"),
    ("div", "z"),       # x̂
    (None, "g"),        # θ.γ, a field: no op recorded
    ("mul", "w"),
    (None, "b"),        # θ.β
    ("add", "y"),
]


def sites_of(env):
    """The envelope's sites of `LayerNorm.apply`, by this script's names."""
    ops = env.get("ops", {})
    mags = env["sites"]
    mins = env.get("minima", {})
    prefix = "LayerNorm.apply/%"
    lets = sorted((int(k[len(prefix):]), k) for k in mags if k.startswith(prefix))
    if not lets:
        sys.exit("gappa_layernorm.py: no site of LayerNorm.apply in the envelope")
    # Walk the lets in order, matching the body's operations; a field read
    # has no op and may or may not be a site.
    out = {}
    i = 0
    for op, name in BODY:
        if i >= len(lets):
            if op is None:
                continue
            sys.exit(f"gappa_layernorm.py: the envelope stops before `{op}` ({name})")
        n, key = lets[i]
        got = ops.get(key)
        if op is None:
            if got is None:
                out[name] = (mags[key], mins.get(key))
                i += 1
            continue
        if got != op:
            sys.exit(f"gappa_layernorm.py: site {key} is `{got}`, the body expected `{op}` ({name}): not the layer norm of examples/spec/layernorm/")
        out[name] = (mags[key], mins.get(key))
        i += 1
    if i != len(lets):
        sys.exit(f"gappa_layernorm.py: {len(lets) - i} site(s) of LayerNorm.apply after the body's last let")
    return out


def dec(x):
    """A decimal literal Gappa reads exactly, slightly outward for a ceiling.

    Gappa parses decimal numbers as exact rationals; the envelope's values
    are f64 numbers printed by Rust shortest round-trip, so `repr` is
    exact. A ceiling is widened by one part in 10^12, so that a hypothesis
    is not tighter than the measurement it stands for.
    """
    return repr(x)


def ceil(x):
    return repr(x * (1 + 1e-12)) if x > 0 else "0"


def floor(x):
    return repr(x * (1 - 1e-12)) if x > 0 else "0"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("env", help="the envelope `presk calibrate --precision` wrote (ENV.json)")
    ap.add_argument("--width", type=int, required=True, help="D, the row's length")
    ap.add_argument("--realize", default="f32", choices=sorted(FORMATS), help="the realization (default f32)")
    ap.add_argument("--hypotheses", default="sites", choices=["sites", "input"], help="the envelope's sites (default), or the inputs' ranges alone")
    ap.add_argument("--eps", default="1e-5", help="the entry's `eps` literal (default 1e-5, the example's)")
    ap.add_argument("--prove", default=None, help="write the goal as `|Y0 - y0| <= BOUND` (a proof) instead of a query")
    ap.add_argument("--all-rows", action="store_true", help="a goal for every element of the row, not only the first")
    ap.add_argument("--by-site", action="store_true", help="query every intermediate's error too (mu, x - mu, the square, the variance, the root, the quotient), the compiler's sites")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    env = json.load(open(a.env))
    D = a.width
    s = sites_of(env)
    fmt = FORMATS[a.realize]
    mu_max = s["mu"][0]
    d_max = max(s["d"][0], s["d2"][0])
    x_max = d_max + mu_max
    g_max = s["g"][0] if "g" in s else 1.0
    b_max = s["b"][0] if "b" in s else 1.0

    L = []
    w = L.append
    w(f"# {a.env}: LayerNorm{{{D}}}.apply in {a.realize}, hypotheses: {a.hypotheses}")
    w(f"# Written by experiments/gappa_layernorm.py (D108); the kernels' order is")
    w(f"# the one `presk dump --stage kernel` prints, mirrored in the real tree.")
    w(f"@rnd = {fmt};")
    w("")
    w("# The inputs, rounded once into the realization's type as they cross the")
    w("# seam; eps is the binary64 number the literal denotes, rounded once more.")
    for i in range(D):
        w(f"X{i} = rnd(x{i});")
    for i in range(D):
        w(f"G{i} = rnd(g{i});")
    for i in range(D):
        w(f"B{i} = rnd(b{i});")
    w(f"eps = float<ieee_64,ne>({a.eps});")
    w("E = rnd(eps);")
    w("")
    w(f"# mean axis model: s = 0; s = s + x[i], i in 0..{D}; mu = s / {D}")
    tx = "0"
    tX = "0"
    for i in range(D):
        tx = f"({tx} + x{i})"
        tX = f"rnd({tX} + X{i})"
    w(f"sx = {tx};")
    w(f"SX = {tX};")
    w(f"mu = sx / {D};")
    w(f"MU = rnd(SX / {D});")
    w("")
    w("# fused 2 steps: q[i] = (x[i] - mu) * (x[i] - mu)")
    for i in range(D):
        w(f"d{i} = x{i} - mu;")
        w(f"D{i} = rnd(X{i} - MU);")
    for i in range(D):
        w(f"q{i} = d{i} * d{i};")
        w(f"Q{i} = rnd(D{i} * D{i});")
    w("")
    w(f"# mean axis model: s = 0; s = s + q[i], i in 0..{D}; v = s / {D}")
    tq = "0"
    tQ = "0"
    for i in range(D):
        tq = f"({tq} + q{i})"
        tQ = f"rnd({tQ} + Q{i})"
    w(f"sq = {tq};")
    w(f"SQ = {tQ};")
    w(f"v = sq / {D};")
    w(f"V = rnd(SQ / {D});")
    w("")
    w("# fused 2 steps: r = sqrt(v + eps)")
    w("ve = v + eps;")
    w("VE = rnd(V + E);")
    w("r = sqrt(ve);")
    w("R = rnd(sqrt(VE));")
    w("")
    w("# fused 4 steps: y[i] = (((x[i] - mu) / r) * g[i]) + b[i]; x[i] - mu is")
    w("# computed again and is the same value, so D[i] stands for it.")
    rows = range(D) if a.all_rows else [0]
    for i in rows:
        w(f"z{i} = d{i} / r;")
        w(f"Z{i} = rnd(D{i} / R);")
        w(f"w{i} = z{i} * g{i};")
        w(f"W{i} = rnd(Z{i} * G{i});")
        w(f"y{i} = w{i} + b{i};")
        w(f"Y{i} = rnd(W{i} + B{i});")
    w("")

    hyp = []
    hyp.append(f"# the inputs: |x| from the sites x - mu and mu, γ and β at the magnitudes the envelope records for θ.γ and θ.β")
    for i in range(D):
        hyp.append(f"|x{i}| <= {ceil(x_max)}")
    for i in range(D):
        hyp.append(f"|g{i}| <= {ceil(g_max)}")
    for i in range(D):
        hyp.append(f"|b{i}| <= {ceil(b_max)}")
    if a.hypotheses == "sites":
        hyp.append("# the sites, as the exact run recorded them (the ceiling of |v| at every site; a floor where the compiler's quotient and square-root rules read one)")
        hyp.append(f"|mu| <= {ceil(s['mu'][0])}")
        for i in range(D):
            hyp.append(f"|d{i}| <= {ceil(d_max)}")
        for i in range(D):
            hyp.append(f"q{i} in [0, {ceil(s['q'][0])}]")
        hyp.append(f"v in [{floor(s['v'][1] or 0)}, {ceil(s['v'][0])}]")
        hyp.append(f"ve in [{floor(s['ve'][1] or 0)}, {ceil(s['ve'][0])}]")
        hyp.append(f"r in [{floor(s['r'][1] or 0)}, {ceil(s['r'][0])}]")
        for i in rows:
            hyp.append(f"|z{i}| <= {ceil(s['z'][0])}")
            hyp.append(f"|w{i}| <= {ceil(s['w'][0])}")
            hyp.append(f"|y{i}| <= {ceil(s['y'][0])}")
    if a.prove:
        goals = [f"|Y{i} - y{i}| <= {a.prove}" for i in rows]
    else:
        goals = [f"|Y{i} - y{i}| in ?" for i in rows]
    if a.by_site:
        goals = [f"|{A} - {b}| in ?" for A, b in [("MU", "mu"), ("D0", "d0"), ("Q0", "q0"), ("V", "v"), ("VE", "ve"), ("R", "r"), ("Z0", "z0"), ("W0", "w0")]] + goals

    w("{")
    conj = [h for h in hyp if not h.startswith("#")]
    for h in hyp:
        if h.startswith("#"):
            w("  " + h)
        else:
            w("  " + h + (" /\\" if h is not conj[-1] else ""))
    w("  ->")
    w("  " + " /\\\n  ".join(goals))
    w("}")
    w("")
    w("# Hints. The sum's first addition, 0 + X, is exact: Gappa would charge")
    w("# it a rounding otherwise.")
    w("0 + X0 -> X0;")
    w("0 + Q0 -> Q0;")
    w(f"# A quotient by the constant {D}: Gappa does not pair rnd(a / {D}) with")
    w(f"# a / {D} on its own; the error is the sum's over {D}, plus the quotient's")
    w(f"# own rounding (zero when {D} is a power of two, which Gappa still charges).")
    w(f"MU - mu -> (SX - sx) / {D} + (MU - SX / {D});")
    w(f"V - v -> (SQ - sq) / {D} + (V - SQ / {D});")
    w("# The quotient x̂ = (x - mu) / r in absolute terms: its numerator has no")
    w("# floor (an element can sit at the mean), so Gappa's own rule, which needs")
    w("# the numerator's relative error, gives nothing. This is the form the")
    w("# compiler's quotient rule calls `abs` (propagate.rs, PrimOp::Div):")
    w("# err(x - mu) / r + |x - mu| · err(r) / r².")
    for i in rows:
        w(f"Z{i} - z{i} -> (Z{i} - D{i} / R) + ((D{i} - d{i}) / R + d{i} * (r - R) / (R * r));")
    with open(a.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"gappa_layernorm.py: wrote {a.out} (D = {D}, {a.realize}, hypotheses {a.hypotheses}, {len(goals)} goal(s))")


if __name__ == "__main__":
    main()
