#!/usr/bin/env python3
"""The arithmetic of experiments/machine_types.sh (plan 0.18.0 step 6, D209):
the standard library only, and margins.py's functions.

  stars MARGINS δ…            ε*(δ) from the exact margins of MARGINS/exact,
                              by D118's rule (Clopper–Pearson, α = 0.05), as
                              JSON {δ: ε*}
  contractions                the contractions of `presk dump --stage core`
                              on stdin, one site a line: each `dot` and each
                              `ein` of two operands
  reading R SITES FORMAT      the realization R with every contraction of
                              SITES whose product format is a float's summed
                              in FORMAT, its products and its additions
  report OUT --record F --names "…"   the report, from the solves' listings
                              and margins.py's report (OUT/margins.json)
"""

import json
import os
import re
import sys
from decimal import Decimal

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margins  # noqa: E402

ALPHA = 0.05


def stars(d, deltas):
    ms = []
    for c in sorted(f[:-4] for f in os.listdir(os.path.join(d, "exact")) if f.endswith(".tsv")):
        for line in open(os.path.join(d, "exact", c + ".tsv")):
            ms.append(Decimal(line.split("\t")[2]))
    sm = sorted(ms)
    out = {}
    for s in deltas:
        k = margins.k_clopper_pearson(len(ms), float(s), ALPHA)
        out[s] = f"{float(margins.eps_star(sm, k)):.4e}"
    print(json.dumps(out))


def contractions():
    fn = None
    for line in sys.stdin:
        m = re.match(r"fn ([^{(\s]+)", line)
        if m:
            fn = m.group(1)
            continue
        m = re.match(r"\s+(%\d+) : .* = (dot |ein\[[^\]]*\]\(\[[^\]]*\] \* \[[^\]]*\]\) )", line)
        if m and fn:
            print(f"{fn}/{m.group(1)}")


def reading(r_path, sites_path, fmt):
    r = json.load(open(r_path))
    sites = set(open(sites_path).read().split())
    integer = re.compile(r"^(u|i)\d+(/sat)?$|^fixed\(")
    for s, f in list(r.get("product", {}).items()):
        if s in sites and not integer.match(f):
            r["product"][s] = fmt
            r.setdefault("accum", {})[s] = fmt
    print(json.dumps(r, indent=1, ensure_ascii=False))


def listing(path):
    """The figures of a `presk solve` listing."""
    t = open(path).read()
    g = lambda pat: (re.search(pat, t, re.M).group(1) if re.search(pat, t, re.M) else None)
    return {
        "budget": g(r"under budget (\S+) absolute"),
        "formats": g(r"^  formats\s+(.*)$"),
        "sites": g(r"^  sites\s+(.*)$"),
        "params": g(r"^  parameters\s+(.*)$"),
        "rest": g(r"parameters .*: (\S+) bytes at rest"),
        "cost": g(r"^  cost\s+(\S+) against"),
        "ratio": g(r"^  cost .*\((\S+)x\)"),
        "measured": g(r"^  measured\s+(.*)$"),
        "modes": g(r"^  modes\s+(.*)$"),
        "emulated": g(r"^  emulated\s+(.*)$"),
    }


def report(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="machine_types.py report")
    ap.add_argument("dir")
    ap.add_argument("--record", required=True)
    ap.add_argument("--names", required=True)
    a = ap.parse_args(argv)
    m = json.load(open(os.path.join(a.dir, "margins.json")))
    rows = {r["name"]: r for r in m["rows"]}
    st = json.load(open(os.path.join(a.dir, "stars.json")))
    L = []
    w = L.append
    w(f"# The course GPT's logits at D = 64 in a machine's types (D209)\n")
    w(f"{m['windows']} held-out windows of the Code civil, {m['n']} positions; ε*(δ) from their exact margins (D118's rule, α = {ALPHA}): "
      + ", ".join(f"ε*({d}) = {e}" for d, e in st.items()) + ".\n")
    names = a.names.split()
    mt5 = [n for n in names if n.startswith("mt5-")]
    if mt5:
        w("## MT5: the solve in the target's types, and the accumulator's readings\n")
        tj = os.path.join(a.dir, "target.json")
        if os.path.exists(tj):
            w(f"Target `{json.load(open(tj))['name']}`.\n")
        for d in st:
            n = f"mt5-{d}"
            p = os.path.join(a.dir, "solve", n + ".txt")
            if not os.path.exists(p):
                continue
            s = listing(p)
            ex = open(os.path.join(a.dir, "solve", n + ".exit")).read().strip()
            w(f"**δ = {d}, ε* = {st[d]}** (exit {ex}). {s['formats']}. Sites: {s['sites']}. Parameters: {s['params']}.")
            w(f"Traffic {s['cost']} bytes ({s['ratio']}× under all-f64); measured: {s['measured']}. Modes: {s['modes']}. Emulated: {s['emulated']}.\n")
        w("| realization | accumulator | gap on the logits (461 windows) | ε* certified | flips | bound #{m < 2ε} | correct |")
        w("|---|---|---|---|---|---|---|")
        for n in mt5:
            r = rows[n]
            d, acc = n[len("mt5-"):].rsplit("-", 1)
            cert = ", ".join(k for k, v in r["certified"].items() if v) or "none"
            w(f"| solved at {d} | {acc} | {r['eps_meas']:.3e} | {cert} | {r['flips']} | {r['bound']} | {r['correct']} |")
        w("")
    if "mt6" in names:
        s, rec = listing(os.path.join(a.dir, "solve", "mt6.txt")), listing(a.record)
        r = rows["mt6"]
        inside = s["measured"] is not None and "inside the budget" in s["measured"]
        fewer_rest = float(s["rest"]) < float(rec["rest"])
        fewer_cost = float(s["cost"]) < float(rec["cost"])
        w("## MT6: binary16 and the 16-bit-significand float on the menu\n")
        w(f"MT6's solve: {s['sites']}; parameters: {s['params']}; traffic {s['cost']} ({s['ratio']}×); measured: {s['measured']}.")
        w(f"D170's solve of record: {rec['sites']}; parameters: {rec['params']}; traffic {rec['cost']} ({rec['ratio']}×).\n")
        # The bytes the solver counts (the plan's §3.5): each stored value
        # of a sample and the parameters once; those at rest are reported.
        w(f"- fewer bytes stored: {s['cost']} against {rec['cost']}, the parameters once and each stored value of a sample ({'yes' if fewer_cost else 'NO'}); "
          f"of which the parameters at rest {s['rest']} against {rec['rest']} ({'fewer' if fewer_rest else 'the same' if s['rest'] == rec['rest'] else 'MORE'});")
        w(f"- inside its budget by measurement: {'yes' if inside else 'NO'};")
        w(f"- on the 461 windows: gap {r['eps_meas']:.3e}, {r['flips']} flips (bound {r['bound']}): {'no decision changed' if r['flips'] == 0 else 'decisions changed'}.\n")
        holds = fewer_cost and inside and r["flips"] == 0
        w(f"MT6 {'holds' if holds else 'fails'}.")
    print("\n".join(L))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "stars":
        stars(rest[0], rest[1:])
    elif cmd == "contractions":
        contractions()
    elif cmd == "reading":
        reading(*rest)
    elif cmd == "report":
        report(rest)
    else:
        sys.exit(__doc__)
