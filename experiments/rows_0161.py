#!/usr/bin/env python3
"""The rows of record solved again at 0.16.1 (plan 0.16.1 step 2, D170), old
beside new: the standard library only, nothing run.

  rows_0161.py [--out DIR]

Reads 0.16.0's solves and realizations (moved to out/pre-0.16.1/, or kept in
their own directories: out/v6-0.16.0, out/margins-course-0.16.0/logits,
out/paper-0.16.0) and 0.16.1's (out/step2-0.16.1/phaseA.sh's), and the
margins rows on both texts of record (the Code civil's 461 held-out windows,
the Code pénal's first 461, D168) at each version. For every row:

  Z1  the values the rules leave out (rule 1, rule 2 with each one's first
      step's gain beside θ·B), the traffic forgone as the solver counts it
      (their saving in bf16) and against 0.16.0's realization (what 0.16.0's
      types saved on them);
  Z4  the decisions that moved, the traffic ratio, the estimate, the gap
      measured on the envelope, the largest gap, flips and bound on each
      text, certification at 10⁻³ (the gap at most the Code civil's ε*);
  Z2  for the two certificate rows, the four conditions.

The report is Markdown on stdout; the files read and their sha256 at its end.
"""

import argparse
import hashlib
import json
import os
import re
import sys

BYTES = {"f64": 8, "f32": 4, "bf16": 2}
BUDGETS = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7"]
PRE = "out/pre-0.16.1"
READ = []


def read(path):
    READ.append(path)
    return open(path).read()


def solve_text(path):
    """The figures of a `presk solve` report."""
    t = read(path)
    f = {}
    m = re.search(r"sites\s+(\d+) narrowable, (\d+) assigned away from f64(?: \((\d+) stored values, (\d+) not offered\))?", t)
    f["offered"], f["assigned"] = int(m[1]), int(m[2])
    f["stored"] = int(m[3]) if m[3] else int(m[1])
    m = re.search(r"parameters\s+(\d+) offered(?: of (\d+))?, (\d+) stored away from f64 \((\d+) f32, (\d+) bf16\)", t)
    if m:
        f["p_offered"], f["p_all"] = int(m[1]), int(m[2] or m[1])
        f["p_away"], f["p_f32"], f["p_bf16"] = int(m[3]), int(m[4]), int(m[5])
    m = re.search(r"cost\s+(\S+) against (\S+) all-f64 \(([\d.]+)x\)", t)
    f["cost"], f["base"], f["ratio"] = float(m[1]), float(m[2]), float(m[3])
    f["estimate"] = float(re.search(r"estimate\s+(\S+)", t)[1])
    m = re.search(r"measured\s+(\S+) against the exact reference, after (\d+) widening", t)
    f["measured"], f["rounds"] = (float(m[1]), int(m[2])) if m else (None, None)
    m = re.search(r"\(rule 1\): (.*)", t)
    f["rule1"] = m[1].split(", ") if m else []
    m = re.search(r"θ·B = \S+ × \S+ = (\S+) bytes \(rule 2, --min-gain\), the bytes it saves: (.*)", t)
    f["theta_b"] = float(m[1]) if m else None
    f["rule2"] = [(p.rsplit(" ", 1)[0], float(p.rsplit(" ", 1)[1])) for p in m[2].split(", ")] if m else []
    m = re.search(r"at most (\S+) bytes forgone .*?, ([\d.]+) % of the realization's traffic, (\S+) bytes", t)
    f["forgone"], f["forgone_pct"], f["traffic"] = (float(m[1]), float(m[2]), float(m[3])) if m else (0.0, 0.0, None)
    return f


def realization(path):
    r = json.loads(read(path))
    out = {k: v for k, v in r.get("storage", {}).items()}
    out.update({k: v for k, v in r.get("params", {}).items()})
    return out, r.get("default", "f64")


def moved(old, new):
    (o, od), (n, nd) = old, new
    keys = sorted(set(o) | set(n))
    return [(k, o.get(k, od), n.get(k, nd)) for k in keys if o.get(k, od) != n.get(k, nd)]


def against_old(f, old):
    """What 0.16.0's types saved on the values the rules leave out: rule 1's
    are one element, rule 2's have gain / 4 elements (the first step saves 4
    bytes an element)."""
    o, od = old
    s = sum(8 - BYTES[o.get(v, od)] for v in f["rule1"])
    s += sum(g / 4 * (8 - BYTES[o.get(v, od)]) for v, g in f["rule2"])
    return s


def margins(d):
    p = os.path.join(d, "margins.json")
    if not os.path.exists(p):
        return None
    m = json.loads(read(p))
    return {"eps": float(m["star"]["0.001"]["eps_cp"]), "rows": {r["name"]: r for r in m["rows"]}}


def e3(x):
    return "—" if x is None else f"{x:.3e}"


def row_margins(name, civil, penal, eps):
    out = {}
    for t, M in (("civil", civil), ("penal", penal)):
        r = M["rows"].get(name) if M else None
        out[t] = (r["eps_meas"], r["flips"], r["bound"]) if r else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    w = print
    M = {
        ("d64", "0.16.0"): (margins("out/margins-0.16.0"), margins("out/margins-penal")),
        ("d64", "0.16.1"): (margins("out/margins-0.16.1"), margins("out/margins-penal-0.16.1")),
        ("course", "0.16.0"): (margins("out/margins-course-0.16.0"), margins("out/margins-course-penal")),
        ("course", "0.16.1"): (margins("out/margins-course-0.16.1"), margins("out/margins-course-penal-0.16.1")),
    }
    eps = {s: M[(s, "0.16.0")][0]["eps"] for s in ("d64", "course")}
    rows = [
        # (section, scale, margins name, old solve, new solve, old realization, new realization)
        ("V6, the logits at D = 64 at ε\\*", "d64", "logits-4.67e-4", "out/v6-0.16.0/solve.txt", "out/v6-0.16.1/solve.txt",
         "out/v6-0.16.0/solved.json", "out/v6-0.16.1/solved.json"),
        ("The course logits at ε\\*", "course", "logits-4.96e-4", "out/margins-course-0.16.0/logits/solve.txt",
         "out/margins-course-0.16.1/logits/solve.txt", "out/margins-course-0.16.0/logits/solved.json",
         "out/margins-course-0.16.1/logits/solved.json"),
    ]
    for b in BUDGETS:
        rows.append((f"the trained curve at D = 64, {b}", "d64", f"d016-{b}", f"{PRE}/curve-trained_{b}.txt",
                     f"out/curve-trained_{b}.txt", f"{PRE}/curve-trained_{b}.json", f"out/curve-trained_{b}.json"))
    for b in BUDGETS:
        rows.append((f"the trained course curve, {b}", "course", f"ct-{b}", f"{PRE}/curve-course-trained_{b}.txt",
                     f"out/curve-course-trained_{b}.txt", f"{PRE}/curve-course-trained_{b}.json", f"out/curve-course-trained_{b}.json"))
    for b in BUDGETS:
        rows.append((f"D129's course curve, {b}", "course", f"curve-{b}", f"{PRE}/curve-course_{b}.txt",
                     f"out/curve-course_{b}.txt", f"{PRE}/curve-course_{b}.json", f"out/curve-course_{b}.json"))

    w("# The rows of record solved again at 0.16.1, old beside new")
    w()
    w(f"ε\\*(10⁻³), the Code civil's, corrected: {eps['d64']:.4e} at D = 64, {eps['course']:.4e} at the course constants. "
      "A gap is the largest elementwise gap of the native logits to the exact ones on a text's 461 windows; "
      "*cert.* is that gap at most ε\\* on the Code civil; Y1 on the Code pénal the gap at most ε\\* and no flip; "
      "Y2 the Code pénal's gap within [0.5×, 2×] of the Code civil's.")
    w()
    w("## Z1: what the rules leave out")
    w()
    w("| row | rule 1 | rule 2 | θ·B (bytes) | largest gain left out / θ·B | forgone, the solver's count | of the traffic | "
      "what 0.16.0's types saved on them |")
    w("|---|---|---|---|---|---|---|---|")
    table = []
    for sec, scale, name, so, sn, ro, rn in rows:
        if not (os.path.exists(sn) and os.path.exists(rn)):
            w(f"| {sec} | not solved yet | | | | | | |")
            continue
        fo, fn = solve_text(so), solve_text(sn)
        old, new = realization(ro), realization(rn)
        top = max((g for _, g in fn["rule2"]), default=None)
        w(f"| {sec} | {len(fn['rule1'])} | {len(fn['rule2'])} | {e3(fn['theta_b'])} | "
          f"{'—' if top is None else f'{top / fn['theta_b']:.2f}'} | {e3(fn['forgone'])} | {fn['forgone_pct']:.2f} % | "
          f"{e3(against_old(fn, old))} |")
        table.append((sec, scale, name, fo, fn, old, new))
    w()
    w("## Z4: each row against 0.16.0's")
    w()
    w("| row | moved | traffic | estimate | on the envelope | gap, civil | flips, bound | cert. | gap, pénal | flips, bound | Y1 | ratio | Y2 |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sec, scale, name, fo, fn, old, new in table:
        for v, f in (("0.16.0", fo), ("0.16.1", fn)):
            civil, penal = M[(scale, v)]
            r = row_margins(name, civil, penal, eps[scale])
            mv = "" if v == "0.16.0" else str(len(moved(old, new)))
            c, p = r["civil"], r["penal"]
            cert = "—" if c is None else ("yes" if c[0] <= eps[scale] else "no")
            y1 = "—" if (c is None or p is None or cert != "yes") else ("holds" if p[0] <= eps[scale] and p[1] == 0 else "fails")
            ratio = None if (c is None or p is None) else p[0] / c[0]
            y2 = "—" if ratio is None else ("holds" if 0.5 <= ratio <= 2.0 else "fails")
            w(f"| {sec}, {v} | {mv} | {f['ratio']:.2f}× | {e3(f['estimate'])} | {e3(f['measured'])} | "
              f"{e3(c[0]) if c else '—'} | {f'{c[1]}, {c[2]}' if c else '—'} | {cert} | {e3(p[0]) if p else '—'} | "
              f"{f'{p[1]}, {p[2]}' if p else '—'} | {y1} | {'—' if ratio is None else f'{ratio:.2f}×'} | {y2} |")
    w()
    w("## The decisions that moved")
    w()
    for sec, scale, name, fo, fn, old, new in table:
        mv = moved(old, new)
        w(f"- **{sec}** ({len(mv)}): " + ", ".join(f"`{k}` {a}→{b}" for k, a, b in mv))
    w()
    w("## Files read")
    w()
    for p in sorted(set(READ)):
        w(f"- `{p}` {hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]}")


if __name__ == "__main__":
    sys.exit(main())
