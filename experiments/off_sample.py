#!/usr/bin/env python3
"""The certificate off-sample against the size of the calibration sample
(D156), what experiments/off_sample.sh hands to Python: the standard
library only.

  merge OUT ENV...            fold envelopes of disjoint windows into one:
                              magnitudes, rounding errors and sums of terms
                              by maximum, minima by minimum, samples added,
                              the data slice from the first's start; the
                              element counts left out, since a site counts
                              its batch's elements or the model's (the mask)
                              and an envelope does not say which, so the
                              fold is an envelope to compare, not to solve on
  same A B                    whether two envelopes hold the same figures,
                              field by field, bit for bit, on the fields both
                              have (the day measured and the data slice
                              apart)
  unique OUT --sr SR          the distinct realizations of OUT/solve/*.json,
                              each under a name its content gives, written to
                              OUT/margins/realizations/; on stdout per line
                              the name, the solves that found it and the name
                              of D155's row with the same realization, if any
  report OUT --sizes … --budgets … --forms … --sr SR
                              the report, in Markdown on stdout

The gap of a realization on a set of windows is the largest elementwise gap
of its native logits to the exact ones over those windows' positions
(margins.py native's per-position maxima); in sample is windows 0 … N − 1,
the envelope's, out of sample the 461 − N others.
"""

import hashlib
import json
import math
import os
import re
import sys
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tuning"))
import margins  # noqa: E402  (k_clopper_pearson, eps_star)
import search  # noqa: E402  (parse_space, BYTES)

T = 256
DELTAS = [1e-2, 1e-3, 1e-4]
ALPHA = 0.05
FIELDS = ("sites", "minima", "rounding", "elems", "per_sample", "ops", "terms")


def load(path):
    return json.load(open(path))


def read(path, default=""):
    try:
        return open(path).read()
    except OSError:
        return default


def sci(x, d=2):
    return "—" if x is None else f"{x:.{d}e}"


# ---------------------------------------------------------------------------
# merge, same


def merge(out, paths):
    envs = [load(p) for p in paths]
    m = json.loads(json.dumps(envs[0]))
    for e in envs[1:]:
        for k in ("program", "entry", "bits", "dims", "consts", "weights"):
            if e.get(k) != m.get(k):
                sys.exit(f"merge: {k} differs: {e.get(k)} against {m.get(k)}")
        for s, v in e["sites"].items():
            m["sites"][s] = max(m["sites"].get(s, 0.0), v)
        for s, v in e.get("minima", {}).items():
            m["minima"][s] = min(m["minima"].get(s, math.inf), v)
        for s, r in e.get("rounding", {}).items():
            q = m["rounding"].setdefault(s, {"f64": 0.0, "f32": 0.0, "bf16": 0.0})
            for d in ("f64", "f32", "bf16"):
                q[d] = max(q[d], r[d])
        for s, op in e.get("ops", {}).items():
            m["ops"].setdefault(s, op)
        for s, t in e.get("terms", {}).items():
            q = m["terms"].setdefault(s, {"abs": 0.0, "rss": 0.0})
            q["abs"] = max(q["abs"], t["abs"])
            q["rss"] = max(q["rss"], t["rss"])
            if "norms" in t:
                n = q.setdefault("norms", [])
                n.extend([0.0] * (len(t["norms"]) - len(n)))
                for i, v in enumerate(t["norms"]):
                    n[i] = max(n[i], v)
        m["samples"] += e["samples"]
    # One sample's counts (0.16.0, D158) are the same in every envelope of
    # the same program: an envelope of one sample has them as its elems.
    per = {}
    for e in envs:
        for s, n in (e.get("per_sample") or (e.get("elems", {}) if e["samples"] == 1 else {})).items():
            per[s] = max(per.get(s, 0), n)
    if per:
        m["per_sample"] = per
    m.pop("elems", None)
    if m.get("data"):
        m["data"]["samples"] = m["samples"]
    json.dump(m, open(out, "w"), indent=2, ensure_ascii=False)


def same(a, b):
    A, B = load(a), load(b)
    fields = [k for k in FIELDS if k in A and k in B]
    diff = [k for k in fields if A.get(k) != B.get(k)]
    if A["samples"] != B["samples"]:
        diff.append("samples")
    name = f"{os.path.basename(a)} ({A['samples']} windows) against {b} ({B['samples']} windows)"
    if diff:
        detail = []
        for k in diff:
            if k == "samples":
                continue
            x, y = A.get(k, {}), B.get(k, {})
            d = [s for s in set(x) | set(y) if x.get(s) != y.get(s)]
            detail.append(f"{k}: {len(d)} of {len(set(x) | set(y))} differ ({', '.join(sorted(d)[:4])}{'…' if len(d) > 4 else ''})")
        print(f"{name}: NOT the same: {'; '.join(detail) or 'samples'}")
        return 1
    print(f"{name}: the same, {', '.join(fields)} bit for bit")
    return 0


# ---------------------------------------------------------------------------
# unique


def key(real):
    return json.dumps({k: real.get(k) for k in ("default", "storage", "params", "order", "quant")}, sort_keys=True)


def digest(real):
    return "r" + hashlib.sha256(key(real).encode()).hexdigest()[:8]


def unique(out, sr):
    known = {}
    for f in sorted(os.listdir(os.path.join(sr, "solve"))) if sr else []:
        if f.endswith(".json"):
            known.setdefault(key(load(os.path.join(sr, "solve", f))), "solver-" + f[:-5])
    rdir = os.path.join(out, "margins", "realizations")
    os.makedirs(rdir, exist_ok=True)
    found = {}
    for f in sorted(os.listdir(os.path.join(out, "solve"))):
        if not f.endswith(".json"):
            continue
        real = load(os.path.join(out, "solve", f))
        found.setdefault(digest(real), (real, []))[1].append(f[:-5])
    for name, (real, solves) in sorted(found.items()):
        p = os.path.join(rdir, name + ".json")
        if not os.path.exists(p):
            json.dump(real, open(p, "w"), indent=2, ensure_ascii=False)
        print(f"{name}\t{','.join(solves)}\t{known.get(key(real), '')}")


# ---------------------------------------------------------------------------
# report


class Rows:
    """The exact margins and every realization's per-position rows."""

    def __init__(self, mdir):
        self.dir = mdir
        self.chunks = sorted(f[:-4] for f in os.listdir(os.path.join(mdir, "exact")) if f.endswith(".tsv"))
        self.top1, self.margin = [], []
        for c in self.chunks:
            for line in open(os.path.join(mdir, "exact", c + ".tsv")):
                i, _, m, _ = line.split("\t")
                self.top1.append(int(i))
                self.margin.append(Decimal(m))
        self.n = len(self.margin)
        self.windows = self.n // T
        self.cache = {}
        self.star = {}

    def of(self, name):
        if name not in self.cache:
            gap = []
            for c in self.chunks:
                for line in open(os.path.join(self.dir, "native", name, c + ".tsv")):
                    gap.append(float(line.split("\t")[0]))
            dec = [int(x) for x in open(os.path.join(self.dir, "decisions", name + ".txt"))][: self.n]
            assert len(gap) == self.n and len(dec) == self.n, name
            win = [max(gap[w * T:(w + 1) * T]) for w in range(self.windows)]
            self.cache[name] = (gap, dec, win)
        return self.cache[name]

    def eps_star(self, lo, hi):
        """ε*(δ), corrected (Clopper–Pearson, α = 0.05), from the margins of
        windows lo … hi − 1."""
        if (lo, hi) not in self.star:
            ms = sorted(self.margin[lo * T:hi * T])
            self.star[(lo, hi)] = {d: float(margins.eps_star(ms, margins.k_clopper_pearson(len(ms), d, ALPHA))) for d in DELTAS}
        return self.star[(lo, hi)]

    def judge(self, name, N):
        gap, dec, win = self.of(name)
        cut = N * T
        g_in = max(gap[:cut])
        g_out = max(gap[cut:])
        out_pos = range(cut, self.n)
        flips = sum(1 for p in out_pos if dec[p] != self.top1[p])
        two = 2 * Decimal(repr(g_out))
        bound = sum(1 for p in out_pos if self.margin[p] < two)
        star = self.eps_star(N, self.windows)
        above = sum(1 for w in win[N:] if w > g_in)
        return {"in": g_in, "out": g_out, "ratio": g_out / g_in if g_in > 0 else math.inf, "flips": flips, "bound": bound,
                "n_out": self.n - cut, "cert": {d: g_out <= star[d] for d in DELTAS}, "star": star,
                "above": above, "windows_out": self.windows - N,
                "flips_in": sum(1 for p in range(cut) if dec[p] != self.top1[p])}


def parse_solve(text):
    sv = re.search(r"sites\s+(\d+) narrowable, (\d+) assigned", text)
    cost = re.search(r"cost\s+(\S+) against (\S+) all-f64 \(([0-9.]+)x\)", text)
    est = re.search(r"estimate\s+(\S+)", text)
    mea = re.search(r"measured\s+(\S+) against the exact reference, after (\d+) widening", text)
    # D155's line, and 0.16.0's with the margin between (D161).
    nar = re.search(r"(\d+) decision\(s\) narrowed one step by measurement in (\d+) pass\(es\)[^;]*; (\d+) measurements", text)
    held = re.search(r"held out\s+(?:(\d+) narrowing\(s\) undone|every narrowing undone \((\d+)\))", text)
    return {
        "sites": int(sv.group(1)), "assigned": int(sv.group(2)),
        "cost_own": float(cost.group(3)),
        "bytes": float(cost.group(1)), "baseline": float(cost.group(2)),
        "estimate": float(est.group(1)),
        "measured": float(mea.group(1)), "widening": int(mea.group(2)),
        "narrowed": int(nar.group(1)) if nar else 0, "passes": int(nar.group(2)) if nar else 0,
        "measurements": int(nar.group(3)) if nar else int(mea.group(2)) + 1,
        "undone": int(held.group(1) or held.group(2)) if held else 0,
    }


def report(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="off_sample.py report")
    ap.add_argument("out")
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--forms", required=True)
    ap.add_argument("--sr", default="")
    a = ap.parse_args(argv)
    O = a.out
    sizes = [int(x) for x in a.sizes.split()]
    budgets = a.budgets.split()
    forms = a.forms.split()
    FORM = {"unit-off": "unit/off", "measured-off": "measured/off", "measured-measured": "measured/narrow"}

    sites, params = search.parse_space(os.path.join(O, "space.txt"))
    elems = dict(sites + params)
    snames, pnames = [n for n, _ in sites], [n for n, _ in params]
    pset = set(pnames)

    def typ(r, n):
        return (r.get("params", {}) if n in pset else r.get("storage", {})).get(n, r.get("default", "f64"))

    def traffic(r):
        return sum(e * search.BYTES[typ(r, n)] for n, e in elems.items())

    base = traffic({"default": "f64"})

    def counts(r):
        c = {"f64": 0, "f32": 0, "bf16": 0}
        for n in snames + pnames:
            c[typ(r, n)] += 1
        return c

    names = {}
    for line in read(os.path.join(O, "realizations.tsv")).splitlines():
        name, solves, _ = (line.split("\t") + [""])[:3]
        for s in solves.split(","):
            names[s] = name
    rows = Rows(os.path.join(O, "margins"))
    env = {n: load(os.path.join(O, "env", f"{n}.json")) for n in sizes}
    pop = load(os.path.join(O, "env", "all.json")) if os.path.exists(os.path.join(O, "env", "all.json")) else None

    out = []
    w = out.append
    w("# The certificate off-sample against the size of the calibration sample (D156)\n")
    w(f"{read(os.path.join(O, 'machine.txt')).strip().replace(chr(10), '; ')}. The program: `{env[sizes[0]]['program']}`, "
      f"`{env[sizes[0]]['entry']}`, at {env[sizes[0]]['bits']} bits; the envelopes on the first N held-out windows of "
      f"`{env[sizes[0]]['data']['path']}`; the realizations judged on all {rows.windows} ({rows.n} positions), "
      "the gap split by window index.\n")

    # --- the envelopes ---------------------------------------------------------------
    w("## The envelopes\n")
    for f in ("check-4.txt", "check-32.txt"):
        t = read(os.path.join(O, "env", f)).strip()
        if t:
            w(f"- {t}")
    w("")

    def grow(a, b, field="sites", sub=None):
        """Sites whose figure is larger in b than in a, and the largest ratio b/a."""
        x, y = a.get(field, {}), b.get(field, {})
        n, worst, at = 0, 1.0, "—"
        for s, va in x.items():
            vb = y.get(s)
            if vb is None:
                continue
            if sub:
                va, vb = va[sub], vb[sub]
            if vb > va:
                n += 1
                r = vb / va if va > 0 else math.inf
                if r > worst:
                    worst, at = r, s
        return n, worst, at

    w("Each envelope against the next (N against 2N) and against the population (all "
      f"{pop['samples'] if pop else '—'} windows): how many of the {len(env[sizes[0]]['sites'])} sites' magnitudes "
      "max|x| grow, the largest ratio and its site; the same for the rounding error in `f32` and `bf16` (D155's measured term).\n")
    w("| N | calibration (note) | vs 2N: sites growing | largest ratio | at | `f32` rounding growing, largest | `bf16` rounding growing, largest | vs all: sites growing | largest ratio | at | `f32` / `bf16` rounding, largest |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    agreement = {}
    for i, n in enumerate(sizes):
        e = env[n]
        tm = read(os.path.join(O, "env", f"{n}.time")).strip().splitlines()
        tm = tm[-1] if tm else "—"
        if i + 1 < len(sizes):
            nxt = env[sizes[i + 1]]
            g, r, at = grow(e, nxt)
            g32, r32, _ = grow(e, nxt, "rounding", "f32")
            gb, rb, _ = grow(e, nxt, "rounding", "bf16")
            nx = f"{g} | {r:.3f} | `{at}` | {g32}, {r32:.3f} | {gb}, {rb:.3f}"
        else:
            g = r = None
            nx = "— | — | — | — | —"
        if pop:
            pg, pr, pat = grow(e, pop)
            _, p32, _ = grow(e, pop, "rounding", "f32")
            _, pb, _ = grow(e, pop, "rounding", "bf16")
            px = f"{pg} | {pr:.3f} | `{pat}` | {p32:.3f} / {pb:.3f}"
        else:
            pg = pr = None
            px = "— | — | — | —"
        agreement[n] = {"next": (g, r), "pop": (pg, pr)}
        w(f"| {n} | {tm} | {nx} | {px} |")
    w("")

    # --- the solves ------------------------------------------------------------------
    w("## The solves\n")
    w("Traffic is `cost_at` without a profile (Σ elements × bytes) at the 4-window envelope's elements, the convention of V6 and "
      "D154–D155, so that the rows compare across N: an envelope's element counts are its batch's, N windows' worth at each "
      "stored value and one model's worth at each parameter. The solver's own `cost` line prices one sample's elements "
      "since D158, the same at every N (its ratio is the column 'own'). The in-sample gap is the solver's measurement on the "
      "envelope's N windows; 'native, in' is the same realization's largest gap on those windows in the 461-window rows.\n")
    w(f"| budget | N | form | realization | `f64` / `f32` / `bf16` of {len(snames) + len(pnames)} | traffic | own | estimate | measured, in | native, in | widening rounds | narrowed (passes) | measurements | wall (note) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    S = {}
    for b in budgets:
        for f in forms:
            for n in sizes:
                s = f"{n}-{b}-{f}"
                p = os.path.join(O, "solve", s)
                if not os.path.exists(p + ".json"):
                    w(f"| {b} | {n} | {FORM[f]} | missing | | | | | | | | | | |")
                    continue
                real = load(p + ".json")
                v = parse_solve(read(p + ".txt"))
                name = names.get(s)
                v.update({"real": real, "name": name, "traffic": base / traffic(real), "counts": counts(real)})
                try:
                    v["j"] = rows.judge(name, n)
                except (OSError, AssertionError, KeyError, TypeError):
                    v["j"] = None
                S[(b, f, n)] = v
                c = v["counts"]
                wall = read(p + ".time").strip().splitlines()
                wall = wall[-1].split(",")[0] if wall else "—"
                nar = f"{v['narrowed']} ({v['passes']})" if f.endswith("measured") else "—"
                nin = sci(v["j"]["in"]) if v["j"] else "—"
                w(f"| {b} | {n} | {FORM[f]} | `{name}` | {c['f64']} / {c['f32']} / {c['bf16']} | {v['traffic']:.2f}× | {v['cost_own']:.2f}× | "
                  f"{sci(v['estimate'])} | {sci(v['measured'])} | {nin} | {v['widening']} | {nar} | {v['measurements']} | {wall} |")
    w("")

    # --- the two tables ----------------------------------------------------------------
    w("## In sample and out of sample\n")
    w("Each cell: traffic; the gap on the N calibration windows (in); on the 461 − N others (out); out/in; whether out is "
      "inside the budget.\n")
    for b in budgets:
        w(f"### Budget {b}\n")
        w("| N | " + " | ".join(FORM[f] for f in forms) + " |")
        w("|---|" + "---|" * len(forms))
        for n in sizes:
            cells = []
            for f in forms:
                v = S.get((b, f, n))
                if not v or not v["j"]:
                    cells.append("—")
                    continue
                j = v["j"]
                ok = "holds" if j["out"] <= float(b) else "**fails**"
                cells.append(f"{v['traffic']:.2f}×, {sci(j['in'])}, {sci(j['out'])}, {j['ratio']:.2f}×, {ok}")
            w(f"| {n} | " + " | ".join(cells) + " |")
        w("")

    w("### Out of sample: flips, bound, certification\n")
    w(f"On the 461 − N windows only: the flips of `decisions` against the exact argmax, the bound #{{m < 2ε}} at the "
      "out-of-sample gap, the windows whose own largest gap exceeds the in-sample gap, and certification: the "
      "out-of-sample gap at most ε*(δ) of those windows' own margins (Clopper–Pearson, α = 0.05). "
      "Flips in sample are given apart.\n")
    w("| budget | N | form | out | flips out (of positions) | bound #{m < 2ε} out | windows above in | cert. 10⁻² | cert. 10⁻³ | cert. 10⁻⁴ | ε*(10⁻³) of the out windows | flips in |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in budgets:
        for f in forms:
            for n in sizes:
                v = S.get((b, f, n))
                if not v or not v["j"]:
                    continue
                j = v["j"]
                cert = " | ".join("yes" if j["cert"][d] else "no" for d in DELTAS)
                w(f"| {b} | {n} | {FORM[f]} | {sci(j['out'])} | {j['flips']} ({j['n_out']}) | {j['bound']} | {j['above']} of {j['windows_out']} | "
                  f"{cert} | {sci(j['star'][1e-3], 3)} | {j['flips_in']} |")
    w("")

    # --- the realization against N ---------------------------------------------------
    w("## The realization against N\n")
    ref = 4 if 4 in sizes else sizes[0]
    for b in budgets:
        for f in forms:
            if (b, f, ref) not in S:
                continue
            r0 = S[(b, f, ref)]["real"]
            parts = []
            for n in sizes:
                if n == ref or (b, f, n) not in S:
                    continue
                r = S[(b, f, n)]["real"]
                d = [f"`{x}` {typ(r0, x)} → {typ(r, x)}" for x in snames + pnames if typ(r, x) != typ(r0, x)]
                parts.append(f"N = {n}: " + (f"{len(d)} differ: " + ", ".join(d) if d else "the same"))
            w(f"- at {b}, {FORM[f]}, against N = {ref}: " + "; ".join(parts) + ".")
    w("")
    # V6 and D155 at N = 4
    if a.sr:
        for b in budgets:
            for f in forms:
                p = os.path.join(a.sr, "solve", f"{b}-{f}.json")
                if (b, f, 4) in S and os.path.exists(p):
                    w(f"- N = 4, {b}, {FORM[f]} against D155's: {'the same realization' if key(load(p)) == key(S[(b, f, 4)]['real']) else 'NOT the same'}.")
    w("")

    # --- a fixed realization against N ------------------------------------------------
    w("## A fixed realization against N\n")
    w("The realizations solved on the 4-window envelope (V6's and D155's), held fixed, their gap over the first N windows "
      "against the gap over the others: how fast the largest gap of a sample of N windows approaches the population's, "
      "apart from what the solver does with the sample.\n")
    ks = sorted(set(sizes) | {64, 128, 256})
    ks = [k for k in ks if k < rows.windows]
    w("| budget | form | " + " | ".join(f"N = {k}" for k in ks) + f" | all {rows.windows} |")
    w("|---|---|" + "---|" * (len(ks) + 1))
    for b in budgets:
        for f in forms:
            v = S.get((b, f, 4))
            if not v or not v["name"]:
                continue
            try:
                gap, _, win = rows.of(v["name"])
            except (OSError, AssertionError):
                continue
            cells = []
            for k in ks:
                g_in, g_out = max(win[:k]), max(win[k:])
                cells.append(f"{sci(g_in)} → {g_out / g_in:.2f}×")
            worst = max(range(rows.windows), key=lambda i: win[i])
            above = sorted(win, reverse=True)
            cells.append(f"{sci(max(win))}, window {worst}; {sum(1 for x in win if x > 0.9 * above[0])} within 10 %")
            w(f"| {b} | {FORM[f]} | " + " | ".join(cells) + " |")
    w("")
    w("Each cell: the gap over the first N windows, and the ratio of the gap over the others to it; the last, the largest "
      "gap over all the windows, the window it is in and how many windows come within 10 % of it.\n")

    # --- the figures the sentences quote -----------------------------------------------
    w("## The figures the sentences quote\n")
    dis = [(abs(v["measured"] - v["j"]["in"]) / v["j"]["in"], k) for k, v in S.items() if v["j"] and v["j"]["in"] > 0]
    if dis:
        d, (b, f, n) = max(dis)
        over = [f"{b2}, N = {n2}, {FORM[f2]}" for (b2, f2, n2), v in S.items() if v["j"] and v["measured"] <= float(b2) < v["j"]["in"]]
        w(f"- The solver's measurement (the kernel interpreter; until D159 on the system's `exp`, `log`, `tanh`, since then on "
          f"the `libm` crate's, the native backend's) against the same realization's native gap on the same windows: at most {100 * d:.1f} % apart ({b}, N = {n}, {FORM[f]}); "
          f"{len(dis)} solves compared. Inside the budget by the solver's measurement and outside it natively on the same windows: "
          + (", ".join(over) if over else "none") + ".")
    for f in forms:
        rs = [(S[(b, f, n)]["j"]["ratio"], b, n) for b in budgets for n in sizes if (b, f, n) in S and S[(b, f, n)]["j"]]
        if rs:
            r, b, n = max(rs)
            w(f"- {FORM[f]}: the largest out/in ratio {r:.2f}× (at {b}, N = {n}); "
              + ", ".join(f"{b2} N={n2}: {r2:.2f}×" for r2, b2, n2 in sorted(rs, key=lambda t: (t[1], t[2]))) + ".")
    w("")
    json.dump({f"{n}-{b}-{f}": {k: v for k, v in S[(b, f, n)].items() if k != "real"} for (b, f, n) in S},
              open(os.path.join(O, "report.json"), "w"), indent=1, default=str)
    print("\n".join(out))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "merge":
        merge(rest[0], rest[1:])
    elif cmd == "same":
        sys.exit(same(*rest))
    elif cmd == "unique":
        import argparse
        ap = argparse.ArgumentParser(prog="off_sample.py unique")
        ap.add_argument("out")
        ap.add_argument("--sr", default="")
        u = ap.parse_args(rest)
        unique(u.out, u.sr)
    elif cmd == "report":
        report(rest)
    else:
        sys.exit(__doc__)
