#!/usr/bin/env python3
"""The report of experiments/solver_rounding.sh (D155), in Markdown on
stdout: the sites whose rounding term the measurement changes, the solves in
their four forms, the 461 windows, the masks.

  solver_rounding.py OUT --budgets "4.67e-4 1e-6" --tuning TUNING --weights W

TUNING is D154's output (`make precision-tuning`): its export names the
weights' leaves, its searches' realizations are the rows set beside these.
The standard library only.
"""

import argparse
import array
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tuning"))
import search  # noqa: E402  (D154's parse_space and BYTES)

U = {"f64": 2.0 ** -53, "f32": 2.0 ** -24, "bf16": 2.0 ** -8}
FORMS = [("unit", "off"), ("measured", "off"), ("unit", "measured"), ("measured", "measured")]


def read(path, default=""):
    try:
        return open(path).read()
    except OSError:
        return default


def sci(x, d=2):
    return "—" if x is None else f"{x:.{d}e}"


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    lsb = (bits >> 16) & 1
    return struct.unpack("<f", struct.pack("<I", (((bits + 0x7FFF + lsb) >> 16) << 16) & 0xFFFFFFFF))[0]


def weights_leaves(export, path):
    """The checkpoint's leaves, by name, in the order the C export reads
    them (`main.c`'s `WEIGHTS` sizes, `model.h`'s names)."""
    h = read(os.path.join(export, "model.h"))
    names = re.findall(r"const double \*theta_\w+; /\* θ\.(\S+) :", h)
    sizes = [int(n) for n in re.findall(r"\{0, (\d+)\}", re.search(r"WEIGHTS\[\] = \{(.*?)\};", read(os.path.join(export, "main.c"))).group(1))]
    with open(path, "rb") as f:
        f.read(6)
        (n,) = struct.unpack("<I", f.read(4))
        f.read(n)
        f.read(8)
        data = array.array("d")
        data.frombytes(f.read())
    out, k = {}, 0
    for name, size in zip(names, sizes):
        out[name] = data[k:k + size]
        k += size
    return out


def counts(r, snames, pnames):
    c = {"f64": 0, "f32": 0, "bf16": 0}
    for n in snames:
        c[r.get("storage", {}).get(n, r.get("default", "f64"))] += 1
    for n in pnames:
        c[r.get("params", {}).get(n, r.get("default", "f64"))] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--tuning", required=True)
    ap.add_argument("--weights", required=True)
    a = ap.parse_args()
    O = a.out
    budgets = a.budgets.split()
    env = json.load(open(os.path.join(O, "env_logits.json")))
    sites, params = search.parse_space(os.path.join(O, "space.txt"))
    elems = dict(sites + params)
    snames, pnames = [n for n, _ in sites], [n for n, _ in params]
    pset = set(pnames)

    def typ(r, n):
        return (r.get("params", {}) if n in pset else r.get("storage", {})).get(n, r.get("default", "f64"))

    def traffic(r):
        return sum(e * search.BYTES[typ(r, n)] for n, e in elems.items())

    base = traffic({"default": "f64"})
    bounds = dict(l.split()[:2] for l in read(os.path.join(O, "bounds.txt")).splitlines() if len(l.split()) >= 2)
    out = []
    w = out.append
    w("# The estimate's rounding term measured on the envelope, and a narrowing step (D155)\n")
    w(f"{read(os.path.join(O, 'machine.txt')).strip().replace(chr(10), '; ')}. The envelope: `{env['program']}`, `{env['entry']}`, "
      f"{env['samples']} samples at {env['bits']} bits ({read(os.path.join(O, 'calibrate.time')).strip()}).\n")

    # --- the term ------------------------------------------------------------------
    w("## Where the measured term departs from `u·max|x|` by more than 10×\n")
    w("The sites of the envelope and the parameters (from the weights, as the solver reads them) whose largest rounding error in `f32` or `bf16` is below a tenth of `u(d)·max|x|`.\n")
    w("| site or parameter | elements | max\\|x\\| | `f32`: max\\|fl(x) − x\\| | ÷ u·max\\|x\\| | `bf16`: max\\|fl(x) − x\\| | ÷ u·max\\|x\\| | the values |")
    w("|---|---|---|---|---|---|---|---|")
    rows = []
    for s_, m in env["sites"].items():
        r = env.get("rounding", {}).get(s_)
        if not r or m <= 0:
            continue
        q32, qb = r["f32"] / (U["f32"] * m), r["bf16"] / (U["bf16"] * m)
        if min(q32, qb) < 0.1:
            what = "every value exact" if r["bf16"] == 0 else ""
            rows.append((s_, env.get("elems", {}).get(s_, 0), m, r["f32"], q32, r["bf16"], qb, env.get("ops", {}).get(s_, ""), what))
    leaves = weights_leaves(os.path.join(a.tuning, "c-f64"), a.weights)
    for name, vals in leaves.items():
        m = max(abs(x) for x in vals)
        e32 = max(abs(x - f32(x)) for x in vals)
        eb = max(abs(x - bf16(f32(x))) for x in vals)
        if m > 0 and min(e32 / (U["f32"] * m), eb / (U["bf16"] * m)) < 0.1:
            rows.append((name, len(vals), m, e32, e32 / (U["f32"] * m), eb, eb / (U["bf16"] * m), "parameter", ""))
    for r in sorted(rows, key=lambda r: r[6]):
        op = f"`{r[7]}`" if r[7] else ""
        w(f"| `{r[0]}` | {r[1]} | {r[2]:.4g} | {sci(r[3])} | {r[4]:.3f} | {sci(r[5])} | {r[6]:.3f} | {op} {r[8]} |")
    w("")

    # --- the solves ------------------------------------------------------------------
    w("## The solves\n")
    w(f"| budget | `--rounding` | `--narrow` | stored values away from `f64` | `f64` / `f32` / `bf16` of {len(snames) + len(pnames)} | traffic vs all-`f64` | estimate | measured, the envelope's windows | widening rounds | narrowed (passes) | measurements | wall (note) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    solved = {}
    for b in budgets:
        for r, n in FORMS:
            p = os.path.join(O, "solve", f"{b}-{r}-{n}")
            t = read(p + ".txt")
            real = json.load(open(p + ".json"))
            solved[(b, r, n)] = real
            c = counts(real, snames, pnames)
            sv = re.search(r"sites\s+(\d+) narrowable, (\d+) assigned", t)
            est = re.search(r"estimate\s+(\S+)", t)
            mea = re.search(r"measured\s+(\S+) against the exact reference, after (\d+) widening", t)
            # Since D161 the line names the margin before the count.
            nar = re.search(r"(\d+) decision\(s\) narrowed one step by measurement in (\d+) pass\(es\)[^;]*; (\d+) measurements", t)
            wall = read(p + ".time").strip().split(",")[0]
            narrowed = f"{nar.group(1)} ({nar.group(2)})" if nar else "—"
            meas = nar.group(3) if nar else str(int(mea.group(2)) + 1)
            w(f"| {b} | {r} | {n} | {sv.group(2)} of {sv.group(1)} | {c['f64']} / {c['f32']} / {c['bf16']} | {base / traffic(real):.2f}× | {est.group(1)} | {mea.group(1)} | {mea.group(2)} | {narrowed} | {meas} | {wall} |")
    w("")
    for b in budgets:
        d154 = json.load(open(os.path.join(a.tuning, "search", f"craft-params-{b}", "final.json")))
        for r, n in FORMS:
            real = solved[(b, r, n)]
            diff = [f"`{x}` {typ(real, x)} (searches: {typ(d154, x)})" for x in snames + pnames if typ(real, x) != typ(d154, x)]
            w(f"- at {b}, `--rounding {r} --narrow {n}` against D154's searches' realization: "
              + (f"{len(diff)} decisions differ: " + ", ".join(diff) if diff else "the same realization") + ".")
    w("")

    # --- the 461 windows ---------------------------------------------------------------
    w("## Judged on the 461 held-out windows\n")
    head, cells = None, {}
    for line in read(os.path.join(O, "margins.md")).splitlines():
        if line.startswith("| realization"):
            head = [c.strip() for c in line.strip("|").split("|")]
        elif head and line.startswith("| ") and not line.startswith("|---"):
            c = [x.strip() for x in line.strip("|").split("|")]
            cells[c[0]] = dict(zip(head, c))
        elif head and not line.startswith("|"):
            head = None if cells else head
    w("| realization | gap on the logits | flips | bound #{m < 2ε} | cert. 10⁻² | cert. 10⁻³ | cert. 10⁻⁴ | traffic vs all-`f64` | bound `cond` |")
    w("|---|---|---|---|---|---|---|---|---|")
    for line in read(os.path.join(O, "rows.tsv")).splitlines():
        name, label = line.split("\t")
        c = cells.get(label, {})
        rp = os.path.join(O, "margins", "realizations", name + ".json")
        if os.path.exists(rp):
            tr = f"{base / traffic(json.load(open(rp))):.2f}×"
        else:
            tr = f"{base / traffic({'default': name, 'params': {q: name for q in pnames}}):.2f}×"
        w(f"| {label} | {c.get('gap on the logits', '—')} | {c.get('flips', '—')} | {c.get('bound #{m < 2ε}', '—')} | {c.get('cert. 1e-02', '—')} | "
          f"{c.get('cert. 1e-03', '—')} | {c.get('cert. 1e-04', '—')} | {tr} | {bounds.get(name, '—')} |")
    w("")

    # --- the masks -----------------------------------------------------------------------
    masks = [s_ for s_, op in env.get("ops", {}).items() if op == "dropout_mask" and s_ in elems]
    w("## The dropout masks\n")
    if not masks:
        w("The envelope has no site of `dropout_mask`: the entry is compiled in the inference mode, where `dropout` is the identity (0.16.0, D160).\n")
        print("\n".join(out))
        return
    w(f"The sites of `dropout_mask` in the envelope: {', '.join('`' + m + '`' for m in masks)}; "
      f"{sum(elems[m] for m in masks)} elements of the {sum(elems[n] for n in snames)} the stored values hold.\n")
    w("| realization | the masks' types | their bytes | share of the traffic | traffic without them vs all-`f64` |")
    w("|---|---|---|---|---|")
    for b in budgets:
        for r, n in FORMS:
            real = solved[(b, r, n)]
            mb = sum(elems[m] * search.BYTES[typ(real, m)] for m in masks)
            tot = traffic(real)
            types = ", ".join(sorted({typ(real, m) for m in masks}))
            w(f"| the solver at {b}, {r} / {n} | {types} | {mb} | {100 * mb / tot:.1f} % | {base / (tot - mb):.2f}× |")
        d154 = json.load(open(os.path.join(a.tuning, "search", f"craft-params-{b}", "final.json")))
        mb = sum(elems[m] * search.BYTES[typ(d154, m)] for m in masks)
        tot = traffic(d154)
        w(f"| D154's searches at {b} | {', '.join(sorted({typ(d154, m) for m in masks}))} | {mb} | {100 * mb / tot:.1f} % | {base / (tot - mb):.2f}× |")
    w("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
