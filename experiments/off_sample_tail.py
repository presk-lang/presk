#!/usr/bin/env python3
"""Which decisions of a realization carry its tail of windows (step 0b of the
course-scale certificate study), what experiments/off_sample_tail.sh hands
to Python: the standard library only.

  variants SOLVED OTHER OUT   the realizations to run, from SOLVED (the
                              realization studied) and OTHER (the one it is
                              compared with), written to OUT/<name>.json; on
                              stdout per line the name and what it changes
  diff SOLVED OTHER           the two realizations decision by decision, in
                              Markdown
  report MARGINS --names … [--labels …] [--ref NAME=DIR …] [--tail …] [--controls …]
                              the report, in Markdown on stdout

The variants: SOLVED as is (`as-is`), OTHER as is (`other`), SOLVED with each
of its `bf16` decisions, stored value or parameter, put back to `f32` one at
a time (`f32-<site>`), all of them at once (`f32-all-bf16`), and the
decisions in which SOLVED and OTHER differ put back to OTHER's types, all
at once (`as-other`) and those whose value is not exact in `bf16` alone
(`as-other-inexact`), if any of the differing ones is.

A window's gap is the largest elementwise gap of its native logits to the
exact ones over its T = 256 positions (`margins.py native`'s first column);
in sample is windows 0 … 3, the envelope's.
"""

import argparse
import json
import os
import struct
import sys

T = 256
N_IN = 4


def load(p):
    return json.load(open(p))


def exact_bf16(x):
    """Whether the f64 value x is a bf16 value: its low 16 bits of f32, and
    the f32 conversion, exact."""
    f = struct.unpack("<f", struct.pack("<f", x))[0]
    if f != x:
        return False
    return struct.unpack("<I", struct.pack("<f", f))[0] & 0xFFFF == 0


def decisions(r):
    out = {}
    for k, v in r.get("storage", {}).items():
        out[("storage", k)] = v
    for k, v in r.get("params", {}).items():
        out[("params", k)] = v
    return out


def with_types(r, changes):
    r = json.loads(json.dumps(r))
    for (sec, k), t in changes.items():
        if t is None:
            r[sec].pop(k, None)
        else:
            r.setdefault(sec, {})[k] = t
    return r


def envelope_values(solved_path):
    """The one-element stored values' magnitudes, from the envelope beside
    the realization, if there is one (the solve's `env_logits.json`)."""
    p = os.path.join(os.path.dirname(solved_path), "env_logits.json")
    if not os.path.exists(p):
        return {}
    e = load(p)
    return {k: v for k, v in e.get("sites", {}).items() if e.get("elems", {}).get(k) == 1}


def short(sec, k):
    return (k if sec == "params" else k.replace(".apply/", ".")).replace("%", "").replace("/", ".")


def variants(solved, other, out):
    a, b = load(solved), load(other)
    os.makedirs(out, exist_ok=True)
    one = envelope_values(solved)
    da, db = decisions(a), decisions(b)
    names = []

    def put(name, r, what):
        json.dump(r, open(os.path.join(out, name + ".json"), "w"), indent=1, ensure_ascii=False)
        names.append(name)
        print(f"{name}\t{what}")

    put("as-is", a, "the realization studied")
    put("other", b, "the one compared with")
    narrow = [k for k, t in da.items() if t == "bf16"]
    for k in sorted(narrow):
        put("f32-" + short(*k), with_types(a, {k: "f32"}), f"{k[0]} {k[1]}: bf16 → f32")
    put("f32-all-bf16", with_types(a, {k: "f32" for k in narrow}), f"all {len(narrow)} bf16 decisions → f32")
    common = [k for k in da if k in db and da[k] != db[k]]
    only_a = [k for k in da if k not in db]
    if common or only_a:
        ch = {k: db[k] for k in common}
        ch.update({k: None for k in only_a})
        put("as-other", with_types(a, ch), "the decisions that differ put back to the other's types: "
            + ", ".join(f"{k[1]} {da[k]} → {db.get(k, 'f64')}" for k in common + only_a))
        inexact = [k for k in common if not (k[0] == "storage" and k[1] in one and exact_bf16(one[k[1]]))]
        if 0 < len(inexact) < len(common):
            put("as-other-inexact", with_types(a, {k: db[k] for k in inexact}),
                "only those whose value is not exact in bf16: " + ", ".join(k[1] for k in inexact))


def diff(solved, other):
    a, b = load(solved), load(other)
    da, db = decisions(a), decisions(b)
    one = envelope_values(solved)
    w = print
    for k in ("default", "order", "envelope"):
        w(f"- `{k}`: {'the same' if a.get(k) == b.get(k) else f'{a.get(k)} against {b.get(k)}'}.")
    w("")
    w("| | decision | studied | other | its value, if one element |")
    w("|---|---|---|---|---|")
    for k in sorted(set(da) | set(db)):
        if da.get(k) != db.get(k):
            v = one.get(k[1]) if k[0] == "storage" else None
            ex = "" if v is None else f"{v!r}" + (", exact in bf16" if exact_bf16(v) else "")
            w(f"| {k[0]} | `{k[1]}` | {da.get(k, '—')} | {db.get(k, '—')} | {ex} |")
    same = [k for k in set(da) & set(db) if da[k] == db[k]]
    w("")
    w(f"The same in both: {len(same)} decisions, of which in `bf16`: "
      + (", ".join(f"`{k[1]}`" for k in sorted(same) if da[k] == "bf16") or "none") + ".")


class Rows:
    def __init__(self, mdir, exact):
        self.dir = mdir
        self.exact_top = []
        for c in sorted(f for f in os.listdir(exact) if f.endswith(".tsv")):
            for line in open(os.path.join(exact, c)):
                self.exact_top.append(int(line.split("\t")[0]))

    def of(self, d):
        gap, top = [], []
        for c in sorted(f for f in os.listdir(d) if f.endswith(".tsv")):
            for line in open(os.path.join(d, c)):
                g, t, _ = line.split("\t")
                gap.append(float(g))
                top.append(int(t))
        n = len(gap) // T
        assert len(gap) == len(self.exact_top), (d, len(gap), len(self.exact_top))
        win = [max(gap[i * T:(i + 1) * T]) for i in range(n)]
        flips = sum(1 for p in range(len(top)) if top[p] != self.exact_top[p])
        return gap, win, flips


def med(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def report(argv):
    ap = argparse.ArgumentParser(prog="off_sample_tail.py report")
    ap.add_argument("margins")
    ap.add_argument("--exact", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--ref", nargs="*", default=[], help="NAME=DIR: rows of record to compare with, bit for bit")
    ap.add_argument("--tail", default="406,385,455,387,402")
    ap.add_argument("--controls", default="")
    a = ap.parse_args(argv)
    rows = Rows(a.margins, a.exact)
    tail = [int(x) for x in a.tail.split(",")]
    ctrl = [int(x) for x in a.controls.split(",")] if a.controls else []
    labels = dict(x.split("=", 1) for x in a.labels.split("|") if x) if a.labels else {}
    w = print
    res = {}
    for n in a.names:
        res[n] = rows.of(os.path.join(a.margins, "native", n))
    refs = dict(x.split("=", 1) for x in a.ref)
    if refs:
        w("**Against the rows of record.**")
        for n, d in refs.items():
            same = all(open(os.path.join(a.margins, "native", n, f), "rb").read() == open(os.path.join(d, f), "rb").read()
                       for f in sorted(os.listdir(d)) if f.endswith(".tsv"))
            w(f"- `{n}` against `{d}`: {'the same rows, byte for byte' if same else 'NOT the same rows'}.")
        w("")

    def key(n):
        d = os.path.join(a.margins, "native", n)
        return tuple(open(os.path.join(d, f), "rb").read() for f in sorted(os.listdir(d)) if f.endswith(".tsv"))
    classes = {}
    for n in a.names:
        classes.setdefault(key(n), []).append(n)
    w("**The variants with the same rows, byte for byte:** "
      + "; ".join(", ".join(f"`{n}`" for n in c) for c in classes.values() if len(c) > 1) + ". "
      + f"{len(classes)} distinct rows among {len(a.names)} variants.\n")
    head = ("| variant | in (0–3) | worst (window) | median window | out/in N = 4 | N = 16 | "
            + " | ".join(f"w{t}" for t in tail) + (" | " + " | ".join(f"w{t}" for t in ctrl) if ctrl else "")
            + " | positions > 10⁻⁴ | flips |")
    w(head)
    w("|---" * (head.count("|") - 1) + "|")
    for n in a.names:
        gap, win, flips = res[n]
        worst = max(range(len(win)), key=lambda i: win[i])
        g_in = max(win[:N_IN])
        r4 = max(win[N_IN:]) / g_in
        r16 = max(win[16:]) / max(win[:16])
        cells = [f"{win[t]:.3e}" for t in tail] + [f"{win[t]:.3e}" for t in ctrl]
        w(f"| `{n}`{' ' + labels[n] if n in labels else ''} | {g_in:.3e} | {win[worst]:.3e} ({worst}) | {med(win):.3e} | "
          f"{r4:.2f}× | {r16:.2f}× | " + " | ".join(cells) + f" | {sum(1 for x in gap if x > 1e-4)} | {flips} |")
    w("")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or [""]
    if cmd == "variants":
        variants(*rest)
    elif cmd == "diff":
        diff(*rest)
    elif cmd == "report":
        report(rest)
    else:
        sys.exit(__doc__)
