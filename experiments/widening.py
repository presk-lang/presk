#!/usr/bin/env python3
"""The tables of experiments/widening.sh (D128), from the outputs it keeps.

    widening.py culprit EXPLAIN.md OUT.json [SITE=TYPE ...]
        The proposal the solve of EXPLAIN.md measured first, with only the
        sites named widened to the types given (none: the proposal itself),
        as a realization; exit 1, writing nothing, when that solve did not
        widen.
    widening.py summary OUT BUDGETS ENTRY:ROWS ...
        The summary, in Markdown, on stdout.

Everything is read from `presk explain`'s §6.3 (the decisions' table and
the moves, one bullet per decision: each narrowing tried, its move number,
what it added to the estimate and the estimate after it; each widening, its
round, from, to, and the gap that forced it), from `presk solve`'s report and
from `presk check`'s. The cost of a realization is the solver's own without a
profile: the bytes of the stored values it decides, each its elements times
its type's width; the script checks that it gives the solver's cost of the
realization the solve ended on before computing the culprit's.
"""
import json
import re
import sys

WIDTH = {"f64": 8, "f32": 4, "bf16": 2}


def section(md):
    """§6.3 of an explain document."""
    m = re.search(r"### 6\.3 .*?(?=\n### 6\.4 )", md, re.S)
    return m.group(0) if m else ""


def decisions(md):
    """The decisions' table: site -> (type, elements)."""
    out = {}
    s = section(md)
    t = s.split("**The decisions**", 1)[1].split("**Why each decision", 1)[0]
    for line in t.splitlines():
        m = re.match(r"\| `([^`]*)` \| [^|]* \| `(\w+)` \| ([0-9]+) \|", line)
        if m:
            out[m.group(1)] = (m.group(2), int(m.group(3)))
    return out


def moves(md):
    """Per decision: the proposal's type, the estimate after each taken
    narrowing (move number -> estimate), and the widening rounds."""
    s = section(md).split("**Why each decision has its type.**", 1)[1]
    s = s.split("**The values rounded elsewhere", 1)[0]
    proposal, spent, rounds = {}, {}, []
    for line in s.splitlines():
        m = re.match(r"- `([^`]*)`: (.*)", line)
        if not m:
            continue
        site, text = m.groups()
        proposal[site] = "f64"
        for to, n, after in re.findall(r"narrowed to `(\w+)` \(move ([0-9]+)\): saves [^,]*, adds [^ ]* to the estimate, then ([0-9.]+e[+-]?[0-9]+)", text):
            proposal[site] = to
            spent[int(n)] = float(after)
        for fr, to, r, gap in re.findall(r"widened from `(\w+)` to `(\w+)` at round ([0-9]+), the gap ([0-9.]+e[+-]?[0-9]+|inf|NaN)", text):
            rounds.append((int(r), site, fr, to, float(gap)))
    rounds.sort()
    return proposal, spent, rounds


def cost(types, dec):
    return sum(n * WIDTH[types.get(site, "f64")] for site, (_, n) in dec.items())


def culprit(argv):
    md = open(argv[0]).read()
    proposal, _, rounds = moves(md)
    if not rounds:
        return 1
    r = dict(proposal)
    for kv in argv[2:]:
        site, t = kv.split("=")
        r[site] = t
    json.dump({"default": "f64", "storage": {k: v for k, v in sorted(r.items()) if v != "f64"}, "order": {"kind": "sequential"}}, open(argv[1], "w"), indent=2)
    return 0


def check(path):
    """`presk check`'s four rules, its measured gap and the magnitude."""
    t = open(path).read()
    num = r"([0-9.e+-]+|inf|NaN)"
    out = {r: float(m.group(1)) for r in ["k", "normwise", "cond", "rss"] if (m := re.search(rf"\b{r}\s+{num}", t))}
    out["measured"] = float(re.search(rf"measured\s+{num}", t).group(1))
    out["mag"] = float(re.search(rf"magnitude\s+{num}", t).group(1))
    return out


def fmt(x):
    return "∞" if x == float("inf") else f"{x:.2e}"


def ratio(a, b):
    if b == 0 or a == float("inf") or b == float("inf"):
        return "—"
    r = a / b
    return f"{r:.2g}×" if r < 10 else f"{r:.1e}×" if r >= 1e4 else f"{r:.0f}×"


def summary(argv):
    out, budgets, entries = argv[0], argv[1].split(), argv[2:]
    ok = True
    for er in entries:
        e, rows = er.split(":")
        print(f"## `{e}`, {rows} rows\n")
        print("Under one type everywhere, the four rules against the measured gap (`presk check --model all --measure`):\n")
        print("| type | `k` | `normwise` | `cond` | `rss` | measured | `cond` / measured | `rss` / measured |")
        print("|---|---|---|---|---|---|---|---|")
        for t in ["f64", "f32", "bf16"]:
            c = check(f"{out}/check_{e}_{t}.txt")
            print(f"| `{t}` | {fmt(c['k'])} | {fmt(c['normwise'])} | {fmt(c['cond'])} | {fmt(c['rss'])} | {fmt(c['measured'])} | {ratio(c['cond'], c['measured'])} | {ratio(c['rss'], c['measured'])} |")
        print()
        print("The solve at each budget (relative to the output's magnitude; the absolute budget beside it): the proposal's estimate and measured gap, the widening rounds, where it lands.\n")
        print("| budget | absolute | proposal: estimate | proposal: measured | proposal: `cond`, `rss` | rounds | widened, in order (the gap that forced each) | final gap | final cost vs all-`f64` | culprit only: gap, cost |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for b in budgets:
            s = open(f"{out}/solve_{e}_{b}.txt").read()
            md = open(f"{out}/explain_{e}_{b}/explain.md").read()
            dec = decisions(md)
            proposal, spent, rounds = moves(md)
            absolute = float(re.search(r"\(([0-9.e+-]+) absolute", s).group(1))
            final = float(re.search(r"^  measured\s+([0-9.e+-]+|inf)", s, re.M).group(1))
            m = re.search(r"^  cost\s+([0-9.e+-]+) against ([0-9.e+-]+) all-f64", s, re.M)
            c_final, c_base = float(m.group(1)), float(m.group(2))
            solved = json.load(open(f"{out}/solved_{e}_{b}.json")).get("storage", {})
            if abs(cost(solved, dec) - c_final) > 1e-4 * c_final or abs(cost({}, dec) - c_base) > 1e-4 * c_base:
                print(f"\n(the cost of {e} at {b} is not the decisions' bytes: {cost(solved, dec)} against {c_final})\n")
                ok = False
            est = spent[max(spent)] if spent else 0.0
            first = rounds[0][4] if rounds else final
            seq = "; ".join(f"`{site.split('/')[1]}` {fr}→{to} ({fmt(g)})" for _, site, fr, to, g in rounds) or "—"
            pb = ""
            try:
                pc = check(f"{out}/proposal_{e}_{b}.txt")
                pb = f"{fmt(pc['cond'])}, {fmt(pc['rss'])}"
                if abs(pc["measured"] - first) > 1e-3 * first:
                    print(f"\n(the proposal of {e} at {b} measures {pc['measured']} under check, {first} under solve)\n")
                    ok = False
            except FileNotFoundError:
                pass
            cul = ""
            try:
                cc = check(f"{out}/culprit_{e}_{b}.txt")
                cr = json.load(open(f"{out}/culprit_{e}_{b}.json"))["storage"]
                cul = f"{fmt(cc['measured'])}, {c_base / cost(cr, dec):.2f}×" + ("" if cc["measured"] <= absolute else " (over)")
            except FileNotFoundError:
                pass
            print(f"| {b} | {absolute:.2e} | {est:.2e} | {fmt(first)} | {pb} | {len(rounds)} | {seq} | {fmt(final)} | {c_base / c_final:.2f}× | {cul} |")
        print()
    print("The proposal's estimate is the solver's after its last narrowing; its measured gap the first measurement, which forced round 1 when there is one. A culprit's cost is the bytes of the stored values, as the solver prices them without a profile; checked against the solver's own (printed to five digits) on every final realization:", "hold" if ok else "VIOLATED")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    sys.exit({"culprit": culprit, "summary": summary}[cmd](rest))
