"""SR3's report (experiments/opt_measured.sh, D195): the e-graph optimizer on
the course GPT, in gap and in time, under all-f32 and all-f64 and the default
layout, at D = 64 and at the course constants.

    python3 experiments/opt_measured.py out/opt-measured 7 > out/opt-measured/report.md

SR3 has no statement and no prediction: it is reported. The rules that fired
are read off the optimizer's report and `presk explain`'s provenance table;
the gap is the relative distance of the loss to the exact loss (the reference
interpreter at 128 bits) on the envelopes' two windows; the time is `presk
bench`'s median call in each of five processes, the median of the five
(D126), and beside it `presk train`'s median step. The loss's launches,
counted by `presk explain` with and without --opt, show where its time goes.
"""
import json
import os
import re
import statistics
import sys
from decimal import Decimal, getcontext

getcontext().prec = 80
ALL_SCALES = {"d64": "D = 64, H = 2, L = 2", "course": "D = 384, H = 6, L = 6"}
PROGRAMS = {"loss": "the loss (`05_gpt.psk`'s `loss`)", "grad": "its gradient (`gpt_grad.psk`'s `loss_grad`)"}
# The rules whose equation `presk explain`'s table does not give, in the words
# of crates/presk-opt/src/rules.rs.
OTHER = {
    "div-into-contraction": "(ein a b) / s = ein (a / s) b = ein a (b / s), s a real scalar (one division either way under operation counts)",
    "field-record": "{…, n = v, …}.n = v",
    "set-record": "{…, n = u, …}[n := v] = {…, n = v, …}",
    "set-zeros-record": "(0_t)[n := v] = the record with v in field n and 0_{t.m} in every other field m",
}
BENCH_ROWS = ["native f64", "native f64 +lanes", "native f32", "native f32 +lanes", "native bf16"]


def read(out, *path):
    return open(os.path.join(out, *path)).read()


def report_lines(text):
    """The optimizer's report: its summary line and, per function, the lets,
    the cost and how saturation stopped."""
    head = re.search(r"^presk-opt: .*$", text, re.M).group(0)[len("presk-opt: "):]
    funcs = {}
    for m in re.finditer(r"^  (\S+): (\d+) → (\d+) lets, cost ([0-9.]+) → ([0-9.]+) \((.*)\)$", text, re.M):
        funcs[m.group(1)] = (int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5)), m.group(6))
    return head, funcs


def provenance(out, key):
    """Per function the optimizer visited in the entry's program, the rules
    that applied and its lets by how they were made; and every rule's equation."""
    d = json.load(open(os.path.join(out, "rules", key, "explain.json")))
    rules, made = {}, {}
    for row in d["links"]:
        child = row["child"]
        if not child.startswith("opt:"):
            continue
        name, _, let = child[4:].partition("/")
        if not let:
            rules[name] = row.get("rules", [])
        else:
            made.setdefault(name, {}).setdefault(row["how"], 0)
            made[name][row["how"]] += 1
    return rules, made, d["rules"]["rewrite"]


def plain(tex):
    """A rule's equation as `presk explain` writes it for the page, in text."""
    tex = re.sub(r"\\code\{(.*?)\}", r"\1", tex)
    return tex.replace("\\_", "_").replace("\\ ", " ").replace("\\,", "·")


def value(out, name):
    """The scalar `presk run` printed, as written."""
    return re.search(r"^ℝ\[\] = (\S+)$", read(out, "gap", name), re.M).group(1)


def rel(x, e):
    return abs(Decimal(x) - Decimal(e)) / abs(Decimal(e))


def flat(x, acc):
    if isinstance(x, dict):
        if "data" in x and "axes" in x:
            flat(x["data"], acc)
        else:
            for k in sorted(x):
                flat(x[k], acc)
    elif isinstance(x, list):
        for y in x:
            flat(y, acc)
    else:
        acc.append(x)
    return acc


def gradient(out, name):
    return flat(json.load(open(os.path.join(out, "gap", name)))["result"], [])


def norms(xs, es):
    """‖x − e‖₂ / ‖e‖₂ and max |x − e| / max |e|."""
    d = [Decimal(a) - Decimal(b) for a, b in zip(xs, es)]
    e = [Decimal(b) for b in es]
    return (sum(v * v for v in d).sqrt() / sum(v * v for v in e).sqrt(),
            max(abs(v) for v in d) / max(abs(v) for v in e))


def bench(out, key):
    """`presk bench`'s median call per row, in ms."""
    rows = {}
    for m in re.finditer(r"^(native \S+(?: \+lanes)?)\s+([0-9.]+)\s+([0-9.]+)\s", read(out, "log", key + ".log"), re.M):
        rows[m.group(1)] = float(m.group(2))
    return rows


def train(out, key):
    text = read(out, "log", key + ".log")
    ms = re.search(r"^presk train: ([0-9.]+) ms a step, median of (\d+)", text, re.M)
    launches = re.search(r"^presk train: (\d+) kernel launches a step, (\d+) without fusion", text, re.M)
    loss = re.search(r"final loss ([0-9.]+)", text)
    opt = re.search(r"^presk-opt: (.*)$", text, re.M)
    return float(ms.group(1)), int(launches.group(1)), int(launches.group(2)), loss.group(1), opt.group(1) if opt else None


def peak_gb(out, key):
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", read(out, "log", key + ".time"))
    return int(m.group(1)) / 1e6


def kernels(out, key):
    """`presk explain`'s counts on the loss: launches before and after fusion,
    distinct kernels, and the values that reach memory (its fusion section),
    each as (value, line, why)."""
    text = read(out, "kernels", key, "explain.md")
    fusion = re.search(r"^\| Fusion \| (\d+) launches without it, (\d+) with it", text, re.M)
    distinct = re.search(r"^\| Kernels \| (\d+) distinct", text, re.M)
    section = re.search(r"^## 7 Fusion$(.*?)^## ", text, re.M | re.S).group(1)
    stored = [tuple(c.strip().strip("`") for c in (m.group(1), m.group(2), m.group(3)))
              for m in re.finditer(r"^\| (`[^|]*`) \| (`[^|]*`) \| [^|]* \| ([^|]*) \|$", section, re.M)]
    return int(fusion.group(1)), int(fusion.group(2)), int(distinct.group(1)), stored


def sums(out):
    s = {}
    p = os.path.join(out, "ckpt.sha256")
    for line in open(p) if os.path.exists(p) else []:
        digest, path = line.split()
        s[path.split("/")[0]] = digest
    return s


def main():
    global SCALES
    out, repeat = sys.argv[1], int(sys.argv[2])
    SCALES = {s: c for s, c in ALL_SCALES.items() if os.path.exists(os.path.join(out, "rules", f"{s}-loss.txt"))}
    print("# SR3: the e-graph optimizer on the course GPT, in gap and in time\n")
    print(read(out, "machine.txt").strip().replace("\n", "  \n"), "\n")
    print("Reported: no statement, no prediction (SR3).\n")

    print("## What the optimizer rewrites (B = 16, T = 256)\n")
    equations = {}
    for p, what in PROGRAMS.items():
        print(f"### {what[0].upper() + what[1:]}\n")
        reps = {s: report_lines(read(out, "rules", f"{s}-{p}.txt")) for s in SCALES}
        provs = {s: provenance(out, f"{s}-{p}") for s in SCALES}
        for s in SCALES:
            print(f"- {s} ({SCALES[s]}): {reps[s][0]}.")
            equations.update(provs[s][2])
            equations.update({r: OTHER.get(r, "(no equation given)") for rs in provs[s][0].values() for r in rs if r not in equations})
        names = list(dict.fromkeys([n for s in SCALES for n in reps[s][1]] +
                                   [n for s in SCALES for n, r in provs[s][0].items() if r]))
        same_rules = all(provs[s][0] == provs["d64"][0] for s in SCALES)
        print("\n| function | " + " | ".join(f"lets, {s}" for s in SCALES) + " | " +
              " | ".join(f"cost, {s}" for s in SCALES) + " | how saturation stopped (d64) | the lets (d64): unchanged, rewritten, introduced | the rules that applied |")
        print("|---|" + "---|" * (2 * len(SCALES) + 3))
        for n in names:
            lets = [f"{reps[s][1][n][0]} → {reps[s][1][n][1]}" if n in reps[s][1] else "unchanged" for s in SCALES]
            cost = [f"{reps[s][1][n][2]:.2f} → {reps[s][1][n][3]:.2f}" if n in reps[s][1] else "unchanged" for s in SCALES]
            stop = reps["d64"][1][n][4] if n in reps["d64"][1] else "—"
            rules, made = provs["d64"][0], provs["d64"][1]
            if n in rules:
                m = made.get(n, {})
                kinds = f"{m.get('unchanged', 0)}, {m.get('rewritten', 0)}, {m.get('introduced', 0)}"
                applied = ", ".join(f"`{r}`" for r in rules[n]) or "none"
            else:
                kinds, applied = "—", "not in the entry's program"
            print(f"| `{n}` | " + " | ".join(lets) + " | " + " | ".join(cost) + f" | {stop} | {kinds} | {applied} |")
        print("\n*Unchanged*: the optimizer's report lists only the functions whose lets or cost changed; rules applied in these, "
              "and `presk explain`'s table counts the lets they rewrote. *None*: no rule applied; the lets the function lost "
              "are the e-graph's own doing, a definition outside the result's term not written back and identical terms one class.")
        print(f"\nThe rules each function applied are the same at every scale: {'yes' if same_rules else '**no**'}.\n")
    print("The rules that applied, as `presk explain` states them (those it does not, as `presk-opt`'s rules do):\n")
    for name in sorted(equations):
        print(f"- `{name}`: {plain(equations[name])}")

    print("\n## The gap to the exact loss (windows 0 and 1 of the held-out tenth, the trained weights)\n")
    print("| scale | row | loss | relative gap to the exact loss | the same bits as without `--opt` |")
    print("|---|---|---|---|---|")
    for s in SCALES:
        e = value(out, f"{s}-loss-exact-plain.txt")
        for r in ("f32", "f64"):
            for o in ("plain", "opt"):
                x = value(out, f"{s}-loss-{r}-{o}.txt")
                same = "" if o == "plain" else ("yes" if x == value(out, f"{s}-loss-{r}-plain.txt") else "**no**")
                print(f"| {s} | {r}{' `--opt`' if o == 'opt' else ''} | {x} | {rel(x, e):.4e} | {same} |")
        eo = value(out, f"{s}-loss-exact-opt.txt")
        print(f"| {s} | exact (128 bits) | {e} | | |")
        print(f"| {s} | exact `--opt` | {eo} | {rel(eo, e):.3e} | {'yes' if eo == e else 'no'} |")
    print()
    for s in SCALES:
        e = Decimal(value(out, f"{s}-loss-exact-plain.txt"))
        x = Decimal(value(out, f"{s}-loss-f64-plain.txt"))
        print(f"- {s}: all-`f64`'s loss is the exact loss rounded to the nearest double: {'yes' if float(e) == float(x) else 'no'}.")

    print("\n### Beside: the gradient at d64, the same windows\n")
    ex = gradient(out, "d64-grad-exact-plain.json")
    print(f"{len(ex)} elements. Against the exact gradient (128 bits):\n")
    print("| row | ‖g − g*‖₂ / ‖g*‖₂ | max \\|g − g*\\| / max \\|g*\\| | elements with the same bits as without `--opt` |")
    print("|---|---|---|---|")
    for r in ("f32", "f64"):
        base = gradient(out, f"d64-grad-{r}-plain.json")
        for o in ("plain", "opt"):
            g = base if o == "plain" else gradient(out, f"d64-grad-{r}-opt.json")
            n2, ninf = norms(g, ex)
            same = "" if o == "plain" else f"{sum(1 for a, b in zip(g, base) if a == b)} of {len(g)}"
            print(f"| {r}{' `--opt`' if o == 'opt' else ''} | {n2:.4e} | {ninf:.4e} | {same} |")
    eo = gradient(out, "d64-grad-exact-opt.json")
    n2, ninf = norms(eo, ex)
    print(f"| exact `--opt` | {n2:.3e} | {ninf:.3e} | |")

    print(f"\n## The time, ten X925 pinned, ten threads, the default layout\n")
    rounds = sorted({int(f.split("-")[0][1:]) for f in os.listdir(os.path.join(out, "log")) if re.match(r"r\d+-", f)})
    print(f"*`presk bench`*: the median of {repeat} calls after one warm-up, one process a round and a variant, "
          f"{len(rounds)} rounds, the order of the two swapped each round; the median of the processes, the fastest, "
          "and the slowest over the fastest; *rounds faster with*, the rounds whose process with `--opt` beat the one without, "
          "the two adjacent in time. In milliseconds a call, the loss at B = 16, T = 256.\n")
    print("| scale | row | without `--opt` | with | with ÷ without | rounds faster with | fastest without | fastest with | max ÷ min without | max ÷ min with |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in SCALES:
        b = {o: [bench(out, f"r{k}-{s}-loss-{o}") for k in rounds] for o in ("plain", "opt")}
        for row in BENCH_ROWS:
            xs = {o: [p[row] for p in b[o]] for o in b}
            med = {o: statistics.median(xs[o]) for o in xs}
            faster = sum(1 for a, b in zip(xs["plain"], xs["opt"]) if b < a)
            print(f"| {s} | {row} | {med['plain']:.2f} | {med['opt']:.2f} | **{med['opt'] / med['plain']:.3f}** | "
                  f"{faster} of {len(rounds)} | {min(xs['plain']):.2f} | {min(xs['opt']):.2f} | {max(xs['plain']) / min(xs['plain']):.3f} | "
                  f"{max(xs['opt']) / min(xs['opt']):.3f} |")
    print("\nThe rows with `+lanes` are `presk bench`'s own, the bit-exact vector lanes; the `bf16` row is beside the protocol's `f32` and `f64`.\n")

    if all(os.path.exists(os.path.join(out, "kernels", f"{s}-loss-{r}-{o}", "explain.md"))
           for s in SCALES for r in ("f32", "f64") for o in ("plain", "opt")):
        print("### The loss's launches, counted (`presk explain --no-exact`, static)\n")
        print("| scale | row | launches a call before fusion (without, with `--opt`) | after fusion | distinct kernels | values that reach memory |")
        print("|---|---|---|---|---|---|")
        ks = {}
        for s in SCALES:
            for r in ("f32", "f64"):
                k = {o: kernels(out, f"{s}-loss-{r}-{o}") for o in ("plain", "opt")}
                ks[(s, r)] = k
                print(f"| {s} | {r} | {k['plain'][0]}, {k['opt'][0]} | {k['plain'][1]}, {k['opt'][1]} | "
                      f"{k['plain'][2]}, {k['opt'][2]} | {len(k['plain'][3])}, {len(k['opt'][3])} |")
        norm = lambda line: re.sub(r"%\d+", "%", line)
        print()
        for s in SCALES:
            k = ks[(s, "f64")]
            count = {o: {} for o in k}
            for o in k:
                for v, _, _ in k[o][3]:
                    count[o][v.split("/")[0]] = count[o].get(v.split("/")[0], 0) + 1
            grew = [f for f in count["opt"] if count["opt"][f] > count["plain"].get(f, 0)]
            for f in grew:
                lines = {norm(line) for v, line, _ in k["plain"][3] if v.startswith(f + "/")}
                new = [(line, why) for v, line, why in k["opt"][3] if v.startswith(f + "/") and norm(line) not in lines]
                print(f"- {s}: `{f}` stores {count['plain'].get(f, 0)} values without `--opt` and {count['opt'][f]} with it; "
                      "with it and not without: " + "; ".join(f"`{line}`, {why}" for line, why in new) + ".")
        print("\nA function's values counted once, whatever the number of its calls.\n")

    print(f"### Beside: the step, `presk train`, {repeat + 1} steps from the initialization\n")
    print(f"*Median step*: `presk train`'s median of {repeat} steps after the first, one process a round and a variant, the median of the {len(rounds)}. "
          "*Rounds faster with*: the rounds whose process with `--opt` beat the one without, the two adjacent in time.\n")
    print("| scale | row | without `--opt` | with | with ÷ without | rounds faster with | max ÷ min (without, with) | launches a step (without, with) | peak GB (without, with) | final loss (without, with) | weights: one checksum a variant; the same with and without |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    s_ck = sums(out)
    notes = {}
    for s in SCALES:
        for r in ("f32", "f64"):
            t = {o: [train(out, f"r{k}-{s}-step-{r}-{o}") for k in rounds] for o in ("plain", "opt")}
            ms = {o: [x[0] for x in t[o]] for o in t}
            med = {o: statistics.median(ms[o]) for o in ms}
            launch = {o: sorted({x[1] for x in t[o]}) for o in t}
            loss = {o: sorted({x[3] for x in t[o]}) for o in t}
            peak = {o: statistics.median(peak_gb(out, f"r{k}-{s}-step-{r}-{o}") for k in rounds) for o in t}
            ck = {o: {s_ck.get(f"r{k}-{s}-step-{r}-{o}") for k in rounds} for o in t}
            notes[(s, r)] = t["opt"][0][4]
            one = "yes" if all(len(ck[o]) == 1 for o in ck) else "**no**"
            same = "the same" if ck["plain"] == ck["opt"] else "different"
            faster = sum(1 for a, b in zip(ms["plain"], ms["opt"]) if b < a)
            print(f"| {s} | {r} | {med['plain']:.1f} | {med['opt']:.1f} | **{med['opt'] / med['plain']:.3f}** | "
                  f"{faster} of {len(rounds)} | {max(ms['plain']) / min(ms['plain']):.3f}, {max(ms['opt']) / min(ms['opt']):.3f} | "
                  f"{'/'.join(map(str, launch['plain']))}, {'/'.join(map(str, launch['opt']))} | "
                  f"{peak['plain']:.2f}, {peak['opt']:.2f} | {'/'.join(loss['plain'])}, {'/'.join(loss['opt'])} | {one}; {same} |")
    print("\nIn milliseconds a step. The optimizer's report on the step's program (`presk train --opt`):\n")
    for (s, r), line in notes.items():
        print(f"- {s}, {r}: {line}.")


if __name__ == "__main__":
    main()
