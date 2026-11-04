#!/usr/bin/env python3
"""The course GPT's timings across processes (presk D126's protocol, D127).

`experiments/pytorch_baseline.sh` (T2, T4) and `experiments/j_step.sh` (J′)
run N times, one round after the other, every row of a round a process of
its own; this combines the rounds. A row's median is the median of its N
processes' medians, its fastest the smallest of them; a ratio between rows
reads their medians, and beside it their fastest. A step a side times as
two programs (`loss_grad` + `adamw.step`) is a round's sum, as the
baseline's report adds them; with N odd its median is a round's, whose two
programs the label shows.

    python3 experiments/retime_processes.py DIR [--table4 OUT.md [--record REPORT.md]] > TABLES.md

DIR holds `baseline-round1..N/` (each a `pytorch_baseline.sh` OUT) and
`j_step-round1..N.md` with the `.times.txt` and `.gaps.txt` each round left
(`out/j_step_times.txt`, `out/j_step_gaps.txt`, copied). `--table4` writes
the forward pass, the step and T4 in the baseline report's own format
(`threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch`),
the median of the processes in `median ms`, the fastest in `best ms`, so
that what reads the report reads it unchanged.

Rounds run with `LLVM=1` (plan 0.17.0 step 8, D181) carry the LLVM
backend's rows under the plans; they are then shown beside the native
backend's, and `--table4` adds NB2 and NB3: Cranelift and LLVM against the
C program, the C program's workers waiting as the native pool's.
"""
import argparse
import glob
import os
import re
import sys

B, T, L, D, H, HD, V = 16, 256, 6, 384, 6, 64, 92  # experiments/pytorch_baseline.py's
FWD = B * T * L * 24 * D * D + L * 4 * B * H * T * T * HD + 2 * B * T * D * V
STEP = 3 * FWD
WAIT = "GOMP_SPINCOUNT=8M"
SLOW = 1.5  # a process in the slow mode: a round more than 1.5 times another on the same row


def thousands(x, digits=1):
    return f"{x:,.{digits}f}".replace(",", " ")


def rounds(d, pattern):
    return sorted(glob.glob(os.path.join(d, pattern)), key=lambda p: int(re.search(r"round(\d+)", p).group(1)))


def results(path):
    res = {}
    for line in open(path):
        key, *vals = line.split()
        res[key] = vals
    return res


class Row:
    """One row across the rounds: its value in each (ms or s), and for a sum its parts."""

    def __init__(self, values, parts=None):
        self.values = values
        self.parts = parts  # per round: (loss_grad, adamw.step)
        n = len(values)
        order = sorted(range(n), key=lambda i: values[i])
        self.mid = order[n // 2]
        self.low = order[0]
        self.median = values[self.mid]
        self.fastest = values[self.low]
        self.spread = max(values) / min(values)

    def label(self, which):
        g, o = self.parts[which]
        return f"`loss_grad` {thousands(g, 0)} + `adamw.step` {thousands(o, 0)}"


def baseline_rows(res):
    """{(workload, threads): [(side, layout, Row)]} from the rounds' results.txt."""
    ms = lambda r, k: float(r[k][0])
    one = lambda k: Row([ms(r, k) for r in res])
    two = lambda g, a: Row([ms(r, g) + ms(r, a) for r in res], [(ms(r, g), ms(r, a)) for r in res])
    out = {}
    for th in ("10", "20"):
        llvm = all(f"fwd.llvm.plan.{th}" in r for r in res)
        f = [("native", lay, one(f"fwd.native.{lay}.{th}")) for lay in ("default", "plan")]
        f += [("LLVM", "plan", one(f"fwd.llvm.plan.{th}"))] if llvm else []
        f += [("C", lay, one(f"fwd.c.{lay}.{th}")) for lay in ("default", "plan")]
        f += [(f"C, `{WAIT}`", lay, one(f"fwd.c.{lay}.{th}.w")) for lay in ("default", "plan")]
        f += [("PyTorch", "its own", one(f"fwd.pytorch.{th}"))]
        out["fwd", th] = f
        s = [("native, `presk train`", lay, one(f"step.native.{lay}.{th}")) for lay in ("default", "plan")]
        s += [("LLVM, `presk train`", "plan", one(f"step.llvm.plan.{th}"))] if llvm else []
        s += [("native", lay, two(f"grad.native.{lay}.{th}", f"adam.native.{lay}.{th}")) for lay in ("default", "plan")]
        s += [("LLVM", "plan", two(f"grad.llvm.plan.{th}", f"adam.llvm.plan.{th}"))] if llvm else []
        s += [("C", lay, two(f"grad.c.{lay}.{th}", f"adam.c.{lay}.{th}")) for lay in ("default", "plan")]
        s += [(f"C, `{WAIT}`", lay, two(f"grad.c.{lay}.{th}.w", f"adam.c.{lay}.{th}.w")) for lay in ("default", "plan")]
        s += [("PyTorch, `train.py`", "its own", one(f"step.pytorch.{th}"))]
        out["step", th] = s
    lin = {}
    for i, o in ((384, 384), (384, 1536), (1536, 384)):
        lin[i, o] = [one(f"lin.native.{i}.{o}"), one(f"kern.native.lin.{i}.{o}"), one(f"lin.c.{i}.{o}"), one(f"lin.pytorch.{i}.{o}")]
    return out, lin


WHERE = {"10": "10 (ten X925, pinned)", "20": "20 (twenty, unpinned)"}
TITLE = {"fwd": "The forward pass", "step": "One training step"}


def side_label(side, row, which):
    return f"{side}, {row.label(which)}" if row.parts else side


def find(rows, side, layout):
    return next(r for s, lay, r in rows if s == side and lay == layout)


def t4_rows(b, th, wait):
    """(workload, native Row, C Row) for T4 at a thread setting and a waiting ('' or '.w')."""
    c = f"C, `{WAIT}`" if wait else "C"
    return [
        ("the forward pass", find(b["fwd", th], "native", "plan"), find(b["fwd", th], c, "plan")),
        ("the step: `loss_grad` + `adamw.step`", find(b["step", th], "native", "plan"), find(b["step", th], c, "plan")),
        ("beside: `presk train`'s step, drawing its masks", find(b["step", th], "native, `presk train`", "plan"), find(b["step", th], c, "plan")),
    ]


def nb_rows(b, th, wait):
    """(statement, workload, backend Row, C Row, threshold) for NB2 and NB3, when the rounds have LLVM's rows."""
    c = f"C, `{WAIT}`" if wait else "C"
    if not any(side == "LLVM" for side, _, _ in b["fwd", th]):
        return []
    out = []
    for nb, side, limit in (("NB2", "native", 1.3), ("NB3", "LLVM", 1.0)):
        out.append((nb, "the forward pass", find(b["fwd", th], side, "plan"), find(b["fwd", th], c, "plan"), limit))
        out.append((nb, "the step: `loss_grad` + `adamw.step`", find(b["step", th], side, "plan"), find(b["step", th], c, "plan"), limit))
    return out


def table4(b, n, machine, version, record, out):
    with open(out, "w", encoding="utf-8") as f:
        p = lambda s="": print(s, file=f)
        p(f"> The course GPT on three sides across {n} processes (presk D127, D126's protocol): `experiments/pytorch_baseline.sh` run {n} times, one round after the other, every row of a round a process of its own; {machine}. Columns as in the baseline's report (`docs/pytorch-baseline-0.10.0.md`), but `median ms` is the median of the {n} processes' medians (each the median of seven calls or steps) and `best ms` the fastest of them; GFLOP/s and ÷ PyTorch read the medians. A step timed as two programs is a round's sum, and the label names the median round's two programs." + (f" The whole record: `{record}`." if record else ""))
        p()
        p(f"# The course GPT on three sides: {version} against PyTorch, the median of {n} processes")
        for w, total in (("fwd", FWD), ("step", STEP)):
            p()
            p(f"## {TITLE[w]}")
            p()
            p("| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |")
            p("|---|---|---|---|---|---|---|")
            for th in ("10", "20"):
                rows = b[w, th]
                base = rows[-1][2].median
                for side, lay, r in rows:
                    p(f"| {WHERE[th]} | {side_label(side, r, r.mid)} | {lay} | {thousands(r.median)} | {thousands(r.fastest)} | {total / 1e9 / (r.median / 1e3):.1f} | {r.median / base:.2f} |")
        p()
        p("## T4: the native backend against the C program")
        p()
        p("Judged on the ten X925 cores pinned, the C program's workers waiting as the native pool's, the row of record; native ÷ C from the medians, and from the fastest beside it.")
        p()
        p("| threads | the C program's waiting | workload | native ms | C ms | native ÷ C | fastest ÷ fastest | T4 (≤ 1.3×) |")
        p("|---|---|---|---|---|---|---|---|")
        for th in ("10", "20"):
            for wait, name in ((".w", f"`{WAIT}`"), ("", "OpenMP's default")):
                for work, nat, c in t4_rows(b, th, wait):
                    of_record = th == "10" and wait == ".w"
                    r = nat.median / c.median
                    v = ("holds" if r <= 1.3 else "fails") if of_record else "(reported)"
                    if work.startswith("beside"):
                        v = "—"
                    p(f"| {WHERE[th]} | {name} | {work} | {thousands(nat.median)} | {thousands(c.median)} | {r:.2f} | {nat.fastest / c.fastest:.2f} | {v} |")
        if nb_rows(b, "10", ".w"):
            p()
            p("## NB2 and NB3: Cranelift with level 2 and the LLVM backend against the C program")
            p()
            p("Plan 0.17.0's statements (D181), judged as T4 is: on the ten X925 cores pinned, the C program's workers waiting as the native pool's, from the medians; NB2 at most 1.3×, NB3 at most 1.0×.")
            p()
            p("| statement | threads | the C program's waiting | workload | backend ms | C ms | backend ÷ C | fastest ÷ fastest | verdict |")
            p("|---|---|---|---|---|---|---|---|---|")
            for th in ("10", "20"):
                for wait, name in ((".w", f"`{WAIT}`"), ("", "OpenMP's default")):
                    for nb, work, x, c, limit in nb_rows(b, th, wait):
                        r = x.median / c.median
                        of_record = th == "10" and wait == ".w"
                        v = (f"holds (≤ {limit}×)" if r <= limit else f"fails (> {limit}×)") if of_record else "(reported)"
                        p(f"| {nb} | {WHERE[th]} | {name} | {work} | {thousands(x.median)} | {thousands(c.median)} | {r:.3f} | {x.fastest / c.fastest:.3f} | {v} |")


def jstep(d):
    """{realization: [(t1, t4)] per round}, {realization: gaps line per round}."""
    times, gaps = {}, {}
    for path in rounds(d, "j_step-round*.times.txt"):
        for line in open(path):
            m = re.match(r"(\S+) steps=1 ([0-9.]+) s, steps=4 ([0-9.]+) s", line)
            if m:
                times.setdefault(m.group(1), []).append((float(m.group(2)), float(m.group(3))))
    for path in rounds(d, "j_step-round*.gaps.txt"):
        for line in open(path):
            f = line.split()
            if f and (f[0] in times):
                gaps.setdefault(f[0], []).append(tuple(f[1:4]))
    return times, gaps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir")
    ap.add_argument("--table4")
    ap.add_argument("--record", help="the report this table belongs to, named in its header")
    a = ap.parse_args()
    dirs = rounds(a.dir, "baseline-round*/results.txt")
    n = len(dirs)
    if n % 2 == 0:
        sys.exit(f"{a.dir}: {n} baseline rounds; the median of an even number is no process's")
    res = [results(p) for p in dirs]
    machines = [open(os.path.join(os.path.dirname(p), "machine.txt")).read().splitlines() for p in dirs]
    b, lin = baseline_rows(res)
    pr = lambda s="": print(s)

    first = machines[0][0].split()[1]
    rev = machines[0][1].split()[1].rstrip(",")
    loads = ", ".join(m[next(i for i, l in enumerate(m) if l.startswith("load "))].split()[1] for m in machines)
    version = machines[0][1].split(", ", 1)[1]
    machine = f"presk `{rev}`, {first[:16].replace('T', ' ')} onwards, the one-minute load at each round's start {loads}"
    if a.table4:
        table4(b, n, machine, version, a.record, a.table4)

    pr(f"## The baseline across {n} processes")
    pr()
    pr(f"Each value is a round's median of seven calls or steps (its `report.md`'s); a row's {n} rounds are {n} processes. **median**: the median of the {n}; **fastest**: the smallest; **max ÷ min**: the row's spread; ÷ PyTorch from the medians, PyTorch's {n} treated alike. A step timed as two programs is the round's sum.")
    head = " | ".join(f"round {i + 1}" for i in range(n))
    for w in ("fwd", "step"):
        for th in ("10", "20"):
            pr()
            pr(f"### {TITLE[w]}, {WHERE[th].split(' ', 1)[1].strip('()')}")
            pr()
            pr(f"| side | layout | {head} | median | fastest | max ÷ min | ÷ PyTorch |")
            pr("|---|---|" + "---|" * n + "---|---|---|---|")
            rows = b[w, th]
            base = rows[-1][2]
            for side, lay, r in rows:
                s = f"{side}, `loss_grad` + `adamw.step`" if r.parts else side
                pr(f"| {s} | {lay} | " + " | ".join(thousands(v) for v in r.values) + f" | **{thousands(r.median)}** | {thousands(r.fastest)} | {r.spread:.3f} | {r.median / base.median:.2f} |")

    pr()
    pr("### T2 and T4, from the medians and from the fastest")
    pr()
    pr("| statement | threads | the C program's waiting | median | fastest ÷ fastest | per round |")
    pr("|---|---|---|---|---|---|")
    for th in ("10", "20"):
        for wait, name in (("", "OpenMP's default"), (".w", f"`{WAIT}`")):
            c = find(b["step", th], f"C, `{WAIT}`" if wait else "C", "plan")
            pt = find(b["step", th], "PyTorch, `train.py`", "its own")
            per = ", ".join(f"{x / y:.2f}" for x, y in zip(c.values, pt.values))
            pr(f"| T2: the C step ÷ PyTorch's | {WHERE[th]} | {name} | **{c.median / pt.median:.2f}** | {c.fastest / pt.fastest:.2f} | {per} |")
    for th in ("10", "20"):
        for wait, name in ((".w", f"`{WAIT}`"), ("", "OpenMP's default")):
            for work, nat, c in t4_rows(b, th, wait):
                per = ", ".join(f"{x / y:.2f}" for x, y in zip(nat.values, c.values))
                pr(f"| T4: native ÷ C, {work} | {WHERE[th]} | {name} | **{nat.median / c.median:.2f}** | {nat.fastest / c.fastest:.2f} | {per} |")

    for th in ("10", "20"):
        for wait, name in ((".w", f"`{WAIT}`"), ("", "OpenMP's default")):
            for nb, work, x, c, _ in nb_rows(b, th, wait):
                per = ", ".join(f"{u / v:.3f}" for u, v in zip(x.values, c.values))
                pr(f"| {nb}: {'LLVM' if nb == 'NB3' else 'native'} ÷ C, {work} | {WHERE[th]} | {name} | **{x.median / c.median:.3f}** | {x.fastest / c.fastest:.3f} | {per} |")

    pr()
    pr("### The linear products alone, ten threads pinned (ms, the median of the processes; max ÷ min)")
    pr()
    pr("| product | native | its kernels | C | PyTorch |")
    pr("|---|---|---|---|---|")
    for (i, o), rs in lin.items():
        pr(f"| [16, 256, {i}] · [{i}, {o}] | " + " | ".join(f"{r.median:.2f} ({r.spread:.3f})" for r in rs) + " |")

    pr()
    pr("### The slow mode")
    pr()
    allrows = [(f"{TITLE[w].lower()}, {th} threads, {side}, {lay}", r) for (w, th), rows in b.items() for side, lay, r in rows]
    allrows += [(f"linear [{i}, {o}], {k}", r) for (i, o), rs in lin.items() for k, r in zip(("native", "its kernels", "C", "PyTorch"), rs)]
    worst = sorted(allrows, key=lambda x: -x[1].spread)
    slow = [x for x in worst if x[1].spread > SLOW]
    pr(f"Rows whose slowest round is more than {SLOW}× its fastest: {len(slow)} of {len(allrows)}" + (": " + "; ".join(f"{k} {r.spread:.2f}" for k, r in slow) if slow else "") + ".")
    pr("The five widest spreads: " + "; ".join(f"{k} {r.spread:.3f}" for k, r in worst[:5]) + ".")

    times, gaps = jstep(a.dir)
    if times:
        m = len(next(iter(times.values())))
        per = {k: Row([(t4 - t1) / 3 for t1, t4 in v]) for k, v in times.items()}
        base = per["f64"]
        pr()
        pr(f"## J′ across {m} rounds")
        pr()
        pr("Per step: a round's four-step time minus its one-step time, over three, each a `presk train` process of its own; the ratio to all-`f64` from the medians, and from the fastest beside it.")
        pr()
        h = " | ".join(f"round {i + 1}" for i in range(m))
        pr(f"| realization | {h} | median | fastest | max ÷ min | vs all-`f64` | fastest vs fastest | update gap |")
        pr("|---|" + "---|" * m + "---|---|---|---|---|---|")
        for k, r in per.items():
            g = sorted(set(x[0] for x in gaps.get(k, [])))
            gap = g[0] if len(g) == 1 else " / ".join(g)
            pr(f"| {k} | " + " | ".join(f"{v:.2f}" for v in r.values) + f" | **{r.median:.2f}** | {r.fastest:.2f} | {r.spread:.3f} | {base.median / r.median:.2f}× | {base.fastest / r.fastest:.2f}× | {gap} |")
        pr()
        pr("The one-step and four-step processes, seconds of wall time:")
        pr()
        pr(f"| realization | steps | {h} | max ÷ min |")
        pr("|---|---|" + "---|" * m + "---|")
        for k, v in times.items():
            for j, name in ((0, "1"), (1, "4")):
                xs = [x[j] for x in v]
                pr(f"| {k} | {name} | " + " | ".join(f"{x:.2f}" for x in xs) + f" | {max(xs) / min(xs):.3f} |")


if __name__ == "__main__":
    main()
