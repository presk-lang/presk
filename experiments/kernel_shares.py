"""Each kernel's share of the native backend's gap to the C program (plan
0.17.0 step 1, D174; statement NB1): the report of kernel_shares.sh.

    python3 experiments/kernel_shares.py OUT --calls N --rounds R

Reads, for each entry KEY (fwd, grad, adam) and each round:
- OUT/native/KEY-SETTING-ROUND.json, the profile `presk bench --calibrate`
  wrote: each kernel's signature and its median launch;
- OUT/c/KEY-SETTING/times/SETTING-BUILD-ROUND.txt, the timed C program's
  launches in order (`K seconds`, N calls, the first left out as
  kernel_time.py does), and OUT/c/KEY-SETTING/kernels.c, each kernel K's
  signature (its `// kernel` line); the program exported under the
  setting's plan.

The two sides are joined by the signature: the C program's host is recorded
from the native backend's run under the same plan, so each signature is
launched as often on both, and a native kernel's time a call is its median
launch times the C program's launches of it a call. A signature launched by
several kernels of kernels.c is one row. Each round gives every row its time
a call on each side; the report gives the median over the rounds, and the
spread of the totals (max ÷ min).

Markdown on stdout: per entry and setting, the totals; then the rows that
carry the gap, sorted by their share of it (native − GCC), with each C
build's time, until four fifths of the gap are named, and the rest summed;
then the same by kind of kernel.
"""
import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

ENTRIES = [("fwd", "the forward pass `GPT.apply`"), ("grad", "the gradient `loss_grad`"), ("adam", "the optimizer `adamw.step`")]
SETTINGS = {
    "ten": ("ten X925, ten threads", ["gcc-O2"]),
    "one": ("one X925, one thread", ["gcc-O2", "clang-O2", "clang-O3"]),
}
SIG = re.compile(r"^// (kernel .*?) \{\n(?:(?://.*)\n)*(?:static|PRESK_KERNEL) void k(\d+)\(", re.M)


def kind(sig):
    """`dot`, `ein[…]`, `dot_vjp_l` …: the word after `kernel`, an epilogue's
    `+add` kept apart from its product."""
    name = sig.split(" ", 2)[1]
    if name.startswith("ein["):
        return "ein"
    return name


def c_side(d, times, calls):
    """Signature → (launches a call, seconds a call) of one timed run."""
    sig = {int(k): s for s, k in SIG.findall(open(os.path.join(d, "kernels.c")).read())}
    records = [(int(k), float(s)) for k, s in (line.split() for line in open(times))]
    assert records and len(records) % calls == 0, (times, len(records), calls)
    per_call = len(records) // calls
    out = defaultdict(lambda: [0, 0.0])
    for k, s in records[per_call:]:
        out[sig[k]][0] += 1
        out[sig[k]][1] += s
    n = calls - 1
    return {s: (c // n, t / n) for s, (c, t) in out.items()}


def native_side(path):
    """Signature → median launch in seconds."""
    p = json.load(open(path))
    return {k["signature"]: k["seconds"] for k in p["kernels"]}


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def ms(x):
    return f"{1e3 * x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--calls", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    a = ap.parse_args()
    w = print
    w("# Each kernel's share of the native backend's gap to the C program")
    w()
    w(open(os.path.join(a.out, "machine.txt")).read().strip().replace("\n", "  \n"))
    w()
    w(f"A row is a kernel signature. Its native time a call is its median launch (`presk bench --calibrate`) times the C "
      f"program's launches of it a call; the C times are each launch timed in the program (`kernel_time.py`), "
      f"{a.calls} calls with the first left out. Every figure is the median of {a.rounds} processes; the gap is native − GCC. "
      f"The kernels' times only: neither side's host is in them.")
    missing = []
    summary = []
    for setting, (label, builds) in SETTINGS.items():
        for key, what in ENTRIES:
            d = os.path.join(a.out, "c", f"{key}-{setting}")
            rows = defaultdict(lambda: {"launches": None, "native": [], **{b: [] for b in builds}})
            totals = {"native": [], **{b: [] for b in builds}}
            for r in range(1, a.rounds + 1):
                nat = os.path.join(a.out, "native", f"{key}-{setting}-{r}.json")
                if not os.path.exists(nat):
                    missing.append(nat)
                    continue
                med = native_side(nat)
                cs = {}
                for b in builds:
                    t = os.path.join(d, "times", f"{setting}-{b}-{r}.txt")
                    if not os.path.exists(t):
                        missing.append(t)
                        continue
                    cs[b] = c_side(d, t, a.calls)
                if builds[0] not in cs:
                    continue
                launches = {s: c for s, (c, _) in cs[builds[0]].items()}
                tot_n = 0.0
                for s, c in launches.items():
                    if s not in med:
                        missing.append(f"{nat}: {s}")
                        continue
                    rows[s]["launches"] = c
                    rows[s]["native"].append(med[s] * c)
                    tot_n += med[s] * c
                totals["native"].append(tot_n)
                for b, side in cs.items():
                    for s, (_, t) in side.items():
                        rows[s][b].append(t)
                    totals[b].append(sum(t for _, t in side.values()))
            if not totals["native"]:
                continue
            tn, tg = median(totals["native"]), median(totals[builds[0]])
            gap = tn - tg
            summary.append((label, what, tn, {b: median(totals[b]) for b in builds}, totals))
            w()
            w(f"## {what}, {label}")
            w()
            cols = " | ".join(f"{b} (ms)" for b in builds)
            w(f"Native {ms(tn)} ms, GCC {ms(tg)} ms: **{tn / tg:.2f}×**, a gap of {ms(gap)} ms."
              + "".join(f" {b}: {ms(median(totals[b]))} ms ({tn / median(totals[b]):.2f}× native ÷ it)." for b in builds[1:]))
            w()
            w(f"| kernel | launches a call | native (ms) | {cols} | native ÷ GCC | share of the gap | cumulative |")
            w("|---|---|---|" + "---|" * len(builds) + "---|---|---|")
            ranked = sorted(rows.items(), key=lambda kv: -(median(kv[1]["native"]) - median(kv[1][builds[0]])))
            cum = 0.0
            rest = defaultdict(float)
            rest_n = 0
            for s, v in ranked:
                n, g = median(v["native"]), median(v[builds[0]])
                share = (n - g) / gap if gap else float("nan")
                if cum < 0.8:
                    cum += share
                    sig = s if len(s) <= 120 else s[:119] + "…"
                    w(f"| `{sig}` | {v['launches']} | {ms(n)} | " + " | ".join(ms(median(v[b])) for b in builds)
                      + f" | {n / g:.2f}× | {100 * share:.1f} % | {100 * cum:.1f} % |")
                else:
                    rest_n += 1
                    rest["native"] += n
                    for b in builds:
                        rest[b] += median(v[b])
            if rest_n:
                w(f"| the other {rest_n} | | {ms(rest['native'])} | " + " | ".join(ms(rest[b]) for b in builds)
                  + f" | {rest['native'] / rest[builds[0]]:.2f}× | {100 * (rest['native'] - rest[builds[0]]) / gap:.1f} % | 100 % |")
            w()
            w(f"| kind | launches a call | native (ms) | {cols} | native ÷ GCC | share of the gap |")
            w("|---|---|---|" + "---|" * len(builds) + "---|---|")
            by = defaultdict(lambda: defaultdict(float))
            for s, v in rows.items():
                k = kind(s)
                by[k]["launches"] += v["launches"] or 0
                by[k]["native"] += median(v["native"])
                for b in builds:
                    by[k][b] += median(v[b])
            for k, v in sorted(by.items(), key=lambda kv: -(kv[1]["native"] - kv[1][builds[0]])):
                w(f"| `{k}` | {int(v['launches'])} | {ms(v['native'])} | " + " | ".join(ms(v[b]) for b in builds)
                  + f" | {v['native'] / v[builds[0]]:.2f}× | {100 * (v['native'] - v[builds[0]]) / gap:.1f} % |")
    w()
    w("## The totals, and their spread over the processes")
    w()
    w("| entry | setting | side | median (ms) | max ÷ min |")
    w("|---|---|---|---|---|")
    for label, what, tn, tb, totals in summary:
        for side, xs in totals.items():
            if xs:
                w(f"| {what} | {label} | {side} | {ms(median(xs))} | {max(xs) / min(xs):.3f} |")
    if missing:
        w()
        w("## Missing")
        w()
        for m in missing:
            w(f"- `{m}`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
