"""SR1's report (experiments/bound_depth.sh, D193): the four rules and the
measured gap on the course GPT at D = 64 with L = 1, 2, 4 and 6 blocks.

    python3 experiments/bound_depth.py out/bound-depth > out/bound-depth/report.md

SR1a: on the trained models, log10(cond/rss) on the loss grows from each L to
the next, and its least-squares slope over the four L lies between 2.2 and
4.2 orders a block, in f64 and in f32. SR1b: at every L of SR1a, in f64 and
f32, rss is within 30x of the measured gap. The models at initialization
are reported beside, without a statement.
"""
import math
import os
import re
import sys

LS = [1, 2, 4, 6]
TYPES = ["f64", "f32"]
SLOPE = (2.2, 4.2)
WITHIN = 30.0
RULES = ["k", "normwise", "cond", "rss"]


def read(path):
    """The four rules' values and the measured gap of one `presk check`."""
    out = {}
    for line in open(path):
        m = re.match(r"^\s+(?:bound|estimate)\s+(\S+)\s+(\S+)", line)
        if m and m.group(1) in RULES:
            out[m.group(1)] = float(m.group(2))
        m = re.match(r"^\s+measured\s+(\S+)", line)
        if m:
            out["measured"] = float(m.group(1))
    return out


def slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def table(out, kind):
    rows = {}
    print(f"| L | type | measured | `k` ÷ gap | `normwise` ÷ gap | `cond` ÷ gap | `rss` ÷ gap | log₁₀(`cond`/`rss`) |")
    print("|---|---|---|---|---|---|---|---|")
    for t in TYPES:
        for L in LS:
            p = os.path.join(out, f"check-{kind}-L{L}-{t}.txt")
            if not os.path.exists(p):
                print(f"| {L} | {t} | not run | | | | | |")
                continue
            r = read(p)
            g = r["measured"]
            orders = math.log10(r["cond"] / r["rss"])
            rows[(t, L)] = (orders, r["rss"] / g)
            print(f"| {L} | {t} | {g:.4e} | {r['k'] / g:.3g} | {r['normwise'] / g:.3g} | "
                  f"{r['cond'] / g:.3g} | {r['rss'] / g:.3g} | {orders:.2f} |")
    return rows


def main():
    out = sys.argv[1]
    print("# SR1: the bound's distance to the estimate against depth\n")
    print("The course GPT at D = 64, H = 2, on two windows of the Code civil; each rule's value on the loss "
          "as a multiple of the gap measured against the exact reference.\n")
    print("## Trained (`make gpt-trained`'s recipe; L = 2 the weights and envelope of record)\n")
    trained = table(out, "trained")
    print("\n## At initialization (reported, no statement)\n")
    init = table(out, "init")
    complete = all((t, L) in trained for t in TYPES for L in LS)
    print("\n## The statements\n")
    if not complete:
        print("**SR1: not every check has run.**")
        return
    a_ok, b_ok = True, True
    for t in TYPES:
        ys = [trained[(t, L)][0] for L in LS]
        s = slope(LS, ys)
        grows = all(b > a for a, b in zip(ys, ys[1:]))
        inside = SLOPE[0] <= s <= SLOPE[1]
        a_ok &= grows and inside
        print(f"- {t}: log₁₀(`cond`/`rss`) {', '.join(f'{y:.2f}' for y in ys)} at L = {', '.join(map(str, LS))}; "
              f"slope {s:.2f} orders a block (intercept {sum(ys) / 4 - s * sum(LS) / 4:.2f}); "
              f"grows at every step: {'yes' if grows else 'no'}; slope in [{SLOPE[0]}, {SLOPE[1]}]: {'yes' if inside else 'no'}.")
        ratios = [trained[(t, L)][1] for L in LS]
        within = all(1 / WITHIN <= q <= WITHIN for q in ratios)
        b_ok &= within
        print(f"- {t}: `rss` ÷ gap {', '.join(f'{q:.3g}' for q in ratios)}; within {WITHIN:g}×: {'yes' if within else 'no'}.")
        if all((t, L) in init for L in LS):
            yi = [init[(t, L)][0] for L in LS]
            print(f"- {t} at initialization (reported): log₁₀(`cond`/`rss`) {', '.join(f'{y:.2f}' for y in yi)}, "
                  f"slope {slope(LS, yi):.2f}; `rss` ÷ gap {', '.join(f'{init[(t, L)][1]:.3g}' for L in LS)}.")
    print(f"\n**SR1a {'holds' if a_ok else 'fails'}. SR1b {'holds' if b_ok else 'fails'}.**")


if __name__ == "__main__":
    main()
