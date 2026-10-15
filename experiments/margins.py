#!/usr/bin/env python3
"""ε from the margins (D118), the arithmetic `experiments/margins.sh` hands
to Python: the standard library only, so that any `python3` runs it.

  exact JSON PREFIX           an exact chunk of `logits` (`presk run --bits
                              128 --json --digits 30`): PREFIX.hilo, each
                              logit as two doubles whose sum is it to about
                              10⁻³² relative, and PREFIX.tsv, per position
                              the top index, the second, the margin and the
                              largest |logit|
  native JSON EXACT PREFIX    a native chunk of `logits` (`--digits 17`,
                              which gives every double back): per position
                              the largest gap to the exact logits, the
                              argmax of the realization's logits and the
                              sum of the gaps, in PREFIX.tsv
  decisions JSON OUT          `decisions`' indices, one per line
  report DIR …                every figure of the report (below)

The margin is m = y₍₁₎ − y₍₂₎, the top exact logit minus the largest of the
others, computed in decimal from the 30 digits printed (the 128-bit values
to 10⁻³⁰, the subtraction exact). A realization ŷ with ‖ŷ − y‖∞ ≤ ε at a
position can change its argmax only if m < 2ε there (m ≤ 2ε with ties).
"""

import array
import json
import math
import os
import sys
from decimal import Decimal, getcontext

getcontext().prec = 60


def rows(path):
    """The result of a `presk run --json` of `logits`, rows of V strings in
    batch-major order (window, then position)."""
    data = json.load(open(path))["result"]["data"]
    for window in data:
        for row in window:
            yield row


def top(values):
    """The greatest index among the maxima (the reference's `argmax`)."""
    best = 0
    for i in range(1, len(values)):
        if values[i] >= values[best]:
            best = i
    return best


def exact(src, prefix):
    hilo = array.array("d")
    with open(prefix + ".tsv.part", "w") as out:
        for row in rows(src):
            d = [Decimal(s) for s in row]
            a = top(d)
            b = top([v if i != a else Decimal("-Infinity") for i, v in enumerate(d)])
            m = d[a] - d[b]
            for s, v in zip(row, d):
                hi = float(s)
                hilo.append(hi)
                hilo.append(float(v - Decimal(hi)))
            ymax = max(abs(v) for v in d)
            out.write(f"{a}\t{b}\t{m:.24e}\t{float(ymax)!r}\n")
    with open(prefix + ".hilo", "wb") as f:
        hilo.tofile(f)
    os.replace(prefix + ".tsv.part", prefix + ".tsv")


def native(src, exact_prefix, prefix):
    hilo = array.array("d")
    with open(exact_prefix + ".hilo", "rb") as f:
        hilo.frombytes(f.read())
    k = 0
    with open(prefix + ".tsv.part", "w") as out:
        for row in rows(src):
            y = [float(s) for s in row]
            worst, total = 0.0, 0.0
            for v in y:
                # (ŷ − hi) is exact when the two are within a factor of two
                # (Sterbenz), and rounded at 2⁻⁵³ of the gap otherwise.
                e = abs((v - hilo[k]) - hilo[k + 1])
                k += 2
                total += e
                if e > worst:
                    worst = e
            out.write(f"{worst!r}\t{top(y)}\t{total!r}\n")
    if k != len(hilo):
        sys.exit(f"{src}: {k // 2} logits against {len(hilo) // 2} exact ones")
    os.replace(prefix + ".tsv.part", prefix + ".tsv")


def decisions(src, dst):
    data = json.load(open(src))["result"]["data"]
    with open(dst + ".part", "w") as out:
        for window in data:
            for i in window:
                out.write(f"{int(i)}\n")
    os.replace(dst + ".part", dst)


# ---------------------------------------------------------------------------
# The report.


def k_clopper_pearson(n, delta, alpha):
    """The largest k such that k events in n draws bound the event's
    probability by δ at confidence 1 − α, by the one-sided Clopper–Pearson
    bound: the largest k with P[Bin(n, δ) ≤ k] ≤ α (−1 if there is none).
    In logarithms, since (1 − δ)ⁿ underflows at δ = 10⁻², n = 118 016."""
    la, lp, lq = math.log(alpha), math.log(delta), math.log1p(-delta)
    lpmf = n * lq
    lcdf = lpmf
    if lcdf > la:
        return -1
    k = 0
    while k < n:
        lpmf += math.log(n - k) - math.log(k + 1) + lp - lq
        hi, lo = max(lcdf, lpmf), min(lcdf, lpmf)
        nxt = hi + math.log1p(math.exp(lo - hi))
        if nxt > la:
            return k
        lcdf = nxt
        k += 1
    return n


def eps_star(sorted_m, k):
    """Half the (k+1)-th smallest margin: the largest ε with at most k
    margins below 2ε."""
    if k < 0:
        return Decimal(0)
    return sorted_m[min(k, len(sorted_m) - 1)] / 2


def sci(x, d=3):
    return f"{float(x):.{d}e}"


def report(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="margins.py report")
    ap.add_argument("dir")
    ap.add_argument("--val", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--loss-gaps", default="")
    ap.add_argument("--deltas", default="1e-2 1e-3 1e-4")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    names = a.names.split()
    labels = a.labels.split("|")
    loss_gaps = a.loss_gaps.split() if a.loss_gaps else [""] * len(names)
    deltas = [float(x) for x in a.deltas.split()]
    T = a.seq

    chunks = sorted(f[:-4] for f in os.listdir(os.path.join(a.dir, "exact")) if f.endswith(".tsv"))
    top1, top2, margin, ymax = [], [], [], []
    for c in chunks:
        for line in open(os.path.join(a.dir, "exact", c + ".tsv")):
            i, j, m, y = line.split("\t")
            top1.append(int(i)); top2.append(int(j)); margin.append(Decimal(m)); ymax.append(float(y))
    n = len(margin)
    windows = n // T
    samples = json.load(open(a.val))["samples"][:windows]
    target = [t for x, y in samples for t in y]
    assert len(target) == n, (len(target), n)
    exact_acc = sum(p == t for p, t in zip(top1, target))
    sm = sorted(margin)
    ties = sum(1 for m in margin if m == 0)
    Y = max(ymax)

    # ε*(δ), with and without the correction.
    star = {}
    for d in deltas:
        k_raw = math.floor(d * n)
        k_cp = k_clopper_pearson(n, d, a.alpha)
        star[d] = {"k_raw": k_raw, "eps_raw": eps_star(sm, k_raw), "k_cp": k_cp, "eps_cp": eps_star(sm, k_cp)}

    def below(e, ms=margin):
        two = 2 * Decimal(repr(float(e))) if not isinstance(e, Decimal) else 2 * e
        return sum(1 for m in ms if m < two)

    rows_ = []
    for name, label, lg in zip(names, labels, loss_gaps):
        eps_pos, arg, tot = [], [], 0.0
        for c in chunks:
            for line in open(os.path.join(a.dir, "native", name, c + ".tsv")):
                e, g, s = line.split("\t")
                eps_pos.append(float(e)); arg.append(int(g)); tot += float(s)
        dec = [int(l) for l in open(os.path.join(a.dir, "decisions", name + ".txt"))][:n]
        assert len(eps_pos) == n and len(dec) == n, name
        em = max(eps_pos)
        flips = [p for p in range(n) if dec[p] != top1[p]]
        flips_tie = sum(1 for p in flips if margin[p] == 0)
        bound = below(em)
        # the same at each position's own gap: what taking the largest gap
        # over the positions costs the bound
        bound_pos = sum(1 for m, e in zip(margin, eps_pos) if m < 2 * Decimal(repr(e)))
        # the argument at each position: a flip needs m ≤ 2 ε at that position
        outside = sum(1 for p in flips if margin[p] > 2 * Decimal(repr(eps_pos[p])))
        disagree = sum(1 for p in range(n) if dec[p] != arg[p])
        acc = sum(p == t for p, t in zip(dec, target))
        rows_.append({
            "name": name, "label": label, "loss_gap": lg,
            "eps_meas": em, "mean_gap": tot / (n * 92), "rel": em / Y,
            "flips": len(flips), "flips_at_ties": flips_tie, "bound": bound, "bound_per_position": bound_pos,
            "outside": outside, "argmax_vs_decisions": disagree, "correct": acc,
            "certified": {str(d): em <= float(star[d]["eps_cp"]) for d in deltas},
            "certified_raw": {str(d): em <= float(star[d]["eps_raw"]) for d in deltas},
            "flip_positions": flips,
        })

    # The statement of step 0.
    d0 = 1e-3
    cert = [r for r in rows_ if r["name"].startswith("curve") and r["certified"][str(d0)]]
    clause1 = all(r["flips"] <= d0 * n for r in cert)
    clause2 = all(r["flips"] <= r["bound"] for r in rows_ if r["name"].startswith("curve"))
    every2 = all(r["flips"] <= r["bound"] for r in rows_)
    argument = all(r["outside"] == 0 for r in rows_)

    # Out of sample: ε* from the even windows, counted on the odd ones, and
    # the other way round.
    halves = []
    for name_a, pa, pb in (("even", 0, 1), ("odd", 1, 0)):
        ma = [margin[p] for p in range(n) if (p // T) % 2 == pa]
        mb = [margin[p] for p in range(n) if (p // T) % 2 == pb]
        sa = sorted(ma)
        for d in deltas:
            e = eps_star(sa, k_clopper_pearson(len(ma), d, a.alpha))
            cnt = below(e, mb)
            halves.append({"from": name_a, "delta": d, "eps": e, "n_a": len(ma), "n_b": len(mb),
                           "below_b": cnt, "frac_b": cnt / len(mb)})

    budgets = [10.0 ** -i for i in range(1, 8)]
    cdf = [(e, below(e)) for e in budgets]

    L = []
    w = L.append
    w(f"# ε from the margins: {windows} windows, {n} positions\n")
    w(f"Exact model: the reference at 128 bits; its accuracy {exact_acc}/{n} = {exact_acc / n:.6f}.")
    w(f"Largest |logit| {Y:.6g}. Margins: smallest {sci(sm[0], 4)}, median {sci(sm[n // 2], 4)}, "
      f"largest {sci(sm[-1], 4)}; exactly 0 (ties): {ties}.\n")
    w("Quantiles of the margins:\n")
    w("| q | margin |")
    w("|---|---|")
    for q in (1e-4, 1e-3, 1e-2, 1e-1, 0.5):
        w(f"| {q:g} | {sci(sm[min(n - 1, math.floor(q * n))], 4)} |")
    w("\nThe fraction of positions whose margin is below 2ε:\n")
    w("| ε | m < 2ε | fraction |")
    w("|---|---|---|")
    for e, c in cdf:
        w(f"| {e:.0e} | {c} | {c / n:.3e} |")
    w(f"\nε*(δ), α = {a.alpha}: the largest ε with at most k margins below 2ε; uncorrected k = ⌊δn⌋, "
      "corrected k the largest with P[Bin(n, δ) ≤ k] ≤ α (Clopper–Pearson, one-sided).\n")
    w("| δ | δn | k uncorrected | ε* uncorrected | k corrected | ε* corrected | ratio |")
    w("|---|---|---|---|---|---|---|")
    for d in deltas:
        s = star[d]
        ratio = float(s["eps_raw"]) / float(s["eps_cp"]) if s["eps_cp"] > 0 else float("inf")
        w(f"| {d:.0e} | {d * n:.1f} | {s['k_raw']} | {sci(s['eps_raw'], 4)} | {s['k_cp']} | {sci(s['eps_cp'], 4)} | {ratio:.2f} |")
    w("\nPer realization. The gap to the exact loss is the foil's, relative, on two held-out windows; the gap on the logits "
      "the largest |ŷ − y| over every position and logit, relative to the largest |y|, and its mean; the flips are "
      "`decisions`' natively against the exact argmax; the bound is #{m < 2ε} at that largest gap, and 'per position' "
      "#{m(x) < 2ε(x)} at each position's own largest gap:\n")
    hdr = "| realization | gap to the exact loss | gap on the logits | relative | mean | flips | at ties | bound #{m < 2ε} | bound / flips | per position | " + \
          " | ".join(f"cert. {d:.0e}" for d in deltas) + " | accuracy | vs exact |"
    w(hdr)
    w("|" + "---|" * (hdr.count("|") - 1))
    for r in rows_:
        ratio = f"{r['bound'] / r['flips']:.1f}" if r["flips"] else ("—" if r["bound"] == 0 else "∞")
        cert_ = " | ".join("yes" if r["certified"][str(d)] else "no" for d in deltas)
        w(f"| {r['label']} | {r['loss_gap']} | {sci(r['eps_meas'])} | {sci(r['rel'], 2)} | {sci(r['mean_gap'], 2)} | "
          f"{r['flips']} | {r['flips_at_ties']} | {r['bound']} | {ratio} | {r['bound_per_position']} | {cert_} | {r['correct'] / n:.6f} | {r['correct'] - exact_acc:+d} |")
    w("\nChecks: every flip at a position whose margin is at most twice that position's gap: "
      f"{'yes' if argument else 'NO'}; `decisions` against the argmax of the realization's printed logits: "
      + (", ".join(f"{r['name']} {r['argmax_vs_decisions']}" for r in rows_ if r["argmax_vs_decisions"]) or "none differ") + ".")
    w("\nOut of sample: ε* (corrected) from one half of the windows, the margins of the other half below 2ε*:\n")
    w("| ε* from | δ | ε* | other half: m < 2ε* | fraction | ≤ δ? |")
    w("|---|---|---|---|---|---|")
    for h in halves:
        w(f"| {h['from']} windows ({h['n_a']}) | {h['delta']:.0e} | {sci(h['eps'], 4)} | {h['below_b']} of {h['n_b']} | {h['frac_b']:.2e} | {'yes' if h['frac_b'] <= h['delta'] else 'no'} |")
    w(f"\nThe statement (δ = 10⁻³, α = {a.alpha}): the curve's realizations certified at 10⁻³: "
      + (", ".join(r["label"] for r in cert) or "none") +
      f"; each flips at most δn = {d0 * n:.1f}: {'yes' if clause1 else 'NO'}; every curve realization's flips at most its bound: "
      f"{'yes' if clause2 else 'NO'} (every row's: {'yes' if every2 else 'NO'}). "
      f"The statement {'holds' if clause1 and clause2 else 'fails'}.")
    text = "\n".join(L) + "\n"
    open(a.out, "w").write(text)
    print(text)
    json.dump({"n": n, "windows": windows, "exact_correct": exact_acc, "ymax": Y, "ties": ties,
               "smallest": str(sm[0]), "median": str(sm[n // 2]),
               "star": {str(d): {k: (str(v) if isinstance(v, Decimal) else v) for k, v in s.items()} for d, s in star.items()},
               "cdf": [[e, c] for e, c in cdf],
               "rows": [{k: v for k, v in r.items() if k != "flip_positions"} for r in rows_],
               "halves": [{k: (str(v) if isinstance(v, Decimal) else v) for k, v in h.items()} for h in halves],
               "statement": {"certified": [r["name"] for r in cert], "clause1": clause1, "clause2": clause2,
                             "every_row_within_bound": every2, "argument_per_position": argument}},
              open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    # The files of record: the margins, and every realization's decisions.
    rec = os.path.join(a.dir, "record")
    os.makedirs(rec, exist_ok=True)
    with open(os.path.join(rec, "margins.tsv"), "w") as f:
        f.write("window\tposition\ttarget\ttop\tsecond\tmargin\n")
        for p in range(n):
            f.write(f"{p // T}\t{p % T}\t{target[p]}\t{top1[p]}\t{top2[p]}\t{margin[p]:.24e}\n")
    with open(os.path.join(rec, "flips.tsv"), "w") as f:
        f.write("realization\twindow\tposition\texact\trealization's\tmargin\n")
        for r in rows_:
            dec = [int(l) for l in open(os.path.join(a.dir, "decisions", r["name"] + ".txt"))]
            for p in r["flip_positions"]:
                f.write(f"{r['name']}\t{p // T}\t{p % T}\t{top1[p]}\t{dec[p]}\t{margin[p]:.24e}\n")


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "exact":
        exact(*rest)
    elif cmd == "native":
        native(*rest)
    elif cmd == "decisions":
        decisions(*rest)
    elif cmd == "report":
        report(rest)
    else:
        sys.exit(__doc__)
