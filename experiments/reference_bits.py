"""SR2's report (experiments/reference_bits.sh, D192): the eight solves at
D = 64 with the reference at 53 and 64 bits against the record's at 128.

    python3 experiments/reference_bits.py out/reference-bits > out/reference-bits/report.md

For each row and precision: the realization byte for byte against the one
of record, the measured gap against the one read at 128 bits, the widening
rounds, the wall time and the peak. SR2 holds when every realization is the
record's byte for byte and every gap is within 1 % of the record's.
"""
import os
import re
import sys

BUDGETS = ["1e-1", "1e-2", "1e-3", "1e-4", "1e-5", "1e-6", "1e-7"]
TOLERANCE = 0.01


def rows(out):
    """(name, record realization, record report, file stem under bits-N/)."""
    for b in BUDGETS:
        yield f"curve {b}", f"out/curve-trained_{b}.json", f"out/curve-trained_{b}.txt", f"curve-trained_{b}"
    yield "V6 at 4.67e-4", "out/v6-0.16.1/solved.json", os.path.join(out, "record/v6/solve.txt"), "v6"


def measured(path):
    """The gap and the widening rounds of a solve's report."""
    text = open(path).read()
    m = re.search(r"^  measured\s+(\S+) against the exact reference, after (\d+) widening round", text, re.M)
    if not m:
        return None, None
    return float(m.group(1)), int(m.group(2))


def timed(path):
    """Wall seconds and peak GB from /usr/bin/time -v."""
    if not os.path.exists(path):
        return None, None
    text = open(path).read()
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    secs = None
    if wall:
        secs = 0.0
        for part in wall.group(1).split(":"):
            secs = secs * 60 + float(part)
    return secs, (int(rss.group(1)) / 1e6 if rss else None)


def fmt(x, spec):
    return "—" if x is None else format(x, spec)


def main():
    out = sys.argv[1]
    bits = sorted(int(d[5:]) for d in os.listdir(out) if d.startswith("bits-"))
    print("# SR2: the eight solves at D = 64 with the reference at 53 and 64 bits\n")
    print("The record is D170's (the reference at 128 bits). *Same* is the realization's JSON byte for byte; "
          "*Δ gap* is the measured gap's relative difference to the record's.\n")
    print("| row | bits | same | gap (128) | gap | Δ gap | widening rounds (128 → bits) | wall s | peak GB |")
    print("|---|---|---|---|---|---|---|---|---|")
    holds = True
    missing = False
    for name, rec_json, rec_txt, stem in rows(out):
        rgap, rwid = measured(rec_txt)
        rec = open(rec_json, "rb").read()
        for n in bits:
            d = os.path.join(out, f"bits-{n}")
            j, t = os.path.join(d, stem + ".json"), os.path.join(d, stem + ".txt")
            if not (os.path.exists(j) and os.path.exists(t)):
                print(f"| {name} | {n} | not run | {fmt(rgap, '.4g')} | | | | | |")
                missing = True
                continue
            same = open(j, "rb").read() == rec
            gap, wid = measured(t)
            delta = None if gap is None or rgap is None else abs(gap - rgap) / rgap
            wall, peak = timed(os.path.join(d, stem + ".time"))
            ok = same and delta is not None and delta <= TOLERANCE
            holds &= ok
            print(f"| {name} | {n} | {'yes' if same else '**no**'} | {fmt(rgap, '.4g')} | {fmt(gap, '.4g')} | "
                  f"{fmt(delta, '.2%')} | {rwid} → {wid} | {fmt(wall, '.1f')} | {fmt(peak, '.2f')} |")
    rwall, rpeak = timed(os.path.join(out, "record/v6/solve.time"))
    print(f"\nV6's solve of record at 128 bits: {fmt(rwall, '.1f')} s, {fmt(rpeak, '.2f')} GB "
          "(the curve's solves of record were not timed).\n")
    if missing:
        print("**SR2: not every solve has run.**")
    else:
        print(f"**SR2 {'holds' if holds else 'fails'}.**")


if __name__ == "__main__":
    main()
