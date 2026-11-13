"""SR4's report (experiments/step_plan.sh, D194): the course GPT's step under
the cost model's plans, all-f64, all-f32 and the step solved at 1e-3.

    python3 experiments/step_plan.py out/step-plan 7 64 > out/step-plan/report.md

SR4: under its plan the solved step is at least 1.3x faster than all-f64
under its plan (J′'s threshold), its ratio to all-f32 reported; read off the
medians of the processes' median steps (D126). The prediction: the solved
step within 3 % of all-f32. J′'s own measure (the four-step process minus
the one-step one, over three) is reported beside, and so, outside the
statement, are a longer process a row and the host by phase.
"""
import json
import os
import re
import statistics
import sys

ROWS = ["f64", "f32", "solved"]
THRESHOLD = 1.3
PREDICTION = 0.03
SLOW = 1.5


def walls(out):
    w = {}
    for line in open(os.path.join(out, "wall.txt")):
        key, secs = line.split()
        w[key] = float(secs)
    return w


def sums(out):
    s = {}
    for line in open(os.path.join(out, "ckpt.sha256")):
        digest, path = line.split()
        s[path.split("/")[0]] = digest
    return s


def step_ms(out, key):
    """The median step `presk train` prints, the first excluded, and its loss."""
    text = open(os.path.join(out, "log", key + ".log")).read()
    m = re.search(r"^presk train: ([0-9.]+) ms a step, median of (\d+)", text, re.M)
    loss = re.search(r"final loss ([0-9.]+)", text)
    return (float(m.group(1)), int(m.group(2))) if m else (None, None), loss.group(1) if loss else None


def peak_gb(out, key):
    p = os.path.join(out, "log", key + ".time")
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", open(p).read()) if os.path.exists(p) else None
    return int(m.group(1)) / 1e6 if m else None


def host(out, key):
    """`--host-profile`'s phases (ms a step, MB a step) and its two summary lines."""
    phases, notes = {}, []
    for line in open(os.path.join(out, "log", key + ".log")):
        m = re.match(r"^(\s*[a-z][a-z.\-]*)\s+([0-9.]+)\s+[0-9.]+ %\s+(\S+)", line)
        if m:
            phases[m.group(1).rstrip()] = (float(m.group(2)), m.group(3))
        elif line.startswith("-- ") and ("page faults" in line or "launches a step" in line):
            notes.append(line[3:].strip())
    return phases, notes


def layouts(out):
    p = os.path.join(out, "plans")
    return {r: json.load(open(os.path.join(p, f"step-{r}.json")))["layout"] for r in ROWS}


def main():
    out, repeat = sys.argv[1], int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else None
    w, s = walls(out), sums(out)
    rounds = sorted({int(k[1:].split("-")[0]) for k in w if re.match(r"r\d+-", k)})
    print("# SR4: the budgeted step under the cost model's plan\n")
    print(open(os.path.join(out, "machine.txt")).read().strip().replace("\n", "  \n"), "\n")

    print("## The plans\n")
    lay = layouts(out)
    for r in ROWS:
        diff = sum(1 for k in lay[r] if lay[r][k] != lay["f32"].get(k))
        print(f"- {r}: {len(lay[r])} nests, {diff} with another layout than all-`f32`'s.")

    print("\n## A layout moves no bit (two steps, the weights after them)\n")
    bits = True
    for r in ROWS:
        same = f"bits-plan-{r}" in s and s.get(f"bits-default-{r}") == s[f"bits-plan-{r}"]
        bits &= same
        print(f"- {r}: the default layout's and the plan's weights {'the same bits' if same else '**differ**'}.")
    long = f"s{repeat + 1}"
    pairs = [("bits-plan-{}", 2), (f"r{rounds[0]}-{{}}-{long}", repeat + 1)]
    if steps and "long-solved" in s:
        pairs.append(("long-{}", steps))
    for key, n in pairs:
        same = s.get(key.format("solved")) == s.get(key.format("f32"))
        print(f"- solved against all-`f32` under the plans, after {n} steps: {'the same bits' if same else 'different bits'}.")
    for r in ROWS:
        ds = {s.get(f"r{k}-{r}-{long}") for k in rounds}
        print(f"- {r}: the {repeat + 1}-step weights of the {len(rounds)} rounds: "
              f"{'one checksum' if len(ds) == 1 else f'{len(ds)} checksums'}.")

    print(f"\n## The step under its plan, ten X925 pinned, {len(rounds)} processes a row\n")
    print(f"*Median step*: `presk train`'s median of {repeat} steps after the first, one process a round. "
          "*J′'s measure*: the four-step process's wall time minus the one-step one's, over three.\n")
    print("| row | " + " | ".join(f"round {k}" for k in rounds) + " | median | fastest | max ÷ min | vs all-`f64` | vs all-`f32` |")
    print("|---|" + "---|" * len(rounds) + "---|---|---|---|---|")
    med, fast, spread, jm = {}, {}, {}, {}
    for r in ROWS:
        xs = [step_ms(out, f"r{k}-{r}-{long}")[0][0] for k in rounds]
        med[r], fast[r], spread[r] = statistics.median(xs), min(xs), max(xs) / min(xs)
        jm[r] = [(w[f"r{k}-{r}-s4"] - w[f"r{k}-{r}-s1"]) / 3 for k in rounds]
    for r in ROWS:
        xs = [step_ms(out, f"r{k}-{r}-{long}")[0][0] for k in rounds]
        print(f"| {r} | " + " | ".join(f"{x:.1f}" for x in xs) + f" | **{med[r]:.1f}** | {fast[r]:.1f} | {spread[r]:.3f} | "
              f"{med['f64'] / med[r]:.2f}× | {med['f32'] / med[r]:.3f}× |")
    print("\nIn milliseconds a step.\n")
    print("| row (J′'s measure) | " + " | ".join(f"round {k}" for k in rounds) + " | median | max ÷ min | vs all-`f64` | vs all-`f32` |")
    print("|---|" + "---|" * len(rounds) + "---|---|---|---|")
    hm = {r: statistics.median(jm[r]) for r in ROWS}
    for r in ROWS:
        print(f"| {r} | " + " | ".join(f"{x:.3f}" for x in jm[r]) + f" | **{hm[r]:.3f}** | {max(jm[r]) / min(jm[r]):.3f} | "
              f"{hm['f64'] / hm[r]:.2f}× | {hm['f32'] / hm[r]:.3f}× |")
    print("\nIn seconds a step.\n")
    print("| row | final loss after " + f"{repeat + 1} steps | peak GB ({repeat + 1} steps, round 1) |")
    print("|---|---|---|")
    for r in ROWS:
        print(f"| {r} | {step_ms(out, f'r{rounds[0]}-{r}-{long}')[1]} | {peak_gb(out, f'r{rounds[0]}-{r}-{long}'):.2f} |")

    print("\n## The statement\n")
    ratio = med["f64"] / med["solved"]
    to32 = med["solved"] / med["f32"]
    slow = max(spread.values()) > SLOW
    print(f"- The solved step {med['solved']:.1f} ms against all-`f64`'s {med['f64']:.1f}: **{ratio:.2f}×** "
          f"(fastest against fastest {fast['f64'] / fast['solved']:.2f}×; J′'s measure {hm['f64'] / hm['solved']:.2f}×). "
          f"At least {THRESHOLD}×: {'yes' if ratio >= THRESHOLD else 'no'}.")
    print(f"- Its ratio to all-`f32` ({med['f32']:.1f} ms): the solved step takes {to32:.4f} of its time "
          f"(J′'s measure {hm['solved'] / hm['f32']:.4f}). The prediction, within {PREDICTION:.0%}: "
          f"{'met' if abs(to32 - 1) <= PREDICTION else 'missed'}.")
    print(f"- No process in a slow mode (no row's slowest round {SLOW}× its fastest): {'yes' if not slow else 'no'}.")
    print(f"- A layout moves no bit: {'yes' if bits else 'no'}, so the plan keeps J′'s update gap.")
    print(f"\n**SR4 {'holds' if ratio >= THRESHOLD else 'fails'}.**")
    if steps is None or f"long-solved" not in w:
        return

    print(f"\n## Beside the statement: {steps} steps, and the host by phase\n")
    print(f"| row | median step, {repeat} steps (the rounds' median) | median step, {steps - 1} steps (one process) | "
          f"vs all-`f64` | vs all-`f32` | peak GB, {repeat + 1} steps | peak GB, {steps} steps |")
    print("|---|---|---|---|---|---|---|")
    lm = {r: step_ms(out, f"long-{r}")[0][0] for r in ROWS}
    for r in ROWS:
        print(f"| {r} | {med[r]:.1f} | {lm[r]:.1f} | {lm['f64'] / lm[r]:.2f}× | {lm['f32'] / lm[r]:.3f}× | "
              f"{peak_gb(out, f'r{rounds[0]}-{r}-{long}'):.2f} | {peak_gb(out, f'long-{r}'):.2f} |")
    keys = [f"hp{n}-{r}" for n in (repeat + 1, steps) for r in ("f32", "solved")]
    prof = {k: host(out, k) for k in keys if os.path.exists(os.path.join(out, "log", k + ".log"))}
    if len(prof) < len(keys):
        return
    print(f"\n`presk train --host-profile`, ms a step (MB a step), the mean of the steps after the first; "
          "the phases where the two rows differ by more than 1 ms, and the totals.\n")
    print("| phase | " + " | ".join(f"{k.split('-')[1]}, {k[2:].split('-')[0]} steps" for k in keys) + " |")
    print("|---|" + "---|" * len(keys))
    names = list(prof[keys[0]][0])
    for name in names:
        vals = [prof[k][0].get(name, (0.0, "—")) for k in keys]
        if name.strip() != "total" and max(abs(vals[1][0] - vals[0][0]), abs(vals[3][0] - vals[2][0])) <= 1.0:
            continue
        print(f"| `{name.strip()}` | " + " | ".join(f"{v:.1f}" + (f" ({mb})" if mb != "—" else "") for v, mb in vals) + " |")
    print()
    for k in keys:
        print(f"- {k}: " + "; ".join(prof[k][1]) + ".")


if __name__ == "__main__":
    main()
