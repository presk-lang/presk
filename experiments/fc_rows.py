"""The report of experiments/fc_rows.sh (plan 0.18.2 step 5, D219; its phase
`fc4`, FC4's rows alone, step 6, D220).

    python3 experiments/fc_rows.py OUT > OUT/report.md

Per row: the median of its processes' medians and the fastest call, in ms
(D126); each ratio is of the medians. The statements' verdicts are FC4's
and FC5's thresholds (0.85x); the rest is reported.
"""
import re
import statistics
import sys
from collections import defaultdict

out = sys.argv[1]
rows = defaultdict(list)
failed = defaultdict(int)
for line in open(f"{out}/times/rows.txt"):
    r, key, med, best = line.split()
    if med == "failed":
        failed[key] += 1
        continue
    rows[key].append((float(med), float(best)))


def med(key):
    return statistics.median(m for m, _ in rows[key]) if rows[key] else float("nan")


def best(key):
    return min(b for _, b in rows[key]) if rows[key] else float("nan")


def cell(key):
    n = len(rows[key])
    f = f", {failed[key]} failed" if failed[key] else ""
    return f"{med(key):.2f} | {best(key):.2f} | {n}{f}"


def ratio(a, b):
    return med(a) / med(b)


def listing(path):
    """The solver's bytes: the realization's traffic and all-f64's, the parameters at rest."""
    text = open(path).read()
    cost = re.search(r"^  cost\s+([0-9.e+-]+) against ([0-9.e+-]+) all-f64", text, re.M)
    rest = re.search(r"([0-9.e+-]+) bytes at rest against ([0-9.e+-]+) in f64", text)
    gap = re.search(r"^  measured\s+([0-9.e+-]+)", text, re.M)
    return {
        "bytes": float(cost.group(1)), "f64": float(cost.group(2)),
        "rest": float(rest.group(1)) if rest else None, "rest64": float(rest.group(2)) if rest else None,
        "gap": float(gap.group(1)) if gap else None,
    }


print("# Plan 0.18.2's rows: FC4, FC5, FC6's cost and FC7\n" if any(k.startswith("fc5") for k in rows) else "# Plan 0.18.2's rows: FC4 again (step 6, D220)\n")
print(open(f"{out}/machine.txt").read().strip().replace("\n", "  \n"), "\n")
print("Each row: the median of its processes' medians, the fastest call, in ms, and the processes (D126).\n")
if failed:
    print(f"**Failed processes:** {', '.join(f'{k} ({n})' for k, n in sorted(failed.items()))}; their logs under `log/`.\n")


def table(title, keys):
    print(f"## {title}\n")
    print("| row | median ms | fastest ms | processes |\n|---|---|---|---|")
    for k, label in keys:
        print(f"| {label} | {cell(k)} |")
    print()


table("FC4: the AdamW step at the course constants", [
    ("fc4.llvm.f32", "LLVM, all-`f32`"), ("fc4.llvm.fc4", "LLVM, binary16 where offered"),
    ("fc4.native.f32", "native, all-`f32`"), ("fc4.native.fc4", "native, binary16 where offered"),
    ("fc4.c.f32", "C (GCC 16.2), all-`f32`"), ("fc4.c.fc4", "C (GCC 16.2), binary16 where offered"),
])
r4 = ratio("fc4.llvm.fc4", "fc4.llvm.f32")
print(f"**FC4 {'holds' if r4 <= 0.85 else 'fails'}:** {r4:.3f}x all-`f32` on the LLVM backend (at most 0.85x). "
      f"Reported: native {ratio('fc4.native.fc4', 'fc4.native.f32'):.3f}x, C {ratio('fc4.c.fc4', 'fc4.c.f32'):.3f}x.\n")
try:
    print(f"The realization's result on the export's inputs, natively: {open(f'{out}/fc4/finite.txt').read().strip()}\n")
except FileNotFoundError:
    pass
# `fc_rows.sh fc4` (plan 0.18.2 step 6, D220) times FC4's rows alone.
if not any(k.startswith("fc5") for k in rows):
    sys.exit(0)

table("FC5: the forward pass, B = 16, T = 256", [
    ("fc5.llvm.f32", "LLVM, default layouts, all-`f32`"), ("fc5.llvm.f16", "LLVM, default layouts, all-binary16"),
    ("fc5.llvm.e8m15", "LLVM, default layouts, all-`e8m15`"),
    ("fc5.llvm-plan.f32", "LLVM, the profile's plan, all-`f32`"), ("fc5.llvm-plan.f16", "LLVM, the profile's plan, all-binary16"),
    ("fc5.native.f32", "native, default layouts, all-`f32`"), ("fc5.native.f16", "native, default layouts, all-binary16"),
    ("fc5.native.e8m15", "native, default layouts, all-`e8m15`"),
])
r5 = ratio("fc5.llvm.f16", "fc5.llvm.f32")
print(f"**FC5 {'holds' if r5 <= 0.85 else 'fails'}:** {r5:.3f}x all-`f32` on the LLVM backend under the default layouts (at most 0.85x). "
      f"Reported: under the profile's plan {ratio('fc5.llvm-plan.f16', 'fc5.llvm-plan.f32'):.3f}x; natively {ratio('fc5.native.f16', 'fc5.native.f32'):.3f}x; "
      f"in `e8m15` {ratio('fc5.llvm.e8m15', 'fc5.llvm.f32'):.3f}x on LLVM and {ratio('fc5.native.e8m15', 'fc5.native.f32'):.3f}x natively.\n")

table("FC6: the guard's cost, FC5's forward pass in binary16 in the exported C (GCC 16.2)", [
    ("fc6.c.f32", "all-`f32`"), ("fc6.c.guard", "all-binary16, with the guard"),
    ("fc6.c.noguard", "all-binary16, `-DPRESK_NO_GUARD`"),
])
g = ratio("fc6.c.guard", "fc6.c.noguard")
print(f"**FC6's cost:** the guarded program takes {g:.4f}x the unguarded one's time ({100 * (g - 1):+.2f} %); "
      f"binary16 against `f32` in C: {ratio('fc6.c.guard', 'fc6.c.f32'):.3f}x.\n")

print("## FC7: what a certified realization buys in time\n")
L = {
    "d64.d170": listing("out/v6-0.16.1/solve.txt"),
    "d64.fc3": listing("out/fc3-0.18.2/solve/mt6.txt"),
    "course.d170": listing("out/margins-course-0.16.1/logits/solve.txt"),
    "course.fc7": listing(f"{out}/fc7/course/solve.txt"),
}
print("`logits`, B = 16, T = 256, the default layouts. Bytes: the solver's traffic (the parameters once and one sample's values, D157) "
      "and the parameters at rest; all-`f32`'s are all-`f64`'s halved.\n")
print("| scale | realization | bytes | at rest | gap measured | LLVM ms | fastest | vs `f32` | C ms | fastest | vs `f32` |\n|---|---|---|---|---|---|---|---|---|---|---|")
for scale, name in [("d64", "D = 64"), ("course", "course")]:
    any_l = L[f"{scale}.d170"]
    for r, label in [("f32", "all-`f32`"), ("d170", "D170's"), ("fc3" if scale == "d64" else "fc7", "FC3's (MT6)" if scale == "d64" else "solved with binary16 and `e8m15`")]:
        if r == "f32":
            b, rest, gap = any_l["f64"] / 2, any_l["rest64"] / 2, None
        else:
            l = L[f"{scale}.{r}"]
            b, rest, gap = l["bytes"], l["rest"], l["gap"]
        kl, kc = f"fc7.{scale}.llvm.{r}", f"fc7.{scale}.c.{r}"
        gs = f"{gap:.3e}" if gap is not None else "—"
        print(f"| {name} | {label} | {b:.4e} | {rest:.4e} | {gs} | {med(kl):.2f} | {best(kl):.2f} | {ratio(kl, f'fc7.{scale}.llvm.f32'):.3f}x "
              f"| {med(kc):.2f} | {best(kc):.2f} | {ratio(kc, f'fc7.{scale}.c.f32'):.3f}x |")
print()
for scale, new in [("d64", "fc3"), ("course", "fc7")]:
    print(f"- {scale}: against D170's, {ratio(f'fc7.{scale}.llvm.{new}', f'fc7.{scale}.llvm.d170'):.3f}x on LLVM, "
          f"{ratio(f'fc7.{scale}.c.{new}', f'fc7.{scale}.c.d170'):.3f}x in C; bytes {L[f'{scale}.{new}']['bytes'] / L[f'{scale}.d170']['bytes']:.3f}x.")
