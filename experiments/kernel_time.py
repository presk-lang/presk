"""Where an exported C program's time goes, by kind of kernel (the PyTorch
baseline at 0.9.0, D87).

    python3 experiments/kernel_time.py instrument DIR/model.c > DIR/model_timed.c
    python3 experiments/kernel_time.py report DIR TIMES CALLS [--totals]

`instrument` copies the model.c of a program `presk export --target c`
wrote, each launch timed on the thread that makes it: a line calling a
kernel `kN(…)`, or the `#ifdef _OPENMP … #endif` block of an OpenMP region
that divides one (so the time is the region's, its team's start and its
wait for the slowest thread included). The copy is built in place of
model.c with experiments/kernel_time.c, whose clock keeps the launches in
order; the kernels are not touched.

`report` reads what the program wrote (TIMES) after CALLS calls of the
entry (`model -r CALLS`), leaves out the first call's launches (the first
touch of the arena's pages, OpenMP's team started), and prints a Markdown
table: each kind of kernel's launches, operations, time and rate per call
and its share of the kernels' time, then the kernels that take the most,
the contraction kernels' operations counting every launch. The call's own
time is not known here: with --totals, one line, `kernels S launches N`,
seconds and launches a call.
"""
import math
import re
import sys

KINDS = [
    ("the contractions `dot` and `ein`", lambda k: product(k) == "dot" or product(k).startswith("ein")),
    ("the derivatives of `dot` (`dot_vjp_l`, `dot_vjp_r`)", lambda k: product(k).startswith("dot_vjp")),
    ("every other kernel", lambda k: True),
]


def product(kind):
    """A kernel's primitive: `dot+add`, a product with the bias as its
    epilogue (plan 0.10.0 step 7), is a `dot`."""
    return kind.removesuffix("+add")

CALL = re.compile(r"^(\s*)k(\d+)\(.*\);\s*$")


def instrument(path):
    lines = open(path).read().split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("#ifdef _OPENMP"):
            j = i
            while not lines[j].startswith("#endif"):
                j += 1
            block = lines[i : j + 1]
            ks = {m.group(2) for m in map(CALL.match, block) if m}
            if any(l.startswith("#pragma omp parallel") for l in block):
                assert len(ks) == 1, block
                k = ks.pop()
                out.append("    { double kt_t0 = kt_now();")
                out.extend(block)
                out.append(f"    kt_launch({k}, kt_now() - kt_t0); }}")
            else:
                assert not ks, block
                out.extend(block)
            i = j + 1
            continue
        m = CALL.match(line)
        if m:
            call = line.strip()
            out.append(f"{m.group(1)}{{ double kt_t0 = kt_now(); {call} kt_launch({m.group(2)}, kt_now() - kt_t0); }}")
        else:
            out.append(line)
            if line.startswith('#include "model.h"'):
                out.append("double kt_now(void);")
                out.append("void kt_launch(int kernel, double seconds);")
        i += 1
    text = "\n".join(out)
    assert "kt_now(void);" in text, "no #include \"model.h\" in model.c"
    return text


def shapes(h):
    """The operands' and the result's axes of a kernel's header, as lists of
    (name or None, size)."""
    out = []
    for m in re.finditer(r"%\d+ : \w+\[([^\]]*)\]", h):
        axes = []
        for a in m.group(1).split(","):
            name, _, n = a.strip().rpartition(":")
            axes.append((name.strip() or None, int(n)))
        out.append(axes)
    return out


def flops(kind, h):
    """A contraction kernel's floating-point operations, a multiply and an
    add per term; 0 for any other kernel."""
    kind = product(kind)
    if not (kind == "dot" or kind.startswith("ein") or kind.startswith("dot_vjp")):
        return 0
    # The operands, then the result; an epilogue's operand after the two.
    s = shapes(h)
    a, r = s[0], s[-1]
    size = lambda axes: math.prod(n for _, n in axes)
    if kind in ("dot", "dot_vjp_l"):
        k = a[-1][1]
    elif kind == "dot_vjp_r":
        k = size(a[:-1])
    else:
        kept = set(kind[4:-1].replace(" ", "").split(","))
        k = math.prod(n for name, n in a if name not in kept)
    return 2 * size(r) * k


def report(out, times, calls, totals):
    kind, header = {}, {}
    text = open(f"{out}/kernels.c").read()
    # `PRESK_KERNEL void kN(` since 0.10.0 step 7's follow-up, `static void
    # kN(` before.
    for m in re.finditer(r"^// kernel (.*?) (\((?:%0 .*?)?\) -> .*?) \{\n(?:(?://.*)\n)*(?:static|PRESK_KERNEL) void k(\d+)\(", text, re.M):
        kind[int(m.group(3))] = m.group(1)
        header[int(m.group(3))] = f"{m.group(1)} {m.group(2)}"
    records = [(int(k), float(s)) for k, s in (line.split() for line in open(times))]
    assert len(records) % calls == 0, (len(records), calls)
    per_call = len(records) // calls
    n = calls - 1
    per = {}
    for k, s in records[per_call:]:
        c, t = per.get(k, (0, 0.0))
        per[k] = (c + 1, t + s)
    total = sum(t for _, t in per.values()) / n
    if totals:
        print(f"kernels {total:.6f} launches {per_call}")
        return
    print("| kernels | launches a call | GFLOP a call | s a call | GFLOP/s | share of the kernels' time |")
    print("|---|---|---|---|---|---|")
    left = dict(per)
    for label, pred in KINDS:
        mine = {k: v for k, v in left.items() if pred(kind[k])}
        for k in mine:
            del left[k]
        launches = sum(c for c, _ in mine.values()) // n
        s = sum(t for _, t in mine.values()) / n
        g = sum(c * flops(kind[k], header[k]) for k, (c, _) in mine.items()) / n / 1e9
        rate = f"{g / s:.0f}" if g else ""
        print(f"| {label} | {launches} | {g:.1f} | {s:.3f} | {rate} | {100 * s / total:.1f} % |")
    print(f"| every kernel | {per_call} | | **{total:.3f}** | | 100 % |")
    print()
    print("| kernel | launches a call | GFLOP/s | s a call | share |")
    print("|---|---|---|---|---|")
    for k, (c, t) in sorted(per.items(), key=lambda kv: -kv[1][1])[:8]:
        h = header[k]
        f = flops(kind[k], h)
        rate = f"{c * f / t / 1e9:.0f}" if f else ""
        h = h if len(h) <= 110 else h[:109] + "…"
        print(f"| `k{k}`: `{h}` | {c // n} | {rate} | {t / n:.3f} | {100 * t / n / total:.1f} % |")


def main():
    if sys.argv[1:2] == ["instrument"]:
        print(instrument(sys.argv[2]))
    elif sys.argv[1:2] == ["report"]:
        report(sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5:] == ["--totals"])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
