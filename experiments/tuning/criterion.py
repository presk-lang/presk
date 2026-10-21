#!/usr/bin/env python3
"""The criterion a tuning tool tests (D154): the largest elementwise gap of
the logits to the exact reference's, on the export's conformance inputs.

  criterion.py OUT.psk.d EXACT.psk.d [--f64 NATIVE.psk.d] [--budget EPS]
      prints `gap`, `relative` (the gap over the largest |exact logit|),
      `mean`, and with --f64 the same gap measured against the `f64` run
      (the reference a tool takes when it has no other); with --budget,
      exits 0 when the gap is at most EPS and 1 when it is not

The gap is absolute, as ε*(δ) is: a number on the logits, which bounds the
decisions a realization changes through the margins (D118), and which
`presk solve --budget-abs` takes (V6). The standard library only.
"""

import array
import struct
import sys


def reals(path):
    """The real values of a one-sample `.psk.d` of one real tensor, as
    doubles."""
    with open(path, "rb") as f:
        if f.read(4) != b"PSKD":
            raise SystemExit(f"{path}: not a .psk.d file")
        (version,) = struct.unpack("<H", f.read(2))
        (tlen,) = struct.unpack("<I", f.read(4))
        ty = f.read(tlen).decode("utf-8")
        (count,) = struct.unpack("<Q", f.read(8))
        if not ty.startswith("ℝ") or count != 1:
            raise SystemExit(f"{path}: {count} samples of {ty}, not one real tensor")
        a = array.array("d")
        a.frombytes(f.read())
        if sys.byteorder != "little":
            a.byteswap()
        return a


def gap(a, e):
    worst = total = 0.0
    for x, y in zip(a, e):
        d = abs(x - y)
        total += d
        if d > worst:
            worst = d
    return worst, total / len(e)


def main(argv):
    out, ex = argv[0], argv[1]
    native = budget = None
    rest = argv[2:]
    while rest:
        k, v = rest[0], rest[1]
        rest = rest[2:]
        if k == "--f64":
            native = v
        elif k == "--budget":
            budget = float(v)
        else:
            raise SystemExit(f"criterion.py: unknown option {k}")
    a, e = reals(out), reals(ex)
    if len(a) != len(e):
        raise SystemExit(f"{len(a)} values against {len(e)}")
    worst, mean = gap(a, e)
    mag = max(abs(y) for y in e)
    print(f"gap {worst:.6e}")
    print(f"relative {worst / mag:.6e}")
    print(f"mean {mean:.6e}")
    if native:
        w64, _ = gap(a, reals(native))
        print(f"gap_f64 {w64:.6e}")
    if budget is not None:
        ok = worst <= budget
        print(f"budget {budget:.6e} {'pass' if ok else 'fail'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
