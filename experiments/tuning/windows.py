#!/usr/bin/env python3
"""A tool's C program judged on the held-out windows (D154, step 3): the
program a tool rewrote, run as it stands on every window of `make
margins`, its logits against the exact ones as experiments/margins.py
compares a native run's.

  windows.py inputs VAL.psk.d EXPORT DIR
      the windows of VAL two by two as the export's inputs
      (DIR/NNNN.psk.d, NNNN the first window; the last, alone, paired
      with itself)
  windows.py judge PROGRAM WEIGHTS DIR EXACT NATIVE DECISIONS [CORE]
      PROGRAM (a built export) on every input of DIR; per chunk of the
      exact logits EXACT/NNNN.hilo, NATIVE/NNNN.tsv in margins.py's
      `native` format (per position the largest gap, the argmax, the sum
      of the gaps); DECISIONS the argmax per position, margins.py's
      `decisions` format. The argmax of the program's logits stands for the
      `decisions` entry, which the program does not export.

The standard library only.
"""

import array
import os
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import criterion  # noqa: E402

V, T = 92, 256


def header(path):
    with open(path, "rb") as f:
        f.read(6)
        (n,) = struct.unpack("<I", f.read(4))
        ty = f.read(n)
        (count,) = struct.unpack("<Q", f.read(8))
        return ty, count, f.read()


def inputs(val, export, out):
    ty, count, data = header(val)
    ety, _, edata = header(os.path.join(export, "conformance", "inputs.psk.d"))
    size = len(data) // count  # x then y, a byte per token
    half = size // 2
    if len(edata) != 2 * size:
        raise SystemExit(f"the export's inputs hold {len(edata)} bytes, not two windows of {size}")
    os.makedirs(out, exist_ok=True)
    for i in range(0, count, 2):
        j = i + 1 if i + 1 < count else i
        wi, wj = data[i * size:(i + 1) * size], data[j * size:(j + 1) * size]
        body = wi[:half] + wj[:half] + wi[half:] + wj[half:]
        with open(os.path.join(out, f"{i:04d}.psk.d"), "wb") as f:
            f.write(b"PSKD" + struct.pack("<H", 1) + struct.pack("<I", len(ety)) + ety + struct.pack("<Q", 1) + body)
    return count


def top(values):
    best = 0
    for i in range(1, len(values)):
        if values[i] >= values[best]:
            best = i
    return best


def judge(program, weights, dirn, exact, native, decisions, core=None):
    files = sorted(f for f in os.listdir(dirn) if f.endswith(".psk.d"))
    chunks = sorted(f[:-5] for f in os.listdir(exact) if f.endswith(".hilo"))
    count = sum(os.path.getsize(os.path.join(exact, c + ".hilo")) // (16 * T * V) for c in chunks)
    windows = {}
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "out.psk.d")
    for f in files:
        i = int(f[:4])
        cmd = ([] if core is None else ["taskset", "-c", str(core)]) + [os.path.join(program, "model"), weights, os.path.join(dirn, f), out]
        subprocess.run(cmd, check=True, capture_output=True)
        y = criterion.reals(out)
        windows[i] = y[: T * V]
        if i + 1 < count:
            windows[i + 1] = y[T * V:]
    os.makedirs(native, exist_ok=True)
    with open(decisions + ".part", "w") as dec:
        for c in chunks:
            hilo = array.array("d")
            with open(os.path.join(exact, c + ".hilo"), "rb") as f:
                hilo.frombytes(f.read())
            n = len(hilo) // (2 * T * V)
            k = 0
            with open(os.path.join(native, c + ".tsv.part"), "w") as tsv:
                for w in range(int(c), int(c) + n):
                    y = windows[w]
                    for p in range(T):
                        row = y[p * V:(p + 1) * V]
                        worst = total = 0.0
                        for v in row:
                            e = abs((v - hilo[k]) - hilo[k + 1])
                            k += 2
                            total += e
                            if e > worst:
                                worst = e
                        a = top(row)
                        tsv.write(f"{worst!r}\t{a}\t{total!r}\n")
                        dec.write(f"{a}\n")
            os.replace(os.path.join(native, c + ".tsv.part"), os.path.join(native, c + ".tsv"))
    os.replace(decisions + ".part", decisions)
    os.remove(out)
    os.rmdir(tmp)


def main(argv):
    if len(argv) == 4 and argv[0] == "inputs":
        print(inputs(argv[1], argv[2], argv[3]), "windows")
    elif len(argv) in (7, 8) and argv[0] == "judge":
        judge(*argv[1:7], core=argv[7] if len(argv) == 8 else None)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
