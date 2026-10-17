#!/usr/bin/env python3
"""Regroup the exact chunks `make margins` wrote into larger ones (D129).

    margins_rechunk.py SRC DST --from-chunk 4 --to-chunk 32 --windows 461

SRC holds experiments/margins.py exact's files of chunks of `--from-chunk`
windows (PREFIX.hilo, PREFIX.tsv, PREFIX the chunk's first window, four
digits); DST receives the same files for chunks of `--to-chunk` windows,
each the byte concatenation of the consecutive smaller ones it covers, in
order. Both files are flat in row order (a row per position: the hilo pairs
of its logits, a line of the tsv), so the concatenation is the file the
larger chunk's exact run would have written, byte for byte.

At the course constants the exact logits are taken in chunks of 4 windows,
so that twelve runs share the machine; the native realizations are run by
chunks of 32, because each `presk run --json` prints the 10.8 M parameters
with its result and fewer runs print them fewer times
(experiments/margins_course.sh with CHUNK=32 on DST). It refuses a chunk
that is missing or a larger chunk that does not start on a smaller one.
"""
import argparse
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--from-chunk", type=int, required=True)
ap.add_argument("--to-chunk", type=int, required=True)
ap.add_argument("--windows", type=int, required=True)
a = ap.parse_args()
if a.to_chunk % a.from_chunk:
    sys.exit("margins_rechunk.py: --to-chunk must be a multiple of --from-chunk")
os.makedirs(a.dst, exist_ok=True)
for big in range(0, a.windows, a.to_chunk):
    parts = list(range(big, min(big + a.to_chunk, a.windows), a.from_chunk))
    for ext in ("hilo", "tsv"):
        out = os.path.join(a.dst, f"{big:04d}.{ext}")
        with open(out + ".part", "wb") as o:
            for p in parts:
                src = os.path.join(a.src, f"{p:04d}.{ext}")
                if not os.path.exists(src):
                    sys.exit(f"margins_rechunk.py: missing {src}")
                with open(src, "rb") as f:
                    o.write(f.read())
        os.replace(out + ".part", out)
    print(f"{big:04d}: windows {big}..{min(big + a.to_chunk, a.windows) - 1} from {len(parts)} chunks")
