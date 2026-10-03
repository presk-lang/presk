#!/usr/bin/env python3
"""The report of experiments/host_profile.sh (plan 0.10.0 steps 5 and 6, D95, D96).

    python3 experiments/host_profile.py OUT > OUT/report.md

Reads the tables `presk bench --host-profile` and `presk train
--host-profile` printed into OUT/log/, and writes one table per
configuration that ran (CONFIGS): each phase's milliseconds a call (or
step) and its share, for the forward pass, the gradient, the optimizer's
step and the training step.
"""

import re
import sys
from pathlib import Path

WORKLOADS = [("fwd", "forward pass"), ("grad", "gradient"), ("adam", "`adamw.step`"), ("step", "training step")]
ROW = re.compile(r"^(\s*)(\S+)\s+([\d.]+)\s+([\d.]+) %\s+(\S+)\s+(\S+)\s*(.*)$")


def parse(path):
    """The phases of one log: {key: (ms, share, MB, n, what)}, in order, and the head lines."""
    rows, head = {}, []
    if not path.exists():
        return rows, head
    for line in path.read_text().splitlines():
        if line.startswith("presk bench --host-profile") or line.startswith("-- a call") or line.startswith("-- the outputs") or line.startswith("presk train"):
            head.append(line)
        if line.startswith("-- ") and ("launches a" in line or "page faults" in line):
            head.append(line)
        m = ROW.match(line)
        if m and m.group(2) not in ("phase",):
            indent, key, ms, share, mb, n, what = m.groups()
            rows[key] = (float(ms), float(share), mb, n, what, bool(indent))
    return rows, head


def main():
    out = Path(sys.argv[1])
    print("# The native host, measured (plan 0.10.0)\n")
    print("```")
    print((out / "machine.txt").read_text().rstrip())
    print("```\n")
    print("The course GPT at `B = 16`, `T = 256`, `f32`, under the plans the frozen profile gives; "
          "each phase's mean a call (a step for the training step) in ms, and its share of the call. "
          "`kernels` is the kernels' own code; groups 1 to 6 are the host (plan 0.10.0 §3.3), "
          "`left.interpreter` by difference.\n")
    configs = (
        ("10", "10 threads, ten X925 cores, pinned"),
        ("20", "20 threads, every core, unpinned"),
        ("10-heap", "10 threads, ten X925 cores, pinned, glibc keeping its pages (`GLIBC_TUNABLES`: no block mapped apart, none returned)"),
    )
    for th, title in configs:
        if not (out / "log" / f"fwd.{th}.log").exists():
            continue
        logs = {}
        for key, _ in WORKLOADS:
            name = f"step.profiled.{th}.log" if key == "step" else f"{key}.{th}.log"
            logs[key] = parse(out / "log" / name)
        print(f"## {title}\n")
        print("| phase | " + " | ".join(w for _, w in WORKLOADS) + " | what |")
        print("|---|" + "---:|" * len(WORKLOADS) + "---|")
        keys = []
        for key, _ in WORKLOADS:
            for k in logs[key][0]:
                if k not in keys:
                    keys.append(k)
        for k in keys:
            cells, what, sub = [], "", False
            for key, _ in WORKLOADS:
                r = logs[key][0].get(k)
                if r is None:
                    cells.append("—")
                else:
                    ms, share, _, _, w, sub = r
                    what = what or w
                    cells.append(f"{ms:.1f} ({share:.1f} %)")
            name = f"&nbsp;&nbsp;{k}" if sub else f"**{k}**"
            print(f"| {name} | " + " | ".join(cells) + f" | {what} |")
        print()
        for key, w in WORKLOADS:
            for line in logs[key][1]:
                print(f"- {w}: {line}")
        plain = (out / "log" / f"step.plain.{th}.log")
        if plain.exists():
            m = re.search(r"presk train: ([\d.]+) ms a step", plain.read_text())
            if m:
                print(f"- training step without the timers: {m.group(1)} ms a step (the median, the first left out)")
        print()


if __name__ == "__main__":
    main()
