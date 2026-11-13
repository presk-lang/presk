#!/usr/bin/env python3
"""SR5's reader (D201; experiments/sr5.sh writes what it reads).

  sr5.py slow OUT ROW            exit 0 if a process of ROW is slow, 1 if none, 2 on an error
  sr5.py report OUT --d190 DIR   OUT/report.md and OUT/report.json

The slow mode, as sr5.sh fixes it before the runs: a process whose call's
median without the timers exceeds the fastest process's of the same row by
more than 20 %. The median and the fastest across processes are D127's
(retime_processes.py's rule): the median of the processes' figures, and the
smallest.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

SLOW = 1.20
NAMES = ["f32", "bf16-params-f32", "int8-absmax", "solve-eps-1e-2", "solve-eps-1e-2-mingain0"]
LABEL = {
    "f32": "`f32`",
    "bf16-params-f32": "`bf16-params-f32` (by hand)",
    "int8-absmax": "`int8` absmax",
    "solve-eps-1e-2": "A at ε\\*(10⁻²)",
    "solve-eps-1e-2-mingain0": "B (the eighth)",
}
SUMMARY = re.compile(r"compiled in ([0-9.]+) s \((\d+) lengths?, (\d+) kernels?\), computed in ([0-9.]+) s")
TIME = re.compile(r"^([0-9.]+) s (\d+) kB$", re.M)
CALLS = re.compile(r"(\d+) calls timed by phase and (\d+) not")
CALL = re.compile(r"-- a call: ([0-9.]+) ms without the timers \(median; ([0-9.]+)–([0-9.]+)\), ([0-9.]+) ms with them")
THP = re.compile(r"-- transparent huge pages: (\S+); the process's anonymous memory in huge pages after the calls: (\d+) kB")
CPU = re.compile(r"cpu (\d+)'s L1d and L2 set strides, (\d+) KiB and (\d+) KiB")
MOVED = re.compile(r"-- (\d+) of (\d+) launches had a large buffer at another address in another call")
LAUNCH = re.compile(r"^\s*(\d+)\s+k(\d+)\s(.*?)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+`([^`]*)`\s*$")
BUFFER = re.compile(r"^\s+(in\d+|out)\s+(\d+)\s+0x([0-9a-f]+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s*$")


def text(path):
    try:
        return open(path).read()
    except FileNotFoundError:
        return None


def med(xs):
    return statistics.median(xs) if xs else None


def num(x, d=1):
    return "—" if x is None else f"{x:,.{d}f}".replace(",", " ")


def timev(path):
    """/usr/bin/time -v's fields we read: wall s, peak kB, context switches, faults."""
    t = text(path) or ""
    def field(name):
        m = re.search(re.escape(name) + r": (\S+)", t)
        return m.group(1) if m else None
    w = field("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    wall = sum(float(x) * 60 ** i for i, x in enumerate(reversed(w.split(":")))) if w else None
    return {
        "ok": "Exit status: 0" in t,
        "wall_s": wall,
        "peak_kb": int(field("Maximum resident set size (kbytes)") or 0),
        "voluntary": int(field("Voluntary context switches") or 0),
        "involuntary": int(field("Involuntary context switches") or 0),
        "minor_faults": int(field("Minor (reclaiming a frame) page faults") or 0),
        "major_faults": int(field("Major (requiring I/O) page faults") or 0),
    }


def modes_process(path):
    t = text(path) or ""
    c = CALL.search(t)
    if not c:
        return None
    thp, cpu, moved, calls = THP.search(t), CPU.search(t), MOVED.search(t), CALLS.search(t)
    launches, cur = [], None
    for line in t.splitlines():
        m = LAUNCH.match(line)
        if m:
            cur = {"index": int(m.group(1)), "kernel": int(m.group(2)), "name": m.group(3).strip(), "median_us": float(m.group(4)),
                   "fastest_us": float(m.group(5)), "threads": int(m.group(6)), "nest": m.group(7), "buffers": []}
            launches.append(cur)
            continue
        m = BUFFER.match(line)
        if m and cur is not None:
            cur["buffers"].append({"slot": m.group(1), "bytes": int(m.group(2)), "address": int(m.group(3), 16),
                                   "mod4k": int(m.group(4)), "mod64k": int(m.group(5)), "modl1": int(m.group(6)), "modl2": int(m.group(7))})
    key = os.path.basename(path)[:-4]
    return {
        "key": key,
        "calls": int(calls.group(2)) if calls else None,
        "call_ms": float(c.group(1)), "call_min_ms": float(c.group(2)), "call_max_ms": float(c.group(3)), "timed_ms": float(c.group(4)),
        "thp": thp.group(1) if thp else None, "anon_huge_kb": int(thp.group(2)) if thp else None,
        "cpu": int(cpu.group(1)) if cpu else None, "l1_kib": int(cpu.group(2)) if cpu else None, "l2_kib": int(cpu.group(3)) if cpu else None,
        "moved": int(moved.group(1)) if moved else None, "launches_n": int(moved.group(2)) if moved else len(launches),
        "launches": launches,
        "time": timev(os.path.join(os.path.dirname(path), "log", key + ".time")),
    }


def modes_row(out, row):
    files = sorted(f for f in glob.glob(f"{out}/modes/{row}-*.txt") if re.fullmatch(re.escape(row) + r"-\d+\.txt", os.path.basename(f)))
    ps = [p for p in (modes_process(f) for f in files) if p]
    if ps:
        fastest = min(p["call_ms"] for p in ps)
        for p in ps:
            p["slow"] = p["call_ms"] > SLOW * fastest
    return ps


def serve_process(path):
    t = text(path)
    if t is None:
        return None
    s, tm = SUMMARY.search(t), TIME.search(t)
    if not (s and tm):
        return None
    lines = [l for l in t.splitlines() if re.fullmatch(r"[0-9 ]+", l.strip()) and " " in l.strip()]
    elapsed, compile_s, compute_s = float(tm.group(1)), float(s.group(1)), float(s.group(4))
    return {"elapsed_s": elapsed, "peak_kb": int(tm.group(2)), "compile_s": compile_s, "compute_s": compute_s,
            "rest_s": elapsed - compile_s - compute_s, "lengths": int(s.group(2)), "kernels": int(s.group(3)),
            "tokens": lines[0].split() if lines else None}


def d190_process(path):
    t = text(path)
    if t is None:
        return None
    tm = TIME.search(t)
    lines = [l for l in t.splitlines() if re.fullmatch(r"[0-9 ]+", l.strip()) and " " in l.strip()]
    return {"elapsed_s": float(tm.group(1)) if tm else None, "tokens": lines[0].split() if lines else None}


def slow(args):
    ps = modes_row(args.out, args.row)
    if not ps:
        print(f"sr5.py: no process of {args.row} in {args.out}/modes", file=sys.stderr)
        sys.exit(2)
    n = sum(p["slow"] for p in ps)
    print(f"{args.row}: {n} slow of {len(ps)}, calls {min(p['call_ms'] for p in ps):.1f}–{max(p['call_ms'] for p in ps):.1f} ms")
    sys.exit(0 if n else 1)


def report(args):
    out, md, js = args.out, [], {"serve": {}, "modes": {}}
    machine = (text(f"{out}/machine.txt") or "").strip()
    md.append("# SR5: the serving split and the two modes (D201)\n")
    md.append("Plan 0.17.2 §3.3, on D200's timers, `experiments/sr5.sh`; the machine:\n")
    md.append("```\n" + machine + "\n```\n")

    # --- SR5a ---------------------------------------------------------------
    rows = {}
    for name in NAMES:
        ps = [(r, serve_process(f"{out}/serve/serve-{name}-{r}.txt")) for r in range(1, 10)]
        ps = [(r, p) for r, p in ps if p]
        ds = [(r, d190_process(f"{args.d190}/serve-{name}-{r}.txt")) for r in range(1, 4)]
        ds = [(r, d) for r, d in ds if d and d["elapsed_s"] is not None]
        if ps:
            rows[name] = (ps, ds)
    if rows:
        md.append("## SR5a: `presk serve` for 32 tokens, split\n")
        md.append("D190's command, one thread on core 5, a process each; `elapsed` is `/usr/bin/time`'s, `compile` and `compute` the summary line's (D200: compiling is each new length's `prepare` and each call's lowering and Cranelift's compiling, computing the calls' time less it), `rest` the elapsed time less both (the weights read, the initializer, the process). In s.\n")
        md.append("| realization | round | elapsed | compile | compute | rest | lengths | kernels | peak GB | tokens = D190's |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, (ps, ds) in rows.items():
            ref = ds[0][1]["tokens"] if ds else None
            for r, p in ps:
                same = "yes" if ref and p["tokens"] == ref else ("**no**" if ref else "—")
                md.append(f"| {LABEL[name]} | {r} | {p['elapsed_s']:.2f} | {p['compile_s']:.3f} | {p['compute_s']:.2f} | {p['rest_s']:.2f} | {p['lengths']} | {p['kernels']} | {p['peak_kb'] / 1e6:.2f} | {same} |")
        md.append("\n**Across the processes** (the median, then the fastest), and SR5-0: the median elapsed time against D190's, within 2 %.\n")
        md.append("| realization | elapsed | compile | compute | rest | D190's elapsed | ÷ D190 | within 2 % | tokens |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        base = {}
        holds = True
        for name, (ps, ds) in rows.items():
            agg = {k: (med([p[k] for _, p in ps]), min(p[k] for _, p in ps)) for k in ("elapsed_s", "compile_s", "compute_s", "rest_s")}
            d = med([x["elapsed_s"] for _, x in ds]) if ds else None
            ratio = agg["elapsed_s"][0] / d if d else None
            within = ratio is not None and abs(ratio - 1) <= 0.02
            ref = ds[0][1]["tokens"] if ds else None
            toks = bool(ref) and all(p["tokens"] == ref for _, p in ps) and all(x["tokens"] == ref for _, x in ds)
            holds = holds and within and toks
            base[name] = agg
            js["serve"][name] = {"processes": [dict(p, round=r) for r, p in ps], "median": {k: v[0] for k, v in agg.items()},
                                 "fastest": {k: v[1] for k, v in agg.items()}, "d190_median_s": d, "ratio_d190": ratio,
                                 "within_2pc": within, "tokens_d190": toks}
            md.append(f"| {LABEL[name]} | **{agg['elapsed_s'][0]:.2f}** / {agg['elapsed_s'][1]:.2f} | {agg['compile_s'][0]:.3f} / {agg['compile_s'][1]:.3f} | "
                      f"**{agg['compute_s'][0]:.2f}** / {agg['compute_s'][1]:.2f} | {agg['rest_s'][0]:.2f} / {agg['rest_s'][1]:.2f} | {num(d, 2)} | "
                      f"{'—' if ratio is None else f'{ratio:.4f}'} | {'yes' if within else '**no**'} | {'D190’s' if toks else '**not D190’s**'} |")
        js["sr5_0_serve_holds"] = holds
        md.append(f"\n**SR5-0, serving: {'holds' if holds else 'fails'}** (every row's tokens D190's and its median elapsed time within 2 % of D190's).\n")
        if "f32" in base:
            f = base["f32"]
            md.append("**÷ `f32`**, on the medians:\n")
            md.append("| realization | elapsed | compile | compute | rest | compute's share of elapsed |")
            md.append("|---|---|---|---|---|---|")
            for name, agg in base.items():
                r = lambda k: agg[k][0] / f[k][0] if f[k][0] else None
                md.append(f"| {LABEL[name]} | {r('elapsed_s'):.3f} | {r('compile_s'):.3f} | {r('compute_s'):.3f} | {r('rest_s'):.3f} | {100 * agg['compute_s'][0] / agg['elapsed_s'][0]:.1f} % |")
                js["serve"][name]["over_f32"] = {k: r(k) for k in ("elapsed_s", "compile_s", "compute_s", "rest_s")}
            md.append("")

    ctl = [(r, d190_process(f"{out}/serve/control-f32-{r}.txt")) for r in range(1, 10)]
    ctl = [(r, c) for r, c in ctl if c and c["elapsed_s"] is not None]
    if ctl and "f32" in rows:
        ps, ds = rows["f32"]
        ref = ds[0][1]["tokens"] if ds else None
        e = [c["elapsed_s"] for _, c in ctl]
        now = med([p["elapsed_s"] for _, p in ps])
        then = med([x["elapsed_s"] for _, x in ds])
        md.append("**SR5-0's control:** `f32`'s serving row under D190's own binary (v0.17.1's code), each process in its own scope as here, run after this run's rows.\n")
        md.append("| binary | when | per process (s) | median | fastest | ÷ D190's median | tokens |")
        md.append("|---|---|---|---|---|---|---|")
        j = lambda xs: ", ".join(f"{x:.2f}" for x in xs)
        md.append(f"| D190's | D190 | {j(x['elapsed_s'] for _, x in ds)} | {then:.2f} | {min(x['elapsed_s'] for _, x in ds):.2f} | 1.0000 | — |")
        md.append(f"| D190's | this run, after it | {j(e)} | {med(e):.2f} | {min(e):.2f} | {med(e) / then:.4f} | {'D190’s' if ref and all(c['tokens'] == ref for _, c in ctl) else '**not D190’s**'} |")
        md.append(f"| this one | this run | {j(p['elapsed_s'] for _, p in ps)} | {now:.2f} | {min(p['elapsed_s'] for _, p in ps):.2f} | {now / then:.4f} | {'D190’s' if ref and all(p['tokens'] == ref for _, p in ps) else '**not D190’s**'} |")
        md.append(f"\nD190's binary now against this one now: {med(e) / now:.4f}.\n")
        js["control"] = {"processes": [dict(c, round=r) for r, c in ctl], "median_s": med(e), "fastest_s": min(e), "over_d190": med(e) / then, "over_this": med(e) / now}

    # --- SR5b ---------------------------------------------------------------
    spread = {}
    for row in ("f32", "solved-0.11.0", "f32-huge", "f32-4k", "f32-huge-fixed", "f32-4k-fixed"):
        ps = modes_row(out, row)
        if not ps:
            continue
        calls = [p["call_ms"] for p in ps]
        spread[row] = (ps, max(calls) / min(calls))
        nslow = sum(p["slow"] for p in ps)
        js["modes"][row] = {"processes": ps, "slow": nslow}
        md.append(f"## SR5b: the decoder block, {row}, {len(ps)} processes\n")
        md.append(f"`presk bench --host-profile --launches` on `Llama.apply` at the 7B width (D124's program, `coverage/llama.psk` as tagged `v0.12.0`), B = 1, T = 384, ten threads on the ten X925 cores, under `presk plan`'s plan with the frozen profile. *Call*: the median of the {ps[0]['calls']} calls without the timers, then its range and the median with the timers, in ms. A process is slow when its call exceeds the fastest process's by more than 20 %. *cpu*: the core the main thread last ran on. *Huge*: AnonHugePages after the calls.\n")
        md.append("| process | call | range | with timers | mode | cpu | huge kB | moved launches | involuntary switches | minor faults | wall s | peak GB |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for p in ps:
            t = p["time"]
            md.append(f"| {p['key']} | {p['call_ms']:.1f} | {p['call_min_ms']:.1f}–{p['call_max_ms']:.1f} | {p['timed_ms']:.1f} | {'**slow**' if p['slow'] else 'fast'} | {p['cpu']} | {p['anon_huge_kb']} | {p['moved']} of {p['launches_n']} | {t['involuntary']} | {t['minor_faults']} | {num(t['wall_s'])} | {t['peak_kb'] / 1e6:.2f} |")
        calls = [p["call_ms"] for p in ps]
        md.append(f"\n**{nslow} of {len(ps)} processes slow**; the calls from {min(calls):.1f} to {max(calls):.1f} ms, median {med(calls):.1f}, max ÷ min {max(calls) / min(calls):.4f}.\n")
        fast = [p for p in ps if not p["slow"]]
        slow_ = [p for p in ps if p["slow"]]
        n = len(ps[0]["launches"])
        same = all(len(p["launches"]) == n and all(a["kernel"] == b["kernel"] for a, b in zip(p["launches"], ps[0]["launches"])) for p in ps)
        md.append(f"Every process launches the same {n} kernels in the same order: {'yes' if same else '**no**'}.\n")
        groups = [("fast", fast), ("slow", slow_)] if slow_ else [("all", ps)]
        # Each launch: the median over a group's processes of the launch's median.
        md.append("**The launches**, each the median over the group's processes of its median µs" + (", slow against fast, largest difference first" if slow_ else ", largest first") + ":\n")
        hdr = " | ".join(f"{g} µs" for g, _ in groups)
        md.append(f"| launch | kernel | nest | threads | {hdr} |" + (" slow ÷ fast | slow − fast ms |" if slow_ else " share of the call |"))
        md.append("|---|---|---|---|" + "---|" * len(groups) + ("---|---|" if slow_ else "---|"))
        rows_ = []
        for i in range(n):
            l0 = ps[0]["launches"][i]
            gm = [med([p["launches"][i]["median_us"] for p in g]) for _, g in groups]
            rows_.append((i, l0, gm))
        if slow_:
            rows_.sort(key=lambda x: -(x[2][1] - x[2][0]))
        else:
            rows_.sort(key=lambda x: -x[2][0])
        launch_js = []
        for i, l0, gm in rows_:
            launch_js.append({"index": l0["index"], "kernel": l0["kernel"], "name": l0["name"], "nest": l0["nest"], "group_median_us": dict(zip([g for g, _ in groups], gm))})
        for i, l0, gm in rows_[:12]:
            cells = " | ".join(num(x, 1) for x in gm)
            extra = f" {gm[1] / gm[0]:.3f} | {(gm[1] - gm[0]) / 1e3:.1f} |" if slow_ and gm[0] else f" {100 * gm[0] / 1e3 / med(calls):.1f} % |"
            md.append(f"| {l0['index']} | k{l0['kernel']} {l0['name']} | `{l0['nest']}` | {l0['threads']} | {cells} |{extra}")
        js["modes"][row]["launches"] = launch_js
        # The large buffers of the launches that take the most: their residues per group.
        md.append("\n**The large buffers** of the six launches above, their virtual addresses' residues in each process (%64K, %L1, %L2 in bytes; %4K is " +
                  ", ".join(sorted({str(b['mod4k']) for p in ps for l in p['launches'] for b in l['buffers']})) + " everywhere listed):\n")
        md.append("| launch | buffer | bytes | group | %64K | %L1 | %L2 |")
        md.append("|---|---|---|---|---|---|---|")
        for i, l0, gm in rows_[:6]:
            for j, b in enumerate(l0["buffers"]):
                for g, gp in groups:
                    vals = [p["launches"][i]["buffers"][j] for p in gp if len(p["launches"][i]["buffers"]) > j]
                    show = lambda k: ", ".join(str(v[k]) for v in vals)
                    md.append(f"| {l0['index']} | {b['slot']} | {b['bytes']} | {g} | {show('mod64k')} | {show('modl1')} | {show('modl2')} |")
        md.append("")

    if "f32" in spread:
        # The launch whose time varies most between f32's processes, in every row.
        ps0 = spread["f32"][0]
        n = len(ps0[0]["launches"])
        var = max(range(n), key=lambda i: max(p["launches"][i]["median_us"] for p in ps0) - min(p["launches"][i]["median_us"] for p in ps0))
        l0 = ps0[0]["launches"][var]
        md.append(f"## SR5b: the launch that varies, and placement\n")
        md.append(f"The launch whose time varies most between f32's processes is launch {l0['index']}, k{l0['kernel']} `{l0['name']}` at `{l0['nest']}`. For each row, in ms: the call, that launch, and the call less it, each from the fastest process to the slowest, with max ÷ min. *Huge* is the processes' memory in huge pages after the calls.\n")
        md.append("| row | processes | call | max ÷ min | the launch | max ÷ min | the call less it | huge pages, MB | slow by the 20 % rule |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        js["placement"] = {"launch": {"index": l0["index"], "kernel": l0["kernel"], "name": l0["name"], "nest": l0["nest"]}}
        for row, (ps, sp) in spread.items():
            c = [p["call_ms"] for p in ps]
            k = [p["launches"][var]["median_us"] / 1e3 for p in ps]
            r = [p["call_ms"] - p["launches"][var]["median_us"] / 1e3 for p in ps]
            h = [p["anon_huge_kb"] / 1024 for p in ps if p["anon_huge_kb"] is not None]
            md.append(f"| {row} | {len(ps)} | {min(c):.1f}–{max(c):.1f} | {sp:.4f} | {min(k):.1f}–{max(k):.1f} | {max(k) / min(k):.4f} | {min(r):.1f}–{max(r):.1f} | {min(h):.0f}–{max(h):.0f} | {sum(p['slow'] for p in ps)} |")
            js["placement"][row] = {"call_spread": sp, "launch_ms": k, "rest_ms": r, "huge_mb": h}
        if "f32-huge" in spread and "f32-4k" in spread:
            hs, ks = spread["f32-huge"][1], spread["f32-4k"][1]
            hslow = sum(p["slow"] for p in spread["f32-huge"][0])
            verdict = "named as the cause" if hs < 1.05 and ks > 1.05 else "ruled out" if hslow else "left open"
            js["placement"]["verdict"] = verdict
            if "f32-huge-fixed" in spread:
                fs = spread["f32-huge-fixed"][1]
                v2 = "the buffers' placement within their pages named as the cause" if fs < 1.05 else "left open, between the physical placement above 2 MiB and a cause other than placement"
                js["placement"]["verdict_fixed"] = v2
                md.append(f"\n**With the addresses fixed** (randomization off): under huge pages the calls' max ÷ min is {fs:.4f}, under 4 KiB pages {spread.get('f32-4k-fixed', (None, float('nan')))[1]:.4f}: {v2} (the rule fixed before the runs: named if under 1.05 under huge pages).\n")
            md.append(f"\n**Placement: {verdict}.** Under huge pages the calls' max ÷ min is {hs:.4f} ({hslow} slow), under 4 KiB pages alternated with them {ks:.4f}. The rule fixed before the runs: named if under 1.05 with huge pages while 4 KiB pages spread; ruled out if a process under huge pages is slow by the 20 % rule; left open between.\n")

    open(f"{out}/report.md", "w").write("\n".join(md) + "\n")
    json.dump(js, open(f"{out}/report.json", "w"), indent=1)
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slow")
    s.add_argument("out")
    s.add_argument("row")
    r = sub.add_parser("report")
    r.add_argument("out")
    r.add_argument("--d190", required=True)
    a = ap.parse_args()
    try:
        (slow if a.cmd == "slow" else report)(a)
    except SystemExit:
        raise
    except Exception as e:
        print(f"sr5.py: {e!r}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
