#!/usr/bin/env python3
"""The report of experiments/tuning/tuning.sh (D154): its tables, from the
run's outputs, in Markdown on stdout.

  report.py OUT --budgets "4.67e-4 1e-6"

The standard library only.
"""

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import criterion  # noqa: E402
import retype  # noqa: E402
import search  # noqa: E402


def sci(x, d=2):
    return "—" if x is None else f"{x:.{d}e}"


def read(path, default=""):
    try:
        return open(path).read()
    except OSError:
        return default


def margins_rows(path):
    """label → the row's cells, from margins.py report's table."""
    rows, head = {}, None
    for line in read(path).splitlines():
        if line.startswith("| realization"):
            head = [c.strip() for c in line.strip("|").split("|")]
            continue
        if head and line.startswith("| ") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            rows[cells[0]] = dict(zip(head, cells))
        elif head and not line.startswith("|"):
            head = None if rows else head
    return rows


def conformance_gap(presk, realize, export, weights):
    out = subprocess.run([presk, "run", "examples/experiments/margins.psk", "--entry", "logits", "--const", "D=64", "--const", "H=2",
                          "--const", "L=2", "--dim", "B=2", "--dim", "T=256", "--weights", weights, "--backend", "native",
                          "--threads", "1", "--realize", realize, "--seed", "0", "--json", "--digits", "17"],
                         capture_output=True, text=True, check=True).stdout
    y = [float(x) for w in json.loads(out)["result"]["data"] for r in w for x in r]
    return criterion.gap(y, criterion.reals(os.path.join(export, "conformance", "exact.psk.d")))[0]


def counts(r, names, params):
    c = {"f64": 0, "f32": 0, "bf16": 0}
    for n in names:
        t = (r.get("params", {}) if n in params else r.get("storage", {})).get(n, r.get("default", "f64"))
        c[t] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--presk", default="./target/release/presk")
    ap.add_argument("--weights", default="out/curve_train_f64/final.psk.d")
    a = ap.parse_args()
    O = a.out
    budgets = a.budgets.split()
    X = os.path.join(O, "c-f64")
    prog = retype.Program(X)
    sites, params = search.parse_space(os.path.join(O, "solver", "space.txt"))
    elems = dict(sites + params)
    pset = {n for n, _ in params}
    snames = [n for n, _ in sites]
    pnames = [n for n, _ in params]

    def traffic(r):
        return sum(e * search.BYTES[(r.get("params", {}) if n in pset else r.get("storage", {})).get(n, r.get("default", "f64"))]
                   for n, e in elems.items())

    base = traffic({"default": "f64"})
    bounds = dict(line.split()[:2] for line in read(os.path.join(O, "bounds.txt")).splitlines() if len(line.split()) >= 2)
    times = {}
    for line in read(os.path.join(O, "c-times.txt")).splitlines():
        p = line.split()
        if len(p) == 2:
            times[p[0]] = float(p[1])
    t64 = next((v for k, v in times.items() if k.endswith("c-f64")), None)
    out = []
    w = out.append

    w("# A precision-tuning tool on the exported C (D154)\n")
    w(f"The machine: {read(os.path.join(O, 'machine.txt')).splitlines()[0] if read(os.path.join(O, 'machine.txt')) else '—'}.")
    tools = []
    for l in read(os.path.join(O, "tools.txt")).splitlines():
        f = l.split()
        tools.append(f"Ruby {f[2]}" if f[0] == "ruby" else f"{f[0]} `{f[1][:7]}` ({f[2]}, {f[3]})")
    w("The tools: " + "; ".join(tools) + ".\n")
    w(f"The program: `logits` of examples/experiments/margins.psk at D = 64, B = 2, T = 256, exported in `f64`: "
      f"{len(prog.host_order)} host functions, {len(prog.kern_order)} kernels; its floating-point variables, "
      f"{sum(1 for v in prog.variables if v['file'] == 'model.c')} buffers of `model.c` and "
      f"{sum(1 for v in prog.variables if v['file'] == 'kernels.c')} locals of the kernels. "
      f"{(lambda t: t[:1].upper() + t[1:])(read(os.path.join(O, 'margins', 'control.txt')).strip())}.\n")

    # --- the solver ------------------------------------------------------------------
    w("## The solver\n")
    w(f"| budget | stored values away from f64 | parameters away from f64 | f64 / f32 / bf16 (of {len(snames) + len(pnames)}) | traffic vs all-f64 | estimate | measured (envelope's windows) | widening rounds | gap on the conformance inputs |")
    w("|---|---|---|---|---|---|---|---|---|")
    for b in budgets:
        t = read(os.path.join(O, "solver", f"solve-{b}.txt"))
        r = json.load(open(os.path.join(O, "solver", f"solved-{b}.json")))
        c = counts(r, snames + pnames, pset)
        sv = re.search(r"sites\s+(\d+) narrowable, (\d+) assigned", t)
        pv = re.search(r"parameters\s+(\d+) offered, (\d+) stored away", t)
        est = re.search(r"estimate\s+(\S+)", t)
        mea = re.search(r"measured\s+(\S+) against the exact reference, after (\d+) widening", t)
        g = conformance_gap(a.presk, "@" + os.path.join(O, "solver", f"solved-{b}.json"), X, a.weights)
        w(f"| {b} | {sv.group(2)} of {sv.group(1)} | {pv.group(2)} of {pv.group(1)} | {c['f64']} / {c['f32']} / {c['bf16']} | "
          f"{base / traffic(r):.2f}× | {est.group(1)} | {mea.group(1)} | {mea.group(2)} | {sci(g)} |")
    w("")

    # --- tier A --------------------------------------------------------------------
    w("## Tier A: CRAFT's variable mode on the C program\n")
    w("| budget | strategy | configurations tested | passed / failed | wall (note) | buffers in float (of 39) | locals in float (of 24) | gap, conformance inputs | against the f64 run | ms per call (note) | ÷ the f64 export |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    finals = {}
    for b in budgets:
        for s in ("bin_simple", "ddebug"):
            d = os.path.join(O, "craft", f"{s}-{b}")
            co = read(os.path.join(d, "craft.out"))
            tested = re.search(r"Total configs tested:\s+(\d+)", co)
            passed = re.search(r"Total passed:\s+(\d+)", co)
            failed = re.search(r"Total failed:\s+(\d+)", co)
            fin = json.load(open(os.path.join(d, "craft_final.json")))
            low = {x["handle"] for x in fin["actions"] if x.get("to_type") == "float"}
            finals[(s, b)] = low
            nb = sum(1 for v in prog.variables if v["file"] == "model.c" and v["handle"] in low)
            nk = sum(1 for v in prog.variables if v["file"] == "kernels.c" and v["handle"] in low)
            last = read(os.path.join(d, "trials.tsv")).strip().splitlines()[-1].split("\t")
            ms = times.get(os.path.join(d, "final"))
            w(f"| {b} | `{s}` | {tested.group(1) if tested else '—'} | {passed.group(1) if passed else '—'} / {failed.group(1) if failed else '—'} | "
              f"{read(os.path.join(d, 'wall.txt')).split()[1] if read(os.path.join(d, 'wall.txt')) else '—'} s | {nb} | {nk} | "
              f"{sci(float(last[2]))} | {sci(float(last[3]))} | {ms if ms else '—'} | {f'{t64 / ms:.2f}×' if ms and t64 else '—'} |")
    w("")
    for (s, b), low in finals.items():
        keep = [v["handle"] for v in prog.variables if v["handle"] not in low]
        w(f"- `{s}` at {b} keeps in `double`: {', '.join('`' + h + '`' for h in keep) if keep else 'nothing'}.")
    w("")
    ex = read(os.path.join(O, "adapt", "example.txt"))
    rec = re.findall(r"^\s+((?:Replace|DO NOT replace)\s+variable\s+\S+)", ex, re.M)
    w("ADAPT: its own example (`sum2pi_x`) " + (f"builds and runs, and recommends: {'; '.join(' '.join(r.split()) for r in rec)}" if rec else "does not run")
      + ". The export compiled as C++, as ADAPT needs: "
      + "; ".join(l.strip("= ").replace("g++ -std=c++17 -x c++ -fsyntax-only ", "`") .replace(": ", "`: ", 1) for l in read(os.path.join(O, "adapt", "export-as-cxx.txt")).splitlines() if l.startswith("=="))
      + ".\n")

    # --- tier B --------------------------------------------------------------------
    w("## Tier B: the tools' searches on presk's space\n")
    w(f"| budget | search | space | trials | distinct realizations | f64 / f32 / bf16 (of {len(snames) + len(pnames)}) | traffic vs all-f64 | gap, conformance inputs |")
    w("|---|---|---|---|---|---|---|---|")
    for b in budgets:
        for s in ("precimonious", "craft"):
            for p in ("sites", "params"):
                d = os.path.join(O, "search", f"{s}-{p}-{b}")
                sm = json.load(open(os.path.join(d, "summary.json")))
                w(f"| {b} | {s} | {f'{len(snames)} stored values and {len(pnames)} parameters' if p == 'params' else f'{len(snames)} stored values'} | {sm['trials']} | {sm['distinct']} | "
                  f"{sm['f64']} / {sm['f32']} / {sm['bf16']} | {sm['ratio']:.2f}× | {sci(sm['gap'])} |")
    w("")
    for b in budgets:
        sol = json.load(open(os.path.join(O, "solver", f"solved-{b}.json")))
        for s in ("precimonious", "craft"):
            f = json.load(open(os.path.join(O, "search", f"{s}-params-{b}", "final.json")))
            diff = []
            for n in snames + pnames:
                tf = (f.get("params", {}) if n in pset else f.get("storage", {})).get(n, "f64")
                ts = (sol.get("params", {}) if n in pset else sol.get("storage", {})).get(n, "f64")
                if tf != ts:
                    diff.append(f"`{n}` {ts} → {tf}")
            w(f"- {s} at {b}, against the solver: {', '.join(diff) if diff else 'the same realization'}.")
    w("")

    # --- the 461 windows ---------------------------------------------------------------
    w("## Judged by the compiler on the 461 held-out windows\n")
    rows = margins_rows(os.path.join(O, "margins.md"))
    w("| realization | gap on the logits | flips | bound #{m < 2ε} | cert. 1e-2 | cert. 1e-3 | cert. 1e-4 | accuracy | traffic vs all-f64 | bound `cond` |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for line in read(os.path.join(O, "rows.tsv")).splitlines():
        name, label = line.split("\t")
        r = rows.get(label, {})
        rp = os.path.join(O, "margins", "realizations", name + ".json")
        if os.path.exists(rp):
            tr = f"{base / traffic(json.load(open(rp))):.2f}×"
        elif name in ("f64", "f32", "bf16"):
            tr = f"{base / traffic({'default': name, 'params': {q: name for q in pnames}}):.2f}×"
        else:
            tr = "—"
        w(f"| {label} | {r.get('gap on the logits', '—')} | {r.get('flips', '—')} | {r.get('bound #{m < 2ε}', '—')} | {r.get('cert. 1e-02', '—')} | "
          f"{r.get('cert. 1e-03', '—')} | {r.get('cert. 1e-04', '—')} | {r.get('accuracy', '—')} | {tr} | {bounds.get(name, '—')} |")
    w("")

    # --- the spaces ----------------------------------------------------------------------
    vs = prog.value_sites()
    held = {}
    for h, ss in vs.items():
        for s_ in ss:
            held.setdefault(s_, []).append(h)
    multi = {s_: hs for s_, hs in held.items() if len(hs) > 1}
    shared = {h: ss for h, ss in vs.items() if len(ss) > 1}
    w("## The two spaces\n")
    w(f"- presk's: {len(snames)} stored values and {len(pnames)} parameters, each f64, f32 or bf16.")
    w(f"- the C program's: {sum(1 for v in prog.variables if v['file'] == 'model.c')} buffers and "
      f"{sum(1 for v in prog.variables if v['file'] == 'kernels.c')} kernel locals, each double or float; "
      f"the buffers hold {len(held)} of the {len(snames)} sites; "
      + "; ".join(f"`{s_}` is held by {len(hs)} declarations" for s_, hs in sorted(multi.items()))
      + "; " + "; ".join(f"`{h}` holds {', '.join(ss)}" for h, ss in shared.items()) + ".")
    w(f"- the parameters (`theta_*`, {len(pnames)}) are inputs of the C program, read as f64; the locals have no site.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
