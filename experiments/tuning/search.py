#!/usr/bin/env python3
"""Two tuning tools' searches on the compiler's own space (D154's tier B):
not the tools, but their search strategies reimplemented as a driver over
presk's realizations, each trial `presk run --backend native` of the
realization on the export's conformance inputs and the criterion of
criterion.py against the exact reference.

  search.py --space SPACE.txt --export EXPORT --budget EPS --strategy S
            [--params] --out DIR

SPACE.txt is `presk solve --sites 100` at a budget every site fits under
(each stored value and parameter it offers, with its elements). The space
is those stored values, and with --params the parameters too, each `f64`,
`f32` or `bf16`. The searches minimise the solver's own cost, the traffic
(`cost_at` in presk-precision's solve.rs, without a profile: Σ elements ×
bytes over the decisions), where the tools minimise a measured runtime;
the two agree on what is narrower. S is
- `precimonious`: the delta-debugging search of Rubio-González et al.
  (SC'13), in the order and with the stopping rule of its implementation
  (`scripts/dd2.py` of corvette-berkeley/precimonious, 5ac56e1): level by
  level of the type vectors (f64 → f32, then f32 → bf16), every variable
  lowered one level at once; if that fails, a change set (the variables kept
  at the higher type) split in `div` parts, each part and (above two parts)
  each complement kept high with the rest lowered, the passing one of least
  cost taken and searched again from two parts, `div` doubled when none
  passes, the set kept high when `div` reaches its size; a variable left at
  the higher type is fixed there for the next levels. A configuration is
  taken only if it passes and costs less than the original;
- `craft`: the binary search of CRAFT's `bin_simple` strategy
  (crafthpc/craft, 03f2514, scripts/craft_strategies.rb): the whole program
  lowered; a failed node split into its children (the Presk functions, then
  their sites; the parameters by their tree), a child with one child
  replaced by it, more than nine children merged into two halves, a failed
  merged set split into its members; each configuration lowers its node
  alone, the rest at the level's start; the level's result is the union of
  the passing nodes (CRAFT's `build_final_config`), tested, and if the
  union fails, the passing configuration of least cost (what CRAFT's
  variable mode returns, `build_fastest_config`). Two levels, f64 → f32,
  then f32 → bf16 over what the first lowered.
Every run the search makes is a trial, as the tools count them; a
realization run twice is measured once (the native backend is
deterministic), and both counts are reported. DIR/trials.tsv holds every
trial, DIR/final.json the realization found, DIR/summary.json the figures.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import criterion  # noqa: E402
import retype  # noqa: E402

BYTES = {"f64": 8, "f32": 4, "bf16": 2}
LEVELS = ["bf16", "f32", "f64"]


def parse_space(path):
    """(stored values, parameters) as lists of (name, elements), in the
    solver's listing."""
    sites, params, part = [], [], None
    for line in open(path):
        if line.startswith("site "):
            part = sites
            continue
        if line.startswith("parameter "):
            part = params
            continue
        m = re.match(r"^(\S+)\s+(f64|f32|bf16)\s+(\d+)\s", line)
        if m and part is not None:
            part.append((m.group(1), int(m.group(3))))
    return sites, params


class Runner:
    def __init__(self, a, sites, params):
        self.a = a
        self.elems = dict(sites + params)
        self.params = {n for n, _ in params}
        self.trials = 0
        self.cache = {}
        os.makedirs(a.out, exist_ok=True)
        self.log = open(os.path.join(a.out, "trials.tsv"), "w")
        self.log.write("trial\tstatus\tgap\trelative\tcost\tf64\tf32\tbf16\trealization\n")
        self.exact = criterion.reals(os.path.join(a.export, "conformance", "exact.psk.d"))
        self.mag = max(abs(y) for y in self.exact)

    def realization(self, cfg):
        r = {"default": "f64", "storage": {}, "order": {"kind": "sequential"}, "params": {}}
        for v, t in sorted(cfg.items()):
            if t != "f64":
                r["params" if v in self.params else "storage"][v] = t
        return r

    def cost(self, cfg):
        return sum(self.elems[v] * BYTES[cfg.get(v, "f64")] for v in self.elems)

    def run(self, cfg):
        """One trial: the realization natively on the conformance inputs."""
        self.trials += 1
        key = tuple(sorted((v, t) for v, t in cfg.items() if t != "f64"))
        if key not in self.cache:
            path = os.path.join(self.a.out, "r.json")
            json.dump(self.realization(cfg), open(path, "w"), ensure_ascii=False)
            cmd = [self.a.presk, "run", self.a.file, "--entry", "logits", *self.a.dims, "--weights", self.a.weights,
                   "--backend", "native", "--threads", "1", "--realize", "@" + path, "--seed", "0", "--json", "--digits", "17"]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
            data = json.loads(out)["result"]["data"]
            y = [float(x) for w in data for r in w for x in r]
            worst, _ = criterion.gap(y, self.exact)
            self.cache[key] = worst
        worst = self.cache[key]
        ok = worst <= self.a.budget
        counts = {t: sum(1 for v in self.elems if cfg.get(v, "f64") == t) for t in BYTES}
        self.log.write(f"{self.trials}\t{'pass' if ok else 'fail'}\t{worst:.6e}\t{worst / self.mag:.6e}\t{self.cost(cfg):.6e}\t"
                       f"{counts['f64']}\t{counts['f32']}\t{counts['bf16']}\t{json.dumps(dict(key), ensure_ascii=False)}\n")
        self.log.flush()
        return ok, worst


# ---- Precimonious ---------------------------------------------------------------


def precimonious(run, cost, order):
    types = {v: list(LEVELS) for v in order}
    cfg = {v: "f64" for v in order}

    def high(vs):
        for v in vs:
            if types[v]:
                cfg[v] = types[v][-1]

    def second(vs):
        for v in vs:
            if len(types[v]) > 1:
                cfg[v] = types[v][-2]

    run(dict(cfg))
    original = cost(cfg)

    def good(c):
        ok, _ = run(dict(c))
        return ok and cost(c) < original

    def dd(change, div):
        size = int(math.ceil(len(change) / div))
        parts = [change[i:i + size] for i in range(0, len(change), size)]
        best, best_score = None, None
        for part in parts:
            inv = [v for v in change if v not in part]
            for chosen, allowed in ((part, True), (inv, div > 2)):
                if not chosen or not allowed:
                    continue
                second(change)
                high(chosen)
                if good(cfg):
                    s = cost(cfg)
                    if best_score is None or s < best_score:
                        best, best_score = chosen, s
        if best is not None:
            second(change)
            if len(best) > 1:
                dd(best, 2)
            else:
                high(best)
            return
        if div >= len(change):
            high(change)
            return
        dd(change, 2 * div)

    while any(len(t) > 1 for t in types.values()):
        second(order)
        ok, _ = run(dict(cfg))
        if not ok or cost(cfg) > original:
            dd(list(order), 2)
        for v in order:
            if types[v] and cfg[v] == types[v][-1]:
                types[v] = []
        for v in order:
            if types[v]:
                types[v].pop()
    run(dict(cfg))
    return cfg


# ---- CRAFT's binary search ------------------------------------------------------


def tree_of(order, params):
    """CRAFT's structure: the program, then the stored values by Presk
    function and the parameters by the first part of their path, then
    each variable."""
    def group(names, key):
        out = {}
        for n in names:
            out.setdefault(key(n), []).append(n)
        return [(g, [(n, None) for n in ns]) for g, ns in out.items()]
    stored = [n for n in order if n not in params]
    ps = [n for n in order if n in params]
    kids = []
    if stored:
        kids.append(("stored values", group(stored, lambda n: n.split("/")[0])))
    if ps:
        kids.append(("parameters", group(ps, lambda n: ".".join(n.split(".")[:2]) if n.startswith("blocks.") else n)))
    return ("program", kids)


def leaves(node):
    name, kids = node
    return [name] if kids is None else [x for k in kids for x in leaves(k)]


def craft_binary(run, cost, order, params):
    cfg = {v: "f64" for v in order}
    run(dict(cfg))  # the baseline
    for frm, to in (("f64", "f32"), ("f32", "bf16")):
        cand = [v for v in order if cfg[v] == frm]
        if not cand:
            continue
        start = dict(cfg)
        root = tree_of(cand, params)
        while root[1] is not None and len(root[1]) == 1:
            root = root[1][0]  # a single part of the program is the program
        queue = [[root]]  # a configuration: the nodes it lowers
        passed = []
        while queue:
            nodes = queue.pop(0)
            c = dict(start)
            for n in nodes:
                for v in leaves(n):
                    c[v] = to
            ok, _ = run(c)
            if ok:
                passed.append((nodes, cost(c)))
                continue
            if len(nodes) == 1 and nodes[0][1] is None:
                continue  # a single variable: no further split
            if len(nodes) > 1:
                children = [[n] for n in nodes]
            else:
                children = []
                for k in nodes[0][1]:
                    while k[1] is not None and len(k[1]) == 1:
                        k = k[1][0]
                    children.append([k])
            if len(children) > 9:
                pivot = len(children) // 2
                children = [[n for c in children[:pivot] for n in c], [n for c in children[pivot:] for n in c]]
            queue.extend(children)
        union = dict(start)
        for nodes, _ in passed:
            for n in nodes:
                for v in leaves(n):
                    union[v] = to
        ok, _ = run(dict(union))
        if ok:
            cfg = union
        elif passed:
            nodes, _ = min(passed, key=lambda p: p[1])
            cfg = dict(start)
            for n in nodes:
                for v in leaves(n):
                    cfg[v] = to
            print(f"craft: the union of the passing nodes fails at {frm} → {to}; the cheapest passing configuration kept")
        else:
            cfg = start
    return cfg


def program_order(export, sites, params):
    """The variables in the order the C program declares them (its
    functions callees first, as an LLVM pass would list them), the
    parameters first, in the weights' order."""
    p = retype.Program(export)
    seen = []
    for h, ss in p.value_sites().items():
        for s in ss:
            if s not in seen:
                seen.append(s)
    names = {n for n, _ in sites}
    order = [n for n, _ in params] + [s for s in seen if s in names]
    order += [n for n, _ in sites if n not in order]
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--export", required=True)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--strategy", choices=["precimonious", "craft"], required=True)
    ap.add_argument("--params", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--presk", default="./target/release/presk")
    ap.add_argument("--file", default="examples/experiments/margins.psk")
    ap.add_argument("--weights", default="out/curve_train_f64/final.psk.d")
    ap.add_argument("--dims", default="--const D=64 --const H=2 --const L=2 --dim B=2 --dim T=256")
    a = ap.parse_args()
    a.dims = a.dims.split()
    sites, params = parse_space(a.space)
    # the cost counts every decision the solver prices; the search changes
    # the stored values, and the parameters only with --params
    r = Runner(a, sites, params)
    order = program_order(a.export, sites, params if a.params else [])
    pset = {n for n, _ in params} if a.params else set()
    if a.strategy == "precimonious":
        cfg = precimonious(r.run, r.cost, order)
    else:
        cfg = craft_binary(r.run, r.cost, order, pset)
    worst = r.cache[tuple(sorted((v, t) for v, t in cfg.items() if t != "f64"))]
    real = r.realization(cfg)
    json.dump(real, open(os.path.join(a.out, "final.json"), "w"), indent=2, ensure_ascii=False)
    base = r.cost({})
    summary = {
        "strategy": a.strategy, "budget": a.budget, "params_in_space": a.params,
        "trials": r.trials, "distinct": len(r.cache), "gap": worst, "relative": worst / r.mag, "pass": worst <= a.budget,
        "cost": r.cost(cfg), "cost_f64": base, "ratio": base / r.cost(cfg),
        "f64": sum(1 for v in r.elems if cfg.get(v, "f64") == "f64"),
        "f32": sum(1 for v in r.elems if cfg.get(v, "f64") == "f32"),
        "bf16": sum(1 for v in r.elems if cfg.get(v, "f64") == "bf16"),
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
