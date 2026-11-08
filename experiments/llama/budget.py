#!/usr/bin/env python3
"""budget.py — the arithmetic of plan 0.17.1 (presk D185 onward) that
`experiments/llama/llama3_budget.sh` hands to Python: Llama 3.2 1B's margins
on the held-out sample, ε*(δ), every realization's gap and flips against the
exact logits, the weights' bytes, the perplexities and the statements.

  eps DIR [--deltas …] [--alpha A]       the exact margins of DIR/exact/*.tsv
                                         (`margins.py exact`'s), ε*(δ) corrected
                                         (Clopper–Pearson, one-sided) and raw:
                                         DIR/margins.json and margins.md
  field BIN EXACT_DIR OUT_DIR            a tool's float32 logits of the held-out
                                         windows (window after window, position
                                         after position) as `margins.py native`
                                         rows, one OUT_DIR/NNNN.tsv a window
  native JSON EXACT.hilo OUT.tsv         the same for one window of `presk run
                                         --json` (NumPy: margins.py's pure
                                         Python takes minutes a window here)
  nll-field BIN QUALITY.txt OUT.json     a tool's float32 logits of the quality
                                         windows' contexts: −log softmax at each
                                         target, in binary64
  nll-presk OUT.json JSON…               `presk run --entry nll`'s results, in
                                         order
  ppl-check IDS CHUNKS [--logits BIN]    llama-perplexity's figure from our
                                         reading of its logits on its windows
  bytes NAME CONFIG [REALIZATION|GGUF]   the weights' bytes at rest
  report DIR                             every table and statement: report.md,
                                         report.json, figure.json
  time DIR --names …                     RW9: DIR/time/'s processes, the median
                                         of their medians and the fastest (D126),
                                         the traffic ratio beside: time.md,
                                         time.json

The margins' arithmetic is margins.py's (D118), imported from it; a gap is
`margins.py native`'s, the logit against its exact value held as two doubles.
"""

import argparse
import json
import math
import os
import sys
from decimal import Decimal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import margins as M  # noqa: E402

DELTAS = (1e-1, 3e-2, 1e-2)


# ---------------------------------------------------------------------------
# The exact margins and ε*(δ).


def exact_rows(d):
    """Per held-out position, in order: (top, second, margin, max |logit|)."""
    files = sorted(f for f in os.listdir(os.path.join(d, "exact")) if f.endswith(".tsv"))
    rows = []
    for f in files:
        for line in open(os.path.join(d, "exact", f)):
            i, j, m, y = line.split("\t")
            rows.append((int(i), int(j), Decimal(m), float(y)))
    return files, rows


def eps(argv):
    ap = argparse.ArgumentParser(prog="budget.py eps")
    ap.add_argument("dir")
    ap.add_argument("--deltas", default=" ".join(map(str, DELTAS)))
    ap.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    files, rows = exact_rows(a.dir)
    n = len(rows)
    sm = sorted(r[2] for r in rows)
    out = {"windows": len(files), "positions": n, "alpha": a.alpha, "ties": sum(1 for m in sm if m == 0),
           "ymax": max(r[3] for r in rows), "quantiles": {}, "eps": {}}
    for q in (0.001, 0.003, 0.01, 0.03, 0.1, 0.25, 0.5):
        out["quantiles"][str(q)] = float(sm[min(n - 1, int(q * n))])
    for d in (float(x) for x in a.deltas.split()):
        k_raw = math.floor(d * n)
        k_cp = M.k_clopper_pearson(n, d, a.alpha)
        out["eps"][str(d)] = {"delta": d, "allowed": math.floor(d * n), "k_raw": k_raw, "eps_raw": float(M.eps_star(sm, k_raw)),
                              "k_cp": k_cp, "eps_cp": float(M.eps_star(sm, k_cp))}
    json.dump(out, open(os.path.join(a.dir, "margins.json"), "w"), indent=1)
    L = [f"# The exact margins: {len(files)} windows, {n} positions\n",
         f"Largest |logit| {out['ymax']:.6g}; margins exactly 0: {out['ties']}; smallest {float(sm[0]):.4e}, median {float(sm[n // 2]):.4e}, largest {float(sm[-1]):.4e}.\n",
         "| δ | decisions allowed | k (Clopper–Pearson, α = %g) | ε*(δ) | k raw | ε raw |" % a.alpha, "|---|---|---|---|---|---|"]
    for e in out["eps"].values():
        L.append(f"| {e['delta']:g} | {e['allowed']} | {e['k_cp']} | {e['eps_cp']:.4e} | {e['k_raw']} | {e['eps_raw']:.4e} |")
    L += ["", "| quantile | margin |", "|---|---|"] + [f"| {q} | {v:.4e} |" for q, v in out["quantiles"].items()]
    open(os.path.join(a.dir, "margins.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------------------
# A tool's logits on the held-out windows, as margins.py's native rows.


def rows_tsv(z, hilo_path, out):
    """`margins.py native`'s rows for one window, in NumPy: per position the
    largest |ŷ − y| (ŷ − hi, then − lo, as margins.py computes it), the
    greatest index among the maxima of ŷ (the reference's `argmax`), and the
    sum of the gaps."""
    import numpy as np
    hl = np.fromfile(hilo_path, dtype="<f8").reshape(z.shape[0], z.shape[1], 2)
    e = np.abs((z - hl[..., 0]) - hl[..., 1])
    worst = e.max(-1)
    arg = z.shape[1] - 1 - np.argmax(z[:, ::-1], -1)
    total = e.sum(-1)
    with open(out + ".part", "w") as f:
        for w, g, t in zip(worst.tolist(), arg.tolist(), total.tolist()):
            f.write(f"{w!r}\t{g}\t{t!r}\n")
    os.replace(out + ".part", out)


def field(argv):
    import numpy as np
    ap = argparse.ArgumentParser(prog="budget.py field")
    ap.add_argument("bin")
    ap.add_argument("exact_dir")
    ap.add_argument("out_dir")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(f[:-5] for f in os.listdir(a.exact_dir) if f.endswith(".hilo"))
    z = np.memmap(a.bin, dtype="<f4", mode="r")
    at = 0
    for f in files:
        positions = sum(1 for _ in open(os.path.join(a.exact_dir, f + ".tsv")))
        n = os.path.getsize(os.path.join(a.exact_dir, f + ".hilo")) // 16
        if at + n > z.size:
            sys.exit(f"{a.bin}: ends before window {f}")
        rows_tsv(np.asarray(z[at:at + n], dtype=np.float64).reshape(positions, n // positions), os.path.join(a.exact_dir, f + ".hilo"), os.path.join(a.out_dir, f + ".tsv"))
        at += n
    if at != z.size:
        sys.exit(f"{a.bin}: {z.size - at} logits more than the {len(files)} windows")


def native(argv):
    """`presk run --json`'s logits of one window as its rows."""
    import numpy as np
    src, hilo, out = argv
    data = json.load(open(src))["result"]["data"]
    z = np.array([[float(x) for x in row] for window in data for row in window], dtype=np.float64)
    rows_tsv(z, hilo, out)


# ---------------------------------------------------------------------------
# Perplexity.


def nll_field(argv):
    import numpy as np
    ap = argparse.ArgumentParser(prog="budget.py nll-field")
    ap.add_argument("bin")
    ap.add_argument("quality")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    windows = [[int(t) for t in line.split()] for line in open(a.quality) if line.strip()]
    T = len(windows[0]) - 1
    z = np.memmap(a.bin, dtype="<f4", mode="r")
    V = z.size // (len(windows) * T)
    assert V * len(windows) * T == z.size, (z.size, len(windows), T)
    z = z.reshape(len(windows), T, V)
    rows = []
    for k, w in enumerate(windows):
        x = z[k].astype(np.float64)
        m = x.max(-1, keepdims=True)
        lse = (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[:, 0]
        rows.append((lse - x[np.arange(T), np.array(w[1:])]).tolist())
    save_nll(a.out, rows)


def nll_presk(argv):
    out, srcs = argv[0], argv[1:]
    rows = []
    for s in srcs:
        for window in json.load(open(s))["result"]["data"]:
            rows.append([float(x) for x in window])
    save_nll(out, rows)


def ppl_check(argv):
    """llama-perplexity's figure computed again from our reading of its
    logits: its windows (n_ctx tokens of the whole text, BOS first, the first
    token of each window replaced by BOS), its scoring (positions n_ctx/2 to
    n_ctx − 2, each predicting the next token), our log-softmax in binary64.
    Writes CHUNKS.txt for llamacpp_logits, or with BIN reads its logits."""
    import numpy as np
    ap = argparse.ArgumentParser(prog="budget.py ppl-check")
    ap.add_argument("ids", help="llama-tokenize --ids's output: the whole text's ids, BOS first")
    ap.add_argument("chunks", help="the windows, one a line (written, or read with --logits)")
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--logits")
    ap.add_argument("--hf-ids", help="our tokenizer's ids of the same text (texts.py's rule), BOS first, to compare")
    a = ap.parse_args(argv)
    ids = json.loads(open(a.ids).read().strip())
    if a.hf_ids:
        hf = [int(t) for t in open(a.hf_ids).read().split()]
        k = min(len(hf), len(ids))
        same = sum(1 for x, y in zip(hf[:k], ids[:k]) if x == y)
        print(f"tokenizers: llama.cpp {len(ids)} ids, ours {len(hf)}; the first {k} the same at {same}"
              + ("" if same == k and len(hf) == len(ids) else " — they differ"))
    bos = ids[0]
    win = []
    for i in range(a.n):
        w = ids[i * a.ctx:(i + 1) * a.ctx]
        w = [bos] + w[1:]
        win.append(w)
    if not a.logits:
        with open(a.chunks, "w") as f:
            for w in win:
                f.write(" ".join(map(str, w)) + "\n")
        return
    z = np.memmap(a.logits, dtype="<f4", mode="r")
    V = z.size // (a.n * a.ctx)
    z = z.reshape(a.n, a.ctx, V)
    first = a.ctx // 2
    tot, cnt = 0.0, 0
    for i in range(a.n):
        x = z[i, first:a.ctx - 1].astype(np.float64)
        m = x.max(-1, keepdims=True)
        lse = (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[:, 0]
        # the targets are the text's own tokens, not the BOS put in their place
        tgt = np.array(ids[i * a.ctx + first + 1:(i + 1) * a.ctx])
        tot += float((lse - x[np.arange(len(tgt)), tgt]).sum())
        cnt += len(tgt)
    print(f"our reading of its logits: {a.n} windows of {a.ctx}, {cnt} targets, PPL = {math.exp(tot / cnt):.4f}")


def save_nll(out, rows):
    flat = [x for r in rows for x in r]
    mean = math.fsum(flat) / len(flat)
    json.dump({"windows": len(rows), "targets": len(flat), "mean_nll": mean, "perplexity": math.exp(mean), "nll": rows}, open(out, "w"))
    print(f"{out}: {len(rows)} windows, {len(flat)} targets, mean nll {mean:.6f}, perplexity {math.exp(mean):.4f}")


# ---------------------------------------------------------------------------
# The weights' bytes at rest.

FLOAT = {"f64": 8, "f32": 4, "bf16": 2}


def leaves(cfg):
    """`coverage/llama3.psk`'s parameter leaves at the checkpoint's
    configuration: path, shape."""
    D, FF, L, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
    KV = cfg["num_key_value_heads"] * (D // cfg["num_attention_heads"])
    return [("tok", (V, D)), ("blocks.attn.wq", (L, D, D)), ("blocks.attn.wk", (L, D, KV)), ("blocks.attn.wv", (L, D, KV)),
            ("blocks.attn.wo", (L, D, D)), ("blocks.ff.gate", (L, D, FF)), ("blocks.ff.up", (L, D, FF)),
            ("blocks.ff.down", (L, FF, D)), ("blocks.n1.γ", (L, D)), ("blocks.n2.γ", (L, D)), ("norm.γ", (D,))]


def presk_bytes(cfg, spec):
    """A presk realization's parameters at rest: a float leaf at its type, an
    integer one packed with its `f32` scales (`QTensor::bytes`)."""
    r = {"default": spec} if spec in FLOAT else json.load(open(spec.lstrip("@")))
    total, per = 0, {}
    for path, shape in leaves(cfg):
        n = math.prod(shape)
        q = r.get("quant", {}).get(path)
        if q:
            over = shape[q.get("over", -2)]
            b = (n * q["bits"] + 7) // 8 + 4 * (n // over)
            per[path] = f"int{q['bits']}/{q.get('scheme', 'absmax')}"
        else:
            t = r.get("params", {}).get(path, r.get("default", "f64"))
            b = n * FLOAT[t]
            per[path] = t
        total += b
    return total, per


def gguf_bytes(path, gguf_py):
    sys.path.insert(0, gguf_py)
    from gguf import GGUFReader
    rd = GGUFReader(path)
    per = {}
    total = 0
    for t in rd.tensors:
        total += int(t.n_bytes)
        per[t.name] = t.tensor_type.name
    return total, per


def bytes_(argv):
    ap = argparse.ArgumentParser(prog="budget.py bytes")
    ap.add_argument("name")
    ap.add_argument("config")
    ap.add_argument("what", help="f64|f32|bf16|@REALIZATION.json|FILE.gguf|hf:float32|hf:bfloat16")
    ap.add_argument("--gguf-py", default=".llamacpp/src/gguf-py")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    if a.what.endswith(".gguf"):
        total, per = gguf_bytes(a.what, a.gguf_py)
    elif a.what.startswith("hf:"):
        b = {"float32": 4, "bfloat16": 2}[a.what[3:]]
        total = sum(math.prod(s) for _, s in leaves(cfg)) * b
        per = {"all": a.what[3:]}
    else:
        total, per = presk_bytes(cfg, a.what)
    print(json.dumps({"name": a.name, "bytes": total, "types": per}))


# ---------------------------------------------------------------------------
# The report.


def report(argv):
    ap = argparse.ArgumentParser(prog="budget.py report")
    ap.add_argument("dir")
    ap.add_argument("--names", required=True, help="the realizations, in the table's order")
    a = ap.parse_args(argv)
    d = a.dir
    files, ex = exact_rows(d)
    n = len(ex)
    margin = [r[2] for r in ex]
    mj = json.load(open(os.path.join(d, "margins.json")))
    eps_of = {k: v["eps_cp"] for k, v in mj["eps"].items()}
    by_bytes = {}
    if os.path.exists(os.path.join(d, "bytes.jsonl")):
        for line in open(os.path.join(d, "bytes.jsonl")):
            r = json.loads(line)
            by_bytes[r["name"]] = r
    budgets = json.load(open(os.path.join(d, "budgets.json"))) if os.path.exists(os.path.join(d, "budgets.json")) else {}
    nll = {}
    for name in a.names.split():
        p = os.path.join(d, "nll", name + ".json")
        if os.path.exists(p):
            j = json.load(open(p))
            nll[name] = (j["mean_nll"], j["perplexity"], j["targets"])
    rows = []
    for name in a.names.split():
        nd = os.path.join(d, "native", name)
        if not all(os.path.exists(os.path.join(nd, f)) for f in files):
            continue
        worst, arg, tot, rel_w = [], [], 0.0, []
        k = 0
        for f in files:
            wmax, ymax = 0.0, 0.0
            for line in open(os.path.join(nd, f)):
                e, g, s = line.split("\t")
                worst.append(float(e)); arg.append(int(g)); tot += float(s)
                wmax = max(wmax, float(e)); ymax = max(ymax, ex[k][3])
                k += 1
            rel_w.append(wmax / ymax)
        assert len(worst) == n, name
        g = max(worst)
        two_g = 2 * Decimal(repr(g))
        flips = [p for p in range(n) if arg[p] != ex[p][0]]
        row = {"name": name, "gap": g, "relative": max(rel_w), "relative_global": g / mj["ymax"], "mean_gap": tot / (n * 128256),
               "flips": len(flips), "bound": sum(1 for m in margin if m < two_g), "flip_positions": flips,
               # the δ at which the realization's own held-out gap certifies
               # it, #{m < 2g}/N: what its guarantee is worth, beside its flips
               "certified_delta": sum(1 for m in margin if m < two_g) / n,
               "outside": sum(1 for p in flips if margin[p] > 2 * Decimal(repr(worst[p])))}
        if name in by_bytes:
            row["bytes"] = by_bytes[name]["bytes"]
        if name in budgets:
            b = budgets[name]["eps"]
            row["budget"] = b
            row["bound_at_budget"] = sum(1 for m in margin if m < 2 * Decimal(repr(b)))
        if name in nll:
            row["mean_nll"], row["perplexity"], row["targets"] = nll[name]
        rows.append(row)
    ref = next((r for r in rows if r["name"] == "f64"), None)
    for r in rows:
        if ref and "mean_nll" in r and "mean_nll" in ref:
            r["dlogppl"] = r["mean_nll"] - ref["mean_nll"]
            r["rw8_gap"] = abs(r["dlogppl"]) <= 2 * r["gap"]
            if "budget" in r:
                r["rw8_budget"] = abs(r["dlogppl"]) <= 2 * r["budget"]
    # The statements.
    st = {}
    for dk, e in eps_of.items():
        delta = float(dk)
        allowed = math.floor(delta * n)
        solved = next((r for r in rows if budgets.get(r["name"], {}).get("delta") == delta and budgets[r["name"]].get("statement")), None)
        if solved is None:
            continue
        rw6 = solved["flips"] <= allowed and solved["gap"] <= solved["budget"]
        cands = [r for r in rows if r["flips"] <= allowed and "bytes" in r]
        best = min(cands, key=lambda r: r["bytes"]) if cands else None
        st[dk] = {"delta": delta, "eps": e, "allowed": allowed, "solved": solved["name"], "rw6": rw6,
                  "rw7": best is not None and best["name"] == solved["name"], "rw7_fewest": best["name"] if best else None,
                  "candidates": sorted((r["name"] for r in cands), key=lambda x: by_bytes.get(x, {}).get("bytes", 0))}
    json.dump({"positions": n, "eps": mj["eps"], "rows": rows, "statements": st}, open(os.path.join(d, "report.json"), "w"), indent=1)
    fig = [{k: r.get(k) for k in ("name", "bytes", "flips", "gap", "relative", "perplexity", "budget", "bound", "bound_at_budget", "certified_delta")} for r in rows]
    json.dump(fig, open(os.path.join(d, "figure.json"), "w"), indent=1)
    L = [f"# Llama 3.2 1B against the exact logits: {len(files)} windows, {n} positions\n",
         "| realization | weights (bytes) | gap | relative | flips | #{m < 2g} | δ certified, #{m < 2g}/N | budget | #{m < 2ε} | perplexity | Δ log ppl | ≤ 2g | ≤ 2ε |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda k, fmt: (fmt % r[k]) if r.get(k) is not None else "—"  # noqa: E731
        yn = lambda k: ("yes" if r[k] else "NO") if k in r else "—"  # noqa: E731
        L.append(f"| {r['name']} | {f('bytes', '%.4e')} | {r['gap']:.3e} | {r['relative']:.2e} | {r['flips']} | {r['bound']} | {r['certified_delta']:.3g} | {f('budget', '%.3e')} | {f('bound_at_budget', '%d')} | {f('perplexity', '%.4f')} | {f('dlogppl', '%+.2e')} | {yn('rw8_gap')} | {yn('rw8_budget')} |")
    L += ["", "| δ | ε*(δ) | allowed | solved | RW6 | RW7 | fewest bytes within δ·N |", "|---|---|---|---|---|---|---|"]
    for s in st.values():
        L.append(f"| {s['delta']:g} | {s['eps']:.4e} | {s['allowed']} | {s['solved']} | {'holds' if s['rw6'] else 'fails'} | {'holds' if s['rw7'] else 'fails'} | {s['rw7_fewest']} |")
    open(os.path.join(d, "report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ---------------------------------------------------------------------------
# RW9's times.

UNIFORM_ROWS = ("native f64", "native f64 +lanes", "native f32", "native f32 +lanes", "native bf16")
WIDTH = {"f64": 8, "f32": 4, "bf16": 2}


def bench_rows(path):
    """`presk bench`'s table: label → (median ms, best ms)."""
    rows = {}
    for line in open(path).read().splitlines()[1:]:
        parts = line.rsplit(None, 3)
        if len(parts) == 4 and parts[3].endswith("x"):
            rows[parts[0].strip()] = (float(parts[1]), float(parts[2]))
    return rows


def across(values):
    """D126: the median of the processes' medians, the fastest of them, the spread."""
    v = sorted(values)
    return {"processes": values, "median": v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2,
            "fastest": v[0], "spread": v[-1] / v[0]}


def solve_cost(path):
    """A solve's `cost` line: (its traffic, the all-f64 traffic), in bytes."""
    for line in open(path):
        if line.strip().startswith("cost"):
            w = line.split()
            return float(w[1]), float(w[3])
    raise ValueError(path)


def time_(argv):
    ap = argparse.ArgumentParser(prog="budget.py time")
    ap.add_argument("dir")
    ap.add_argument("--names", required=True, help="the realizations, in the table's order")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args(argv)
    d, T = a.dir, os.path.join(a.dir, "time")
    names = a.names.split()
    rounds = range(1, a.rounds + 1)
    by_bytes = {json.loads(line)["name"]: json.loads(line) for line in open(os.path.join(d, "bytes.jsonl"))}
    # The traffic in the solver's unit (D157): the parameters at rest, each
    # stored value at one sample's elements. The solves print theirs; for the
    # others the values' bytes in f64 are the all-f64 traffic less f64's
    # parameters, taken at the realization's default width.
    solved = sorted(f for f in os.listdir(os.path.join(d, "solve")) if f.endswith(".txt"))
    _, all_f64 = solve_cost(os.path.join(d, "solve", solved[0]))
    values_f64 = all_f64 - by_bytes["f64"]["bytes"]
    out = {"all_f64_traffic": all_f64, "values_f64_bytes": values_f64, "forward": {}, "serve": {}, "uniform_everywhere": {}}
    for name in names:
        runs = [bench_rows(os.path.join(T, f"forward-{name}-{r}.txt")) for r in rounds]
        own = [k for k in runs[0] if k not in UNIFORM_ROWS]
        if name in WIDTH:
            label = f"native {name}"
            own = [label] + ([label + " +lanes"] if label + " +lanes" in runs[0] else [])
        row = {"rows": {}}
        for k in own:
            row["rows"][k] = across([r[k][0] for r in runs])
            row["rows"][k]["best_call"] = min(r[k][1] for r in runs)
        if name.startswith("solve-"):
            t, _ = solve_cost(os.path.join(d, "solve", name[len("solve-"):] + ".txt"))
            row["traffic"], row["traffic_from"] = t, "the solve"
        else:
            default = "f32" if name not in WIDTH else name
            row["traffic"] = by_bytes[name]["bytes"] + values_f64 * WIDTH[default] / 8
            row["traffic_from"] = f"the parameters at rest, the values at {default}"
        row["traffic_ratio"] = all_f64 / row["traffic"]
        row["bytes"] = by_bytes[name]["bytes"]
        out["forward"][name] = row
    # The uniform rows, timed again in every name's process: D126's modes.
    for k in UNIFORM_ROWS:
        v = [bench_rows(os.path.join(T, f"forward-{n}-{r}.txt"))[k][0] for n in names for r in rounds]
        out["uniform_everywhere"][k] = across(v)
    for name in names:
        ps = [os.path.join(T, f"serve-{name}-{r}.txt") for r in rounds]
        if not all(os.path.exists(p) and os.path.getsize(p) for p in ps):
            continue
        secs, kb, toks = [], [], set()
        for p in ps:
            lines = open(p).read().splitlines()
            s, _, k, _ = lines[-1].split()
            secs.append(float(s)); kb.append(int(k)); toks.add(lines[1].strip())
        assert len(toks) == 1, name
        out["serve"][name] = dict(across(secs), peak_kb=max(kb), tokens=toks.pop())
    json.dump(out, open(os.path.join(d, "time.json"), "w"), indent=1)
    f64 = out["forward"]["f64"]["rows"]["native f64"]["median"]
    L = [f"# RW9: Llama 3.2 1B's forward pass at B = 1, T = 16, one thread on one X925 core, {a.rounds} processes a row\n",
         "Each process the median of five calls after a warmup (`presk bench --warmup 1 --repeat 5`); **median** the median of the processes', **fastest** the smallest (D126). Traffic in the solver's unit (D157), its ratio against all-`f64`'s.\n",
         "| realization | row | weights (bytes) | traffic ratio | per process (ms) | median ms | fastest ms | max ÷ min | ÷ f64 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, row in out["forward"].items():
        for k, r in row["rows"].items():
            L.append(f"| {name} | `{k[len('native '):]}` | {row['bytes']} | {row['traffic_ratio']:.3f} | "
                     + " ".join(f"{v:.1f}" for v in r["processes"])
                     + f" | **{r['median']:.1f}** | {r['fastest']:.1f} | {r['spread']:.4f} | {f64 / r['median']:.3f} |")
    n = len(names) * a.rounds
    L += ["", f"The uniform rows in every realization's process ({n} processes each):", "",
          "| row | median ms | fastest ms | slowest ms | max ÷ min |", "|---|---|---|---|---|"]
    for k, r in out["uniform_everywhere"].items():
        L.append(f"| `{k[len('native '):]}` | {r['median']:.1f} | {r['fastest']:.1f} | {max(r['processes']):.1f} | {r['spread']:.4f} |")
    L += ["", f"`presk serve`, 32 tokens from prompt 1 at temperature 0, one thread on one X925 core, {a.rounds} processes:", "",
          "| realization | per process (s) | median s | fastest s | max ÷ min | ÷ f32 | peak (kB) | tokens |", "|---|---|---|---|---|---|---|---|"]
    f32s = out["serve"].get("f32", {}).get("median")
    for name, r in out["serve"].items():
        L.append(f"| {name} | " + " ".join(f"{v:.2f}" for v in r["processes"])
                 + f" | **{r['median']:.2f}** | {r['fastest']:.2f} | {r['spread']:.4f} | {(r['median'] / f32s) if f32s else float('nan'):.3f} | {r['peak_kb']} | {r['tokens']} |")
    open(os.path.join(d, "time.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    cmd, rest = sys.argv[1], sys.argv[2:]
    {"eps": eps, "field": field, "native": native, "ppl-check": ppl_check, "nll-field": nll_field, "nll-presk": nll_presk, "bytes": bytes_, "report": report,
     "time": time_}[cmd](rest)


if __name__ == "__main__":
    main()
