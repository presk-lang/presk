#!/usr/bin/env python3
"""PyTorch compiled and in mixed precision, beside eager (D153, experiments/pytorch_modes.sh).

    python pytorch_modes.py forward --export DIR --weights W.psk.d --inputs IDS.psk.d --threads N --mode MODE [--out Y.npy]
    python pytorch_modes.py step --export DIR --threads N --mode MODE [--steps 8] [--no-dropout]
    python pytorch_modes.py profile --export DIR --threads N --mode MODE [--no-dropout] --out TABLE.txt
    python pytorch_modes.py explain --export DIR --inputs IDS.psk.d --threads N --mode MODE
    python pytorch_modes.py train --export DIR --data TRAIN.psk.d --steps 600 --seed S --threads N --mode MODE --out W.pt
    python pytorch_modes.py evaluate --export DIR --data VAL.psk.d --threads N NAME=W.pt|W.psk.d ...
    python pytorch_modes.py gap --export DIR --weights W.psk.d --data VAL.psk.d --exact MARGINS/exact
                                --threads N --mode MODE --name NAME --out OUT
    python pytorch_modes.py report OUT

The course GPT as `presk export --target pytorch --mode readable --train`
writes it (DIR: `_05_gpt.py`, `pskd.py`, `train.py`), run by PyTorch in four
modes, nothing of the export changed:

    eager         the model as exported, float32
    compile       `torch.compile`, its defaults: the backend inductor, the
                  default mode, on the CPU
    amp           `torch.autocast("cpu", dtype=torch.bfloat16)` around the
                  forward pass and the loss, the parameters, the gradient
                  and AdamW in float32: the recipe as PyTorch documents it
    compile+amp   both, the autocast region inside the compiled function

`forward` times the forward pass in eval mode under `torch.no_grad()`, with
presk's checkpoint and the token indices of the C export's conformance
inputs, as `pytorch_baseline.py forward` does: the first call (which
compiles, under `compile`) timed apart, then REPEAT calls, the median and
the best. `step` times train.py's step: its loop, schedule, AdamW constants
and `clip_grad_norm_` (imported from the exported train.py), the model
initialized by its seed, the batches of its data file in its order; the
objective calls the model directly, where train.py's `loss` calls it through
`torch.func.functional_call` on a GPT(T) it builds at every step (the
exported train.py itself is the control of the step, run by the shell
driver). The first step (which compiles) is timed apart; the median of the
rest is printed in train.py's form. `explain` runs `torch._dynamo.explain`
on the forward pass and on the objective: graphs and graph breaks.
`--no-dropout` calls the model with `training=False` in the objective, the
rest of the step unchanged: a training step that draws no dropout masks,
as the compiler's `loss_grad` + `adamw.step` draws none. `profile` runs
three steps and gives the third's ops by self CPU time (torch.profiler):
where a step's time goes, not a timing of record.

`train` trains the exported model from its seed for STEPS steps in a mode,
train.py's loop on TRAIN at its batch, and writes the final state dict;
`evaluate` gives each model's validation loss and next-character accuracy
in float64 on VAL (the weights cast up), a presk checkpoint (.psk.d) read
as `pytorch_baseline.py` reads one. `gap` writes, for one mode, the rows
`experiments/margins.py report` reads, as `pytorch_gap.py logits` does for
eager. `report` combines the rounds of the shell driver.
"""
import argparse
import array
import glob
import os
import re
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pytorch_baseline import LAYOUT, load  # noqa: E402

MODES = ("eager", "compile", "amp", "compile+amp")


def compiled(mode):
    return mode.startswith("compile")


def amp(mode):
    return mode.endswith("amp")


def import_export(export):
    sys.path.insert(0, os.path.abspath(export))
    import _05_gpt
    import pskd
    import train

    return _05_gpt, pskd, train


def forward_of(model, mode):
    def f(ids):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp(mode)):
            return model(ids, None)

    return torch.compile(f) if compiled(mode) else f


def objective_of(model, mode, dropout=True):
    """The exported `loss`, the model called directly: the cross-entropy of
    the logits, per position, then the mean; under autocast in the amp modes;
    its dropout off with `dropout=False`."""

    def objective(x, y):
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp(mode)):
            logits = model(x, None, training=dropout)
            return F.cross_entropy(logits.flatten(0, -2), y.flatten(), reduction="none").view(y.shape).mean()

    return torch.compile(objective) if compiled(mode) else objective


def header(kind, mode):
    return f"pytorch_modes {kind}: torch {torch.__version__}, {torch.get_num_threads()} threads, float32, mode {mode}"


def dynamo_line():
    """What dynamo did in this process: frames, graphs, graph breaks."""
    from torch._dynamo.utils import counters

    st, fr = counters["stats"], counters["frames"]
    breaks = sum(counters["graph_break"].values())
    reasons = "; ".join(f"{k} ({v})" for k, v in counters["graph_break"].items())
    return f"dynamo: frames {fr.get('total', 0)} ({fr.get('ok', 0)} ok), unique graphs {st.get('unique_graphs', 0)}, graph breaks {breaks}" + (f": {reasons}" if reasons else "")


def compile_times():
    from torch._dynamo.utils import compile_times as ct

    return ct(repr="str", aggregate=True)


def forward(a):
    torch.set_num_threads(a.threads)
    model, ids = load(a.export, a.weights, a.inputs, torch.float32)
    f = forward_of(model, a.mode)
    times = []
    with torch.no_grad():
        t0 = time.perf_counter()
        f(ids)
        first = 1e3 * (time.perf_counter() - t0)
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            f(ids)
            times.append(1e3 * (time.perf_counter() - t0))
        y = f(ids)
    if a.out:
        np.save(a.out, y.double().numpy())
    print(f"{header('forward', a.mode)}, logits {str(y.dtype).replace('torch.', '')}, {a.repeat} calls after the first, the first {first:.1f} ms, median {statistics.median(times):.3f} ms, best {min(times):.3f} ms")
    if compiled(a.mode):
        print(dynamo_line())
        print(compile_times())


def loader_of(pskd, train, data, batch=None):
    dataset = pskd.PskdDataset(data, torch.float32)
    g = torch.Generator().manual_seed(train.SHUFFLE_SEED) if train.SHUFFLE_SEED is not None else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch or train.BATCH, shuffle=train.SHUFFLE_SEED is not None, drop_last=True, generator=g)


def loop(a, model, objective, loader, train, total, epochs, log_every, out=sys.stdout):
    """train.py's loop: the step timed from the learning rate to opt.step()."""
    opt = torch.optim.AdamW(model.parameters(), lr=train.LR, betas=train.BETAS, eps=train.EPS, weight_decay=train.WEIGHT_DECAY)
    t, times, value = 0, [], None
    for epoch in range(1, epochs + 1):
        for x, y in loader:
            t += 1
            t0 = time.perf_counter()
            for group in opt.param_groups:
                group["lr"] = train.LR * train.schedule(t, total)
            opt.zero_grad()
            value = objective(x, y)
            value.backward()
            if train.GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train.GRAD_CLIP)
            opt.step()
            times.append(time.perf_counter() - t0)
            if log_every and t % log_every == 0:
                print(f"step {t}/{total}  epoch {epoch}  loss {value.item():.6f}  lr {opt.param_groups[0]['lr']:.3e}", file=out, flush=True)
            if t >= total:
                return times, value
    return times, value


def step(a):
    _, pskd, train = import_export(a.export)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    loader = loader_of(pskd, train, os.path.join(a.export, "data.psk.d"))
    model = train.make_model()
    objective = objective_of(model, a.mode, not a.no_dropout)
    times, value = loop(a, model, objective, loader, train, a.steps, 1, 1)
    ms = [1e3 * s for s in times]
    print(f"{header('step', a.mode)}{', no dropout' if a.no_dropout else ''}, the first step {ms[0]:.1f} ms, {statistics.median(ms[1:]):.1f} ms a step, median of {len(ms) - 1} (the first excluded), best {min(ms[1:]):.1f} ms, final loss {value.item():.6f}")
    if compiled(a.mode):
        print(dynamo_line())
        print(compile_times())


def profile(a):
    _, pskd, train = import_export(a.export)
    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    loader = loader_of(pskd, train, os.path.join(a.export, "data.psk.d"))
    model = train.make_model()
    objective = objective_of(model, a.mode, not a.no_dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=train.LR, betas=train.BETAS, eps=train.EPS, weight_decay=train.WEIGHT_DECAY)
    batches = iter(loader)

    def one():
        x, y = next(batches)
        opt.zero_grad()
        objective(x, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train.GRAD_CLIP)
        opt.step()

    one()
    one()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        one()
    with open(a.out, "w") as f:
        f.write(f"{header('profile', a.mode)}{', no dropout' if a.no_dropout else ''}: the third step, ops by self CPU time\n")
        f.write(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25))


def explain(a):
    torch.set_num_threads(a.threads)
    import torch._dynamo as dynamo

    model, ids = load(a.export, None, a.inputs, torch.float32)
    _, pskd, train = import_export(a.export)
    mode = a.mode.replace("compile", "eager")  # explain traces; the mode's autocast stays
    with torch.no_grad():
        e = dynamo.explain(forward_of(model, mode))(ids)
    print(f"explain forward, mode {a.mode}: graphs {e.graph_count}, graph breaks {e.graph_break_count}, ops {e.op_count}" + "".join(f"\n  break: {r.reason}" for r in e.break_reasons))
    dynamo.reset()
    x, y = next(iter(loader_of(pskd, train, os.path.join(a.export, "data.psk.d"))))
    model.train()
    e = dynamo.explain(objective_of(model, mode))(x, y)
    print(f"explain step's objective, mode {a.mode}: graphs {e.graph_count}, graph breaks {e.graph_break_count}, ops {e.op_count}" + "".join(f"\n  break: {r.reason}" for r in e.break_reasons))


def train_run(a):
    _, pskd, train = import_export(a.export)
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    loader = loader_of(pskd, train, a.data, a.batch)
    model = train.make_model()
    objective = objective_of(model, a.mode)
    print(f"{header('train', a.mode)}, seed {a.seed}, {len(loader.dataset)} samples of {a.data}, batch {loader.batch_size}, {a.steps} steps", flush=True)
    t0 = time.time()
    times, value = loop(a, model, objective, loader, train, a.steps, a.epochs, a.log_every)
    torch.save(model.state_dict(), a.out)
    print(f"pytorch_modes train: {len(times)} steps in {time.time() - t0:.1f} s, {1e3 * statistics.median(times[1:]):.1f} ms a step, final loss {value.item():.6f}, weights in {a.out}")


def evaluate(a):
    """Validation loss and next-character accuracy in float64."""
    torch.set_num_threads(a.threads)
    _, pskd, train = import_export(a.export)
    data = pskd.PskdDataset(a.data, torch.float64)
    print("| model | validation loss (float64) | next-character accuracy | positions |")
    print("|---|---|---|---|")
    for spec in a.models:
        name, _, path = spec.partition("=")
        model = train.make_model().double()
        if path.endswith(".psk.d"):
            state = pskd.read_params(path, LAYOUT, model)
        else:
            state = {k: v.double() for k, v in torch.load(path).items()}
        model.load_state_dict(state)
        model.eval()
        ce, hits, n = 0.0, 0, 0
        with torch.no_grad():
            for s in range(0, len(data), 32):
                x = torch.stack([data[i][0] for i in range(s, min(s + 32, len(data)))])
                y = torch.stack([data[i][1] for i in range(s, min(s + 32, len(data)))])
                z = model(x, None)
                ce += float(F.cross_entropy(z.flatten(0, -2), y.flatten(), reduction="sum"))
                hits += int((z.argmax(-1) == y).sum())
                n += y.numel()
        print(f"| {name} | {ce / n:.7f} | {hits / n:.6f} | {n} |")


def gap(a):
    """pytorch_gap.py logits, in a mode."""
    torch.set_num_threads(a.threads)
    _, pskd, train = import_export(a.export)
    data = pskd.PskdDataset(a.data)
    x0, _ = data[0]
    T = x0.shape[0]
    from _05_gpt import GPT

    model = GPT(T)
    model.load_state_dict(pskd.read_params(a.weights, LAYOUT, model))
    model.eval()
    f = forward_of(model, a.mode)
    os.makedirs(os.path.join(a.out, "native", a.name), exist_ok=True)
    os.makedirs(os.path.join(a.out, "decisions"), exist_ok=True)
    chunks = sorted(c[:-5] for c in os.listdir(a.exact) if c.endswith(".hilo"))
    print(f"{header('gap', a.mode)}, {len(chunks)} chunks of {a.exact}")
    decisions, total_s, dtype = [], 0.0, None
    V = 92
    for c in chunks:
        hilo = array.array("d")
        with open(os.path.join(a.exact, c + ".hilo"), "rb") as fh:
            hilo.frombytes(fh.read())
        rows = len(hilo) // (2 * V)
        start, count = int(c), rows // T
        ids = torch.stack([data[i][0] for i in range(start, start + count)])
        t0 = time.perf_counter()
        with torch.no_grad():
            y = f(ids)
        total_s += time.perf_counter() - t0
        dtype = y.dtype
        y = y.double().numpy().reshape(rows, V)
        ex = np.frombuffer(hilo, dtype=np.float64).reshape(rows, V, 2)
        g = np.abs((y - ex[:, :, 0]) - ex[:, :, 1])
        top = (V - 1) - np.argmax(y[:, ::-1], axis=1)
        part = os.path.join(a.out, "native", a.name, c + ".tsv.part")
        with open(part, "w") as out:
            for r in range(rows):
                out.write(f"{float(g[r].max())!r}\t{int(top[r])}\t{float(g[r].sum())!r}\n")
        os.replace(part, part[: -len(".part")])
        decisions.extend(int(t) for t in top)
        print(f"  {c}: {count} windows, {rows} positions, largest gap {g.max():.3e}")
    with open(os.path.join(a.out, "decisions", a.name + ".txt"), "w") as out:
        out.write("".join(f"{d}\n" for d in decisions))
    print(f"pytorch_modes gap: {len(decisions)} positions, logits {str(dtype).replace('torch.', '')}, the forward passes {total_s:.2f} s in all")


# --- The report -------------------------------------------------------------

B, T, L, D, H, HD, V = 16, 256, 6, 384, 6, 64, 92  # pytorch_baseline.py's
FWD = B * T * L * 24 * D * D + L * 4 * B * H * T * T * HD + 2 * B * T * D * V
STEP = 3 * FWD
# D127's rows of record (docs/retime-2026-10-15-table4.md), ten X925 pinned.
D127 = {
    "fwd.pytorch.10": (151.2, 142.1),
    "fwd.pytorch.20": (263.0, 259.6),
    "step.pytorch.10": (1026.9, 1010.1),
    "step.pytorch.20": (1396.2, 1371.8),
    "step.c.plan.10": (1003.5, 991.7),
    "step.c.plan.10.w": (846.9, 839.6),
    "step.native.plan.10": (1354.3, 1350.3),
    "fwd.c.plan.10.w": (226.0, 225.8),
    "fwd.native.plan.10": (343.4, 340.9),
}
LABEL = {
    "baseline": "eager, `pytorch_baseline.py` (D127's row)",
    "train_py": "eager, the exported `train.py` (D127's row)",
    "eager": "eager",
    "compile": "`torch.compile`",
    "amp": "AMP `bfloat16`",
    "compile+amp": "`torch.compile` + AMP `bfloat16`",
    "eager-nodrop": "eager, without dropout",
    "compile-nodrop": "`torch.compile`, without dropout",
}


def thousands(x, digits=1):
    return f"{x:,.{digits}f}".replace(",", " ")


class Row:
    def __init__(self, values):
        self.values = values
        n = len(values)
        self.median = sorted(values)[n // 2]
        self.fastest = min(values)
        self.spread = max(values) / min(values)


def report(a):
    dirs = sorted(glob.glob(os.path.join(a.dir, "round*/results.txt")), key=lambda p: int(re.search(r"round(\d+)", p).group(1)))
    n = len(dirs)
    if n % 2 == 0:
        sys.exit(f"{a.dir}: {n} rounds; the median of an even number is no process's")
    res = []
    for i, p in enumerate(dirs):
        r = {}
        # The rows without dropout, taken after the rounds in rounds of their own.
        for q in (p, os.path.join(a.dir, f"nodrop-round{i + 1}", "results.txt")):
            if os.path.exists(q):
                for line in open(q):
                    k, *v = line.split()
                    r[k] = v
        res.append(r)
    keys = [k for k in res[0] if all(k in r for r in res)]
    col = lambda k, i: Row([float(r[k][i]) for r in res])
    pr = print
    head = " | ".join(f"round {i + 1}" for i in range(n))
    pr(f"Each value is a process's median of seven calls or steps; a row's {n} values are {n} processes, one per round. **median**: the median of the {n}; **fastest**: the smallest; **max ÷ min**: the row's spread; ÷ eager from the medians. *first*: the first call or step of the median process, which compiles under `torch.compile` (its median across the processes in brackets).")
    for w, title, total in (("fwd", "The forward pass", FWD), ("step", "One training step", STEP)):
        for th in ("10", "20"):
            rows = [k for k in keys if k.startswith(f"{w}.pytorch.") and k.endswith(f".{th}")]
            if not rows:
                continue
            pr()
            pr(f"### {title}, {'ten X925 pinned' if th == '10' else 'twenty threads, unpinned'}")
            pr()
            pr(f"| mode | {head} | median | fastest | max ÷ min | GFLOP/s | ÷ eager | first ms |")
            pr("|---|" + "---|" * n + "---|---|---|---|---|---|")
            base = col(f"{w}.pytorch.eager.{th}", 0)
            for k in rows:
                r = col(k, 0)
                mode = k.split(".")[2]
                first = "—"
                if res[0][k][2] != "-":
                    f = col(k, 2)
                    first = f"{thousands(f.values[r.values.index(r.median)])} ({thousands(f.median)})"
                pr(f"| {LABEL[mode]} | " + " | ".join(thousands(v) for v in r.values) + f" | **{thousands(r.median)}** | {thousands(r.fastest)} | {r.spread:.3f} | {total / 1e9 / (r.median / 1e3):.1f} | {r.median / base.median:.2f} | {first} |")
    ctl = [k for k in keys if ".c." in k or ".native." in k]
    if ctl:
        pr()
        pr("### The compiler's step, ten X925 pinned (the control: D127's rows timed again)")
        pr()
        pr(f"| row | {head} | median | fastest | max ÷ min | D127's median | D127's fastest |")
        pr("|---|" + "---|" * n + "---|---|---|---|---|")
        for side, key in (("C, under the plan, OpenMP's default", "c.plan.10"), ("C, under the plan, `GOMP_SPINCOUNT=8M`", "c.plan.10.w"), ("native, under the plan", "native.plan.10")):
            g, o = f"grad.{key}", f"adam.{key}"
            if g not in keys:
                continue
            r = Row([float(x[g][0]) + float(x[o][0]) for x in res])
            m, f = D127[f"step.{key}"]
            pr(f"| {side}, `loss_grad` + `adamw.step` | " + " | ".join(thousands(v) for v in r.values) + f" | **{thousands(r.median)}** | {thousands(r.fastest)} | {r.spread:.3f} | {thousands(m)} | {thousands(f)} |")
    pr()
    pr("### The compiler's step against each mode's (T2's ratio), ten X925 pinned")
    pr()
    pr("The compiler's rows are D127's (C under the plan 1 003.5 ms, at OpenMP's default waiting as T2 reads it; with its workers spinning 846.9; native 1 354.3), each mode's step its median here, and beside it fastest ÷ fastest.")
    pr()
    pr("| PyTorch's step | ms | C ÷ it | fastest | C, spinning, ÷ it | native ÷ it |")
    pr("|---|---|---|---|---|---|")
    for k in [k for k in keys if k.startswith("step.pytorch.") and k.endswith(".10")]:
        r = col(k, 0)
        c, cf = D127["step.c.plan.10"]
        w, _ = D127["step.c.plan.10.w"]
        nat, _ = D127["step.native.plan.10"]
        pr(f"| {LABEL[k.split('.')[2]]} | {thousands(r.median)} | **{c / r.median:.2f}** | {cf / r.fastest:.2f} | {w / r.median:.2f} | {nat / r.median:.2f} |")


    quality = os.path.join(a.dir, "quality", "quality.md")
    if os.path.exists(quality):
        text = open(quality).read()
        runs = {}
        for m, s, loss, acc in re.findall(r"^\| PyTorch (\S+), seed (\d+) \| ([0-9.]+) \| ([0-9.]+) \|", text, re.M):
            runs.setdefault(m, {})[int(s)] = (float(loss), float(acc))
        presk = [(float(x), float(y)) for x, y in re.findall(r"^\| presk all-f64[^|]*\| ([0-9.]+) \| ([0-9.]+) \|", text, re.M)]
        seeds = sorted(runs["eager"])
        pr()
        pr("### The quality at D = 64: validation loss and next-character accuracy in float64, 461 held-out windows")
        pr()
        pr("Each mode's model trained from the same seed, in the same batch order; − eager is the paired difference, seed by seed. The seeds' spread is the largest loss less the smallest, over the seeds of one mode.")
        pr()
        h = " | ".join(f"seed {s}" for s in seeds)
        pr(f"| mode | {h} | mean | − eager, per seed | mean − eager | seeds' spread | accuracy, mean | accuracy − eager, mean |")
        pr("|---|" + "---|" * len(seeds) + "---|---|---|---|---|---|")
        e = runs["eager"]
        for m in [m for m in MODES if m in runs]:
            r = runs[m]
            loss = [r[s][0] for s in seeds]
            acc = [r[s][1] for s in seeds]
            d = [r[s][0] - e[s][0] for s in seeds]
            da = [r[s][1] - e[s][1] for s in seeds]
            pr(f"| {LABEL[m]} | " + " | ".join(f"{x:.7f}" for x in loss) + f" | {statistics.mean(loss):.7f} | " + ", ".join(f"{x:+.2e}" for x in d) + f" | {statistics.mean(d):+.2e} | {max(loss) - min(loss):.4f} | {statistics.mean(acc):.6f} | {statistics.mean(da):+.6f} |")
        if presk:
            loss = [x for x, _ in presk]
            cells = [f"{x:.7f}" for x in loss[: len(seeds)]] + ["—"] * max(0, len(seeds) - len(loss))
            pr(f"| presk all-`f64`, the curve's runs ({len(presk)} seeds: " + ", ".join(f"{x:.7f}" for x in loss) + ") | " + " | ".join(cells) + f" | {statistics.mean(loss):.7f} | — | — | {max(loss) - min(loss):.4f} | {statistics.mean(y for _, y in presk):.6f} | — |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forward")
    f.add_argument("--export", required=True)
    f.add_argument("--weights", required=True)
    f.add_argument("--inputs", required=True)
    f.add_argument("--threads", type=int, required=True)
    f.add_argument("--mode", choices=MODES, required=True)
    f.add_argument("--repeat", type=int, default=7)
    f.add_argument("--out")
    s = sub.add_parser("step")
    s.add_argument("--export", required=True)
    s.add_argument("--threads", type=int, required=True)
    s.add_argument("--mode", choices=MODES, required=True)
    s.add_argument("--steps", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-dropout", action="store_true")
    q = sub.add_parser("profile")
    q.add_argument("--export", required=True)
    q.add_argument("--threads", type=int, required=True)
    q.add_argument("--mode", choices=MODES, required=True)
    q.add_argument("--no-dropout", action="store_true")
    q.add_argument("--out", required=True)
    e = sub.add_parser("explain")
    e.add_argument("--export", required=True)
    e.add_argument("--inputs", required=True)
    e.add_argument("--threads", type=int, required=True)
    e.add_argument("--mode", choices=MODES, required=True)
    t = sub.add_parser("train")
    t.add_argument("--export", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--steps", type=int, default=600)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--batch", type=int)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, required=True)
    t.add_argument("--mode", choices=MODES, required=True)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--out", required=True)
    v = sub.add_parser("evaluate")
    v.add_argument("--export", required=True)
    v.add_argument("--data", required=True)
    v.add_argument("--threads", type=int, required=True)
    v.add_argument("models", nargs="+", metavar="NAME=FILE")
    g = sub.add_parser("gap")
    g.add_argument("--export", required=True)
    g.add_argument("--weights", required=True)
    g.add_argument("--data", required=True)
    g.add_argument("--exact", required=True)
    g.add_argument("--threads", type=int, required=True)
    g.add_argument("--mode", choices=MODES, required=True)
    g.add_argument("--name", required=True)
    g.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("dir")
    a = ap.parse_args()
    {"forward": forward, "step": step, "profile": profile, "explain": explain, "train": train_run, "evaluate": evaluate, "gap": gap, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
