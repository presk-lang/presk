"""PyTorch's side of the baseline at 0.9.0 (D87, experiments/pytorch_baseline.sh).

    python pytorch_baseline.py forward --export DIR --weights W.psk.d --inputs IDS.psk.d --threads N [--out Y.npy]
    python pytorch_baseline.py forward --model llama --export DIR --batch B --seq T --threads N
    python pytorch_baseline.py product --in 384 --out 1536 --threads N
    python pytorch_baseline.py compare --reference F64.psk.d NAME=FILE ...
    python pytorch_baseline.py flops --export DIR --inputs IDS.psk.d
    python pytorch_baseline.py flops --model llama --export DIR --batch B --seq T
    python pytorch_baseline.py report OUT

`forward` times the course GPT's forward pass as `presk export --target
pytorch` writes it (DIR, the export's `_05_gpt.py` and `pskd.py`), in eval
mode under `torch.no_grad()`, with presk's checkpoint and the token indices
the C export's conformance inputs hold: WARMUP calls, then REPEAT timed
ones, printing the median and the best in ms and writing the logits.
`product` times `x @ w + b`, the model's `Linear`, at x : [16, 256, IN].
`compare` prints each logits file's distance to the reference. `flops`
counts the floating-point operations of the forward pass and of one
training step's forward and backward with PyTorch's own counter. `report`
writes experiments/pytorch_baseline.sh's report from what it left in OUT.

`--model llama` (D107, experiments/pytorch_decoder.sh) builds the Llama 2
decoder instead, `llama.Llama(T)` as `presk export --target pytorch` writes
examples/coverage/llama.psk, with its own initialization (every matrix normal(std =
0.02), every γ ones, the distributions presk draws from) after
`torch.manual_seed(0)`, on token indices drawn by `torch.randint(0, 32000,
(B, T))` from a generator seeded 0: no checkpoint and no inputs file, since
at the 7B width neither is written. `forward` then prints the logits' size,
whether they are finite and their largest magnitude in place of writing
them, and `flops` counts the forward pass only.
"""
import argparse
import os
import re
import statistics
import sys
import time

import numpy as np
import torch

# The parameter tree of `params GPT` as `pskd.read_params` reads it: the
# module's `param`s in their order (tok, pos, head), then its `use`s
# (blocks, a stack{6} of Block, and lnf), each module's fields in their
# order. `presk export --serve` prints this literal; the course GPT has no
# `inference serve`, so it is written here, and the float64 forward pass
# against the native backend's f64 checks it (a leaf out of place moves the
# logits by far more than rounding).
LINEAR = {"w": None, "b": None}
NORM = {"γ": None, "β": None}
LAYOUT = {
    "tok": None,
    "pos": None,
    "head": None,
    "blocks": ["stack", {"attn": {"wq": LINEAR, "wk": LINEAR, "wv": LINEAR, "wo": LINEAR}, "ff": {"up": LINEAR, "down": LINEAR}, "ln1": NORM, "ln2": NORM}],
    "lnf": NORM,
}


def timed(f, warmup, repeat):
    for _ in range(warmup):
        f()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        f()
        times.append(1e3 * (time.perf_counter() - t0))
    return statistics.median(times), min(times)


def load_llama(export, batch, seq, dtype):
    """The decoder, its own seeded draw, and seeded token indices."""
    sys.path.insert(0, os.path.abspath(export))
    from llama import V, Llama

    torch.manual_seed(0)
    model = Llama(seq).to(dtype)
    model.eval()
    ids = torch.randint(0, V, (batch, seq), generator=torch.Generator().manual_seed(0))
    return model, ids


def load(export, weights, inputs, dtype):
    sys.path.insert(0, os.path.abspath(export))
    from _05_gpt import GPT
    from pskd import PskdDataset, read_params

    ids = PskdDataset(inputs)[0]
    model = GPT(ids.shape[1]).to(dtype)
    if weights:
        model.load_state_dict(read_params(weights, LAYOUT, model))
    model.eval()
    return model, ids


def forward_llama(a):
    torch.set_num_threads(a.threads)
    model, ids = load_llama(a.export, a.batch, a.seq, getattr(torch, a.dtype))
    with torch.no_grad():
        med, best = timed(lambda: model(ids), a.warmup, a.repeat)
        y = model(ids)
    finite = bool(torch.isfinite(y).all())
    print(f"pytorch forward: torch {torch.__version__}, {torch.get_num_threads()} threads, {a.dtype}, {a.repeat} calls after {a.warmup}, median {med:.3f} ms, best {best:.3f} ms")
    print(f"pytorch logits: {list(y.shape)}, finite {'yes' if finite else 'NO'}, max|y| {float(y.abs().max()):.4f}")


def forward(a):
    if a.model == "llama":
        return forward_llama(a)
    torch.set_num_threads(a.threads)
    model, ids = load(a.export, a.weights, a.inputs, getattr(torch, a.dtype))
    with torch.no_grad():
        med, best = timed(lambda: model(ids, None), a.warmup, a.repeat)
        y = model(ids, None)
    if a.out:
        np.save(a.out, y.double().numpy())
    print(f"pytorch forward: torch {torch.__version__}, {torch.get_num_threads()} threads, {a.dtype}, {a.repeat} calls after {a.warmup}, median {med:.3f} ms, best {best:.3f} ms")
    # The exported `loss` calls the model through torch.func.functional_call
    # on a module it builds at every call, GPT(T), whose initializers draw
    # every parameter: a cost inside train.py's step that eval does not pay.
    from _05_gpt import GPT

    built, _ = timed(lambda: GPT(ids.shape[1]), a.warmup, a.repeat)
    print(f"pytorch GPT({ids.shape[1]}): built in {built:.3f} ms, median")


def product(a):
    torch.set_num_threads(a.threads)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 256, a.inp, generator=g)
    w = torch.randn(a.outp, a.inp, generator=g).t().contiguous() / a.inp**0.5
    b = torch.randn(a.outp, generator=g)
    with torch.no_grad():
        med, best = timed(lambda: x @ w + b, a.warmup, a.repeat)
    print(f"pytorch product: [16, 256, {a.inp}] · [{a.inp}, {a.outp}] + b, {torch.get_num_threads()} threads, {a.repeat} calls after {a.warmup}, median {med:.3f} ms, best {best:.3f} ms")


def logits(path):
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path)).double()
    return read_pskd_reals(path)


def read_pskd_reals(path):
    """The one sample of a .psk.d file of reals, as float64 (the leading
    header is magic, version, type text, count; in version 2 the leaf's
    encoding follows, 0 f64, 1 f32, 2 bf16, presk D180)."""
    import struct

    with open(path, "rb") as f:
        assert f.read(4) == b"PSKD", path
        (version,) = struct.unpack("<H", f.read(2))
        (n,) = struct.unpack("<I", f.read(4))
        text = f.read(n).decode()
        (count,) = struct.unpack("<Q", f.read(8))
        code = 0
        if version == 2:
            (k,) = struct.unpack("<I", f.read(4))
            (code,) = f.read(k)
        body = f.read()
        if code == 2:
            data = (np.frombuffer(body, dtype="<u2").astype(np.uint32) << 16).view("<f4").astype(np.float64)
        else:
            data = np.frombuffer(body, dtype="<f8" if code == 0 else "<f4").astype(np.float64)
    assert count == 1, (path, count)
    shape = [int(s.rpartition(":")[2]) for s in text[text.index("[") + 1 : text.rindex("]")].split(",")]
    return torch.from_numpy(data.reshape(shape).copy())


def compare(a):
    ref = logits(a.reference)
    print("| side | ‖Δ‖₂ / ‖y‖₂ | max \\|Δ\\| | max \\|Δ\\| / max \\|y\\| | positions whose largest logit moves |")
    print("|---|---|---|---|---|")
    top = ref.argmax(-1)
    for spec in a.sides:
        name, _, path = spec.partition("=")
        y = logits(path)
        assert y.shape == ref.shape, (name, y.shape, ref.shape)
        d = y - ref
        moved = int((y.argmax(-1) != top).sum())
        print(f"| {name} | {float(d.norm() / ref.norm()):.2e} | {float(d.abs().max()):.2e} | {float(d.abs().max() / ref.abs().max()):.2e} | {moved} of {top.numel()} |")


def flops(a):
    from torch.utils.flop_counter import FlopCounterMode

    if a.model == "llama":
        model, ids = load_llama(a.export, a.batch, a.seq, torch.float32)
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(ids)
        fwd = fc.get_total_flops()
        print(f"pytorch flops: forward {fwd} ({fwd / 1e9:.3f} GFLOP)")
        return
    model, ids = load(a.export, None, a.inputs, torch.float32)
    sys.path.insert(0, os.path.abspath(a.export))
    from _05_gpt import loss

    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(ids, None)
    fwd = fc.get_total_flops()
    theta = dict(model.named_parameters())
    with FlopCounterMode(display=False) as fc:
        loss(theta, None, (ids, ids), training=True).backward()
    step = fc.get_total_flops()
    print(f"pytorch flops: forward {fwd} ({fwd / 1e9:.3f} GFLOP), forward and backward of the loss {step} ({step / 1e9:.3f} GFLOP)")


# The course GPT's constants (examples/corpus/nlp/05_gpt.psk) at the timed size.
B, T, L, D, H, HD, V = 16, 256, 6, 384, 6, 64, 92


def model_flops():
    """The forward pass's floating-point operations: its contractions, a
    multiply and an add a term (the six blocks' four D x D products and
    two D x 4D ones, attention's two T x T contractions computed in full,
    the head); and one training step's, each contraction with its two
    derivatives, of its size. Normalizations, softmax, GELU, biases and
    residual additions are not counted, on any side."""
    fwd = B * T * L * 24 * D * D + L * 4 * B * H * T * T * HD + 2 * B * T * D * V
    return fwd, 3 * fwd


def thousands(x, digits=1):
    """A number as the manual writes it, thousands apart: 1 094.1."""
    return f"{x:,.{digits}f}".replace(",", " ")


def report(a):
    out = a.dir
    res = {}
    for line in open(f"{out}/results.txt"):
        key, *vals = line.split()
        res[key] = vals
    ms = lambda k: float(res[k][0])
    best = lambda k: res[k][1]
    fwd, step = model_flops()
    counted = [int(n) for n in re.findall(r"(\d+) \(", open(f"{out}/flops.txt").read())]
    machine = open(f"{out}/machine.txt").read().strip()
    libs = open(f"{out}/torch-libs.txt").read().strip()
    head = lambda n, text: print(f"\n{'#' * n} {text}\n")
    date = machine.splitlines()[0].split()[1][:10]
    rev = machine.splitlines()[1].split()[1].rstrip(",")
    version = machine.splitlines()[1].split(", ", 1)[1]  # `presk --version`
    flags = re.search(r"^cflags (.*)$", machine, re.M)
    flags = flags.group(1).replace(" -Wall -Wextra", "") if flags else "-O2"
    # The C compiler as `gcc --version` named it (D184: GCC 16.2 beside 13.3).
    gcc = re.search(r"^gcc \(.*\) (\S+)$", machine, re.M)
    gcc = f"GCC {gcc.group(1)}" if gcc else "GCC"
    # Since 0.10.0 step 7's follow-up: every C program timed under OpenMP's
    # default waiting and under C_WAIT, the native pool's (keys `….w`).
    wait = re.search(r"^c_wait (.*)$", machine, re.M)
    wait = wait.group(1) if wait else None
    waited = lambda k: f"{k}.w" if wait and f"{k}.w" in res else k
    print(f"# The course GPT on three sides: {version} against PyTorch")
    print()
    print(f"The course GPT (`examples/corpus/nlp/05_gpt.psk`, D = {D}, {H} heads, {L} blocks, V = {V}) at B = {B}, T = {T}, `f32` on every side, with the checkpoint `out/course_probe_f32/final.psk.d` and the token indices the C export synthesizes; {date}, presk `{rev}`, `experiments/pytorch_baseline.sh`. Three sides: presk's native backend (`presk bench --plan-only`, `presk train`), the C program `presk export --target c` writes ({gcc}, `{flags}`, OpenMP), and PyTorch {torch.__version__} running the model `presk export --target pytorch` writes. Each time is the median of seven calls or steps, the first call or step, which compiles natively, left out on the native and PyTorch sides; the C program's seven calls are all counted.")
    print()
    if wait:
        print(f"**The C program's threads.** Each C program is timed twice: under the OpenMP runtime's default, by which an idle worker sleeps after about 0.1 ms (libgomp's 300 000 spins on these cores), and with `{wait}`, by which it spins about 2 ms before it sleeps, as the native backend's pool does (presk D97's addendum). T4 compares the native backend with the second, both sides waiting alike; T2 compares PyTorch, at its own runtime's default, with the first. The same bits either way.")
        print()
    print(f"**Operations.** The forward pass is {fwd / 1e9:.2f} GFLOP of contractions, a training step {step / 1e9:.2f} (each contraction and its two derivatives); PyTorch's counter (`torch.utils.flop_counter`) gives {counted[0] / 1e9:.2f} and {counted[1] / 1e9:.2f}. The rates below divide these by the median, whatever a side executes: presk's gradient recomputes the forward pass's products (below, where the C program's time goes).")

    def rows(kind, sides, total, unit, with_best):
        print(f"| threads | side | layout | {unit} |" + (" best ms |" if with_best else "") + " GFLOP/s | ÷ PyTorch |")
        print("|---|---|---|---|" + ("---|" if with_best else "") + "---|---|")
        for th in ("10", "20"):
            where = "ten X925, pinned" if th == "10" else "twenty, unpinned"
            base = ms(f"{kind}.pytorch.{th}")
            for side, layout, key, b in sides(th):
                t = key if isinstance(key, float) else ms(key)
                bcol = f" {thousands(float(b))} |" if with_best else ""
                print(f"| {th} ({where}) | {side} | {layout} | {thousands(t)} |{bcol} {total / 1e9 / (t / 1e3):.1f} | {t / base:.2f} |")

    head(2, "The forward pass")

    def fwd_sides(th):
        out = []
        for layout in ("default", "plan"):
            out.append(("native", layout, f"fwd.native.{layout}.{th}", best(f"fwd.native.{layout}.{th}")))
        for layout in ("default", "plan"):
            out.append(("C", layout, f"fwd.c.{layout}.{th}", best(f"fwd.c.{layout}.{th}")))
        if wait:
            for layout in ("default", "plan"):
                out.append((f"C, `{wait}`", layout, f"fwd.c.{layout}.{th}.w", best(f"fwd.c.{layout}.{th}.w")))
        out.append(("PyTorch", "its own", f"fwd.pytorch.{th}", best(f"fwd.pytorch.{th}")))
        return out

    rows("fwd", fwd_sides, fwd, "median ms", True)
    head(2, "One training step")

    def step_sides(th):
        out = []
        for layout in ("default", "plan"):
            out.append(("native, `presk train`", layout, f"step.native.{layout}.{th}", ""))
        for layout in ("default", "plan"):
            if f"grad.native.{layout}.{th}" in res:
                g, o = ms(f"grad.native.{layout}.{th}"), ms(f"adam.native.{layout}.{th}")
                out.append((f"native, `loss_grad` {thousands(g, 0)} + `adamw.step` {thousands(o, 0)}", layout, g + o, ""))
        for layout in ("default", "plan"):
            g, o = ms(f"grad.c.{layout}.{th}"), ms(f"adam.c.{layout}.{th}")
            out.append((f"C, `loss_grad` {thousands(g, 0)} + `adamw.step` {thousands(o, 0)}", layout, g + o, ""))
        if wait:
            for layout in ("default", "plan"):
                g, o = ms(f"grad.c.{layout}.{th}.w"), ms(f"adam.c.{layout}.{th}.w")
                out.append((f"C, `{wait}`, `loss_grad` {thousands(g, 0)} + `adamw.step` {thousands(o, 0)}", layout, g + o, ""))
        out.append(("PyTorch, `train.py`", "its own", f"step.pytorch.{th}", ""))
        return out

    rows("step", step_sides, step, "median ms", False)
    built = re.search(r"built in ([0-9.]+) ms", open(f"{out}/log/fwd.pytorch.10.log").read())
    print()
    print(f"The C step is its two programs' medians added, and so is the native step by `presk bench` (since 0.10.0 step 6, T4's step like for like); neither drops (the export drops keys, so its dropout is the expectation, and `presk bench` does not train), where `presk train` and `train.py` drop with their own masks. `train.py`'s step builds the module `GPT(256)` at every call of the exported `loss` (`torch.func.functional_call` on a fresh module, its initializers drawing every parameter): {float(built.group(1)):.1f} ms on ten threads, inside its time.")
    same = res.get("layouts.same", ["?"])[0] == "yes" and res.get("train.same", ["?"])[0] == "yes"
    print()
    print(f"Every layout and thread count gives the same `f32` logits and the same gradient, bit for bit, and the native training run on ten threads under the plan ends on the default layout's weights byte for byte: {'yes' if same else 'NO'}.")
    if "grad.native.plan.10" in res:
        head(2, "T4: the native backend against the C program")
        print("Plan 0.10.0's statement T4 (§3.5, as amended after steps 5 and 7): the native backend's forward pass and training step under the plans take at most 1.3× the C program's, same plans, same cores; the step compared like for like, the native `loss_grad` and `adamw.step` timed by `presk bench` against the C program's, neither drawing dropout masks, `presk train`'s step reported beside it. Judged on the ten X925 cores pinned" + (f", the C program's workers waiting as the native pool's (`{wait}`), the row of record; the C program's default waiting and every core unpinned reported." if wait else ", the row of record; every core unpinned reported."))
        print()
        waits = ((".w", f"`{wait}`"), ("", "OpenMP's default")) if wait else (("", None),)
        print("| threads |" + (" the C program's waiting |" if wait else "") + " workload | native ms | C ms | native ÷ C | T4 (≤ 1.3×) |")
        print("|---|" + ("---|" if wait else "") + "---|---|---|---|---|")
        for th in ("10", "20"):
            where = "ten X925, pinned" if th == "10" else "twenty, unpinned"
            for w in waits:
                col = f" {w[1]} |" if w[1] else ""
                fn, fc = ms(f"fwd.native.plan.{th}"), ms(f"fwd.c.plan.{th}{w[0]}")
                sn = ms(f"grad.native.plan.{th}") + ms(f"adam.native.plan.{th}")
                sc = ms(f"grad.c.plan.{th}{w[0]}") + ms(f"adam.c.plan.{th}{w[0]}")
                tr = ms(f"step.native.plan.{th}")
                record = th == "10" and (w[0] == ".w" or not wait)
                verdict = lambda r: ("holds" if r <= 1.3 else "fails") if record else "(reported)"
                print(f"| {th} ({where}) |{col} the forward pass | {thousands(fn)} | {thousands(fc)} | {fn / fc:.2f} | {verdict(fn / fc)} |")
                print(f"| {th} ({where}) |{col} the step: `loss_grad` + `adamw.step` | {thousands(sn)} | {thousands(sc)} | {sn / sc:.2f} | {verdict(sn / sc)} |")
                print(f"| {th} ({where}) |{col} beside: `presk train`'s step, drawing its masks | {thousands(tr)} | {thousands(sc)} | {tr / sc:.2f} | — |")
    head(2, "The linear products alone (ten threads, pinned)")
    print("`Linear{In,Out}.apply` of the course GPT, `x · w + b` at x : [16, 256, In], under the layout the forward plan gives `Linear.apply` (one layout for its three shapes); PyTorch `x @ w + b`.")
    print()
    print("A call's time on each side, its product and its bias together; for the native backend also its kernels' alone (`presk bench --calibrate`: the product's and the bias's, or one when the plan states the bias as the product's epilogue, plan 0.10.0 step 7), the rest of its call being its host's buffers and copies. GFLOP/s: the product's operations over the call's time, or over the kernels'.")
    print()
    print("| product | GFLOP | native ms | its kernels ms | C ms | PyTorch ms | native GFLOP/s | its kernels | C | PyTorch |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for i, o in ((384, 384), (384, 1536), (1536, 384)):
        g = 2 * B * T * i * o / 1e9
        t = [ms(f"lin.native.{i}.{o}"), ms(f"kern.native.lin.{i}.{o}"), ms(f"lin.c.{i}.{o}"), ms(f"lin.pytorch.{i}.{o}")]
        print(f"| [16, 256, {i}] · [{i}, {o}] | {g:.2f} | " + " | ".join(f"{x:.2f}" for x in t) + " | " + " | ".join(f"{g / (x / 1e3):.0f}" for x in t) + " |")
    head(2, "Kernels and host, the forward pass under its plan (ten threads, pinned)")
    print(f"The same kernels on both presk sides (C1), so what differs is who compiles them (Cranelift against GCC `{flags}`) and what runs around them. The native kernels' time is `presk bench --calibrate`'s (every launch timed, its threads' start included); the C kernels' is the instrumented program's (below, three calls after a first; the program's median call is uninstrumented). The host is the call's median less its kernels. Under the plan only: `--calibrate` runs a nest no plan states scalar, where the native backend gives it its natural lanes, so under the default layout it times other kernels; the forward plan leaves only the host's small kernels (the causal mask, the positions) unstated.")
    if wait:
        print()
        print(f"The C program's call is its time with `{wait}`, under which its kernels were timed.")
    print()
    print("| side | the call, ms | its kernels, ms | the rest, the host's, ms |")
    print("|---|---|---|---|")
    for side, call, kern in (("native", "fwd.native.plan.10", "kern.native.fwd-plan"), ("C", waited("fwd.c.plan.10"), "kern.c.fwd-plan")):
        c, k = ms(call), ms(kern)
        print(f"| {side} | {thousands(c)} | {thousands(k)} | {thousands(c - k)} |")
    head(2, "Distances to the native backend's `f64` logits")
    print(open(f"{out}/distances.md").read().strip())
    print()
    print("The checkpoint is a probe a few steps from the initialization (its head's largest weight is 1.8·10⁻⁵), so the logits are of order 10⁻³ and a position whose largest logit moves is a near-tie.")
    head(2, "Where the C program's time goes (ten threads, pinned, three calls after a first)")
    print("A copy of `model.c` with every launch timed on the thread that makes it, a kernel's call or the OpenMP region that divides it, its team's start and its wait for the slowest thread included (`experiments/kernel_time.py`, `kernel_time.c`); the kernels are the program's, and its outputs are unchanged. Four calls, the first left out" + (f", with `{wait}`" if wait else "") + ". GFLOP counts every launch, recomputations included.")
    for w, title in (("fwd-default", "The forward pass, default layout"), ("fwd-plan", "The forward pass under its plan"), ("grad-default", "The gradient, `loss_grad`, default layout"), ("grad-plan", "The gradient under its plan")):
        head(3, title)
        k, call = (float(x) for x in res[f"kern.c.{w}"][:2])
        print(f"The instrumented program: {thousands(call)} ms a call (the median of its four); its kernels {thousands(k)} ms; the rest, the host's, {thousands(call - k)} ms.")
        print()
        print(open(f"{out}/kernels-{w}.md").read().strip())
    head(2, "The machine and PyTorch's libraries")
    print("```")
    print(machine)
    print(libs)
    print("```")
    print()
    print("What PyTorch's forward pass hands to oneDNN, one call (`ONEDNN_VERBOSE=1`: count, primitive, implementation, shapes); the linear products are not among them, so they are the wheel's OpenBLAS:")
    print()
    print("```")
    print(open(f"{out}/onednn.txt").read().rstrip())
    print("```")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forward")
    f.add_argument("--model", default="gpt", choices=["gpt", "llama"])
    f.add_argument("--export", required=True)
    f.add_argument("--weights")
    f.add_argument("--inputs")
    f.add_argument("--batch", type=int)
    f.add_argument("--seq", type=int)
    f.add_argument("--threads", type=int, required=True)
    f.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    f.add_argument("--warmup", type=int, default=1)
    f.add_argument("--repeat", type=int, default=7)
    f.add_argument("--out")
    p = sub.add_parser("product")
    p.add_argument("--in", dest="inp", type=int, required=True)
    p.add_argument("--out", dest="outp", type=int, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=7)
    c = sub.add_parser("compare")
    c.add_argument("--reference", required=True)
    c.add_argument("sides", nargs="+", metavar="NAME=FILE")
    n = sub.add_parser("flops")
    n.add_argument("--model", default="gpt", choices=["gpt", "llama"])
    n.add_argument("--export", required=True)
    n.add_argument("--inputs")
    n.add_argument("--batch", type=int)
    n.add_argument("--seq", type=int)
    r = sub.add_parser("report")
    r.add_argument("dir")
    a = ap.parse_args()
    if a.cmd in ("forward", "flops"):
        # The GPT reads a checkpoint and the C export's inputs; the decoder draws both.
        need = ["batch", "seq"] if a.model == "llama" else ["inputs"] + (["weights"] if a.cmd == "forward" else [])
        for k in need:
            if getattr(a, k) is None:
                ap.error(f"{a.cmd} --model {a.model} needs --{k}")
    {"forward": forward, "product": product, "compare": compare, "flops": flops, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
