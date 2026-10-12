#!/bin/bash
# PyTorch eager on the Llama 2 decoder block (D107): D100's protocol
# (pytorch_baseline.sh, the course GPT's forward pass on three sides) on one
# block of coverage/llama.psk at the 7B configuration's width (G = 32, L = 1:
# D = 4096, 32 heads, SwiGLU's 11008, V = 32 000), the model D104 and D106
# ran exactly and natively. The forward pass `Llama.apply` at T tokens
# (default 384, D104's) and batch 1 and 20, `f32`. A measurement, not a
# statement: it decides nothing, and it is no performance claim — PyTorch
# eager on the CPU, no `torch.compile`, no GPU, no training step.
#
#   experiments/pytorch_decoder.sh OUT
#
# Rows, per batch size, on the ten X925 cores pinned (CORES, `taskset`, ten
# threads), the row of record as T2's:
#   native   `presk bench --native-only --plan-only --warmup 1`, `f32`,
#            under the default layouts (a plan stating no nest: the native
#            backend's own, its natural lanes and a kernel's outermost loop
#            divided among the threads, which at batch 1 is the batch axis,
#            so one core) and under `presk plan`'s plan with the frozen
#            profile (PROFILE), where it gives one
#   int8     the same with `int8` weights on every matrix, `f32` elsewhere:
#            experiments/llama.sh's `int8_all.json` (D106), written here in
#            the same form, under the default layouts and under a plan if
#            `presk plan` accepts the quantised realization
#   PyTorch  the module `presk export --target pytorch --mode readable`
#            writes, `llama.Llama(T)` with its own draw of the distributions
#            presk draws from (seed 0), in eval mode under `torch.no_grad()`,
#            on token ids drawn by `torch.randint` (seed 0): ten threads, and
#            at batch 1 also one thread pinned to one X925 (ONE), beside the
#            native backend's one-core batch 1
# Each row: one warmup call, then REPEAT calls (REPEAT20 for the native rows
# at batch 20, where seven may take more than half an hour), the median and
# the best. The native process first draws the parameters exactly, outside
# the timed calls; PyTorch's draw is outside them too. Besides the times:
# PyTorch's counter's FLOPs (`torch.utils.flop_counter`) and the compiler's
# own count of the contractions; the logits' size, finiteness and largest
# magnitude on PyTorch's first call.
#
# Every command runs in a systemd scope of its own (MemoryMax=MEM, no swap)
# under `/usr/bin/time -v`, its wall and peak in OUT/log/KEY.time, its
# output in OUT/log/KEY.log, its command line in OUT/log/KEY.cmd. A command
# whose `.time` says it exited 0 is not run again: delete its files to take
# it again. A command that fails is recorded and the others still run.
# T=16 is the gate (D107's step 2), which sizes the run of record. A quiet
# machine: nothing beside it. Writes OUT/report.md.
set -u -o pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PRESK=${PRESK:-target/release/presk}
case $PRESK in /*) ;; *) PRESK=$ROOT/$PRESK ;; esac
PY=$ROOT/${PY:-.venv/bin/python}
PROFILE=${PROFILE:-$HOME/presk-data/ra4-model-0.8.0/profile.json}
CORES=${CORES:-5-9,15-19}
ONE=${ONE:-15}
T=${T:-384}
BATCHES=${BATCHES:-1 20}
REPEAT=${REPEAT:-7}
REPEAT20=${REPEAT20:-$REPEAT}
MEM=${MEM:-108G}
OUT=${1:?usage: experiments/pytorch_decoder.sh OUT}
FILE=coverage/llama.psk
CONSTS=(--const G=32 --const L=1)

for f in "$PRESK" "$PY" "$PROFILE"; do
    [ -e "$f" ] || { echo "pytorch_decoder.sh: missing $f" >&2; exit 1; }
done
mkdir -p "$OUT/log" "$OUT/plans"
OUT=$(cd "$OUT" && pwd)
say() { echo "-- $(date +%H:%M:%S) $*"; }
ok() { [ -s "$OUT/log/$1.time" ] && grep -q "Exit status: 0" "$OUT/log/$1.time"; }
run() { # key cores -- command: in its own scope, pinned to `cores` (`-`: not pinned)
    local key=$1 cores=$2 rc
    shift 2
    if ok "$key"; then
        say "$key: done before, kept"
        return 0
    fi
    local pin=()
    [ "$cores" = - ] || pin=(taskset -c "$cores")
    echo "${pin[*]} $*" | sed "s|$ROOT/||g; s|^ ||" > "$OUT/log/$key.cmd"
    grep -qx "$key" "$OUT/order.txt" 2>/dev/null || echo "$key" >> "$OUT/order.txt"
    say "$key"
    systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 \
        /usr/bin/time -v -o "$OUT/log/$key.time" "${pin[@]}" "$@" > "$OUT/log/$key.log" 2>&1
    rc=$?
    [ "$rc" = 0 ] || say "$key: exit $rc"
    return 0
}

# --- The machine -----------------------------------------------------------
# The machine of record is the first run's; a later run, which reuses its
# outputs and takes what is missing, appends a line.
say "the machine"
if [ -s "$OUT/machine.txt" ]; then
    echo "later run $(date -Iseconds), presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates experiments || echo -dirty), load $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/machine.txt"
else {
    echo "date $(date -Iseconds)"
    echo "presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates experiments || echo -dirty), $("$PRESK" --version)"
    echo "kernel $(uname -sr), $(nproc) cores"
    lscpu | grep -E "^(Nom de modèle|Model name|Vitesse maximale|CPU max MHz)" | sed 's/  */ /g'
    echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
    echo "T $T, batches $BATCHES, repeat $REPEAT, repeat at batch 20 (native) $REPEAT20, cores $CORES, one-thread core $ONE"
} > "$OUT/machine.txt"
fi
"$PY" -c "import torch; print('torch', torch.__version__); print(torch.__config__.show()); print(torch.__config__.parallel_info())" > "$OUT/torch-config.txt"
TORCH_LIB=$("$PY" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
{
    "$PY" -c "import ctypes, sys; l = ctypes.CDLL(sys.argv[1]); l.openblas_get_config.restype = ctypes.c_char_p; print(l.openblas_get_config().decode())" "$TORCH_LIB/libopenblas.so.0"
    grep -a -m1 -o "arm_compute_version=v[0-9.]*" "$TORCH_LIB/libarm_compute.so"
} > "$OUT/torch-libs.txt"
cat "$OUT/machine.txt" "$OUT/torch-libs.txt"

# --- The export and the realizations -----------------------------------------
# The module, once: it does not depend on T (the position table's length is
# `Llama(T)`'s argument). No conformance file at this width: it would print
# every parameter as JSON. The quantised realization in experiments/llama.sh's
# form, checked equal to its file where that exists.
P=$OUT/plans
python3 -c "import json,sys; print(json.dumps({'default':'f32','order':{'kind':'sequential'},'quant':{p:{'bits':8,'over':-2} for p in json.loads(sys.argv[1])}}))" \
    '["tok","pos","blocks.attn.wq","blocks.attn.wk","blocks.attn.wv","blocks.attn.wo","blocks.ff.gate.w","blocks.ff.up.w","blocks.ff.down.w","head"]' > "$P/int8_all.json"
if [ -s out/llama/int8_all.json ]; then
    cmp -s "$P/int8_all.json" out/llama/int8_all.json && { grep -q "^int8_all.json" "$OUT/machine.txt" || echo "int8_all.json out/llama's" >> "$OUT/machine.txt"; } \
        || { echo "pytorch_decoder.sh: $P/int8_all.json differs from out/llama/int8_all.json" >&2; exit 1; }
fi
echo '{"realization": "f32", "layout": {}}' > "$P/f32-default.json"
echo "{\"realization\": \"@$P/int8_all.json\", \"layout\": {}}" > "$P/int8-default.json"
[ -s "$OUT/pt/llama.py" ] || rm -f "$OUT/log/export.time"
run export - "$PRESK" export "$FILE" --target pytorch --mode readable "${CONSTS[@]}" -o "$OUT/pt"

# --- Plans -----------------------------------------------------------------
# The frozen profile's plan per batch size, `f32` and the quantised
# realization; a refusal is recorded and that row left out.
for b in $BATCHES; do
    run "plan.f32.B$b" - "$PRESK" plan "$FILE" --entry Llama.apply "${CONSTS[@]}" --dim B="$b" --dim T="$T" \
        --realize f32 --profile "$PROFILE" --out "$P/f32-plan-B$b.json"
    run "plan.int8.B$b" - "$PRESK" plan "$FILE" --entry Llama.apply "${CONSTS[@]}" --dim B="$b" --dim T="$T" \
        --realize "@$P/int8_all.json" --profile "$PROFILE" --out "$P/int8-plan-B$b.json"
done

# --- Timing ----------------------------------------------------------------
for b in $BATCHES; do
    rep=$REPEAT
    [ "$b" = 20 ] && rep=$REPEAT20
    say "batch $b"
    run "pytorch.10.B$b" "$CORES" "$PY" experiments/pytorch_baseline.py forward --model llama --export "$OUT/pt" \
        --batch "$b" --seq "$T" --threads 10 --warmup 1 --repeat "$REPEAT"
    if [ "$b" = 1 ]; then
        run "pytorch.1.B$b" "$ONE" "$PY" experiments/pytorch_baseline.py forward --model llama --export "$OUT/pt" \
            --batch "$b" --seq "$T" --threads 1 --warmup 1 --repeat "$REPEAT"
    fi
    for r in f32 int8; do
        for layout in default plan; do
            plan=$P/$r-$layout.json
            [ "$layout" = plan ] && plan=$P/$r-plan-B$b.json
            [ -s "$plan" ] || { say "native.$r.$layout.B$b: no plan, left out"; continue; }
            run "native.$r.$layout.B$b" "$CORES" "$PRESK" bench "$FILE" --entry Llama.apply "${CONSTS[@]}" --dim B="$b" --dim T="$T" \
                --plan "@$plan" --plan-only --native-only --warmup 1 --repeat "$rep" --threads 10
        done
    done
    run "flops.B$b" "$CORES" "$PY" experiments/pytorch_baseline.py flops --model llama --export "$OUT/pt" --batch "$b" --seq "$T"
done

# --- The native kernels alone ------------------------------------------------
# D100's: `presk bench --calibrate` times every launch (its threads' start
# included, its host's buffers and copies not) and prints each kernel's
# median launch; the profile it writes, one per batch size in OUT/prof, is
# used by nothing else. Under the `f32` plan only: `--calibrate` runs a nest
# no plan states scalar, so under the default layouts it would time other
# kernels. After every timed row, so that it moves none of them.
mkdir -p "$OUT/prof"
for b in $BATCHES; do
    [ -s "$P/f32-plan-B$b.json" ] || continue
    run "kernels.f32.plan.B$b" "$CORES" "$PRESK" bench "$FILE" --entry Llama.apply "${CONSTS[@]}" --dim B="$b" --dim T="$T" \
        --plan "@$P/f32-plan-B$b.json" --threads 10 --repeat "$REPEAT" --calibrate "$OUT/prof/f32-plan-B$b.json"
done
# The host by phase (`--host-profile`, D95) under the `f32` plan at batch 1:
# what of a call is the kernels' and what the host's, measured in the call.
if [ -s "$P/f32-plan-B1.json" ] && [[ " $BATCHES " == *" 1 "* ]]; then
    run "hostprof.f32.plan.B1" "$CORES" "$PRESK" bench "$FILE" --entry Llama.apply "${CONSTS[@]}" --dim B=1 --dim T="$T" \
        --plan "@$P/f32-plan-B1.json" --threads 10 --warmup 1 --repeat 3 --host-profile
fi

# --- The report ------------------------------------------------------------
python3 - "$OUT" "$T" "$BATCHES" "$ONE" > "$OUT/report.md" <<'EOF'
import glob, os, re, sys
out, T, batches, one = sys.argv[1], int(sys.argv[2]), [int(b) for b in sys.argv[3].split()], sys.argv[4]
log = lambda k: f"{out}/log/{k}"
D, FF, V, L = 4096, 11008, 32000, 1

def text(path):
    try:
        return open(path).read()
    except FileNotFoundError:
        return None

def ok(k):
    t = text(log(k) + ".time")
    return t is not None and "Exit status: 0" in t

def wall(k):
    t = text(log(k) + ".time") or ""
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", t)
    return sum(float(x) * 60 ** i for i, x in enumerate(reversed(m.group(1).split(":")))) if m else None

def peak(k):
    t = text(log(k) + ".time") or ""
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    return int(m.group(1)) / 2**20 if m else None

def status(k):
    t = text(log(k) + ".time") or ""
    m = re.search(r"Exit status: (\d+)", t)
    if m:
        return m.group(1)
    m = re.search(r"Command terminated by signal (\d+)", t)
    return f"signal {m.group(1)}" if m else "not run"

def times(k):
    """(median, best) in ms: PyTorch's line, or the native plan row."""
    t = text(log(k) + ".log")
    if not ok(k) or t is None:
        return None
    m = re.search(r"median ([0-9.]+) ms, best ([0-9.]+) ms", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    rows = [l.split() for l in t.splitlines() if re.search(r" plan\s", l)]
    return (float(rows[-1][-3]), float(rows[-1][-2])) if rows else None

def hm(s):
    if s < 1:
        return "under 1 s"
    s = round(s)
    return f"{s // 3600} h {s % 3600 // 60:02d} min {s % 60:02d} s" if s >= 3600 else f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"

def gb(m):
    return f"{m:.1f} GB" if m >= 1 else f"{m * 1024:.0f} MB"

def thousands(x, digits=1):
    return f"{x:,.{digits}f}".replace(",", " ")

def flops(b):
    t = text(log(f"flops.B{b}") + ".log") or ""
    m = re.search(r"forward (\d+) \(", t)
    return int(m.group(1)) if m else None

def model_flops(b):
    """The contractions of the forward pass, a multiply and an add a term:
    the block's four D x D projections (G = 32, so the keys and values are
    D wide too) and SwiGLU's three D x FF products, attention's two T x T
    contractions computed in full, the head. Normalizations, softmax, SiLU
    and residual additions are not counted, on any side."""
    return b * T * (L * (8 * D * D + 6 * D * FF + 4 * T * D) + 2 * D * V)

machine = open(f"{out}/machine.txt").read().strip()
libs = open(f"{out}/torch-libs.txt").read().strip()
first = machine.splitlines()
date, rev, version = first[0].split()[1][:10], first[1].split()[1].rstrip(","), first[1].split(", ", 1)[1]
print(f"# The Llama 2 decoder block against PyTorch eager: {version}")
print()
print(f"One block of the Llama 2 decoder at the 7B configuration's width (`coverage/llama.psk`, `--const G=32 --const L=1`: D = {D}, 32 heads each its own key head, SwiGLU's {FF}, V = {V}), its forward pass `Llama.apply` at T = {T}, `f32`; {date}, presk `{rev}`, `experiments/pytorch_decoder.sh`. Two sides on the ten X925 cores pinned (`taskset -c 5-9,15-19`, ten threads): presk's native backend (`presk bench --native-only --plan-only`), under its default layouts and under `presk plan`'s plan with the frozen profile, in `f32` and with `int8` weights on every matrix; and PyTorch running the module `presk export --target pytorch --mode readable` writes, in eval mode under `torch.no_grad()`, with its own draw of the same distributions (every matrix normal(std = 0.02), every γ ones; seed 0) and token ids drawn by `torch.randint` (seed 0). Each time is the median of the calls after one warmup call, which compiles natively; the native process's exact draw of the parameters comes before and is not timed.")
print()
counted = {b: flops(b) for b in batches}
print("**Operations.** The forward pass's contractions, a multiply and an add a term: " + "; ".join(
    f"batch {b}: {model_flops(b) / 1e9:.2f} GFLOP by the compiler's count of them, {counted[b] / 1e9:.2f} by PyTorch's counter (`torch.utils.flop_counter`)" if counted[b] else f"batch {b}: {model_flops(b) / 1e9:.2f} GFLOP by the compiler's count"
    for b in batches) + ". The rates below divide PyTorch's count by the median, whatever a side executes.")

SIDES = (("native", "f32"), ("native, int8 every matrix", "int8"))
for b in batches:
    print(f"\n## The forward pass, B = {b}\n")
    base = times(f"pytorch.10.B{b}")
    fl = counted[b] or model_flops(b)
    print("| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for side, r in SIDES:
        for layout in ("default", "plan"):
            rows.append(("10 (ten X925, pinned)", side, layout, f"native.{r}.{layout}.B{b}"))
    rows.append(("10 (ten X925, pinned)", "PyTorch", "its own", f"pytorch.10.B{b}"))
    if b == 1:
        rows.append((f"1 (one X925, cpu {one}, pinned)", "PyTorch", "its own", f"pytorch.1.B{b}"))
    for th, side, layout, k in rows:
        t = times(k)
        if t is None:
            why = "no plan" if layout == "plan" and not ok(k.replace("native.", "plan.").replace(f".plan.B{b}", f".B{b}")) else f"not measured ({status(k)})"
            print(f"| {th} | {side} | {layout} | {why} | | | |")
            continue
        ratio = f"{t[0] / base[0]:.2f}" if base else ""
        print(f"| {th} | {side} | {layout} | {thousands(t[0])} | {thousands(t[1])} | {fl / 1e9 / (t[0] / 1e3):.1f} | {ratio} |")
    lg = re.search(r"pytorch logits: (.*)", text(log(f"pytorch.10.B{b}") + ".log") or "")
    if lg:
        print(f"\nPyTorch's logits on its last call, ten threads: {lg.group(1)}.")

print("\n## Where the native time goes, under the `f32` plan\n")
print("Every launch timed on its own (`presk bench --calibrate`, D100's: a launch's threads' start included, the host's buffers and copies not), each kernel's median launch; the kernels' time a call is each kernel's median launch times its launches. The largest kernels by their launch's time; a kernel is one shape of a nest, so `wk` and `wv` (their output axis `kvout`) are one kernel launched twice, and so are SwiGLU's `gate` and `up`; every other kernel here is launched once. GFLOP: a product's multiplies and adds, from its operands' shapes (a `dot`'s left operand's elements times its right operand's columns, an `ein`'s every axis once), twice its terms; the kernel's own operation count, which weighs index arithmetic too, is not used.")
hp = text(log("hostprof.f32.plan.B1") + ".log") or ""
hk = re.search(r"^kernels\s+([0-9.]+)", hp, re.M)
ht = re.search(r"^total\s+([0-9.]+)", hp, re.M)
hh = re.search(r"the host ([0-9.]+) ms", hp)
if ok("hostprof.f32.plan.B1") and hk and ht and hh:
    print()
    print(f"In the call itself (`presk bench --host-profile`, D95's, at batch 1 under the `f32` plan, three calls timed by phase): the kernels {thousands(float(hk.group(1)))} ms of {thousands(float(ht.group(1)))}, the host {float(hh.group(1)):.1f} ms. So what a call takes beyond its kernels timed alone is not the host's: inside the call a launch takes longer than the same launch timed on its own.")

def contraction_flops(sig):
    """Twice the terms of a `dot` or an `ein`, from the full signature's shapes; None for any other kernel.
    A `dot`: its left operand's elements times its right operand's columns. An `ein`: its output's
    elements times the axes of operand 0 its output does not keep (its two operands may name an axis
    alike that the output keeps twice, as the scores' `seq` and `seq2`)."""
    head, _, out = sig.partition(" -> ")
    shape = lambda t: [[a.strip() for a in o.split(",") if a.strip()] for o in re.findall(r"\b(?:f64|f32|bf16|i64)\[([^\]]*)\]", t)]
    ln = lambda a: int(a.rpartition(":")[2])
    ops, res = shape(head.split("(", 1)[-1]), shape(out)
    prod = lambda xs: __import__("math").prod(xs)
    if head.startswith("dot ") and len(ops) == 2:
        return 2 * prod(ln(a) for a in ops[0]) * ln(ops[1][-1])
    m = re.match(r"ein\[([^\]]*)\]", head)
    if m and len(ops) == 2 and res:
        keep = [x.strip() for x in m.group(1).split(",")]
        summed = [ln(a) for a in ops[0] if a.rpartition(":")[0].strip() not in keep]
        return 2 * prod(ln(a) for a in res[0]) * prod(summed)
    return None

for b in batches:
    k = f"kernels.f32.plan.B{b}"
    t = text(log(k) + ".log") or ""
    call = re.search(r"-- a call: (\d+) launches, the kernels' time ([0-9.]+) ms", t)
    prof = text(f"{out}/prof/f32-plan-B{b}.json")
    if not (ok(k) and call and prof):
        print(f"\n- batch {b}: not measured ({status(k)}).")
        continue
    import json
    ks = sorted(json.loads(prof)["kernels"], key=lambda x: -x["seconds"])
    row = times(f"native.f32.plan.B{b}")
    kern = float(call.group(2))
    print(f"\n### B = {b}\n")
    print(f"{call.group(1)} launches a call, the kernels' time {thousands(kern)} ms timed alone" + (f", the plan's row {thousands(row[0])} ms." if row else "."))
    print()
    print("| kernel | GFLOP a launch | ms a launch | share of the kernels' time | GFLOP/s |")
    print("|---|---|---|---|---|")
    for x in ks[:10]:
        sig = x["signature"].removeprefix("kernel ").split(" -> ")[0]
        sig = re.sub(r"%\d+ : ", "", sig)
        f = contraction_flops(x["signature"].removeprefix("kernel "))
        n = 2 if re.search(r"kvout: 4096\]\)$|\[4096, 11008\]\)$", sig) else 1
        share = f"{100 * n * x['seconds'] * 1e3 / kern:.1f} %" + (f" ({n} launches)" if n > 1 else "")
        print(f"| `{sig}` | {f / 1e9:.2f} | {thousands(x['seconds'] * 1e3)} | {share} | {f / 1e9 / x['seconds']:.1f} |" if f else f"| `{sig}` | — | {thousands(x['seconds'] * 1e3)} | {share} | — |")

print("\n## The plans\n")
for b in batches:
    for r in ("f32", "int8"):
        k = f"plan.{r}.B{b}"
        t = text(log(k) + ".log") or ""
        if ok(k):
            n = len(re.findall(r'"[^"]+/%[0-9]+"', text(f"{out}/plans/{r}-plan-B{b}.json") or ""))
            print(f"- `{r}`, batch {b}: a plan stating {n} nests, in {hm(wall(k))}.")
        else:
            lines = [l for l in t.splitlines() if l.strip()]
            print(f"- `{r}`, batch {b}: no plan ({status(k)}): " + (f"`{lines[0].strip()[:300]}`" if lines else "no output") + ".")

print("\n## The processes\n")
print("Every command, its exit status, its wall time and its peak memory (`/usr/bin/time -v`); the native rows' wall includes the exact draw of the parameters and the warmup call.")
print()
print("| command | exit | wall | peak |")
print("|---|---|---|---|")
# In the order the script runs them (OUT/order.txt), then any other.
order = [k for k in (text(f"{out}/order.txt") or "").split() if text(log(k) + ".cmd") is not None]
order += sorted(set(os.path.basename(f)[:-4] for f in glob.glob(f"{out}/log/*.cmd")) - set(order))
for k in order:
    w, m = wall(k), peak(k)
    print(f"| `{k}` | {status(k)} | {hm(w) if w is not None else ''} | {gb(m) if m is not None else ''} |")
print()
print("```")
for k in order:
    print(open(log(k) + ".cmd").read().strip())
print("```")

print("\n## The machine and PyTorch's libraries\n")
print("```")
print(machine)
print(libs)
print("```")
EOF
cat "$OUT/report.md"
