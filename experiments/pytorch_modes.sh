#!/bin/bash
# PyTorch compiled and in mixed precision, beside eager (D153): the PyTorch
# side of experiments/pytorch_baseline.sh (D87, D100, D127) in four modes,
# `eager`, `compile` (`torch.compile`, inductor, its default mode, on the
# CPU), `amp` (`torch.autocast("cpu", dtype=torch.bfloat16)` around the
# forward pass and the loss, the parameters and AdamW in float32) and
# `compile+amp`, on the model `presk export --target pytorch` writes, the
# export unchanged (experiments/pytorch_modes.py drives it). A measurement:
# it decides nothing of the compiler.
#
#   experiments/pytorch_modes.sh OUT
#
# 1. Setup (OUT/setup.done): the machine, torch's configuration and the
#    inductor's; the exports (the course GPT with --train, at its course
#    constants and at D = 64); the plans and the C programs T2 times
#    (`loss_grad` and `adamw.step` under the plans, ten threads), checked
#    against D127's (D127, below: their sources byte for byte, their
#    outputs); the native backend's f64 logits; dynamo's graphs and breaks.
# 2. The time (OUT/round1..ROUNDS, D126's protocol: every row of a round a
#    process of its own, the rounds one after the other, each after a quiet
#    machine): the course GPT at B = 16, T = 256, the forward pass and one
#    training step in the four modes, beside D127's own PyTorch rows run
#    again (`pytorch_baseline.py forward`, the exported `train.py`); on the
#    ten X925 cores pinned (CORES) and on twenty unpinned; and on ten, the
#    compiler's step under the plans (the C program and the native backend)
#    as the control of D127's rows. REPEAT calls or steps, the first apart.
#    Then (OUT/nodrop-round1..ROUNDS) the step in eager and compiled with its
#    dropout off, as the compiler's `loss_grad` + `adamw.step` draws no
#    masks: like for like with T2's C step, on ten threads pinned.
# 3. The identity: each mode's logits on the conformance inputs against the
#    native backend's f64 ones (`pytorch_baseline.py compare`); at D = 64 on
#    the 461 held-out windows, each mode's gap to the exact logits, its
#    flips and bound (`margins.py report`, beside D130's rows).
# 4. The quality at D = 64: the exported model trained 600 steps in each
#    mode from SEEDS (train.py's loop, batch and schedule, on the 90 % of
#    the curve), its validation loss and next-character accuracy in
#    float64 on the held-out windows, beside presk's all-f64 runs of the
#    curve (out/curve_train_f64, seeds 1 and 2) read by the same code.
# 5. Where a step's time goes (OUT/attribution): one step in eager and
#    compiled, with and without dropout, under torch.profiler; not timed.
# Writes OUT/report.md.
set -eu -o pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PRESK=${PRESK:-target/release/presk}
case $PRESK in /*) ;; *) PRESK=$ROOT/$PRESK ;; esac
PY=$ROOT/${PY:-.venv/bin/python}
PROFILE=${PROFILE:-$HOME/presk-data/ra4-model-0.8.0/profile.json}
CKPT=${CKPT:-out/course_probe_f32/final.psk.d}
case $CKPT in /*) ;; *) CKPT=$ROOT/$CKPT ;; esac
CORES=${CORES:-5-9,15-19}
C_WAIT=${C_WAIT:-GOMP_SPINCOUNT=8M}
REPEAT=${REPEAT:-7}
ROUNDS=${ROUNDS:-5}
STEPS=${STEPS:-600}
SEEDS=${SEEDS:-"0 1 2"}
D127=${D127:-out/retime-0.12.0/baseline-round1}
MARGINS=${MARGINS:-out/margins}
D130=${D130:-out/pytorch-gap/margins}
W64=${W64:-out/curve_train_f64/final.psk.d}
TRAIN=${TRAIN:-out/civil_train_T256.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
PRESK_ROWS=${PRESK_ROWS:-"f64 f32 bf16 d124-1e-3 d132-1e-3 logits-4.67e-4"}
OUT=${1:?usage: experiments/pytorch_modes.sh OUT}
FILE=corpus/nlp/05_gpt.psk
GRAD=experiments/gpt_grad.psk
DIMS=(--dim B=16 --dim T=256)
D64=(--const D=64 --const H=2 --const L=2)
read -ra THREADS <<< "${THREADS:-10 20}"
MODES=(eager compile amp compile+amp)
# Every compile in a process of its own, cold: no cache across processes.
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

for f in "$PRESK" "$PY" "$PROFILE" "$CKPT" "$W64" "$TRAIN" "$VAL" "$D127/results.txt" "$MARGINS/val.json" "$D130/decisions/pytorch-f32.txt"; do
    [ -e "$f" ] || { echo "pytorch_modes.sh: missing $f" >&2; exit 1; }
done
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
say() { echo "-- $(date +%T) $*"; }
pin() { if [ "$1" = 10 ]; then echo "taskset -c $CORES"; fi; }
quiet() { # wait for a one-minute load under 1.5, at most 20 minutes; log it
    local i l
    for i in $(seq 1 120); do
        l=$(cut -d' ' -f1 /proc/loadavg)
        python3 -c "import sys; sys.exit(0 if $l < 1.5 else 1)" && break
        sleep 10
    done
    {
        echo "$(date -Iseconds) $1: load $(cut -d' ' -f1-3 /proc/loadavg), waited $((10 * (i - 1))) s"
        ps -eo pid,pcpu,etime,user,comm --sort=-pcpu | sed -n 2,6p
    } >> "$OUT/driver.log"
}
PT=$OUT/pt
PT64=$OUT/pt64
C=$OUT/c
P=$OUT/plans
INPUTS=$C/fwd-plan/conformance/inputs.psk.d

# --- 1. Setup ----------------------------------------------------------------
if [ ! -f "$OUT/setup.done" ]; then
    say "the machine"
    {
        echo "date $(date -Iseconds)"
        echo "presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates experiments/pytorch_baseline.py experiments/gpt_grad.psk || echo -dirty), $("$PRESK" --version)"
        echo "kernel $(uname -sr), $(nproc) cores"
        lscpu | grep -E "^(Nom de modèle|Model name|Vitesse maximale|CPU max MHz)" | sed 's/  */ /g'
        echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
        gcc --version | sed -n 1p
        g++ --version | sed -n 1p
        echo "c_wait $C_WAIT"
    } > "$OUT/machine.txt"
    sha256sum "$CKPT" "$W64" "$TRAIN" "$VAL" "$PROFILE" corpus/nlp/data/civil_T256.psk.d > "$OUT/inputs.sha256"
    "$PY" -c "import torch; print('torch', torch.__version__); print(torch.__config__.show()); print(torch.__config__.parallel_info())" > "$OUT/torch-config.txt"
    "$PY" - > "$OUT/inductor-config.txt" <<'EOF'
import torch, torch._inductor.config as c, torch._dynamo.config as d
from torch._inductor import cpu_vec_isa
print("torch", torch.__version__)
print("the vector ISA inductor picks:", cpu_vec_isa.pick_vec_isa())
for k in ("cpp.cxx", "cpp.simdlen", "cpp.threads", "cpp.dynamic_threads", "cpp.enable_kernel_profile", "cpp_wrapper",
          "max_autotune", "freezing", "fallback_random", "compile_threads", "fx_graph_cache", "force_disable_caches",
          "epilogue_fusion", "aggressive_fusion", "pattern_matcher", "split_reductions", "triton.cudagraphs"):
    o = c
    for p in k.split("."):
        o = getattr(o, p)
    print(f"inductor.{k} = {o!r}")
for k in ("cache_size_limit", "automatic_dynamic_shapes", "assume_static_by_default"):
    print(f"dynamo.{k} = {getattr(d, k)!r}")
EOF
    cat "$OUT/machine.txt" "$OUT/inductor-config.txt"

    say "plans (the frozen profile), against D127's"
    mkdir -p "$P"
    "$PRESK" plan "$FILE" --entry GPT.apply "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/forward.json" > "$P/forward.log" 2>&1
    "$PRESK" plan "$GRAD" --entry loss_grad "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/grad.json" > "$P/grad.log" 2>&1
    "$PRESK" plan "$FILE" --entry adamw.step "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/adam.json" > "$P/adam.log" 2>&1
    for p in forward grad adam; do
        if cmp -s "$P/$p.json" "$D127/plans/$p.json"; then echo "plan $p: as D127's, byte for byte"; else echo "plan $p: DIFFERS from D127's"; fi
    done | tee -a "$OUT/identity.txt"

    say "exports"
    "$PRESK" export "$FILE" --target pytorch --mode readable --train -o "$PT" > "$PT.log" 2>&1
    "$PRESK" export "$FILE" --target pytorch --mode readable --train "${D64[@]}" -o "$PT64" > "$PT64.log" 2>&1
    for f in _05_gpt.py pskd.py train.py; do
        if diff -q "$PT/$f" "$D127/pt/$f" > /dev/null; then echo "pytorch export $f: as D127's"; else echo "pytorch export $f: differs from D127's"; diff "$D127/pt/$f" "$PT/$f" | head -20; fi
        if diff -q "$PT64/$f" out/pytorch-gap/export/$f > /dev/null; then echo "pytorch export at D = 64, $f: as D130's"; else echo "pytorch export at D = 64, $f: differs from D130's"; diff out/pytorch-gap/export/$f "$PT64/$f" | head -20; fi
    done | tee -a "$OUT/identity.txt"
    mkdir -p "$C"
    export_c() { # dir file entry plan
        local dir=$1 file=$2 entry=$3 plan=$4
        "$PRESK" export "$file" --target c --entry "$entry" "${DIMS[@]}" --plan "@$P/$plan.json" --weights "$CKPT" --threads 10 --no-exact -o "$dir" 2> "$dir.log"
        make -s -C "$dir" model
        (cd "$dir" && sh conformance/check.sh) > "$dir.check" 2>&1
        grep -q "C1 holds" "$dir.check" || { echo "pytorch_modes.sh: $dir: C1 fails" >&2; cat "$dir.check" >&2; exit 1; }
    }
    export_c "$C/fwd-plan" "$FILE" GPT.apply forward
    export_c "$C/grad-plan" "$GRAD" loss_grad grad
    export_c "$C/adam-plan" "$FILE" adamw.step adam
    mkdir -p "$C/f64"
    "$PRESK" export "$FILE" --target c --entry GPT.apply "${DIMS[@]}" --realize f64 --weights "$CKPT" --no-exact -o "$C/f64/fwd" 2> "$C/f64/fwd.log"
    # The C programs T2 times, against D127's: their sources but for the
    # lines naming presk's version, and the native backend's outputs, which
    # each export records, bit for bit.
    {
        for w in fwd-plan grad-plan adam-plan f64/fwd; do
            d=$D127/c/10/$w
            [ "$w" = f64/fwd ] && d=$D127/c/f64/fwd
            for f in model.c model.h kernels.c main.c presk_rt.c presk_rt.h presk_libm.c Makefile README.md conformance/check.sh; do
                [ -f "$C/$w/$f" ] || continue
                if cmp -s "$C/$w/$f" "$d/$f"; then s="byte for byte"
                else
                    n=$(diff "$d/$f" "$C/$w/$f" | grep -c '^[<>]' || true)
                    v=$(diff "$d/$f" "$C/$w/$f" | grep '^[<>]' | grep -vc -E 'presk (0\.1[2-5]\.0|[0-9a-f]{7})' || true)
                    s="differs in $n lines, $v of them not a version line"
                fi
                echo "C $w/$f: $s"
            done
            for f in inputs.psk.d native.psk.d; do
                cmp -s "$C/$w/conformance/$f" "$d/conformance/$f" && echo "C $w/conformance/$f: bit for bit" || echo "C $w/conformance/$f: DIFFERS"
            done
        done
    } | tee -a "$OUT/identity.txt"
    grep -h "^presk export" "$C"/*.log "$C"/f64/*.log > "$OUT/exports.txt" || true

    say "dynamo: graphs and graph breaks"
    for m in compile compile+amp; do
        taskset -c "$CORES" "$PY" experiments/pytorch_modes.py explain --export "$PT" --inputs "$INPUTS" --threads 10 --mode "$m" > "$OUT/explain-$m.txt" 2>&1
        grep "^explain" "$OUT/explain-$m.txt"
    done
    touch "$OUT/setup.done"
fi

# --- 2. The time ---------------------------------------------------------------
fwd_row() { # round threads mode
    local R=$1 th=$2 m=$3 key=fwd.pytorch.$3.$2 out=
    [ "${R##*round}" = 1 ] && out=(--out "$OUT/logits/$m-$th.npy") || out=()
    $(pin "$th") "$PY" experiments/pytorch_modes.py forward --export "$PT" --weights "$CKPT" --inputs "$INPUTS" --threads "$th" --mode "$m" --repeat "$REPEAT" "${out[@]}" > "$R/log/$key.log" 2>&1
    echo "$key $(sed -n 's/^pytorch_modes forward.* the first \([0-9.]*\) ms, median \([0-9.]*\) ms, best \([0-9.]*\) ms.*/\2 \3 \1/p' "$R/log/$key.log")" >> "$R/results.txt"
}
step_row() { # round threads mode [--no-dropout]
    local R=$1 th=$2 m=$3 key=step.pytorch.$3${4:+-nodrop}.$2
    $(pin "$th") "$PY" experiments/pytorch_modes.py step --export "$PT" --threads "$th" --mode "$m" --steps $((REPEAT + 1)) ${4:+"$4"} > "$R/log/$key.log" 2>&1
    echo "$key $(sed -n 's/^pytorch_modes step.* the first step \([0-9.]*\) ms, \([0-9.]*\) ms a step, median of [0-9]* (the first excluded), best \([0-9.]*\) ms.*/\2 \3 \1/p' "$R/log/$key.log")" >> "$R/results.txt"
}
c_run() { # round key dir: under OpenMP's default waiting (key) and under C_WAIT (key.w)
    local R=$1 key=$2 dir=$3 w=$CKPT k wait
    [ -f "$dir/conformance/weights.psk.d" ] && w=conformance/weights.psk.d
    for k in "$key" "$key.w"; do
        wait=
        [ "$k" = "$key" ] || wait=$C_WAIT
        # shellcheck disable=SC2086 # C_WAIT is words, VAR=VALUE each
        (cd "$dir" && env $wait OMP_NUM_THREADS=10 taskset -c "$CORES" ./model -r "$REPEAT" "$w" conformance/inputs.psk.d conformance/out.psk.d) > "$R/log/$k.log" 2>&1
        echo "$k $(sed -n 's/.*median \([0-9.]*\) ms, best \([0-9.]*\) ms.*/\1 \2/p' "$R/log/$k.log") -" >> "$R/results.txt"
    done
}
native_bench() { # round key -- presk bench args
    local R=$1 key=$2
    shift 2
    taskset -c "$CORES" "$PRESK" bench "$@" --native-only --plan-only --warmup 1 --repeat "$REPEAT" --threads 10 > "$R/log/$key.log" 2>&1
    echo "$key $(awk '/ plan / { print $(NF-2), $(NF-1) }' "$R/log/$key.log") -" >> "$R/results.txt"
}
mkdir -p "$OUT/logits"
for N in $(seq 1 "$ROUNDS"); do
    R=$OUT/round$N
    [ -f "$R/done" ] && continue
    rm -rf "$R"
    mkdir -p "$R/log"
    quiet "round$N"
    : > "$R/results.txt"
    echo "load $(cut -d' ' -f1-3 /proc/loadavg)" > "$R/load.txt"
    for th in "${THREADS[@]}"; do
        say "round $N: the forward pass, $th threads"
        key=fwd.pytorch.baseline.$th
        $(pin "$th") "$PY" experiments/pytorch_baseline.py forward --export "$PT" --weights "$CKPT" --inputs "$INPUTS" --repeat "$REPEAT" --threads "$th" > "$R/log/$key.log" 2>&1
        echo "$key $(sed -n 's/.*median \([0-9.]*\) ms, best \([0-9.]*\) ms.*/\1 \2/p' "$R/log/$key.log") -" >> "$R/results.txt"
        for m in "${MODES[@]}"; do fwd_row "$R" "$th" "$m"; done
        say "round $N: one training step, $th threads"
        key=step.pytorch.train_py.$th
        (cd "$PT" && $(pin "$th") "$PY" train.py --steps $((REPEAT + 1)) --log-every 1 --threads "$th" --ckpt "") > "$R/log/$key.log" 2>&1
        echo "$key $(sed -n 's/^train.py: \([0-9.]*\) ms a step.*/\1/p' "$R/log/$key.log") - -" >> "$R/results.txt"
        for m in "${MODES[@]}"; do step_row "$R" "$th" "$m"; done
        if [ "$th" = 10 ]; then
            say "round $N: the compiler's step under the plans, 10 threads (the control)"
            c_run "$R" grad.c.plan.10 "$C/grad-plan"
            c_run "$R" adam.c.plan.10 "$C/adam-plan"
            native_bench "$R" grad.native.plan.10 "$GRAD" --entry loss_grad "${DIMS[@]}" --plan "@$P/grad.json" --weights "$CKPT"
            native_bench "$R" adam.native.plan.10 "$FILE" --entry adamw.step "${DIMS[@]}" --plan "@$P/adam.json" --weights "$CKPT"
        fi
    done
    cat "$R/results.txt"
    touch "$R/done"
done

# The step without dropout, in rounds of its own after the five.
for N in $(seq 1 "$ROUNDS"); do
    R=$OUT/nodrop-round$N
    [ -f "$R/done" ] && continue
    rm -rf "$R"
    mkdir -p "$R/log"
    quiet "nodrop-round$N"
    : > "$R/results.txt"
    say "round $N: one training step without dropout, 10 threads"
    for m in eager compile; do step_row "$R" 10 "$m" --no-dropout; done
    cat "$R/results.txt"
    touch "$R/done"
done

# --- 3. The identity -------------------------------------------------------------
say "distances to the native backend's f64 logits"
args=()
for th in "${THREADS[@]}"; do
    for m in "${MODES[@]}"; do args+=("PyTorch $m, $th threads=$OUT/logits/$m-$th.npy"); done
done
"$PY" experiments/pytorch_baseline.py compare --reference "$C/f64/fwd/conformance/native.psk.d" \
    "native f32, and the C program=$C/fwd-plan/conformance/native.psk.d" "${args[@]}" > "$OUT/distances.md"
"$PY" - "$OUT/logits" "${THREADS[@]}" > "$OUT/modes-vs-eager.md" <<'EOF'
import sys, numpy as np
d, ths = sys.argv[1], sys.argv[2:]
print("| mode | threads | positions | max |Δ| to eager float32 | bit for bit |")
print("|---|---|---|---|---|")
for th in ths:
    e = np.load(f"{d}/eager-{th}.npy")
    for m in ("compile", "amp", "compile+amp"):
        y = np.load(f"{d}/{m}-{th}.npy")
        print(f"| {m} | {th} | {e.shape[0] * e.shape[1]} | {np.abs(y - e).max():.3e} | {'yes' if np.array_equal(y, e) else 'no'} |")
    if th != ths[0]:
        e0 = np.load(f"{d}/eager-{ths[0]}.npy")
        print(f"| eager | {th} against {ths[0]} | {e.shape[0] * e.shape[1]} | {np.abs(e - e0).max():.3e} | {'yes' if np.array_equal(e, e0) else 'no'} |")
EOF
cat "$OUT/distances.md" "$OUT/modes-vs-eager.md"

say "the gap to the exact logits at D = 64, 461 windows"
G=$OUT/gap
mkdir -p "$G/margins/native" "$G/margins/decisions"
M=$(cd "$MARGINS" && pwd)
ln -sfn "$M/exact" "$G/margins/exact"
ROWS=()
for r in $PRESK_ROWS; do
    [ -d "$M/native/$r" ] || { echo "  no row $r in $MARGINS: left out"; continue; }
    ln -sfn "$M/native/$r" "$G/margins/native/$r"
    ln -sf "$M/decisions/$r.txt" "$G/margins/decisions/$r.txt"
    ROWS+=("$r")
done
D130A=$(cd "$D130" && pwd)
ln -sfn "$D130A/native/pytorch-f32" "$G/margins/native/d130-pytorch-f32"
ln -sf "$D130A/decisions/pytorch-f32.txt" "$G/margins/decisions/d130-pytorch-f32.txt"
ROWS+=(d130-pytorch-f32)
for m in "${MODES[@]}"; do
    n=pytorch-${m/+/-}
    [ -f "$G/margins/decisions/$n.txt" ] || taskset -c "$CORES" "$PY" experiments/pytorch_modes.py gap --export "$PT64" --weights "$W64" --data "$VAL" \
        --exact "$G/margins/exact" --threads 10 --mode "$m" --name "$n" --out "$G/margins" > "$G/$n.log" 2>&1
    head -1 "$G/$n.log"; tail -1 "$G/$n.log"
    ROWS+=("$n")
done
# The eager row is D130's, bit for bit, or the comparison says otherwise.
{
    same=yes
    for f in "$G/margins/native/pytorch-eager"/*.tsv; do cmp -s "$f" "$D130A/native/pytorch-f32/$(basename "$f")" || same=no; done
    cmp -s "$G/margins/decisions/pytorch-eager.txt" "$D130A/decisions/pytorch-f32.txt" || same=no
    echo "the eager row at D = 64 against D130's PyTorch float32 row, every gap and decision: bit for bit $same"
} | tee -a "$OUT/identity.txt"
LABELS=""
GAPS=""
for r in "${ROWS[@]}"; do
    case $r in
        d130-pytorch-f32) l="PyTorch float32, ten threads (D130)" ;;
        pytorch-eager) l="PyTorch eager float32, ten threads" ;;
        pytorch-compile) l="PyTorch torch.compile float32, ten threads" ;;
        pytorch-amp) l="PyTorch AMP bfloat16, ten threads" ;;
        pytorch-compile-amp) l="PyTorch torch.compile + AMP bfloat16, ten threads" ;;
        d124-*) l="presk, solved at ${r#d124-} on the loss (D124)" ;;
        d132-*) l="presk, solved at ${r#d132-} on the loss, the trained envelope (D132)" ;;
        logits-*) l="presk, solved at ${r#logits-} on the logits" ;;
        *) l="presk, all-$r" ;;
    esac
    LABELS="${LABELS:+$LABELS|}$l"
    v=$(cat "$M/decisions/$r.loss" 2>/dev/null || true)
    GAPS="$GAPS${v:-—} "
done
python3 experiments/margins.py report "$G/margins" --val "$M/val.json" --names "${ROWS[*]}" --labels "$LABELS" \
    --loss-gaps "$GAPS" --out "$G/margins.md"
sed -n '/^| realization/,/^$/p' "$G/margins.md"

# --- 4. The quality at D = 64 ------------------------------------------------------
say "the quality at D = 64: $STEPS steps from scratch, seeds $SEEDS"
Q=$OUT/quality
mkdir -p "$Q"
models=()
for s in $SEEDS; do
    for m in "${MODES[@]}"; do
        n=${m/+/-}-seed$s
        [ -f "$Q/$n.pt" ] || taskset -c "$CORES" "$PY" experiments/pytorch_modes.py train --export "$PT64" --data "$TRAIN" --steps "$STEPS" \
            --seed "$s" --threads 10 --mode "$m" --out "$Q/$n.pt" > "$Q/$n.log" 2>&1
        tail -1 "$Q/$n.log"
        models+=("PyTorch $m, seed $s=$Q/$n.pt")
    done
done
presk=("presk all-f64 (out/curve_train_f64)=$W64")
for s in 1 2; do
    [ -f "out/curve_train_f64_seed$s/final.psk.d" ] && presk+=("presk all-f64, seed $s (out/curve_train_f64_seed$s)=out/curve_train_f64_seed$s/final.psk.d")
done
taskset -c "$CORES" "$PY" experiments/pytorch_modes.py evaluate --export "$PT64" --data "$VAL" --threads 10 "${presk[@]}" "${models[@]}" > "$Q/quality.md"
cat "$Q/quality.md"

# --- 5. Where a step's time goes ----------------------------------------------------
say "where a step's time goes: torch.profiler, 10 threads"
A=$OUT/attribution
mkdir -p "$A"
for m in eager compile; do
    for d in "" --no-dropout; do
        f=$A/profile-$m${d:+-nodrop}.txt
        [ -f "$f" ] || taskset -c "$CORES" "$PY" experiments/pytorch_modes.py profile --export "$PT" --threads 10 --mode "$m" $d --out "$f" > "${f%.txt}.log" 2>&1
        sed -n 1p "$f"
    done
done

# --- The report -------------------------------------------------------------------
"$PY" experiments/pytorch_modes.py report "$OUT" > "$OUT/report.md"
cat "$OUT/report.md"
