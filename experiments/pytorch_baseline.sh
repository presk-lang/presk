#!/bin/bash
# The PyTorch baseline at 0.9.0 (D87; the manual's `experiments` §3,
# "Three sides at 0.9.0"): the course GPT at B = 16, T = 256 in f32, timed
# on three sides — presk's native backend, the C program `presk export
# --target c` writes, and PyTorch running the model `presk export --target
# pytorch` writes — for two workloads, the forward pass and one training
# step. A measurement, not a statement: it decides nothing.
#
#   experiments/pytorch_baseline.sh OUT
#
# The forward pass is `GPT.apply` with the checkpoint CKPT and the token
# indices the C export synthesizes (its conformance inputs), the same on
# every side: native `presk bench --plan-only`, the C program's `-r` loop,
# PyTorch in eval mode under no_grad. The training step is `presk train`
# natively, the exported `train.py` for PyTorch, and for C the gradient
# (`loss_grad` of experiments/gpt_grad.psk, exported with CKPT) plus
# `adamw.step`, the step's two entries, timed apart and added. Since 0.10.0
# step 6 the native backend's two entries are timed the same way, by `presk
# bench`, neither side drawing dropout masks: T4's step, like for like
# (plan 0.10.0 §3.5), `presk train`'s step reported beside it.
#
# Layouts: the default, and plans `presk plan` gives with the frozen profile
# (PROFILE): for the forward pass, GPT.apply's; for the step, loss_grad's and
# adamw.step's, merged. Since 0.10.0 step 1 the derivatives of `dot` are
# contraction nests of the layout space, and the gradient's plan states
# their layouts (D89); under the default layout they are 0.9.0's kernels.
#
# Threads, the same on every side, two rows: the ten X925 cores (CORES)
# pinned with taskset, ten threads, which is what the plans divide among;
# and every core unpinned, twenty threads (the plans' launches still on ten
# workers, the other nests on twenty; PyTorch on twenty, as on 23 September).
#
# Timing: REPEAT calls or steps, the median and the best; the native
# backend's first call compiles, so it is left out (`--warmup 1`, and `presk
# train` leaves out its first step); PyTorch's first call or step too (its
# first step is left out by train.py); the C program compiles nothing and
# counts every call. A quiet machine: nothing beside it.
#
# The C programs are built by their Makefiles, `-O2` (CFLAGS empty), or with
# CFLAGS in place of the Makefile's (`-O3 -mcpu=native`, plan 0.10.0 step 7);
# the flags bit identity rests on (STRICT: ISO C11, no contraction) stay, and
# every program is checked against the native backend's outputs (C1).
#
# Their OpenMP workers' waiting (plan 0.10.0 step 7's follow-up, D97's
# addendum): every C program is timed twice, under the OpenMP runtime's
# default, by which an idle worker sleeps after about 0.1 ms (libgomp's
# 300 000 spins on the X925 cores), and under C_WAIT, the environment that
# makes it wait as the native backend's pool does, spinning about 2 ms
# before it sleeps (`GOMP_SPINCOUNT=8M`, measured on these cores), its
# rows keyed `….w`. T4 compares the native backend with the second, the
# two sides waiting alike; T2 compares PyTorch, which runs at its own
# runtime's default, with the first. Where the C program's time goes is
# timed under C_WAIT.
#
# Besides the times: each side's distance to the native backend's f64
# logits; the model's floating-point operations, counted, and PyTorch's
# counter's; the model's three linear products timed alone on each side;
# and where the C program's time goes, kernel by kernel (kernel_time.c).
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
CFLAGS=${CFLAGS:-}
C_WAIT=${C_WAIT:-GOMP_SPINCOUNT=8M}
REPEAT=${REPEAT:-7}
OUT=${1:?usage: experiments/pytorch_baseline.sh OUT}
FILE=corpus/nlp/05_gpt.psk
GRAD=experiments/gpt_grad.psk
DATA=corpus/nlp/data/civil_T256.psk.d
DIMS=(--dim B=16 --dim T=256)
THREADS=(10 20)

for f in "$PRESK" "$PY" "$PROFILE" "$CKPT" "$DATA"; do
    [ -e "$f" ] || { echo "pytorch_baseline.sh: missing $f (the data: make experiment-gpt; the checkpoint: CKPT)" >&2; exit 1; }
done
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
RES=$OUT/results.txt
: > "$RES"
say() { echo "-- $*"; }
pin() { # threads -> the prefix that pins a run of that many threads
    if [ "$1" = 10 ]; then echo "taskset -c $CORES"; fi
}

# --- The machine -----------------------------------------------------------
say "the machine"
{
    echo "date $(date -Iseconds)"
    echo "presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates experiments || echo -dirty), $("$PRESK" --version)"
    echo "kernel $(uname -sr), $(nproc) cores"
    lscpu | grep -E "^(Nom de modèle|Model name|Vitesse maximale|CPU max MHz)" | sed 's/  */ /g'
    echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
    gcc --version | sed -n 1p
    echo "cflags ${CFLAGS:--O2 -Wall -Wextra}"
    echo "c_wait $C_WAIT"
} > "$OUT/machine.txt"
"$PY" -c "import torch; print('torch', torch.__version__); print(torch.__config__.show()); print(torch.__config__.parallel_info())" > "$OUT/torch-config.txt"
# The libraries the wheel carries: its OpenBLAS (asked for its configuration)
# and the Arm Compute Library under oneDNN (its version string).
TORCH_LIB=$("$PY" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
{
    "$PY" -c "import ctypes, sys; l = ctypes.CDLL(sys.argv[1]); l.openblas_get_config.restype = ctypes.c_char_p; print(l.openblas_get_config().decode())" "$TORCH_LIB/libopenblas.so.0"
    grep -a -m1 -o "arm_compute_version=v[0-9.]*" "$TORCH_LIB/libarm_compute.so"
} > "$OUT/torch-libs.txt"
cat "$OUT/machine.txt" "$OUT/torch-libs.txt"

# --- Plans -----------------------------------------------------------------
say "plans (the frozen profile)"
P=$OUT/plans
mkdir -p "$P"
echo '{"realization": "f32", "layout": {}}' > "$P/default.json"
"$PRESK" plan "$FILE" --entry GPT.apply "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/forward.json" > "$P/forward.log" 2>&1
"$PRESK" plan "$GRAD" --entry loss_grad "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/grad.json" > "$P/grad.log" 2>&1
"$PRESK" plan "$FILE" --entry adamw.step "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/adam.json" > "$P/adam.log" 2>&1
# The step's plan: the gradient's layouts and the optimizer's, whose nests are
# named apart; `presk train` runs both, and names its functions as the
# wrapper does (`loss.fwd`, `GPT.apply.bwd`, ..., since the gradient keeps
# its residuals, 0.10.0 step 4). The linear products' plan: the
# forward plan's layouts of Linear.apply, the product and its bias.
python3 - "$P" <<'EOF'
import json, sys
p = sys.argv[1]
g, a, f = (json.load(open(f"{p}/{n}.json")) for n in ("grad", "adam", "forward"))
assert not set(g["layout"]) & set(a["layout"])
step = dict(g, layout={**g["layout"], **a["layout"]})
json.dump(step, open(f"{p}/step.json", "w"), indent=2)
lin = dict(f, layout={k: v for k, v in f["layout"].items() if k.startswith("Linear.apply/")})
json.dump(lin, open(f"{p}/linear.json", "w"), indent=2)
EOF

# --- Exports ---------------------------------------------------------------
# Each C program is recorded on the native backend's run with the checkpoint,
# built (GCC, -O2, OpenMP) and checked against the native backend's outputs
# bit for bit (C1). The default layout's workers are the thread count's, so
# every program is exported once per row.
say "exports"
"$PRESK" export "$FILE" --target pytorch --mode readable --train -o "$OUT/pt" > "$OUT/pt.log" 2>&1
C=$OUT/c
rm -rf "$C"
export_c() { # dir file entry threads layout weights [extra args]; weights `-`: synthesized
    local dir=$1 file=$2 entry=$3 th=$4 layout=$5 weights=$6
    shift 6
    local how=(--realize f32)
    [ "$layout" = plan ] && how=(--plan "@$P/$(plan_of "$entry").json")
    [ "$weights" = - ] || how+=(--weights "$weights")
    "$PRESK" export "$file" --target c --entry "$entry" "$@" "${how[@]}" --threads "$th" --no-exact -o "$dir" 2> "$dir.log"
    make -s -C "$dir" ${CFLAGS:+"CFLAGS=$CFLAGS"} model
    (cd "$dir" && sh conformance/check.sh) > "$dir.check" 2>&1
    grep -q "C1 holds" "$dir.check" || { echo "pytorch_baseline.sh: $dir: C1 fails" >&2; cat "$dir.check" >&2; exit 1; }
}
plan_of() {
    case $1 in GPT.apply) echo forward ;; loss_grad) echo grad ;; adamw.step) echo adam ;; *) echo linear ;; esac
}
for th in "${THREADS[@]}"; do
    mkdir -p "$C/$th"
    for layout in default plan; do
        export_c "$C/$th/fwd-$layout" "$FILE" GPT.apply "$th" "$layout" "$CKPT" "${DIMS[@]}"
        export_c "$C/$th/grad-$layout" "$GRAD" loss_grad "$th" "$layout" "$CKPT" "${DIMS[@]}"
        export_c "$C/$th/adam-$layout" "$FILE" adamw.step "$th" "$layout" "$CKPT" "${DIMS[@]}"
    done
done
PRODUCTS=("384 384" "384 1536" "1536 384")
for io in "${PRODUCTS[@]}"; do
    read -r i o <<< "$io"
    export_c "$C/10/lin-$i-$o" "$FILE" "Linear{$i,$o}.apply" 10 plan - "${DIMS[@]}" --dim batch=16 --dim seq=256
done
# The native backend's f64 logits, the reference of the distances: the
# export records the native run, whose outputs it writes (the exact
# reference would take hours at this size).
mkdir -p "$C/f64"
"$PRESK" export "$FILE" --target c --entry GPT.apply "${DIMS[@]}" --realize f64 --weights "$CKPT" --no-exact -o "$C/f64/fwd" 2> "$C/f64/fwd.log"
grep -h "^presk export" "$C"/*/*.log > "$OUT/exports.txt"
# Every side's forward pass reads these token indices: the C programs' and
# `presk bench`'s are synthesized from the same seed, PyTorch reads the file.
INPUTS=$C/10/fwd-default/conformance/inputs.psk.d
for th in "${THREADS[@]}"; do
    for layout in default plan; do
        cmp "$C/$th/fwd-$layout/conformance/inputs.psk.d" "$INPUTS"
        cmp "$C/$th/grad-$layout/conformance/inputs.psk.d" "$C/10/grad-default/conformance/inputs.psk.d"
    done
done

# --- Timing ----------------------------------------------------------------
native_bench() { # key threads -- presk bench args: "median best" in ms
    local key=$1 th=$2
    shift 2
    $(pin "$th") "$PRESK" bench "$@" --native-only --plan-only --warmup 1 --repeat "$REPEAT" --threads "$th" > "$OUT/log/$key.log" 2>&1
    echo "$key $(awk '/ plan / { print $(NF-2), $(NF-1) }' "$OUT/log/$key.log")" >> "$RES"
}
c_run() { # key threads dir: the program's weights are CKPT, or those its export synthesized;
    # timed under OpenMP's default waiting (key) and under C_WAIT (key.w)
    local key=$1 th=$2 dir=$3 w=$CKPT k wait
    [ -f "$dir/conformance/weights.psk.d" ] && w=conformance/weights.psk.d
    for k in "$key" "$key.w"; do
        wait=
        [ "$k" = "$key" ] || wait=$C_WAIT
        # shellcheck disable=SC2086 # C_WAIT is words, VAR=VALUE each
        (cd "$dir" && env $wait OMP_NUM_THREADS=$th $(pin "$th") ./model -r "$REPEAT" "$w" conformance/inputs.psk.d conformance/out.psk.d) > "$OUT/log/$k.log" 2>&1
        echo "$k $(sed -n 's/.*median \([0-9.]*\) ms, best \([0-9.]*\) ms.*/\1 \2/p' "$OUT/log/$k.log")" >> "$RES"
    done
}
py_run() { # key threads -- pytorch_baseline.py args
    local key=$1 th=$2
    shift 2
    $(pin "$th") "$PY" experiments/pytorch_baseline.py "$@" --threads "$th" > "$OUT/log/$key.log" 2>&1
    echo "$key $(sed -n 's/.*median \([0-9.]*\) ms, best \([0-9.]*\) ms.*/\1 \2/p' "$OUT/log/$key.log")" >> "$RES"
}
mkdir -p "$OUT/log" "$OUT/train"
for th in "${THREADS[@]}"; do
    say "forward pass, $th threads"
    for layout in default plan; do
        plan=$P/$([ "$layout" = plan ] && echo forward || echo default).json
        native_bench "fwd.native.$layout.$th" "$th" "$FILE" --entry GPT.apply "${DIMS[@]}" --plan "@$plan" --weights "$CKPT"
        c_run "fwd.c.$layout.$th" "$th" "$C/$th/fwd-$layout"
    done
    py_run "fwd.pytorch.$th" "$th" forward --export "$OUT/pt" --weights "$CKPT" --inputs "$INPUTS" --repeat "$REPEAT" --out "$OUT/pytorch-$th.npy"
    say "training step, $th threads"
    (cd "$OUT/pt" && $(pin "$th") "$PY" train.py --steps $((REPEAT + 1)) --log-every 1 --threads "$th" --ckpt "$OUT/train/pytorch-$th") > "$OUT/log/step.pytorch.$th.log" 2>&1
    echo "step.pytorch.$th $(sed -n 's/^train.py: \([0-9.]*\) ms a step.*/\1/p' "$OUT/log/step.pytorch.$th.log") -" >> "$RES"
    for layout in default plan; do
        c_run "grad.c.$layout.$th" "$th" "$C/$th/grad-$layout"
        c_run "adam.c.$layout.$th" "$th" "$C/$th/adam-$layout"
    done
    # The native step like for like (T4): the C programs' two entries, timed
    # by `presk bench` under the same plans, on the same synthesized inputs.
    for layout in default plan; do
        g=default a=default
        [ "$layout" = plan ] && g=grad a=adam
        native_bench "grad.native.$layout.$th" "$th" "$GRAD" --entry loss_grad "${DIMS[@]}" --plan "@$P/$g.json" --weights "$CKPT"
        native_bench "adam.native.$layout.$th" "$th" "$FILE" --entry adamw.step "${DIMS[@]}" --plan "@$P/$a.json" --weights "$CKPT"
    done
    for layout in default plan; do
        how=(--realize f32)
        [ "$layout" = plan ] && how=(--plan "@$P/step.json")
        $(pin "$th") "$PRESK" train "$FILE" --steps $((REPEAT + 1)) --log-every 1 --backend native "${how[@]}" --threads "$th" --ckpt "$OUT/train/native-$layout-$th" > "$OUT/log/step.native.$layout.$th.log" 2>&1
        echo "step.native.$layout.$th $(sed -n 's/^presk train: \([0-9.]*\) ms a step.*/\1/p' "$OUT/log/step.native.$layout.$th.log") -" >> "$RES"
    done
done
say "the linear products alone, 10 threads"
for io in "${PRODUCTS[@]}"; do
    read -r i o <<< "$io"
    native_bench "lin.native.$i.$o" 10 "$FILE" --entry "Linear{$i,$o}.apply" "${DIMS[@]}" --dim batch=16 --dim seq=256 --plan "@$P/linear.json"
    c_run "lin.c.$i.$o" 10 "$C/10/lin-$i-$o"
    py_run "lin.pytorch.$i.$o" 10 product --in "$i" --out "$o" --repeat "$REPEAT"
done

# --- Where the C program's time goes ---------------------------------------
# A copy of model.c with every launch timed on the thread that makes it (a
# kernel's call, or the OpenMP region that divides it), built with
# kernel_time.c's clock; the kernels are untouched and the outputs checked
# unchanged. Pinned on ten threads, four calls, the first left out, the
# workers waiting under C_WAIT, so that a region's time is its kernel's and
# not a sleeping worker's wake.
say "the C program kernel by kernel, 10 threads"
for w in fwd-default fwd-plan grad-default grad-plan; do
    d=$C/10/$w
    python3 experiments/kernel_time.py instrument "$d/model.c" > "$d/model_timed.c"
    # shellcheck disable=SC2086 # the flags are words
    (cd "$d" && gcc -std=c11 -ffp-contract=off ${CFLAGS:--O2 -Wall -Wextra} -fopenmp -o model_timed main.c model_timed.c presk_rt.c presk_libm.c "$ROOT/experiments/kernel_time.c" -lm \
        && env $C_WAIT KERNEL_TIME=kernel_time.txt OMP_NUM_THREADS=10 taskset -c "$CORES" ./model_timed -r 4 "$CKPT" conformance/inputs.psk.d conformance/out_timed.psk.d > /dev/null 2> timed.log \
        && cmp -s conformance/out_timed.psk.d conformance/native.psk.d)
    python3 experiments/kernel_time.py report "$d" "$d/kernel_time.txt" 4 > "$OUT/kernels-$w.md"
    echo "kern.c.$w $(python3 experiments/kernel_time.py report "$d" "$d/kernel_time.txt" 4 --totals | awk '{ print 1e3 * $2 }') $(sed -n 's/.*median \([0-9.]*\) ms.*/\1/p' "$d/timed.log")" >> "$RES"
done
# The native backend's kernels alone: `presk bench --calibrate` times every
# launch (its thread spawns included, its host's buffers and copies not) and
# prints each kernel's median launch times its launches a call; the profile
# it writes is not used. Under the plans only: `--calibrate` runs a nest the
# plan does not state scalar, where the native backend gives it its natural
# lanes, so under the default layout it would time other kernels. The
# forward plan leaves only the host's small kernels unstated.
say "the native kernels alone, 10 threads"
mkdir -p "$OUT/prof"
native_kernels() { # key -- presk bench args
    local key=$1
    shift
    taskset -c "$CORES" "$PRESK" bench "$@" --threads 10 --repeat "$REPEAT" --calibrate "$OUT/prof/$key.json" > "$OUT/log/$key.log" 2>&1
    echo "$key $(sed -n "s/^-- a call: .* the kernels' time \([0-9.]*\) ms.*/\1/p" "$OUT/log/$key.log") -" >> "$RES"
}
native_kernels "kern.native.fwd-plan" "$FILE" --entry GPT.apply "${DIMS[@]}" --plan "@$P/forward.json" --weights "$CKPT"
for io in "${PRODUCTS[@]}"; do
    read -r i o <<< "$io"
    native_kernels "kern.native.lin.$i.$o" "$FILE" --entry "Linear{$i,$o}.apply" "${DIMS[@]}" --dim batch=16 --dim seq=256 --plan "@$P/linear.json"
done

# --- Distances and counts ---------------------------------------------------
say "distances to the native backend's f64 logits"
"$PY" experiments/pytorch_baseline.py forward --export "$OUT/pt" --weights "$CKPT" --inputs "$INPUTS" --threads 10 --dtype float64 --repeat 1 --out "$OUT/pytorch-f64.npy" > "$OUT/log/pytorch-f64.log" 2>&1
"$PY" experiments/pytorch_baseline.py compare --reference "$C/f64/fwd/conformance/native.psk.d" \
    "PyTorch float64 (a check of the comparison)=$OUT/pytorch-f64.npy" \
    "native f32, and the C program=$C/10/fwd-default/conformance/native.psk.d" \
    "PyTorch float32=$OUT/pytorch-10.npy" > "$OUT/distances.md"
{
    same=yes
    for th in "${THREADS[@]}"; do
        for layout in default plan; do
            cmp -s "$C/$th/fwd-$layout/conformance/native.psk.d" "$C/10/fwd-default/conformance/native.psk.d" || same=no
            cmp -s "$C/$th/grad-$layout/conformance/native.psk.d" "$C/10/grad-default/conformance/native.psk.d" || same=no
        done
    done
    echo "layouts.same $same -"
    cmp -s "$OUT/train/native-default-10/final.psk.d" "$OUT/train/native-plan-10/final.psk.d" && echo "train.same yes -" || echo "train.same no -"
} >> "$RES"
"$PY" experiments/pytorch_baseline.py flops --export "$OUT/pt" --inputs "$INPUTS" > "$OUT/flops.txt"
# What PyTorch's forward pass hands to oneDNN, primitive by primitive (the
# products it does not hand over are OpenBLAS's).
ONEDNN_VERBOSE=1 "$PY" experiments/pytorch_baseline.py forward --export "$OUT/pt" --weights "$CKPT" --inputs "$INPUTS" --threads 10 --warmup 0 --repeat 1 2>&1 \
    | awk -F, '$3 == "primitive" && $4 == "exec" { print $6, $7, $(NF-1) }' | sort | uniq -c > "$OUT/onednn.txt"

"$PY" experiments/pytorch_baseline.py report "$OUT" > "$OUT/report.md"
cat "$OUT/report.md"
