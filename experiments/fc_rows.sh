#!/bin/bash
# Plan 0.18.2 step 5 (D219): the version's rows, alone on falicon.
#
# FC4  the course GPT's AdamW step (`adamw.step` of corpus/nlp/05_gpt.psk at
#      its own constants, D = 384, L = 6), each value in binary16 where D208's
#      range rules offer it and in f32 elsewhere, against all-f32, on the
#      LLVM backend under the default layouts: at most 0.85x. The rules read
#      D178's envelope of the training step (out/step5-0.17.0/step_env.json,
#      at D = 64, the only envelope of a step of record); the optimizer's
#      offers are its constants' and its divisors', which do not depend on
#      the width. Reported beside it: the native backend and the exported C.
# FC5  the forward pass (`GPT.apply`, B = 16, T = 256), every value in
#      binary16, the default layouts, against all-f32, on the LLVM backend:
#      at most 0.85x. Reported: the same under the profile's plan (f32's by
#      the frozen profile; binary16's by a copy of it with binary16's layout
#      groups calibrated as `make calibrate-ra4` calibrates f32's), on the
#      native backend, and in e8m15.
# FC6  the guard's cost: FC5's forward pass in binary16 in the exported C,
#      built with and without -DPRESK_NO_GUARD (GCC 16.2, the export's flags).
# FC7  two certified realizations, each timed on the LLVM backend and in the
#      exported C (GCC 16.2) against all-f32 and D170's realization of the
#      same entry (`logits` of experiments/margins.psk, B = 16, T = 256, the
#      default layouts), with the solver's bytes: FC3's at D = 64 (D209's
#      MT6), and the solve at the course constants at epsilon*(1e-3) =
#      4.96e-4 with binary16 and e8m15 on the menu, on D170's envelope and
#      weights (written by the phase `solve`, measured natively).
#
# Every timed row: ten X925 (taskset -c 5-9,15-19); D126's protocol, PROCS
# processes per row (5), the rows alternated round by round, each process a
# warm-up call and REPEAT (7) timed ones, its median and fastest; per row
# the median of the medians and the fastest. `presk bench --native-only
# --plan-only` for the two backends, the C program's own `-r` loop under
# OMP_NUM_THREADS=10 and GOMP_SPINCOUNT=8M (T4's row of record) for C.
# Inputs: the checkpoint of record of these rows (CKPT, the course probe's),
# the other arguments drawn from the seed, the same on every side.
#
#   experiments/fc_rows.sh [solve|calib|prep|time|report|all|fc4]
#
# `solve` is not a timing: it runs on the A725 cores (0-4, 10-14) and may
# run beside anything but `calib` and `time`, which must run alone. Outputs
# present are reused, as every experiment script here does. `fc4` (plan
# 0.18.2 step 6, D220) takes FC4's rows again into its own OUT, alone: its
# realization, plans and C programs, the realization's non-finite values
# counted, and its six rows by the same protocol.
set -u -o pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PRESK=${PRESK:-target/release/presk}
case $PRESK in /*) ;; *) PRESK=$ROOT/$PRESK ;; esac
PY=${PY:-python3}
OUT=${OUT:-out/fc-rows-0.18.2}
case $OUT in /*) ;; *) OUT=$ROOT/$OUT ;; esac
PROCS=${PROCS:-5}
REPEAT=${REPEAT:-7}
CORES=${CORES:-5-9,15-19}
SIDE=${SIDE:-0-4,10-14}
G16=$ROOT/.gcc/bin/gcc
G16LIB=$(dirname "$("$G16" -print-file-name=libgomp.so)")
FROZEN=${FROZEN:-$HOME/presk-data/ra4-model-0.8.0/profile.json}
CKPT=${CKPT:-$ROOT/out/course_probe_f32/final.psk.d}
GPT=examples/corpus/nlp/05_gpt.psk
MARGINS=examples/experiments/margins.psk
STEP_ENV=out/step5-0.17.0/step_env.json
W64=$ROOT/out/curve_train_f64/final.psk.d
WCOURSE=$ROOT/out/course_train_f64/final.psk.d
FC3=out/fc3-0.18.2/solve/mt6.json
D170_64=out/v6-0.16.1/solved.json
D170_COURSE=out/margins-course-0.16.1/logits/solved.json
COURSE_ENV=out/margins-course-0.16.1/logits/env_logits.json
MENU=f64,f32,bf16,f16,e8m15
CFLAGS_C="-std=c11 -ffp-contract=off -O2 -Wall -Wextra -fopenmp"
FWD=("$GPT" --entry GPT.apply --dim B=16 --dim T=256 --weights "$CKPT")
ADAM=("$GPT" --entry adamw.step --dim B=16 --dim T=256 --weights "$CKPT")
L64=("$MARGINS" --entry logits --const D=64 --const H=2 --const L=2 --dim B=16 --dim T=256 --weights "$W64")
LCOURSE=("$MARGINS" --entry logits --const D=384 --const H=6 --const L=6 --dim B=16 --dim T=256 --weights "$WCOURSE")
P=$OUT/plans
C=$OUT/c
mkdir -p "$OUT/log" "$OUT/times" "$P" "$C" "$OUT/fc4" "$OUT/fc7/course" "$OUT/calib"
say() { echo "== $(date -Is) $*"; }
scoped() { systemd-run --user --scope -q -p MemoryMax="${MEM:-32G}" -p MemorySwapMax=0 "$@"; }
for f in "$PRESK" "$G16" "$FROZEN" "$CKPT" "$W64" "$WCOURSE" "$STEP_ENV" "$FC3" "$D170_64" "$D170_COURSE" "$COURSE_ENV"; do
    [ -e "$f" ] || { echo "fc_rows.sh: missing $f" >&2; exit 1; }
done

# --- solve: FC7's second realization (not a timing) --------------------------
solve() {
    local o=$OUT/fc7/course
    if [ -s "$o/solved.json" ]; then say "the course solve: reused"; return; fi
    say "the course solve at 4.96e-4 with $MENU, measured natively"
    scoped taskset -c "$SIDE" /usr/bin/time -v -o "$o/solve.time" "$PRESK" solve "$MARGINS" --precision "$COURSE_ENV" --budget-abs 4.96e-4 \
        --weights "$WCOURSE" --sites 60 --menu "$MENU" --measure native --out "$o/solved.json" > "$o/solve.txt" 2> "$o/solve.err"
}

# --- calib: binary16's layout groups, for FC5's plan row (timed, alone) ------
# `make calibrate-ra4`'s programs, sizes and model, in binary16, into a copy
# of the frozen profile: its samples kept, the model refitted with binary16's.
# Only binary16's plan is planned from it; f32's is the frozen profile's.
calib() {
    local p=$OUT/calib/profile-f16.json
    if [ -s "$OUT/calib/done" ]; then say "the binary16 calibration: reused"; return; fi
    cp "$FROZEN" "$p"
    chmod u+w "$p"
    local model=(--load-term group --combine pnorm --rules lines,overhead) start
    start=$(date +%s)
    for n in 256 512 1024 2048; do
        w=1
        [ "$n" = 2048 ] && w=2
        say "calibration: GEMM N=$n in f16"
        taskset -c "$CORES" "$PRESK" bench examples/experiments/gemm.psk --entry gemm --dim N=$n --realize f16 --calibrate "$p" --space calib "${model[@]}" \
            --min-workers $w --repeat 5 > "$OUT/calib/gemm-$n.log" 2>&1 || { echo "calibration failed: $OUT/calib/gemm-$n.log"; exit 1; }
    done
    for s in I=16,K=4096,J=4096 I=4096,K=4096,J=16; do
        say "calibration: product $s in f16"
        # shellcheck disable=SC2046 # the dims are words
        taskset -c "$CORES" "$PRESK" bench examples/experiments/matmul.psk --entry matmul $(echo $s | tr ',' ' ' | sed 's/[^ ]*/--dim &/g') --realize f16 \
            --calibrate "$p" --space calib "${model[@]}" --repeat 5 > "$OUT/calib/product-$s.log" 2>&1 || { echo "calibration failed: $OUT/calib/product-$s.log"; exit 1; }
    done
    echo "$(( $(date +%s) - start )) s" > "$OUT/calib/done"
}

# --- prep: realizations, plans, exports, builds (not timed) ------------------
wrap() { # realization.json plan.json: a plan of the default layouts
    $PY -c "import json, sys; json.dump({'realization': json.load(open(sys.argv[1])), 'layout': {}}, open(sys.argv[2], 'w'), indent=1)" "$1" "$2"
}
c_export() { # name realization-spec args...: export, build with GCC 16.2, C1
    local name=$1 spec=$2 d=$C/$1
    shift 2
    if [ ! -f "$d/conformance/native.psk.d" ]; then
        "$PRESK" export "$@" --target c --realize "$spec" --threads 10 --no-exact -o "$d" > "$C/$name.log" 2>&1 || { echo "export $name failed: $C/$name.log"; exit 1; }
    fi
    c_build "$name" model ""
}
c_build() { # name binary extra-flags
    local d=$C/$1 a
    # shellcheck disable=SC2086 # the flags are words
    (cd "$d" && "$G16" $CFLAGS_C $3 -o "$2" main.c model.c presk_rt.c presk_libm.c -lm) || { echo "build $1/$2 failed"; exit 1; }
    a=$(c_args "$d" "conformance/c1-$2.psk.d")
    (cd "$d" && eval "LD_LIBRARY_PATH=$G16LIB OMP_NUM_THREADS=10 ./$2 $a" && cmp -s "conformance/c1-$2.psk.d" conformance/native.psk.d) \
        || { echo "$1/$2: C1 fails"; exit 1; }
    echo "$1/$2: built by GCC 16.2, C1 holds"
}
c_args() { # dir out: the program's arguments as its check.sh gives them
    grep '^\./model ' "$1/conformance/check.sh" | sed "s#^\./model ##; s#conformance/out.psk.d#$2#"
}
fc4_realization() {
    say "FC4's realization: D208's range rules on $STEP_ENV"
    "$PRESK" solve "$GPT" --precision "$STEP_ENV" --menu f32,f16 --offers "$OUT/fc4/offers.json" > "$OUT/fc4/offers.txt" 2>&1
    $PY - "$OUT/fc4/offers.json" "$OUT/fc4/realization.json" <<'EOF'
import json, sys
o = json.load(open(sys.argv[1]))
refused = {v["site"]: "f32" for v in o["values"] if v["site"].startswith("adamw.") and "f16" not in v["offered"]}
json.dump({"default": "f16", "storage": refused, "order": {"kind": "sequential"}}, open(sys.argv[2], "w"), indent=1)
print(f"{sum(v['site'].startswith('adamw.') for v in o['values'])} stored values of the optimizer, {len(refused)} refused binary16: {', '.join(refused)}")
EOF
}
finite() { # result.psk.d: the non-finite reals of an AdamW step's result, the new parameters (its first third) and all
    local n
    n=$("$PRESK" data info "$1" | sed -n 's/^sample *\([0-9]*\) bytes.*/\1/p')
    $PY - "$1" "$n" <<'EOF'
import math, os, sys
from array import array
path, n = sys.argv[1], int(sys.argv[2])
a = array("d")
with open(path, "rb") as f:
    f.seek(os.path.getsize(path) - n)
    a.frombytes(f.read(n))
third = len(a) // 3
def count(xs):
    return sum(math.isinf(x) for x in xs), sum(math.isnan(x) for x in xs)
(pi, pn), (ai, an) = count(a[:third]), count(a)
print(f"{os.path.basename(os.path.dirname(os.path.dirname(path)))}: the new parameters, {third} reals: {pi} infinities, {pn} NaNs; the whole result, {len(a)} reals: {ai} infinities, {an} NaNs")
EOF
}
prep() {
    [ -s "$OUT/fc7/course/solved.json" ] || { echo "prep: run the phase solve first"; exit 1; }
    [ -s "$OUT/calib/done" ] || { echo "prep: run the phase calib first"; exit 1; }
    fc4_realization
    for r in f32 f16 e8m15; do echo "{\"realization\": \"$r\", \"layout\": {}}" > "$P/$r.json"; done
    wrap "$OUT/fc4/realization.json" "$P/fc4.json"
    wrap "$FC3" "$P/d64-fc3.json"
    wrap "$D170_64" "$P/d64-d170.json"
    wrap "$OUT/fc7/course/solved.json" "$P/course-fc7.json"
    wrap "$D170_COURSE" "$P/course-d170.json"
    say "FC5's plans"
    "$PRESK" plan "${FWD[@]:0:7}" --realize f32 --profile "$FROZEN" --out "$P/fwd-plan-f32.json" > "$P/fwd-plan-f32.log" 2>&1 || { echo "plan f32 failed"; exit 1; }
    "$PRESK" plan "${FWD[@]:0:7}" --realize f16 --profile "$OUT/calib/profile-f16.json" --out "$P/fwd-plan-f16.json" > "$P/fwd-plan-f16.log" 2>&1 || { echo "plan f16 failed"; exit 1; }
    say "the C programs"
    c_export fwd-f32 f32 "${FWD[@]}"
    c_export fwd-f16 f16 "${FWD[@]}"
    c_build fwd-f16 model-noguard -DPRESK_NO_GUARD
    c_export adam-f32 f32 "${ADAM[@]}"
    c_export adam-fc4 "@$OUT/fc4/realization.json" "${ADAM[@]}"
    c_export d64-f32 f32 "${L64[@]}"
    c_export d64-d170 "@$D170_64" "${L64[@]}"
    c_export d64-fc3 "@$FC3" "${L64[@]}"
    c_export course-f32 f32 "${LCOURSE[@]}"
    c_export course-d170 "@$D170_COURSE" "${LCOURSE[@]}"
    c_export course-fc7 "@$OUT/fc7/course/solved.json" "${LCOURSE[@]}"
    grep -h "^presk export" "$C"/*.log > "$OUT/exports.txt"
    touch "$OUT/prep.done"
}

# --- time: the rows, alone ----------------------------------------------------
# key | how | what
ROWS=(
    "fc4.llvm.f32|bench --llvm|adam f32"           "fc4.llvm.fc4|bench --llvm|adam fc4"
    "fc4.native.f32|bench|adam f32"                "fc4.native.fc4|bench|adam fc4"
    "fc4.c.f32|c model|adam-f32"                   "fc4.c.fc4|c model|adam-fc4"
    "fc5.llvm.f32|bench --llvm|fwd f32"            "fc5.llvm.f16|bench --llvm|fwd f16"
    "fc5.llvm.e8m15|bench --llvm|fwd e8m15"
    "fc5.llvm-plan.f32|bench --llvm|fwd fwd-plan-f32" "fc5.llvm-plan.f16|bench --llvm|fwd fwd-plan-f16"
    "fc5.native.f32|bench|fwd f32"                 "fc5.native.f16|bench|fwd f16"
    "fc5.native.e8m15|bench|fwd e8m15"
    "fc6.c.f32|c model|fwd-f32"                    "fc6.c.guard|c model|fwd-f16"
    "fc6.c.noguard|c model-noguard|fwd-f16"
    "fc7.d64.llvm.f32|bench --llvm|l64 f32"        "fc7.d64.llvm.d170|bench --llvm|l64 d64-d170"
    "fc7.d64.llvm.fc3|bench --llvm|l64 d64-fc3"
    "fc7.d64.c.f32|c model|d64-f32"                "fc7.d64.c.d170|c model|d64-d170"
    "fc7.d64.c.fc3|c model|d64-fc3"
    "fc7.course.llvm.f32|bench --llvm|lcourse f32" "fc7.course.llvm.d170|bench --llvm|lcourse course-d170"
    "fc7.course.llvm.fc7|bench --llvm|lcourse course-fc7"
    "fc7.course.c.f32|c model|course-f32"          "fc7.course.c.d170|c model|course-d170"
    "fc7.course.c.fc7|c model|course-fc7"
)
row() { # round key how what
    local r=$1 key=$2 how=$3 what=$4 log=$OUT/log/$2.$1.log res
    case $how in
        bench*)
            local flag=${how#bench} args plan
            read -r args plan <<< "$what"
            case $args in
                adam) args=("${ADAM[@]}") ;;
                fwd) args=("${FWD[@]}") ;;
                l64) args=("${L64[@]}") ;;
                lcourse) args=("${LCOURSE[@]}") ;;
            esac
            # shellcheck disable=SC2086 # the flag is a word or none
            taskset -c "$CORES" "$PRESK" bench "${args[@]}" --plan "@$P/$plan.json" $flag --native-only --plan-only --warmup 1 --repeat "$REPEAT" --threads 10 > "$log" 2>&1
            res=$(awk '/ plan / { print $(NF-2), $(NF-1) }' "$log")
            ;;
        c*)
            local bin=${how#c } d=$C/$what a
            a=$(c_args "$d" "conformance/time-$bin.psk.d")
            (cd "$d" && eval "LD_LIBRARY_PATH=$G16LIB OMP_NUM_THREADS=10 GOMP_SPINCOUNT=8M taskset -c $CORES ./$bin -r $REPEAT $a") > "$log" 2>&1
            res=$(sed -n 's/.*median \([0-9.]*\) ms, best \([0-9.]*\) ms.*/\1 \2/p' "$log")
            ;;
    esac
    [ -n "$res" ] || res="failed -"
    echo "$r $key $res" | tee -a "$OUT/times/rows.txt"
}
machine() {
    {
        echo "date $(date -Iseconds)"
        echo "presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates experiments || echo -dirty), $("$PRESK" --version)"
        echo "kernel $(uname -sr), $(nproc) cores"
        echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
        "$G16" --version | sed -n 1p
        clang --version 2>/dev/null | sed -n 1p
        echo "c flags $CFLAGS_C"
    } >> "$OUT/machine.txt"
}
timed() {
    [ -f "$OUT/prep.done" ] || { echo "time: run the phase prep first"; exit 1; }
    : > "$OUT/times/rows.txt"
    machine
    for r in $(seq 1 "$PROCS"); do
        say "round $r of $PROCS"
        for spec in "${ROWS[@]}"; do
            IFS='|' read -r key how what <<< "$spec"
            row "$r" "$key" "$how" "$what"
        done
    done
    echo "load after $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/machine.txt"
}

# --- fc4: FC4's rows again, alone (plan 0.18.2 step 6, D220) -------------------
fc4() {
    fc4_realization
    echo '{"realization": "f32", "layout": {}}' > "$P/f32.json"
    wrap "$OUT/fc4/realization.json" "$P/fc4.json"
    say "FC4's C programs"
    c_export adam-f32 f32 "${ADAM[@]}"
    c_export adam-fc4 "@$OUT/fc4/realization.json" "${ADAM[@]}"
    say "FC4's realization on the export's inputs, natively"
    finite "$C/adam-fc4/conformance/native.psk.d" | tee "$OUT/fc4/finite.txt"
    : > "$OUT/times/rows.txt"
    machine
    for r in $(seq 1 "$PROCS"); do
        say "round $r of $PROCS"
        for spec in "${ROWS[@]:0:6}"; do
            IFS='|' read -r key how what <<< "$spec"
            row "$r" "$key" "$how" "$what"
        done
    done
    echo "load after $(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT/machine.txt"
}

# --- report -------------------------------------------------------------------
report() {
    $PY experiments/fc_rows.py "$OUT" > "$OUT/report.md"
    cat "$OUT/report.md"
}

case ${1:-all} in
    solve) solve ;;
    calib) calib ;;
    prep) prep ;;
    time) timed ;;
    report) report ;;
    all) solve; calib; prep; timed; report ;;
    fc4) fc4; report ;;
    *) echo "usage: experiments/fc_rows.sh [solve|calib|prep|time|report|all|fc4]" >&2; exit 2 ;;
esac
say "done ${1:-all}"
