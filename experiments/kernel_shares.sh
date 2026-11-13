#!/bin/bash
# Each kernel's share of the native backend's gap to the C program (plan
# 0.17.0 step 1, level 1, D174; statement NB1).
#
#   experiments/kernel_shares.sh OUT
#
# The course GPT at B = 16, T = 256 in f32, under the plans the frozen
# profile gives (as pytorch_baseline.sh plans them): the forward pass
# `GPT.apply`, the gradient `loss_grad` (examples/experiments/gpt_grad.psk) and the
# optimizer `adamw.step`, the step's two entries. Each is timed launch by
# launch on both sides:
# - natively, by `presk bench --plan-only --calibrate`, which times every
#   launch, the pool's wakes included, and keeps a kernel's median launch,
#   without timing the kernels again under the other realizations;
# - in the C program `presk export --target c` writes, by kernel_time.py's
#   copy of model.c, which times each kernel's call or OpenMP region.
#
# Two settings:
# - "ten": ten X925 cores pinned (CORES), ten threads, the C program built
#   by GCC -O2 with OpenMP, its workers waiting as the native pool's
#   (GOMP_SPINCOUNT=8M): T4's setting (D127);
# - "one": one X925 core (CORE), one thread, under the same plans with every
#   split `none` (a split of ten workers is refused on one, condition 5), the
#   C program exported from those plans and built without OpenMP by GCC -O2,
#   by clang -O2 and by clang -O3 (CLANG, LLVM's release of `make llvm`): the
#   code generators alone, clang's columns predicting level 3.
#
# Every C program is checked against the native backend's outputs bit for
# bit (C1) before it is timed, and every build's timed copy too. ROUNDS
# processes of each (D126), the machine quiet: a round starts when the load
# is under 1.5. kernel_shares.py joins the sides by the kernel's signature
# and writes OUT/report.md. A measurement: it decides nothing.
#
# BENCH, words added to the native side's `presk bench` (plan 0.17.0 step 8,
# D181; statement NB4): `--llvm` times the LLVM backend's kernels in place
# of Cranelift's; PRESK another build times another code generator. The C
# side is the same either way.
set -eu -o pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PRESK=${PRESK:-target/release/presk}
case $PRESK in /*) ;; *) PRESK=$ROOT/$PRESK ;; esac
CLANG=${CLANG:-$ROOT/.llvm/bin/clang}
PROFILE=${PROFILE:-$HOME/presk-data/ra4-model-0.8.0/profile.json}
CKPT=${CKPT:-out/course_probe_f32/final.psk.d}
case $CKPT in /*) ;; *) CKPT=$ROOT/$CKPT ;; esac
CORES=${CORES:-5-9,15-19}
CORE=${CORE:-5}
C_WAIT=${C_WAIT:-GOMP_SPINCOUNT=8M}
REPEAT=${REPEAT:-7}
CALLS=${CALLS:-4}
ROUNDS=${ROUNDS:-3}
BENCH=${BENCH:-}
OUT=${1:?usage: experiments/kernel_shares.sh OUT}
FILE=examples/corpus/nlp/05_gpt.psk
GRAD=examples/experiments/gpt_grad.psk
DIMS=(--dim B=16 --dim T=256)
STRICT=(-std=c11 -ffp-contract=off -Wall -Wextra)

for f in "$PRESK" "$CLANG" "$PROFILE" "$CKPT"; do
    [ -e "$f" ] || { echo "kernel_shares.sh: missing $f (clang: make llvm; the checkpoint: CKPT)" >&2; exit 1; }
done
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
say() { echo "-- $(date +%T) $*"; }
quiet() { # wait until the one-minute load is under 1.5
    while awk '{ exit !($1 >= 1.5) }' /proc/loadavg; do sleep 30; done
}

{
    echo "date $(date -Iseconds)"
    echo "presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates experiments || echo -dirty), $("$PRESK" --version)"
    echo "kernel $(uname -sr), $(nproc) cores"
    gcc --version | sed -n 1p
    "$CLANG" --version | sed -n 1p
    echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
    echo "cores $CORES, core $CORE, c_wait $C_WAIT, repeat $REPEAT, calls $CALLS, rounds $ROUNDS"
    [ -z "$BENCH" ] || echo "bench $BENCH"
} > "$OUT/machine.txt"
cat "$OUT/machine.txt"

say "plans (the frozen profile)"
P=$OUT/plans
mkdir -p "$P"
"$PRESK" plan "$FILE" --entry GPT.apply "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/fwd.json" > "$P/fwd.log" 2>&1
"$PRESK" plan "$GRAD" --entry loss_grad "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/grad.json" > "$P/grad.log" 2>&1
"$PRESK" plan "$FILE" --entry adamw.step "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/adam.json" > "$P/adam.log" 2>&1
# The one-thread plans: the same layouts, no split.
for key in fwd grad adam; do
    mv "$P/$key.json" "$P/$key-ten.json"
    python3 -c 'import json, sys; p = json.load(open(sys.argv[1])); [v.update(split="none") for v in p["layout"].values()]; json.dump(p, open(sys.argv[2], "w"), indent=2)' "$P/$key-ten.json" "$P/$key-one.json"
done

# The three entries: key, file, entry.
ENTRIES=("fwd $FILE GPT.apply" "grad $GRAD loss_grad" "adam $FILE adamw.step")

say "exports, checked (C1)"
C=$OUT/c
rm -rf "$C"
mkdir -p "$C"
for e in "${ENTRIES[@]}"; do
    read -r key file entry <<< "$e"
    for setting in ten one; do
        d=$C/$key-$setting
        th=10
        [ "$setting" = one ] && th=1
        "$PRESK" export "$file" --target c --entry "$entry" "${DIMS[@]}" --plan "@$P/$key-$setting.json" --weights "$CKPT" --threads $th --no-exact -o "$d" 2> "$d.log"
        make -s -C "$d" model
        (cd "$d" && sh conformance/check.sh) > "$d.check" 2>&1
        grep -q "C1 holds" "$d.check" || { echo "kernel_shares.sh: $d: C1 fails" >&2; cat "$d.check" >&2; exit 1; }
        python3 experiments/kernel_time.py instrument "$d/model.c" > "$d/model_timed.c"
    done
done

# build DIR NAME CC FLAGS...: the timed copy of DIR's program, DIR/NAME.
build() {
    local d=$1 name=$2 cc=$3
    shift 3
    (cd "$d" && "$cc" "${STRICT[@]}" "$@" -o "$name" main.c model_timed.c presk_rt.c presk_libm.c "$ROOT/experiments/kernel_time.c" -lm)
}
BUILDS=("ten gcc-O2 gcc -O2 -fopenmp" "one gcc-O2 gcc -O2" "one clang-O2 $CLANG -O2" "one clang-O3 $CLANG -O3")
say "the timed builds"
for e in "${ENTRIES[@]}"; do
    read -r key _ _ <<< "$e"
    for b in "${BUILDS[@]}"; do
        read -r setting name cc flags <<< "$b"
        # shellcheck disable=SC2086 # the flags are words
        build "$C/$key-$setting" "model_$setting-$name" "$cc" $flags
    done
done

# c_time DIR SETTING NAME ROUND: one process of the timed copy, its launches
# into DIR/times/SETTING-NAME-ROUND.txt, its outputs checked unchanged.
c_time() {
    local d=$1 setting=$2 name=$3 round=$4 pin env
    mkdir -p "$d/times"
    if [ "$setting" = ten ]; then pin="taskset -c $CORES"; env="$C_WAIT OMP_NUM_THREADS=10"; else pin="taskset -c $CORE"; env="OMP_NUM_THREADS=1"; fi
    # shellcheck disable=SC2086 # the prefixes are words
    (cd "$d" && env $env KERNEL_TIME="times/$setting-$name-$round.txt" $pin "./model_$setting-$name" -r "$CALLS" "$CKPT" conformance/inputs.psk.d "conformance/out_$setting-$name.psk.d" > /dev/null 2> "times/$setting-$name-$round.log" \
        && cmp -s "conformance/out_$setting-$name.psk.d" conformance/native.psk.d) \
        || { echo "kernel_shares.sh: $d: the timed $setting-$name program's outputs differ from the native backend's" >&2; exit 1; }
}
# native_time KEY FILE ENTRY SETTING ROUND: `presk bench --calibrate`'s
# profile, each kernel's median launch, into OUT/native/KEY-SETTING-ROUND.json.
native_time() {
    local key=$1 file=$2 entry=$3 setting=$4 round=$5 pin th
    mkdir -p "$OUT/native"
    if [ "$setting" = ten ]; then pin="taskset -c $CORES"; th=10; else pin="taskset -c $CORE"; th=1; fi
    rm -f "$OUT/native/$key-$setting-$round.json"
    # shellcheck disable=SC2086 # the prefix and BENCH are words
    $pin "$PRESK" bench "$file" --entry "$entry" "${DIMS[@]}" --plan "@$P/$key-$setting.json" --weights "$CKPT" --threads $th --repeat "$REPEAT" --plan-only $BENCH --calibrate "$OUT/native/$key-$setting-$round.json" > "$OUT/native/$key-$setting-$round.log" 2>&1
}

for round in $(seq 1 "$ROUNDS"); do
    quiet
    say "round $round of $ROUNDS (load $(cut -d' ' -f1 /proc/loadavg))"
    for e in "${ENTRIES[@]}"; do
        read -r key file entry <<< "$e"
        for setting in ten one; do
            native_time "$key" "$file" "$entry" "$setting" "$round"
        done
        for b in "${BUILDS[@]}"; do
            read -r setting name _ <<< "$b"
            c_time "$C/$key-$setting" "$setting" "$name" "$round"
        done
    done
done

say "the report"
python3 experiments/kernel_shares.py "$OUT" --calls "$CALLS" --rounds "$ROUNDS" > "$OUT/report.md"
say "done: $OUT/report.md"
