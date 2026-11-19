#!/bin/bash
# The course GPT's logits at D = 64 in a machine's types (plan 0.18.0 step 6,
# D209): statements MT5 and MT6. Both read `make margins`' exact logits on
# the Code civil's 461 held-out windows (D118) and V6's envelope and weights
# (D170), and judge each realization as margins.sh judges a row, its logits
# by chunks against the exact ones and its decisions on every window, but on
# the kernel interpreter, which emulates every format.
#   MT5 (reported; TARGET, a target description, presk solve --target):
#     1. ε*(δ) for δ = 10⁻¹, 10⁻², 10⁻³ from the exact margins, by D118's
#        rule (Clopper–Pearson at α = 0.05; 10⁻² and 10⁻³ are D118's);
#     2. the solve at each ε*(δ), writing the realization, or the one of
#        smallest gap where none in the target's types is inside
#        (--refused);
#     3. the accumulator's readings on the same contractions: each
#        contraction (a `dot` or an `ein` of two operands) whose product
#        format is a float's summed in f32, or exactly in a fixed-point
#        format (READINGS), its formats otherwise unchanged;
#   MT6 (holds or fails; no target): the solve at ε*(10⁻³) with MENU (at
#   V6's budget of record, 4.67·10⁻⁴), the
#   default menu with binary16 and the 16-bit-significand float of f32's
#   exponent, against D170's solve of record (RECORD): fewer bytes, inside
#   its budget by measurement, no decision changed on the 461 windows.
#   Then margins.py report on every row, and machine_types.py report.
# Nothing is timed. Every presk process runs in its own scope (MemoryMax=MEM,
# no swap, oom_score_adj 1000). Outputs present are reused: delete OUT to
# start again. MEASURE=native measures the solves on the native backend and
# BACKEND=native runs the rows there (plan 0.18.2 step 4, D218: statement
# FC3, experiments/fc3.sh); the kernel interpreter by default, D209's.
#
#   TARGET=examples/tests/targets/accelerator.json ./experiments/machine_types.sh
#   ./experiments/machine_types.sh               MT6 alone
#   REPORT_ONLY=1 TARGET=… ./experiments/machine_types.sh
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
PY=${PY:-python3}
FILE=examples/experiments/margins.psk
DIMS=(--const D=64 --const H=2 --const L=2 --dim T=256)
W=${W:-out/curve_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
ENV=${ENV:-out/v6-0.16.1/env_logits.json}
RECORD=${RECORD:-out/v6-0.16.1/solve.txt}
MARGINS=${MARGINS:-out/margins}
TARGET=${TARGET:-}
MENU=${MENU:-f64,f32,bf16,f16,e8m15}
MT6_BUDGET=${MT6_BUDGET:-4.67e-4}      # V6's, ε*(10⁻³) of record (D118, D170)
DELTAS=${DELTAS:-1e-1 1e-2 1e-3}
READINGS=${READINGS:-f32 fixed(i48,frac=18)}
OUT=${OUT:-out/machine-types}
MEASURE=${MEASURE:-kernel}
BACKEND=${BACKEND:-kernel}
CHUNK=${CHUNK:-16}
JOBS=${JOBS:-16}
MEM=${MEM:-8G}

for f in "$W" "$VAL" "$ENV" "$RECORD" "$MARGINS/val.json"; do [ -f "$f" ] || { echo "missing $f (the record of D118 and D170)"; exit 1; }; done
[ -d "$MARGINS/exact" ] || { echo "missing $MARGINS/exact: run make margins first"; exit 1; }
mkdir -p "$OUT/native" "$OUT/decisions" "$OUT/json" "$OUT/solve"
[ -e "$OUT/exact" ] || ln -s "$(realpath "$MARGINS/exact")" "$OUT/exact"
[ -n "$TARGET" ] && cp "$TARGET" "$OUT/target.json"
scoped() { systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 choom -n 1000 -- "$@"; }
say() { echo "== $(date -Is) $*"; }
N=$($PY -c "import json, sys; print(int(json.load(open(sys.argv[1]))['count']))" "$MARGINS/val.json")
chunks() { for ((f = 0; f < N; f += CHUNK)); do echo "$f:$(( N - f < CHUNK ? N - f : CHUNK ))"; done; }
running=0
spawn() { "$@" & running=$((running + 1)); if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi; }
drain() { wait; running=0; }

# 1. ε*(δ).
[ -s "$OUT/stars.json" ] || $PY experiments/machine_types.py stars "$MARGINS" $DELTAS > "$OUT/stars.json" || exit 1
star() { $PY -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$OUT/stars.json" "$1"; }
[ -s "$OUT/contractions.txt" ] || "$PRESK" dump --stage core "$FILE" --const D=64 --const H=2 --const L=2 2>/dev/null \
    | $PY experiments/machine_types.py contractions > "$OUT/contractions.txt" || exit 1

NAMES=()      # the report's rows
JUDGE=()      # the realizations judged: one of several identical files
declare -A SEEN
if [ "${REPORT_ONLY:-0}" != 1 ]; then say "machine types on $("$PRESK" --version)"; fi
# 2. The solves.
solve() { # name budget args…
    local n=$1 b=$2; shift 2
    [ -s "$OUT/solve/$n.txt" ] && return 0
    say "solve $n at $b"
    scoped /usr/bin/time -v -o "$OUT/solve/$n.time" "$PRESK" solve "$FILE" --precision "$ENV" --budget-abs "$b" --weights "$W" \
        --sites 60 --measure "$MEASURE" "$@" > "$OUT/solve/$n.txt" 2> "$OUT/solve/$n.err"
    echo $? > "$OUT/solve/$n.exit"
}
if [ -n "$TARGET" ]; then
    for d in $DELTAS; do
        b=$(star "$d")
        [ "${REPORT_ONLY:-0}" = 1 ] || solve "mt5-$d" "$b" --target "$OUT/target.json" --out "$OUT/solve/mt5-$d.json" --refused "$OUT/solve/mt5-$d.refused.json"
        r=$OUT/solve/mt5-$d.json
        [ -s "$r" ] || r=$OUT/solve/mt5-$d.refused.json
        [ -s "$r" ] || { echo "the solve at $d wrote no realization: $OUT/solve/mt5-$d.err"; continue; }
        # 3. The accumulator's readings. A realization the same as an
        # earlier budget's is judged once, its rows linked.
        cp "$r" "$OUT/solve/mt5-$d-f16.json"
        h=$(sha256sum < "$r" | cut -c1-64)
        first=${SEEN[$h]:-}
        [ -n "$first" ] || SEEN[$h]=$d
        for k in f16 $(for a in $READINGS; do echo "${a%%(*}"; done); do
            if [ "$k" != f16 ] && [ ! -s "$OUT/solve/mt5-$d-$k.json" ]; then
                a=$(for x in $READINGS; do [ "${x%%(*}" = "$k" ] && echo "$x"; done)
                $PY experiments/machine_types.py reading "$r" "$OUT/contractions.txt" "$a" > "$OUT/solve/mt5-$d-$k.json" || exit 1
            fi
            NAMES+=("mt5-$d-$k")
            if [ -n "$first" ]; then
                for t in native decisions; do [ -e "$OUT/$t/mt5-$d-$k" ] || ln -s "mt5-$first-$k" "$OUT/$t/mt5-$d-$k"; done
            else
                JUDGE+=("mt5-$d-$k")
            fi
        done
    done
fi
[ "${REPORT_ONLY:-0}" = 1 ] || solve mt6 "$MT6_BUDGET" --menu "$MENU" --out "$OUT/solve/mt6.json"
[ -s "$OUT/solve/mt6.json" ] && NAMES+=(mt6) && JUDGE+=(mt6)

# 4. Every realization on the 461 windows, on the kernel interpreter.
native() { # name from count
    local p; p=$OUT/native/$1/$(printf %04d "$2")
    [ -f "$p.tsv" ] && return 0
    scoped "$PRESK" run "$FILE" --entry logits "${DIMS[@]}" --weights "$W" --data "$VAL" --from "$2" --samples "$3" \
        --backend "$BACKEND" --realize "@$OUT/solve/$1.json" --json --digits 17 > "$OUT/json/$1_$2.json" 2> "$p.err" \
        && $PY experiments/margins.py native "$OUT/json/$1_$2.json" "$OUT/exact/$(printf %04d "$2")" "$p" \
        && rm -f "$OUT/json/$1_$2.json" || echo "the $1 run of windows $2.. failed: $p.err"
}
decide() { # name from count
    local d; d=$OUT/decisions/$1/$(printf %04d "$2")
    [ -f "$d.txt" ] && return 0
    scoped "$PRESK" run "$FILE" --entry decisions "${DIMS[@]}" --weights "$W" --data "$VAL" --from "$2" --samples "$3" \
        --backend "$BACKEND" --realize "@$OUT/solve/$1.json" --json > "$OUT/json/dec_$1_$2.json" 2> "$d.err" \
        && $PY experiments/margins.py decisions "$OUT/json/dec_$1_$2.json" "$d.txt" && rm -f "$OUT/json/dec_$1_$2.json" \
        || echo "the $1 decisions of windows $2.. failed: $d.err"
}
if [ "${REPORT_ONLY:-0}" != 1 ]; then
    # The exact chunks are margins.sh's, of 32 windows: the logits are judged by them.
    ECHUNK=32
    for n in "${JUDGE[@]}"; do
        say "judge $n"
        mkdir -p "$OUT/native/$n" "$OUT/decisions/$n"
        for ((f = 0; f < N; f += ECHUNK)); do spawn native "$n" "$f" "$(( N - f < ECHUNK ? N - f : ECHUNK ))"; done
        for fc in $(chunks); do spawn decide "$n" "${fc%:*}" "${fc#*:}"; done
    done
    drain
    for n in "${NAMES[@]}"; do cat "$OUT/decisions/$n"/*.txt > "$OUT/decisions/$n.txt"; done
fi

# 5. The reports.
$PY experiments/margins.py report "$OUT" --val "$MARGINS/val.json" --names "${NAMES[*]}" --labels "$(IFS='|'; echo "${NAMES[*]}")" \
    --deltas "$DELTAS" --out "$OUT/margins.md" > /dev/null
$PY experiments/machine_types.py report "$OUT" --record "$RECORD" --names "${NAMES[*]}" > "$OUT/report.md"
cat "$OUT/report.md"
say "done"
