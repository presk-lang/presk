#!/bin/bash
# Statement FC3 (plan 0.18.2 step 4, D218): D209's MT6 solve (the course
# GPT's logits at D = 64, binary16 and e8m15 on the menu, at V6's budget of
# record, 4.67·10⁻⁴), measured on the native backend, gives byte for byte
# D209's realization, its measured gap, and its rows and decisions on the
# Code civil's 461 held-out windows, which run on the native backend too.
# Reported: its time against the kernel interpreter's.
#   1. experiments/machine_types.sh, MT6 alone, MEASURE=native BACKEND=native,
#      into OUT;
#   2. the realization, the measured gap and every row and decision against
#      RECORD's (D209's run, out/machine-types), with cmp;
#   3. ROUNDS processes of the solve under each of --measure kernel and
#      native, alternated, and ROUNDS of the measurement alone (the
#      realization's run on the envelope's four samples, presk run) on each
#      backend: their medians and fastest (D126), into OUT/report.md.
# The timed processes run alone, one after the other, on an otherwise quiet
# machine. Every presk process in its own scope (MemoryMax=MEM, no swap).
#
#   ./experiments/fc3.sh
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
PY=${PY:-python3}
RECORD=${RECORD:-out/machine-types}
OUT=${OUT:-out/fc3-0.18.2}
ROUNDS=${ROUNDS:-3}
MEM=${MEM:-8G}
FILE=examples/experiments/margins.psk
DIMS=(--const D=64 --const H=2 --const L=2 --dim T=256)
W=${W:-out/curve_train_f64/final.psk.d}
ENV=${ENV:-out/v6-0.16.1/env_logits.json}
BUDGET=4.67e-4
MENU=f64,f32,bf16,f16,e8m15
for f in "$RECORD/solve/mt6.json" "$RECORD/solve/mt6.txt" "$RECORD/decisions/mt6.txt" "$W" "$ENV"; do [ -f "$f" ] || { echo "missing $f (D209's record)"; exit 1; }; done
mkdir -p "$OUT/times"
scoped() { systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 choom -n 1000 -- "$@"; }
say() { echo "== $(date -Is) $*"; }
say "FC3 on $("$PRESK" --version)"

# 1. MT6 natively.
MEASURE=native BACKEND=native OUT="$OUT" PRESK="$PRESK" ./experiments/machine_types.sh > "$OUT/machine_types.log" 2>&1 || { echo "machine_types.sh failed: $OUT/machine_types.log"; exit 1; }

# 2. Byte for byte.
same=0 differ=0
check() { # what file record
    if cmp -s "$2" "$3"; then same=$((same + 1)); else differ=$((differ + 1)); echo "DIFFERENT $1: $2 $3"; fi
}
check "the realization" "$OUT/solve/mt6.json" "$RECORD/solve/mt6.json"
grep '^  measured' "$OUT/solve/mt6.txt" > "$OUT/measured.txt"
grep '^  measured' "$RECORD/solve/mt6.txt" > "$OUT/measured-record.txt"
check "the measured gap" "$OUT/measured.txt" "$OUT/measured-record.txt"
for t in "$RECORD"/native/mt6/*.tsv; do check "rows $(basename "$t")" "$OUT/native/mt6/$(basename "$t")" "$t"; done
check "the decisions" "$OUT/decisions/mt6.txt" "$RECORD/decisions/mt6.txt"
say "compared: $same the same, $differ different"

# 3. Times.
EDATA=$($PY -c "import json, sys; d = json.load(open(sys.argv[1]))['data']; print(d['path'], d['from'], d['samples'])" "$ENV")
read -r DPATH DFROM DSAMPLES <<< "$EDATA"
for r in $(seq 1 "$ROUNDS"); do
    for m in kernel native; do
        say "round $r: the solve measured on $m"
        scoped /usr/bin/time -f "%e %M" -o "$OUT/times/solve-$m-$r.time" "$PRESK" solve "$FILE" --precision "$ENV" --budget-abs "$BUDGET" --weights "$W" \
            --sites 60 --menu "$MENU" --measure "$m" --out "$OUT/times/solve-$m-$r.json" > "$OUT/times/solve-$m-$r.txt" 2> "$OUT/times/solve-$m-$r.err"
        cmp -s "$OUT/times/solve-$m-$r.json" "$RECORD/solve/mt6.json" || echo "DIFFERENT the realization of round $r on $m"
        say "round $r: the measurement alone on $m"
        scoped /usr/bin/time -f "%e %M" -o "$OUT/times/run-$m-$r.time" "$PRESK" run "$FILE" --entry logits "${DIMS[@]}" --weights "$W" --data "$DPATH" --from "$DFROM" --samples "$DSAMPLES" \
            --backend "$m" --realize "@$RECORD/solve/mt6.json" --json --result-only > "$OUT/times/run-$m-$r.json" 2> "$OUT/times/run-$m-$r.err"
    done
done
for m in kernel native; do cmp -s "$OUT/times/run-$m-1.json" "$OUT/times/run-kernel-1.json" || echo "DIFFERENT the measurement's logits on $m"; done
$PY - "$OUT" "$ROUNDS" "$same" "$differ" > "$OUT/report.md" <<'EOF'
import statistics, sys
out, rounds, same, differ = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
def times(kind, m):
    return [float(open(f"{out}/times/{kind}-{m}-{r}.time").read().split()[0]) for r in range(1, rounds + 1)]
print("# FC3: D209's MT6 solve measured on the native backend\n")
print(f"Byte for byte against D209's record: {same} files the same, {differ} different (the realization, the measured gap, {int(same) + int(differ) - 3} chunks of rows, the decisions).\n")
print(f"| what | executor | median s | fastest s | {rounds} processes |\n|---|---|---|---|---|")
for kind, what in [("solve", "the solve"), ("run", "the measurement alone")]:
    for m in ["kernel", "native"]:
        t = times(kind, m)
        print(f"| {what} | {m} | {statistics.median(t):.2f} | {min(t):.2f} | {', '.join(f'{x:.2f}' for x in t)} |")
k, n = statistics.median(times("run", "kernel")), statistics.median(times("run", "native"))
print(f"\nThe measurement natively takes {n / k:.3f} of the kernel interpreter's time (medians), {k / n:.1f} times faster.")
EOF
cat "$OUT/report.md"
say "done"
