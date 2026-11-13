#!/bin/bash
# Every realization's validation loss on the held-out windows (D129): `loss`
# of examples/corpus/nlp/05_gpt.psk run natively under each realization on every
# window of VAL at once (`presk run --data` binds the batch to them all), as
# experiments/curve.sh's quality axis reads a model's validation loss, so
# that each row of the margins' table has the accuracy figure beside its
# gap. The exact model's own loss on the same windows is
# experiments/margins_loss.py's, from the exact logits.
#
#   NAME=SPEC ...   the realizations (a type, or @FILE), as arguments
#
# Into OUT/<NAME>.txt, one number each, and OUT/val_loss.md, the table. A
# row whose file exists is not run again. JOBS runs at a time, THREADS
# workers each (a native f64 run at the course constants on the 461 windows
# holds 14 GB). Nothing is timed.
set -u -o pipefail
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
MODEL=examples/corpus/nlp/05_gpt.psk
CONSTS=(${CONSTS:---const D=384 --const H=6 --const L=6})
W=${W:-out/course_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
OUT=${OUT:-out/margins-course/val-loss}
JOBS=${JOBS:-2}
THREADS=${THREADS:-4}
mkdir -p "$OUT"

one() { # name spec
    [ -s "$OUT/$1.txt" ] && return 0
    "$PRESK" run "$MODEL" --entry loss "${CONSTS[@]}" --dim T=256 --weights "$W" --data "$VAL" \
        --backend native --threads "$THREADS" --realize "$2" --digits 12 2> "$OUT/$1.err" \
        | awk '{print $NF}' > "$OUT/$1.txt.part" && [ -s "$OUT/$1.txt.part" ] \
        && mv "$OUT/$1.txt.part" "$OUT/$1.txt" && rm -f "$OUT/$1.err" || echo "the $1 run failed: $OUT/$1.err"
}
running=0
names=()
for ns in "$@"; do
    names+=("${ns%%=*}")
    one "${ns%%=*}" "${ns#*=}" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
done
wait
{
    echo "| realization | validation loss | vs all-f64 |"
    echo "|---|---|---|"
    base=$(cat "$OUT/f64.txt" 2>/dev/null)
    for n in "${names[@]}"; do
        v=$(cat "$OUT/$n.txt" 2>/dev/null)
        d=""
        [ -n "$base" ] && [ -n "$v" ] && d=$(python3 -c "print(f'{float(\"$v\") - float(\"$base\"):+.3e}')")
        echo "| $n | ${v:-—} | $d |"
    done
} > "$OUT/val_loss.md"
cat "$OUT/val_loss.md"
