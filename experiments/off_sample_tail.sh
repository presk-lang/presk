#!/bin/bash
# Which decisions of the course GPT's logits realization (D163 §5) carry its
# tail of windows: step 0b of the course-scale certificate study (presk-project
# docs/assessment-0.16.0-2026-10-22.md §6.2, docs/off-sample-course-step0b-
# 2026-10-22.md). Nothing is calibrated or solved: the variants are the
# realization with some of its decisions changed, each run natively on the
# 461 held-out windows by margins_course.sh's native step and judged against
# the exact logits of record (D129, out/margins-course/exact).
#
#   1. the variants (experiments/off_sample_tail.py variants): the realization
#      as is, D129's, each bf16 decision back to f32 one at a time, all of
#      them, and the decisions that differ from D129's put back to its types;
#   2. per variant, its native logits on every window, by chunks of CHUNK
#      windows, JOBS at a time, the gap per position (margins.py native);
#   3. the report, against the rows of record for the realization as is and
#      for D129's.
#
#   ./experiments/off_sample_tail.sh
#   PRESK=out/bin/presk-22dd345 ./experiments/off_sample_tail.sh     the binary of record (default)
#   EXTRA="unit=out/off-sample-course-step0d/solved.json" ./experiments/off_sample_tail.sh
#                                                     step 0d's realization beside them
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-out/bin/presk-22dd345}
PY=${PY:-python3}
FILE=examples/experiments/margins.psk
DIMS=(--const D=384 --const H=6 --const L=6 --dim T=256)
W=${W:-out/course_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
SEED=${SEED:-0}
SOLVED=${SOLVED:-out/margins-course-0.16.0/logits/solved.json}
OTHER=${OTHER:-out/margins-course/logits/solved.json}
EXACT=${EXACT:-out/margins-course/exact}
CHUNK=${CHUNK:-32}
JOBS=${JOBS:-8}
THREADS=${THREADS:-2}
OUT=${OUT:-out/off-sample-course-step0b}
N=461

for f in "$PRESK" "$W" "$VAL" "$SOLVED" "$OTHER" "$EXACT/0000.hilo"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
mkdir -p "$OUT/native" "$OUT/json" "$OUT/realizations"
"$PRESK" --version > "$OUT/presk.txt"
$PY experiments/off_sample_tail.py variants "$SOLVED" "$OTHER" "$OUT/realizations" > "$OUT/variants.tsv" || exit 1
$PY experiments/off_sample_tail.py diff "$SOLVED" "$OTHER" > "$OUT/diff.md" || exit 1
# EXTRA="name=realization.json …": more realizations run beside the variants
# (step 0d's solve under --rounding unit), each also compared with SOLVED.
for nr in ${EXTRA:-}; do
    cp "${nr#*=}" "$OUT/realizations/${nr%%=*}.json" || exit 1
    printf '%s\t%s\n' "${nr%%=*}" "\`${nr#*=}\`" >> "$OUT/variants.tsv"
    $PY experiments/off_sample_tail.py diff "${nr#*=}" "$SOLVED" > "$OUT/diff-${nr%%=*}.md" || exit 1
done
mapfile -t NAMES < <(cut -f1 "$OUT/variants.tsv")

running=0
spawn() { "$@" & running=$((running + 1)); if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi; }
drain() { wait; running=0; }
chunks() { for ((f = 0; f < N; f += CHUNK)); do echo "$f:$(( N - f < CHUNK ? N - f : CHUNK ))"; done; }

# margins_course.sh's native step, its command line unchanged.
native() { # name spec from count
    local p; p=$OUT/native/$1/$(printf %04d "$3")
    [ -f "$p.tsv" ] && return 0
    "$PRESK" run "$FILE" --entry logits "${DIMS[@]}" --weights "$W" --data "$VAL" --from "$3" --samples "$4" \
        --seed "$SEED" --backend native --threads "$THREADS" --realize "$2" --json --digits 17 > "$OUT/json/$1_$3.json" 2> "$p.err" \
        && $PY experiments/margins.py native "$OUT/json/$1_$3.json" "$EXACT/$(printf %04d "$3")" "$p" \
        && rm -f "$OUT/json/$1_$3.json" "$p.err" || echo "the $1 run of windows $3.. failed: $p.err"
}
echo "== native $(date -Is)"
for n in "${NAMES[@]}"; do
    mkdir -p "$OUT/native/$n"
    for fc in $(chunks); do spawn native "$n" "@$OUT/realizations/$n.json" "${fc%:*}" "${fc#*:}"; done
done
drain
echo "== report $(date -Is)"
{
    echo "# Step 0b: the course realization's tail against its decisions"
    echo
    echo "$(cat "$OUT/presk.txt") (\`$PRESK\`); the realization \`$SOLVED\`, compared with \`$OTHER\`."
    echo
    echo "## The two realizations, decision by decision"
    echo
    cat "$OUT/diff.md"
    echo
    for nr in ${EXTRA:-}; do
        echo "## \`${nr%%=*}\` against the realization studied"
        echo
        cat "$OUT/diff-${nr%%=*}.md"
        echo
    done
    echo "## The variants"
    echo
    awk -F'\t' '{print "- `" $1 "`: " $2 "."}' "$OUT/variants.tsv"
    echo
    echo "## Their gaps on the 461 windows"
    echo
    $PY experiments/off_sample_tail.py report "$OUT" --exact "$EXACT" --names "${NAMES[@]}" \
        --ref "as-is=out/margins-course-0.16.0/native/logits-4.96e-4" "other=out/margins-course/native/logits-4.96e-4" --controls 0,1,2,3,100,200,300
} > "$OUT/report.md"
echo "== done $(date -Is)"
