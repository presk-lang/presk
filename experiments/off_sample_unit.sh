#!/bin/bash
# One solve to separate R from C in the course realization's choice of the
# queries' bias in bf16 (step 0d of the course-scale certificate study,
# docs/off-sample-course-step0b-2026-10-22.md §"R or C"): D163 §5's logits
# solve, its command, its binary and its envelope unchanged (windows 0-3 of
# civil_val_T256, the trained course weights), but `--rounding unit`, the
# estimate's rounding term of 0.12.0-0.15.0 (u(d)·max|x|); the traffic stays
# 0.16.0's (C, one sample's). Nothing calibrated, nothing timed of record.
#
#   ./experiments/off_sample_unit.sh                   into out/off-sample-course-step0d/
#   EXTRA="unit=out/off-sample-course-step0d/solved.json" ./experiments/off_sample_tail.sh
#                                                      then its native rows beside step 0b's
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-out/bin/presk-22dd345}
ENV=${ENV:-out/margins-course-0.16.0/logits/env_logits.json}
W=${W:-out/course_train_f64/final.psk.d}
BUDGET=${BUDGET:-4.96e-4}
OUT=${OUT:-out/off-sample-course-step0d}

for f in "$PRESK" "$ENV" "$W"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
[ -f "$OUT/solved.json" ] && { echo "$OUT/solved.json exists"; exit 1; }
mkdir -p "$OUT"
"$PRESK" --version > "$OUT/presk.txt"
echo "== solve $(date -Is)"
/usr/bin/time -v -o "$OUT/solve.time" "$PRESK" solve experiments/margins.psk --precision "$ENV" --budget-abs "$BUDGET" \
    --weights "$W" --sites 60 --rounding unit --narrow off --out "$OUT/solved.json" > "$OUT/solve.txt" 2> "$OUT/solve.err" \
    || { echo "the solve failed: $OUT/solve.err"; exit 1; }
echo "== done $(date -Is)"
