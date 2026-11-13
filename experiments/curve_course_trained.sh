#!/bin/bash
# The course-scale curve on the trained model's envelope (D132): a copy of
# experiments/curve_course.sh whose calibration reads the trained weights on
# two held-out windows of text (`--weights W --data out/civil_val_T256.psk.d
# --samples 2 --from 0`, windows 0 and 1) where curve_course.sh calibrates
# on the initialization and random tokens; a solve runs on its envelope's
# inputs, so its sensitivities and measurement move to those windows too.
# Realizations under out/curve-course-trained_<b>.json, the envelope
# out/curve-course-trained_env.json; a second table gives each solve's
# parameters' types and widening rounds. Otherwise curve_course.sh's text:
#
# The curve's solver columns at the course constants (D129): a copy of
# experiments/curve.sh's QUALITY=0 path for the course GPT at D = 384,
# H = 6, L = 6, on the weights trained at those constants (600 steps in f64,
# out/course_train_f64, trained by the command docs/margins-course-2026-10-16.md
# gives, 4 h), for experiments/margins_course.sh to read as make margins
# reads make curve's.
#
# As curve.sh does: the envelope of `loss` on the seed's synthesized
# inputs at B = 2, T = 256, 128 bits; then `presk solve` at each of the
# seven budgets with the trained weights, the sensitivities and the
# measurement on them; out/curve-course_<b>.json each. No quality axis: no
# training per budget. The solves are independent and run JOBS at a time
# (one thread each; D101: 45 min each at these constants on one X925 core,
# the calibration 25 min); a solve whose realization exists is not run
# again. Nothing is timed.
set -u -o pipefail
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
REMAT=${REMAT:-store}
MODEL=examples/corpus/nlp/05_gpt.psk
CONSTS=(--const D=384 --const H=6 --const L=6)
DIMS=(--dim B=2 --dim T=256)
ENV=out/curve-course-trained_env.json
VAL=${VAL:-out/civil_val_T256.psk.d}
W=${W:-out/course_train_f64/final.psk.d}
OUT=${1:-out/curve-course-trained.md}
JOBS=${JOBS:-7}
BUDGETS=(${BUDGETS:-1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7})   # a subset: BUDGETS="1e-3"

[ -f "$W" ] || { echo "no trained weights at $W: train at the course constants first (docs/margins-course-2026-10-16.md)"; exit 1; }

if [ ! -f "$ENV" ]; then
    echo "== calibrate $(date -Is)"
    "$PRESK" calibrate --precision "$MODEL" --entry loss "${CONSTS[@]}" --dim T=256 \
        --weights "$W" --data "$VAL" --samples 2 --from 0 \
        --bits 128 --remat "$REMAT" --out "$ENV" > "${ENV%.json}.log" 2>&1 || { echo "calibrate failed"; exit 1; }
fi
BOUND=$("$PRESK" check --precision "$ENV" "$MODEL" --realize f64 --model cond 2>/dev/null \
    | awk '$1 == "bound" {for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]/) {print $i; exit}}')

solve() { # budget
    [ -f "out/curve-course-trained_$1.json" ] && return 0
    echo "== solve $1 $(date -Is)"
    "$PRESK" solve "$MODEL" --precision "$ENV" --budget "$1" --weights "$W" \
        --out "out/curve-course-trained_$1.json" --sites 0 --remat "$REMAT" > "out/curve-course-trained_$1.txt" 2>&1 \
        || echo "solve $1 failed: out/curve-course-trained_$1.txt"
    echo "== solved $1 $(date -Is)"
}
running=0
for b in "${BUDGETS[@]}"; do
    solve "$b" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
done
wait

{
  echo "| budget | sites narrowed | cost | vs all-f64 | measured gap | inside? | bound (f64, cond) |"
  echo "|---|---|---|---|---|---|---|"
  for b in "${BUDGETS[@]}"; do
      r=$(cat "out/curve-course-trained_$b.txt")
      n=$(echo "$r" | awk '/^  sites/ {print $4}')
      c=$(echo "$r" | awk '/^  cost/ {print $2}')
      x=$(echo "$r" | awk '/^  cost/ {print $6}' | tr -d '(),')
      g=$(echo "$r" | awk '/^  measured/ {print $2}')
      i=$(echo "$r" | grep -q "inside the budget" && echo "yes" || echo "NO")
      echo "| $b | $n | $c | $x | $g | $i | $BOUND |"
  done
  echo
  echo "Solver columns on $W at T = 256, the course constants D = 384, H = 6, L = 6: the envelope, the"
  echo "sensitivities and the measurement on windows 0 and 1 of $VAL (D132)."
  echo
  echo "| budget | stored values narrowed (bf16) | parameters typed (bf16) | estimate | widening rounds |"
  echo "|---|---|---|---|---|"
  for b in "${BUDGETS[@]}"; do
      python3 - "out/curve-course-trained_$b.json" "out/curve-course-trained_$b.txt" "$b" <<'PY'
import json, re, sys
r = json.load(open(sys.argv[1])); t = open(sys.argv[2]).read()
st, pa = list(r.get("storage", {}).values()), list(r.get("params", {}).values())
est = re.search(r"^  estimate\s+([0-9.e+-]+)", t, re.M).group(1)
rounds = re.search(r"after ([0-9]+) widening", t).group(1)
print(f"| {sys.argv[3]} | {len(st)} ({st.count('bf16')}) | {len(pa)} ({pa.count('bf16')}) | {est} | {rounds} |")
PY
  done
} > "$OUT"
cat "$OUT"
