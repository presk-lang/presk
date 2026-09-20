#!/bin/bash
# The ε-to-quality curve (plan 0.5.0 §3.5, step 8).
#
# One program, one parameter varying. That is the whole point: the
# specification's complaint about the literature is that every quantisation
# level is a different pipeline with different code and different bugs, so a
# sweep compares artefacts rather than precisions. Here the program is
# identical at every point and only the budget moves.
#
# Per budget: what the solver chose, what it costs in traffic, what the
# propagated bound says, and what the distance to the exact reference
# actually is. The measured columns stand on their own; the bound is
# reported beside them and is known to be loose (D39).
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
MODEL=corpus/nlp/05_gpt.psk
CONSTS=(--const D=64 --const H=2 --const L=2)
DIMS=(--dim B=2 --dim T=32)
ENV=out/curve_env.json
W=out/gpt_tiny/final.psk.d
OUT=${1:-out/curve.md}

"$PRESK" calibrate --precision "$MODEL" --entry loss "${CONSTS[@]}" "${DIMS[@]}" \
    --bits 128 --out "$ENV" >/dev/null 2>&1 || { echo "calibrate failed"; exit 1; }

{
  echo "| budget | sites narrowed | traffic | vs all-f64 | measured gap | inside? | bound (f64) |"
  echo "|---|---|---|---|---|---|---|"
} > "$OUT"

BOUND=$("$PRESK" check --precision "$ENV" "$MODEL" --realize f64 2>/dev/null | awk '/bound/ {print $2}')

for b in 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7; do
    r=$("$PRESK" solve "$MODEL" --precision "$ENV" --budget "$b" --weights "$W" \
        --out "out/curve_$b.json" --sites 0 2>/dev/null)
    narrowed=$(echo "$r" | awk '/sites/ {print $4}')
    traffic=$(echo "$r" | awk '/traffic/ {print $2}')
    ratio=$(echo "$r" | awk '/traffic/ {print $7}' | tr -d '(),')
    gap=$(echo "$r" | awk '/measured/ {print $2}')
    verdict=$(echo "$r" | grep -q "inside the budget" && echo "yes" || echo "NO")
    echo "| $b | $narrowed | $traffic | $ratio | $gap | $verdict | $BOUND |" >> "$OUT"
    printf "%-6s narrowed %-4s traffic %-10s %-7s gap %-10s %s\n" "$b" "$narrowed" "$traffic" "$ratio" "$gap" "$verdict"
done
echo
echo "wrote $OUT"
