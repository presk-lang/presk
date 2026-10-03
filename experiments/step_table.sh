#!/bin/bash
# The step table of plan 0.6.0 step 7b (§3.7, D52): one optimizer step of the
# corpus GPT under each realization, against the exact step from the same
# state.
#
# update gap    ‖θ̂′ − θ′‖₂ / ‖θ′ − θ‖₂ — what a budget in a training run binds
# storage floor the same with θ′ exact and only rounded to the type the
#               optimizer stores it in: the part no arithmetic can remove
# grad gap      ‖ĝ − g‖₂ / ‖g‖₂ — where the error enters
# noise         ‖g_B − g_B′‖₂ / ‖g_B‖₂ between two batches: the scale
#
# The state is `make gpt-trained`'s weights plus 100 steps of the run in f64
# (the schedule's warm-up, so the measured step is at the peak learning rate
# and AdamW's second moment is not zero), the step is number 101 of 600, and
# its batch is two windows of the Code civil.
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
# What a gradient does with its forward pass's values (`--remat`, 0.10.0 step
# 4): `store`, the default, or `recompute`, the policy of the runs D48–D55
# recorded (the re-measurement after defect 3f, D93).
REMAT=${REMAT:-store}
MODEL=corpus/nlp/05_gpt.psk
W=${W:-out/gpt_trained/final.psk.d}
PROF=${PROF:-out/gpt_prof.json}
OUT=${1:-out/step_table.md}
[ -f "$W" ] || { echo "no trained weights at $W: make gpt-trained"; exit 1; }
[ -f "$PROF" ] || { echo "no machine profile at $PROF: presk bench --calibrate on the corpus GPT"; exit 1; }
STEP=(--const D=64 --const H=2 --const L=2 --weights "$W" --warmup 100 --steps 600
      --samples 2 --from 4000 --bits 128)

# TABLE_ONLY=1 rebuilds the table from the outputs of a previous run.
if [ "${TABLE_ONLY:-0}" != 1 ]; then
"$PRESK" calibrate --step "$MODEL" "${STEP[@]}" --profile "$PROF" --remat "$REMAT" \
    --realize f64 --realize f32 --realize bf16 --realize recipe \
    --out out/step_env.json 2>/dev/null | tee out/step_calibrate.txt
for b in 1e-3 1e-2; do
    "$PRESK" solve --step "$MODEL" --precision out/step_env.json --budget "$b" --bits 128 --remat "$REMAT" \
        --profile "$PROF" --out "out/step_$b.json" 2>/dev/null | tee "out/step_solve_$b.txt"
done
# The recipe as a file, for the course-scale timing (j_step.sh): the forward
# and backward functions in bf16, the optimizer (step and schedule) in f32.
python3 - <<'EOF'
import json
env = json.load(open('out/step_env.json'))
fns = sorted(f for f in {s.rsplit('/', 1)[0] for s in env['sites']} if not f.startswith('adamw.'))
json.dump({'default': 'f32', 'order': {'kind': 'sequential'}, 'functions': {f: 'bf16' for f in fns}},
          open('out/step_recipe.json', 'w'), indent=2)
EOF
fi

NOISE=$(awk '/^  noise/ {print $2}' out/step_calibrate.txt)
{
  echo "| realization | update gap | storage floor | grad gap | predicted s/step | vs all-f64 | inside 1e-3? | inside 1e-2? |"
  echo "|---|---|---|---|---|---|---|---|"
  base=$(awk '$1 == "f64" {print $5}' out/step_calibrate.txt)
  for r in f64 f32 bf16 recipe; do
      awk -v r="$r" -v base="$base" '$1 == r {printf "| %s | %s | %s | %s | %s | %.2fx | %s | %s |\n", r, $2, $3, $4, $5, base / $5, ($2 <= 1e-3 ? "yes" : "NO"), ($2 <= 1e-2 ? "yes" : "NO")}' out/step_calibrate.txt
  done
  for b in 1e-3 1e-2; do
      f=out/step_solve_$b.txt
      gap=$(awk '/^  measured/ {print $2}' "$f")
      floor=$(awk '/^  measured/ {gsub(/[),]/, "", $7); print $7}' "$f")
      grad=$(awk '/^  measured/ {print $8}' "$f")
      cost=$(awk '/^  cost/ {print $2}' "$f")
      chosen=$(awk '/^  chosen/ {print $2" f64, "$4" f32, "$6" bf16"}' "$f")
      echo "| solved at $b ($chosen) | $gap | $floor | $grad | $cost | $(python3 -c "print(f'{$base / $cost:.2f}x')") | $(python3 -c "print('yes' if $gap <= 1e-3 else 'NO')") | $(python3 -c "print('yes' if $gap <= 1e-2 else 'NO')") |"
  done
  echo
  echo "One step (101 of 600) of the corpus GPT at D = 64 from $W plus 100 warm-up steps in f64,"
  echo "on two windows of the Code civil; against the exact step at 128 bits from the same state rounded"
  echo "to the realization's default. Sampling noise between two such batches: $NOISE."
  echo "Predicted time: the per-dtype roofline $PROF over the step's sites."
} > "$OUT"
echo
cat "$OUT"
