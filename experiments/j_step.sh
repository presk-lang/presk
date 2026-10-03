#!/bin/bash
# Statement J′ of plan 0.6.0 (§3.6, step 7b, D52): at course scale, the step
# realization solved under a 1e-3 update budget against all-f64, all-f32 and
# the mixed-precision recipe — time per training step, and the update gap.
#
# time    per step, from the difference of a one- and a four-step run, so
#         that the front end, the JIT and the checkpoint are not counted (D50)
# gap     the update gap of one step natively against a native f64 step from
#         the same state: the exact step is not affordable at this size, and
#         f64 is within 1e-12 of it wherever both were measured (7b.1)
#
# The realizations are the ones step_table.sh solved at D = 64: a realization
# is keyed by function, not by shape, so it applies unchanged here.
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
# What a gradient does with its forward pass's values (`--remat`, 0.10.0 step
# 4): `store`, the default, or `recompute`, the policy of the runs D48–D55
# recorded (the re-measurement after defect 3f, D93).
REMAT=${REMAT:-store}
MODEL=corpus/nlp/05_gpt.psk
OUT=${1:-out/j_step.md}
SOLVED=${SOLVED:-out/step_1e-3.json}
RECIPE=${RECIPE:-out/step_recipe.json}
# A course-scale checkpoint for the gap: D50's probe (one f64 step from
# initialisation); the warm-up is shorter than at D = 64 since a step costs
# a minute here, so the learning rate is a tenth of its peak and the storage
# floor ten times higher — it is printed beside the gap.
CKPT=${CKPT:-out/course_probe_f64/final.psk.d}
for f in "$SOLVED" "$RECIPE" "$CKPT"; do [ -f "$f" ] || { echo "missing $f: run experiments/step_table.sh first"; exit 1; }; done

SPECS=(f64 "@$SOLVED" f32 "@$RECIPE")
# TABLE_ONLY=1 rebuilds the table from the outputs of a previous run.
if [ "${TABLE_ONLY:-0}" != 1 ]; then
declare -A T1 T4
timed() { # realize steps -> seconds
    local s e
    s=$(date +%s.%N)
    "$PRESK" train "$MODEL" --steps "$2" --backend native --realize "$1" --remat "$REMAT" --ckpt out/j_step_tmp >/dev/null 2>&1
    e=$(date +%s.%N)
    python3 -c "print(f'{$e - $s:.2f}')"
}
for spec in "${SPECS[@]}"; do
    T1[$spec]=$(timed "$spec" 1)
    T4[$spec]=$(timed "$spec" 4)
    echo "$spec steps=1 ${T1[$spec]} s, steps=4 ${T4[$spec]} s"
done | tee out/j_step_times.txt

"$PRESK" calibrate --step "$MODEL" --weights "$CKPT" --warmup 10 --samples 2 --from 4000 --remat "$REMAT" \
    --reference f64 --realize f64 --realize "@$SOLVED" --realize f32 --realize "@$RECIPE" \
    2>/dev/null | tee out/j_step_gaps.txt
fi

{
  echo "| realization | 1 step (s) | 4 steps (s) | per step (s) | vs all-f64 | update gap | storage floor | grad gap |"
  echo "|---|---|---|---|---|---|---|---|"
  base=$(python3 -c "print(($(awk '$1 == "f64" {print $6}' out/j_step_times.txt) - $(awk '$1 == "f64" {print $3}' out/j_step_times.txt)) / 3)")
  for spec in "${SPECS[@]}"; do
      t1=$(awk -v s="$spec" '$1 == s {print $3}' out/j_step_times.txt)
      t4=$(awk -v s="$spec" '$1 == s {print $6}' out/j_step_times.txt)
      per=$(python3 -c "print(f'{($t4 - $t1) / 3:.1f}')")
      ratio=$(python3 -c "print(f'{$base / (($t4 - $t1) / 3):.2f}x')")
      gaps=$(awk -v s="$spec" '$1 == s {print "| "$2" | "$3" | "$4}' out/j_step_gaps.txt)
      echo "| $spec | $t1 | $t4 | $per | $ratio $gaps |"
  done
  echo
  J=$(python3 -c "print(f'{$base / (($(awk -v s="@$SOLVED" '$1 == s {print $6}' out/j_step_times.txt) - $(awk -v s="@$SOLVED" '$1 == s {print $3}' out/j_step_times.txt)) / 3):.2f}')")
  G=$(awk -v s="@$SOLVED" '$1 == s {print $2}' out/j_step_gaps.txt)
  echo "Per step: the four-step time minus the one-step time, over three. Gaps: one step from $CKPT"
  echo "plus 10 warm-up steps in f64, on two windows, against a native f64 step from the same state."
  echo
  echo "Statement J′ (the step solved at 1e-3 at least 1.3x faster than all-f64 per step, and inside its budget):"
  echo "$J x, update gap $G — $(python3 -c "print('HOLDS' if $J >= 1.3 and $G <= 1e-3 else 'FAILS')")"
} > "$OUT"
cat "$OUT"
