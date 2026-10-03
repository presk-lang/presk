#!/bin/bash
# The trajectory of plan 0.6.0 step 7b (D52): a per-step budget does not bound
# a training run, so the run is measured. The corpus GPT is trained from
# scratch for 600 steps under the step realization solved at 1e-3, exactly as
# curve.sh trained D50's rows (same initialisation, seed, batch order and
# data split), and compared with them:
#
# validation   loss on the held-out 10 % in f64, against all-f64 and two
#              other seeds (D50: the seed moves it a hundred times more than
#              any forward budget did)
# distance     ‖θ − θ_f64‖₂ / ‖θ_f64‖₂ of the final weights, against the
#              same distance between two seeds of all-f64
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
# What a gradient does with its forward pass's values (`--remat`, 0.10.0 step
# 4): `store`, the default, or `recompute`, the policy of the runs D48–D55
# recorded (the re-measurement after defect 3f, D93).
REMAT=${REMAT:-store}
MODEL=corpus/nlp/05_gpt.psk
CONSTS=(--const D=64 --const H=2 --const L=2)
SOLVED=${SOLVED:-out/step_1e-3.json}
STEPS=${STEPS:-600}
OUT=${1:-out/step_train.md}
for f in "$SOLVED" out/civil_train_T256.psk.d out/civil_val_T256.psk.d out/curve_train_f64/final.psk.d \
         out/curve_train_f64_seed1/final.psk.d out/curve_train_f64_seed2/final.psk.d; do
    [ -f "$f" ] || { echo "missing $f: run experiments/step_table.sh and experiments/curve.sh first"; exit 1; }
done
train() { # name realize-spec
    [ -f "out/curve_train_$1/final.psk.d" ] && return
    "$PRESK" train "$MODEL" "${CONSTS[@]}" --data out/civil_train_T256.psk.d \
        --steps "$STEPS" --epochs 10 --log-every 100 --backend native \
        --realize "$2" --remat "$REMAT" --ckpt "out/curve_train_$1" > "out/curve_train_$1.log" 2>&1
}
train step_1e-3 "@$SOLVED"
train step_recipe "@out/step_recipe.json"
val() {
    "$PRESK" run "$MODEL" --entry loss "${CONSTS[@]}" --dim T=256 --weights "out/curve_train_$1/final.psk.d" \
        --data out/civil_val_T256.psk.d --backend native --realize f64 --digits 8 2>/dev/null | awk '{print $NF}'
}
dist() { "$PRESK" data diff "out/curve_train_f64/final.psk.d" "out/curve_train_$1/final.psk.d" | awk '{print $10}'; }
{
  echo "| trained under | validation loss | vs all-f64 | ‖θ − θ_f64‖ / ‖θ_f64‖ |"
  echo "|---|---|---|---|"
  base=$(val f64)
  echo "| all-f64 | $base | | 0 |"
  for n in step_1e-3 step_recipe f64_seed1 f64_seed2; do
      v=$(val "$n")
      echo "| $n | $v | $(python3 -c "print(f'{$v - $base:+.2e}')") | $(dist "$n") |"
  done
  echo
  echo "$STEPS steps from scratch on 90 % of the Code civil at D = 64 (seed 0 unless named), loss on the other 10 % in f64."
} > "$OUT"
cat "$OUT"
