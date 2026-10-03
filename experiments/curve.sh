#!/bin/bash
# The ε-to-quality curve (plan 0.5.0 §3.5, step 8; its quality axis is plan
# 0.6.0 step 7).
#
# One program, one parameter varying. That is the whole point: the
# specification's complaint about the literature is that every quantisation
# level is a different pipeline with different code and different bugs, so a
# sweep compares artefacts rather than precisions. Here the program is
# identical at every point and only the budget moves.
#
# Per budget: what the solver chose, what it costs, what the propagated bound
# says, and what the distance to the exact reference actually is — and then
# the quality axis: the model trained from scratch *under that realization*
# on 90 % of the Code civil, and its loss on the held-out 10 %, evaluated in
# f64. The solver columns are measured on trained weights at T = 256, the
# dimension they were trained at (D48); the realizations are solved once, on
# those weights, and every training run starts from the same initialisation,
# seed and batch order, so the only thing that differs between two rows is
# the realization.
#
#   QUALITY=0 ./experiments/curve.sh   the solver columns only (minutes)
#   STEPS=600 JOBS=3                   the training runs (hours; the default)
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
# What a gradient does with its forward pass's values (`--remat`, 0.10.0 step
# 4): `store`, the default, or `recompute`, the policy of the runs D48–D55
# recorded (the re-measurement after defect 3f, D93).
REMAT=${REMAT:-store}
MODEL=corpus/nlp/05_gpt.psk
CONSTS=(--const D=64 --const H=2 --const L=2)
DIMS=(--dim B=2 --dim T=256)
ENV=out/curve_env.json
W=${W:-out/gpt_trained/final.psk.d}
OUT=${1:-out/curve.md}
QUALITY=${QUALITY:-1}
STEPS=${STEPS:-600}
JOBS=${JOBS:-3}
BUDGETS=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)

[ -f "$W" ] || { echo "no trained weights at $W: make gpt-trained"; exit 1; }

"$PRESK" calibrate --precision "$MODEL" --entry loss "${CONSTS[@]}" "${DIMS[@]}" \
    --bits 128 --remat "$REMAT" --out "$ENV" >/dev/null 2>&1 || { echo "calibrate failed"; exit 1; }
# The bound line reads `bound <rule> <value> (<relative> relative)` since
# 0.6.0 step 6: take the first number, not the second field (the rule's name).
BOUND=$("$PRESK" check --precision "$ENV" "$MODEL" --realize f64 --model cond 2>/dev/null \
    | awk '$1 == "bound" {for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]/) {print $i; exit}}')

# The solver columns.
declare -A NARROWED COST RATIO GAP INSIDE
for b in "${BUDGETS[@]}"; do
    r=$("$PRESK" solve "$MODEL" --precision "$ENV" --budget "$b" --weights "$W" \
        --out "out/curve_$b.json" --sites 0 --remat "$REMAT" 2>/dev/null)
    NARROWED[$b]=$(echo "$r" | awk '/^  sites/ {print $4}')
    COST[$b]=$(echo "$r" | awk '/^  cost/ {print $2}')
    RATIO[$b]=$(echo "$r" | awk '/^  cost/ {print $6}' | tr -d '(),')
    GAP[$b]=$(echo "$r" | awk '/^  measured/ {print $2}')
    INSIDE[$b]=$(echo "$r" | grep -q "inside the budget" && echo "yes" || echo "NO")
    printf "%-6s narrowed %-4s cost %-10s %-7s gap %-10s %s\n" "$b" "${NARROWED[$b]}" "${COST[$b]}" "${RATIO[$b]}" "${GAP[$b]}" "${INSIDE[$b]}"
done

# The quality axis: a training run per realization, and all-f64 as the
# baseline every row is compared with.
declare -A VAL
if [ "$QUALITY" = 1 ]; then
    [ -f out/civil_train_T256.psk.d ] && [ -f out/civil_val_T256.psk.d ] || {
        python3 -c "
t = open('corpus/nlp/data/civil_tokens.txt').read().split()
cut = len(t) * 9 // 10
open('out/civil_train.txt', 'w').write(' '.join(t[:cut]) + '\n')
open('out/civil_val.txt', 'w').write(' '.join(t[cut:]) + '\n')"
        "$PRESK" data lm-windows --vocab 92 --seq 256 out/civil_train.txt -o out/civil_train_T256.psk.d >/dev/null
        "$PRESK" data lm-windows --vocab 92 --seq 256 out/civil_val.txt -o out/civil_val_T256.psk.d >/dev/null
    }
    threads=$(( $(nproc) / JOBS )); [ "$threads" -ge 1 ] || threads=1
    train() { # name realize-spec
        "$PRESK" train "$MODEL" "${CONSTS[@]}" --data out/civil_train_T256.psk.d \
            --steps "$STEPS" --epochs 10 --log-every 100 --backend native --threads "$threads" \
            --realize "$2" --remat "$REMAT" --ckpt "out/curve_train_$1" > "out/curve_train_$1.log" 2>&1
    }
    names=(f64 "${BUDGETS[@]}")
    running=0
    for n in "${names[@]}"; do
        if [ "$n" = f64 ]; then spec=f64; else spec="@out/curve_$n.json"; fi
        if [ ! -f "out/curve_train_$n/final.psk.d" ]; then
            train "$n" "$spec" &
            running=$((running + 1))
            if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
        fi
    done
    wait
    for n in "${names[@]}"; do
        VAL[$n]=$("$PRESK" run "$MODEL" --entry loss "${CONSTS[@]}" --dim T=256 \
            --weights "out/curve_train_$n/final.psk.d" --data out/civil_val_T256.psk.d \
            --backend native --realize f64 --digits 8 2>/dev/null | awk '{print $NF}')
        echo "trained under $n: validation loss ${VAL[$n]}"
    done
fi

{
  echo "| budget | sites narrowed | cost | vs all-f64 | measured gap | inside? | bound (f64, cond) | validation loss | vs all-f64 |"
  echo "|---|---|---|---|---|---|---|---|---|"
  [ "$QUALITY" = 1 ] && echo "| all-f64 | 0 | | 1.00x | | | | ${VAL[f64]} | |"
  for b in "${BUDGETS[@]}"; do
      q=""; d=""
      if [ "$QUALITY" = 1 ]; then
          q=${VAL[$b]}
          d=$(python3 -c "print(f'{float(\"$q\") - float(\"${VAL[f64]}\"):+.2e}')")
      fi
      echo "| $b | ${NARROWED[$b]} | ${COST[$b]} | ${RATIO[$b]} | ${GAP[$b]} | ${INSIDE[$b]} | $BOUND | $q | $d |"
  done
  echo
  echo "Solver columns on $W at T = 256 (random tokens, B = 2); quality: $STEPS steps"
  echo "from scratch on 90 % of the Code civil under each realization, loss on the other 10 % in f64."
} > "$OUT"
echo
cat "$OUT"
