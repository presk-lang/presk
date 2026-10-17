#!/bin/bash
# The curve's solver columns on the trained model's envelope (D132): a copy
# of experiments/curve.sh whose calibration reads the weights the solves
# read, `--weights W`, on two held-out windows of text (`--data
# out/civil_val_T256.psk.d --samples 2 --from 0`, windows 0 and 1), where
# curve.sh calibrates on the initialization and one batch of random tokens.
# A solve runs on its envelope's inputs, so the sensitivities and the
# measurement move to those two windows as well. The solves, the budgets and
# the table are curve.sh's; the realizations go
# to out/curve-trained_<b>.json and the envelope to
# out/curve-trained_env.json, so that nothing of make curve's is
# overwritten. QUALITY=1 adds curve.sh's quality axis: the model trained
# from scratch under each realization (less its `params`), 600 steps, into
# out/curve-trained_train_<b>, and its validation loss in f64; the all-f64
# baseline is make curve's own out/curve_train_f64 (the same
# initialization, seed and batch order), read and not trained again. A
# budget whose realization and solve report exist is not solved again.
# After the table, a second one gives each solve's parameters' types and
# widening rounds.
# Otherwise curve.sh's text:
#
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
#   STEPS=600 JOBS=3                   the training runs (half an hour since 0.10.0; the default)
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
ENV=out/curve-trained_env.json
W=${W:-out/gpt_trained/final.psk.d}
OUT=${1:-out/curve-trained.md}
QUALITY=${QUALITY:-0}
# A training run's checkpoint: make curve's for all-f64, this script's own
# for a realization.
ck() { if [ "$1" = f64 ]; then echo out/curve_train_f64; else echo "out/curve-trained_train_$1"; fi; }
VAL=${VAL:-out/civil_val_T256.psk.d}
STEPS=${STEPS:-600}
JOBS=${JOBS:-3}
BUDGETS=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)

[ -f "$W" ] || { echo "no trained weights at $W: make gpt-trained"; exit 1; }

[ -f "$ENV" ] || "$PRESK" calibrate --precision "$MODEL" --entry loss "${CONSTS[@]}" --dim T=256 \
    --weights "$W" --data "$VAL" --samples 2 --from 0 \
    --bits 128 --remat "$REMAT" --out "$ENV" > "${ENV%.json}.log" 2>&1 || { echo "calibrate failed"; exit 1; }
# The bound line reads `bound <rule> <value> (<relative> relative)` since
# 0.6.0 step 6: take the first number, not the second field (the rule's name).
BOUND=$("$PRESK" check --precision "$ENV" "$MODEL" --realize f64 --model cond 2>/dev/null \
    | awk '$1 == "bound" {for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]/) {print $i; exit}}')

# The solver columns.
declare -A NARROWED COST RATIO GAP INSIDE
for b in "${BUDGETS[@]}"; do
    if [ -f "out/curve-trained_$b.json" ] && [ -s "out/curve-trained_$b.txt" ]; then
        r=$(cat "out/curve-trained_$b.txt")
    else
        r=$("$PRESK" solve "$MODEL" --precision "$ENV" --budget "$b" --weights "$W" \
            --out "out/curve-trained_$b.json" --sites 0 --remat "$REMAT" 2>/dev/null | tee "out/curve-trained_$b.txt")
    fi
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
            --realize "$2" --remat "$REMAT" --ckpt "$(ck "$1")" > "$(ck "$1").log" 2>&1
    }
    names=(f64 "${BUDGETS[@]}")
    running=0
    for n in "${names[@]}"; do
        # A training run keeps its parameters in the default type: they are
        # the master weights it updates, and `presk train` refuses a
        # realization that types them (plan 0.12.0, §5). So it trains under
        # the solved realization less its `params`, the nests' types, which
        # is what every run before 0.12.0 trained under, its parameters
        # being in the default type then.
        if [ "$n" = f64 ]; then
            spec=f64
        else
            python3 -c "import json, sys; r = json.load(open(sys.argv[1])); r.pop('params', None); json.dump(r, open(sys.argv[2], 'w'), indent=2)" "out/curve-trained_$n.json" "out/curve-trained_$n.train.json"
            spec="@out/curve-trained_$n.train.json"
        fi
        if [ ! -f "$(ck "$n")/final.psk.d" ]; then
            train "$n" "$spec" &
            running=$((running + 1))
            if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
        fi
    done
    wait
    for n in "${names[@]}"; do
        VAL[$n]=$("$PRESK" run "$MODEL" --entry loss "${CONSTS[@]}" --dim T=256 \
            --weights "$(ck "$n")/final.psk.d" --data out/civil_val_T256.psk.d \
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
  echo "Solver columns on $W at T = 256: the envelope, the sensitivities and the measurement on"
  echo "windows 0 and 1 of $VAL, a solve running on its envelope's inputs (D132)."
  [ "$QUALITY" = 1 ] && echo "Quality: $STEPS steps from scratch on 90 % of the Code civil under each realization, loss on the other 10 % in f64;" \
      && echo "all-f64 is make curve's out/curve_train_f64."
  echo
  echo "| budget | stored values narrowed (bf16) | parameters typed (bf16) | estimate | widening rounds |"
  echo "|---|---|---|---|---|"
  for b in "${BUDGETS[@]}"; do
      python3 - "out/curve-trained_$b.json" "out/curve-trained_$b.txt" "$b" <<'PY'
import json, re, sys
r = json.load(open(sys.argv[1])); t = open(sys.argv[2]).read()
st, pa = list(r.get("storage", {}).values()), list(r.get("params", {}).values())
est = re.search(r"^  estimate\s+([0-9.e+-]+)", t, re.M).group(1)
rounds = re.search(r"after ([0-9]+) widening", t).group(1)
print(f"| {sys.argv[3]} | {len(st)} ({st.count('bf16')}) | {len(pa)} ({pa.count('bf16')}) | {est} | {rounds} |")
PY
  done
} > "$OUT"
echo
cat "$OUT"
