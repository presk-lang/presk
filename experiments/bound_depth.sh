#!/bin/bash
# SR1, the short runs (2026-11-10;
# D193): the bound's distance to the estimate against depth. D51
# explains the nine orders between `cond` and the measured gap by `k`
# against `√k` over the sums in series; SR1 tests it as a law. The course
# GPT at D = 64, H = 2 with L = 1, 2, 4 and 6 blocks:
#
#   trained   by make gpt-trained's recipe (600 steps, 3 epochs, f32, seed 0),
#             into $OUT/train-L<L>/; L = 2 is make gpt-trained's own weights
#             (out/gpt_trained/final.psk.d), not trained again;
#   envelope  calibrated as make precision calibrates (two windows of the Code
#             civil, 128 bits), into $OUT/env-L<L>.json; L = 2 is make
#             precision's envelope of record (out/gpt_env.json), not taken again;
#   init      the same L at initialization (no weights, seed 0), calibrated
#             likewise into $OUT/env-init-L<L>.json: reported, no statement;
#   check     `presk check --model all --measure` under f64 and f32 on every
#             envelope: the four rules and the gap measured against the exact
#             reference, into $OUT/check-<trained|init>-L<L>-<type>.txt.
#
# experiments/bound_depth.py reads the checks into $OUT/report.md, with
# SR1a's and SR1b's verdicts. Each training, calibration and check runs
# alone in its own scope (MemoryMax 108G, no swap, oom_score_adj 1000),
# timed by /usr/bin/time -v. An output that exists is not taken again.
#
#   ./experiments/bound_depth.sh              everything, then the report
#   LS="1 4" ./experiments/bound_depth.sh     some depths
#   REPORT_ONLY=1 ./experiments/bound_depth.sh
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
OUT=${OUT:-out/bound-depth}
read -r -a LS <<< "${LS:-1 2 4 6}"
MEM=${MEM:-108G}
REMAT=${REMAT:-store}
MODEL=examples/corpus/nlp/05_gpt.psk
DATA=examples/corpus/nlp/data/civil_T256.psk.d
W2=out/gpt_trained/final.psk.d
ENV2=out/gpt_env.json

scoped() { systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 choom -n 1000 -- "$@"; }
say() { echo "== $(date -Is) $*"; }
consts() { echo --const D=64 --const H=2 --const L="$1"; }

for f in "$W2" "$ENV2" "$DATA"; do [ -f "$f" ] || { echo "missing $f (make gpt-trained, make precision)"; exit 1; }; done
mkdir -p "$OUT"

if [ "${REPORT_ONLY:-0}" != 1 ]; then
    say "SR1 on $("$PRESK" --version)"
    for L in "${LS[@]}"; do
        if [ "$L" = 2 ]; then
            W=$W2; ENV=$ENV2
        else
            W=$OUT/train-L$L/final.psk.d; ENV=$OUT/env-L$L.json
            if [ ! -f "$W" ]; then
                say "L = $L: training"
                # shellcheck disable=SC2046
                scoped /usr/bin/time -v -o "$OUT/train-L$L.time" \
                    "$PRESK" train "$MODEL" $(consts "$L") --steps 600 --epochs 3 --log-every 100 \
                    --backend native --realize f32 --remat "$REMAT" --ckpt "$OUT/train-L$L" \
                    > "$OUT/train-L$L.log" 2>&1 || { echo "training L = $L failed"; exit 1; }
            fi
            if [ ! -f "$ENV" ]; then
                say "L = $L: the envelope on the trained weights"
                # shellcheck disable=SC2046
                scoped /usr/bin/time -v -o "$OUT/env-L$L.time" \
                    "$PRESK" calibrate --precision "$MODEL" --entry loss $(consts "$L") --dim T=256 --bits 128 \
                    --data "$DATA" --samples 2 --from 0 --weights "$W" --out "$ENV" \
                    > "$OUT/env-L$L.log" 2>&1 || { echo "calibration L = $L failed"; exit 1; }
            fi
        fi
        INIT=$OUT/env-init-L$L.json
        if [ ! -f "$INIT" ]; then
            say "L = $L: the envelope at initialization"
            # shellcheck disable=SC2046
            scoped /usr/bin/time -v -o "$OUT/env-init-L$L.time" \
                "$PRESK" calibrate --precision "$MODEL" --entry loss $(consts "$L") --dim T=256 --bits 128 \
                --data "$DATA" --samples 2 --from 0 --out "$INIT" \
                > "$OUT/env-init-L$L.log" 2>&1 || { echo "calibration at initialization L = $L failed"; exit 1; }
        fi
        for kind in trained init; do
            if [ "$kind" = trained ]; then e=$ENV; else e=$INIT; fi
            for R in f64 f32; do
                c=$OUT/check-$kind-L$L-$R
                [ -s "$c.txt" ] && continue
                say "L = $L: check, $kind, $R"
                scoped /usr/bin/time -v -o "$c.time" \
                    "$PRESK" check --precision "$e" "$MODEL" --realize "$R" --model all --measure \
                    > "$c.txt" 2> "$c.err" || { echo "check $kind L = $L $R failed"; exit 1; }
            done
        done
    done
    say "done"
fi
python3 experiments/bound_depth.py "$OUT" > "$OUT/report.md"
cat "$OUT/report.md"
