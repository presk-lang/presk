#!/bin/bash
# ε from the margins at the course constants (D129): a copy of
# experiments/margins.sh for the course GPT at D = 384, H = 6, L = 6, on the
# weights trained at those constants (out/course_train_f64, 600 steps in
# f64), the realizations of experiments/curve_course.sh
# (out/curve-course_<b>.json) in place of make curve's, into
# out/margins-course. The exact chunks' JSON (30 digits) are kept: at these
# constants they are hours of the reference (D101: 178.6 s a window on one
# X925 core), not margins.sh's four minutes. Otherwise margins.sh's text:
#
# ε from the margins (D118; the argument is presk-project's
# docs/presk-epsilon-from-quality-2026-10-01.md §1). The course GPT decides
# argmax_v y_v(x) at every position; a realization ŷ with ‖ŷ − y‖∞ ≤ ε can
# change that decision only where the margin m = y₍₁₎ − y₍₂₎ is below 2ε. So
# the exact model's margins on held-out data, taken once, bound the decisions
# any realization inside ε changes: ε*(δ) is half the δ-quantile of the
# margins, corrected for the sample (Clopper–Pearson, one-sided), and every
# realization whose gap on the logits is at most ε*(δ) flips at most a
# fraction δ of the decisions, with confidence 1 − α.
#
# The model and the data are `make foil`'s: the all-f64 run of `make curve`
# (600 steps on 90 % of the Code civil), the 461 held-out windows of 256,
# 118 016 positions. Then
#   1. the exact logits, `logits` of experiments/margins.psk on the reference
#      at 128 bits, once, by chunks of CHUNK windows, JOBS at a time; their
#      margins, ε*(δ) and the fraction of margins below 2ε;
#   2. per realization (the foil's rows and the curve's seven), the native
#      logits against the exact ones: the largest gap, its mean, the argmax;
#   3. per realization `decisions` natively on every window, the flips
#      against the exact decisions, the predicted bound #{m < 2 ε_meas},
#      the accuracy; and beside them the gap to the exact loss on two
#      held-out windows, relative, the foil's own measure (`presk quantize`).
# Nothing is timed. Outputs present are reused, as every experiment script
# here does: delete OUT to start again.
#
#   DELTAS="1e-2 1e-3 1e-4" ALPHA=0.05 ./experiments/margins.sh
#   WINDOWS=32 OUT=out/margins-32 ./experiments/margins.sh     the first 32 windows only
#   EXTRA="logits-4.67e-4=@out/v6_solved.json" ./experiments/margins.sh
#       further realizations, NAME=SPEC each, measured as the fourteen and
#       labelled by their names (plan 0.12.0's V6: a realization solved on
#       the logits)
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
PY=${PY:-python3}
FILE=experiments/margins.psk
MODEL=corpus/nlp/05_gpt.psk
DIMS=(--const D=384 --const H=6 --const L=6 --dim T=256)
W=${W:-out/course_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
SEED=${SEED:-0}
BITS=${BITS:-128}
DELTAS=${DELTAS:-1e-2 1e-3 1e-4}
ALPHA=${ALPHA:-0.05}
WINDOWS=${WINDOWS:-}           # empty: every window of VAL
CHUNK=${CHUNK:-32}
JOBS=${JOBS:-8}
THREADS=${THREADS:-2}          # a native run's workers, JOBS runs at a time
OUT=${OUT:-out/margins-course}
BUDGETS=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)
# The course curve's realizations, CURVE_<b>.json (plan 0.16.1: 0.16.0's
# moved aside into out/pre-0.16.1/ are run again from there on the further
# texts).
CURVE=${CURVE:-out/curve-course}

for f in "$W" "$VAL"; do [ -f "$f" ] || { echo "missing $f: train at the course constants first"; exit 1; }; done
# EXACT_ONLY=1: stage 1 alone, the exact logits, which need no realization
# and take hours here, so that they can run while curve_course.sh solves.
if [ "${EXACT_ONLY:-0}" != 1 ]; then
for b in "${BUDGETS[@]}"; do
    [ -f "${CURVE}_$b.json" ] || { echo "missing ${CURVE}_$b.json: run experiments/curve_course.sh first"; exit 1; }
done
fi
mkdir -p "$OUT/exact" "$OUT/native" "$OUT/decisions" "$OUT/json"
[ -f "$OUT/val.json" ] || "$PRESK" data export "$VAL" --out "$OUT/val.json" >/dev/null || exit 1
N=$($PY -c "import json, sys; print(int(json.load(open(sys.argv[1]))['count']))" "$OUT/val.json")
[ -n "$WINDOWS" ] && N=$WINDOWS

# The foil's realizations (experiments/foil.sh): every matrix quantised, f32
# elsewhere.
MATS='["tok","pos","head","blocks.attn.wq.w","blocks.attn.wk.w","blocks.attn.wv.w","blocks.attn.wo.w","blocks.ff.up.w","blocks.ff.down.w"]'
realization() {
    python3 -c "import json,sys; print(json.dumps({'default':'f32','order':{'kind':'sequential'},'quant':{p:{'bits':int(sys.argv[1]),'over':-2,'scheme':sys.argv[2]} for p in json.loads(sys.argv[3])}}))" "$1" "$2" "$MATS" > "$OUT/int$1_$2.json"
    echo "@$OUT/int$1_$2.json"
}
NAMES=(f64 f32 bf16 int8 int8-pow2 int4 int4-pow2)
LABELS="f64|f32|bf16|int8, f32 elsewhere|int8 pow2|int4, f32 elsewhere|int4 pow2"
SPECS=(f64 f32 bf16 "$(realization 8 absmax)" "$(realization 8 pow2)" "$(realization 4 absmax)" "$(realization 4 pow2)")
for b in "${BUDGETS[@]}"; do
    NAMES+=("curve-$b"); LABELS="$LABELS|solved at $b"; SPECS+=("@${CURVE}_$b.json")
done
for nr in ${EXTRA:-}; do
    NAMES+=("${nr%%=*}"); LABELS="$LABELS|${nr%%=*}"; SPECS+=("${nr#*=}")
done

# A pool of JOBS background jobs.
running=0
spawn() { "$@" & running=$((running + 1)); if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi; }
drain() { wait; running=0; }
chunks() { for ((f = 0; f < N; f += CHUNK)); do echo "$f:$(( N - f < CHUNK ? N - f : CHUNK ))"; done; }

# A chunk's JSON as kept: `presk run --json` prints the entry's inputs with
# its result, here the 10.8 M parameters at 30 digits (623 MB a chunk, the
# same in every chunk and in out/course_train_f64 already); the parameter
# tree θ is dropped, the tokens, the key, the header and the logits kept.
slim() {
    python3 -c "import json, os, sys; d = json.load(open(sys.argv[1])); d['inputs'] = {k: v for k, v in d['inputs'].items() if k != 'θ'}; json.dump(d, open(sys.argv[1] + '.part', 'w')); os.replace(sys.argv[1] + '.part', sys.argv[1])" "$1"
}

# 1. The exact logits, once.
exact() { # from count
    local p; p=$OUT/exact/$(printf %04d "$1")
    [ -f "$p.tsv" ] && return 0
    "$PRESK" run "$FILE" --entry logits "${DIMS[@]}" --weights "$W" --data "$VAL" --from "$1" --samples "$2" \
        --seed "$SEED" --bits "$BITS" --json --digits 30 > "$OUT/json/exact_$1.json" 2> "$p.err" \
        && $PY experiments/margins.py exact "$OUT/json/exact_$1.json" "$p" && slim "$OUT/json/exact_$1.json" && rm -f "$p.err" \
        || echo "the exact run of windows $1.. failed: $p.err"
}
for fc in $(chunks); do spawn exact "${fc%:*}" "${fc#*:}"; done; drain
[ "${EXACT_ONLY:-0}" = 1 ] && exit 0

# 2. The native logits, per realization.
native() { # name spec from count
    local p; p=$OUT/native/$1/$(printf %04d "$3")
    [ -f "$p.tsv" ] && return 0
    "$PRESK" run "$FILE" --entry logits "${DIMS[@]}" --weights "$W" --data "$VAL" --from "$3" --samples "$4" \
        --seed "$SEED" --backend native --threads "$THREADS" --realize "$2" --json --digits 17 > "$OUT/json/$1_$3.json" 2> "$p.err" \
        && $PY experiments/margins.py native "$OUT/json/$1_$3.json" "$OUT/exact/$(printf %04d "$3")" "$p" \
        && rm -f "$OUT/json/$1_$3.json" "$p.err" || echo "the $1 run of windows $3.. failed: $p.err"
}
# 3. Its decisions on every window at once, and its gap to the exact loss.
decide() { # name spec
    local d=$OUT/decisions/$1.txt
    [ -f "$d" ] || { "$PRESK" run "$FILE" --entry decisions "${DIMS[@]}" --weights "$W" --data "$VAL" --samples "$N" \
        --seed "$SEED" --backend native --threads "$THREADS" --realize "$2" --json > "$OUT/json/dec_$1.json" 2> "$d.err" \
        && $PY experiments/margins.py decisions "$OUT/json/dec_$1.json" "$d" && rm -f "$OUT/json/dec_$1.json" "$d.err" \
        || echo "the $1 decisions failed: $d.err"; }
    [ -f "$OUT/decisions/$1.loss" ] || "$PRESK" quantize "$MODEL" --entry loss "${DIMS[@]}" --weights "$W" --data "$VAL" \
        --samples 2 --from 0 --realize "$2" --backend native 2>/dev/null | awk '/total/ {print $3}' | tr -d '(' > "$OUT/decisions/$1.loss"
}
for i in "${!NAMES[@]}"; do
    mkdir -p "$OUT/native/${NAMES[$i]}"
    for fc in $(chunks); do spawn native "${NAMES[$i]}" "${SPECS[$i]}" "${fc%:*}" "${fc#*:}"; done
    spawn decide "${NAMES[$i]}" "${SPECS[$i]}"
done
drain

GAPS=$(for n in "${NAMES[@]}"; do v=$(cat "$OUT/decisions/$n.loss" 2>/dev/null); echo -n "${v:-—} "; done)
$PY experiments/margins.py report "$OUT" --val "$OUT/val.json" --names "${NAMES[*]}" --labels "$LABELS" \
    --loss-gaps "$GAPS" --deltas "$DELTAS" --alpha "$ALPHA" --out "$OUT/margins.md"
