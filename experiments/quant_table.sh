#!/bin/bash
# The gap table of plan 0.6.0 step 5d: every realization of the corpus GPT's
# loss measured against the exact denotation, the gap split in two.
#
# representation — the exact denotation on the dequantised weights against
#                  the exact denotation on the true ones: rounding the
#                  weights, and nothing else. Zero for a float realization.
# arithmetic     — the realization run natively on those weights against
#                  the exact denotation on them: the machine's arithmetic.
# total          — the native run against the exact denotation on the true
#                  weights, which is what a budget is checked against (D41).
#
# Only a compiler that keeps an exact reference can separate the first two;
# a quantisation toolchain sees their sum. Statement K is the `int8` row's
# total against a 1e-2 budget.
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
MODEL=examples/corpus/nlp/05_gpt.psk
# The dimensions the weights were trained at: `pos : ℝ[T, D]` makes T part of
# the parameters, and a checkpoint opened at another T is refused (D48).
# Two windows of the Code civil: a trained model's loss on random tokens is not
# the loss it was trained for. The run saw every window during training (three
# epochs), so this is a statement about numerics, not about generalisation.
ARGS=(--entry loss --const D=64 --const H=2 --const L=2 --dim T=256
      --data examples/corpus/nlp/data/civil_T256.psk.d --samples 2 --from 4000)
W=${W:-out/gpt_trained/final.psk.d}
OUT=${1:-out/quant_table.md}
MATS='["tok","pos","head","blocks.attn.wq.w","blocks.attn.wk.w","blocks.attn.wv.w","blocks.attn.wo.w","blocks.ff.up.w","blocks.ff.down.w"]'

realization() { # bits scheme -> a file quantising every matrix, f32 elsewhere
    python3 -c "import json,sys; print(json.dumps({'default':'f32','order':{'kind':'sequential'},'quant':{p:{'bits':int(sys.argv[1]),'over':-2,'scheme':sys.argv[2]} for p in json.loads(sys.argv[3])}}))" "$1" "$2" "$MATS" > "out/quant_$1_$2.json"
    echo "@out/quant_$1_$2.json"
}

{
  echo "| realization | weights (bytes) | representation | arithmetic | total | inside 1e-2? |"
  echo "|---|---|---|---|---|---|"
} > "$OUT"
row() { # label realize-spec
    r=$("$PRESK" quantize "$MODEL" "${ARGS[@]}" --weights "$W" --realize "$2" --backend native 2>/dev/null)
    bytes=$(echo "$r" | awk '/^  weights/ {print $2}')
    rep=$(echo "$r" | awk '/representation/ {print $3}' | tr -d '(')
    ari=$(echo "$r" | awk '/arithmetic/ {print $3}' | tr -d '(')
    tot=$(echo "$r" | awk '/total/ {print $3}' | tr -d '(')
    inside=$(python3 -c "print('yes' if float('$tot') <= 1e-2 else 'NO')")
    echo "| $1 | $bytes | $rep | $ari | $tot | $inside |" >> "$OUT"
}
row "f64" f64
row "f32" f32
row "bf16" bf16
row "int8, f32 elsewhere" "$(realization 8 absmax)"
row "int8 pow2, f32 elsewhere" "$(realization 8 pow2)"
row "int4, f32 elsewhere" "$(realization 4 absmax)"
row "int4 pow2, f32 elsewhere" "$(realization 4 pow2)"
echo >> "$OUT"
echo "Relative to the exact loss on two windows of the Code civil; every matrix quantised over its second-to-last axis." >> "$OUT"
K=$(awk -F'|' '/^\| int8, f32/ {gsub(/ /,"",$6); print $6}' "$OUT")
echo "Statement K (int8 inside a 1e-2 budget, by measurement): total $K — $(python3 -c "print('HOLDS' if float('$K') <= 1e-2 else 'FAILS')")" >> "$OUT"
cat "$OUT"
