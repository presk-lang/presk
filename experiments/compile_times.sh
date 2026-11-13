#!/bin/bash
# The time the compiler's commands take per model: `presk check`,
# `presk calibrate --precision` (the magnitude envelope,
# measured on the exact interpreter), `presk solve` (a realization under a
# 1e-3 budget, verified by measurement) and the exact reference alone (`presk
# run --bits 128` on one sample), each a wall time of one command, start-up
# included, on four models:
#
#   gpt-course     the course GPT at the course's constants (D = 384, 6 heads, 6 blocks)
#   gpt-reduced    the course GPT at the precision targets' constants (D = 64, 2, 2),
#                  on `make gpt-trained`'s weights where they exist
#   minigpt        the tiny MiniGPT of the examples (the Makefile's TINY constants)
#   bert-large-L2  BERT-large at MLPerf's constants with two of its 24 blocks, at its
#                  initialization, on its SQuAD loss (`squad_loss`, a scalar for `presk
#                  solve`, D101) over SQuAD-shaped features (experiments/bert.sh's) with
#                  an answer's span (bert_features.py --span)
#   llama-7b-L<N>  the Llama 2 decoder of examples/coverage/llama.psk at the 7B configuration's width
#                  (G = 32), N of its 32 blocks, at its initialization, on its next-token
#                  loss (`loss`) over random token ids (experiments/llama.sh's, T = LLAMA_T,
#                  default 384; D104)
#
# It is a measurement: run it on the measurement machine when it is quiet, like
# `make j-step`, and never while another experiment runs. The report goes to
# docs/compile-times-<date>.md (by the model key in its first column).
#
#   MODELS="gpt-reduced minigpt" ./experiments/compile_times.sh   a subset
#   SAMPLES=2                                                      the envelope's batch (default 2)
#   DATE=2026-10-02                                                the report's date (default today, UTC)
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
MODELS=${MODELS:-gpt-course gpt-reduced minigpt bert-large-L2}
SAMPLES=${SAMPLES:-2}
DATE=${DATE:-$(date -u +%F)}
OUT=${1:-docs/compile-times-$DATE.md}
D=${CT_OUT:-out/compile_times}
mkdir -p "$D"
[ -x "$PRESK" ] || { echo "no $PRESK: cargo build --release -p presk"; exit 1; }

GPT=examples/corpus/nlp/05_gpt.psk
CIVIL=examples/corpus/nlp/data/civil_T256.psk.d
W=out/gpt_trained/final.psk.d
TINY=(--const V=64 --const D=16 --const H=2 --const L=2)

# BERT's features, as experiments/bert.sh makes them, with the answer's span.
if [[ " $MODELS " == *" bert-large-L2 "* ]] && [ ! -s out/bert/features_loss.psk.d ]; then
    mkdir -p out/bert
    python3 experiments/bert_features.py 30522 384 20 0 --span > out/bert/features_loss.txt
    "$PRESK" data pack --type '(idx[30522][seq: 384], idx[2][seq: 384], ℝ[seq2: 384], idx[384][ends: 2])' out/bert/features_loss.txt -o out/bert/features_loss.psk.d >/dev/null
fi

# timed NAME CMD...: runs the command, keeps its output in $D/NAME.txt, prints the wall seconds;
# /usr/bin/time's report, the peak resident memory with it, goes to $D/NAME.time (D104).
timed() {
    local name=$1; shift
    local s e
    s=$(date +%s.%N)
    /usr/bin/time -v -o "$D/$name.time" "$@" > "$D/$name.txt" 2>&1
    local rc=$?
    e=$(date +%s.%N)
    [ $rc -eq 0 ] || { echo "  $name failed (exit $rc): see $D/$name.txt" >&2; echo "failed"; return; }
    python3 -c "print(f'{$e - $s:.3f}')"
}

# per model: file, constants, entry, dims, data, weights (or -)
describe() {
    case $1 in
        gpt-course)    FILE=$GPT; CONSTS=(--const D=384 --const H=6 --const L=6); ENTRY=loss; DIMS=(--dim T=256); DATA=$CIVIL; WEIGHTS=-;
                       LABEL="the course GPT, course constants (D = 384, H = 6, L = 6, T = 256)";;
        gpt-reduced)   FILE=$GPT; CONSTS=(--const D=64 --const H=2 --const L=2); ENTRY=loss; DIMS=(--dim T=256); DATA=$CIVIL;
                       WEIGHTS=$([ -s "$W" ] && echo "$W" || echo -);
                       LABEL="the course GPT, reduced constants (D = 64, H = 2, L = 2, T = 256)$([ "$WEIGHTS" != - ] && echo ', trained weights')";;
        minigpt)       FILE=examples/spec/minigpt/train_minigpt.psk; CONSTS=("${TINY[@]}"); ENTRY=loss; DIMS=(--dim B=2 --dim T=8); DATA=-; WEIGHTS=-;
                       LABEL="the tiny MiniGPT (V = 64, D = 16, H = 2, L = 2, B = 2, T = 8)";;
        bert-large-L2) FILE=examples/coverage/bert.psk; CONSTS=(--const L=2); ENTRY=squad_loss; DIMS=(--dim T=384); DATA=out/bert/features_loss.psk.d; WEIGHTS=-;
                       LABEL="BERT-large at MLPerf's constants, L = 2 of 24, one feature, its SQuAD loss";;
        llama-7b-L*)   FILE=examples/coverage/llama.psk; CONSTS=(--const G="${LLAMA_G:-32}" --const L="${1#llama-7b-L}" ${LLAMA_CONSTS:-}); ENTRY=loss; DIMS=(--dim T="${LLAMA_T:-384}");
                       DATA=${LLAMA_OUT:-out/llama}/ids_loss_T${LLAMA_T:-384}.psk.d; WEIGHTS=-;
                       LABEL="the Llama 2 decoder at the 7B configuration's width, L = ${1#llama-7b-L} of 32, T = ${LLAMA_T:-384}, its next-token loss";;
        *) echo "unknown model $1"; exit 1;;
    esac
}

declare -A CHECK CALIBRATE SOLVE EXACT LABELS
for m in $MODELS; do
    describe "$m"; LABELS[$m]=$LABEL
    data=(); [ "$DATA" != - ] && data=(--data "$DATA" --samples "$SAMPLES")
    weights=(); [ "$WEIGHTS" != - ] && weights=(--weights "$WEIGHTS")
    one=(); [ "$DATA" != - ] && one=(--data "$DATA" --samples 1)
    echo "== $m: $LABEL"
    CHECK[$m]=$(timed "${m}_check" "$PRESK" check "$FILE" "${CONSTS[@]}")
    echo "  presk check            ${CHECK[$m]} s"
    CALIBRATE[$m]=$(timed "${m}_calibrate" "$PRESK" calibrate --precision "$FILE" --entry "$ENTRY" "${CONSTS[@]}" "${DIMS[@]}" \
        "${data[@]}" "${weights[@]}" --bits 128 --out "$D/${m}_env.json")
    echo "  presk calibrate        ${CALIBRATE[$m]} s  (the envelope on $SAMPLES sample(s) at 128 bits)"
    SOLVE[$m]=$(timed "${m}_solve" "$PRESK" solve "$FILE" --precision "$D/${m}_env.json" --budget 1e-3 --bits 128 \
        "${weights[@]}" --sites 0 --out "$D/${m}_solved.json")
    echo "  presk solve            ${SOLVE[$m]} s  (budget 1e-3, verified by measurement)"
    EXACT[$m]=$(timed "${m}_exact" "$PRESK" run "$FILE" --entry "$ENTRY" "${CONSTS[@]}" "${DIMS[@]}" "${one[@]}" "${weights[@]}" --bits 128)
    echo "  the exact reference    ${EXACT[$m]} s  (one sample at 128 bits)"
done

{
  echo "> The time the compiler's commands take per model, measured on $DATE by \`make compile-times\`"
  echo "> (\`experiments/compile_times.sh\`) on presk \`$(git rev-parse --short HEAD 2>/dev/null)\` (\`$("$PRESK" --version 2>/dev/null | head -1)\`),"
  echo "> on $(uname -m), $(nproc) cores. Wall seconds of one command each, start-up included: \`presk check\`;"
  echo "> \`presk calibrate --precision\` (the magnitude envelope of the entry on $SAMPLES sample(s) at 128 bits);"
  echo "> \`presk solve --budget 1e-3\` (a realization proposed and verified by measurement against the reference"
  echo "> at 128 bits); and the exact reference alone, \`presk run --bits 128\` on one sample. The first column is"
  echo "> the model's key."
  echo
  echo "# Compile and solve times, $DATE"
  echo
  echo "| model | constants | presk check (s) | presk solve (s) | presk calibrate (s) | exact reference (s) |"
  echo "|---|---|---|---|---|---|"
  for m in $MODELS; do
      echo "| $m | ${LABELS[$m]} | ${CHECK[$m]} | ${SOLVE[$m]} | ${CALIBRATE[$m]} | ${EXACT[$m]} |"
  done
  echo
  echo "The outputs of every command are under \`$D/\` (not committed). A time is of the"
  echo "command as a reader would run it, the front end, the JIT and the reading of the data"
  echo "included; \`presk solve\` includes the reference run it verifies against."
} > "$OUT"
echo
cat "$OUT"
