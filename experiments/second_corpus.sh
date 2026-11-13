#!/bin/bash
# The certificate on a second text (presk-project
# docs/prompt-second-corpus-2026-10-22.md, docs/second-corpus-2026-10-22.md):
# the Code pénal through the Code civil's models, realizations and ε*,
# unchanged. Nothing is calibrated, solved, trained or timed.
#
#   1. data: the Code pénal from the course's container (the one that holds
#      civil.md.zip), the characters outside the Code civil's vocabulary
#      deleted and counted, the tokens on civil_vocab.json, its windows of
#      256 (`presk data lm-windows`, the Makefile's recipe), and the judge
#      set, the first 461 of them, as many as the Code civil's held-out;
#   2. d64: experiments/margins.sh on the D = 64 model of record
#      (out/curve_train_f64) with the judge set, every realization of
#      out/margins-0.16.0 (D163 §4), into out/margins-penal;
#   3. exact: the course GPT's exact logits on the judge set
#      (margins_course.sh's stage 1, D129's way: chunks of 4 windows, 12 at
#      a time, then regrouped by 32), into out/margins-course-penal;
#   4. course: margins_course.sh's native stages on every realization of
#      out/margins-course-0.16.0 (D163 §5);
#   5. window: the Code pénal's worst position at the course constants
#      under step 0b's variants, one chunk (below);
#   6. report: the Code civil's rows of record against the Code pénal's, per
#      scale, into out/second-corpus/.
#
#   ./experiments/second_corpus.sh                 every stage in order
#   ./experiments/second_corpus.sh data d64        some of them
#
# Outputs present are reused, as every experiment script here does.
set -u
cd "$(dirname "$0")/.."
export PRESK=${PRESK:-out/bin/presk-22dd345}
PY=${PY:-python3}
URL=${URL:-https://storage.gra.cloud.ovh.net/v1/AUTH_4d7d1bcd41914ee184ef80e2c75c4fb1/dila-legi-codes/penal.md.zip}
SHA=${SHA:-be64baff45de6e8dfc03704e5d37d21d4ff39559af64102139d1775c1539243d}   # penal.md, 2026-10-22
D=examples/corpus/nlp/data
VOCAB=$D/civil_vocab.json
JUDGE=${JUDGE:-out/penal_461_T256.psk.d}
N=461
R=out/second-corpus
M64=out/margins-penal
MC=out/margins-course-penal
BUDGETS=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)
say() { echo "== $(date -Is) $*"; }
mkdir -p "$R"

data() {
    say "data"
    [ -f "$D/penal.md" ] || $PY experiments/second_corpus.py fetch "$URL" penal.md "$D/penal.md" > "$R/fetch.txt" || exit 1
    echo "$SHA  $D/penal.md" | sha256sum -c --quiet || { echo "penal.md is not the text of 2026-10-22 (SHA=…)"; exit 1; }
    [ -f "$D/penal_civil92.md" ] || $PY experiments/second_corpus.py clean "$D/penal.md" "$VOCAB" "$D/penal_civil92.md" \
        "$R/deleted.tsv" > "$R/clean.txt" || exit 1
    [ -f "$D/penal_tokens.txt" ] || "$PRESK" data encode "$D/penal_civil92.md" --vocab "$VOCAB" -o "$D/penal_tokens.txt" || exit 1
    [ -f "$D/penal_T256.psk.d" ] || "$PRESK" data lm-windows --vocab 92 --seq 256 "$D/penal_tokens.txt" -o "$D/penal_T256.psk.d" || exit 1
    if [ ! -f "$JUDGE" ]; then
        $PY experiments/second_corpus.py head "$D/penal_tokens.txt" $N 256 "$R/penal_461_tokens.txt" > "$R/head.txt" || exit 1
        "$PRESK" data lm-windows --vocab 92 --seq 256 "$R/penal_461_tokens.txt" -o "$JUDGE" || exit 1
        "$PRESK" data export "$D/penal_T256.psk.d" --out "$R/penal_T256.json" >/dev/null || exit 1
        "$PRESK" data export "$JUDGE" --out "$R/penal_461.json" >/dev/null || exit 1
        $PY experiments/second_corpus.py same "$R/penal_T256.json" "$R/penal_461.json" $N > "$R/same.txt" || { rm -f "$JUDGE"; exit 1; }
        rm -f "$R/penal_T256.json" "$R/penal_461.json" "$R/penal_461_tokens.txt"
    fi
    "$PRESK" data info "$D/penal_T256.psk.d" --samples 0 > "$R/penal_T256.info"
    "$PRESK" data info "$JUDGE" --samples 0 > "$R/penal_461.info"
    sha256sum "$D/penal.md" "$D/penal_civil92.md" "$D/penal_tokens.txt" "$D/penal_T256.psk.d" "$JUDGE" "$VOCAB" > "$R/data.sha256"
}

d64() {
    say "D = 64: margins.sh on the Code pénal"
    local extra="logits-4.67e-4=@out/v6-0.16.0/solved.json"
    for b in "${BUDGETS[@]}"; do extra="$extra d016-$b=@out/curve-trained_$b.json"; done
    EXTRA="$extra" OUT="$M64" W=out/curve_train_f64/final.psk.d VAL="$JUDGE" JOBS=8 experiments/margins.sh > "$R/d64.log" 2>&1 || exit 1
}

exact() {
    say "course: the exact logits, chunks of 4, 12 at a time"
    if [ ! -f "$MC/exact/0448.tsv" ]; then
        systemd-run --user --scope -q -p MemoryMax=100G -p MemorySwapMax=0 \
            env EXACT_ONLY=1 CHUNK=4 JOBS=12 OUT="$MC/c4" W=out/course_train_f64/final.psk.d VAL="$JUDGE" \
            experiments/margins_course.sh > "$R/exact.log" 2>&1 || exit 1
        $PY experiments/margins_rechunk.py "$MC/c4/exact" "$MC/exact" --from-chunk 4 --to-chunk 32 --windows $N >> "$R/exact.log" || exit 1
    fi
}

course() {
    say "course: margins_course.sh's native stages on the Code pénal"
    local extra="logits-4.96e-4=@out/margins-course-0.16.0/logits/solved.json"
    for b in "${BUDGETS[@]}"; do extra="$extra ct-$b=@out/curve-course-trained_$b.json"; done
    systemd-run --user --scope -q -p MemoryMax=108G -p MemorySwapMax=0 \
        env EXTRA="$extra" OUT="$MC" CHUNK=32 JOBS=8 W=out/course_train_f64/final.psk.d VAL="$JUDGE" \
        experiments/margins_course.sh > "$R/course.log" 2>&1 || exit 1
}

# 5. window: the Code pénal's worst position at the course constants (window
#    261, position 176), whose chunk (windows 256-287) is run under step 0b's
#    variants of two realizations (experiments/off_sample_tail.py variants:
#    each bf16 decision back to f32, all of them, D129's types): D163 §5's
#    logits at ε* and the course curve's 10⁻⁴ (D129's curve, solved at
#    0.16.0). Native runs only, into out/second-corpus/w261.
window() {
    say "course: the worst window's chunk under step 0b's variants"
    local O=$R/w261 from=256 count=32 c=0256
    mkdir -p "$O/json" "$O/native"
    : > "$O/variants.tsv"
    for s in "logits=out/margins-course-0.16.0/logits/solved.json" "curve-1e-4=out/curve-course_1e-4.json"; do
        local n=${s%%=*} f=${s#*=}
        $PY experiments/off_sample_tail.py variants "$f" out/margins-course/logits/solved.json "$O/realizations/$n" \
            | sed "s/^/$n:/" >> "$O/variants.tsv" || exit 1
    done
    run() { # name realization
        local p=$O/native/$1
        [ -f "$p.tsv" ] && return 0
        "$PRESK" run examples/experiments/margins.psk --entry logits --const D=384 --const H=6 --const L=6 --dim T=256 \
            --weights out/course_train_f64/final.psk.d --data "$JUDGE" --from $from --samples $count --seed 0 \
            --backend native --threads 2 --realize "@$2" --json --digits 17 > "$O/json/$1.json" 2> "$p.err" \
            && $PY experiments/margins.py native "$O/json/$1.json" "$MC/exact/$c" "$p" && rm -f "$O/json/$1.json" "$p.err" \
            || echo "the $1 run failed: $p.err"
    }
    local running=0
    while IFS=$'\t' read -r name _; do
        run "$name" "$O/realizations/${name%%:*}/${name#*:}.json" &
        running=$((running + 1)); [ "$running" -ge 8 ] && { wait -n; running=$((running - 1)); }
    done < "$O/variants.tsv"
    wait
    $PY experiments/second_corpus.py chunk "$O/native" --first $from --window 261 --position 176 --what "$O/variants.tsv" > "$O/report.md"
}

report() {
    say "report"
    [ -f "$R/loss-civil-d64.txt" ] || $PY experiments/margins_loss.py out/margins/exact out/margins/val.json > "$R/loss-civil-d64.txt"
    [ -f "$R/loss-penal-d64.txt" ] || $PY experiments/margins_loss.py "$M64/exact" "$M64/val.json" > "$R/loss-penal-d64.txt"
    $PY experiments/second_corpus.py report --civil out/margins-0.16.0 --penal "$M64" --vocab "$VOCAB" --label "D = 64" \
        --loss-civil "$R/loss-civil-d64.txt" --loss-penal "$R/loss-penal-d64.txt" --focus logits-4.67e-4 --windows-out "$R/windows-d64" > "$R/d64.md" || exit 1
    if [ -f "$MC/margins.json" ]; then
        [ -f "$R/loss-civil-course.txt" ] || $PY experiments/margins_loss.py out/margins-course/exact out/margins-course/val.json > "$R/loss-civil-course.txt"
        [ -f "$R/loss-penal-course.txt" ] || $PY experiments/margins_loss.py "$MC/exact" "$MC/val.json" > "$R/loss-penal-course.txt"
        $PY experiments/second_corpus.py report --civil out/margins-course-0.16.0 --penal "$MC" --vocab "$VOCAB" \
            --label "the course constants" --loss-civil "$R/loss-civil-course.txt" --loss-penal "$R/loss-penal-course.txt" \
            --focus logits-4.96e-4 --windows-out "$R/windows-course" > "$R/course.md" || exit 1
    fi
}

"$PRESK" --version > "$R/presk.txt"
for s in ${@:-data d64 exact course window report}; do "$s"; done
say "done"
