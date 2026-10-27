#!/bin/bash
# The third and fourth texts (plan 0.16.1 step 3, §7; D171): the Code du
# travail and the Code de procédure pénale through the Code civil's models,
# realizations and ε*, as D168 took the Code pénal (experiments/
# second_corpus.sh), on the realizations of 0.16.1 (D170) and of 0.16.0 for
# comparison. Nothing is calibrated, solved, trained or timed.
#
#   TEXT=travail ./experiments/further_texts.sh                every stage
#   TEXT=procedure_penale ./experiments/further_texts.sh data d64
#
#   1. data: the text from the container that holds civil.md.zip, its sha256
#      the one fixed when it was first fetched (2026-10-22); the typographic
#      characters mapped and the rest outside the Code civil's vocabulary
#      deleted (experiments/second_corpus.py map: both counts per character,
#      and how many fall in the judge set); the tokens on civil_vocab.json,
#      its windows of 256, and the judge set, its first 461 windows, cut from
#      the same stream and checked against the whole file's;
#   2. d64: experiments/margins.sh on the D = 64 model of record with the
#      judge set: the exact logits, the uniform and integer rows, v0.10.0's
#      curve, and 0.16.1's realizations (V6 and the trained curve's), into
#      out/margins-<TEXT>; then 0.16.0's under the same names into
#      out/margins-<TEXT>-0.16.0, the exact logits and the unmoved rows
#      linked;
#   3. exact: the course GPT's exact logits on the judge set, D129's way
#      (chunks of 4 windows, 12 at a time, then regrouped by 32), into
#      out/margins-course-<TEXT>; alone: no other exact run shares its cores;
#   4. course: margins_course.sh's native stages on 0.16.1's course
#      realizations (the logits at ε*, the trained curve's, D129's curve),
#      then on 0.16.0's into out/margins-course-<TEXT>-0.16.0;
#   5. report: each text against the Code civil, per scale and per version,
#      into out/further-<TEXT>/.
#
# Outputs present are reused, as every experiment script here does.
set -u
cd "$(dirname "$0")/.."
export PRESK=${PRESK:-out/bin/presk-0.16.1-rules}
# 0.16.0's realizations run on the binary that wrote their rows on the Code
# civil (D163), as D168's did; Z0 (D169) found the native bits the same.
PRESK016=${PRESK016:-out/bin/presk-22dd345}
PY=${PY:-python3}
TEXT=${TEXT:?TEXT=travail or TEXT=procedure_penale}
case $TEXT in
    travail) SHA=df267bfea80613ce36a14864ba3cbeff7a8234da32849d14f1bbbd023afdd8a5            # travail.md, 2026-10-22
        NAME="Code du travail"; SHORT=travail ;;
    procedure_penale) SHA=0937f8892e6b27adfa1b4d37609d5a14a1062b86ffe0e277af7da6fcf0a3a1ae   # procedure_penale.md, 2026-10-22
        NAME="Code de procédure pénale"; SHORT="proc. pénale" ;;
    *) echo "TEXT is travail or procedure_penale, not $TEXT"; exit 2 ;;
esac
URL=https://storage.gra.cloud.ovh.net/v1/AUTH_4d7d1bcd41914ee184ef80e2c75c4fb1/dila-legi-codes/$TEXT.md.zip
D=corpus/nlp/data
VOCAB=$D/civil_vocab.json
JUDGE=out/${TEXT}_461_T256.psk.d
N=461
R=out/further-$TEXT
M64=out/margins-$TEXT
MC=out/margins-course-$TEXT
W64=out/curve_train_f64/final.psk.d
WC=out/course_train_f64/final.psk.d
PRE=out/pre-0.16.1
BUDGETS=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)
UNMOVED=(f64 f32 bf16 int8 int4 int8-pow2 int4-pow2)
V10=(curve-1e-1 curve-1e-2 curve-1e-3 curve-1e-4 curve-1e-5 curve-1e-6 curve-1e-7)
say() { echo "== $(date -Is) $*"; }
mkdir -p "$R"

# The realizations of each version, by the names of record.
extra64() { # 0.16.1 | 0.16.0
    if [ "$1" = 0.16.1 ]; then
        local e="logits-4.67e-4=@out/v6-0.16.1/solved.json"
        for b in "${BUDGETS[@]}"; do e="$e d016-$b=@out/curve-trained_$b.json"; done
    else
        local e="logits-4.67e-4=@out/v6-0.16.0/solved.json"
        for b in "${BUDGETS[@]}"; do e="$e d016-$b=@$PRE/curve-trained_$b.json"; done
    fi
    echo "$e"
}
extrac() {
    if [ "$1" = 0.16.1 ]; then
        local e="logits-4.96e-4=@out/margins-course-0.16.1/logits/solved.json"
        for b in "${BUDGETS[@]}"; do e="$e ct-$b=@out/curve-course-trained_$b.json"; done
    else
        local e="logits-4.96e-4=@out/margins-course-0.16.0/logits/solved.json"
        for b in "${BUDGETS[@]}"; do e="$e ct-$b=@$PRE/curve-course-trained_$b.json"; done
    fi
    echo "$e"
}

# D163's: a directory with another's exact logits and unmoved rows linked.
link_rows() { # from to name...
    local from=$1 to=$2; shift 2
    mkdir -p "$to/native" "$to/decisions"
    local fa; fa=$(cd "$from" && pwd)
    ln -sfn "$fa/exact" "$to/exact"
    [ -e "$fa/val.json" ] && ln -sf "$fa/val.json" "$to/val.json"
    for r in "$@"; do
        [ -d "$fa/native/$r" ] && ln -sfn "$(readlink -f "$fa/native/$r")" "$to/native/$r"
        for e in txt loss; do [ -e "$fa/decisions/$r.$e" ] && ln -sf "$(readlink -f "$fa/decisions/$r.$e")" "$to/decisions/$r.$e"; done
    done
    return 0
}

data() {
    say "data: $TEXT"
    [ -f "$D/$TEXT.md" ] || $PY experiments/second_corpus.py fetch "$URL" "$TEXT.md" "$D/$TEXT.md" > "$R/fetch.txt" || exit 1
    echo "$SHA  $D/$TEXT.md" | sha256sum -c --quiet || { echo "$TEXT.md is not the text of 2026-10-22 (SHA=…)"; exit 1; }
    [ -f "$D/${TEXT}_civil92.md" ] || $PY experiments/second_corpus.py map "$D/$TEXT.md" "$VOCAB" "$D/${TEXT}_civil92.md" \
        "$R/mapped.tsv" $((N * 256 + 1)) > "$R/map.txt" || exit 1
    [ -f "$D/${TEXT}_tokens.txt" ] || "$PRESK" data encode "$D/${TEXT}_civil92.md" --vocab "$VOCAB" -o "$D/${TEXT}_tokens.txt" || exit 1
    [ -f "$D/${TEXT}_T256.psk.d" ] || "$PRESK" data lm-windows --vocab 92 --seq 256 "$D/${TEXT}_tokens.txt" -o "$D/${TEXT}_T256.psk.d" || exit 1
    if [ ! -f "$JUDGE" ]; then
        $PY experiments/second_corpus.py head "$D/${TEXT}_tokens.txt" $N 256 "$R/${TEXT}_461_tokens.txt" > "$R/head.txt" || exit 1
        "$PRESK" data lm-windows --vocab 92 --seq 256 "$R/${TEXT}_461_tokens.txt" -o "$JUDGE" || exit 1
        "$PRESK" data export "$D/${TEXT}_T256.psk.d" --out "$R/all.json" >/dev/null || exit 1
        "$PRESK" data export "$JUDGE" --out "$R/judge.json" >/dev/null || exit 1
        $PY experiments/second_corpus.py same "$R/all.json" "$R/judge.json" $N > "$R/same.txt" || { rm -f "$JUDGE"; exit 1; }
        rm -f "$R/all.json" "$R/judge.json" "$R/${TEXT}_461_tokens.txt"
    fi
    "$PRESK" data info "$D/${TEXT}_T256.psk.d" --samples 0 > "$R/${TEXT}_T256.info"
    "$PRESK" data info "$JUDGE" --samples 0 > "$R/${TEXT}_461.info"
    sha256sum "$D/$TEXT.md" "$D/${TEXT}_civil92.md" "$D/${TEXT}_tokens.txt" "$D/${TEXT}_T256.psk.d" "$JUDGE" "$VOCAB" > "$R/data.sha256"
}

d64() {
    say "D = 64: margins.sh on $TEXT, 0.16.1's realizations"
    EXTRA="$(extra64 0.16.1)" OUT="$M64" W="$W64" VAL="$JUDGE" JOBS=8 experiments/margins.sh > "$R/d64.log" 2>&1 || exit 1
    say "D = 64: 0.16.0's realizations"
    link_rows "$M64" "$M64-0.16.0" "${UNMOVED[@]}" "${V10[@]}"
    PRESK=$PRESK016 EXTRA="$(extra64 0.16.0)" OUT="$M64-0.16.0" W="$W64" VAL="$JUDGE" JOBS=8 experiments/margins.sh > "$R/d64-0.16.0.log" 2>&1 || exit 1
}

exact() {
    say "course: the exact logits on $TEXT, chunks of 4, 12 at a time"
    if [ ! -f "$MC/exact/0448.tsv" ]; then
        systemd-run --user --scope -q -p MemoryMax=100G -p MemorySwapMax=0 \
            env EXACT_ONLY=1 CHUNK=4 JOBS=12 OUT="$MC/c4" W="$WC" VAL="$JUDGE" \
            experiments/margins_course.sh > "$R/exact.log" 2>&1 || exit 1
        $PY experiments/margins_rechunk.py "$MC/c4/exact" "$MC/exact" --from-chunk 4 --to-chunk 32 --windows $N >> "$R/exact.log" || exit 1
    fi
}

course() {
    say "course: margins_course.sh's native stages on $TEXT, 0.16.1's realizations"
    systemd-run --user --scope -q -p MemoryMax=108G -p MemorySwapMax=0 \
        env EXTRA="$(extrac 0.16.1)" OUT="$MC" CHUNK=32 JOBS=8 W="$WC" VAL="$JUDGE" \
        experiments/margins_course.sh > "$R/course.log" 2>&1 || exit 1
    say "course: 0.16.0's realizations"
    link_rows "$MC" "$MC-0.16.0" "${UNMOVED[@]}"
    systemd-run --user --scope -q -p MemoryMax=108G -p MemorySwapMax=0 \
        env PRESK=$PRESK016 EXTRA="$(extrac 0.16.0)" CURVE="$PRE/curve-course" OUT="$MC-0.16.0" CHUNK=32 JOBS=8 W="$WC" VAL="$JUDGE" \
        experiments/margins_course.sh > "$R/course-0.16.0.log" 2>&1 || exit 1
}

report() {
    say "report: $TEXT"
    [ -f "$R/loss-d64.txt" ] || $PY experiments/margins_loss.py "$M64/exact" "$M64/val.json" > "$R/loss-d64.txt"
    local lc=out/second-corpus/loss-civil-d64.txt lcc=out/second-corpus/loss-civil-course.txt
    for v in 0.16.1 0.16.0; do
        local t=$M64 c=out/margins-$v
        [ $v = 0.16.0 ] && t=$M64-0.16.0
        $PY experiments/second_corpus.py report --civil "$c" --penal "$t" --vocab "$VOCAB" --label "D = 64, $v's realizations" --name "$NAME" --short "$SHORT" \
            --loss-civil "$lc" --loss-penal "$R/loss-d64.txt" --focus logits-4.67e-4 --windows-out "$R/windows-d64-$v" > "$R/d64-$v.md" || exit 1
    done
    if [ -f "$MC/margins.json" ]; then
        [ -f "$R/loss-course.txt" ] || $PY experiments/margins_loss.py "$MC/exact" "$MC/val.json" > "$R/loss-course.txt"
        for v in 0.16.1 0.16.0; do
            local t=$MC c=out/margins-course-$v
            [ $v = 0.16.0 ] && t=$MC-0.16.0
            [ -f "$t/margins.json" ] || continue
            $PY experiments/second_corpus.py report --civil "$c" --penal "$t" --vocab "$VOCAB" --label "the course constants, $v's realizations" --name "$NAME" --short "$SHORT" \
                --loss-civil "$lcc" --loss-penal "$R/loss-course.txt" --focus logits-4.96e-4 --windows-out "$R/windows-course-$v" > "$R/course-$v.md" || exit 1
        done
    fi
}

"$PRESK" --version > "$R/presk.txt"
for s in ${@:-data d64 exact course report}; do "$s"; done
say "done"
