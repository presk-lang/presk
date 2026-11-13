#!/bin/bash
# SR6 of the short runs (2026-11-10,
# prompt-sr6-weights-width-2026-11-11.md): the weights' certified width. How
# many significant bits do the course GPT's weights need at rest for the
# all-f32 realization to stay certified at δ = 10⁻³? No type is added: a
# parameter is rounded once, when the program receives it, so a storage width
# is a transformation of the checkpoint. The trained weights of record
# (`make margins`' at D = 64, margins_course.sh's at the course constants) are
# rounded to k significant bits (experiments/weights_width.py round: f64 → f32
# → k bits, to nearest, ties to even, as the compiler rounds to bf16), and the
# tag's all-f32 realization runs on them natively, as margins.sh's steps 2 and
# 3 run a realization, against the exact logits of the UNROUNDED weights of
# record (nothing exact is computed again but the loss at D = 64 on two
# windows). Per scale:
#   0. the leaves (`presk dump --stage elab`), the windows, the record's exact
#      chunks linked;
#   1. the rounded checkpoints, variant M (the nine matrices of margins.sh's
#      MATS) and A (all twenty-one parameters) at each k of KS, checksummed;
#   2. per row the native logits by chunks against the exact ones (margins.py
#      native, each chunk's JSON checksummed before it is removed), the
#      decisions on every window, the loss on two windows; the rows are
#      all-f32 on the unrounded weights, the compiler's realizations with the
#      same parameters at rest in bf16 (SR6a's references, bf16p-M and
#      bf16p-A), and the ladder;
#   3. margins.py report on the rows (ε* of record from the exact margins);
#   4. the Code pénal's 461 windows for M at k = 16 and at the smallest
#      certified k, against its exact chunks of record (D168);
#   5. weights_width.py report: SR6a's checks, the table with the bytes, SR6b.
# Nothing is timed or solved. Every presk process runs in its own scope
# (MemoryMax=MEM, no swap, oom_score_adj 1000). Outputs present are reused:
# delete OUT to start again.
#
#   ./experiments/weights_width.sh
#   SCALES=d64 KS="8 16 24" ./experiments/weights_width.sh
#   REPORT_ONLY=1 ./experiments/weights_width.sh
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
PY=${PY:-python3}
OUT=${OUT:-out/weights-width}
MEM=${MEM:-108G}
read -ra SCALES <<< "${SCALES:-d64 course}"
read -ra KS <<< "${KS:-8 10 12 14 16 18 20 24}"
read -ra VARIANTS <<< "${VARIANTS:-M A}"
CHUNK=${CHUNK:-32}
JOBS=${JOBS:-8}
THREADS=${THREADS:-2}          # a native run's workers, JOBS runs at a time
SEED=0
FILE=examples/experiments/margins.psk
MODEL=examples/corpus/nlp/05_gpt.psk
VAL=${VAL:-out/civil_val_T256.psk.d}
PENAL=${PENAL:-out/penal_461_T256.psk.d}
declare -A CONSTS=([d64]="--const D=64 --const H=2 --const L=2" [course]="--const D=384 --const H=6 --const L=6")
declare -A WEIGHTS=([d64]=out/curve_train_f64/final.psk.d [course]=out/course_train_f64/final.psk.d)
# The rows of record: the exact chunks, all-f32's native rows, the Code
# pénal's exact chunks, a logits solve for the all-f64 traffic.
declare -A EXACT=([d64]=out/margins/exact [course]=out/margins-course/exact)
declare -A REC_F32=([d64]=out/margins/native/f32 [course]=out/margins-course/native/f32)
declare -A EXACT_PENAL=([d64]=out/margins-penal/exact [course]=out/margins-course-penal/exact)
declare -A SOLVE=([d64]=out/v6-0.16.1/solve.txt [course]=out/margins-course-0.16.1/logits/solve.txt)
# The exact loss of the unrounded weights on the two windows, when a run of
# record already has it (SR3's, D195: the same weights, windows and binary).
declare -A EXACT_LOSS=([course]=out/opt-measured/gap/course-loss-exact-plain.txt)
MATS='["tok","pos","head","blocks.attn.wq.w","blocks.attn.wk.w","blocks.attn.wv.w","blocks.attn.wo.w","blocks.ff.up.w","blocks.ff.down.w"]'

scoped() { systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 choom -n 1000 -- "$@"; }
say() { echo "== $(date -Is) $*"; }
running=0
spawn() { "$@" & running=$((running + 1)); if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi; }
drain() { wait; running=0; }

for s in "${SCALES[@]}"; do
    for f in "${WEIGHTS[$s]}" "$VAL" "${SOLVE[$s]}"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
    for d in "${EXACT[$s]}" "${REC_F32[$s]}"; do [ -d "$d" ] || { echo "missing $d"; exit 1; }; done
done

# One scale's rows on one data file: native chunks, decisions, the loss.
#   rows BASE SCALE DATA EXACTDIR N NAME=WEIGHTS=SPEC …
rows() {
    local base=$1 s=$2 data=$3 exact=$4 n=$5; shift 5
    local -a C; read -ra C <<< "${CONSTS[$s]}"
    local spec w name
    for nws in "$@"; do
        name=${nws%%=*}; spec=${nws##*=}; w=${nws#*=}; w=${w%=*}
        mkdir -p "$base/native/$name" "$base/decisions" "$base/json"
        for ((f = 0; f < n; f += CHUNK)); do
            spawn native_chunk "$base" "$name" "$w" "$spec" "$data" "$exact" "$f" $(( n - f < CHUNK ? n - f : CHUNK )) "${C[@]}"
        done
        spawn decide "$base" "$name" "$w" "$spec" "$data" "$n" "${C[@]}"
    done
    drain
}
native_chunk() { # base name weights spec data exact from count consts…
    local base=$1 name=$2 w=$3 spec=$4 data=$5 exact=$6 from=$7 count=$8; shift 8
    local p; p=$base/native/$name/$(printf %04d "$from")
    [ -f "$p.tsv" ] && return 0
    local j=$base/json/${name}_$from.json
    scoped "$PRESK" run "$FILE" --entry logits "$@" --dim T=256 --weights "$w" --data "$data" --from "$from" --samples "$count" \
        --seed "$SEED" --backend native --threads "$THREADS" --realize "$spec" --json --digits 17 --result-only > "$j" 2> "$p.err" \
        && $PY experiments/margins.py native "$j" "$exact/$(printf %04d "$from")" "$p" \
        && sha256sum < "$j" | cut -d' ' -f1 > "$p.json.sha256" && rm -f "$j" "$p.err" \
        || echo "the $name run of windows $from.. failed: $p.err"
}
decide() { # base name weights spec data n consts…
    local base=$1 name=$2 w=$3 spec=$4 data=$5 n=$6; shift 6
    local d=$base/decisions/$name.txt j=$base/json/dec_$name.json
    [ -f "$d" ] || { scoped "$PRESK" run "$FILE" --entry decisions "$@" --dim T=256 --weights "$w" --data "$data" --samples "$n" \
        --seed "$SEED" --backend native --threads "$THREADS" --realize "$spec" --json --result-only > "$j" 2> "$d.err" \
        && $PY experiments/margins.py decisions "$j" "$d" && rm -f "$j" "$d.err" || echo "the $name decisions failed: $d.err"; }
}
loss() { # base name weights spec consts…
    local base=$1 name=$2 w=$3 spec=$4; shift 4
    local f=$base/loss/$name.txt how
    [ -s "$f" ] && return 0
    if [ "$spec" = exact ]; then how=(--bits 128); else how=(--backend native --threads "$THREADS" --realize "$spec"); fi
    scoped "$PRESK" run "$MODEL" --entry loss "$@" --dim T=256 --weights "$w" --data "$VAL" --samples 2 --from 0 \
        --digits 40 "${how[@]}" > "$f" 2> "$f.err" && rm -f "$f.err" || echo "the $name loss failed: $f.err"
}
count() { $PY -c "import json, sys; print(int(json.load(open(sys.argv[1]))['count']))" "$1"; }

if [ "${REPORT_ONLY:-0}" != 1 ]; then
for s in "${SCALES[@]}"; do
    base=$OUT/$s
    read -ra C <<< "${CONSTS[$s]}"
    mkdir -p "$base/w" "$base/loss" "$base/realizations"
    say "$s: the leaves, the windows, the exact chunks of record"
    [ -s "$base/leaves.tsv" ] || { "$PRESK" dump --stage elab "$MODEL" "${C[@]}" --dim T=256 > "$base/dump-elab.txt" \
        && $PY experiments/weights_width.py leaves "$base/dump-elab.txt" --model GPT --dim T=256 > "$base/leaves.tsv"; } || exit 1
    [ -f "$base/val.json" ] || "$PRESK" data export "$VAL" --out "$base/val.json" >/dev/null || exit 1
    [ -e "$base/exact" ] || ln -s "$(realpath "${EXACT[$s]}")" "$base/exact"
    N=$(count "$base/val.json")

    say "$s: the rounded checkpoints"
    for v in "${VARIANTS[@]}"; do
        if [ "$v" = M ]; then names=$MATS; else names=$(cut -f1 "$base/leaves.tsv" | $PY -c "import json, sys; print(json.dumps([l.strip() for l in sys.stdin]))"); fi
        echo "$names" > "$base/realizations/names-$v.json"
        [ -f "$base/realizations/bf16p-$v.json" ] || $PY experiments/weights_width.py realization --names "$names" > "$base/realizations/bf16p-$v.json"
        for k in "${KS[@]}"; do
            ck=$base/w/$v-$k.psk.d
            [ -f "$ck" ] || $PY experiments/weights_width.py round "${WEIGHTS[$s]}" "$ck" --leaves "$base/leaves.tsv" --k "$k" --names "$names" \
                > "$base/w/$v-$k.json" || exit 1
        done
    done
    ( cd "$base/w" && sha256sum ./*.psk.d > ckpt.sha256 )

    say "$s: the rows on the Code civil ($N windows)"
    R=("f32=${WEIGHTS[$s]}=f32")
    for v in "${VARIANTS[@]}"; do R+=("bf16p-$v=${WEIGHTS[$s]}=@$base/realizations/bf16p-$v.json"); done
    for v in "${VARIANTS[@]}"; do for k in "${KS[@]}"; do R+=("$v-$k=$base/w/$v-$k.psk.d=f32"); done; done
    rows "$base" "$s" "$VAL" "${EXACT[$s]}" "$N" "${R[@]}"

    say "$s: the loss on two windows"
    if [ -n "${EXACT_LOSS[$s]:-}" ] && [ -s "${EXACT_LOSS[$s]}" ]; then
        [ -s "$base/loss/exact.txt" ] || { cp "${EXACT_LOSS[$s]}" "$base/loss/exact.txt"; echo "${EXACT_LOSS[$s]}" > "$base/loss/exact.from"; }
    else
        loss "$base" exact "${WEIGHTS[$s]}" exact "${C[@]}"
    fi
    for r in "${R[@]}"; do
        name=${r%%=*}; spec=${r##*=}; w=${r#*=}; w=${w%=*}
        spawn loss "$base" "$name" "$w" "$spec" "${C[@]}"
    done
    drain

    say "$s: margins.py report"
    names=""; labels=""; gaps=""
    for r in "${R[@]}"; do
        name=${r%%=*}; names="$names $name"; labels="$labels|$name"
        g=$($PY -c "import sys; l = lambda p: float(open(p).read().split('=')[1]); print(f'{abs(l(sys.argv[1]) - l(sys.argv[2])) / abs(l(sys.argv[2])):.3e}')" \
            "$base/loss/$name.txt" "$base/loss/exact.txt" 2>/dev/null || echo —)
        gaps="$gaps $g"
    done
    $PY experiments/margins.py report "$base" --val "$base/val.json" --names "${names# }" --labels "${labels#|}" \
        --loss-gaps "${gaps# }" --deltas "1e-2 1e-3 1e-4" --alpha 0.05 --out "$base/margins.md" > /dev/null || exit 1

    pk=(16 $($PY experiments/weights_width.py smallest "$base/margins.json" --variant M --ks "${KS[*]}"))
    if [ -f "$PENAL" ] && [ -d "${EXACT_PENAL[$s]}" ] && [[ " ${VARIANTS[*]} " == *" M "* ]]; then
        pb=$OUT/$s-penal
        mkdir -p "$pb"
        [ -f "$pb/val.json" ] || "$PRESK" data export "$PENAL" --out "$pb/val.json" >/dev/null || exit 1
        [ -e "$pb/exact" ] || ln -s "$(realpath "${EXACT_PENAL[$s]}")" "$pb/exact"
        NP=$(count "$pb/val.json")
        P=("f32=${WEIGHTS[$s]}=f32")
        for k in $(printf '%s\n' "${pk[@]}" | sort -nu); do
            [[ " ${KS[*]} " == *" $k "* ]] && P+=("M-$k=$base/w/M-$k.psk.d=f32")
        done
        say "$s: the Code pénal ($NP windows): ${P[*]%%=*}"
        rows "$pb" "$s" "$PENAL" "${EXACT_PENAL[$s]}" "$NP" "${P[@]}"
        names=""; labels=""
        for r in "${P[@]}"; do names="$names ${r%%=*}"; labels="$labels|${r%%=*}"; done
        $PY experiments/margins.py report "$pb" --val "$pb/val.json" --names "${names# }" --labels "${labels#|}" \
            --deltas "1e-2 1e-3 1e-4" --alpha 0.05 --out "$pb/margins.md" > /dev/null || exit 1
    fi
done
fi

rec=""; sol=""
for s in "${SCALES[@]}"; do rec="$rec $s=$(realpath "${REC_F32[$s]}")"; sol="$sol $s=${SOLVE[$s]}"; done
$PY experiments/weights_width.py report "$OUT" --scales "${SCALES[*]}" --ks "${KS[*]}" --variants "${VARIANTS[*]}" \
    --mats "$MATS" --record-f32 "${rec# }" --solve "${sol# }"
