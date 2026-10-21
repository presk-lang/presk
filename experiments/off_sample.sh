#!/bin/bash
# The certificate off-sample against the size of the calibration sample
# (D156), on D154–D155's program: the course GPT's logits at D = 64 on
# `make curve`'s all-f64 weights (`make margins`' model). The solver's
# measured claims are claims on the envelope's sample, the first N held-out
# windows; the realizations are judged on the 461 − N others.
#
#   1. the envelopes: `presk calibrate --precision` at 128 bits on the first
#      N = SIZES held-out windows (`--from 0 --samples N`), and with POP=1
#      the population, all 461 windows, by chunks of 32 calibrated apart and
#      folded together (experiments/off_sample.py merge: the envelope's
#      fields are maxima, minima and element counts, so the fold is exact;
#      checked on a chunk of 16 against N = 32);
#   2. `presk solve --budget-abs` on each envelope at ε*(10⁻³) = 4.67·10⁻⁴
#      and at 10⁻⁶, in three forms: `--rounding unit --narrow off` (the
#      solver of record), `--rounding measured --narrow off` and `--rounding
#      measured --narrow measured` (D155); and the space, every decision
#      listed (a solve at 10³ unverified on the 4-window envelope);
#   3. every distinct realization run natively on the 461 held-out windows
#      by experiments/margins.sh (a realization two solves found is run
#      once; D155's rows reused where the realization is one of them);
#   4. the report, OUT/report.md (experiments/off_sample.py report): the gap
#      split by window index, in sample (the N calibration windows) and out
#      of sample (the 461 − N others), the flips, the bound #{m < 2ε} and
#      the certification on the out-of-sample windows only.
# Nothing here is timed of record: a calibration's or a solve's wall time
# is a note. Outputs present are reused: delete OUT to start again. Needs
# `make margins` (MARGINS) and, for its rows, `make solver-rounding` (SR).
#
#   experiments/off_sample.sh [OUT]     default out/off-sample
#   SIZES="1 2 4 8 16 32" POP=1 BUDGETS="4.67e-4 1e-6" JOBS=12 MARGINS=out/margins SR=out/solver-rounding-0.15.0
set -u
cd "$(dirname "$0")/.."
OUT=${1:-out/off-sample}
PRESK=${PRESK:-./target/release/presk}
PY=${PY:-python3}
W=${W:-out/curve_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
MARGINS=${MARGINS:-out/margins}
SR=${SR:-out/solver-rounding-0.15.0}
SIZES=${SIZES:-1 2 4 8 16 32}
POP=${POP:-1}
BUDGETS=${BUDGETS:-4.67e-4 1e-6}
FORMS=${FORMS:-unit-off measured-off measured-measured}
JOBS=${JOBS:-12}
# The narrowing's margin (0.16.0, D161): D155's and D156's narrowing kept a
# step while the gap was inside the budget itself, a margin of 1; since
# 0.16.0 `--narrow measured` keeps it under the budget over `--margin`, 2.5
# by default, so the forms of record pass 1.
NARROW_MARGIN=${NARROW_MARGIN:-1}
CHUNK=32
FILE=experiments/margins.psk
CONSTS=(--const D=64 --const H=2 --const L=2)
mkdir -p "$OUT/env" "$OUT/solve"
say() { echo "== $(date +%H:%M:%S) $*"; }
for f in "$W" "$VAL" "$MARGINS/exact/0000.hilo" "$MARGINS/val.json"; do
    [ -e "$f" ] || { echo "missing $f: run make curve and make margins first"; exit 1; }
done
echo "presk $("$PRESK" --version) at $(git rev-parse --short HEAD)" > "$OUT/machine.txt"
uname -srm >> "$OUT/machine.txt"
WINDOWS=$($PY -c "import json, sys; print(int(json.load(open(sys.argv[1]))['count']))" "$MARGINS/val.json")

running=0
spawn() { "$@" & running=$((running + 1)); if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi; }
calibrate() { # name from samples
    local e=$OUT/env/$1
    [ -f "$e.json" ] && return 0
    /usr/bin/time -f "%e s, %M KB" "$PRESK" calibrate --precision "$FILE" --entry logits "${CONSTS[@]}" --dim T=256 \
        --weights "$W" --data "$VAL" --from "$2" --samples "$3" --bits 128 --out "$e.json.part" > "$e.txt" 2> "$e.time" \
        && mv "$e.json.part" "$e.json" || echo "the calibration $1 failed: $e.time"
}

say "1. the envelopes"
# the largest first, so that the pool's last jobs are short
for n in $(echo $SIZES | tr ' ' '\n' | sort -rn); do spawn calibrate "$n" 0 "$n"; done
if [ "$POP" = 1 ]; then
    spawn calibrate chunk-16 16 16
    for ((f = CHUNK; f < WINDOWS; f += CHUNK)); do
        spawn calibrate "chunk-$f" "$f" "$(( WINDOWS - f < CHUNK ? WINDOWS - f : CHUNK ))"
    done
fi
wait; running=0
for n in $SIZES; do [ -f "$OUT/env/$n.json" ] || { echo "missing the envelope on $n windows"; exit 1; }; done
if [ "$POP" = 1 ]; then
    "$PY" experiments/off_sample.py merge "$OUT/env/check-32.json" "$OUT/env/16.json" "$OUT/env/chunk-16.json" || exit 1
    "$PY" experiments/off_sample.py same "$OUT/env/check-32.json" "$OUT/env/32.json" > "$OUT/env/check-32.txt"; true
    cat "$OUT/env/check-32.txt"
    "$PY" experiments/off_sample.py merge "$OUT/env/all.json" "$OUT/env/32.json" \
        $(for ((f = CHUNK; f < WINDOWS; f += CHUNK)); do echo "$OUT/env/chunk-$f.json"; done) || exit 1
fi
[ -f "$SR/env_logits.json" ] && { "$PY" experiments/off_sample.py same "$OUT/env/4.json" "$SR/env_logits.json" > "$OUT/env/check-4.txt"; true; } \
    && cat "$OUT/env/check-4.txt"

say "2. the solves"
[ -f "$OUT/space.json" ] || "$PRESK" solve "$FILE" --precision "$OUT/env/4.json" --budget-abs 1e3 --no-verify --weights "$W" \
    --sites 100 --out "$OUT/space.json" > "$OUT/space.txt" 2>&1 || exit 1
solve() { # n budget rounding narrow
    local s=$OUT/solve/$1-$2-$3-$4
    [ -f "$s.json" ] && return 0
    /usr/bin/time -f "%e s, %M KB" "$PRESK" solve "$FILE" --precision "$OUT/env/$1.json" --budget-abs "$2" --weights "$W" \
        --sites 100 --rounding "$3" --narrow "$4" --margin "$NARROW_MARGIN" --out "$s.json.part" > "$s.txt" 2> "$s.time" \
        && mv "$s.json.part" "$s.json" || echo "the solve $1-$2-$3-$4 failed: $s.time"
}
# the narrowing's solves on the largest samples first: they are the longest
for f in measured-measured unit-off measured-off; do
    case " $FORMS " in *" $f "*) ;; *) continue ;; esac
    for n in $(echo $SIZES | tr ' ' '\n' | sort -rn); do
        for b in $BUDGETS; do spawn solve "$n" "$b" "${f%-*}" "${f#*-}"; done
    done
done
wait; running=0
for f in "$OUT"/solve/*.txt; do
    echo "$(basename "$f" .txt): $(grep -o '([0-9.]*x)' "$f" | head -1) $(grep -o 'measured  *[0-9.e+-]*' "$f" | head -1) $(grep -o '[0-9]* decision(s) narrowed.*in all' "$f")"
done

say "3. the 461 windows"
M=$OUT/margins
mkdir -p "$M/native" "$M/decisions" "$M/realizations"
MA=$(cd "$MARGINS" && pwd)
ln -sfn "$MA/exact" "$M/exact"
ln -sf "$MA/val.json" "$M/val.json"
# `make margins`' rows, and D155's where a realization is one of its,
# linked: margins.sh finds them and runs only the others; it writes its
# int*.json itself, never linked.
for r in "$MA"/native/*/; do
    r=$(basename "$r")
    [ -e "$M/native/$r" ] || ln -sfn "$MA/native/$r" "$M/native/$r"
    for e in txt loss; do [ -e "$MA/decisions/$r.$e" ] && ln -sf "$MA/decisions/$r.$e" "$M/decisions/$r.$e"; done
done
"$PY" experiments/off_sample.py unique "$OUT" --sr "$SR" > "$OUT/realizations.tsv" || exit 1
EXTRA=""
while IFS=$'\t' read -r name solves d155; do
    if [ -n "$d155" ] && [ -d "$SR/margins/native/$d155" ]; then
        SA=$(cd "$SR/margins" && pwd)
        [ -e "$M/native/$name" ] || ln -sfn "$SA/native/$d155" "$M/native/$name"
        for e in txt loss; do [ -e "$SA/decisions/$d155.$e" ] && ln -sf "$SA/decisions/$d155.$e" "$M/decisions/$name.$e"; done
    fi
    EXTRA="$EXTRA $name=@$M/realizations/$name.json"
done < "$OUT/realizations.tsv"
EXTRA="$EXTRA" OUT="$M" W="$W" VAL="$VAL" PRESK="$PRESK" experiments/margins.sh > "$M/margins.log" 2>&1 || { tail "$M/margins.log"; exit 1; }
sed -n '/^| realization/,/^$/p' "$M/margins.md" | cut -d'|' -f2,4,7,9,12,13,14

say "4. the report"
"$PY" experiments/off_sample.py report "$OUT" --sizes "$SIZES" --budgets "$BUDGETS" --forms "$FORMS" --sr "$SR" > "$OUT/report.md" || exit 1
say "wrote $OUT/report.md"
