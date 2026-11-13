#!/bin/bash
# The estimate's rounding term measured on the envelope, and a narrowing
# step decided by measurement (D155), on D154's program: the course GPT's
# logits at D = 64 on `make curve`'s all-f64 weights (`make margins`' model).
#
#   1. the envelope by V6's protocol (D124): `presk calibrate --precision` on
#      the first 4 held-out windows at 128 bits, which since D155 also
#      records each site's largest rounding error per type;
#   2. `presk solve --budget-abs` at ε*(10⁻³) = 4.67·10⁻⁴ and at 10⁻⁶, in four
#      forms each: `--rounding unit` (the solver of record, V6's realization),
#      `--rounding measured`, and each with `--narrow measured`; and the space,
#      every decision listed (a solve at 10³ unverified);
#   3. every realization run natively on the 461 held-out windows by
#      experiments/margins.sh, beside D154's rows (the searches of
#      Precimonious and CRAFT on presk's space, linked from TUNING), with
#      `make margins`' report and `presk check --precision`'s bound;
#   4. the report, OUT/report.md (experiments/solver_rounding.py).
# Nothing here is timed of record: a solve's wall time is a note. Outputs
# present are reused: delete OUT to start again. Needs `make margins`
# (MARGINS) and D154's outputs (TUNING) for its rows and its export.
#
#   experiments/solver_rounding.sh [OUT]     default out/solver-rounding
#   BUDGETS="4.67e-4 1e-6" JOBS=4 MARGINS=out/margins TUNING=out/precision-tuning-0.15.0
set -u
cd "$(dirname "$0")/.."
OUT=${1:-out/solver-rounding}
PRESK=${PRESK:-./target/release/presk}
PY=${PY:-python3}
W=${W:-out/curve_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
MARGINS=${MARGINS:-out/margins}
TUNING=${TUNING:-out/precision-tuning-0.15.0}
BUDGETS=${BUDGETS:-4.67e-4 1e-6}
JOBS=${JOBS:-4}
# The narrowing's margin (0.16.0, D161): D155's narrowing kept a step while
# the gap was inside the budget itself, a margin of 1; since 0.16.0
# `--narrow measured` keeps it under the budget over `--margin`, 2.5 by
# default, so the forms of record pass 1.
NARROW_MARGIN=${NARROW_MARGIN:-1}
FILE=examples/experiments/margins.psk
CONSTS=(--const D=64 --const H=2 --const L=2)
mkdir -p "$OUT/solve"
say() { echo "== $(date +%H:%M:%S) $*"; }
for f in "$W" "$VAL" "$MARGINS/exact/0000.hilo" "$MARGINS/val.json" "$TUNING/c-f64/model.h"; do
    [ -e "$f" ] || { echo "missing $f: run make curve, make margins and make precision-tuning first"; exit 1; }
done
echo "presk $("$PRESK" --version) at $(git rev-parse --short HEAD)" > "$OUT/machine.txt"
uname -srm >> "$OUT/machine.txt"

say "1. the envelope"
[ -f "$OUT/env_logits.json" ] || /usr/bin/time -f "%e s, %M KB" "$PRESK" calibrate --precision "$FILE" --entry logits "${CONSTS[@]}" --dim T=256 \
    --weights "$W" --data "$VAL" --samples 4 --bits 128 --out "$OUT/env_logits.json" > "$OUT/calibrate.txt" 2> "$OUT/calibrate.time" || exit 1
sed -n '/largest rounding error/,$p' "$OUT/calibrate.txt"

say "2. the solves"
running=0
for b in $BUDGETS; do
    for r in unit measured; do
        for n in off measured; do
            s=$OUT/solve/$b-$r-$n
            [ -f "$s.json" ] && continue
            /usr/bin/time -f "%e s, %M KB" "$PRESK" solve "$FILE" --precision "$OUT/env_logits.json" --budget-abs "$b" --weights "$W" \
                --sites 100 --rounding "$r" --narrow "$n" --margin "$NARROW_MARGIN" --out "$s.json" > "$s.txt" 2> "$s.time" &
            running=$((running + 1))
            if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
        done
    done
done
wait
[ -f "$OUT/space.json" ] || "$PRESK" solve "$FILE" --precision "$OUT/env_logits.json" --budget-abs 1e3 --no-verify --weights "$W" \
    --sites 100 --out "$OUT/space.json" > "$OUT/space.txt" 2>&1 || exit 1
for f in "$OUT"/solve/*.txt; do
    echo "$(basename "$f" .txt): $(grep -o '([0-9.]*x)' "$f" | head -1) $(grep -o 'measured  *[0-9.e+-]*' "$f" | head -1) $(grep -o '[0-9]* decision(s) narrowed.*in all' "$f")"
done

say "3. the 461 windows"
M=$OUT/margins
mkdir -p "$M/native" "$M/decisions" "$M/realizations"
MA=$(cd "$MARGINS" && pwd)
ln -sfn "$MA/exact" "$M/exact"
ln -sf "$MA/val.json" "$M/val.json"
# `make margins`' rows and D154's, linked: margins.sh finds the first and
# runs only the new ones; it writes its int*.json itself, never linked.
for r in "$MA"/native/*/; do
    r=$(basename "$r")
    [ -e "$M/native/$r" ] || ln -sfn "$MA/native/$r" "$M/native/$r"
    for e in txt loss; do [ -e "$MA/decisions/$r.$e" ] && ln -sf "$MA/decisions/$r.$e" "$M/decisions/$r.$e"; done
done
TA=$(cd "$TUNING/margins" && pwd)
for r in "$TA"/native/search-*; do
    r=$(basename "$r")
    [ -e "$M/native/d154-$r" ] || ln -sfn "$TA/native/$r" "$M/native/d154-$r"
    for e in txt loss; do [ -e "$TA/decisions/$r.$e" ] && ln -sf "$TA/decisions/$r.$e" "$M/decisions/d154-$r.$e"; done
    cp "$TA/realizations/$r.json" "$M/realizations/d154-$r.json"
done
EXTRA=""
for b in $BUDGETS; do
    for r in unit measured; do
        for n in off measured; do
            cp "$OUT/solve/$b-$r-$n.json" "$M/realizations/solver-$b-$r-$n.json"
            EXTRA="$EXTRA solver-$b-$r-$n=@$M/realizations/solver-$b-$r-$n.json"
        done
    done
done
EXTRA="$EXTRA" OUT="$M" W="$W" VAL="$VAL" PRESK="$PRESK" experiments/margins.sh > "$M/margins.log" 2>&1 || { tail "$M/margins.log"; exit 1; }
NAMES="f64 f32 bf16"
LABELS="presk, all-f64|presk, all-f32|presk, all-bf16"
for b in $BUDGETS; do
    for r in unit measured; do
        for n in off measured; do
            NAMES="$NAMES solver-$b-$r-$n"; LABELS="$LABELS|the solver at $b, --rounding $r --narrow $n"
        done
    done
    for s in craft precimonious; do
        NAMES="$NAMES d154-search-$s-params-$b"; LABELS="$LABELS|D154, $s's search at $b, stored values and parameters"
    done
done
paste <(echo "$NAMES" | tr ' ' '\n') <(echo "$LABELS" | tr '|' '\n') > "$OUT/rows.tsv"
GAPS=$(for n in $NAMES; do v=$(cat "$M/decisions/$n.loss" 2>/dev/null); echo -n "${v:-—} "; done)
"$PY" experiments/margins.py report "$M" --val "$M/val.json" --names "$NAMES" --labels "$LABELS" --loss-gaps "$GAPS" --out "$OUT/margins.md"
sed -n '/^| realization/,/^$/p' "$OUT/margins.md" | cut -d'|' -f2,4,7,9,12,13,14

say "bounds"
{
    for r in f64 f32 bf16; do echo "$r $("$PRESK" check "$FILE" --precision "$OUT/env_logits.json" --realize "$r" | awk '/bound/ {print $3}')"; done
    for f in "$M"/realizations/*.json; do
        echo "$(basename "$f" .json) $("$PRESK" check "$FILE" --precision "$OUT/env_logits.json" --realize "@$f" | awk '/bound/ {print $3}')"
    done
} > "$OUT/bounds.txt"

say "4. the report"
"$PY" experiments/solver_rounding.py "$OUT" --budgets "$BUDGETS" --tuning "$TUNING" --weights "$W" > "$OUT/report.md" || exit 1
say "wrote $OUT/report.md"
