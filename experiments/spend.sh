#!/bin/bash
# Three ways to spend the budget, judged out of sample (plan 0.16.0 step 4,
# D161): the logits of the course GPT at D = 64 on `make curve`'s all-f64
# weights (`make margins`' model), solved on the envelopes of the first N
# held-out windows and judged on the windows no form saw.
#
#   1. the envelopes: `presk calibrate --precision` at 128 bits on the first
#      N = SIZES held-out windows (`--from 0 --samples N`);
#   2. `presk solve --budget-abs` on each at ε*(10⁻³) = 4.67·10⁻⁴ and at
#      10⁻⁶, in seven forms (FORMS): `unit-off` (the solver of 0.12.0–0.15.0),
#      `measured-off` (0.16.0's default), `rss` (`--estimate rss`, the
#      proposal's estimate a root-sum-square aimed at budget/k), `narrow`
#      (`--narrow measured`, kept under budget/k), `heldout` (`--narrow
#      heldout`, the narrowing undone until the N windows after the
#      envelope's are under budget/k), `rss-narrow` and `rss-heldout`, with
#      k = MARGIN;
#   3. every distinct realization run natively on the 461 held-out windows
#      by experiments/margins.sh (`make margins`' exact logits linked);
#   4. the report, OUT/report.md (experiments/spend.py report): per form,
#      size and budget, the traffic, the measurements, the gap in sample
#      (the N windows) and out (windows 2N to 460, which no form saw: the
#      held-out form measures windows N to 2N − 1), out/in, the flips, the
#      bound #{m < 2ε} and the certification; then the rule of the plan's
#      §1: a form is a default only if it holds out of sample in every cell
#      with 1.5× left and its realization is the same at the two sizes.
# Nothing here is timed of record. Outputs present are reused: delete OUT
# to start again. Needs `make curve` and `make margins` (MARGINS).
#
#   experiments/spend.sh [OUT]     default out/spend
#   SIZES="4 16" BUDGETS="4.67e-4 1e-6" MARGIN=2.5 JOBS=8 MARGINS=out/margins
set -u
cd "$(dirname "$0")/.."
OUT=${1:-out/spend}
PRESK=${PRESK:-./target/release/presk}
PY=${PY:-python3}
W=${W:-out/curve_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
MARGINS=${MARGINS:-out/margins}
SIZES=${SIZES:-4 16}
BUDGETS=${BUDGETS:-4.67e-4 1e-6}
MARGIN=${MARGIN:-2.5}
FORMS=${FORMS:-unit-off measured-off rss narrow heldout rss-narrow rss-heldout}
JOBS=${JOBS:-8}
FILE=examples/experiments/margins.psk
CONSTS=(--const D=64 --const H=2 --const L=2)
mkdir -p "$OUT/env" "$OUT/solve"
say() { echo "== $(date +%H:%M:%S) $*"; }
for f in "$W" "$VAL" "$MARGINS/exact/0000.tsv" "$MARGINS/val.json"; do
    [ -e "$f" ] || { echo "missing $f: run make curve and make margins first"; exit 1; }
done
echo "presk $("$PRESK" --version) at $(git rev-parse --short HEAD)" > "$OUT/machine.txt"
uname -srm >> "$OUT/machine.txt"

running=0
spawn() { "$@" & running=$((running + 1)); if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi; }
flags() { # form
    case "$1" in
        unit-off) echo "--rounding unit" ;;
        measured-off) echo "" ;;
        rss) echo "--estimate rss --margin $MARGIN" ;;
        narrow) echo "--narrow measured --margin $MARGIN" ;;
        heldout) echo "--narrow heldout --margin $MARGIN" ;;
        rss-narrow) echo "--estimate rss --narrow measured --margin $MARGIN" ;;
        rss-heldout) echo "--estimate rss --narrow heldout --margin $MARGIN" ;;
        *) echo "unknown form $1" >&2; exit 2 ;;
    esac
}

say "1. the envelopes"
for n in $SIZES; do
    e=$OUT/env/$n
    [ -f "$e.json" ] || spawn bash -c "/usr/bin/time -f '%e s, %M KB' '$PRESK' calibrate --precision $FILE --entry logits ${CONSTS[*]} --dim T=256 \
        --weights '$W' --data '$VAL' --from 0 --samples $n --bits 128 --out '$e.json.part' > '$e.txt' 2> '$e.time' && mv '$e.json.part' '$e.json'"
done
wait; running=0
for n in $SIZES; do [ -f "$OUT/env/$n.json" ] || { echo "missing the envelope on $n windows"; exit 1; }; done

say "2. the solves"
solve() { # n budget form
    local s=$OUT/solve/$1-$2-$3
    [ -f "$s.json" ] && return 0
    # shellcheck disable=SC2046
    /usr/bin/time -f "%e s, %M KB" "$PRESK" solve "$FILE" --precision "$OUT/env/$1.json" --budget-abs "$2" --weights "$W" \
        --sites 100 $(flags "$3") --out "$s.json.part" > "$s.txt" 2> "$s.time" \
        && mv "$s.json.part" "$s.json" || echo "the solve $1-$2-$3 failed: $s.time"
}
# the narrowings on the largest sample first: they are the longest
for f in rss-heldout heldout rss-narrow narrow rss measured-off unit-off; do
    case " $FORMS " in *" $f "*) ;; *) continue ;; esac
    for n in $(echo $SIZES | tr ' ' '\n' | sort -rn); do
        for b in $BUDGETS; do spawn solve "$n" "$b" "$f"; done
    done
done
wait; running=0
for f in "$OUT"/solve/*.txt; do
    echo "$(basename "$f" .txt): $(grep -o '([0-9.]*x)' "$f" | head -1) $(grep -o 'measured  *[0-9.e+-]*' "$f" | head -1)"
done

say "3. the 461 windows"
M=$OUT/margins
mkdir -p "$M/native" "$M/decisions" "$M/realizations"
MA=$(cd "$MARGINS" && pwd)
ln -sfn "$MA/exact" "$M/exact"
ln -sf "$MA/val.json" "$M/val.json"
# `make margins`' rows linked, so that margins.sh runs only the realizations
# below; it writes its own reports, never linked.
for r in "$MA"/native/*/; do
    r=$(basename "$r")
    [ -e "$M/native/$r" ] || ln -sfn "$MA/native/$r" "$M/native/$r"
    for e in txt loss; do [ -e "$MA/decisions/$r.$e" ] && ln -sf "$MA/decisions/$r.$e" "$M/decisions/$r.$e"; done
done
"$PY" experiments/spend.py unique "$OUT" > "$OUT/realizations.tsv" || exit 1
EXTRA=""
while IFS=$'\t' read -r name _; do EXTRA="$EXTRA $name=@$M/realizations/$name.json"; done < "$OUT/realizations.tsv"
EXTRA="$EXTRA" OUT="$M" W="$W" VAL="$VAL" PRESK="$PRESK" experiments/margins.sh > "$M/margins.log" 2>&1 || { tail "$M/margins.log"; exit 1; }

say "4. the report"
"$PY" experiments/spend.py report "$OUT" --sizes "$SIZES" --budgets "$BUDGETS" --forms "$FORMS" --margin "$MARGIN" > "$OUT/report.md" || exit 1
say "wrote $OUT/report.md"
