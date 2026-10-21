#!/bin/bash
# A precision-tuning tool on the exported C (D154): what CRAFT, the one of
# the C tools that runs here, finds on the C program `presk export` writes
# for the logits of the course GPT at D = 64, under the paper's criterion,
# beside what the solver finds; the searches of Precimonious and CRAFT
# reimplemented on the compiler's own space; and everything they found
# judged by the compiler on the 461 held-out windows of `make margins`.
#
# The criterion: the largest elementwise gap of the logits to the exact
# reference's, absolute, at most ε*(10⁻³) = 4.67·10⁻⁴ (V6's `--budget-abs`;
# D118's margins), on the export's conformance inputs, against the exact
# outputs the export ships (conformance/exact.psk.d). A second budget, 10⁻⁶,
# below every all-f32's gap, so that the searches have something to search.
#
#   1. the export, `f64`, and its check (C1);
#   2. the solver at each budget (V6's protocol: the envelope on the first 4
#      held-out windows at 128 bits, `presk solve --budget-abs`), and its
#      space, every decision listed (a solve at 10³ unverified);
#   3. tier A, CRAFT's variable mode on the C (experiments/tuning/craft.sh:
#      the variables retype.py lists, its builder and test), both its
#      strategies, `bin_simple` and `ddebug`; ADAPT looked at (its example
#      built and run, the export compiled as C++);
#   4. tier B, the two searches on presk's realizations
#      (experiments/tuning/search.py), over the stored values and over the
#      stored values with the parameters;
#   5. every realization found run natively on the 461 windows by
#      experiments/margins.sh, beside each tool's C program as it stands
#      (experiments/tuning/windows.py), with `make margins`' report; the
#      bound each realization has (`presk check --precision`);
#   6. the report, OUT/report.md (experiments/tuning/report.py).
# Nothing here is a timing of record: a C program's milliseconds and a
# search's wall time are notes, the searches' trials are counts. Outputs
# present are reused, as every experiment script here does: delete OUT to
# start again. Needs `make margins` (MARGINS: its exact logits and rows)
# and the tools (experiments/tuning/setup.sh, network once).
#
#   experiments/tuning/tuning.sh [OUT]       default out/precision-tuning
#   BUDGETS="4.67e-4 1e-6" CORE=5 TOOLS=.tuning MARGINS=out/margins
set -u
cd "$(dirname "$0")/../.."
OUT=${1:-out/precision-tuning}
PRESK=${PRESK:-./target/release/presk}
PY=${PY:-python3}
W=${W:-out/curve_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
MARGINS=${MARGINS:-out/margins}
BUDGETS=${BUDGETS:-4.67e-4 1e-6}
CORE=${CORE:-5}
TOOLS=${TOOLS:-.tuning}
FILE=experiments/margins.psk
CONSTS=(--const D=64 --const H=2 --const L=2)
T=experiments/tuning
mkdir -p "$OUT"
say() { echo "== $(date +%H:%M:%S) $*"; }

for f in "$W" "$VAL" "$MARGINS/exact/0000.hilo" "$MARGINS/val.json"; do
    [ -e "$f" ] || { echo "missing $f: run make curve and make margins first"; exit 1; }
done

say "the tools"
"$T/setup.sh" "$TOOLS" > "$OUT/tools.txt" || exit 1
TOOLS=$(cd "$TOOLS" && pwd)
cat "$OUT/tools.txt"
{
    uname -srm
    lscpu | grep -E "^(Model name|CPU\(s\)|Thread|Core)" || true
    gcc --version | head -1
    "$PY" --version
    echo "presk $("$PRESK" --version) at $(git rev-parse --short HEAD)"
} > "$OUT/machine.txt"

say "1. the export, f64"
X=$OUT/c-f64
[ -f "$X/model.c" ] || "$PRESK" export "$FILE" --entry logits "${CONSTS[@]}" --dim B=2 --dim T=256 --weights "$W" \
    --realize f64 --target c --out "$X" > "$OUT/c-f64.export" 2>&1 || exit 1
make -s -C "$X" check > "$OUT/c-f64.check" 2>&1; tail -1 "$OUT/c-f64.check"
[ -x "$X/model" ] || exit 1
wc -l "$X"/*.c "$X"/*.h | tail -1

say "2. the solver"
S=$OUT/solver
mkdir -p "$S"
[ -f "$S/env_logits.json" ] || "$PRESK" calibrate --precision "$FILE" --entry logits "${CONSTS[@]}" --dim T=256 --weights "$W" \
    --data "$VAL" --samples 4 --bits 128 --out "$S/env_logits.json" > "$S/calibrate.txt" 2>&1 || exit 1
for b in $BUDGETS; do
    [ -f "$S/solved-$b.json" ] || /usr/bin/time -v "$PRESK" solve "$FILE" --precision "$S/env_logits.json" --budget-abs "$b" \
        --weights "$W" --sites 100 --out "$S/solved-$b.json" > "$S/solve-$b.txt" 2> "$S/solve-$b.time" || exit 1
    sed -n '1,8p' "$S/solve-$b.txt"
done
[ -f "$S/space.json" ] || "$PRESK" solve "$FILE" --precision "$S/env_logits.json" --budget-abs 1e3 --no-verify \
    --weights "$W" --sites 100 --out "$S/space.json" > "$S/space.txt" 2>&1 || exit 1
sed -n '2,3p' "$S/space.txt"

say "3. tier A: CRAFT's variable mode on the C"
for b in $BUDGETS; do
    for s in bin_simple ddebug; do
        d=$OUT/craft/$s-$b
        [ -f "$d/wall.txt" ] || { rm -rf "$d"; CORE=$CORE RUBY="$TOOLS/env/bin/ruby" CRAFT="$TOOLS/craft" "$T/craft.sh" "$X" "$W" "$b" "$s" "$d" || exit 1; }
        echo "$s at $b: $(grep -E 'Total configs tested' "$d/craft.out" | tr -s ' ')  $(cat "$d/wall.txt")"
    done
done
A=$OUT/adapt
mkdir -p "$A"
if [ ! -f "$A/example.txt" ]; then
    ( cd "$TOOLS/adapt-fp/sum2pi_x" && rm -f sum2pi_x-adapt && make CODIPACK_HOME=../../CoDiPack \
        ADAPTFLAGS="--std=c++17 -I../../CoDiPack/include -I../ -DCODI_ReversalZeroesAdjoints=0" sum2pi_x-adapt && ./sum2pi_x-adapt ) > "$A/example.txt" 2>&1
    for f in model.c presk_rt.c main.c; do
        echo "== g++ -std=c++17 -x c++ -fsyntax-only $f: $(g++ -std=c++17 -x c++ -fsyntax-only -I"$X" "$X/$f" 2>&1 | grep -c 'error') errors"
        g++ -std=c++17 -x c++ -fsyntax-only -I"$X" "$X/$f" 2>&1 | grep 'error' | sed 's/.*error: //' | sort | uniq -c | sort -rn | head -6
    done > "$A/export-as-cxx.txt"
fi
grep -A 4 "BEGIN ADAPT REPORT" "$A/example.txt" | head -5; grep "^==" "$A/export-as-cxx.txt"

say "4. tier B: the searches on presk's space"
mkdir -p "$OUT/search"
for b in $BUDGETS; do
    for s in precimonious craft; do
        for p in sites params; do
            d=$OUT/search/$s-$p-$b
            [ -f "$d/summary.json" ] || "$PY" "$T/search.py" --space "$S/space.txt" --export "$X" --budget "$b" --strategy "$s" \
                $([ $p = params ] && echo --params) --out "$d" --presk "$PRESK" --weights "$W" > "$d.log" 2>&1 || { cat "$d.log"; exit 1; }
            tail -1 "$d.log"
        done
    done
done

say "5. the 461 windows"
M=$OUT/margins
mkdir -p "$M/native" "$M/decisions" "$M/realizations"
MA=$(cd "$MARGINS" && pwd)
ln -sfn "$MA/exact" "$M/exact"
ln -sf "$MA/val.json" "$M/val.json"
# the rows of `make margins`, linked: margins.sh finds them and runs only the new ones
for r in "$MA"/native/*/; do
    r=$(basename "$r")
    [ -e "$M/native/$r" ] || ln -sfn "$MA/native/$r" "$M/native/$r"
    for e in txt loss; do [ -e "$MA/decisions/$r.$e" ] && ln -sf "$MA/decisions/$r.$e" "$M/decisions/$r.$e"; done
done
EXTRA=""
for b in $BUDGETS; do
    cp "$S/solved-$b.json" "$M/realizations/solver-$b.json"
    for s in bin_simple ddebug; do
        "$PY" "$T/retype.py" realization "$X" "$OUT/craft/$s-$b/craft_final.json" > "$M/realizations/craft-$s-$b.json" 2> "$M/realizations/craft-$s-$b.apart"
    done
    for s in precimonious craft; do
        for p in sites params; do cp "$OUT/search/$s-$p-$b/final.json" "$M/realizations/search-$s-$p-$b.json"; done
    done
done
for f in "$M"/realizations/*.json; do n=$(basename "$f" .json); EXTRA="$EXTRA $n=@$f"; done
EXTRA="$EXTRA" OUT="$M" W="$W" VAL="$VAL" PRESK="$PRESK" "experiments/margins.sh" > "$M/margins.log" 2>&1 || { tail "$M/margins.log"; exit 1; }
"$PY" "$T/windows.py" inputs "$VAL" "$X" "$M/inputs" > /dev/null
for b in $BUDGETS; do
    for s in bin_simple ddebug; do
        n=c-craft-$s-$b
        [ -f "$M/decisions/$n.txt" ] || "$PY" "$T/windows.py" judge "$OUT/craft/$s-$b/final" "$W" "$M/inputs" "$M/exact" "$M/native/$n" "$M/decisions/$n.txt" "$CORE"
    done
done
n=c-f64
[ -f "$M/decisions/$n.txt" ] || "$PY" "$T/windows.py" judge "$X" "$W" "$M/inputs" "$M/exact" "$M/native/$n" "$M/decisions/$n.txt" "$CORE"
same=yes
for f in "$M/native/c-f64"/*.tsv; do cmp -s "$f" "$MA/native/f64/$(basename "$f")" || same=no; done
cmp -s "$M/decisions/c-f64.txt" "$MA/decisions/f64.txt" || same=no
echo "the export's own C program judged as a tool's is the native f64 row, bit for bit: $same" | tee "$M/control.txt"

# the report on the rows this run is about
NAMES="f64 f32 bf16"
LABELS="presk, all-f64|presk, all-f32|presk, all-bf16"
for b in $BUDGETS; do
    NAMES="$NAMES solver-$b"; LABELS="$LABELS|the solver at $b"
    for s in precimonious craft; do
        for p in sites params; do
            NAMES="$NAMES search-$s-$p-$b"; LABELS="$LABELS|tier B, $s's search at $b, $([ $p = params ] && echo 'stored values and parameters' || echo 'stored values')"
        done
    done
    for s in bin_simple ddebug; do
        NAMES="$NAMES craft-$s-$b c-craft-$s-$b"
        LABELS="$LABELS|tier A, CRAFT $s at $b, as a realization|tier A, CRAFT $s at $b, its C program"
    done
done
NAMES="$NAMES c-f64"; LABELS="$LABELS|the export's own C program (control)"
paste <(echo "$NAMES" | tr ' ' '\n') <(echo "$LABELS" | tr '|' '\n') > "$OUT/rows.tsv"
GAPS=$(for n in $NAMES; do v=$(cat "$M/decisions/$n.loss" 2>/dev/null); echo -n "${v:-—} "; done)
"$PY" experiments/margins.py report "$M" --val "$M/val.json" --names "$NAMES" --labels "$LABELS" --loss-gaps "$GAPS" --out "$OUT/margins.md"

say "bounds"
{
    for r in f64 f32 bf16; do echo "$r $("$PRESK" check "$FILE" --precision "$S/env_logits.json" --realize "$r" | awk '/bound/ {print $3}')"; done
    for f in "$M"/realizations/*.json; do
        echo "$(basename "$f" .json) $("$PRESK" check "$FILE" --precision "$S/env_logits.json" --realize "@$f" | awk '/bound/ {print $3}')"
    done
} > "$OUT/bounds.txt"

say "C programs' times (notes, ${CORE})"
{
    for d in "$X" "$OUT"/craft/*/final; do
        echo "$d $( { OMP_NUM_THREADS=1 taskset -c "$CORE" "$d/model" -r 21 "$W" "$X/conformance/inputs.psk.d" /dev/null; } 2>&1 | sed -n 's/.*median \([0-9.]*\) ms.*/\1/p')"
    done
    for r in f32 "@$S/solved-4.67e-4.json"; do
        n=$(echo "$r" | sed 's/.*\///; s/\.json//; s/@//')
        [ -d "$OUT/c-presk-$n" ] || "$PRESK" export "$FILE" --entry logits "${CONSTS[@]}" --dim B=2 --dim T=256 --weights "$W" \
            --realize "$r" --target c --no-exact --out "$OUT/c-presk-$n" > /dev/null 2>&1
        make -s -C "$OUT/c-presk-$n" OPENMP= model
        echo "$OUT/c-presk-$n $( { taskset -c "$CORE" "$OUT/c-presk-$n/model" -r 21 "$W" "$X/conformance/inputs.psk.d" /dev/null; } 2>&1 | sed -n 's/.*median \([0-9.]*\) ms.*/\1/p')"
    done
} > "$OUT/c-times.txt"
cat "$OUT/c-times.txt"

say "6. the report"
"$PY" "$T/report.py" "$OUT" --budgets "$BUDGETS" > "$OUT/report.md" || exit 1
say "wrote $OUT/report.md"
