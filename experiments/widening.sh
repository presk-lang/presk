#!/bin/bash
# Where the measurement changes the solver's decision (D128): the six
# entries of experiments/widening.psk, three pairs, each solved at seven
# budgets and verified against the exact reference, beside the four rules'
# bounds and the measured gap under each uniform type.
#
# For each entry, every output under out/widening/:
#
#   env_<E>.json             `presk calibrate --precision` at 128 bits on the
#                            seed's inputs, B rows (8, 64 or 4: below).
#   check_<E>_<T>.txt        `presk check --model all --measure` under the
#                            uniform type T = f64, f32, bf16: the bounds `k`,
#                            `normwise`, `cond`, the estimate `rss` and the
#                            gap of the kernels against the exact reference.
#   solve_<E>_<b>.txt        `presk solve --budget b`, and the realization it
#   solved_<E>_<b>.json      ends on.
#   explain_<E>_<b>/         `presk explain --budget b`, whose §6.3 gives every
#                            move of that solve: each narrowing tried and
#                            taken, each widening round (the site, from, to,
#                            the gap that forced it).
#   proposal_<E>_<b>.json    where the solver widened: its proposal, before
#   proposal_<E>_<b>.txt     any widening, under `presk check --model all
#                            --measure`: the four rules on what the
#                            measurement refused.
#   culprit_<E>_<b>.json     the proposal with only the value the estimate
#   culprit_<E>_<b>.txt      misjudged widened (named below, CULPRIT), and
#                            its measured gap: what widening could have kept.
#   summary.md               the tables (experiments/widening.py summary).
#
# Nothing is timed; any machine, a few minutes. A step whose output exists
# is not run again (delete out/widening to start over).
set -u -o pipefail
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
OUT=${OUT:-out/widening}
FILE=experiments/widening.psk
BUDGETS=${BUDGETS:-"1e-1 3e-2 1e-2 1e-3 1e-4 1e-5 1e-6"}
# entry:rows. The attention row and the long sums at a few rows; the
# standard deviation at 64, where the row of the smallest variance, whose
# error is the largest, is far enough from the mean row the estimate is of.
ENTRIES=${ENTRIES:-"attn_raw:8 attn_softmax:8 std_two_pass:64 std_one_pass:64 clock:4 noisy:4"}
# The value each widening entry's estimate misjudges, and the type it needs:
# the sum of exp and the nest that reads it (p = e / Σe is fused into p · v);
# the mean μ; the long sum.
declare -A CULPRIT=(
    [attn_raw]="attn_raw/%6=f64 attn_raw/%8=f64"
    [std_two_pass]="std_two_pass/%4=f32"
    [clock]="clock/%7=f64"
)
[ -x "$PRESK" ] || { echo "widening.sh: no $PRESK: make build"; exit 1; }
mkdir -p "$OUT"
$PRESK --version > "$OUT/presk-version.txt"

for er in $ENTRIES; do
    E=${er%:*}; N=${er#*:}
    echo "== $E, B = $N"
    ENV=$OUT/env_$E.json
    [ -f "$ENV" ] || $PRESK calibrate --precision $FILE --entry $E --dim B=$N --bits 128 --out "$ENV" > /dev/null || exit 1
    for T in f64 f32 bf16; do
        C=$OUT/check_${E}_$T.txt
        [ -f "$C" ] || $PRESK check --precision "$ENV" $FILE --realize $T --model all --measure --sites 20 > "$C" || exit 1
    done
    for b in $BUDGETS; do
        S=$OUT/solve_${E}_$b.txt
        [ -f "$S" ] || $PRESK solve --precision "$ENV" $FILE --budget $b --out "$OUT/solved_${E}_$b.json" --sites 20 > "$S" 2>&1 || exit 1
        X=$OUT/explain_${E}_$b
        [ -f "$X/explain.md" ] || $PRESK explain $FILE --entry $E --dim B=$N --budget $b --out "$X" > /dev/null 2>&1 || exit 1
        # explain solves again, on its own calibration: the same seed, rows
        # and bits, so the same envelope and the same realization.
        cmp -s <(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])).get('storage', {}), sort_keys=True))" "$OUT/solved_${E}_$b.json") \
               <(python3 -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1])).get('storage', {}), sort_keys=True))" "$X/artifacts/realization.json") \
            || { echo "  $b: explain's realization differs from solve's"; exit 1; }
        printf "  %-5s %s\n" "$b" "$(sed -n 's/^  measured *//p' "$S")"
    done
    # Where the solve widened: its proposal as a realization, under the
    # four rules (the bound against the gap the estimate missed), and the
    # proposal with only the culprit widened.
    for b in $BUDGETS; do
        J=$OUT/proposal_${E}_$b.json
        python3 experiments/widening.py culprit "$OUT/explain_${E}_$b/explain.md" "$J" || continue
        [ -f "${J%.json}.txt" ] || $PRESK check --precision "$ENV" $FILE --realize "@$J" --model all --measure --sites 0 > "${J%.json}.txt" || exit 1
        [ -n "${CULPRIT[$E]:-}" ] || continue
        J=$OUT/culprit_${E}_$b.json
        python3 experiments/widening.py culprit "$OUT/explain_${E}_$b/explain.md" "$J" ${CULPRIT[$E]} || continue
        [ -f "${J%.json}.txt" ] || $PRESK check --precision "$ENV" $FILE --realize "@$J" --model cond --measure --sites 0 > "${J%.json}.txt" || exit 1
    done
done

python3 experiments/widening.py summary "$OUT" "$BUDGETS" $ENTRIES | tee "$OUT/summary.md"
