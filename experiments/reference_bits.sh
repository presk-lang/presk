#!/bin/bash
# SR2, the short runs (2026-11-10;
# D192): would a reference at f64's precision have solved as the one at 128
# bits does? The eight solves of record at D = 64 (D170), solved again with
# the reference interpreter at `--bits 53` and at `--bits 64`, on the
# envelopes of record and with the solves' own options:
#
#   the curve on the trained envelope at 1e-1 … 1e-7 (experiments/curve_trained.sh's
#   solves: out/curve-trained_<b>.json and .txt are the record);
#   V6, the logits at ε*(1e-3) = 4.67e-4 (out/step2-0.16.1/phaseA.sh's solve:
#   out/v6-0.16.1/solved.json, its report in presk-project data/rows-0.16.1.tar.xz).
#
# experiments/reference_bits.py then compares each realization with the one
# of record byte for byte, and each measured gap with the one read at 128
# bits. The rows of record are read, never written: everything goes to
# $OUT. Each solve runs alone in its own scope (MemoryMax 108G, no swap,
# oom_score_adj 1000), timed by /usr/bin/time -v. A solve whose realization
# and report exist is not solved again.
#
#   ./experiments/reference_bits.sh            the 16 solves, then the report
#   BITS=53 ./experiments/reference_bits.sh    one precision
#   REPORT_ONLY=1 ./experiments/reference_bits.sh
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
OUT=${OUT:-out/reference-bits}
read -r -a BITS <<< "${BITS:-53 64}"
MEM=${MEM:-108G}
REMAT=${REMAT:-store}
MODEL=examples/corpus/nlp/05_gpt.psk
ENV=out/curve-trained_env.json
W=out/gpt_trained/final.psk.d
V6=out/v6-0.16.1
W64=out/curve_train_f64/final.psk.d
BUDGETS=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7)
# The record's V6 report is in the 0.16.1 archive only.
ARCHIVE=${ARCHIVE:-../presk-project/data/rows-0.16.1.tar.xz}

scoped() { systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 choom -n 1000 -- "$@"; }
say() { echo "== $(date -Is) $*"; }

for f in "$ENV" "$W" "$V6/env_logits.json" "$V6/solved.json" "$W64"; do
    [ -f "$f" ] || { echo "missing $f (the record of D170)"; exit 1; }
done
mkdir -p "$OUT/record"
[ -f "$OUT/record/v6/solve.txt" ] || tar -xJf "$ARCHIVE" -C "$OUT/record" --strip-components=1 rows-0.16.1/v6 \
    || { echo "cannot read V6's report of record from $ARCHIVE"; exit 1; }

if [ "${REPORT_ONLY:-0}" != 1 ]; then
    say "SR2 on $("$PRESK" --version)"
    for bits in "${BITS[@]}"; do
        D=$OUT/bits-$bits
        mkdir -p "$D"
        for b in "${BUDGETS[@]}"; do
            [ -s "$D/curve-trained_$b.json" ] && [ -s "$D/curve-trained_$b.txt" ] && continue
            say "the curve at $b, --bits $bits"
            scoped /usr/bin/time -v -o "$D/curve-trained_$b.time" \
                "$PRESK" solve "$MODEL" --precision "$ENV" --budget "$b" --weights "$W" \
                --out "$D/curve-trained_$b.json" --sites 0 --remat "$REMAT" --bits "$bits" \
                > "$D/curve-trained_$b.txt" 2> "$D/curve-trained_$b.err"
        done
        if ! { [ -s "$D/v6.json" ] && [ -s "$D/v6.txt" ]; }; then
            say "V6, --bits $bits"
            scoped /usr/bin/time -v -o "$D/v6.time" \
                "$PRESK" solve examples/experiments/margins.psk --precision "$V6/env_logits.json" \
                --budget-abs 4.67e-4 --weights "$W64" --sites 60 --out "$D/v6.json" --bits "$bits" \
                > "$D/v6.txt" 2> "$D/v6.err"
        fi
    done
    say "done"
fi
python3 experiments/reference_bits.py "$OUT" > "$OUT/report.md"
cat "$OUT/report.md"
