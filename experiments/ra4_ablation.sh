#!/bin/bash
# The ablation of plan 0.8.0 §3.4 (step 4): the development set priced under
# every form of the layout cost model, each fitted on the same calibration,
# and 0.7.0's own model beside them. Nothing is measured; it decides nothing.
#
#   experiments/ra4_ablation.sh OUT
#
# CALIBRATION is the definitive calibration's profile (D68), archived
# read-only. Each form is `presk refit CALIBRATION --load-term L --combine C`
# into OUT/profile-L-C.json, then experiments/ra4_dev.sh prices 0.7.0's
# ten-core days under it into OUT/L-C (its twenty-core day is left out,
# D68). A form L-C+R names rules of the planned cost too (step 4, 4c): the
# calibration is first costed again under them, its times kept (`presk
# refit --recost examples/experiments/matmul.psk --rules R`), then refitted in L-C.
# OUT/0.7.0 is priced by 0.7.0's profile, and must reproduce
# docs/ra4-2026-09-24.md. experiments/ra4_ablation.py then gathers the
# reports and the re-priced lines into OUT/ablation.md.
set -u -o pipefail
cd "$(dirname "$0")/.."
PRESK=${PRESK:-target/release/presk}
RA4_DEV=${RA4_DEV:-$HOME/presk-data/ra4-0.7.0}
CALIBRATION=${CALIBRATION:-$HOME/presk-data/ra4-0.8.0-calibration/out/profile.json}
FORMS=${FORMS:-none-max group-max machine-max none-sum group-sum machine-sum none-pnorm group-pnorm machine-pnorm}
OUT=${1:?usage: experiments/ra4_ablation.sh OUT}
[ -f "$CALIBRATION" ] || { echo "missing $CALIBRATION: the definitive calibration is archived there (D68)"; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT"
{ date -Is; echo "calibration $CALIBRATION $(sha256sum "$CALIBRATION" | cut -c1-64)"; git rev-parse --short HEAD; } > "$OUT/ablation.txt"
RA4_DEV=$RA4_DEV PROFILE=$RA4_DEV/out/profile.json PROFILE_20= experiments/ra4_dev.sh "$OUT/0.7.0" | tail -1 || exit 1
for f in $FORMS; do
  form=${f%%+*}
  recost=()
  [ "$form" != "$f" ] && recost=(--recost examples/experiments/matmul.psk --rules "${f#*+}")
  "$PRESK" refit "$CALIBRATION" "${recost[@]}" --load-term "${form%-*}" --combine "${form#*-}" --out "$OUT/profile-$f.json" | grep -E '^-- (fit|uncertainty|[0-9]+ samples re-costed)' || exit 1
  RA4_DEV=$RA4_DEV PROFILE=$OUT/profile-$f.json PROFILE_20= experiments/ra4_dev.sh "$OUT/$f" | tail -1 || exit 1
done
python3 experiments/ra4_ablation.py "$OUT" 0.7.0 $FORMS > "$OUT/ablation.md" || exit 1
echo "-- the ablation in $OUT/ablation.md"
