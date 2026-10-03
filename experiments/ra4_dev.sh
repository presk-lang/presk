#!/bin/bash
# The development set of plan 0.8.0 (§3.4; step 1, D64): 0.7.0's days, archived
# read-only by step 0 (D63), priced again under a profile by `presk reprice`
# (every recorded point lowered again from its layout and costed, its times
# copied) and ranked by `presk rank`, in about a minute and without measuring
# anything. A model is compared with 0.7.0's on it; it decides nothing. The
# attention is the course GPT's as 0.7.0 measured it, its mask added to the
# scores, frozen in experiments/ra4_attention.psk (plan 0.10.0 step 7b,
# D99): its nests are the ones the days name.
#
#   experiments/ra4_dev.sh OUT
#
# RA4_DEV is the archive. PROFILE prices the sweep on the ten X925 cores
# (out/ra4: two days and the interpreter's check), PROFILE_20 the one on all
# twenty (out/ra4-20, one day); both default to the profiles 0.7.0 priced them
# with. An empty PROFILE_20 leaves the twenty-core day out: 0.8.0's
# development prices only the ten-core days, since no statement is about
# twenty cores and 0.7.0's twenty-core samples have no load count (D68). The
# re-priced days go to OUT/out/ra4/day1, ... and are ranked from
# OUT, so that a report quotes its days as 0.7.0's did: OUT/ra4.md and
# OUT/ra4-20.md. Under 0.7.0's profiles every re-priced line must be the
# archived one, byte for byte — its prediction, times and bits — but for its
# per-kernel records, which gained the load count in 0.8.0 step 2 (D66), and
# each report the one docs/ holds; the script fails otherwise. Under the
# frozen model's profile (FROZEN, archived read-only by 0.8.0 step 4, D71),
# the report must be docs/ra4-dev-model-0.8.0.md, byte for byte: the check
# plan 0.8.0 §3.4 asks of a defect fixed after the freeze.
set -u -o pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PRESK=$ROOT/${PRESK:-target/release/presk}
RA4_DEV=${RA4_DEV:-$HOME/presk-data/ra4-0.7.0}
PROFILE=${PROFILE:-$RA4_DEV/out/profile.json}
PROFILE_20=${PROFILE_20-$RA4_DEV/out/profile-20.json}
FROZEN=${FROZEN:-$HOME/presk-data/ra4-model-0.8.0/profile.json}
OUT=${1:?usage: experiments/ra4_dev.sh OUT}
[ -f "$RA4_DEV/SHA256SUMS" ] || { echo "missing $RA4_DEV/SHA256SUMS: the development set is archived there (D63)"; exit 1; }
(cd "$RA4_DEV" && sha256sum -c --quiet SHA256SUMS) || { echo "$RA4_DEV does not match its checksums"; exit 1; }
rm -rf "$OUT"
mkdir -p "$OUT"
{ date -Is; echo "profiles $PROFILE $PROFILE_20"; git rev-parse --short HEAD; } > "$OUT/dev.txt"
for d in ra4/day1 ra4/day2; do
  "$PRESK" reprice experiments/ra4_attention.psk --entry SelfAttention.apply --day "$RA4_DEV/out/$d" --profile "$PROFILE" --out "$OUT/out/$d" | grep '^--' || exit 1
  "$PRESK" reprice experiments/gemm.psk --entry gemm --day "$RA4_DEV/out/$d" --profile "$PROFILE" --out "$OUT/out/$d" | grep '^--' || exit 1
done
days="ra4/day1 ra4/day2"
if [ -n "$PROFILE_20" ]; then
  "$PRESK" reprice experiments/ra4_attention.psk --entry SelfAttention.apply --day "$RA4_DEV/out/ra4-20/day1" --profile "$PROFILE_20" --out "$OUT/out/ra4-20/day1" | grep '^--' || exit 1
  days="$days ra4-20/day1"
fi
cd "$OUT" || exit 1
"$PRESK" rank out/ra4/day1 out/ra4/day2 --check "$RA4_DEV/out/ra4/check" --out ra4.md > /dev/null || exit 1
if [ -n "$PROFILE_20" ]; then
  "$PRESK" rank out/ra4-20/day1 out/ra4-20/day1 --out ra4-20.md > /dev/null || exit 1
  echo "-- ranked into $OUT/ra4.md and $OUT/ra4-20.md"
else
  echo "-- ranked into $OUT/ra4.md (the twenty-core day left out)"
fi
if [ -f "$FROZEN" ] && cmp -s "$PROFILE" "$FROZEN"; then
  cmp ra4.md "$ROOT/docs/ra4-dev-model-0.8.0.md" || { echo "-- the frozen model's report differs from docs/ra4-dev-model-0.8.0.md"; exit 1; }
  echo "-- the frozen model's profile: the report identical to docs/ra4-dev-model-0.8.0.md"
fi
if cmp -s "$PROFILE" "$RA4_DEV/out/profile.json" && { [ -z "$PROFILE_20" ] || cmp -s "$PROFILE_20" "$RA4_DEV/out/profile-20.json"; }; then
  # 0.7.0's model: every point priced as it was, and the same reports. A
  # line's per-kernel records (the model's inputs and terms) are left out.
  strip() { sed -E 's/"kernels":\[.*\],"key":/"key":/' "$1"; }
  lines=0
  for d in $days; do
    [ "$(cd "$RA4_DEV/out/$d" && find . -name '*.jsonl' | sort)" = "$(cd "out/$d" && find . -name '*.jsonl' | sort)" ] || { echo "-- $d: not every nest was re-priced"; exit 1; }
    for f in $(cd "out/$d" && find . -name '*.jsonl' | sort); do
      cmp -s <(strip "out/$d/$f") <(strip "$RA4_DEV/out/$d/$f") || { echo "-- out/$d/$f differs from the archived one"; exit 1; }
      lines=$((lines + $(wc -l < "out/$d/$f")))
    done
  done
  cmp ra4.md "$ROOT/docs/ra4-2026-09-24.md" || exit 1
  if [ -n "$PROFILE_20" ]; then
    cmp ra4-20.md "$ROOT/docs/ra4-20cores-2026-09-24.md" || exit 1
    echo "-- 0.7.0's profiles: $lines re-priced lines identical to the archived ones but for their per-kernel records, both reports identical to docs/ra4-2026-09-24.md and docs/ra4-20cores-2026-09-24.md"
  else
    echo "-- 0.7.0's profile: $lines re-priced lines identical to the archived ones but for their per-kernel records, the report identical to docs/ra4-2026-09-24.md"
  fi
fi
