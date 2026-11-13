#!/bin/bash
# The dry run before the nights of a fresh sweep (plan 0.8.0 §3.5; step 5,
# D72): one nest's sweep timed as a night times it, and the night sized from
# it, so that a configuration too slow to sweep whole is known before it is
# run. Nothing it measures enters the test.
#
#   experiments/ra4_dry.sh DIR
#
# 1. Every configuration is priced by `presk plan --cost`, pinned as the
#    sweep is, so that its points are the sweep's: DIR/plans/<config>/ (its
#    nests NEST names, as ra4.sh's, every nest when empty).
# 2. The nest whose sweep the model predicts longest is chosen, or DRY_NEST
#    at AT in DRY_TYPE when all three are given.
# 3. It is swept by experiments/ra4.sh (--nest), pinned, timed, checked
#    natively: DIR/nest/.
# 4. experiments/ra4_dry.py sizes each configuration from it into
#    DIR/dry.md: the nest's measured time over its predicted, applied to
#    every point's prediction, plus its time a point outside the kernels.
# The configuration variables are ra4.sh's (RA4_CORES, PROFILE, FILE, ENTRY,
# CONSTS, SHAPES, TYPES, SPACE, REPEAT, EXTRA, NEST).
set -u -o pipefail
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
CORES=${RA4_CORES:-5-9,15-19}
FILE=${FILE:-examples/experiments/ra4_attention.psk}
ENTRY=${ENTRY:-SelfAttention.apply}
CONSTS=${CONSTS:-}
SHAPES=${SHAPES:-B=1,T=256 B=16,T=256}
TYPES=${TYPES:-f32 f64}
SPACE=${SPACE:-full}
REPEAT=${REPEAT:-7}
PROFILE=${PROFILE:-out/profile.json}
NEST=${NEST:-}
DRY_NEST=${DRY_NEST:-}
AT=${AT:-}
DRY_TYPE=${DRY_TYPE:-}
DIR=${1:?usage: experiments/ra4_dry.sh DIR}
[ -f "$PROFILE" ] || { echo "missing $PROFILE: calibrate first"; exit 1; }
rm -rf "$DIR"
mkdir -p "$DIR/plans"
common=(--entry "$ENTRY")
for c in $CONSTS; do common+=(--const "$c"); done
for n in $NEST; do common+=(--nest "$n"); done
for shape in $SHAPES; do
  dims=(); for d in ${shape//,/ }; do dims+=(--dim "$d"); done
  for t in $TYPES; do
    name=$(echo "$CONSTS $shape" | tr -d '=' | tr ' ,' '__' | sed 's/^_//')_$t
    mkdir -p "$DIR/plans/$name"
    echo "$shape $t" > "$DIR/plans/$name/config.txt"
    taskset -c "$CORES" "$PRESK" plan "$FILE" "${common[@]}" "${dims[@]}" --realize "$t" --profile "$PROFILE" \
      --space "$SPACE" --cost > "$DIR/plans/$name/plan.txt" || { tail -3 "$DIR/plans/$name/plan.txt"; exit 1; }
  done
done
if [ -z "$DRY_NEST" ] || [ -z "$AT" ] || [ -z "$DRY_TYPE" ]; then
  read -r DRY_NEST AT DRY_TYPE < <(python3 experiments/ra4_dry.py choose "$DIR" "$REPEAT") || exit 1
fi
echo "-- the dry run: $DRY_NEST at $AT, $DRY_TYPE, pinned to $CORES"
RA4_CORES=$CORES PROFILE=$PROFILE FILE=$FILE ENTRY=$ENTRY CONSTS=$CONSTS SPACE=$SPACE REPEAT=$REPEAT EXTRA=${EXTRA:-} \
  NEST=$DRY_NEST SHAPES=$AT TYPES=$DRY_TYPE INTERP=0 experiments/ra4.sh "$DIR/nest" || exit 1
python3 experiments/ra4_dry.py estimate "$DIR" "$REPEAT" > "$DIR/dry.md" || exit 1
cat "$DIR/dry.md"
echo "-- the estimate in $DIR/dry.md"
