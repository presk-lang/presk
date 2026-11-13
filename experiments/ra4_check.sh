#!/bin/bash
# Statement L checked apart from the timed sweeps (plan 0.7.0 step 6, D62).
# Every point of a configuration runs once, untimed, natively and on the
# kernel interpreter, and is compared bit for bit with point 0. The kernel
# interpreter does about 40 M multiply-adds a second, 30 hours for the
# attention at B = 16 in two types, so the check is split into shards run in
# parallel on every core: this script prints one command per shard, and the
# caller runs them (`make ra4-check`: `xargs -P`). Each shard states
# PRESK_THREADS=WORKERS, the pinned sweep's worker count, so it enumerates
# the same splits; the sample of EXTRA is drawn from the profile's
# predictions, so it is the same sample. The configuration variables are
# ra4.sh's, NEST among them.
#
#   experiments/ra4_check.sh DIR    commands writing DIR/<config>/shard-<k>/
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
FILE=${FILE:-examples/experiments/ra4_attention.psk}
ENTRY=${ENTRY:-SelfAttention.apply}
CONSTS=${CONSTS:-}
SHAPES=${SHAPES:-B=1,T=256 B=16,T=256}
TYPES=${TYPES:-f32 f64}
SPACE=${SPACE:-full}
PROFILE=${PROFILE:-out/profile.json}
EXTRA=${EXTRA:-}
NEST=${NEST:-}
SHARDS=${SHARDS:-40}
WORKERS=${WORKERS:-10}
DIR=${1:?usage: experiments/ra4_check.sh DIR}
common=(--entry "$ENTRY")
for c in $CONSTS; do common+=(--const "$c"); done
for n in $NEST; do common+=(--nest "$n"); done
for shape in $SHAPES; do
  dims=(); for d in ${shape//,/ }; do dims+=(--dim "$d"); done
  for t in $TYPES; do
    name=$(echo "$CONSTS $shape" | tr -d '=' | tr ' ,' '__' | sed 's/^_//')_$t
    for ((k = 0; k < SHARDS; k++)); do
      out=$DIR/$name/shard-$k
      cmd=(env PRESK_THREADS="$WORKERS" "$PRESK" autotune "$FILE" "${common[@]}" "${dims[@]}" --realize "$t" --space "$SPACE"
           --repeat 0 --interp --shard "$k/$SHARDS" --profile "$PROFILE" $EXTRA --out "$out")
      printf 'mkdir -p %q && %s > %q 2>&1 || { echo "!! %s shard %d: $(tail -1 %q)"; exit 1; }\n' \
        "$out" "$(printf '%q ' "${cmd[@]}")" "$out/log.txt" "$name" "$k" "$out/log.txt"
    done
  done
done
