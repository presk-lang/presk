#!/bin/bash
# The RA4 sweep of plan 0.7.0 (§3.6; step 5, D61): one day of measurement.
# For every shape and every type, `presk autotune` runs the entry once to
# capture its launches, then times every point of every nest's layout space
# on the captured buffers and checks each output bit-identical to point 0's
# (statement L) — natively, and on the kernel interpreter too at the shapes
# INTERP names (1: all of them, 0: none);
# `presk plan --cost` chooses over the same points from the layout cost
# model, for the record of what the static choice costs. Both are pinned to
# the RA4 cores with taskset. Given the directory of an earlier day,
# `presk rank` then compares the model with both days.
#
#   experiments/ra4.sh DAY                   one day, into DAY/<shape>_<type>/
#   experiments/ra4.sh DAY EARLIER OUT.md    ...then rank EARLIER against DAY
#
# Each DAY/<constants>_<shape>_<type>/ holds one <nest>.jsonl per nest (a
# JSON line per point), sweep.json (the times, the machine, u), plan.json and
# plan.txt (`presk plan --cost`); DAY/day.txt the date, the load and the
# commit of each run into DAY. EXTRA is passed to `presk autotune` (a sample:
# "--top 8 --random 32"). NEST names the nests to sweep, separated by spaces,
# every nest when empty: one sweep of several nests into one configuration
# (0.10.0 step 2, T3; `presk autotune` records them in sweep.json).
# The defaults are the measurement of §3.4: SelfAttention.apply of the
# course GPT at its constants (D = 384, H = 6), T = 256, B ∈ {1, 16}, f32
# and f64, the whole space; the attention as the sweeps measured it, its
# mask a real tensor added to the scores, frozen in
# examples/experiments/ra4_attention.psk since plan 0.10.0 step 7b (D99). Needs a
# quiet machine and a calibrated profile (`make calibrate-ra4`).
set -u -o pipefail
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
CORES=${RA4_CORES:-5-9,15-19}
FILE=${FILE:-examples/experiments/ra4_attention.psk}
ENTRY=${ENTRY:-SelfAttention.apply}
CONSTS=${CONSTS:-}                       # e.g. "D=1024 H=16"
SHAPES=${SHAPES:-B=1,T=256 B=16,T=256}   # one --dim set per word
TYPES=${TYPES:-f32 f64}
SPACE=${SPACE:-full}
REPEAT=${REPEAT:-7}
PROFILE=${PROFILE:-out/profile.json}
INTERP=${INTERP:-1}                      # 1, 0, or the shapes to check on the interpreter, e.g. "B=1,T=256"
NEST=${NEST:-}                           # nests only, e.g. "SelfAttention.apply/%13 SelfAttention.apply/%23"
EXTRA=${EXTRA:-}
DAY=${1:?usage: experiments/ra4.sh DAY [EARLIER OUT.md]}
[ -f "$PROFILE" ] || { echo "missing $PROFILE: run make calibrate-ra4 first"; exit 1; }
mkdir -p "$DAY"
# The commit, marked -dirty when the working tree differs from it: `presk
# rank --frozen` refuses such a day (plan 0.8.0 §3.4).
{ date -Is; uptime; echo "$ENTRY $CONSTS $SHAPES $TYPES${NEST:+, nests $NEST}, cores $CORES"
  echo "$(git rev-parse --short HEAD)$([ -z "$(git status --porcelain)" ] || echo -dirty)"; } >> "$DAY/day.txt"
common=(--entry "$ENTRY")
for c in $CONSTS; do common+=(--const "$c"); done
for n in $NEST; do common+=(--nest "$n"); done
for shape in $SHAPES; do
  interp=(); { [ "$INTERP" = 1 ] || [[ " $INTERP " == *" $shape "* ]]; } && interp=(--interp)
  dims=(); for d in ${shape//,/ }; do dims+=(--dim "$d"); done
  for t in $TYPES; do
    dir=$DAY/$(echo "$CONSTS $shape" | tr -d '=' | tr ' ,' '__' | sed 's/^_//')_$t
    mkdir -p "$dir"
    echo "-- $ENTRY at $shape, $t, space $SPACE, pinned to $CORES${interp:+, checked on the interpreter}"
    taskset -c "$CORES" "$PRESK" plan "$FILE" "${common[@]}" "${dims[@]}" --realize "$t" --profile "$PROFILE" \
      --space "$SPACE" --cost --out "$dir/plan.json" > "$dir/plan.txt" || { tail -3 "$dir/plan.txt"; exit 1; }
    grep '^-- wall' "$dir/plan.txt"
    taskset -c "$CORES" "$PRESK" autotune "$FILE" "${common[@]}" "${dims[@]}" --realize "$t" --space "$SPACE" \
      --repeat "$REPEAT" --profile "$PROFILE" "${interp[@]}" $EXTRA --out "$dir" | grep '^--' || exit 1
  done
done
if [ $# -ge 3 ]; then
  "$PRESK" rank "$2" "$DAY" --out "$3" --entry "$ENTRY" > /dev/null || exit 1
  echo "-- $2 and $DAY ranked into $3"
fi
