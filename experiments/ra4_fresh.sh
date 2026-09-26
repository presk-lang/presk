#!/bin/bash
# The checks before a day of a fresh sweep of plan 0.8.0 (§3.4–3.5; step 5,
# D72), so that a night is not spent on a day the test cannot use. It
# measures nothing; the caller runs experiments/ra4.sh when it passes.
#
#   experiments/ra4_fresh.sh DAY           before day one
#   experiments/ra4_fresh.sh DAY DAY1      before day two, DAY1 its day one
#
# It refuses, with exit code 1:
# - DAY when it exists: a day is measured once, in one run;
# - a working tree that differs from HEAD, or a HEAD that does not descend
#   from TAG (the freeze, `ra4-model-0.8.0`): `presk rank --frozen` would
#   refuse the day after the night;
# - day two without day one, or on the calendar date day one started on:
#   the two days of §3.5 are two different days. SAME_DATE=1 accepts day
#   two on day one's date, hours after it: the user's decision for F2 and
#   F1 (2026-09-25, D72), which amends §3.5.
# TRIAL=1 is a trial of the targets on a tiny configuration, not a day of
# the test: only the first check applies.
set -u
cd "$(dirname "$0")/.."
TAG=${TAG:-ra4-model-0.8.0}
TRIAL=${TRIAL:-}
SAME_DATE=${SAME_DATE:-}
DAY=${1:?usage: experiments/ra4_fresh.sh DAY [DAY1]}
DAY1=${2:-}
[ -e "$DAY" ] && { echo "-- $DAY exists: a day is measured once; move it aside first"; exit 1; }
if [ -n "$TRIAL" ]; then
  echo "-- a trial (TRIAL=$TRIAL): the tree, the tag and the calendar are not checked, and this is not a day of the test"
  exit 0
fi
[ -z "$(git status --porcelain)" ] || { echo "-- the working tree differs from HEAD: a fresh day runs on a committed tree (presk rank --frozen)"; git status --short | head -5; exit 1; }
git merge-base --is-ancestor "$TAG" HEAD 2>/dev/null || { echo "-- HEAD $(git rev-parse --short HEAD) does not descend from $TAG: the model is not the frozen one"; exit 1; }
if [ -n "$DAY1" ]; then
  [ -f "$DAY1/day.txt" ] || { echo "-- $DAY1/day.txt is missing: day two needs day one"; exit 1; }
  first=$(head -1 "$DAY1/day.txt" | cut -c1-10)
  if [ "$first" = "$(date +%F)" ]; then
    [ -n "$SAME_DATE" ] || { echo "-- day one started on $first, today: day two is measured on another calendar day (SAME_DATE=1 accepts today)"; exit 1; }
    first="$first, today (SAME_DATE: day two on day one's date)"
  fi
fi
echo "-- $(git rev-parse --short HEAD) descends from $TAG, the tree is clean${DAY1:+, day one started on $first}"
