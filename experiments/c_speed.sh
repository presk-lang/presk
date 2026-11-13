#!/bin/bash
# Statement C4 of plan 0.9.0 (step 5, D81): the wall time of the exported
# course GPT's forward pass at B = 16, T = 256, with the course's trained
# checkpoint, against the native backend's, on one thread and on every
# thread. Reported; it decides nothing.
#
#   experiments/c_speed.sh OUT
#
# Both run the same plan: the one `presk plan` gives with the frozen profile
# (PROFILE), which divides most launches among ten workers, the profile's
# cores. The C program is exported under it, built by its Makefile (GCC,
# -O2, OpenMP) and timed by its own `-r` loop: the entry called REPEAT times,
# the arena released between calls, the median and the best. The native
# backend is timed by `presk bench --plan-only`: the same plan, the same
# weights, REPEAT calls, the median and the best (its first call compiles
# the kernels, so the median of three or more leaves it out).
#
# - Every thread: the C program's teams are the plan's workers, the native
#   backend's splits the same; neither is pinned.
# - One thread: the C program with OMP_NUM_THREADS=1. The native backend
#   cannot run the plan on one worker (`--threads 1` refuses a split among
#   ten, condition 5), so it runs the plan with every split stated as none,
#   which lowers the same kernels; both pinned to one X925 core (CORE).
#
# Neither side's time holds the loading of the weights. The native side's
# holds its host's copies of values (a slice of a stacked tree, a view) and
# the clone of the arguments per call, which the C program does not make.
# Timing: a quiet machine, nothing beside it.
set -eu -o pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PRESK=$ROOT/${PRESK:-target/release/presk}
PROFILE=${PROFILE:-$HOME/presk-data/ra4-model-0.8.0/profile.json}
CKPT=${CKPT:-out/course_probe_f32/final.psk.d}
case $CKPT in /*) ;; *) CKPT=$ROOT/$CKPT ;; esac
REPEAT=${REPEAT:-7}
CORE=${CORE:-5}
OUT=${1:?usage: experiments/c_speed.sh OUT}
FILE=examples/corpus/nlp/05_gpt.psk
DIMS=(--dim B=16 --dim T=256)

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
"$PRESK" plan "$FILE" --entry GPT.apply "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$OUT/plan.json" > "$OUT/plan.log" 2>&1
# The same layouts, every split stated as none.
python3 - "$OUT/plan.json" "$OUT/plan-one.json" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
for layout in p["layout"].values():
    layout["split"] = "none"
json.dump(p, open(sys.argv[2], "w"), indent=2)
EOF
rm -rf "$OUT/c"
"$PRESK" export "$FILE" --target c --entry GPT.apply "${DIMS[@]}" --plan "@$OUT/plan.json" --weights "$CKPT" --no-exact -o "$OUT/c" 2> "$OUT/export.log"
tail -1 "$OUT/export.log"
make -s -C "$OUT/c" model
(cd "$OUT/c" && sh conformance/check.sh)

# `median best` in ms, from each side's report.
native() {
    "$@" | awk '/plan/ { print $(NF-2), $(NF-1) }'
}
c_prog() {
    (cd "$OUT/c" && "$@" 2>&1 >/dev/null | sed -n 's/.*median \([0-9.]*\) ms, best \([0-9.]*\) ms.*/\1 \2/p')
}
BENCH=(bench "$FILE" --entry GPT.apply "${DIMS[@]}" --native-only --plan-only --weights "$CKPT" --repeat "$REPEAT")
RUN=(./model -r "$REPEAT" "$CKPT" conformance/inputs.psk.d conformance/out.psk.d)

read -r n_all n_all_best < <(native "$PRESK" "${BENCH[@]}" --plan "@$OUT/plan.json")
read -r c_all c_all_best < <(c_prog "${RUN[@]}")
read -r n_one n_one_best < <(native taskset -c "$CORE" "$PRESK" "${BENCH[@]}" --threads 1 --plan "@$OUT/plan-one.json")
read -r c_one c_one_best < <(c_prog env OMP_NUM_THREADS=1 taskset -c "$CORE" "${RUN[@]}")
for t in "$n_all" "$c_all" "$n_one" "$c_one"; do
    [ -n "$t" ] || { echo "c_speed.sh: a run gave no time" >&2; exit 1; }
done

ratio() { python3 -c "print(f'{$1 / $2:.2f}')"; }
{
    echo "# Statement C4: the exported course GPT against the native backend"
    echo
    echo "The course GPT's forward pass (\`$FILE\`, \`GPT.apply\`) at B = 16, T = 256, in \`f32\`, with the trained checkpoint \`${CKPT#$ROOT/}\`, under the plan \`presk plan\` gives with the frozen profile; $REPEAT calls each, wall time in ms. $(date -I), \`$(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)\`, $(nproc) cores, $(gcc --version | head -1)."
    echo
    echo "| threads | native median | native best | C median | C best | native / C |"
    echo "|---|---|---|---|---|---|"
    echo "| every thread (the plan's ten workers) | $n_all | $n_all_best | $c_all | $c_all_best | $(ratio "$n_all" "$c_all") |"
    echo "| one thread, core $CORE | $n_one | $n_one_best | $c_one | $c_one_best | $(ratio "$n_one" "$c_one") |"
    echo
    echo "Speed-up with threads: native $(ratio "$n_one" "$n_all"), C $(ratio "$c_one" "$c_all")."
} > "$OUT/report.md"
cat "$OUT/report.md"
