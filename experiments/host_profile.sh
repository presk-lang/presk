#!/bin/bash
# The native host measured (plan 0.10.0 step 5, D95; the manual's `cli`,
# `presk bench --host-profile`): the course GPT at B = 16, T = 256, f32,
# under the plans the frozen profile gives (made as `make pytorch-baseline`
# makes them), its native backend's time split by phase between the kernels
# and the host that launches them. A measurement, not a statement: it
# decides nothing, and orders step 6's fixes.
#
#   experiments/host_profile.sh OUT
#
# Four workloads: the forward pass (`GPT.apply` with the checkpoint CKPT),
# the gradient (`loss_grad` of examples/experiments/gpt_grad.psk), the optimizer's
# step (`adamw.step`), each by `presk bench --host-profile` (REPEAT calls
# timed by phase and REPEAT not, alternated, after one warm-up call, the
# outputs of the two kinds of call compared bit for bit); and the training
# step, `presk train --host-profile` over STEPS steps (the first left out),
# beside a run without the flag whose weights must be the same bytes.
#
# Three configurations: as in the baseline, the ten X925 cores (CORES)
# pinned with taskset, ten threads, and every core unpinned, twenty threads;
# then the ten X925 cores again with glibc's allocator told never to map a
# block of its own nor return memory to the system (HEAP), so that after
# the warm-up no page is touched for the first time: the difference is what
# the page faults of the host's allocations cost, phase by phase. CONFIGS
# names the configurations to run, all three by default (plan 0.10.0 step
# 6 measures each fix pinned alone: CONFIGS=10). PRESK, the binary, may be
# another build's (an absolute path). A quiet machine: nothing beside it.
# Writes OUT/report.md.
set -eu -o pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
PRESK=${PRESK:-target/release/presk}
case $PRESK in /*) ;; *) PRESK=$ROOT/$PRESK ;; esac
PROFILE=${PROFILE:-$HOME/presk-data/ra4-model-0.8.0/profile.json}
CKPT=${CKPT:-out/course_probe_f32/final.psk.d}
case $CKPT in /*) ;; *) CKPT=$ROOT/$CKPT ;; esac
CORES=${CORES:-5-9,15-19}
REPEAT=${REPEAT:-7}
STEPS=${STEPS:-8}
OUT=${1:?usage: experiments/host_profile.sh OUT}
FILE=examples/corpus/nlp/05_gpt.psk
GRAD=examples/experiments/gpt_grad.psk
DATA=examples/corpus/nlp/data/civil_T256.psk.d
DIMS=(--dim B=16 --dim T=256)
read -r -a CONFIGS <<< "${CONFIGS:-10 20 10-heap}"
HEAP=${HEAP:-glibc.malloc.mmap_max=0:glibc.malloc.trim_threshold=1099511627776}

for f in "$PRESK" "$PROFILE" "$CKPT" "$DATA"; do
    [ -e "$f" ] || { echo "host_profile.sh: missing $f (the data: make experiment-gpt; the checkpoint: CKPT)" >&2; exit 1; }
done
mkdir -p "$OUT/log" "$OUT/train"
OUT=$(cd "$OUT" && pwd)
say() { echo "-- $*"; }
run() { # config -- command: the command under the configuration's threads, pinning and allocator
    local config=$1
    shift
    case $config in
        10) taskset -c "$CORES" "$@" --threads 10 ;;
        20) "$@" --threads 20 ;;
        10-heap) GLIBC_TUNABLES=$HEAP taskset -c "$CORES" "$@" --threads 10 ;;
        *) echo "host_profile.sh: unknown configuration $config (10, 20, 10-heap)" >&2; exit 1 ;;
    esac
}

say "the machine"
{
    echo "date $(date -Iseconds)"
    echo "presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates experiments || echo -dirty), $("$PRESK" --version), $PRESK"
    echo "configurations ${CONFIGS[*]}"
    echo "kernel $(uname -sr), $(nproc) cores"
    lscpu | grep -E "^(Nom de modèle|Model name)" | sed 's/  */ /g'
    echo "load $(cut -d' ' -f1-3 /proc/loadavg)"
} > "$OUT/machine.txt"
cat "$OUT/machine.txt"

# The plans, as experiments/pytorch_baseline.sh makes them: the forward
# pass's, the gradient's, the optimizer's, and the step's (the two merged).
say "plans (the frozen profile)"
P=$OUT/plans
mkdir -p "$P"
"$PRESK" plan "$FILE" --entry GPT.apply "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/forward.json" > "$P/forward.log" 2>&1
"$PRESK" plan "$GRAD" --entry loss_grad "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/grad.json" > "$P/grad.log" 2>&1
# `presk train`'s plan, of the program with its masks (0.16.0, D160, D164).
"$PRESK" plan "$GRAD" --entry loss_grad "${DIMS[@]}" --realize f32 --mode training --profile "$PROFILE" --out "$P/grad-train.json" > "$P/grad-train.log" 2>&1
"$PRESK" plan "$FILE" --entry adamw.step "${DIMS[@]}" --realize f32 --profile "$PROFILE" --out "$P/adam.json" > "$P/adam.log" 2>&1
python3 - "$P" <<'EOF'
import json, sys
p = sys.argv[1]
g, a = (json.load(open(f"{p}/{n}.json")) for n in ("grad-train", "adam"))
assert not set(g["layout"]) & set(a["layout"])
json.dump(dict(g, layout={**g["layout"], **a["layout"]}), open(f"{p}/step.json", "w"), indent=2)
EOF

profile() { # key config -- presk bench args
    local key=$1 config=$2
    shift 2
    run "$config" "$PRESK" bench "$@" --warmup 1 --repeat "$REPEAT" --host-profile > "$OUT/log/$key.log" 2>&1
}
for c in "${CONFIGS[@]}"; do
    say "forward pass, $c"
    profile "fwd.$c" "$c" "$FILE" --entry GPT.apply "${DIMS[@]}" --plan "@$P/forward.json" --weights "$CKPT"
    say "gradient, $c"
    profile "grad.$c" "$c" "$GRAD" --entry loss_grad "${DIMS[@]}" --plan "@$P/grad.json" --weights "$CKPT"
    say "optimizer step, $c"
    profile "adam.$c" "$c" "$FILE" --entry adamw.step "${DIMS[@]}" --plan "@$P/adam.json" --weights "$CKPT"
    say "training step, $c, without and with the timers"
    for how in plain profiled; do
        flag=()
        [ "$how" = profiled ] && flag=(--host-profile)
        run "$c" "$PRESK" train "$FILE" --steps "$STEPS" --log-every 1 --backend native --plan "@$P/step.json" --ckpt "$OUT/train/$how-$c" "${flag[@]}" > "$OUT/log/step.$how.$c.log" 2>&1
    done
    cmp "$OUT/train/plain-$c/final.psk.d" "$OUT/train/profiled-$c/final.psk.d"
done
# Every configuration trains the same weights.
for c in "${CONFIGS[@]}"; do cmp "$OUT/train/plain-${CONFIGS[0]}/final.psk.d" "$OUT/train/plain-$c/final.psk.d"; done

python3 experiments/host_profile.py "$OUT" > "$OUT/report.md"
cat "$OUT/report.md"
