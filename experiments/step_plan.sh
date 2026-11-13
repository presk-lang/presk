#!/bin/bash
# SR4, the short runs (2026-11-10;
# D194): the budgeted step under the cost model's plan. J′'s 1.75× (D127:
# the step solved at 1e-3 on the update against all-f64) was timed under the
# default layout, about ten times slower than a planned step. At the course
# constants (B = 16, T = 256), three rows, each under its own plan:
#
#   f64      all-f64;
#   f32      all-f32;
#   solved   the step solved at 1e-3 (SOLVED, experiments/step_table.sh's).
#
# plans    `presk plan --mode training` on loss_grad (examples/experiments/gpt_grad.psk)
#          and `presk plan` on adamw.step, under the row's realization and the
#          frozen profile, merged as experiments/pytorch_baseline.sh merges
#          them, into $OUT/plans/step-<row>.json;
# bits     two steps under the default layout and two under the plan, the
#          weights compared bit for bit: a layout moves no bit, so the plan
#          keeps J′'s update gap;
# rounds   ROUNDS rounds, each the three rows in turn, on the ten X925 cores
#          pinned with ten threads, the machine alone (the driver waits for
#          the one-minute load under 1.5): per row a process of REPEAT + 1
#          steps, whose median step after the first `presk train` prints
#          (D126's protocol, T4's `presk train` row), then a process of one
#          step and one of four, whose wall times give J′'s own measure
#          (experiments/j_step.sh's: the four minus the one, over three);
# beside   not the statement's: a process of LONG steps a row (its median step
#          and its peak), and `presk train --host-profile` at REPEAT + 1 and at
#          LONG steps for f32 and solved, where the solved step's time over
#          all-f32's goes.
#
# experiments/step_plan.py reads $OUT into $OUT/report.md, with SR4's
# verdict. Each run is alone in its own scope (MemoryMax 108G, no swap,
# oom_score_adj 1000). The checkpoints are checksummed into $OUT/ckpt.sha256
# and removed. A round whose times exist is not taken again; machine.txt is
# the first run's, and every run's start is appended to $OUT/runs.txt.
#
#   ./experiments/step_plan.sh                everything, then the report
#   ROUNDS=1 ./experiments/step_plan.sh       one round
#   REPORT_ONLY=1 ./experiments/step_plan.sh
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
OUT=${OUT:-out/step-plan}
PROFILE=${PROFILE:-$HOME/presk-data/ra4-model-0.8.0/profile.json}
SOLVED=${SOLVED:-out/step_1e-3.json}
CORES=${CORES:-5-9,15-19}
ROUNDS=${ROUNDS:-5}
REPEAT=${REPEAT:-7}
LONG=${LONG:-64}
MEM=${MEM:-108G}
REMAT=${REMAT:-store}
MODEL=examples/corpus/nlp/05_gpt.psk
GRAD=examples/experiments/gpt_grad.psk
DIMS=(--dim B=16 --dim T=256)
ROWS=(f64 f32 solved)

scoped() { systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 choom -n 1000 -- "$@"; }
say() { echo "== $(date -Is) $*"; }
spec() { if [ "$1" = solved ]; then echo "@$SOLVED"; else echo "$1"; fi; }
quiet() { while python3 -c "import sys; sys.exit(float(open('/proc/loadavg').read().split()[0]) < 1.5)"; do sleep 10; done; }
# train KEY STEPS LAYOUT-ARGS...: one process, its wall time into $OUT/wall.txt,
# its weights' checksum into $OUT/ckpt.sha256, the weights removed
train() {
    local key=$1 steps=$2 s e
    shift 2
    s=$(date +%s.%N)
    scoped /usr/bin/time -v -o "$OUT/log/$key.time" taskset -c "$CORES" \
        "$PRESK" train "$MODEL" --steps "$steps" --log-every 1 --backend native "$@" --threads 10 \
        --remat "$REMAT" --ckpt "$OUT/ckpt/$key" > "$OUT/log/$key.log" 2>&1 || { echo "$key failed"; exit 1; }
    e=$(date +%s.%N)
    echo "$key $(python3 -c "print(f'{$e - $s:.3f}')")" >> "$OUT/wall.txt"
    (cd "$OUT/ckpt" && sha256sum "$key/final.psk.d") >> "$OUT/ckpt.sha256"
    rm -rf "${OUT:?}/ckpt/$key"
}

for f in "$PRESK" "$PROFILE" "$SOLVED"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
mkdir -p "$OUT/plans" "$OUT/log" "$OUT/ckpt"

if [ "${REPORT_ONLY:-0}" != 1 ]; then
    say "SR4 on $("$PRESK" --version)"
    echo "run $(date -Iseconds)" >> "$OUT/runs.txt"
    [ -s "$OUT/machine.txt" ] || {
        echo "date $(date -Iseconds)"
        echo "presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates || echo -dirty), $("$PRESK" --version)"
        echo "kernel $(uname -sr), $(nproc) cores, pinned on $CORES"
        echo "solved $(sha256sum "$SOLVED")"
        echo "profile $(sha256sum "$PROFILE")"
    } > "$OUT/machine.txt"
    cp "$SOLVED" "$OUT/plans/solved-realization.json"

    for row in "${ROWS[@]}"; do
        P=$OUT/plans
        [ -s "$P/step-$row.json" ] && continue
        say "plans, $row"
        "$PRESK" plan "$GRAD" --entry loss_grad "${DIMS[@]}" --realize "$(spec "$row")" --mode training \
            --profile "$PROFILE" --out "$P/grad-$row.json" > "$P/grad-$row.log" 2>&1 || { echo "plan $row failed"; exit 1; }
        "$PRESK" plan "$MODEL" --entry adamw.step "${DIMS[@]}" --realize "$(spec "$row")" \
            --profile "$PROFILE" --out "$P/adam-$row.json" > "$P/adam-$row.log" 2>&1 || { echo "plan $row failed"; exit 1; }
        python3 - "$P" "$row" <<'EOF'
import json, sys
p, row = sys.argv[1:]
g, a = (json.load(open(f"{p}/{n}-{row}.json")) for n in ("grad", "adam"))
assert not set(g["layout"]) & set(a["layout"]) and g["realization"] == a["realization"]
json.dump(dict(g, layout={**g["layout"], **a["layout"]}), open(f"{p}/step-{row}.json", "w"), indent=2)
EOF
    done

    for row in "${ROWS[@]}"; do
        grep -q "^bits-plan-$row " "$OUT/wall.txt" 2>/dev/null && continue
        quiet
        say "bits, $row: two steps under the default layout and under the plan"
        train "bits-default-$row" 2 --realize "$(spec "$row")"
        train "bits-plan-$row" 2 --plan "@$OUT/plans/step-$row.json"
    done

    for r in $(seq 1 "$ROUNDS"); do
        for row in "${ROWS[@]}"; do
            grep -q "^r$r-$row-s4 " "$OUT/wall.txt" 2>/dev/null && continue
            quiet
            say "round $r, $row"
            train "r$r-$row-s$((REPEAT + 1))" $((REPEAT + 1)) --plan "@$OUT/plans/step-$row.json"
            train "r$r-$row-s1" 1 --plan "@$OUT/plans/step-$row.json"
            train "r$r-$row-s4" 4 --plan "@$OUT/plans/step-$row.json"
        done
    done

    for row in "${ROWS[@]}"; do
        grep -q "^long-$row " "$OUT/wall.txt" 2>/dev/null && continue
        quiet
        say "beside: $LONG steps, $row"
        train "long-$row" "$LONG" --plan "@$OUT/plans/step-$row.json"
    done
    for row in f32 solved; do
        for n in $((REPEAT + 1)) "$LONG"; do
            grep -q "^hp$n-$row " "$OUT/wall.txt" 2>/dev/null && continue
            quiet
            say "beside: the host by phase, $n steps, $row"
            train "hp$n-$row" "$n" --plan "@$OUT/plans/step-$row.json" --host-profile
        done
    done
    say "done"
fi
python3 experiments/step_plan.py "$OUT" "$REPEAT" "$LONG" > "$OUT/report.md"
cat "$OUT/report.md"
