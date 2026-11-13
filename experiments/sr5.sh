#!/bin/bash
# SR5, the short runs (2026-11-10;
# plan 0.17.2 §3.3, D201), on D200's two timers, alone on falicon:
#
#   serve   SR5a: D190's five serving rows, each `presk serve` for 32 tokens
#           greedily from D180's first prompt (16 tokens) under one
#           realization, one thread pinned to core 5 (an X925), by D190's
#           own command, ROUNDS rounds of the five rows in turn (f32, the
#           realization by hand, int8 absmax, A = the solve at ε*(1e-2), B =
#           the eighth solve). Per process: `/usr/bin/time`'s elapsed
#           seconds and peak, the tokens, and the summary line's compiling
#           and computing (D200); the rest is the elapsed time less both.
#   modes   SR5b: the Llama 2 decoder block at the 7B width, G = 32, L = 1,
#           batch 1, T = 384, all-f32 under `presk plan`'s plan with the
#           frozen profile (D124's V5 setting), on the ten X925 cores pinned
#           with ten threads (D100's). The program is D124's, coverage/llama.psk
#           as tagged v0.12.0 (MODES_REV), taken from git into $OUT/modes: the
#           version's examples/coverage/llama.psk has RoPE since 0.17.0, 53
#           launches a call against 34, and 0.11.0's realization names the
#           older program's values. Under this binary both plans are V5's
#           byte for byte. PROCS processes one after another, each
#           `presk bench --host-profile --launches --warmup 1 --repeat 7`.
#           Per process: the call's median without the timers and with them,
#           every launch's time and its large buffers' addresses, the main
#           thread's last core, the huge pages, and `/usr/bin/time -v`. If
#           no process of the PROCS is slow, PROCS more run under 0.11.0's
#           solved realization (D119 saw the two modes in it too).
#   placement  SR5b's test of placement, added after `modes` found the
#           spread in one launch (the attention's output projection, an `ein`
#           at one lane reading the 64 MiB weight column by column): PROCS
#           processes of `modes`' command under huge pages
#           (GLIBC_TUNABLES=glibc.malloc.hugetlb=1, which has glibc madvise
#           its mappings; the machine's mode is `madvise`), row f32-huge,
#           alternated with PROCS more under the default 4 KiB pages, row
#           f32-4k. Under 4 KiB pages a buffer's cache sets above its page
#           offset come from the physical frames the kernel gives it, which
#           differ from process to process and stay for its life; under
#           2 MiB pages they are its virtual address's. Fixed before the
#           runs: if the placement of the pages is the cause, f32-huge's
#           calls are within 5 % of each other (max ÷ min under 1.05) while
#           f32-4k's spread as f32's; if f32-huge's spread too, with a slow
#           process among them by the 20 % rule, it is ruled out; between
#           the two, it is left open. The level under huge pages is not
#           predicted.
#   aslr    SR5b's second test of placement, added after `placement` found
#           the spread under huge pages too (1.31x): address-space
#           randomization still moves the buffers from process to process,
#           and under 2 MiB pages those virtual offsets are their cache
#           sets. PROCS processes under huge pages with randomization off
#           (`setarch -R`, the same addresses in every process), row
#           f32-huge-fixed, alternated with PROCS under 4 KiB pages with it
#           off, row f32-4k-fixed. Fixed before the runs: if f32-huge-fixed's
#           calls are within 5 % of each other, the buffers' placement within
#           their pages is named as the cause; if not, the cause is left open
#           between the physical placement above 2 MiB (the memory's
#           channels and banks) and a cause other than placement.
#   control  SR5-0's control, added after `serve`'s first round found f32
#           5.3 % faster than D190: f32's serving row under D190's own binary
#           (CONTROL_PRESK, v0.17.1's code), ROUNDS processes, each in its
#           own scope as `serve`'s are. D190's time stage ran its serving
#           processes in one scope of 108 GB after its `presk bench`
#           processes, and B's, 0.9 % from this run's, each in a scope of its
#           own. If D190's binary takes D190's time here, the timers or the
#           build differ; if it takes `serve`'s, the setting does.
#   report  experiments/sr5.py: $OUT/report.md and report.json, with SR5-0's
#           verdict and SR5a's and SR5b's reports.
#
# The slow mode, fixed before the runs: a process is slow when its call's
# median without the timers exceeds the fastest process's of the same row by
# more than 20 % (D124: 2.44 s against 3.7-3.9 s, 1.5x; the spread between
# processes of one mode is under 2 %, D190's max ÷ min 1.004-1.018).
#
# Each process runs alone in its own scope (MemoryMax 108G, no swap,
# oom_score_adj 1000), after the machine was idle for two seconds (under 5 %
# of one core busy beyond the driver). A process whose output exists is not
# taken again; machine.txt is the first run's, every run's start is appended
# to runs.txt. Nothing here is a row of record: D190's and
# D124's stay; these are new rows beside them.
#
#   ./experiments/sr5.sh                     modes, serve, placement, control, aslr, then the report
#   STAGES=serve ./experiments/sr5.sh        one stage
#   STAGES=report ./experiments/sr5.sh
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
OUT=${OUT:-out/sr5-0.17.2}
STAGES=${STAGES:-modes serve placement control aslr report}
ROUNDS=${ROUNDS:-3}
PROCS=${PROCS:-20}
REPEAT=${REPEAT:-7}
MEM=${MEM:-108G}
PROFILE=${PROFILE:-$HOME/presk-data/ra4-model-0.8.0/profile.json}
SOLVED011=${SOLVED011:-out/v5-0.12.0/realizations/solved-0.11.0.json}
CORES=${CORES:-5-9,15-19}
ONE=${ONE:-5}
CONTROL_PRESK=${CONTROL_PRESK:-out/bin/presk-0.17.1-1d865e3}
W=${W:-out/llama3/llama-3.2-1b.psk.d}
PROMPTS=${PROMPTS:-out/llama3/prompts.txt}
BUDGET=${BUDGET:-out/llama3-budget}
LLAMA3=examples/coverage/llama3.psk
MODES_REV=${MODES_REV:-v0.12.0}
NAMES=(f32 bf16-params-f32 int8-absmax solve-eps-1e-2 solve-eps-1e-2-mingain0)

want() { [[ " $STAGES " == *" $1 "* ]]; }
scoped() { systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 choom -n 1000 -- "$@"; }
say() { echo "== $(date -Is) $*"; }
# The machine idle for two seconds: the busy jiffies of every core, less the
# driver's own sampling, under 5 % of one core.
quiet() {
    while ! python3 - <<'EOF'
import os, sys, time
def busy():
    f = open("/proc/stat").readline().split()[1:]
    v = [int(x) for x in f]
    return sum(v) - v[3] - v[4], sum(v)
b0, t0 = busy(); time.sleep(2); b1, t1 = busy()
cores = os.cpu_count()
sys.exit(0 if (b1 - b0) / ((t1 - t0) / cores) < 0.05 else 1)
EOF
    do sleep 5; done
}
spec() { # a serving row's name -> --realize's argument, as D190's time stage
    case $1 in
        f32) echo f32 ;;
        int*) echo "@out/llama3/realize/$1.json" ;;
        bf16-params-f32) echo "@experiments/llama/$1.json" ;;
        solve-*) echo "@$BUDGET/solve/${1#solve-}.json" ;;
    esac
}

need=("$PRESK")
want serve && need+=("$W" "$PROMPTS" experiments/llama/bf16-params-f32.json out/llama3/realize/int8-absmax.json "$BUDGET/solve/eps-1e-2.json" "$BUDGET/solve/eps-1e-2-mingain0.json")
want modes && need+=("$PROFILE" "$SOLVED011")
want control && need+=("$CONTROL_PRESK" "$W" "$PROMPTS")
for f in "${need[@]}"; do [ -e "$f" ] || { echo "sr5.sh: missing $f" >&2; exit 1; }; done
mkdir -p "$OUT/serve" "$OUT/modes/plans" "$OUT/modes/log"
LLAMA=$OUT/modes/llama-$MODES_REV.psk
if want modes; then
    [ -s "$LLAMA" ] || git show "$MODES_REV:coverage/llama.psk" > "$LLAMA" || { echo "sr5.sh: no coverage/llama.psk at $MODES_REV" >&2; exit 1; }
fi

if want modes || want serve; then
    say "SR5 on $("$PRESK" --version), stages $STAGES"
    echo "run $(date -Iseconds) stages $STAGES" >> "$OUT/runs.txt"
    [ -s "$OUT/machine.txt" ] || {
        echo "date $(date -Iseconds)"
        echo "presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates || echo -dirty), $("$PRESK" --version)"
        echo "binary $(sha256sum "$PRESK")"
        echo "kernel $(uname -sr), $(nproc) cores, governor $(cat /sys/devices/system/cpu/cpu5/cpufreq/scaling_governor 2>/dev/null)"
        echo "transparent huge pages $(cat /sys/kernel/mm/transparent_hugepage/enabled), defrag $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
        echo "serve: one thread on core $ONE, rounds $ROUNDS; modes: cores $CORES, ten threads, $PROCS processes, repeat $REPEAT"
        echo "weights $(sha256sum "$W" | cut -d' ' -f1)  $W"
        echo "profile $(sha256sum "$PROFILE")"
        echo "decoder $MODES_REV:coverage/llama.psk $(git rev-parse "$MODES_REV:coverage/llama.psk")"
    } > "$OUT/machine.txt"
fi

# --- SR5b: the two modes -------------------------------------------------------
# A realization's plan: modes_plan NAME REALIZATION
modes_plan() {
    local plan=$OUT/modes/plans/$1-plan-B1.json
    [ -s "$plan" ] && return 0
    say "plan, $1"
    "$PRESK" plan "$LLAMA" --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 \
        --realize "$2" --profile "$PROFILE" --out "$plan" > "$OUT/modes/plans/$1.log" 2>&1 || { echo "plan $1 failed"; exit 1; }
}
# One process: modes_proc KEY PLAN TUNABLES [PREFIX...]
modes_proc() {
    local key=$1 plan=$OUT/modes/plans/$2-plan-B1.json
    [ -s "$OUT/modes/$key.txt" ] && grep -q "Exit status: 0" "$OUT/modes/log/$key.time" 2>/dev/null && return 0
    quiet
    local tun=$3
    shift 3
    say "modes, $key${tun:+, GLIBC_TUNABLES=$tun}${1:+, $*}"
    GLIBC_TUNABLES=$tun scoped /usr/bin/time -v -o "$OUT/modes/log/$key.time" taskset -c "$CORES" "$@" \
        "$PRESK" bench "$LLAMA" --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 \
        --plan "@$plan" --threads 10 --warmup 1 --repeat "$REPEAT" --host-profile --launches \
        > "$OUT/modes/$key.txt" 2>&1 < /dev/null || echo "modes $key failed: $OUT/modes/$key.txt"
}
# One row's processes: modes_row NAME REALIZATION
modes_row() {
    local i
    modes_plan "$1" "$2"
    for i in $(seq -w 1 "$PROCS"); do modes_proc "$1-$i" "$1" ""; done
}
if want modes; then
    modes_row f32 f32
    python3 experiments/sr5.py slow "$OUT" f32
    case $? in
    0) say "modes: a slow process among f32's $PROCS; no further row" ;;
    1)
        say "modes: no slow process among f32's $PROCS; $PROCS more under 0.11.0's solved realization"
        cp "$SOLVED011" "$OUT/modes/plans/solved-0.11.0.realization.json"
        modes_row solved-0.11.0 "@$OUT/modes/plans/solved-0.11.0.realization.json"
        ;;
    *) echo "sr5.sh: f32's processes could not be read" >&2; exit 1 ;;
    esac
fi

# --- SR5b: placement ---------------------------------------------------------------
if want placement; then
    modes_plan f32 f32
    for i in $(seq -w 1 "$PROCS"); do
        modes_proc "f32-huge-$i" f32 glibc.malloc.hugetlb=1
        modes_proc "f32-4k-$i" f32 ""
    done
fi

# --- SR5b: placement with the addresses fixed ----------------------------------------
if want aslr; then
    modes_plan f32 f32
    for i in $(seq -w 1 "$PROCS"); do
        modes_proc "f32-huge-fixed-$i" f32 glibc.malloc.hugetlb=1 setarch "$(uname -m)" -R
        modes_proc "f32-4k-fixed-$i" f32 "" setarch "$(uname -m)" -R
    done
fi

# --- SR5a: the serving split ----------------------------------------------------
if want serve; then
    head -1 "$PROMPTS" > "$OUT/serve/prompt.txt"
    cmp -s "$OUT/serve/prompt.txt" "$BUDGET/time/prompt.txt" || { echo "sr5.sh: the prompt is not D190's" >&2; exit 1; }
    for round in $(seq 1 "$ROUNDS"); do
        for name in "${NAMES[@]}"; do
            o=$OUT/serve/serve-$name-$round.txt
            [ -s "$o" ] && grep -q " kB$" "$o" && continue
            quiet
            say "serve, $name, round $round"
            scoped bash -c '{ /usr/bin/time -f "%e s %M kB" taskset -c "$0" "$@" < /dev/null; }' "$ONE" \
                "$PRESK" serve "$LLAMA3" --weights "$W" --prompt "$(cat "$OUT/serve/prompt.txt")" --tokens 32 \
                --temperature 0 --backend native --realize "$(spec "$name")" > "$o" 2>&1 || echo "serve $name failed: $o"
        done
    done
fi

# --- SR5-0: the control -----------------------------------------------------------
if want control; then
    head -1 "$PROMPTS" > "$OUT/serve/prompt.txt"
    grep -q "^control " "$OUT/machine.txt" || echo "control $("$CONTROL_PRESK" --version), $(sha256sum "$CONTROL_PRESK")" >> "$OUT/machine.txt"
    for round in $(seq 1 "$ROUNDS"); do
        o=$OUT/serve/control-f32-$round.txt
        [ -s "$o" ] && grep -q " kB$" "$o" && continue
        quiet
        say "control, f32 under $CONTROL_PRESK, round $round"
        scoped bash -c '{ /usr/bin/time -f "%e s %M kB" taskset -c "$0" "$@" < /dev/null; }' "$ONE" \
            "$CONTROL_PRESK" serve "$LLAMA3" --weights "$W" --prompt "$(cat "$OUT/serve/prompt.txt")" --tokens 32 \
            --temperature 0 --backend native --realize f32 > "$o" 2>&1 || echo "control $round failed: $o"
    done
fi

if want report; then
    say report
    python3 experiments/sr5.py report "$OUT" --d190 "$BUDGET/time" || exit 1
fi
say done
