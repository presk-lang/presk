#!/bin/bash
# SR3, the short runs (2026-11-10;
# D195): the e-graph optimizer measured where it is allowed. `--opt` is
# refused with a plan, with `--budget` and with a realization that types
# sites (D133, D134), and it gives no bound, since the precision stages name
# the program before it. So SR3 has no statement: it reports what `--opt`
# changes on the course GPT, in gap and in time, under all-f32 and all-f64
# and the default layout, at two scales:
#
#   d64      D = 64, H = 2, L = 2, `make gpt-trained`'s weights;
#   course   D = 384, H = 6, L = 6, out/course_train_f64 (D129's weights).
#
# rules    `presk dump --stage core-opt`, the optimizer's report, and `presk
#          explain --opt --no-exact --format json`, whose provenance table
#          names the rules each rewritten function applied (D134), on the loss
#          (examples/corpus/nlp/05_gpt.psk) and on its gradient
#          (examples/experiments/gpt_grad.psk's loss_grad, named as `presk
#          train` names it), at B = 16, T = 256;
# kernels  `presk explain --no-exact` on the loss with and without --opt, at
#          each scale in f32 and f64: the launches a call before and after
#          fusion and the values that reach memory, which the time below
#          follows (static, added after the record's run);
# gap      the loss on the inputs of the envelopes of record (D132: windows 0
#          and 1 of out/civil_val_T256.psk.d, the trained weights), natively in
#          f32 and f64 with and without --opt, printed whole, against the exact
#          loss (the reference interpreter at 128 bits); the exact loss of the
#          rewritten program too, which differs from the other only by the
#          128-bit rounding, the rules being exact over ℝ. Beside, at d64
#          only, the gradient the same way;
# time     ROUNDS rounds, the machine alone (the driver waits for the
#          one-minute load under 1.5), on the ten X925 cores pinned with ten
#          threads: at each scale, `presk bench --native-only` on the loss at
#          B = 16, T = 256 with and without --opt, REPEAT calls after one
#          warm-up, the median (D126's protocol, five processes); the order of
#          the two swapped from one round to the next;
# beside   per round, at each scale and in f32 and f64, `presk train` for
#          REPEAT + 1 steps with and without --opt, under the default layout
#          (J′'s path, D127: `--opt` is refused with a plan): the median step
#          after the first, the kernel launches a step, the weights'
#          checksum.
#
# experiments/opt_measured.py reads $OUT into $OUT/report.md. Each process
# is alone in its own scope (MemoryMax 108G, no swap, oom_score_adj 1000).
# The checkpoints are checksummed into $OUT/ckpt.sha256 and removed. An
# output that exists is not taken again; machine.txt is the first run's, and
# every run's start is appended to $OUT/runs.txt.
#
#   ./experiments/opt_measured.sh                everything, then the report
#   ROUNDS=1 ./experiments/opt_measured.sh       one round
#   SCALES=d64 ./experiments/opt_measured.sh     one scale (the gradient's gap is d64's)
#   REPORT_ONLY=1 ./experiments/opt_measured.sh
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
OUT=${OUT:-out/opt-measured}
VAL=${VAL:-out/civil_val_T256.psk.d}
CORES=${CORES:-5-9,15-19}
ROUNDS=${ROUNDS:-5}
REPEAT=${REPEAT:-7}
MEM=${MEM:-108G}
REMAT=${REMAT:-store}
MODEL=examples/corpus/nlp/05_gpt.psk
GRAD=examples/experiments/gpt_grad.psk
read -ra SCALES <<< "${SCALES:-d64 course}"
declare -A CONSTS=([d64]="--const D=64 --const H=2 --const L=2" [course]="--const D=384 --const H=6 --const L=6")
declare -A WEIGHTS=([d64]=out/gpt_trained/final.psk.d [course]=out/course_train_f64/final.psk.d)
WINDOWS=(--data "$VAL" --samples 2 --from 0)

scoped() { systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 choom -n 1000 -- "$@"; }
say() { echo "== $(date -Is) $*"; }
quiet() { while python3 -c "import sys; sys.exit(float(open('/proc/loadavg').read().split()[0]) < 1.5)"; do sleep 10; done; }
opt() { if [ "$1" = opt ]; then echo --opt; fi; }
# run FILE ARGS...: one process into FILE, its /usr/bin/time beside; FILE
# removed if it fails
run() {
    local f=$1
    shift
    scoped /usr/bin/time -v -o "$f.time" "$@" > "$f" 2> "$f.err" || { echo "$f failed"; rm -f "$f"; exit 1; }
}

for f in "$PRESK" "$VAL" "${WEIGHTS[@]}"; do [ -e "$f" ] || { echo "missing $f"; exit 1; }; done
mkdir -p "$OUT/rules" "$OUT/kernels" "$OUT/gap" "$OUT/log" "$OUT/ckpt"

if [ "${REPORT_ONLY:-0}" != 1 ]; then
    say "SR3 on $("$PRESK" --version)"
    echo "run $(date -Iseconds)" >> "$OUT/runs.txt"
    [ -s "$OUT/machine.txt" ] || {
        echo "date $(date -Iseconds)"
        echo "presk $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- crates || echo -dirty), $("$PRESK" --version)"
        echo "kernel $(uname -sr), $(nproc) cores, pinned on $CORES"
        for s in "${SCALES[@]}"; do echo "weights $s $(sha256sum "${WEIGHTS[$s]}")"; done
        echo "windows $(sha256sum "$VAL")"
    } > "$OUT/machine.txt"

    for s in "${SCALES[@]}"; do
        read -ra C <<< "${CONSTS[$s]}"
        for p in loss grad; do
            if [ "$p" = loss ]; then src=(examples/corpus/nlp/05_gpt.psk --entry loss); else src=("$GRAD" --entry loss_grad); fi
            [ -s "$OUT/rules/$s-$p/explain.json" ] && continue
            say "rules, $s, $p"
            "$PRESK" dump "${src[@]}" "${C[@]}" --dim B=16 --dim T=256 --stage core-opt 2> "$OUT/rules/$s-$p.txt" > "$OUT/rules/$s-$p.core-opt" \
                || { echo "dump $s $p failed"; exit 1; }
            "$PRESK" explain "${src[@]}" "${C[@]}" --dim B=16 --dim T=256 --opt --no-exact --format json \
                --out "$OUT/rules/$s-$p" > "$OUT/rules/$s-$p.explain.log" 2>&1 || { echo "explain $s $p failed"; exit 1; }
        done
    done

    for s in "${SCALES[@]}"; do
        read -ra C <<< "${CONSTS[$s]}"
        for r in f32 f64; do
            for o in plain opt; do
                d=$OUT/kernels/$s-loss-$r-$o
                [ -s "$d/explain.md" ] && continue
                say "kernels, $s, the loss, $r, $o"
                "$PRESK" explain "$MODEL" --entry loss "${C[@]}" --dim B=16 --dim T=256 --realize "$r" $(opt $o) --no-exact \
                    --format json --out "$d" > "$d.log" 2>&1 || { echo "explain $s $r $o failed"; exit 1; }
            done
        done
    done

    for s in "${SCALES[@]}"; do
        read -ra C <<< "${CONSTS[$s]}"
        common=("${C[@]}" --dim T=256 --weights "${WEIGHTS[$s]}" "${WINDOWS[@]}" --digits 40 --remat "$REMAT")
        for r in f32 f64; do
            for o in plain opt; do
                f=$OUT/gap/$s-loss-$r-$o.txt
                [ -s "$f" ] && continue
                say "gap, $s, the loss, $r, $o"
                run "$f" "$PRESK" run "$MODEL" --entry loss "${common[@]}" --backend native --realize "$r" $(opt $o)
            done
        done
        for o in plain opt; do
            f=$OUT/gap/$s-loss-exact-$o.txt
            [ -s "$f" ] && continue
            say "gap, $s, the exact loss, $o"
            run "$f" "$PRESK" run "$MODEL" --entry loss "${common[@]}" --bits 128 $(opt $o)
        done
    done
    # beside: the gradient at d64
    read -ra C <<< "${CONSTS[d64]}"
    common=("${C[@]}" --dim T=256 --weights "${WEIGHTS[d64]}" "${WINDOWS[@]}" --digits 40 --remat "$REMAT" --json --result-only)
    for r in f32 f64 exact; do
        for o in plain opt; do
            f=$OUT/gap/d64-grad-$r-$o.json
            [ -s "$f" ] && continue
            say "gap, d64, the gradient, $r, $o"
            if [ "$r" = exact ]; then how=(--bits 128); else how=(--backend native --realize "$r"); fi
            run "$f" "$PRESK" run "$GRAD" --entry loss_grad "${common[@]}" "${how[@]}" $(opt $o)
        done
    done

    for k in $(seq 1 "$ROUNDS"); do
        if [ $((k % 2)) = 1 ]; then order=(plain opt); else order=(opt plain); fi
        for s in "${SCALES[@]}"; do
            read -ra C <<< "${CONSTS[$s]}"
            for o in "${order[@]}"; do
                key=r$k-$s-loss-$o
                grep -q "^native bf16" "$OUT/log/$key.log" 2>/dev/null && continue
                quiet
                say "round $k, $s, the loss, $o"
                scoped /usr/bin/time -v -o "$OUT/log/$key.time" taskset -c "$CORES" \
                    "$PRESK" bench "$MODEL" --entry loss "${C[@]}" --dim B=16 --dim T=256 --weights "${WEIGHTS[$s]}" \
                    --native-only --repeat "$REPEAT" --warmup 1 --threads 10 $(opt $o) > "$OUT/log/$key.log" 2>&1 \
                    || { echo "$key failed"; exit 1; }
            done
        done
        for s in "${SCALES[@]}"; do
            read -ra C <<< "${CONSTS[$s]}"
            for r in f32 f64; do
                for o in "${order[@]}"; do
                    key=r$k-$s-step-$r-$o
                    grep -q "ms a step" "$OUT/log/$key.log" 2>/dev/null && continue
                    quiet
                    say "round $k, beside: $s, the step, $r, $o"
                    scoped /usr/bin/time -v -o "$OUT/log/$key.time" taskset -c "$CORES" \
                        "$PRESK" train "$MODEL" "${C[@]}" --steps $((REPEAT + 1)) --log-every 1 --backend native \
                        --realize "$r" --threads 10 --remat "$REMAT" $(opt $o) --ckpt "$OUT/ckpt/$key" > "$OUT/log/$key.log" 2>&1 \
                        || { echo "$key failed"; exit 1; }
                    (cd "$OUT/ckpt" && sha256sum "$key/final.psk.d") >> "$OUT/ckpt.sha256"
                    rm -rf "${OUT:?}/ckpt/$key"
                done
            done
        done
    done
    say "done"
fi
python3 experiments/opt_measured.py "$OUT" "$REPEAT" > "$OUT/report.md"
cat "$OUT/report.md"
