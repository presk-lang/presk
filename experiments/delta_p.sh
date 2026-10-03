#!/bin/bash
# δ_p, the reference's own gap, measured (the reference's chapter 4, W16;
# D98). The reference interpreter at p bits is a realization of the
# denotation, not the denotation, and every budget is discharged against it
# with its gap δ_p taken as 0. The denotation is not computable, so δ_p is
# estimated as the distance between the reference at p bits and the
# reference at a higher precision P: every rounding shrinks by 2^-(P-p) when
# p rises to P, so |f_p − f_P| ≈ δ_p, and the ratio δ_p / δ_2p checks it
# (near 2^-p: the roundings of one run are the only thing that moves).
#
# Three entries of the course GPT at D = 64 on `make gpt-trained`'s weights
# and the first eight windows of the Code civil, the inputs the calibrations
# of the `precision` chapter use, each printed at 330 digits (more than
# 1024 bits resolve) and compared in experiments/delta_p.py:
#
# loss     `loss`, the entry the forward budgets bind: |L_p − L_P| / |L_P|,
#          p = 128, 256, 512 against P = 1024
# logits   `GPT.apply` in its inference reading, the entry a serve's budget
#          binds: max_i |y_p[i] − y_P[i]| / max_i |y_P[i]|, the same p and P
# grad     the gradient of `loss` at the weights, in the 2-norm relative to
#          the gradient's and in the ∞-norm, p = 128, 256, 512 against P =
#          1000: at p ≥ 1016 the interpreter's π for gelu′ (prims.rs,
#          `machin_pi`) compares its terms with 2^-(p+8) computed in f64,
#          which is 0, and never stops (recorded in D98; not fixed here).
#          The exact step itself (`presk calibrate --step`) prints its gaps
#          at four digits and writes no parameter at full precision, so the
#          step is checked, not measured: the all-f64 row at each p must
#          read the same.
#
# `presk run` evaluates in the inference reading (dropout masks are 1), so
# nothing on these paths is drawn; the weights are binary64 numbers read
# exactly and the literals denote binary64 numbers (the reference's
# definition "What a literal denotes"): the function approximated is the
# same at every p. A run whose output exists is not run again; delete
# out/delta_p to start over. The time and the peak memory of each run are
# recorded beside it (out/delta_p/<entry>_<bits>.time).
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
MODEL=corpus/nlp/05_gpt.psk
W=${W:-out/gpt_trained/final.psk.d}
DATA=corpus/nlp/data/civil_T256.psk.d
OUT=${OUT:-out/delta_p}
SAMPLES=${SAMPLES:-8}
DIGITS=${DIGITS:-330}
BITS_FWD=${BITS_FWD:-"128 256 512 1024"}
BITS_GRAD=${BITS_GRAD:-"128 256 512 1000"}
BITS_STEP=${BITS_STEP:-"128 256 512 1000"}
[ -f "$W" ] || { echo "no trained weights at $W: make gpt-trained"; exit 1; }
[ -f "$DATA" ] || { echo "no data at $DATA: make corpus-gpt"; exit 1; }
mkdir -p "$OUT"
# The two entries the program lacks, appended to a copy of it: the logits
# with the batch as the last argument, so that --data fills it, and the
# gradient of `loss` as a value `presk run` prints.
WRAP=$OUT/gpt_delta_p.psk
{
    cat "$MODEL"
    cat <<'EOF'

-- Appended by experiments/delta_p.sh (D98): the logits with the batch last,
-- so that `presk run --data` fills it, and the gradient of `loss` as an
-- entry, so that `presk run` prints it at --digits.
fn logits(θ : params GPT, k : key, batch : (Tokens, Tokens)) -> ℝ[batch: B, seq: T, vocab: V] uses Rand =
  let (x, y) = batch
  GPT.apply(θ, x, k)

fn loss_grad(θ : params GPT, k : key, batch : (Tokens, Tokens)) -> params GPT uses Rand =
  grad(λ θ. loss(θ, k, batch))(θ)
EOF
} > "$WRAP"
# The copy reads the training run's data path relative to itself.
[ -e "$OUT/data" ] || ln -s ../../corpus/nlp/data "$OUT/data"
ARGS=(--const D=64 --const H=2 --const L=2 --dim T=256 --weights "$W" --data "$DATA" --samples "$SAMPLES")
# timed NAME CMD...: runs CMD, its stdout to $OUT/NAME.json, its wall time
# in seconds and peak resident set in kB to $OUT/NAME.time.
timed() {
    local name=$1; shift
    [ -s "$OUT/$name.json" ] && [ -s "$OUT/$name.time" ] && return
    echo "== $name: $*"
    python3 - "$OUT/$name.json" "$OUT/$name.time" "$@" <<'EOF'
import resource, subprocess, sys, time
out, tfile, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
t = time.monotonic()
with open(out, "w") as f:
    r = subprocess.run(cmd, stdout=f)
s = time.monotonic() - t
kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
with open(tfile, "w") as f:
    f.write(f"{s:.1f} {kb}\n")
print(f"   {s:.1f} s, {kb / 1024:.0f} MB, exit {r.returncode}")
sys.exit(r.returncode)
EOF
}
for b in $BITS_FWD; do
    timed "loss_$b" "$PRESK" run "$WRAP" --entry loss "${ARGS[@]}" --bits "$b" --digits "$DIGITS" --json
    timed "logits_$b" "$PRESK" run "$WRAP" --entry logits "${ARGS[@]}" --bits "$b" --digits "$DIGITS" --json
done
for b in $BITS_GRAD; do
    timed "grad_$b" "$PRESK" run "$WRAP" --entry loss_grad "${ARGS[@]}" --bits "$b" --digits "$DIGITS" --json
done
# The step: D52's state (step 101 of 600 from the weights, 100 warm-up steps
# in f64, two windows from 4000; step_table.sh), the all-f64 realization
# against the exact step at each p. Four digits, so a check and not a
# measurement.
for b in $BITS_STEP; do
    timed "step_$b" "$PRESK" calibrate --step "$MODEL" --const D=64 --const H=2 --const L=2 --weights "$W" \
        --warmup 100 --steps 600 --samples 2 --from 4000 --bits "$b" --realize f64
done
python3 experiments/delta_p.py "$OUT" > "$OUT/report.md"
cat "$OUT/report.md"
