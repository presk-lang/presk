#!/bin/bash
# BERT-large inference, run (plan 0.6.0 step 8, D53). The one MLPerf benchmark
# within reach, at MLPerf's own constants — 24 blocks, width 1024, 16 heads,
# context 384, vocabulary 30522 — on its initialization and on SQuAD-shaped
# features with random token ids (experiments/bert_features.py). It measures
# the compiler's reach and what each acceptance rule costs on the benchmark's
# model, not its accuracy: there are no trained weights, no SQuAD and no F1.
#
# exact  the denotation of one feature at 128 bits, by the reference
#        interpreter: about two hours
# gaps   f64, f32, bf16, and int8 and int4 weights with f32 arithmetic,
#        natively on that feature: max|ŷ − y| / max|y| over the SQuAD
#        logits, which is what a budget on an entry binds (spec §6.1)
# split  the quantised gaps split into representation and arithmetic with
#        `presk quantize`, on two of the 24 blocks: at 24 an exact run on the
#        dequantised weights would cost two more hours per scheme
# time   `presk bench` per sequence of 384 tokens, at batch 1 — on one core:
#        threads split a kernel's outermost loop, and at batch 1 that loop is
#        the batch axis — and at batch 20, which occupies the cores
#
# STAGES="exact gaps split time" selects stages (default all; STAGES= only
# rebuilds the table); an exact output already in out/bert is reused unless
# FRESH=1. L=2 runs every stage at two blocks, in minutes rather than hours.
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
MODEL=examples/coverage/bert.psk
L=${L:-24}
STAGES=${STAGES-exact gaps split time}
OUT=${1:-out/bert.md}
D=out/bert
mkdir -p "$D"
has() { [[ " $STAGES " == *" $1 "* ]]; }

if [ "${FRESH:-0}" = 1 ] || [ ! -s "$D/features.psk.d" ]; then
    python3 experiments/bert_features.py 30522 384 20 0 > "$D/features.txt"
    "$PRESK" data pack --type '(idx[30522][seq: 384], idx[2][seq: 384], ℝ[seq2: 384])' "$D/features.txt" -o "$D/features.psk.d" >/dev/null
fi
# Statement K's realization on BERT: every matrix quantised over the axis it
# is contracted over, f32 elsewhere.
MATS='["bert.emb.tok","bert.emb.pos","bert.emb.seg","bert.blocks.attn.wq.w","bert.blocks.attn.wk.w","bert.blocks.attn.wv.w","bert.blocks.attn.wo.w","bert.blocks.up.w","bert.blocks.down.w","head.span.w"]'
for b in 8 4; do
    python3 -c "import json,sys; print(json.dumps({'default':'f32','order':{'kind':'sequential'},'quant':{p:{'bits':int(sys.argv[1]),'over':-2} for p in json.loads(sys.argv[2])}}))" "$b" "$MATS" > "$D/int$b.json"
done
SPECS=(f64 f32 bf16 "@$D/int8.json" "@$D/int4.json")
LABELS=(f64 f32 bf16 "int8 weights, f32 elsewhere" "int4 weights, f32 elsewhere")
name() { basename "${1#@}" .json; }
RUN=(--entry squad --const L="$L" --dim T=384 --data "$D/features.psk.d" --samples 1)

if has exact && { [ "${FRESH:-0}" = 1 ] || [ ! -s "$D/exact_L$L.txt" ]; }; then
    /usr/bin/time -v "$PRESK" run "$MODEL" "${RUN[@]}" --bits 128 --digits 17 > "$D/exact_L$L.txt" 2> "$D/exact_L$L.time"
fi

if has gaps; then
    for spec in "${SPECS[@]}"; do
        /usr/bin/time -v "$PRESK" run "$MODEL" "${RUN[@]}" --backend native --realize "$spec" --digits 17 > "$D/native_L${L}_$(name "$spec").txt" 2> "$D/native_L${L}_$(name "$spec").time"
    done
fi

if has split; then
    for spec in "@$D/int8.json" "@$D/int4.json"; do
        "$PRESK" quantize "$MODEL" --entry squad --const L=2 --dim T=384 --data "$D/features.psk.d" --samples 1 \
            --realize "$spec" --backend native > "$D/split_L2_$(name "$spec").txt" 2>/dev/null
    done
fi

if has time; then
    for b in 1 20; do
        "$PRESK" bench "$MODEL" --entry BertSquad.apply --const L="$L" --dim B=$b --dim T=384 --native-only --repeat 2 \
            --realize "@$D/int8.json" > "$D/bench_L${L}_B$b.txt" 2>/dev/null
    done
    "$PRESK" bench "$MODEL" --entry BertSquad.apply --const L=2 --dim B=20 --dim T=384 --native-only --repeat 2 \
        --realize "@$D/int4.json" > "$D/bench_L2_B20_int4.txt" 2>/dev/null
fi

python3 - "$D" "$L" "$OUT" "${LABELS[@]}" <<'EOF'
import os, re, sys
d, L, out, labels = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
files = ["f64", "f32", "bf16", "int8", "int4"]

def logits(path):
    # Every number from the first `[[`: presk prints an exact integer without
    # a decimal point, so a pattern that requires one drops values and
    # misaligns every comparison after them. None if the run left no output.
    try:
        t = open(path).read()
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", t[t.index("[["):])]
    except (FileNotFoundError, ValueError):
        return None

def wall(path):
    try:
        for line in open(path):
            if "Elapsed" in line:
                h = line.split()[-1].split(":")
                return sum(float(x) * 60 ** i for i, x in enumerate(reversed(h)))
    except FileNotFoundError:
        pass
    return None

def peak(path):
    try:
        for line in open(path):
            if "Maximum resident" in line:
                return int(line.split()[-1]) / 2**20
    except FileNotFoundError:
        pass
    return None

rows = []
ex = logits(f"{d}/exact_L{L}.txt")
if ex:
    scale = max(abs(v) for v in ex)
    w, m = wall(f"{d}/exact_L{L}.time"), peak(f"{d}/exact_L{L}.time")
    rows += [f"Exact reference: {len(ex)} logits of one feature at 128 bits, max|y| = {scale:.4f}"
             + (f", {w / 60:.0f} min and {m:.0f} GB" if w and m else "") + ".", "",
             "| realization | gap to the exact logits | inside 1e-2 |", "|---|---|---|"]
    for f, lab in zip(files, labels):
        p = f"{d}/native_L{L}_{f}.txt"
        v = logits(p)
        if v is None:
            print(f"no output in {p}", file=sys.stderr)
            continue
        assert len(v) == len(ex), (p, len(v), len(ex))
        g = max(abs(a - b) for a, b in zip(v, ex)) / scale
        rows.append(f"| {lab} | {g:.3e} | {'yes' if g <= 1e-2 else 'NO'} |")
    rows.append("")

split = [(lab, f"{d}/split_L2_{f}.txt") for f, lab in zip(files, labels) if os.path.exists(f"{d}/split_L2_{f}.txt") and os.path.getsize(f"{d}/split_L2_{f}.txt")]
if split:
    rows += ["Split on two of the 24 blocks (presk quantize, relative to max|y|):", "",
             "| realization | representation | arithmetic | total |", "|---|---|---|---|"]
    for lab, p in split:
        t = open(p).read()
        rel = {k: re.search(k + r"\s+\S+ \((\S+) relative\)", t).group(1) for k in ("representation", "arithmetic", "total")}
        rows.append(f"| {lab} | {rel['representation']} | {rel['arithmetic']} | {rel['total']} |")
    rows.append("")

for b in (1, 20):
    p = f"{d}/bench_L{L}_B{b}.txt"
    if os.path.exists(p) and os.path.getsize(p):
        rows += [f"`presk bench` at batch {b}, {L} blocks (median ms of two calls; per sequence = median / {b}):", "", "```"]
        rows += [line.rstrip() for line in open(p) if line.strip()]
        rows += ["```", ""]
p = f"{d}/bench_L2_B20_int4.txt"
if os.path.exists(p) and os.path.getsize(p):
    rows += ["int4 at batch 20, two blocks:", "", "```"] + [line.rstrip() for line in open(p) if line.strip()] + ["```", ""]

open(out, "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
EOF
