#!/bin/bash
# The Llama 2 decoder at the 7B configuration's width, run (D104): the
# second model of the paper's RQ5, by BERT-large's protocol (bert.sh, D53).
# coverage/llama.psk with G = 32, so that the width (4096), the heads (32,
# each its own key head), SwiGLU's width (11008) and the vocabulary (32 000)
# are Llama 2 7B's published configuration; L of its 32 blocks, as the
# machine affords them; a learned position table in place of RoPE, which
# needs `concat`. On its initialization, the head drawn as the other
# matrices (the file's head was `zeros` until D104), and on random token
# ids (experiments/llama_ids.py). Not MLPerf's constants: MLPerf's Llama 2
# is the 70B. It measures the compiler's reach and what each realization
# costs on a decoder the project did not design, not its accuracy: there
# are no trained weights, no text and no perplexity.
#
# exact  the denotation of one sequence of T tokens at 128 bits, by the
#        reference interpreter, with its wall time and peak memory
# gaps   f64, f32, bf16, and int8 and int4 weights with f32 arithmetic,
#        natively on that sequence: max|ŷ − y| / max|y| over its T × V
#        logits, which is what a budget on an entry binds (spec §6.1).
#        Statement K's scheme twice: on every matrix but `wo`, the one the
#        file reads through `split`, which the native backend computed
#        wrongly once quantised at the tag v0.10.0 (D104), and on every
#        matrix, which it computes since 0.10.1 (D105's step 6, D106)
# kgaps  the int8 and int4 schemes on every matrix, on the kernel
#        interpreter, which was right at the tag: D104's whole scheme, and
#        since 0.10.1 the oracle the native rows of every matrix equal (up
#        to the two backends' `libm`s)
# split  the quantised gaps split into representation and arithmetic with
#        `presk quantize`, the whole scheme, at SPLIT_L blocks (default 1).
#        At the tag it held the parameters three times at 128 bits, more
#        than the machine had at one block (killed at 108 GB on 16 tokens),
#        so D104's split ran at no block (the embeddings, the final norm and
#        the head; SPLIT_L=0 reads its files); since 0.10.1 one block takes
#        about four hours a scheme and 64 GB (D106)
# time   `presk bench` per sequence at batch 1 (one core: threads split a
#        kernel's outermost loop, the batch axis at batch 1) and batch 20,
#        f32 and int8 among the rows: int8 on every matrix (D106; D104's
#        files, int8 but `wo`, are read too)
# solve  `make compile-times`' four commands on the next-token loss
#        (`loss`, a scalar): check, the envelope, `presk solve --budget
#        1e-3` and the reference, timed for the paper's setup table (model
#        key llama-7b-L<L>; their outputs in out/llama/compile_times_T<T>/),
#        then `presk check --precision` on the solved realization for the
#        bound beside its measured gap
#
# STAGES="exact gaps kgaps split time solve" selects stages (default all
# but solve: at the tag it did not fit in falicon's 119 GB at one block even
# on 16 tokens, D104; since 0.10.1 it does, in 14 h and 72 GB at one block
# and 384 tokens, and not at two, D106; STAGES= only rebuilds the table);
# an output that exists is reused unless FRESH=1. L= the blocks (default 1, the dry run's choice),
# T= the sequence (default 384, BERT's), SAMPLES= the envelope's sequences
# for `solve` (default 1). Run it alone on the measurement machine: the
# exact stage and the solve hold tens of gigabytes, and the time stage
# needs the cores. For a check at tiny constants, G= the key heads,
# LLAMA_CONSTS= the other constants and V= the vocabulary of the ids
# (`G=2 LLAMA_CONSTS="--const D=32 --const H=2 --const FF=64 --const V=100"
# V=100 T=16 LLAMA_OUT=out/llama_tiny`).
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
MODEL=coverage/llama.psk
L=${L:-1}
SPLIT_L=${SPLIT_L:-1}
T=${T:-384}
V=${V:-32000}
G=${G:-32}
SAMPLES=${SAMPLES:-1}
STAGES=${STAGES-exact gaps kgaps split time}
OUT=${1:-out/llama.md}
D=${LLAMA_OUT:-out/llama}
mkdir -p "$D"
has() { [[ " $STAGES " == *" $1 "* ]]; }

# Two sequences of T random ids, seed 0: the first is the sequence measured,
# the second only the envelope's when SAMPLES=2. The same ids twice, as the
# plain `Tokens` that `Llama.apply` takes and as the loss's (inputs, next
# tokens) pair.
if [ "${FRESH:-0}" = 1 ] || [ ! -s "$D/ids_T$T.psk.d" ]; then
    python3 experiments/llama_ids.py $V "$T" 2 0 > "$D/ids_T$T.txt"
    "$PRESK" data pack --type "idx[$V][seq: $T]" "$D/ids_T$T.txt" -o "$D/ids_T$T.psk.d" >/dev/null
    python3 experiments/llama_ids.py $V "$T" 2 0 --shift > "$D/ids_loss_T$T.txt"
    "$PRESK" data pack --type "(idx[$V][seq: $T], idx[$V][seq: $T])" "$D/ids_loss_T$T.txt" -o "$D/ids_loss_T$T.psk.d" >/dev/null
fi
# Statement K's scheme: every matrix quantised over the axis it is
# contracted over (an embedding over its rows, as BERT's), f32 elsewhere.
# `int8`/`int4` leave out `wo`, which the file reads through `split` and
# the native backend at the tag computed wrongly once quantised (D104);
# `int8_all`/`int4_all` are the whole scheme, on every backend since 0.10.1.
MATS='["tok","pos","blocks.attn.wq","blocks.attn.wk","blocks.attn.wv","blocks.ff.gate.w","blocks.ff.up.w","blocks.ff.down.w","head"]'
ALL='["tok","pos","blocks.attn.wq","blocks.attn.wk","blocks.attn.wv","blocks.attn.wo","blocks.ff.gate.w","blocks.ff.up.w","blocks.ff.down.w","head"]'
for b in 8 4; do
    for set in "int$b:$MATS" "int${b}_all:$ALL"; do
        python3 -c "import json,sys; print(json.dumps({'default':'f32','order':{'kind':'sequential'},'quant':{p:{'bits':int(sys.argv[1]),'over':-2} for p in json.loads(sys.argv[2])}}))" "$b" "${set#*:}" > "$D/${set%%:*}.json"
    done
done
SPECS=(f64 f32 bf16 "@$D/int8.json" "@$D/int4.json" "@$D/int8_all.json" "@$D/int4_all.json")
name() { basename "${1#@}" .json; }
CONSTS=(--const G="$G" --const L="$L" ${LLAMA_CONSTS:-})
RUN=(--entry Llama.apply "${CONSTS[@]}" --dim T="$T" --data "$D/ids_T$T.psk.d" --samples 1)

if has exact && { [ "${FRESH:-0}" = 1 ] || [ ! -s "$D/exact_L${L}_T$T.txt" ]; }; then
    /usr/bin/time -v "$PRESK" run "$MODEL" "${RUN[@]}" --bits 128 --digits 17 > "$D/exact_L${L}_T$T.txt" 2> "$D/exact_L${L}_T$T.time"
fi

if has gaps; then
    for spec in "${SPECS[@]}"; do
        f="$D/native_L${L}_T${T}_$(name "$spec")"
        [ "${FRESH:-0}" != 1 ] && [ -s "$f.txt" ] && continue
        /usr/bin/time -v "$PRESK" run "$MODEL" "${RUN[@]}" --backend native --realize "$spec" --digits 17 > "$f.txt" 2> "$f.time"
    done
fi

if has kgaps; then
    for spec in "@$D/int8_all.json" "@$D/int4_all.json"; do
        f="$D/kernel_L${L}_T${T}_$(name "$spec")"
        [ "${FRESH:-0}" != 1 ] && [ -s "$f.txt" ] && continue
        /usr/bin/time -v "$PRESK" run "$MODEL" "${RUN[@]}" --backend kernel --realize "$spec" --digits 17 > "$f.txt" 2> "$f.time"
    done
fi

if has split; then
    for spec in "@$D/int8_all.json" "@$D/int4_all.json"; do
        f="$D/split_L${SPLIT_L}_T${T}_$(name "$spec")"
        [ "${FRESH:-0}" != 1 ] && [ -s "$f.txt" ] && continue
        /usr/bin/time -v "$PRESK" quantize "$MODEL" --entry Llama.apply --const G="$G" --const L="$SPLIT_L" ${LLAMA_CONSTS:-} --dim T="$T" \
            --data "$D/ids_T$T.psk.d" --samples 1 --realize "$spec" --backend native > "$f.txt" 2> "$f.time"
    done
fi

if has time; then
    for b in 1 20; do
        f="$D/bench_L${L}_T${T}_B${b}_int8_all.txt"
        [ "${FRESH:-0}" != 1 ] && [ -s "$f" ] && continue
        "$PRESK" bench "$MODEL" --entry Llama.apply "${CONSTS[@]}" --dim B=$b --dim T="$T" --native-only --repeat 2 \
            --realize "@$D/int8_all.json" > "$f" 2>/dev/null
    done
fi

if has solve; then
    key=llama-7b-L$L
    ct=$D/compile_times_T$T
    if [ "${FRESH:-0}" = 1 ] || [ ! -s "$ct/${key}_solved.json" ]; then
        CT_OUT=$ct LLAMA_G=$G LLAMA_OUT=$D LLAMA_T=$T MODELS=$key SAMPLES=$SAMPLES ./experiments/compile_times.sh "$D/compile_times_L${L}_T$T.md"
    fi
    if [ -s "$ct/${key}_solved.json" ]; then
        /usr/bin/time -v "$PRESK" check "$MODEL" "${CONSTS[@]}" --precision "$ct/${key}_env.json" \
            --realize "@$ct/${key}_solved.json" > "$D/bound_L${L}_T$T.txt" 2> "$D/bound_L${L}_T$T.time"
    fi
fi

python3 - "$D" "$L" "$T" "$OUT" "$SPLIT_L" <<'EOF'
import os, re, sys
from array import array
d, L, T, out, SL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
tag = f"L{L}_T{T}"
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?|-?nan|-?inf")

def values(path):
    # Every number from the first `[[`: presk prints an exact integer without
    # a decimal point, so a pattern that requires one drops values and
    # misaligns every comparison after them. A NaN or an infinity is kept,
    # so that a broken run shows as one. None if the run left no output.
    try:
        t = open(path).read()
        return (float(m.group()) for m in NUM.finditer(t, t.index("[[")))
    except (FileNotFoundError, ValueError):
        return None

def field(path, key):
    try:
        for line in open(path):
            if key in line:
                return line.split()[-1]
    except FileNotFoundError:
        pass
    return None

def wall(path):
    v = field(path, "Elapsed")
    return v and sum(float(x) * 60 ** i for i, x in enumerate(reversed(v.split(":"))))

def peak(path):
    v = field(path, "Maximum resident")
    return v and int(v) / 2**20

def hm(s):
    s = round(s)
    return f"{s // 3600} h {s % 3600 // 60:02d} min {s % 60:02d} s" if s >= 3600 else f"{s // 60} min {s % 60:02d} s" if s >= 60 else f"{s} s"

rows = []
ex = values(f"{d}/exact_{tag}.txt")
if ex is not None:
    ex = array("d", ex)
    scale = max(abs(v) for v in ex)
    w, m = wall(f"{d}/exact_{tag}.time"), peak(f"{d}/exact_{tag}.time")
    rows += [f"Exact reference: {len(ex)} logits of one sequence of {T} tokens, {L} block(s), at 128 bits, max|y| = {scale:.4f}"
             + (f", {hm(w)} and {m:.1f} GB" if w and m else "") + ".", "",
             "| realization | backend | gap to the exact logits | inside 1e-2 | wall | peak |", "|---|---|---|---|---|---|"]
    runs = [("f64", "native", "f64"), ("f32", "native", "f32"), ("bf16", "native", "bf16"),
            ("int8 weights but `wo`, f32 elsewhere", "native", "int8"), ("int4 weights but `wo`, f32 elsewhere", "native", "int4"),
            ("int8 weights, every matrix, f32 elsewhere", "native", "int8_all"), ("int4 weights, every matrix, f32 elsewhere", "native", "int4_all"),
            ("int8 weights, every matrix, f32 elsewhere", "kernel", "int8_all"), ("int4 weights, every matrix, f32 elsewhere", "kernel", "int4_all")]
    for lab, be, f in runs:
        p = f"{d}/{be}_{tag}_{f}"
        v = values(p + ".txt")
        if v is None:
            continue
        n, g = 0, 0.0
        for a, b in zip(v, ex):
            n += 1
            g = max(g, abs(a - b)) if a == a else float("nan")
        assert n == len(ex), (p, n, len(ex))
        g /= scale
        w, m = wall(p + ".time"), peak(p + ".time")
        rows.append(f"| {lab} | {be} | {g:.3e} | {'yes' if g <= 1e-2 else 'NO'} | {hm(w) if w else ''} | {f'{m:.1f} GB' if m else ''} |")
    rows.append("")

stag = f"L{SL}_T{T}"
but = "" if SL == "0" else " but `wo`"   # no block, no `wo` to leave out
def present(f):
    return os.path.exists(f"{d}/split_{stag}_{f}.txt") and os.path.getsize(f"{d}/split_{stag}_{f}.txt")
# The whole scheme since D106; D104's files, `int8.json` and `int4.json` at
# no block, where there is no `wo` to leave out, when it is absent.
split = []
for b in (8, 4):
    if present(f"int{b}_all"):
        split.append((f"int{b} weights, every matrix", f"{d}/split_{stag}_int{b}_all"))
    elif present(f"int{b}"):
        split.append((f"int{b} weights" + but, f"{d}/split_{stag}_int{b}"))
if split:
    rows += [f"Split at {SL} block(s) (presk quantize, relative to the largest exact logit at {SL} block(s)):", "",
             "| realization | representation | arithmetic | total | wall | peak |", "|---|---|---|---|---|---|"]
    for lab, p in split:
        t = open(p + ".txt").read()
        rel = {k: re.search(k + r"\s+\S+ \((\S+) relative\)", t).group(1) for k in ("representation", "arithmetic", "total")}
        w, m = wall(p + ".time"), peak(p + ".time")
        rows.append(f"| {lab} | {rel['representation']} | {rel['arithmetic']} | {rel['total']} | {hm(w) if w else ''} | {f'{m:.1f} GB' if m else ''} |")
    rows.append("")

benches = [(b, s, w) for b in (1, 20) for s, w in (("", "int8 weights but `wo`"), ("_int8_all", "int8 weights, every matrix"))]
for b, suffix, what in benches:
    p = f"{d}/bench_{tag}_B{b}{suffix}.txt"
    if os.path.exists(p) and os.path.getsize(p):
        rows += [f"`presk bench` at batch {b}, {L} block(s), T = {T} (median ms of two calls; per sequence = median / {b}; "
                 f"the quantised row is {what}):", "", "```"]
        rows += [line.rstrip() for line in open(p) if line.strip()]
        rows += ["```", ""]

ct = f"{d}/compile_times_{tag}.md"
if os.path.exists(ct):
    rows += [f"The solve (`make compile-times`' commands on `loss`, {L} block(s); their outputs in {d}/compile_times_T{T}/):", ""]
    rows += [line.rstrip() for line in open(ct) if line.startswith("|")]
    for name in ("calibrate", "solve", "exact"):
        tp = f"{d}/compile_times_T{T}/llama-7b-L{L}_{name}.time"
        w, m = wall(tp), peak(tp)
        if w and m:
            rows.append(f"- {name}: {hm(w)}, peak {m:.1f} GB")
    sp = f"{d}/compile_times_T{T}/llama-7b-L{L}_solve.txt"
    if os.path.exists(sp):
        rows += ["", "```"] + [line.rstrip() for line in open(sp) if line.strip()][-25:] + ["```"]
    bp = f"{d}/bound_{tag}.txt"
    if os.path.exists(bp) and os.path.getsize(bp):
        rows += ["", "The bound on the solved realization (`presk check --precision`):", "", "```"]
        rows += [line.rstrip() for line in open(bp) if line.strip()][-12:] + ["```"]
    rows.append("")

open(out, "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
EOF
