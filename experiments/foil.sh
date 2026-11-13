#!/bin/bash
# Two acceptance rules on one model (plan 0.6.0 step 8, D53). MLPerf's closed
# division admits a realization whose task accuracy stays within 99 % (or
# 99.9 %) of the FP32 reference's over the whole validation set; a budget
# admits one whose measured distance to the exact denotation, on the
# envelope's inputs, stays within ε. Both are applied here to the same
# realizations of the corpus GPT, with what each costs.
#
# The weights are D50's all-f64 run (`make curve`): 600 steps on 90 % of the
# Code civil, so the other 10 % — 461 windows — is held out, which an
# accuracy rule needs and `make gpt-trained`'s weights, trained on every
# window, cannot give.
#
# budget side  `presk quantize` on two held-out windows: the gap to the exact
#              loss, split into representation and arithmetic (zero
#              representation for a float realization). It costs one exact
#              run of those windows, once, and one native run of them per
#              realization.
# MLPerf side  next-character accuracy and loss over the 461 held-out
#              windows, against the f32 run's: one pass over the validation
#              set per realization. The model runs in the realization and the
#              metric in f64 — MLPerf's harness scores a model's outputs
#              outside it, and a bf16 mean over 118 016 positions would
#              otherwise round the metric itself to eight bits. A loss is not
#              what MLPerf thresholds; it is given beside the accuracy because
#              perplexity on a held-out corpus is the roadmap's declared NLP
#              measure.
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
MODEL=examples/corpus/nlp/05_gpt.psk
DIMS=(--const D=64 --const H=2 --const L=2 --dim T=256)
W=${W:-out/curve_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
SOLVED=${SOLVED:-out/curve_1e-3.json}   # statement D's realization, solved by `make curve`
OUT=${1:-out/foil.md}
for f in "$W" "$VAL" "$SOLVED"; do [ -f "$f" ] || { echo "missing $f: run make curve first"; exit 1; }; done
MATS='["tok","pos","head","blocks.attn.wq.w","blocks.attn.wk.w","blocks.attn.wv.w","blocks.attn.wo.w","blocks.ff.up.w","blocks.ff.down.w"]'

realization() { # bits scheme -> a file quantising every matrix, f32 elsewhere (statement K's)
    python3 -c "import json,sys; print(json.dumps({'default':'f32','order':{'kind':'sequential'},'quant':{p:{'bits':int(sys.argv[1]),'over':-2,'scheme':sys.argv[2]} for p in json.loads(sys.argv[3])}}))" "$1" "$2" "$MATS" > "out/foil_$1_$2.json"
    echo "@out/foil_$1_$2.json"
}
harness() { # spec -> the same realization with the metric's functions in f64
    python3 - "$1" "$2" <<'PY'
import json, sys
spec, out = sys.argv[1], sys.argv[2]
r = json.load(open(spec[1:])) if spec.startswith("@") else {"default": spec, "order": {"kind": "sequential"}}
metric = ("loss", "accuracy", "hits")
for m in ("storage", "accum"):   # a site named here would override the function's type
    if m in r:
        r[m] = {s: t for s, t in r[m].items() if s.split("/")[0] not in metric}
r["functions"] = {**r.get("functions", {}), **{f: "f64" for f in metric}}
json.dump(r, open(out, "w"))
PY
    echo "@$2"
}
now() { date +%s.%N; }
since() { python3 -c "print(f'{$(now) - $1:.2f}')"; }
value() { awk '{print $NF}'; } # the scalar `presk run` prints last

# The exact reference on the envelope's inputs, once: the one cost the budget
# pays that the accuracy rule does not.
t=$(now)
EXACT=$("$PRESK" run "$MODEL" --entry loss "${DIMS[@]}" --weights "$W" --data "$VAL" --samples 2 --bits 128 --digits 17 2>/dev/null | value)
T_EXACT=$(since "$t")
echo "exact loss on two held-out windows: $EXACT ($T_EXACT s at 128 bits)"

LABELS=("f64" "f32" "bf16" "int8, f32 elsewhere" "int8 pow2" "int4, f32 elsewhere" "int4 pow2" "solved at 1e-3 (statement D)")
SPECS=(f64 f32 bf16 "$(realization 8 absmax)" "$(realization 8 pow2)" "$(realization 4 absmax)" "$(realization 4 pow2)" "@$SOLVED")
declare -a REP ARI TOT TCHK LOSS ACC TPASS
for i in "${!SPECS[@]}"; do
    spec=${SPECS[$i]}
    r=$("$PRESK" quantize "$MODEL" --entry loss "${DIMS[@]}" --weights "$W" --data "$VAL" --samples 2 --from 0 --realize "$spec" --backend native 2>/dev/null)
    REP[$i]=$(echo "$r" | awk '/representation/ {print $3}' | tr -d '(')
    ARI[$i]=$(echo "$r" | awk '/arithmetic/ {print $3}' | tr -d '(')
    TOT[$i]=$(echo "$r" | awk '/total/ {print $3}' | tr -d '(')
    t=$(now); "$PRESK" run "$MODEL" --entry loss "${DIMS[@]}" --weights "$W" --data "$VAL" --samples 2 --backend native --realize "$spec" >/dev/null 2>&1; TCHK[$i]=$(since "$t")
    h=$(harness "$spec" "out/foil_harness_$i.json")
    t=$(now); ACC[$i]=$("$PRESK" run "$MODEL" --entry accuracy "${DIMS[@]}" --weights "$W" --data "$VAL" --backend native --realize "$h" --digits 10 2>/dev/null | value); TPASS[$i]=$(since "$t")
    LOSS[$i]=$("$PRESK" run "$MODEL" --entry loss "${DIMS[@]}" --weights "$W" --data "$VAL" --backend native --realize "$h" --digits 10 2>/dev/null | value)
    echo "${LABELS[$i]}: gap ${TOT[$i]} (${REP[$i]} + ${ARI[$i]}), check ${TCHK[$i]} s; accuracy ${ACC[$i]}, loss ${LOSS[$i]}, pass ${TPASS[$i]} s"
done

python3 - "$OUT" "$EXACT" "$T_EXACT" <<EOF
import sys
out, exact, t_exact = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
labels = """$(printf '%s\n' "${LABELS[@]}")""".splitlines()
col = lambda s: s.split()
rep, ari, tot = col("""${REP[*]}"""), col("""${ARI[*]}"""), col("""${TOT[*]}""")
tchk, acc, loss, tpass = col("""${TCHK[*]}"""), col("""${ACC[*]}"""), col("""${LOSS[*]}"""), col("""${TPASS[*]}""")
ref = labels.index("f32")
a32, l32 = float(acc[ref]), float(loss[ref])
rows = ["| realization | gap to the exact loss | representation | arithmetic | check (s) | accuracy | vs f32 | 99 % | 99.9 % | loss vs f32 | pass (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
for i, name in enumerate(labels):
    a, l = float(acc[i]), float(loss[i])
    rows.append(f"| {name} | {tot[i]} | {rep[i]} | {ari[i]} | {tchk[i]} | {a:.6f} | {a / a32 - 1:+.2e} | {'yes' if a >= 0.99 * a32 else 'NO'} | {'yes' if a >= 0.999 * a32 else 'NO'} | {l / l32 - 1:+.2e} | {tpass[i]} |")
rows += ["",
         f"Budget side: two held-out windows, against the exact loss {exact:.10f} computed once in {t_exact:.1f} s at 128 bits;",
         "'check' is one native run of those windows in the realization. MLPerf side: the 461 held-out windows,",
         "natively; 'pass' is one pass over them (the accuracy entry). The f32 run is the FP32 reference."]
open(out, "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
EOF
