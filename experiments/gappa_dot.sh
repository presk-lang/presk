#!/bin/bash
# A contraction's bound proved in Gappa, beside the compiler's bound and the
# measured gap (D131): a copy of experiments/gappa.sh (D108's layer norm) for
# `dot` of experiments/gappa_dot.psk, [M, K] · [K, N] in f32, at K = 8, 16,
# 32, 64, 128, on an 8 × K and a K × 8 matrix synthesized from seed 0, under
# the sequential order and under interleaved(n) for n = 4, 8, 16 (where n
# divides K and K/n ≥ 2).
#
# For each width, in order, every output under out/gappa-dot/:
#
#   env_K<K>.json            `presk calibrate --precision` on the entry, at 128
#                            bits: the operands' and the output's magnitudes,
#                            the hypotheses Gappa is given.
#   order_<O>.json           the realization: f32, the reduction order O.
#   check_K<K>_<O>.txt       `presk check --precision … --model all --measure`
#                            under it: the four rules' bounds and the measured
#                            gap of the native f32 kernel on the envelope's
#                            inputs.
#   kernel_K<K>_<O>.txt      `presk dump --stage kernel` under it: the loops
#                            and the order of the sums the Gappa file mirrors;
#                            its `dot` kernel must be the bare product's
#                            (checked below).
#   dot_K<K>_<O>_<H>.g       the Gappa file experiments/gappa_dot.py writes, H
#   dot_K<K>_<O>_<H>.out     = operands (the operands' magnitudes) or sites (the
#                            output's too), and Gappa's answer to `|Y - y| in ?`;
#   dot_K<K>_<O>_<H>.proof.g    the same with the bound found as its goal, a
#   dot_K<K>_<O>_<H>.proof.out  proof Gappa checks (its exit code is the check).
#   summary.md               the table: width × order × {measured, k, normwise,
#                            cond, rss, proved}, the ratios, and the checks
#                            (proved ≥ measured, cond ≥ measured).
#
# Nothing here is timed; any machine. WIDTHS=, ROWS=, ORDERS=, HYPS= and
# REALIZE= select; a `.g` whose `.out` exists is not run again (delete
# out/gappa-dot to start over).
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
GAPPA=${GAPPA:-gappa}
OUT=${OUT:-out/gappa-dot}
WIDTHS=${WIDTHS:-"8 16 32 64 128"}
ROWS=${ROWS:-8}
ORDERS=${ORDERS:-"sequential interleaved(4) interleaved(8) interleaved(16)"}
HYPS=${HYPS:-"operands sites"}
REALIZE=${REALIZE:-f32}
FILE=experiments/gappa_dot.psk
command -v "$GAPPA" > /dev/null || { echo "gappa_dot.sh: no gappa on the PATH (apt install gappa, or GAPPA=…)"; exit 1; }
[ -x "$PRESK" ] || { echo "gappa_dot.sh: no $PRESK: make build"; exit 1; }
mkdir -p "$OUT"
$GAPPA --version 2>&1 | head -1 > "$OUT/gappa-version.txt"
$PRESK --version > "$OUT/presk-version.txt"
rc_all=0

# The bound Gappa prints for a query, as the exact dyadic `Mb-E` of the
# result's upper end (gappa.sh's, D108).
bound_of() {
    python3 - "$1" <<'PY'
import re, sys
best = None
for m, e in re.findall(r"in \[0, ([0-9]+)b(-?[0-9]+) \{", open(sys.argv[1]).read()):
    v = int(m) * 2.0 ** int(e)
    if best is None or v > best[0]:
        best = (v, f"{m}b{e}")
print(best[1] if best else "")
PY
}
# The `dot` kernel of a dump, its text alone.
dot_kernel() { sed -n '/^kernel dot /,/^}/p' "$1"; }
tag() { echo "$1" | tr -d '()'; }

for K in $WIDTHS; do
    ENV=$OUT/env_K$K.json
    echo "== K = $K, $ROWS × $K · $K × $ROWS, $REALIZE"
    [ -f "$ENV" ] || $PRESK calibrate --precision $FILE --entry dot --dim M=$ROWS --dim N=$ROWS --const K=$K --bits 128 --out "$ENV" > /dev/null || exit 1
    for O in $ORDERS; do
        case $O in
            sequential) n=1; R='{"default": "'$REALIZE'", "order": {"kind": "sequential"}}' ;;
            interleaved*) n=${O#interleaved(}; n=${n%)}
                { [ $((K % n)) -eq 0 ] && [ $((K / n)) -ge 2 ]; } || continue
                R='{"default": "'$REALIZE'", "order": {"kind": "interleaved", "partials": '$n'}}' ;;
        esac
        T=$(tag "$O")
        RJ=$OUT/order_$T.json
        echo "$R" > "$RJ"
        C=$OUT/check_K${K}_$T.txt
        $PRESK check --precision "$ENV" $FILE --realize "@$RJ" --model all --measure --sites 10 > "$C" || exit 1
        $PRESK dump --stage kernel $FILE --entry dot --dim M=$ROWS --dim N=$ROWS --const K=$K --realize "@$RJ" > "$OUT/kernel_K${K}_$T.txt" || exit 1
        $PRESK dump --stage kernel $FILE --entry bare --dim M=$ROWS --dim N=$ROWS --const K=$K --realize "@$RJ" > "$OUT/kernel_bare_K${K}_$T.txt" || exit 1
        if cmp -s <(dot_kernel "$OUT/kernel_K${K}_$T.txt") <(dot_kernel "$OUT/kernel_bare_K${K}_$T.txt"); then
            k_ok="the bare product's"
        else
            k_ok="DIFFERS from the bare product's"; rc_all=1
        fi
        echo "  $O: measured $(sed -n 's/^  measured *\([^ ]*\).*/\1/p' "$C"), kernel $k_ok"
        for H in $HYPS; do
            G=$OUT/dot_K${K}_${T}_$H
            python3 experiments/gappa_dot.py "$ENV" --width $K --order "$O" --realize $REALIZE --hypotheses $H -o "$G.g" > /dev/null || exit 1
            if [ ! -f "$G.out" ]; then
                start=$(date +%s)
                $GAPPA "$G.g" > "$G.out" 2>&1
                echo "gappa: rc=$? in $(( $(date +%s) - start )) s" >> "$G.out"
            fi
            B=$(bound_of "$G.out")
            [ -n "$B" ] || { echo "    $H: Gappa found no bound (see $G.out)"; rc_all=1; continue; }
            python3 experiments/gappa_dot.py "$ENV" --width $K --order "$O" --realize $REALIZE --hypotheses $H --prove "$B" -o "$G.proof.g" > /dev/null || exit 1
            # The proof with Gappa's change threshold at 10⁻⁴, as gappa.sh does.
            if $GAPPA -Echange-threshold=0.0001 "$G.proof.g" > "$G.proof.out" 2>&1; then
                echo "    $H: proved |Y - y| <= $B = $(python3 -c "m,e='$B'.split('b'); print(f'{int(m)*2.0**int(e):.5e}')") ($(sed -n 's/gappa: rc=[0-9]* in \(.*\)/\1/p' "$G.out"))"
            else
                echo "    $H: the proof of $B FAILED (see $G.proof.out)"; rc_all=1
            fi
        done
    done
done

# The table.
python3 - "$OUT" $ROWS $REALIZE "$ORDERS" $WIDTHS <<'EOF' | tee "$OUT/summary.md"
import os, re, sys
out, rows, realize, orders, *widths = sys.argv[1:]
print("| K | order | measured | `k` | `normwise` | `cond` | `rss` (estimate) | proved, operands | proved, sites | `cond` / measured | proved / measured | proved / `cond` |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
ok = True
def x(r):
    return f"{r:.1f}×" if r < 10 else f"{r:.0f}×" if r < 1e4 else f"{r:.1e}×"
for K in widths:
    for O in orders.split():
        T = O.replace("(", "").replace(")", "")
        try:
            t = open(f"{out}/check_K{K}_{T}.txt").read()
        except FileNotFoundError:
            continue
        def rule(name):
            return float(re.search(rf"\b{name}\s+([0-9.e+-]+)", t).group(1))
        meas = float(re.search(r"measured\s+([0-9.e+-]+)", t).group(1))
        k, nw, cond, rss = rule("k"), rule("normwise"), rule("cond"), rule("rss")
        proved = {}
        for H in ["operands", "sites"]:
            try:
                g = open(f"{out}/dot_K{K}_{T}_{H}.out").read()
            except FileNotFoundError:
                continue
            bs = [int(m) * 2.0 ** int(e) for m, e in re.findall(r"in \[0, ([0-9]+)b(-?[0-9]+) \{", g)]
            if bs:
                proved[H] = max(bs)
        p = min(proved.values()) if proved else float("nan")
        if (proved and p < meas) or cond < meas:
            ok = False
        po = f"{proved['operands']:.2e}" if "operands" in proved else ""
        ps = f"{proved['sites']:.2e}" if "sites" in proved else ""
        print(f"| {K} | {O} | {meas:.2e} | {k:.2e} | {nw:.2e} | {cond:.2e} | {rss:.2e} | {po} | {ps} | {x(cond / meas)} | {x(p / meas) if proved else ''} | {f'{p / cond:.2f}×' if proved else ''} |")
print()
print(f"{rows} × K · K × {rows}, {realize}; the ratios read the tighter proved bound. Proved ≥ measured and `cond` ≥ measured: {'hold' if ok else 'VIOLATED'}")
EOF
grep -q VIOLATED "$OUT/summary.md" && rc_all=1
exit $rc_all
