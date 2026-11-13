#!/bin/bash
# One nest's bound proved in Gappa, beside the compiler's bound and the
# measured gap (D108): `LayerNorm{D}.apply` of examples/spec/layernorm/,
# in f32, at D = 8 and 64, on a batch of rows synthesized from seed 0.
#
# For each width, in order, every output under out/gappa/:
#
#   env_D<D>.json        `presk calibrate --precision` on the entry, at 128
#                        bits: the magnitude envelope the bound is conditional
#                        on (every site's largest and smallest |v|, each
#                        sum's Σ|terms|), the hypotheses Gappa is given.
#   check_D<D>.txt       `presk check --precision … --model all --measure`:
#                        the four rules' bounds and the measured gap of the
#                        native f32 kernels against the reference on the
#                        envelope's own inputs, the sites' table under `cond`.
#   kernel_D<D>.txt      `presk dump --stage kernel` of the entry in f32: the
#   realize_D<D>.txt     loops the kernels run and the order of their sums,
#                        which the Gappa file mirrors; the dump of the
#                        example's own entry `LayerNorm{D}.apply` must be
#                        identical to the wrapper's (checked below).
#   ln_D<D>_<H>.g        the Gappa file experiments/gappa_layernorm.py writes
#   ln_D<D>_<H>.out      from the envelope, hypotheses H = sites (the
#                        envelope's magnitudes at every site) or input (the
#                        inputs' ranges alone), with Gappa's answer to the
#                        query `|Y0 - y0| in ?`;
#   ln_D<D>_<H>.proof.g  the same file with the bound found as its goal, a
#   ln_D<D>_<H>.proof.out  proof Gappa checks (its exit code is the check).
#   check_example_D<D>.txt  `presk check --precision` on the example's own
#                        entry `LayerNorm{D}.apply`, whose module dimension
#                        the command does not bind (D108): the bound it
#                        prints there is not its rule's.
#   summary.md           the table: width × hypotheses × {measured, k,
#                        normwise, cond, rss, proved}, with the ratios, and
#                        the two checks (proved ≥ measured, cond ≥ measured).
#
# Nothing here is timed; any machine. The reference at D = 64 on 8 rows takes
# seconds; Gappa takes seconds at D = 8 and minutes at D = 64. The files the
# record keeps (the .g, the .out, the dumps, the summary) are copied to
# docs/gappa/ by hand with the report. WIDTHS=, BATCH=, HYPS= and REALIZE=
# select; a `.g` whose `.out` exists is not run again (delete out/gappa to
# start over).
set -u
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
GAPPA=${GAPPA:-gappa}
OUT=${OUT:-out/gappa}
WIDTHS=${WIDTHS:-"8 64"}
BATCH=${BATCH:-8}
HYPS=${HYPS:-"sites input"}
REALIZE=${REALIZE:-f32}
FILE=examples/experiments/gappa_layernorm.psk
EXAMPLE=examples/spec/layernorm/layernorm.psk
command -v "$GAPPA" > /dev/null || { echo "gappa.sh: no gappa on the PATH (apt install gappa, or GAPPA=…)"; exit 1; }
[ -x "$PRESK" ] || { echo "gappa.sh: no $PRESK: make build"; exit 1; }
mkdir -p "$OUT"
$GAPPA --version 2>&1 | head -1 > "$OUT/gappa-version.txt"
$PRESK --version > "$OUT/presk-version.txt"
rc_all=0

# The bound Gappa prints for a query, the largest over the goals, as the
# exact dyadic `Mb-E` of the result's upper end — not the six-digit decimal
# in braces beside it, which Gappa rounds to nearest and which a proof's
# goal then fails by a hair (it did, on a bf16 row).
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

for D in $WIDTHS; do
    ENV=$OUT/env_D$D.json
    echo "== D = $D, batch $BATCH, $REALIZE"
    [ -f "$ENV" ] || $PRESK calibrate --precision $FILE --entry 'Norm.apply' --dim batch=$BATCH --const D=$D --bits 128 --out "$ENV" || exit 1
    $PRESK check --precision "$ENV" $FILE --realize $REALIZE --model all --measure --sites 20 > "$OUT/check_D$D.txt" || exit 1
    sed -n '1,8p' "$OUT/check_D$D.txt"
    # The kernels and their order, the wrapper's and the example's.
    $PRESK dump --stage kernel $FILE --entry 'Norm.apply' --dim batch=$BATCH --const D=$D --realize $REALIZE > "$OUT/kernel_D$D.txt" || exit 1
    $PRESK dump --stage realize $FILE --entry 'Norm.apply' --dim batch=$BATCH --const D=$D --realize $REALIZE > "$OUT/realize_D$D.txt" || exit 1
    $PRESK dump --stage kernel $EXAMPLE --entry "LayerNorm{$D}.apply" --dim batch=$BATCH --realize $REALIZE > "$OUT/kernel_example_D$D.txt" || exit 1
    if cmp -s "$OUT/kernel_D$D.txt" "$OUT/kernel_example_D$D.txt"; then
        echo "  kernels: the wrapper's are the example's, byte for byte"
    else
        echo "  kernels: the wrapper's DIFFER from the example's (see $OUT/kernel_*_D$D.txt)"; rc_all=1
    fi
    # The example's own entry under check: its module dimension unbound (D108).
    $PRESK calibrate --precision $EXAMPLE --entry "LayerNorm{$D}.apply" --dim batch=$BATCH --bits 128 --out "$OUT/env_example_D$D.json" > /dev/null || exit 1
    $PRESK check --precision "$OUT/env_example_D$D.json" $EXAMPLE --realize $REALIZE --model all --measure --sites 0 > "$OUT/check_example_D$D.txt" || exit 1
    for H in $HYPS; do
        G=$OUT/ln_D${D}_$H
        ALL=""; [ "$D" -le 8 ] && ALL="--all-rows"
        python3 experiments/gappa_layernorm.py "$ENV" --width $D --realize $REALIZE --hypotheses $H $ALL -o "$G.g" > /dev/null || exit 1
        if [ ! -f "$G.out" ]; then
            start=$(date +%s)
            $GAPPA "$G.g" > "$G.out" 2>&1
            echo "gappa: rc=$? in $(( $(date +%s) - start )) s" >> "$G.out"
        fi
        B=$(bound_of "$G.out")
        [ -n "$B" ] || { echo "  $H: Gappa found no bound (see $G.out)"; rc_all=1; continue; }
        # The proof: the same file with the bound as its goal (the largest
        # over the rows when every row is a goal).
        python3 experiments/gappa_layernorm.py "$ENV" --width $D --realize $REALIZE --hypotheses $H $ALL --prove "$B" -o "$G.proof.g" > /dev/null || exit 1
        # Gappa stops refining an enclosure when a step improves it by less
        # than its change threshold (1 % by default); with a goal to meet
        # instead of a query it stopped short of the query's own bound on
        # one row (8.12e-6 against 7.96e-6, D = 16, one row), so the proof
        # runs with the threshold at 10⁻⁴ and reaches it.
        if $GAPPA -Echange-threshold=0.0001 "$G.proof.g" > "$G.proof.out" 2>&1; then
            echo "  $H: proved |y - exact| <= $B = $(python3 -c "m,e='$B'.split('b'); print(f'{int(m)*2.0**int(e):.5e}')") ($(grep -c Warning "$G.out") warning(s), $(sed -n 's/gappa: rc=[0-9]* in \(.*\)/\1/p' "$G.out"))"
        else
            echo "  $H: the proof of $B FAILED (see $G.proof.out)"; rc_all=1
        fi
    done
done

# The table.
python3 - "$OUT" $BATCH $REALIZE $WIDTHS <<'EOF' | tee "$OUT/summary.md"
import re, sys
out, batch, realize, *widths = sys.argv[1:]
print(f"| D | hypotheses | measured | `k` | `normwise` | `cond` | `rss` (estimate) | proved (Gappa) | `cond` / measured | proved / measured | proved / `cond` |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
ok = True
for D in widths:
    t = open(f"{out}/check_D{D}.txt").read()
    def row(name):
        m = re.search(rf"\b{name}\s+([0-9.e+-]+)", t)
        return float(m.group(1))
    meas = float(re.search(r"measured\s+([0-9.e+-]+)", t).group(1))
    k, nw, cond, rss = row("k"), row("normwise"), row("cond"), row("rss")
    for H in ["sites", "input"]:
        try:
            g = open(f"{out}/ln_D{D}_{H}.out").read()
        except FileNotFoundError:
            continue
        bs = [int(m) * 2.0 ** int(e) for m, e in re.findall(r"in \[0, ([0-9]+)b(-?[0-9]+) \{", g)]
        if not bs:
            continue
        p = max(bs)
        if p < meas or cond < meas:
            ok = False
        def x(r):
            return f"{r:.0f}×" if r < 1e4 else f"{r:.1e}×"
        if H == "sites":
            print(f"| {D} | {H} | {meas:.2e} | {k:.2e} | {nw:.2e} | {cond:.2e} | {rss:.2e} | **{p:.2e}** | {x(cond/meas)} | {x(p/meas)} | {p/cond:.2f}× |")
        else:
            print(f"| {D} | {H} | {meas:.2e} | | | | | **{p:.2e}** | | {x(p/meas)} | {x(p/cond)} |")
print()
print(f"batch {batch}, {realize}; proved ≥ measured and `cond` ≥ measured: {'hold' if ok else 'VIOLATED'}")
EOF
# The checks, once more, as the exit code.
grep -q VIOLATED "$OUT/summary.md" && rc_all=1
exit $rc_all
