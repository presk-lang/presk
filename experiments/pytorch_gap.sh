#!/bin/bash
# PyTorch's own gap to the exact model, and its flips (D130): the course GPT
# at D = 64, H = 2, L = 2, the margins' model (`make margins`, D118, D125:
# out/curve_train_f64, the 461 held-out windows of 256, 118 016 positions),
# exported by `presk export --target pytorch --mode readable` and run by
# PyTorch eager in eval mode on presk's checkpoint; its logits against the
# exact ones make margins took at 128 bits, beside the compiler's own
# realizations of the same model on the same windows.
#
#   MARGINS=out/margins   make margins' output: the exact logits
#                         (exact/*.hilo), the compiler's rows (native/,
#                         decisions/), val.json; run make margins first
#                         with EXTRA="d124-1e-3=@CURVE/curve_1e-3.json
#                         logits-4.67e-4=@out/v6-0.12.0/solved.json", the
#                         curve's realization at 10⁻³ at 0.12.0 (D124; in
#                         presk-project data/epsilon-curve-0.12.0.tar.xz)
#                         and the one solved on the logits (V6)
#
# Into out/pytorch-gap/:
#   export/               the PyTorch module, `--train` for its pskd.py reader
#   margins/              make margins' layout for margins.py report: the
#                         exact logits and the compiler's rows linked from
#                         MARGINS, PyTorch's rows written by
#                         experiments/pytorch_gap.py
#   <row>.log             each PyTorch run: torch's version, threads, dtype
#   margins.md            experiments/margins.py report on those rows: ε*,
#                         every row's largest gap, flips, bound #{m < 2ε}
#   gaps.md               the distribution of the per-position largest gap,
#                         row by row
#
# PyTorch's rows: float32 on ten X925 cores pinned (the baseline's, D100), on
# one (cpu 15), and float64 on ten (the control: the export's identity).
# Nothing is timed; the forward passes take seconds.
set -u -o pipefail
cd "$(dirname "$0")/.."
PRESK=${PRESK:-./target/release/presk}
PY=${PY:-.venv/bin/python}
MARGINS=${MARGINS:-out/margins}
OUT=${OUT:-out/pytorch-gap}
W=${W:-out/curve_train_f64/final.psk.d}
VAL=${VAL:-out/civil_val_T256.psk.d}
# The compiler's rows: the uniform types, D124's curve realization at 10⁻³
# and V6's solved on the logits, which make margins measures when EXTRA
# names them (below).
PRESK_ROWS=${PRESK_ROWS:-"f64 f32 bf16 d124-1e-3 logits-4.67e-4"}
CORES=${CORES:-5-9,15-19}
# The model's constants: D = 64's by default; the course constants with
# MARGINS=out/margins-course W=out/course_train_f64/final.psk.d (D129).
CONSTS=${CONSTS:-"--const D=64 --const H=2 --const L=2"}
for f in "$W" "$VAL" "$MARGINS/val.json"; do [ -e "$f" ] || { echo "pytorch_gap.sh: missing $f (make margins first)"; exit 1; }; done
ls "$MARGINS"/exact/*.hilo > /dev/null 2>&1 || { echo "pytorch_gap.sh: no exact logits in $MARGINS/exact (make margins first)"; exit 1; }
mkdir -p "$OUT/margins/native" "$OUT/margins/decisions"
"$PRESK" --version > "$OUT/presk-version.txt"

[ -f "$OUT/export/_05_gpt.py" ] || "$PRESK" export examples/corpus/nlp/05_gpt.psk --target pytorch --mode readable --train \
    $CONSTS -o "$OUT/export" > "$OUT/export.log" 2>&1 || { echo "export failed"; exit 1; }

# The exact logits and the compiler's rows, linked.
M=$(cd "$MARGINS" && pwd)
ln -sfn "$M/exact" "$OUT/margins/exact"
ROWS=()
for r in $PRESK_ROWS; do
    [ -d "$M/native/$r" ] || { echo "  no row $r in $MARGINS: left out"; continue; }
    ln -sfn "$M/native/$r" "$OUT/margins/native/$r"
    ln -sf "$M/decisions/$r.txt" "$OUT/margins/decisions/$r.txt"
    ROWS+=("$r")
done

# PyTorch's rows.
pt() { # name dtype threads cores
    [ -f "$OUT/margins/decisions/$1.txt" ] && return 0
    taskset -c "$4" "$PY" experiments/pytorch_gap.py logits --export "$OUT/export" --weights "$W" --data "$VAL" \
        --exact "$OUT/margins/exact" --dtype "$2" --threads "$3" --name "$1" --out "$OUT/margins" > "$OUT/$1.log" 2>&1 \
        || { echo "pytorch_gap.sh: the $1 run failed: $OUT/$1.log"; exit 1; }
    head -1 "$OUT/$1.log"; tail -1 "$OUT/$1.log"
}
pt pytorch-f32 float32 10 "$CORES"
pt pytorch-f32-1t float32 1 15
pt pytorch-f64 float64 10 "$CORES"
ROWS+=(pytorch-f32 pytorch-f32-1t pytorch-f64)

LABELS=""
for r in "${ROWS[@]}"; do
    case $r in
        pytorch-f32) l="PyTorch float32, ten threads" ;;
        pytorch-f32-1t) l="PyTorch float32, one thread" ;;
        pytorch-f64) l="PyTorch float64, ten threads" ;;
        curve-*) l="presk, solved at ${r#curve-} on the loss" ;;
        d124-*) l="presk, solved at ${r#d124-} on the loss (D124)" ;;
        logits-*) l="presk, solved at ${r#logits-} on the logits" ;;
        *) l="presk, all-$r" ;;
    esac
    LABELS="${LABELS:+$LABELS|}$l"
done
GAPS=$(for r in "${ROWS[@]}"; do v=$(cat "$M/decisions/$r.loss" 2>/dev/null); echo -n "${v:-—} "; done)
python3 experiments/margins.py report "$OUT/margins" --val "$M/val.json" --names "${ROWS[*]}" --labels "$LABELS" \
    --loss-gaps "$GAPS" --out "$OUT/margins.md" || exit 1

# The per-position largest gap, its distribution.
python3 - "$OUT/margins" "${ROWS[@]}" <<'EOF' | tee "$OUT/gaps.md"
import os, sys
d, rows = sys.argv[1], sys.argv[2:]
chunks = sorted(f[:-4] for f in os.listdir(os.path.join(d, "exact")) if f.endswith(".tsv"))
print("| row | positions | median | 90 % | 99 % | 99.9 % | largest | mean of the largest |")
print("|---|---|---|---|---|---|---|---|")
for r in rows:
    g = sorted(float(l.split("\t")[0]) for c in chunks for l in open(os.path.join(d, "native", r, c + ".tsv")))
    n = len(g)
    q = lambda p: g[min(n - 1, int(p * n))]
    print(f"| {r} | {n} | {q(0.5):.3e} | {q(0.9):.3e} | {q(0.99):.3e} | {q(0.999):.3e} | {g[-1]:.3e} | {sum(g) / n:.3e} |")
EOF
