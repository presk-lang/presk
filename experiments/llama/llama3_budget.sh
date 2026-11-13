#!/bin/bash
# Llama 3.2 1B solved whole, the budget read off a quality target, against
# the field's recipes on the exact scale (plan 0.17.1, presk-specs
# implementation/presk-plan-0.17.1.md; presk D185 onward). The checkpoint and
# its conversion are D180's (`make llama3 STAGE=convert`); WikiText-2 raw is
# the user's download:
#
#     hf download Salesforce/wikitext --repo-type dataset \
#         --include 'wikitext-2-raw-v1/*' --local-dir ~/presk-data/wikitext-2-raw
#
# and llama.cpp is built by experiments/llama/setup_llamacpp.sh.
#
# The stages, each reusing the outputs that exist (FRESH=1 redoes them):
# texts       the three samples (experiments/llama/texts.py, .venv-hf): 4
#             windows of 16 of the train split (the envelope's), 32 of 32 of
#             the test split (the certificate's, 1 024 positions), the next 128
#             of 129 (the perplexity's), each opened by the beginning-of-text
#             token; their .psk.d files
# exact       the held-out windows' logits on the reference at 128 bits, one
#             window a run, one run at a time, each in a scope of its own
#             (MemoryMax=108G, no swap, oom_score_adj 1000); per window its
#             margins and its logits as two doubles (experiments/margins.py
#             exact)
# margins     the exact margins, ε*(δ) for DELTAS at ALPHA (budget.py eps)
# gguf        llama.cpp's BF16, then Q8_0, Q4_K_M and Q4_0 (its converter in
#             .venv-hf with its own gguf-py, and llama-quantize)
# calibrate   the envelope on the calibration sample at 128 bits, alone
# solve       the seven solves of the plan's §3.3 and the eighth (ε*(1e-2),
#             `--min-gain 0`, the user's, 2026-11-06), one at a time, each in its
#             scope, under the one menu, `--sweep f64 --measure native`; each
#             realization's `presk explain`; budgets.json
# measure     every realization on the held-out windows: ours natively, one
#             window and one thread a run, JOBS side by side on the X925 cores;
#             llama.cpp's four and transformers' two by their programs; the
#             gaps as margins.py's rows; the weights' bytes (bytes.jsonl)
# perplexity  `nll` on the quality windows, B = 8 a run, under ours; the
#             field's from their logits of the contexts; llama-perplexity's own
#             figure as a check (nll/)
# time        RW9, alone on the machine: the forward pass at T = 16 under ours
#             and `presk serve` for 32 tokens under f32, the realization by
#             hand, int8, the solve at ε*(1e-2) and the eighth solve, three
#             processes each (time/); the median and the fastest (budget.py
#             time: time.md, time.json)
# report      the tables and the statements (budget.py report)
#
#   usage: experiments/llama/llama3_budget.sh [STAGE|all]
#
# CK names the checkpoint's directory, WIKI WikiText-2's, OUT the outputs',
# PRESK the binary (a copy under out/bin while a stage runs), LLAMACPP
# llama.cpp's directory. A running stage runs from a copy of this script.
set -u
cd "$(dirname "$0")/../.."
CK=${CK:-$HOME/presk-data/llama-3.2-1b}
WIKI=${WIKI:-$HOME/presk-data/wikitext-2-raw}
OUT=${OUT:-out/llama3-budget}
P=${PRESK:-target/release/presk}
W=${W:-out/llama3/llama-3.2-1b.psk.d}
LC=${LLAMACPP:-.llamacpp}
TX=$OUT/texts
STAGE=${1:-all}
MEM=${MEM:-108G}
DELTAS=${DELTAS:-1e-1 3e-2 1e-2}
ALPHA=${ALPHA:-0.05}
JOBS=${JOBS:-8}
CORES=(5 6 7 8 9 15 16 17 18 19)
PY=.venv-hf/bin/python
mkdir -p "$OUT"
fresh() { [ "${FRESH:-0}" = 1 ] || [ ! -e "$1" ]; }
want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
say() { echo "== $(date -Is) $*"; }
[ -d "$CK" ] || { echo "no checkpoint at $CK: download meta-llama/Llama-3.2-1B there (the user's step), or set CK"; exit 2; }
[ -f "$W" ] || { echo "no $W: run make llama3 STAGE=convert first"; exit 2; }
VOCAB=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['vocab_size'])" "$CK/config.json") || exit 1

# A command in a memory scope of its own, the first the kernel kills.
scoped() { systemd-run --user --scope -q -p MemoryMax="$MEM" -p MemorySwapMax=0 choom -n 1000 -- "$@"; }

# Our realizations: D180's uniform six and f64, one by hand, then the solved ones.
UNIFORM="f64 f32 bf16 int8-absmax int8-pow2 int4-absmax int4-pow2"
# Written by hand (the user, 2026-11-06): every parameter bf16 at rest, every
# stored value f32 (experiments/llama/bf16-params-f32.json).
HAND="bf16-params-f32"
# The plan's seven, then the eighth (the user, 2026-11-06): ε*(1e-2) with
# rule 2 off, the activations offered, run last. SOLVES chooses among them.
SOLVES=${SOLVES:-"eps-1e-1 eps-3e-2 eps-1e-2 rel-1e-2 rel-1e-3 eps-1e-2-defaults eps-1e-2-margin1 eps-1e-2-mingain0"}
FIELD="llamacpp-BF16 llamacpp-Q8_0 llamacpp-Q4_K_M llamacpp-Q4_0 hf-float32 hf-bfloat16"
spec() { # name -> --realize's argument
  case $1 in
    f64 | f32 | bf16) echo "$1" ;;
    int*) echo "@out/llama3/realize/$1.json" ;;
    bf16-params-f32) echo "@experiments/llama/$1.json" ;;
    *) echo "@$OUT/solve/$1.json" ;;
  esac
}

# --- texts -------------------------------------------------------------------
if want texts && fresh "$TX/texts.json"; then
  say texts
  [ -d "$WIKI/wikitext-2-raw-v1" ] || { echo "no WikiText-2 at $WIKI: download it there (the user's step), or set WIKI"; exit 2; }
  $PY experiments/llama/texts.py "$WIKI" "$CK" "$TX" || exit 1
fi
for s in calib:16 heldout:32 quality:129; do
  n=${s%:*}; len=${s#*:}
  # Packed again under FRESH=1 only by the texts stage: another stage's
  # FRESH=1 must not rewrite a file that a run beside it may be reading.
  if [ -f "$TX/$n.txt" ] && { [ ! -e "$TX/$n.psk.d" ] || { want texts && fresh "$TX/$n.psk.d"; }; }; then
    $P data pack --type "idx[$VOCAB][seq: $len]" -o "$TX/$n.psk.d" "$TX/$n.txt" > /dev/null < /dev/null || exit 1
  fi
done
# The quality windows' contexts, their first 128 ids, for the field's tools.
[ -f "$TX/quality.txt" ] && { [ -f "$TX/quality_ctx.txt" ] || cut -d' ' -f1-128 "$TX/quality.txt" > "$TX/quality_ctx.txt"; }

# --- exact -------------------------------------------------------------------
if want exact; then
  say exact
  D=$OUT/exact
  mkdir -p "$D"
  N=$(wc -l < "$TX/heldout.txt")
  for ((k = 0; k < N; k++)); do
    p=$D/$(printf %04d "$k")
    { [ -f "$p.tsv" ] && [ -f "$p.hilo" ] && [ "${FRESH:-0}" != 1 ]; } && continue
    echo "-- exact, window $k $(date)"
    if fresh "$p.json"; then
      scoped /usr/bin/time -v $P run examples/coverage/llama3.psk --entry Llama3.apply --dim T=32 --weights "$W" \
        --data "$TX/heldout.psk.d" --from "$k" --samples 1 --json --result-only --digits 30 \
        --backend interp --bits 128 > "$p.json.tmp" 2> "$p.err" < /dev/null && mv "$p.json.tmp" "$p.json"
      grep -E "Elapsed|Maximum resident|Exit status" "$p.err"
      [ -f "$p.json" ] || { echo "the exact run of window $k failed: $p.err"; exit 1; }
    fi
    python3 experiments/margins.py exact "$p.json" "$p" || exit 1
  done
fi

# --- margins -----------------------------------------------------------------
if want margins; then
  n=$(ls "$OUT"/exact/*.tsv 2>/dev/null | wc -l)
  [ "$n" = "$(wc -l < "$TX/heldout.txt")" ] || { echo "margins: $n exact windows of $(wc -l < "$TX/heldout.txt")"; [ "$STAGE" = margins ] && exit 1; }
  if [ "$n" = "$(wc -l < "$TX/heldout.txt")" ] && fresh "$OUT/margins.json"; then
    say margins
    $PY experiments/llama/budget.py eps "$OUT" --deltas "$DELTAS" --alpha "$ALPHA" || exit 1
  fi
fi

# --- gguf --------------------------------------------------------------------
G=$OUT/gguf
if want gguf; then
  mkdir -p "$G"
  [ -x "$LC/build/bin/llama-quantize" ] || { echo "no llama.cpp in $LC: run experiments/llama/setup_llamacpp.sh"; exit 2; }
  if fresh "$G/llama-3.2-1b-BF16.gguf"; then
    say gguf BF16
    MEM=8G scoped env PYTHONPATH="$LC/src/gguf-py:experiments/llama/stub" $PY "$LC/src/convert_hf_to_gguf.py" "$CK" \
      --outtype bf16 --outfile "$G/llama-3.2-1b-BF16.gguf" > "$G/convert.log" 2>&1 || exit 1
  fi
  for q in Q8_0 Q4_K_M Q4_0; do
    if fresh "$G/llama-3.2-1b-$q.gguf"; then
      say gguf $q
      MEM=8G scoped "$LC/build/bin/llama-quantize" "$G/llama-3.2-1b-BF16.gguf" "$G/llama-3.2-1b-$q.gguf" $q 4 > "$G/quantize-$q.log" 2>&1 || exit 1
    fi
  done
fi

# --- calibrate ---------------------------------------------------------------
ENV=$OUT/envelope/calib_env.json
if want calibrate && fresh "$ENV"; then
  say calibrate
  mkdir -p "$OUT/envelope"
  scoped /usr/bin/time -v -o "$OUT/envelope/calibrate.time" $P calibrate --precision examples/coverage/llama3.psk --entry Llama3.apply \
    --dim T=16 --data "$TX/calib.psk.d" --samples 4 --weights "$W" --bits 128 --out "$ENV.tmp" \
    > "$OUT/envelope/calibrate.txt" 2>&1 < /dev/null && mv "$ENV.tmp" "$ENV"
  grep -E "Elapsed|Maximum resident|Exit status" "$OUT/envelope/calibrate.time"
  [ -f "$ENV" ] || { echo "the calibration failed: $OUT/envelope/calibrate.txt"; exit 1; }
fi

# --- solve -------------------------------------------------------------------
eps_of() { python3 -c "import json, sys; print(repr(json.load(open(sys.argv[1]))['eps'][str(float(sys.argv[2]))]['eps_cp']))" "$OUT/margins.json" "$1"; }
if want solve; then
  [ -f "$ENV" ] && [ -f "$OUT/margins.json" ] || { echo "solve: run the margins and calibrate stages first"; exit 1; }
  S=$OUT/solve
  mkdir -p "$S"
  MENU=f64,f32,bf16,int8,int4
  for name in $SOLVES; do
    case $name in
      eps-1e-1) b="--budget-abs $(eps_of 1e-1)"; o="--narrow measured" ;;
      eps-3e-2) b="--budget-abs $(eps_of 3e-2)"; o="--narrow measured" ;;
      eps-1e-2) b="--budget-abs $(eps_of 1e-2)"; o="--narrow measured" ;;
      rel-1e-2) b="--budget 1e-2"; o="--narrow measured" ;;
      rel-1e-3) b="--budget 1e-3"; o="--narrow measured" ;;
      eps-1e-2-defaults) b="--budget-abs $(eps_of 1e-2)"; o="" ;;
      eps-1e-2-margin1) b="--budget-abs $(eps_of 1e-2)"; o="--narrow measured --margin 1" ;;
      eps-1e-2-mingain0) b="--budget-abs $(eps_of 1e-2)"; o="--narrow measured --min-gain 0" ;;
    esac
    if fresh "$S/$name.json"; then
      say "solve $name: $b $o"
      # shellcheck disable=SC2086 # the options are words
      scoped /usr/bin/time -v -o "$S/$name.time" $P solve examples/coverage/llama3.psk --precision "$ENV" $b --menu "$MENU" $o \
        --sweep f64 --measure native --weights "$W" --sites 40 --out "$S/$name.json.tmp" > "$S/$name.txt" 2>&1 < /dev/null \
        && mv "$S/$name.json.tmp" "$S/$name.json"
      grep -E "Elapsed|Maximum resident|Exit status" "$S/$name.time"
      grep -E "^  (measured|parameters|narrowing|cost)" "$S/$name.txt"
      [ -f "$S/$name.json" ] || { echo "the solve $name failed: $S/$name.txt"; exit 1; }
    fi
    if fresh "$S/explain-$name/explain.md"; then
      MEM=40G scoped $P explain examples/coverage/llama3.psk --entry Llama3.apply --dim B=1 --dim T=16 --weights "$W" \
        --realize "@$S/$name.json" --envelope "$ENV" --no-exact --target native --out "$S/explain-$name" \
        > "$S/explain-$name.log" 2>&1 < /dev/null || echo "presk explain $name failed: $S/explain-$name.log"
    fi
  done
  # Each solve's budget, absolute, for the report.
  python3 - "$S" "$OUT/budgets.json" "$OUT/margins.json" <<'EOF'
import json, re, sys
s, out, mj = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
b = {}
for name in "eps-1e-1 eps-3e-2 eps-1e-2 rel-1e-2 rel-1e-3 eps-1e-2-defaults eps-1e-2-margin1 eps-1e-2-mingain0".split():
    try:
        head = open(f"{s}/{name}.txt").readline()
    except OSError:
        continue
    # ε*(δ) as passed, from the margins; a relative budget's absolute value
    # as the solve prints it (three digits).
    of = {"eps-1e-1": 0.1, "eps-3e-2": 0.03, "eps-1e-2": 0.01, "eps-1e-2-defaults": 0.01, "eps-1e-2-margin1": 0.01,
          "eps-1e-2-mingain0": 0.01}.get(name)
    if of is not None:
        eps = mj["eps"][str(of)]["eps_cp"]
    else:
        m = re.search(r"\(([0-9.e+-]+) absolute", head)
        eps = float(m.group(1)) if m else None
    delta = {"eps-1e-1": 0.1, "eps-3e-2": 0.03, "eps-1e-2": 0.01}.get(name)
    b["solve-" + name] = {"eps": eps, "delta": delta, "statement": delta is not None, "first_line": head.strip()}
json.dump(b, open(out, "w"), indent=1)
EOF
fi

# --- measure -----------------------------------------------------------------
# A pool of JOBS runs, each pinned to one X925 core: the job is a shell
# function, so its subshell pins itself and its children inherit the core.
running=0; slot=0
spawn() { local c=${CORES[$((slot % ${#CORES[@]}))]}; slot=$((slot + 1)); ( taskset -pc "$c" "$BASHPID" > /dev/null && "$@" ) & running=$((running + 1)); if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi; }
drain() { wait; running=0; }
native_window() { # name spec k
  local d=$OUT/native/$1 k; k=$(printf %04d "$3")
  [ -f "$d/$k.tsv" ] && [ "${FRESH:-0}" != 1 ] && return 0
  $P run examples/coverage/llama3.psk --entry Llama3.apply --dim T=32 --weights "$W" --data "$TX/heldout.psk.d" --from "$3" --samples 1 \
    --backend native --threads 1 --realize "$2" --json --result-only --digits 17 > "$d/$k.json" 2> "$d/$k.err" < /dev/null \
    && $PY experiments/llama/budget.py native "$d/$k.json" "$OUT/exact/$k.hilo" "$d/$k.tsv" && rm -f "$d/$k.json" "$d/$k.err" \
    || echo "the $1 run of window $3 failed: $d/$k.err"
}
if want measure; then
  N=$(wc -l < "$TX/heldout.txt")
  names="$UNIFORM $HAND"
  for s in $SOLVES; do [ -f "$OUT/solve/$s.json" ] && names="$names solve-$s"; done
  for name in $names; do
    sp=$(spec "${name#solve-}")
    mkdir -p "$OUT/native/$name"
    say "measure $name ($sp)"
    for ((k = 0; k < N; k++)); do spawn native_window "$name" "$sp" "$k"; done
    drain
  done
  F=$OUT/field
  mkdir -p "$F"
  for q in BF16 Q8_0 Q4_K_M Q4_0; do
    name=llamacpp-$q
    if fresh "$F/$name.heldout.f32"; then
      say "measure $name"
      MEM=8G scoped "$LC/bin/llamacpp_logits" "$G/llama-3.2-1b-$q.gguf" "$TX/heldout.txt" "$F/$name.heldout.f32" 8 2> "$F/$name.heldout.log" || exit 1
    fi
    [ -f "$OUT/native/$name/0031.tsv" ] && [ "${FRESH:-0}" != 1 ] || $PY experiments/llama/budget.py field "$F/$name.heldout.f32" "$OUT/exact" "$OUT/native/$name" || exit 1
  done
  for d in float32 bfloat16; do
    name=hf-$d
    if fresh "$F/$name.heldout.f32"; then
      say "measure $name"
      MEM=16G scoped $PY experiments/llama/rw2.py "$CK" - "$TX/heldout.txt" --dtype $d --logits "$F/$name.heldout.f32" --threads 8 > "$F/$name.heldout.log" 2>&1 || exit 1
    fi
    [ -f "$OUT/native/$name/0031.tsv" ] && [ "${FRESH:-0}" != 1 ] || $PY experiments/llama/budget.py field "$F/$name.heldout.f32" "$OUT/exact" "$OUT/native/$name" || exit 1
  done
  # The weights' bytes at rest.
  : > "$OUT/bytes.jsonl"
  for name in $names; do $PY experiments/llama/budget.py bytes "$name" "$CK/config.json" "$(spec "${name#solve-}")" >> "$OUT/bytes.jsonl"; done
  for q in BF16 Q8_0 Q4_K_M Q4_0; do $PY experiments/llama/budget.py bytes "llamacpp-$q" "$CK/config.json" "$G/llama-3.2-1b-$q.gguf" --gguf-py "$LC/src/gguf-py" >> "$OUT/bytes.jsonl"; done
  for d in float32 bfloat16; do $PY experiments/llama/budget.py bytes "hf-$d" "$CK/config.json" "hf:$d" >> "$OUT/bytes.jsonl"; done
fi

# --- perplexity --------------------------------------------------------------
if want perplexity; then
  NQ=$(wc -l < "$TX/quality.txt")
  B=8
  names="$UNIFORM $HAND"
  for s in $SOLVES; do [ -f "$OUT/solve/$s.json" ] && names="$names solve-$s"; done
  mkdir -p "$OUT/nll"
  for name in $names; do
    sp=$(spec "${name#solve-}")
    d=$OUT/nll/$name
    mkdir -p "$d"
    fresh "$OUT/nll/$name.json" || continue
    say "perplexity $name ($sp)"
    for ((f = 0; f < NQ; f += B)); do
      o=$d/$(printf %04d "$f").json
      [ -s "$o" ] && [ "${FRESH:-0}" != 1 ] && continue
      taskset -c "$(IFS=,; echo "${CORES[*]:0:8}")" $P run examples/coverage/llama3.psk --entry nll --dim T=128 --weights "$W" --data "$TX/quality.psk.d" \
        --from "$f" --samples "$B" --backend native --threads 8 --realize "$sp" --json --result-only --digits 17 > "$o.tmp" 2> "$o.err" < /dev/null \
        && mv "$o.tmp" "$o" && rm -f "$o.err" || { echo "the nll run $name from $f failed: $o.err"; exit 1; }
    done
    $PY experiments/llama/budget.py nll-presk "$OUT/nll/$name.json" $(ls "$d"/*.json | sort) || exit 1
  done
  F=$OUT/field
  for q in BF16 Q8_0 Q4_K_M Q4_0; do
    name=llamacpp-$q
    if fresh "$OUT/nll/$name.json"; then
      say "perplexity $name"
      fresh "$F/$name.quality.f32" && { MEM=8G scoped "$LC/bin/llamacpp_logits" "$G/llama-3.2-1b-$q.gguf" "$TX/quality_ctx.txt" "$F/$name.quality.f32" 8 2> "$F/$name.quality.log" || exit 1; }
      $PY experiments/llama/budget.py nll-field "$F/$name.quality.f32" "$TX/quality.txt" "$OUT/nll/$name.json" || exit 1
      rm -f "$F/$name.quality.f32"
    fi
  done
  # The check of our reading: llama-perplexity's own figure on its own
  # windows of the whole test text (n_ctx 128, one window a batch, as
  # llamacpp_logits decodes), against the same figure from llamacpp_logits'
  # logits of those windows; and its tokenizer against ours on the text.
  C=$OUT/check
  mkdir -p "$C"
  if fresh "$C/check.md"; then
    say "perplexity: llama-perplexity's check"
    $PY - "$WIKI" "$CK" "$C" <<'PYEOF' || exit 1
import os, sys, pyarrow.parquet as pq
from tokenizers import Tokenizer
wiki, ck, c = sys.argv[1:]
text = "\n\n".join(pq.read_table(os.path.join(wiki, "wikitext-2-raw-v1", "test-00000-of-00001.parquet")).column("text").to_pylist())
open(os.path.join(c, "wiki.test.txt"), "w").write(text)
ids = [128000] + Tokenizer.from_file(os.path.join(ck, "tokenizer.json")).encode(text, add_special_tokens=False).ids
open(os.path.join(c, "ours.ids"), "w").write(" ".join(map(str, ids)))
PYEOF
    "$LC/build/bin/llama-tokenize" -m "$G/llama-3.2-1b-Q8_0.gguf" -f "$C/wiki.test.txt" --ids --log-disable > "$C/llamacpp.ids" 2> "$C/tokenize.err" || exit 1
    $PY experiments/llama/budget.py ppl-check "$C/llamacpp.ids" "$C/chunks.txt" --hf-ids "$C/ours.ids" > "$C/check.tmp" || exit 1
    for q in BF16 Q8_0 Q4_K_M Q4_0; do
      for b in 128 2048; do
        MEM=8G scoped "$LC/build/bin/llama-perplexity" -m "$G/llama-3.2-1b-$q.gguf" -f "$C/wiki.test.txt" -c 128 -b $b --chunks 32 -t 4 > "$C/ppl-b$b-$q.log" 2>&1
        echo "$q, llama-perplexity -b $b: $(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$C/ppl-b$b-$q.log")" >> "$C/check.tmp"
      done
      MEM=8G scoped "$LC/bin/llamacpp_logits" "$G/llama-3.2-1b-$q.gguf" "$C/chunks.txt" "$C/chunks-$q.f32" 4 2>/dev/null || exit 1
      echo "$q, $($PY experiments/llama/budget.py ppl-check "$C/llamacpp.ids" "$C/chunks.txt" --logits "$C/chunks-$q.f32")" >> "$C/check.tmp"
      rm -f "$C/chunks-$q.f32"
    done
    mv "$C/check.tmp" "$C/check.md"
    cat "$C/check.md"
  fi
  for d in float32 bfloat16; do
    name=hf-$d
    if fresh "$OUT/nll/$name.json"; then
      say "perplexity $name"
      fresh "$F/$name.quality.f32" && { MEM=24G scoped $PY experiments/llama/rw2.py "$CK" - "$TX/quality_ctx.txt" --dtype $d --logits "$F/$name.quality.f32" --threads 8 > "$F/$name.quality.log" 2>&1 || exit 1; }
      $PY experiments/llama/budget.py nll-field "$F/$name.quality.f32" "$TX/quality.txt" "$OUT/nll/$name.json" || exit 1
      rm -f "$F/$name.quality.f32"
    fi
  done
fi

# --- time --------------------------------------------------------------------
# RW9, alone on the machine: each
# measurement a process of its own, three of each, one thread at batch 1
# (nothing is divided there, D180).
if want time; then
  T=$OUT/time
  mkdir -p "$T"
  names="$UNIFORM $HAND"
  for s in $SOLVES; do [ -f "$OUT/solve/$s.json" ] && names="$names solve-$s"; done
  head -1 out/llama3/prompts.txt > "$T/prompt.txt"
  for round in 1 2 3; do
    for name in $names; do
      o=$T/forward-$name-$round.txt
      fresh "$o" || continue
      say "time forward $name, round $round"
      taskset -c 5 $P bench examples/coverage/llama3.psk --entry Llama3.apply --dim B=1 --dim T=16 --weights "$W" --realize "$(spec "${name#solve-}")" \
        --native-only --threads 1 --warmup 1 --repeat 5 > "$o" 2>&1 < /dev/null || echo "bench $name failed: $o"
    done
    # The eighth solve's serve is the user's addition (2026-11-08, D190): in
    # step 5 its three processes ran after the others' three rounds.
    for name in f32 $HAND int8-absmax solve-eps-1e-2 solve-eps-1e-2-mingain0; do
      [ "${name#solve-}" != "$name" ] && [ ! -f "$OUT/solve/${name#solve-}.json" ] && continue
      o=$T/serve-$name-$round.txt
      fresh "$o" || continue
      say "time serve $name, round $round"
      { /usr/bin/time -f "%e s %M kB" taskset -c 5 $P serve examples/coverage/llama3.psk --weights "$W" --prompt "$(cat "$T/prompt.txt")" --tokens 32 \
          --temperature 0 --backend native --realize "$(spec "${name#solve-}")" < /dev/null; } > "$o" 2>&1 || echo "serve $name failed: $o"
    done
  done
  $PY experiments/llama/budget.py time "$OUT" --names "$names" || exit 1
fi

# --- report ------------------------------------------------------------------
if want report && [ -f "$OUT/margins.json" ]; then
  say report
  names="$UNIFORM $HAND"
  for s in $SOLVES; do names="$names solve-$s"; done
  $PY experiments/llama/budget.py report "$OUT" --names "$names $FIELD" || exit 1
fi
say done
