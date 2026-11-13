#!/bin/bash
# Llama 3.2 1B on its published weights (plan 0.17.0 step 7, presk D180):
# examples/coverage/llama3.psk on meta-llama/Llama-3.2-1B, the base model, which the
# user downloads (its licence accepted, its token theirs):
#
#     hf download meta-llama/Llama-3.2-1B --local-dir ~/presk-data/llama-3.2-1b
#
# The stages, each reusing the outputs that exist (FRESH=1 redoes them):
# convert  the checkpoint as a .psk.d of bf16 leaves (version 2), its leaves
#          in presk's order (experiments/llama/hf_to_pskd.py)
# prompts  four fixed texts, 16 tokens each with the beginning-of-text
#          token, by the checkpoint's tokenizer (experiments/llama/tokens.py)
# rw2      RW2: presk run natively in f64 against transformers'
#          LlamaForCausalLM in float64 on the same ids, the largest gap at
#          most 1e-8 of the largest logit (experiments/llama/rw2.py,
#          .venv-hf)
# rw4      RW4: 32 greedy tokens from each prompt by presk serve --backend
#          native and by the C export's generate.c (exported at T = 48, the
#          context padded), in f32 and in int8, the same tokens
# rw3      RW3, reported: the logits' gap to the exact reference at 128 bits
#          (each exact run alone, under a 108 GB scope) under f32, bf16, and
#          int8 and int4 weights, absmax and pow2, over the contraction axis
#          (experiments/llama/rw3.py)
#
#   usage: experiments/llama/llama3.sh [convert|prompts|rw2|rw4|rw3|all]
#
# CK names the checkpoint's directory, OUT the outputs', CONSTS presk's
# --const overrides of the program (a small checkpoint of the same family,
# to try the stages: then give OUT/prompts.txt, ids one prompt a line).
set -u
cd "$(dirname "$0")/../.."
CK=${CK:-$HOME/presk-data/llama-3.2-1b}
OUT=${OUT:-out/llama3}
P=${PRESK:-target/release/presk}
W=$OUT/llama-3.2-1b.psk.d
PROMPTS=$OUT/prompts.txt
STAGE=${1:-all}
CONSTS=${CONSTS:-}
mkdir -p "$OUT"
fresh() { [ "${FRESH:-0}" = 1 ] || [ ! -e "$1" ]; }
want() { [ "$STAGE" = all ] || [ "$STAGE" = "$1" ]; }
[ -d "$CK" ] || { echo "no checkpoint at $CK: download meta-llama/Llama-3.2-1B there (the user's step), or set CK"; exit 2; }
VOCAB=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['vocab_size'])" "$CK/config.json") || exit 1
CF=""
for c in $CONSTS; do CF="$CF --const $c"; done

if want convert && fresh "$W"; then
  echo "== convert $(date)"
  .venv/bin/python experiments/llama/hf_to_pskd.py "$CK" -o "$W" --presk "$P" $CF || exit 1
  $P data info "$W" | head -4
fi

if want prompts && fresh "$PROMPTS"; then
  echo "== prompts $(date)"
  : > "$PROMPTS"
  for text in \
    "The capital of France is Paris, and the capital of Italy is Rome. The capital of Spain is" \
    "In mathematics, a prime number is a natural number greater than one that has no positive divisors other than" \
    "def fibonacci(n): if n < 2: return n; return fibonacci(n - 1) + fibonacci(n - 2) # the" \
    "Le compilateur traduit un programme écrit dans un langage de haut niveau en un programme"; do
    .venv/bin/python experiments/llama/tokens.py encode "$CK" "$text" --length 16 >> "$PROMPTS" || exit 1
  done
  cat "$PROMPTS"
fi

# The realizations of RW3 and RW4: every matrix of the blocks in int8 or
# int4, its scales along its contraction axis (-2: `model`, `ff` or
# `headhd`); the token table, read by `embed` and by the tied head, and the
# norms' gains stay in f32.
mkdir -p "$OUT/realize"
for bits in 8 4; do
  for scheme in absmax pow2; do
    f=$OUT/realize/int${bits}-${scheme}.json
    q=""
    for m in attn.wq attn.wk attn.wv attn.wo ff.gate ff.up ff.down; do
      q="$q${q:+, }\"blocks.$m\": {\"bits\": $bits, \"over\": -2, \"scheme\": \"$scheme\"}"
    done
    echo "{\"default\": \"f32\", \"order\": {\"kind\": \"sequential\"}, \"quant\": {$q}}" > "$f"
  done
done

if want rw2 && fresh "$OUT/rw2/rw2.json"; then
  echo "== rw2 $(date)"
  .venv-hf/bin/python experiments/llama/rw2.py "$CK" "$W" "$PROMPTS" --presk "$P" --out "$OUT/rw2" $CF
  echo "rw2 exit $?"
fi

if want rw4 && fresh "$OUT/rw4.md"; then
  echo "== rw4 $(date)"
  {
    echo "| realization | prompt | presk serve --backend native | generate.c | same |"
    echo "|---|---|---|---|---|"
  } > "$OUT/rw4.tmp"
  for rz in f32 int8-absmax; do
    case $rz in f32) R=f32 ;; *) R=@$OUT/realize/$rz.json ;; esac
    C=$OUT/serve_c_$rz
    if fresh "$C/generate"; then
      $P export examples/coverage/llama3.psk --target c --serve $CF --dim T=48 --weights "$W" --realize "$R" --no-exact -o "$C" < /dev/null || exit 1
      make -s -C "$C" generate || exit 1
    fi
    k=0
    while read -r ids; do
      a=$($P serve examples/coverage/llama3.psk $CF --weights "$W" --prompt "$ids" --tokens 32 --temperature 0 --backend native --realize "$R" 2>/dev/null < /dev/null)
      b=$("$C/generate" --weights "$W" --prompt "$ids" --tokens 32 --temperature 0 2>/dev/null < /dev/null)
      same=no; [ -n "$a" ] && [ "$a" = "$b" ] && same=yes
      echo "| $rz | $k | $a | $b | $same |" >> "$OUT/rw4.tmp"
      k=$((k + 1))
    done < "$PROMPTS"
  done
  mv "$OUT/rw4.tmp" "$OUT/rw4.md"
  cat "$OUT/rw4.md"
fi

if want rw3; then
  echo "== rw3 $(date)"
  D=$OUT/rw3
  mkdir -p "$D"
  k=0
  while read -r ids; do
    n=$(echo "$ids" | wc -w)
    echo "$ids" > "$D/ids$k.txt"
    $P data pack --type "idx[$VOCAB][batch: 1, seq: $n]" -o "$D/ids$k.psk.d" "$D/ids$k.txt" > /dev/null < /dev/null || exit 1
    run="$P run examples/coverage/llama3.psk --entry Llama3.apply $CF --dim B=1 --dim T=$n --weights $W --data $D/ids$k.psk.d --json --result-only --digits 17"
    if fresh "$D/exact.$k.json"; then
      echo "-- exact, prompt $k $(date)"
      /usr/bin/time -v systemd-run --user --scope -q -p MemoryMax=108G -p MemorySwapMax=0 \
        $run --backend interp --bits 128 > "$D/exact.$k.json.tmp" 2> "$D/exact.$k.err" < /dev/null && mv "$D/exact.$k.json.tmp" "$D/exact.$k.json"
      grep -E "Elapsed|Maximum resident" "$D/exact.$k.err"
    fi
    for rz in f32 bf16 int8-absmax int8-pow2 int4-absmax int4-pow2; do
      case $rz in f32 | bf16) R=$rz ;; *) R=@$OUT/realize/$rz.json ;; esac
      fresh "$D/$rz.$k.json" && $run --backend native --realize "$R" > "$D/$rz.$k.json" < /dev/null
    done
    k=$((k + 1))
  done < "$PROMPTS"
  .venv/bin/python experiments/llama/rw3.py "$D"
fi
echo "== done $(date)"
