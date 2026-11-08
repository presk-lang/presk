#!/bin/bash
# llama.cpp, built from a release into DIR (default .llamacpp, not in git), for
# plan 0.17.1's field points (presk D185): the Llama 3.2 1B checkpoint
# converted by its convert_hf_to_gguf.py to BF16 and quantized by its
# llama-quantize to Q8_0, Q4_K_M and Q4_0, and the logits of given ids read
# through libllama by experiments/llama/llamacpp_logits.c. MIT licence.
#   - the release tag below, cloned at depth 1 from GitHub and checked against
#     its commit (the user approved the fetch, 2026-11-05: the latest release
#     that day);
#   - a CPU build by the system's cmake and GCC, at the build's defaults
#     (GGML_NATIVE on, OpenMP), no network library (LLAMA_CURL off);
#   - DIR/bin/llamacpp_logits, the logits program, linked to DIR/build's
#     libllama with an rpath.
# Network access is needed once; a build already made is kept. A few minutes
# on falicon's cores; run it in a scope of its own beside heavy jobs.
# The converter runs in presk's .venv-hf (torch, transformers, numpy,
# safetensors) with the release's own gguf-py on PYTHONPATH:
#   experiments/llama/llama3_budget.sh gguf
set -eu
TAG=b12132
COMMIT=2ada83079afc79295b3c07bae8a74457c61cba92
DIR=${1:-.llamacpp}
JOBS=${JOBS:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$DIR"
DIR=$(cd "$DIR" && pwd)
if [ ! -d "$DIR/src/.git" ]; then
    git clone -q --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$DIR/src"
fi
got=$(git -C "$DIR/src" rev-parse HEAD)
[ "$got" = "$COMMIT" ] || { echo "llama.cpp at $got, not $TAG's $COMMIT"; exit 1; }
if [ ! -x "$DIR/build/bin/llama-quantize" ] || [ ! -x "$DIR/build/bin/llama-perplexity" ] || [ ! -x "$DIR/build/bin/llama-tokenize" ]; then
    cmake -S "$DIR/src" -B "$DIR/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF > "$DIR/configure.log" 2>&1
    cmake --build "$DIR/build" -j"$JOBS" --target llama-quantize llama-perplexity llama-tokenize llama > "$DIR/build.log" 2>&1
fi
mkdir -p "$DIR/bin"
gcc -O2 -Wall -o "$DIR/bin/llamacpp_logits" "$HERE/llamacpp_logits.c" \
    -I"$DIR/src/include" -I"$DIR/src/ggml/include" \
    -L"$DIR/build/bin" -Wl,-rpath,"$DIR/build/bin" -lllama -lggml -lggml-base
echo "llama.cpp $TAG ($COMMIT) in $DIR"
