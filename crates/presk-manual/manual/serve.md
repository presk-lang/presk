# Serving: `inference serve`, `presk serve`, `serve.py`

The `inference serve` block of a file names a model and its weights; `presk
serve` generates tokens with them, on any backend, through the file's own
*generation function*; `presk export --serve` writes the same loop in
PyTorch. This is the minimal form of the specification's `serve` (plan 0.3.0
§3.6, question 37): one prompt, no sessions, no batching, no cache.

## The block, resolved

<!-- psk: examples/spec/minigpt/serve_minigpt.psk#inference serve, same tokens -->
```text
inference serve {
  model   = m.MiniGPT
  weights = "ckpt/final"
  precision = budget(abs_error = 1e-2) | quantize(int8, calibration = "data/calib.psk.d")
  target    = gpu(sm_90) | cpu(avx512)
  batching  = dynamic(max_batch = 64, timeout = 5ms)
  cache     = kv_cache(axis seq)
}
```

`presk check` resolves the block as it resolves a `training run`:

- `model` is a plain module; `weights` is a path relative to the file, with
  `.psk.d` appended (`ckpt/final.psk.d`), a one-sample file of type `params
  M` as `presk train` writes it. It need not exist at check time.
- The **generation function** is the top-level function the field
  `generate = name` names, or, without the field, the first function of the
  file, in declaration order, whose signature is `(θ : params M, [key :
  key,] ctx : idx[V][batch: B, seq: T], … with defaults) ->
  idx[V][batch: B]` (the field since 0.10.0 step 8, the reference's decision
  W4). A name that is not a top-level function of the file, or whose
  signature does not fit, is an error (`E2010`). It is `next_token` in the
  example, which
  applies the model to the context, keeps the last position and samples from
  its key. A key, when the function takes one, comes second, and is given a
  key of its own at every token (0.10.0 step 7b; the reference's definition
  "Generation function"); a function without one is served as before, and
  cannot sample, since `sample` takes a key. A file with
  the block but no such function gets `W0003`; `presk serve --entry FN` then
  names one.
- `precision`, `target`, `batching`, `cache` pass the schema and are listed
  as *ignored* in `--summary`: they have no effect in this version.

```text
-- inference --
inference serve: model MiniGPT
    weights : ckpt/final (examples/spec/minigpt/ckpt/final.psk.d, not found)
    generate: next_token(θ, key, ctx, temperature)
    ignored : precision, target, batching, cache (checked, without effect in this version)
```

## `presk serve`

```sh
presk serve FILE.psk [--const NAME=N]... [--weights FILE.psk.d] --prompt "t₁ t₂ …" [--tokens N]
     [--temperature x] [--entry FN] [--dim NAME=N]... [--seed S] [--bits 256]
     [--backend interp|kernel|native] [--realize f64|f32|bf16|@R.json] [--json]
     [--opt [--profile FILE.json]]
```

A realization file whose `quant` map names parameters serves them as
integers with per-channel scales, computed from the loaded weights (0.6.0
step 5, `precision` chapter, "Integer weights"): the reference interpreter
runs on their dequantised values, and the native kernels load the integers
themselves, bit-identical to running on those values (D45).

The loop, in `presk-serve`:

```text
θ   = load(weights, shaped like M.init(key))
ctx = prompt
k   = key(--seed)
repeat --tokens times, i = 1, 2, …:
  next = generate(θ, child(k, i), ctx[-T:], temperature)
  ctx  = ctx ++ [next]
```

Everything with a semantics is the program's: the model, the generation
function, the initializer that gives the weights their shapes. The crate only
sequences calls through the `Backend` seam, so the same command runs on the
reference interpreter (`--bits`), on the kernel interpreter or natively
(`native` chapter). Two forms of the step:

- **sampling** (`--temperature x > 0`, default: the function's own default,
  `0.8` in the example): the function is called as written, token `i` (from
  1) given the child `i` of the key `--seed` names, as a training run gives
  step `t` the child `t` of its own (the reference's definition
  "Generation"). Its `sample` draws from that key, so every backend draws
  alike (it drew from a stream of each backend's until 0.10.0 step 7b); the
  kernel backends draw with the `libm` crate's `exp`, the one their kernels
  call, so their tokens do not depend on the machine's C library (since
  0.9.0 step 6), and they sum the probabilities in `f64` where the
  interpreter sums them at its precision, so a token can differ only where a
  draw falls within that rounding of a cumulative sum;
- **greedy** (`--temperature 0`): a copy of the function in which every
  `sample(key, x)` is an `argmax(x)` is called instead (`next_token.greedy`,
  appended to the Core program; the temperature is passed as `1`, the key
  is read by nothing). Every backend and realization agrees on this path, up
  to exact ties in the logits.

**The context length.** In the example the positional embedding is `param
pos : ℝ[T, D]` with the file's `dim T <= 1024`: the parameters fix `T`, and a
checkpoint trained on windows of 8 tokens only applies to contexts of 8. When
a leaf of the model's parameter tree has the context's `seq` dimension in
its type, `presk serve` takes `T` from `--dim T=N` (default: the prompt's
length), loads the weights at that shape and passes the model the **last
`T` tokens** of the context at every step: a sliding window, reported as
`context window 8`. A prompt shorter than `T` is refused (`E7001`), a longer
one slides. When the parameters do not depend on `T`, as a model with RoPE's
(`examples/coverage/llama3.psk`), the whole context is passed and grows, reported as
`whole context`, and the file's `dim T <= …` bound is checked at every step.
Until 0.17.0 step 7 (D180) the test read the initializer's dimensions,
which are every dimension of its file, so every model was served over a
window, the prompt's length by default. Spec question 43
records the observation; `ℝ[Tmax, D]` with a slice (question 9) would let the
same model read any context up to `Tmax`.

Output: the generated tokens on stdout, one line, space-separated; a summary
line on stderr; `--json` gives `{model, weights, prompt, generated, window,
greedy, vocab, compile_s, compute_s, compilations, compiled_kernels}`. Exit
codes: 0; 1 a diagnostic or a run-time error (a token outside the
vocabulary, a prompt shorter than the window, a violated bound); 2 no
`inference serve`, no weights file, or a malformed `--prompt`/`--dim`.

**Compiling and computing** (0.17.2 step 4, D200). The summary line ends
with the loop's time in two parts, and `--json` gives them in seconds:
- **compiled** (`compile_s`): each context length the loop has not seen is
  prepared, and inside the call that follows, each kernel new at that
  length is lowered to PSK-Kernel and, natively, compiled by Cranelift (or
  by `clang` with `--backend llvm`) at its first launch. The backend counts
  that time as it goes. The line also gives the lengths prepared
  (`compilations`) and the kernels compiled (`compiled_kernels`). Over a
  window the length is the window's, so one; over a growing context, one a
  token. The reference interpreter compiles nothing, so its line says 0
  kernels.
- **computed** (`compute_s`): the calls' time less that.

The rest of a process's time is neither: reading the weights, the
initializer's call that gives them their shapes, and the program's checking
and lowering before the loop. The timers are a clock read on each side of
each call and each new kernel, so the tokens are the same bits.

## On the tiny MiniGPT

With the checkpoint of `make train-examples` (`out/ckpt/final.psk.d`: twenty
steps on the example tokens, `V=64 D=16 H=2 L=2`, windows of 8):

```sh
presk serve examples/spec/minigpt/serve_minigpt.psk --const V=64 --const D=16 --const H=2 --const L=2 \
     --weights out/ckpt/final.psk.d --prompt "1 2 3 4 5 6 7 8" --tokens 8 --temperature 0 --backend native
```

```text
presk serve: MiniGPT from out/ckpt/final.psk.d, prompt of 8 tokens, 8 generated, greedy, context window 8; compiled in 0.011 s (1 length, 28 kernels), computed in 0.004 s
33 33 33 33 0 0 0 0
```

The times vary from run to run; the tokens do not.

The same eight tokens come out of the reference interpreter at 64 bits, of
the native backend in `f64` and in `f32`, and of the exported `serve.py` in
float64 and in float32 (`make serve-examples`, `make serve-pt` compare the
outputs byte for byte). Twenty training steps at a warming-up learning rate
leave a nearly uniform model, hence the repetitions: the point here is the
agreement, not the text. With sampling (without `--temperature 0`), `--seed`
fixes the tokens, the same on the interpreter at 64 bits and natively in
`f64` and in `f32` (they were each backend's own until 0.10.0 step 7b,
`19 55 1 63 57 1 54 29` natively):

```text
presk serve: MiniGPT from out/ckpt/final.psk.d, prompt of 8 tokens, 8 generated, sampled, context window 8; compiled in 0.005 s (1 length, 27 kernels), computed in 0.002 s
6 7 23 37 52 34 57 19
```

A test (`crates/presk-serve/tests/serve.rs`) does the same on a fresh
initialization: greedy tokens identical on the interpreter and natively in
both realizations, sampled tokens identical on the interpreter and natively
for three seeds and different from one seed to another, the window rule and
its error.

## In PyTorch: `presk export --serve`

```sh
presk export examples/spec/minigpt/serve_minigpt.psk --target pytorch --mode readable --serve \
     --const V=64 --const D=16 --const H=2 --const L=2 -o out/serve_pt
python out/serve_pt/serve.py --weights out/ckpt/final.psk.d --prompt "1 2 3 4 5 6 7 8" --tokens 8 --temperature 0 --dtype float64
```

`--serve` adds `serve.py` and `pskd.py` to the readable export. `serve.py`
builds the exported class with its dimension arguments (`MiniGPT(T)`, `T`
from `--dim T=N` or the prompt's length), loads the weights, and runs the
same loop: `next_token(θ, None, ctx, temperature)` for sampling (the key
passed as `None`, read by nothing; PyTorch draws from its own generator,
seeded by `--seed`, so the sampled tokens are not `presk serve`'s), `model(ctx)[:, -1].argmax(-1)` for
`--temperature 0`, with the same sliding window. The weights are either
`final.pt` from `train.py` or a `.psk.d` checkpoint of `presk train`:
`pskd.read_params` decodes the latter with the **layout of the parameter
tree** generated into the script (`LAYOUT = {"head": None, "emb": {...},
"blocks": ["stack", {...}], ...}`: a leaf is `None`, a module a dict of its
fields in the initializer's order, `stack{L}(M)` a `["stack", …]` whose
leaves carry a leading axis of length `L`) and the shapes of the model's own
parameters, then splits stacked leaves over the `nn.ModuleList`. The file
itself carries only `params MiniGPT` as its type, which is why the layout
travels with the script (`data` chapter, question 28).

## Not in this version

Sessions, batching (`batching = dynamic(…)`), the KV cache (`cache =
kv_cache(axis seq)`, proposed as an e-graph rewrite for 0.4.0), the block's
`precision` and `target` (weights are quantised through `--realize @R.json`
instead, above, and `quantize(int8, calibration = …)` in the block is not
read), a server process: the block's fields are checked and ignored, and
`presk serve` is a command-line loop, not a service. What it fixes is the seam: the generation function of the language,
run through `Backend`, agreeing bit for bit with the reference on the
deterministic path.

## In C: `presk export --target c --serve`

```sh
presk export examples/spec/minigpt/serve_minigpt.psk --target c --serve \
     --const V=64 --const D=16 --const H=2 --const L=2 --dim T=8 --weights out/ckpt/final.psk.d -o out/serve_c
make -C out/serve_c
out/serve_c/generate --weights out/ckpt/final.psk.d --prompt "1 2 3 4 5 6 7 8" --tokens 8 --temperature 0
```

The C export (plan 0.9.0 step 6; `cli` chapter) writes the generation
function and its greedy copy as C functions, at one sequence and the window
the parameters fix (`--dim T=8`, which a C program of one shape needs where
`presk serve` takes the prompt's length), and `generate`, the same loop,
token `i` given the child `i` of the key `--seed` names (`presk_key_child`).
Its draws are the native backend's from the same keys, and its output is
`presk serve --backend native`'s byte for byte, in every realization,
greedy and sampled, `--json` included but for `presk serve`'s four fields of
compiling and computing (statement C5): the eight greedy tokens
above are `33 33 33 33 0 0 0 0` again, and with `--seed` the sampled ones are
the native backend's for that seed. A model whose parameters do not fix the
context's length is exported at the length `--dim T=n` gives (0.17.0 step
7, D180), and refused without it (`E8009`). The export then takes two
copies of the generation function, `next_token.at` and its greedy
`next_token.greedy.at`:
- each reads the logits at a position it is given, an indicator `at :
  ℝ[seq: T]` (1 at the position, 0 elsewhere), where the function read the
  last one: `x[seq: -1]` becomes `sum(axis seq, x · at)`, exactly `x` at
  that position, since a product by 0 or 1 is exact and so is adding zeros;
- `generate.c` pads the context after its last token to `n`, reads the
  logits at that token, and stops when the `n` positions are full;
- by causality those logits are the context's alone, so its tokens are
  `presk serve`'s over the whole, growing context, greedy and sampled,
  `--json` included (`"window": null`).

The function must select the last position itself, once (`E8009`
otherwise). No KV cache comes with it: each step computes the whole padded
context again.

## On Llama 3.2 1B's published weights

`make llama3`'s stage `rw4` (`experiments` chapter, §25; plan 0.17.0
step 7, D180) serves `examples/coverage/llama3.psk` on `meta-llama/Llama-3.2-1B`,
converted to a `.psk.d` of `bf16` leaves (`data` chapter), from four
prompts of 16 tokens:

```sh
presk serve examples/coverage/llama3.psk --weights out/llama3/llama-3.2-1b.psk.d \
     --prompt "128000 791 6864 315 9822 374 12366 11 323 279 6864 315 15704 374 22463 13" \
     --tokens 32 --temperature 0 --backend native --realize f32
presk export examples/coverage/llama3.psk --target c --serve --dim T=48 \
     --weights out/llama3/llama-3.2-1b.psk.d --realize f32 --no-exact -o out/llama3/serve_c_f32
make -C out/llama3/serve_c_f32 generate
```

`presk serve` and `generate.c` give the same 32 greedy tokens from each
prompt, in `f32` and with the blocks' seven matrices in `int8`: " The
capital of the United States is Washington, D.C. The capital of the United
Kingdom is London. …" after "The capital of France is Paris, and the
capital of Italy is Rome." (statement RW4). Neither is fast:
- **One thread.** Both run on one thread. A kernel's split divides its
  outermost loop, which at batch 1 is the batch, so nothing is divided.
- **No KV cache.** Each token computes the whole context again.
- **`presk serve`** also compiles the kernels again at each new length:
  16 to 28 minutes for 32 tokens in `f32`, 8 in `int8`, whose weights are
  a quarter of the bytes. Since 0.17.2 its summary line splits that time
  between compiling and computing (above).
- **`generate.c`** computes all 48 positions at every step, in 21 to 33
  minutes.
- **Memory.** `presk serve` holds about 29 GB in `f32`. The C program's
  arena is 5.0 GB in `f32` and 2.1 GB in `int8`.

A KV cache, and a split over the sequence at batch 1, are later work.
