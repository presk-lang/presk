# Serving: `inference serve`, `presk serve`, `serve.py`

The `inference serve` block of a file names a model and its weights; `presk
serve` generates tokens with them, on any backend, through the file's own
*generation function*; `presk export --serve` writes the same loop in
PyTorch. This is the minimal form of the specification's `serve` (plan 0.3.0
§3.6, question 37): one prompt, no sessions, no batching, no cache.

## The block, resolved

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
- The **generation function** is the top-level function of the file whose
  signature is `(θ : params M, ctx : idx[V][batch: B, seq: T], … with
  defaults) -> idx[V][batch: B]`: `next_token` in the example, which applies
  the model to the context, keeps the last position and samples. A file with
  the block but no such function gets `W0003`; `presk serve --entry FN` then
  names one.
- `precision`, `target`, `batching`, `cache` pass the schema and are listed
  as *ignored* in `--summary`: they have no effect in this version.

```text
-- inference --
inference serve: model MiniGPT
    weights : ckpt/final (tests/examples/minigpt/ckpt/final.psk.d, not found)
    generate: next_token(θ, ctx, temperature)
    ignored : precision, target, batching, cache (checked, without effect in this version)
```

## `presk serve`

```sh
presk serve FILE.psk [--const NAME=N]... [--weights FILE.psk.d] --prompt "t₁ t₂ …" [--tokens N]
     [--temperature x] [--entry FN] [--dim NAME=N]... [--seed S] [--bits 256]
     [--backend interp|kernel|native] [--realize f64|f32] [--json]
```

The loop, in `presk-serve`:

```text
θ   = load(weights, shaped like M.init(key))
ctx = prompt
repeat --tokens times:
  next = generate(θ, ctx[-T:], temperature)
  ctx  = ctx ++ [next]
```

Everything with a semantics is the program's: the model, the generation
function, the initializer that gives the weights their shapes. The crate only
sequences calls through the `Backend` seam, so the same command runs on the
reference interpreter (`--bits`), on the kernel interpreter or natively
(`native` chapter). Two forms of the step:

- **sampling** (`--temperature x > 0`, default: the function's own default,
  `0.8` in the example): the function is called as written; its `sample`
  draws from the backend's stream seeded by `--seed`, so two backends need
  not produce the same tokens (question 41), but one backend is deterministic
  for a seed;
- **greedy** (`--temperature 0`): a copy of the function in which every
  `sample` is an `argmax` is called instead (`next_token.greedy`, appended to
  the Core program; the temperature is passed as `1`). Every backend and
  realization agrees on this path, up to exact ties in the logits.

**The context length.** In the example the positional embedding is `param
pos : ℝ[T, D]` with the file's `dim T <= 1024`: the parameters fix `T`, and a
checkpoint trained on windows of 8 tokens only applies to contexts of 8. When
the initializer's dimensions include the context's `seq` dimension, `presk
serve` takes `T` from `--dim T=N` (default: the prompt's length), loads the
weights at that shape and passes the model the **last `T` tokens** of the
context at every step: a sliding window, reported as `context window 8`. A
prompt shorter than `T` is refused (`E7001`), a longer one slides. When the
parameters do not depend on `T`, the whole context is passed and grows, and
the file's `dim T <= …` bound is checked at every step. Spec question 43
records the observation; `ℝ[Tmax, D]` with a slice (question 9) would let the
same model read any context up to `Tmax`.

Output: the generated tokens on stdout, one line, space-separated; a summary
line on stderr; `--json` gives `{model, weights, prompt, generated, window,
greedy, vocab}`. Exit codes: 0; 1 a diagnostic or a run-time error (a token
outside the vocabulary, a prompt shorter than the window, a violated bound);
2 no `inference serve`, no weights file, or a malformed `--prompt`/`--dim`.

## On the tiny MiniGPT

With the checkpoint of `make train-examples` (`out/ckpt/final.psk.d`: twenty
steps on the example tokens, `V=64 D=16 H=2 L=2`, windows of 8):

```sh
presk serve tests/examples/minigpt/serve_minigpt.psk --const V=64 --const D=16 --const H=2 --const L=2 \
     --weights out/ckpt/final.psk.d --prompt "1 2 3 4 5 6 7 8" --tokens 8 --temperature 0 --backend native
```

```text
presk serve: MiniGPT from out/ckpt/final.psk.d, prompt of 8 tokens, 8 generated, greedy, context window 8
33 33 33 33 0 0 0 0
```

The same eight tokens come out of the reference interpreter at 64 bits, of
the native backend in `f64` and in `f32`, and of the exported `serve.py` in
float64 and in float32 (`make serve-examples`, `make serve-pt` compare the
outputs byte for byte). Twenty training steps at a warming-up learning rate
leave a nearly uniform model, hence the repetitions: the point here is the
agreement, not the text. With sampling, `--seed` fixes the tokens of one
backend:

```text
presk serve: MiniGPT from out/ckpt/final.psk.d, prompt of 8 tokens, 8 generated, sampled, context window 8
19 55 1 63 57 1 54 29
```

A test (`crates/presk-serve/tests/serve.rs`) does the same on a fresh
initialization: greedy tokens identical on the interpreter and natively in
both realizations, sampling deterministic per seed, the window rule and its
error.

## In PyTorch: `presk export --serve`

```sh
presk export tests/examples/minigpt/serve_minigpt.psk --target pytorch --mode readable --serve \
     --const V=64 --const D=16 --const H=2 --const L=2 -o out/serve_pt
python out/serve_pt/serve.py --weights out/ckpt/final.psk.d --prompt "1 2 3 4 5 6 7 8" --tokens 8 --temperature 0 --dtype float64
```

`--serve` adds `serve.py` and `pskd.py` to the readable export. `serve.py`
builds the exported class with its dimension arguments (`MiniGPT(T)`, `T`
from `--dim T=N` or the prompt's length), loads the weights, and runs the
same loop: `next_token(θ, ctx, temperature, generator)` for sampling (a
seeded `torch.Generator`), `model(ctx)[:, -1].argmax(-1)` for
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
kv_cache(axis seq)`, proposed as an e-graph rewrite for 0.4.0), quantization
and targets (`precision`, `target`), a server process: the block's fields
are checked and ignored, and `presk serve` is a command-line loop, not a
service. What it fixes is the seam: the generation function of the language,
run through `Backend`, agreeing bit for bit with the reference on the
deterministic path.
