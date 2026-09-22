# `minigpt/` — A decoder-only GPT, end to end

The reference example: the description of the model, its training, its serving, and
its compilation toward an ARM+GPU SoC. It is the complete use case that §6 and §7 of the
specification have in mind.

## Files

| File | Role |
|---|---|
| [`minigpt.psk`](minigpt.psk) | The model: `Embedding`, causal `SelfAttention`, `Block`, `stack{L}(Block)`, logits head. |
| [`train_minigpt.psk`](train_minigpt.psk) | `loss` (cross-entropy), `optimizer adamw`, `training run`, budget + parallelism. |
| [`serve_minigpt.psk`](serve_minigpt.psk) | `inference serve`, int8 quantization, `kv_cache`, `sample` under the `Rand` effect. |
| [`target/jetson-agx.pskt`](target/jetson-agx.pskt) | Hardware description for `presk build` (§8.5). |
| [`data/tokens.txt`](data/tokens.txt), [`data/tokens.psk.d`](data/tokens.psk.d) | A small stream of 2048 tokens (vocabulary 64, repeated sentences with a little noise) and its `(x, y)` windows of length 8 in the `.psk.d` format, so that `training run` has data: `presk data lm-windows --vocab 64 --seq 8 --stride 4 data/tokens.txt -o data/tokens.psk.d`. The bounds of the file (64) are below `V` (50257): a sample of a smaller vocabulary is a valid sample. |

The `Linear`/`MLP` and `LayerNorm` modules are imported from [`../mlp/`](../mlp/) and
[`../layernorm/`](../layernorm/): one example reuses the building blocks of the previous ones.

## Points of the specification shown here

- **Tokens = indices, not reals** (§3.5): input and targets are `idx[V]`; `embed` denotes
  a one-hot contraction compiled to a `gather`, with a `scatter-add` gradient on the table.
- **Shapes in the types** (§3.1–3.2): named axes `batch/seq/heads/hd/model/vocab`, the
  attention contraction written with `ein[…]` and checked axis by axis, `static assert D % H == 0`.
- **Exact semantics ⇒ FlashAttention and the KV cache are not kernels** but scheduling
  and certified transformations (§8.2, §7).
- **Precision = budget** (§6, §8.2): no dtype in the source; `budget(rel_error=…)` for
  training, `budget(abs_error=…) | quantize(int8, …)` for inference. A budget is
  discharged by measuring the distance to the exact denotation on the envelope's inputs,
  and it covers only those inputs (§8.3).
- **Sampling** (§2.2): unlike `dropout(rate, key, x)`, `sample` does not take a key yet;
  the compiler draws it from a stream seeded by the run's seed, which holds only as long
  as sampling is not differentiated (§10, open question no. 9).
- **Third-party build** (§8.5): the target is a separate `.pskt` file; `presk build`
  produces a `.pskb` bundle.

## Commands

```sh
# 1. type checking (shapes, regularity, effects)
presk check minigpt.psk

# 2. training
presk run train_minigpt.psk

# 3. pedagogical export to readable PyTorch, with shape asserts (§8.4)
presk export --target pytorch --mode readable --shapes=check minigpt.psk -o out/minigpt_pt/

# 4. cross build (x86 → ARM) toward the SoC, hermetic .pskb bundle (§8.5)
presk build --target target/jetson-agx.pskt serve_minigpt.psk

# 5. validate on the exact reference interpreter BEFORE having the hardware
presk build --target target/jetson-agx.pskt --emulate serve_minigpt.psk
```
