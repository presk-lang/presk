# `layernorm/` — Layer normalization

Shows how Presk handles numerical stability: **it is not in the code**.
The semantics is the division by the exact real standard deviation; the compiler chooses,
under the error budget, how to evaluate it without overflow (§8.2).

## What it illustrates

- Reductions over a **named axis**: `mean(axis model, …)`.
- Vector `param`s (gain `γ`, bias `β`), broadcast over the leading axes.
- The explicit injection `real : f64 -> ℝ` to mix a hyperparameter (`eps`) with a real.
- An `RMSNorm` variant on the same pattern.

## Files

- [`layernorm.psk`](layernorm.psk) — `LayerNorm` and `RMSNorm`.

## Commands

```sh
presk check layernorm.psk
```
