# `mlp/` — The smallest complete Presk program

A two-layer perceptron, to be read first: it shows only the foundation of the language
and nothing specific to Transformers.

## What it illustrates

- A `module` with `param` parameters and their `init` (§4.1).
- Composition of modules through `use` (§4.2).
- The parameter pytree `params net`, generated automatically (§3.4, §6).
- Typed `grad`: `mse` is of class `C∞`, hence differentiable; the regularity is inferred,
  not declared (§3.3, §5).

## Files

- [`mlp.psk`](mlp.psk) — the `Linear` and `MLP` modules, the `mse` objective and its gradient.

## Commands

```sh
presk check mlp.psk
presk export --target pytorch --mode readable --shapes=check mlp.psk -o out/mlp_pt/
```
