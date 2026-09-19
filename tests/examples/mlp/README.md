# `mlp/` — Le plus petit programme Presk complet

Un perceptron à deux couches. À lire en premier : il ne montre que le socle du langage
et rien de spécifique aux Transformers.

## Ce que ça illustre

- `module` avec paramètres `param` et leur `init` (§4.1).
- Composition de modules par `use` (§4.2).
- Le pytree de paramètres `params net` engendré automatiquement (§6).
- `grad` typé : `mse` est de classe `C∞`, donc dérivable — la régularité est inférée (§3.3, §5).

## Fichiers

- [`mlp.psk`](mlp.psk) — les modules `Linear`, `MLP`, l'objectif `mse` et son gradient.

## Commandes (prospectives)

```sh
presk check mlp.psk
presk export --target pytorch --mode readable --shapes=check mlp.psk -o out/mlp_pt/
```
