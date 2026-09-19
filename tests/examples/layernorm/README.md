# `layernorm/` — Normalisation de couche

Montre comment Presk traite la stabilité numérique : **elle n'est pas dans le code**.
La sémantique est la division par l'écart-type réel exact ; le compilateur choisit
sous le budget d'erreur comment l'évaluer sans débordement (§8.2).

## Ce que ça illustre

- Réductions sur un **axe nommé** : `mean(axis model, …)`.
- `param` vecteurs (gain `γ`, biais `β`) et broadcast sur les axes de tête.
- L'injection explicite `real : f64 -> ℝ` pour mêler un hyperparamètre (`eps`) à un réel.
- Une variante `RMSNorm` sur le même patron.

## Fichiers

- [`layernorm.psk`](layernorm.psk) — `LayerNorm` et `RMSNorm`.

## Commandes (prospectives)

```sh
presk check layernorm.psk
```
