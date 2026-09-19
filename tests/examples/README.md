# Presk — Exemples

Code source Presk illustrant les constructions de la [spécification](../presk-draft-spec.md).
Chaque exemple vit dans **son propre dossier**, avec un `README.md` local qui explique
ce qu'il montre et les commandes prospectives associées.

> **Statut.** Presk n'a pas encore d'implémentation : ces fichiers ne compilent pas,
> ils *spécifient par l'exemple*. Ils sont pensés pour rester valides une fois le
> front-end écrit, et pour exercer chaque coin du système de types.

## Progression suggérée

| Dossier | Ce qu'il illustre |
|---|---|
| [`mlp/`](mlp/) | Le minimum : `module`, `param`, `fn apply`, composition, `grad` typé. |
| [`layernorm/`](layernorm/) | Réductions sur axe nommé, `param` vecteur, stabilité laissée au compilateur. |
| [`minigpt/`](minigpt/) | Le cœur : attention causale, blocs Transformer, `stack{N}`, `idx[V]`, embeddings, **entraînement**, **inférence** et **build** vers un SoC ARM+GPU. |

## Conventions rappelées

- `ℝ[…]` est un tenseur réel **exact** dans la sémantique ; aucun `float32` dans le source.
- Les axes sont **nommés** (`batch:`, `seq:`, `model:`, `heads:`, `hd:`) — le broadcasting
  et les contractions `ein[…]` sont vérifiés axe par axe.
- Les entiers du graphe sont des **indices** `idx[N]` (tokens, labels), jamais mélangés
  arithmétiquement aux réels sans `real(…)`.
- `let x = …` est une liaison immuable (shadowing autorisé) ; il n'y a jamais de mutation.
- La précision est un **budget** déclaré au niveau `training`/`inference`, pas un dtype dans le code.
