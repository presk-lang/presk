# `minigpt/` — Un GPT décodeur-seul, de bout en bout

L'exemple de référence : la description du modèle, son entraînement, son service, et
sa compilation vers un SoC ARM+GPU. C'est le cas d'usage complet évoqué au §6 de la spec.

## Fichiers

| Fichier | Rôle |
|---|---|
| [`minigpt.psk`](minigpt.psk) | Le modèle : `Embedding`, `SelfAttention` causale, `Block`, `stack{L}(Block)`, tête logits. |
| [`train_minigpt.psk`](train_minigpt.psk) | `loss` (cross-entropy), `optimizer adamw`, `training run`, budget + parallélisme. |
| [`serve_minigpt.psk`](serve_minigpt.psk) | `inference serve`, quantification int8, `kv_cache`, `sample` sous effet `Rand`. |
| [`target/jetson-agx.pskt`](target/jetson-agx.pskt) | Description matérielle pour `presk build` (§8.5). |

Les modules `Linear`/`MLP` et `LayerNorm` sont importés depuis [`../mlp/`](../mlp/) et
[`../layernorm/`](../layernorm/) — un exemple réutilise les briques des précédents.

## Points de la spec mis en scène

- **Tokens = indices, pas réels** (§3.5) : entrée et cibles en `idx[V]` ; `embed` dénote
  une contraction one-hot compilée en `gather`, gradient `scatter-add` sur la table.
- **Formes dans les types** (§3.1–3.2) : axes nommés `batch/seq/heads/hd/model/vocab`,
  contraction d'attention par `ein[…]` vérifiée axe par axe, `static assert D % H == 0`.
- **Sémantique exacte ⇒ FlashAttention et KV-cache ne sont pas des noyaux** mais des
  transformations d'ordonnancement/certifiées (§8.2, §7).
- **Précision = budget** (§6, §8.2) : aucun dtype dans le source ; `budget(rel_error=…)`
  à l'entraînement, `budget(abs_error=…) | quantize(int8, …)` à l'inférence.
- **Build tiers** (§8.5) : la cible est un `.pskt` séparé ; `presk build` produit un `.pskb`.

## Commandes (prospectives)

```sh
# 1. vérification de types (formes, régularité, effets)
presk check minigpt.psk

# 2. entraînement
presk run train_minigpt.psk

# 3. export pédagogique vers PyTorch lisible, avec asserts de shapes (§8.4)
presk export --target pytorch --mode readable --shapes=check minigpt.psk -o out/minigpt_pt/

# 4. build croisé (x86 → ARM) vers le SoC, bundle .pskb hermétique (§8.5)
presk build --target target/jetson-agx.pskt serve_minigpt.psk

# 5. valider sur l'interprète de référence exact AVANT d'avoir le matériel
presk build --target target/jetson-agx.pskt --emulate serve_minigpt.psk
```
