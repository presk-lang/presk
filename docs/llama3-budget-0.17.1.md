# Llama 3.2 1B solved whole, against the field's recipes (plan 0.17.1)

The record of plan 0.17.1 (`presk-specs/implementation/presk-plan-0.17.1.md`),
run on falicon by `experiments/llama/llama3_budget.sh` (`make
llama3-budget`), its outputs in `out/llama3-budget/`. Each step's decision
holds the details: D185 (the version opened, the options, OP1), D186 (the
exact logits and ε\*(δ)), D187 (the envelope and the solves, RW5), D188 (the
measurements, RW6 and RW7), D189 (perplexity, RW8, and its addendum: `nll` in `f64` for the six rows
whose default is `f32`), D190 (time, RW9), D191 (the version). This file
gathers the figures as each step lands.

## 1. The samples

WikiText-2 raw (`Salesforce/wikitext`, the user's download), each split's
`text` joined by "\n\n" and tokenized whole by the checkpoint's tokenizer
(`texts.py`; `texts.json` holds the Parquet files' sha256 and every window
decoded). Every window is the beginning-of-text token (128000) and the
split's next tokens, contiguous and disjoint.

| sample | windows | split | its tokens | used by |
|---|---|---|---|---|
| calibration | 4 of 16 | train | 0–60 | the envelope (step 2) |
| held-out | 32 of 32 | test | 0–992 | the margins, ε\*(δ), every realization's gap and flips (steps 1, 3) |
| quality | 128 of 129 | test | 992–17 376 | perplexity: 128 of context, 128 targets a window, 16 384 predictions (step 4) |

## 2. The exact logits and ε\*(δ) (step 1, D186)

The 32 held-out windows on the reference at 128 bits, one window a run and
one run at a time, each in a scope of 108 GB, from 2026-11-05 20:17 to
2026-11-06 18:15:49 (`target/release/presk` at `35b9af3`, D185). Each run
took 40 min 51 s to 42 min 27 s and 96.65 GB; all 32 exited cleanly.

**The margins** m = y₍₁₎ − y₍₂₎ of the 1 024 exact decisions
(`margins.json`, `margins.md`):

| | value |
|---|---|
| positions | 1 024 (32 windows of 32) |
| ties (m = 0) | none |
| largest exact \|logit\| | 30.24 |
| smallest margin | 1.888·10⁻³ (window 7) |
| 0.1 %, 0.3 %, 1 %, 3 % quantiles | 5.29·10⁻³, 7.48·10⁻³, 1.13·10⁻², 3.21·10⁻² |
| 10 %, 25 %, median | 0.108, 0.287, 0.821 |
| largest margin | 9.40 |

The first position of every window predicts from the beginning-of-text
token alone, so 32 of the 1 024 decisions are one decision (its margin
1.033). The windows are consecutive in one text: the positions are not
independent draws, which the Clopper–Pearson bound assumes.

**ε\*(δ)** is half the (k+1)-th smallest margin, with k from the one-sided
Clopper–Pearson bound at α = 0.05 (`margins.py`'s rule), in logit units:

| δ | decisions allowed, δ·N | k | ε\*(δ) | relative to 30.24 | raw: k = δ·N, ε |
|---|---|---|---|---|---|
| 10⁻¹ | 102 | 86 | 4.880·10⁻² | 1.61·10⁻³ | 5.389·10⁻² |
| 3·10⁻² | 30 | 21 | 1.091·10⁻² | 3.61·10⁻⁴ | 1.607·10⁻² |
| 10⁻² | 10 | 4 | 3.807·10⁻³ | 1.26·10⁻⁴ | 5.674·10⁻³ |

At ε\*(10⁻¹) the bound #{m < 2ε} counts 86 positions in 30 windows; at
ε\*(3·10⁻²), 21 in 17; at ε\*(10⁻²), 4 in 4.

## 3. The envelope and the solves (step 2, D187)

On `out/bin/presk-0.17.1-1d865e3`, each run alone on falicon in its own
scope of 108 GB, from 2026-11-06 20:55:50 to 2026-11-07 20:46:12:

| run | time | peak |
|---|---|---|
| `presk calibrate --precision --bits 128`, the 4 calibration windows of 16 | 6 h 34 min 32 s | 98.65 GB |
| each of the plan's seven solves | 2 h 05 min 34 s to 2 h 06 min 58 s | 96.74 GB |
| the eighth (`--min-gain 0`) | 2 h 18 min 37 s | 96.74 GB |

The envelope holds 123 sites; its worst condition number is 6.89
(`SwiGLU.apply/%3`); 37 of its 113 rounded sites are exact in `bf16`.

Every solve ran under `--menu f64,f32,bf16,int8,int4 --sweep f64
--measure native`; the relative budgets are of the calibration sample's
largest exact |logit|, 27.55:

| solve | budget, absolute | setting | measurements | realization | measured gap |
|---|---|---|---|---|---|
| `eps-1e-1` | 4.880·10⁻² | `--narrow measured` | 9 | A | 2.753·10⁻¹³ |
| `eps-3e-2` | 1.091·10⁻² | `--narrow measured` | 9 | A | 2.753·10⁻¹³ |
| `eps-1e-2` | 3.807·10⁻³ | `--narrow measured` | 9 | A | 2.753·10⁻¹³ |
| `rel-1e-2` | 0.2755 | `--narrow measured` | 9 | A | 2.753·10⁻¹³ |
| `rel-1e-3` | 2.755·10⁻² | `--narrow measured` | 9 | A | 2.753·10⁻¹³ |
| `eps-1e-2-defaults` | 3.807·10⁻³ | no narrowing | 1 | A | 2.753·10⁻¹³ |
| `eps-1e-2-margin1` | 3.807·10⁻³ | `--narrow measured --margin 1` | 9 | A | 2.753·10⁻¹³ |
| `eps-1e-2-mingain0` | 3.807·10⁻³ | `--narrow measured --min-gain 0` | 90 | B | 1.515·10⁻⁴ |

**Realization A**, the plan's seven, one file byte for byte: the token
table and the seven block matrices stored in `bf16`, the three norms'
γ in `f64`, every value in `f64`. 2 472 034 304 bytes at rest; the
traffic 2.4982·10⁹ bytes against 9.9126·10⁹ in `f64` (3.97×).

**Realization B**, the eighth: every parameter in `bf16`, the norms too
(2 471 628 800 bytes, the realization by hand's); 32 of the 44 values the
rules leave in `f32` (every product with a weight, the logits, the
attention's scores and softmax, the norms, the residual sums, SwiGLU,
RoPE's arithmetic); 12 in `bf16`: Q, K and V at their reshapes before the
attention, its output before `wo`, RoPE's input and output, the
embedding and two integers made real (exact), and one term under RoPE's
clamp. The 15 values of one element stay `f64`. Traffic 2.4842·10⁹
(3.99×).

**Why one realization for seven budgets.** Every budget lies between the
two rungs the weights can take. `bf16` is free: the checkpoint is `bf16`,
so its term at rest is 0. `int8` per column is far: its term at rest is
7.44 on the token table and 62.7 to 1.39·10³ on the block matrices, and
measured, each matrix alone in `int8` moves a logit of the calibration
sample by more than 0.1102 (`rel-1e-2`'s narrowing threshold, which no
candidate met). Rule 2 keeps every value in `f64`: the largest, the
logits, saves 8.2·10⁶ bytes in `f32`, under θ·B = 9.9·10⁶. So the menu
binds, not the budget. RW5 holds.

## 4. Every realization on the held-out sample (step 3, D188; the uniform and the field's rows taken at step 1, D186)

Natively for ours, one window and one thread a run. The field's from their
logits by their own programs: llama.cpp b12132 through `libllama`
(`llamacpp_logits`, each window from an empty cache in one batch),
transformers in its dtype at batch 1 (`rw2.py --dtype`). The gap is the
largest |z − exact| over 1 024 × 128 256 logits. "Relative" is the largest
over windows of the window's gap over its largest exact |logit|. The bound
is #{m < 2g}, and its share of the 1 024 is the δ at which the
realization's own gap certifies it. The bytes are the weights at rest:
presk's parameters with their scales, the GGUF's tensors, the state dict.

| realization | weights (bytes) | gap | relative | flips | #{m < 2g} | δ certified |
|---|---|---|---|---|---|---|
| `f64` | 9 886 515 200 | 3.666·10⁻¹³ | 1.52·10⁻¹⁴ | 0 | 0 | 0 |
| `f32` | 4 943 257 600 | 4.277·10⁻⁴ | 1.57·10⁻⁵ | 0 | 0 | 0 |
| `bf16-params-f32` (by hand) | 2 471 628 800 | 4.277·10⁻⁴ | 1.57·10⁻⁵ | 0 | 0 | 0 |
| `bf16` | 2 471 628 800 | 0.7060 | 2.63·10⁻² | 32 | 743 | 0.726 |
| `int8` absmax | 2 025 529 344 | 2.100 | 7.19·10⁻² | 27 | 973 | 0.950 |
| `int8` pow2 | 2 025 529 344 | 3.055 | 0.112 | 48 | 1 009 | 0.985 |
| `int4` absmax | 1 538 990 080 | 19.65 | 0.783 | 624 | 1 024 | 1 |
| `int4` pow2 | 1 538 990 080 | 19.94 | 0.819 | 814 | 1 024 | 1 |
| llama.cpp BF16 | 2 471 764 096 | 0.2569 | 9.57·10⁻³ | 0 | 391 | 0.382 |
| llama.cpp Q8_0 | 1 313 251 456 | 1.810 | 7.01·10⁻² | 28 | 959 | 0.937 |
| llama.cpp Q4_K_M | 799 862 912 | 11.07 | 0.379 | 163 | 1 024 | 1 |
| llama.cpp Q4_0 | 763 097 216 | 15.61 | 0.611 | 266 | 1 024 | 1 |
| transformers `float32` | 4 943 257 600 | 2.401·10⁻⁴ | 1.01·10⁻⁵ | 0 | 0 | 0 |
| transformers `bfloat16` | 2 471 628 800 | 0.5660 | 2.25·10⁻² | 32 | 659 | 0.644 |

`bf16-params-f32` is every parameter `bf16` at rest and every stored value
`f32` (`experiments/llama/bf16-params-f32.json`, the user's, 2026-11-06).
The checkpoint is `bf16`, so it stores `f32`'s values, and its rows are
`f32`'s byte for byte on all 32 windows: `f32`'s certificate at half its
weight.

Against ε\*(δ):
- `f32`, `bf16-params-f32` and transformers `float32` are inside all three
  ε\*(δ).
- Every other realization of 2.47 GB or less, ours or the field's, is
  outside all three. llama.cpp BF16, the closest, is 5.3 times
  ε\*(10⁻¹).
- The flips are a different count. `bf16` (32), `int8` absmax (27),
  llama.cpp Q8_0 (28) and transformers `bfloat16` (32) are within 10⁻¹'s
  102 decisions, and `int8` absmax and Q8_0 within 3·10⁻²'s 30, without a
  certificate.

**The solved realizations** (step 3, D188). The eight solved names ran on
2026-11-07, 20:51 to 22:37: each new realization's first window alone, then
three windows at a time. The plan's seven give one row, byte for byte, on
32 of 32 windows, and that row is `f64`'s byte for byte.

| realization | solves | weights (bytes) | gap | relative | mean gap | flips | #{m < 2g} |
|---|---|---|---|---|---|---|---|
| A | the plan's seven | 2 472 034 304 | 3.666·10⁻¹³ | 1.52·10⁻¹⁴ | 1.73·10⁻¹⁴ | 0 | 0 |
| B | the eighth (`--min-gain 0`) | 2 471 628 800 | 7.703·10⁻⁴ | 2.87·10⁻⁵ | 1.56·10⁻⁵ | 0 | 0 |

- **B** measures 1.515·10⁻⁴ on the calibration sample and 7.703·10⁻⁴ held
  out. That is 5.1 times more off its sample, and still 4.9 times under its
  budget.
- **The bound at each budget**, #{m < 2ε}, at the budget the solve took (the
  nearest double to ε\*):
  - 88, 21 and 5 at ε\*(10⁻¹), ε\*(3·10⁻²), ε\*(10⁻²), against 102, 30
    and 10 allowed;
  - 411 at the relative 10⁻² (0.2755) and 57 at 10⁻³.

  At ε\* exactly the counts are 86, 21 and 4, the Clopper–Pearson k. The
  double lies just above the (k+1)-th margin's half, and at 10⁻¹ two
  positions share that margin.
- **The certificate is sound on every row.** Every flip of all 22 rows, the
  field's among them, lies where the exact margin is under twice that
  position's gap.
- **Where a row flips, the bound is 21 to 36 times its flips.** The
  `int4` formats are the exception: every position lies under 2g there.

**The statements:**

| δ | allowed | ε\*(δ) | solved | its flips | RW6 | lightest within δ·N | RW7 |
|---|---|---|---|---|---|---|---|
| 10⁻¹ | 102 | 4.880·10⁻² | A | 0 | holds | llama.cpp Q8_0, 1 313 251 456 (28 flips) | fails: A is 9th of 12 |
| 3·10⁻² | 30 | 1.091·10⁻² | A | 0 | holds | llama.cpp Q8_0 | fails: 6th of 9 |
| 10⁻² | 10 | 3.807·10⁻³ | A | 0 | holds | `bf16-params-f32` and B, 2 471 628 800 (0 flips) | fails: 4th of 7 |

- **RW6 holds at every δ.** A changes no held-out decision, and its gap
  is under each budget.
- **RW7 fails at every δ.**
  - *At 10⁻¹ and 3·10⁻², the cause is the certificate.* Rows whose gaps
    are far above ε\*(δ) stay within the flips allowed:
    - Q8_0 at 1.81, 37 times ε\*(10⁻¹);
    - our per-column `int8` absmax at 2.10, 43 times ε\*(10⁻¹), with 27
      flips, 446 504 960 bytes under A;
    - at 10⁻¹, every `bf16` row too.

    The plan's exception, grouped scales, is not the reason: our
    per-column `int8` has none and beats A at both δ.
  - *At 10⁻², the cause is rule 2.* A keeps the norms' γ in `f64`, since
    their first narrowing saves under θ·B. That is A's 405 504 bytes over
    the realization by hand and 270 208 over llama.cpp BF16.
    - The eighth solve, with rule 2 off, ties the realization by hand for
      the fewest bytes.
    - It is not the statement's row.

## 5. Perplexity on the quality text (step 4, D189)

`Llama3.nll` on the 128 quality windows, B = 8 a run on 8 X925 cores, one
run at a time. Its sites beyond `Llama3.apply`'s take the realization's
default type. The field's six are computed from their logits on the same
windows, with the log-softmax in binary64 (`budget.py nll-field`). The runs
took place from 2026-11-07 22:52 to 2026-11-08 08:10, each realization new
to `nll` first alone (21.75 to 23.85 GB). The plan's seven solves give
`f64`'s 16 runs byte for byte.

g is each row's largest absolute gap on the held-out sample (§4). `f64`'s
perplexity is 24.769920.

| realization | weights (bytes) | perplexity | Δ log ppl | 2g | orders under 2g |
|---|---|---|---|---|---|
| `f64` | 9 886 515 200 | 24.7699 | 0 | 7.3·10⁻¹³ | — |
| `f32` | 4 943 257 600 | 24.7601 | −3.961·10⁻⁴ | 8.553·10⁻⁴ | 0.33 |
| `bf16-params-f32` (by hand) | 2 471 628 800 | 24.7601 | −3.961·10⁻⁴ | 8.553·10⁻⁴ | 0.33 |
| `bf16` | 2 471 628 800 | 24.8159 | +1.854·10⁻³ | 1.412 | 2.9 |
| `int8` absmax | 2 025 529 344 | 24.8638 | +3.782·10⁻³ | 4.200 | 3.0 |
| `int8` pow2 | 2 025 529 344 | 24.9791 | +8.411·10⁻³ | 6.110 | 2.9 |
| `int4` absmax | 1 538 990 080 | 75.5661 | +1.115 | 39.30 | 1.5 |
| `int4` pow2 | 1 538 990 080 | 596.2051 | +3.181 | 39.88 | 1.1 |
| A (the plan's seven) | 2 472 034 304 | 24.7699 | 0 | 7.3·10⁻¹³ | — |
| B (the eighth) | 2 471 628 800 | 24.7699 | +4.51·10⁻⁷ | 1.541·10⁻³ | 3.5 |
| llama.cpp BF16 | 2 471 764 096 | 24.7661 | −1.542·10⁻⁴ | 0.5138 | 3.5 |
| llama.cpp Q8_0 | 1 313 251 456 | 24.8192 | +1.986·10⁻³ | 3.620 | 3.3 |
| llama.cpp Q4_K_M | 799 862 912 | 25.9183 | +4.532·10⁻² | 22.14 | 2.7 |
| llama.cpp Q4_0 | 763 097 216 | 27.9202 | +0.1197 | 31.22 | 2.4 |
| transformers `float32` | 4 943 257 600 | 24.7699 | −7.9·10⁻⁹ | 4.801·10⁻⁴ | 4.8 |
| transformers `bfloat16` | 2 471 628 800 | 24.7883 | +7.43·10⁻⁴ | 1.132 | 3.2 |

A's Δ is 0 against every 2ε, the smallest 7.61·10⁻³; B's is 5.9·10⁻⁵ of its
2ε.

**RW8 holds.** Every perplexity lies within e^{±2g} of `f64`'s, and every
solved realization's within e^{±2ε}.

**The prediction**, one to two orders under the bound, holds or is passed
for every row but two.
- The rows with a gap from 0.25 to 16 are 2.4 to 3.5 orders under it.
- `int4`'s bound is vacuous, e^{±39}.
- `f32` and the realization by hand are at 0.46 of theirs.

**Their offset is `nll`'s log-softmax, not their logits.** `nll` sums
exp(z − max) over the 128 256 tokens in the realization's default type and
order. Under `f32`, that is one accumulator, in sequence, and the sum comes
out low.
- On window 0 (`out/llama3-budget/nll-bias/`), the split is:
  - the `f32` logits with the log-softmax in binary64: −8.2·10⁻⁷;
  - `nll`'s own log-softmax: −3.918·10⁻⁴, under at 128 of 128 targets;
  - a sequential `f32` sum in numpy reproduces our `nll` within 1.9·10⁻⁶.
- On the whole text, `f32`'s `nll` is under `f64`'s at 16 358 of 16 384
  targets. transformers `float32`, whose log-softmax is binary64's, is
  centred.
- `int8` absmax carries the same −3.90·10⁻⁴ on window 0, so does every
  row of ours whose default is `f32`: about −3.9·10⁻⁴ in log, 0.04 % of
  the perplexity. `f64`, A and B take the log-softmax in binary64, as the
  field's do.
- Per target, a narrower `nll` can pass 2g: `f32` at 18 of the 16 384
  targets, `bf16` at 4. The bound counts the logits' error only. Here g is
  the held-out sample's gap; the quality text has no exact logits.

**The check.** On llama.cpp's own windows of the whole WikiText-2 test
text (32 of 128 tokens, 2 016 targets), the tokenizers agree on all
289 077 ids. Our reading of llama.cpp's logits gives `llama-perplexity`'s
figure at `-b 128` for all four GGUFs: 16.5726, 16.5793, 16.9761 and
18.5607. At its default `-b 2048` it prints 16.5639, 16.5911, 17.0154 and
18.5478 (D186: its figure follows the micro-batch).

### D189's addendum: `nll`'s sites in `f64` (2026-11-08, after D191)

The user asked, after step 6, for the six rows whose default is `f32` to
be run again with `nll`'s own sites typed `f64` by per-site storage
(`{"nll/%3": "f64", "nll/%5": "f64"}`: the logits `nll` receives and its
`cross_entropy`). `Llama3.apply`'s kernels do not change; the
log-softmax of the row's `f32` logits is then taken in binary64, as the
field's is. The rows above stay of record; these sit beside them
(`out/llama3-budget/nll-f64sites/`, `compare.md`). The runs took 2 h 50 min
alone on falicon, 16 runs a row, each row's first alone at 23.71 to
26.99 GB.

| realization | perplexity, D189 | perplexity | Δ log ppl, D189 | Δ log ppl | the log-softmax's share | orders under 2g, D189 | orders under 2g |
|---|---|---|---|---|---|---|---|
| `f32` | 24.7601 | 24.7699 | −3.961·10⁻⁴ | −1.46·10⁻⁷ | −3.960·10⁻⁴ | 0.33 | 3.8 |
| `bf16-params-f32` (by hand) | 24.7601 | 24.7699 | −3.961·10⁻⁴ | −1.46·10⁻⁷ | −3.960·10⁻⁴ | 0.33 | 3.8 |
| `int8` absmax | 24.8638 | 24.8736 | +3.782·10⁻³ | +4.178·10⁻³ | −3.964·10⁻⁴ | 3.0 | 3.0 |
| `int8` pow2 | 24.9791 | 24.9890 | +8.411·10⁻³ | +8.808·10⁻³ | −3.963·10⁻⁴ | 2.9 | 2.8 |
| `int4` absmax | 75.5661 | 75.6076 | +1.115 | +1.116 | −5.49·10⁻⁴ | 1.5 | 1.5 |
| `int4` pow2 | 596.2051 | 596.5223 | +3.181 | +3.181 | −5.32·10⁻⁴ | 1.1 | 1.1 |

- **RW8 holds on the six**, as it did.
- **`f32` and the realization by hand stand 3.8 orders under their
  bound**, where D189 had 0.33, so the prediction now holds for them.
  - They are under `f64` at 8 342 of 16 384 targets, sd 1.42·10⁻⁵.
  - No target passes 2g; D189 had 18.
- **The share is the term D189 named.** It is negative at all but at most
  two targets in every row: −3.96·10⁻⁴ for `f32`, by hand and `int8`, and
  −5.3 to −5.5·10⁻⁴ for `int4`.
- **On window 0** the addendum's `nll` is numpy's binary64 log-softmax of
  the same logits within 5.3·10⁻¹⁴ a target (D189's split: −8.199·10⁻⁷
  for `f32`, −4.313·10⁻³ for `int8` absmax, both found again).
- **The realization by hand's runs are `f32`'s**, 16 of 16 byte for byte.
- **Against transformers `float32`**, `f32` now differs per target by mean
  −1.4·10⁻⁷ and sd 1.39·10⁻⁵. Both are centred on `f64`.
  - So D189's 24.7601 against 24.7699 was the log-softmax's difference
    alone.
  - Every row of ours but `bf16`, whose `nll` stays in `bf16`, is now the
    field's function of its logits.

## 6. Time (step 5, D190)

RW9, alone on falicon, on 2026-11-08: one thread pinned to one X925 core
(core 5), three processes for each row. Each figure is the median of the
processes and the fastest of them (D126).
- **The forward pass:** `presk bench --native-only --threads 1 --warmup 1
  --repeat 5` at B = 1, T = 16. Each process gives the median of its five
  calls.
- **Serving:** `presk serve` for 32 greedy tokens from D180's first
  prompt, timed by `/usr/bin/time`. It runs the scalar code at lengths 16
  to 47 and compiles at each length.
- **The traffic ratio** is the all-`f64` traffic over the realization's,
  in the solver's unit: the parameters at rest and each stored value at
  one sample's elements.

The outputs are in `out/llama3-budget/time/`, and `time.md` gathers them
(`budget.py time`).

**The forward pass**, in seconds, the median of three processes (the
fastest in brackets):

| realization | traffic ratio | scalar | with lanes | ÷ `f64`, scalar | ÷ `f64`, lanes |
|---|---|---|---|---|---|
| `f64` | 1.000 | 14.617 (14.616) | 14.434 (14.410) | 1.000 | 1.013 |
| `f32` | 2.000 | 11.512 (11.511) | 10.135 (10.108) | 1.270 | 1.442 |
| `bf16` | 4.000 | 10.220 (10.202) | — | 1.430 | — |
| `int8` absmax | 4.863 | 7.751 (7.741) | 5.431 (5.420) | 1.886 | 2.692 |
| `int8` pow2 | 4.863 | 7.767 (7.766) | 5.425 (5.405) | 1.882 | 2.694 |
| `int4` absmax | 6.387 | 9.903 (9.899) | 4.829 (4.825) | 1.476 | 3.027 |
| `int4` pow2 | 6.387 | 9.907 (9.886) | 4.819 (4.812) | 1.475 | 3.033 |
| `bf16-params-f32` (by hand) | 3.990 | 9.977 (9.909) | 7.733 (7.728) | 1.465 | 1.890 |
| A at ε\*(10⁻²) | 3.968 | 9.874 (9.857) | 10.256 (10.249) | 1.480 | 1.425 |
| B (the eighth) | 3.990 | 9.821 (9.768) | 7.775 (7.757) | 1.488 | 1.880 |

- A's seven copies, one realization, give 9.820 to 9.897 s scalar across
  21 processes. That 0.8 % is the noise between two rows here.
- The uniform rows, timed again in all 48 processes, vary by at most
  1.8 %, so no process ran in D119's slow mode.

**Serving 32 tokens**, in seconds:

| realization | median | fastest | ÷ `f32` | peak |
|---|---|---|---|---|
| `f32` | 806.30 | 804.47 | 1.000 | 31.12 GB |
| A at ε\*(10⁻²) | 723.91 | 723.78 | 0.898 | 23.85 GB |
| `bf16-params-f32` (by hand) | 548.63 | 545.06 | 0.680 | 19.02 GB |
| B (the eighth) | 547.07 | 547.04 | 0.678 | 23.85 GB |
| `int8` absmax | 428.81 | 428.62 | 0.532 | 24.40 GB |

`f32`, A, B and the realization by hand generate D180's `f32` tokens.
`int8` gives D180's `int8` tokens, which differ at the last one.

**RW9, reported.**
- **A against `f64` and `f32`.** A, the realization solved at every
  budget, runs the forward pass 1.48× faster than `f64` and 1.17× faster
  than `f32`, both scalar. It serves 1.11× faster than `f32`.
- **The time does not follow the traffic.** A's ratio is 3.97 and its
  time ratio 1.48. `bf16`, at a ratio of 4.00, is 1.43×, and scalar
  `int4`, at 6.39, is no faster than A.
- **Lanes.** A's lanes, two `f64`, make it 4 % slower. The realizations
  whose values are `f32` gain: by hand and B 1.26–1.29×, `int8` 1.43×,
  `int4` 2.05×.
- **Serving.** Its ratios between realizations are not the forward
  pass's at T = 16: `int8` is 0.53 of `f32` serving and 0.67 in the
  forward pass. B serves 1.32× faster than A, though their scalar
  forward passes are within 0.6 %. The split between compiling and
  computing is not measured.

The field's gaps, flips and perplexities, which RW9 sets beside these, are
§4's and §5's. llama.cpp is not timed, as the plan decided.

**Time against §7.** The plan gave about 6 h. The planned runs took 9 h
46 min, and B's serve 28 min more:
- `presk bench` times the five uniform rows in every process, about 4 h
  over the three rounds;
- A's six extra copies took about 2 h 50 min, their uniform rows
  included.

## 7. The figure's data (`figure.json`)

Every row of the result, A standing for the plan's seven solves (one
realization). The gaps, flips and bounds are §4's, on the held-out sample.
The perplexities are §5's. The times are §6's medians, at T = 16 on one
X925 core; llama.cpp and transformers are not timed.

| realization | weights (bytes) | gap g | flips | #{m < 2g} | δ certified | perplexity | forward, s | with lanes, s |
|---|---|---|---|---|---|---|---|---|
| `f64` | 9 886 515 200 | 3.666·10⁻¹³ | 0 | 0 | 0 | 24.7699 | 14.617 | 14.434 |
| `f32` | 4 943 257 600 | 4.277·10⁻⁴ | 0 | 0 | 0 | 24.7601 | 11.512 | 10.135 |
| `bf16` | 2 471 628 800 | 0.7060 | 32 | 743 | 0.726 | 24.8159 | 10.220 | — |
| `int8` absmax | 2 025 529 344 | 2.100 | 27 | 973 | 0.950 | 24.8638 | 7.751 | 5.431 |
| `int8` pow2 | 2 025 529 344 | 3.055 | 48 | 1 009 | 0.985 | 24.9791 | 7.767 | 5.425 |
| `int4` absmax | 1 538 990 080 | 19.65 | 624 | 1 024 | 1 | 75.5661 | 9.903 | 4.829 |
| `int4` pow2 | 1 538 990 080 | 19.94 | 814 | 1 024 | 1 | 596.2051 | 9.907 | 4.819 |
| `bf16-params-f32` (by hand) | 2 471 628 800 | 4.277·10⁻⁴ | 0 | 0 | 0 | 24.7601 | 9.977 | 7.733 |
| A (the plan's seven solves) | 2 472 034 304 | 3.666·10⁻¹³ | 0 | 0 | 0 | 24.7699 | 9.874 | 10.256 |
| B (the eighth solve) | 2 471 628 800 | 7.703·10⁻⁴ | 0 | 0 | 0 | 24.7699 | 9.821 | 7.775 |
| llama.cpp BF16 | 2 471 764 096 | 0.2569 | 0 | 391 | 0.382 | 24.7661 | — | — |
| llama.cpp Q8_0 | 1 313 251 456 | 1.810 | 28 | 959 | 0.937 | 24.8192 | — | — |
| llama.cpp Q4_K_M | 799 862 912 | 11.07 | 163 | 1 024 | 1 | 25.9183 | — | — |
| llama.cpp Q4_0 | 763 097 216 | 15.61 | 266 | 1 024 | 1 | 27.9202 | — | — |
| transformers `float32` | 4 943 257 600 | 2.401·10⁻⁴ | 0 | 0 | 0 | 24.7699 | — | — |
| transformers `bfloat16` | 2 471 628 800 | 0.5660 | 32 | 659 | 0.644 | 24.7883 | — | — |

ε\*(10⁻¹), ε\*(3·10⁻²) and ε\*(10⁻²) are 4.880·10⁻², 1.091·10⁻² and
3.807·10⁻³ (§2). Every row's perplexity carries `nll`'s own arithmetic
(§5): the rows of ours whose default is `f32` (`f32`, by hand, `int8`,
`int4`) about −3.9·10⁻⁴ in log from its `f32` log-softmax, the others
none. D189's addendum (§5) takes those six again with `nll`'s sites in
`f64`: `f32` and by hand 24.7699, `int8` 24.8736 and 24.9890, `int4`
75.6076 and 596.5223. This table keeps D189's rows of record.

## 8. The solved realizations in words

**A, the plan's seven solves.** The three absolute budgets ε\*(δ), the
two relative ones and the two variants all return the same file:
- the token table and the seven matrices of every block stored in `bf16`,
  which is the checkpoint's own type, so the weights are not rounded at
  all;
- the norms' gains and every value in `f64`.

It weighs 2.47 GB, a quarter of `f64`'s. Its outputs, held out and on the
quality text, are `f64`'s byte for byte. It runs the forward pass 1.48×
faster than `f64` and 1.17× faster than `f32` on one core, and serves
32 tokens 1.11× faster than `f32`.

Every budget asks for the same thing because of the menu, not the
budget. `bf16` costs nothing on a `bf16` checkpoint, and each matrix in
`int8` moves a logit by more than the largest budget's narrowing
threshold. Rule 2 (D169) keeps every value in `f64`, since the largest,
the logits, saves less than θ·B. A has the fewest bytes the solver could
give under rule 2. It is not the lightest realization within the flips
allowed (RW7, §4).

**B, the eighth solve** (ε\*(10⁻²) with `--min-gain 0`, the user's
addition). With rule 2 off, B stores:
- every parameter in `bf16`, the norms too;
- 32 values in `f32` and 12 in `bf16` (Q, K and V before the attention,
  its output, RoPE's input and output among them).

It weighs the realization by hand's 2.47 GB, 405 504 bytes under A. Its
gap is 7.70·10⁻⁴ held out, 4.9 times under its budget, with no flip, and
its perplexity is `f64`'s within 4.5·10⁻⁷ in log. Its forward pass takes
A's time without lanes and is 1.27× faster with them. It serves 32 tokens
1.32× faster than A, in the realization by hand's time.

## 9. The statements

| statement | verdict | decision | what was measured |
|---|---|---|---|
| OP0 | holds | D185, D191 | X1's harness, the quant and step tables and the tutorial on step 0's build and on the version's tree give `v0.17.0`'s outputs but for version and commit lines, the envelopes' dates and the tutorial's measurements |
| OP1 | holds | D185 | the five rows of record under `--sweep f64 --measure native`: the same realizations and measured gaps byte for byte |
| RW5 | holds | D187 | each of the seven solves inside its budget, under 24 h and 108 GB (2 h 06 min, 96.74 GB) |
| RW6 | holds at every δ | D188 | A changes no held-out decision; its gap 3.67·10⁻¹³ |
| RW7 | fails at every δ, as written | D188 | the lightest within δ·N flips is llama.cpp Q8_0 at 10⁻¹ and 3·10⁻² (the certificate bounds the worst logit, not the flips), the realization by hand and B at 10⁻² (the norms rule 2 keeps in `f64`) |
| RW8 | holds | D189 and its addendum | every perplexity within e^{±2g} of `f64`'s, every solved one's within e^{±2ε}, on all 22 rows; with `nll`'s sites in `f64` (the addendum), `f32` and the realization by hand 3.8 orders under their bound |
| RW9 | reported | D190 | the times of §6 |

## 10. The cost against the plan's §7

| step | the plan's falicon | measured | the difference |
|---|---|---|---|
| 0 | OP1 ≈ 4 h; the build and the GGUFs ≈ 1 h | OP1's five solves 50 min (beside the exact runs) | `--measure native` cut BERT's solve from 8 h 13 min to 32 min |
| 1 | 32 × 41 min ≈ 22 h | 21 h 58 min (40 min 51 s to 42 min 27 s a window) | as planned |
| 2 | ≈ 28 h | 23 h 50 min: the calibration 6 h 35 min, eight solves of 2 h 06 min to 2 h 19 min | the eighth solve is the user's |
| 3 | ≈ 3 h for 14 names | 3 h 38 min for 16 | the realization by hand and B are the user's |
| 4 | ≈ 8 h for 14 names | 9 h 17 min for 16 | A's seven copies took 3 h 52 min |
| 5 | ≈ 6 h | 10 h 14 min | `presk bench` times the five uniform rows in every process (≈ 4 h); B's serving is the user's |
| 6 | — | OP0 and the checks of record, 35 min alone; the archive | |

That is about 70 hours of falicon, counting each step's own runs, where
the plan gave about 75. Of these, 45 h 48 min were exact runs and solves
one at a time under 108 GB, where the plan gave 50. The plan started on
2026-11-05 and ended on 2026-11-08: four days elapsed, where the plan gave
9 to 11 with an agent session.
