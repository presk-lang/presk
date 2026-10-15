# ε from the margins: 461 windows, 118016 positions

Exact model: the reference at 128 bits; its accuracy 28612/118016 = 0.242442.
Largest |logit| 5.20979. Margins: smallest 1.1943e-05, median 4.5839e-01, largest 1.0835e+00; exactly 0 (ties): 0.

Quantiles of the margins:

| q | margin |
|---|---|
| 0.0001 | 8.5495e-05 |
| 0.001 | 1.0763e-03 |
| 0.01 | 1.1319e-02 |
| 0.1 | 9.5351e-02 |
| 0.5 | 4.5839e-01 |

The fraction of positions whose margin is below 2ε:

| ε | m < 2ε | fraction |
|---|---|---|
| 1e-01 | 38242 | 3.240e-01 |
| 1e-02 | 2150 | 1.822e-02 |
| 1e-03 | 202 | 1.712e-03 |
| 1e-04 | 21 | 1.779e-04 |
| 1e-05 | 1 | 8.473e-06 |
| 1e-06 | 0 | 0.000e+00 |
| 1e-07 | 0 | 0.000e+00 |

ε*(δ), α = 0.05: the largest ε with at most k margins below 2ε; uncorrected k = ⌊δn⌋, corrected k the largest with P[Bin(n, δ) ≤ k] ≤ α (Clopper–Pearson, one-sided).

| δ | δn | k uncorrected | ε* uncorrected | k corrected | ε* corrected | ratio |
|---|---|---|---|---|---|---|
| 1e-02 | 1180.2 | 1180 | 5.6597e-03 | 1123 | 5.3615e-03 | 1.06 |
| 1e-03 | 118.0 | 118 | 5.3814e-04 | 99 | 4.6699e-04 | 1.15 |
| 1e-04 | 11.8 | 11 | 4.2747e-05 | 5 | 2.4423e-05 | 1.75 |

Per realization. The gap to the exact loss is the foil's, relative, on two held-out windows; the gap on the logits the largest |ŷ − y| over every position and logit, relative to the largest |y|, and its mean; the flips are `decisions`' natively against the exact argmax; the bound is #{m < 2ε} at that largest gap, and 'per position' #{m(x) < 2ε(x)} at each position's own largest gap:

| realization | gap to the exact loss | gap on the logits | relative | mean | flips | at ties | bound #{m < 2ε} | bound / flips | per position | cert. 1e-02 | cert. 1e-03 | cert. 1e-04 | accuracy | vs exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| f64 | 3.18e-16 | 5.894e-15 | 1.13e-15 | 5.57e-16 | 0 | 0 | 0 | — | 0 | yes | yes | yes | 0.242442 | +0 |
| f32 | 4.68e-8 | 3.115e-06 | 5.98e-07 | 2.96e-07 | 0 | 0 | 0 | — | 0 | yes | yes | yes | 0.242442 | +0 |
| bf16 | 1.71e-3 | 5.033e-02 | 9.66e-03 | 6.07e-03 | 480 | 0 | 13661 | 28.5 | 4916 | no | no | no | 0.242425 | -2 |
| int8, f32 elsewhere | 3.90e-5 | 7.808e-02 | 1.50e-02 | 2.16e-03 | 161 | 0 | 33682 | 209.2 | 1488 | no | no | no | 0.242493 | +6 |
| int8 pow2 | 2.97e-5 | 1.149e-01 | 2.21e-02 | 3.25e-03 | 168 | 0 | 41830 | 249.0 | 2061 | no | no | no | 0.242348 | -11 |
| int4, f32 elsewhere | 3.99e-4 | 1.240e+00 | 2.38e-01 | 4.32e-02 | 2968 | 0 | 118016 | 39.8 | 45620 | no | no | no | 0.241289 | -136 |
| int4 pow2 | 8.66e-4 | 2.083e+00 | 4.00e-01 | 6.89e-02 | 3488 | 0 | 118016 | 33.8 | 59989 | no | no | no | 0.243416 | +115 |
| solved at 1e-1 | 4.23e-6 | 3.180e-02 | 6.10e-03 | 1.38e-03 | 69 | 0 | 6933 | 100.5 | 581 | no | no | no | 0.242442 | +0 |
| solved at 1e-2 | 1.23e-5 | 1.422e-02 | 2.73e-03 | 5.11e-04 | 36 | 0 | 3090 | 85.8 | 254 | no | no | no | 0.242484 | +5 |
| solved at 1e-3 | 2.50e-7 | 3.345e-04 | 6.42e-05 | 2.86e-05 | 1 | 0 | 70 | 70.0 | 6 | yes | yes | no | 0.242442 | +0 |
| solved at 1e-4 | 5.29e-8 | 7.509e-05 | 1.44e-05 | 3.00e-06 | 0 | 0 | 18 | ∞ | 0 | yes | yes | no | 0.242442 | +0 |
| solved at 1e-5 | 1.06e-8 | 1.562e-06 | 3.00e-07 | 8.88e-08 | 0 | 0 | 0 | — | 0 | yes | yes | yes | 0.242442 | +0 |
| solved at 1e-6 | 4.39e-9 | 7.285e-07 | 1.40e-07 | 2.39e-08 | 0 | 0 | 0 | — | 0 | yes | yes | yes | 0.242442 | +0 |
| solved at 1e-7 | 4.84e-11 | 6.712e-08 | 1.29e-08 | 5.07e-09 | 0 | 0 | 0 | — | 0 | yes | yes | yes | 0.242442 | +0 |

Checks: every flip at a position whose margin is at most twice that position's gap: yes; `decisions` against the argmax of the realization's printed logits: none differ.

Out of sample: ε* (corrected) from one half of the windows, the margins of the other half below 2ε*:

| ε* from | δ | ε* | other half: m < 2ε* | fraction | ≤ δ? |
|---|---|---|---|---|---|
| even windows (59136) | 1e-02 | 5.1529e-03 | 539 of 58880 | 9.15e-03 | yes |
| even windows (59136) | 1e-03 | 4.1941e-04 | 44 of 58880 | 7.47e-04 | yes |
| even windows (59136) | 1e-04 | 1.7995e-05 | 1 of 58880 | 1.70e-05 | yes |
| odd windows (58880) | 1e-02 | 5.2413e-03 | 559 of 59136 | 9.45e-03 | yes |
| odd windows (58880) | 1e-03 | 4.5514e-04 | 49 of 59136 | 8.29e-04 | yes |
| odd windows (58880) | 1e-04 | 2.2127e-05 | 2 of 59136 | 3.38e-05 | yes |

The statement (δ = 10⁻³, α = 0.05): the curve's realizations certified at 10⁻³: solved at 1e-3, solved at 1e-4, solved at 1e-5, solved at 1e-6, solved at 1e-7; each flips at most δn = 118.0: yes; every curve realization's flips at most its bound: yes (every row's: yes). The statement holds.
