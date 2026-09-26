# RA4: static ranking against measurement

Days: `out/ra4/day1` and `out/ra4/day2`. A close pair differs in one decision; it is *distinguished* when its medians differ by more than twice the larger MAD on both days, with the same sign (question 65). *Right*: the model orders it as the measurement does; a pair it predicts equal (*tied*) is not ranked, so not right (question 70). *Decided*: its predicted ratio lies outside 1 ± u, u = 0.132. The model's pick is `presk plan`'s: the best prediction, a tie broken by the space's order. Times are the mean of the two days' medians. *vs 0.6.0*: 0.6.0's default layout, its split left to the executor, over the best point measured.

| nest | config | operands | type | points | close pairs | distinguished | right | right, decided | undecided | tied | Spearman | top-1 regret | top-k regret (k) | vs 0.6.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Linear.apply/%3` | B=1,T=256 | [1, 256, 384] [384, 384] | f32 | 1029 | 8454 | 6952 | 89.4 % | 99.7 % | 40.7 % | 0.0 % | 0.983 | 1.02 | 1.00 (20) | 12.79× |
| `Linear.apply/%3` | B=1,T=256 | [1, 256, 384] [384, 384] | f64 | 1029 | 8454 | 6984 | 87.3 % | 97.2 % | 44.0 % | 0.0 % | 0.966 | 1.00 | 1.00 (34) | 18.87× |
| `Linear.apply/%3` | D=1024,H=16,B=1,T=384 | [1, 384, 1024] [1024, 1024] | f32 | 41 | 20 | 17 | 64.7 % | 100.0 % | 41.2 % | 0.0 % | 0.874 | 1.00 | 1.00 (1) | 34.58× |
| `Linear.apply/%3` | B=16,T=256 | [16, 256, 384] [384, 384] | f32 | 1563 | 14705 | 13598 | 87.4 % | 99.6 % | 40.6 % | 0.0 % | 0.981 | 1.07 | 1.00 (9) | 2.56× |
| `Linear.apply/%3` | B=16,T=256 | [16, 256, 384] [384, 384] | f64 | 1563 | 14705 | 13508 | 86.2 % | 97.0 % | 41.9 % | 0.0 % | 0.961 | 1.07 | 1.00 (26) | 2.67× |
| `Linear.apply/%5` | B=1,T=256 | [1, 256, 384] [384] | f32 | 5 | 10 | 8 | 50.0 % | 100.0 % | 50.0 % | 0.0 % | 0.200 | 1.00 | 1.00 (1) | 1.01× |
| `Linear.apply/%5` | B=1,T=256 | [1, 256, 384] [384] | f64 | 5 | 10 | 8 | 50.0 % | 100.0 % | 50.0 % | 0.0 % | 0.000 | 1.00 | 1.00 (1) | 1.06× |
| `Linear.apply/%5` | D=1024,H=16,B=1,T=384 | [1, 384, 1024] [1024] | f32 | 5 | 10 | 6 | 0.0 % | 0.0 % | 66.7 % | 0.0 % | -0.500 | 1.24 | 1.24 (1) | 1.22× |
| `Linear.apply/%5` | B=16,T=256 | [16, 256, 384] [384] | f32 | 7 | 21 | 15 | 66.7 % | 85.7 % | 53.3 % | 0.0 % | 0.393 | 1.05 | 1.03 (3) | 2.42× |
| `Linear.apply/%5` | B=16,T=256 | [16, 256, 384] [384] | f64 | 7 | 21 | 12 | 66.7 % | 71.4 % | 41.7 % | 0.0 % | 0.286 | 1.00 | 1.00 (2) | 1.81× |
| `SelfAttention.apply/%12` | B=1,T=256 | [1, 256, 384] | f32 | 5 | 10 | 7 | 57.1 % | 100.0 % | 42.9 % | 0.0 % | 0.300 | 1.00 | 1.00 (1) | 1.02× |
| `SelfAttention.apply/%12` | B=1,T=256 | [1, 256, 384] | f64 | 5 | 10 | 7 | 57.1 % | 100.0 % | 42.9 % | 0.0 % | 0.300 | 1.00 | 1.00 (1) | 1.02× |
| `SelfAttention.apply/%12` | D=1024,H=16,B=1,T=384 | [1, 384, 1024] | f32 | 5 | 10 | 9 | 11.1 % | 0.0 % | 55.6 % | 0.0 % | -0.800 | 1.71 | 1.71 (1) | 1.70× |
| `SelfAttention.apply/%12` | B=16,T=256 | [16, 256, 384] | f32 | 7 | 21 | 16 | 100.0 % | 100.0 % | 25.0 % | 0.0 % | 0.964 | 1.00 | 1.00 (4) | 3.16× |
| `SelfAttention.apply/%12` | B=16,T=256 | [16, 256, 384] | f64 | 7 | 21 | 10 | 90.0 % | 100.0 % | 20.0 % | 0.0 % | 0.536 | 1.24 | 1.00 (4) | 2.69× |
| `SelfAttention.apply/%13` | D=1024,H=16,B=1,T=384 | [1, 16, 384, 64] [1, 16, 384, 64] | f32 | 41 | 17 | 13 | 84.6 % | 100.0 % | 30.8 % | 0.0 % | 0.982 | 1.00 | 1.00 (1) | 64.17× |
| `SelfAttention.apply/%13` | B=1,T=256 | [1, 6, 256, 64] [1, 6, 256, 64] | f32 | 1049 | 8942 | 7287 | 89.9 % | 99.8 % | 45.1 % | 0.0 % | 0.973 | 1.08 | 1.08 (9) | 47.41× |
| `SelfAttention.apply/%13` | B=1,T=256 | [1, 6, 256, 64] [1, 6, 256, 64] | f64 | 1049 | 8942 | 7450 | 88.5 % | 98.3 % | 47.9 % | 0.1 % | 0.967 | 1.03 | 1.00 (21) | 30.22× |
| `SelfAttention.apply/%13` | B=16,T=256 | [16, 6, 256, 64] [16, 6, 256, 64] | f32 | 1515 | 14521 | 12925 | 89.3 % | 99.8 % | 42.8 % | 0.2 % | 0.973 | 1.00 | 1.00 (13) | 9.19× |
| `SelfAttention.apply/%13` | B=16,T=256 | [16, 6, 256, 64] [16, 6, 256, 64] | f64 | 1515 | 14521 | 9321 | 87.7 % | 91.2 % | 31.5 % | 0.1 % | 0.757 | 1.05 | 1.01 (35) | 4.23× |
| `SelfAttention.apply/%16` | D=1024,H=16,B=1,T=384 | [] | f32 | 1 | 0 | 0 | — | — | — | — | 0.000 | 1.00 | 1.00 (1) | 1.12× |
| `SelfAttention.apply/%16` | B=1,T=256 | [] | f64 | 1 | 0 | 0 | — | — | — | — | 0.000 | 1.00 | 1.00 (1) | 1.12× |
| `SelfAttention.apply/%19` | D=1024,H=16,B=1,T=384 | [1, 16, 384, 384] [] [384, 384] | f32 | 5 | 10 | 9 | 100.0 % | 100.0 % | 0.0 % | 0.0 % | 1.000 | 1.00 | 1.00 (1) | 3.97× |
| `SelfAttention.apply/%19` | B=1,T=256 | [1, 6, 256, 256] [] [256, 256] | f32 | 5 | 10 | 8 | 62.5 % | 66.7 % | 62.5 % | 0.0 % | 0.200 | 1.30 | 1.00 (3) | 1.62× |
| `SelfAttention.apply/%19` | B=1,T=256 | [1, 6, 256, 256] [] [256, 256] | f64 | 5 | 10 | 8 | 62.5 % | 100.0 % | 62.5 % | 0.0 % | 0.500 | 1.32 | 1.05 (2) | 1.66× |
| `SelfAttention.apply/%19` | B=16,T=256 | [16, 6, 256, 256] [] [256, 256] | f32 | 7 | 21 | 20 | 100.0 % | 100.0 % | 15.0 % | 0.0 % | 1.000 | 1.00 | 1.00 (1) | 1.02× |
| `SelfAttention.apply/%19` | B=16,T=256 | [16, 6, 256, 256] [] [256, 256] | f64 | 7 | 21 | 5 | 100.0 % | 100.0 % | 40.0 % | 0.0 % | 0.857 | 1.29 | 1.29 (1) | 1.00× |
| `SelfAttention.apply/%20` | D=1024,H=16,B=1,T=384 | [1, 16, 384, 384] | f32 | 10 | 25 | 25 | 100.0 % | 100.0 % | 0.0 % | 0.0 % | 1.000 | 1.00 | 1.00 (1) | 10.97× |
| `SelfAttention.apply/%20` | B=1,T=256 | [1, 6, 256, 256] | f32 | 10 | 25 | 25 | 100.0 % | 100.0 % | 8.0 % | 0.0 % | 1.000 | 1.00 | 1.00 (1) | 8.35× |
| `SelfAttention.apply/%20` | B=1,T=256 | [1, 6, 256, 256] | f64 | 10 | 25 | 25 | 100.0 % | 100.0 % | 8.0 % | 0.0 % | 1.000 | 1.00 | 1.00 (1) | 8.89× |
| `SelfAttention.apply/%20` | B=16,T=256 | [16, 6, 256, 256] | f32 | 14 | 49 | 49 | 100.0 % | 100.0 % | 0.0 % | 0.0 % | 0.991 | 1.00 | 1.00 (1) | 1.53× |
| `SelfAttention.apply/%20` | B=16,T=256 | [16, 6, 256, 256] | f64 | 14 | 49 | 45 | 100.0 % | 100.0 % | 4.4 % | 0.0 % | 0.978 | 1.00 | 1.00 (1) | 1.30× |
| `SelfAttention.apply/%22` | D=1024,H=16,B=1,T=384 | [1, 16, 384, 384] [1, 16, 384, 384] | f32 | 5 | 10 | 5 | 100.0 % | 100.0 % | 0.0 % | 0.0 % | 0.600 | 1.01 | 1.01 (1) | 2.90× |
| `SelfAttention.apply/%22` | B=1,T=256 | [1, 6, 256, 256] [1, 6, 256, 256] | f32 | 5 | 10 | 6 | 33.3 % | 50.0 % | 33.3 % | 0.0 % | -0.600 | 1.32 | 1.32 (1) | 1.38× |
| `SelfAttention.apply/%22` | B=1,T=256 | [1, 6, 256, 256] [1, 6, 256, 256] | f64 | 5 | 10 | 6 | 33.3 % | 100.0 % | 66.7 % | 0.0 % | -0.500 | 1.27 | 1.00 (4) | 1.26× |
| `SelfAttention.apply/%22` | B=16,T=256 | [16, 6, 256, 256] [16, 6, 256, 256] | f32 | 7 | 21 | 16 | 93.8 % | 92.9 % | 12.5 % | 0.0 % | 0.821 | 1.11 | 1.11 (1) | 0.99× |
| `SelfAttention.apply/%22` | B=16,T=256 | [16, 6, 256, 256] [16, 6, 256, 256] | f64 | 7 | 21 | 10 | 90.0 % | 100.0 % | 20.0 % | 0.0 % | 0.857 | 1.26 | 1.26 (2) | 1.01× |
| `SelfAttention.apply/%23` | D=1024,H=16,B=1,T=384 | [1, 16, 384, 384] [1, 16, 384, 64] | f32 | 41 | 18 | 16 | 81.2 % | 100.0 % | 43.8 % | 0.0 % | 0.977 | 1.00 | 1.00 (2) | 118.65× |
| `SelfAttention.apply/%23` | B=1,T=256 | [1, 6, 256, 256] [1, 6, 256, 64] | f32 | 1183 | 10473 | 8649 | 90.7 % | 99.7 % | 38.7 % | 0.0 % | 0.981 | 1.02 | 1.00 (9) | 77.47× |
| `SelfAttention.apply/%23` | B=1,T=256 | [1, 6, 256, 256] [1, 6, 256, 64] | f64 | 1183 | 10473 | 8973 | 89.7 % | 97.5 % | 43.4 % | 0.0 % | 0.970 | 1.03 | 1.01 (20) | 52.32× |
| `SelfAttention.apply/%23` | B=16,T=256 | [16, 6, 256, 256] [16, 6, 256, 64] | f32 | 1717 | 17032 | 14868 | 89.6 % | 99.7 % | 36.6 % | 0.0 % | 0.979 | 1.01 | 1.00 (19) | 16.50× |
| `SelfAttention.apply/%23` | B=16,T=256 | [16, 6, 256, 256] [16, 6, 256, 64] | f64 | 1717 | 17032 | 15520 | 89.4 % | 96.9 % | 41.0 % | 0.0 % | 0.966 | 1.11 | 1.00 (33) | 8.71× |
| `SelfAttention.apply/%24` | D=1024,H=16,B=1,T=384 | [1, 16, 384, 64] | f32 | 3 | 3 | 3 | 0.0 % | 0.0 % | 33.3 % | 0.0 % | -1.000 | 1.58 | 1.58 (1) | 1.61× |
| `SelfAttention.apply/%24` | B=1,T=256 | [1, 6, 256, 64] | f32 | 3 | 3 | 3 | 66.7 % | 100.0 % | 33.3 % | 0.0 % | 0.500 | 1.00 | 1.00 (1) | 1.04× |
| `SelfAttention.apply/%24` | B=1,T=256 | [1, 6, 256, 64] | f64 | 3 | 3 | 3 | 66.7 % | 100.0 % | 33.3 % | 0.0 % | 0.500 | 1.00 | 1.00 (1) | 1.06× |
| `SelfAttention.apply/%24` | B=16,T=256 | [16, 6, 256, 64] | f32 | 5 | 10 | 6 | 100.0 % | 100.0 % | 16.7 % | 0.0 % | 1.000 | 1.00 | 1.00 (3) | 3.01× |
| `SelfAttention.apply/%24` | B=16,T=256 | [16, 6, 256, 64] | f64 | 5 | 10 | 5 | 100.0 % | 100.0 % | 20.0 % | 0.0 % | 0.900 | 1.00 | 1.00 (3) | 2.44× |
| `SelfAttention.apply/%6` | B=1,T=256 | [1, 256, 384] | f32 | 5 | 10 | 7 | 57.1 % | 100.0 % | 42.9 % | 0.0 % | 0.300 | 1.00 | 1.00 (1) | 0.99× |
| `SelfAttention.apply/%6` | B=1,T=256 | [1, 256, 384] | f64 | 5 | 10 | 8 | 50.0 % | 100.0 % | 50.0 % | 0.0 % | 0.100 | 1.00 | 1.00 (1) | 1.02× |
| `SelfAttention.apply/%6` | D=1024,H=16,B=1,T=384 | [1, 384, 1024] | f32 | 5 | 10 | 8 | 0.0 % | 0.0 % | 50.0 % | 0.0 % | -0.900 | 1.66 | 1.66 (1) | 1.65× |
| `SelfAttention.apply/%6` | B=16,T=256 | [16, 256, 384] | f32 | 7 | 21 | 15 | 100.0 % | 100.0 % | 26.7 % | 0.0 % | 0.964 | 1.03 | 1.00 (4) | 3.13× |
| `SelfAttention.apply/%6` | B=16,T=256 | [16, 256, 384] | f64 | 7 | 21 | 11 | 100.0 % | 100.0 % | 18.2 % | 0.0 % | 0.857 | 1.10 | 1.00 (4) | 2.66× |
| `SelfAttention.apply/%9` | B=1,T=256 | [1, 256, 384] | f32 | 5 | 10 | 8 | 50.0 % | 100.0 % | 50.0 % | 0.0 % | 0.100 | 1.00 | 1.00 (1) | 1.02× |
| `SelfAttention.apply/%9` | B=1,T=256 | [1, 256, 384] | f64 | 5 | 10 | 7 | 57.1 % | 100.0 % | 42.9 % | 0.0 % | 0.300 | 1.00 | 1.00 (1) | 1.02× |
| `SelfAttention.apply/%9` | D=1024,H=16,B=1,T=384 | [1, 384, 1024] | f32 | 5 | 10 | 9 | 11.1 % | 0.0 % | 55.6 % | 0.0 % | -0.900 | 1.68 | 1.68 (1) | 1.68× |
| `SelfAttention.apply/%9` | B=16,T=256 | [16, 256, 384] | f32 | 7 | 21 | 17 | 100.0 % | 100.0 % | 29.4 % | 0.0 % | 0.964 | 1.04 | 1.00 (4) | 3.15× |
| `SelfAttention.apply/%9` | B=16,T=256 | [16, 256, 384] | f64 | 7 | 21 | 11 | 90.9 % | 100.0 % | 27.3 % | 0.0 % | 0.643 | 1.33 | 1.00 (4) | 2.89× |
| `gemm/%2` | N=1024 | [1024, 1024] [1024, 1024] | f32 | 1333 | 11566 | 10745 | 87.1 % | 98.8 % | 39.4 % | 0.4 % | 0.962 | 1.03 | 1.00 (10) | 5.49× |
| **all** | | | | | 160509 | 137307 | 88.5 % | 98.0 % | 40.7 % | 0.1 % | | | | |

## Statements (plan 0.7.0 §3.7)

| | statement | measured | verdict |
|---|---|---|---|
| L | every point bit-identical to point 0, natively and on the kernel interpreter | 17879 of 17879 points of day 1 checked on the interpreter; 0 outputs differing, natively or on the interpreter, in any sweep | **holds** |
| M | more than 90 % of the distinguished close pairs ranked right, each `ein` of `SelfAttention.apply` at B=16,T=256, f32 | `SelfAttention.apply/%13` 89.3 % of 12925; `SelfAttention.apply/%23` 89.6 % of 14868 | **fails** |
| N | choosing statically at least 10³× cheaper than the exhaustive sweep of the same points, `SelfAttention.apply` | B=16,T=256 f32 717×; B=16,T=256 f64 607×; B=1,T=256 f32 89×; B=1,T=256 f64 76×; B=16,T=256 f32 697×; B=16,T=256 f64 582×; B=1,T=256 f32 87×; B=1,T=256 f64 75× | **fails** |
| O | the model's pick within 1.25× of the best measured, every contraction nest of `SelfAttention.apply` at B=16,T=256, f32 | `Linear.apply/%3` 1.07; `SelfAttention.apply/%13` 1.00; `SelfAttention.apply/%23` 1.01 | **holds** |
| P | the best point of the GEMM space at least 3× the 0.6.0 default, `gemm` at N=1024, f32 | 5.49× | **holds** |

## Per decision class

| class | close pairs | distinguished | right | right, decided | undecided | tied |
|---|---|---|---|---|---|---|
| kblock | 5136 | 2998 | 65.9 % | 16.7 % | 99.8 % | 0.0 % |
| lanes | 7401 | 7289 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| order | 31154 | 17565 | 67.0 % | 92.3 % | 82.8 % | 0.4 % |
| pack | 1761 | 1148 | 57.8 % | — | 100.0 % | 0.0 % |
| passes | 29 | 27 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| split | 40499 | 39144 | 97.7 % | 98.5 % | 5.0 % | 0.0 % |
| tile | 74529 | 69136 | 89.1 % | 97.6 % | 50.9 % | 0.0 % |

## Per nest and class

| nest | config | type | class | close pairs | distinguished | right | right, decided | undecided | tied |
|---|---|---|---|---|---|---|---|---|---|
| `Linear.apply/%3` | B=1,T=256 | f32 | kblock | 418 | 158 | 50.0 % | 0.0 % | 98.7 % | 0.0 % |
| `Linear.apply/%3` | B=1,T=256 | f32 | lanes | 486 | 484 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `Linear.apply/%3` | B=1,T=256 | f32 | order | 1577 | 847 | 63.6 % | 97.0 % | 92.2 % | 0.0 % |
| `Linear.apply/%3` | B=1,T=256 | f32 | split | 1599 | 1444 | 99.9 % | 100.0 % | 5.0 % | 0.0 % |
| `Linear.apply/%3` | B=1,T=256 | f32 | tile | 4374 | 4019 | 91.3 % | 99.7 % | 45.3 % | 0.0 % |
| `Linear.apply/%3` | B=1,T=256 | f64 | kblock | 418 | 138 | 37.7 % | — | 100.0 % | 0.0 % |
| `Linear.apply/%3` | B=1,T=256 | f64 | lanes | 486 | 478 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `Linear.apply/%3` | B=1,T=256 | f64 | order | 1577 | 900 | 65.1 % | 92.2 % | 84.3 % | 0.0 % |
| `Linear.apply/%3` | B=1,T=256 | f64 | split | 1599 | 1467 | 98.0 % | 99.9 % | 6.2 % | 0.0 % |
| `Linear.apply/%3` | B=1,T=256 | f64 | tile | 4374 | 4001 | 88.5 % | 94.9 % | 52.2 % | 0.0 % |
| `Linear.apply/%3` | D=1024,H=16,B=1,T=384 | f32 | kblock | 3 | 0 | — | — | — | — |
| `Linear.apply/%3` | D=1024,H=16,B=1,T=384 | f32 | lanes | 1 | 1 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `Linear.apply/%3` | D=1024,H=16,B=1,T=384 | f32 | order | 3 | 3 | 33.3 % | — | 100.0 % | 0.0 % |
| `Linear.apply/%3` | D=1024,H=16,B=1,T=384 | f32 | split | 6 | 6 | 50.0 % | 100.0 % | 50.0 % | 0.0 % |
| `Linear.apply/%3` | D=1024,H=16,B=1,T=384 | f32 | tile | 7 | 7 | 85.7 % | 100.0 % | 14.3 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f32 | kblock | 646 | 508 | 53.1 % | 0.0 % | 99.4 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f32 | lanes | 738 | 738 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f32 | order | 2755 | 1971 | 59.8 % | 96.6 % | 92.6 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f32 | split | 3924 | 3837 | 99.2 % | 99.6 % | 6.5 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f32 | tile | 6642 | 6544 | 90.0 % | 99.6 % | 45.0 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f64 | kblock | 646 | 431 | 42.0 % | — | 100.0 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f64 | lanes | 738 | 738 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f64 | order | 2755 | 2051 | 64.3 % | 91.6 % | 80.9 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f64 | split | 3924 | 3820 | 98.7 % | 100.0 % | 6.9 % | 0.0 % |
| `Linear.apply/%3` | B=16,T=256 | f64 | tile | 6642 | 6468 | 87.1 % | 93.5 % | 51.1 % | 0.0 % |
| `Linear.apply/%5` | B=1,T=256 | f32 | split | 10 | 8 | 50.0 % | 100.0 % | 50.0 % | 0.0 % |
| `Linear.apply/%5` | B=1,T=256 | f64 | split | 10 | 8 | 50.0 % | 100.0 % | 50.0 % | 0.0 % |
| `Linear.apply/%5` | D=1024,H=16,B=1,T=384 | f32 | split | 10 | 6 | 0.0 % | 0.0 % | 66.7 % | 0.0 % |
| `Linear.apply/%5` | B=16,T=256 | f32 | split | 21 | 15 | 66.7 % | 85.7 % | 53.3 % | 0.0 % |
| `Linear.apply/%5` | B=16,T=256 | f64 | split | 21 | 12 | 66.7 % | 71.4 % | 41.7 % | 0.0 % |
| `SelfAttention.apply/%12` | B=1,T=256 | f32 | split | 10 | 7 | 57.1 % | 100.0 % | 42.9 % | 0.0 % |
| `SelfAttention.apply/%12` | B=1,T=256 | f64 | split | 10 | 7 | 57.1 % | 100.0 % | 42.9 % | 0.0 % |
| `SelfAttention.apply/%12` | D=1024,H=16,B=1,T=384 | f32 | split | 10 | 9 | 11.1 % | 0.0 % | 55.6 % | 0.0 % |
| `SelfAttention.apply/%12` | B=16,T=256 | f32 | split | 21 | 16 | 100.0 % | 100.0 % | 25.0 % | 0.0 % |
| `SelfAttention.apply/%12` | B=16,T=256 | f64 | split | 21 | 10 | 90.0 % | 100.0 % | 20.0 % | 0.0 % |
| `SelfAttention.apply/%13` | D=1024,H=16,B=1,T=384 | f32 | lanes | 1 | 1 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%13` | D=1024,H=16,B=1,T=384 | f32 | order | 3 | 0 | — | — | — | — |
| `SelfAttention.apply/%13` | D=1024,H=16,B=1,T=384 | f32 | pack | 1 | 1 | 100.0 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%13` | D=1024,H=16,B=1,T=384 | f32 | split | 2 | 2 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%13` | D=1024,H=16,B=1,T=384 | f32 | tile | 10 | 9 | 77.8 % | 100.0 % | 33.3 % | 0.0 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f32 | lanes | 324 | 323 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f32 | order | 1798 | 619 | 76.3 % | 100.0 % | 87.9 % | 0.3 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f32 | pack | 360 | 279 | 50.5 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f32 | split | 1924 | 1885 | 99.9 % | 100.0 % | 2.1 % | 0.0 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f32 | tile | 4536 | 4181 | 89.3 % | 99.7 % | 58.1 % | 0.0 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f64 | lanes | 324 | 323 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f64 | order | 1798 | 805 | 60.7 % | 74.4 % | 90.3 % | 0.6 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f64 | pack | 360 | 280 | 63.2 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f64 | split | 1924 | 1906 | 99.9 % | 100.0 % | 1.5 % | 0.0 % |
| `SelfAttention.apply/%13` | B=1,T=256 | f64 | tile | 4536 | 4136 | 89.5 % | 97.2 % | 61.3 % | 0.0 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f32 | lanes | 468 | 468 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f32 | order | 2726 | 1433 | 70.3 % | 100.0 % | 92.7 % | 1.5 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f32 | pack | 520 | 468 | 56.4 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f32 | split | 4255 | 4202 | 99.6 % | 99.9 % | 3.8 % | 0.0 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f32 | tile | 6552 | 6354 | 88.5 % | 99.6 % | 56.3 % | 0.0 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f64 | lanes | 468 | 376 | 99.2 % | 99.2 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f64 | order | 2726 | 453 | 67.5 % | 63.3 % | 93.4 % | 1.5 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f64 | pack | 520 | 120 | 67.5 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f64 | split | 4255 | 4072 | 87.3 % | 88.1 % | 3.7 % | 0.0 % |
| `SelfAttention.apply/%13` | B=16,T=256 | f64 | tile | 6552 | 4300 | 89.8 % | 96.0 % | 52.1 % | 0.0 % |
| `SelfAttention.apply/%19` | D=1024,H=16,B=1,T=384 | f32 | split | 10 | 9 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%19` | B=1,T=256 | f32 | split | 10 | 8 | 62.5 % | 66.7 % | 62.5 % | 0.0 % |
| `SelfAttention.apply/%19` | B=1,T=256 | f64 | split | 10 | 8 | 62.5 % | 100.0 % | 62.5 % | 0.0 % |
| `SelfAttention.apply/%19` | B=16,T=256 | f32 | split | 21 | 20 | 100.0 % | 100.0 % | 15.0 % | 0.0 % |
| `SelfAttention.apply/%19` | B=16,T=256 | f64 | split | 21 | 5 | 100.0 % | 100.0 % | 40.0 % | 0.0 % |
| `SelfAttention.apply/%20` | D=1024,H=16,B=1,T=384 | f32 | passes | 5 | 5 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%20` | D=1024,H=16,B=1,T=384 | f32 | split | 20 | 20 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%20` | B=1,T=256 | f32 | passes | 5 | 5 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%20` | B=1,T=256 | f32 | split | 20 | 20 | 100.0 % | 100.0 % | 10.0 % | 0.0 % |
| `SelfAttention.apply/%20` | B=1,T=256 | f64 | passes | 5 | 5 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%20` | B=1,T=256 | f64 | split | 20 | 20 | 100.0 % | 100.0 % | 10.0 % | 0.0 % |
| `SelfAttention.apply/%20` | B=16,T=256 | f32 | passes | 7 | 7 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%20` | B=16,T=256 | f32 | split | 42 | 42 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%20` | B=16,T=256 | f64 | passes | 7 | 5 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%20` | B=16,T=256 | f64 | split | 42 | 40 | 100.0 % | 100.0 % | 5.0 % | 0.0 % |
| `SelfAttention.apply/%22` | D=1024,H=16,B=1,T=384 | f32 | split | 10 | 5 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%22` | B=1,T=256 | f32 | split | 10 | 6 | 33.3 % | 50.0 % | 33.3 % | 0.0 % |
| `SelfAttention.apply/%22` | B=1,T=256 | f64 | split | 10 | 6 | 33.3 % | 100.0 % | 66.7 % | 0.0 % |
| `SelfAttention.apply/%22` | B=16,T=256 | f32 | split | 21 | 16 | 93.8 % | 92.9 % | 12.5 % | 0.0 % |
| `SelfAttention.apply/%22` | B=16,T=256 | f64 | split | 21 | 10 | 90.0 % | 100.0 % | 20.0 % | 0.0 % |
| `SelfAttention.apply/%23` | D=1024,H=16,B=1,T=384 | f32 | kblock | 3 | 2 | 100.0 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%23` | D=1024,H=16,B=1,T=384 | f32 | lanes | 1 | 1 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%23` | D=1024,H=16,B=1,T=384 | f32 | order | 3 | 2 | 0.0 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%23` | D=1024,H=16,B=1,T=384 | f32 | split | 1 | 1 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%23` | D=1024,H=16,B=1,T=384 | f32 | tile | 10 | 10 | 90.0 % | 100.0 % | 30.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f32 | kblock | 494 | 220 | 91.8 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f32 | lanes | 558 | 557 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f32 | order | 2261 | 1030 | 68.6 % | 99.0 % | 71.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f32 | split | 2138 | 2089 | 100.0 % | 100.0 % | 3.9 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f32 | tile | 5022 | 4753 | 90.2 % | 99.4 % | 48.6 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f64 | kblock | 494 | 232 | 84.9 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f64 | lanes | 558 | 554 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f64 | order | 2261 | 1355 | 75.0 % | 94.7 % | 72.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f64 | split | 2138 | 2101 | 100.0 % | 100.0 % | 3.8 % | 0.0 % |
| `SelfAttention.apply/%23` | B=1,T=256 | f64 | tile | 5022 | 4731 | 88.4 % | 94.9 % | 55.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f32 | kblock | 722 | 500 | 90.4 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f32 | lanes | 810 | 810 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f32 | order | 3439 | 1810 | 61.3 % | 100.0 % | 82.2 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f32 | split | 4771 | 4636 | 99.4 % | 99.8 % | 4.1 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f32 | tile | 7290 | 7112 | 89.1 % | 99.5 % | 45.9 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f64 | kblock | 722 | 455 | 80.0 % | — | 100.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f64 | lanes | 810 | 807 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f64 | order | 3439 | 2599 | 74.8 % | 84.9 % | 76.1 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f64 | split | 4771 | 4651 | 99.5 % | 99.9 % | 4.2 % | 0.0 % |
| `SelfAttention.apply/%23` | B=16,T=256 | f64 | tile | 7290 | 7008 | 87.5 % | 94.4 % | 53.3 % | 0.0 % |
| `SelfAttention.apply/%24` | D=1024,H=16,B=1,T=384 | f32 | split | 3 | 3 | 0.0 % | 0.0 % | 33.3 % | 0.0 % |
| `SelfAttention.apply/%24` | B=1,T=256 | f32 | split | 3 | 3 | 66.7 % | 100.0 % | 33.3 % | 0.0 % |
| `SelfAttention.apply/%24` | B=1,T=256 | f64 | split | 3 | 3 | 66.7 % | 100.0 % | 33.3 % | 0.0 % |
| `SelfAttention.apply/%24` | B=16,T=256 | f32 | split | 10 | 6 | 100.0 % | 100.0 % | 16.7 % | 0.0 % |
| `SelfAttention.apply/%24` | B=16,T=256 | f64 | split | 10 | 5 | 100.0 % | 100.0 % | 20.0 % | 0.0 % |
| `SelfAttention.apply/%6` | B=1,T=256 | f32 | split | 10 | 7 | 57.1 % | 100.0 % | 42.9 % | 0.0 % |
| `SelfAttention.apply/%6` | B=1,T=256 | f64 | split | 10 | 8 | 50.0 % | 100.0 % | 50.0 % | 0.0 % |
| `SelfAttention.apply/%6` | D=1024,H=16,B=1,T=384 | f32 | split | 10 | 8 | 0.0 % | 0.0 % | 50.0 % | 0.0 % |
| `SelfAttention.apply/%6` | B=16,T=256 | f32 | split | 21 | 15 | 100.0 % | 100.0 % | 26.7 % | 0.0 % |
| `SelfAttention.apply/%6` | B=16,T=256 | f64 | split | 21 | 11 | 100.0 % | 100.0 % | 18.2 % | 0.0 % |
| `SelfAttention.apply/%9` | B=1,T=256 | f32 | split | 10 | 8 | 50.0 % | 100.0 % | 50.0 % | 0.0 % |
| `SelfAttention.apply/%9` | B=1,T=256 | f64 | split | 10 | 7 | 57.1 % | 100.0 % | 42.9 % | 0.0 % |
| `SelfAttention.apply/%9` | D=1024,H=16,B=1,T=384 | f32 | split | 10 | 9 | 11.1 % | 0.0 % | 55.6 % | 0.0 % |
| `SelfAttention.apply/%9` | B=16,T=256 | f32 | split | 21 | 17 | 100.0 % | 100.0 % | 29.4 % | 0.0 % |
| `SelfAttention.apply/%9` | B=16,T=256 | f64 | split | 21 | 11 | 90.9 % | 100.0 % | 27.3 % | 0.0 % |
| `gemm/%2` | N=1024 | f32 | kblock | 570 | 354 | 50.3 % | 100.0 % | 99.7 % | 0.0 % |
| `gemm/%2` | N=1024 | f32 | lanes | 630 | 630 | 100.0 % | 100.0 % | 0.0 % | 0.0 % |
| `gemm/%2` | N=1024 | f32 | order | 2033 | 1687 | 64.6 % | 90.6 % | 78.5 % | 2.4 % |
| `gemm/%2` | N=1024 | f32 | split | 2663 | 2571 | 96.0 % | 98.7 % | 9.8 % | 0.0 % |
| `gemm/%2` | N=1024 | f32 | tile | 5670 | 5503 | 90.8 % | 99.6 % | 41.8 % | 0.0 % |

## Decision cost

Static: the entry run once to capture its launches, then every point lowered and costed (`presk plan`). Exhaustive: every point lowered, compiled and timed (`presk autotune`, without the interpreter's check). A sampled sweep is marked: its exhaustive side is not the whole space.

| day | entry | config | type | points | exhaustive (s) | static (s) | ratio |
|---|---|---|---|---|---|---|---|
| `out/ra4/day1` | `SelfAttention.apply` | B=16,T=256 | f32 | 4857 | 1048.3 | 1.462 | 717 |
| `out/ra4/day1` | `SelfAttention.apply` | B=16,T=256 | f64 | 4857 | 1284.0 | 2.115 | 607 |
| `out/ra4/day1` | `SelfAttention.apply` | B=1,T=256 | f32 | 3305 | 58.4 | 0.657 | 89 |
| `out/ra4/day1` | `SelfAttention.apply` | B=1,T=256 | f64 | 3305 | 66.7 | 0.878 | 76 |
| `out/ra4/day1` | `SelfAttention.apply` | D=1024,H=16,B=1,T=384 | f32 | 167 (sample) | 22.5 | 1.815 | 12 |
| `out/ra4/day1` | `gemm` | N=1024 | f32 | 1333 | 701.4 | 0.456 | 1538 |
| `out/ra4/day2` | `SelfAttention.apply` | B=16,T=256 | f32 | 4857 | 1047.5 | 1.504 | 697 |
| `out/ra4/day2` | `SelfAttention.apply` | B=16,T=256 | f64 | 4857 | 1288.2 | 2.213 | 582 |
| `out/ra4/day2` | `SelfAttention.apply` | B=1,T=256 | f32 | 3305 | 58.3 | 0.667 | 87 |
| `out/ra4/day2` | `SelfAttention.apply` | B=1,T=256 | f64 | 3305 | 66.3 | 0.884 | 75 |
| `out/ra4/day2` | `SelfAttention.apply` | D=1024,H=16,B=1,T=384 | f32 | 167 (sample) | 22.4 | 1.820 | 12 |
| `out/ra4/day2` | `gemm` | N=1024 | f32 | 1333 | 701.1 | 0.480 | 1460 |
