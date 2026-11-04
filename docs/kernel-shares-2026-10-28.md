# Each kernel's share of the native backend's gap to the C program

> Each kernel's share of the native backend's gap to the C program (presk D174, plan 0.17.0
> step 1, statement NB1), by `make kernel-shares` (`experiments/kernel_shares.sh`, `.py`). The
> run used the release binary of this step's tree (`out/bin/presk-0.17.0-step1`: `dcb4d54` and
> `presk bench --calibrate --plan-only`), from 09:17 to 09:30 on 2026-10-28, alone on falicon.
> The tree was `-dirty` by this step's uncommitted files. Every output but the C programs'
> outputs and binaries is archived in presk-project `data/kernel-shares-0.17.0.tar.xz`.
>
> The kernels' times only. D166's calls of the step, less these kernels' times, leave about
> 30 ms to the native host and about 110 ms to the C program's, so the kernels' gap on the step
> (1.90×) is wider than the calls' (1.67×).


date 2026-10-28T09:17:15+01:00  
presk dcb4d54-dirty, presk 0.17.0-dev  
kernel Linux 6.17.0-1021-nvidia, 20 cores  
gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0  
clang version 23.1.2 (https://github.com/llvm/llvm-project 85ac560262434c9ccfc0c183ec22d4138ed647fb)  
load 0.78 1.86 2.06  
cores 5-9,15-19, core 5, c_wait GOMP_SPINCOUNT=8M, repeat 7, calls 4, rounds 3

A row is a kernel signature. Its native time a call is its median launch (`presk bench --calibrate`) times the C program's launches of it a call; the C times are each launch timed in the program (`kernel_time.py`), 4 calls with the first left out. Every figure is the median of 3 processes; the gap is native − GCC. The kernels' times only: neither side's host is in them.

## the forward pass `GPT.apply`, ten X925, ten threads

Native 331.19 ms, GCC 202.08 ms: **1.64×**, a gap of 129.11 ms.

| kernel | launches a call | native (ms) | gcc-O2 (ms) | native ÷ GCC | share of the gap | cumulative |
|---|---|---|---|---|---|---|
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 256,…` | 24 | 74.70 | 42.84 | 1.74× | 24.7 % | 24.7 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq: 25…` | 6 | 75.24 | 45.50 | 1.65× | 23.0 % | 47.7 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 25…` | 6 | 74.23 | 44.53 | 1.67× | 23.0 % | 70.7 % |
| `kernel softmax axis seq2 where, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : i64[seq: 256, seq2…` | 6 | 45.17 | 20.54 | 2.20× | 19.1 % | 89.8 % |
| the other 19 | | 61.34 | 48.76 | 1.26× | 9.7 % | 100 % |

| kind | launches a call | native (ms) | gcc-O2 (ms) | native ÷ GCC | share of the gap |
|---|---|---|---|---|---|
| `dot+add` | 36 | 224.17 | 132.87 | 1.69× | 70.7 % |
| `softmax` | 6 | 45.17 | 20.54 | 2.20× | 19.1 % |
| `ein` | 12 | 24.98 | 17.28 | 1.45× | 6.0 % |
| `gelu` | 6 | 16.81 | 13.57 | 1.24× | 2.5 % |
| `pack` | 6 | 4.99 | 3.29 | 1.52× | 1.3 % |
| `dot` | 1 | 1.33 | 0.55 | 2.41× | 0.6 % |
| `embed` | 2 | 0.94 | 0.45 | 2.11× | 0.4 % |
| `fused` | 39 | 2.46 | 2.05 | 1.20× | 0.3 % |
| `mean` | 26 | 2.17 | 1.95 | 1.11× | 0.2 % |
| `causal_mask` | 1 | 0.04 | 0.02 | 1.72× | 0.0 % |
| `sqrt` | 6 | 0.00 | 0.00 | 7.76× | 0.0 % |
| `positions` | 1 | 0.00 | 0.00 | 4.16× | 0.0 % |
| `merge_heads` | 6 | 0.64 | 0.84 | 0.76× | -0.2 % |
| `split_heads` | 18 | 2.81 | 3.15 | 0.89× | -0.3 % |
| `add` | 13 | 1.78 | 2.43 | 0.73× | -0.5 % |
| `div` | 6 | 2.39 | 3.18 | 0.75× | -0.6 % |

## the gradient `loss_grad`, ten X925, ten threads

Native 1297.02 ms, GCC 678.04 ms: **1.91×**, a gap of 618.97 ms.

| kernel | launches a call | native (ms) | gcc-O2 (ms) | native ÷ GCC | share of the gap | cumulative |
|---|---|---|---|---|---|---|
| `kernel ein[batch, heads, seq2, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, seq:…` | 6 | 153.31 | 35.24 | 4.35× | 19.1 % | 19.1 % |
| `kernel sum_to (%0 : f32[batch: 16, seq: 256, model: 384], %1 : f32[batch: 16, seq: 256]) -> %2 : f32[batch: 16, seq: 25…` | 39 | 134.55 | 27.66 | 4.87× | 17.3 % | 36.3 % |
| `kernel dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 77.12 | 42.89 | 1.80× | 5.5 % | 41.9 % |
| `kernel dot_vjp_l (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 6 | 78.09 | 44.40 | 1.76× | 5.4 % | 47.3 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 256,…` | 24 | 74.92 | 42.92 | 1.75× | 5.2 % | 52.5 % |
| `kernel dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 78.28 | 47.34 | 1.65× | 5.0 % | 57.5 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 25…` | 6 | 75.01 | 44.65 | 1.68× | 4.9 % | 62.4 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq: 25…` | 6 | 75.52 | 45.38 | 1.66× | 4.9 % | 67.3 % |
| `kernel dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[384, 384]` | 24 | 70.73 | 44.71 | 1.58× | 4.2 % | 71.5 % |
| `kernel softmax axis seq2 where, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : i64[seq: 256, seq2…` | 6 | 45.18 | 20.99 | 2.15× | 3.9 % | 75.4 % |
| `kernel dot_vjp_r (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 384]) -> %2 : f32[1536, 384]` | 6 | 68.32 | 44.47 | 1.54× | 3.9 % | 79.2 % |
| `kernel dot_vjp_r (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[384, 1536]` | 6 | 71.44 | 48.22 | 1.48× | 3.8 % | 83.0 % |
| the other 78 | | 296.29 | 188.19 | 1.57× | 17.5 % | 100 % |

| kind | launches a call | native (ms) | gcc-O2 (ms) | native ÷ GCC | share of the gap |
|---|---|---|---|---|---|
| `sum_to` | 102 | 186.08 | 41.54 | 4.48× | 23.4 % |
| `ein` | 36 | 215.43 | 78.06 | 2.76× | 22.2 % |
| `dot_vjp_l` | 37 | 234.29 | 135.06 | 1.73× | 16.0 % |
| `dot+add` | 36 | 225.45 | 132.95 | 1.70× | 14.9 % |
| `dot_vjp_r` | 37 | 212.04 | 138.07 | 1.54× | 12.0 % |
| `softmax` | 6 | 45.18 | 20.99 | 2.15× | 3.9 % |
| `zeros` | 21 | 17.23 | 2.56 | 6.74× | 2.4 % |
| `mean_vjp` | 27 | 15.05 | 1.99 | 7.55× | 2.1 % |
| `fused` | 274 | 63.43 | 52.08 | 1.22× | 1.8 % |
| `add` | 160 | 13.11 | 8.89 | 1.48× | 0.7 % |
| `pack` | 49 | 18.63 | 14.61 | 1.27× | 0.6 % |
| `gelu` | 6 | 16.90 | 13.53 | 1.25× | 0.5 % |
| `neg` | 26 | 5.95 | 3.22 | 1.84× | 0.4 % |
| `dot` | 1 | 1.33 | 0.58 | 2.30× | 0.1 % |
| `embed_vjp` | 2 | 0.95 | 0.22 | 4.38× | 0.1 % |
| `split_heads` | 24 | 4.45 | 3.91 | 1.14× | 0.1 % |
| `embed` | 2 | 0.87 | 0.45 | 1.92× | 0.1 % |
| `sum` | 6 | 1.97 | 1.76 | 1.12× | 0.0 % |
| `cross_entropy` | 1 | 0.20 | 0.08 | 2.44× | 0.0 % |
| `cross_entropy_vjp` | 1 | 0.23 | 0.15 | 1.54× | 0.0 % |
| `causal_mask` | 1 | 0.04 | 0.02 | 1.80× | 0.0 % |
| `sqrt` | 6 | 0.00 | 0.00 | 2.42× | 0.0 % |
| `positions` | 1 | 0.00 | 0.00 | 4.04× | 0.0 % |
| `broadcast_like` | 6 | 2.14 | 2.16 | 0.99× | -0.0 % |
| `merge_heads` | 24 | 2.92 | 3.02 | 0.97× | -0.0 % |
| `sub` | 13 | 0.91 | 1.14 | 0.80× | -0.0 % |
| `pow` | 13 | 0.93 | 1.63 | 0.57× | -0.1 % |
| `mul` | 32 | 7.32 | 8.26 | 0.89× | -0.2 % |
| `div` | 19 | 3.57 | 4.81 | 0.74× | -0.2 % |
| `mean` | 26 | 2.17 | 5.33 | 0.41× | -0.5 % |

## the optimizer `adamw.step`, ten X925, ten threads

Native 15.24 ms, GCC 12.45 ms: **1.22×**, a gap of 2.79 ms.

| kernel | launches a call | native (ms) | gcc-O2 (ms) | native ÷ GCC | share of the gap | cumulative |
|---|---|---|---|---|---|---|
| `kernel mul (%0 : f32[6, 384, 384], %1 : f32[]) -> %2 : f32[6, 384, 384]` | 4 | 1.49 | 0.54 | 2.76× | 34.0 % | 34.0 % |
| `kernel mul (%0 : f32[6, 1536, 384], %1 : f32[]) -> %2 : f32[6, 1536, 384]` | 1 | 1.47 | 0.53 | 2.79× | 33.6 % | 67.6 % |
| `kernel mul (%0 : f32[6, 384, 1536], %1 : f32[]) -> %2 : f32[6, 384, 1536]` | 1 | 1.45 | 0.53 | 2.75× | 33.1 % | 100.7 % |
| the other 46 | | 10.90 | 10.60 | 1.03× | 10.7 % | 100 % |

| kind | launches a call | native (ms) | gcc-O2 (ms) | native ÷ GCC | share of the gap |
|---|---|---|---|---|---|
| `mul` | 21 | 4.52 | 1.63 | 2.77× | 103.5 % |
| `fused` | 67 | 5.08 | 4.87 | 1.04× | 7.7 % |
| `sum_squares` | 21 | 5.70 | 5.70 | 1.00× | 0.1 % |
| `sub` | 2 | 0.00 | 0.00 | 2.16× | 0.0 % |

## the forward pass `GPT.apply`, one X925, one thread

Native 3036.34 ms, GCC 2181.73 ms: **1.39×**, a gap of 854.61 ms. clang-O2: 1660.69 ms (1.83× native ÷ it). clang-O3: 1657.86 ms (1.83× native ÷ it).

| kernel | launches a call | native (ms) | gcc-O2 (ms) | clang-O2 (ms) | clang-O3 (ms) | native ÷ GCC | share of the gap | cumulative |
|---|---|---|---|---|---|---|---|---|
| `kernel softmax axis seq2 where, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : i64[seq: 256, seq2…` | 6 | 432.30 | 129.16 | 133.50 | 133.87 | 3.35× | 35.5 % | 35.5 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 256,…` | 24 | 702.54 | 553.86 | 389.68 | 388.27 | 1.27× | 17.4 % | 52.9 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq: 25…` | 6 | 703.36 | 558.11 | 411.21 | 400.92 | 1.26× | 17.0 % | 69.9 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 25…` | 6 | 701.73 | 558.67 | 394.09 | 409.21 | 1.26× | 16.7 % | 86.6 % |
| the other 19 | | 496.69 | 382.26 | 332.75 | 325.48 | 1.30× | 13.4 % | 100 % |

| kind | launches a call | native (ms) | gcc-O2 (ms) | clang-O2 (ms) | clang-O3 (ms) | native ÷ GCC | share of the gap |
|---|---|---|---|---|---|---|---|
| `dot+add` | 36 | 2107.63 | 1670.64 | 1194.99 | 1198.40 | 1.26× | 51.1 % |
| `softmax` | 6 | 432.30 | 129.16 | 133.50 | 133.87 | 3.35× | 35.5 % |
| `gelu` | 6 | 164.67 | 130.65 | 147.43 | 144.26 | 1.26× | 4.0 % |
| `ein` | 12 | 219.34 | 188.22 | 134.79 | 130.62 | 1.17× | 3.6 % |
| `fused` | 39 | 27.10 | 8.21 | 8.16 | 8.19 | 3.30× | 2.2 % |
| `split_heads` | 18 | 16.85 | 5.42 | 3.53 | 3.53 | 3.11× | 1.3 % |
| `div` | 6 | 15.76 | 9.76 | 9.74 | 9.74 | 1.61× | 0.7 % |
| `merge_heads` | 6 | 5.66 | 1.14 | 1.12 | 1.11 | 4.98× | 0.5 % |
| `add` | 13 | 8.69 | 4.53 | 4.56 | 4.57 | 1.92× | 0.5 % |
| `mean` | 26 | 20.63 | 18.37 | 15.49 | 15.50 | 1.12× | 0.3 % |
| `pack` | 6 | 5.25 | 3.71 | 2.43 | 2.46 | 1.41× | 0.2 % |
| `embed` | 2 | 0.81 | 0.15 | 0.15 | 0.15 | 5.24× | 0.1 % |
| `causal_mask` | 1 | 0.04 | 0.02 | 0.03 | 0.03 | 1.94× | 0.0 % |
| `sqrt` | 6 | 0.00 | 0.00 | 0.00 | 0.00 | 11.01× | 0.0 % |
| `positions` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 3.50× | 0.0 % |
| `dot` | 1 | 11.90 | 12.07 | 5.32 | 5.31 | 0.99× | -0.0 % |

## the gradient `loss_grad`, one X925, one thread

Native 10100.01 ms, GCC 6725.00 ms: **1.50×**, a gap of 3375.01 ms. clang-O2: 6003.71 ms (1.68× native ÷ it). clang-O3: 5978.46 ms (1.69× native ÷ it).

| kernel | launches a call | native (ms) | gcc-O2 (ms) | clang-O2 (ms) | clang-O3 (ms) | native ÷ GCC | share of the gap | cumulative |
|---|---|---|---|---|---|---|---|---|
| `kernel ein[batch, heads, seq2, hd] (%0 : f32[batch: 16, heads: 6, seq: 256, hd: 64], %1 : f32[batch: 16, heads: 6, seq:…` | 6 | 1912.53 | 324.52 | 1269.08 | 1274.72 | 5.89× | 47.1 % | 47.1 % |
| `kernel softmax axis seq2 where, two passes (%0 : f32[batch: 16, heads: 6, seq: 256, seq2: 256], %1 : i64[seq: 256, seq2…` | 6 | 431.06 | 129.10 | 133.55 | 133.66 | 3.34× | 8.9 % | 56.0 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 256,…` | 24 | 702.65 | 554.10 | 403.73 | 387.82 | 1.27× | 4.4 % | 60.4 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[1536, 384], %2 : f32[384]) -> %3 : f32[batch: 16, seq: 25…` | 6 | 702.26 | 557.36 | 395.79 | 409.32 | 1.26× | 4.3 % | 64.7 % |
| `kernel dot+add (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536], %2 : f32[1536]) -> %3 : f32[batch: 16, seq: 25…` | 6 | 702.95 | 558.24 | 399.19 | 400.04 | 1.26× | 4.3 % | 69.0 % |
| `kernel dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 384]) -> %2 : f32[batch: 16, seq: 256, 384]` | 24 | 673.99 | 553.38 | 402.17 | 385.71 | 1.22× | 3.6 % | 72.6 % |
| `kernel dot_vjp_l (%0 : f32[batch: 16, seq: 256, 384], %1 : f32[384, 1536]) -> %2 : f32[batch: 16, seq: 256, 1536]` | 6 | 675.75 | 556.41 | 407.49 | 398.53 | 1.21× | 3.5 % | 76.1 % |
| `kernel sum_to (%0 : f32[batch: 16, seq: 256, model: 384], %1 : f32[batch: 16, seq: 256]) -> %2 : f32[batch: 16, seq: 25…` | 39 | 134.07 | 27.63 | 23.28 | 23.32 | 4.85× | 3.2 % | 79.2 % |
| `kernel fused 2 steps (%0 : f32[batch: 16, seq: 256, 1536], %1 : f32[batch: 16, seq: 256, 1536]) -> %2 : f32[batch: 16, …` | 6 | 331.57 | 237.19 | 259.36 | 257.89 | 1.40× | 2.8 % | 82.0 % |
| the other 81 | | 3833.01 | 3225.98 | 2310.26 | 2308.97 | 1.19× | 18.0 % | 100 % |

| kind | launches a call | native (ms) | gcc-O2 (ms) | clang-O2 (ms) | clang-O3 (ms) | native ÷ GCC | share of the gap |
|---|---|---|---|---|---|---|---|
| `ein` | 36 | 2466.19 | 809.49 | 1610.77 | 1608.69 | 3.05× | 49.1 % |
| `dot+add` | 36 | 2107.85 | 1669.70 | 1198.72 | 1197.18 | 1.26× | 13.0 % |
| `dot_vjp_l` | 37 | 2007.06 | 1688.63 | 1194.58 | 1178.62 | 1.19× | 9.4 % |
| `softmax` | 6 | 431.06 | 129.10 | 133.55 | 133.66 | 3.34× | 8.9 % |
| `dot_vjp_r` | 37 | 1995.20 | 1816.41 | 1239.04 | 1237.24 | 1.10× | 5.3 % |
| `sum_to` | 102 | 185.84 | 40.38 | 36.24 | 36.17 | 4.60× | 4.3 % |
| `fused` | 274 | 432.89 | 293.80 | 315.82 | 314.43 | 1.47× | 4.1 % |
| `gelu` | 6 | 164.78 | 132.06 | 149.52 | 148.44 | 1.25× | 1.0 % |
| `merge_heads` | 24 | 26.32 | 7.62 | 5.49 | 5.52 | 3.46× | 0.6 % |
| `mean_vjp` | 27 | 21.51 | 3.41 | 3.25 | 3.22 | 6.31× | 0.5 % |
| `broadcast_like` | 6 | 20.12 | 3.31 | 3.24 | 3.24 | 6.08× | 0.5 % |
| `split_heads` | 24 | 22.49 | 5.75 | 4.85 | 4.86 | 3.91× | 0.5 % |
| `neg` | 26 | 21.86 | 5.98 | 6.02 | 5.95 | 3.66× | 0.5 % |
| `mul` | 32 | 33.66 | 17.87 | 17.63 | 17.59 | 1.88× | 0.5 % |
| `zeros` | 21 | 17.21 | 2.56 | 2.55 | 2.54 | 6.72× | 0.4 % |
| `add` | 160 | 26.51 | 13.46 | 13.45 | 13.50 | 1.97× | 0.4 % |
| `div` | 19 | 24.21 | 15.10 | 15.04 | 15.04 | 1.60× | 0.3 % |
| `pow` | 13 | 10.92 | 2.92 | 2.93 | 2.96 | 3.74× | 0.2 % |
| `sub` | 13 | 8.46 | 2.92 | 2.86 | 2.87 | 2.90× | 0.2 % |
| `pack` | 49 | 18.49 | 14.89 | 11.91 | 11.88 | 1.24× | 0.1 % |
| `sum` | 6 | 18.89 | 15.53 | 11.98 | 11.97 | 1.22× | 0.1 % |
| `mean` | 26 | 20.62 | 18.47 | 15.56 | 15.54 | 1.12× | 0.1 % |
| `cross_entropy` | 1 | 1.84 | 0.72 | 1.18 | 1.17 | 2.57× | 0.0 % |
| `cross_entropy_vjp` | 1 | 2.17 | 1.37 | 2.01 | 2.01 | 1.58× | 0.0 % |
| `embed_vjp` | 2 | 0.94 | 0.21 | 0.22 | 0.22 | 4.48× | 0.0 % |
| `embed` | 2 | 0.81 | 0.15 | 0.15 | 0.16 | 5.22× | 0.0 % |
| `causal_mask` | 1 | 0.04 | 0.02 | 0.03 | 0.03 | 1.88× | 0.0 % |
| `sqrt` | 6 | 0.00 | 0.00 | 0.00 | 0.00 | 11.26× | 0.0 % |
| `positions` | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 3.68× | 0.0 % |
| `dot` | 1 | 11.89 | 12.07 | 5.32 | 5.26 | 0.99× | -0.0 % |

## the optimizer `adamw.step`, one X925, one thread

Native 34.39 ms, GCC 25.68 ms: **1.34×**, a gap of 8.71 ms. clang-O2: 25.61 ms (1.34× native ÷ it). clang-O3: 25.72 ms (1.34× native ÷ it).

| kernel | launches a call | native (ms) | gcc-O2 (ms) | clang-O2 (ms) | clang-O3 (ms) | native ÷ GCC | share of the gap | cumulative |
|---|---|---|---|---|---|---|---|---|
| `kernel fused 3 steps (%0 : f32[6, 384, 384], %1 : f32[], %2 : f32[6, 384, 384], %3 : f32[]) -> %4 : f32[6, 384, 384]` | 4 | 1.98 | 0.90 | 0.91 | 0.92 | 2.19× | 12.3 % | 12.3 % |
| `kernel fused 3 steps (%0 : f32[6, 1536, 384], %1 : f32[], %2 : f32[6, 1536, 384], %3 : f32[]) -> %4 : f32[6, 1536, 384]` | 1 | 1.97 | 0.96 | 0.95 | 0.96 | 2.06× | 11.7 % | 24.0 % |
| `kernel fused 4 steps (%0 : f32[6, 384, 384], %1 : f32[], %2 : f32[6, 384, 384], %3 : f32[]) -> %4 : f32[6, 384, 384]` | 4 | 1.98 | 0.97 | 0.96 | 0.98 | 2.04× | 11.6 % | 35.6 % |
| `kernel fused 4 steps (%0 : f32[6, 1536, 384], %1 : f32[], %2 : f32[6, 1536, 384], %3 : f32[]) -> %4 : f32[6, 1536, 384]` | 1 | 1.98 | 0.98 | 0.98 | 0.99 | 2.02× | 11.5 % | 47.1 % |
| `kernel fused 3 steps (%0 : f32[6, 384, 1536], %1 : f32[], %2 : f32[6, 384, 1536], %3 : f32[]) -> %4 : f32[6, 384, 1536]` | 1 | 1.94 | 0.96 | 0.95 | 0.95 | 2.03× | 11.3 % | 58.4 % |
| `kernel fused 4 steps (%0 : f32[6, 384, 1536], %1 : f32[], %2 : f32[6, 384, 1536], %3 : f32[]) -> %4 : f32[6, 384, 1536]` | 1 | 1.95 | 0.99 | 0.98 | 0.99 | 1.97× | 11.0 % | 69.4 % |
| `kernel mul (%0 : f32[6, 384, 384], %1 : f32[]) -> %2 : f32[6, 384, 384]` | 4 | 1.49 | 0.59 | 0.58 | 0.59 | 2.54× | 10.4 % | 79.8 % |
| `kernel mul (%0 : f32[6, 1536, 384], %1 : f32[]) -> %2 : f32[6, 1536, 384]` | 1 | 1.46 | 0.58 | 0.58 | 0.59 | 2.51× | 10.1 % | 89.9 % |
| the other 41 | | 19.64 | 18.73 | 18.71 | 18.74 | 1.05× | 10.4 % | 100 % |

| kind | launches a call | native (ms) | gcc-O2 (ms) | clang-O2 (ms) | clang-O3 (ms) | native ÷ GCC | share of the gap |
|---|---|---|---|---|---|---|---|
| `fused` | 67 | 24.18 | 18.16 | 18.18 | 18.26 | 1.33× | 69.1 % |
| `mul` | 21 | 4.52 | 1.79 | 1.77 | 1.80 | 2.53× | 31.4 % |
| `sub` | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 1.41× | 0.0 % |
| `sum_squares` | 21 | 5.70 | 5.71 | 5.65 | 5.65 | 1.00× | -0.1 % |

## The totals, and their spread over the processes

| entry | setting | side | median (ms) | max ÷ min |
|---|---|---|---|---|
| the forward pass `GPT.apply` | ten X925, ten threads | native | 331.19 | 1.009 |
| the forward pass `GPT.apply` | ten X925, ten threads | gcc-O2 | 202.08 | 1.001 |
| the gradient `loss_grad` | ten X925, ten threads | native | 1297.02 | 1.007 |
| the gradient `loss_grad` | ten X925, ten threads | gcc-O2 | 678.04 | 1.012 |
| the optimizer `adamw.step` | ten X925, ten threads | native | 15.24 | 1.024 |
| the optimizer `adamw.step` | ten X925, ten threads | gcc-O2 | 12.45 | 1.192 |
| the forward pass `GPT.apply` | one X925, one thread | native | 3036.34 | 1.001 |
| the forward pass `GPT.apply` | one X925, one thread | gcc-O2 | 2181.73 | 1.002 |
| the forward pass `GPT.apply` | one X925, one thread | clang-O2 | 1660.69 | 1.001 |
| the forward pass `GPT.apply` | one X925, one thread | clang-O3 | 1657.86 | 1.001 |
| the gradient `loss_grad` | one X925, one thread | native | 10100.01 | 1.000 |
| the gradient `loss_grad` | one X925, one thread | gcc-O2 | 6725.00 | 1.001 |
| the gradient `loss_grad` | one X925, one thread | clang-O2 | 6003.71 | 1.000 |
| the gradient `loss_grad` | one X925, one thread | clang-O3 | 5978.46 | 1.001 |
| the optimizer `adamw.step` | one X925, one thread | native | 34.39 | 1.001 |
| the optimizer `adamw.step` | one X925, one thread | gcc-O2 | 25.68 | 1.004 |
| the optimizer `adamw.step` | one X925, one thread | clang-O2 | 25.61 | 1.004 |
| the optimizer `adamw.step` | one X925, one thread | clang-O3 | 25.72 | 1.000 |
