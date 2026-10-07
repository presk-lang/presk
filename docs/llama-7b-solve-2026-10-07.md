> A budget of 10⁻³ solved and discharged on the Llama 2 decoder at the 7B configuration's width, one of
> its 32 blocks, 384 tokens, and the whole quantisation scheme run natively: the measurement version
> 0.10.1 was made for (decision D106, step 8 of plan 0.10.1; D105 for the version). Measured from
> 2026-10-05 19:35 to 2026-10-07 09:04 by `out/llama/run-0.10.1.sh` on presk `4610f5e` (the branch
> `0.10.1-step-6`: steps 1–3 and 6 of the plan, the version's code; the release binary `presk
> 0.10.1-dev`, kept as `out/u1/presk-final`), on falicon (aarch64, ten Cortex-X925 and ten Cortex-A725
> cores, 119 GB), alone. Each stage ran in a `systemd-run --user --scope -p MemoryMax=108G -p
> MemorySwapMax=0`, the resident memory of each `presk` sampled every ten seconds
> (`out/llama/rss-0.10.1.log`). Every number below names the file it is read from, under `out/llama/`
> (the run of record) or `out/llama_gate/` (D105's gate, on 16 tokens); none of them is committed, and
> all but the logits are archived as `presk-project/data/llama-7b-0.10.1.tar.xz`. Walls and peaks are
> `/usr/bin/time -v`'s; GB are 2³⁰ bytes, as in D104's report.

# The Llama 2 decoder at the 7B width, solved, 2026-10-07

## The model and the data

D104's (`docs/llama-7b-2026-10-04.md`): `coverage/llama.psk` with `--const G=32` (`D = 4096`, `H = 32`,
`HD = 128`, `FF = 11008`, `V = 32000`), a learned position table in place of RoPE, the parameters from
`--seed 0` (466 104 320 at `L = 1`, `T = 384`), the head drawn as the other matrices. The sequence is
D104's first line of `python3 experiments/llama_ids.py 32000 384 2 0`, as `Tokens`
(`out/llama/ids_T384.psk.d`) and as the loss's (inputs, next tokens) pair (`out/llama/ids_loss_T384.psk.d`).

## The gate (`out/llama_gate/`, `T = 16`)

`out/llama_gate/gate.sh`, 2026-10-05 12:29–16:18, with the binary of steps 1–3 and 6
(`gate.log`); D104's dry run beside it (`out/llama_probe/`, D104's report):

| command | `L` | 0.10.0 (D104) | 0.10.1 | file |
|---|---|---|---|---|
| `presk run --entry Llama.apply --bits 128` | 1 | 6 min 37 s, 85.0 GB | 5 min 11 s, 29.0 GB | `exact_L1_T16.time` |
| the same | 2 | killed at 100 GB | 8 min 16 s, 41.1 GB | `exact_L2_T16.time` |
| `presk run --backend native --realize f32` | 1 | 1 min 29 s, 71.3 GB | 46 s, 36.7 GB | `native_L1_T16_f32.time` |
| the same | 2 | — | 1 min 09 s, 52.9 GB | `native_L2_T16_f32.time` |
| `presk calibrate --precision --entry loss --samples 1 --bits 128` | 1 | 24 min 54 s, 70.1 GB | 22 min 22 s, 30.5 GB | `calibrate_L1_T16.time` |
| the same | 2 | — | 36 min 40 s, 42.5 GB | `calibrate_L2_T16.time` |
| `presk solve --budget 1e-3 --bits 128` | 1 | killed at 108 GB after 18 min | 50 min 43 s, 71.2 GB | `solve_L1_T16.{txt,time}` |
| the same | 2 | — | killed at 108 GB, 68 min after its start | `gate.log`, `oom-journal.txt` |
| `presk quantize --realize @int8.json` | 1 | killed at 108 GB | 10 min 18 s, 61.1 GB | `split_L1_T16_int8.time` |
| the same | 2 | — | 17 min 23 s, 87.7 GB | `split_L2_T16_int8.time` |
| the same | 0 | 5 min 29 s, 74.1 GB | 4 min 10 s, 35.0 GB | `split_L0_T16_int8.time` |
| `presk run --backend kernel --realize @int8_all.json` | 1 | 3 min 51 s, 105.4 GB | 2 min 50 s, 36.2 GB | `kernel_L1_T16_int8_all.time` |

The one-block solve on 16 tokens (`solve_L1_T16.txt`): 56 narrowable sites, 54 away from `f64`,
4.6127·10⁹ bytes against 7.6058·10⁹ all-`f64` (1.65×), the estimate 1.071·10⁻², measured 1.125·10⁻⁵
after 0 widening rounds, inside the budget of 1.089·10⁻² absolute. The exact logits, the native logits,
the calibration's output and envelope (but for its date and data path), the split at no block and the
kernel interpreter's logits are D104's dry run's, byte for byte (D105, step 4).

## The run of record (`out/llama/`, `T = 384`)

`out/llama/run-0.10.1.log`:

| stage | start | end |
|---|---|---|
| `gaps_int8_all`, `gaps_int4_all` | 2026-10-05 19:35:06 | 19:38:23 |
| `bench_all_B1`, `bench_all_B20` | 19:38:23 | 20:45:13 |
| `solve` | 20:45:13 | 2026-10-06 21:10:10 |
| `exact_L2` | 21:10:10 | 2026-10-07 00:46:39 |
| `gaps_L2`, `gaps_L2_int8_all`, `gaps_L2_int4_all` | 00:46:39 | 01:09:49 |
| `split_L1_int8_all`, `split_L1_int4_all` | 01:09:49 | 09:04:02 |

Every stage exited 0; 37 h 29 min in all.

### The solve

`STAGES=solve L=1 T=384 ./experiments/llama.sh out/llama.md`, which runs
`experiments/compile_times.sh` under the key `llama-7b-L1` with one sample (its report
`compile_times_L1_T384.md`, its outputs in `compile_times_T384/`), then the bound:

| command | wall | peak | file |
|---|---|---|---|
| `presk check coverage/llama.psk --const G=32 --const L=1` | 0.006 s | 7 MB | `compile_times_T384/llama-7b-L1_check.time` |
| `presk calibrate --precision … --entry loss --dim T=384 --data out/llama/ids_loss_T384.psk.d --samples 1 --bits 128` | 8 h 23 min 13 s (30 193.4 s) | 32.0 GB | `llama-7b-L1_calibrate.{txt,time}` |
| `presk solve … --precision llama-7b-L1_env.json --budget 1e-3 --bits 128 --sites 0` | **14 h 06 min 51 s** (50 811.6 s) | **71.7 GB** | `llama-7b-L1_solve.{txt,time}` |
| `presk run … --entry loss … --samples 1 --bits 128`, the exact reference alone | 1 h 53 min 48 s (6 828.2 s) | 29.3 GB | `llama-7b-L1_exact.{txt,time}` |
| `presk check … --precision llama-7b-L1_env.json --realize @llama-7b-L1_solved.json` | under 0.01 s | 8 MB | `bound_L1_T384.{txt,time}` |

The sampled resident memory (`rss-0.10.1.log`) reached its 71.7 GB at 2026-10-06 16:20:10, 11 h 12
min into the solve. The parameters and their cotangent at 64 bytes each are 55.6 GB of it.

- **The loss**, `llama-7b-L1_exact.txt`: `ℝ[] = 11.1761830595`.
- **The envelope**, `llama-7b-L1_calibrate.txt`: 70 sites over one sample; the largest magnitudes
  `Attention.apply/%15` 1.2800·10², `RMSNorm.apply/%2` 1.2030·10², `Attention.apply/%13` 1.0188·10²;
  the worst condition numbers of the ten sums and contractions `Attention.apply/%23` 1.0049·10¹,
  `Dense.apply/%3` 9.7994.
- **The solve**, `llama-7b-L1_solve.txt`: budget 10⁻³ relative, 1.118·10⁻² absolute; 56 narrowable sites,
  52 assigned away from `f64`; cost 4.1394·10⁹ bytes against 8.2284·10⁹ all-`f64` (**1.99×**), the
  relaxation floor 2.0571·10⁹; the estimate 1.098·10⁻²; **measured 4.324·10⁻⁶** against the exact
  reference, after 0 widening rounds, inside the budget.
- **The realization**, `llama-7b-L1_solved.json`: default `f64`, sequential order, 52 sites in its
  `storage`: 50 in `f32` and 2 in `bf16`, two scalar constants (`presk dump --stage core`):
  `Attention.apply/%15`, `real 128`, the head width before the square root that divides the scores,
  exact in `bf16`, and `RMSNorm.apply/%5`, `real(1e-5)`, the norm's ε, which `bf16` rounds to
  1.0014·10⁻⁵ (1.4·10⁻³ relative).
- **The bound**, `bound_L1_T384.txt`: `loss` under `f64+sequential+52sites`, magnitude 1.1176·10¹,
  `cond` 6.2390·10⁵ (5.58·10⁴ relative), the envelope over 1 sample and 70 sites.

The ratios: the solve 7.44 times the reference, calibrate 4.42 times (50 811.6 / 6 828.2, 30 193.4 /
6 828.2). D101: BERT-large's two blocks 17.0 and 8.2, the course GPT 15.1 and 8.4, on two samples.

### The whole scheme, natively

`presk run … --entry Llama.apply --const G=32 --const L=1 --dim T=384 --data out/llama/ids_T384.psk.d
--samples 1 --backend native --realize @out/llama/int{8,4}_all.json --digits 17`
(`native_L1_T384_int{8,4}_all.{txt,time}`), against D104's exact logits (`exact_L1_T384.txt`, the largest
|y| 6.882258) and the kernel interpreter's (`kernel_L1_T384_int{8,4}_all.txt`), by
`python3 out/u5/compare_all.py` (`out/u5/compare_all.txt`):

| realization | gap | the kernel interpreter's gap | max \|native − kernel\| | identical logits | wall | peak |
|---|---|---|---|---|---|---|
| `int8` on every matrix | 5.2730·10⁻² | 5.2730·10⁻² | 1.907·10⁻⁵ (2.77·10⁻⁶ of max\|y\|) | 160 438 of 12 288 000 | 1 min 43 s | 36.5 GB |
| `int4` on every matrix | 7.5646·10⁻¹ | 7.5646·10⁻¹ | 1.788·10⁻⁵ (2.60·10⁻⁶) | 163 708 | 1 min 33 s | 36.5 GB |

No NaN among the 12 288 000 logits of either. D104's table rounds the `int4` row to 7.57·10⁻¹, from the
four digits 7.565·10⁻¹: it is 0.75646.

### The time, every matrix in `int8`

`presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=… --dim T=384
--native-only --repeat 2 --realize @out/llama/int8_all.json` (`bench_L1_T384_B1_int8_all.txt`,
`bench_L1_T384_B20_int8_all.txt`), the median of two calls in ms:

| row | batch 1 (one core) | batch 20 | batch 20, per sequence |
|---|---|---|---|
| native f64 | 132 014.3 | 304 231.8 | 15 211.6 |
| native f64 +lanes | 118 547.6 | 310 833.4 | 15 541.7 |
| native f32 | 101 672.3 | 190 192.2 | 9 509.6 |
| native f32 +lanes | 80 907.3 | 188 044.0 | 9 402.2 |
| native bf16 | 84 236.1 | 136 757.0 | 6 837.9 |
| native f32+sequential+10quant | 60 619.4 | 99 226.2 | 4 961.3 |
| native f32+sequential+10quant +lanes | 47 026.4 | 80 381.1 | 4 019.1 |

The better of scalar and four lanes: `f32` 80 907.3 ms at batch 1 and 9 402.2 a sequence at batch 20,
`int8` on every matrix 47 026.4 and 4 019.1: **1.72×** and **2.34×**. D104's run (int8 but `wo`):
1.59× and 1.86×. Its `f32` at batch 20 was 8 481.7 ms a sequence: this run's is 10.9 % slower, where
0.10.1 changes no kernel but those of integer weights read through `split`, so a ratio is read within
one run.

### Two blocks

`L=2 STAGES=exact` and `STAGES=gaps` of `experiments/llama.sh` into `out/llama_L2.md`, then the whole
scheme by hand (`stage_gaps_L2_int{8,4}_all`): `exact_L2_T384.{txt,time}`, `native_L2_T384_*.{txt,time}`.
The exact logits: 12 288 000, the largest |y| 7.208059, in 3 h 36 min 23 s and 41.4 GB. The gaps,
`max|ŷ − y| / max|y|` (`out/llama_L2.md`; the last two rows by `out/u5/compare_all.py`):

| realization | gap at two blocks | at one block | wall | peak |
|---|---|---|---|---|
| `f64` | 1.606·10⁻¹⁴ | 1.265·10⁻¹⁴ | 4 min 49 s | 56.2 GB |
| `f32` | 7.793·10⁻⁶ | 6.181·10⁻⁶ | 3 min 27 s | 53.7 GB |
| `bf16` | 3.360·10⁻² | 2.451·10⁻² | 3 min 45 s | 52.3 GB |
| `int8` weights but `wo` | 5.620·10⁻² | 4.852·10⁻² | 2 min 42 s | 51.7 GB |
| `int4` weights but `wo` | 9.353·10⁻¹ | 7.140·10⁻¹ | 2 min 41 s | 51.7 GB |
| `int8` on every matrix | 5.6892·10⁻² | 5.2730·10⁻² | 2 min 45 s | 51.6 GB |
| `int4` on every matrix | 9.8248·10⁻¹ | 7.5646·10⁻¹ | 2 min 19 s | 51.6 GB |

The one-block column is D104's native rows and, for every matrix, this run's.

### The split at one block, the whole scheme

`presk quantize coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim T=384 --data
out/llama/ids_T384.psk.d --samples 1 --realize @out/llama/int{8,4}_all.json --backend native`
(`split_L1_T384_int{8,4}_all.{txt,time}`); the largest exact logit 6.882258 (`|exact|` in both):

| realization | representation | arithmetic | total | storage | wall | peak |
|---|---|---|---|---|---|---|
| `int8` on every matrix | 3.629·10⁻¹ (5.27·10⁻² relative) | 4.380·10⁻⁵ (6.36·10⁻⁶) | 3.629·10⁻¹ (5.27·10⁻²) | 1 864 368 128 → 466 422 784 bytes (4.00×) | 4 h 12 min 08 s | 63.5 GB |
| `int4` on every matrix | 5.206 (7.56·10⁻¹) | 4.214·10⁻⁵ (6.12·10⁻⁶) | 5.206 (7.56·10⁻¹) | 1 864 368 128 → 233 376 768 bytes (7.99×) | 3 h 42 min 04 s | 63.2 GB |

D104's at no block (`split_L0_T384_int{8,4}.txt`): 1.18·10⁻² and 3.29·10⁻⁶ in `int8`, 2.25·10⁻¹ and
2.85·10⁻⁶ in `int4`, 1 h 22 min and 76.4 GB each.

## Not run

- The solve at two blocks: killed at 108 GB on 16 tokens (the gate).
- The split at two blocks: 87.7 GB on 16 tokens, two exact runs of 3 h 36 min a scheme.
- The kernel interpreter at two blocks, and `kgaps` again at one: the native rows equal D104's kernel
  interpreter's (above), and the native test of integer weights runs D104's two-line case,
  `tests/quant/split.psk`, against it (D105, step 6).
- The 32 blocks, in any mode.

## The archive

`presk-project/data/llama-7b-0.10.1.tar.xz` holds 0.10.1's files of `out/llama/` (the run's script, log
and resident memory, the stage logs, the solve's outputs, the bound, the `.time` files of 2026-10-05 to
07, the bench and split outputs, the tables `out/llama.md` and `out/llama_L2.md`, the realizations
`int*.json`) and `out/llama_gate/` but for the logits, with their checksums in
`SHA256SUMS-logits-0.10.1` (`exact_L2_T384.txt`, `native_L{1,2}_T384_*.txt` of 0.10.1 and the gate's),
and the user journal's lines of the gate's kill (`llama_gate/oom-journal.txt`). The logits are a
deterministic output of the compiler (the gate's are D104's dry run's, byte for byte), and stay on
falicon. D104's archive, `llama-7b-0.10.0.tar.xz`, is unchanged.
