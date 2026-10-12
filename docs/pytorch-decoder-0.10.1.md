> PyTorch eager on one block of the Llama 2 decoder at the 7B width, beside presk's native backend (D107: `experiments/pytorch_decoder.sh out/pytorch-decoder` on presk `25dbeda`, version 0.10.1's code and the script, begun 2026-10-07 at 16:35 on falicon, alone, the timed rows done at 17:59; the one-minute load of 7.2 at its start left by `make test`, which had finished; the native kernels alone, 18:03–20:23, and the host profile, 20:24, after every timed row, from `18d83c1`, the same binary), copied from `out/pytorch-decoder/report.md`. After it: the gate on 16 tokens, the checks of correctness at small constants, what these rows are not, and PyTorch's configuration. A measurement: it decides nothing.

# The Llama 2 decoder block against PyTorch eager: presk 0.10.1

One block of the Llama 2 decoder at the 7B configuration's width (`coverage/llama.psk`, `--const G=32 --const L=1`: D = 4096, 32 heads each its own key head, SwiGLU's 11008, V = 32000), its forward pass `Llama.apply` at T = 384, `f32`; 2026-10-07, presk `25dbeda`, `experiments/pytorch_decoder.sh`. Two sides on the ten X925 cores pinned (`taskset -c 5-9,15-19`, ten threads): presk's native backend (`presk bench --native-only --plan-only`), under its default layouts and under `presk plan`'s plan with the frozen profile, in `f32` and with `int8` weights on every matrix; and PyTorch running the module `presk export --target pytorch --mode readable` writes, in eval mode under `torch.no_grad()`, with its own draw of the same distributions (every matrix normal(std = 0.02), every γ ones; seed 0) and token ids drawn by `torch.randint` (seed 0). Each time is the median of the calls after one warmup call, which compiles natively; the native process's exact draw of the parameters comes before and is not timed.

**Operations.** The forward pass's contractions, a multiply and an add a term: batch 1: 258.50 GFLOP by the compiler's count of them, 258.50 by PyTorch's counter (`torch.utils.flop_counter`); batch 20: 5170.07 GFLOP by the compiler's count of them, 5170.07 by PyTorch's counter (`torch.utils.flop_counter`). The rates below divide PyTorch's count by the median, whatever a side executes.

## The forward pass, B = 1

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 95 806.6 | 95 745.3 | 2.7 | 348.25 |
| 10 (ten X925, pinned) | native | plan | 3 744.1 | 3 719.4 | 69.0 | 13.61 |
| 10 (ten X925, pinned) | native, int8 every matrix | default | 48 617.0 | 48 326.0 | 5.3 | 176.72 |
| 10 (ten X925, pinned) | native, int8 every matrix | plan | 1 870.6 | 1 828.5 | 138.2 | 6.80 |
| 10 (ten X925, pinned) | PyTorch | its own | 275.1 | 270.1 | 939.6 | 1.00 |
| 1 (one X925, cpu 15, pinned) | PyTorch | its own | 1 910.5 | 1 909.1 | 135.3 | 6.94 |

PyTorch's logits on its last call, ten threads: [1, 384, 32000], finite yes, max|y| 6.8378.

## The forward pass, B = 20

| threads | side | layout | median ms | best ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 214 063.7 | 212 926.5 | 24.2 | 47.14 |
| 10 (ten X925, pinned) | native | plan | 65 195.9 | 65 079.6 | 79.3 | 14.36 |
| 10 (ten X925, pinned) | native, int8 every matrix | default | 94 453.5 | 94 396.1 | 54.7 | 20.80 |
| 10 (ten X925, pinned) | native, int8 every matrix | plan | 37 602.9 | 37 532.6 | 137.5 | 8.28 |
| 10 (ten X925, pinned) | PyTorch | its own | 4 541.2 | 4 507.4 | 1138.5 | 1.00 |

PyTorch's logits on its last call, ten threads: [20, 384, 32000], finite yes, max|y| 7.8442.

## Where the native time goes, under the `f32` plan

Every launch timed on its own (`presk bench --calibrate`, D100's: a launch's threads' start included, the host's buffers and copies not), each kernel's median launch; the kernels' time a call is each kernel's median launch times its launches. The largest kernels by their launch's time; a kernel is one shape of a nest, so `wk` and `wv` (their output axis `kvout`) are one kernel launched twice, and so are SwiGLU's `gate` and `up`; every other kernel here is launched once. GFLOP: a product's multiplies and adds, from its operands' shapes (a `dot`'s left operand's elements times its right operand's columns, an `ein`'s every axis once), twice its terms; the kernel's own operation count, which weighs index arithmetic too, is not used.

In the call itself (`presk bench --host-profile`, D95's, at batch 1 under the `f32` plan, three calls timed by phase): the kernels 3 824.1 ms of 3 827.3, the host 0.8 ms. So what a call takes beyond its kernels timed alone is not the host's: inside the call a launch takes longer than the same launch timed on its own.

### B = 1

34 launches a call, the kernels' time 2 470.2 ms timed alone, the plan's row 3 744.1 ms.

| kernel | GFLOP a launch | ms a launch | share of the kernels' time | GFLOP/s |
|---|---|---|---|---|
| `ein[batch, seq, model] (f32[batch: 1, seq: 384, groups: 32, per: 1, hd: 128], f32[groups: 32, per: 1, hd: 128, model: 4096])` | 12.88 | 1 724.0 | 69.8 % | 7.5 |
| `dot (f32[batch: 1, seq: 384, model: 4096], f32[4096, 32000])` | 100.66 | 255.3 | 10.3 % | 394.3 |
| `ein[batch, seq, groups, per, hd] (f32[batch: 1, groups: 32, per: 1, seq: 384, seq2: 384], f32[batch: 1, seq: 384, groups: 32, hd: 128])` | 1.21 | 89.0 | 3.6 % | 13.6 |
| `dot (f32[batch: 1, seq: 384, 11008], f32[11008, 4096])` | 34.63 | 87.4 | 3.5 % | 396.0 |
| `dot (f32[batch: 1, seq: 384, 4096], f32[4096, 11008])` | 34.63 | 83.9 | 6.8 % (2 launches) | 412.6 |
| `ein[batch, groups, per, seq, seq2] (f32[batch: 1, seq: 384, groups: 32, per: 1, hd: 128], f32[batch: 1, seq: 384, groups: 32, hd: 128])` | 1.21 | 39.7 | 1.6 % | 30.5 |
| `dot (f32[batch: 1, seq: 384, model: 4096], f32[model: 4096, kvout: 4096])` | 12.88 | 30.9 | 2.5 % (2 launches) | 417.3 |
| `dot (f32[batch: 1, seq: 384, model: 4096], f32[model: 4096, qout: 4096])` | 12.88 | 30.8 | 1.2 % | 418.8 |
| `softmax axis seq2 where, two passes (f32[batch: 1, groups: 32, per: 1, seq: 384, seq2: 384], i64[seq: 384, seq2: 384])` | — | 6.3 | 0.3 % | — |
| `fused 2 steps (f32[batch: 1, seq: 384, 11008])` | — | 3.1 | 0.1 % | — |

### B = 20

34 launches a call, the kernels' time 44 007.7 ms timed alone, the plan's row 65 195.9 ms.

| kernel | GFLOP a launch | ms a launch | share of the kernels' time | GFLOP/s |
|---|---|---|---|---|
| `ein[batch, seq, model] (f32[batch: 20, seq: 384, groups: 32, per: 1, hd: 128], f32[groups: 32, per: 1, hd: 128, model: 4096])` | 257.70 | 29 413.7 | 66.8 % | 8.8 |
| `dot (f32[batch: 20, seq: 384, model: 4096], f32[4096, 32000])` | 2013.27 | 4 940.7 | 11.2 % | 407.5 |
| `ein[batch, seq, groups, per, hd] (f32[batch: 20, groups: 32, per: 1, seq: 384, seq2: 384], f32[batch: 20, seq: 384, groups: 32, hd: 128])` | 24.16 | 1 792.7 | 4.1 % | 13.5 |
| `dot (f32[batch: 20, seq: 384, 11008], f32[11008, 4096])` | 692.56 | 1 768.8 | 4.0 % | 391.5 |
| `dot (f32[batch: 20, seq: 384, 4096], f32[4096, 11008])` | 692.56 | 1 696.5 | 7.7 % (2 launches) | 408.2 |
| `ein[batch, groups, per, seq, seq2] (f32[batch: 20, seq: 384, groups: 32, per: 1, hd: 128], f32[batch: 20, seq: 384, groups: 32, hd: 128])` | 24.16 | 659.7 | 1.5 % | 36.6 |
| `dot (f32[batch: 20, seq: 384, model: 4096], f32[model: 4096, qout: 4096])` | 257.70 | 610.8 | 1.4 % | 421.9 |
| `dot (f32[batch: 20, seq: 384, model: 4096], f32[model: 4096, kvout: 4096])` | 257.70 | 608.2 | 2.8 % (2 launches) | 423.7 |
| `softmax axis seq2 where, two passes (f32[batch: 20, groups: 32, per: 1, seq: 384, seq2: 384], i64[seq: 384, seq2: 384])` | — | 103.4 | 0.2 % | — |
| `fused 2 steps (f32[batch: 20, seq: 384, 11008])` | — | 61.8 | 0.1 % | — |

## The plans

- `f32`, batch 1: a plan stating 22 nests, in 18 s.
- `int8`, batch 1: a plan stating 22 nests, in 49 s.
- `f32`, batch 20: a plan stating 22 nests, in 19 s.
- `int8`, batch 20: a plan stating 22 nests, in 50 s.

## The processes

Every command, its exit status, its wall time and its peak memory (`/usr/bin/time -v`); the native rows' wall includes the exact draw of the parameters and the warmup call.

| command | exit | wall | peak |
|---|---|---|---|
| `export` | 0 | under 1 s | 6 MB |
| `plan.f32.B1` | 0 | 18 s | 2.1 GB |
| `plan.int8.B1` | 0 | 49 s | 31.1 GB |
| `plan.f32.B20` | 0 | 19 s | 2.1 GB |
| `plan.int8.B20` | 0 | 50 s | 31.1 GB |
| `pytorch.10.B1` | 0 | 5 s | 2.2 GB |
| `pytorch.1.B1` | 0 | 20 s | 2.2 GB |
| `native.f32.default.B1` | 0 | 13 min 27 s | 29.9 GB |
| `native.f32.plan.B1` | 0 | 1 min 14 s | 29.9 GB |
| `native.int8.default.B1` | 0 | 7 min 16 s | 31.1 GB |
| `native.int8.plan.B1` | 0 | 1 min 03 s | 31.1 GB |
| `flops.B1` | 0 | 4 s | 2.2 GB |
| `pytorch.10.B20` | 0 | 44 s | 4.5 GB |
| `native.f32.default.B20` | 0 | 29 min 15 s | 33.7 GB |
| `native.f32.plan.B20` | 0 | 9 min 25 s | 33.7 GB |
| `native.int8.default.B20` | 0 | 13 min 22 s | 32.5 GB |
| `native.int8.plan.B20` | 0 | 5 min 49 s | 32.5 GB |
| `flops.B20` | 0 | 8 s | 4.0 GB |
| `kernels.f32.plan.B1` | 0 | 41 min 31 s | 37.2 GB |
| `kernels.f32.plan.B20` | 0 | 1 h 38 min 50 s | 45.3 GB |
| `hostprof.f32.plan.B1` | 0 | 1 min 21 s | 29.9 GB |

```
target/release/presk export coverage/llama.psk --target pytorch --mode readable --const G=32 --const L=1 -o out/pytorch-decoder/pt
target/release/presk plan coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 --realize f32 --profile ~/presk-data/ra4-model-0.8.0/profile.json --out out/pytorch-decoder/plans/f32-plan-B1.json
target/release/presk plan coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 --realize @out/pytorch-decoder/plans/int8_all.json --profile ~/presk-data/ra4-model-0.8.0/profile.json --out out/pytorch-decoder/plans/int8-plan-B1.json
target/release/presk plan coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=20 --dim T=384 --realize f32 --profile ~/presk-data/ra4-model-0.8.0/profile.json --out out/pytorch-decoder/plans/f32-plan-B20.json
target/release/presk plan coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=20 --dim T=384 --realize @out/pytorch-decoder/plans/int8_all.json --profile ~/presk-data/ra4-model-0.8.0/profile.json --out out/pytorch-decoder/plans/int8-plan-B20.json
taskset -c 5-9,15-19 .venv/bin/python experiments/pytorch_baseline.py forward --model llama --export out/pytorch-decoder/pt --batch 1 --seq 384 --threads 10 --warmup 1 --repeat 7
taskset -c 15 .venv/bin/python experiments/pytorch_baseline.py forward --model llama --export out/pytorch-decoder/pt --batch 1 --seq 384 --threads 1 --warmup 1 --repeat 7
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 --plan @out/pytorch-decoder/plans/f32-default.json --plan-only --native-only --warmup 1 --repeat 7 --threads 10
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 --plan @out/pytorch-decoder/plans/f32-plan-B1.json --plan-only --native-only --warmup 1 --repeat 7 --threads 10
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 --plan @out/pytorch-decoder/plans/int8-default.json --plan-only --native-only --warmup 1 --repeat 7 --threads 10
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 --plan @out/pytorch-decoder/plans/int8-plan-B1.json --plan-only --native-only --warmup 1 --repeat 7 --threads 10
taskset -c 5-9,15-19 .venv/bin/python experiments/pytorch_baseline.py flops --model llama --export out/pytorch-decoder/pt --batch 1 --seq 384
taskset -c 5-9,15-19 .venv/bin/python experiments/pytorch_baseline.py forward --model llama --export out/pytorch-decoder/pt --batch 20 --seq 384 --threads 10 --warmup 1 --repeat 7
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=20 --dim T=384 --plan @out/pytorch-decoder/plans/f32-default.json --plan-only --native-only --warmup 1 --repeat 7 --threads 10
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=20 --dim T=384 --plan @out/pytorch-decoder/plans/f32-plan-B20.json --plan-only --native-only --warmup 1 --repeat 7 --threads 10
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=20 --dim T=384 --plan @out/pytorch-decoder/plans/int8-default.json --plan-only --native-only --warmup 1 --repeat 7 --threads 10
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=20 --dim T=384 --plan @out/pytorch-decoder/plans/int8-plan-B20.json --plan-only --native-only --warmup 1 --repeat 7 --threads 10
taskset -c 5-9,15-19 .venv/bin/python experiments/pytorch_baseline.py flops --model llama --export out/pytorch-decoder/pt --batch 20 --seq 384
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 --plan @out/pytorch-decoder/plans/f32-plan-B1.json --threads 10 --repeat 7 --calibrate out/pytorch-decoder/prof/f32-plan-B1.json
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=20 --dim T=384 --plan @out/pytorch-decoder/plans/f32-plan-B20.json --threads 10 --repeat 7 --calibrate out/pytorch-decoder/prof/f32-plan-B20.json
taskset -c 5-9,15-19 target/release/presk bench coverage/llama.psk --entry Llama.apply --const G=32 --const L=1 --dim B=1 --dim T=384 --plan @out/pytorch-decoder/plans/f32-plan-B1.json --threads 10 --warmup 1 --repeat 3 --host-profile
```

## The machine and PyTorch's libraries

```
date 2026-10-07T16:34:54+02:00
presk 25dbeda, presk 0.10.1
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Vitesse maximale du processeur en MHz : 3900,0000
Nom de modèle : Cortex-A725
Vitesse maximale du processeur en MHz : 2808,0000
load 7.16 6.77 3.24
T 384, batches 1 20, repeat 7, repeat at batch 20 (native) 7, cores 5-9,15-19, one-thread core 15
int8_all.json out/llama's
later run 2026-10-07T18:03:10+02:00, presk 18d83c1, load 6.46 13.38 11.12 (the kernels alone)
later run 2026-10-07T20:24:20+02:00, presk 18d83c1, load 3.56 7.51 8.47 (the host profile, by hand, as the script's stage runs it)
later run 2026-10-07T20:28:56+02:00, presk 7f35ba3, load 0.31 3.67 6.65 (the report written again)
OpenBLAS 0.3.34 DYNAMIC_ARCH NO_AFFINITY USE_OPENMP neoversev2 MAX_THREADS=256
arm_compute_version=v53.2.0
```

## The gate, on 16 tokens

Every command of the script once at T = 16 (`T=16 REPEAT=1 experiments/pytorch_decoder.sh out/pytorch-decoder-T16`, 16:21–16:31, one call after one warmup), to size the run of record: every command exited 0, a native process held 30–31 GB, PyTorch 2.1 GB, and `presk plan` gave a plan in `f32` and in the quantised realization at both batch sizes (18 s and 47 s). Scaled to 384 tokens, the run of record came to an hour and a half to two hours, the longest row, native `f32` under the default layouts at batch 20, about thirty minutes for seven calls: so seven calls everywhere (it took 29 min 15 s).

### B = 1, T = 16

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 4 182.8 | 2.6 | 172.20 |
| 10 (ten X925, pinned) | native | plan | 165.0 | 64.7 | 6.79 |
| 10 (ten X925, pinned) | native, int8 every matrix | default | 1 989.7 | 5.4 | 81.91 |
| 10 (ten X925, pinned) | native, int8 every matrix | plan | 130.4 | 81.8 | 5.37 |
| 10 (ten X925, pinned) | PyTorch | its own | 24.3 | 439.5 | 1.00 |
| 1 (one X925, cpu 15, pinned) | PyTorch | its own | 125.3 | 85.2 | 5.16 |

### B = 20, T = 16

| threads | side | layout | median ms | GFLOP/s | ÷ PyTorch |
|---|---|---|---|---|---|
| 10 (ten X925, pinned) | native | default | 9 313.3 | 22.9 | 45.45 |
| 10 (ten X925, pinned) | native | plan | 2 744.0 | 77.8 | 13.39 |
| 10 (ten X925, pinned) | native, int8 every matrix | default | 3 887.1 | 54.9 | 18.97 |
| 10 (ten X925, pinned) | native, int8 every matrix | plan | 1 486.9 | 143.6 | 7.26 |
| 10 (ten X925, pinned) | PyTorch | its own | 204.9 | 1042.0 | 1.00 |

10.67 GFLOP at batch 1 and 213.49 at batch 20, by both counts. PyTorch's logits finite, their largest magnitude 5.98 and 7.11.

## Correctness

The rows time two draws of the same distributions, so their values are not compared. What the PyTorch side computes is checked instead:

- **The export at small constants.** `presk export coverage/llama.psk --target pytorch --mode readable --conformance-entry Llama.apply` writes `test_conformance.py`, which runs the module on presk's own parameters and inputs and compares its outputs with the reference interpreter's. It passes (1 test) at `--const V=100 --const D=32 --const H=2 --const G=2 --const FF=64 --const L=2 --dim B=2 --dim T=16`, and at `--const V=1000 --const D=256 --const H=4 --const G=4 --const FF=688 --const L=1 --dim B=2 --dim T=32`, whose conformance file is 52 MB (`out/pytorch-decoder-conformance/`). At the 7B width the file would print every parameter, so the module timed is the same printer's at other constants.
- **The logits at 384 tokens.** On PyTorch's draw they are finite, their largest magnitude 6.84 at batch 1 and 7.84 at batch 20; on presk's own draw, the exact logits' largest magnitude is 6.88 at batch 1 (D104). The same order, as two draws of one distribution give.
- **The FLOPs.** PyTorch's counter gives 258 503 344 128 a sequence, the compiler's count of the forward pass's contractions exactly: the module computes the products presk's program states, no more.
- **The native rows.** Every point of the layout space computes the same bits as the default layout (statement L, 0.7.0, checked on the RA4 sweeps' nests by `make ra4-check` and its successors); it is not checked again on this block, whose rows are timed and not compared.
- **`pytorch_baseline.py`'s course GPT.** Its forward pass and FLOPs on D100's checkpoint and inputs, one call before and after the decoder was added: the logits byte for byte, the FLOP count and D100's report from `out/pytorch-baseline` identical (`out/pytorch-decoder-regress/`).

## What these rows are not

PyTorch eager on the CPU, as practitioners run it without a compiler: no `torch.compile`, no GPU, no other library. The forward pass only, no training step, since at this width no budget can be discharged on a step without the exact step. One block of 32, at random ids, on initializations: the head and the token table are 56 % of these parameters and 4 % of the 7B's. PyTorch's parameters are its own draw of the distributions presk draws from, not presk's values. Neither side is the C program. Nothing here is a performance claim beyond the rows measured.

## PyTorch's configuration

`out/pytorch-decoder/torch-config.txt`, the build's compiler flags left out:

```
torch 2.14.0+cpu
PyTorch built with:
  - GCC 13.3
  - C++ Version: 202002
  - Intel(R) MKL-DNN v3.12.0 (Git Hash 80afa71049cd69a3df32adcccb623b12cd7baa22)
  - OpenMP 201511 (a.k.a. OpenMP 4.5)
  - LAPACK is enabled (usually provided by MKL)
  - NNPACK is enabled
  - CPU capability usage: SVE128

ATen/Parallel:
	at::get_num_threads() : 20
	at::get_num_interop_threads() : 20
OpenMP 201511 (a.k.a. OpenMP 4.5)
	omp_get_max_threads() : 20
Intel(R) MKL-DNN v3.12.0 (Git Hash 80afa71049cd69a3df32adcccb623b12cd7baa22)
std::thread::hardware_concurrency() : 20
Environment variables:
	OMP_NUM_THREADS : [not set]
ATen parallel backend: OpenMP

```
