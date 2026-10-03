> The native host measured at 0.10.0 step 5 (D95: `make host-profile` on presk `163a61f`, begun 2026-10-01 at 12:52, the run of record), copied from `out/host-profile/report.md`. The plans are made afresh by the script with the frozen profile, as `make pytorch-baseline` makes them; the C side to compare with is step 4b's baseline, `docs/pytorch-baseline-0.10.0-step-4b.md` (D94). Every call with the timers gave the same bits as a call without them, and every training run, with or without the timers, in each configuration, ended on the same weights byte for byte.

# The native host, measured (plan 0.10.0 step 5)

```
date 2026-10-01T12:52:25+02:00
presk 163a61f, presk 0.10.0-dev
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Nom de modèle : Cortex-A725
load 1.62 2.22 2.05
```

The course GPT at `B = 16`, `T = 256`, `f32`, under the plans the frozen profile gives; each phase's mean a call (a step for the training step) in ms, and its share of the call. `kernels` is the kernels' own code; groups 1 to 6 are the host (plan 0.10.0 §3.3), `left.interpreter` by difference.

## 10 threads, ten X925 cores, pinned

| phase | forward pass | gradient | `adamw.step` | training step | what |
|---|---:|---:|---:|---:|---|
| **kernels** | 341.4 (51.4 %) | 1449.6 (17.4 %) | 21.8 (10.8 %) | 1452.1 (18.2 %) | the kernels' code |
| **find** | 2.1 (0.3 %) | 9.3 (0.1 %) | 0.6 (0.3 %) | 18.9 (0.2 %) | 1. finding the kernel |
| &nbsp;&nbsp;find.site | 0.2 (0.0 %) | 1.6 (0.0 %) | 0.0 (0.0 %) | 1.6 (0.0 %) | a let's site named, its type found |
| &nbsp;&nbsp;find.groups | 0.2 (0.0 %) | 0.5 (0.0 %) | 0.0 (0.0 %) | 9.6 (0.1 %) | a function's fusion groups cloned at its call |
| &nbsp;&nbsp;find.layout | 0.5 (0.1 %) | 2.1 (0.0 %) | 0.1 (0.0 %) | 2.1 (0.0 %) | a nest's kind, its layout in the plan |
| &nbsp;&nbsp;find.key | 1.2 (0.2 %) | 5.2 (0.1 %) | 0.5 (0.2 %) | 5.7 (0.1 %) | the cache key formatted, the kernel looked up |
| &nbsp;&nbsp;find.lower | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | a kernel lowered and compiled (first launch) |
| **buffers** | 27.0 (4.1 %) | 594.9 (7.1 %) | 16.6 (8.2 %) | 421.8 (5.3 %) | 2. buffers allocated, converted, freed |
| &nbsp;&nbsp;buffers.alloc | 26.5 (4.0 %) | 59.9 (0.7 %) | 1.6 (0.8 %) | 64.4 (0.8 %) | output buffers allocated |
| &nbsp;&nbsp;buffers.convert | 0.1 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.0 %) | 0.4 (0.0 %) | inputs converted to the kernel's type |
| &nbsp;&nbsp;buffers.free | 0.5 (0.1 %) | 534.6 (6.4 %) | 15.0 (7.4 %) | 357.1 (4.5 %) | values freed when their function returns |
| **views** | 3.8 (0.6 %) | 691.3 (8.3 %) | 0.0 (0.0 %) | 716.0 (9.0 %) | 3. stacked trees sliced, views materialized |
| &nbsp;&nbsp;views.slice | 3.8 (0.6 %) | 691.3 (8.3 %) | 0.0 (0.0 %) | 716.0 (9.0 %) | stacked layers sliced, scans' results stacked |
| &nbsp;&nbsp;views.axes | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | axes merged, split, reshaped (copied) |
| **copies** | 212.7 (32.0 %) | 5326.8 (63.8 %) | 151.3 (75.1 %) | 4212.6 (52.8 %) | 4. arguments, tuples, fields, accumulators copied |
| &nbsp;&nbsp;copies.entry | 6.2 (0.9 %) | 6.7 (0.1 %) | 27.2 (13.5 %) | 29.9 (0.4 %) | the entry's arguments copied and typed |
| &nbsp;&nbsp;copies.operands | 89.4 (13.5 %) | 501.4 (6.0 %) | 16.1 (8.0 %) | 497.7 (6.2 %) | a launch's operands copied from the environment |
| &nbsp;&nbsp;copies.as-tensor | 86.7 (13.1 %) | 716.6 (8.6 %) | 41.4 (20.6 %) | 673.2 (8.4 %) | the same operands copied again as tensors |
| &nbsp;&nbsp;copies.args | 19.3 (2.9 %) | 978.2 (11.7 %) | 0.0 (0.0 %) | 508.0 (6.4 %) | a call's arguments, a closure's captures |
| &nbsp;&nbsp;copies.tuples | 9.2 (1.4 %) | 2582.3 (30.9 %) | 66.6 (33.0 %) | 1939.4 (24.3 %) | tuples, records, fields, a tree's leaves |
| &nbsp;&nbsp;copies.fold | 1.9 (0.3 %) | 541.6 (6.5 %) | 0.0 (0.0 %) | 564.2 (7.1 %) | a fold's arguments, a scan's accumulators |
| **threads** | 43.1 (6.5 %) | 125.9 (1.5 %) | 10.8 (5.3 %) | 138.5 (1.7 %) | 5. a launch dispatched, its threads started and joined |
| &nbsp;&nbsp;threads.dispatch | 1.2 (0.2 %) | 3.9 (0.1 %) | 0.1 (0.1 %) | 4.0 (0.1 %) | the executor's dispatch of a launch |
| &nbsp;&nbsp;threads.start-join | 41.9 (6.3 %) | 122.0 (1.5 %) | 10.6 (5.3 %) | 134.5 (1.7 %) | a divided launch's threads started and joined |
| **left** | 33.8 (5.1 %) | 148.4 (1.8 %) | 0.4 (0.2 %) | 1013.6 (12.7 %) | 6. what is left |
| &nbsp;&nbsp;left.dropout | 31.8 (4.8 %) | 31.0 (0.4 %) | 0.0 (0.0 %) | 890.2 (11.2 %) | dropout masks drawn on the host |
| &nbsp;&nbsp;left.host-prims | 0.0 (0.0 %) | 109.2 (1.3 %) | 0.0 (0.0 %) | 105.0 (1.3 %) | other primitives the host computes |
| &nbsp;&nbsp;left.interpreter | 1.9 (0.3 %) | 8.2 (0.1 %) | 0.4 (0.2 %) | 18.4 (0.2 %) | the host program's interpreter, by difference |
| **total** | 664.0 (100.0 %) | 8346.1 (100.0 %) | 201.5 (100.0 %) | 7973.5 (100.0 %) | a call, timed around it |

- forward pass: presk bench --host-profile: `GPT.apply` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- forward pass: -- a call: 657.054 ms without the timers (median; 639.391–704.290), 658.066 ms with them (653.570–686.813): the timers' cost +1.012 ms, +0.15 %
- forward pass: -- the outputs of a call with the timers and of one without: the same bits
- forward pass: -- without the timers, 1 minor page faults a call
- forward pass: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- forward pass: -- 228 launches a call, 201 divided among 2010 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 384.498 ms, the host 279.460 ms
- gradient: presk bench --host-profile: `loss_grad` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- gradient: -- a call: 8493.081 ms without the timers (median; 7386.096–8823.872), 8509.559 ms with them (7511.938–9107.842): the timers' cost +16.478 ms, +0.19 %
- gradient: -- the outputs of a call with the timers and of one without: the same bits
- gradient: -- without the timers, 3327625 minor page faults a call
- gradient: -- 3252022 minor page faults a call (pages touched for the first time, 13320.3 MB of 4 KB pages)
- gradient: -- 1061 launches a call, 569 divided among 5626 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1575.437 ms, the host 6770.689 ms
- `adamw.step`: presk bench --host-profile: `adamw.step` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- `adamw.step`: -- a call: 210.001 ms without the timers (median; 175.377–214.688), 205.746 ms with them (187.458–212.013): the timers' cost -4.255 ms, -2.03 %
- `adamw.step`: -- the outputs of a call with the timers and of one without: the same bits
- `adamw.step`: -- without the timers, 73408 minor page faults a call
- `adamw.step`: -- 73408 minor page faults a call (pages touched for the first time, 300.7 MB of 4 KB pages)
- `adamw.step`: -- 111 launches a call, 63 divided among 438 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 32.585 ms, the host 168.914 ms
- training step: presk train --host-profile: the host by phase, the mean of steps 2 to 8 (the first left out)
- training step: -- 2232313 minor page faults a step (pages touched for the first time, 9143.6 MB of 4 KB pages)
- training step: -- 1174 launches a step, 632 divided among 6064 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1590.577 ms, the host 6382.895 ms
- training step: presk train: 1181 kernel launches a step, 2013 without fusion (41% removed)
- training step: presk train: 7937.4 ms a step, median of 7 (the first excluded)
- training step: presk train: 8 steps of GPT, final loss 4.493265, weights in out/host-profile/train/profiled-10/final.psk.d
- training step without the timers: 7847.8 ms a step (the median, the first left out)

## 20 threads, every core, unpinned

| phase | forward pass | gradient | `adamw.step` | training step | what |
|---|---:|---:|---:|---:|---|
| **kernels** | 754.2 (67.3 %) | 2692.8 (29.1 %) | 27.4 (12.7 %) | 2708.1 (28.8 %) | the kernels' code |
| **find** | 3.1 (0.3 %) | 10.7 (0.1 %) | 0.7 (0.3 %) | 11.4 (0.1 %) | 1. finding the kernel |
| &nbsp;&nbsp;find.site | 0.5 (0.0 %) | 2.1 (0.0 %) | 0.0 (0.0 %) | 2.0 (0.0 %) | a let's site named, its type found |
| &nbsp;&nbsp;find.groups | 0.2 (0.0 %) | 0.5 (0.0 %) | 0.0 (0.0 %) | 0.5 (0.0 %) | a function's fusion groups cloned at its call |
| &nbsp;&nbsp;find.layout | 0.6 (0.1 %) | 2.3 (0.0 %) | 0.1 (0.0 %) | 2.4 (0.0 %) | a nest's kind, its layout in the plan |
| &nbsp;&nbsp;find.key | 1.8 (0.2 %) | 5.9 (0.1 %) | 0.6 (0.3 %) | 6.5 (0.1 %) | the cache key formatted, the kernel looked up |
| &nbsp;&nbsp;find.lower | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | a kernel lowered and compiled (first launch) |
| **buffers** | 31.7 (2.8 %) | 503.3 (5.5 %) | 20.2 (9.4 %) | 400.3 (4.3 %) | 2. buffers allocated, converted, freed |
| &nbsp;&nbsp;buffers.alloc | 31.0 (2.8 %) | 58.9 (0.6 %) | 1.6 (0.8 %) | 67.4 (0.7 %) | output buffers allocated |
| &nbsp;&nbsp;buffers.convert | 0.1 (0.0 %) | 0.4 (0.0 %) | 0.0 (0.0 %) | 0.4 (0.0 %) | inputs converted to the kernel's type |
| &nbsp;&nbsp;buffers.free | 0.5 (0.1 %) | 444.1 (4.8 %) | 18.6 (8.6 %) | 332.4 (3.5 %) | values freed when their function returns |
| **views** | 4.0 (0.4 %) | 715.1 (7.7 %) | 0.0 (0.0 %) | 698.2 (7.4 %) | 3. stacked trees sliced, views materialized |
| &nbsp;&nbsp;views.slice | 4.0 (0.4 %) | 715.1 (7.7 %) | 0.0 (0.0 %) | 698.2 (7.4 %) | stacked layers sliced, scans' results stacked |
| &nbsp;&nbsp;views.axes | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | axes merged, split, reshaped (copied) |
| **copies** | 235.8 (21.0 %) | 4996.4 (54.1 %) | 150.9 (69.8 %) | 4345.3 (46.2 %) | 4. arguments, tuples, fields, accumulators copied |
| &nbsp;&nbsp;copies.entry | 7.0 (0.6 %) | 7.8 (0.1 %) | 27.3 (12.6 %) | 39.5 (0.4 %) | the entry's arguments copied and typed |
| &nbsp;&nbsp;copies.operands | 98.9 (8.8 %) | 497.1 (5.4 %) | 16.0 (7.4 %) | 487.7 (5.2 %) | a launch's operands copied from the environment |
| &nbsp;&nbsp;copies.as-tensor | 96.8 (8.6 %) | 712.5 (7.7 %) | 35.5 (16.4 %) | 635.0 (6.8 %) | the same operands copied again as tensors |
| &nbsp;&nbsp;copies.args | 20.9 (1.9 %) | 785.0 (8.5 %) | 0.0 (0.0 %) | 538.8 (5.7 %) | a call's arguments, a closure's captures |
| &nbsp;&nbsp;copies.tuples | 10.2 (0.9 %) | 2414.6 (26.1 %) | 72.2 (33.4 %) | 2069.3 (22.0 %) | tuples, records, fields, a tree's leaves |
| &nbsp;&nbsp;copies.fold | 2.1 (0.2 %) | 579.3 (6.3 %) | 0.0 (0.0 %) | 575.1 (6.1 %) | a fold's arguments, a scan's accumulators |
| **threads** | 53.9 (4.8 %) | 171.5 (1.9 %) | 16.4 (7.6 %) | 195.7 (2.1 %) | 5. a launch dispatched, its threads started and joined |
| &nbsp;&nbsp;threads.dispatch | 1.3 (0.1 %) | 4.2 (0.1 %) | 0.2 (0.1 %) | 4.1 (0.0 %) | the executor's dispatch of a launch |
| &nbsp;&nbsp;threads.start-join | 52.5 (4.7 %) | 167.3 (1.8 %) | 16.2 (7.5 %) | 191.6 (2.0 %) | a divided launch's threads started and joined |
| **left** | 38.2 (3.4 %) | 148.6 (1.6 %) | 0.5 (0.2 %) | 1039.9 (11.1 %) | 6. what is left |
| &nbsp;&nbsp;left.dropout | 35.6 (3.2 %) | 31.4 (0.3 %) | 0.0 (0.0 %) | 897.1 (9.5 %) | dropout masks drawn on the host |
| &nbsp;&nbsp;left.host-prims | 0.0 (0.0 %) | 107.9 (1.2 %) | 0.0 (0.0 %) | 111.4 (1.2 %) | other primitives the host computes |
| &nbsp;&nbsp;left.interpreter | 2.6 (0.2 %) | 9.4 (0.1 %) | 0.5 (0.2 %) | 31.4 (0.3 %) | the host program's interpreter, by difference |
| **total** | 1120.8 (100.0 %) | 9238.4 (100.0 %) | 216.2 (100.0 %) | 9398.9 (100.0 %) | a call, timed around it |

- forward pass: presk bench --host-profile: `GPT.apply` on the native backend, f32 under the plan, 20 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- forward pass: -- a call: 1122.580 ms without the timers (median; 1115.888–1135.320), 1121.775 ms with them (1113.579–1126.465): the timers' cost -0.805 ms, -0.07 %
- forward pass: -- the outputs of a call with the timers and of one without: the same bits
- forward pass: -- without the timers, 0 minor page faults a call
- forward pass: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- forward pass: -- 228 launches a call, 201 divided among 2010 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 808.058 ms, the host 312.780 ms
- gradient: presk bench --host-profile: `loss_grad` on the native backend, f32 under the plan, 20 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- gradient: -- a call: 9359.425 ms without the timers (median; 8218.122–10250.094), 8968.122 ms with them (8394.370–10359.950): the timers' cost -391.303 ms, -4.18 %
- gradient: -- the outputs of a call with the timers and of one without: the same bits
- gradient: -- without the timers, 2690967 minor page faults a call
- gradient: -- 2649256 minor page faults a call (pages touched for the first time, 10851.4 MB of 4 KB pages)
- gradient: -- 1061 launches a call, 569 divided among 5626 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 2864.245 ms, the host 6374.155 ms
- `adamw.step`: presk bench --host-profile: `adamw.step` on the native backend, f32 under the plan, 20 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- `adamw.step`: -- a call: 216.666 ms without the timers (median; 201.237–224.145), 216.820 ms with them (200.976–226.153): the timers' cost +0.154 ms, +0.07 %
- `adamw.step`: -- the outputs of a call with the timers and of one without: the same bits
- `adamw.step`: -- without the timers, 72545 minor page faults a call
- `adamw.step`: -- 72545 minor page faults a call (pages touched for the first time, 297.1 MB of 4 KB pages)
- `adamw.step`: -- 111 launches a call, 63 divided among 438 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 43.831 ms, the host 172.353 ms
- training step: presk train --host-profile: the host by phase, the mean of steps 2 to 8 (the first left out)
- training step: -- 2148426 minor page faults a step (pages touched for the first time, 8800.0 MB of 4 KB pages)
- training step: -- 1174 launches a step, 632 divided among 6064 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 2903.739 ms, the host 6495.116 ms
- training step: presk train: 1181 kernel launches a step, 2013 without fusion (41% removed)
- training step: presk train: 9486.7 ms a step, median of 7 (the first excluded)
- training step: presk train: 8 steps of GPT, final loss 4.493265, weights in out/host-profile/train/profiled-20/final.psk.d
- training step without the timers: 9481.6 ms a step (the median, the first left out)

## 10 threads, ten X925 cores, pinned, glibc keeping its pages (`GLIBC_TUNABLES`: no block mapped apart, none returned)

| phase | forward pass | gradient | `adamw.step` | training step | what |
|---|---:|---:|---:|---:|---|
| **kernels** | 337.9 (51.1 %) | 1316.1 (26.2 %) | 14.6 (11.5 %) | 1330.1 (23.7 %) | the kernels' code |
| **find** | 2.2 (0.3 %) | 9.1 (0.2 %) | 0.4 (0.3 %) | 9.6 (0.2 %) | 1. finding the kernel |
| &nbsp;&nbsp;find.site | 0.3 (0.0 %) | 1.5 (0.0 %) | 0.0 (0.0 %) | 1.5 (0.0 %) | a let's site named, its type found |
| &nbsp;&nbsp;find.groups | 0.2 (0.0 %) | 0.5 (0.0 %) | 0.0 (0.0 %) | 0.5 (0.0 %) | a function's fusion groups cloned at its call |
| &nbsp;&nbsp;find.layout | 0.5 (0.1 %) | 2.0 (0.0 %) | 0.0 (0.0 %) | 2.0 (0.0 %) | a nest's kind, its layout in the plan |
| &nbsp;&nbsp;find.key | 1.3 (0.2 %) | 5.2 (0.1 %) | 0.3 (0.3 %) | 5.6 (0.1 %) | the cache key formatted, the kernel looked up |
| &nbsp;&nbsp;find.lower | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | a kernel lowered and compiled (first launch) |
| **buffers** | 27.5 (4.2 %) | 76.7 (1.5 %) | 2.6 (2.0 %) | 77.9 (1.4 %) | 2. buffers allocated, converted, freed |
| &nbsp;&nbsp;buffers.alloc | 27.0 (4.1 %) | 72.1 (1.4 %) | 2.5 (1.9 %) | 73.6 (1.3 %) | output buffers allocated |
| &nbsp;&nbsp;buffers.convert | 0.1 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.0 %) | 0.4 (0.0 %) | inputs converted to the kernel's type |
| &nbsp;&nbsp;buffers.free | 0.5 (0.1 %) | 4.3 (0.1 %) | 0.1 (0.1 %) | 4.0 (0.1 %) | values freed when their function returns |
| **views** | 3.8 (0.6 %) | 590.3 (11.8 %) | 0.0 (0.0 %) | 572.7 (10.2 %) | 3. stacked trees sliced, views materialized |
| &nbsp;&nbsp;views.slice | 3.8 (0.6 %) | 590.3 (11.8 %) | 0.0 (0.0 %) | 572.7 (10.2 %) | stacked layers sliced, scans' results stacked |
| &nbsp;&nbsp;views.axes | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | axes merged, split, reshaped (copied) |
| **copies** | 215.4 (32.6 %) | 2817.3 (56.1 %) | 101.0 (79.5 %) | 2529.4 (45.1 %) | 4. arguments, tuples, fields, accumulators copied |
| &nbsp;&nbsp;copies.entry | 6.1 (0.9 %) | 6.4 (0.1 %) | 26.7 (21.0 %) | 23.6 (0.4 %) | the entry's arguments copied and typed |
| &nbsp;&nbsp;copies.operands | 91.0 (13.8 %) | 397.9 (7.9 %) | 15.8 (12.5 %) | 412.7 (7.4 %) | a launch's operands copied from the environment |
| &nbsp;&nbsp;copies.as-tensor | 88.0 (13.3 %) | 312.6 (6.2 %) | 12.2 (9.6 %) | 323.0 (5.8 %) | the same operands copied again as tensors |
| &nbsp;&nbsp;copies.args | 18.9 (2.9 %) | 376.1 (7.5 %) | 0.0 (0.0 %) | 303.6 (5.4 %) | a call's arguments, a closure's captures |
| &nbsp;&nbsp;copies.tuples | 9.4 (1.4 %) | 1594.3 (31.8 %) | 46.2 (36.4 %) | 1335.0 (23.8 %) | tuples, records, fields, a tree's leaves |
| &nbsp;&nbsp;copies.fold | 1.9 (0.3 %) | 130.1 (2.6 %) | 0.0 (0.0 %) | 131.5 (2.3 %) | a fold's arguments, a scan's accumulators |
| **threads** | 40.2 (6.1 %) | 122.5 (2.4 %) | 8.2 (6.4 %) | 133.0 (2.4 %) | 5. a launch dispatched, its threads started and joined |
| &nbsp;&nbsp;threads.dispatch | 1.2 (0.2 %) | 4.0 (0.1 %) | 0.1 (0.1 %) | 3.9 (0.1 %) | the executor's dispatch of a launch |
| &nbsp;&nbsp;threads.start-join | 39.0 (5.9 %) | 118.5 (2.4 %) | 8.1 (6.4 %) | 129.1 (2.3 %) | a divided launch's threads started and joined |
| **left** | 34.4 (5.2 %) | 87.9 (1.8 %) | 0.3 (0.2 %) | 956.5 (17.1 %) | 6. what is left |
| &nbsp;&nbsp;left.dropout | 32.4 (4.9 %) | 31.5 (0.6 %) | 0.0 (0.0 %) | 890.8 (15.9 %) | dropout masks drawn on the host |
| &nbsp;&nbsp;left.host-prims | 0.0 (0.0 %) | 48.4 (1.0 %) | 0.0 (0.0 %) | 48.0 (0.9 %) | other primitives the host computes |
| &nbsp;&nbsp;left.interpreter | 2.0 (0.3 %) | 8.0 (0.2 %) | 0.3 (0.2 %) | 17.6 (0.3 %) | the host program's interpreter, by difference |
| **total** | 661.3 (100.0 %) | 5019.9 (100.0 %) | 127.1 (100.0 %) | 5609.2 (100.0 %) | a call, timed around it |

- forward pass: presk bench --host-profile: `GPT.apply` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- forward pass: -- a call: 666.761 ms without the timers (median; 655.374–668.876), 659.250 ms with them (655.993–672.822): the timers' cost -7.511 ms, -1.13 %
- forward pass: -- the outputs of a call with the timers and of one without: the same bits
- forward pass: -- without the timers, 1 minor page faults a call
- forward pass: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- forward pass: -- 228 launches a call, 201 divided among 2010 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 378.079 ms, the host 283.238 ms
- gradient: presk bench --host-profile: `loss_grad` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- gradient: -- a call: 4925.371 ms without the timers (median; 4822.015–5040.704), 5033.554 ms with them (4952.897–5059.762): the timers' cost +108.183 ms, +2.20 %
- gradient: -- the outputs of a call with the timers and of one without: the same bits
- gradient: -- without the timers, 1146 minor page faults a call
- gradient: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- gradient: -- 1061 launches a call, 569 divided among 5626 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1438.556 ms, the host 3581.341 ms
- `adamw.step`: presk bench --host-profile: `adamw.step` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- `adamw.step`: -- a call: 132.803 ms without the timers (median; 117.554–135.252), 132.079 ms with them (117.925–133.762): the timers' cost -0.723 ms, -0.54 %
- `adamw.step`: -- the outputs of a call with the timers and of one without: the same bits
- `adamw.step`: -- without the timers, 1 minor page faults a call
- `adamw.step`: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- `adamw.step`: -- 111 launches a call, 63 divided among 438 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 22.741 ms, the host 104.326 ms
- training step: presk train --host-profile: the host by phase, the mean of steps 2 to 8 (the first left out)
- training step: -- 1030 minor page faults a step (pages touched for the first time, 4.2 MB of 4 KB pages)
- training step: -- 1174 launches a step, 632 divided among 6064 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1463.059 ms, the host 4146.177 ms
- training step: presk train: 1181 kernel launches a step, 2013 without fusion (41% removed)
- training step: presk train: 5597.4 ms a step, median of 7 (the first excluded)
- training step: presk train: 8 steps of GPT, final loss 4.493265, weights in out/host-profile/train/profiled-10-heap/final.psk.d
- training step without the timers: 5581.8 ms a step (the median, the first left out)

