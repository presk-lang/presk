> The native host measured at 0.10.0 step 6, after its fixes (D96: `make host-profile` on presk `ceade37`, begun 2026-10-01 at 15:59, the run of record, the three configurations), copied from `out/host-profile/report.md`. The plans are made afresh by the script with the frozen profile, as `make pytorch-baseline` makes them. Step 5's report, before the fixes, is `docs/host-profile-0.10.0-step-5.md`; each fix measured apart, pinned, against `main`'s binary of the same day, is in D96. Every call with the timers gave the same bits as a call without them, and every training run, with or without the timers, in each configuration, ended on the same weights byte for byte.

# The native host, measured (plan 0.10.0)

```
date 2026-10-01T15:59:01+02:00
presk ceade37, presk 0.10.0-dev, target/release/presk
configurations 10 20 10-heap
kernel Linux 6.17.0-1021-nvidia, 20 cores
Nom de modèle : Cortex-X925
Nom de modèle : Cortex-A725
load 3.71 7.40 8.22
```

The course GPT at `B = 16`, `T = 256`, `f32`, under the plans the frozen profile gives; each phase's mean a call (a step for the training step) in ms, and its share of the call. `kernels` is the kernels' own code; groups 1 to 6 are the host (plan 0.10.0 §3.3), `left.interpreter` by difference.

## 10 threads, ten X925 cores, pinned

| phase | forward pass | gradient | `adamw.step` | training step | what |
|---|---:|---:|---:|---:|---|
| **kernels** | 328.5 (98.8 %) | 1281.0 (97.3 %) | 15.0 (93.0 %) | 1302.4 (91.0 %) | the kernels' code |
| **find** | 1.2 (0.4 %) | 4.4 (0.3 %) | 0.2 (1.3 %) | 4.8 (0.3 %) | 1. finding the kernel |
| &nbsp;&nbsp;find.site | 0.1 (0.0 %) | 0.6 (0.0 %) | 0.0 (0.0 %) | 0.6 (0.0 %) | a let's site named, its type found |
| &nbsp;&nbsp;find.groups | 0.1 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.1 %) | 0.3 (0.0 %) | a function's fusion groups cloned at its call |
| &nbsp;&nbsp;find.layout | 0.2 (0.1 %) | 0.9 (0.1 %) | 0.0 (0.1 %) | 0.9 (0.1 %) | a nest's kind, its layout in the plan |
| &nbsp;&nbsp;find.key | 0.7 (0.2 %) | 2.6 (0.2 %) | 0.2 (1.1 %) | 2.9 (0.2 %) | the cache key formatted, the kernel looked up |
| &nbsp;&nbsp;find.lower | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | a kernel lowered and compiled (first launch) |
| **buffers** | 0.2 (0.1 %) | 1.8 (0.1 %) | 0.1 (0.5 %) | 2.3 (0.2 %) | 2. buffers allocated, converted, freed |
| &nbsp;&nbsp;buffers.alloc | 0.1 (0.0 %) | 0.6 (0.1 %) | 0.0 (0.2 %) | 1.1 (0.1 %) | output buffers allocated |
| &nbsp;&nbsp;buffers.convert | 0.0 (0.0 %) | 0.2 (0.0 %) | 0.0 (0.1 %) | 0.2 (0.0 %) | inputs converted to the kernel's type |
| &nbsp;&nbsp;buffers.free | 0.1 (0.0 %) | 1.0 (0.1 %) | 0.0 (0.3 %) | 1.0 (0.1 %) | values freed when their function returns |
| **views** | 0.0 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.0 %) | 0.3 (0.0 %) | 3. stacked trees sliced, views materialized |
| &nbsp;&nbsp;views.slice | 0.0 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.0 %) | 0.3 (0.0 %) | stacked layers sliced, scans' results stacked |
| &nbsp;&nbsp;views.axes | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | axes merged, split, reshaped (copied) |
| **copies** | 0.4 (0.1 %) | 3.0 (0.2 %) | 0.2 (1.0 %) | 3.2 (0.2 %) | 4. arguments, tuples, fields, accumulators (copied until step 6) |
| &nbsp;&nbsp;copies.entry | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.2 %) | 0.1 (0.0 %) | the entry's arguments typed |
| &nbsp;&nbsp;copies.operands | 0.2 (0.1 %) | 1.2 (0.1 %) | 0.0 (0.3 %) | 1.2 (0.1 %) | a launch's operands read from the environment |
| &nbsp;&nbsp;copies.as-tensor | 0.1 (0.0 %) | 0.5 (0.0 %) | 0.0 (0.1 %) | 0.5 (0.0 %) | the same operands taken again as tensors |
| &nbsp;&nbsp;copies.args | 0.0 (0.0 %) | 0.2 (0.0 %) | 0.0 (0.0 %) | 0.2 (0.0 %) | a call's arguments, a closure's captures |
| &nbsp;&nbsp;copies.tuples | 0.1 (0.0 %) | 1.0 (0.1 %) | 0.1 (0.4 %) | 1.1 (0.1 %) | tuples, records, fields, a tree's leaves |
| &nbsp;&nbsp;copies.fold | 0.0 (0.0 %) | 0.1 (0.0 %) | 0.0 (0.0 %) | 0.1 (0.0 %) | a fold's arguments, a scan's accumulators |
| **threads** | 0.9 (0.3 %) | 21.6 (1.6 %) | 0.5 (3.1 %) | 25.7 (1.8 %) | 5. a launch dispatched, its threads started and joined |
| &nbsp;&nbsp;threads.dispatch | 0.6 (0.2 %) | 4.0 (0.3 %) | 0.0 (0.2 %) | 6.3 (0.4 %) | the executor's dispatch of a launch |
| &nbsp;&nbsp;threads.start-join | 0.3 (0.1 %) | 17.6 (1.3 %) | 0.5 (2.9 %) | 19.3 (1.4 %) | a divided launch's threads started and joined |
| **left** | 1.2 (0.4 %) | 4.9 (0.4 %) | 0.2 (1.0 %) | 92.3 (6.5 %) | 6. what is left |
| &nbsp;&nbsp;left.dropout | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 79.0 (5.5 %) | dropout masks drawn on the host |
| &nbsp;&nbsp;left.host-prims | 0.0 (0.0 %) | 0.2 (0.0 %) | 0.0 (0.0 %) | 0.2 (0.0 %) | other primitives the host computes |
| &nbsp;&nbsp;left.interpreter | 1.2 (0.3 %) | 4.7 (0.3 %) | 0.2 (1.0 %) | 13.0 (0.9 %) | the host program's interpreter, by difference |
| **total** | 332.5 (100.0 %) | 1317.0 (100.0 %) | 16.1 (100.0 %) | 1431.0 (100.0 %) | a call, timed around it |

- forward pass: presk bench --host-profile: `GPT.apply` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- forward pass: -- a call: 331.608 ms without the timers (median; 331.059–332.901), 332.377 ms with them (332.143–333.097): the timers' cost +0.769 ms, +0.23 %
- forward pass: -- the outputs of a call with the timers and of one without: the same bits
- forward pass: -- without the timers, 0 minor page faults a call
- forward pass: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- forward pass: -- 228 launches a call, 201 divided among 2010 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 329.440 ms, the host 3.058 ms
- gradient: presk bench --host-profile: `loss_grad` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- gradient: -- a call: 1314.790 ms without the timers (median; 1312.459–1317.294), 1316.683 ms with them (1316.062–1318.273): the timers' cost +1.892 ms, +0.14 %
- gradient: -- the outputs of a call with the timers and of one without: the same bits
- gradient: -- without the timers, 0 minor page faults a call
- gradient: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- gradient: -- 1061 launches a call, 569 divided among 5626 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1302.561 ms, the host 14.442 ms
- `adamw.step`: presk bench --host-profile: `adamw.step` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- `adamw.step`: -- a call: 15.845 ms without the timers (median; 15.785–15.940), 16.104 ms with them (15.912–16.172): the timers' cost +0.259 ms, +1.64 %
- `adamw.step`: -- the outputs of a call with the timers and of one without: the same bits
- `adamw.step`: -- without the timers, 0 minor page faults a call
- `adamw.step`: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- `adamw.step`: -- 111 launches a call, 63 divided among 438 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 15.476 ms, the host 0.615 ms
- training step: presk train --host-profile: the host by phase, the mean of steps 2 to 8 (the first left out)
- training step: -- 6048 minor page faults a step (pages touched for the first time, 24.8 MB of 4 KB pages)
- training step: -- 1174 launches a step, 632 divided among 6064 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1328.086 ms, the host 102.892 ms
- training step: presk train: 1181 kernel launches a step, 2013 without fusion (41% removed)
- training step: presk train: 1427.7 ms a step, median of 7 (the first excluded)
- training step: presk train: 8 steps of GPT, final loss 4.493265, weights in out/host-profile/train/profiled-10/final.psk.d
- training step without the timers: 1421.2 ms a step (the median, the first left out)

## 20 threads, every core, unpinned

| phase | forward pass | gradient | `adamw.step` | training step | what |
|---|---:|---:|---:|---:|---|
| **kernels** | 328.0 (98.8 %) | 1285.4 (97.3 %) | 20.2 (92.7 %) | 1304.8 (90.7 %) | the kernels' code |
| **find** | 1.2 (0.3 %) | 4.4 (0.3 %) | 0.2 (1.0 %) | 5.2 (0.4 %) | 1. finding the kernel |
| &nbsp;&nbsp;find.site | 0.1 (0.0 %) | 0.6 (0.0 %) | 0.0 (0.0 %) | 0.6 (0.0 %) | a let's site named, its type found |
| &nbsp;&nbsp;find.groups | 0.1 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.1 %) | 0.3 (0.0 %) | a function's fusion groups cloned at its call |
| &nbsp;&nbsp;find.layout | 0.2 (0.1 %) | 0.9 (0.1 %) | 0.0 (0.1 %) | 1.0 (0.1 %) | a nest's kind, its layout in the plan |
| &nbsp;&nbsp;find.key | 0.7 (0.2 %) | 2.6 (0.2 %) | 0.2 (0.8 %) | 3.2 (0.2 %) | the cache key formatted, the kernel looked up |
| &nbsp;&nbsp;find.lower | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | a kernel lowered and compiled (first launch) |
| **buffers** | 0.3 (0.1 %) | 1.8 (0.1 %) | 0.1 (0.4 %) | 2.4 (0.2 %) | 2. buffers allocated, converted, freed |
| &nbsp;&nbsp;buffers.alloc | 0.1 (0.0 %) | 0.6 (0.0 %) | 0.0 (0.1 %) | 1.2 (0.1 %) | output buffers allocated |
| &nbsp;&nbsp;buffers.convert | 0.0 (0.0 %) | 0.2 (0.0 %) | 0.0 (0.1 %) | 0.2 (0.0 %) | inputs converted to the kernel's type |
| &nbsp;&nbsp;buffers.free | 0.1 (0.0 %) | 1.0 (0.1 %) | 0.0 (0.2 %) | 1.0 (0.1 %) | values freed when their function returns |
| **views** | 0.0 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.0 %) | 0.3 (0.0 %) | 3. stacked trees sliced, views materialized |
| &nbsp;&nbsp;views.slice | 0.0 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.0 %) | 0.3 (0.0 %) | stacked layers sliced, scans' results stacked |
| &nbsp;&nbsp;views.axes | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | axes merged, split, reshaped (copied) |
| **copies** | 0.4 (0.1 %) | 3.0 (0.2 %) | 0.2 (0.8 %) | 3.4 (0.2 %) | 4. arguments, tuples, fields, accumulators (copied until step 6) |
| &nbsp;&nbsp;copies.entry | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.1 %) | 0.1 (0.0 %) | the entry's arguments typed |
| &nbsp;&nbsp;copies.operands | 0.2 (0.1 %) | 1.2 (0.1 %) | 0.0 (0.2 %) | 1.3 (0.1 %) | a launch's operands read from the environment |
| &nbsp;&nbsp;copies.as-tensor | 0.1 (0.0 %) | 0.5 (0.0 %) | 0.0 (0.1 %) | 0.6 (0.0 %) | the same operands taken again as tensors |
| &nbsp;&nbsp;copies.args | 0.0 (0.0 %) | 0.2 (0.0 %) | 0.0 (0.0 %) | 0.2 (0.0 %) | a call's arguments, a closure's captures |
| &nbsp;&nbsp;copies.tuples | 0.1 (0.0 %) | 1.0 (0.1 %) | 0.1 (0.4 %) | 1.1 (0.1 %) | tuples, records, fields, a tree's leaves |
| &nbsp;&nbsp;copies.fold | 0.0 (0.0 %) | 0.1 (0.0 %) | 0.0 (0.0 %) | 0.1 (0.0 %) | a fold's arguments, a scan's accumulators |
| **threads** | 1.0 (0.3 %) | 20.6 (1.6 %) | 1.0 (4.4 %) | 28.7 (2.0 %) | 5. a launch dispatched, its threads started and joined |
| &nbsp;&nbsp;threads.dispatch | 0.6 (0.2 %) | 4.0 (0.3 %) | 0.0 (0.2 %) | 6.6 (0.5 %) | the executor's dispatch of a launch |
| &nbsp;&nbsp;threads.start-join | 0.3 (0.1 %) | 16.6 (1.3 %) | 0.9 (4.2 %) | 22.1 (1.5 %) | a divided launch's threads started and joined |
| **left** | 1.2 (0.4 %) | 4.9 (0.4 %) | 0.2 (0.8 %) | 94.3 (6.6 %) | 6. what is left |
| &nbsp;&nbsp;left.dropout | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 80.6 (5.6 %) | dropout masks drawn on the host |
| &nbsp;&nbsp;left.host-prims | 0.0 (0.0 %) | 0.2 (0.0 %) | 0.0 (0.0 %) | 0.2 (0.0 %) | other primitives the host computes |
| &nbsp;&nbsp;left.interpreter | 1.2 (0.3 %) | 4.6 (0.3 %) | 0.2 (0.8 %) | 13.5 (0.9 %) | the host program's interpreter, by difference |
| **total** | 332.0 (100.0 %) | 1320.4 (100.0 %) | 21.8 (100.0 %) | 1439.1 (100.0 %) | a call, timed around it |

- forward pass: presk bench --host-profile: `GPT.apply` on the native backend, f32 under the plan, 20 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- forward pass: -- a call: 331.369 ms without the timers (median; 331.211–331.586), 331.783 ms with them (331.682–333.056): the timers' cost +0.414 ms, +0.12 %
- forward pass: -- the outputs of a call with the timers and of one without: the same bits
- forward pass: -- without the timers, 0 minor page faults a call
- forward pass: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- forward pass: -- 228 launches a call, 201 divided among 2010 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 328.910 ms, the host 3.069 ms
- gradient: presk bench --host-profile: `loss_grad` on the native backend, f32 under the plan, 20 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- gradient: -- a call: 1319.150 ms without the timers (median; 1316.493–1322.594), 1320.710 ms with them (1317.369–1321.447): the timers' cost +1.560 ms, +0.12 %
- gradient: -- the outputs of a call with the timers and of one without: the same bits
- gradient: -- without the timers, 0 minor page faults a call
- gradient: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- gradient: -- 1061 launches a call, 569 divided among 5626 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1305.998 ms, the host 14.389 ms
- `adamw.step`: presk bench --host-profile: `adamw.step` on the native backend, f32 under the plan, 20 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- `adamw.step`: -- a call: 21.736 ms without the timers (median; 21.173–22.072), 21.917 ms with them (21.384–22.087): the timers' cost +0.181 ms, +0.83 %
- `adamw.step`: -- the outputs of a call with the timers and of one without: the same bits
- `adamw.step`: -- without the timers, 0 minor page faults a call
- `adamw.step`: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- `adamw.step`: -- 111 launches a call, 63 divided among 438 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 21.186 ms, the host 0.636 ms
- training step: presk train --host-profile: the host by phase, the mean of steps 2 to 8 (the first left out)
- training step: -- 6048 minor page faults a step (pages touched for the first time, 24.8 MB of 4 KB pages)
- training step: -- 1174 launches a step, 632 divided among 6064 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1333.500 ms, the host 105.649 ms
- training step: presk train: 1181 kernel launches a step, 2013 without fusion (41% removed)
- training step: presk train: 1435.5 ms a step, median of 7 (the first excluded)
- training step: presk train: 8 steps of GPT, final loss 4.493265, weights in out/host-profile/train/profiled-20/final.psk.d
- training step without the timers: 1432.2 ms a step (the median, the first left out)

## 10 threads, ten X925 cores, pinned, glibc keeping its pages (`GLIBC_TUNABLES`: no block mapped apart, none returned)

| phase | forward pass | gradient | `adamw.step` | training step | what |
|---|---:|---:|---:|---:|---|
| **kernels** | 330.4 (98.8 %) | 1287.1 (97.2 %) | 14.9 (92.8 %) | 1301.3 (91.1 %) | the kernels' code |
| **find** | 1.2 (0.4 %) | 4.5 (0.3 %) | 0.2 (1.3 %) | 4.7 (0.3 %) | 1. finding the kernel |
| &nbsp;&nbsp;find.site | 0.1 (0.0 %) | 0.6 (0.0 %) | 0.0 (0.0 %) | 0.6 (0.0 %) | a let's site named, its type found |
| &nbsp;&nbsp;find.groups | 0.1 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.1 %) | 0.3 (0.0 %) | a function's fusion groups cloned at its call |
| &nbsp;&nbsp;find.layout | 0.2 (0.1 %) | 0.9 (0.1 %) | 0.0 (0.1 %) | 0.9 (0.1 %) | a nest's kind, its layout in the plan |
| &nbsp;&nbsp;find.key | 0.7 (0.2 %) | 2.7 (0.2 %) | 0.2 (1.1 %) | 2.9 (0.2 %) | the cache key formatted, the kernel looked up |
| &nbsp;&nbsp;find.lower | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | a kernel lowered and compiled (first launch) |
| **buffers** | 0.3 (0.1 %) | 1.8 (0.1 %) | 0.1 (0.5 %) | 2.3 (0.2 %) | 2. buffers allocated, converted, freed |
| &nbsp;&nbsp;buffers.alloc | 0.1 (0.0 %) | 0.6 (0.0 %) | 0.0 (0.2 %) | 1.1 (0.1 %) | output buffers allocated |
| &nbsp;&nbsp;buffers.convert | 0.0 (0.0 %) | 0.2 (0.0 %) | 0.0 (0.1 %) | 0.2 (0.0 %) | inputs converted to the kernel's type |
| &nbsp;&nbsp;buffers.free | 0.1 (0.0 %) | 1.0 (0.1 %) | 0.0 (0.3 %) | 1.0 (0.1 %) | values freed when their function returns |
| **views** | 0.0 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.0 %) | 0.3 (0.0 %) | 3. stacked trees sliced, views materialized |
| &nbsp;&nbsp;views.slice | 0.0 (0.0 %) | 0.3 (0.0 %) | 0.0 (0.0 %) | 0.3 (0.0 %) | stacked layers sliced, scans' results stacked |
| &nbsp;&nbsp;views.axes | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | axes merged, split, reshaped (copied) |
| **copies** | 0.4 (0.1 %) | 3.0 (0.2 %) | 0.2 (1.1 %) | 3.2 (0.2 %) | 4. arguments, tuples, fields, accumulators (copied until step 6) |
| &nbsp;&nbsp;copies.entry | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.2 %) | 0.1 (0.0 %) | the entry's arguments typed |
| &nbsp;&nbsp;copies.operands | 0.2 (0.1 %) | 1.2 (0.1 %) | 0.0 (0.3 %) | 1.2 (0.1 %) | a launch's operands read from the environment |
| &nbsp;&nbsp;copies.as-tensor | 0.1 (0.0 %) | 0.5 (0.0 %) | 0.0 (0.1 %) | 0.5 (0.0 %) | the same operands taken again as tensors |
| &nbsp;&nbsp;copies.args | 0.0 (0.0 %) | 0.2 (0.0 %) | 0.0 (0.0 %) | 0.2 (0.0 %) | a call's arguments, a closure's captures |
| &nbsp;&nbsp;copies.tuples | 0.1 (0.0 %) | 1.0 (0.1 %) | 0.1 (0.5 %) | 1.1 (0.1 %) | tuples, records, fields, a tree's leaves |
| &nbsp;&nbsp;copies.fold | 0.0 (0.0 %) | 0.1 (0.0 %) | 0.0 (0.0 %) | 0.1 (0.0 %) | a fold's arguments, a scan's accumulators |
| **threads** | 1.0 (0.3 %) | 21.9 (1.6 %) | 0.5 (3.2 %) | 24.6 (1.7 %) | 5. a launch dispatched, its threads started and joined |
| &nbsp;&nbsp;threads.dispatch | 0.6 (0.2 %) | 4.0 (0.3 %) | 0.0 (0.2 %) | 6.3 (0.4 %) | the executor's dispatch of a launch |
| &nbsp;&nbsp;threads.start-join | 0.4 (0.1 %) | 17.9 (1.4 %) | 0.5 (3.0 %) | 18.3 (1.3 %) | a divided launch's threads started and joined |
| **left** | 1.2 (0.4 %) | 4.9 (0.4 %) | 0.2 (1.0 %) | 92.3 (6.5 %) | 6. what is left |
| &nbsp;&nbsp;left.dropout | 0.0 (0.0 %) | 0.0 (0.0 %) | 0.0 (0.0 %) | 79.0 (5.5 %) | dropout masks drawn on the host |
| &nbsp;&nbsp;left.host-prims | 0.0 (0.0 %) | 0.2 (0.0 %) | 0.0 (0.0 %) | 0.2 (0.0 %) | other primitives the host computes |
| &nbsp;&nbsp;left.interpreter | 1.2 (0.4 %) | 4.7 (0.3 %) | 0.2 (1.0 %) | 13.0 (0.9 %) | the host program's interpreter, by difference |
| **total** | 334.6 (100.0 %) | 1323.5 (100.0 %) | 16.1 (100.0 %) | 1428.6 (100.0 %) | a call, timed around it |

- forward pass: presk bench --host-profile: `GPT.apply` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- forward pass: -- a call: 334.568 ms without the timers (median; 333.050–335.231), 334.703 ms with them (333.451–335.776): the timers' cost +0.136 ms, +0.04 %
- forward pass: -- the outputs of a call with the timers and of one without: the same bits
- forward pass: -- without the timers, 0 minor page faults a call
- forward pass: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- forward pass: -- 228 launches a call, 201 divided among 2010 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 331.478 ms, the host 3.141 ms
- gradient: presk bench --host-profile: `loss_grad` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- gradient: -- a call: 1323.919 ms without the timers (median; 1318.854–1325.905), 1323.535 ms with them (1321.466–1326.385): the timers' cost -0.384 ms, -0.03 %
- gradient: -- the outputs of a call with the timers and of one without: the same bits
- gradient: -- without the timers, 0 minor page faults a call
- gradient: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- gradient: -- 1061 launches a call, 569 divided among 5626 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1308.961 ms, the host 14.544 ms
- `adamw.step`: presk bench --host-profile: `adamw.step` on the native backend, f32 under the plan, 10 threads; 7 calls timed by phase and 7 not, alternated, after 1 warm-up
- `adamw.step`: -- a call: 15.830 ms without the timers (median; 15.742–16.195), 16.115 ms with them (15.936–16.205): the timers' cost +0.285 ms, +1.80 %
- `adamw.step`: -- the outputs of a call with the timers and of one without: the same bits
- `adamw.step`: -- without the timers, 0 minor page faults a call
- `adamw.step`: -- 0 minor page faults a call (pages touched for the first time, 0.0 MB of 4 KB pages)
- `adamw.step`: -- 111 launches a call, 63 divided among 438 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 15.451 ms, the host 0.638 ms
- training step: presk train --host-profile: the host by phase, the mean of steps 2 to 8 (the first left out)
- training step: -- 6048 minor page faults a step (pages touched for the first time, 24.8 MB of 4 KB pages)
- training step: -- 1174 launches a step, 632 divided among 6064 threads; the kernels as `--calibrate` times them (code, dispatch, threads) 1325.864 ms, the host 102.734 ms
- training step: presk train: 1181 kernel launches a step, 2013 without fusion (41% removed)
- training step: presk train: 1424.3 ms a step, median of 7 (the first excluded)
- training step: presk train: 8 steps of GPT, final loss 4.493265, weights in out/host-profile/train/profiled-10-heap/final.psk.d
- training step without the timers: 1427.1 ms a step (the median, the first left out)

