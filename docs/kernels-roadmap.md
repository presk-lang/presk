# Matching a tuned BLAS: what it will take, and what it costs us

> Logged 2026-09-20, after the decisive experiment of 0.4.0 step 8. Not a
> plan: a note of the target, the measured distance to it, and the one
> constraint that makes it a research question here rather than an
> engineering chore. To be turned into a plan when a version takes it on.

## The measurement that motivates it

One training step of the course GPT, `f32`, twelve cores:

| | a step | a 1024³ matmul |
|---|---|---|
| PyTorch CPU | 0.405 s (656 GFLOP/s) | 1309–1454 GFLOP/s |
| presk native | 37.9 s (7.0 GFLOP/s) | 47–56 GFLOP/s |

PyTorch's figure barely moves with the thread count, because it is not
thread parallelism: the platform BLAS reaches Apple's matrix coprocessor.
Presk emits scalar loops. About twenty-six of the ninety-four is that one
difference, and the rest is presk's non-contraction work.

## What we will have to build

Our own tuned kernels, not a dependency: the tool has to be able to say what
its kernels compute, which is the reason a vendor BLAS was rejected (D32).
Three layers, in the order their absence hurts:

1. **A vectorized CPU GEMM.** Packed panels of both operands into contiguous
   buffers, a register-blocked micro-kernel over vector lanes (NEON here,
   AVX-512 elsewhere), cache blocking sized from the calibrated profile
   rather than from constants. This is the factor of ten.
2. **The matrix coprocessor.** On Apple silicon, AMX through the accelerate
   path or its instruction encodings; elsewhere the equivalent tensor units.
   This is the rest of the factor of twenty-six, and it is the least
   portable piece, so it belongs behind the realization contract rather than
   inside the kernel lowering.
3. **GPU.** Metal compute kernels on Apple, and one other target to keep the
   abstraction honest. PSK-Kernel's loop nests already carry enough
   structure to emit a grid; what is missing is the memory hierarchy in the
   IR (threadgroup and register tiles), which is PSK-Realize's business and is
   why fusion groups came first.

## The constraint that makes this ours

Every one of those three reassociates a sum. A vector lane holds a partial;
a packed micro-kernel accumulates several; a GPU reduces across threads. In
this compiler a reassociated sum is not an implementation detail — it is a
different **realization**, owing a declared reduction order and a measured
gap to the exact reference, exactly as `f32` owes one today (D31, D32).

That is the cost of the position, and also the only reason to build our own
rather than link someone else's. A BLAS whose reduction order is part of its
contract, and whose error against an exact reference is a published number
per shape and per machine, is something no vendor library offers. If the
project is right that a realization should be measured rather than trusted,
then the GEMM is exactly where that claim has to be made good — it is the
kernel everybody uses and nobody bounds.

So the work is not "write a fast GEMM". It is: give the realization contract
a reduction-order field, make the conformance suite measure a kernel per
declared order, and then write fast kernels under it — vector, coprocessor,
GPU — each with its number.

## Order this suggests

Vectorized CPU GEMM first, because it is the largest single factor and the
smallest new surface; the realization contract extended at the same time,
since the first vector kernel is what forces it. The coprocessor and the GPU
after, and the GPU only once PSK-Realize has a memory hierarchy, since a GPU
kernel without one is a slow GPU kernel. `presk bench --calibrate` already
fits the constants these would be tuned against.

## Two things BERT-large showed (2026-09-22, 0.6.0 step 8)

Running BERT-large's inference at MLPerf's constants (D53) measured two
properties of today's kernels that the course GPT, at batch 16 and width 384,
had not exposed.

- **At batch 1 a forward pass runs on one core.** A kernel's threads split its
  outermost exact axis (D31), and at batch 1 that axis has length one: one
  sequence of 384 tokens takes 62 s in `f32` on one core of twenty. Splitting
  the next exact axis when the outermost is too short — a contraction's rows,
  `seq` — is exact for the same reason the batch split is, since every output
  still accumulates its terms in order. It is the obvious fix, and it is not
  in 0.6.0.
- **At BERT-large's width the contraction waits for memory.** `int8` weights
  run a sequence 2.2× faster than `f32`, at batch 1 as at batch 20; at batch
  20 `bf16`, which only halves the bytes, is faster than `f32` too, and `f32`'s
  vector lanes are slower than its scalar code. The blocked kernel loops over
  the rows of the result, then over blocks of eight columns, then over the
  contracted axis, so it reads the whole weight matrix again for every row: at
  1024 × 4096, 16 MB in `f32` for each of 7 680 rows at batch 20, it streams
  from memory, and an `int8` matrix is a quarter of the bytes. Packed panels
  and cache blocking, item 1 above, would make it compute-bound there, and the
  gain from narrow storage would shrink to what a memory-bound regime leaves.
  The speedup measured is therefore partly a statement about the blocking this
  kernel lacks, and it has to be measured again when item 1 exists.

## What 0.7.0 built of item 1 (2026-09-24)

Version 0.7.0 (D56–D62) built the register-blocked micro-kernel of item 1 as
a layout space rather than as one kernel: register tiles up to `[4, 16]` over
vector lanes, a k-block, the block loops in any order, a pack of one operand
(the NT one, so that lanes can read it), and a thread split at any depth, the
fix named for batch one above. Every point is bit-identical to the scalar
sequential kernel, because lanes run across output columns and every sum
keeps its order, so none of it needed a new realization: the constraint
below applies to what is still to come, not to this. On falicon's ten X925
cores the best point of a 1024³ `f32` product runs at 460 GFLOP/s, 5.49× the
0.6.0 kernel (84) and within 1.6× of PyTorch's 710 on twenty cores; at batch
one the attention's two `ein`s run 47× and 77× faster than 0.6.0's. Packing
**both** operands into contiguous panels, cache blocking sized from the
profile, the matrix coprocessor and the GPU are still to do, with memory
placement, which is what turns the pack's copy into a scratch buffer
(question 66).

What 0.7.0 did not get is a model that chooses among those points. Its
layout cost model ranks lanes and thread splits and cannot see a register
tile (D62), which is the design below.

## A register-load term for the layout cost model (design, for a later version)

> Recorded 2026-09-24, when 0.7.0 closed on D62's results. The user decided
> that the term goes to a version after 0.7.0, and that 0.7.0's statements
> stand as measured. Not a plan and not code: to be planned when a version
> takes it on.

**What fails, and why.** The layout cost model of 0.7.0
(`presk_kernel::cost::planned_cost`, fitted by
`presk_opt::profile::fit_layout`) predicts

```text
T̂ = λ + spawn·[p > 1] + max(W/π_g, Q_L2/β_L2, p·Q_L3/β_L3, p·Q_mem/β_mem)
```

with `W` one worker's weighted operations and `Q_c` its traffic from each
cache level. Two conventions make it blind to the register tile. A load
weighs nothing in `W` (`Expr::Load`, `LoadI` and `VLoad` weigh 0, as index
arithmetic does: 0.3.0's roofline convention), and traffic is counted from L2,
where every tile of a form has the same footprint. So at the attention's
shapes every point is predicted work-bound, and the points of a form that
differ only in their tile, order or k-block get the same prediction. RA4's
statement M failed on exactly those decisions: tiles 27.7 % right and 46.7 %
tied, orders 30.0 % and 61.8 %, k-blocks 0.6 % and 97.0 %, where lanes and
splits are 93–97 % right.

**The evidence for a load term.** On the values `ein` at `B = 16` in `f32`
(the course GPT's attention, day one and day two of the measurement, `out/ra4`,
not committed), thirty points with four lanes and the split at depth 2 over
ten workers are all predicted 2.27 ms. Grouped by tile, with the loads the
micro-kernel issues per vector multiply-add — each `k` loads `m_i` elements
of operand 0 (broadcast) and `n_j/ℓ` vectors of operand 1, for `m_i·n_j/ℓ`
vector multiply-adds — they measure (mean of the two days' medians, five
points a tile: the orders and k-blocks):

| tile | loads per vector multiply-add | measured |
|---|---|---|
| `[1, 4]` | 2.00 | 6.89–7.30 ms |
| `[1, 8]` | 1.50 | 4.55–4.76 ms |
| `[2, 4]` | 1.50 | 4.55–5.22 ms |
| `[1, 16]` | 1.25 | 3.36–3.48 ms |
| `[2, 8]` | 1.00 | 3.12–4.07 ms |
| `[2, 16]` | 0.75 | 2.53–2.90 ms |

The load count orders the tiles almost exactly where the model sees one
number; the one overlap is `[2, 8]` without a k-block, which is slower than
the count says and faster with a k-block of 128 — an order or cache effect, the
second blind spot below. The calibration saw the same thing: its two tiles,
`[1, 8]` and `[4, 16]`, share a group and have the same work, and differ by a
median of 1.26× (a quarter of the 288 pairs by more than 1.85×), which is part
of what `u` = 0.185 is.

**The term.** Count, per worker, the loads the kernel issues into registers,
`N_ld` (a scalar load, a broadcast and a vector load one each; the pack's copy
kernel counted as it is), from the same walk that computes `W` — the loops are
affine, so it is the same product of trip counts. Charge them against a load
rate, fitted like the other constants:

```text
T̂ = λ + spawn·[p > 1] + max(W/π_g, N_ld/ρ_g, Q_L2/β_L2, p·Q_L3/β_L3, p·Q_mem/β_mem)
```

`ρ_g` per group, as `π_g` is (a vector load and a scalar load may cost the
same, so the width is in the group already), or one `ρ` per machine if the fit
cannot tell the groups apart. The calibration GEMM needs more tiles than its
two for `ρ` to be identified: `[1, 4]`, `[2, 8]` and `[2, 16]` besides `[1, 8]`
and `[4, 16]`, which the calibration space can take at the same cost by
dropping some orders. The pack is not helped by it: its close pairs are at
one lane, since lanes on an NT nest need the pack, and there the pack changes
where the loads read, not how many there are. What it buys at one lane is
contiguous loads, which is the L1 question below; with lanes, the load count
does credit it, by the lane width.

**What it will not fix alone.** Loop orders and k-blocks change which operand
stays in L1 and L2, and their traffic terms never bind under the `max`
because the work term is always larger at these shapes; with a load term
the in-core term grows and would still dominate. Two changes are candidates, to be
judged on the calibration first: a footprint analysis that starts at L1
(the bytes from L2 into L1, `Q_L1`, which is what an order changes), and a
combination that lets a non-binding term separate two points the binding one
ties — a sum of the in-core and the memory terms, or a smooth maximum whose
exponent is fitted. Index arithmetic, also weighed 0, is the third suspect;
the kernel text computes every address from the loop variables, and whether
Cranelift strength-reduces them is a question for the emitted code, not the
model.

**How to test it, which is the point.** The term is designed after seeing
where 0.7.0's model failed, so ranking 0.7.0's measurements again would be a
fit, not a test. The measurements can still serve during development: each
point's kernel is lowered again from its layout key, and its times are in
`out/ra4*`. The test is a **fresh sweep** the design never saw, decided by the
same rule (`presk rank`, questions 65 and 70): other shapes — BERT-large's
attention, sampled in 0.7.0 and never swept whole, or a model of another
width — or another machine. falicon has
one within reach: its ten Cortex-A725 cores (CPUs 0–4 and 10–14) are another
microarchitecture, with a quarter of the L2, and a sweep pinned to them, with
its own calibration, is a new machine for the model.

**The other result worth building on.** The model already works as a
**pruner**: on every contraction nest at the attention's shapes, its
candidates within `1 + u` of its best held the best measured point or one
within 1.08× of it, at 2.4–15 % of the space. A chooser that prunes statically
and measures the candidates — `presk autotune --profile P --top N` already
measures the model's `N` best points — is the hybrid those numbers support, and it is a different claim from RA4's,
which asks the model to decide alone. A version that adds the load term should
report both: the rank accuracy RA4 asks for, and the pruned search's cost and
regret.
