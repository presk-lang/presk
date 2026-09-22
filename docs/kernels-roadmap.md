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
   IR (threadgroup and register tiles), which is PSK-Sched's business and is
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
after, and the GPU only once PSK-Sched has a memory hierarchy, since a GPU
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
