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
> stand as measured. Not a plan and not code. Version 0.8.0 took it on
> (`presk-specs/implementation/presk-plan-0.8.0.md`, D63), with the second
> fix below. Its step 0 corrected what this note said about where the
> model's traffic starts (D63). What 0.8.0 built and found, and what it
> left for later, is the next section.

**What fails, and why.** The layout cost model of 0.7.0
(`presk_kernel::cost::planned_cost`, fitted by
`presk_opt::profile::fit_layout`) predicts

```text
T̂ = λ + spawn·[p > 1] + max(W/π_g, Q_L2/β_L2, p·Q_L3/β_L3, p·Q_mem/β_mem)
```

with `W` one worker's weighted operations and `Q_c` its traffic from each
cache level: `Q_L2` the bytes brought into L1 from L2 (`traffic(worker,
m.l1)`), `Q_L3` into L2, `Q_mem` into a worker's share of L3, from the
footprints of the affine accesses. Two conventions make it blind to the
register tile. A load weighs nothing in `W` (`Expr::Load`, `LoadI` and `VLoad`
weigh 0, as index arithmetic does: 0.3.0's roofline convention), and nothing
counts the loads from L1 into registers, which is what a register tile
saves. So at the attention's shapes every point is predicted work-bound, and
the points of a form that differ only in their tile, order or k-block get the
same prediction, whatever their traffic. RA4's
statement M failed on exactly those decisions: tiles 27.7 % right and 46.7 %
tied, orders 30.0 % and 61.8 %, k-blocks 0.6 % and 97.0 %, where lanes and
splits are 93–97 % right.

**The evidence for a load term.** On the values `ein` at `B = 16` in `f32`
(the course GPT's attention, day one and day two of the measurement, `out/ra4`,
not committed, archived in 0.8.0 step 0: D63), thirty points with four lanes
and the split at depth 2 over ten workers are all predicted 2.27 ms. Grouped by tile, with the loads the
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
contiguous loads, which the traffic terms do not see either, since a
footprint counts elements, not cache lines; with lanes, the load count does
credit it, by the lane width.

**What it will not fix alone.** Loop orders and k-blocks change which operand
stays in L1 and L2, and the model already counts what they change: the bytes
from L2 into L1 are `Q_L2`, each buffer counted apart. On the values `ein` at
`B = 16` in `f32`, every one of the 133 sets of points that differ only in
their order and k-block has more than one `Q_L2` among its points, and so do
187 of the scores `ein`'s 203. But all 1 717 and 1 515 points are predicted
work-bound, and the `max` hides every term but the largest; with a load term
the in-core term grows and would still dominate. What hides the orders is the
combination, not a missing level. The change is a combination that lets a
non-binding term separate two points the binding one ties — a sum of the
in-core and the memory terms, or a smooth maximum whose exponent is fitted —
with a calibration that identifies the traffic constants, which 0.7.0's
barely did (D60). (Until 0.8.0 step 0 this paragraph also asked for "a
footprint analysis that starts at L1 (the bytes from L2 into L1, `Q_L1`)":
those bytes are `Q_L2`, counted since 0.7.0 step 4; D63.) Index arithmetic,
also weighed 0, is the third suspect;
the kernel text computes every address from the loop variables, and whether
Cranelift strength-reduces them is a question for the emitted code, not the
model.

**How to test it, which is the point.** The term is designed after seeing
where 0.7.0's model failed, so ranking 0.7.0's measurements again would be a
fit, not a test. The measurements can still serve during development: each
point's kernel is lowered again from its layout key, and its times are in
`out/ra4*`, archived with 0.7.0's profiles in `~/presk-data/ra4-0.7.0` on
falicon (D63). The test is a **fresh sweep** the design never saw, decided by the
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

## What 0.8.0 built of this design, and what it left (2026-09-26)

Version 0.8.0 (D63–D74) built the design above, froze it at the tag
`ra4-model-0.8.0` before measuring anything new, and tested it on two fresh
sweeps: 0.7.0's shapes on the ten A725 cores (F2), and BERT-large's
attention and a head of 128 on the ten X925 cores (F1).
- **The model frozen**: one worker's register loads and stores against a
  rate per group, `N_ld/ρ_g` (D66); the five terms combined by a p-norm
  whose exponent is fitted, `q` = 1.06, nearly a sum (D67, D69); two rules
  of the planned cost, footprints in whole cache lines and one operation of
  work per loop iteration (D70); calibrated on a GEMM widened to five tiles,
  sizes to 2048 and two thin products (D68), `u` = 0.132 (D71).
- **What it gave** (D74): statement M, more than 90 % of the attention
  `ein`s' distinguished close pairs right, holds on F2 (92.7 % and 91.7 %)
  and fails on F1 (85.4–88.8 %). Ties fell from 34 % to 0.1 %, and the
  pairs the model decides are about 99 % right. The pick is within 1.05× of
  the best on every contraction nest M names (O), and the candidates within
  `1 + u` hold a point within 1.03× of the best at under 2 % of the sweep's
  cost (Q). The static choice is 218–252 times cheaper than the sweep at
  batch one on the A725 cores, short of 10³ (N).
- **What `β_mem` is**: not identified on either kind of core. No GEMM this
  compiler emits is bound by memory on falicon, not even a two-row product
  (D67), so a large, loosely fitted constant keeps the memory term
  small.

**Recorded for after 0.8.0.** Each is a change to the model, so each needs
a model frozen again and a sweep it was not designed on, as 0.8.0 did.

1. **The distinct lines an iteration's loads touch** (D70, D71), which would
   see the pack. No rule of 0.8.0 does: the body kernel costs the same with
   the pack and without it, so the model predicts the pack slower in every
   pair, and it wins in 44 % of them on the development set, by 5 % in the
   median, all with tiles of 4 to 16 columns. At each step of `k` such a
   tile reads one value from each of its rows of the unpacked operand, where
   the packed panel holds them in one line. That is how an iteration's
   loads fall, not what stays in cache, so neither `lines` nor `shared`
   sees it. The pack is 48–63 % right on the fresh sweeps.
2. **Why the maximum forms rank k-blocks better than the p-norm** (D71):
   73–75 % against 66 % on the development set under the rules kept, though
   they lose on every other count. On the fresh sweeps the p-norm ranks
   k-blocks 53–64 % right, almost all within `u`.
3. **How the L2 term combines with the load term at one-row tiles** (D74).
   The one class of pairs the frozen model decides and gets wrong: at F1b's
   scores `ein` (`hd` = 128), 67 of 121 decided loop-order pairs, all
   one-row tiles (`[1, 8]`, `[1, 16]`), packed, in four lanes. The order with
   `I` outside `J` reads the whole packed operand once a row of the output;
   the model predicts it 15–23 % slower for its L2 term, and it measures
   1–16 % faster. Near `q` = 1 the p-norm adds the L2 time to the load time,
   and a one-row kernel does not pay it on top of its loads. It shows only
   where the packed operand of one head exceeds L1: 256 KiB at F1b, 96 KiB
   at F1a (8 of 30 decided order pairs wrong there), against 64 KiB of L1;
   at 0.7.0's shapes it is 64 KiB, and every decided order pair is right.
   The question is the combination per pair of terms, not the terms: which
   resources a core overlaps, and whether that depends on the tile.
4. **The order of the pairs within `u`**, which decides M (D74). 97–99 % of
   the misses on both sweeps are pairs the model predicts within `1 + u` of
   each other; it orders 69–73 % of those right on F1 and 78–81 % on F2, and
   they are 39–46 % of F1's distinguished pairs. Reaching 90 % on F1 would
   have taken 134 to 1 064 more right pairs a nest, among tiles (about 10 %
   wrong, on the largest counts), loop orders, the pack and k-blocks. A
   smaller `u` is not the aim: what separates close points is what the
   model does not count (items 1–3).
5. **The costing of a point**, most of N's static side at batch one (D74).
   At `B = 1` on the A725 cores a sweep takes 143–170 s and the static side
   0.57–0.78 s, of which the capture of shapes is 0.03–0.05 s; costing 3 305
   points, 0.16–0.22 ms each, is the rest. Clearing 10³ needs costing four
   to five times faster, for instance by sharing the walk between points
   that differ in one decision, or the footprints across points as 0.8.0
   step 1 shared them across cache levels (D64).

## Work a gradient does and discards (2026-09-26, 0.9.0 step 3)

The C export's recording of the host (D79) showed that the kernel backend
computes, for a gradient, values that nothing reads.
- **The counts.** 15 of `mse_grad`'s 46 launches and 125 of `loss_grad`'s
  767 (`tests/plan/minigpt_grad.psk`) compute values the result does not
  depend on. The native backend launches them, and step 3's C program
  copies them launch for launch, so that C4 compares the same work and a
  disagreement can be traced launch by launch.
- **Where they are launched**, in `loss_grad`:
  - 89 in forward functions: `Linear.apply` 32, `LayerNorm.apply` 25,
    `SelfAttention.apply` 18, `Model.apply` 6, `Block.apply` 4,
    `MLP.apply` 2, `loss` 2;
  - 36 in the `.vjp` functions and the fold's backward body.
- **What they are.** They come from two sources: the forward pass whose
  value, the primal loss, `grad` discards; and primal values computed
  inside the `.vjp` functions that no caller reads.

**The fix** is dead-code elimination in PSK-Core after automatic
differentiation, in two parts.
- **Within a function**, a `let` nothing reads can go. PSK-Core is pure,
  since a key is an argument, so this is safe.
- **Across calls**, a `.vjp` whose caller reads only its cotangents needs a
  specialized copy without the primal computation. That is the larger part.

The native backend and the C export would both gain, and the C would still
copy the native host launch for launch.

**Not measured yet.**
- Their share of a gradient's time. The forward pass is about a third of a
  forward-and-backward step's work, so a duplicated one could be a large
  part of a training step.
- Whether `presk-opt`'s extraction (`--opt`) already removes some of them.

**Its cost.** It changes launch counts that goldens, `presk dump --stage
kernel` and the tutorial quote. Every timing taken since 0.3.0 would then
include less work. So it wants its own step, with a measurement before and
after, not a place inside the C export's steps.

**Measured on the course GPT (2026-09-28, D87).** The work is larger than
a duplicated forward pass. Each `.vjp` recomputes its callee's forward
pass at every level of calls, so in `loss_grad` at `B = 16`, `T = 256` a
block's linear products run six times and attention's five. The gradient
executes 590 GFLOP of `dot` and `ein` where the forward pass is 97; 386 of
its 2 220 launches are dead. In time, those forward-type contractions take
1.4 s of the C gradient's 12.1 under a plan, since they run at the plan's
400–570 GFLOP/s. The derivatives of `dot` take 9.7 s, at 18 GFLOP/s,
outside the layout space: they come first, and removing the recomputation
second. Removing it wants more than dead-code elimination: a gradient
that keeps the values its forward pass computed.

**Done, the dead code, in 0.10.0 step 3 (2026-09-30, D91).** A pass after
differentiation (`presk_core::ad::prune`) removes what a gradient computes
and never reads: forward values its sweep does not read, cotangents no
caller reads (a `.vjp`'s result narrows to what its callers read), and
values read for their shape only (`sum_to` between equal types, a
template pointed at another value of its type). It keeps every fusion
group of what the result reads, since a group holds `bf16`'s
intermediates in `f32`, and holds what would change one. The course GPT's
gradient: 2 220 launches, 1 616 after, 12 of them dead (386 before), the
12 held for groups; its arena 1.22 GB, 1.02 GB after. `presk train` calls
the objective's `value_vjp` once a step, where it called the objective,
then its `.vjp`. What is left is the recomputation itself, each `.vjp`'s
forward pass that its sweep reads: step 4's residuals.

**Done, the recomputation, in 0.10.0 step 4 (2026-09-30, D92).** A
gradient keeps what its forward pass computed: each `.vjp` is split into a
`.fwd` that returns its value and its residuals and a `.bwd` that reads
them (`presk_core::ad::store`, `--remat store`, the default), a fold
stacking its layers' residuals. The course GPT's gradient: 1 182
launches, its `dot` and `ein` 150.1 GFLOP where they were 311.9, its arena
3.04 GB (1.02). What is left: the values the recomputing copy computed in
other fusion groups than the forward pass, which `bf16`'s exactness makes
the `.fwd` compute again (in the course GPT, 33.8 GFLOP of attention's
`a · v` and the feed-forward's first product, most of it because of the
launches step 3 holds for `bf16`'s groups: without them the gradient runs
15 % faster in C, the same `f32` bits); a scan that copies each layer's
residuals into its slot, where a fold writing them in place would not; and
the native host's copies of the values it reads (two of them removed, the
rest step 6's).

## What the C export's speed says (2026-09-27, 0.9.0 step 9)

Statement C4 timed the exported course GPT against the native backend on
the same kernels, under the same plan (D81): at `B = 16`, `T = 256` in
`f32`, 290 ms against 1 094 ms on ten threads, 1 700 against 3 132 ms on one
X925 core. Nothing about those kernels differs but who compiles them and
what runs around them, so the gaps point at the native backend itself.
- **On one thread, 1.8×.** Two candidates, not measured apart:
  - the code generator: GCC `-O2` on the C text of each kernel, against
    Cranelift on the same loop nest, which neither unrolls loops nor
    rewrites their induction variables;
  - the host: the native host copies each stacked layer's parameters out
    of the stack at every layer (`slice_tree`) and materializes views,
    where the C program passes `w + l * chunk`.
- **On ten threads, 3.8×.** Beside those two, the native backend starts a
  scoped OS thread per piece at every divided launch (201 of the 228), where
  the C program's OpenMP team is a pool.

**How to measure the shares.** The C program is the instrument. Made to
copy each slice as the native host does (an option of the export), it
gives the host's share on one thread; the per-launch spawn shows as the
ten-thread gap less the one-thread one, and a thread pool in
`presk-cranelift` would close it; the code generator's share is what
remains. None of it changes a bit, so each is a change to measure, not to
check.

**Measured on ten X925 cores (2026-09-28, D87).** Each side's kernels
were timed launch by launch, under the same plan, on the forward pass:
- the native backend takes 697–710 ms, of which its kernels 362 (`presk
  bench --calibrate`, each launch's thread starts included) and its host
  335–348;
- the C program takes 281–285 ms, of which its kernels 268–274 and its
  host 7–17.

So Cranelift's code is 1.3× GCC's, spawns included, and the host is about
four fifths of the gap. The host's cost is not page faults, whose count
does not grow with the calls; its copies and its per-launch work are
candidates, not measured apart.

## What is left between the native backend and the C program (2026-10-02, 0.10.0 step 8)

0.10.0 removed the native host's share (step 6, D96: values sharing their
buffers, views, a free list, a pool of workers), so at the version's final
measurement (D100, `docs/pytorch-baseline-0.10.0.md`) the forward pass of
the course GPT under its plan is all kernels on the native side: 340.4 ms,
of which its kernels 339.9 and its host 0.6, against the C program's 223.5
(its kernels 211.7, its host 11.8), with the C program's threads waiting as
the native pool's. The same kernels, so the 1.61× is the code generator:
Cranelift on each loop nest as PSK-Kernel states it, against GCC `-O2`
on the C text, every kernel inlined into its caller and scheduled after
register allocation only (D97's addendum: where GCC ran short of vector
registers a product's sum lived on the stack). T4 asked 1.3× and fails on
it, as decision 2 of plan 0.10.0 accepted: the kernels' code was not the
version's. The candidates, none measured: the register tile's inner loop
unrolled and its sums kept in registers by construction in the lowering,
which Cranelift's allocator would then not have to discover; a backend
whose code generator schedules and allocates as GCC's does (LLVM, at the
seam the C export already uses); or the C export as the fast path, its
bits the native backend's (C1).

## A launch inside a call slower than alone (2026-10-15, D107, D124, D126)

The Llama 2 decoder block's forward pass at the 7B width, all-`f32` under
the frozen profile's plan, at batch 1:
- *D107:* its kernels took 3 824 ms of a 3 827 ms call, where the same
  kernels timed one by one added up to 2 470 ms.
- *D124 (V5):* in 7 of 8 processes the call took 3.7 to 3.9 s, and in the
  eighth 2 439 ms. 0.12.0's solved realization, under the same plan but
  for one scalar nest, took 2.4 to 2.9 s in all of its 8.

So a call runs in one of two modes, chosen per process and kept for its
life. In the fast mode the call is near its kernels' one-by-one sum.
- *Not the host:* it is under 1 % of the call in both.
- *Not page faults:* all-`f32` has none a call.
- *A guess, untested:* where the allocator places the large buffers. That
  could make two launches' inputs and outputs collide in the caches, the
  output projection's especially, two thirds of the kernels' time.

**What would tell it.** The host profile times phases, not launches. A
per-launch timer inside a call (`presk bench --host-profile` gaining the
kernels' times as `--calibrate` takes them, but in the call's order and
memory), in the fast and the slow mode of the same plan, would name the
kernels that slow down. Beside it, the buffers' addresses modulo the page
and the cache sizes. If it is placement, an allocator that aligns the
large buffers apart, or a plan that states it, would put every process in
the fast mode, about 1.5× on this block.

**Until then** (D126): every timing of record runs each row in several
processes and reports the median across them and the fastest. A single
process, as D100's protocol ran it, may land in either mode.
