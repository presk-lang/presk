# Experiments: exact versus almost

What the tool can show about the approach, with numbers, and what it cannot.
§1 and §2 are the first two experiments: `make experiment` runs both (after
`make venv`), with the script `experiments/exact_vs_almost.py`, which takes
any output directory of `presk export … --conformance-entry train:K`. §3 is
0.4.0's decisive experiment against PyTorch (`make experiment-gpt`), taken
again on three sides at 0.9.0 (`make pytorch-baseline`), §4 the
statements of 0.5.0, §5 MLPerf's acceptance rule set beside a budget on one
model (`make foil`) and an accuracy target derived as a budget from the
exact model's margins (`make margins`), §6 the statements of 0.6.0, §7 those of 0.7.0, the
first test of the research programme's hypothesis RA4 (`make ra4`), §8
those of 0.8.0, its second, with a model frozen before two fresh sweeps
(`make ra4-f2`, `make ra4-f1`), §9 those of 0.9.0, a C export
(`make c-conformance`, `make c-speed`), and §10 those of 0.10.0, a training
step that pays (`make pytorch-baseline`, `make ra4-t3`). Each
version named its statements before it began, with what would count as
failure, and each section reports them as measured.

## 1. Shapes as types

The RMSNorm of the tutorial divides `x : ℝ[..., model: D]` by `ms : ℝ[...]`,
the per-row mean of squares. In Presk the two align by axis name, and the
checker proves it for every `B` and `D`. In plain PyTorch the natural line
`x / torch.sqrt(ms)` broadcasts from the trailing axis: it divides column `j`
by the norm of row `j`. When the batch happens to equal the model width, it
runs without any error.

```text
plain PyTorch, batch == model width: `x / sqrt(ms)` runs without error, max |wrong - right| = 0.773
Presk: `x / sqrt(ms)` with x : ℝ[..., model: D], ms : ℝ[...] aligns by name; the export inserts unsqueeze(-1)
```

The export had to learn the same lesson: it inserts `unsqueeze(-1)` exactly
where Presk's alignment (by leading prefix or by name) and PyTorch's (from the
trailing axis) differ. The `E2003` and `E2006` diagnostics of the tutorial,
with their symbolic counterexamples, are the compile-time side of the same
point: a shape error is a type error, found before the first batch.

## 2. The realization gap

`presk export --conformance-entry train:K` records the initial parameters,
the sample indices of every batch and the parameters after `K` steps of the
reference interpreter at 256 bits. The script replays those `K` steps in
PyTorch, on the same batches, with the same optimizer and schedule, in
float32 and in float64, and measures the largest absolute gap to the exact
parameters. Since 0.3.0 the export also runs the same steps on presk's own
native backend (`--backend native`, `native` chapter) in `f64` and `f32`,
and the script measures those too: four realizations of one program against
one reference.

| Run | Steps | PyTorch float32 | PyTorch float64 | presk `f32` | presk `f64` | Parameter scale |
|---|---|---|---|---|---|---|
| Tutorial tower (`tutorial/07_train.psk`, AdamW, cosine warm-up) | 10 | 3.2e-7 | 4.4e-16 | 2.0e-7 | 4.4e-16 | 1.0 |
| Tiny MiniGPT (`train_minigpt.psk`, weight decay 0.1, clipping 1.0, cosine warm-up) | 20 | 9.6e-7 | 8.9e-16 | 3.7e-7 | 6.7e-16 | 1.0 |

Read against the machine epsilons (6e-8 for float32, 1.1e-16 for float64),
all four arithmetics lose a handful of units in the last place over ten to
twenty AdamW steps on these small problems: float64 is exact to fifteen
digits, float32 to six. Presk's kernels are two to three times closer than
PyTorch's in float32, because they follow the reference operation for
operation (no fused or reassociated kernel); this is a property of naive
kernels that the e-graph pass will have to preserve or account for. Two
remarks on the measurement itself:

- The gap is *absolute*, with the model's scale given for context. An
  elementwise relative gap is meaningless here: some parameters are kept near
  zero by both arithmetics (the zero-initialized head of MiniGPT in its first
  steps), and dividing by them turns rounding noise into enormous ratios.
- What is measured is the arithmetic alone. Shuffling, initialization and the
  optimizer's definition are identical on both sides, because the export
  carries the interpreter's batches and parameters and the optimizer is a
  function of the language tested against PyTorch's (`adamw.step`, rtol
  1e-9). In an ordinary PyTorch workflow there is no reference to measure
  against; the "right" answer is whatever float32 produced.

These are the first two points of the specification's "exact versus almost"
experiment, and of its ε-to-quality curve: the error of a realization is a
number attached to a program, measured, not argued. Longer runs, larger
models, bfloat16 and mixed precision are the obvious next points; they need
the PyTorch side only (`train.py --dtype`), the reference being fixed.

## 3. The decisive experiment: against PyTorch

Version 0.4.0 set three falsifiable statements and trained the course GPT to
answer them (`D = 384`, 6 heads, 6 blocks, context 256, batch 16, the
character-level Code civil, `f32`, twelve cores, `presk train` against the
`train.py` the export writes). The answers:

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| A | presk trains within 3× of PyTorch CPU per step | ≤ 3× | **94×** (82× with `--opt`) | fails |
| B | `--opt --profile` is worth at least 1.3× | ≥ 1.30× | **1.14×** | fails |
| C | the profile predicted that speedup within 30 % | ≤ 30 % | **5 %** | holds |

One training step is about 266 GFLOP. PyTorch did it in 0.405 s, presk in
37.9 s, 33.3 s with the optimizer: 656 GFLOP/s against 7.0. Those are the
numbers of that machine and that version; the re-measurement below is this
machine's.

### The same experiment on falicon, 23 September 2026

The numbers above are 0.4.0's, on the machine it was written on. This is the
same experiment on the machine the compiler now runs on (falicon: aarch64,
twenty cores — ten Cortex-X925 and ten Cortex-A725 at 2.8 GHz — SVE at
128 bits, no matrix coprocessor reachable by either side; PyTorch 2.14, a
CPU-only wheel, so the machine's GB10 GPU takes no part). Same model, same
data file, same batches, same six steps, `f32` and twenty threads on both
sides, each side reporting the median of its five timed steps with the first
excluded, three repetitions of each row:

| side | the three runs (ms) | median (s) | vs PyTorch | GFLOP/s |
|---|---|---|---|---|
| PyTorch 2.14 `float32` | 1332, 1338, 1342 | **1.34** | 1.0× | 199 |
| presk `f32` | 29469, 29960, 31440 | **29.96** | 22.4× | 8.9 |
| presk, step solved at 10⁻³ (J′) | 30075, 30611, 30995 | **30.61** | 22.9× | 8.7 |

**The factor is 22×, not 94×, and fourteen of it is one kernel.** On this
machine a 1024³ `f32` product runs at **710 GFLOP/s** under PyTorch and at
**49.6 GFLOP/s** under presk's blocked kernel with lanes (`presk bench
out/gemm.psk --entry gemm --realize f32`), a factor of 14.3 on the
contraction alone. PyTorch's side of that is not a matrix unit: this wheel
carries its own OpenBLAS 0.3.34 (built `DYNAMIC_ARCH USE_OPENMP`, selecting
its `neoversev2` kernels) and oneDNN 3.12 over Arm Compute Library 53.2, and
turning oneDNN off leaves 680 GFLOP/s, so the two paths are within 4 % of
each other here. The cores report `sve sve2 i8mm bf16` and **no `sme`**, so
710 GFLOP/s is about four fifths of what 128-bit SVE fused multiply-add
allows on twenty cores at 2.8 GHz, and presk's 49.6 is about a twentieth of
it. On the machine of 0.4.0 the same product ran at 1309–1454 GFLOP/s under
PyTorch, through a matrix coprocessor its platform BLAS could reach, and at
47–56 under presk.

So neither side moved for a reason inside the compiler. Presk's matmul rate
is where it was (47–56 → 49.6) and PyTorch's halved for want of that
coprocessor (≈1400 → 710); that accounts for a factor of 14 of the 22, and
the remaining 1.6 is everything that is not a contraction. Each side's
training step runs below its own matmul rate — presk's at 18 % of it, 8.9
against 49.6, PyTorch's at 28 %, 199 against 710 — where on the earlier
machine the same two figures were 14 % and 45 %. Presk has gained a little
on the rest of the step and PyTorch has lost a good deal of it, and that,
with the missing coprocessor, is what turned 94 into 22. The honest reading
of §3 is unchanged in kind and smaller in size: a mature stack is still an
order of magnitude ahead of these kernels, and the reason is the
instruction set they reach, not the discipline.

The third row is the realization the solver chose for one training step
under a 10⁻³ budget on its update, statement J′ of §6. Against PyTorch it
buys nothing, and that is not news: J′'s 1.99× is against all-`f64`, and
0.6.0 already reported the solved step level with all-`f32`. Here it is
30.61 s against 29.96, inside the spread of the three `f32` runs
(29.47 to 31.44 s). A budget bought time against the exact end of the menu,
not against the type a practitioner would have written.

Two things make this comparison fairer than 0.4.0's. Both sides now **drop**:
`make experiment-gpt` had been broken since 0.5.0, and the exported loop ran
its dropout at the expectation while `presk train` dropped, so PyTorch was
doing less work (D54). The masks differ — PyTorch draws its own, presk
derives its from the step's key — so the **losses are comparable to about
four digits and no further** (4.507609 against 4.507883 after six steps,
measured again in D93 with the right gradient; 4.507878 in D55, whose presk
side trained with the gradient defect 3f of D92 made wrong),
while the timings are of the same computation. And both sides are timed per
step, the first step excluded, rather than by total wall time, so neither
pays for the other's warm-up (the native backend compiles its kernels on the
first step, PyTorch allocates on its).

**Where the factor of ninety-four is.** It is not distributed through the
compiler; most of it is one thing. PyTorch's CPU matmul reaches Apple's
matrix coprocessor through the platform BLAS and runs a 1024³ `f32` product
at 1309–1454 GFLOP/s — a rate that barely moves with the thread count,
because it is not thread parallelism. Presk's own blocked kernel, scalar
loops over twelve threads, runs the same product at 47–56 GFLOP/s. That is
a factor of about twenty-six, and it is an *instruction-set access* gap, not
a compiler-cleverness gap.

The remaining factor of three is presk's, and it is visible in the same
numbers: PyTorch's training step runs at 45 % of its own matmul rate, presk's
at 14 % of its own. The difference is everything that is not a contraction —
softmax, normalization, the elementwise chains reductions keep splitting, and
2274 kernel launches a step.

**What that does and does not say about the thesis.** The claim under test is
that exact semantics plus rewriting plus static cost models buy speed. This
version bought 7.3× (289 s a step to 39.8 s) and *every* transformation it
shipped is exact: fusion, threads, blocked contractions and the scalar
rewrite all leave the training losses identical to every digit printed. The
discipline held under a sevenfold speedup, which is the part worth keeping.

What it did not do is close the gap to a mature stack, and the reason is
recorded rather than excused. The kernels this compiler emits are scalar
because the two ways to go faster both reassociate a sum — vectorizing the
contraction's inner loop, and calling a BLAS whose summation order is not
ours — and a reassociated sum is a different realization in this system,
owing a declared order and a measured gap (decisions D31 and D32). That is a
real cost of the position, and the honest form of the result is: the
discipline is affordable at 7.3×, and the next order of magnitude requires
admitting a new realization and measuring it, not writing a faster loop.
(0.5.0 then found a third way that does not reassociate — vector lanes across
the output columns of the blocked form, D42 — and 0.6.0 emits it; §6 says
what it bought.)

Matching a tuned BLAS is a target this project will have to take on, for the
CPU's vector units and matrix coprocessor and for the GPU, with its own
kernels rather than a dependency; `docs/kernels-roadmap.md` records what that
needs and why it is a research question here — every one of those paths
reassociates a sum, so each is a realization owing a declared order and a
measured gap.

Statement A, re-posed on falicon, still fails: 22× against the 3× it asked
for. Statement C is the one that held, and it is the smallest of the three: asked
to predict its own speedup, the calibrated model said 1.087× and measurement
said 1.14×. The model is right about what it is asked; it is asked very
little, because the rule set offers few decisions that turn on size
(`optimization` chapter).

### Three sides at 0.9.0, 28 September 2026

23 September's comparison is the native backend at its default layout.
0.7.0 has since added layout plans and 0.9.0 a C export, so the course GPT
was timed again at `B = 16`, `T = 256` in `f32` (`make pytorch-baseline`,
D87). There are three sides: the native backend, the C program `presk
export --target c` writes, and PyTorch 2.14 running the model `presk export
--target pytorch` writes. There are two workloads: the forward pass, in eval
mode on PyTorch's side, and one training step, where the C program's is its
gradient plus `adamw.step`. Each runs under the default layout and under the
plans the frozen profile gives. The runs use the ten X925 cores, pinned,
with ten threads on every side, since the plans divide among ten workers.
Each figure is the median of seven calls or steps; the native backend's
first call, which compiles, is left out, and PyTorch's first too. Since
D126 a timing of record also runs each row in several processes, at least
three, and reports the median across them and the fastest, since a
process may run its kernels in a slower mode for its whole life (D124). The two
runs, in ms, and the factor over PyTorch:

| side | forward pass | one training step |
|---|---|---|
| PyTorch 2.14 `float32` | 156, 153 (621–634 GFLOP/s) | 1 016, 1 013 (286–287 GFLOP/s) |
| C program, under the plan | 281, 285: **1.8–1.9×** | 12 235, 12 247: **12.0–12.1×** |
| native, under the plan | 697, 710: 4.5–4.6× | 23 025, 22 292: 22.0–22.7× |
| C program, default layout | 894, 880: 5.7–5.8× | 16 129, 16 392: 15.9–16.2× |
| native, default layout | 1 728, 1 807: 11.1–11.8× | 30 249, 29 895: 29.5–29.8× |

The counts are 96.93 GFLOP a forward pass and 290.78 a step, the
contractions counted in full, a multiply and an add each. PyTorch's own
counter gives the same, and the 266 GFLOP of 0.4.0, used above, was 8.5 %
low. On every core unpinned, 23 September's setting, the native step at the
default layout is 22.3× and 22.8× PyTorch's, against 22.4× then: at the
default layout a training step has not moved. The C program's forward pass
there is 1.13× PyTorch's: PyTorch loses 1.6× on the slower cores, while the
plan's launches stay on ten threads. Every layout and thread count gives
the same `f32` logits and gradient, bit for bit. Against the native
backend's `f64` logits, PyTorch's `float32` is at a relative distance of
1.6·10⁻⁷ and the native backend's, which is the C program's, at 3.3·10⁻⁷.
presk's sums run in sequence, as the realization declares, and a blocked
library's run in shorter chains. PyTorch's `float64` is at 6.5·10⁻¹⁶, which
checks that the sides run the same model. The checkpoint is C4's, a probe a
few steps from the initialization, not a trained model.

**Where the gap is now.**
- **The contraction rate**, fourteen of the 22 on 23 September, is now the
  smallest factor. The model's linear products alone run at 713–1 070
  GFLOP/s under PyTorch, which reaches the wheel's OpenBLAS for them and
  oneDNN over Arm Compute Library for attention's. The C program runs them
  under the plan's layout at 510–568 GFLOP/s, and at 567 inside its
  forward pass. The C program's remaining 1.8× is that factor and the
  kernels that are not contractions.
- **The native backend's host.** Under the plan, the host takes 335–348 of
  the native backend's 697–710 ms; the C program's host takes 7–17 of
  281–285. The kernels themselves, as Cranelift compiles them, take 1.3×
  the C program's. So about four fifths of the native backend's lead over
  C is its host, the share C4 left unmeasured (§9). What in the host
  takes the time is not measured apart.
- **The training step: the derivatives of `dot`.** They take 9.7 s, 80 % of
  the C gradient's kernels under the plan, at 18 GFLOP/s. They keep their
  own lowering, scalar loops outside the layout space (`realize` chapter),
  so a plan buys 1.3× on a step where it buys 2.5–3.2× on a forward pass.
- **Then recomputation.** Each `.vjp` recomputes its callee's forward
  pass, at every level of calls, so a block's linear products run six
  times a gradient: a gradient executes 590 GFLOP of forward-type
  contractions where the forward pass is 97. PyTorch keeps its
  activations.

By arithmetic, not measured: at the rate the planned products reach, the
derivatives would take about 0.35 s and the C step about 2.9 s. The shape
of §3's result has changed. The contraction rate is no longer the
distance: the exported program is within 2× of PyTorch on inference. On a
training step, one lowering the layout space does not reach yet is most of
the distance.

### Addendum: the native host by phase (0.10.0 step 5)

The host's share, which this section left unmeasured, is measured by
`presk bench --host-profile` (`cli` chapter) and `make host-profile` (D95,
`docs/host-profile-0.10.0-step-5.md`), the course GPT under the frozen
profile's plans, the ten X925 cores pinned. Of the forward pass's 664 ms,
the kernels' own code is 341 ms; of the rest, the host copies values for
213 ms, 6.2 GB a call: each launch's operands are copied out of its
environment and copied again as tensors. The threads' start and join take
43 ms, the dropout masks of ones the host draws outside training 32,
the outputs' allocation 27, and finding each kernel by its key 2. On the
gradient (no dropout, as the C program's), the copies are 5.3 of 8.3 s,
73 GB a call, most of them the stored residuals carried in tuples, and the
gradient touches 3.25 million fresh pages a call. With glibc told to keep
its pages it takes 5.0 s, and its kernels' code 1.32 s, 1.33× the C
program's 0.99, as the forward pass's is 1.27×. On every core unpinned the
native forward pass's kernels take 2.2× their pinned time while the C
program loses nothing, and its threads' start is still 4.7 % of it: the
loss is in the kernels, perhaps where each fresh thread is placed, which
step 6's pool of workers will tell. Step 6 takes the fixes in an order
from these measurements.

### Addendum: the native host fixed, T4 (0.10.0 step 6)

Step 6 fixed the host in that order, each change measured apart (D96,
`docs/host-profile-0.10.0-step-6.md`). Values share their buffers, so a
launch's operands, a call's arguments and a tuple's components are no
longer copied; a stacked layer's slice is a view of its stack and a scan's
stack its parts; a launch's output comes from a free list of the buffers
freed before; the training dropout masks are drawn on the workers, bit for
bit, and outside training not drawn at all; a pool of persistent workers
runs a divided launch's pieces; the entry's arguments already in their
type are not converted. Pinned, the forward pass takes 332 ms where it took
661, its host 4 ms where it took 323, and touches no fresh page; the
gradient 1.32 s where it took 8.2; a training step 1.43 s where it took
7.8. On every core unpinned the times are the same: the forward pass's
kernels take 326 ms there against 755 before the pool, so the loss above
was where fresh threads ran. **T4 holds** (`docs/pytorch-baseline-0.10.0-step-6.md`):
under the plans, the native forward pass takes 1.16× the C program's (329.9
ms against 284.5), and the step compared like for like, the native
`loss_grad` and `adamw.step` against the C program's, 1.17× (1 335.5 ms
against 1 137.8); `presk train`'s step, which draws its masks, 1.26×. What is
left between the two is the kernels' code, Cranelift's against GCC's:
1.24× on the forward pass, 1.30× on the gradient. T2 again: the C step
1.13× PyTorch's.

### Addendum: the bias as an epilogue, the C flags (0.10.0 step 7)

Step 7 adds a linear layer's bias in its product's kernel (D97; the
`realize` chapter's "The bias as an epilogue"), which `presk plan` now
states, and measures the C programs' flags
(`docs/pytorch-baseline-0.10.0-step-7.md`, and `…-o3-native.md`, `…-o3.md`
beside it). Against `main`'s programs, interleaved on the same cores, the
epilogue saves the 36 bias launches of a forward pass, about 10 ms of it on
both sides (the C forward pass 286.2 → 277.8 ms, the native 336.6 →
325.8), and the C gradient takes 1 011 ms where it took 1 124. Of those
113 ms, the bias kernels were 15; the rest is GCC compiling the same
products and `dot_vjp_l`s about 25 % faster once the program holds no
`add` kernels: it inlines most kernels into the host, and the program's
other kernels changed its choices (built `-fno-inline`, the two take the
same time). **T2: the C training step 1.02× PyTorch's** (1 017.0 ms against
994.8). **T4** holds on the forward pass, 1.16×, and on the step reads
1.305× in the run of record, 1.30× and 1.29× in the flags' runs: at its
threshold, by the C side's 100 ms of inlining, the native step 9 ms faster
than at step 6. `-O3` moves a C program by −4 % to +2 % and `-mcpu=native`
adds nothing to it, so `-O2` stays the default; C1 holds under each. And
the C program's OpenMP workers sleep between its regions by libgomp's
default: with `OMP_WAIT_POLICY=active` the forward pass takes 229 ms and
the gradient 836, the same bits. The baseline keeps the default.

### Addendum: the C program made deterministic, its threads waiting as the native ones (0.10.0 step 7's follow-up)

The user decided on step 7's findings (D97's addendum): T4 is judged at
step 8 on the final measurement; the C program's inlining made
deterministic; OpenMP's waiting matched to the native pool in the
measurement. Measured first: the 100 ms GCC's inlining moved were the
register allocator's margin. A product's inner loop under a 4×16 tile
holds about as many values as there are vector registers, and wherever
GCC ran short (a kernel out of line, or inlined after the scheduling that
runs before allocation) one sum lived on the stack and the product took
about 30 % longer. The export now always inlines every kernel and, on
GCC, compiles `model.c` without that scheduling (the `export_c` chapter):
the course GPT's gradient under the plans with and without their
epilogues then takes 827 and 822 ms, where today's programs took 844 and
959, and the forward pass 222 and 228 ms (230, 234), with the threads
waiting as below; no bit moves. libgomp's idle workers spin about 0.1 ms
before they sleep, the native ones about 2 ms; `GOMP_SPINCOUNT=8M` makes
libgomp's spin as long, and `make pytorch-baseline` times every C program
both ways (`docs/pytorch-baseline-0.10.0-step-7-followup.md`, the run of
record). **T2: the C training step 1.01× PyTorch's** (1 006.9 ms against
1 000.1, both at their runtime's default; 0.85× with the threads waiting
as the native ones). **T4**, in its row of record, the C program's threads
waiting as the native backend's: 1.47× on the forward pass (323.4 ms
against 219.9) and 1.57× on the step (1 340.6 against 852.8), both over
1.3; by OpenMP's default 1.21× and 1.33×. What is left between the two
sides is the kernels' code: 323.5 ms of native kernels on the forward pass
against GCC's 207.8, 1.56×, which decision 2 leaves for later. T4 is
judged at step 8, on the final measurement.

### PyTorch compiled and in mixed precision (D153)

Eager is not what a PyTorch user who cares about speed runs. `make
pytorch-modes` (`experiments/pytorch_modes.sh`, the report
`docs/pytorch-modes-2026-10-18.md`) runs the exported course GPT in four
modes, on the same inputs and the same ten X925 cores, by D126's
protocol:
- eager;
- `torch.compile` with its defaults (inductor, on the CPU);
- AMP, `torch.autocast` in `bfloat16` around the forward pass and the
  loss;
- both.

D127's PyTorch rows and the compiler's step are timed again as controls,
and reproduce D127 within its spread.

**The time.**
- *Compiled*, PyTorch's training step takes 558.2 ms, 0.57× eager's
  987.6, and its forward pass 122.4 ms, 0.81×. So **the C step under the
  plans (1 003.5 ms) is 1.80× the compiled step**, where it is 0.98×
  eager's (T2).
- *With AMP as well*, PyTorch's step takes 466.2 ms, so the C step is
  2.15× it.
- *Like for like*, PyTorch's step with its dropout off, as the C step
  draws no masks, takes 685.0 ms in eager and 483.8 compiled, so the C
  step is 1.46× and 2.07× them.
- *The compiles*, cold, take 3.7 s for the forward pass and 8.3 s for the
  step; each is one graph, with no graph break.

**Where the compiled step wins.** The compiled graph sends every product to
ATen's `mm` and `bmm`, as eager does, about two thirds of its step. It wins
on two things: the elementwise passes, fused into its C++ kernels, and the
dropout masks, drawn inside them. In eager, `bernoulli_` takes a quarter
of the step.

**The accuracy at D = 64**, on the 461 held-out windows against the exact
logits:
- *Compiled `float32`* is 3.3·10⁻⁶ from the exact logits, eager's
  2.9·10⁻⁶ and presk's all-`f32` 3.1·10⁻⁶, with no flip.
- *AMP's `bfloat16`* is 4.0·10⁻² from them and flips 505 decisions, as the
  compiler's all-`bf16` does (480). It certifies nothing at ε\*(10⁻³).

**The quality.** Trained 600 steps at D = 64 from three seeds, AMP's
validation loss is eager's +1.4·10⁻³. That is ten times the curve's
realizations' 1.3·10⁻⁴, and inside the 10⁻² a different dropout draw alone
makes.

## 4. Precision as a resource: the four statements of 0.5.0

Version 0.5.0 named four falsifiable statements before it began
(`presk-specs/implementation/presk-plan-0.5.0.md` §3.7), in the form 0.4.0
used. One holds, two fail, and one cannot be asked of anything that exists.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| D | the solved realization's **measured** gap is inside a 1e-3 budget | ≤ 4.52e-3 | **1.05e-7** (0.5.0, D41; on the weights of the corrected gradient **1.89·10⁻⁶** with 2.08× less traffic, D93, and at the tag `v0.10.0` **2.12·10⁻⁵** with 2.00×, D102) | holds |
| E | the solved realization is faster per step than all-`f64` | ≥ 1.5× | **0.96×** | fails |
| F | the propagated bound is within 10³ of the measured gap on half the entries | ≥ 11 of 21 | **10 of 21**, median 1.8e3 | fails |
| G | the vectorized GEMM's gap is inside its declared order's bound | — | — | not askable |

**D holds, and it is the one that matters most.** A budget declared in the
program becomes an assignment whose distance from the exact denotation is
*measured*, not asserted, and the measurement is inside the budget by four
orders of magnitude at 0.5.0, and by more than three at D93's
re-measurement (1.89·10⁻⁶ under the same relative budget of 10⁻³, on a loss
of about 2.8), and by two at the tag `v0.10.0`, whose solver prices `bf16`
at its corrected unit roundoff (2.1·10⁻⁵, D102). No other compiler can run this check, because no other
compiler keeps the exact denotation to measure against.

**E fails, and the reason is the version's own finding turned against it.**
The solver minimises **traffic**, because traffic is what a narrower type
reduces. At these shapes the kernels are not traffic-bound, so narrowing buys
nothing and the conversions cost: the solved realization runs at 5.63 ms
against all-`f64`'s 5.39, and uniform `bf16` at 6.05. Step 2 measured exactly
this on a single kernel and D41 wrote it down as the reason the objective is
traffic and not time. The consequence was still not drawn: **optimising the
right proxy in the wrong regime is not optimising**. Closing E needs two
things, both already named — the roofline's `W` per site so the objective can
be time, and vector lanes so that arithmetic width buys time at all.

**F fails narrowly and informatively.** Ten of twenty-one entries are within
10³ and the median ratio is 1.8e3; on a normaliser the bound is within a
factor of 27, on the corpus GPT it is 10²⁰. The threshold was set generously
on purpose, because the interesting outcome was the number and not the
verdict. The number says the bound is worth having on a kernel and worthless
on a network, and locates why: a contraction of `k` terms multiplies error by
`k` where committed errors grow like `√k`.

**G cannot be asked.** It is about a vectorized GEMM, and there is none. The
step that would have built it found instead that vectorizing across output
*columns* is bit-exact (D42), so the kernel G describes should be built under
a sequential order and would have no realization gap for G to measure. The
statement needs restating before the next version can answer it — against the
exact vector kernel, with the margin printed beside the verdict rather than a
bare pass. It became 0.6.0's statement H (§6, D43).

**What the four together say.** The claim that precision can be *declared*
and *verified* is demonstrated: D holds by measurement. The claim that it
**pays** is not: E fails, and it fails for a reason this version measured
twice before believing it. That is the same shape as 0.4.0, where the
discipline was shown affordable and the speed was not there yet, and it puts
the same item at the top of the next version's list: make a narrower type buy
time, not only bytes.


## 5. Two acceptance rules, one model

MLPerf's closed division accepts a realization by its task accuracy over the
whole validation set, within 99 % or 99.9 % of an FP32 reference's; a budget
accepts it by its measured distance to the exact denotation, on the
envelope's inputs. The `coverage` chapter has the argument and D53 the
decision (0.6.0 step 8). `make foil` applies both rules to the same eight
realizations of the corpus GPT at `D = 64`. The weights are the all-`f64` run
of the `precision` chapter's curve, trained for 600 steps on 90 % of the Code
civil, so the other tenth — 461 windows, 118 016 next-character predictions —
is held out.

- **MLPerf's side** is next-character accuracy over the held-out windows (a
  new entry of `corpus/nlp/05_gpt.psk`, `accuracy`), with the loss beside it
  because perplexity on a held-out corpus is the roadmap's declared NLP
  measure. The model runs in the realization and the metric in `f64`, as
  MLPerf's harness scores a model from outside it: in `bf16`, the mean over
  118 016 positions would round the metric itself to eight bits.
- **The budget's side** is `presk quantize` on two held-out windows: the gap
  to the exact loss, split into the weights' rounding and the arithmetic.

| realization | gap to the exact loss | representation | arithmetic | check (s) | accuracy | vs `f32` | 99 % | 99.9 % | loss vs `f32` | pass (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `f64` | 3.18·10⁻¹⁶ | 0 | 3.18·10⁻¹⁶ | 0.10 | 0.242442 | 0 | yes | yes | +7.2·10⁻¹⁰ | 1.7 |
| `f32` | 4.68·10⁻⁸ | 0 | 4.68·10⁻⁸ | 0.08 | 0.242442 | — | yes | yes | — | 1.4 |
| `bf16` | 1.71·10⁻³ | 0 | 1.71·10⁻³ | 0.07 | 0.242425 | −7.0·10⁻⁵ | yes | yes | +2.3·10⁻⁵ | 1.7 |
| `int8`, `f32` elsewhere | 3.90·10⁻⁵ | 3.89·10⁻⁵ | 6.56·10⁻⁸ | 0.06 | 0.242493 | +2.1·10⁻⁴ | yes | yes | +2.1·10⁻⁵ | 1.5 |
| `int8` `pow2` | 2.97·10⁻⁵ | 3.00·10⁻⁵ | 2.98·10⁻⁷ | 0.09 | 0.242348 | −3.8·10⁻⁴ | yes | yes | −2.5·10⁻⁵ | 1.5 |
| `int4`, `f32` elsewhere | 3.99·10⁻⁴ | 3.99·10⁻⁴ | 2.67·10⁻⁷ | 0.06 | 0.241289 | −4.8·10⁻³ | yes | **no** | +2.3·10⁻⁴ | 1.5 |
| `int4` `pow2` | 8.66·10⁻⁴ | 8.66·10⁻⁴ | 4.65·10⁻⁷ | 0.06 | 0.243416 | +4.0·10⁻³ | yes | yes | +1.2·10⁻³ | 1.5 |
| solved at 10⁻³ (statement D's) | 2.50·10⁻⁷ | 0 | 2.50·10⁻⁷ | 0.07 | 0.242442 | 0 | yes | yes | +8.5·10⁻⁸ | 3.3 |

The exact loss of the two windows, 2.7921108, takes 7.6 s at 128 bits, once.
A check is one native run of those windows in the realization; a pass is one
run of the accuracy entry over the 461 held-out windows. Both are wall times
of one `presk run`, start-up included. One prediction is 3.5·10⁻⁵ of `f32`'s
28 612 correct ones. (This table is D102's, at the tag `v0.10.0`, statement
D's realization the curve's of that day; D93's, on the same all-`f64` run of
the curve trained with the right gradient, had every other row to every
digit, the passes 3.8–11.7 s, and statement D's realization at 4.68·10⁻⁸;
D53's, on a checkpoint defect 3f trained
wrong (D92), had `f32` right on 18 615 predictions and every realization
within seven of it.)

- **The accuracy rule admits seven of the eight at 99.9 %, and all eight at
  99 %.** `int4` weights lose 136 predictions in 118 016, 0.48 % of `f32`'s
  correct ones, where their gap to the exact loss is 3.99·10⁻⁴, inside a
  budget of 10⁻³; the other six change by at most eleven. The budget
  separates the same realizations over thirteen orders of magnitude, four
  and a half of them between `f32` and `bf16`, from two windows.
- **The accuracy is not monotone in the gap.** `int4 pow2` is twice as far
  from the exact loss as `int4` and gains 115 predictions where `int4` loses
  136; `int8` gains six. A thresholded metric reads the computation coarsely
  and with noise, which is what it is designed to do: it asks whether the
  task survived, not how far the arithmetic went.
- **Attribution.** Every quantised gap is the weights' and not the kernels':
  the representation part is a hundred to two thousand times the arithmetic
  part. The accuracy rule has no way to say so, because it has a reference
  score and no reference computation.
- **Cost.** Per realization a check takes about 0.1 s and a pass about 4 s;
  the exact reference, paid once per checkpoint and sample, costs about two
  passes. On BERT-large the same accounting is in hours (`coverage` chapter).
- **Statement K, on other windows.** On this checkpoint and these two held-out
  windows `int8` weights are 3.90·10⁻⁵ from the exact loss, where K measured
  2.73·10⁻⁵ on `make gpt-trained`'s weights and two training windows: 1.4
  times more, and 256 times inside 10⁻². The error of a quantised loss is a
  signed sum of many terms (D49), and a measurement of it on two windows is
  one draw: on 0.6.0's weights K's windows had cancelled well, 3.23·10⁻⁶
  against 1.46·10⁻⁴ held out (D53). `int4` is 3.99·10⁻⁴ here against
  4.73·10⁻⁴ there.
- **`bf16` on a loss entry.** All of its 1.71·10⁻³ is the output's own
  rounding: the exact loss, 2.7921, is 1.71·10⁻³ from the nearest `bf16`
  number, 2.796875. A budget binds what the entry returns, and this entry
  returns one number stored in eight bits. With the metric in `f64`, the
  `bf16` model's loss is 2.3·10⁻⁵ from `f32`'s.

(Re-run at 0.11.0 on 2026-10-12, with D117's statement D realization:
the same table but for its wall times.)

### ε from the margins: an accuracy target as a budget (D118)

The accuracy rule tests the realization it ran. A budget on the logits
says more, once it is derived from the exact model.
- **A realization within ε of every logit** changes a decision only
  where the exact margin m, the top logit minus the second, is below 2ε.
  So its accuracy is at least the exact model's less the fraction of
  positions with m < 2ε, a property of the exact model alone.
- **ε\*(δ)** is half the δ-quantile of the exact margins, corrected so
  that the fraction with m < 2ε\* is at most δ with confidence 1 − α
  (one-sided Clopper–Pearson). Every realization within ε\* changes at
  most a fraction δ of the decisions.

`make margins` (`experiments/margins.sh`, five minutes) measures it on
the foil's model and windows. The exact logits come from the reference
at 128 bits, the margins from them. Each realization of the foil and of
the curve gets its gap on the logits (the largest of 10 857 472), its
flips (where `argmax` natively differs from the exact one) and the
argument's bound, #{m < 2ε}. The report is `docs/epsilon-2026-10-12.md`,
the tables `docs/epsilon/`.

The margins have no ties: the smallest is 1.2·10⁻⁵, the median 0.46.
With α = 0.05, ε\* is 5.4·10⁻³ at δ = 10⁻², **4.7·10⁻⁴ at 10⁻³** and
2.4·10⁻⁵ at 10⁻⁴ (5.7·10⁻³, 5.4·10⁻⁴ and 4.3·10⁻⁵ before the
correction). Taken from half the windows, ε\* leaves under δ of the
other half's margins below 2ε\*, at every δ.

| realization | gap on the logits | flips | bound #{m < 2ε} | certified at 10⁻³ | MLPerf 99.9 % |
|---|---|---|---|---|---|
| `f32` | 3.1·10⁻⁶ | 0 | 0 | yes | yes |
| `bf16` | 5.0·10⁻² | 480 | 13 661 | no | yes |
| `int8`, `f32` elsewhere | 7.8·10⁻² | 161 | 33 682 | no | yes |
| `int8` `pow2` | 1.1·10⁻¹ | 168 | 41 830 | no | yes |
| `int4`, `f32` elsewhere | 1.24 | 2 968 | 118 016 | no | **no** |
| `int4` `pow2` | 2.08 | 3 488 | 118 016 | no | yes |
| the curve at 10⁻¹, 10⁻² | 3.2·10⁻², 1.4·10⁻² | 69, 36 | 6 933, 3 090 | no | yes |
| the curve at 10⁻³ (statement D's) | **3.3·10⁻⁴** | **1** | 70 | **yes** | yes |
| the curve at 10⁻⁴ … 10⁻⁷ | 7.5·10⁻⁵ … 6.7·10⁻⁸ | 0 | 18 … 0 | yes | yes |

The curve's rows are 0.11.0's (D118). Measured again on 0.12.0's curve
(D124's realizations, `make margins` on 2026-10-15, D125; the foil's rows
the same, figure for figure):

| realization | gap on the logits | flips | bound #{m < 2ε} | certified at 10⁻³ |
|---|---|---|---|---|
| the curve at 10⁻¹, 10⁻² | 3.6·10⁻², 1.5·10⁻² | 76, 38 | 7 818, 3 174 | no |
| the curve at 10⁻³ (statement D's) | **2.5·10⁻⁴** | **0** | 51 | **yes** |
| the curve at 10⁻⁴ … 10⁻⁷ | 7.5·10⁻⁵ … 5.9·10⁻⁸ | 0 | 18 … 0 | yes |
| solved on the logits at 4.67·10⁻⁴ (V6) | 7.5·10⁻⁵ | 0 | 18 | yes |

Five of seven are within ε\*(10⁻³) again, and S holds.

**What it shows:**
- **Every realization's flips are within its bound**, by 28 to 250
  times, and every flip is at a position whose margin is at most twice
  the gap there. The statement fixed beforehand, S, holds: the curve's
  realizations within ε\*(10⁻³), five of seven, flip at most δ of the
  decisions (one at most), and each flips within its bound.
- **The margins order the realizations by their gap and bound what
  changes.** The accuracy rule reads a net score, which is not monotone
  in any gap.
  - `bf16` and the two `int8` rows change 480, 161 and 168 decisions,
    more than 10⁻³ of them. MLPerf admits them on net counts of −2, +6
    and −11; the margins refuse them, rightly at 10⁻³ and conservatively
    at 10⁻², where their gaps are 9–21× ε\*.
  - The two `int4` rows, which MLPerf separates, change 2.5 % and 3.0 %
    of the decisions, the admitted one more.
- **A budget on the loss does not bound the logits.** The curve's two
  loosest realizations are inside their budgets on the loss and 30 to 70
  times ε\*(10⁻³) on the logits. A loss averages the errors that each
  decision sees alone.

`presk solve` cannot solve at ε\* yet. It takes a scalar entry, ranking
its sites by `∂L/∂v`, and `max`, which would write the largest gap as a
scalar, is not implemented.

## 6. Make it pay: the statements of 0.6.0

Version 0.6.0 named four falsifiable statements before it began
(`presk-specs/implementation/presk-plan-0.6.0.md` §3.6), in the form the two
previous versions used, and added a fifth, J′, once J's failure had shown
what J was really asking. Three hold and two fail.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| H | the vector kernel is **bit-identical** to the scalar blocked one, in `f32` and `f64` | every bit | every bit, on both executors, integer weights included | holds |
| I | a 1024³ `f32` matmul at 3× the scalar figure of 0.4.0 (47–56 GFLOP/s) | ≥ 3× | lanes **1.15–1.25×** over scalar (before D46); on falicon after it 41–46 GFLOP/s, below 1× | fails, retired |
| J | the solved realization at a 10⁻³ budget faster per training step than all-`f64`, on the corpus GPT | ≥ 1.3× | **1.06×** at `D = 64`, **1.13×** at course scale | fails |
| J′ | the same, with the budget binding the training step's **update** and the solver free over the whole step, at course scale | ≥ 1.3×, inside 10⁻³ | **1.99×**, 2.02× re-measured; 2.2·10⁻⁴; on the corrected gradient (D93) 1.95×, 3.0·10⁻⁴; at the tag `v0.10.0`, under `--remat store` (D103), **1.75×**, 3.0·10⁻⁴ | holds |
| K | `int8` weights, per channel, inside a 10⁻² budget on the corpus GPT's loss, with trained weights | ≤ 10⁻² | **3.23·10⁻⁶**; 1.46·10⁻⁴ on held-out windows; on weights trained with the corrected gradient (D93) **2.73·10⁻⁵**, 3.90·10⁻⁵ held out | holds |

**H holds, and it is the claim the design supports.** 0.5.0's G asked for a
vector kernel's measured gap to fall inside its declared order's bound. D42
showed that lanes across output columns do not reassociate anything, so the
claim became an equality rather than a tolerance (D43). Tests assert it at
two and four lanes in `f32` and `f64`, on the kernel interpreter and in the
emitted code, and the integer-weight kernels are asserted bit-identical to
the dequantised path the same way (D45, D47). Since lanes change no bit,
they are on by default. What they buy depends on the kernel and the batch:
on BERT-large they make `int8` weights 1.5× faster, and scalar `f32` code is
15 % faster than `f32` in lanes at batch 20.

**I fails, and it is retired rather than pursued.** On the development
machine the lanes gave 1.25× in `f64` and 1.15× in `f32`, measured before
D46 took the conversion of `bench`'s inputs out of the timing. On falicon
after it, lanes give 1.1–1.2× in `f64` and nothing measurable in `f32`, and
`bf16` is the fastest row: at 1024³ the kernel waits for memory (`native`
chapter). The blocked form already ran eight independent accumulator
chains, so the parallelism lanes add was mostly there. The factor I asked
for is an instruction set's — the matrix coprocessor PyTorch reaches through
the platform BLAS (§3) — which `docs/kernels-roadmap.md` puts after this
version, behind the realization contract, since it reassociates.

**J fails, and J′ says why.** Solved at `D = 64` with the time objective,
the realization was 1.06× at `D = 64` (with D46's dilution, and inside noise
several per cent wide) and 1.13× at course scale, where all-`f32` was 1.87×.
The budget constrained the forward loss, so the solver had no licence over
the reverse sweep and the optimizer, which are most of a step (D50, question
60). J′ gives the budget the step itself: one optimizer step's parameter
update against the exact step from the same state (D52).

**J′ holds.** The step solved at 10⁻³ is `f32` in every function. Per step at
course scale, from the difference of a one- and a four-step run: all-`f64`
59.7 s, solved 30.0 s, all-`f32` 31.0 s, the mixed-precision recipe 34.6 s
(D52), and 61.1, 30.3, 30.8 and 35.2 s re-measured at the end of the
version: **1.99×**, then 2.02×. It is inside its budget by measurement, 2.2·10⁻⁴
against the exact step at `D = 64` and 6.7·10⁻⁵ against a native `f64` step at
course scale. Measured again in 0.10.0 (D93), after defect 3f of D92 was
found to have made every training of this model use a wrong gradient under
dropout since 0.5.0: the step solved on weights trained right, and with
the right gradient, is again sixteen functions `f32`, 1.95× faster per step
than all-`f64` (44.2 s against 22.7, every step faster than in 0.6.0), and
3.0·10⁻⁴ and 6.8·10⁻⁵ from the exact and the `f64` step. At the tag
`v0.10.0` (D103), under `--remat store`, it is 1.75× (24.4 s against 13.9),
level with all-`f32` again, every gap the same: the compiler's steps since
sped all-`f64` up more than all-`f32`, and the policy moves neither ratio
(all-`f32` 1.76× under `recompute`). Two limits, stated in D53. The solved step is level with
all-`f32`: the budget licenses leaving `f64` and the time objective refuses
`bf16`, which the profile predicts slower, so J′ is faster than `f64` and not
than FP32, MLPerf's reference. And at course scale the discharge is against
`f64`, because the exact step is not affordable there (question 62).

**K holds, and its margin was one favourable measurement.** On `make
gpt-trained`'s weights and two training windows, every matrix in `int8` and
`f32` elsewhere is 3.23·10⁻⁶ from the exact loss, three thousand times inside
10⁻². On D50's checkpoint and two held-out windows it is 1.46·10⁻⁴, seventy
times inside: the error of a quantised loss is a signed sum of many
terms (D49), and on K's windows it cancelled well. On weights trained with
the right gradient (D93) the two measurements agree better, 2.73·10⁻⁵ on
the training windows and 3.90·10⁻⁵ held out (§5), 366 and 256 times
inside. K is a claim about a
loss, a scalar averaged over 512 positions. On BERT-large's logits, at its
initialization, the same scheme is 5.7·10⁻², outside 10⁻² (`coverage`
chapter). There `int8` weights are also 2.2× faster than `f32`, the first
type narrower than `f32` to buy time on a model, because that kernel reads
the whole weight matrix again for every row of the result, and 4 MB of
`int8` stay in cache where 16 MB of `f32` do not.

**What the five together say.** The version's purpose was to make a
narrower type buy time inside a declared budget. It does, twice, each time
for a reason the plan did not expect. Not through vector lanes, which are
exact and free and buy little. Through the budget's scope: once the budget
binds the whole training step, the solver may leave `f64` wherever the
measurement allows, and a step runs 1.99× faster. And through memory: `int8`
weights run 2.2× faster on BERT-large, where the kernel streams its weights.
Against FP32 — MLPerf's reference, and every practitioner's — neither is yet
both inside a tight budget and faster: the solved step is level with `f32`,
and `int8` on BERT-large's logits is inside 10⁻¹, not 10⁻². The part of the
speed question that is a compiler question at the kernel level has an
answer — lanes are exact and buy little — and what is left is an instruction
set.

## 7. Ranking, not speed: the statements of 0.7.0

The research programme's central hypothesis, RA4, is that on regular tensor
kernels a **static cost model** orders more than 90 % of the pairs of
variants that autotuning distinguishes, at a decision cost lower by orders
of magnitude. No version had tested it: until 0.7.0 there was no space of
variants to rank. 0.7.0 built one — a layout space per loop nest (register
tile, k-block, loop order, lanes, pack, a thread split at a depth, softmax
passes), every point exact by construction (`realize` and `native`
chapters) — and a model that reads a layout, calibrated on a GEMM the test
never enters (`optimization` chapter). Five statements were fixed before the
version began (`presk-specs/implementation/presk-plan-0.7.0.md` §3.7). Two
hold and three fail.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| L | every point of every nest's space **bit-identical** to the scalar sequential kernel, natively and on the kernel interpreter | every bit | 17 879 of 17 879 points of day one checked on the interpreter; no output differing in any sweep, on either executor | holds |
| M | on each attention `ein` at `B = 16`, `f32`, the static model orders the distinguished close pairs right | > 90 % | scores **47.9 %** of 12 925 pairs, values **43.9 %** of 14 868 | fails |
| N | choosing statically costs less wall time than the exhaustive sweep of the same points | ≥ 10³× | **582–717×** at `B = 16`, **75–89×** at `B = 1` | fails |
| O | the model's pick within reach of the best measured point, every contraction of the entry at `B = 16`, `f32` | ≤ 1.25× | `Linear` **2.87**, scores **2.18**, values **3.01** | fails |
| P | the best measured point of the GEMM space at 1024³ `f32` against 0.6.0's point | ≥ 3× | **5.49×**, 460 GFLOP/s against 84 | holds |

**How it was measured** (D62; `docs/ra4-2026-09-24.md` is the report, written
by `presk rank` and not by hand). One night on falicon, 23–24 September 2026.
`make ra4` swept the course GPT's `SelfAttention.apply` at `D = 384`, `T = 256`,
`B` = 1 and 16, in `f32` and `f64` — every point of every nest's space, 3 305
points a type at `B = 1` and 4 857 at `B = 16` — then the GEMM at 1024³ in `f32`
(1 333 points) and a BERT-large-shaped attention on a sample of 41 points per
contraction, pinned to the ten Cortex-X925 cores, each point timed seven
times on the entry's own captured inputs. `make ra4-20` swept the attention
once more on all twenty cores with its own calibration; `make ra4-check` ran
every point of day one again on the kernel interpreter, in 130 shards (about
45 CPU-hours); and `make ra4-day2` repeated day one four hours later, in the
same night rather than on another day as the plan asks. The two runs agree
within 0.1 % in each configuration's sweep time, and 137 307 of their 160 509
close pairs — two points of a nest differing in one decision — are
*distinguished*: their median times differ by more than twice the larger
median absolute deviation on both runs, with the same sign (question 65).

**L holds.** No point of any sweep computed a different bit from the scalar
sequential kernel, natively or on the kernel interpreter: every register
tile, k-block, loop order, lane width, pack and thread split of the space.
Exactness by construction survives a layout space, which is what makes the
other statements about time alone.

**M fails, under any reading.** A pair the model predicts equal is *tied* and
counted as not right (question 70), and 37.4 % and 46.1 % of the two `ein`s'
distinguished pairs are ties. Leaving the ties out, the model orders 76.5 %
and 81.4 % of the rest right; on the pairs it *decides* (a predicted ratio
outside `1 ± u`), 79.0 % and 82.1 %. No contraction nest reaches 90 % on any
reading, at any shape or type; the closest is 88.1 %, on the decided pairs
of the `f64` values `ein` and of the `f64` `Linear` dot. By decision, over
the whole measurement:

| Decision | Distinguished pairs | Right | Tied |
|---|---|---|---|
| thread split | 39 144 | 96.7 % | 1.3 % |
| lanes | 7 289 | 93.2 % | 6.8 % |
| softmax passes | 27 | 100 % | 0 % |
| register tile | 69 136 | 27.7 % | 46.7 % |
| loop order | 17 565 | 30.0 % | 61.8 % |
| k-block | 2 998 | 0.6 % | 97.0 % |
| pack | 1 148 | 57.8 % | 0 % |

The model is right on the decisions its terms read — a rate per lane width
and per worker count, the cost of starting workers — and blind to the rest.
Every point of both `ein`s at `B = 16` in `f32` is predicted work-bound: a load
weighs nothing in its work count, and its traffic is counted from L2, where
every register tile of a form has the same footprint. So the values `ein`'s
tiles `[1, 4]`, `[1, 16]` and `[2, 8]` are all predicted 2.27 ms and measure
7.34, 3.39 and 4.66 ms. What a register tile saves is loads from L1 into
registers per multiply-add, and no term counts them (`optimization` chapter,
"What 0.7.0's layout model could not see"). That is the first risk the plan named,
at the one level the model does not have.

**N fails, and grows with the kernel.** The static side is `presk plan`'s:
the entry run once to capture its launches, then every point lowered and
costed, about 0.2 ms a point, 1.5 s for the whole entry at `B = 16`. The
exhaustive side is `presk autotune` without the interpreter's check: eight
launches a point of a few milliseconds each, 1 048 s at `B = 16` in `f32`.
The ratio is 582–717 at `B = 16` and 75–89 at `B = 1`, where the kernels are
short. On the GEMM at 1024³, whose kernels are long, it is 1 460–1 538.

**O fails.** The model's pick is `presk plan`'s: its best prediction, a tie
broken by the space's order. On the attention's three contractions at `B = 16`
in `f32` it is 2.87, 2.18 and 3.01 times slower than the best point measured.
On the GEMM, 21 points tie at its best prediction; the first, the tile
`[1, 4]`, runs in 28.6 ms, slower than 0.6.0's own point.

**P holds.** The best measured point of the GEMM space — the tile `[4, 16]`
in four lanes, order `JI`, split over ten workers at depth 1 — runs 1024³ in
4.67 ms, 460 GFLOP/s, against 84 GFLOP/s for 0.6.0's point in the same sweep:
5.49×. It is 0.6.0's retired statement I re-posed, its factor of three
found in the layout rather than in the lanes, and it is within 1.6× of the 710
GFLOP/s PyTorch reaches on twenty cores (§3). P is a by-product: it is about
the best point, which measuring found, not about the model's pick.

**Elsewhere, the same picture.** On all twenty cores: M 48.4 % and 45.8 %, O
4.13, 2.31 and 2.80, N 91–762×, the split right on 93.6 % of its pairs —
the unlike cores did not make the thread term unlearnable
(`docs/ra4-20cores-2026-09-24.md`). The BERT-shaped sample gives 13–17
distinguished close pairs per contraction, too few to rank, and top-1 regrets
of 1.90–4.96.

**What does hold.** The model is a good **pruner**, which RA4 does not claim:
on every contraction nest at the attention's shapes, the points it predicts
within `1 + u` of its best contain the best measured point or one within
1.08× of it; at `B = 16` in `f32` and on the GEMM, the best itself, among
36–201 candidates of 1 333–1 717 points. Measuring its candidates alone
finds the best point at 2.4–15 % of the sweep's cost. And **the space pays**:
its best point beats 0.6.0's by 9.2× and 16.5× on the two `ein`s at `B = 16`,
by 47× and 77× at `B = 1`, where 0.6.0 ran them on one worker with the tile
`(1, 1)`, by 2.6× on the `Linear` dot and by 5.5× on the GEMM.

**Two corrections made before the statements were read.** The first reading
of the two runs counted a tie as right, since Rust's `0.0.signum()` is `1.0`,
and took the first *key* among tied predictions as the pick. It gave M 73.9 %
and 74.2 % and O 1.73, 1.25 and 1.41. A model that predicts two variants
equal has not ordered them, and the pick is the one `presk plan` makes, so
both were corrected and the report regenerated from the same measurements;
every verdict is the same under both readings. Counting a tie as half right,
as concordance indices do, would give M about 66 % and 67 %, still a failure.

**What the five together say.** Exactness costs nothing here: every one of
the 17 879 points measured is bit-identical, and the best point of each
contraction is 2.6 to 77 times faster than what 0.6.0 emitted. The static model does not choose
among them as RA4 says, and the reason is named rather than guessed: it has
no term for the loads a register tile saves. The model ranks what it can see,
and it narrows the search to a few per cent of the space that holds the best
point. Whether a load term makes it rank is not something these measurements
can answer, since a model designed after seeing where this one failed would
be fitted to its test; the term is recorded in `docs/kernels-roadmap.md`, for
a later version and a fresh sweep. 0.7.0 closes on these results, and 0.8.0
took them up (§8).

## 8. Ranking, re-tested: the statements of 0.8.0

0.7.0's model failed RA4 for a named reason: it weighed no load from the
first-level cache into registers, and under a maximum its traffic terms
never bound, so it tied register tiles, loop orders and k-blocks (§7).
0.8.0 changed the whole model (`optimization` chapter): a term for one
worker's register loads and stores, `N_ld/ρ_g`; a p-norm of the five terms
in place of the maximum, its exponent fitted (`q` = 1.06); a calibration
widened to five tiles, sizes to 2048 and two thin products; and two rules
of the planned cost, footprints in whole cache lines and one operation of
work per loop iteration. It was designed on 0.7.0's measurements alone,
which became the **development set** and decide nothing, and **frozen**
at the tag `ra4-model-0.8.0` on 2026-09-25 before any new measurement.
`presk rank --frozen` refuses a day measured on a commit that does not
descend from it. Then two fresh sweeps tested it. Six statements were fixed
before the version began (`presk-specs/implementation/presk-plan-0.8.0.md`
§3.6): L, M, N and O are 0.7.0's, with the same thresholds; Q is new; D
reports the development figures. Three hold and two fail.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| L | every point of both fresh sweeps **bit-identical** to the scalar sequential kernel, natively and on the kernel interpreter | every bit | 9 286 of 9 286 points of F1's day one and 16 368 of 16 368 of F2's checked on the interpreter; no output differing on any day or shard | holds |
| M | on each attention `ein` of F1a, F1b and F2 at `B = 16`, `f32`, the frozen model orders the distinguished close pairs right | > 90 % | F2 **92.7 %** and **91.7 %**; F1a **87.4 %** and **88.5 %**; F1b **85.4 %** and **88.8 %** | fails |
| N | choosing statically costs less wall time than the exhaustive sweep, each fresh configuration, each day | ≥ 10³× | F1a 1 013–1 016×, F1b 2 880–2 933×, F2 at `B = 16` 2 168–2 734×, F2 at `B = 1` **218–252×** | fails |
| O | the model's pick within reach of the best measured point, every contraction nest of the configurations M names | ≤ 1.25× | at most **1.05** | holds |
| Q | measuring only the points predicted within `1 + u` of the model's best finds a point within reach of the best, at a fraction of the sweep's time, every contraction nest of every fresh configuration | ≤ 1.05×, ≤ 10 % | 18 nests: the best candidate within **1.03×**, at most **1.8 %** of the sweep | holds |
| D | the same, on the development set under the frozen model | reported | M 89.3 % and 89.6 %; O 1.07, 1.00, 1.01; Q on 12 of 13 nests | — |

**How it was measured** (D72–D74; the reports are
`docs/ra4-f2-2026-09-25.md`, `docs/ra4-f1a-2026-09-25.md` and
`docs/ra4-f1b-2026-09-25.md`, written by `presk rank`).
- **F2**, 0.7.0's shapes on the other cores: the course GPT's
  `SelfAttention.apply` at `D = 384`, `H = 6`, `T = 256`, `B` = 1 and 16,
  `f32` and `f64`, every point of every space, 16 324 points a day, pinned
  to the ten Cortex-A725 cores. Those cores have a quarter of the X925's L2
  and their own calibration, made in the frozen form and space after the
  freeze: `u` = 0.168, where the X925's is 0.132. It tests whether the model
  carries over to another microarchitecture.
- **F1**, new shapes on the ten X925 cores, priced by the frozen profile
  itself: BERT-large's attention (F1a: `D = 1024`, `H = 16`, `T = 384`,
  `B = 1`), which 0.7.0 only sampled, and a head dimension of 128, which the
  development set never had (F1b: `D = 1024`, `H = 8`, `T = 512`, `B = 2`),
  both in `f32`, every point, 9 264 points a day. It tests whether the model
  carries over to other shapes.
- **Twice each**, on 2026-09-25: F2 from 12:07 and from 16:45, F1 from 20:02
  and from 22:35, the other cores idle. The plan asked for two dates, and
  the user decided on one, hours apart (D72): a date is only a proxy for
  independent conditions, and a run at midnight would have been little more
  independent. The days' sweep times agree within 1.3 %.
- **Before each**, a dry run of the longest nest sized the day (1 h 41 min
  estimated for F2, 1 h 45 min measured; 52 min and 48 min for F1), so both
  were swept whole and no M rests on a sample.
- **Statement L's check** ran apart on the kernel interpreter: 88 shards for
  F2 (1 h 58 min) and 100 for F1 (1 h 44 min), on every core.
- **Every day, check and calibration is archived** read-only with its
  checksums, since none can be taken again, and ranked in step 7 from
  copies checked against those checksums.

**L holds.** No point of either sweep, 25 588 a day and 0.6.0's beside
them, computed a different bit from the scalar sequential kernel, natively
or on the kernel interpreter, on the A725 cores as on the X925.

**M fails: the model carried over to another core, not to new shapes.** On
F2 it ranks 92.7 % and 91.7 % of the two `ein`s' distinguished pairs right,
above RA4's 90 % and above its own development figure (89.3 % and 89.6 %),
where 0.7.0's model ranked 47.9 % and 43.9 % of the same shapes' pairs. On
F1 it ranks 85.4–88.8 %, 0.8–3.9 points below the development figure. More
than 90 % would have taken 200 and 134 more right pairs at F1a, 1 064 and
183 at F1b. By decision, over every nest:

| Decision | F1 | F2 | development set | 0.7.0's model (§7) |
|---|---|---|---|---|
| register tile | 90.7 % | 92.4 % | 89.1 % | 27.7 % |
| thread split | 99.1 % | 98.1 % | 97.7 % | 96.7 % |
| lanes | 100 % | 98.7 % | 100 % | 93.2 % |
| loop order | 60.6 % | 72.6 % | 67.0 % | 30.0 % |
| k-block | 64.3 % | 53.4 % | 65.9 % | 0.6 % |
| pack | 47.9 % | 62.9 % | 57.8 % | 57.8 % |
| every pair | 87.6 % of 76 830 | 90.4 % of 134 022 | 88.5 % of 137 307 | 50.8 % |

Ties are gone: 0.1 % of the pairs, where 0.7.0's model tied 34 %. The
pairs the model decides, predicted apart by more than `u`, are 99.2 % right
on F1 and 98.8 % on F2. **What M is decided by** is how the model orders
the pairs it predicts within `1 + u` of each other: 97–98 % of the misses on
F1, and 99 % on F2, are such pairs. It orders 69–73 % of them right on F1
and 78–81 % on F2, and they are 39–46 % of F1's distinguished pairs. They
are tiles (about 10 % wrong, on the largest counts), loop orders (47–71 %
right in M's nests), the pack and k-blocks, whose pairs almost all lie
within `u`.

**One decided failure, which names a term.** At F1b's scores `ein` the model
decides 121 loop-order pairs and gets 67 wrong. All 67 are one-row tiles,
`[1, 8]` and `[1, 16]`, packed, in four lanes; every decided order pair of
a two- or four-row tile is right. The order with `I` outside `J` reads the
whole packed operand once a row of the output, where the other keeps a
panel of it. The model predicts it 15–23 % slower, for its L2 term, and it
measures 1–16 % faster. With `q` = 1.06 the p-norm is nearly a sum, which
adds the L2 time to the load time, and a one-row kernel does not pay it on
top of its loads. The packed operand of one head is 256 KiB at F1b and
96 KiB at F1a, against 64 KiB of L1. At 0.7.0's shapes it is 64 KiB, and on
F2 and the development set every decided order pair at M's configuration is
right.

**N fails at batch one on the A725 cores**, as the plan foresaw. The static
side is now a capture of the entry's shapes, which runs no kernel (D65),
and the costing of every point: 0.45–0.85 s on the X925's configurations,
0.96–1.55 s on the A725's at `B = 16`. At `B = 1` on the A725 a sweep takes 143–170 s and
the static side 0.57–0.78 s, of which the capture is 0.03–0.05 s: costing
3 305 points at 0.16–0.22 ms each is the rest. Clearing 10³ there would take
costing four to five times faster. At F1a, `B = 1` on the X925, the ratio
clears 10³ by 1–2 %. With the capture that runs the entry, as 0.7.0 counted
it, the ratios would be 161–173× on F2 at `B = 1`.

**O holds.** The model's pick, `presk plan`'s, is within 1.00–1.03× of the
best point measured on F1a's three contractions, 1.00–1.05× on F1b's and
1.00–1.01× on F2's at `B = 16` in `f32`, where 0.7.0's was 2.18–3.01× off.
Elsewhere the worst top-1 regret on a contraction nest is 1.11, on F2's
values `ein` in `f64`. The space pays on the new shapes as on the old: the
best point beats 0.6.0's kernel by 27–132× on F1's three contractions, and
by 3.2–13.7× on F2's at `B = 16` in `f32`.

**Q holds.** Q is not RA4's claim. It tests the chooser 0.7.0's data
suggested: prune statically, then measure the candidates, the points
predicted within `1 + u` of the model's best. Its cost is question 72's:
the whole configuration's static side, charged to each nest, plus the
candidates' measured time, over the nest's exhaustive sweep, on each day
(D74).

| configuration | candidates | best candidate / best | cost (question 72) | candidates' share of the timed launches |
|---|---|---|---|---|
| F1a, `B = 1`, `T = 384` | 12–18 | 1.00 | 0.3–0.6 % | 0.10–0.16 % |
| F1b, `B = 2`, `T = 512` | 11–13 | 1.00 | 0.1–0.2 % | 0.05–0.06 % |
| F2, `B = 1`, `f32` | 5–15 | 1.00 | 1.2–1.5 % | 0.05–0.16 % |
| F2, `B = 1`, `f64` | 13–19 | 1.00–1.01 | 1.5–1.8 % | 0.21–0.31 % |
| F2, `B = 16`, `f32` | 13–19 | 1.00 | 0.2 % | 0.10–0.12 % |
| F2, `B = 16`, `f64` | 19–28 | 1.00–1.03 | 0.4–0.6 % | 0.20–0.44 % |

Where the sweep is short, the static side is most of the cost: the dearest
nest, the scores `ein` at `B = 1` in `f64` on the A725, spends 0.77 s
choosing and 0.15 s measuring its candidates, against 50.4 s swept.

**D, the development figures, decided nothing.** They foresaw F1's M within
four points and understated F2's. They showed Q's one miss, the scores
`ein` at `B = 1` in `f32` at 1.08×, which the fresh sweeps do not repeat.
The development report under the frozen profile is
`docs/ra4-dev-model-0.8.0.md`, which `make ra4-dev-frozen` regenerates byte
for byte.

**What the six together say.** The load term and the combination did what
0.7.0's failure asked of them: the model now reads register tiles, ties
almost nothing, and is right on nearly every pair it decides, on a
machine it was not designed on. RA4's 90 % is met on the other cores and
missed on new shapes, by the pairs the model predicts within its own
uncertainty. Its pick is within 1.05× of the best, and measuring its
candidates finds the best at under 2 % of a sweep's cost. The static choice
is cheap enough at batch 16, and costing a point is what keeps it from 10³ at
batch one. What the failures point to (the L2 term beside the loads of a
one-row tile, the order of pairs within `u`, the cost of costing a point) is
recorded in `docs/kernels-roadmap.md`, for a model to be frozen and tested
again on a sweep it was not designed on.

### Addendum: RA4's third fresh test, T3 (0.10.0 step 2)

0.10.0 step 1 made the derivatives of `dot` contraction nests of the layout
space (`dot_vjp_l`, the builder's `nt` form; `dot_vjp_r`, a sum over every
leading axis). They are the first nests the frozen model has never seen:
it was fitted on a GEMM and tested on attention. Plan 0.10.0 asks O and Q
of them, as stated above, on the course GPT's gradient at `B = 16`, `T =
256`, `f32`, on the ten X925 cores, priced by the frozen profile, neither
changed nor refitted (statement T3, D90). The gradient launches four
derivative nests at eight shapes: `Linear.apply.vjp/%11` (`dot_vjp_l`) and
`%12` (`dot_vjp_r`) at the blocks' three product shapes, and the head's
`GPT.apply.vjp/%28` and `%29`, 11 028 points. They were swept whole twice on
2026-09-29 (3 h 05 min each, within 0.3 % of each other), every point
checked on the kernel interpreter in between (7 h 06 min on every core),
and ranked by `presk rank --frozen`; the report is
`docs/ra4-t3-2026-09-29.md`, and the measurements are archived read-only in
`~/presk-data/ra4-t3-0.10.0`.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| O | the model's pick within reach of the best measured point, every derivative nest and shape | ≤ 1.25× | 1.00–**1.19** | holds |
| Q | the candidates, the points predicted within `1 + u` of the model's best, hold a point within reach of the best, at a fraction of the nest's sweep | ≤ 1.05×, ≤ 10 % | within **1.01×** on all eight; cost 0.2–1.7 % on the blocks' six, **16.4–19.9 %** and **9.9–11.9 %** on the head's two | fails |
| L | every point bit-identical to point 0, natively and on the kernel interpreter | every bit | 11 036 of 11 036 points checked, none differing | holds |
| — | close pairs ranked right, reported with no threshold (M's 90 % was set for attention's `ein`s) | reported | **88.5 %** of 92 279, 85.2–95.4 % per nest; 98.6 % of those the model decides | — |

**T3 fails, on Q's cost alone.** On every nest the candidates, 2 to 24
points, hold the best within 1.01×, and measuring them takes 0.03–3.1 s.
But Q's cost charges each nest the whole configuration's static side
(question 72), and choosing statically for a gradient means capturing the
shapes of the whole gradient, 147 nests at 179 shapes, 3.9–5.0 s, before
costing four of them in 1.1 s. The head's two nests sweep whole in 30 and 51
s, so that side alone is 16–20 % and 10–12 % of their sweep. The blocks'
nests, 6 to 49 minutes a sweep, cost 0.2–1.7 %. The rule was not changed
after the fact, and the model was not refitted.

**The model carried over to the derivatives.** 88.5 % of the pairs right
is attention's figure (F1 87.6 %, F2 90.4 %, the development set 88.5 %),
and the classes fall as they did: the thread split 98.4 %, the lanes
99.1 %, the register tile 90.3 %, the loop order 73.2 %, the pack 73.4 %,
the k-block 53.9 %. 92.4 % of the misses are pairs the model predicts
within `u` of each other. The pick is 1.19× off at worst, on the head's
`dot_vjp_r`. There, and on one of the blocks' shapes (1.17×), it loses by
decisions the model predicts within 7 % and 2 % of the best. On the square
`dot_vjp_r` (1.16×) the pick's launches alternate between two speeds on
both days, and the point best on average, the order `IKJ`, is a decided
miss: predicted 1.75× slower than the pick through its L3 and memory
terms, it measures faster.

**What the space buys the gradient.** Over one gradient's derivative
launches, the layouts `presk plan` states (one per nest over its three
shapes) take 479 ms natively on the captured buffers, 1.05× the best point
of each shape, where 0.9.0's derivative kernels take 11.0 s, 24× the best.

## 9. A C program: the statements of 0.9.0

0.9.0 is not a test of a hypothesis of the research programme. It gives an
entry a way out of the compiler, as a C program that runs without it
(`export_c` chapter), and its five statements, fixed before the version
began (`presk-specs/implementation/presk-plan-0.9.0.md` §3.8), say what the
program owes: the native backend's bits, the exact reference's band, a
clean build on the compilers that matter, and `presk serve`'s tokens; its
speed is reported and decides nothing. The test set is the eleven entries
`make plan-check` compares (an MLP's loss and its gradient, a LayerNorm
and an RMSNorm, MiniGPT's forward pass, its training loss, that loss's
gradient through attention, an optimizer step and `next_token`, and the
course GPT's attention and forward pass, all at small constants) and the
course GPT at its course constants. Four hold; the fifth
decides nothing.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| C1 | every exported entry of the test set, in `f64`, `f32` and `bf16`, under the default layout and under a plan from `presk plan`, computes outputs **bit-identical** to `presk run --backend native` with the same realization, plan, dimensions, seed and weights | every bit | every entry in the three types, on synthesized parameters and with their zeros drawn, under a mixed realization, under `presk plan`'s plans, both hosts, with OpenMP and without, on one thread and every thread; the course GPT at `T = 256` with its trained checkpoint, at `B = 16` under its plans (201 of 228 launches divided among ten threads); 42 programs with integer weights; every nest in the SCoP form: no bit differs | holds |
| C2 | the same outputs lie within the realization's conformance band of the exact reference, as the native backend's do | no output outside the band where the native backend's is inside | the outputs are the native backend's, so outside the band on exactly its elements: 34, 121 and 99 of `loss_grad`'s 8 768 in `f64`, `f32` and `bf16`, and a few cancelling logits of the forward passes (question 75) | holds |
| C3 | the exported C compiles without warnings (`-Wall -Wextra -std=c11`) with GCC and Clang on aarch64, and cross-compiles for x86-64 | no warning or error | GCC 13.3 with and without OpenMP; Clang 21.1 (`zig cc`), linked, run and bit-identical, `generate` giving `presk serve`'s tokens; x86-64 (`x86_64-linux-musl`), linked; the SCoP mode's files on Clang 23.1.2 too: none warns | holds |
| C4 | the exported program's wall time against the native backend's, one thread and every thread, on the course GPT's forward at `B = 16` | reported | every thread: native 1 094 ms, C 290 ms; one thread: 3 132 and 1 700 ms (`f32`, `T = 256`, under the frozen profile's plan) | — |
| C5 | the tokens the exported `next_token` generates over a sliding window are those `presk serve` generates, greedy and sampled with the same seed | every token | `generate`'s `--json` document is `presk serve --backend native`'s byte for byte, greedy and at three temperatures and four seeds, in the three types, through both hosts, on the example with its zeros drawn and on the trained checkpoint, with integer weights too | holds |

**How it was measured** (D77–D85). `make test` exports, builds and runs
every program through the binary and compares its output file with the
native backend's with `cmp`; `make c-conformance` adds the course GPT, the
trained checkpoints, the plans, Clang and x86-64 through zig from PyPI, and
Polly from LLVM's release; `make c-speed` measures C4. Each check was
checked by mutations of the program that each change its outputs: a
dropped launch, a slice off by a layer, a loop bound off by one, an early
release, a gap between threads' pieces, a skipped draw, a scale one channel
off, and others (`export_c` chapter, "What is checked").

**What makes C1 hold.** Two things the plan named, and one it did not
know. The program is built without floating-point contraction (`-std=c11
-ffp-contract=off`; GCC contracts in its default GNU mode, and 14 of 246
primitive runs then move). It ships the transcendental functions the
native backend calls through the `libm` crate, which ports musl as it was
before 1.1.23: current musl's `exp` and `log` differ from it on 8–10 % of
inputs, glibc's on 1.2–9.8 %, and musl 1.1.22's on none of 10⁶ inputs a
function. And the native backend's own `sample` called the platform's
`exp`, not the crate's, which step 6 changed: 540 generations of 32 tokens
did not move.

**C2 became a check of a copy.** The band is per element, and the native
backend's outputs cross it where large terms cancel, in gradients and in
forward passes, where the kernel interpreter, another conforming
executor, crosses it on other elements. So C2 was judged against the
native backend's own gap from step 0 on, and question 75 asks how
conformance should judge an output that cancels.

**C4, reported.** The C program is 3.8 times faster on ten threads and 1.8
times on one. On one thread both run the same kernels, so the difference
is the code generator (GCC `-O2` against Cranelift) and the host (the
native host copies each stacked layer's slice, the program slices by
pointer); on ten, the native backend also starts a thread per piece at
each divided launch, where OpenMP keeps a pool. The shares are not
measured apart.

**What the five together say.** A Presk entry, a gradient or a sampling
loop included, can leave the compiler as a small C program that computes
the native backend's bits on any compiler and thread count, so the exact
reference's measurements of the native backend (its gap, a budget's
verdict) carry over to it unchanged. The polyhedral comparison the SCoP
mode prepares, a tool's transformation of the same nests against RA4's
ranking, comes after 0.9.0.

## 10. A training step that pays: the statements of 0.10.0

0.10.0 took the causes the PyTorch baseline at 0.9.0 found (§3, "Three
sides at 0.9.0", D87) in order. The derivatives of `dot`, nine tenths of
the C gradient at the default layout and outside the layout space, became
nests of it (step 1), which the frozen model ranks (step 2); a gradient
stopped recomputing its forward pass, keeping its residuals instead
(`--remat store`, steps 3, 4 and 4b); the native host was measured, then
fixed (steps 5 and 6); the bias of a linear layer became its product's
epilogue (step 7); and, at the user's request, the language's four
decisions of 2026-09-29 went in (step 7b). Its six statements were fixed
before the version began, the sixth with step 7b
(`presk-specs/implementation/presk-plan-0.10.0.md` §3.5). Every gap the
version measured is against the exact reference, and the reference's own
gap was measured too (D98, `make delta-p`): five to ten units of roundoff
at the reference's precision, 3·10⁻³⁸ at 128 bits, 21 orders of magnitude
below the resolution of binary64 at which every gap is read (the
`precision` chapter, "The reference's own gap"). Three hold, two fail, one
decides nothing.

| # | Statement | Asked | Measured | |
|---|---|---|---|---|
| T1 | every change of 0.10.0 is exact: the derivative nests at every point of their spaces, the gradients under `--remat store` and `recompute` and without their dead code, bit-identical to 0.9.0's native outputs on the test set in `f64`, `f32` and `bf16`; C1 keeps holding | every bit | at each step against `main`'s binary, on the test set's eleven entries on the three executors in the three types and under plans, the course GPT's forward pass and gradient, and training runs (87 to 184 comparisons a step; 154 at step 8); every point of the derivative nests' spaces (4 280) and of the products with their epilogues; C1 on the test set and the course GPT: no bit moved but by design (the gradient defects 3a and 3f fixed, D91, D92) | holds |
| T2 | the course GPT's training step at `B = 16`, `T = 256`, `f32`, in C under the frozen profile's plans, within 2× of PyTorch's, ten X925 cores pinned, the median of seven | ≤ 2× | 995.4 ms against 1 008.6: **0.99×** (12× at 0.9.0); 0.83× with the C program's threads waiting as the native ones; every core unpinned 0.76× | holds |
| T3 | on each derivative nest of the course GPT's gradient, 0.8.0's statements O and Q, the frozen model on nests it has never seen | its pick ≤ 1.25× the best; its candidates within 1.05× at ≤ 10 % of the sweep's time | O holds, 1.00–1.19×; Q fails on the head's two nests (16.4/19.9 % and 9.9/11.9 % of their sweeps' time), by its static side, the whole gradient's capture charged to nests that sweep in 30 to 51 s; the blocks' nests 0.2–1.7 % (§8's addendum, D90) | fails |
| T4 | the native backend's forward pass and training step under the plans within 1.3× of the C program's, same plans and cores, the step like for like, the C program's threads waiting as the native ones | ≤ 1.3× on both | the forward pass 340.4 ms against 223.5, **1.52×**; the step 1 340.6 against 836.8, **1.60×**; by OpenMP's default waiting 1.25× and 1.35× | fails |
| T5 | the gradient's peak memory under `store` and `recompute`: the C program's arena, the course GPT's gradient at `B = 16`, `T = 256`, `f32` | reported | 2.49 GB storing, 0.91 GB recomputing (2 486 774 208 and 905 256 128 bytes) | — |
| T6 | the language's four decisions change no value: every output without a draw of `sample` bit-identical to the additive mask's, three types, C1, the amended examples checked | every bit | 147 comparisons identical, only `next_token`'s draws moving, now its key's (D99) | holds |

**How it was measured.** `make pytorch-baseline`, at every step that
could move T2 or T4 and once more on the version's committed tree (D100,
`docs/pytorch-baseline-0.10.0.md`, the run of record): the course GPT on
three sides, presk's native backend, the C program `presk export --target
c` writes, and PyTorch 2.14 running the model the PyTorch export writes,
ten X925 cores pinned, the median of seven, the C program timed under
OpenMP's default waiting and with its workers spinning as the native
pool's. T3 by two days of `presk autotune` on the derivative nests and
`presk rank --frozen` (`make ra4-t3`, archived in
`~/presk-data/ra4-t3-0.10.0`). T1 and T6 against `main`'s binary at each
step, on outputs, byte for byte.

**What the compiler itself takes.** Beside the reference's own gap, the
time of the compiler's commands was measured on the version's tree (D101,
`make compile-times`, the report `docs/compile-times-2026-10-02.md`):
`presk check`, `presk calibrate --precision` on two samples, `presk solve
--budget 1e-3` and the exact reference on one sample, wall seconds on one
X925 core, on the course GPT at its course and reduced constants, the tiny
MiniGPT, and two of BERT-large's blocks. The front end is milliseconds
(7 ms on the course GPT). The rest is the reference interpreter at 128
bits: on the course GPT at its course constants, 3 minutes for the
reference on one window, 25 to calibrate on two, 45 to solve (a reverse
sweep, the reference and the proposal's measurement, each on two
windows); on BERT-large's two blocks, on their SQuAD loss, 10 minutes for
the reference, 85 to calibrate and 2 h 54 min to solve, which puts their 63
sites away from `f64` (59 in `f32`, 4 in `bf16`) and measures 5.1·10⁻⁷
under a budget of 6.3·10⁻³. At 0.11.0, the solver offering the values the
program stores alone (D112), the same solve in the same time puts all 25
away from `f64` (21 in `f32`, 4 in `bf16`), 2.03× less traffic, and
measures 1.05·10⁻⁵: two of the 25 have another type, the query `bf16`
and the layer norm's mean `f32` (D117, `docs/compile-times-2026-10-09.md`).
At 0.12.0, the solver offering the functions' results and the parameters
too (D123), it puts all 28 stored values and all 23 parameters away from
`f64`, 2.00× less traffic with the weights counted, in 2 h 49 min, and
measures 9.2·10⁻⁷ (D124, `docs/compile-times-2026-10-13.md`). Until
0.12.0 `presk solve` took a scalar entry, a loss, and refused any other at
once (D101). BERT's `squad`, which returns its logits, was refused that
way, so its row is measured on `squad_loss`, the loss `run_squad.py`
trains it on (the `coverage` chapter); since 0.12.0 a tensor entry is
solved under a budget on its largest elementwise gap (D123), and the row
stays on the loss, D101's.

**A second model, after the tag.** The Llama 2 decoder of
`coverage/llama.psk`, at the 7B configuration's width and one of its 32
blocks, ran by BERT's protocol on 2026-10-04 (D104, `make llama`, the
`coverage` chapter's *The decoder, run*): its exact logits in 1 h 55 min
and 87.6 GB, `f32` the one narrowed type inside 10⁻² (6.2·10⁻⁶), `int8`
weights 1.6× to 1.9× faster than `f32` and 4.9·10⁻² away, the quantised
gap the weights' rounding. It had no solve: at the tag `presk solve` held
more than the machine has at one block of that width, even on 16 tokens.
On the way it found a defect of the native backend, a quantised parameter
read through `split` computed wrongly.

**The second model's budget, at 0.10.1.** Version 0.10.1 holds each
exact value once (D105) and fixes that defect, and on 2026-10-05–07 the
decoder's budget was solved (D106, the `coverage` chapter's *The decoder,
run*): `presk solve --budget 1e-3` on its next-token loss, one block, 384
tokens, in 14 h 07 min and 71.7 GB, 52 of 56 sites narrowed, 1.99× less
traffic than all-`f64`, measured 4.3·10⁻⁶ against 1.1·10⁻² (at 0.11.0,
the solver offering the stored values alone: 24 of 25 at 1.96×, measured
2.0·10⁻⁵, D117; at 0.12.0, the results and the parameters offered too: 31
of 32 and the 13 parameters at 1.99× with the weights counted, measured
2.6·10⁻⁵, D124), beside BERT's two blocks above (5.1·10⁻⁷, 2.00×;
1.05·10⁻⁵, 2.03× at 0.11.0; 9.2·10⁻⁷, 2.00× at 0.12.0) and statement D's
course GPT (2.1·10⁻⁵, 2.00×, D102 and D117; 2.1·10⁻⁶ at 0.12.0, D124). Every output 0.10.0 gives is 0.10.1's, so every
number of this chapter stands.

**The second model against PyTorch, after 0.10.1.** The same block's
forward pass by §3's protocol, the ten X925 cores pinned (D107, `make
pytorch-decoder`, the `coverage` chapter's *The decoder, run*,
`docs/pytorch-decoder-0.10.1.md`): under the frozen model's plan the native
backend takes 13.6× PyTorch eager's at batch 1 and 14.4× at batch 20, 6.8×
and 8.3× with `int8` weights on every matrix, where the course GPT's
forward pass took 2.32× (D100); two thirds of its kernels' time is the
attention's output projection, an `ein` over three summed axes that the
layout space does not tile.

**The decoder block under its solved realization, at 0.12.0** (V5, D124;
`ROWS=realizations` of `experiments/pytorch_decoder.sh`, each row in
several processes). Its forward pass at batch 1, in ms:

| realization | three rounds | batch 20 |
|---|---|---|
| all-`f64` | 7 023 / 8 919 / 8 874 | 105 662 |
| all-`f32` | 2 439 / 3 880 / 3 825 | 67 554 |
| 0.11.0's solved | 9 714 / 9 670 / 9 800 | 108 394 |
| 0.12.0's solved | 2 441 / 2 798 / 2 524 | 46 631 |

- *Against 0.11.0's solved realization, 3.5 to 4 times faster.* 0.11.0
  left the functions' results in `f64`, the output projection among them,
  and converted every weight at the kernels' edges, 944 MB a call. 0.12.0
  narrows the results and stores the weights in `f32`, read in place: what
  it converts is the residual stream, 18.9 MB a call, for the one `f64`
  nest.
- *As fast as all-`f32`, not faster.* The two plans are the same but for
  one scalar nest. All-`f32` ran at 3.7–3.9 s in 7 of its 8 processes
  (D107's 3 744 ms) and at 2 439 ms in one; the solved realization at
  2.4–2.9 s in all 8. The cause of all-`f32`'s slow processes is not
  found (`docs/kernels-roadmap.md`, "A launch inside a call slower than
  alone").
- *So every timing of record runs each row in several processes* since
  D126, at least three, and reports the median across them and the
  fastest. D107's 13.6× above is one process, in the slow mode. Against
  V5's eight all-`f32` processes, the median gives 13.9× and the fastest
  8.9×, PyTorch's side not timed again.

**What T2 says.** A training step of the course GPT in C is level with
PyTorch's on these cores, where 0.9.0 left it 12× behind. In order of what
each bought on the C step: the derivatives of `dot` in the layout space
(12.2 s → 2.85 s, step 1), the dead code after differentiation (1.88 s,
step 3), the residuals stored (1.36 s, step 4), the fusion groups kept only
where a realization can see them (1.13 s, step 4b), the bias as an
epilogue and the C program's kernels always inlined (1.01 s, step 7 and its
follow-up), each measured on the run of its step. Under the plan the
derivatives of `dot` run at 633 GFLOP/s, faster than the gradient's other
contractions (539), and the gradient's contractions are 70 % of its
kernels' time. Every change was a layout or the same operations in the
same order, so the step that is level with PyTorch computes 0.9.0's bits.

**What T4 says.** The native backend's host, which step 5 found to be most
of its lead over the C program (copies of values and the fresh pages they
touch), is gone: under the plan its forward pass is 340.4 ms of kernels and
0.6 ms of host. What is left is the kernels' code, Cranelift's against
GCC's on the same kernels, 1.61× on the forward pass (339.9 ms against
211.7), wider than at step 6 (1.24×) since GCC's kernels lost their
spills (§3's addendum). Plan 0.10.0's decision 2 kept the kernels to the
epilogue and the C flags, so the version could not close it; it is
reported as it fell (`docs/kernels-roadmap.md`, the last section).

**What T3 says.** §8's addendum: the frozen model picks within 1.19× of
the best on nests it was never shown, and fails Q only by what it charges
for deciding, the whole gradient's capture, on nests whose sweep is short.

## 11. One nest's bound proved: Gappa beside the compiler's bound and the measurement (D108)

The compiler's bound is a first-order analysis conditional on a measured
envelope, sound wherever it was checked and never proved (§4, §5). On
2026-10-09 one nest's bound was proved in Gappa and set beside it
(`make gappa`, `experiments/gappa.sh`; the report `docs/gappa-2026-10-09.md`;
the statements and Gappa's answers under `docs/gappa/`): `LayerNorm{D}.apply`
of the examples in `f32`, at `D = 8, 16, 32, 64`, on eight rows synthesized
from seed 0. For each width: `presk calibrate --precision` measures the
envelope at 128 bits; `presk check --precision … --model all --measure`
prints the four rules' bounds and the native kernels' gap to the reference
on the envelope's inputs; `presk dump --stage kernel` gives the operations
and the order of the sums; `experiments/gappa_layernorm.py` writes the Gappa
statement — the same tree over the reals and with one rounding per
operation in the kernels' order, the inputs rounded once, `eps` the `f32`
of the binary64 the literal denotes — whose hypotheses are the envelope's
magnitudes at every site (and, as a second run, the inputs' ranges alone),
and whose goal is `|ŷ₀ − y₀|`; Gappa answers, and the same statement with
the answer as its goal is a proof it checks. Nothing is timed; the machine
does not matter.

| D | hypotheses | measured | `k` | `cond` | proved (Gappa) | `cond` / measured | proved / measured | proved / `cond` |
|---|---|---|---|---|---|---|---|---|
| 8 | the envelope's sites | 1.4·10⁻⁷ | 5.0·10⁻⁶ | 2.8·10⁻⁶ | **9.2·10⁻⁵** | 19× | 637× | 33× |
| 64 | the envelope's sites | 4.1·10⁻⁷ | 1.0·10⁻⁴ | 1.4·10⁻⁵ | **6.9·10⁻⁵** | 35× | 168× | 4.8× |
| 8 | one row's sites | 6.8·10⁻⁸ | 5.8·10⁻⁶ | 2.4·10⁻⁶ | **2.3·10⁻⁶** | 35× | 34× | 0.98× |
| 64 | one row's sites | 2.4·10⁻⁷ | 8.8·10⁻⁵ | 1.3·10⁻⁵ | **3.3·10⁻⁵** | 53× | 134× | 2.5× |
| 8 | the inputs alone | 1.4·10⁻⁷ | | | **3.2·10³** | | 2.3·10¹⁰× | |
| 64 | the inputs alone | 4.1·10⁻⁷ | | | **4.5·10³** | | | |

**What it says.** Proved ≥ measured and `cond` ≥ measured everywhere (`k`
and `normwise` too). On one row's envelope the proof and the compiler's
bound agree (0.98–2.5×); on a sample's the proved bound is above the
compiler's: the two analyses agree within a factor of two on every sum
and product and part at the square root and the quotient, whose rules in
the compiler are in relative terms at the site's maximum — a value's error
taken as proportional to its magnitude — where a proof must hold for a row
with the envelope's smallest variance and the largest errors at once,
which a sample's box admits. So the compiler's bound on a sample is not a
theorem of the sample's envelope; the proportionality it assumes is what
the measurement supports and a proof does not have. From the inputs' ranges alone the proved bound is vacuous
(nothing keeps the variance from 0 and the denominator from `√eps`): the
envelope's floor on the variance buys 3.5·10⁷. The proved bound does not
grow with `D` (it is the quotient's term, `|x − μ|ₘₐₓ · err(√) / √ₘᵢₙ²`,
a property of the sample's spread), where `k` grows with the sum's length,
so proved/`cond` falls from 33× at `D = 8` to 4.8× at 64. In `f64` the
picture is `f32`'s (33× at `D = 8`); in `bf16` the proved bound is vacuous
(866) while `cond` is 6.2× the measurement, the sum of squares' error
under `u = 2⁻⁸` exceeding the variance's floor. Gappa's time grows about
sixfold per doubling of the width (4 s, 13 s, 86 s, 10 min), so 384 was
not run.

**What it does not say.** Nothing about the loss or a block (§4's nine
orders are eight sums in series, not this nest); nothing proved about the
envelope, which both bounds are conditional on; no Gappa export (the
generator knows one nest). Four findings about the compiler are recorded
in D108. Two were fixed in 0.11.0 (D110): `presk check --precision` did
not bind a module's dimension arguments, so on the entry
`LayerNorm{8}.apply` itself the mean's length fell back to 1 024 terms and
the bound printed was 93× its rule's (the soundness test binds them and
gives §4's table; the paper's entries are top-level functions and
unaffected; `presk solve` bound them through its run and was not
affected); and `presk calibrate --precision --samples N` without `--data`
drew nothing, and is now refused. Two are left: the entry's arguments are not sites of the envelope, so an
input's rounding is charged at `u · 1` whatever its range; and the bound
for `ŷ₀` is every element's only to within 0.3 %, the sequential sum not
being symmetric in them.

## 12. Where the measurement decides: three programs the solver widens (D128)

Every solve §4–§10 report was accepted as proposed. `make widening`
(`experiments/widening.sh`, the report `docs/widening-2026-10-16.md`)
solves six small entries of `experiments/widening.psk` at seven budgets,
three pairs in which one form makes the first-order estimate wrong and the
other does not:

| program | budget | the estimate admits | measured | widening | beside it |
|---|---|---|---|---|---|
| an unshifted softmax of logits near 84 | 10⁻³ | `f32` (6.3·10⁻⁶ against 3.7·10⁻⁴) | 0.372: the row's sum of `exp` past `f32`'s range | 4 rounds, all-`f64` | the `softmax` primitive: none, 2.00× |
| the two-pass standard deviation, rows at 1 000 | 10⁻³ | `bf16` for the mean, whose cotangent is zero | 4.3·10⁻²: (δμ)²/2σ | 5 rounds, the mean last, 1.00× | the one-pass formula: none |
| a sum of 4 096 equal terms | 10⁻⁵ | `f32` for the sum | 1.6·10⁻²: correlated roundings | 1 round, 2.00× | noisy terms: none |

Under one type everywhere, the unshifted softmax in `f32` puts every rule
of `presk check`, the bounds included, below the gap, by 4·10⁴ to 10⁵ (no
rule compares a value's magnitude with its type's range); on the equal
terms `rss` is 10× below it and `cond` 6.4× above. The widening goes by
the estimate's terms, so the misjudged value is widened last and nothing
is narrowed again: widening it alone would hold the budget at 1.33× and
2.00× where the solver ends at 1.00×. The run first found and fixed a
defect: a tensor first argument was offered to the solver as a parameter,
which no executor runs, and every measurement narrowing it read ∞.

*At 0.16.0* (D163): the measured term lowers the estimates. The measurement still refuses the proposals D128 found refused, and two more at the largest budgets. `attn_raw` widens two values instead of four (§20, `docs/rows-0.16.0.md` §8).

## 13. PyTorch's own gap to the exact model (D130)

`make pytorch-gap` (`experiments/pytorch_gap.sh`, the report
`docs/pytorch-gap-2026-10-16.md`) runs `make margins`' model, the course
GPT at D = 64 on its trained weights, exported to PyTorch and run by
PyTorch 2.14 eager in eval mode, on the 461 held-out windows, and sets its
logits against the exact ones `make margins` took at 128 bits. PyTorch's
`float32` run is 2.95·10⁻⁶ from them at worst over 118 016 positions,
flips no decision and lies 158 times inside ε\*(10⁻³) = 4.67·10⁻⁴; on one
thread or ten, bit for bit the same. The compiler's all-`f32` is 3.12·10⁻⁶.
PyTorch's `float64` run is 5.9·10⁻¹⁵ from the exact logits, as the native
`f64` is: PyTorch could approximate its `float32` gap by its own `float64`
run, but only the exact reference says how close that `float64` is.

*At 0.16.0* (D163): PyTorch's rows and the compiler's uniform ones are the same digits. V6's row moves to 1.4·10⁻⁴ (§20).

## 14. The course GPT trained at course scale, through the margins (D129)

The course GPT at D = 384, H = 6, L = 6 (10.8 M parameters), trained 600
steps natively in `f64` on 90 % of the Code civil (4 h), then taken
through §4's curve and the margins (`experiments/curve_course.sh`,
`experiments/margins_course.sh`; the report
`docs/margins-course-2026-10-16.md`). On the 461 held-out windows the
exact model's validation loss is 2.3307 and its accuracy 0.3037. The
exact logits at 128 bits took 2 h 28 min on twelve cores. ε\*(10⁻³) is
4.96·10⁻⁴. The curve's realization at 10⁻³ (1.98× less traffic, eight
parameters in `bf16`) is 2.8·10⁻⁴ from the exact logits and the one solved
on the logits at ε\* (1.97×) 2.1·10⁻⁴; neither flips any of the 118 016
decisions, and every certified realization's validation loss is the exact
model's to 2·10⁻⁵. PyTorch's `float32` run of the same model is 6.4·10⁻⁶
from the exact logits, the native all-`f32` 1.8·10⁻⁵.

*At 0.16.0* (D163): the logits' solve gives 1.99× and 4.2·10⁻⁴ on the 461 windows, still inside ε\*. The curve's 10⁻³ realization is 13 ε\* with 29 flips, as before (§20).

## 15. A contraction's bound proved (D131)

`make gappa-dot` (`experiments/gappa_dot.sh`, the report
`docs/gappa-dot-2026-10-16.md`, the files under `docs/gappa-dot/`) does
for `dot` what §11 did for the layer norm: one output element of
`[8, K]·[K, 8]` in `f32`, the products rounded and summed in the kernel's
order, sequential or `interleaved(n)`, proved in Gappa from the
envelope's magnitudes of the operands, at K = 8 to 128. Under the
sequential order the proof is 2.6 to 3.1 times `cond` at every width,
flat over a sixteenfold length; under `interleaved(n)` 3.3 to 6.0 times.
Both bounds grow as K², the measured gap much more slowly: at K = 128 the
proof is 514 times the gap and `cond` 164 times.

## 16. A precision-tuning tool on the exported C (D154)

`make precision-tuning` (`experiments/tuning/tuning.sh`, the report
`docs/precision-tuning-2026-10-19.md`) sets the solver beside the C
precision-tuning tools. Their own input is the program `presk export
--target c` writes: `make margins`' model at D = 64, its logits in `f64`. The
criterion is the paper's: the logits within ε\*(10⁻³) = 4.67·10⁻⁴ of the
exact outputs the export ships, on its conformance inputs.

- *CRAFT on the C.* CRAFT's variable mode runs on it. Its builder is
  `experiments/tuning/retype.py`, a stand-in for TypeForge that retypes the
  C's 39 buffers and 24 kernel locals. Its default search passes the whole
  program in `float` on its first trial, though that program is slower than
  the `f64` one: the weights stay `double`, as inputs. Its delta-debugging
  search, Precimonious's, keeps the products' register tiles in `double` and
  runs 1.29× faster. As realizations, its answers are 1.81–1.95× less traffic,
  against the solver's 1.97×. They hold on the 461 held-out windows.
- *The searches on presk's space.* Precimonious's and CRAFT's searches,
  reimplemented on presk's realizations (`bf16` and the parameters
  included), find **2.17×**, in 70 and 2 858 trials. The realization holds on
  the 461 windows: 3.7·10⁻⁴, no flip, certified at 10⁻³.
  - *Where the gain comes from.* 84 % of the bytes it saves over the solver
    are the eval-mode dropout masks, tensors of ones, in `bf16`. They are
    exact there, but the solver's first-order estimate charges them as if
    they rounded.
- *At a budget of 10⁻⁶* the searches overfit the two windows they test on:
  their realizations fail on the 461. The solver's holds.
- *ADAPT* does not run on the export. ADAPT is C++, the export C11, and the
  export's arena holds bytes where ADAPT needs C++ objects.

*At 0.16.0* (D165): the searches find the same realization and price it at 2.00×, the masks no longer counted. The solver is 2 % short of it, where it was 10 % short (§20).

## 17. The rounding term measured, and narrowing by measurement (D155)

`make solver-rounding` (`experiments/solver_rounding.sh`, the report
`docs/solver-rounding-2026-10-19.md`) takes §16's finding to the compiler.
`presk solve` gains two options, both off by default: `--rounding measured`,
which charges each site the largest rounding error of its sampled values
instead of `u · max|x|`, and `--narrow measured`, which narrows each decision
one step after a passing measurement while the measured gap fits. On the
logits at D = 64, V6's protocol:

| budget | `--rounding` | `--narrow` | traffic vs all-`f64` | measurements | on the 461 windows |
|---|---|---|---|---|---|
| ε\*(10⁻³) | unit (of record) | off | 1.97× | 1 | 7.5·10⁻⁵, no flip |
| ε\*(10⁻³) | measured | off | **2.14×** | 1 | 1.6·10⁻⁴, no flip |
| ε\*(10⁻³) | measured | measured | **2.17×**, §16's searches' realization | 79 | 3.7·10⁻⁴, no flip |
| 10⁻⁶ | unit (of record) | off | 1.27× | 1 | 8.7·10⁻⁷, holds |
| 10⁻⁶ | measured | off | 1.40× | 1 | 8.9·10⁻⁷, holds |
| 10⁻⁶ | measured | measured | 2.07× | 93 | **1.6·10⁻⁶, fails** |

- *The measured term* is safe at both budgets. Its gain is the four dropout
  masks, ones, in `bf16`.
- *The narrowing* reaches the searches' answer and inherits their overfit
  at 10⁻⁶.
- *The masks* are 15 % of the realization of record's bytes. The compiler
  stores them because the program is the same in and out of training: spec
  question 78.

*At 0.16.0* (D165): the measured term gives 1.96× at ε\*, as the unit term does. Its gain here was the masks, which D160 removed. The narrowing still reaches the searches' realization, at 2.00× in 74 measurements, and still fails out of sample at 10⁻⁶ (§20).

## 18. The certificate off-sample against the sample's size (D156)

`make off-sample` (`experiments/off_sample.sh`, the report
`docs/certificate-off-sample-2026-10-19.md`) asks how far a gap the solver
measures on the envelope's windows is from the gap on the others. The
envelope of the logits at D = 64 is calibrated on the first N held-out
windows, the solver run on it at ε\*(10⁻³) and 10⁻⁶, and every realization
judged on the 461 windows, its gap split into the N it was calibrated on
(in) and the others (out). For the solver of record:

| N | ε\*(10⁻³): out/in | 10⁻⁶: out/in | its realization |
|---|---|---|---|
| 1 | 1.72× | 3.76× | another at ε\*, V6's at 10⁻⁶ |
| 2 | 1.72× | 2.57× | another at both |
| 4 | 2.30× | 2.48× | V6's |
| 8 | 2.30× | 1.74× | V6's |
| 16 | 1.92× | 1.53× | V6's |
| 32 | 1.92× | 1.53× | V6's at ε\*, √32 apart at 10⁻⁶ |

- *The ratio falls slowly and in steps.* The out-of-sample gap is one
  window's: V6's largest over the 461 is in window 402, no other within
  10 % of it.
- *The realization is the model's.* From four windows on, the sample sets
  the gap the solver measures, not the realization it picks.
- *The narrowing* (`--narrow measured`) fails out of sample at 10⁻⁶ from
  every N up to 32, 1.4–1.7·10⁻⁶, narrowing less as N grows.
- *The certificate.* No realization inside ε\* flips a decision on the
  windows it was not calibrated on.
- *The two executors.* The solver measures on the kernel interpreter, the
  rows are native, and under `bf16` sites the two gaps differ by up to 15 %
  on the same windows (D76's transcendental functions).

*At 0.16.0* (D165): the default, `measured / off`, is 1.94× out of sample against in at four windows and 1.15× at 32. Its realization is the same from one window to 32, and the solver's measurement is the native gap to the digit (§20).

## 19. Three ways to spend the budget, judged out of sample (D161)

`make spend` (`experiments/spend.sh`, 57 minutes, nothing timed) solves the
logits at D = 64 on the envelopes of the first 4 and 16 held-out windows, at
ε\*(10⁻³) = 4.67·10⁻⁴ and 10⁻⁶, in seven forms. It judges every realization
on windows 2N to 460, which no form saw. The traffic is one sample's, and the
program the inference one, with no mask.

| form | 4.67·10⁻⁴, N = 4 / 16 | 10⁻⁶, N = 4 / 16 |
|---|---|---|
| the default (`measured / off`) | 1.966×, out 0.31 of the budget / the same | 1.266×, 0.89 / the same |
| `--estimate rss` | 1.965×, 0.66, 1 flip / the same | 1.206×, 0.69 / another, 0.75 |
| `--narrow measured` (k = 2.5) | 2.002×, 0.79 / 2.001×, 0.40 | 1.554×, 0.82 / 1.266× (none kept) |
| `--narrow heldout` | as `--narrow measured` | 1.266× (all 22 undone) / 1.266× |

What this shows:
- *Every form holds* in every cell.
- *The margin helps the narrowing.* With k = 2.5 the narrowing holds at
  10⁻⁶, where at k = 1 it failed from every sample size (§18). But it
  then depends on N: at 16 windows it keeps nothing.
- *The rule.* The plan's rule asks for 1.5× of the budget left out of
  sample in every cell and the same realization at both sizes. Nothing
  meets the first condition at 10⁻⁶, the default included (0.89 of the
  budget), and no new form meets the second. So the three options stay
  flags, and the defaults are final.

## 20. The rows of record at 0.16.0 (D162–D166)

Plan 0.16.0's step 5 solved and timed every row of record again, each by
its own script and whole protocol. They ran on presk `22dd345`, the tree
of steps 1 to 4, and the timings on falicon alone. Since 0.16.0 four
things move a row:
- the measured rounding term is the default (D158);
- the traffic is one sample's (D158);
- the kernel interpreter computes the native bits (D159);
- an entry outside a training run has no dropout mask (D160).

The report, with each row's old value, its new one and the paper's
sentence, is `docs/rows-0.16.0.md`. The outputs are archived in
presk-project `data/`.

| row | before | 0.16.0 |
|---|---|---|
| the decoder's solve (D162) | D124: 31 of 32, 13 weights in `f32`, 1.99×, 2.6·10⁻⁵ | the same 31, 13 weights with RMSNorm's three gains (ones at initialization) in `bf16`, 1.99×, 2.5·10⁻⁵ (the native gap, which D124's realization had too) |
| H2, the curve at 10⁻³ (D163) | D132: 33 stored values, 2.12×, 1.6·10⁻⁵ | 27 (the masks gone), **2.04×**, 4.9·10⁻⁶ |
| H5, the margins | four of seven certified; H2's realization 17 ε\*, 30 flips | four of seven; H2's 5.5 ε\*, 9 flips |
| V6 | D124: 29 of 31, 1.97×, 7.5·10⁻⁵ | 22 of 25, **1.97×**, 1.4·10⁻⁴, no decision changed |
| the course GPT through the margins (D129) | the logits' solve 1.97×, 2.1·10⁻⁴ | 1.99×, 4.2·10⁻⁴, still inside ε\* |
| `make precision` | D124: 2.12×, 3.9·10⁻⁶ | **2.02×**, 1.2·10⁻⁶ |
| the quant and step tables, J′ | D124 | byte for byte, so J′ stands |
| D128, D130 | | as before, with lower estimates (§12) and the same PyTorch rows (§13) |
| D154: the tools' searches (D165) | 2.17× | **2.00×**, the same realization, the masks no longer counted |
| D155: the measured term at ε\* | 2.14× against 1.97× | **1.96× either way**: its gain was the masks, ones exact in `bf16` |
| D156: out of sample against in, N = 4 and 32 | 2.30×, 1.92× | 1.94×, 1.15×, the default's realization the same from one window, the solver's measurement the native gap to the digit |
| Table 1 (D166) | D101, D124 | the solves within 7 %; the calibrations 14–19 % slower, measured: D155's rounding table (+5 %) and D158's pass for one sample's counts (+9 %) |
| V5: the decoder block's realization ÷ all-`f32` | D124: as fast | **0.73×**, below all-`f32` in each of eight processes |
| T2 and T4 (Table 4) | D127: 0.98×; 1.52×, 1.60× | **0.93×; 1.63×, 1.67×**, the C program 5 % faster without masks |
| D153: the C step ÷ `torch.compile`'s | 1.80× (2.07× like for like) | 1.70× (1.97×) |
| H6 (D50): course scale, D = 64 | 1.13×, 1.06× (apremont) | 1.02×, 0.74× (falicon) |

What changed the numbers:
- *The masks.* They are gone from every entry outside a training run, so
  they leave the solver's menu, the traffic and the C program.
- *The measured term.* It charges nothing to a value exact in a type, and
  with the masks gone that means the norms' gains at initialization and
  the head width 32.
- *The cost per sample.* It makes the weights weigh more against the
  activations.

What stayed: every row is inside its budget by measurement, H2, H5 and
statement S hold at both scales, and T2 holds and T4 fails as before.
The step-5 runs found one defect. Since D160, `presk plan` had planned
the inference reading, so `presk train` refused the plan of a training
step (`E7002`). `presk plan --mode training` fixes it (D164).

## 21. The certificate on a second text (D168)

`make second-corpus` (`experiments/second_corpus.sh`, the report
`docs/second-corpus-2026-10-22.md`) keeps the models, the realizations and
ε\*(10⁻³) of record and changes the text to the Code pénal.
- *The text.* It comes from the course's container and is put on the Code
  civil's vocabulary, with 32 characters deleted.
- *The judge set.* Its first 461 windows of 256.
- *The runs.* Every realization of §20's margins at both scales, run
  natively against the Code pénal's exact logits.

Two statements were fixed before it ran:
- *Y1:* every realization certified on the Code civil stays inside its ε\*
  and flips nothing;
- *Y2:* every gap is within 2× of its gap on the Code civil.

| | D = 64 | the course constants |
|---|---|---|
| certified realizations above the Code civil's ε\* | none of 12 | **4 of 12**; the logits at ε\* 9.4·10⁻⁴ (1.90 ε\*) |
| their flips | none | 1 or 2 each, at two near-ties |
| the gaps, Code pénal ÷ Code civil | 0.54× to 1.31× | 0.78× to **2.82×**, 7 of 22 above 2× |
| Y1, Y2 | hold | fail |
| the exact model's loss, Code civil → Code pénal | 2.774 → 2.747 | 2.331 → 2.313 |

- *One stretch moves.* At course scale 18 of the 22 realizations have their
  worst gap in one window, from the prediction after "par le J|ournal
  officiel" to its end. The median window moves less than 17 %.
- *LayerNorm's ε in `bf16` carries it.* Back to `f32`, the course curve's
  10⁻⁴ realization falls 25 times there.
- *The flips are near-ties.* The Code pénal's two smallest margins,
  1.1·10⁻⁵ and 2.6·10⁻⁶, are below any of the Code civil's.

The certificate is a statement about the text it was measured on. At
D = 64 it carries to a second text of the same kind. At the course
constants it does not: one stretch, in one window of 461 and never seen by
the envelope, breaks it.

## 22. The rows of record at 0.16.1 (D170)

The two rules of 0.16.1 (D169; the `precision` chapter, "Two rules") keep at
`f64` the values without gain. The rows they touch were solved again on
0.16.0's envelopes (`docs/rows-0.16.1.md`), and judged on the Code civil
and on the Code pénal.
- **The values left out.** On every row 18 to 22 values are left out. They
  forgo at most 0.96 % of the traffic (Z1), and every ratio of record
  moves by at most 0.03×: V6 1.96×, the course logits 1.98×, the curve at
  10⁻³ 2.03×, `make precision` 2.01×.
- **No `bf16` left in the certificate rows.** The certificate rows are now
  `f32` but for the values left out.
  - V6 measures all-`f32`'s gap on the 461 windows (2.88·10⁻⁶; 0.16.0:
    1.43·10⁻⁴).
  - The course logits measure half all-`f32`'s (8.70·10⁻⁶; 0.16.0:
    4.23·10⁻⁴).
  - On the Code pénal the course logits measure 2.24·10⁻⁵ (0.16.0: 1.90
    ε\*).
- **On the Code pénal.** D168's Y1 now holds at both scales and Y2 at
  D = 64. At the course constants Y2 fails on 5 of 22 rows, all at
  2.2·10⁻⁵ in window 261. So does Z2's out/in criterion for the course
  logits: 1.92× from 16 windows on the Code civil, where all-`f32`'s own is
  1.80×. Both are relative criteria at `f32`'s level. In absolute terms
  every certified realization is at least 22 times inside ε\* on both
  texts, and flips nothing.

## 23. The certificate on a third and a fourth text (D171)

The plan's §7 fixed two more texts before the measurement: the Code du
travail and the Code de procédure pénale (`make further-texts`,
`docs/further-texts-2026-10-23.md`). Their typographic characters are
mapped onto the Code civil's vocabulary, and their first 461 windows go
through the Code civil's models. Both 0.16.1's realizations and 0.16.0's
are judged at the Code civil's ε\*.
- **Y1 holds on both texts at both scales with 0.16.1's realizations.**
  Every certified realization stays inside ε\* and flips nothing. 0.16.0's
  failed it: by one or two near-tie flips inside ε\* at D = 64 on both
  texts, and at 1.16 ε\* at the course constants on the Code de procédure
  pénale.
- **Y2 holds at D = 64 and fails at the course constants on both texts,**
  at `f32`'s level: 0.49× and 0.495× on the Code du travail, 2.07× on the
  Code de procédure pénale. So Z3 fails on both.
- **The tails** come after an article's heading and after an opening
  parenthesis, the Code civil's own kinds of position. They never come
  after a capital inside running text, the Code pénal's window 261.

## 24. Each kernel's share of the native backend's gap (D174)

`make kernel-shares` (`docs/kernel-shares-2026-10-28.md`) times every
launch of the course GPT's forward pass, gradient and optimizer on both
sides and joins them by the kernel's signature. The run is at B = 16,
T = 256 in `f32`, under the frozen profile's plans:
- natively, by `presk bench --plan-only --calibrate`;
- in the exported C program, by `kernel_time.py`.

There are two settings: ten X925 against GCC `-O2` with OpenMP, and one
X925 against GCC `-O2`, clang `-O2` and clang `-O3`. Each ran in three
processes. Plan 0.17.0's level 1, statement NB1.
- **On ten X925 the kernels are 1.64× GCC's on the forward pass and
  1.90× on the step.** The step is wider than D166's calls (1.67×), since
  the C program's host takes about 110 ms of its step.
- **The forward pass's gap** is the register-tiled products with their
  bias (71 %, 1.65–1.74× each) and the masked softmax (19 %, 2.2×).
- **The gradient's gap** is spread over five kinds:
  - `sum_to` (23 %, 4.5×), whose innermost loop sums into an output
    element in memory;
  - the attention's `ein` (22 %);
  - the products' derivatives and the products (43 %).

  Simple kernels with one lane in the plan are 4–7.5× slower, where GCC
  vectorizes them at `-O2` anyway.
- **On one core clang's build of the same C is 1.32× faster than GCC's on
  the forward pass and 1.12× on the gradient.** That predicts level 3's
  LLVM route at or under GCC on these kernels.

A measurement; one change it needed: `--calibrate --plan-only` times the
plan's kernels alone (the manual's `cli` chapter).

## 25. A public model's weights: Llama 3.2 1B (D180)

`make llama3` (`experiments/llama/llama3.sh`) runs `coverage/llama3.psk`
on the published weights of `meta-llama/Llama-3.2-1B`, the base model.
The user downloads them, the licence accepted and the token theirs. This is
plan 0.17.0 step 7, statements RW2–RW4.
- **The conversion**, once (`hf_to_pskd.py`): 11 leaves, 1 235 814 400
  parameters, kept in their `bf16` in a `.psk.d` of version 2 (2.47 GB).
- **The prompts**, four of 16 tokens, the first the beginning-of-text
  token, made by the checkpoint's tokenizer (`tokens.py`): "The capital of
  France is Paris, and the capital of Italy is Rome."; "In mathematics, a
  prime number is a natural number greater than one that has"; "def
  fibonacci(n): if n < 2: return n; return fibonacci"; "Le compilateur
  traduit un programme écrit dans un langage de haut niveau".

The statements:
- **RW2 holds: the conversion is right.** `presk run --backend native
  --realize f64` is compared with Hugging Face's `LlamaForCausalLM` in
  `float64` on the same ids (`rw2.py`, in `.venv-hf`).
  - The largest gap of the logits is 1.2–1.8·10⁻¹³, 4.7–6.2·10⁻¹⁵ of their
    largest magnitude (25–30), against a threshold of 10⁻⁸.
  - The argmax is the same at all 64 positions.
  - That is two summation orders in `f64`, and no leaf misplaced.
  - `transformers` computes in `float32` in three places whatever the
    model's type: the rotary angles, `LlamaRMSNorm` and the eager softmax.
    So the harness recomputes the angles in `float64` by Hugging Face's own
    function and removes the norm's cast. Its `float32` frequencies are
    2.1·10⁻⁷ from the `float64` ones.
- **RW4 holds: served the same.** `presk serve --backend native` and the
  exported `generate.c` (at T = 48, the context padded) give the same 32
  greedy tokens from each prompt, in `f32` and with the blocks' seven
  matrices in `int8` (absmax, the scale along the contraction axis).
  - The `f32` model continues the first prompt with " The capital of the
    United States is Washington, D.C. The capital of the United Kingdom is
    London. The capital of Canada is Ottawa. The capital of Australia".
  - The `int8` continuations are the same but for that prompt's last token,
    "Germany".
  - Both programs run on one thread at batch 1 (the `serve` chapter).
- **RW3, reported: the model's gap to the exact reference.** Each
  prompt's logits are compared with the exact reference at 128 bits
  (`presk run --backend interp --bits 128`, 21.5 min and 96.6 GB a prompt,
  each in a memory scope of its own). The `int8` and `int4` realizations quantize the blocks'
  seven matrices with one scale per output column:

  | realization | max \|logit − exact\| / max \|exact\|, prompts 0–3 | argmax changed, of 16 a prompt |
  |---|---|---|
  | `f64` (RW2's run) | 4.8e-15, 6.1e-15, 5.1e-15, 4.8e-15 | 0, 0, 0, 0 |
  | `f32` | 2.3e-06, 4.0e-06, 2.5e-06, 3.0e-06 | 0, 0, 0, 0 |
  | `bf16` | 7.9e-03, 1.3e-02, 1.2e-02, 1.0e-02 | 1, 0, 0, 0 |
  | `int8`, absmax | 1.9e-02, 3.5e-02, 2.5e-02, 3.9e-02 | 0, 0, 0, 0 |
  | `int8`, pow2 | 2.7e-02, 5.0e-02, 5.5e-02, 4.0e-02 | 0, 0, 0, 1 |
  | `int4`, absmax | 4.9e-01, 4.8e-01, 4.0e-01, 4.9e-01 | 6, 7, 7, 9 |
  | `int4`, pow2 | 5.0e-01, 5.9e-01, 5.8e-01, 5.8e-01 | 10, 10, 11, 12 |

  - `f32` changes no decision.
  - `bf16` and `int8` change one argmax of 64.
  - `int4` changes 6 to 12 of each prompt's 16. One scale over the whole
    contraction axis is too coarse for four bits on this model.

One defect, found by the first full-size run. Every command that reads
`--weights` first drew the model's initial parameters in the reference
interpreter, in MPFR, about 80 bytes a number at 256 bits, and only then
read the checkpoint against that tree.
- At 1.2·10⁹ parameters the first run passed 88 GB and was OOM-killed.
- Since D180 the parameters are described, and the checkpoint is read
  against zeros of their shape (the `cli` chapter, `--weights`). RW2's run
  takes 21.7 GB and 74 s.
- `presk run --result-only` keeps a checkpoint out of `--json`'s output.

## 26. The rows of record at 0.17.0 (D181)

Plan 0.17.0's step 8 measured every row the version changes, on the
version's binary (`docs/rows-0.17.0.md`). The timed rows ran alone on
falicon by D127's protocol: five processes, ten X925, the median. No row of
the paper was solved or timed again: the paper stays at 0.16.1.
- **The native code (NB2, NB3 hold).** The course GPT at B = 16,
  T = 256 in `f32`, under the frozen profile's plans, against the exported
  C program under GCC `-O2`, its workers waiting as the native pool's:

  | workload | C program | Cranelift, level 2 | ÷ C | LLVM | ÷ C |
  |---|---|---|---|---|---|
  | the forward pass | 206.5 ms | 246.4 ms | 1.19× | 205.5 ms | 0.995× |
  | the step, `loss_grad` + `adamw.step` | 808.1 ms | 803.4 ms | 0.994× | 703.7 ms | 0.871× |

  - At 0.16.0, T4 failed at 1.63× and 1.67× (§20). NB2's threshold is
    T4's 1.3×, NB3's 1.0×.
  - Kernel for kernel the LLVM backend's step is 0.98× GCC's. The rest of
    its 0.871× is the C program's host, about 110 ms of its step (§24).
  - What remains on Cranelift is the forward pass's register tile (the
    `native` chapter, "The code generator").
  - The LLVM backend compiles a kernel in 37–48 ms through `clang`, once:
    cached, a program starts as fast as on Cranelift (NB4).
- **One menu (QM1 and QM3 hold, QM2 fails).** `presk solve --menu
  f64,f32,bf16,int8,int4` against `--quant`'s two passes, on the envelopes
  of record:
  - every one-menu solve is inside its budget by measurement, with no
    widening round;
  - on the curve at D = 64 it takes integers at 10⁻¹ and 10⁻² (3.55×
    against 0.16.1's 3.33×, 2.55× against 2.51×), and none from 10⁻³ down;
  - on V6 and the course logits at ε\* it takes none, and its realizations
    are 0.16.1's byte for byte: certified on the Code civil, Y1 on the Code
    pénal, the same time;
  - **on BERT-large's two blocks at 10⁻³ it takes none (2.00×), where the
    two passes quantize the ten matrices (4.93×, measured 3.82·10⁻³ inside
    6.25·10⁻³).** The one menu prices an integer by its largest single
    move, a worst case of about the whole budget. The second pass prices
    the moves' root-sum-square, an estimate its measurement checks (the
    `precision` chapter, "One menu").
- **The step per stored value (TS2 fails).** At course scale the
  per-value realization's step takes 7.80 s: 1.92× faster than
  all-`f64`'s, but slower than J′'s per-function 6.99 s.
  - J′ with the optimizer's state kept in `f64`, as the per-value space
    keeps it, already takes 7.58 s.
  - The conversions between types take 3.1 % of the per-value step and
    1.0 % of J′'s (TS3).
  - At D = 64 they take 12.5 % of H6's step, 38 % of its loss against
    all-`f64` (§20).

## What these experiments do not show

- **Speed against a mature stack, beyond one model on one machine.** §3's
  comparison, redone on falicon on 23 September 2026, was 22× a training
  step, 14 of it the contraction; redone on three sides at 0.9.0 (D87), 12×
  for the C program under a plan, four fifths of it the derivatives of
  `dot`; at 0.10.0 the C program's training step is level with PyTorch's
  (0.99×, §10) and its forward pass 1.86× behind, on the course GPT at its
  course constants, in `f32`, on ten X925 cores of one machine, against
  PyTorch eager on the CPU (OpenBLAS and oneDNN). Against PyTorch
  compiled by `torch.compile`, the C step is 1.80× behind, and 2.07× like
  for like without dropout masks (D153, §3); at 0.16.0, the masks gone,
  1.70× and 1.97× (§20). A GPU, a larger model and
  another machine are not shown. The native backend's own kernels, 1.6×
  GCC's at 0.16.0, are at GCC's on the LLVM backend at 0.17.0, and 1.19×
  on Cranelift's forward pass (§26).
- **A static choice as good as measuring.** RA4, the programme's first
  thesis, was tested twice on the attention of one model family, on one
  machine's two kinds of core. 0.7.0's model failed it for a named cause
  (§7). 0.8.0's, frozen before its test, ranks more than 90 % of the
  attention `ein`s' pairs on the other cores and 85–89 % on new shapes, so
  RA4 still fails as stated. Its misses are pairs it predicts within its
  own uncertainty, and its pick and its candidates are within a few per
  cent of the best (§8). Kernels other than attention's and a GEMM,
  machines other than falicon, and the comparison with the MDH and ATF
  autotuners on the same space have not been tried.
- **A budget discharged by a bound.** Every budget here is discharged by
  measurement, on the envelope's inputs (D41, D51); the conditioned bound is
  still ten orders of magnitude above the gap on the trained corpus GPT in
  `f32`. And the reference measured against is the exact denotation only
  where the interpreter reaches: BERT-large's inference, not a course-scale
  training step, where J′ was measured against `f64` (question 62).
- **Quality beyond this scale.** The ε curve's quality axis is flat after
  600 steps of a `D = 64` GPT, within 2.0·10⁻⁴ against a seed spread of
  0.092 (D102, at the tag; 1.9·10⁻⁴ in D93; 1.2·10⁻³ and 0.13 in D50, on a
  gradient defect 3f made wrong). There is no trained BERT, no SQuAD and no F1 here, so MLPerf's own
  threshold on its own benchmark was not applied; §5 applies the rule to the
  corpus GPT instead.
- **Verified transformations** (the KV cache), **parallelism** beyond threads
  inside a kernel, and **energy**: declared in the language, not implemented.

What the tool delivers is the front half of the argument and a first piece
of the back half: a language where shapes, smoothness, randomness and data
are checked before anything runs; an exact reference that turns numerical
precision into a measurement; a declared budget that buys time — against
`f64`, not yet against `f32` inside a tight budget; and a space of kernel
layouts in which exactness costs nothing and the best points are several
times faster than the defaults. That a **static** cost model can choose among
them as well as measuring does, which is what would make the discipline buy
speed without autotuning, is the part still to show. The first test failed
for a reason it named; the second, on a model frozen before it, came within
a few points of the hypothesis's threshold and met it on one of its two
sweeps, and a chooser that prunes by the model and measures the rest found
a point within 1.03× of the best everywhere, at under 2 % of the cost.
