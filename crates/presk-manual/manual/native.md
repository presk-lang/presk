# Native execution: PSK-Kernel v0 and the backends

Version 0.3.0 runs PSK-Core outside the reference interpreter. This chapter
describes the seam every executor implements, the kernel IR the tensor
primitives are lowered to, the machine realizations, the Cranelift backend
that compiles the kernels, and the conformance suite that measures each
backend against the reference.

## The seam: `presk_rt::Backend`

An executor `prepare`s an entry (a PSK-Core function at bound dimensions
and a realization) and `call`s it on runtime values. A runtime value is a
tree (tuples, records) of tensors with named axes whose buffers are `f64`,
`f32`, `bf16`, `i64`, quantised weights (integers with their scales, since
0.6.0) or *exact* MPFR reals. The reference interpreter is the first
backend (`--backend interp`); its exact values cross the seam untouched, so
the training loop written over the seam reproduces 0.2.0's results digit for
digit. Input synthesis stays with the interpreter: any backend runs on the
inputs the reference would.

## The realization: `--realize f64|f32|bf16|@FILE`

Versions 0.1.0 to 0.4.0 had a flag here with two values. Since 0.5.0 step 2
the realization is an **object**, because two separate needs met: the
precision solver has to *produce* a per-site assignment, and a vectorized or
coprocessor kernel has to *declare* its reduction order before it may ship —
which decisions D31 and D32 demanded and had nowhere to write down. One
record answers both (D34):

```json
{
  "default": "f32",
  "storage": { "loss/%17": "bf16", "loss/%23": "f64" },
  "accum":   { "loss/%17": "f32" },
  "order":   { "kind": "interleaved", "partials": 8 },
  "envelope": "out/envelope.json"
}
```

- **`default`** is the storage type of every site the map does not mention,
  so `--realize f64` is exactly this object with an empty map and a
  sequential order. Those spellings mean today what they meant in 0.4.0, and
  the goldens prove it: every measured gap of the conformance table is
  unchanged digit for digit.
- **A site** is `"<function>/%<let>"`, naming a fusion group's root as
  `presk dump --stage core` prints it. The group is the unit because it is
  one loop nest with one set of buffers, so it is the smallest place a type
  is a decision rather than a conversion.
- **`accum`** is the type a contraction sums in. It defaults to the
  *arithmetic* type of the storage, so `bf16` storage accumulates in `f32`
  unless something says otherwise.
- **`order`** is `sequential`, `blocked(width)` or `interleaved(partials)`.
  The first two are bit-identical — register blocking keeps every addition in
  index order (D32) — and the third reassociates, which is a different
  realization owing a measured gap (D31). It is not a performance hint: the
  constant in front of `u` in a contraction's error rule is a function of it.
- **`envelope`** names the magnitude measurements any bound about this
  realization assumed, because a bound without one is a different and
  unsupported claim (D35).

A parameter a realization quantises (`precision` chapter) reaches the
kernels as integers: `presk dump --stage kernel` shows such an input as
`int8/o[…]` or `int4/o[…]`, `o` the axis its scales are constant along, and the kernel loads
each element as `f32(q)·s` — in four lanes at a time when it has them — which
is bit-for-bit the value the reference path substitutes (D45).

`bf16` is **storage only**: buffers hold sixteen bits, arithmetic is `f32`,
and conversion is a shift on the way in and one round to nearest even on the
way out (D38). It halves the traffic of the memory-bound sweeps — `adamw.step`
is a dozen passes over every parameter — while contractions keep their
accuracy. There is no `f16`.

`presk bench` now times each uniform realization, and the first thing it
shows is that a narrower type is not automatically faster. On
`Linear{64, 64}.apply` at batch 64, twelve cores, medians of five:

| realization | median ms |
|---|---|
| native `f64` | 0.273 |
| native `f32` | 0.264 |
| native `bf16` | 0.282 |

`bf16` is *slower* than `f32` here, because this kernel is bound by
arithmetic rather than by traffic and the conversion on every load and store
is pure overhead. That is the whole reason precision has to be *solved* per
site rather than chosen globally: the narrow type pays on the memory-bound
sweeps and costs on the compute-bound contractions, and which is which is a
question for a cost model, not for a flag. It is also why 0.5.0 planned the
vectorized kernel before the curve — without vector lanes there is no regime
in which arithmetic width alone buys time — and why 0.6.0 made the solver's
objective predicted time (`precision` chapter, "Time, not traffic").

Reductions and contractions accumulate in the accumulator type, indices are
`i64`, hyperparameters and `nat` values are converted at their injection.
`gelu` is the exact `x·Φ(x)` through `erf`; `softmax` and `cross_entropy`
subtract the row maximum (a value-preserving rewrite over ℝ). Where a site
hands a buffer to a kernel of another type, the conversion happens at the
kernel's edge, which is where the cost of a mixed assignment actually is.
The interpreter ignores the realization and computes at `--bits`.

## The vector form

Since 0.6.0 step 1 a kernel may hold **vector variables**: registers of
`lanes` reals, with a contiguous load, a broadcast and lane-wise arithmetic.
The one place they are used is the blocked contraction, whose shape was
already right — eight accumulators over eight contiguous output columns, an
inner loop reading one element of the left operand and eight of the right.

**A lane is a different output column, never a partial sum of one.** Every
output still accumulates the contracted axis in increasing order, so a vector
kernel is **bit-identical** to its scalar form, and a test asserts exactly
that at two and four lanes in `f64` and `f32` rather than a tolerance. That
is statement H of plan 0.6.0, and it is why the vector kernel owes no
declared reduction order and no measured gap: it is a layout change, not a
reassociation. Decision D31 feared the other thing — splitting the *inner*
sum — which is `Interleaved` and is still a different realization.

Both executors emit it. Cranelift's vector types are 128 bits wide, so the
lane count is the machine type's: four `f32` or two `f64`, and a request for
anything else is refused rather than quietly rounded. `bf16` stays scalar,
since a narrow lane would need a conversion per element. Lanes are **on by
default** on the native backend (`run`, `train`, `serve`), each kernel at its
own type's width: they change no bit, so there is nothing to declare and
nothing to measure but the time. `presk bench` times every uniform
realization with and without them (the `+lanes` rows). They do not always
pay. On BERT-large at MLPerf's constants (step 8, `coverage` chapter), lanes
make `int8` weights 1.9× faster at batch 1 and 1.5× at batch 20, and `f32`
1.1× faster at batch 1 — but 15 % *slower* at batch 20, where scalar `f32`
code wins. Lanes are chosen for the whole run, not by the cost model per
kernel.

**Statement H holds on both paths**, asserted as an equality: the emitted
vector kernel is bit-identical to the emitted scalar one, and so is the
interpreted one.

`presk bench --native-only` on a 1024³ matmul, twelve cores, medians of
three:

| | median ms |
|---|---|
| `f64` scalar | 104.1 |
| `f64`, two lanes | **83.3** |
| `f32` scalar | 87.2 |
| `f32`, four lanes | **75.8** |
| `bf16` scalar | 97.2 |

**Statement I fails.** It asked for at least three times the scalar figure;
the lanes give 1.25× in `f64` and 1.15× in `f32`. The gain is real and it is
free — the same bits, faster — but it is not the order of magnitude the
kernels roadmap expects from vectorization, and the reason is worth naming
rather than guessing at: the blocked form already ran eight independent
accumulator chains, so the instruction-level parallelism the lanes add was
largely there, and what remains is bound by loads rather than by issue. The
packed panels this step also planned, which would make those loads
unit-stride and aligned, are **not written**, and that is the first thing to
try before concluding anything about the ceiling.

These figures were taken on the development machine before D46 took the
conversion of `bench`'s inputs out of the timing, so their ratios are
diluted. Measured again on falicon (aarch64, twenty cores) at the end of
0.6.0, four runs of the same command: `f64` 78–85 ms scalar and 68–75 ms in
two lanes (1.1–1.2×), `f32` 46–53 ms scalar and 46–59 ms in four lanes (no
measurable gain), `bf16` 40 ms, the fastest row. That is 41–46 GFLOP/s for
the better `f32` row of each run and 54 for `bf16`, about 0.4.0's scalar
figure (47–56) where I asked for three times it. At this size, on this machine, the kernel waits for memory, and a
wider register does not help a load-bound loop. I is retired rather than
pursued (plan 0.6.0, revision of 2026-09-20): the factor it asks for is in
the matrix coprocessor, which `docs/kernels-roadmap.md` puts after this
version.

## PSK-Kernel v0: `--backend kernel`

Every tensor primitive of PSK-Core, at the concrete shapes of its operands
and the chosen machine type, is lowered to a **kernel**: a loop nest over
row-major buffers with scalar operations, reductions as loops with an
accumulator, index arithmetic explicit. Kernels are specialized on first
use and cached by their signature (primitive, operand shapes and names,
dtype). The **host program** is the PSK-Core body itself, evaluated over
runtime values: calls, folds, tuples and records, the structural
primitives of parameter trees (`slice`, `stack`, `set_field`, …), the
initializers and `sample` (on the host, with the interpreter's ChaCha keys,
so native and reference runs draw the same numbers), and one kernel launch
per tensor primitive — since 0.4.0, per fusion group, a chain of elementwise
primitives sharing one loop nest (`scheduling` chapter). `presk dump --stage
kernel` runs an entry and prints the kernels it launched:

```sh
presk dump tutorial/01_linear.psk --stage kernel --entry 'Linear{3, 2}.apply' --dim batch=4 --realize f32
```

```text
-- 2 kernels, 2 launches, f32
-- fusion: 2 launches without it, 0 saved (0%)
kernel dot (%0 : f32[batch: 4, 3], %1 : f32[3, 2]) -> %2 : f32[batch: 4, 2] {
  for i0 in 0..4 {
    for i2 in 0..2 {
      %2[i0, i2] = 0
    }
    for i1 in 0..3 {
      s0 = %0[i0, i1]
      for i2 in 0..2 {
        %2[i0, i2] = (%2[i0, i2] + (s0 * %1[i1, i2]))
      }
    }
  }
}
kernel add (%0 : f32[batch: 4, 2], %1 : f32[2]) -> %2 : f32[batch: 4, 2] {
  for i0 in 0..4 {
    for i1 in 0..2 {
      %2[i0, i1] = (%0[i0, i1] + %1[i1])
    }
  }
}
```

The contraction is in the row-wise form of "Contractions" below, since two
output columns are not a multiple of eight; each output still sums `k` in
increasing order. The lowering follows the interpreter's semantics rule for
rule: its
alignment of axes for broadcasting (trailing positions when names agree,
else by name), the contraction axes of `dot` and its adjoints, the general
`ein`, the row-wise `softmax` and `cross_entropy`, gathers and scatter-adds
for `embed`, `argmax` keeping the last of equal maxima, the axis remaps of
`split_heads`/`merge_heads`, `sum_to` and `broadcast_like`, the masks and
positions. In 0.3.0 step 2 the kernels are *interpreted* by a small
evaluator in the kernel's machine type, every operation rounded to that
type: it is the first native-arithmetic realization and the yardstick of
the code generator to come, not a fast path yet.

Not in v0: tiles, memory spaces other than one global space, collectives, a
dependency graph between launches (they are sequential), the versioned
binary `.pskk` (spec question 35). Fusion groups (0.4.0), a parallel
outermost loop split across threads (0.4.0, below) and the vector form
(0.6.0, above) have been added since.

## Conformance

The conformance suite (`crates/presk-kernel/tests/conformance.rs`) compares
the kernel backend with the reference interpreter at 256 bits:

- every tensor primitive on random operands, in `f64` and `f32` (41 cases);
- every example entry (`mse`, `mse_grad`, the norms, `MiniGPT.apply`, the
  training `loss`, `adamw.step`, `next_token`, the gradients through
  stacked modules, the regression's loss), in both realizations.

The tolerance is `|x − x̂| ≤ ε · (|x| + a)` with `ε = 2⁻⁴⁰, a = 2⁻⁵⁰` for
`f64`, `ε = 2⁻¹⁵, a = 2⁻²⁰` for `f32` and `ε = 2⁻⁴, a = 2⁻⁵` for `bf16`.
Since 0.5.0 step 2 the suite is parameterized by a **realization** rather
than a machine type, so a mixed or reassociated one is reported the same way;
the bands above are those of uniform ones. Measured worst gaps of the
examples (`f64` and `f32` on 2026-09-19, `bf16` on 2026-09-20):

| Entry | f64 | f32 | bf16 |
|---|---|---|---|
| `mse` | 1.2e-16 | 6.4e-8 | 1.7e-3 |
| `mse_grad` | 1.5e-14 | 2.0e-6 | 2.8e-2 |
| `LayerNorm{8}.apply` | 1.2e-15 | 2.4e-7 | 3.2e-2 |
| `RMSNorm{8}.apply` | 2.2e-16 | 8.2e-8 | 6.1e-3 |
| `MiniGPT.apply` (tiny) | 0 | 0 | 0 |
| training `loss` (tiny MiniGPT) | 2.1e-16 | 1.1e-7 | 6.3e-4 |
| `adamw.step` | 1.0e-14 | 1.8e-6 | 7.6e-3 |
| `loss_grad` through a stacked module | 2.6e-14 | 5.3e-6 | 1.1e-2 |

The `f64` and `f32` columns are those of 0.4.0, digit for digit: making the
realization an object changed no arithmetic. The `bf16` column needs its
absolute floor `a` to mean anything — a result near a cancellation has
unbounded *relative* error at any precision, which says nothing about the
kernel, and that is why the bound rules of the precision analysis carry a
magnitude beside an error rather than a ratio alone.

(`MiniGPT.apply` is exactly 0 because its head is zero-initialized; the
gaps are those of the gradient path.) Reading against the machine epsilons
(1.1e-16 and 6e-8), the realizations lose a few units in the last place
per operation and a few dozen through a gradient, as expected. The table is
the interim form of the specification's "error bound carried by each
kernel" (question 39): stated and measured, not derived; the precision
solver will replace it.

## The Cranelift backend: `--backend native`

`presk-cranelift` compiles every kernel once, on first use, to machine code
with Cranelift's JIT, and the host program calls the function through a
pointer: buffers are passed as an array of pointers, the loop nest becomes
basic blocks with an induction variable per loop, scalar variables are
Cranelift variables, `sqrt` is a native instruction, and the transcendental
functions and `erf` are calls into `libm`, the same definitions the kernel
interpreter uses. Everything else, from the host program to the sampling
keys, is the kernel backend's, so a native run is the kernel run with the
launches compiled. Compilation happens in process; there is no object file,
no bundle and no cross-compilation yet (spec question 36).

The native backend passes the same conformance suite as the kernel
interpreter with the same measured gaps, up to the last digits of `libm`'s
functions against the standard library's (`loss_grad` through a stacked
module in f32: 9.4e-6 native, 5.3e-6 interpreted; both under the
tolerance).

## Training natively

`presk train --backend native [--realize f64|f32|bf16|@R.json]` is the training loop of
0.2.0 over the native backend: the objective, its `vjp` and `adamw.step` are
compiled kernels, the folds over stacked blocks and the loop itself run on
the host, the data pipeline, the schedule and the checkpoints are unchanged.
The same seed gives the same `θ₀` and the same batches on every backend, so
a native run is comparable to the reference digit for digit.

Two measurements. `crates/presk-cranelift/tests/native.rs` trains the
regression (`tests/train/mlp_regression.psk`) for 200 steps on the reference
at 256 bits and natively, and compares the final parameters:

| Realization | Worst relative gap after 200 AdamW steps |
|---|---|
| native `f64` | 4.4e-16 |
| native `f32` | 1.6e-7 |

And `presk export --conformance-entry train:K` now also runs the same `K`
steps natively in both realizations and writes the final parameters next to
the reference's (`conformance/native_f64.json`, `conformance/native_f32.json`);
`experiments/exact_vs_almost.py` reads them, so the realization gap of the
`experiments` chapter has four lines: presk's own `f64` and `f32` against the
256-bit reference, and PyTorch's float64 and float32 on the same batches
(tiny MiniGPT, 20 steps: 6.7e-16, 3.7e-7, 8.9e-16, 9.6e-7). Presk's `f32`
run is closer to the exact parameters than PyTorch's: the kernels compute
`gelu` through `erf`, the losses through the exact formula and the optimizer
through the same steps as the reference, with no fused or reassociated
operation that would round differently.

The larger configuration of the plan, `V=64 D=64 H=4 L=4` on the example
tokens with batches of 32, trains for 500 steps in under three minutes
natively (`--epochs 40`: the example's `epochs = 10` stops after 150), the
loss going from `ln 64` to 3.39 under the still-warming cosine schedule
(159 s with the debug build of `presk`, 163 s with the release build: the
time is in the compiled kernels, not in the host program around them):

```text
step 100  epoch 7  loss 4.132775  lr 1.500e-5  |g| 1.4102e0
step 200  epoch 14  loss 4.034732  lr 3.000e-5  |g| 1.4270e0
step 300  epoch 20  loss 3.886361  lr 4.500e-5  |g| 1.4509e0
step 400  epoch 27  loss 3.676601  lr 6.000e-5  |g| 1.3349e0
step 500  epoch 34  loss 3.385110  lr 7.500e-5  |g| 1.5861e0
```

`make train-examples` runs the reference commands of the `training` chapter
and then the tiny MiniGPT natively in `f64` and `f32` and the larger
configuration for 100 steps.

## Contractions: the blocked form

A contraction is where the time is, and the loop nest a naive lowering gives
is the wrong one: for each output element it walks a *column* of the right
operand, one element per cache line. Since 0.4.0 `dot` and `matmul` are
lowered in a **register-blocked** form instead. Eight columns of the result
are held in scalar accumulators across the whole contraction; the inner loop
loads one element of the left operand and a contiguous run of eight from the
right, and stores nothing. A width that is not a multiple of eight takes a
row-wise form, which sweeps a whole row of the result at a time and handles
any width.

Both are **exact**. Each output element still accumulates `k` in increasing
order — what changes is where the partial sums live and which memory the loop
touches, not the order of the additions. A test asserts that the blocked form
and the row-wise one agree bit for bit in `f64` and `f32`, and the conformance
gaps against the reference are unchanged to every digit of the table above.

A 1024³ matmul in `f32` on twelve cores:

| Form | 1 thread | 12 threads |
|---|---|---|
| row-wise | 3.7 GFLOP/s | 26.6 GFLOP/s |
| blocked | 9.9 GFLOP/s | 47–56 GFLOP/s |

The machine's scalar peak is about 84 GFLOP/s (twelve cores, one
fused multiply-add per cycle, two flops each), so the blocked form reaches
roughly two thirds of what this compiler can address without vector
instructions. The remaining factor of four was expected from those. 0.4.0
did not emit them, for the reason given under threads; 0.6.0 does, exactly
(the vector form, above), and they buy 1.1–1.25× rather than four, because
the eight accumulators already gave the loop its parallelism.

## Threads: `--threads N`

Since 0.4.0 a kernel whose outermost loop is *parallel* is split across
workers. A loop is parallel when the kernel is a single loop nest and every
store indexes the output with that loop's variable directly, so different
iterations write different elements. A reduction over the outer axis fails
that test, and so does a scatter such as `embed_vjp`, whose output index is a
loaded value; both stay on one thread.

This is exact, not approximate. Each output element is computed by the same
inner loops, in the same order, on the same values — only on another thread.
One worker and twelve produce bit-identical buffers in `f64` and in `f32`,
which the crate's test asserts element by element rather than within a
tolerance.

Only kernels above four million weighted operations are split: spawning
workers costs tens of microseconds, which a small kernel never earns back, so
the elementwise kernels stay sequential and the contractions parallelize.
`--threads N` sets the count on `run`, `train` and `bench`; the default is the
machine's cores, and `--threads 1` turns the split off.

The corpus GPT on twelve cores, one training step, `f32`:

| Model | 1 thread | 4 threads | 12 threads |
|---|---|---|---|
| `D=64, H=2, L=2`, batch 16, context 256 | 2.97 s | — | 0.83 s |
| the course's own `D=384, H=6, L=6` | — | — | 39.8 s |

A course-scale step took 289 s before this version's scheduling and backend
work and takes 39.8 s after it, seven times faster, with the loss unchanged
to the last digit at every stage: 289 s naive, 58.3 s once the threads split
the parallel axes, 39.8 s once the contractions were blocked. The scaling is well below twelve because only the big kernels are
split, because the outer loop bounds the workers (a batch of sixteen cannot
use more than sixteen), and because what is left — host dispatch, small
kernels, the sequential reductions — does not parallelize.

**Vectorization was not in 0.4.0.** The kernels that would vectorize
exactly, the elementwise ones, are not where the time is; the time is in the
contractions, whose innermost loop is a sum. Vectorizing a sum computes
partial sums per lane and adds them at the end, which *reassociates* it and
changes the result. That is a new realization, not an implementation detail:
it needs its own declared reduction order and its own measured gap to the
exact reference, which is how this tool treats every other realization, and
it is why the work stopped at threads (D31). 0.5.0 then found the other way
to put a contraction in lanes (D42): the blocked form's eight accumulators
are eight different output *columns*, so lanes across them leave every sum
in order. That is the vector form 0.6.0 emits, bit-identical to the scalar
kernel.

## Speed

`presk bench FILE --entry E [--dim …] [--const …] [--bits 64] [--repeat 5]`
runs one entry on every backend and prints the median of the calls. On the
tiny MiniGPT (`V=64 D=16 H=2 L=2, B=2, T=8`), 2026-09-19:

| Entry | Reference, 64 bits | Kernel interpreter f64 | Native f64 | Native f32 |
|---|---|---|---|---|
| `MiniGPT.apply` | 16.0 ms | 5.6 ms (2.9×) | 0.59 ms (27×) | 0.60 ms |
| training `loss` | 16.5 ms | 5.7 ms (2.9×) | 0.64 ms (26×) | 0.63 ms |
| `adamw.step` | 25.1 ms | 6.8 ms (3.7×) | 1.6 ms (16×) | 1.6 ms |

Training end to end, `presk train --backend native`:

| Run | Reference (64 bits) | Kernel interpreter | Native f64 |
|---|---|---|---|
| Regression, 200 steps | 0.5 s | 0.05 s | 0.04 s |
| Tiny MiniGPT, 20 steps | 34 s | 12 s | 0.4 s |
| MiniGPT `V=64 D=64 H=4 L=4`, 100 steps, batches of 32 | — | — | 32 s |
| MiniGPT `V=64 D=64 H=4 L=4`, 500 steps, batches of 32 | — | — | 159 s |

What these numbers are and are not. They are 0.3.0's, taken before fusion,
threads and blocked contractions (0.4.0) and vector lanes (0.6.0); the
`experiments` chapter and the sections above have the later figures. The
kernels were then naive loop nests, one per primitive, without fusion,
tiling or vectorization: `adamw.step` was a dozen elementwise passes over
every parameter and `MiniGPT.apply` a chain of small contractions, and the
host program is still interpreted between launches. Against the reference
the gain is two orders of magnitude, which is what a native path is for;
against PyTorch's BLAS on the same shapes it was far behind, as the plan
expected. `f32` is not faster than `f64` here because the work is dominated
by loop overhead, not by arithmetic width.
