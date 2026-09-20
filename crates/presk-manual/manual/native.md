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
`f32`, `i64` or *exact* MPFR reals. The reference interpreter is the first
backend (`--backend interp`); its exact values cross the seam untouched, so
the training loop written over the seam reproduces 0.2.0's results digit for
digit. Input synthesis stays with the interpreter: any backend runs on the
inputs the reference would.

## Realization: `--realize f64|f32`

A native backend computes every real tensor in one machine type. Reductions
and contractions accumulate in that type (PyTorch's convention), indices
are `i64`, hyperparameters and `nat` values are converted at their
injection. `gelu` is the exact `x·Φ(x)` through `erf`; `softmax` and
`cross_entropy` subtract the row maximum (a value-preserving rewrite over
ℝ). The realization is uniform: per-tensor choices are the precision
solver's job, in a later version. The interpreter ignores the flag and
computes at `--bits`.

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
per tensor primitive. `presk dump --stage kernel` runs an entry and prints
the kernels it launched:

```sh
presk dump tutorial/01_linear.psk --stage kernel --entry 'Linear{3, 2}.apply' --dim batch=4 --realize f32
```

```text
-- 2 kernels, 2 launches, f32
kernel dot (%0 : f32[batch: 4, 3], %1 : f32[3, 2]) -> %2 : f32[batch: 4, 2] {
  for i0 in 0..4 {
    for i1 in 0..2 {
      s0 = 0
      for i2 in 0..3 {
        s0 = (s0 + (%0[i0, i2] * %1[i2, i1]))
      }
      %2[i0, i1] = s0
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

The lowering follows the interpreter's semantics rule for rule: its
alignment of axes for broadcasting (trailing positions when names agree,
else by name), the contraction axes of `dot` and its adjoints, the general
`ein`, the row-wise `softmax` and `cross_entropy`, gathers and scatter-adds
for `embed`, `argmax` keeping the last of equal maxima, the axis remaps of
`split_heads`/`merge_heads`, `sum_to` and `broadcast_like`, the masks and
positions. In 0.3.0 step 2 the kernels are *interpreted* by a small
evaluator in the kernel's machine type, every operation rounded to that
type: it is the first native-arithmetic realization and the yardstick of
the code generator to come, not a fast path yet.

Not in v0: tiles, memory spaces other than one global space, parallel
axes, collectives, a dependency graph between launches (they are
sequential), fusion, the versioned binary `.pskk` (spec question 35).

## Conformance

The conformance suite (`crates/presk-kernel/tests/conformance.rs`) compares
the kernel backend with the reference interpreter at 256 bits:

- every tensor primitive on random operands, in `f64` and `f32` (41 cases);
- every example entry (`mse`, `mse_grad`, the norms, `MiniGPT.apply`, the
  training `loss`, `adamw.step`, `next_token`, the gradients through
  stacked modules, the regression's loss), in both realizations.

The tolerance is `|x − x̂| ≤ ε · (|x| + a)` with `ε = 2⁻⁴⁰, a = 2⁻⁵⁰` for
`f64` and `ε = 2⁻¹⁵, a = 2⁻²⁰` for `f32`, and the measured worst gaps of
the examples on 2026-09-19:

| Entry | f64 | f32 |
|---|---|---|
| `mse` | 1.2e-16 | 6.4e-8 |
| `mse_grad` | 1.5e-14 | 2.0e-6 |
| `LayerNorm{8}.apply` | 1.2e-15 | 2.4e-7 |
| `RMSNorm{8}.apply` | 2.2e-16 | 8.2e-8 |
| `MiniGPT.apply` (tiny) | 0 | 0 |
| training `loss` (tiny MiniGPT) | 2.1e-16 | 1.1e-7 |
| `adamw.step` | 1.0e-14 | 1.8e-6 |
| `loss_grad` through a stacked module | 2.6e-14 | 5.3e-6 |

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

`presk train --backend native [--realize f64|f32]` is the training loop of
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
instructions. The remaining factor of four would come from those, and the
`native` backend does not emit them yet — for the reason given under threads.

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

**Vectorization is not in this version.** The kernels that would vectorize
exactly, the elementwise ones, are not where the time is; the time is in the
contractions, whose innermost loop is a sum. Vectorizing a sum computes
partial sums per lane and adds them at the end, which *reassociates* it and
changes the result. That is a new realization, not an implementation detail:
it needs its own declared reduction order and its own measured gap to the
exact reference, which is how this tool treats every other realization. The
plan of this version says so, and it is the honest reason the work stopped at
threads.

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

What these numbers are and are not. The kernels are naive loop nests, one
per primitive, without fusion, tiling or vectorization: `adamw.step` is a
dozen elementwise passes over every parameter and `MiniGPT.apply` a chain of
small contractions; the host program is still interpreted between launches.
Against the reference the gain is two orders of magnitude, which is what a
native path is for; against PyTorch's BLAS on the same shapes it is far
behind, as the plan expects, and the 0.3.0 e-graph pass with its calibrated
cost model is the first tool for closing that gap. `f32` is not faster than
`f64` here because the work is dominated by loop overhead, not by arithmetic
width.
