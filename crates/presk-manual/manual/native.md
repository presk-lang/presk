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
  *arithmetic* type of the storage, so `bf16` storage accumulates in `f32`.
  Only the error analysis reads it: the kernels sum in the arithmetic type
  of their storage (found in 0.7.0 step 0, D56), so on the kernel and
  native backends a realization that declares any other accumulation type
  (`bf16` over `bf16`, `f64` over `f32`) is refused before anything runs
  (`E7002`, since 0.10.0 step 8; until then it was run as if absent, and
  condition 7, which reads it, admitted layouts whose bits differ from the
  default's, D97). The reference interpreter, which reads no realization,
  runs it. Kernels that honour it are question 67's.
- **`order`** is `sequential`, `blocked(width)` or `interleaved(partials)`.
  The first two are bit-identical — register blocking keeps every addition in
  index order (D32) — and the third reassociates, which is a different
  realization owing a measured gap (D31). It is not a performance hint: the
  constant in front of `u` in a contraction's error rule is a function of it.
- **`envelope`** names the magnitude measurements any bound about this
  realization assumed, because a bound without one is a different and
  unsupported claim (D35).
- **`params`** (since 0.12.0) gives a parameter a floating-point type of its
  own at rest, `f32` or `bf16`, keyed by its path in the tree as `quant`
  keys integer weights: `"params": { "blocks.ff.up.w": "bf16" }`. Every
  other parameter is stored in `default`. The `precision` chapter's
  *Parameters in their own type* says what is checked and what runs. A
  kernel of a wider type reads such a parameter from an input marked as
  narrower, `presk dump --stage kernel` printing it in its own type
  (`%1 : bf16[4, 4]` in an `f32` `dot`), whose loads widen each element
  exactly, in four lanes at a time for `bf16` into `f32` and two for `f32`
  into `f64`; it is never copied at the kernel's edge.

A parameter a realization quantises (`precision` chapter) reaches the
kernels as integers: `presk dump --stage kernel` shows such an input as
`int8/o[…]` or `int4/o[…]`, `o` the axis of the kernel's view its scales are
constant along, and the kernel loads each element as `f32(q)·s` — in four
lanes at a time when it has them — which is bit-for-bit the value the
reference path substitutes (D45). A program that reads the parameter through
`split` of that axis gives the kernel several axes for it, shown `int8/o..p`
(axes `o` to `p − 1`): the Llama decoder reads its output projection so. A
view in which no run of axes is that axis, as after a `merge` of it with
another, reaches the kernel as the weights' values, `f32(q)·s`, which is
what every load of them computes. (Until 0.10.1 the native backend and the C
export took the stored axis's number for the view's, and read past the
scales on such a program: D104's defect, fixed by D105.)

`bf16` is **storage only**: buffers hold sixteen bits, arithmetic is `f32`,
and conversion is a shift on the way in and one round to nearest even on the
way out (D38). A value reaches `bf16` through `f32`: an input or a weight
given in `f64` is rounded to `f32`, then to `bf16`, two roundings, as the
`export_c` chapter's table says and the language reference states as the
rule (its chapter 4). And a `bf16` nest holds its interior values in `f32`:
a fusion group computes every member in its root's arithmetic type and
stores only the root, so a value's bits depend on the partition of its
function into groups, which is why a gradient keeps its forward pass's
groups wherever a realization can see them (D92, D94; the `language`
chapter, "Residuals"). It halves the traffic of the memory-bound sweeps — `adamw.step`
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

Reductions and contractions accumulate in the arithmetic type of their
storage (the default accumulator type; see `accum` above), indices are
`i64`, hyperparameters and `nat` values are converted at their injection.
`gelu` is the exact `x·Φ(x)` through `erf`; `softmax` and `cross_entropy`
subtract the row maximum (a value-preserving rewrite over ℝ). A mask
(`bool[…]`, `causal_mask`, 0.10.0 step 7b) is an index buffer of 0 and 1,
exact in every realization: the masked softmax reads it in selections, its
maximum from the first allowed position and its sum over the allowed ones,
0 where it refuses, a row it empties giving zeros (the interpreter refuses
such a row), so that its bits are the softmax's of the scores with `−∞` at
the refused positions, the additive mask's until then. Where a site
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
code wins. By default lanes are chosen for the whole run; since 0.7.0 a
realization plan states them per nest, one or the register's width, and
`presk plan` chooses them with the other decisions of a nest's layout
(`realize` chapter).

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

**0.7.0 found most of the factor in the layout.** Lanes on the `[1, 8]` tile
had little to add; lanes on a taller tile do. Measured over the whole layout
space of the same product (1024³, `f32`, pinned to falicon's ten Cortex-X925
cores, two runs; D62), the best point is the tile `[4, 16]` in four lanes,
the block loops in the order `JI`, split over ten workers at depth 1: 4.67
ms, **460 GFLOP/s**, against 84 GFLOP/s for 0.6.0's own point (the `[1, 8]`
tile, the executor's split) in the same sweep. That is 5.49×, where
statement P of 0.7.0 asked for 3 (`experiments` chapter §7), and within 1.6×
of the 710 GFLOP/s PyTorch reaches through its BLAS on twenty cores. Four
accumulator rows share each load of the right operand, and sixteen columns in
four registers share each load of the left one: what the tile buys is fewer
loads per multiply-add, which is also the one thing the layout cost model
does not count (`optimization` chapter).

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
primitives sharing one loop nest (`realize` chapter). `presk dump --stage
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

Not in v0: memory spaces other than one global space, collectives, a
dependency graph between launches (they are sequential), the versioned
binary `.pskk` (spec question 35). Fusion groups (0.4.0), a parallel
outermost loop split across threads (0.4.0, below), the vector form (0.6.0,
above), and tile loops, a pack and a split at any depth (0.7.0, below) have
been added since.

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
examples (`f64` and `f32` on 2026-09-19, `bf16` on 2026-09-23, after the
odd-width `bf16` `dot` began accumulating in `f32`, 0.7.0 step 2):

| Entry | f64 | f32 | bf16 |
|---|---|---|---|
| `mse` | 1.2e-16 | 6.4e-8 | 2.4e-3 |
| `mse_grad` | 1.5e-14 | 2.0e-6 | 1.2e-2 |
| `LayerNorm{8}.apply` | 1.2e-15 | 2.4e-7 | 3.2e-2 |
| `RMSNorm{8}.apply` | 2.2e-16 | 8.2e-8 | 6.1e-3 |
| `MiniGPT.apply` (tiny) | 0 | 0 | 0 |
| training `loss` (tiny MiniGPT) | 2.1e-16 | 1.1e-7 | 6.3e-4 |
| `adamw.step` | 1.0e-14 | 1.8e-6 | 7.6e-3 |
| `loss_grad` through a stacked module | 2.6e-14 | 5.3e-6 | 1.2e-2 |

The `f64` and `f32` columns are those of 0.4.0, digit for digit: making the
realization an object changed no arithmetic, and neither did 0.7.0's layout
space. The bands are asserted on the examples in all three types, and on the
compiler's own test programs (`tests/data`, `tests/train`, the MLPerf
transcription) in `f64` and `f32`; their `bf16` gaps are reported. On the
gradient of `stack_prefix_grad`, a small result left over from larger terms
cancelling crosses the `bf16` band on some inputs and not others, with 0.6.0's
kernels as with 0.7.0's (D58). The `bf16` column needs its
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

Since 0.9.0 step 3 the suite also runs the examples with every value their
initializers set to zero drawn (`examples_with_their_zeros_drawn`), so that
MiniGPT's logits are not zero. The band holds on every example in `f64`,
and on every one but `MiniGPT.apply` in `f32` and `bf16`. There a few
logits, 6·10⁻⁴ to 3·10⁻³ against a median of 0.17, are sums that cancel:
4 of 1 024 leave the `f32` band and 35 the `bf16` band, by up to 3.7 times
it. They are reported, not asserted. The native backend leaves the band on
other logits than the kernel interpreter, since their `exp` differ in the
last bits, so no rule holding one executor to the other element by element
would hold either (spec question 75, D79).

## The Cranelift backend: `--backend native`

`presk-cranelift` compiles every kernel once, on first use, to machine code
with Cranelift's JIT, and the host program calls the function through a
pointer: buffers are passed as an array of pointers, the loop nest becomes
basic blocks with an induction variable per loop, scalar variables are
Cranelift variables, `sqrt` is a native instruction, and the transcendental
functions and `erf` are calls into the `libm` crate, a port of musl's C
library as it was before musl 1.1.23 rewrote `exp` and `log` (the C export
ships those functions, bit-identical to the crate: D78). The kernel interpreter calls the same crate for `erf` only, and
Rust's standard library, the system's C library (glibc on Linux), for
`exp`, `log`, `tanh`, `sin` and `cos`. Everything else, from the host program to the sampling
keys, is the kernel backend's, so a native run is the kernel run with the
launches compiled. Compilation happens in process; there is no object file,
no bundle and no cross-compilation yet (spec question 36): an entry that
must run elsewhere leaves as a C program (below).

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

## Contractions: one builder, a layout space

A contraction is where the time is, and the loop nest a naive lowering gives
is the wrong one: for each output element it walks a *column* of the right
operand, one element per cache line. Since 0.4.0 `dot` and `matmul` are
lowered in a **register-blocked** form instead: eight columns of the result
held in accumulators across the whole contraction, the inner loop loading one
element of the left operand and a contiguous run of eight from the right. A
width that is not a multiple of eight took a **row-wise** form, which sweeps a
whole row of the result at a time and handles any width.

Since 0.7.0 step 2 both are points of one **layout space**, and one builder
lowers every point, for a `dot` whose right operand has rank two and for an
`ein` of attention's pattern alike (the language reference's *contraction
nest*: the output is the outer axes, then `I`, then `J`, over one contracted
axis `K`), and since 0.10.0 step 1 for the two derivatives of such a `dot`,
`dot_vjp_r` summing over several leading axes as one `K` (`realize`
chapter). A realization plan (`realize` chapter) chooses, per nest:

- the **form**: row-wise, or tiled;
- the **register tile** `[m_i, n_j]`: `m_i` rows by `n_j` columns of the
  result held in accumulators, `[1, 1]` or `m_i` in 1, 2, 4 and `n_j` in 4,
  8, 16;
- the **k-block**: none, 32 or 128 — a loop over blocks of `K` and a
  within-block loop inside it;
- the **loop order** of the block loops, outermost first (`IJ`, `JI`, or with
  a k-block a permutation of `IJK`);
- the **lanes**: one, or the register's width across the tile's columns;
- since 0.10.0 step 7, the **epilogue**: where one `add` alone reads the
  product's value (a linear layer's bias), the product's kernel adds the
  `add`'s other operand to each output element once its sum is done, and
  the `add` launches nothing. In `f64` and `f32` the operand is added to the
  sum in its register before the store; in `bf16` the sum is stored, which
  rounds it, read back, added to and stored again, as the `add`'s own kernel
  would; under a spill, in the last block of `K` only. So it computes the two
  nests' bits (`realize` chapter, "The bias as an epilogue").

The default is 0.6.0's kernel. A `dot` whose `J` is a multiple of eight is
tiled `[1, 8]`, its text character for character what 0.6.0 printed. Any
other `dot` is row-wise. An `ein` is the tile `[1, 1]`, one accumulator per
output element, which is the naive nest 0.6.0 emitted less a multiplication by
one it carried (exact, so no bit moved). The derivatives of a `dot` are the
tile `[1, 1]` too, and their text at the default is 0.9.0's, character for
character, since a one-element tile of theirs reads its operand 0 inside the
product (0.10.0 step 1, D89). One default changed, on purpose: a
`bf16` `dot` whose `J` is not a multiple of eight is now tiled `[1, 1]` and no
longer row-wise, because the row-wise form keeps its partial sums in the
`bf16` output and so accumulated in `bf16` where the realization says `f32`
(D56). On `x[4, 1024] · w[1024, 12]` its median distance to the exact product
falls from 2.4·10⁻² to 2.7·10⁻³, the tiled form's. The derivative of
`embed` and `sum_to` still keep their running sums in the output buffer, so
on a `bf16` nest they accumulate in `bf16`: they are not contractions, and
have no other form yet.

Every point is **exact**. Every loop counts upward and a within-block loop
runs inside its block loop, so each output element adds its terms in
increasing `k` whatever the tile, the order or the lanes. What changes is
where a running sum lives and which memory the loops touch. When the loop over
the blocks of `K` encloses the one over `I` or `J`, the accumulators are
stored at the end of a block and reloaded at the start of the next (`if kb {
… }` in the kernel text). That is exact only where the storage type is the
accumulation type, so a plan cannot state it for `bf16`, nor the row-wise
form. Lanes run across the tile's columns and never across `k`, and need
operand 1 to be contiguous along `J`. On an NT contraction such as
attention's scores `ein`, whose operand 1 has `K` innermost, lanes need a
**pack**: operand 1 copied, in its own type, to a panel with the axes (the
outer axes it carries…, `K`, `J`), which the contraction then reads as NN. The copy is a kernel of its
own, launched just before the contraction's (`pack` in `dump --stage
kernel`); both launches are the nest's (since 0.7.0 step 3).

With this plan in `pack.json`, the tutorial's scores `ein` is packed, tiled
two rows by four columns in four lanes, and split at depth 1, its heads, over
two workers:

```json
{ "realization": "f32",
  "layout": { "scores/%2": { "tile": [2, 4], "lanes": 4, "pack": true,
                             "split": { "depth": 1, "workers": 2 } } } }
```

```sh
presk dump tutorial/03_attention.psk --stage kernel --entry scores --dim B=1 --dim T=4 --plan @pack.json
```

```text
kernel pack (%0 : f32[batch: 1, heads: 2, seq: 4, hd: 4]) -> %1 : f32[batch: 1, heads: 2, hd: 4, seq: 4] {
  for i0 in 0..1 {
    for i1 in 0..2 {
      for i2 in 0..4 {
        for i3 in 0..4 {
          %1[i0, i1, i2, i3] = %0[i0, i1, i3, i2]
        }
      }
    }
  }
}
kernel ein[batch, heads, seq, seq2] (%0 : f32[batch: 1, heads: 2, seq: 4, hd: 4], %1 : f32[batch: 1, heads: 2, hd: 4, seq: 4]) -> %2 : f32[batch: 1, heads: 2, seq: 4, seq2: 4] {
  for i0 in 0..1 {
    for i1 in 0..2 {
      for i2 in 0..2 {
        for i3 in 0..1 {
          v0 = splat(0)
          v1 = splat(0)
          for i4 in 0..4 {
            s0 = %0[i0, i1, ((i2 * 2) + 0), i4]
            s1 = %0[i0, i1, ((i2 * 2) + 1), i4]
            v2 = %1[i0, i1, i4, ((i3 * 4) + 0)]<v>
            v0 = (v0 + (splat(s0) * v2)<v>)<v>
            v1 = (v1 + (splat(s1) * v2)<v>)<v>
          }
          %2[i0, i1, ((i2 * 2) + 0), ((i3 * 4) + 0)] <v= v0
          %2[i0, i1, ((i2 * 2) + 1), ((i3 * 4) + 0)] <v= v1
        }
      }
    }
  }
}
```

The copy writes `k` with `hd` before `seq`. The contraction then holds its
tile in two vector accumulators, and for each `i4`, the summed `hd`, loads
two elements of `q` and one vector of four from the panel, which both rows
share. The split is not in the text: a kernel carries it, and the Cranelift
backend ranges loop `i1` over the workers with the loop above it in full. It
is checked against the machine when the kernel runs natively, so `presk run
… --backend native --plan @pack.json` needs `--threads 4`, where two workers
are half the machine (condition 5); the kernel interpreter runs everything on
one worker and is not held to it.

A test lowers **every point** of four nests — a `dot` as `Linear` has it,
attention's two `ein`s and its softmax — in `f64`, `f32` and `bf16`, 1 491
points in all, the pack included, each to kernels of its own. It finds every
output bit equal to the default's, on the kernel interpreter and natively. The
softmax's points are its three passes and its two: the exponentials stored
into the output while their sum is taken, then divided in place, each computed
once (`f64` and `f32` only, since a `bf16` output would round them twice). A
fifth of the points is also split at depths 0, 1 and 2 over two and four
workers, natively, with the same result.

A 1024³ matmul in `f32` on twelve cores:

| Form | 1 thread | 12 threads |
|---|---|---|
| row-wise | 3.7 GFLOP/s | 26.6 GFLOP/s |
| blocked | 9.9 GFLOP/s | 47–56 GFLOP/s |

These are 0.4.0's measurements, on another machine. The machine's scalar peak was about 84 GFLOP/s (twelve cores, one
fused multiply-add per cycle, two flops each), so the blocked form reaches
roughly two thirds of what this compiler can address without vector
instructions. The remaining factor of four was expected from those. 0.4.0
did not emit them, for the reason given under threads; 0.6.0 does, exactly
(the vector form, above), and they buy 1.1–1.25× rather than four, because
the eight accumulators already gave the loop its parallelism.

On falicon's ten large cores the layout space of 0.7.0 does better (D62):

| Point, 1024³ `f32` | GFLOP/s |
|---|---|
| 0.6.0's: tile `[1, 8]`, four lanes, the executor's split | 84 |
| 0.7.0's model's pick: tile `[1, 4]`, four lanes, order `JI`, split at depth 1 | 75 |
| 0.8.0's frozen model's pick: tile `[4, 16]`, k-block 128, four lanes, order `KJI`, split at depth 2 | 445 |
| the best measured: tile `[4, 16]`, four lanes, order `JI`, split at depth 1 | 460 |

0.7.0's pick is the first, in the space's order, of 21 points its model
predicts equal, which measure between 8.1 and 28.6 ms: the pick is among
the slowest of them. 0.8.0's model, which counts register loads (the
`optimization` chapter), picks a point 1.03 times slower than the best. That
row prices 0.7.0's measurement again under the frozen model (`make
ra4-dev-frozen`), and the model was designed on those measurements and
calibrated on a GEMM, so it is not a test; the fresh sweeps are
(`experiments` chapter §8). Every point of the space is bit-identical to the
scalar sequential kernel. Packing both operands, as BLAS libraries do, is not in the space
(`docs/kernels-roadmap.md`).

## The host's values (0.10.0 step 6)

The host passes values between launches without copying them (D96). A
tensor's elements are shared by every value that reads them: an operand
taken from the environment, a call's argument, a tuple's component, a
projection, a fold's accumulator (`presk_rt::Elems`, behind a reference
count). A stacked layer's slice is a view of its stack, as the C program
passes `w + l · chunk`, and a scan's stack is its parts laid end to end,
which a fold reads layer by layer without a copy; they are laid out in one
buffer only if something reads the stack whole. Nothing writes elements
another value sees: a kernel writes only its output, and a write to shared
elements copies them first. A launch's output comes from a free list of the
buffers freed before, by length, so after the first call no page is
touched for the first time; it is not zeroed, since every kernel writes its
output before reading it (a debug build fills it with NaN, which the tests
would see). A training dropout mask is drawn in pieces on the workers, each
from its place in the key's stream, the bits of one stream drawn in order;
outside training the mask is ones, one buffer per length, drawn nowhere.
`presk bench --host-profile` times each of these (`cli` chapter).

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

**A pool of workers** (0.10.0 step 6, D96). Until then a divided launch
started a thread for each of its pieces and joined them, about 210 µs a
launch of ten threads, and the scheduler placed each new thread afresh,
often on the slower cores. The executor now keeps `N − 1` workers for its
life (`presk_cranelift::Pool`, on the standard library): a launch posts
piece `t` to worker `t`, runs piece 0 on the calling thread and waits for
the others, about two microseconds beyond its longest piece. The pieces are
the ones the threads ran, so the bits are. A waiting worker spins about two
milliseconds before it parks, as OpenMP's runtime does, and a few
microseconds when the process's workers outnumber its cores. The workers
are not pinned to cores: they run where the process may (`taskset`), and a
persistent worker stays on a fast core, which is what pinning was for.

**A split at a depth** (0.7.0 step 3). A realization plan may state a split
per nest, `{ "depth": d, "workers": p }` (`realize` chapter): the loop at
depth `d` of the nest's kernel is divided among `p` workers, each running the
loops that enclose it in full and a disjoint range of its iterations,
whatever the kernel's work — the four-million threshold is the default's
alone. At batch one the outermost loop has one iteration and the default
splits nothing; a split at depth 1 or 2 (the heads, a block loop of a
contraction) still can. The loop must be *parallel* in the language
reference's sense, checked once the kernel exists: the loops down to it form
a perfect nest, and one position of the output is indexed by `c·v + t`,
`0 ≤ t < c`, in every store to the output and every load from it, so the
block loops of a tiled contraction qualify through their `jb·n_j + t`. A
split on a loop that is not parallel (the loop over the blocks of `K`, a
softmax's row) is refused (`E7002`, condition 8), and so is a worker count
other than the machine's or half of it (condition 5, with `--threads` or
`PRESK_THREADS` giving the machine's count). `"split": "none"` keeps a nest on
one worker. The test that one worker and many agree bit for bit now also
splits every kernel of the course GPT at depths 0, 1 and 2, at batch 4 and at
batch 1.

At batch one this is where most of the layout space's gain is. 0.6.0 ran the
course GPT's two attention `ein`s at `B = 1` on one worker with the tile
`(1, 1)`; the best point of each space, split over the heads or a block loop
and tiled, is **47×** and **77×** faster in `f32` (D62). At batch 16, where
the default already splits the batch, the factors are 9.2× and 16.5×. It is the
fix `docs/kernels-roadmap.md` named after BERT-large, whose forward pass at
batch one ran on one core of twenty.

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

## A C program: `presk export --target c`

Since 0.9.0 an entry can leave the compiler as a C program that computes
this backend's bits: its kernels are this backend's, printed as C, its host
a recording of this backend's own run of the entry, and it ships the
transcendental functions the `libm` crate ports (musl 1.1.22's), built
without floating-point contraction. The `export_c` chapter says how it is
made, why its bits are this backend's, and what is checked.
