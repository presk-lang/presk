# PSK-Realize: fusion groups and realization plans

One kernel per primitive makes every intermediate a round trip through
memory. PSK-Realize v0 (crate `presk-realize`, 0.4.0 plan §3.4) partitions
each PSK-Core body into **fusion groups**: connected sets of elementwise
nodes that share a loop nest, so their intermediates stay in registers and
only the group's result is written. `presk dump --stage realize` prints the
partition. Since 0.7.0 a **realization plan** gives each nest a layout
beside the realization (below, "The realization plan").

The representation was named **PSK-Sched** until 0.6.0, and the crate
`presk-sched`: the language reference renamed it on 23 September 2026,
because "schedule" suggests an ordering in time and what a plan at this
level mostly fixes is arithmetic. `--stage sched` is kept as an alias of
`--stage realize`, and `make sched` of `make realize`; the compiler's
realization files still say "site" where the reference says *nest*.

## What a group is

A group is a set of `let`s of one function such that

- every member is an *elementwise* primitive — `add`, `mul`, `exp`, `gelu`,
  `sqrt`, … A reduction ends a group, because its loop nest is not the nest
  of its consumers;
- every member has the same result type, so one loop nest serves them all.
  Inputs may be smaller and broadcast, exactly as the unfused kernels do;
- every *interior* member, that is every member but the root, is read only by
  other members. A value read from outside has to reach memory for the
  outside reader, so it cannot be interior.

Those three conditions are what make the group safe to launch as one kernel,
and they are what the crate's test asserts on the corpus and the examples.
The partition is structural: it depends on the body, not on the shapes or the
realization, so it is computed once per function and reused by every backend
and every dimension. It is part of what a program computes, not only of how
fast: a value inside a group stays in the kernel's arithmetic type, `f32`
for `bf16`, where a value that reaches memory is rounded to its storage
type. So the dead-code pass after differentiation (0.10.0 step 3, the
`language` chapter) keeps every group of what a gradient's result reads,
and the partition is computed in PSK-Core (`presk_core::fusion`, which this
crate re-exports) so that the pass can check it.

A value read twice *inside* a group appears twice in the emitted expression,
which trades an arithmetic operation for a round trip through memory. That
trade is bounded: past sixty-four nodes the group stops growing. `pow n` with
`n > 2` never joins a group for the same reason — it expands to `n − 1`
products of its argument.

**Parameter trees.** A group whose type is `params M` is mapped over the
leaves of the tree, one launch per leaf for the whole chain. This is where
most of the saving is: an optimizer step is one long elementwise chain over
every parameter, and without it each step of the chain launches again for
every leaf.

## What it removes

On the corpus GPT at `V=92, D=64, H=2, L=2, B=2, T=32`, in `f32`:

| Entry | Launches without fusion | With | Removed |
|---|---|---|---|
| `loss`, the forward pass | 108 | 81 | 25 % |
| `adamw.step` | 414 | 111 | 73 % |
| a whole training step | 1515 | 884 | 42 % |

`presk train` prints the last row for any run on a kernel backend. The plan
of this version asked for half the launches to go; 42 % is short of it, and
the reason is the one below — the forward and backward passes are where the
launches are, and their chains are cut by reductions and by values that two
consumers read.

The forward pass fuses little because reductions cut its chains: the
`(x − μ)² → mean → +ε → sqrt → divide` of a layer normalization is split at
each `mean`, and `x − μ` is read by both the variance and the division, so it
cannot be interior. The optimizer fuses well because it is one chain of
twenty-odd elementwise steps over a tree of parameters, with nothing reading
its intermediates.

## What it buys, measured

On the tiny MiniGPT, native `f64`, medians of 21 calls, three runs each,
before and after the pass:

| Entry | Before | After |
|---|---|---|
| `MiniGPT.apply` | 0.57–0.61 ms | 0.59–0.61 ms |
| training `loss` | 0.58–0.63 ms | 0.60–0.62 ms |
| `adamw.step` | 1.54–1.60 ms | 1.31–1.33 ms |

The forward pass does not move: its time is in the contractions, which fusion
does not touch, and removing a quarter of its launches removes a quarter of
almost nothing. `adamw.step` gains about 15 %, consistently and outside the
run-to-run spread of ±4 %.

That number says something about the cost model. Removing 303 launches at the
measured λ of 50 ns is 15 µs, one percent of the 1.5 ms — so the saving is
*not* launch overhead. It is traffic: each elementwise step over 200 000
parameters reads and writes 1.6 MB, and fusing twenty steps into six groups
removes tens of megabytes. That is the `Q` term of the roofline, and it is
the first measurement in this project where a compiler pass moved the clock
outside the noise.

The training losses are unchanged to the last digit. Fusion reassociates
nothing: the operations happen in the same order, on the same values, in the
same machine type — only the intermediates stay in registers. Every golden
output of the test suite is byte-identical with the pass on.

## The realization plan

A realization plan (0.7.0 step 1; language reference §6.6) is a realization
and a **layout** for each nest: `--plan @P.json` on `presk run`, `bench`,
`train` and `dump`, in place of `--realize`. The layout is kept beside the
realization and not inside it, because a checkpoint, a measured gap and a
served model name the realization, and a change of tile must not change what
they name: a layout moves no bit, so it owes no gap.

```json
{
  "realization": "f32",
  "profile": "out/profile.json",
  "layout": {
    "twice/%2": { "lanes": 4 },
    "twice/%3": { "lanes": 1 },
    "attend/%3": { "passes": 3 }
  }
}
```

- **`realization`**, required: a realization record (`native` chapter), the
  name of a uniform one (`f64`, `f32`, `bf16`) or `@FILE`, as `--realize`
  takes it. `--realize R` is the plan `{ "realization": R }`.
- **`profile`**: the machine profile the plan was chosen against, written
  by `presk plan` and printed by `presk dump --stage realize`. No command
  checks it against the machine: a plan is advisory on another one
  (question 64).
- **`layout`**: a record per nest, named as the realization's sites,
  `"<function>/%<let>"` with the binding as `presk dump --stage core` prints
  it: the root of a fusion group, or a primitive of its own. A decision not
  stated takes its default, and a nest not mentioned takes the default of
  every decision.

## The layout decisions

The decisions and their domains are the language reference's (§6.3, table
`realize-decisions`), and since 0.7.0 step 3 every one of them is lowered. A
nest is a **contraction** (a `dot` whose right operand has rank two, the two
derivatives of such a `dot` since 0.10.0 step 1, or an `ein` whose output is
the outer axes, then `I`, then `J`, summed over one axis `K`: both of
attention's `ein`s), a **softmax**, or anything else (an elementwise group, a
reduction, a contraction of another pattern), whose only decision is the
thread split.

| Decision | Nests | Domain | Default | What it changes |
|---|---|---|---|---|
| `form` | contraction | `"tiled"`, `"row-wise"` | tiled; row-wise for a `dot` whose `J` is not a multiple of eight, except in `bf16` | where the running sums live: in registers, or in the output row from one `k` to the next |
| `tile` | tiled contraction | `[m_i, n_j]`: `[1, 1]`, or `m_i` in 1, 2, 4 and `n_j` in 4, 8, 16 | `[1, 8]` on a `dot` whose `J` is a multiple of eight, `[1, 1]` otherwise | how many results are held in accumulators, so how often each operand element is loaded |
| `kblock` | tiled contraction | `"none"`, 32, 128 | `"none"` | a loop over blocks of `K` and one within each block |
| `order` | tiled contraction | the block loops, outermost first: `"IJ"`, `"JI"`; with a k-block a permutation of `"IJK"` | `"IJ"`, or `"IJK"` with a k-block | which operand stays in cache while the other streams |
| `lanes` | tiled contraction | 1, or the register's width (four in `f32`, two in `f64`, one in `bf16`) | the register's width where it divides `n_j` and operand 1 is contiguous along `J`, else 1 | vector registers across the tile's columns, never across `k` |
| `split` | every nest | `"none"`, or `{ "depth": d, "workers": p }` with `d` at most 2 and `p` the machine's workers or half of them | the executor's: the outermost loop over every worker, above four million weighted operations | which loop is divided among threads |
| `pack` | NT contraction | `true`, `false` | `false` | operand 1 copied, before the contraction, to a panel with the axes (the outer axes it carries…, `K`, `J`), so that `J` is contiguous and lanes can read it |
| `passes` | softmax | 3, 2 | 3 | two passes store `exp(x − m)` while summing it and divide in place: each exponential computed once. A softmax with a mask (`where = m`, since 0.10.0 step 7b) reads it as one more input, indices 0 or 1, in both forms: the maximum and the sum over the allowed positions, 0 where it refuses, the bits of the softmax of the scores with `−∞` at every refused position |
| `epilogue` | contraction whose value one `add` alone reads | `true`, `false` | `false` | the product's kernel adds the `add`'s other operand to each output element once its sum is done, and the `add` launches nothing (since 0.10.0 step 7, below) |

The default of every decision is the kernel 0.6.0 lowered, with one
exception the reference makes: a `bf16` `dot` whose `J` is not a multiple of
eight is tiled `[1, 1]` rather than row-wise, since the row-wise form would
keep every partial sum in the `bf16` output (since 0.7.0 step 2; `native`
chapter, D56, D58). The native backend's lanes are its default layout's, as
wide as each kernel allows; the kernel interpreter's are one, which is why
`presk dump --stage kernel` prints scalar kernels unless a plan states lanes.

**The derivatives of `dot`** (since 0.10.0 step 1, D89). A gradient's
largest contractions are the derivatives of its products, `dot_vjp_l` and
`dot_vjp_r` (`language` chapter, "Automatic differentiation"), and both are
contraction nests wherever the `dot` is one:

- **`dot_vjp_l`**, `ct[…, I, N] · w[J, N]`, the cotangent of the left
  operand, is NT: its summed axis `N` is innermost in both operands, its `I`
  is the cotangent's last leading axis, its `J` the rows of `w`, and the
  cotangent's other leading axes are outer. A pack copies `w` to its
  transpose, a panel `[N, J]`, since `w` carries no outer axis.
- **`dot_vjp_r`**, `x[…, I]ᵀ · ct[…, J]`, the cotangent of the right operand,
  sums over every leading axis of its operands: `batch` and `seq` on a
  transformer. They are one summed axis `K`, their lengths' product, iterated
  as one loop per axis, outermost first. That is the order of their flat
  index, since both operands are row-major and carry them first, so no
  operand is reshaped and the kernel's signature is its operands' own. A
  k-block divides the innermost of them, `seq`.

Their default layout is the tile `[1, 1]` with one lane, and at the
default their kernels are 0.9.0's, character for character: a one-element
tile of a derivative reads its operand 0 inside the product, where an
`ein`'s reads it into a scalar first. So `make plan-check`, every golden, the
C export's programs and the SCoP mode's point zero did not move. A plan may
state their layouts, and `presk plan` states them, so a plan it gives for a
gradient, which `presk train` runs too, tiles them. A derivative of a `dot`
that is not a contraction nest (a right operand of rank above two, a left
operand of rank one) keeps its own lowering. The derivative of an `ein` is an
`ein`, and a contraction nest when its axes follow the pattern.

**The bias as an epilogue** (since 0.10.0 step 7, D97). `Linear.apply`
is `x · w + b`: a product, then an `add` of the bias, which fusion cannot
take, since a group holds elementwise `let`s only and the product is none.
Stated on the product's nest, `"epilogue": true` gives the product's kernel
the `add`: the product's `let` launches nothing, and the `add`'s launches
the product's kernel, under the product's layout and at its site, which
adds the `add`'s other operand to each output element once that element's
sum is done, and binds the `add`'s value. One launch and one pass over the
output instead of two. It applies where `presk dump --stage core` shows:

- the nest is a contraction nest (a `dot`, an `ein` of the pattern, a
  derivative of `dot`) whose value one `let` alone reads, an `add` of it and
  another tensor, in either order, which is no member of a fusion group and
  so launches by itself; the value is not the function's result;
- the `add`'s site has the product's storage type, since one kernel computes
  both; a realization that types one of the two sites apart from the other
  makes the plan refused;
- the other operand broadcasts into the product's output without widening
  it: a bias along `J`, a mask carrying `I` and `J`, a residual of the
  output's shape, an operand carrying the outer axes or `I` alone; and,
  where the tile has lanes, it runs along `J` as its innermost axis or not
  at all.

It is exact. The separate nests round the sum to the storage type, then add
in the arithmetic type and round again; so does the epilogue. In `f64` and
`f32`, whose arithmetic is their storage, the operand is added to the sum in
its register before the store. In `bf16` the sum is stored, rounded, read
back in `f32`, added to and stored again, as the `add`'s kernel would read
it. Under a spill the addition belongs to the last block of `K` (every
block variable at its last value); in the row-wise form, one more sweep of
the output row. The kernels accumulate in their arithmetic type, so the
rounding the epilogue reproduces is the storage type's; a realization that
declares another accumulation type is refused on the kernel and native
backends (`E7002`, since 0.10.0 step 8: condition 7 read the declared type,
and under `bf16` accumulated in `bf16` admitted a spill whose bits differ
from the default's, D97). The kernel's name is the product's with `+add` (`dot+add`).

The epilogue is a layout decision, but no point of a nest's space: like the
split it stays as stated when a layout is resolved, so no point and no key
of `presk autotune`'s spaces carries it, and no archived measurement moves.
Its worth is the `add`'s launch, which a nest timed alone does not show.
`presk plan` states it on every product whose every launch can take it,
over the layout it chooses without it, and prints them (`cli` chapter); the
frozen model prices a nest alone and has no term for it. Unstated, the two
nests run as before, byte for byte.

**Every point is exact.** Each output element adds its terms in increasing
`k` whatever the tile, the k-block, the order, the lanes, the pack or the
split, so every point of a nest's space computes its default's bits (`native`
chapter, "Contractions"), and a plan owes no measured gap. Tests lower every
point of four nests in three types on both executors, and the measurement of
0.7.0 checked 17 879 more (`experiments` chapter §7). Since 0.10.0 step 1
they also lower every point of the derivative nests of three gradients, the
MLP's, MiniGPT's and the course GPT's at small constants (4 280 points on
twenty nest shapes, in three types, on both executors), and run each
gradient with every derivative nest planned at once, packs included. Since
0.10.0 step 7 they lower every point with its epilogue too, on the products
of `tests/plan/epilogue.psk` (a bias after a `dot`, before it, a mask after
the scores `ein`, a residual after the values `ein`, an operand constant
along `J`; 3 208 points in three types and under an `f64` accumulation) and
on the derivative nests given an operand to add (4 280 points), against the
product's kernel followed by the `add`'s, on both executors; and run the
MLP, MiniGPT and the course GPT with every product's epilogue planned at
four points, and MiniGPT's gradient under mixed realizations.

**Well-formedness.** A plan is checked twice, and an ill-formed layout is
refused with `E7002`, never ignored, because it would then be timed and
reported as if it had been applied. Before anything runs, every nest it names
must exist and be a nest, and state only decisions its kind has (a tile on a
contraction, passes on a softmax, an epilogue on a product whose value one
`add` alone reads, launched by itself and of the product's type). When a
nest is launched, its layout is
checked against its shapes and types — the reference's conditions 3 (the
tile and the k-block divide the axes; a row-wise contraction has no tile,
block, order or pack), 4 (the lanes are one or the register's width, divide
the tile's columns, and on an NT contraction need the pack), 7 (a running sum
kept in the output buffer — the row-wise form, or a k-block loop enclosing
the `I` or `J` block loop — and two softmax passes need the storage type to
be the accumulation type, so none of them on `bf16`) and 9 (a pack only on an
NT contraction). A split is checked once the kernel exists, against
condition 8 (its loop at that depth must be parallel, with two iterations or
more) and condition 5 (its workers are the machine's, `p_M`, or half of
them, on the native backend; `--threads` or `PRESK_THREADS` set `p_M`). The
layout is part of the kernel cache's key, so two nests of one shape with two
layouts get two kernels. `make plan-check` lowers every entry of
`tests/plan/entries.txt` with and without `--plan
@tests/plan/default.json` and compares the dumps byte for byte; at 0.7.0
step 1 they were also byte-identical to 0.6.0's, in `f64`, `f32` and `bf16`.

`tests/plan/attention.json` is the language reference's example plan, for the
course GPT's attention: the scores `ein` tiled `[4, 16]` with a k-block of 32
in the order `JIK`, four lanes, packed, split at depth 1 (its heads) over ten
workers, and the softmax in two passes. It gives the default's output, bit for
bit.

```sh
presk run corpus/nlp/05_gpt.psk --entry SelfAttention.apply --dim B=16 --dim T=256 --backend native \
     --plan @tests/plan/attention.json
presk dump tests/plan/two_dots.psk --stage realize --plan @tests/plan/two_dots.json
presk dump tests/plan/two_dots.psk --stage kernel --entry twice --plan @tests/plan/two_dots.json
presk run tests/plan/two_dots.psk --entry twice --backend native --plan @tests/plan/two_dots.json
```

## Choosing a plan

A plan can be written by hand, or chosen by the compiler (0.7.0 step 5; `cli`
chapter). `presk plan` chooses **statically**. It captures each nest's
first launch from the entry's shapes, on zero inputs, without running a
kernel (since 0.8.0 step 1). It lowers every nest under every point of its
space, the thread splits included, and costs each point with the layout cost
model of a calibrated profile (`optimization` chapter), without compiling or
timing anything. It writes the plan of the best predictions. `presk autotune`
is the other side of the comparison: it compiles and times every point on the
captured inputs, checking each bit for bit against the scalar sequential
kernel. `presk rank` compares the two. Tutorial step 17 runs the three on the
tutorial's attention.

That comparison is the research programme's hypothesis RA4. 0.7.0 measured
it on the course GPT's attention (`experiments` chapter §7): its model
ordered the lanes, the thread splits and the softmax passes, and could not
see a register tile, a loop order or a k-block, because it counted no load
from the first-level cache into registers. Its pick on the attention's
contractions at batch 16 was 2.2–3.0 times slower than the best point
measured. 0.8.0's model, with that term and frozen before its test (§8),
picks a point within 1.05× of the best on every contraction nest of two
fresh sweeps. It ranks more than 90 % of the attention `ein`s' close pairs
right on the small cores, but 85–89 % at new shapes, short of RA4's
threshold. The space pays either way: its best point beats 0.6.0's kernel
by 9.2× and 16.5× on the attention's two `ein`s at batch 16, and by 27–132×
on the contractions of BERT-large's attention and of a head of 128. The
model's candidates within its uncertainty hold a point within 1.03× of the
best, so measuring them alone is a chooser too (statement Q). `presk plan` gives each nest
one layout, which holds at every shape the nest is launched at and is
checked again at each. A nest launched at several shapes gets the layout the
model prices at all of them whose predicted time, each shape's weighed by
its launches, is least; until 0.9.0 (D82) the first shape decided, and a
tile chosen there could fail to divide another shape.

## Not in this version

- **Memory placement and collectives**, the third block of the reference's
  table: every buffer is in one global memory space, and launches are
  sequential. The pack is two launches, a copy and the contraction, until
  placement has a domain and the panel becomes a scratch buffer (question 66).
- **The derivative nests ranked.** `presk plan` prices the derivatives of
  `dot` with the frozen model of 0.8.0, which has never seen a nest of their
  shapes, a summed axis of 4 096 against an output of 384 × 1 536 on the
  course GPT. Whether it ranks them as it ranks attention's `ein`s is
  statement T3 of plan 0.10.0, measured in its step 2.
- **A layout chosen with the realization.** `presk solve` chooses types and
  `presk plan` layouts, one after the other; a solver over both at once is
  for a later version.
- **Multi-output kernels.** A value read from inside *and* outside a group
  has to be materialized, which splits the group. Letting one kernel write
  several outputs would keep those chains together; it changes the kernel IR,
  its interpreter and the Cranelift backend, and it is the main reason the
  forward pass fuses as little as it does.
- **Reduction fusion.** A `mean` or `softmax` ends a group. Accumulating a
  reduction inside the producer's nest is a different and more powerful
  scheduler, and it is what would let a layer normalization become one
  kernel.
- **Extraction that knows about fusion.** With groups, `Q` is a property of
  the partition, not of a node, so choosing the cheapest term and choosing
  the partition are one problem (`optimization` chapter, question 49). The
  e-graph still extracts by the per-node roofline and the partition is
  computed afterwards, so the cost model under-counts what fusion will save.

```sh
presk dump corpus/nlp/05_gpt.psk --stage realize --const D=64 --const H=2 --const L=2
presk dump corpus/nlp/05_gpt.psk --stage kernel --entry adamw.step --const D=64 --const H=2 --const L=2 \
     --dim B=2 --dim T=32 --realize f32
```
