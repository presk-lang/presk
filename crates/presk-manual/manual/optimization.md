# Optimization: e-graph rewriting of PSK-Core

`--opt` on `presk run`, `train`, `serve`, `bench` and on the conformance
path of `presk export` rewrites the PSK-Core program before it is executed;
`presk dump --stage core-opt` prints the rewritten program and a report.
The pass is the crate `presk-opt` (0.3.0 plan §3.5, steps 6 and 7):
equality saturation over e-graphs with the crate `egg`, rules that are
*exact over ℝ* as the specification demands, extraction by a cost — operation
counts, or the roofline of a machine profile that `presk bench --calibrate`
fits on measured kernel times — and the reference interpreter as the oracle
of every rewrite.

## What the pass does

One e-graph per function body. The `let`s of the A-normal form become nodes
whose children are the classes of their arguments, so two identical
definitions merge on entry; function parameters are leaves; calls, folds,
closures, literals and the primitives no rule mentions are opaque nodes
(their header interned, their arguments as children). Attributes a rule
needs to see — a `pow` exponent, a field name, a projection index — are leaf
children, so that patterns can bind them.

Every class carries an **analysis**: the PSK-Core type of its value (all the
nodes of a class denote one value, hence one type) and, for scalar
constants, whether it is zero or one. Types come from the `let`s for the
original nodes and are computed for the nodes the rules create; a rule fires
only when the types it needs are known and compatible, so the rewritten
program is typed like the original and the ANF written back carries a type
on every `let`.

Saturation runs under limits (30 iterations, 20 000 nodes, 2 s per
function). The cheapest term of the result's class is then extracted and
written back as fresh ANF, source names and spans preserved where a class
still holds an original definition. A definition that is not in the result's
term is not written back: dead code, including the forward recomputation
that automatic differentiation leaves in every adjoint, disappears without a
dedicated pass. Every rewritten function is verified; one that would not
verify is left as it was and reported.

## The rules

All exact over ℝ, in three families:

- **Algebra**: commutativity and associativity of `add` and `mul`; `x + 0`,
  `x − 0`, `x − x`, `x · 1`, `x · 0`, `x / 1`; `neg (neg x)`; `sub` as
  `add`/`neg` and back; `pow 1 x`, `x · x = x²`, `pow n (pow m x)`;
  `log (exp x)`; `exp a · exp b = exp (a + b)` and back; `√x · √x = x` and
  `(√x)² = x` on the domain of the left-hand side; distributivity in both
  directions (the cost decides), also through `sub`; `a · (b / c) = (a · b) /
  c`, `(a / b) / c = a / (b · c)`.
- **Reductions**: a scalar factor or a negation moves out of `sum` and
  `mean`.
- **Structure** (the adjoint primitives of AD): `sum_to x t = x` and
  `broadcast_like x t = x` when `x` and `t` have the same type; `sum_to` of
  zeros; `x + zeros_like t = x` for equal types; `field`/`set_field` chains
  (`field n (set_field n r v) = v`, `field n (set_field m r v) = field n r`,
  `field n (zeros_like t) = zeros_like (field n t)`); the sum of two
  `set_field`s on zeros into one record; `set_field` on a record, or on
  `zeros_like` of a tree whose fields are known, into a `record`; `proj` of a
  `tuple`.

The guard on every rule that builds a new elementwise operation is the pair
of operand types: equal, or a real scalar against a tensor or a parameter
tree. Under that guard broadcasting is trivial, the algebra of ℝ applies
element by element, and the axis names of the result are those of the tensor
operand in either order. Rules on mixed shapes (`x : ℝ[B, D]` with `b :
ℝ[D]`) are deliberately absent: reassociating them could change the
alignment, and the identity `sum_to x t = x` is applied only when the two
static types agree *including axis names*, because the runtime tensor of
`sum_to` carries the template's names (question 44).

Two rules were tried and removed: `x² = x · x` and moving a negation into a
product. `egg` extracts over trees and does not see sharing, so among forms
of equal cost it may pick one that duplicates a shared subterm; a power now
costs slightly less than a product (it reads one operand), and negations
stay where the program put them.

Since 0.4.0 the backends fuse elementwise chains into one loop nest
(`realize` chapter), which makes `Q` a property of a *group* rather than
of a node. Extraction does not know that yet: it costs each node on its own
and the partition is computed afterwards, so the model under-counts what
fusion saves. Making the two one decision is question 49.

**Not in this version**: the reassociation and merging of consecutive
`ein` contractions (the `(q · k) · v` decision), `merge_heads (split_heads
x) = x`, the `mean_vjp`/`dot_vjp` identities of the plan, and the
"numerics-neutral canonical forms" (softmax shift, GELU through `erf`,
`cross_entropy` as `logsumexp − logit`): the last are properties of the
kernel realization rather than PSK-Core identities, and the kernels already
compute GELU through `erf`. Without a profile the cost is *operation
counts* — structural nodes 0.01, elementwise 1, powers 0.9, transcendental
functions 2, reductions 1, contractions and loops 8, plus the children; with
one it is the roofline below.

## The oracle

Every rewrite must reproduce the interpreter's goldens, and it does
(`crates/presk-opt/tests`):

- the eight golden outputs of `tests/run` (`mlp`, `layernorm`, `minigpt`
  forward, `loss`, `next_token`, the gradient through a stacked module at 512
  bits) are byte-identical under `--opt`, axis names included;
- the twenty-step MiniGPT training log of `presk-train` is reproduced to
  its printed digits, and the regression still converges;
- a property test generates random PSK-Core expressions over two tensors
  and a scalar (sums, products, negations, `exp`, `log ∘ exp`, powers,
  divisions by `exp`, `√·√`, `sum`, `mean`, scalar factors, `sum_to`,
  `broadcast_like`), rewrites them and compares the two programs on the
  interpreter at 128 bits to 2⁻¹⁰⁰ relative; 128 cases per run;
- the native backend agrees with itself: forward pass, `f32` gradient,
  twenty native training steps and greedy serving give the same output with
  and without `--opt` (`make opt-examples`).

## What it changes, in numbers

`presk dump --stage core-opt` on the examples (tiny MiniGPT constants):

| Program | Functions rewritten | `let`s | Cost (counts) |
|---|---|---|---|
| `mlp.psk` (after AD) | 6 of 12 | 89 → 71 | 129 → 101 |
| `train_minigpt.psk` (after AD, before `prepare_training`) | 14 of 28 | 308 → 276 | 407 → 353 |

The adjoints are where the rules bite. `Linear.apply.vjp` goes from 14 to 7
`let`s: the forward `dot` and `add` recomputed by AD are dead and vanish,
`sum_to ct, %4` between equal types is the identity, and the two
`set_field … (zeros_like θ)` summed together become one record `{w = …, b =
…}` with no addition of zeros. `MLP.apply.vjp` (19 → 14) and `mse.vjp` (17 →
14) shrink the same way; the forward passes (`SelfAttention.apply`,
`LayerNorm.apply`) are already minimal for these rules and keep their size.
`adamw.step` hits the node limit (its long sums under associativity and
commutativity do not saturate) and is extracted from the partial e-graph,
cost 139 → 113.

On the tiny shapes of the examples this does not show in time (table
below): the saved operations are cheap elementwise ones and the entries are
dominated by their contractions and by host-side dispatch. The pass is the
mechanism; what it chooses for speed is the cost model's business.

## The cost model, calibrated (v0.1)

The cost-model document (§1.1) writes the time of a group of operations as a
roofline, `T̂ = λ + max(W/π, Q/β)`: `W` the work in operations, `Q` the
bytes moved, `π` the machine's sustained operation rate, `β` its sustained
bandwidth, `λ` a launch overhead — and asks for the constants to be
*learned*, not declared. Version 0.1 does exactly that for PSK-Kernel v0,
where every tensor primitive is one kernel, so the roofline applies node by
node and `Q` is additive (fusion groups, where it is not, come with a fusion
pass).

**`W` and `Q` from the shapes.** On the kernel side (`presk_kernel::cost`),
`W` is the weighted operation count of the loop nest at its concrete
shapes — arithmetic 1, a transcendental function 8, index arithmetic 0 — and
`Q` the bytes of every buffer read plus the output. On the PSK-Core side
(`presk_opt::model`), the same quantities come from the node's *type*: the
element counts of its operands and result, with the dimensions the run binds
(`--dim`, by name as the interpreter binds them) and 64 for any other
dimension variable; `dot` and `ein` count 2 operations per multiply-add over
the contracted axes, `softmax` and the cross-entropy 11 per element, and so
on; structural nodes (fields, tuples, records, calls) launch no kernel and
cost one nanosecond, enough to prefer fewer of them.

**Calibration.** `presk bench --calibrate profile.json` runs the entry
natively with an executor that times every kernel launch, drops the first
launch of each kernel (it includes its JIT compilation), keeps the median of
the others, and fits `λ`, `π`, `β` to the `(W, Q, t)` samples by least
squares on the *relative* error (a log-spaced grid over `π` and `β` refined
twice, `λ` in closed form per cell), so that the microsecond kernels of the
examples weigh as much as the millisecond ones. Calling it again on another
entry with the same file extends the samples and refits. The profile is
JSON: the constants, the machine and realization, the samples, and Spearman's
rank correlation between measured times and three predictors — operation
counts (the cost without a profile), `W` alone, and the fitted roofline. On
the development machine (Apple silicon), the three MiniGPT entries at the
tiny constants:

| Realization | Kernels | π (ops/s) | β (bytes/s) | λ | Spearman: counts | work | roofline |
|---|---|---|---|---|---|---|---|
| `f64` (forward, `loss`, `adamw.step`) | 90 | 7.3e9 | 2.3e10 | 0.05 µs | 0.15 | 0.91 | 0.95 |
| `f32` (`loss`) | 32 | 1.9e9 | 8.5e9 | 0.08 µs | 0.37 | 0.77 | 0.89 |

The plan's check holds: the profile ranks the kernels far better than
operation counts, which know nothing of shapes (0.15), and better than work
alone, which misses the memory-bound elementwise kernels. Two honest notes.
`λ` here is the cost of calling a compiled kernel, about fifty nanoseconds;
the host program's own dispatch (buffer allocation, cache lookup, the
interpretation of the Core body) is outside the model, which is why the
entries take far longer than the sum of their kernels. And `π` is fitted on
naive loop nests: it is the rate of *these* kernels, not of the machine.

**Extraction with a profile.** `--opt --profile profile.json` makes the
extraction cost the predicted seconds of each node under the profile, summed
over the term; the report then prints costs in microseconds. On the tiny
MiniGPT training program the model predicts 740 µs → 511 µs for the sum of
the rewritten functions (these are the model's units, comparable between
alternatives, not the wall time of an entry). Measured, natively, medians
of 21 calls, three runs each:

| Entry | plain | `--opt` (counts) | `--opt --profile` |
|---|---|---|---|
| `MiniGPT.apply` | 0.57–0.61 ms | 0.56–0.57 ms | 0.56 ms |
| training `loss` | 0.58–0.63 ms | 0.56–0.61 ms | 0.57–0.62 ms |
| `adamw.step` | 1.54–1.60 ms | 1.54–1.61 ms | 1.58–1.67 ms |

Every difference is inside the run-to-run spread of about ±4 %: the second
check of the plan — extraction never slows a golden entry beyond the noise —
holds, and no entry is measurably faster either. The rules of step 6 remove
cheap kernels; the decisions that would move these numbers (the `(q · k) ·
v` order, fusion of elementwise chains) need rules and a fusion pass the
model is now ready to arbitrate.

```sh
make calibrate                                               # the profile above, then the two bench lines with it
presk bench tests/examples/minigpt/train_minigpt.psk --entry loss --const V=64 --const D=16 --const H=2 --const L=2 \
     --dim B=2 --dim T=8 --calibrate out/profile.json
presk run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2 --opt --profile out/profile.json
```

**One profile, every machine type.** Since 0.6.0 step 3 a profile holds the
constants of every machine type and lane width of the machine. A narrow
type's `π` is not a wide type's — that is the reason lanes exist — and a cost
model meant to compare machine types cannot have one `π`. `--calibrate` fits
the `--realize` type as before and then times the same kernels in the other
uniform realizations, `f64`, `f32` and `bf16`, each with and without lanes,
keeping one set of constants per type and width; files written before still
load and mean what they said. On the corpus GPT: `f64` π 6.36·10⁹ ops/s,
`f32` 8.03·10⁹, `f32` in four lanes 1.59·10¹⁰, two and a half times the `f64`
figure. The precision solver prices each site with its own type's constants
(`precision` chapter, "Time, not traffic").

## The layout cost model (0.7.0 step 4)

The roofline above reads the shapes and the machine type, and no layout
decision: two register tiles or two loop orders of one nest have the same `W`
and `Q`, so it cannot rank the points of a nest's layout space (`realize`
chapter). Version 0.7.0 tests whether a static model can, which is the
research programme's hypothesis RA4. The model its plan specifies (§3.5)
reads three more things of a kernel (`presk_kernel::cost::planned_cost`),
all of **one worker's part** of it: the kernel with the loop its split
divides cut to one worker's share.

- **Traffic by level.** For each buffer, the footprint of its accesses inside
  every loop, from the affine index expressions the lowering emits, is
  compared with the capacity of each cache level. A buffer whose footprint
  inside a loop fits is brought in once per iteration of the loops enclosing
  it; otherwise the loop's body is looked at once per iteration. That gives
  `Q_L2`, `Q_L3` and `Q_mem`, the bytes from L2, L3 and memory. A taller
  register tile reads the right operand fewer times; a k-block reused inside a
  cache reads it once per block.
- **Threads.** A worker's work and private traffic are those of its share;
  the shared levels, L3 and memory, see every worker's.
- **Register pressure.** The real and vector variables live in the busiest
  innermost loop, against the machine's vector registers. The excess costs
  two level-1 accesses per iteration and per spilled register, `S`: charged
  as work in 0.7.0's model, as loads in a model with the load term. In
  four lanes a `[4, 16]` tile holds 24 registers of 32; without lanes, 84.
- **Register loads** (0.8.0 step 2). `N_ld` counts every load and store the
  worker's kernel executes (`presk_kernel::cost::kernel_loads`), once
  whatever its lanes, times the trip counts of its loops; a broadcast of a
  loaded scalar is part of its load. It is what a register tile changes: a
  tile `[m, n]` in `ℓ` lanes loads, per `k`, `m` scalars and `n/ℓ` vectors
  for `m·n/ℓ` vector multiply-adds.

`T̂ = λ + spawn·[p > 1] + C(W/π_g, (N_ld + S)/ρ_g, Q_L2/β_L2, p·Q_L3/β_L3, p·Q_mem/β_mem)`,
with `π` and `ρ` fitted per group `g`, a machine type, a lane width and a
worker count (the cores are not equal, and one rate per core would need a
schedule). `C` is the maximum, 0.7.0's; since 0.8.0 step 3 it may instead be
the sum of the in-core and the memory terms, `max(work, loads) + max(l2, l3,
mem)`, or a p-norm `(Σ tᵢ^q)^(1/q)` whose `q` is fitted (the next sections).
A profile without `ρ` is 0.7.0's model: no load term, and `(W + S)/π_g` for
the work, which gives 0.7.0's predictions to the last bit.

**The machine** (`presk_kernel::cost::Machine`) is read from the operating
system for the first CPU the process may run on, so that a run pinned to some
cores describes those cores. On falicon, pinned to the ten Cortex-X925, that
is 64 KiB of L1d and 2 MiB of L2 per core and 8 MiB of L3 shared by five of
the pinned cores. Unpinned, it would be `cpu0`, a Cortex-A725 with 512 KiB of
L2. The worker count is `--threads`, `PRESK_THREADS` or the CPUs of the
affinity mask. Since 0.8.0 step 4 it also holds the level-1 data cache's
line, 64 bytes on falicon (`line`; absent from a description written
before). The description is written into the profile (`machine_desc`),
which is where to edit it.

**Calibration** (`presk bench --calibrate --space calib`, `make
calibrate-ra4`; widened in 0.8.0 step 3, below) is out of sample by
construction. In 0.7.0 it was a GEMM at 256, 512 and 1024,
`f32` and `f64`, 96 layouts each (two tiles, every loop order with and without
a k-block of 128, both lane widths, 1, 5 and 10 workers), pinned to the X925
cores. The attention nests the model is tested on are never in it. The
constants are fitted by coordinate descent on a log grid, one at a time, on
the squared relative error, and the median absolute relative residual is the
model's uncertainty `u`. On falicon, 2026-09-23, 576 samples:

| Constant | Fitted |
|---|---|
| `λ`, `spawn` | 0 µs, 182 µs |
| `β_L2`, `β_L3`, `β_mem` | 4.4·10¹⁰, 1.8·10¹¹, 2.9·10¹¹ bytes/s |
| `π`, `f32`, one lane / four lanes, 1 worker | 1.67·10¹⁰ / 4.40·10¹⁰ ops/s |
| `π`, `f32`, four lanes, 5 / 10 workers | 3.91·10¹⁰ / 3.93·10¹⁰ ops/s per worker |
| `π`, `f64`, one lane / two lanes, 1 worker | 1.76·10¹⁰ / 2.42·10¹⁰ ops/s |
| `u` (median relative residual), Spearman | 0.185, 0.980 |

A GEMM at these sizes is bound by its work and its L2 traffic, so `β_L3` and
`β_mem` are barely identified by it (`β_mem` above `β_L3` is the sign): the
calibration set constrains the terms the GEMM exercises. In the same samples,
at 1024³ in `f32` the best layout (`[4, 16]`, k-block 128, order `KIJ`, four
lanes, split at depth 1 over ten workers) runs at 405 GFLOP/s, and 0.6.0's
default layout at 77. These are the calibration's own samples, not a test.
Whether the model ranks the layouts of the attention nests it never saw is
what 0.7.0 step 6 measured, below: it did not.

**What `u` is for.** A prediction comes with the model's uncertainty, and two
commands read it (question 69). `presk plan` picks the best prediction, a tie
broken by the order of the space, and prints how many points it predicts
within `1 + u` of that best: the points it cannot tell from its choice.
`presk rank` counts a pair of points as *decided* when their predicted ratio
lies outside `1 ± u`, and reports the accuracy on every pair the measurement
distinguishes and on the decided ones apart. Whether a chooser should decide
alone above the margin is the chooser's rule, not the model's.

## What 0.7.0's layout model could not see (0.7.0 step 6)

Tested on the course GPT's attention, which the calibration never saw (D62,
`docs/ra4-2026-09-24.md`, the `experiments` chapter's §7), 0.7.0's model
failed the hypothesis. It orders 47.9 % and 43.9 % of the two `ein`s'
distinguished close pairs right, at batch 16 in `f32`, where RA4 asks for
more than 90 %, and its pick on the attention's three contractions is 2.18,
2.87 and 3.01 times slower than the best point measured. The table per
decision, over every nest and configuration of the measurement, says which
terms work:

| Decision | Right | Tied |
|---|---|---|
| lanes | 93.2 % | 6.8 % |
| split | 96.7 % | 1.3 % |
| passes | 100 % | 0 % |
| tile | 27.7 % | 46.7 % |
| order | 30.0 % | 61.8 % |
| k-block | 0.6 % | 97.0 % |
| pack | 57.8 % | 0 % |

The first three are the decisions it has terms for: `π` per lane width, `π`
and the spawn cost per worker count, and the work of a third pass. It cannot
see the others, and the terms each point records say why. At batch 16 in `f32`
every point of both `ein`s is predicted **work-bound**, so the traffic terms
never bind. The work count gives a load no weight (`Expr::Load` weighs 0 in
`presk_kernel::cost`, 0.3.0's roofline convention), and nothing counts the
loads from L1 into registers, which is what a register tile changes. The
traffic terms do count the bytes from L2 into L1, which differ between loop
orders and k-blocks, but under the maximum they never bind. The
values `ein`'s tiles `[1, 4]`, `[1, 16]` and `[2, 8]` have the same work, the
same traffic and the same prediction, 2.27 ms, and measure 7.34, 3.39 and 4.66
ms. Where the predictions of two tiles do differ, it is by the edge work of
four-row tiles, so the model prefers the smaller tile; it is wrong on 57–63 %
of those pairs. With the traffic terms never binding, k-blocks and loop orders
cost nothing at all, and the pack is charged its copy and credited nothing for
the contiguous loads it buys.

What a register tile saves is **loads from L1 into registers per
multiply-add**, and no term counts them. The calibration had seen it too: its
two tiles, `[1, 8]` and `[4, 16]`, share a group and have the same work, and
they differ by a median of 1.26× — a quarter of the 288 pairs by more than
1.85× — which is part of what `u` = 0.185 is. A term for those loads is the
design `docs/kernels-roadmap.md` recorded for a later version. 0.8.0 added
it (next section) and tested it on fresh sweeps, since a model designed
after seeing where this one failed would otherwise be fitted to its test.

Two things hold. **The model is a good pruner**: on every contraction nest at
the attention's shapes, the points it predicts within `1 + u` of its best
hold the best measured point or one within 1.08× of it (at batch 16 in `f32`
and on the GEMM, the best itself, among 36–201 candidates of 1 333–1 717
points, 2.4–15 % of the space), so measuring its candidates alone finds the
best at a fraction of the sweep's cost. And it is **cheap, but not cheap
enough** for statement N: choosing statically costs 582–717 times less wall
time than measuring every point at batch 16, and 75–89 times less at batch 1,
where N asks for 10³, because a point's measurement is eight launches of a few
milliseconds against about 0.2 ms to lower and cost it.

## The load term (0.8.0 step 2)

Version 0.8.0 re-tests RA4 with the model changed, designed on 0.7.0's
measurements, which are its development set, and frozen before two fresh
sweeps decide (`experiments` chapter; `presk-specs/implementation/presk-plan-0.8.0.md`).
Step 2 adds the load term above:

- `N_ld/ρ_g` joins the maximum. `ρ` is fitted per group with the other
  constants (`presk bench --calibrate P.json --space calib --load-term
  group`, the default), once for the machine (`--load-term machine`), when
  the calibration cannot tell the groups' rates apart, or not at all (`none`,
  0.7.0's model).
- Every point `presk autotune` and `presk reprice` record carries `loads`
  and `spill` in its planned cost, and a `loads` term when the model has
  one.
- A calibration into an existing profile drops that profile's samples from
  before this step, which have no load count and hold their spilled
  registers inside their work.

0.7.0's calibration had two register tiles, too few to identify `ρ`. Step 3
widens it, and chooses how the terms combine: the maximum, a sum of the
in-core and memory terms, or a fitted p-norm.

On the development set the count reads the tiles as designed. The values
`ein` at batch 16 in `f32`, four lanes, split at depth 2 over ten workers,
has 90 points. Spearman between one worker's `N_ld` and the measured time is
0.953, where 0.7.0's prediction reaches 0.437. Among the thirty that model
predicted equal, 2.27 ms, it is 0.943. Those are the measurements the term
was designed on, so they show that it is built as designed, not that it
ranks: the fresh sweeps are the test.

## The combination and the wider calibration (0.8.0 step 3)

**How the terms combine** (`--combine`):
- The maximum, 0.7.0's form, assumes a core that overlaps everything, so
  that only the slowest resource counts.
- The sum of the in-core and memory terms assumes a core that does not
  overlap its arithmetic with its misses.
- The p-norm lies between them, from `q = 1`, the sum of all five terms, to
  the maximum as `q` grows.

Each is fitted on the same samples (`presk refit --compare`). A p-norm's `q`
is searched on its own, the rates fitted again at every `q` tried, because
it trades off against all of them at once.

**The calibration** (`make calibrate-ra4`) now times 300 layouts per shape
and type:
- five register tiles, `[1, 4]` to `[4, 16]`;
- ten loop orders and k-blocks;
- both lane widths;
- one, five and ten workers.

The shapes are the GEMM at 256 to 2048 and two thin products, `16 × 4096 ×
4096` and `4096 × 4096 × 16`. On falicon's X925 cores it gave 3 536 samples:

| load term | max | sum | p-norm |
|---|---|---|---|
| none (0.7.0's) | u 0.226 | 0.226 | 0.224 (q = 1.00) |
| per group | **0.133** | 0.137 | 0.136 (q = 1.00) |
| one for the machine | 0.149 | 0.163 | 0.167 (q = 1.00) |

The load term is what improves the fit. The three combinations fit the
calibration about equally, and the p-norm settles at the sum. The form is
chosen on the development set (step 4). Two things this calibration does not
do:

- **It does not identify `ρ` at one lane.** There the work term binds on
  every sample, though the scalar tiles differ by up to 1.75×: under the
  maximum they tie, and a sum or a p-norm would see them.
- **It does not identify the memory constant.** `β_mem` comes out above
  `β_L3`, which cannot be. The thin products meant to be bound by memory
  are not on this machine: ten cores multiply sixteen rows faster than
  memory feeds them. A product of two rows is not bound by memory either:
  it reads memory at 6–8 GB/s on one worker and 32–38 GB/s on ten, of
  about 273. The kernels are bound by their loads and loop overhead long
  before memory, so no product identifies `β_mem` here. The thin products
  are kept, and `β_mem` is recorded as not identifiable on this machine
  (D67, the user's decision).

It took 3 h 13 min. About a third was the kernels' own time; the rest was
converting the exact inputs at every call, now done once. At 2048 the
one-worker layouts took three quarters of the kernels' time, and since 0.8.0
step 4 they are left out there (`--min-workers 2`, the user's decision, D67):
one worker is calibrated at 256 to 1024 and on the thin products.

**The definitive calibration** (0.8.0 step 4, D68), the space as amended and
a profile started empty, took 44.9 min on falicon's X925 cores, 1 329 s of
it in the timed kernels, and gave 3 336 samples. With the load term per
group, `u` is 0.142 under the maximum, 0.141 under the sum and 0.143 under
the p-norm (`q` = 1.06); without it, 0.241. Timed again, the same layouts
differ from step 3's by a median of 2.3 %. It is archived read-only beside
the development set, and every form step 4 compares is a `presk refit` of
it.

**The form chosen** (0.8.0 step 4, D69) on the development set, 0.7.0's
ten-core days priced under each form (`make ra4-ablation`): a load rate per
group under the p-norm. On the two attention `ein`s at batch 16 in `f32`,
0.7.0's model ranked 47.9 % and 43.9 % of the distinguished close pairs
right. The load term alone reaches 55.2 % and 60.3 %, and the p-norm with
it 84.6 % and 84.4 %, with ties down from about 40 % to 3 %. Near `q` = 1
the p-norm adds the work and the loads, which is what tells the scalar
tiles apart; the two-term sum does not. These are development figures,
and the fresh sweeps decided (the last section).

**Rules of the planned cost** (0.8.0 step 4, 4c). Three rules refine what
the model reads of a kernel, each named by a profile (`rules` in its layout
model) and off by default, so that a profile naming none prices every
kernel as before, bit for bit:

- `lines`: a footprint counts whole cache lines of the machine's line size.
  Along the contiguous axis its span is rounded up to the line, so that an
  access whose loops do not walk that axis brings a line per element.
- `shared`: a loop is resident when the footprints of all its buffers,
  summed, fit the cache, where 0.7.0 asks it of each buffer apart.
- `overhead`: each loop iteration weighs one operation of work, its
  increment, compare and branch.

Under `lines` or `shared`, a loop that does not fit while one of its
iterations does brings once what consecutive iterations share: the lines
they share along the contiguous axis, or a buffer the loop does not move.
Without that, the first rule would charge a line per element to every
stream, and the second would evict the output's tile at every step of `k`.
A rule is tried on the definitive calibration without measuring it again:
`presk refit --recost` costs every sample again under it, its time kept
(`cli` chapter).

**The rules kept** (0.8.0 step 4, D70). Each was tried on the definitive
calibration, re-costed, and refitted in the chosen form, then on the
development set. A rule had to lower the calibration's `u` and not lower M
on either attention `ein`.

| rules | `u` | scores `ein` | values `ein` |
|---|---|---|---|
| none | 0.143 | 84.6 % | 84.4 % |
| `lines` | 0.141 | 84.8 % | 84.5 % |
| `shared` | 0.143 (up) | 85.0 % | 87.9 % |
| `overhead` | 0.134 | 88.8 % | 89.2 % |
| `lines,overhead` | 0.132 | 89.3 % | 89.6 % |

`overhead` tells tiles and k-blocks apart: tiles go from 83 % to 89 % right,
k-blocks from 49 % to 66 %. `lines` adds little on these nests, whose rows
are whole lines. `shared` fails in every combination and worsens the pick
on both `ein`s. The user kept `lines` and `overhead`. No rule sees the pack:
its body costs the same with and without it, and what it gains is how each
iteration's loads fall, one line against one per row of the tile.

**The frozen model** (0.8.0 step 4, D71): a load rate per group under the
p-norm (`q` = 1.06), the rules `lines` and `overhead`, the calibration of
D68 re-costed under them (`u` = 0.132). Under those rules the nine forms
were fitted again, and this one still leads. Its profile is archived
read-only in `~/presk-data/ra4-model-0.8.0`, and fresh sweep F1 is priced by
it. On the development set it ranks 89.3 % and 89.6 % of the two attention
`ein`s' distinguished pairs right, with O at 1.07, 1.00 and 1.01. These are
development figures, statement D of 0.8.0, which decides nothing. `make
calibrate-ra4` calibrates in this form and under these rules by default,
and so does `make calibrate-ra4-small` on the A725 cores; since 0.8.0 step 8
they are also the defaults of `presk bench --calibrate --space calib` and
`presk refit` (D75).

## What the fresh sweeps found (0.8.0 steps 5–7)

The frozen model was tested on measurements taken after the freeze (D72–D74,
`experiments` chapter §8): **F1**, new shapes on the ten X925 cores,
priced by the frozen profile itself (BERT-large's attention, F1a, and a
head of 128, F1b); and **F2**, 0.7.0's shapes on the ten A725 cores, priced
by a calibration of those cores in the frozen form and space.

**The A725's calibration** (`make calibrate-ra4-small`, D72): 3 336
samples in 101 minutes, `u` = 0.168 where the X925's is 0.132, and `q` =
1.00, a plain sum. The frozen form fits best there too of the nine. The
constants that are identified are what a smaller core predicts: `π` at one
lane 5.3–6.3 Gop/s a worker, against 12.2–12.9 on the X925, and `ρ` 7.1–10.6
G loads/s at one lane, which the X925's calibration could not identify
(49–80, unphysical). Not identified there: `π` in four lanes in `f32` on
five and ten workers, where loads make most of every sample's time, and
`β_mem`, as on the X925.

**What the model ranks.** Over every nest of each sweep, by decision:

| Decision | F1 (X925) | F2 (A725) | development set |
|---|---|---|---|
| tile | 90.7 % | 92.4 % | 89.1 % |
| split | 99.1 % | 98.1 % | 97.7 % |
| lanes | 100 % | 98.7 % | 100 % |
| order | 60.6 % | 72.6 % | 67.0 % |
| k-block | 64.3 % | 53.4 % | 65.9 % |
| pack | 47.9 % | 62.9 % | 57.8 % |
| every pair | 87.6 % of 76 830 | 90.4 % of 134 022 | 88.5 % of 137 307 |

Almost no pair is tied, and the pairs the model decides, predicted apart by
more than `u`, are 99.2 % right on F1 and 98.8 % on F2. On statement M's
nests, the two attention `ein`s, F2 ranks 92.7 % and 91.7 % of the
distinguished pairs right, above the 90 % RA4 asks. F1 ranks 87.4 % and
88.5 % at F1a and 85.4 % and 88.8 % at F1b, below it: the model carried
over to another core and fell short on new shapes. The development figure,
89.3 % and 89.6 %, foresaw F1 within four points.

**Where it is wrong.** 97–98 % of the misses on F1, and 99 % on F2, are
pairs the model predicts within `1 + u` of each other: M is decided by how
it orders what it predicts to be close. Within `u` it is right on 69–73 % of
the pairs on F1 and 78–81 % on F2, and those pairs are 39–46 % of F1's. The
misses are tiles (about 10 % wrong, on the largest counts), loop orders, the
pack and k-blocks, whose pairs almost all lie within `u`. One class of
decided pairs fails: the loop orders of one-row tiles at F1b's scores
`ein`, 67 wrong of 121. The order that reads the packed operand once a row
of the output (`I` outside `J`) is predicted 15–23 % slower, for its L2
term, and measures 1–16 % faster. Under a p-norm with `q` = 1.06, nearly a
sum, the L2 time is added to the load time, and a one-row kernel does not
pay it on top of its loads. The packed operand of one head is 256 KiB at F1b
and 96 KiB at F1a, against 64 KiB of L1; at F2's shapes it is 64 KiB, and
every decided order pair there is right.

**What holds.** The pick: within 1.05× of the best point on every
contraction nest M names (statement O). And the hybrid chooser 0.7.0's data
suggested, statement Q: the candidates within `1 + u` of the model's best,
5 to 28 points of a nest, hold a point within 1.03× of the best measured on
all eighteen contraction nests of both sweeps, at most 1.8 % of the
exhaustive sweep's time, the static side included. The static choice itself
is 1 013–2 933 times cheaper than sweeping, but only 218–252 times at batch
one on the A725 (statement N): there costing 3 305 points, 0.16–0.22 ms each,
is most of the static side.

What the failures point to is recorded in `docs/kernels-roadmap.md`: how the
L2 term should combine with the load term where the packed operand exceeds
L1, how the model orders pairs within `u`, and the cost of costing a point.

## Moving a scalar across a contraction

The rule the plan of this version named — reassociating `(q·k)·v` into
`q·(k·v)` — turned out to have nowhere to fire. Counted on the corpus GPT
after differentiation, the number of contractions whose operand is itself a
contraction is **zero**: a transformer puts a softmax, a GELU or a
normalization between every pair, and a nonlinearity blocks the
reassociation. That is a fact about the architecture, not about this
compiler, and it is why the rule is not in the set.

The rule that does fire moves a *scalar* across a contraction:

```text
(Σ aᵢ bᵢ) / s  =  Σ (aᵢ / s) bᵢ
```

exact over ℝ, and cheaper whenever an operand holds fewer elements than the
result. Attention is exactly that case. Its scores are divided by `√hd`:

| Tensor | Elements at `B=16, H=6, T=256, hd=64` |
|---|---|
| the scores, `ℝ[batch, heads, seq, seq2]` | 6 291 456 |
| the queries, `ℝ[batch, heads, seq, hd]` | 1 572 864 |

so dividing the queries instead of the scores does a quarter of the work.
The e-graph proposes scaling *every* operand and lets the cost decide.

This is the kind of choice a cost model exists for, because it is a question
about sizes and nothing else. The calibrated roofline prices the two forms
apart — `SelfAttention.apply` at 10 205 µs against 9 489 µs — and prefers the
moved one for a reason. Operation counts price them the *same*, one division
either way, and land on the same form only as a tie-break.

Measured, on the corpus GPT at `D=64`, forty training steps, twelve threads:

| | seconds a step | extraction |
|---|---|---|
| no rewriting | 0.710 | — |
| `--opt` (operation counts) | 0.626 | 0.41 s |
| `--opt --profile` (roofline) | 0.617 | 2.54 s |

The rewriting is worth about 12 %, and the losses of all three runs are
identical to every digit printed. Two honest observations about the profile.
Its *predictions* are much the better: on this program the rank correlation
with measured kernel times is 0.96 against 0.59 for counts. Its *decisions*
are barely better, 0.617 against 0.626 a step, because the rule set offers
few choices where size is what separates the alternatives. And its extraction
costs six times more, since the roofline recomputes each node's work and
traffic from the types; at this rate it takes some 230 steps to repay itself.
A cost model can rank kernels well and still not change which program you
extract.

## Limits, reported

The plan proposed to leave a function unrewritten when saturation hits a
limit. Extraction from a partially saturated e-graph is sound — every node
in it is an exact rewrite of the original — so the pass extracts anyway and
reports it: the summary line of `--opt` counts the functions that stopped on
a limit, and `dump --stage core-opt` names them with their stop reason and
iteration count (decision D24).

```sh
presk dump tests/examples/mlp/mlp.psk --stage core-opt        # program on stdout, report on stderr
presk run tests/examples/mlp/mlp.psk --entry mse_grad --dim D=4 --dim B=2 --opt
presk train tests/examples/minigpt/train_minigpt.psk --const V=64 --const D=16 --const H=2 --const L=2 \
     --steps 20 --backend native --opt
make opt-examples                                            # the comparisons above
```
