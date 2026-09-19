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
