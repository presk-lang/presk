# Architecture decisions

One paragraph per decision, newest last. The plans these come from are
`presk-specs/implementation/presk-plan-0.1.0.md` through
`presk-plan-0.6.0.md`.

## D1 — Scope of 0.1.0 (2026-09-18)

Version 0.1.0 is the front-end (`presk check`), the arbitrary-precision
reference interpreter with automatic differentiation (`presk run`) and the
readable PyTorch export with a generated conformance test (`presk export`).
E-graph rewriting, cost models, code generation, PSK-Sched/PSK-Kernel, the
serialized formats, the Zig runtime and Lean certification come later. The
roadmap's Phase 1 was too large for a first tag.

## D2 — Rust workspace of six crates, Makefile on top of cargo

`presk-syntax ← presk-types ← presk-core ← presk-interp ← presk-cli`, plus
`presk-export-pytorch` depending on `presk-types` only. `presk_core::Program`
is the seam towards the future optimizer and code generator. A `Makefile`
orchestrates cargo (the spec's rule: cargo for the compiler, never a foreign
build system driving it); `just` was considered and dropped because it is not
installed on the development machines while `make` is.

## D3 — Examples copied from presk-specs

`tests/examples/` is a copy of `presk-specs/examples/` (see
`tests/examples/PROVENANCE.md`), refreshed by `make sync-examples`. Copied
rather than symlinked so the repository is self-contained and the relative
`import` paths keep working.

## D4 — Presburger fragment solved in-house

Shape arithmetic (linear forms with rational coefficients, divisibility by
literals, bounds) is decided by a small solver in `presk-types`; no z3. The
fragment is tiny and an external solver would add a C++ dependency and a
translation layer for error messages.

## D5 — Arbitrary precision through a `Real` trait, `rug` as the implementation

The interpreter is generic over a `Real` trait; the 0.1.0 implementation is
`rug::Float` (MPFR), the spec's choice, which also provides `erf` for the
exact GELU. If building `gmp-mpfr-sys` proves a problem on some machine, a
pure-Rust implementation (`astro-float`) can be added behind the same trait.

## D6 — Code-generation backend deferred

Cranelift versus LLVM (`inkwell`) is decided in 0.2 once `presk-opt` exists
and extracted programs can be inspected; 0.1.0 only keeps PSK-Core
backend-neutral.

## D7 — Hand-written lexer and recursive-descent/Pratt parser, newline layout

No parser generator: the Unicode rules (XID identifiers with combining marks,
`·` excluded from identifiers, `ℝ`/`λ` as keywords), the layout rule for
`let` without `in` (an expression ends at a newline unless a binary operator
that cannot start an expression follows; newlines are insignificant inside
`(`, `[` and dimension `{…}`), and the source-located diagnostics are all
easier to get right by hand. Diagnostics are a renderer-independent
`Diagnostic` value rendered with `ariadne` (ASCII, no color in snapshots).

## D8 — ASCII spellings of the Unicode symbols (2026-09-18)

The five non-ASCII symbols of the syntax have ASCII synonyms accepted
everywhere by the lexer: `ℝ` = `Real`, `λ` = `\`, `·` = `@`, `⊗` = `><`,
`∘` = `<<`. Both forms produce the same token, so the AST, the diagnostics
(which print the Unicode form) and a future formatter are unaffected. The
conventional spellings (NumPy's `@`, Haskell's `\`, F#'s `<<`) were preferred
to keyword forms (`dot`, `fun`) and to TeX escapes (`\cdot`). Added to the
specification as the "Notation" paragraph of the spec chapter.

## D9 — The Presburger fragment: rewrite, inspect, search (2026-09-19)

`presk_types::presburger::entails` decides `facts ⊢ goal` over ℕ in three
stages: equality facts become rewriting rules and `k | v` facts become
substitutions `v := k·v'`; the rewritten goal is checked by inspection (an
equality's difference is identically zero; `a ≤ b` has a non-negative
difference, possibly after substituting one declared upper bound; `k | e`
divides every coefficient); otherwise a bounded enumeration of the goal's
variables (and those connected to them through facts) looks for a
counterexample, and the answer is `Unknown` if none is found. `Valid` is
sound by construction and property-tested against random assignments;
`Unknown` is reported as an error ("cannot be proved"), never as a pass.
Constants are evaluated to numbers before any of this, so the symbolic
variables are only `dim` declarations and type parameters, which keeps the
enumeration small.

## D10 — Shape typing by generic checking and per-call instantiation (2026-09-19)

Each function is checked once with its type parameters, its file's `dim`s and
its `...` prefix rigid; a call instantiates the callee's signature with fresh
inference dimensions and a fresh *flexible* prefix (`PrefixVar`), which
unification binds to the argument's leading axes. Dimension equalities are
solved for inference variables when one is present and otherwise decided by
`presburger::entails`. Module references are structural (`ModRef::Inst { def,
args }`) so that substitution reaches instance arguments (`params net` inside
`mse{D}` must become `params MLP{D', 4*D'}` at the call). Broadcasting aligns
from the trailing axis and falls back to alignment by name. The `ein`
relabeling rule renames a name shared by both operands in the right operand
when exactly one `<name><digits>` copy of it is unmatched. Types are kept in
side tables keyed by expression span (no second syntax tree).

## D11 — PSK-Core is dimension-polymorphic with explicit instantiation at calls (2026-09-19)

Core functions keep the symbolic dimensions of their signatures and every
`Call` carries `dim_args`, the callee's dimension variables mapped to the
caller's expressions (recorded by the checker per call site). The
alternative, monomorphizing per instance, would duplicate `Linear.apply` for
every `Linear{…}` and expand shapes early; keeping the IR generic mirrors the
source and lets the interpreter bind dimensions from `--dim` at the entry
point. Axes are designated by name in primitives because runtime tensors
carry their axis names. Parameter trees are explicit (`θ` first parameter,
`θ.w` fields); `stack{L}` is unrolled with `slice_params`; lambdas are lifted
with their free variables first (`closure`/`grad`/`apply`), which is the shape
AD will rewrite. The plan's reserved `Fold` node is not in the IR yet; it will
be added when a loop form replaces unrolling (0.2).

## D12 — Regularity as per-parameter dataflow on PSK-Core with witnesses (2026-09-19)

Classes are computed on the IR rather than on the syntax tree because Core is
first-order with explicit primitives and explicit θ. Each variable carries,
per input parameter, the worst class along any path and the primitive that
raised it (the *witness*), so `grad` errors can point at `argmax` inside a
callee three calls away. Function summaries are memoized (Core is
non-recursive); function values (`closure`) keep the dependencies of their
captured variables so that `grad` checks only the lambda's own parameters.
Effects are a plain set inferred bottom-up and compared with the `uses`
clause. Per-parameter granularity is what makes `loss : C∞ in θ, Disc in
batch` expressible (spec question 11).

## D13 — Reverse-mode AD as VJP functions with structural adjoints (2026-09-19)

`grad` is eliminated by generating, for every function on the differentiated
path, `f.vjp(params…, ct)` returning the tuple of parameter adjoints (forward
copy + reverse sweep with accumulation), and for every differentiated lambda
`f.grad(captured…, θ) = f.vjp(…, 1).θ`; `grad v` becomes a closure over
`f.grad`. Adjoints of parameter trees are built with `zeros_like` /
`set_field` / `set_elem` and combined with a structural `add`, which keeps the
IR small (no per-type adjoint representation) at the cost of some zero
traffic that a reference interpreter can afford. Adjoint primitives
(`sum_to`, `broadcast_like`, `mean_vjp`, `dot_vjp_l/r`, `embed_vjp`,
`cross_entropy_vjp`, `scatter_*`, `gelu'`, `heaviside`, `sign`) are ordinary
PrimOps so that the interpreter, and later the optimizer, treat adjoint code
like any other; their own derivatives are left undefined (`E6002`), which
bounds 0.1.0 to first order. VJPs recompute the forward pass (no tape), the
simplest correct choice for an interpreter.

## D14 — Interpreter over named-axis tensors with MPFR values (2026-09-19)

Runtime tensors carry axis names re-tagged from the type of each binding, so
the primitives designate axes by name and the evaluator never reconstructs
positions from the checker's tables. Dimensions are runtime values bound per
call from `dim_args`, which is what lets Core stay dimension-polymorphic
(D11). Numbers are `rug::Float` behind a `Real` trait (D5); the evaluator
materializes broadcasts and contractions with plain loops — a reference
interpreter optimizes for being obviously right, not fast (the tiny MiniGPT
forward pass takes half a second at 256 bits). Randomness is a tree of
ChaCha20 keys derived from `--seed`, one per entry parameter, split per field
in initializers; `sample` uses a separate seeded stream because the `Rand`
effect carries no explicit key in the source of 0.1.0. `--const` overrides
apply to imported files too, since a model's constants live in its own file
while `loss`/`next_token` live in the training/serving files.

## D15 — PyTorch export as a printer over the typed program (2026-09-19)

The export reads the typed syntax tree and the checker's side tables (types
per expression, resolved callees, axis positions, `ein` relabeling), not
PSK-Core: the specification wants code isomorphic to the source, and Core has
already made θ explicit, unrolled `stack` and lifted lambdas. Two semantic
gaps between Presk and PyTorch are bridged mechanically: file-level
dimensions in parameter shapes become constructor arguments (PyTorch needs
concrete shapes at construction), and Presk's prefix/by-name broadcasting is
turned into PyTorch's trailing broadcasting with `unsqueeze`. The conformance
test is generated from the interpreter's run (`inputs.json`,
`expected.json`) and compares in float64 with a tight tolerance; PyTorch is
never a dependency of the compiler, so the test is byte-compiled here and run
where PyTorch exists.

## D16 — `.psk.d`: samples of one Presk type, `f64` leaves, no windowing in the language (2026-09-19)

A data file is a sequence of samples of one closed Presk type written in the
header in Presk syntax; leaves are `f64` (injected exactly into ℝ, the
hyperparameter rule) and `idx` at the width of their bound. Checkpoints and
weights are one-sample files of type `params M`, whose layout comes from the
program. The language-modelling shift between inputs and targets is a
data-preparation step (`presk data lm-windows`), so that `batch(B)` has one
meaning for every objective: stack `B` samples under a `batch` axis and bind
`B`. Alternatives considered: NumPy `.npy` or safetensors (would tie the
language to a foreign type system and offer no place for `idx` bounds or
tuples); a windowing operator in the pipeline (would make `stream` depend on
the objective's shape). The format is versioned and confined to one crate so
that the specification's answer (question 28) stays a local change.

## D17 — The optimizer is PSK-Core, generated per (block, model), PyTorch's AdamW (2026-09-19)

`optimizer adamw { … }` is lowered to `adamw.init`, `adamw.lr` and
`adamw.step`, ordinary Core functions over `params M` written with the
language's primitives (`relu` for the piecewise schedules, `sum_squares` for
the global norm), so that the interpreter, the export and later code
generation treat the update like the model. The update rule is PyTorch's
AdamW (decoupled decay, bias correction, `eps` outside the root, clipping
with the `1e-6` of `clip_grad_norm_`), because the conformance test needs a
fixed reference and PyTorch is the one users compare with. The block's name
selects the algorithm (the spec writes `optimizer adamw`, `optimizer = adamw`).
Alternatives considered: an interpreter-side optimizer (would not be exported
or compiled, and would duplicate tree arithmetic), a `Select` primitive for
the schedules (not needed: `min`/`max` of the piecewise forms are `relu`).

## D18 — `fold` as the loop of PSK-Core; scan-and-backward-fold differentiation (2026-09-19)

`stack{L}(M).apply` lowers to `Rhs::Fold { body, n, stacked, init, extra }`
rather than `L` unrolled calls: the IR's size no longer depends on `L`, `L`
may be symbolic, and code generation will emit a loop. The construct is kept
minimal — a fold over the leading axis of stacked values with an
accumulator — plus three flags needed by differentiation: `scan` (also
return the accumulators before each iteration), `reverse`, and `index`
(pass `i` to the body). The reverse rule is the classical one: the forward
copy scans, the backward fold runs down from `L − 1`, its generated body
applies the body's `vjp`, writes the stacked cotangent into slice `i`
(`set_slice`, a dynamic-index primitive) and sums the extras' cotangents.
Alternatives considered: recomputing the accumulators in the backward pass
(quadratic, no new construct) and a general `scan` returning per-iteration
outputs (more general than needed). Initializers of stacked modules are
still unrolled (`L` keys, `stack_params`), a `map` form to be added when a
symbolic `L` is actually used.

## D19 — The training loop sequences PSK-Core; nothing trains inside the interpreter (2026-09-19)

`presk train` is a small crate that calls, through the interpreter, the
model's initializer, the objective and its `vjp` (generated on demand by the
same differentiation as `grad`), and the PSK-Core optimizer, over the
pipeline of `presk-data`. It converts `.psk.d` samples into batch values
along the `batch` axis of the objective's type and parameter trees into
checkpoints, and nothing else: no update rule, no schedule, no gradient
code lives in the loop, so the export and later code generation reproduce
training by reproducing PSK-Core. The objective is resolved by its origin
(the elaborated function), not by name, since imports may define a function
of the same name. Two calls per step (the loss, then the `vjp`) recompute the
forward pass; acceptable for a reference, to be shared when speed matters.

## D20 — Training in the export: `train.py` with PyTorch's shuffling, conformance by replaying batches; graph mode through PyTorch (2026-09-19)

`train.py` is a real PyTorch training script over the exported objective and
model: `DataLoader` shuffling, `torch.optim.AdamW`, the schedule as a
multiplier, `clip_grad_norm_`, `torch.save` checkpoints. It does not
reproduce the pipeline's ChaCha permutation (that would mean a ChaCha in
Python for no user benefit); instead the K-step conformance test records
the interpreter's batch indices and `θ₀` and replays them, so the two loops
see the same data and the comparison isolates the arithmetic. Graph mode
follows the decision on question 32: the `.pt2` and `.onnx` files come from
PyTorch's own exporters run on the exported module loaded with the
conformance parameters, in float32; a Presk-native graph waits for solved
dtypes. Alternatives considered: a `.psk.d` writer in `train.py` for
checkpoints interoperable with `presk` (deferred to `inference serve`), and a
float64 ONNX check (ONNX Runtime's double coverage is uneven; float32 with
`rtol 1e-4` is the meaningful test of a float32 deployment graph).

## D21 — One `Backend` seam; exact values cross it untouched (2026-09-19)

Every executor of PSK-Core (the reference interpreter today, the kernel-IR
interpreter and the Cranelift JIT next) implements `presk_rt::Backend`:
`prepare(entry, dims, realize) -> Prepared` then `call(&Prepared, args) ->
Value`. The runtime `Value` is a tree (tuples, records) of tensors whose
buffers are `f64`, `f32`, `i64` — or *exact* MPFR reals, so that the
interpreter behind the trait loses nothing between two calls: the training
loop written over the trait reproduces the 0.2.0 goldens digit for digit,
and the interpreter stays the oracle of the other backends rather than an
approximation of itself. The training loop, `presk run`, `presk serve` and
the conformance suite are written once against the trait; input synthesis
stays with the interpreter (`prepare_run`), since synthesizing inputs is a
reference concern. Alternatives considered: a generic `Value<R>` (would push
the arithmetic type into every consumer), and converting the interpreter's
results to `f64` at the seam (would silently change the reference's
semantics across steps).

## D22 — PSK-Kernel v0: one kernel per primitive at concrete shapes; the host program is the Core body (2026-09-19)

A kernel is the loop nest of one PSK-Core tensor primitive at the concrete
shapes of its operands and one machine type, built when first launched and
cached by its signature. The host program is not a second IR: the PSK-Core
body is evaluated over runtime values (calls, folds, tuples, records,
structural primitives, initializers and sampling on the host with the
reference's ChaCha keys), and every tensor primitive is a launch. This gives
a complete native-arithmetic realization of every program with one lowering
table, keeps the interpreter's semantics rule for rule (the same alignment,
contraction and adjoint definitions), and specializes on shapes that only
become known at run time (open prefixes). What it defers: fusion across
primitives, tiling, a static host program and the versioned `.pskk`, which
belong with PSK-Sched. Alternatives considered: a full host IR mirroring
Core (duplication without benefit before scheduling exists) and static
shape inference for all kernels (impossible through `...` prefixes without
the caller's shapes).

## D23 — Cranelift JIT in process; the kernel backend's host program unchanged (2026-09-19)

The native backend is the kernel backend with its launches compiled: a
`KernelExec` seam lets the host program run kernels through the kernel-IR
interpreter or through Cranelift-compiled functions, so the two realizations
share every semantic decision and differ only in how a loop nest executes.
Kernels are compiled on first use, in process, with buffers passed as an
array of pointers; `sqrt` is native and the transcendental functions are
`libm` calls, the same definitions as the interpreter's. No object files,
bundles or cross-compilation yet: those belong with the `.pskk`/`.pskb`
contract (spec question 36). Cranelift 0.131 is pinned because Cranelift's
minimum Rust version tracks stable closely and 0.135 needs rustc 1.95.
Alternatives considered: LLVM through `inkwell` (better vectorization, an
LLVM installation on every machine, slower builds) and emitting C to compile
with `zig cc` (cross-compilation for free, but a foreign toolchain in the
inner loop of a JIT).

## D24 — E-graph rewriting per function body, typed classes, extraction even under a limit (2026-09-19)

`presk-opt` builds one e-graph (crate `egg`) per PSK-Core function body
and writes the extracted term back in ANF; calls, folds and closures are
opaque nodes, so the pass is intraprocedural and needs no inlining decision.
Every class carries the PSK-Core type of its value as an `egg` analysis:
rules that build a new elementwise operation fire only between equal types
or against a real scalar, so no rule has to reason about broadcasting or
axis alignment, and the identity `sum_to x t = x` requires equal axis names
because the runtime tensor of `sum_to` carries the template's names
(question 44). The rules are exact over ℝ; the interpreter's goldens and a
property test are the oracle. Extraction is by operation counts until the
calibrated model of step 7, over trees (`egg`'s extractor does not see
sharing), which excluded rules that only move an operation around. The plan
proposed to leave a function untouched when saturation hits a limit;
extraction from a partial e-graph is sound, so the pass extracts and
reports the stop reason instead. Alternatives considered: `egglog`
(relational rules; not needed for a few dozen patterns), a hand-written
peephole pass (no equational reasoning, order-dependent), interprocedural
e-graphs with inlining (later, with the cost model).

## D25 — Cost model v0.1: per-kernel roofline with constants fitted on relative error (2026-09-19)

The extraction cost of `presk-opt` follows the cost-model document's
roofline `λ + max(W/π, Q/β)` node by node — legitimate because PSK-Kernel v0
runs one kernel per primitive, so traffic is additive; fusion groups will
change that with a fusion pass. `W` and `Q` are computed twice from the same
definitions: from the kernel IR at concrete shapes for calibration, from the
PSK-Core types (dimensions bound by name from the run, 64 by default) for
extraction, so that the profile fitted on kernels applies to the terms the
e-graph proposes. The constants are fitted by least squares on the
*relative* error over a refined log grid, `λ` in closed form: kernel times
span three orders of magnitude and an absolute fit would see only the
largest. Rank correlation (Spearman) is the reported quality, the roadmap's
metric for cost models; on the examples it moves from 0.15 (operation
counts) to 0.95. `λ` is the measured cost of calling a compiled kernel,
about 50 ns; the host program's dispatch is not modelled, and the model's
totals are not wall-time predictions of an entry. Alternatives considered:
a linear additive model `o + W/π + Q/β` (fits as well on these samples, but
is not the roofline the specification writes); per-primitive overheads (too
many parameters for ninety samples); measuring whole entries rather than
kernels (no way to attribute time to nodes).

## D26 — The NLP corpus lives in the compiler repository (2026-09-20)

`presk/corpus/nlp/` holds the programs of the NLP course, not
`presk-specs/examples/`. The specification's examples are the syntax source of
truth and the acceptance programs of the language; the corpus is benchmark
material driven by a course, it changes with every measurement, and it carries
data. `data/prepare.py` downloads the course files and derives the datasets;
only the 71 KB word list is committed, the 1.2 MB of Code civil and every
derived file are regenerated. `EXPECTED.md` records what each rung produced
and on which date, so a number in the manual can always be traced to a run.

## D27 — MLPerf as specification and foil, not as a submission target (2026-09-20)

Of MLCommons' training and inference suites, only BERT-large inference is
within reach of this compiler, and it was retired from recent rounds;
everything else needs convolution, sparse embeddings, NMS, collectives or a
GPU backend, none of which touch the thesis. What is taken instead: the model
architectures as an expressiveness test (`presk check` on transcriptions,
which is the roadmap's own "coverage of the shape fragment on a corpus of
models"), the methodology of stating accuracy as a fraction of an FP32
reference, and MLPerf's closed division as the *foil* — it admits any
realization whose task accuracy stays within a hand-negotiated per-benchmark
threshold, verified by a full validation run, and cannot tell model fragility
from a buggy fused kernel. Presk's counter-proposal is a computed distance to
the exact denotation, attributable per kernel, known before the dataset run.
That argument needs one benchmark done properly with an exact reference, not
thirty done at vendor quality.

## D28 — `merge` joins adjacent axes only, and names nothing (2026-09-20)

The general axis-flattening operation the specification lacks (question 45) is
`merge(axis a, axis b, x)`, with three deliberate restrictions. The axes must
already be adjacent and in order: joining distant axes would also transpose,
and `merge` is defined as a *retag* — row-major buffers make it metadata-only,
so it launches no kernel on any backend and costs nothing in the model. At
least one length must be constant, because the merged length is their product
and `DimExpr` is a linear form. And the joined axis is unnamed, named by the
signature it is passed to, exactly as `embed`'s and `one_hot`'s are; inventing
`a_b` would put a name in the type that no declaration wrote. The adjoint is
the retag back, `reshape_like(ct, x)`, which is its own adjoint. Alternatives
considered: `merge(axis a, x)` joining `a` with its successor (shorter, but it
silently merges whatever follows, which is the class of error this language
exists to catch); a general `reshape` to an arbitrary shape (loses the
adjacency guarantee and the metadata-only property); making
`split_heads`/`merge_heads` sugar for split/merge, which does not work because
they also permute.

## D29 — Dropout's key is in the signature, and its mask is a value (2026-09-20)

`dropout(rate, key, x)` takes a key, and lowers to `x · dropout_mask(rate,
key, x)`. Both halves of that are forced by differentiation rather than
chosen for style. Reverse-mode AD here recomputes the forward pass inside
the generated `vjp`, so a mask drawn from a hidden RNG stream — the obvious
implementation, and what PyTorch does — would be *different* in the forward
and in the backward, and the gradient would be wrong with nothing to signal
it. A mask that is a function of the key alone recomputes identically, which
makes the scheme correct by construction; putting it in the signature makes
that visible in the type, and `split_key(key) -> (key, key)` lets a function
drop more than once. Making the mask an ordinary value, rather than a
primitive with its own adjoint, means `dropout` needs no differentiation rule
at all: the adjoint of a product already multiplies the cotangent by the
mask. The test that matters is finite differences *while dropping*, which
would fail under any of the rejected designs. Outside a training run the mask
is all ones, which is not a special case but the expectation of the random
function `dropout` denotes, so `presk run`, `presk serve` and the export give
the model's mean; the seam carries that as `Backend::set_training`, off by
default and turned on only by `presk train`. Alternatives considered: a
hidden stream (wrong gradients); a key derived at run time from a call-site
id and a step counter (correct only if every recomputation reproduces the
same dynamic path, which a fold makes fragile, and invisible in the type);
storing the mask from the forward pass instead of recomputing it (a change to
the whole AD strategy for one primitive).

## D30 — Fusion groups are structural, elementwise, and single-output (2026-09-20)

PSK-Sched v0 partitions a PSK-Core body once, on the body alone: the
partition does not depend on the shapes, the realization or the backend, so
it is computed per function and cached, and the kernel backend and the
Cranelift backend necessarily agree. A group is elementwise and
single-output: a reduction ends it, and a value read from outside the group
stays materialized. Both restrictions are what keep the lowering a single
loop nest with one store, needing no change to the kernel IR — multi-output
kernels would touch the IR, its interpreter and the JIT, and are what the
forward pass would need to fuse well. Interior values are duplicated in the
expression tree rather than stored when read twice inside, bounded at
sixty-four nodes, which trades arithmetic for traffic in the direction the
roofline says is right. Fusion reassociates nothing, so every golden output
is byte-identical and the pass needs no tolerance of its own. Groups over
parameter trees map leaf by leaf, which is where the measured gain is: the
optimizer's chain is elementwise over every parameter. Alternatives
considered: fusing at run time by looking ahead in the body (the same
decision taken repeatedly, invisible to `dump`); fusing in the e-graph as a
rewrite (the partition is not a term rewrite, and the cost of a group is not
the sum of its parts); requiring single use rather than all-uses-inside (loses
the chains where a value is read twice within one group).

## D31 — Threads split an exact axis; vectorizing a sum does not (2026-09-20)

The native backend parallelizes a kernel by splitting the range of its
outermost loop, and only when that loop is *exact* to split: the body must be
a single loop nest and every store must index the output with that loop's
variable directly, so the iterations write disjoint elements. A reduction over
the outer axis and a scatter whose index is a loaded value both fail the test
and stay on one thread. Under that condition each output element is computed
by the same inner loops in the same order, so the result is bit-identical to
the sequential one and the conformance suite needs no new tolerance — the test
asserts equality of bits, not closeness. Compiled kernels therefore take the
loop's range as two parameters, and one compiled function serves the
sequential call and every worker. Only kernels above four million weighted
operations are split, because spawning costs tens of microseconds; a
persistent pool would remove that floor and is the obvious refinement.

Vectorization stops here on purpose. The kernels that vectorize exactly are
elementwise, and they are not where the time is; the time is in the
contractions, whose innermost loop is a sum, and vectorizing a sum computes
one partial per lane and adds them at the end. That reassociates the sum and
changes the result. In this compiler a changed result is a different
*realization*, which must declare its reduction order and carry a measured gap
to the exact reference, exactly as `f32` does. Shipping it as an invisible
backend detail would contradict the thesis the tool exists to demonstrate, so
it waits for that treatment rather than for the code.

## D32 — Contractions are blocked, and blocking is exact (2026-09-20)

`dot` and `matmul` are lowered with eight columns of the result held in
scalar accumulators across the whole contraction, so the inner loop reads one
element of the left operand and eight contiguous elements of the right and
writes nothing; widths that are not a multiple of eight take a row-wise form
that sweeps a row of the result. The naive nest this replaces walked a column
of the right operand for every output element, one element per cache line.

The point of both forms is that they are *exact*. Each output element still
accumulates the contraction index in increasing order, so the additions happen
in the same order as before; what changes is where the partial sums live and
which memory the loop touches. That is the line this compiler draws through
the usual GEMM toolbox: loop reordering and register blocking keep it, while
vectorizing the inner sum, splitting the contraction across threads, and
multiple partial accumulators summed at the end do not — those reassociate,
and a reassociated sum is a different realization owing a declared order and a
measured gap (D31). The test therefore asserts bit equality between the two
forms rather than closeness, and the conformance table against the reference
is unchanged to every digit. Alternatives considered: a packed panel layout
(faster still, but it copies operands, which is a scheduling decision that
belongs above the kernel); calling a vendor BLAS (fast, but the tool would
stop being able to say what its own kernels compute).

## D33 — The rewrite worth having moves scalars, not contractions (2026-09-20)

The e-graph rule set gains `(Σ aᵢbᵢ) · s = Σ (aᵢ · s) bᵢ` for `dot`, `matmul`
and `ein`, proposing every operand as the one to scale and leaving the choice
to the cost. It replaces the reassociation of adjacent contractions that the
0.4.0 plan named, which was dropped after counting: the corpus GPT, after
differentiation, contains *no* contraction whose operand is another
contraction, because a transformer separates every pair with a softmax, a
GELU or a normalization, and a nonlinearity blocks associativity. Shipping a
rule that no program in the corpus can use would have been untestable
decoration.

The scalar rule is exact over ℝ and not over the floats — it changes the
order in which a scale is applied — which is the licence this compiler has
and a float-first one does not: the rewrite is legal against the denotation,
and the realization gap is measured rather than assumed. It is also the
first rule whose *right answer depends on sizes*, which makes it the case the
calibrated profile exists to decide; the measurement of what that is worth,
including the finding that the profile's decisions are barely better than
operation counts' while its extraction is six times dearer, is in the
`optimization` chapter. Alternatives considered: `merge_heads(split_heads x)`
(a pure win with no decision to make, so it teaches the cost model nothing);
distributing a contraction over a sum (legal, but it duplicates the
contraction, which the tree-based extractor prices wrongly).


## D34 — The realization is an object, not a flag (2026-09-20)

`presk_rt::Realize` becomes a record: a storage type and an accumulator type
per fusion group, a declared reduction order, and a reference to the
magnitude envelope any bound about it assumed. It is serialized as JSON
beside the calibrated profile, `--realize f64|f32` stays as sugar for a
uniform assignment with a sequential order, and `--realize @FILE` and
`--realize solved` are the new forms; `solved` reads the budget from the
program's own `precision` configuration block, which the checker has accepted
since 0.1.0 and ignored ever since.

The change is forced from two sides at once, and that is the argument for
making it now rather than as part of a larger precision effort. D31 and D32
of 0.4.0 each refused a transformation — vectorizing a contraction's inner
sum, and calling a vendor BLAS — on the ground that a reassociated sum is a
different realization owing a declared order and a measured gap, and each
then had nowhere to write the declaration down. `docs/kernels-roadmap.md`
records the same blockage from the performance side: the fast kernels cannot
ship until the contract exists. Meanwhile the precision solver needs exactly
this record as its output. One object unblocks both.

The granularity is the fusion group (D30) rather than the individual value,
because a group is one loop nest with one set of buffers, so it is the
smallest unit at which a type is a decision rather than a conversion; it also
puts the choice at PSK-Sched, where the specification puts it. Storage `bf16`
is admitted with `f32` arithmetic and round-to-nearest-even on store, which
is the standard mixed-precision idiom and is what gives the solver a third
rung with a real traffic win on the memory-bound sweeps; `f16` is not, its
dynamic range being a separate problem. Alternatives considered: a per-value
assignment (too fine to lower, and it prices conversions it cannot see); a
per-module annotation in the source (it would put a machine type in a
language that has none, which is the position the project exists to test).

## D35 — A bound is conditional on a measured envelope, and says so (2026-09-20)

The first-order error analysis of plan 0.5.0 §3.2 needs a bound on the
magnitude of every intermediate. Static interval arithmetic cannot supply one
for a neural network — weights are unbounded, and intervals grow with depth
until they say nothing — so the magnitudes are measured instead: the
reference interpreter runs on a stated number of calibration batches and
records `max |·|` per PSK-Core value into an envelope file carrying its date
and its sample count.

Every bound the tool computes is therefore conditional on an envelope, and
must be printed with it. This is the version's main exposure, and the answer
is to make the conditionality part of the output rather than a caveat in a
paper: a bound without its envelope is not a weaker claim, it is a different
and unsupported one. Step 3 measures the envelope's stability between two
disjoint sample sets rather than assuming it, because that stability is the
assumption the whole analysis rests on and it is cheap to test.

*Measured (2026-09-20, corpus GPT at reduced constants, `make precision`):*
disjoint slices of the civil code agree to **1.241×** and far-apart ones to
**1.128×**, over 77 compared sites of 101, with 24 identically zero in both
and none present in only one. So the magnitudes are a property of the program
within about a quarter. The decision stands, now with a number attached to
what "conditional" costs.

The specification already proposes this shape for quantisation, calibration
being "an input of the bound rather than of the graph" (RA3). What is new
here is applying it to the deterministic bound as well, and admitting the
consequence: presk guarantees the distance between an execution and the
denotation *over an observed input domain*, not over all inputs. Spec
question 53 asks whether the language should let a program declare that
domain. Alternatives considered: symbolic interval propagation (unusable, as
above); a probabilistic envelope from moments rather than maxima (a real
option, and the specification admits statistical error models as a
realization of the same framework — deferred, because a maximum is checkable
against the conformance suite and a quantile is not).

## D36 — `split` names its halves, and splits chain where merges do not (2026-09-20)

`split(axis a, n1: L1, n2: L2, x)` reads one axis as two, row-major
(`k ↦ (k / L2, k % L2)`), so it is exactly the inverse of `merge` and, like
it, a pure retag that moves no data and launches no kernel. `L1 · L2` must
equal `|a|`, and at least one must be constant, because `DimExpr` is linear —
the same restriction `merge` carries for the same reason.

Unlike `merge`, **the split names both halves**, and the asymmetry is not an
oversight. D28 left a merged axis unnamed because neither input name
describes the product and `a_b` would be arbitrary. A split has the opposite
problem: the two halves are genuinely new axes that the program must be able
to refer to, and only the author knows what they mean — `groups` and `per`,
or `rot` and `pair`. The consequence, found while transcribing Llama, is that
**two splits chain and two merges do not**: `split` produces named axes that
the next `split` can select, while `merge` produces an unnamed one that no
second `merge` can name. Grouped-query attention needs a three-way reading of
one axis and so needs the chaining.

The larger finding is that the split, not `split_heads`, is what expresses a
head layout. `split_heads` is hard-wired to `const H` and `const HD` and also
permutes, so it cannot say "eight key heads shared by thirty-two query
heads" at all. The split can, and needs no transpose to do it, because `ein`
contracts by *name*: the group axis of the queries meets the group axis of
the keys wherever each happens to sit in its operand. `coverage/llama.psk`
now transcribes grouped-query attention exactly and uses neither head
primitive.

## D37 — Per-layer randomness rides in the fold's accumulator (2026-09-20)

A `stack{L}` module lowers to a `Fold`, and a fold hands every iteration the
same extra arguments. A key passed as an extra is therefore one key for all
`L` blocks: every layer drops the same positions, which is wrong and silent.
Question 48 left this open and proposed deriving a key from the loop index.

The answer taken is that **no new construct is needed**: the key is carried in
the accumulator. The accumulator is already threaded from one iteration to
the next, which is exactly the dependency a key chain needs, so a block whose
type is `(Acts, key) -> (Acts, key)` splits its key, uses one half and passes
the other on. `tests/data/stack_keys.psk` measures the difference — four
independent masks at rate ½ zero about fifteen sixteenths of a tensor, one
reused mask about half — so a fold that quietly shared a key fails the test.

The index route was implemented far enough to see that it is worse. `Fold`
already carries an `index` flag, but AD refuses outright to differentiate an
indexed fold (`E6001`), so per-layer keys by index would not have survived
`grad` — and a dropout scheme that cannot be trained through is not a dropout
scheme. The accumulator route differentiates today, through the existing
fold adjoint, with nothing added. An indexed `split_key(key, i)` was written
and then removed for the same reason D33 gives: no program needed it once the
accumulator worked, and a builtin no program can use is decoration.

## D38 — `bf16` is storage, and the conversion is one named function (2026-09-20)

The third rung of the dtype menu is `bf16`, and it is a **storage** type
only: buffers hold sixteen bits, the loop nest is generated in `f32`, and
conversion happens on load and on store. That is the standard mixed-precision
idiom rather than an invention, it needs no arithmetic this compiler does not
have, and it is what makes a narrower type *worth* anything — the memory-bound
sweeps, of which `adamw.step` is a dozen over every parameter, halve their
traffic while the contraction kernels keep their accuracy.

`f16` is not admitted. Its dynamic range is a separate problem needing loss
scaling, which is a training technique rather than a realization, and the
solver would have to model overflow rather than rounding.

Widening is exact, because `bf16` is a prefix of `f32`: a shift. Narrowing is
one round to nearest even, written once in `presk_rt::f32_to_bf16` and emitted
again in Cranelift as the same integer sequence, NaN case included. That
duplication is deliberate and is pinned by a test: the kernel interpreter and
the JIT must store the same bits, or the conformance suite would be comparing
two different programs and calling the difference a realization gap. They
agree digit for digit on a `bf16` gradient today.

Measured worst gap to the exact reference over the conformance entries:
`f64` 2.6e-14, `f32` 5.3e-6, `bf16` 3.3e-2. The `bf16` band needs an absolute
floor of `2⁻⁵` to mean anything, because a result near a cancellation has
unbounded *relative* error at any precision and that says nothing about the
kernel — which is itself a small lesson about what a bound on a ratio can
promise, and the reason step 4's rules carry a magnitude alongside an error.

## D39 — The bound is first-order and relative where it must be, and it is vacuous on a transformer (2026-09-20)

The propagation of plan 0.5.0 §3.2 carries a pair at every value — a ceiling
on `|v|` from the measured envelope and a ceiling on `|executed − exact|` —
and computes the second by first-order rules, dropping products of errors as
this literature does. Three choices in it are worth recording, and the third
is a result rather than a choice.

**Quotients, roots and logarithms propagate in *relative* terms.** The
absolute form of the quotient rule, `da/b + a·db/b²`, is pointwise sound
given a floor for `b`, and it is useless: it pairs the largest numerator with
the smallest divisor, a pairing that never occurs in a tensor. Written that
way, five layer norms in a row put an `f64` transformer's bound at 10³⁰. In
relative terms a quotient simply adds, `d(a/b)/(a/b) = da/a − db/b`, and a
normaliser costs nothing — which is the conditioning the specification asks
to exploit, and it is not a special case for `LayerNorm` but the quotient
rule written in the right variable. Where both forms apply the analysis takes
the tighter; products and powers keep the absolute form, which never divides.

**A mean does not multiply error by `k`.** The sum of `k` errors is `k·e` and
the division by `k` takes it straight back to `e`. Treating a mean like a sum
cost a factor of the batch size before it was noticed. Sums and contractions
do multiply by `k`, and that is worst-case correct: `|Σ wᵢ δxᵢ| ≤ k·max|w|·
max|δx|`.

**On a real transformer the bound is vacuous, and that is the finding.** It
is sound everywhere it can be checked — on every conformance entry, in every
realization, the propagated bound exceeds the measured gap, which a test
asserts. But the ratio between them has a median of 1.8·10³ on small entries
and reaches 10²⁰ on the corpus GPT, where four contractions in series carry
`k⁴` while the errors they commit grow like `√k`, being independent. A bound
of 10²⁰ relative on a loss of 4.5 says nothing.

This is the specification's own risk register, arriving on schedule: "worst-
case bounds are pessimistic by several orders of magnitude". Two consequences
are taken now rather than discovered later. The tightening that would close
the gap is named and is not vague — each sum needs its **condition number**,
`Σ|xᵢwᵢ| / |Σxᵢwᵢ|`, measured into the envelope beside the magnitudes, which
replaces the worst-case `k` with what the program actually does. And step 6's
solver cannot verify a candidate assignment against a bound that is vacuous;
it will have to either carry that tightening first or accept against the
measured gap, and the plan's propose/verify/tighten loop is now known to
depend on it. Step 8's measured curve was already the version's primary
deliverable for exactly this reason.

## D40 — Sensitivities are read off the adjoints a reverse sweep already computes (2026-09-20)

A reverse sweep computes `∂L/∂v` for every intermediate on the way to the
parameters — that is what a reverse sweep *is* — and then discards all of
them but the parameters'. The sensitivity analysis of plan 0.5.0 §3.3 keeps
the correspondence instead: `presk_core::ad::AdjointMap` records, for each
forward value, which variable of the generated `.vjp` holds its adjoint, and
running that vjp once with the 1-norm recorder gives `‖∂L/∂v‖₁` at every
site. No second differentiation, no perturbation per site, and no new
numerical machinery.

Two things this makes explicit, and both are in the code's own words. A
sensitivity **ranks** and a bound **guarantees**; the solver proposes with
the first and verifies with the second, and code that mixes them is wrong.
And a ranking taken at initialization is **vacuous**: the corpus GPT's output
head is zero-initialized, as Karpathy's is, so nothing upstream of it
receives any gradient and every module scores exactly zero. `--weights`
starts from trained parameters, and the same caveat governs the magnitude
envelope.

**Half the mixed-precision folklore is reproduced and half is not.** On the
trained corpus GPT the output head dominates at 5.86, twelve times the whole
block stack and twenty times anything else: keeping it wide is what the
derivative says, not a convention. The token embedding table scores 0.28,
*below* the down-projection that practice is happy to narrow. So the habit of
keeping embeddings in high precision is not explained by the loss's
sensitivity to them; if it is right it is right for another reason — sparse
rows, dynamic range, optimizer state — and that is a claim this measurement
can make and the folklore cannot. It is also the first evidence that a solver
allocating by sensitivity will not simply rediscover current practice.

## D41 — The solver verifies by measurement, not by the bound (2026-09-20)

The plan had the solver *propose* with the sensitivity and *verify* with the
propagated bound. Step 4 established that the bound, though sound, is vacuous
on a real transformer — 10²⁰ relative on a loss of 4.5 (D39) — so verifying
against it would accept every assignment and mean nothing. Question 56 put
three options to the specification; this is the one taken in practice.

**A candidate realization is run and its distance to the exact reference is
measured.** That is a stronger check than a bound, not a weaker one: it is
the error actually committed rather than an envelope around it. It is also
something only this compiler can do, because only this compiler keeps an
exact denotation to measure against — which makes the acceptance rule an
argument for the project's position rather than a retreat from it. The
propagated bound is still computed and reported beside the measurement,
labelled as the weaker artefact.

Three further choices, each with a reason that was measured rather than
assumed.

**The objective is traffic, not time.** Bytes moved is what a narrower type
reduces. Time is not: step 2 measured `bf16` *slower* than `f32` on a
compute-bound kernel, the conversion being pure overhead there. Pricing time
per site needs the calibrated roofline's `W` beside its `Q`, and that is named
work rather than assumed away.

**An unknown sensitivity means leave the site alone.** Reading a missing
measurement as zero makes a site look maximally *safe* to narrow, which is
exactly backwards. With the absence read as zero the solver narrowed
everything, measurement rejected it, and a hundred and three widening rounds
recovered a 1.06× saving; with the absence read as unknown it proposes a
mixed assignment accepted on the first measurement, at 2.67×.

**Tightening widens one step of the menu, not straight to the widest.**
Jumping to the top undoes in one round everything the proposal found.

Measured. On the corpus GPT at `D = 64` with trained weights and a 10⁻³
budget: 2.67× less traffic than all-`f64`, accepted with **zero** widening
rounds, measured gap 1.05·10⁻⁷ against a budget of 4.5·10⁻³, and the solved
realization runs natively to 4.521788597 where `f64` gives 4.521788577. On a
program small enough to search exhaustively the greedy proposal is **exactly
optimal** at a 10⁻³ budget and 1.057× off at 10⁻⁶.

## D42 — Vectorizing across output columns is exact; the inner sum is what reassociates (2026-09-20)

D31 of 0.4.0 refused vectorization on the ground that "vectorizing a
contraction's inner sum reassociates it". That is true and it is not the only
way to vectorize a contraction, which step 7 established while plumbing the
declared order through the kernel lowering.

The blocked form of 0.4.0 already holds `GEMM_TILE = 8` accumulators, and
they are **eight different output columns**, not eight partial sums of one.
Each still accumulates `k` in increasing order. Putting those eight
accumulators in vector lanes is therefore **bit-exact**: the lanes are
independent columns, the inner sum's order is untouched, and no new
realization is owed. The register blocking that 0.4.0 chose for cache
behaviour turns out to be the shape a vector kernel wants anyway.

So the factor the `kernels-roadmap` note expects from vectorization does not
need the realization contract after all — which is a better position than the
plan assumed, not a worse one. The contract is still needed for the matrix
coprocessor and for the GPU, where a reduction genuinely crosses lanes.

**What is landed and what is not.** `Stmt::SetL` now assigns a *loop*
variable in PSK-Kernel, in both executors, which is what an unrolled index
`cb·n + t` needs; on it, `contraction_ordered` emits the interleaved
contraction, and the declared order is part of the kernel cache key, so two
realizations differing only in their order can never share a compiled loop
nest.

**The order reaches the kernel and reassociates**, which a test asserts: on
the embedding MLP at `D = 32` in `f32`, a sequential order gives
1.5494239330292 and `Interleaved(8)` gives 1.5494240522385, and the kernel
interpreter and the JIT agree with each other to the last digit under each.
A declared order the kernel ignored would be the worst outcome this design
could have, so that inequality is the assertion worth having.

An intermediate diagnosis in this step was **wrong and is recorded because
it cost time**: the two orders first appeared bit-identical, and the reason
looked like the order not reaching the fused-group path. It is not. Fusion
groups are elementwise (D30) and contain no reduction at all. The programs
being compared simply did not *depend* on a long contraction: a
zero-initialized head makes the logits identically zero and the training loss
exactly `ln V`, whatever happens upstream — the third time that zero head has
changed how something must be measured, after the magnitude envelope and the
sensitivity ranking.

**Measured, and the measurement settles the step.** `presk bench
--native-only` on a 1024³ matmul, twelve cores, medians of three:

| realization | median ms |
|---|---|
| `f64` | 102.4 |
| `f32` | 87.3 |
| `bf16` | 96.9 |
| `f32`, `Interleaved(8)` | **239.0** |

The reassociating order is **2.7× slower**, and the reason is structural
rather than incidental: an interleaved order cannot use the blocked form,
whose eight accumulators are eight output columns, so it falls back to the
naive per-element contraction and loses the cache behaviour that was the
whole gain of 0.4.0 step 6. Breaking the accumulator's dependency chain buys
less than blocking does, by a wide margin.

So the conclusion of this step is a **negative result, and a useful one**:
for a CPU contraction there is no reason to reassociate at all. The exact
blocked form is both faster and exact, and — by the finding above — it is
also the shape vector lanes want. The `Interleaved` order stays implemented,
declared and measured, because the coprocessor and the GPU will need it and
because a contract one cannot exercise is not a contract; it does not ship
as anything's default. `bf16` slower than `f32` again (96.9 against 87.3)
repeats step 2's measurement on a second kernel.

What is still not landed is the vector lanes, which need a vector form in the
kernel IR. **Statement G, as the plan worded it, is about a vectorized kernel
that does not exist yet and stays unanswered**; what this step can answer, it
has.


## D43 — Statement G is retired; the claim to make about a vector kernel is bit-identity (2026-09-20)

Plan 0.5.0 §3.7 asked, as statement G, that "the vectorized GEMM's measured
gap falls inside the bound its declared `Interleaved(n)` order predicts". It
is withdrawn rather than carried forward, and the reason is a finding of the
step that was meant to answer it.

G assumes the vector kernel reassociates. D42 established that it need not:
the blocked contraction's `GEMM_TILE = 8` accumulators are eight different
output *columns*, each still summing `k` in index order, so putting them in
lanes touches no addition's position. The kernel is exact, its realization is
`Blocked`, and it has no gap for G to measure. Answering G as written would
have meant either building a worse kernel on purpose or reporting a tolerance
that passes vacuously.

Statement **H** of plan 0.6.0 replaces it: the vector kernel is
**bit-identical** to the scalar blocked one, in `f32` and in `f64`. That is
the stronger claim — an equality rather than a tolerance — and it is the one
the design actually supports.

The general lesson is worth keeping separately from the arithmetic. A
falsifiable statement fixed in advance is what these plans are for, and it
still has to be allowed to be the *wrong question*; retiring one with its
reason recorded is not the same as quietly failing it. 0.5.0 reported G as
"not askable" rather than claiming it held, which it trivially would have.

## D44 — Quantisation is weight storage; the scheme is the realization, the scales are data (2026-09-21)

Plan 0.6.0 step 5 adds `int8` and `int4`, and its draft took the scales from
the magnitude envelope. For activations that is right, since an activation's
range is only known from data. For weights it is wrong: the envelope keeps
one maximum per site, not one per channel, and a weight's scale is a function
of the parameter tensor, which is at hand. So the step is fixed before coding
as follows.

**Weights only, as storage.** A quantised parameter is integers plus one
scale per channel, dequantised to `f32` on load; every contraction runs in
`f32` in the Blocked form. This is D38 with a scale — no new arithmetic, no
new reduction order, and every kernel shape of 0.4.0–0.6.0 still applies.
Quantising activations too (W8A8) needs activation scales, `i32`
accumulation and requantisation, and Cranelift's 128-bit integer dot products
are thin; it is after 0.6.0.

**Symmetric absmax per channel along a named axis**, the output axis of the
contraction the parameter feeds: `s_c = max_k |w_kc| / (2^(b−1) − 1)`, round
to nearest. It cannot saturate, since `|w/s| ≤ 2^(b−1) − 1` by construction.
An all-zero channel — the zero-initialized head again — gets `s = 0` and
stores zeros rather than dividing by it. A weight scale needs no calibration
batch, so a weight-only realization is **not** conditional on an envelope
(D35), which is a stronger claim than RA3's "calibration as an input of the
bound"; the envelope becomes the source only when activations are quantised.

**The scheme is the realization; the scales are data** (question 57). The
realization gains a `quant` map keyed by parameter, `{ bits, axis, scheme }`,
beside `storage` and `accum`, which stay keyed by fusion group (D34): a
quantised weight is a property of one tensor feeding a contraction, not of a
loop nest. Scales are not written into the realization file — they are as
many as a layer's channels and change at every training step — but computed
from (realization, checkpoint) deterministically at load; a served model may
cache them beside the checkpoint, keyed by its hash.

**The denotation is dequantise-on-load**, and the kernel interpreter and the
JIT must load the same bits, pinned by a test as `bf16` is. Applying the
scale once after the sum rounds differently and would owe a gap, except with
**power-of-two scales**, where the multiplication is exact: `scheme = pow2`
is offered as an option, and with it the two placements are asserted
bit-identical — the same kind of claim as statement H, and a priced
alternative (a scale up to twice too coarse) to the default.

**`int4` is per channel, reported and not promised.** Per-channel `int4` is
known to be poor, and the practice — groups along the *contracted* axis — has
scales that do not factor out of the sum. Shipping it per channel with its
gap measured is the honest first point; group scales are the named
follow-up.

**The gap is measured in two parts**: the exact denotation on the
dequantised weights against the exact denotation on the true ones (the
representation error), and the quantised run against the former (the
arithmetic error). Only a compiler with an exact reference can separate
them. Statement K is measured on the forward loss with trained weights,
because gradients and optimizer state are never quantised here and an
untrained zero head makes the loss `ln V` whatever the weights are.

*Amended at step 5a (2026-09-21).* "Per channel along a named axis — the
output axis" was wrong twice. The realization names the axis the parameter
is **contracted** over (`over`), and there is one scale per index of every
other axis: a stacked or multi-head parameter has more than one non-contracted
axis, and "every other axis is a channel" is the only reading under which the
scale still factors out of the sum. And the axis cannot be required to be
named, because module parameters are written without names (`param w :
ℝ[In, Out]`); it is a name or a position counted from the end, which survives
the leading axis `stack{L}` adds. A name two axes share is refused.

Alternatives considered: scales in the realization file (unbounded in size
and stale after one step); an `Int8` member of `Dtype` (it would claim a
fusion group *computes* in `int8`, which nothing does); scales from the
envelope for weights (a measured approximation of a number that can be
computed exactly); W8A8 now (a different kernel family, for a speedup this
instruction set would not show).

## D45 — Integer weights are a buffer, and a kernel input is marked, not lowered (2026-09-21)

Step 5b puts D44's integers into the kernels. Three choices, each the
smallest that keeps the step's one promise: that running on the integers is
**bit-identical** to running on their dequantised values, which 5a's
reference path already measured.

**The integers are a `Buf`.** `Buf::Quant` holds the integers and the scales
with their geometry, shared rather than copied, and anything that does not
know about it — the reference interpreter, `to_f64`, a host operation —
reads it as its values `f32(q)·s`. So the reference path needs no special
case, and a buffer that reaches code which cannot use the integers still
computes the right numbers. A stacked matrix's slice stays quantised,
because its scales are constant along an axis other than the stacked one;
without that, `stack{L}` would have handed every block `f64` weights.

**The lowering does not know.** It emits ordinary real loads, and the host
marks an input that receives integers, `BufKind::Quant { bits, over }`, in
the kernel's spec and its cache key. Each executor loads a marked input as
`f32(q)·s`, one `f32` multiplication, widened exactly in an `f64` kernel.
No kernel shape changed, and a kernel written after this one gets integer
weights for nothing. The alternative, a quantised operand kind through the
lowering, would have touched every primitive for a property of one buffer.

**Four lanes at once.** The JIT loads four `int8`, sign-extends them twice,
converts and multiplies by four contiguous scales. The first version
converted lane by lane with `insertlane`, which is the same arithmetic and
made the GPT forward 1.6× *slower* than `f32`; the four-lane load brought it
level. The arithmetic per lane is the scalar load's, so statement H's
discipline holds: a test runs both schemes in `f32` and `f64`, scalar and in
lanes, on both executors, against the dequantised path, and asserts equality.

Measured, lanes on: a 64 × 1024 × 4096 contraction 47.6 → 10.7 ms (4.4×),
the GPT forward at `D = 256` 28.6 → 26.0 ms, at `D = 64` level. The integers
pay where the kernel streams weights from memory and not where they fit in
cache — the regime distinction 0.5.0 learnt from `bf16`, now with the sign
the right way round. `int4` is read as values until step 5c.

## D46 — `presk bench` times the step, not the loading of its inputs (2026-09-21)

Found while timing 5b. `bench` takes its inputs from the reference
interpreter, as exact MPFR reals, and each native backend rounded them to its
machine type **on every call, inside the timing**. The integer rows quantised
their weights once, outside it, and came out 7–13× faster on a single
contraction — a number that was obviously not the arithmetic. It was the
conversion: `f32` on a 64 × 256 × 1024 contraction went from 13.0 to 1.3 ms
once the inputs were rounded outside the loop.

Every row now rounds its inputs, and quantises its weights, once, before it
is timed. Rounding and quantising are loading a checkpoint; a step is what a
benchmark of a step should measure.

The consequence for the record is uncomfortable and is written down rather
than smoothed over. Every `bench` figure before this decision carries the
conversion as a constant on both sides, so its **ratios are diluted**: on this
machine it was most of a 1024³ matmul (`f64` 183 → 68 ms, `f32` 143 → 50).
Statement I's 1.15–1.25× from lanes and statement J's 1.06× were both
measured that way, on apremont, and cannot be re-measured from here; J is
re-measured at course scale in step 7 in any case. Calibrated profiles are
not affected: they time each kernel, and the conversion happened before the
first one. Measured fairly on falicon (aarch64, 20 cores), lanes buy nothing
on a threaded 1024³ matmul and `bf16` is the fastest row — at that size, on
this machine, the kernel waits for memory.

## D47 — `int4` is unpacked by a multiplication, four lanes at a time (2026-09-21)

Step 5c. The kernel interpreter already read nibbles through `QBuf::value`,
so the step is the JIT and the host. The scalar load takes byte `e/2`,
shifts the nibble to the top of an `i32` and back arithmetically, which
sign-extends four bits. The four-lane load cannot use 5b's widening sequence,
because four nibbles may start in the middle of a byte: it reads the 32-bit
word at byte `e/2`, shifts it right by `4·(e mod 2)`, splats it, and
multiplies by `[2²⁸, 2²⁴, 2²⁰, 2¹⁶]` — a per-lane left shift, which vector
shifts in Cranelift cannot express because their amount is one scalar — then
shifts right by 28. The word can reach two bytes past the last element, so an
`int4` buffer carries four zero bytes of padding; the alternative, a branch on
the parity, would have split the loop body for a property of the address.

Lane-by-lane conversion was tried first, as in 5b, and made the GPT forward
2.3× slower than `f32`. With the four-lane load: the 16 MB contraction 52.9
→ 10.1 ms, level with `int8` (10.8); the GPT forward level with `f32` at
`D = 256` and 0.82× at `D = 64`. Halving the bytes again buys nothing once
they already fit in cache, which is the same regime argument as D45 one rung
further down.

The bit-identity test covers `int4` in both schemes, on both executors,
scalar and in lanes, and on a program of odd widths whose every other row
starts mid-byte — the one case where the parity shift does anything, and so
the one a test on even widths would have passed without exercising.

## D48 — A checkpoint is exactly the size of its layout, or it is refused (2026-09-21)

Found at the start of step 5d, and it reaches back into 0.5.0. A `params`
file records its model's *name*, not its leaves' shapes, so the reader takes
the layout from the program at the dimensions it is opened with — and
checked only that the file was not *shorter* than that layout. The corpus
GPT has `pos : ℝ[T, D]`, so `T` is part of its parameters: `out/gpt_tiny` was
trained at `T = 256` (the data's window) and opened at `T = 32` by every
command that measured on "trained weights". The file was longer than the
`T = 32` layout, the check passed, `tok` and the first 32 rows of `pos` were
read correctly, and **every later leaf was read from shifted offsets**. The
head, zero-initialised and barely moved after 12 steps, came back as a slice
of the positional table. `presk solve` went further and ignored a load error
altogether (`if let Ok`).

Now the file must be exactly the size the layout implies (`E9005` naming the
likely cause when it is longer), and `solve` fails on a checkpoint it cannot
load. `make precision` and `experiments/curve.sh` measure at `T = 256`.

**What this touches, stated rather than smoothed over.** Every result that
opened `out/gpt_tiny` at `T = 32` was measured on misaligned weights, which
are neither the initial nor the trained ones:

- 0.5.0 statement D's measured gap (1.05·10⁻⁷ against 4.52·10⁻³, D41). The
  *mechanism* it demonstrates — a budget verified against the denotation —
  is unaffected; the number describes other weights than it says.
- 0.5.0 step 5's sensitivity ranking on "trained" weights, and with it the
  finding that the token embedding ranks *below* the down-projection,
  against the folklore. That finding is **unsupported** until re-measured.
- The ε curve of 0.5.0 step 8 and the solved assignments of 0.6.0 step 4.
- The table of 0.6.0 step 5a, already replaced in the manual.

Timings are not affected (the kernels do the same work on any values), nor
is anything at initialisation or on the tiny MiniGPT. The re-measurements
belong with step 7, which trains the corpus GPT properly anyway; the curve
and the sensitivity ranking should be redone on those weights.

## D49 — Weights are priced by the signed first-order change, and it predicts within ten per cent (2026-09-21)

Step 5d's parameter menu. `presk solve --quant` offers, after the float
assignment, every parameter *declared* of rank two — the declared rank, so
that a stacked bias, whose leaf has rank two, is not taken for a matrix and
scaled across layers — as `int8` or `int4`, over its second-to-last axis.
Acceptance is by measurement of the whole realization, as D41.

The proposal is priced without calibration. Quantising moves each weight by
a known `δ = ŵ − w`, and the reverse sweep the solver already runs gives
`g = ∂L/∂w`, so to first order the loss moves by `Σ g·δ`. The first version
priced a parameter by the bound `Σ |g|·|δ|` and quantised **nothing**: every
matrix's bound exceeded the whole budget. Measured on the trained corpus GPT
(600 steps, `T = 256`), all matrices at once:

| | bound `Σ|g|·|δ|` | signed `Σ g·δ` | `√Σ (g·δ)²` | measured |
|---|---|---|---|---|
| `int8` | 1.19·10⁻¹ | 5.43·10⁻⁴ | 1.06·10⁻³ | 5.17·10⁻⁴ |
| `int4` | 2.16 | 4.01·10⁻² | 1.90·10⁻² | 4.39·10⁻² |

The signed prediction is within five and nine per cent. The linearisation is
right; what makes the bound 230 times too large is only its refusal to let
independent roundings cancel. That is D39's diagnosis — `k` against `√k` —
confirmed with a number, on parameters rather than on sums, and it is direct
evidence for question 58: a statistical reading of the same first-order
analysis is accurate where the worst case is vacuous.

The proposal uses the root-sum-square: within a factor of two of the
measurement, never negative, combined across parameters as a
root-sum-square. The signed sum is more accurate on the total but can cancel
between parameters, and a greedy search over a quantity that can decrease
when something is added would treat a lucky combination as free; the
measurement would catch it, at the cost of widening rounds. At a 10⁻³ budget
the solver quantises all nine matrices, five `int8` and four `int4`, 11×
fewer weight bytes than `f64`, estimated 4.41·10⁻³ and measured 3.11·10⁻³,
accepted without widening.

**Statement K holds.** On two windows of the Code civil, every matrix `int8`
and `f32` elsewhere is 3.23·10⁻⁶ from the exact loss (2.9987), inside 10⁻²
by a factor of three thousand; the representation part is 3.86·10⁻⁶ and the
arithmetic part 6.32·10⁻⁷. `int4` is 8.32·10⁻⁵. `bf16` everywhere is
4.27·10⁻⁴ — not like for like, since it rounds every activation, but the
comparison a practitioner makes: eight-bit weights and `f32` arithmetic are
130 times closer to the exact loss than sixteen bits everywhere, in half the
bytes. Measured on random tokens instead, `int8` is 7.96·10⁻⁵: a trained
model's loss off its data is a different function, which is why `presk
quantize` gained `--data` and the table is measured on text.

## D50 — The quality axis is flat, and J fails for a reason the solver's scope explains (2026-09-21)

Step 7, done where the plan's risk table said it might have to be: the curve
at the reduced constants the rest of the version measures at, since seven
course-scale training runs would take days on this machine (a course-scale
step is 57 s in `f64`, 30 s in `f32`). J is measured at course scale, by
timing alone.

**The curve** (`experiments/curve.sh`, now with the quality column). The
solver columns are re-measured on correctly loaded trained weights at
`T = 256` (D48): every budget from 10⁻¹ to 10⁻⁷ is inside, 3.22× to 1.20×
less traffic than all-`f64`. Each realization then trains the model from
scratch, 600 steps on 90 % of the Code civil, same initialisation, seed and
batch order, and is evaluated in `f64` on the other 10 %. Every validation
loss is within 1.2·10⁻³ of all-`f64`'s 3.0263, with no trend — the most
narrowed realization, 47 of 66 sites in `bf16`, has the lowest — while
all-`f64` at seeds 1 and 2 reaches 2.8967 and 2.9608: the seed moves the
result a hundred times more than any budget did. **There is no knee in
quality to identify** on this model at this length of training; the knee in
error is where it was. The limits are the model's size and 600 steps, short
of convergence, and they are the obvious next measurement rather than a
reason to doubt this one.

**J.** A realization is keyed by function and `let`, not by shape, so one
solved at `D = 64` with the time objective (`--profile` fitted per dtype on
the corpus GPT, 62 sites `f32` and 4 `bf16`, measured gap 2.9·10⁻⁵) applies
unchanged at course scale. Per training step, from the difference of a one-
and a four-step run: all-`f64` 56.9 s, solved 50.2 s, all-`f32` 30.4 s.
**J fails at 1.13×.** The all-`f32` row says why: the gain is there to be
had, and the solved realization does not reach it because it assigns only
the *loss's* sites. A training step is mostly the reverse sweep and the
optimizer, whose sites the realization does not mention and which therefore
run at its default, `f64`. The budget constrains the forward loss and says
nothing about the gradient, so the solver has no licence to narrow the part
of the step that costs. That is not a kernel problem and not a cost-model
problem; it is a question of what a budget binds in a training run
(question 60), and 0.5.0's statement E, which this J restated, was asking
the solver for a speedup outside the scope its budget gives it.

Also new for this step: `presk run --weights` (and `--from` with `--data`),
so a validation loss is one command, and the curve script's traffic columns,
which had been reading a line `solve` renamed in step 4, are read again.

## D51 — The condition number buys twenty orders and not the last ten; a budget is discharged by measurement, and the statistical model proposes (2026-09-21)

Step 6: the answer to questions 56 and 58, as §3.4 asked, with the numbers.

**What was built.** The exact interpreter evaluates every sum and contraction
a second and third time, on `|operands|` and on `operands²`, and once more per
operand with that operand replaced by ones. Every one of these primitives is
linear in each operand it sums over, so that gives, per output element,
`Σ|xᵢwᵢ|`, `√Σ(xᵢwᵢ)²` and the normwise factor `Σ Π_{i≠j}|·|`. `presk
calibrate` records their maxima in the envelope beside the magnitudes (and
now takes `--weights`, since a trained network is another function). The
propagation of `propagate.rs` has four rules for a sum, and changes nothing
else:

| rule | propagated error | rounding | what it is |
|---|---|---|---|
| `k` | `k · max|w| · max|δx|` | `c·u·max(k·|x||w|, |y|)` | 0.5.0's, sound |
| `normwise` | `max|δx| · Σ|wᵢ|` | `γ_c · Σ|xᵢwᵢ|` | sound |
| `cond` (default) | `(|δx|/|x|) · Σ|xᵢwᵢ|` | `γ_c · Σ|xᵢwᵢ|` | sound if each element's error is proportional to its magnitude |
| `rss` | `(|δx|/|x|) · √Σ(xᵢwᵢ)²` | `√c · u · Σ|xᵢwᵢ|` | an estimate: the terms' errors independent |

The componentwise assumption of `cond` holds of a value rounded once and is
only assumed after that, which is why the soundness test was kept asserting
and was the judge (the first recommendation of this step): it held in every
case below. `presk check --precision --model all --measure` prints the four
beside the gap measured natively on the envelope's own inputs and weights.

**Small entries** (the soundness test, 21 comparable cases):

| rule | median bound/gap | worst | within 10³ |
|---|---|---|---|
| `k` | 1.8·10³ | 1.6·10²⁰ | 10 |
| `normwise` | 48 | 1.3·10¹¹ | 16 |
| `cond` | 18 | 5.5·10⁴ | 17 |
| `rss` | 12 | 3.4·10⁴ | 17 |

The tiny MiniGPT's training loss, 8.7·10¹⁹ under `k`, is 8.8 under `cond`.
The estimate never fell below the measurement.

**The corpus GPT** (`D = 64`, `T = 256`, `make gpt-trained`'s weights, two
windows of the Code civil, loss 3.080):

| realization | measured | `k` | `normwise` | `cond` | `rss` |
|---|---|---|---|---|---|
| `f64` | 4.0·10⁻¹⁵ | 9.4·10³²× | 2.7·10²¹× | 2.3·10¹⁰× | **3.7×** |
| `f32` | 1.3·10⁻⁶ | 1.6·10³³× | 4.6·10²¹× | 3.9·10¹⁰× | **6.3×** |
| `bf16` | 1.6·10⁻³ | 3.3·10³³× | 8.2·10²¹× | 1.2·10¹¹× | **10×** |
| solved, 10⁻³ | 8.4·10⁻⁷ | 1.5·10³³× | 4.6·10²²× | 1.6·10¹²× | 2.4·10³× |

(`k` is 10³³ here and was 10²⁰ in D39: that figure was at `D = 32` on the
misaligned weights of D48.)

**So the condition number does not rescue the bound**, and it is §3.4's
expectation to the letter. It buys twenty-three orders on the GPT and the
bound is still ten orders above the error: 1.6·10⁴ relative in `f32`, on a
loss of 3. The summary condition numbers are small — the worst, 36, is
`LayerNorm`'s mean — so cancellation was never the problem. The per-site
tables show what is: `cond` and `rss` share every rule except the sums, and
the ten orders between them build up from about eight sums in series, each
costing `Σ|xᵢwᵢ| / √Σ(xᵢwᵢ)² · √c`, roughly a factor of `k`. That is
the `k` against `√k` of D39, and it lives in the *roundings' independence*,
which no worst-case rule sees, however well measured its magnitudes. In `f64`
the conditioned bound is 2.9·10⁻⁵ relative and discharges a 10⁻³ budget,
which is worth saying and not what a budget is for.

**The statistical estimate is accurate on uniform realizations and not on
mixed ones.** Within 3.7 to 10× on the GPT, as D49 found for weights, it is
the same first-order analysis read statistically. On the solved realization
it is 2.4·10³× high, for a reason with nothing to do with sums: the solver
put the attention logits in `bf16`, and one magnitude per site charges every
logit `u·max|s|` (`max|s|` ≈ 100) and passes it through the softmax at its
1-Lipschitz ceiling. The per-site ∞-norm envelope, not the model of the sum,
is the limit there.

**Recommendation to the specification.**

- **Question 56: option (c).** `budget(rel_error = ε)` is a claim about the
  *measured* distance between the realization and the denotation, on the
  envelope's inputs, and is discharged by measuring it — which only a
  compiler that keeps the exact denotation can do (D41). The deterministic
  bound stays a separate artefact, now by the conditioned rule, and is
  printed as the weaker one: sound in every case tested, useful on
  normalisers, small programs and in `f64`, and vacuous on a transformer in
  `f32` or narrower. Option (a) is closed with its number.
- **Question 58: yes, as a proposal and not as a discharge.** A statistical
  model is a legitimate realization of the same framework (RA3), and it is
  the right analytic tool to propose and to predict with: D49's
  root-sum-square for weights, this one for arithmetic, the same linearisation
  in both. It should not be what a budget is checked against, because it is
  an estimate with no probability attached, and 2.4·10³× high on the one
  mixed realization measured. What would make it closer is finer
  magnitudes (per row or per channel rather than per site), not another rule
  for the sum.

`presk solve` now takes the envelope's data slice as its inputs, so an
envelope measured on text is solved and measured on the same text. Its
proposal still prices with sensitivities (D41); making it propose with the
statistical propagation instead is named, not done.

## D52 — A budget in a training run binds one step's update, and J′ holds at 1.99× (2026-09-21)

Step 7b, the plan's §3.7, and the answer to question 60. J failed in step 7
at 1.13× because the budget constrained the forward loss and the solved
realization therefore left the reverse sweep and the optimizer, most of a
step, at `f64` (D50). This step gives the budget a scope that covers the step.

**What a budget binds.** One optimizer step's parameter update, measured
against the exact step from the same state:
`‖θ̂′ − θ′‖₂ / ‖θ′ − θ‖₂ ≤ ε`, where `θ′` is the objective's `vjp` followed by
`adamw.step` over the reals. It is discharged by measurement (D41, D51). The
state is a trained checkpoint and `N` warm-up steps of the run's own pipeline
natively in `f64`, so that AdamW's second moment is not zero. "The same state"
means the state rounded to the realization's default type: rounding `θ` on
the way in is not an error of the step (an `f32` run's parameters already
live in `f32`), and rounding `θ′` on the way out is, and stays in the
measurement. `presk calibrate --step` records the envelope of the exact step
(509 sites in 16 functions at `D = 64`) and prints, per realization, the
update gap, the gradient gap and the sampling noise between two batches;
`--reference f64` compares against a native `f64` step where the exact one is
not affordable.

**The storage floor, which the plan did not foresee.** The update is a small
difference of large numbers: `‖θ‖ / ‖θ′ − θ‖` is 716 on the corpus GPT at its
peak learning rate and 4.5·10⁵ at step 4 of the tiny MiniGPT's 2000-step
warm-up. Rounding the exact `θ′` to its storage type therefore costs
`u·‖θ‖ / ‖θ′ − θ‖` of the update before any arithmetic happens: on the GPT
3.6·10⁻¹⁴ in `f64`, 1.9·10⁻⁵ in `f32`, 0.74 in `bf16`; on the tiny model at
step 4, 2.2·10⁻¹¹ in `f64` and 1.2·10⁻² in `f32`. The report prints it as
its own column. On the tiny model the update gap *is* the floor to three
digits, since the arithmetic is negligible there. The plan's check that
all-`f64` is "within 10⁻¹² of the exact step" was therefore the floor's to
decide. It passes at a realistic learning rate (3.5·10⁻¹³ after 300 warm-up
steps) and fails at step 4, so the test reads the `f64` gap against the floor.
The consequence reaches beyond the test: a budget relative to the update is
not scale-free along a schedule, since at the end of a cosine decay every
realization fails it, `f64` included. It is stated at a state, here the step
after the schedule's warm-up (question 61).

**How the solver chooses** (`presk solve --step`). Every function of the
step is a group: the `.vjp`s, `adamw.step`, and — a departure from §3.7,
which kept the forward sites on their sensitivity proposal — the forward
functions too. The loss's sensitivities are in the unit of the loss and not
of the update, and the forward functions run again inside the reverse sweep's
recomputation, so they feed the gradient; they are measured like the rest.
Each group is run natively alone in `f32` and in `bf16`, with everything else
in `f64`, against one exact step, which takes 33 native steps at `D = 64`.
Groups are narrowed greedily by predicted time saved per unit of isolated
gap while the gaps' root-sum-square stays inside the budget (D49), and the
combination is then measured and widened as D41. The result is written as
a realization with a new `functions` map: a type per function, overridden by
a site named in the per-site map. `presk train --realize @R.json` needed no
change; a test checks that narrowing only `adamw.step` moves `θ` and that
naming a function the program lacks does not.

**The table** at `D = 64`, step 101 of 600 from `make gpt-trained`'s weights
and 100 warm-up steps, two windows of the Code civil, against the exact step
at 128 bits (`make step-table`). Sampling noise between two such batches:
1.49.

| realization | update gap | floor | gradient gap | predicted time |
|---|---|---|---|---|
| all-`f64` | 5.2·10⁻¹³ | 3.6·10⁻¹⁴ | 1.1·10⁻¹⁵ | 1.00× |
| all-`f32` | 2.2·10⁻⁴ | 1.9·10⁻⁵ | 6.4·10⁻⁷ | 1.01× |
| all-`bf16` | 0.74 | 0.74 | 1.3·10⁻² | 0.52× |
| recipe: `bf16`, the optimizer `f32` | 2.9·10⁻² | 1.9·10⁻⁵ | 1.3·10⁻² | 0.55× |
| solved at 10⁻³: 16 functions `f32` | 2.2·10⁻⁴ | 1.9·10⁻⁵ | 6.3·10⁻⁷ | 1.01× |
| solved at 10⁻²: 15 `f32`, `adamw.lr` `bf16` | 1.8·10⁻³ | 1.9·10⁻⁵ | 6.3·10⁻⁷ | 1.01× |

- **The update and not the gradient was right.** In `f32` the gradient is
  6.4·10⁻⁷ from the exact one and the update 2.2·10⁻⁴: AdamW's division by
  `√v̂` multiplies the relative error by 340. A budget on the gradient,
  question 60's provisional answer, would have judged this step 340 times
  safer than it is.
- **Where the error enters**, from the isolated measurements:
  `SelfAttention.apply.vjp` in `f32` costs 2.4·10⁻⁴ alone, `adamw.step`
  2.8·10⁻⁵ (1.9·10⁻⁵ of it the floor) and `Linear.apply.vjp` 2.0·10⁻⁵. No
  other function reaches 2·10⁻⁷. The root-sum-square of the isolated gaps,
  2.38·10⁻⁴, predicted the combination's 2.19·10⁻⁴ within nine per cent.
- **The folklore row.** Mixed precision as practitioners use it puts the step
  2.9·10⁻² from the exact one: outside a 10⁻² budget, and 130 times further
  than all-`f32`. The master weights in `f32` do their job, since the floor is
  `f32`'s and not `bf16`'s 0.74; the gap is the `bf16` forward and backward.
  It is small against a sampling noise of 1.49, which is the recipe's
  justification. What was missing was the number.
- **The solver takes `f32` everywhere and `bf16` nowhere**, because the
  profile predicts `bf16` slower than `f64` at this size. The budget licenses
  the narrowing and the time objective refuses the part that does not pay,
  which is what §3.7 expected. At 10⁻² it also puts the scalar schedule in
  `bf16` for a predicted 5 µs. That is legal and pointless, and it shows the
  resolution of the time model rather than anything about the budget.

**J′ at course scale** (`make j-step`, `D = 384`, six blocks). The realization
solved at `D = 64` is keyed by function and applies unchanged. Time per step
is the four-step time minus the one-step time, over three. The update gap is
of one step from D50's course-scale checkpoint plus ten warm-up steps,
against a native `f64` step:

| realization | s / step | vs all-`f64` | update gap | floor |
|---|---|---|---|---|
| all-`f64` | 59.7 | 1.00× | — | — |
| solved at 10⁻³ | 30.0 | **1.99×** | 6.7·10⁻⁵ | 4.7·10⁻⁵ |
| all-`f32` | 31.0 | 1.92× | 4.7·10⁻⁵ | 4.7·10⁻⁵ |
| recipe | 34.6 | 1.72× | 5.6·10⁻³ | 4.7·10⁻⁵ |

**Statement J′ holds**: 1.99× per training step, and inside its 10⁻³ budget
by measurement at both scales (2.2·10⁻⁴ against the exact step at `D = 64`,
6.7·10⁻⁵ against `f64` at course scale). The solved step is all-`f32` by
function and runs level with all-`f32`. The recipe is slower than `f32`,
because the `bf16` conversions cost more than the traffic saves. That is
0.5.0's lesson about `bf16`, now on a whole training step. Step 8 has the
realization it needed, inside a budget and faster.

**The trajectory** (`make step-train`). A per-step budget does not bound a
run, so the run was measured: 600 steps from scratch at `D = 64` under the
solved step and under the recipe, exactly as D50's rows (same initialisation,
seed, batch order and split), validation in `f64` on the held-out tenth. The
solved step ends at 3.0260 against all-`f64`'s 3.0263, its final `θ` 5.0·10⁻³
(relative) from the `f64` run's. The recipe ends at 3.0245 and 3.0·10⁻². Two
other seeds end 1.22 away, at 2.8967 and 2.9608. So the solved run ends 245
times closer to `f64` than a change of seed, and the recipe 40 times; both
are well inside the seed's spread. Over 600 steps nothing accumulated into a
different model. A longer run is the next measurement, as D50 said of the
quality axis.

**Also new**: `presk data diff`, the distance between two files of one type
(two checkpoints compared without the program).

**Question 60, answered**: the update, for the reasons measured above.
**Question 61, new**: the state at which a relative update budget is stated.

## D53 — The foil rests on what holds, each result as what it is; BERT-large runs for its reach and its cost, not its accuracy (2026-09-22)

Step 8, the MLPerf comparison. The recommendation of 2026-09-22 narrowed its
claims before it began. BERT-large had only been type-checked; the one MLPerf
benchmark within reach is BERT-large *inference*, while J′ is a *training*
step; and J′'s solved step is level with all-`f32`, which is MLPerf's
reference. So the step decided first what the foil can claim, then measured
both acceptance rules on one model and ran the benchmark's own. No submission
(D27).

**Both rules on one model** (`make foil`). MLPerf's closed division admits a
realization whose task accuracy stays within 99 %, or 99.9 %, of the FP32
reference's over the whole validation set: for BERT, F1 over the 10 833
features of SQuAD v1.1, against 90.874. A budget admits one whose measured
distance to the exact denotation, on the envelope's inputs, stays within ε.
Both were applied to the same realizations of the corpus GPT at `D = 64`, on
D50's all-`f64` checkpoint: it was trained on 90 % of the Code civil, so the
other 461 windows are held out, which an accuracy rule needs and `make
gpt-trained`'s weights, trained on every window, cannot give. The accuracy is
next-character accuracy over their 118 016 positions (a new entry of
`05_gpt.psk`, `accuracy`), with the model in the realization and the metric in
`f64`, as MLPerf's harness scores a model from outside it: a `bf16` mean would
round the metric itself to eight bits. The budget side is `presk quantize` on
two held-out windows.

| realization | gap to the exact loss | representation | arithmetic | accuracy vs `f32` | loss vs `f32` |
|---|---|---|---|---|---|
| `f64` | 8.8·10⁻¹⁶ | 0 | 8.8·10⁻¹⁶ | 0 | 0 |
| `f32` | 2.2·10⁻⁷ | 0 | 2.2·10⁻⁷ | — | — |
| `bf16` | 3.0·10⁻³ | 0 | 3.0·10⁻³ | −1 hit | +6.6·10⁻⁵ |
| `int8` | 1.5·10⁻⁴ | 1.5·10⁻⁴ | 4.9·10⁻⁸ | 0 | +1.5·10⁻⁶ |
| `int8` `pow2` | 1.1·10⁻⁴ | 1.1·10⁻⁴ | 4.1·10⁻⁷ | −1 hit | −4.0·10⁻⁶ |
| `int4` | 1.0·10⁻³ | 1.0·10⁻³ | 1.7·10⁻⁸ | −7 hits | −9.0·10⁻⁵ |
| `int4` `pow2` | 1.6·10⁻³ | 1.6·10⁻³ | 2.9·10⁻⁷ | −1 hit | +9.1·10⁻⁴ |
| solved at 10⁻³ (statement D's) | 2.4·10⁻⁷ | 0 | 2.4·10⁻⁷ | 0 | +5.3·10⁻⁹ |

(`f32` makes 18 615 correct predictions of 118 016; one hit is 5.4·10⁻⁵ of that.)

- **The accuracy rule admits all eight**, at 99.9 % as at 99 %: the largest
  change is seven predictions in 118 016. The budget separates them over
  twelve orders of magnitude from two windows, and says of every quantised gap
  that it is the weights' and not the kernels': the representation part is
  10² to 10⁵ times the arithmetic.
- **The accuracy is not monotone in the gap.** `int4` loses seven predictions,
  `int4 pow2`, further from the exact loss, loses one. A thresholded task
  metric reads the computation through a coarse and noisy lens, which is what
  it is designed to do.
- **What each costs.** The exact reference of the two windows takes 7.5 s at
  128 bits, once per checkpoint and sample; a check, one native run of them,
  about 0.1 s; a pass over the held-out windows 3.7 to 11 s. Per realization
  the budget is some forty times cheaper; its reference costs about two
  passes, once.
- **K restated.** On this checkpoint and these two held-out windows, `int8`
  weights are 1.46·10⁻⁴ from the exact loss, where K measured 3.23·10⁻⁶ on
  `make gpt-trained`'s weights and two training windows: forty-five times
  more, and still seventy times inside 10⁻². D49 says why they can differ so
  much: the error of a quantised loss is a signed sum of many terms, and on
  K's windows it cancelled well. K holds; the factor of three thousand was
  one favourable measurement.
- **`bf16`'s gap on a loss entry is mostly its own output.** The exact loss,
  3.02458, is already 2.2·10⁻³ from the nearest `bf16` number. A budget binds
  what the entry returns, and this entry returns one number stored in eight
  bits.

**BERT-large, run** (`make bert`, `experiments/bert.sh`). The benchmark within
reach, at MLPerf's own constants — 24 blocks, width 1024, 16 heads, context
384, vocabulary 30 522 — on its initialization, and on SQuAD-shaped features
with random WordPiece ids, laid out as MLPerf's harness lays them out
(`experiments/bert_features.py`; a new entry `squad` takes a feature as one
tuple, so that a data file can supply it). Nothing in it is about answering
questions or about a trained BERT, whose activations carry outlier features an
initialization lacks. What it measures is the compiler's reach, and what each
acceptance rule costs on the benchmark's own model.

- The exact logits of one feature at 128 bits: 1 h 57 min and 44 GB, once the
  interpreter was changed as below.
- Over the 768 logits (the largest exact one is 0.477), `max|ŷ − y| /
  max|y|`, which is what a budget on an entry binds: `f64` 1.5·10⁻¹⁴, `f32`
  1.1·10⁻⁵, `bf16` 1.1·10⁻¹, `int8` weights 5.7·10⁻², `int4` weights 2.0. Only
  `f32` is inside 10⁻². On two of the 24 blocks the quantised gaps split into
  the weights' rounding (1.09·10⁻² for `int8`, 2.17·10⁻¹ for `int4`) and the
  arithmetic (below 10⁻⁶ for both). K's scheme, 3.23·10⁻⁶ on the GPT's loss, is
  outside the same budget on the benchmark's output by a factor of six.
- Time per sequence of 384 tokens (`presk bench`, median of two, the better
  of scalar and four-lane code): `f32` 52.3 s at batch 1 and 6.92 s at batch
  20, `int8` weights 23.4 s and 3.10 s — **2.2× at both**, the first type
  narrower than `f32` to buy time on a model; `bf16` 66.5 s and 5.28 s, slower
  than `f32` at batch 1 and 1.3× faster at batch 20; `f64` 80.6 s and 13.1 s.
  At batch 1 a pass runs on one core: threads split a kernel's outermost loop,
  which is then the batch axis. The gain is this kernel's rather than
  `int8`'s: it reads the whole weight matrix again for every row of the
  result, and a 16 MB matrix streams from memory where its 4 MB `int8` copy
  stays in cache (`docs/kernels-roadmap.md`). `int4` is level with `int8`.
- What each rule costs on it: a pass over the 10 833 features is 9.3 h per
  realization in `int8` and 20.8 h in `f32`, at batch 20 on this machine; a
  budget's check is 2 to 3 min per realization against an exact reference of
  1 h 57 min, computed once — a tenth of one pass in `f32`.
- So on the benchmark's own model the version's aim is met at one budget and
  not at the next: `int8` is 2.2× faster than FP32 and inside 10⁻¹, outside
  10⁻², and the error is the weights' rounding. The budget does not say which
  ε the task needs — that is the accuracy rule's question, and here it would
  need a trained BERT and SQuAD — but it prices the speedup in computational
  error, attributes it, and does so before any dataset pass.

**What the foil claims**, each result as what it is:

| | measured | what it is | what it is not |
|---|---|---|---|
| D | 4.48·10⁻⁶ against 10⁻³, 2.12× less traffic (D50) | a budget declared on an entry, discharged by measurement against the exact denotation | independent evidence of the solver, which widens until the measurement passes; a speedup (J: 1.06×, 1.13×) |
| K | `int8` weights 3.23·10⁻⁶ against 10⁻², and 1.46·10⁻⁴ on held-out windows | inside a budget on a **loss**, a scalar averaged over 512 positions | a speed claim (the GPT forward is level with `f32`); a claim about outputs: on BERT-large's logits the same scheme is 5.7·10⁻², outside 10⁻² |
| the split | 3.86·10⁻⁶ representation, 6.32·10⁻⁷ arithmetic (K's); 1.46·10⁻⁴ and 4.9·10⁻⁸ held out | a quantised gap attributed to the weights or to the kernels | exclusive to an exact reference: `f64` would approximate it wherever its own gap is known to be small, and knowing that is the exact reference's work. What cannot split anything is a task-accuracy rule, which has a reference *score* and no reference *computation* |
| J′ | 1.99× per training step against all-`f64`, inside its 10⁻³ update budget (D52) | the budget licensing the move off `f64` by measurement, and the time objective refusing `bf16` | faster than FP32: all-`f32` is 31.0 s against 30.0 s, inside the run-to-run noise, and FP32 is MLPerf's reference; an inference result |

Four sentences of the `coverage` chapter's foil, written in 0.5.0, are
withdrawn. "Checking it costs one run against the exact denotation … in
milliseconds": a check is one native run, a tenth of a second on the GPT and
two to three minutes on BERT-large, against an exact reference of seconds on
the GPT and two hours on BERT-large, once per checkpoint and sample. "A
degradation beyond ε is a compiler bug, provably": by measurement, on the
envelope's inputs, which is all spec §8.3 claims. "It is per kernel, so the
number transfers": a gap is attributed per function by isolated measurement
(D52) or per part by the split; what transfers across scale is the realization,
keyed by function, and its gap is measured again there (J′). And "what it
still needs … is a realization both inside a budget and faster": on
BERT-large `int8` weights are 2.2× faster than FP32 and inside a budget of 10⁻¹
on the logits, not of 10⁻²; J′ is inside its budget and faster than `f64`,
level with FP32.

**What running it took.** The exact interpreter cloned a value at every use.
A field of the parameter tree was a clone of the whole tree, then of the
field; a call cloned its arguments; a fold cloned its stacked parameters; and
the backend seam converted them twice. BERT-large's 335M parameters would have
been held about eight times, some 180 GB on a 119 GB machine: four blocks
already held 40 GB. Tuples and records are now shared (`Arc`) rather than
owned, so cloning a tree copies pointers and never a tensor; a projection or a
field clones only what it selects; and the interpreter backend consumes its
arguments instead of copying them (`Value::into_interp`, which rounds in
place to the same value `to_interp` gives). It is representation only: the
four-block exact run is byte-identical before and after, in 20.3 min against
20.8, at 13.5 GB instead of 40.3, and every golden is unchanged. Two copies
of the parameters remain: the interpreter's own, and the synthesized inputs
that `presk run` keeps for its output.

**Question 62, new**: the exact interpreter now reaches an MLPerf model's
inference at its own constants, and still not a course-scale training step,
where J′ was discharged against native `f64`. Is a discharge against `f64` a
discharge? Provisionally yes, as a two-part claim stated as such: `f64`'s own
gap to the exact denotation where both run (1.5·10⁻¹⁴ on BERT-large's logits),
then the realization's gap to `f64` at scale.
