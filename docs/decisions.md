# Architecture decisions

One paragraph per decision, newest last. The plans these come from are
`presk-specs/implementation/presk-plan-0.1.0.md` through
`presk-plan-0.7.0.md`; the decisions of 0.7.0 start at D56.

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

## D54 — The export's `training` switch: `train.py` drops with PyTorch's masks, everything else stays at the expectation (2026-09-23)

`make experiment-gpt` had failed since 0.5.0 step 1: the corpus GPT's
objective took a per-step key (D29, D37) and the generated `train.py` still
called `loss(theta, batch)`; and had it run, its dropout would have been the
expectation (`training=False`) while `presk train` drops, an unequal
comparison. The decision (user, 2026-09-23): **both sides drop, with
different masks**. The export gives every `uses Rand` function a
`training=False` keyword beside `generator=None`, passes both on to every
`uses Rand` callee (sub-module calls, the kwargs of `functional_call`, the
stacked loop), and emits `F.dropout(x, p, training=training)`; `train.py`
calls the objective as `loss(theta, None, batch, training=True)` — the key
is a value the export threads and never reads — and PyTorch draws the masks
from its global generator, seeded from `--seed`. The switch is not the
generator because the generator also serves `sample` in `serve.py`, where
the model must stay at its expectation (question 33): a served model that
dropped would be a different function. The K-step conformance replay passes
`None` for the key and leaves the switch off: the interpreter's keyed masks
cannot be reproduced either way, so a keyed objective's replay is not a
conformance test, and no example has one. For the comparison of
`experiments` §3 this means the timings are fair — same model, data,
batches, steps and threads, dropout costing the same on both sides — and
the losses are not comparable to the digit, since the masks differ. On the
way, the stacked loop with a tuple accumulator (D37) was found exported as
`hL, _ = block(hL, _, mask)`, the pattern splatted into two arguments; it is
now `block((hL, _), mask, …)`. Both sides now print the median step time
with the first step excluded — `presk train` on standard error and in
`--json` (`ms` per step, `step_ms`), `train.py` after its log — so the
comparison is per step and leaves out the native backend's compilation and
PyTorch's warm-up; `train.py` takes `--threads`, defaulting to the machine's
cores, and prints the thread count. `experiments/exact_vs_almost.py` calls
`OBJECTIVE(theta, batch)` on the tutorial's and the tiny MiniGPT's exports,
whose objectives take no key: unchanged. Alternatives considered: the
generator as the switch (turns dropout on wherever sampling is seeded,
serving included); a module-level flag in the exported file (one per unit,
set from outside by `train.py`); PyTorch's `module.training`
(`functional_call` builds a fresh instance, in training mode by default, so
the expectation would need `.eval()` at every construction and the
conformance tests would drop unless told not to); dropout off on both sides
(then `presk train` would not run the training run the program declares).

## D55 — PyTorch on falicon: the gap is 22×, and the compiler is not what moved it (2026-09-23)

The only speed figure against a mature stack was 0.4.0's 94×, measured on
another machine, and the assessment of 2026-09-21 listed it as stale. It is
re-measured here, on falicon, with `make experiment-gpt` repaired (D54): the
course GPT (D = 384, 6 heads, 6 blocks, T = 256, batch 16), `f32` and twenty
threads on both sides, six steps, the median of the five timed steps with
the first excluded, three repetitions. **PyTorch 2.14 `float32` 1.34 s a
step; presk `f32` 29.96 s a step; a factor of 22.4.** The step solved under
the 10⁻³ update budget of J′ takes 30.61 s, 22.9× PyTorch's and level with
all-`f32` (the three `f32` runs span 29.47 to 31.44 s): J′'s 1.99× is
against all-`f64`, and 0.6.0 already said the solved step is level with
`f32`.

The number to carry is not the factor but its decomposition, and a reviewer's
question about which BLAS PyTorch uses is what produced it. The wheel carries
**its own OpenBLAS 0.3.34** (`BLAS_INFO=open`, `DYNAMIC_ARCH USE_OPENMP`,
selecting its `neoversev2` kernels) and **oneDNN 3.12 over Arm Compute
Library 53.2**; on a 1024³ `f32` product the two are within 4 % of each other
(710 GFLOP/s with oneDNN, 680 with OpenBLAS alone), so naming "the platform
BLAS" was wrong twice over — it is vendored, not the platform's, and it is
not one library. Presk's blocked kernel with lanes does the same product at
**49.6 GFLOP/s**, so **14 of the 22 is the contraction**, and the remaining
1.6 is everything that is not one.

Neither side moved for a reason inside the compiler. Presk's matmul rate is
where 0.4.0 left it (47–56 → 49.6 GFLOP/s) and its step rate rose from 7.0
to 8.9; PyTorch's matmul halved (≈1400 → 710) for want of the matrix
coprocessor the earlier machine's platform BLAS reached, which these cores
do not have — they report `sve sve2 i8mm bf16` and no `sme`, and 710 is
about four fifths of what 128-bit SVE fused multiply-add allows on twenty
cores at 2.8 GHz. What is left is the rest of the step: each side runs it
below its own matmul rate, presk at 18 % and PyTorch at 28 %, where on the
earlier machine those figures were 14 % and 45 %. So the re-measurement
neither vindicates nor accuses the two versions between. It says that
against a mature stack on comparable instruction sets the distance is one
order of magnitude, that it is the kernels' and not the discipline's, and
that the figure to quote from now on is 22× on this machine rather than 94×
on another. The GB10 GPU takes no part: the installed wheel is CPU-only
(`torch 2.14.0+cpu`, `torch.cuda.is_available()` false), and a GPU row would
not be the CPU comparison in any case; the GPU is 0.8.0's, per the plan of
2026-09-23.

The losses are comparable to about four digits and no further, by
construction: both sides drop and the masks differ (D54); after six steps
PyTorch ends at 4.507609 and presk at 4.507878. What the comparison is of,
then, is one training step of the same model on the same batches at the same
arithmetic width, which is what a per-step time should be of. Alternatives
considered: dropout off on both sides (then `presk train` would not be
running the training run the program declares); PyTorch's `--dtype float64`
as the reference (not what a practitioner runs, and MLPerf's reference is
FP32); timing by total wall time (it charges the native backend's first-step
compilation to every step, and 0.4.0's numbers were of that kind).

## D56 — The layout space is defined, and its defaults are 0.6.0's kernels except where those do not realize their plan (2026-09-23)

Plan 0.7.0 step 0 wrote the layout space into the language reference
(`presk-specs/reference/presk-reference.tex`, draft v0.2, chapters 6 and 7):
the domains of the contraction form, the register tile, the k-block, the
loop order, the lanes, the thread split, the pack and the softmax's passes,
each per nest; the exactness rule of plan §3.2 as a requirement; the plan
object `{ realization, profile, layout }` beside the realization's form;
well-formedness conditions 7 (a running sum kept in the output buffer needs
the accumulation type to be the storage type), 8 (a split at depth d needs
loop d parallel with two iterations) and 9 (a pack only on an NT nest); the
parallel loop generalized to a depth, with an output index `c·v + t`,
`0 ≤ t < c`; and tile loops in the kernel grammar. Tile loops take no new
statement: a block loop is a `for` over `E/n` whose variable indexes its
axis as `ib * n + t`, and a k-block's index is set by the assignment to a
loop variable the grammar already has, `k = kb * b + kt`. The kernel
interpreter needs nothing else (plan step 2 says so), and a production the
compiler never printed would have made the grammar and `dump --stage
kernel` disagree on day one.

The step's check was that the default of every new decision reproduces
0.6.0's kernel, read from `lower.rs` and from `presk dump --stage kernel` on
`SelfAttention.apply` at small constants. It does for the `dot` whose J is a
multiple of eight — tiled `(1, 8)`, loops outer, I, the block loop over J,
then k, natural lanes on the native backend — and for everything that has
only a thread split. It did not, as the plan had it, for four things, and
the reference and the plan's §3.1 were corrected: (1) **the `ein`s are the
tile `(1, 1)`**, one scalar accumulator per output element, one lane, not
`(1, 8)` with lanes; the tile domain gains `(1, 1)`, since a default must be
a point of L(P) (its product is `(1 · q) · k`, exact, so a builder that
drops the `1 ·` changes text and not bits). (2) **The row-wise form is
taken whenever J is not a multiple of eight**, not four, and it keeps each
running sum in the output buffer from one k to the next. In `f32` and `f64`
that is exact and stays the default. In `bf16` it rounds every partial sum
to `bf16`: on `x[4, 1024] · w[1024, n]` the median relative gap to the
exact product is 2.4·10⁻² at n = 12 (row-wise) against 3.3·10⁻³ at n = 16
(tiled), where `f32` gives 5.5·10⁻⁷ and 5.8·10⁻⁷. That is a `bf16`
accumulation the realization does not state (its `accum` defaults to
`f32`), so condition 7 makes the `bf16` row-wise form ill-formed and the
default there is tiled `(1, 1)`. The course GPT's logits projection (V = 92)
is the one such nest in the corpus; its `bf16` bits will move, towards the
exact value, when step 2 builds the default, and the conformance suite only
prints `bf16` gaps. `embed`'s derivative and `sum_to` accumulate through
their output the same way and have no other form: they are recorded in the
reference's status, outside 0.7.0. (3) **No kernel reads the realization's
`accum`**: a contraction sums in the arithmetic type of its storage whatever
the file says, and only the error analysis reads the member (the manual
said otherwise and is corrected). (4) The two-pass softmax stores
`exp(x − m)` in the output before dividing, so on a `bf16` nest it rounds
it twice; condition 7 asks two passes to store in the arithmetic type. Also
written down: 0.6.0's lanes are per program and per executor (natural on
the native backend, one on the kernel interpreter and in `dump`), which is
harmless since both are bit-identical; `parallel_outer` accepts an index at
"some position" per store and reads no loads, where the definition now asks
for one position and covers loads, which no 0.6.0 kernel needs; and V(P) is
not the product R(P) × L(P), since conditions 4 and 7 tie a layout to its
realization, so the reference writes {r} × L_r(P).

The sweep's cores. falicon's ten Cortex-X925 (CPU part `0xd85`, 3.9 GHz)
are CPUs **5–9 and 15–19**; the ten Cortex-A725 (`0xd87`, 2.8 GHz) are 0–4
and 10–14 (`/proc/cpuinfo`, `lscpu -e`). L1d is 64 KiB and L2 1.25 MiB per
core; the two 12 MiB L3s are shared by CPUs 0–9 and 10–19, so each group of
five X925 shares its L3 with five A725, and a five-worker split pinned to all
ten may run on one L3 or across both — a source of noise the protocol has to
live with or pin away. `std::thread::available_parallelism`, 0.6.0's default
worker count, honours the affinity mask: 20 unpinned, 10 under `taskset -c
5-9,15-19`, so the default split of a pinned run is p_M = 10 workers, the
domain's `{5, 10}`. The Makefile records the set as `RA4_CORES`, for step
5's `experiments/ra4.sh`. Alternatives considered: keeping `(1, 8)` as the
`ein` default (then the default would not be 0.6.0's kernel, and a plan
stating nothing would change every `ein`'s kernel for a layout nobody
chose); putting the row-wise form
outside L(P) as the plan had it (then a J that is a multiple of four and not
of eight would have a default outside its own space); a `tile` statement in
the kernel grammar (see above).

## D57 — A plan's layout is refused where it cannot be lowered, never ignored (2026-09-23)

Plan 0.7.0 step 1 gives the compiler the plan object of the reference (§6.6):
`presk_rt::Plan { realization, profile, layout }`, the layout a map from nest
names to `presk_rt::Layout`, read by `--plan @P.json` on `run`, `bench`,
`train` and `dump`, and carried by the kernel backend, which reads a nest's
layout at every launch (`layout_at`). The reference defines the whole layout
space; this step lowers one decision of it, the lanes of a tiled `dot`, and
steps 2 and 3 lower the rest. What a plan may say in the meantime is the
decision. **A stated decision is checked, then lowered or refused**: before
anything runs, every nest the plan names must exist and be a nest (the root
of a fusion group or a primitive of its own, condition 1) and may state only
decisions its primitive can have; at the launch, its layout is checked
against its shapes and types — conditions 3, 4, 7 and 9, in
`Layout::check` — and then against the lowering, which accepts the lanes and
every other decision at the value 0.6.0's kernel has. Anything else is
`E7002`, naming the nest, the decision and the kernel it would have had. The
alternative, accepting a plan and lowering what can be lowered, is how a
benchmark would come to time a tile it never built and report it; the rule
is the one D43 and D56 applied to statements, applied to a file. A thread
split is refused before the run, so that a split on a nest the entry never
launches cannot pass unnoticed; `dump --stage realize --plan` checks only
that the nests exist, so a whole plan can be read before it can be run.

The rest follows from the reference. The **lanes** field of the backend
becomes its default layout's (`default_layout.lanes`: `Some(1)` on the
kernel interpreter, unstated on the native backend, which gives each kernel
its natural width where its form has lanes). A **stated** width is held to
condition 4 — one or the register's width, dividing the tile's columns — but
the backend's default is not, since it asks for "as wide as this kernel
allows" rather than for a number. So `dump --stage kernel` stays scalar
unless a plan states lanes, and a plan with the default layout gives
byte-identical dumps: `make plan-check` compares ten entries, and against a
build of `v0.6.0` the dumps are identical in `f64`, `f32` and `bf16`. The
layout lowered is part of the **kernel cache's key**, so two nests of one
shape with two layouts, or one nest under two plans, never share a compiled
kernel (tested on `tests/plan/two_dots.psk`). A plan's `realization` is a
record, `@FILE`, or the name of a uniform realization, which the reference's
definition of a uniform realization already lets a name denote; paths are
read from the current directory, as `--realize @FILE` is. `--plan` and
`--realize` are exclusive: `--realize R` is the plan `{ "realization": R }`.
The `profile` member is recorded and read by no command yet (question 64).
The default layout of a `bf16` `dot` whose `J` is not a multiple of eight is
still 0.6.0's row-wise form, which condition 7 forbids a plan to state (D56);
`presk_kernel::plan::emitted` is the one place that says so, and step 2
replaces it. Alternatives considered: the layout as a member of the
realization file (the reference's question 63 answers no); lanes stated as
"on/off" rather than as a width (the reference's domain is a width, and a
plan must mean the same on another machine or be ill-formed there, question
64); refusing a stated decision even at its current value (a plan listing
0.6.0's kernel explicitly would then be refused, and `make plan-check` could
not show that stating the defaults changes nothing).

## D58 — One contraction builder; every point of the space exact; a band that seed 0 had passed by luck (2026-09-23)

Plan 0.7.0 step 2. A **contraction nest** is recognized once, for `dot` and
`ein` alike (`lower::Contraction::of`: a `dot` whose right operand has rank
two and whose left has rank at least two, or an `ein` whose output is the
outer axes, then `I`, then `J`, over one contracted `K`, under an exact
order), and one builder lowers it (`emit_contraction`) in the form, register
tile, k-block, loop order and lanes its layout states — the reference's
decisions, `Layout::resolve` supplying the defaults the plan leaves unstated.
`lower_vec` became `lower_planned`. The row-wise form is 0.4.0's, generalized
to an `ein`. The tiled form is the outer loops, the block loops in the stated
order, and a micro-kernel of `m_i × n_j` accumulators — scalar, or `n_j/lanes`
vectors per row — over the `k` of a block. With several rows, operand 1's
columns are loaded once per `k` and shared; with a k-block, `k = kb·b + kt`
is set in the within-block loop. When the loop over the blocks of `K` encloses
the one over `I` or `J`, the accumulators are zeroed, reloaded from the output
under `if kb`, and stored after each block: a guard rather than a zero-fill
before the loop, so the nest stays perfect down to the block loops, which a
split at a depth needs (step 3). Loop variables and accumulators are numbered
in 0.6.0's order, so the default prints 0.6.0's text. Contractions of another
pattern, and every contraction under an interleaved order, keep their
lowering.

**The default is 0.6.0's kernel.** Against a build of `v0.6.0`, the kernel
dumps of the ten entries of `tests/plan/entries.txt` in `f64`, `f32` and
`bf16` give 585 identical kernels, every `f64` and `f32` `dot` among them. 30
`ein` kernels differ, all by one thing: 0.6.0 multiplied by one before the
product (`(1 * a) * b`), and the builder loads `a` into a scalar and computes
`s * b`, which is exact. It also takes one operation per term out of an
`ein`'s weighted count `W`, so the cost model's figure for the attention
`ein`s drops by a third in every type alike. Three `bf16` `dot`s differ on
purpose (J = 4 and J = 92). A `bf16` `dot` whose `J` is not a multiple of
eight is now tiled `[1, 1]`, one accumulator per element in `f32`, as the
reference's condition 7 requires. On `x[4, 1024] · w[1024, 12]` its median
distance to the exact product falls from 2.4·10⁻² to 2.7·10⁻³, the tiled
form's (D56).

**Every point is exact.** The plan asked for 64 random points per nest.
The whole space costs four seconds, so `tests/layout.rs` checks every point
of three nests: a `dot` as `Linear` has it (I = 8, J = 48, K = 128) and
attention's scores (NT) and values (NN) `ein`s at I = J = K = 32, in `f64`,
`f32` and `bf16`, 1 142 points (`Layout::space`), each lowered to a kernel of
its own and giving every output bit of the default, on the kernel interpreter
and natively. Distinct kernels are counted, so a layout that was not applied
cannot pass the comparison trivially. `Layout::space`, the enumerator of the
reference's space without the thread split, is also what `presk plan` will
enumerate (step 5).

**Found: lanes on an NT nest need the pack.** Operand 1 is read along `J`,
one vector per load, and the kernel language loads only contiguous lanes;
with `K` innermost its `J` is strided. The reference allowed such lanes;
condition 4 now says "one unless operand 1 is NN or packed" (`presk-specs`,
reference and plan §3.1). At the course shapes the scores `ein`'s well-formed
space is 233 points without splits, the values `ein`'s and a `Linear` dot's
267. Alternatives considered: a strided vector load in the kernel language
(a change of the language for a case the pack covers); vectorizing along `I`
(stores would stride).

**A band that seed 0 had passed by luck.** The `bf16` fix moved the
conformance gaps of the five test programs whose `dot`s have odd widths.
`mse` 1.7 → 2.4·10⁻³, `mse_grad` 2.8 → 1.2·10⁻², `stack_grad` 1.1 →
1.2·10⁻², `split_grad` 6.0 → 5.5·10⁻³, and `stack_prefix_grad`'s
`loss_grad` 3.3 → 7.6·10⁻², over the band (2⁻⁴ with a floor 2⁻⁵) at γ's
gradient, an exact −0.0141 left over from larger terms cancelling. `f64` and
`f32` did not move. Over twenty input seeds that entry is slightly more
accurate than before (median 5.8·10⁻³ against 6.2·10⁻³ over every element),
but it crossed the band on 6 seeds of 20 with 0.6.0's kernels and on 9 now.
Seed 0 had passed by luck, and the dots at K = 4 only reshuffle input
rounding. A `sum_to` accumulating in its arithmetic type was tried, since it
is the other kernel D56 found accumulating through its output. It did not
move the element, and was put back. **The user's decision**: the band is
asserted on the examples, which the reference's conformance definition
covers, in every realization; on the compiler's own test programs
(`tests/data`, `tests/train`, the MLPerf transcription) in `f64` and `f32`,
their `bf16` gap reported and not asserted. The alternatives were to widen
the reference's `bf16` band, to postpone the fix, or to change the seed.

## D59 — The pack, two softmax passes and a split at a depth: every decision of the table lowered (2026-09-23)

Plan 0.7.0 step 3 lowers the three layout decisions that step 2 still
refused, so every decision of the reference's table has a lowering and
`E7002` now means an ill-formed layout, never an unimplemented one.

**The pack** is two launches for one nest (question 66's provisional answer).
First a copy of operand 1, in its own type, to a panel with the axes
(outer…, `K`, `J`), written along `J` (`lower::pack_panel`, a kernel named
`pack`). Then the contraction reading the panel, which is NN
(`Contraction::packed`), so lanes are allowed on it. Quantised operands keep
their marking on the copy's input. The copy carries no split: the nest's
split names the contraction's loops, and the copy takes the executor's
default.

**Two softmax passes** (`softmax_two_pass`): the maximum, then one loop
storing `e = exp(x − m)` into the output while summing `z`, then a loop
dividing the output by `z` in place. The values and roundings are the
three-pass form's where the output is stored in the arithmetic type, which
condition 7 requires; each exponential is computed once instead of twice.

**A split at a depth.** `Kernel::parallel_at(depth)` implements the
reference's definition: a perfect nest down to the loop, its variable never
assigned, and one position of the output indexed by `c·v + t`, `0 ≤ t < c`
over every lane, in every store to the output and every load from it. So a
contraction's block loops qualify through `jb·n_j + t`, and a spill's reloads
are covered. A kernel now carries its split (`ir::Threads`: the executor's
default, one worker, or a loop at a depth over a stated count), which is not
part of its text. The host sets it once the kernel exists and checks it:
condition 8 (the loop at that depth is parallel, with two iterations or
more) against the kernel, condition 5 (the workers are `p_M` or `p_M/2`)
against the executor (`KernelExec::workers`, `None` on the kernel
interpreter, which runs everything on one worker and is not held to it). The
Cranelift kernel is compiled with the loop at that depth ranged and the loops
enclosing it emitted in full (`range_at`), the signature `fn(*const *mut u8,
i64, i64)` unchanged. The run takes the stated workers whatever the work: the
four-million threshold is the default's alone. The default keeps 0.6.0's
analysis (`parallel_outer`, depth 0, the variable itself at some position of
each store), so no default schedule moved. A backend's default layout may
also state a split (the tests do); like its lanes, it applies where the
kernel admits it and is not held to the conditions.

**Checks.** `tests/layout.rs` lowers every point of four nests — a `dot` as
`Linear` has it, attention's NT scores and NN values `ein`s, its softmax — in
`f64`, `f32` and `bf16`: 1 491 points, the pack included, each lowered to
kernels of its own and bit-identical to the default on the kernel interpreter
and natively. The softmax is compared on each executor against its own
default, since the two take `exp` from two libraries. A fifth of the points,
split at depths 0, 1 and 2 over 2 and 4 workers natively, gives 1 618
bit-identical runs, and 200 refused by condition 8, each a split naming a
loop that is not parallel (the loop over the blocks of `K`, a softmax's row).
`threads_do_not_change_a_single_bit` now also splits every kernel of the
course GPT at depths 0, 1 and 2 over 2, 5 and 12 workers, at batch 4 and at
batch 1, where the outermost loop has one iteration. The reference's example
plan (`tests/plan/attention.json`: the scores `ein` tiled `[4, 16]`, a
k-block of 32, the order `JIK`, four lanes, packed, split at depth 1 over ten
workers; the softmax in two passes) runs the course GPT's attention at B = 16,
T = 256 with the default's output. Timed once, unpinned, as a sanity check
and not a measurement, it takes 230 ms against 261 ms for `f32` with lanes at
the default. Alternatives considered: the split as an argument of the
executor rather than a field of the kernel (the reference lists the split
among what a kernel carries, and the compiled code depends on it); the pack
as a scratch buffer inside one kernel (a second output, which no kernel has:
the memory-placement answer to question 66).

## D60 — The layout cost model: the machine read from the cores it runs on, traffic by level, u = 0.185 on a GEMM (2026-09-23)

Plan 0.7.0 step 4 gives the compiler the cost model of plan §3.5, beside
0.6.0's roofline, which it leaves as it was: the optimizer, the precision
solver and every profile written before keep their meaning.

**What the model reads** (`presk_kernel::cost::planned_cost`, a
`PlannedCost` beside `KernelCost`, which keeps its two fields). It reads one
worker's part of a kernel: the kernel with the loop its split divides cut to
a worker's share, the split being the kernel's own or the executor's default
as the executor decides it (the outermost parallel loop over every worker
above `PARALLEL_MIN_WORK`, now in `presk-kernel` so both read one constant).
- *Work*: 0.6.0's weighted count of that share.
- *Traffic* by level, per buffer: the footprint of its accesses inside each
  loop, from the affine indices (a `SetL`'s `k = kb·b + kt` substituted), is
  compared with the capacity of L1, L2 and a worker's share of L3. A buffer
  that fits inside a loop is brought in once per iteration of the loops
  enclosing it; otherwise the loop's body is looked at once per iteration.
  Each buffer counts against the whole capacity, which is optimistic when
  several compete; the per-class table of step 6 will say whether it matters.
- *Register pressure*: the real and vector variables of the busiest
  innermost loop against the machine's vector registers, the excess charged
  as two level-1 accesses per iteration and per spilled register, counted as
  work.

**The machine** (`Machine`) is read from sysfs for the first CPU in the
process's affinity mask, not `cpu0` as the plan wrote. On falicon `cpu0` is a
Cortex-A725 with 512 KiB of L2, and an X925 has 2 MiB. Plan §3.5's 1.25 MiB
was the average `lscpu` prints over both kinds. sysfs gives an L3 of 8 MiB
shared by CPUs 0–9 (`lscpu` says 12 MiB per instance), shared by five of the
pinned cores. `p_M` is `PRESK_THREADS` or the affinity mask, as the native
executor counts it, so condition 5 and the model agree. The description is
stored in the profile (`machine_desc`), which is where it is overridden.

**The fit** (`presk_opt::profile::fit_layout`, a `LayoutFit` in the profile
rather than fields added to each `Fit`, since the `β`s and the spawn cost are
the machine's, one each): `T̂ = λ + spawn·[p > 1] + max(W/π_g, Q_L2/β_L2,
p·Q_L3/β_L3, p·Q_mem/β_mem)`. The group `g` is (machine type, lanes, the
split's stated workers), the plan's decision, whatever share of them a short
loop keeps busy. The fit is coordinate descent on a log grid, one constant at
a time, on the squared relative error, the grid narrowing when a round
improves nothing. It recovers synthetic constants within 10 %. `u` is the
median absolute relative residual.

**The calibration** (`presk bench --calibrate FILE --space calib`, `make
calibrate-ra4`): a GEMM (`experiments/gemm.psk`) at 256, 512 and 1024, `f32`
and `f64`, pinned to the ten X925 cores, 96 layouts each. The layouts are the
tiles `[1, 8]` and `[4, 16]`; no k-block in `IJ` and `JI`, a k-block of 128 in
every order of `IJK`; both lane widths; 1, 5 and 10 workers; five calls
each. The samples accumulate in the profile. On falicon, 576 samples, the fit
gives λ 0, spawn 182 µs, β_L2 4.4·10¹⁰, β_L3 1.8·10¹¹ and β_mem 2.9·10¹¹
bytes/s, and π per worker 1.67·10¹⁰ (`f32`, one lane) to 4.40·10¹⁰ ops/s
(`f32`, four lanes). **u = 0.185, Spearman 0.980.** The GEMM is bound by its
work and its L2 traffic at these sizes, so `β_L3` and `β_mem` are barely
identified (`β_mem` above `β_L3` shows it); the attention nests, never in the
set, may lean on them more. The calibration was run per size and type as it
would be, and the fit printed after each; `u` stayed between 0.185 and 0.27.

**A preview, not a result.** In the calibration's own samples, 1024³ `f32`:
the best layout (`[4, 16]`, k-block 128, `KIJ`, four lanes, split at depth 1
over ten workers) runs at 405 GFLOP/s, and 0.6.0's default layout (`[1, 8]`,
four lanes, ten workers at depth 0) at 77, a factor of 5.3 where statement P
asks for 3. 405 is within 1.8× of the 710 GFLOP/s D55 measured for PyTorch on
twenty cores. The model's pick in-sample is that best layout, and it orders
87 % of the pairs of those 96 that differ by more than 10 %. None of this is
a test of RA4. The GEMM is the set the constants were fitted on, and the
statements are decided in step 6 on the attention nests with the plan's
noise rule. Alternatives considered: `cpu0`'s caches (the wrong core); one
`π` per core (a schedule the model does not have); fitting `λ` in closed form
as the roofline does (with a `max` of five terms, coordinate descent is as
simple and does not special-case one constant).

## D61 — `presk plan`, `presk autotune`, `presk rank`: every nest captured once and every point timed on its own buffers (2026-09-23)

Plan 0.7.0 step 5 gives the compiler the three commands of the RA4 test and
the script that runs them. Nothing is decided yet: step 6 does the measuring.

**Capture.** All three commands read an entry the same way. It runs once
natively under its default layout, with the kernel backend recording the
first launch of every nest at every operand shape (`presk_kernel::Launch`):
the nest's name, its primitive or fusion group, its operands, its types and
reduction order, and its input buffers as the kernel received them. Each
captured launch is then lowered again under a layout by the function the
backend itself now calls (`plan::lower_launch`: the checks, `Layout::resolve`,
the lowering, a pack as two kernels, integer weights marked, the split
checked by condition 8). So a point is the kernel a `--plan` run would
compile. The plan asked for each point to be timed on the captured buffers
rather than through the entry. That is sound because an entry's kernels run
one after another, so a nest's time does not depend on another nest's layout.
It is also what makes a sweep tractable.

**The points.** A point is one of `Layout::space` (the reference's `L_r`)
combined with one of the splits: none, or at depth 0, 1 or 2 over `p_M` or
`⌊p_M/2⌋` workers. `--space small` is the plan's daily subset: the tiles
`{1, 4} × {8, 16}`, no k-block or 128, the orders `IJ JI KIJ IKJ IJK`, and
the splits at depths 0 and 1 over `p_M`. A point is kept if it lowers at the
launch's shapes. Condition 8 refuses a split whose loop is not parallel or
has one iteration: at `B` = 1, a split at depth 0. At the shapes of plan §3.4,
per type, the entry has twelve nests and 3 305 points at `B` = 1 and 4 857 at
`B` = 16. At `B` = 16 in `f32` that is the scores `ein` 1 515, the values
1 717, the `Linear` dot 1 563, the softmax 14 and each elementwise nest 7,
where §3.4 counted 2 135, 1 869, 1 869, 14 and 7 before the splits were
checked against the kernels. Point 0 is the default layout with one lane, one
worker and no pack: 0.6.0's scalar sequential kernel. Every other point is
compared with it bit for bit, natively, and with `--interp` on the kernel
interpreter too. Each executor is compared with its own point 0, since the
two take `exp` from different libraries (D59).

**The record.** `presk autotune` writes one JSON line per point: the layout,
every timed launch, the median and the MAD, and the prediction. For each of
the point's kernels it also records the model's group, its inputs, the terms
of the prediction (`λ`, spawn, work, L2, L3, memory) and the bound that set
it. Plan §3.6 asked for "the prediction with its components", and step 6's
per-class table needs to say which term fails.

**The rank.** A close pair is two points of one nest, at one shape and type,
whose layouts differ in exactly one member (`tile`, `kblock`, `order`,
`lanes`, `split`, `pack` or `passes`). A row-wise point differs from a tiled
one in several members, so it is in no close pair with one. The rule for
*distinguished* is question 65's, and *decided* (the predicted ratio outside
`1 ± u`) is question 69's. Spearman is taken over the points both days
measured, against the mean of the two medians. The top-k regret uses the
points within `1 + u` of the model's best. The **decision cost** compares
the two sides over the same points. The static side is the capture plus the
lowering and costing of every point, which is what `presk plan` does. The
exhaustive side is all of `presk autotune` except the interpreter's check,
which belongs to statement L and not to tuning. Both come from `sweep.json`,
so every number of `out/ra4.md` traces to a file. `experiments/ra4.sh` also
keeps each `presk plan --cost` run's own output beside its sweep
(`plan.txt`, its last line the wall time), as §3.6 defines the cost.
`presk plan` gives each nest one layout, chosen at the first shape the nest
launches at.

**First runs.** `make ra4-small` sweeps the tutorial's attention (`D` = 8,
so `K` = 4) at `B` = 1 and 16, `T` = 256, in `f32` and `f64`, twice: 312
points a day, every one bit-identical natively and, on day one, on the
kernel interpreter, in 55 s. Ranked: 906 close pairs, 570 distinguished,
75.3 % ordered right, 84.5 % of the decided ones, 33.2 % undecided. By
class: lanes 100 %, passes 100 %, split 88.8 %, tile 65.6 %. Order is right
on 30 % and pack on 0 %, and the model decides no pair of either: at `K` = 4
it predicts them equal. These sizes test the tools, not RA4.

**What the first runs say about step 6.** The full space of the course GPT's
scores `ein` in `f32`, natively and pinned, took 18.2 s at `B` = 1 (1 049
points) and 348.9 s at `B` = 16 (1 515 points). The same points cost 0.32 s
and 0.97 s statically (capture included), ratios of 57 and 360. The eight
launches a point account for 16.6 s and 313 s of those sweeps, so
compilation is about a millisecond a point, and the sweep's cost is the
kernels' own run time. Statement N asks 10³ for the whole entry. The capture
is paid once per entry, so the whole entry's ratio will lie between the
per-nest ratio and 915 at `B` = 16 (348 s against 0.38 s of costing alone),
and be of the order of 10² at `B` = 1. **N is at risk at `B` = 16 and will fail at `B` = 1**
unless the costing gets cheaper: it lowers every point to its kernel IR and
walks it, about 0.2 ms a point. The kernel interpreter does about 40 M
multiply-adds a second (the whole entry: 4.7 s at `B` = 1, 18.8 s at
`B` = 4). Checking every point on it would take about 40 minutes per type at
`B` = 1 and about 15 hours per type at `B` = 16. So `make ra4` checks the
interpreter at the shapes of `RA4_INTERP`, `B` = 1 by default, and every
shape natively. That is a narrowing of statement L, which says "natively and
on the kernel interpreter" at every point. It is for the user to accept or
to pay for with a third night (`RA4_INTERP=1`).

Alternatives considered: timing through the entry, which the plan ruled out
and which would make a point's time depend on the other nests; capturing
every launch rather than the first per shape, which gives the same kernels
and more memory; taking the decision cost from the `presk plan` process's
wall time, which is kept beside the sweep but is not a JSON record.

## D62 — The RA4 measurement: every point exact, the ranking fails on what the model cannot see (2026-09-24)

Plan 0.7.0 step 6. The measurement was run on falicon over one night. The
statements are decided by `presk rank` in `out/ra4.md`, whose copy is
`docs/ra4-2026-09-24.md`; the 20-core sweep's report is
`docs/ra4-20cores-2026-09-24.md`.

**What was run** (`make ra4`, `ra4-20`, `ra4-check`, `ra4-day2`, commit
`fe5634b`):

- **Day one**, 2026-09-23, 20:56–21:49, pinned to the ten X925 cores. The
  course GPT's `SelfAttention.apply` at the constants of plan §3.4 (`B` = 1
  and 16, `T` = 256, `f32` and `f64`, the whole space: 3 305 and 4 857
  points a type). Then the GEMM at 1024³ in `f32` (1 333 points), and the
  BERT-large-shaped attention (`D` = 1024, `H` = 16, `T` = 384, `B` = 1,
  `f32`) on 41 points per contraction nest: point 0, the model's eight best
  and thirty-two drawn by the seed. Each nest also timed 0.6.0's own point,
  the default layout with its split left to the executor.
- **The 20-core sweep**, 21:49–22:41. It was calibrated on all twenty cores
  (`out/profile-20.json`: u = 0.201, Spearman 0.979 over 576 samples, the
  machine read as `cpu0`, an A725) and swept the attention once.
- **The check of statement L**, 22:41–01:05. Every point of day one ran
  again, untimed, natively and on the kernel interpreter, in 130 shards,
  twenty at a time on every core (about 45 CPU-hours).
- **Day two**, 2026-09-24, 01:05–01:59: day one repeated. It ran four hours
  after day one, in the same night, and not on another day as §3.6 asks.
  The two days agree within 0.1 % in each configuration's sweep time (the
  attention at B = 16 in f32: 1 046.8 s and 1 045.9 s; the GEMM: 700.9 s
  and 700.5 s). 137 307 of the 160 509 close pairs were distinguished.

**Two corrections to `presk rank`, made before the statements were read.**
The first reading of the two days counted a pair the model predicts equal
as right whenever the key order agreed with the measurement: Rust's
`0.0.signum()` is `1.0`. A model that predicts two variants equal has not
ordered them, so such a pair is now *tied* and not right. The report gives
the tied fraction, and the pairs the model *decides* (outside `1 ± u`) are
unchanged. The first reading also took the model's pick as the first
*key* among tied best predictions, where `presk plan` takes the first
point in the space's order. The pick is `presk plan`'s now. Both reports
were regenerated from the same measurements, and question 70 asks the
specification to state how ties count. The first reading gave M 73.9 % and
74.2 %, and O 1.73, 1.25 and 1.41. Every verdict is the same under both
readings.

**The statements** (plan §3.7):

| | measured | verdict |
|---|---|---|
| L | 17 879 of 17 879 points and 0.6.0 points of day one checked on the interpreter; no output differing in day one, day two, the 20-core sweep or the check, natively or on the interpreter | **holds** |
| M | the scores `ein` (`%13`): 47.9 % of 12 925 distinguished pairs; the values `ein` (`%23`): 43.9 % of 14 868 | **fails** |
| N | 582–717× at B = 16, 75–89× at B = 1, per configuration and day | **fails** |
| O | top-1 regret at B = 16, f32: `Linear` 2.87, scores 2.18, values 3.01 | **fails** |
| P | best GEMM point 4.67 ms (460 GFLOP/s) against 0.6.0's 25.5 ms (84 GFLOP/s): 5.49× | **holds** |

M fails under any reading. Of the two `ein`s' distinguished pairs, 37.4 %
and 46.1 % are ties. Leaving the ties out, the model orders 76.5 % and
81.4 % of the rest right; on the pairs it decides, 79.0 % and 82.1 %. No
contraction nest reaches 90 % on any reading, at any shape or type. The
closest is 88.1 %, on the decided pairs of the `f64` values `ein` and of
the `f64` `Linear` dot at B = 1.

**Why it fails: the model cannot see a register tile.** It is right on the
decisions its terms read: lanes (93.2 % right, the rest ties), threads
(split 96.7 %) and softmax passes (100 %), that is `π` per type, lanes and
workers, and the spawn cost. It is blind to the rest:

| class | right | tied |
|---|---|---|
| tile | 27.7 % | 46.7 % |
| order | 30.0 % | 61.8 % |
| k-block | 0.6 % | 97.0 % |
| pack | 57.8 % | 0 % |

The terms each point records say why. At B = 16 in `f32` every point of
both `ein`s is predicted work-bound (1 515 of 1 515 and 1 717 of 1 717): the traffic
terms never bind. The work count weighs a load zero (`cost.rs`, as 0.6.0's
roofline did), and traffic is counted from L2, where every tile of a form
has the same footprint. The values `ein`'s tiles `[1, 4]`, `[1, 16]` and
`[2, 8]` have the same work (8.18·10⁷), the same traffic (9.49 MB) and the
same prediction (2.27 ms), and they measure 7.34, 3.39 and 4.66 ms. The
four-row tiles add a little edge work (8.81·10⁷), so where the predictions
of a tile pair differ the model prefers the smaller tile. On day one's
three contraction nests at B = 16 in `f32`, it is wrong on 57–63 % of
those pairs. What a register tile saves is loads from L1 into
registers per multiply-add, and no term counts them. That is plan §7's
first risk ("the model cannot separate tilings without a cache term")
at the one level the model does not have. The calibration saw it too: its
two tiles, `[1, 8]` and `[4, 16]`, share a group and the same work, and
differ by a median of 1.26× (a quarter of the 288 pairs by more than
1.85×), part of what `u` = 0.185 is. The traffic terms never binding also makes k-blocks and loop orders
cost nothing. The pack is charged its copy and credited nothing for the
contiguous loads it buys.

**The same picture elsewhere.** On twenty cores: M 48.4 % and 45.8 %, O
4.13, 2.31 and 2.80, N 91–762×, split 93.6 % right. Heterogeneous cores
did not make the split term unlearnable. The BERT-shaped sample gives 13–17
distinguished close pairs per nest, too few to rank, and top-1 regrets of
1.90–4.96.

**What does hold.**

- *The model is a good pruner.* The best measured point is always among
  the points it predicts within `1 + u` of its best. The top-k regret is
  1.00 on all four contraction nests at B = 16 in `f32`, and at most 1.08
  anywhere at the attention's shapes, with `k` = 36–201 of 1 333–1 717
  points. Measuring the model's candidates alone finds the best point at
  2.4–15 % of the sweep's cost. RA4 does not claim this, but it is what the
  data supports.
- *The space pays.* The best point beats 0.6.0's by 9.2× and 16.5× on the
  `ein`s at B = 16 in `f32`, 47× and 77× at B = 1, where 0.6.0 ran them on
  one worker with the tile `(1, 1)`, 2.6× on the `Linear` dot and 5.5× on
  the GEMM (P). The model's own pick on the GEMM (28.6 ms, the tile
  `[1, 4]`) is slower than 0.6.0's point.
- *N* fails because a point's sweep is short. Eight launches of a few
  milliseconds against about 0.2 ms to lower and cost it, plus one capture
  per entry. The ratio grows with the kernel: the GEMM at 1024³ reaches
  1 460–1 538×.

**What it suggests, not done here.** An L1 term, loads into registers per
multiply-add against a load bandwidth or a load-port count, would give the
tiles and the pack a cost. Fitted on the GEMM, it could be judged on these
measurements without measuring again: each point's kernel is lowered again
from its layout, and its times are on file. But the model would then be designed after
seeing where this one failed, so it would need a fresh sweep, new shapes
or another machine, to be a test and not a fit. That is the user's call:
0.7.0's statements are decided as they stand.

Alternatives considered: counting ties as half right, which gives M about
66 % and 67 % and is still a failure; breaking the pick's ties by the
model's second-best term, which is a different chooser from `presk plan`'s;
and re-running day two on a later day, which the 0.1 % agreement of the two
days did not seem to need but which `make ra4-day2` allows, `presk rank`
regenerating the report.

## D63 — Version 0.8.0 re-tests RA4; 0.7.0's measurements archived as its development set (2026-09-24)

Plan 0.8.0 step 0 (`presk-specs/implementation/presk-plan-0.8.0.md`,
presk-specs `90a24d2`). Five decisions of the user frame the version:

1. **0.8.0 re-tests RA4**; the GPU and the matrix unit come later, 0.9.0 at
   the earliest. This reverses plan 0.7.0 §8.
2. **The full model change, frozen first.** A load term `N_ld/ρ_g`, one
   worker's register loads and stores (spills among them) against a load
   rate fitted per group; a combination of the terms other than the
   maximum (a two-term sum, or a p-norm whose exponent is fitted), chosen on
   the development set; and a calibration that identifies the traffic
   constants (tiles `[1, 4]`, `[1, 8]`, `[2, 8]`, `[2, 16]` and `[4, 16]`,
   sizes up to 2048, two thin products). It is designed on 0.7.0's
   measurements alone and frozen in one commit tagged `ra4-model-0.8.0` at
   the end of step 4, which is time-boxed to three weeks, before any new
   measurement is taken.
3. **Two fresh measurements**, each on two different days: F1, new shapes on
   the ten X925 cores (BERT-large's attention, `D = 1024`, `H = 16`, `T =
   384`, `B = 1`, swept whole; a head dimension of 128, `D = 1024`, `H = 8`,
   `T = 512`, `B = 2`), and F2, 0.7.0's shapes on the ten A725 cores with
   their own calibration.
4. **A statement Q** for the hybrid chooser: measuring only the points the
   model predicts within `1 + u` of its best finds a point within 1.05× of
   the best measured, at no more than 10 % of the sweep's time. M, N and O
   keep 0.7.0's thresholds, on measurements the model never saw; P is
   retired; D reports the development figures and decides nothing.
5. **MDH and ATF deferred again**: the baseline stays the exhaustive
   in-house sweep of the same space.

**The development set, archived.** 0.7.0's raw measurements were only in
`out/`, which git ignores and which the experiment targets rewrite: `make
ra4`, `ra4-day2`, `ra4-check` and `ra4-20` write the same directories, and
`make calibrate` starts with `rm -f out/profile.json`, the file that holds
0.7.0's fitted layout model and its 576 calibration samples. Before anything
else in this step, they were copied to `~/presk-data/ra4-0.7.0/` on falicon,
outside the repository and read-only, under their `out/...` paths:
`out/ra4/day1`, `day2` and `check`, `out/ra4-20/day1`, and also
`out/profile.json`, `out/profile-20.json`, `out/ra4.md` and `out/ra4-20.md`,
which the plan's list does not name but which re-pricing "with 0.7.0's
profile" needs. That is 1 605 files and 75 835 032 bytes. A checksum list,
`SHA256SUMS`, was computed on the source and checked on the copy; its own
SHA-256 is `e73fffc0ca796b3cfd8ef09a2b674c271d1bf3ace05058f0d1b8b571fede80b7`.
A `README.md` there gives each directory's provenance. `out/` itself is left
as it was.

The check that the archive holds what the ranking reads: `presk rank` at
v0.7.0, run from the archive's root (a report quotes its day paths),
regenerates `docs/ra4-2026-09-24.md` and `docs/ra4-20cores-2026-09-24.md`
byte for byte. The step's own check, `make ra4-dev` with 0.7.0's profile
regenerating the first of them, is not done here: it re-lowers every point
and prices it again, which is `presk reprice`, step 1's tool, and it is
step 1's check.

**A sentence of the roadmap corrected.** `docs/kernels-roadmap.md`'s design
for a register-load term asked, for loop orders and k-blocks, for "a
footprint analysis that starts at L1 (the bytes from L2 into L1, `Q_L1`)".
The model has counted those bytes since 0.7.0 step 4 (D60): `Q_L2` is
`traffic(worker, m.l1)`, the bytes a cache of L1's size brings in, named
after the level they come from. It also tells the orders apart. At B = 16 in
`f32`, on the development set, all 133 sets of the values `ein`'s points
that differ only in their order and k-block have more than one `Q_L2`, and so
do 187 of the scores `ein`'s 203. But all 1 717 and 1 515 points are
predicted work-bound, and the maximum hides the traffic terms. What hides
the orders is the combination, as plan 0.8.0 §1 says, and the note now says
so. The same note, and D62, also said that "traffic is counted from L2, where
every tile of a form has the same footprint". The first half is right in
that nothing counts the loads from L1 into registers. The second is not
exact: the tile's block loops change the footprints, and points that differ
only in their tile have different `Q_L2` in 27 of the scores `ein`'s 156 such
sets and in 90 of the values `ein`'s 180. Their predictions tie because every
point is work-bound, with one work count for all its tiles but D62's edge
work. The note is corrected; D62 stays as recorded. Its pack paragraph
pointed to the removed request. What a pack buys at one lane is contiguous
loads, which no traffic term sees, since a footprint counts elements, not
cache lines.

**Bookkeeping.** Version `0.8.0-dev`; the CHANGELOG's "Unreleased" section;
spec questions 71 (what makes a cost model static, and what the measurements
that test it may have been used for) and 72 (a chooser that measures among
statically pruned candidates) drafted from plan §6. `RA4_CORES_SMALL =
0-4,10-14` beside `RA4_CORES`: the Cortex-A725 cores, CPU part `0xd87`,
2.8 GHz, 512 KiB of L2 each where an X925 has 2 MiB (`/proc/cpuinfo`, sysfs,
read on 2026-09-24). No target uses the variable yet: F2 does (steps 5–6).
The manual's `status` chapter gains 0.8.0's table and statements. No change
to the cost model's code. The tutorial still quotes `presk --help` as 0.7.0
prints it; step 8 refreshes its outputs, as 0.7.0's step 7 did.

Alternatives considered: committing the raw measurements (78 MB of JSON lines
in 1 605 files, into a code repository; the checksum list's hash, recorded
here, lets any copy be checked instead); a git-ignored directory inside the
repository, which `git clean -x` reaches; archiving only the three directories
the plan lists, which would have left the profile that the re-pricing check
needs one `make calibrate` from being lost.

## D64 — `presk reprice`: 0.7.0's predictions reproduced from its measurements, and a static side twice as cheap (2026-09-24)

Plan 0.8.0 step 1. Three tools, and the check step 0 left to this step.

**`presk reprice`.** It prices a recorded day again under another profile
without measuring anything, so that a candidate model is judged on 0.7.0's
measurements in the time it takes to cost them (plan §3.4). For each sweep of
the entry under the day, it captures the entry once, lowers every recorded
point again from its `layout`, costs it, and writes the line again with
`predicted` and the per-kernel records (cost, group, terms, bound) replaced.
Four choices:

- The configuration comes from the sweep's `sweep.json` (constants, dims,
  realization), not from `--dim` and `--const` as the plan's synopsis has it.
  A day records what it was swept at, and pricing it at other constants
  would price points against launches they were not measured at. The file
  and the entry stay on the command line, since `sweep.json` does not record
  the file. A day that holds sweeps of two entries, the attention and the
  GEMM, takes one call per entry.
- The worker count is the sweep's (`machine` in `sweep.json`): it decided
  which splits were well-formed and how wide the executor's default split
  was. The model's machine is the profile's, as in `presk autotune`.
- Every other field is copied **as written**. `serde_json` parses floats
  best-effort by default, so parsing a line and printing it again could move
  a time's last digit. `presk reprice` reads a line as raw fields (the
  `raw_value` feature, in `presk-cli` only) and replaces two of them. The
  alternative, the `float_roundtrip` feature, parses exactly, but it is
  global: every profile would be read with it too. Tried on the development
  set, it moved 127 of the 1 564 predictions of one nest by one ulp (up to
  1.9·10⁻¹⁶), because 0.7.0 read its profile's constants with the default
  parser. So a model's re-pricing must read its profile as the sweep did.
- `sweep.json` keeps the sweep's own capture and costing times, which `presk
  rank` reads for statement N, and gains the profile's `u` and a `repriced`
  record: the profile, its `u`, the points, and how long the capture and
  the costing took. `presk plan`'s files are not copied: they are the old
  model's choice.

**The check.** `make ra4-dev` (`experiments/ra4_dev.sh`) re-prices the
archived days (D63) into `out/ra4-dev/out/…` with 0.7.0's two profiles and
ranks them from `out/ra4-dev`, so that each report quotes its days as 0.7.0's
did. It first checks the archive against its checksums. **Every one of the
52 142 re-priced lines is the archived one, byte for byte** (day one and day
two of the X925 sweep, 16 551 attention and 1 334 GEMM lines each; the
20-core day, 16 372). **Both reports are the committed ones:**
`docs/ra4-2026-09-24.md` and `docs/ra4-20cores-2026-09-24.md`. That is step
0's deferred check and this step's, in 28 s. The 20-core day tests the worker
count taken from the sweep, since its splits are over 10 and 20 workers and
its profile's machine is `cpu0`, an A725. Under any other profile the script
re-prices and ranks without comparing.

**A cheaper static side.** `presk plan` and `presk autotune` now lower a
layout once for all its thread splits (`lower_launch_splits`). No lowering
reads the split and neither do the layout's checks: it only sets the main
kernel's `threads`, so each split is that lowering with its threads set and
checked as before (`split_as`). That alone saved 6 %. Timers put the rest of
the costing in `traffic`: 70–80 % of it, three walks per kernel (into L1,
into L2, into a worker's share of L3), each recomputing footprints that do
not depend on the capacity. `traffic_at` walks once per capacity with a
shared memo of footprints, keyed by where the statements start in the kernel
and the buffer. Each walk adds the same terms in the same order, collected
and then summed as `.sum()` summed them, so that an empty sum keeps its
sign. (`perf` is closed to users on falicon, `perf_event_paranoid` = 4.)
`presk plan` on the course GPT's attention, pinned to the X925 cores, median
of five runs against the v0.8.0-dev `main` binary:

| config | points | costing, before → after | capture | static side, before → after |
|---|---|---|---|---|
| B = 1, `f32` | 3 305 | 0.438 → 0.255 s (1.72×) | 0.15 s | 0.58 → 0.40 s |
| B = 1, `f64` | 3 305 | 0.607 → 0.341 s (1.78×) | 0.17 s | 0.78 → 0.52 s |
| B = 16, `f32` | 4 857 | 0.928 → 0.472 s (1.97×) | 0.57 s | 1.48 → 1.05 s |
| B = 16, `f64` | 4 857 | 1.312 → 0.600 s (2.19×) | 0.87 s | 2.20 → 1.47 s |

Every prediction is unchanged: `make ra4-dev` compares against lines that
0.7.0's code computed. Unit tests also compare, on the tutorial's
attention, every point of every nest with 0.7.0's one-`lower_launch`-per-point
lowering (same layouts, same kernels, same order), and `traffic_at` with one
`traffic` call per capacity.

**What is left of the static side is the capture.** At B = 16 in `f32` it is
0.58 s of 1.05 s: synthesizing the inputs in exact arithmetic (0.17 s),
converting them (0.10 s), then compiling and running the entry (0.32 s).
Choosing statically needs the launches' shapes, not their data, so a capture
that walks the host program without running a kernel could remove most of
it. The plan does not ask for it and this step does not do it. For statement
N it means that the sweep times of 0.7.0 (1 048 s and 1 284 s at B = 16, 58
and 67 s at B = 1) would give ratios of about 10³ at B = 16 in `f32`, 870 in
`f64`, and 130–150 at B = 1. N at B = 1 still fails, as plan §3.6 expects.
Those ratios are estimates from the development set, not a verdict.

**`presk rank --frozen TAG`.** Before reading anything, it refuses a day
whose `day.txt` names no commit, or a commit that does not descend from
`TAG` (`git merge-base --is-ancestor`, in the repository it runs in), or a
commit marked `-dirty`. Every run of `experiments/ra4.sh` into a day adds its
commit, and every one is checked. The script now marks the commit `-dirty`
when `git status --porcelain` is not empty, since a commit does not name the
code that ran if the tree differed from it. A unit test builds a repository
with a commit before the tag, the tag and a commit after it, and checks each
case. The tag itself is step 4's.

Alternatives considered: re-pricing inside `presk rank`, one command instead
of two (a re-priced day on disk can be compared with the measured one and
ranked like it, which is how the check above works); keeping `--dim` and
`--const` on `reprice` (see above); a capture of shapes only (see above; the
user's call).

## D65 — Statement N's static side is a capture of shapes: the host program on zero inputs, no kernel run (2026-09-24)

Plan 0.8.0 step 1, decided by the user before any fresh measurement.

**The decision.** N compares what choosing statically costs with what the
exhaustive sweep costs. Until now the static side included a native run of
the entry: inputs drawn in exact arithmetic (the parameters by their
modules' initializers), converted to machine types, and every kernel
compiled and run. That was because `presk plan` reused `presk autotune`'s
capture, and autotune needs values to time a point on them. Choosing needs
only the launches' shapes. On 2026-09-24 the user decided that N's static
side is what choosing needs: the launches captured from their shapes, then
every point lowered and costed. The capture that runs the entry is reported
beside the ratio. The reason is RA4's own wording, the cost of deciding
statically: a compiler choosing a layout does not run the program. The
decision was taken knowing that the development set put N near 10³ at
B = 16 in `f32` (D64). The entry-run ratio stays in every report, so that a
reader sees what the rule changes. Plan 0.8.0 §3.6 is amended to say so.

**The capture of shapes** (`ra4::Capture::Shapes`). It is the kernel
backend with the native backend's defaults and an executor that runs
nothing. Its inputs come from `presk_interp::run::prepare_shapes`, a sibling
of `prepare_run`: the same entry, dimensions, keys and scalars, with each
tensor described by its axes and each parameter tree by its module's
initializer, dimensions and key. The code that binds an initializer's
dimensions is now shared with `synthesize`. The tensors are built as zeros
in the realization's type. Each tree is built by calling its initializer on
the backend that runs nothing, so its tensors are zeros of their shapes
too. Nothing is computed in exact arithmetic. A realization with integer
weights keeps drawn inputs, since zeros have no scale. Every launch after
the first has zero inputs, which no static use reads. `presk autotune` keeps
the native capture for its buffers.

**Checks.**

1. A unit test: on the ten entries of `tests/plan/entries.txt`, in `f32`,
   `f64` and `bf16`, both captures record the same launches. That is, the
   same nests, operations, operands, types, orders and accumulation types,
   and input buffers of the same kinds and lengths.
2. `presk plan --cost` prints the same prediction for every point, and
   writes the same plan, as `main`'s binary. Checked at B = 1 in `f32`, at
   B = 16 in `f32` and `f64`, and at the fresh shapes F1a and F1b.
3. `make ra4-dev` is unchanged: 52 142 lines and both reports, byte for
   byte, with `presk reprice` now capturing shapes.

**Where it is used.** `presk plan` and `presk reprice` capture shapes.
`presk autotune` still runs the entry. After it has taken the sweep's wall
time, it captures shapes and records how long that took as
`static_capture_seconds` in `sweep.json`. When a sweep records it, `presk
rank` takes N's static side as that capture plus the costing, and prints
the entry-run ratio beside it: in N's cell, and in two more columns of the
decision cost table. A sweep without it, 0.7.0's and so the development
set, is costed as before, and its report is unchanged byte for byte.

**Numbers.** `presk plan` on the course GPT's attention, pinned to the X925
cores, median of five runs, `main` → this branch:

| config | points | costing | capture | static side |
|---|---|---|---|---|
| B = 1, `f32` | 3 305 | 0.445 → 0.251 s | 0.143 → 0.024 s | 0.588 → 0.275 s (2.14×) |
| B = 1, `f64` | 3 305 | 0.616 → 0.343 s | 0.171 → 0.037 s | 0.787 → 0.380 s (2.07×) |
| B = 16, `f32` | 4 857 | 0.942 → 0.357 s | 0.568 → 0.184 s | 1.510 → 0.541 s (2.79×) |
| B = 16, `f64` | 4 857 | 1.318 → 0.467 s | 0.884 → 0.462 s | 2.202 → 0.929 s (2.37×) |
| F1a, `f32` (D = 1024, H = 16, B = 1, T = 384) | 3 305 | 0.575 → 0.296 s | 1.183 → 0.156 s | 1.758 → 0.452 s (3.89×) |
| F1b, `f32` (D = 1024, H = 8, B = 2, T = 512) | 5 959 | 1.455 → 0.526 s | 1.486 → 0.201 s | 2.941 → 0.727 s (4.05×) |

At F1a most of the old capture was the initializers drawing 4 M weights in
exact arithmetic. The costing is also faster in a process that has not run
the entry natively (0.36 s against 0.47 s at B = 16 in `f32`). `presk
autotune` costs after its native run, so the static side it records is the
larger of the two. On 0.7.0's sweep times these figures would put N at about
1.9·10³ at B = 16 in `f32`, 1.4·10³ in `f64`, and 175–210 at B = 1, where N
still fails, as plan §3.6 expects. Those ratios are estimates from the
development set, not verdicts. In `f64` the capture keeps 0.46 s at B = 16:
the host still converts and copies buffers at each kernel's edge, and
`f64` buffers are twice as wide. Removing that would take a host that walks
shapes without buffers, which is not done.

Alternatives considered:

- Keeping the native run and reporting the capture apart only, as the
  plan's first wording had it. N would then keep measuring the tooling, not
  the decision.
- Running the initializers in exact arithmetic and zeroing only the data
  tensors. At D = 1024 the initializers were 0.37 s of a 0.64 s capture.
- A host that walks shapes without buffers. It would remove the rest of the
  `f64` capture, at the price of a second interpretation of the host
  program to keep in step with the first.

## D66 — The load term: register loads and stores against a rate per group, and 0.7.0's model kept bit for bit without it (2026-09-24)

Plan 0.8.0 step 2 (§3.1).

**The count.** `presk_kernel::cost::kernel_loads` counts every `Load`,
`LoadI`, `VLoad`, `Store`, `StoreI` and `VStore` a kernel executes, once
whatever its lanes, times the trip counts of its loops. It uses its own walk,
beside the one that counts work, for two reasons: work gives a load no weight
and does not look inside a load's index, and a gather's index load is a
load. A broadcast of a loaded scalar adds nothing to its load, and a branch
is counted as taken, as work is. `planned_cost` records one worker's count
as `PlannedCost.loads`, on the kernel the split has cut.

**The spill moves.** 0.7.0 charged the two level-1 accesses per spilled
register per iteration of the busiest loop as work. They are now
`PlannedCost.spill`, apart from `work`, and a model with the load term charges
them as loads.

**The model.** `LayoutFit.rho` is a rate per group, fitted as `π` is, and
the maximum gains `(N_ld + S)/ρ_g`. A model with an empty `rho` has no load
term and charges `(W + S)/π_g`. That is the sum 0.7.0 formed, `work +
spill`, with the same operands in the same order, so a profile written by
0.7.0 predicts exactly what it predicted. A sample in such a profile has no
`loads` or `spill` field, which reads as zero, and its `work` already holds
its spill, so it means what it meant. `Terms.loads` is absent from a model
without the term, and `bound()` can name `loads`.

**The fit.** `fit_layout(samples, LoadTerm)` fits `ρ` per group (`Group`),
once for all groups (`Machine`, for a calibration that cannot tell their
rates apart), or not at all (`None`), with the other constants, by the same
coordinate descent. Each rate starts at the largest `(N_ld + S)/t` of its
samples. `presk bench --calibrate --space calib --load-term` chooses, and
`group` is the default. A calibration into an existing profile drops that
profile's samples recorded before this step. A contraction always loads, so
a sample with no load count is one whose `work` includes its spill. Fitting
it beside new ones would mix the two meanings.

**Checks.**

- `the_load_count_reads_the_micro_kernel`, on the values `ein` of
  `tests/plan/contractions.psk` (I = J = K = 32, four heads) with four lanes.
  The six tiles of plan §1 count `blocks · (K·(m + n/4) + m·n/4)` exactly:
  per `k`, `m` broadcasts and `n/4` vector loads, then `m·n/4` vector
  stores, which is 2, 1.5, 1.5, 1.25, 1 and 0.75 loads per vector
  multiply-add. The pack of the NT scores `ein` counts one load and one
  store per element, and the packed contraction the NN micro-kernel's
  count. A `[4, 16]` tile at one lane keeps 84 values in its inner loop, 52
  over 32 registers, a spill of `2·52·2048`.
- `the_layout_fit_recovers_a_load_rate`: kernels bound by each term in turn,
  the load term among them, over three groups. The fit recovers every `ρ`,
  `π` and `β` within 10 %, u < 0.02, and one shared `ρ` with `Machine`. The
  terms add up to the prediction and name the bound.
- `a_model_without_a_load_rate_predicts_as_0_7_0`, on a sample recorded now
  and on one in 0.7.0's JSON form. 0.7.0's synthetic test runs with
  `LoadTerm::None`.
- `make ra4-dev`: 0.7.0's profiles re-price the development set to the same
  prediction at every point, and both reports byte for byte. The lines
  themselves are no longer the archived ones: their per-kernel records gain
  `loads` and `spill`, and a spilled kernel's `work` no longer includes its
  spill. So `experiments/ra4_dev.sh` now compares each line with its
  `kernels` field left out, which keeps the prediction, the times and the
  bits, byte for byte.

**On the development set**, what the count sees, which decides nothing.
Take the values `ein` at B = 16 in `f32`, four lanes, split at depth 2 over
ten workers: 90 points. Spearman between one worker's `N_ld` and the mean of
the two days' medians is 0.953, where 0.7.0's prediction reaches 0.437.
Among the thirty points that model predicted equal (2.27 ms, plan §1's
table) it is 0.943. The fewest-loads variant of each tile measures, in order,
`[1, 4]` 20.5 M loads a worker and 6.9 ms, `[1, 8]` and `[2, 4]` 15.4 M and
4.6 ms, `[4, 4]` 13.8 M and 3.9 ms, `[1, 16]` 12.8 M and 3.4 ms, `[2, 8]`
10.3 M and 3.1 ms, `[4, 8]` 8.3 M and 2.9 ms, `[2, 16]` 7.7 M and 2.5 ms, and
`[4, 16]` 5.5 M and 2.4 ms. A k-block that keeps its running sums in the
output reads and writes them back at every block, and the count sees that
too: those variants carry up to 2.5 times the loads and measure slower. This
is the data the term was designed on. It shows that the term is built as
designed; the fresh sweeps are the test.

**Not in this step.** `ρ` is not identified yet: 0.7.0's calibration has two
tiles, and step 3 widens it and chooses the combination. A smoke run shows
it. The GEMM at 256 and 512 in `f32`, pinned to the X925 cores, 192
samples, fits `u` = 0.088 and Spearman 0.986. Then:

- With four lanes the load term binds on 39 of the 48 `[1, 8]` samples, and
  `ρ` comes out 3.7–4.4·10⁹ loads/s.
- At one lane the work term binds on every sample, so the fitted `ρ`
  (1.1–1.2·10¹⁰) is only a value large enough not to bind, not a rate.

The other samples are bound by the work or by the traffic from L2 or L3. The default
`--load-term group` means that `make calibrate-ra4` and the tutorial's
calibration now fit a `ρ` on those two tiles; the tutorial's quoted outputs
are refreshed in step 8. Nothing is frozen until step 4.

Alternatives considered:

- Keeping the spill inside `work` and adding the load term beside it. The
  spill would then be charged twice in a model with the term, or subtracted
  back out of `work`, which is not exact in floating point.
- One `ρ` per machine only. The groups differ in lanes and workers, as
  their `π` do, and `Machine` is there for a calibration that cannot tell
  them apart.
- Counting a vector load as its lanes. It is one instruction, and the plan's
  §3.1 counts it once.

## D67 — How the terms combine, `presk refit`, and a wider calibration that does not identify the memory constant on this machine (2026-09-24)

Plan 0.8.0 step 3 (§3.2, §3.3).

**The combination.** `LayoutFit.combine` takes one of three forms:

- **max**, the default. A profile written before reads as it, and predicts
  bit for bit what it predicted.
- **sum**: `max(work, loads) + max(l2, l3, mem)`.
- **p-norm**: `(Σ tᵢ^q)^(1/q)`, computed scaled by the largest term so that
  no power under- or overflows, with `q` fitted in [1, 100].

`λ` and the spawn cost are added in every form. `Terms` records the
combination when it is not the maximum, and `bound()` names the largest term.

**The fit.** `fit_layout(samples, load term, combine)`. A p-norm's `q` trades
off against every rate at once, and coordinate descent, one constant at a
time, drifts along that valley. The first attempts gave `q` = 5.3, then 4.1,
for a true 3, at a `u` near the noise. So `q` is searched on its own: ten
values on log q over [1, 100], then ten between the best one's neighbours.
Each value fits the rates again, starting from the maximum's fit, and the ten
fits of a stage run on ten threads.

Two things had to be fixed on the way:

- **The synthetic check failed at first because of the test.** Its samples
  drew the five terms from one fractional sequence with a fixed stride, so
  the terms always stood in the same pattern relative to one another, and a
  p-norm fitted the sum's samples as well as the sum. Drawn independently
  (splitmix64), the sum and a p-norm with `q` = 3 are both recovered: every
  rate within 10 %, and `q` within 10 %.
- **A round of coordinate descent now narrows its grid when it improves the
  error by less than a millionth**, not only when it improves nothing. In
  the p-norm's valley every round improved a little, and every fit ran its
  200 rounds. This changes when every fit stops, the maximum's included, and
  the synthetic tests pass as they did.

**`presk refit P.json`** fits a profile's stored samples again in another
form, writing to `--out` or back. With `--compare`, it fits nine forms (three
load terms under three combinations) and writes nothing. Every form is thus
fitted on the same calibration, and step 4's ablation re-prices under each.
It leaves out samples recorded before step 2, as a calibration does. `presk
bench --calibrate --space calib` gains `--combine`, max by default.

**The calibration space** (plan §3.3) is now 300 layouts per shape and type,
where 0.7.0 had 96:

- the tiles `[1, 4]`, `[1, 8]`, `[2, 8]`, `[2, 16]` and `[4, 16]`;
- no k-block in `IJ` and `JI`, a k-block of 128 in the six orders of `IJK`,
  and of 32 in `IJK` and `KIJ`;
- one lane and the type's width;
- one worker, five and ten.

`make calibrate-ra4` runs it on the GEMM at 256, 512, 1024 and 2048, and on
the thin products 16 × 4096 × 4096 and 4096 × 4096 × 16, in `f32` and
`f64`, then prints `presk refit --compare` and the time taken. The thin
products are the entry `matmul` of a sibling file,
`experiments/matmul.psk`: `gemm.psk`'s file-level `N` would bind every
function of that file. The run gave 3 536 samples: at 4096 × 4096 × 16, 268
of the 300 layouts, since a split of a loop of one iteration is refused.

**The measurement**, on falicon's X925 cores, 2026-09-24:

| load term | combination | u | Spearman | β_L2 | β_L3 | β_mem |
|---|---|---|---|---|---|---|
| none | max | 0.226 | 0.976 | 3.78·10¹⁰ | 1.29·10¹¹ | 2.65·10¹¹ |
| none | sum | 0.226 | 0.978 | 7.43·10¹⁰ | 1.93·10¹¹ | 4.20·10¹¹ |
| none | p-norm, q = 1.00 | 0.224 | 0.979 | 7.92·10¹⁰ | 3.88·10¹¹ | 8.04·10¹¹ |
| group | max | 0.133 | 0.985 | 3.86·10¹⁰ | 1.38·10¹¹ | 2.65·10¹¹ |
| group | sum | 0.137 | 0.985 | 1.69·10¹¹ | 2.50·10¹¹ | 5.51·10¹¹ |
| group | p-norm, q = 1.00 | 0.136 | 0.986 | 1.67·10¹¹ | 4.14·10¹¹ | 8.02·10¹¹ |
| machine | max | 0.149 | 0.980 | 3.76·10¹⁰ | 1.34·10¹¹ | 2.70·10¹¹ |
| machine | sum | 0.163 | 0.980 | 1.04·10¹¹ | 2.24·10¹¹ | 4.64·10¹¹ |
| machine | p-norm, q = 1.00 | 0.167 | 0.983 | 1.15·10¹¹ | 4.08·10¹¹ | 8.55·10¹¹ |

`λ` comes out 0 in every form, and the spawn cost 138–190 µs.

1. **The load term cuts `u` from 0.226 to 0.133**, and a rate per group
   beats one for the machine (0.149).
2. **The three combinations fit the calibration about equally**: 0.133,
   0.137 and 0.136 with the load term. The p-norm's `q` settles at 1.00, the
   edge of its range, which is the sum of all five terms, under every load
   term. As the plan says, the development set's M chooses the form (step 4).
3. **Where the terms bind** in the fitted model (group, max): the work on
   1 806 samples, the loads on 827, the traffic from L2 on 564, from L3 on
   307, from memory on 32.
   - At one lane the work binds on 1 488 of 1 768 samples and the loads
     never, so the one-lane `ρ` is not identified. Yet the scalar tiles
     differ. At 1024³ in `f32`, one worker, the medians over orders are
     `[1, 4]` 302 ms, `[1, 8]` 228, `[2, 16]` 188, `[4, 16]` 187 (it
     spills) and `[2, 8]` 173, and the maximum predicts them alike.
   - With four lanes the load term binds on 531 of 884 samples, and the
     tiles order as their loads: 290, 147, 81, 54 and 45 ms.
   - So a form in which the loads add to the work (sum, p-norm) can see
     the scalar tiles, and the maximum cannot. Step 4 weighs that on the
     development set.
4. **The step's check fails: `β_mem` comes out above `β_L3`**, in every
   form (with the load term and the max, 2.65·10¹¹ against 1.38·10¹¹
   bytes/s).
   - The thin products do not do what §3.3 meant them to. On this machine
     they are bound by their work and loads, not by memory.
   - Their implied memory bandwidth (`p·Q_mem` over the measured time) has
     medians of 2–118 GB/s by type and worker count, where the machine's
     memory gives about 273 GB/s.
   - Sixteen rows of a 4096 × 4096 operand in `f32` is 8 flops per byte of
     it, and ten X925 cores multiply faster than memory feeds them at that
     intensity. Only 32 samples of the calibration are memory-bound.
   - A product bound by memory here needs an intensity below the machine's
     balance point: one or two rows or columns, a matrix–vector product.
     That changes §3.3's space, so it is the user's call.
   - Separately, some 2048² `f64` samples imply 313 GB/s from memory, more
     than it gives. The footprint rule over-counts memory bytes somewhere.
     This is noted, not chased.
5. **Time: 11 597 s**, where §3.3 asked for under an hour.
   - 3 753 s were the kernels' own time (six calls × 300 layouts × twelve
     shapes), 3 195 s of it at 2048.
   - Nearly all the rest was the backend converting the exact inputs to
     machine numbers at every call, outside the kernels it times. It is
     fixed: the inputs are rounded once, as the backend rounds them, and no
     sample changes. At 1024³ in `f32` a calibration's wall time went from
     318 to 166 s.
   - The whole would take about 70 minutes, still over the hour. To fit in
     it, for example, three timed calls instead of five at 2048 (about 52
     minutes). That is the user's call too, with the space.

This calibration is in `out/profile.json` (load term per group, max). It is
not the one step 4 will freeze: that one waits for the decisions above. The
tutorial's step 17 calibrates at 256 and 512, and it now times 300 layouts
where it timed 96. Its outputs are refreshed in step 8.

Alternatives considered:

- Fitting `q` with the other constants by coordinate descent: it drifted
  (above).
- A Levenberg–Marquardt fit: the maximum is not differentiable where it
  changes term, and the grid search treats every form alike.
- Replacing the thin products now: that is the plan's space and the user's
  call; this report says why.

**The user's two decisions** (2026-09-24, after this report):

1. **The thin products stay as §3.3 has them, and `β_mem` is recorded as
   not identifiable on this machine.** The recommendation first made was to
   replace them with 2-row products. A 2 × 4096 × 4096 product in `f32`
   (1 flop per byte, a 64 MB operand far over the L3) was then calibrated
   into a scratch profile, and it is not bound by memory either. Its
   implied bandwidth has a median of 6–8 GB/s on one worker and 32–38 GB/s
   on ten, where memory gives about 273 GB/s. The kernels this compiler
   emits are bound by their loads and loop overhead long before memory, so
   no product identifies `β_mem` here. A 2-column product admits no tile:
   every calibration tile has n ≥ 4. The memory term does not bind on
   these kernels in reality either, and a large, loosely fitted `β_mem`
   keeps it small. No streaming kernel is added to the calibration: it
   would serve only the nests no statement judges, and the calibration
   stays a GEMM.
2. **At 2048 the one-worker layouts are left out.** They are 76 % of the
   kernel time there: 2 433 of 3 196 s. One-worker groups stay calibrated
   at 256–1024 and on the thin products, and 2048's large working sets are
   kept by its 5- and 10-worker layouts. The estimate is about 30 minutes
   on the X925 cores. The A725 cores of step 5, about twice as slow, would
   then take about an hour, where every other option leaves them over. The
   timing stays at five calls and their median.

Plan 0.8.0 §3.3 is amended accordingly in presk-specs at the start of step
4. The 2048 exception is step 4's first change, and the definitive `make
calibrate-ra4` runs before any development work.

## D68 — The definitive calibration on the X925 cores: one run of the amended space, before any development (2026-09-24)

Plan 0.8.0 step 4, its first part: §3.3 as amended in presk-specs on the
user's two decisions (D67).

**The change.**

- `presk bench --space calib --min-workers N` leaves out the layouts stated
  on fewer than `N` workers, and prints the workers it keeps. `make
  calibrate-ra4` passes 2 at the sizes of `CALIB_NO_ONE_WORKER`, 2048. The
  GEMM at 256–1024 and the thin products keep their one-worker layouts.
- **A calibration is one run.** `bench --calibrate` keeps a profile's
  earlier samples unless it times the same layout again, which is how the
  shapes of one calibration add up. Across two runs it also keeps stale
  ones: step 3's one-worker samples at 2048 would have stayed in this fit.
  `make calibrate-ra4` now moves an existing profile to `*-prev.json` first.
- Nothing else changed: not the model, the fit, nor the other layouts of the
  space.

**The run**: falicon's X925 cores (5-9,15-19), 2026-09-24 18:28–19:14, on
presk `5777531` with the step's working-tree diff (SHA-256 `c0c98015…`,
archived with the log). Nothing else ran; the load average was 1.1 before.

- **3 336 samples**: step 3's 3 536 less the 200 one-worker layouts at 2048,
  100 per type. `4096 × 4096 × 16` again gave 268 of its 300.
- **2 693 s (44.9 min)**, where D67 estimated about 30. The kernels' own time
  is 1 329 s, six calls a layout: 760 s at 2048, 314 at 1024, 215 on the thin
  products, 39 at 512 and 256. The other 1 364 s are spent outside the timed
  kernels: lowering, compiling and each call's host work. D67 took that
  share from 1024³ in `f32`; it is larger elsewhere, and it is not broken
  down here. §3.3's hour holds on these cores. On the A725 (step 5), about
  twice as slow, the same shares would give about an hour and a half, not
  an hour.

**The fits** (`presk refit --compare`, the same 3 336 samples):

| load term | combination | u | Spearman | spawn µs | β_L2 | β_L3 | β_mem |
|---|---|---|---|---|---|---|---|
| none | max | 0.241 | 0.970 | 114 | 3.85·10¹⁰ | 1.23·10¹¹ | 2.51·10¹¹ |
| none | sum | 0.234 | 0.971 | 129 | 8.05·10¹⁰ | 1.85·10¹¹ | 3.49·10¹¹ |
| none | p-norm, q = 1.06 | 0.232 | 0.973 | 131 | 7.75·10¹⁰ | 3.23·10¹¹ | 6.93·10¹¹ |
| group | max | **0.142** | 0.980 | 78 | 4.06·10¹⁰ | 1.29·10¹¹ | 2.52·10¹¹ |
| group | sum | 0.141 | 0.981 | 97 | 3.44·10¹¹ | 2.42·10¹¹ | 5.96·10¹¹ |
| group | p-norm, q = 1.06 | 0.143 | 0.982 | 100 | 1.72·10¹¹ | 3.48·10¹¹ | 7.16·10¹¹ |
| machine | max | 0.159 | 0.975 | 92 | 3.89·10¹⁰ | 1.27·10¹¹ | 2.52·10¹¹ |
| machine | sum | 0.172 | 0.974 | 110 | 1.16·10¹¹ | 2.18·10¹¹ | 4.66·10¹¹ |
| machine | p-norm, q = 1.00 | 0.173 | 0.978 | 112 | 1.27·10¹¹ | 3.80·10¹¹ | 8.36·10¹¹ |

`λ` is 0 in every form.

1. **As in step 3, the load term is what improves the fit**: `u` from 0.241
   to 0.142 with a rate per group, 0.159 with one for the machine.
2. **The combinations fit alike again**: 0.142, 0.141 and 0.143. The
   p-norm's `q` settles at 1.06 (1.00 with one `ρ`), next to the sum. The
   development set chooses the form (§3.4).
3. **`β_mem` is above `β_L3` in every form**, 2.52·10¹¹ against 1.29·10¹¹
   bytes/s with the load term and the maximum: not identifiable here, as
   decided. The thin products' implied memory bandwidth has medians of
   2.4–117 GB/s by type and workers, as in step 3.
4. **Where the terms bind** (group, max): the work on 1 883 samples, the
   loads on 740, the traffic from L2 on 350, from L3 on 331, from memory on
   32. At one lane the loads bind on none of 1 668 samples, so `ρ` there is
   still not identified under the maximum. The scalar tiles still differ at
   1024³ in `f32` on one worker (medians over orders): `[1, 4]` 301 ms,
   `[1, 8]` 230, `[2, 16]` 188, `[4, 16]` 191, `[2, 8]` 172. In four lanes
   they order as their loads: 285, 147, 80, 57 and 48 ms.
5. **Against step 3, layout by layout** (the 3 336 layouts both timed): the
   time ratio has a median of 1.002, a median `|r − 1|` of 2.3 % and a 90th
   percentile of 18 %. By shape the median ratio goes from 0.94 (256, `f32`)
   to 1.05 (1024, `f64`). 2048 agrees best, with a median `|r − 1|` of
   1.2–1.3 %, and 256 and 1024 least, 2.7–7.5 %. That is the noise between
   two runs of the calibration on one day. `u` is six times it: most of the residual is the
   model's.
6. **Without the one-worker samples at 2048**, the fit moved a little. `u`
   (group, max) went from 0.133 to 0.142, a median over another set of
   samples. `π` and `ρ` per group are within 9 % of step 3's, except the
   ten-worker `π` in four and two lanes, 15 % lower. The spawn cost fell
   from 138 to 78 µs.

**What this profile is.** It is `out/profile.json`, archived read-only in
`~/presk-data/ra4-0.8.0-calibration/` on falicon, with the log, the diff,
step 3's profile and a `SHA256SUMS` whose own SHA-256 is `d6378e1d…`. The
form fitted into it, a load rate per group under the maximum, is the
default, not a choice. Step 4's development prices the development set
under refits of these samples (`presk refit out/profile.json --load-term L
--combine C --out …`), and the freeze names the form.

Alternatives considered:

- A rule in the code instead of a flag, for example no one worker above
  1024³ multiply-adds. It would carry this machine's timing into the
  definition of the space. The flag keeps the exception in the Makefile,
  beside `CALIB_SIZES`.
- Moving the old profile away by hand for this run only. The next
  calibration, on the A725 in step 5 or a repeat here, would meet the same
  trap.

**The user's three decisions** (2026-09-24, after this report, each as
recommended):

1. **F1 is priced by this calibration**, refitted in the frozen form, not
   by a new one after the freeze. The freeze then fixes the model's
   constants as well as its form before any fresh measurement: what F1
   tests is what the development saw and the tag names. A second
   calibration would differ from this one by the run-to-run noise above (a
   median of 2.3 %, 18 % at the 90th percentile), and part of the gap
   between the development figure D and F1 would be that noise. Plan §5's
   `make calibrate-ra4` stands as the procedure being reproducible. The
   A725 has no calibration yet, so step 5 fits its constants after the
   freeze, with the frozen form and space (question 71).
2. **The development set's twenty-core day is left out of development.**
   No statement of 0.8.0 is about twenty cores. 0.7.0's twenty-core profile
   has only samples recorded before the load count, which `presk refit`
   drops, so keeping the day would take a new calibration on cores 0–19.
   Its groups would mix X925 and A725 cores, and the day was measured once,
   ranked against itself. `experiments/ra4_dev.sh` skips it when
   `PROFILE_20` is empty (`make ra4-dev RA4_DEV_PROFILE_20=`), and the byte
   check then covers the ten-core days: 35 770 lines and `docs/ra4-2026-09-24.md`
   under 0.7.0's profile. With both profiles given it is unchanged (52 142
   lines, both reports). 0.7.0's twenty-core report stands as recorded.
3. **The archive is copied into presk-project**, as the development set is,
   as `data/ra4-calibration-0.8.0.tar.xz` with its `.sha256`: the frozen
   model depends on it, and the only other copy is on falicon.

## D69 — The ablation on the development set, and the form chosen: a load rate per group under a p-norm (2026-09-24)

Plan 0.8.0 step 4, §3.2 and §3.4. The development set is 0.7.0's ten-core
days (D68 leaves the twenty-core day out). Every form of the model is fitted
on the definitive calibration (D68) by `presk refit`. `make ra4-ablation`
(`experiments/ra4_ablation.sh` and `ra4_ablation.py`, about 3½ minutes,
nothing measured) prices the days under each form, and 0.7.0's own profile
beside them. The report is `docs/ra4-dev-ablation-2026-09-24.md`. Its 0.7.0
row reproduces `docs/ra4-2026-09-24.md` byte for byte. These are
development figures: they decide nothing.

**Statement M's nests** (B=16, T=256, `f32`), right (tied), and O:

| form | u | scores `ein` (%13) | values `ein` (%23) | O, top-1 regret of %3, %13, %23 |
|---|---|---|---|---|
| 0.7.0's profile | 0.185 | 47.9 % (37.4 %) | 43.9 % (46.1 %) | 2.87, 2.18, 3.01 |
| no load term, max | 0.241 | 47.9 % (37.4 %) | 44.0 % (46.1 %) | 2.87, 2.18, 3.01 |
| load term per group, max | 0.142 | 55.2 % (41.0 %) | 60.3 % (37.4 %) | 1.18, 1.04, 1.04 |
| no load term, two-term sum | 0.234 | 48.9 % (36.0 %) | 54.8 % (29.7 %) | 2.87, 2.18, 3.01 |
| no load term, p-norm | 0.232 | 51.2 % (32.6 %) | 55.1 % (29.3 %) | 2.87, 2.18, 3.01 |
| per group, two-term sum | 0.141 | 57.7 % (38.1 %) | 66.8 % (27.3 %) | 1.07, 1.00, 1.01 |
| **per group, p-norm, q = 1.06** | 0.143 | **84.6 % (2.6 %)** | **84.4 % (3.5 %)** | **1.07, 1.00, 1.01** |
| one for the machine, p-norm | 0.173 | 84.8 % (2.6 %) | 84.2 % (3.5 %) | 1.17, 1.04, 1.04 |

1. **Neither change works alone.** The load term under the maximum reaches
   55–60 %, and a new combination without it 49–55 %. Together they reach
   about 85 %, and the tied pairs fall from about 40 % to 3 %.
2. **Only the p-norm works, not the two-term sum.** With `q` near 1 it adds
   all five terms, so the work and the loads add. That is what tells the
   scalar tiles apart (D67). The two-term sum still takes the maximum of
   the work and the loads, and of the three traffic terms.
3. **Per class under the chosen form**, over every nest: tiles 83.4 %
   right, orders 63.6 %, k-blocks 49.3 %, lanes 100 %, splits 97.6 %, the
   pack 57.8 %. The pack is the same in every form: it is predicted within
   `u` everywhere.
4. **The pairs the model separates by more than `u` are 97–100 % right.**
   Every miss is a pair it predicts within its own uncertainty. On the two
   M nests the misses are mostly orders (52–68 % right), then tiles (79–83
   %) and the pack (56 %). To reach 90 % the scores `ein` needs about 700
   more pairs right, the values `ein` about 830.
5. **O holds** under every form with the load term per group, and fails
   without it (2.2–3.0×).
6. **Q's rule on the development set, under the chosen form.** It measures
   only the points predicted within 1 + `u` of the best prediction, at
   most 0.6 % of the sweep's kernel time. The best of them is within 1.01×
   of the best measured point on 12 of the 13 contraction nests swept
   whole. The scores `ein` at B=1 in `f32` misses, at 1.08×, as it does
   in most forms.

**The form chosen: a load rate per group under the p-norm**, its `q`
fitted on the calibration (1.06), not on the development set. It leads on
M, O and Q. One load rate for the machine does as well on M, but its O
(1.17 on `Linear`) and its candidate sets are worse.

**What follows (the user's decision, 2026-09-24).** At most a week of
refinements aimed at what the per-class table points to: loop orders and
the pack. A refinement is accepted only if it also lowers the calibration's
`u` in the chosen form, so that the calibration is the gate and the
development set the judge. Then the freeze, whatever the development
figure. A refinement changes the planned cost, so the calibration's stored
costs must first be computed again from each sample's layout, its time
kept: a tool comes first. The plan is amended in presk-specs (plan 0.8.0,
step 4, "after the ablation").

Alternatives considered:

- **Freezing the chosen form now**, at 84.6 % and 84.4 %. It is what a
  refinement that fails its gate leaves.
- **Choosing `q` on the development set.** It would fit the model's form to
  the data that is meant only to judge it. The fitted `q` stays.

## D70 — The calibration re-costed, and two rules of the planned cost kept: cache lines and loop overhead (2026-09-24)

Plan 0.8.0 step 4, 4b and 4c. The review report is
`docs/ra4-dev-rules-2026-09-24.md`; its tables are `make ra4-ablation` over
the forms `group-pnorm+R`. These are development figures: they decide
nothing.

**4b. The calibration re-costed, its times kept** (`presk refit P.json
--recost experiments/matmul.psk [--rules R]`). A rule of the planned cost
changes the cost of every sample the model is fitted on. Measuring again
would replace the definitive calibration, which F1 is priced by (D68). So
each sample's layout is read back from its key (`Layout::from_key`) and
lowered at the product its signature names, on `matmul`. It is costed on
the profile's own machine description, and must have the sample's
signature and group; its time is kept. Under the current planned cost all
3 336 costs come back identical. Refitted in its stored form, the profile
is byte-identical to the archived calibration, so the fit and `u` agree to
the last digit. It takes 1.3 s.

**4c. Three rules, each off by default.** A profile names the rules its
samples were costed under (`rules` in `LayoutFit`), and every prediction
made from it is costed under them. A profile naming none prices every
kernel as before, bit for bit: `make ra4-dev` under 0.7.0's profiles still
gives its 52 142 lines and both reports byte for byte.

- `lines`: a footprint counts whole lines of the machine's line size (64
  bytes on falicon, read from sysfs into `Machine.line`). Along the
  contiguous axis, its span is rounded up to the line. The archived
  description has no line size, so `--recost` reads it for `cpu5` and
  records it.
- `shared`: a list of statements, or a loop, is resident when every
  buffer's footprint in it, summed, fits the cache.
- `overhead`: each loop iteration adds one operation to the work.

One addition to the plan's wording, under `lines` or `shared` only. A loop
that does not fit while one of its iterations does brings once what
consecutive iterations share: the lines they share along the contiguous
axis, or a buffer the loop does not move. The loop is charged the lesser of
its footprint and its iterations' footprints summed. Without it, `lines`
would charge a line per element to every stream along a row, and `shared`
would evict the output's tile at every step of `k`.

**The gate** (the user's, D69): the re-costed calibration's `u`, refitted
in the chosen form, must fall below 0.1425, and M must not fall on either
nest (84.6 % and 84.4 %).

| rules | u | scores `ein` (%13) | values `ein` (%23) | O, %3, %13, %23 | gate |
|---|---|---|---|---|---|
| none | 0.1425 | 84.6 % | 84.4 % | 1.07, 1.00, 1.01 | — |
| `lines` | 0.1407 | 84.8 % | 84.5 % | 1.07, 1.00, 1.01 | passes |
| `shared` | 0.1432 | 85.0 % | 87.9 % | 1.07, 1.07, 1.05 | fails: `u` rises |
| `overhead` | 0.1340 | 88.8 % | 89.2 % | 1.07, 1.00, 1.01 | passes |
| **`lines,overhead`** | **0.1319** | **89.3 %** | **89.6 %** | 1.07, 1.00, 1.01 | passes |
| `lines,shared,overhead` | 0.1299 | 87.6 % | 90.6 % | 1.07, 1.07, 1.05 | `shared` lowers %13 |

1. **`overhead`** does what it was aimed at. Over every nest, tiles go from
   83.4 % to 88.7 % right and k-blocks from 49.3 % to 65.5 %, and the
   k-blocks' ties (25 %) are gone. It improves 2 194 of the 3 336
   calibration samples.
2. **`lines` adds little.** Orders gain 0.9 points and the pack does not
   move: the attention's rows are whole lines. On the calibration it moves
   564 samples, with narrow tiles at 2048 and on the thin products, whose
   L2 traffic grows 2–4× (up to 17×). Those samples fit worse after the
   refit (median |r| 0.272 → 0.299), the others better. The gain in `u`
   comes from `β_L2` refitted. The lines it counts are real (a unit test
   counts them exactly), but that L2 stream does not bound those kernels on
   this machine.
3. **`shared` passes in no combination.** On its own and on top of
   `overhead`, `u` rises. On top of `lines`, or of both, M falls on the
   scores `ein`. It helps the values `ein`'s orders (52 → 68 %) and hurts
   the scores `ein`'s (68 → 52 %). It worsens O on both `ein`s and drives
   `q` to 1.00.
4. **The pack is reached by no rule** (56.4 % on %13). The body kernel costs
   the same with the pack and without it, so the model predicts the pack
   slower in all 468 pairs. It wins in 204 of them, by 5 % in the median and
   16 % at most, all with tiles of 4 to 16 columns. At each step of `k` such
   a tile reads one value from each of its rows of the unpacked operand,
   where the packed panel holds them in one line. That is how an
   iteration's loads fall, not what stays in cache.
5. **Q** is unchanged by `lines` and `overhead`: within 1.01× of the best
   everywhere but the scores `ein` at B=1 in `f32` (1.08×, in every form),
   at most 1.1 % of the sweep's time.

**Decided (the user, 2026-09-24): `lines` and `overhead` are kept, `shared`
is not, and no fourth rule is tried.** Both pass the gate on their own and
on top of each other. Together they are best on every criterion the gate's
second half names: M, the per-class table, O and Q. On the development
set, M is still 86 and 65 right pairs short of 90 % on the two nests. That
decides nothing: the fresh sweeps do.

Alternatives considered:

- **`overhead` alone**, since `lines` pays little and indirectly. It is
  about 90 % of the gain in M. Setting aside a rule that passes the gate
  set in advance would be choosing on the development set.
- **All three rules.** It has the lowest `u` (0.1299) and reaches 90.6 % on
  the values `ein`, but `shared` fails the gate in every combination.
- **A fourth rule: the distinct lines each iteration's loads touch**, which
  would see the pack. It is not in the plan, and designing it now would fit
  the model's form to the development set. It is recorded for after
  0.8.0.

## D71 — The freeze: 0.8.0's layout model, and statement D on the development set (2026-09-24)

Plan 0.8.0 step 4, 4d and 4e, and §3.4. This record is statement D. What
follows is measured only on the fresh sweeps F1 and F2, which the model has
not seen.

**4d. The nine forms again, under the rules kept** (`make ra4-ablation
FORMS="…+lines,overhead"`, `docs/ra4-dev-ablation-lines-overhead-2026-09-24.md`).
Each form is refitted on the calibration re-costed under `lines` and
`overhead`. The chosen form still leads, so nothing was left to choose:

| form, under `lines,overhead` | u | scores `ein` | values `ein` | O, %3, %13, %23 |
|---|---|---|---|---|
| no load term, max | 0.254 | 76.7 % | 83.1 % | 1.18, 1.04, 1.04 |
| per group, max | 0.138 | 81.9 % | 85.7 % | 1.18, 1.04, 1.04 |
| per group, two-term sum | 0.136 | 83.1 % | 86.6 % | 1.07, 1.00, 1.01 |
| one for the machine, p-norm | 0.157 | 86.7 % | 87.3 % | 1.17, 1.04, 1.04 |
| **per group, p-norm** | **0.132** | **89.3 %** | **89.6 %** | **1.07, 1.00, 1.01** |

The loop overhead lifts every form: without the load term, under the
maximum, M's nests go from 47.9 % and 44.0 % to 76.7 % and 83.1 %. The
maximum forms rank k-blocks better (73–75 % against 66 %), and lose on
every other count.

**The frozen model.**
- **The planned cost**: `cost.rs` under the rules `lines` and `overhead`
  (D70), `Machine` with its line size.
- **The form**: a load rate per group under the p-norm, `q` fitted on the
  calibration (1.0585), `LayoutFit` and its fit as of D67.
- **The calibration's space**: D68's, unchanged. Its method: `make
  calibrate-ra4`, whose defaults are now this form and these rules
  (`bench --rules`, `RA4_RULES`).
- **`presk plan`'s rule**: the least predicted time, a tie broken by the
  space's order, unchanged.

**The frozen profile** is the definitive calibration (D68), costed again
under the rules kept (`presk refit --recost`, D70), its times kept, and
refitted in that form: u = 0.1319, Spearman 0.984, 3 336 samples. Nothing
was measured to make it.
- Archived read-only in `~/presk-data/ra4-model-0.8.0/`: `profile.json`,
  with the command's log, the working-tree diff it ran with, a README and
  `SHA256SUMS`.
- SHA-256 of `profile.json`: `cfc9237f7c6e90c958b1e29f0031939ed1351d664eb493f85018311c807bcdb1`.
  Of `SHA256SUMS`: `8235d8c6815b6714c907323b0795aff63fb30f8c7dd960ace7ce4aaba801dfbe`.
- A copy is in presk-project `data/ra4-model-0.8.0.tar.xz`, SHA-256
  `f04420cf295c48ab87f91d80e89f6c4d3bade817e23724b340529a60195422e9`.
- F1 is priced by it (D68). F2's A725 profile is calibrated in step 5 in
  this form and under these rules.

The development report under it is `docs/ra4-dev-model-0.8.0.md`, byte for
byte what `make ra4-dev-frozen` writes. A defect fixed before both fresh
sweeps are ranked must leave that check passing, or it is recorded and
fixed after step 7 (plan §3.4). The freeze starts at the tag
`ra4-model-0.8.0`, which the user places on the commit that holds all of
this.

**Statement D: the development set under the frozen profile.** 0.7.0's
ten-core days, re-priced, nothing measured. These figures are the model's
design data and decide nothing.

| | measured | 0.8.0's threshold |
|---|---|---|
| M | scores `ein` 89.3 % of 12 925 distinguished pairs, values `ein` 89.6 % of 14 868 (0.7.0's model: 47.9 % and 43.9 %) | above 90 %: not met, by 86 and 65 pairs |
| O | 1.07 (`Linear`), 1.00, 1.01 | within 1.25×: met |
| Q | the best candidate within 1.01× of the best measured on 12 of the 13 contraction nests swept whole; the scores `ein` at B=1 in `f32` 1.08×; candidates at most 1.1 % of the sweep's time | within 1.05× at ≤ 10 %: not met on one nest |
| N | 75× to 717× on the attention, 1 460× to 1 538× on the GEMM | ≥ 10³×: these are 0.7.0's static times, recorded before its static side was cut to a capture of shapes (D65); D64 and D65 estimate about 1.9·10³ at B=16 in `f32` and 175–210 at B=1 |

**Per class, every nest**: tiles 89.1 % right, orders 67.0 %, k-blocks
65.9 %, lanes 100 %, splits 97.7 %, the pack 57.8 %. Only 0.1 % of the
pairs are tied. The pairs predicted apart by more than `u` are 98.0 % right
over every nest: tiles 97.6 %, orders 92.3 %, splits 98.5 %. Almost every
k-block pair (99.8 %) and every pack pair lies within `u`. On M's nests
the misses are tiles (731 and 775), orders (426 and 700), the pack (204 on
the scores `ein`) and a few splits.

**The choices of 4b–4d, and why.**
- **4b**: re-costing rather than measuring again, so that the calibration F1
  is priced by stays the one measured before any development. It gives the
  calibration back byte for byte.
- **4c**: `lines` and `overhead` kept, `shared` not (D70). The gate was the
  calibration's `u` and M on the development set, fixed before any result.
- **4d**: the form chosen in D69 still leads under the rules kept.

**Tried and not kept**:
- the rule `shared`, which fails its gate in every combination (D70);
- `overhead` without `lines`, and all three rules (D70);
- one load rate for the machine, and the two-term sum (D69, and again in
  4d);
- `q` chosen on the development set (D69).

**Recorded for after 0.8.0**:
- a rule counting the distinct lines each iteration's loads touch, which
  would see the pack (D70);
- why the maximum forms rank k-blocks better than the p-norm (73–75 %
  against 66 %, 4d).

## D72 — F2's targets: the A725's calibration, a dry run that sizes a day, and checks before a fresh day (2026-09-25)

Plan 0.8.0 step 5, its first part (§3.5). This is experiment plumbing only.
Nothing the freeze covers changes: not `cost.rs`'s planned cost and its
rules, `LayoutFit`, the combination, the calibration space, nor `presk
plan`'s rule. `make ra4-dev-frozen` still writes
`docs/ra4-dev-model-0.8.0.md` byte for byte. Neither does anything in
`experiments/ra4.sh`, `ra4_check.sh` or `presk rank` change.

**The targets**, on the model of 0.7.0's:
- `make calibrate-ra4-small` is `make calibrate-ra4` with `RA4_CORES` set
  to `RA4_CORES_SMALL` (`0-4,10-14`) and `RA4_PROFILE` set to
  `RA4_PROFILE_SMALL` (`out/profile-a725.json`). It keeps the same GEMM
  sizes and thin products, the same exception at 2048, and the frozen form
  and rules, which are `calibrate-ra4`'s defaults since D71. It refuses a
  `RA4_PROFILE_SMALL` that is `RA4_PROFILE`. The machine description is
  read for cpu0, an A725: 64 KiB of L1, 512 KiB of L2, 64-byte lines, and
  the 8 MiB L3 of CPUs 0–9, which five of the sweep's workers share, as on
  the X925 cores.
- `make ra4-f2` and `ra4-f2-day2` run `experiments/ra4.sh` into
  `out/ra4-f2/day1` and `day2`, pinned to the A725 cores and priced by
  their profile. The configuration is `RA4_F2_RUN` at `RA4_F2_SHAPES`,
  0.7.0's: `SelfAttention.apply` of `corpus/nlp/05_gpt.psk`, `B` = 1 and
  16, `T` = 256, `f32` and `f64`, the whole space.
- Day two ends with `make ra4-f2-rank`: `presk rank --frozen
  ra4-model-0.8.0`, the check included, into `out/ra4-f2.md`. The ranking
  can also be run again alone.
- `make ra4-f2-check` runs `experiments/ra4_check.sh` with 4 and 40
  shards, as 0.7.0's check did.

**The checks before a fresh day** (`experiments/ra4_fresh.sh`). `presk
rank --frozen` refuses a day that cannot count only after it is measured,
a night later. The script refuses it before:
- a day directory that exists: a day is one run;
- a working tree that differs from its commit, or a commit that does not
  descend from the tag;
- day two without day one, or on the calendar date day one started on
  (§3.5's two days).

Every F2 target that measures or checks, the calibration among them, also
refuses to start while a `presk` process runs (`pgrep -x presk`). `RA4_TRIAL=1` marks a
trial of the targets on a tiny configuration. A trial skips every check
but the first, and ranks without `--frozen`.

**The dry run** (`make ra4-f2-dry`: `experiments/ra4_dry.sh`,
`ra4_dry.py`). §3.5 asks for one nest's sweep timed before the first night.
- **Which nest.** Every configuration is priced by `presk plan --cost`, and
  the nest with the longest predicted sweep is timed as a night times it,
  pinned.
- **The estimate.** A configuration is estimated as `(R + 1) · r · Σ
  predictions + c · points`, plus its captures and `presk plan`:
  - `R + 1` is the warm-up launch and the `R` timed ones;
  - `r` is the nest's sum of medians over its sum of predictions;
  - `c` is its time a point outside the kernels.
- **Why this works.** A sweep is almost all kernel time. On 0.7.0's day
  one, `(R + 1)` times the sum of the medians is within 2 % of each
  configuration's sweep at `B` = 16, and within 7 % at `B` = 1. What is
  left is how far the model's predictions are from the times, and one
  nest's ratio measures that on the machine.
- **Checked on 0.7.0's day one**, priced by the frozen model
  (`out/ra4-dev`, the X925 cores). The nest chosen is `Linear.apply/%3` at
  `B` = 16 in `f64`, `r` = 1.226. The estimates are 1 020, 1 186, 55 and
  63 s, against 1 047, 1 282, 58 and 66 s measured. The day comes out at
  38 min 56 s against 40 min 57 s, 5 % short. The contraction nests'
  ratios there range from 1.16 to 1.60 (the scores `ein` at `B` = 16 in
  `f64` is the highest), so an estimate from one nest can be off by about a
  third on a configuration.

A configuration too slow to sweep whole is swept on a sample (`ra4.sh`'s
`EXTRA="--top N --random M"`), and its M is inconclusive (§3.5). **`presk
rank` now says so** (the user's decision, 2026-09-25, taken before any
fresh measurement). Until then it decided M on the count of distinguished
pairs alone, and a sample with 10³ pairs could have held.
- M's configuration swept on a sample, on either day, makes M
  "inconclusive: swept on a sample (plan 0.8.0 §3.5)", whatever its pairs
  say.
- M's and O's cells are marked `(sample)`. O keeps its verdict, since the
  model's pick is among a sample's points, but its best point is only the
  sample's, which flatters the model. Whether O on a sample should count is
  left to step 7.
- It is fixed now, before F2 and F1 are measured, so that the rules that
  judge the fresh sweeps are fixed before their data, as the model is.
  Nothing the freeze covers changes. No statement configuration of the
  development set is sampled, so `make ra4-dev` (0.7.0's reports) and `make
  ra4-dev-frozen` come back byte for byte.
- A unit test ranks one sweep sampled and then whole.

**The trial** on a tiny configuration, before any measurement of the test:
the tutorial's attention (`attend`), `T` = 256, `B` = 1 and 16, `f32` and
`f64`, the small space, into `out/ra4-f2-trial`.
- **The calibration**: `make calibrate-ra4-small CALIB_SIZES=256
  CALIB_THIN=` took 59 s. Its profile records cpu0's description and the
  rules `lines` and `overhead`.
- **`bench` and `refit --recost` price alike.** `presk refit --recost
  --rules lines,overhead` gives its 600 costs back identical, and the
  profile byte for byte. This matters because the X925's frozen profile
  was made by the second (D71) and the A725's is made by the first.
- **The dry run** estimated the four configurations at 3.1, 3.4, 0.7 and
  0.7 s; day one measured 3.0, 3.5, 0.3 and 0.3 s. At `B` = 1 a point's
  overhead, measured at `B` = 16, is too large.
- **The refusals**, each seen: an uncommitted tree, a day or a check that
  exists, a `presk` process running, and `presk rank --frozen` on the
  trial's `-dirty` days. Three the presk tree cannot reach while it is
  uncommitted were run in a scratch repository, with the cases that pass:
  day two without day one, day two on day one's date, and a commit not
  descending from the tag.
- **The check and the ranking**: 12 shards, then day two and the ranking.
  L holds, 332 of 332 points checked on the interpreter.

**Day two on day one's date** (the user's decision, 2026-09-25, after day
one and the check). §3.5 asks for two different days, and the check before
a fresh day refused day two on day one's calendar date. The user chose to
measure F2's day two the same afternoon, a few hours after day one, rather
than after midnight. The date was only a proxy for independent conditions,
and a run at 00:05 would have been little more independent than one the
same afternoon. What a second day guards against is state two close runs
share: where physical memory places the buffers, the frequency and thermal
regime, background activity.
- `RA4_SAME_DATE=1` (the script's `SAME_DATE`) accepts day two on day one's
  date. The default still refuses it. Every other check stands: the clean
  tree, the tag, day one's presence, no `presk` running.
- §3.5 is amended in presk-specs. F1 may also run its two days on one date:
  the user's decision of 2026-09-25, taken after F2 was ranked.
- The report states that F2's days ran hours apart on one date, as 0.7.0's
  did back to back.

**Archives.** Each piece is archived read-only as soon as it is measured,
since none can be taken again: the A725 calibration as D68 archived the
X925's, then each day and the check. The places and checksums are added to
this record as they are made.

**The A725 calibration** (`make calibrate-ra4-small`, 2026-09-25
07:35–09:17, presk `8e3c9e1`, a clean tree). It ran alone: the X925 cores,
which share the L3 with cores 0–9, were more than 99 % idle (`sar`).
- **3 336 samples in 6 073 s** (101 min, against about 1.5 h estimated in
  D68), the same layouts as the X925's.
- **The fit.** In the frozen form, u = 0.168 and Spearman 0.977; the X925's
  frozen profile has 0.132 and 0.984. `q` = 1.00 is the lower edge of its
  search, so on these cores the p-norm is a plain sum (1.06 on the X925).
  Of the nine forms of `presk refit --compare`, the frozen one fits best
  here too: group sum 0.185, group max 0.190, 0.7.0's form without the load
  term 0.252.
- **Costing.** `presk refit --recost --rules lines,overhead` gives the
  profile back byte for byte.

| | A725 | X925 (frozen) |
|---|---|---|
| spawn, λ | 189 µs, 0 | 101 µs, 0 |
| β_L2, β_L3, β_mem | 196, 318, 334 GB/s | 214, 354, 711 GB/s |
| π, one lane | 5.3–6.3 Gop/s a worker | 12.2–12.9 |
| π, vector lanes | 25–203 | 33–181 |
| ρ, one lane | 7.1–10.6 G loads/s | 49–80 |
| ρ, vector lanes | 1.5–2.3 | 4.5–10.1 |

**What the calibration identifies, and what it does not.** Measured by each
term's share of the p-norm's sum over the samples:
- **π at one lane is identified.** The work term is the largest in every
  one-lane sample, about 60 % of the sum.
- **ρ at one lane is identified here**, unlike on the X925. Loads are 19–34 %
  of the sum, and the rate, about 2.5–3.8 loads a cycle at 2.8 GHz, is
  plausible; the X925's 49–80 G/s is not (D67).
- **ρ with vector lanes is identified.** Loads are the largest term in every
  vector sample, 54–80 % of the sum.
- **π with vector lanes is not, on 5 and 10 workers in `f32`.** The work
  term is about 5 % of the sum there, and the fit puts it at 202–203 Gop/s a
  worker, twice its one-worker value, which is not physical. In `f64` it is
  15–21 % of the sum, and loosely held.
- **β_L2 and β_L3 are loose.** They are never the largest term, and 1–11 % of
  the sum.
- **β_mem is not identified**, as on the X925 (D67). It comes out above β_L3,
  and above the machine's memory bandwidth of about 273 GB/s.

The thin `16 × 4096 × 4096` product fits worst: a median residual of 0.47,
against 0.32 on the X925.

Archived read-only in `~/presk-data/ra4-0.8.0-a725-calibration` (the
profile, the log, a README and `SHA256SUMS`). SHA-256 of `profile-a725.json`:
`ad64964edbe94438c3e1985c615f80c4911f163da4f06cbc4b6d4d5c2ea1c6c4`. Of
`SHA256SUMS`: `e0def2c002597c3af305a71845b62a5d431f876d9ba24b36333f7e855e0495d4`.
A copy is in presk-project `data/ra4-a725-calibration-0.8.0.tar.xz`
(`cb15a0df…`).

**The dry run** (`make ra4-f2-dry`, 09:19–09:41) swept `Linear.apply/%3`
at `B` = 16 in `f64`, the nest the model predicts longest: 1 563 points in
1 258 s, where the X925 took about 470 s. Its measured time over its
predicted is 1.154, which gives an estimate of 1 h 41 min a day. F2 was
therefore swept whole.

**The days.** Both ran from a clean tree descending from the tag, and every
output was bit-identical to point 0 natively. On each day the X925 cores
were more than 99 % idle on average.

| | day one | day two |
|---|---|---|
| run | 2026-09-25 12:07–13:52 | 16:45–18:30, the same date (above) |
| commit | `8e3c9e1` | `524ea5a` (the check before a fresh day and docs only) |
| B=1 `f32`, `f64` | 142.0 s, 168.2 s | 142.2 s, 168.7 s |
| B=16 `f32`, `f64` | 2 621.2 s, 3 365.6 s | 2 621.3 s, 3 323.2 s |

- **Against the estimate.** Day one measured 1 h 45 min against 1 h 41 min
  estimated. The estimate was 7 % and 4 % long at `B` = 1, 1 % long in
  `f32` at `B` = 16, and 8 % short in `f64` there.
- **The check** of statement L (`make ra4-f2-check`, 13:53–15:51, 88 shards
  on every core) found none of day one's 16 324 points differing, natively
  or on the kernel interpreter.
- **A caveat on day two.** The `sar` sample from 16:40 to 16:50 shows the
  X925 cores 92–97 % idle. That window includes the end of `make test`,
  just before day two started at 16:45:32. Day two's first two sweeps fall
  in it, and their times match day one's to within 0.3 %.

Archived read-only in `~/presk-data/ra4-f2-0.8.0`, with one checksum list
per piece and a README:

| piece | `SHA256SUMS` | presk-project `data/` |
|---|---|---|
| day one and the dry run | `.day1` `fc36e593…`, `.dry` `57e20f13…` | `ra4-f2-day1-0.8.0.tar.xz` `f1876f60…` |
| the check | `.check` `49ebd1a0…` | `ra4-f2-check-0.8.0.tar.xz` `dd538a2c…` |
| day two and the ranking | `.day2` `0c4a709e…` | `ra4-f2-day2-0.8.0.tar.xz` `02d15943…` |

**The ranking** (`make ra4-f2-rank`: `presk rank --frozen ra4-model-0.8.0`
with the check, `out/ra4-f2.md`). These figures are reported here; step 7
decides the statements.
- **L**: all 16 368 points of day one, 0.6.0's baselines included, checked
  on the interpreter; none differing.
- **M's nests**, B=16 in `f32`: the scores `ein` ranks 92.7 % of 13 488
  distinguished pairs right, the values `ein` 91.7 % of 15 826. On the
  development set under the frozen profile they were 89.3 % and 89.6 %
  (D71); under 0.7.0's model on the X925, 47.9 % and 43.9 %.
- **O**, at M's configuration: 1.00, 1.01 and 1.00 (`Linear.apply/%3`,
  `%13`, `%23`).
- **N**: 2 168–2 734× at `B` = 16, but 218–252× at `B` = 1, below 10³ as
  plan §7 foresaw. With the capture that runs the entry, the ratios are
  1 395–1 546× and 161–173×.
- **Every nest**: 90.4 % of 134 022 distinguished pairs right, 98.8 % of
  the pairs predicted apart by more than u, 0.1 % tied.
- **By class**: tiles 92.4 %, splits 98.1 %, lanes 98.7 %, orders 72.6 %,
  the pack 62.9 %, k-blocks 53.4 %. Every k-block pair and every pack pair
  lies within u.
- **In `f64` at B=16**, which M does not name: the scores `ein` 85.9 %, the
  values `ein` 86.3 %.
- **The worst top-1 regret** on a contraction nest is 1.11 (`%23` at B=1 in
  `f64`, and at B=16 in `f64`).
- **Not yet computed**: statement Q, which `presk rank` does not decide
  before step 7.

## D73 — F1's targets: two configurations in one day, a dry run each, and two ranks (2026-09-25)

Plan 0.8.0 step 6, its first part (§3.5). This is experiment plumbing only.
Nothing the freeze covers changes, and neither do `experiments/ra4.sh`,
`ra4_check.sh`, `ra4_fresh.sh` nor `presk rank`. The only script that
changes is `experiments/ra4_dry.py`, which gains a mode; its estimate of
F2's archived dry run comes back byte for byte.

**F1** is `SelfAttention.apply` of `corpus/nlp/05_gpt.psk` in `f32`, the
whole space, pinned to the ten X925 cores (`RA4_CORES`, `5-9,15-19`), at
two configurations:
- **F1a**, BERT-large's attention: `D` = 1024, `H` = 16, `T` = 384, `B` = 1.
  0.7.0's day one swept it on a sample of 41 points a nest, which is part
  of the development set; the plan keeps the shape, and sweeps it whole.
- **F1b**, a head of 128: `D` = 1024, `H` = 8, `T` = 512, `B` = 2.

**The profile is the frozen model's own**, `~/presk-data/ra4-model-0.8.0/
profile.json` (`RA4_F1_PROFILE`, by default `RA4_FROZEN`). The X925 is not
calibrated again after the freeze: the user's decision of 2026-09-24 (D68).
The targets read it where it is archived, read-only.

**The targets**, on the model of F2's (D72):
- **Configurations.** `RA4_F1A` and `RA4_F1B` are written
  `constants:shape` (`D=1024,H=16:B=1,T=384`). Their constants differ, so
  each is its own run of `experiments/ra4.sh`, both into one day's
  directory, as 0.7.0's day held three runs. `RA4_F1_RUN` holds the rest:
  the file, the entry, the type, the space.
- **`make ra4-f1` and `ra4-f1-day2`** run F1a then F1b into
  `out/ra4-f1/day1` and `day2`. They check the day first with
  `experiments/ra4_fresh.sh`, as F2's did: a day directory that exists, an
  uncommitted tree, a commit not descending from the tag, day two without
  day one or on day one's date without `RA4_SAME_DATE=1`. F1 may run both
  days on one date, the user's decision of 2026-09-25 (D72).
- **`make ra4-f1-rank`** ranks the two days twice. `presk rank` decides M
  and O at one configuration a run (`--at`), so one rank is at F1a into
  `out/ra4-f1a.md`, the other at F1b into `out/ra4-f1b.md`. Both use
  `--frozen ra4-model-0.8.0` and the check. L and N, which cover every
  configuration swept whole, and the tables of nests and classes are the
  same in both reports. No code changes to rank F1: `presk rank` stays
  the tool that ranked F2. Day two ends with it.
- **`make ra4-f1-check`** is statement L on the kernel interpreter,
  sharded over every core, F1b's shards first so that the longest run first.
- Every target refuses to start beside a `presk` process, and `RA4_TRIAL=1`
  tries them on a tiny configuration, as for F2.

**The dry run, one per configuration** (`make ra4-f1-dry`). §3.5 asks for
a configuration too slow to sweep whole to be known before the first
night.
- **One nest per configuration.** `experiments/ra4_dry.sh` runs once for
  each configuration, into `out/ra4-f1/dry/f1a` and `f1b`, and times that
  configuration's longest-predicted nest. F2 scaled four configurations of
  one shape family by one nest's ratio. F1's two shapes are further apart,
  and the dev set's ratios ranged from 1.16 to 1.60 (D72).
- **The day** (`ra4_dry.py day`, into `out/ra4-f1/dry/dry.md`) adds the
  configurations' estimates. It keeps them whole, the quickest first, while
  the day fits in `RA4_F1_BUDGET` hours, 8 by default, a night. The
  others are marked to be swept on a sample.
- **A sample**, if one is needed, is `RA4_F1A_SAMPLE` or
  `RA4_F1B_SAMPLE`, `ra4.sh`'s `EXTRA` (`--top 8 --random 32` was 0.7.0's).
  It is set by a commit before day one, and not on the command line: both
  days and the check must take the same points, and a committed default is
  what `ra4_fresh.sh`'s clean-tree check guarantees. `presk rank` then
  leaves that configuration's M inconclusive (D72) and its N out.

**Sizes, before the dry run** (`presk plan --cost` under the frozen
profile, nothing timed):

| | points | nests | predicted, the whole space once | `presk plan`'s wall time |
|---|---|---|---|---|
| F1a | 3 305 | 12 | 40.9 s | 0.46 s |
| F1b | 5 959 | 12 | 245.8 s | 0.76 s |

A sweep launches each point eight times (the warm-up and seven timed). At
the dev set's ratios F1a's kernels take about 6–9 minutes and F1b's 38–52,
so a day should take about an hour, well within a night, and both
configurations should be swept whole. The longest-predicted nests, which
the dry run times, are `Linear.apply/%3` in both: about 3–5 minutes at F1a and
16–22 at F1b.

**The check's shards.** The kernel interpreter ran F2's check at about 41 M
multiply-adds a second a process in `f32` (56 013 s of interpreter time for
2.3·10¹² multiply-adds at `B` = 16, over 40 shards of 1 300–1 490 s each).
Each shard also runs point 0 of every nest again.

| | multiply-adds, every point | shards | per shard |
|---|---|---|---|
| F1a | 7.5·10¹¹ (`Linear` 1 029 × 4.0·10⁸, the `ein`s 2 232 × 1.5·10⁸) | 20 | about 15 min |
| F1b | 4.0·10¹² (`Linear` 1 563 × 1.07·10⁹, the `ein`s 4 334 × 5.4·10⁸) | 80 | about 21 min |

The shards are sized to about F2's, and 100 of them fill twenty processes
five times: about 1 h 40 min on every core, against 1 h 58 min for F2.

**The trial** on a tiny configuration, before any measurement of the test:
05_gpt's attention at `D` = 32 with `H` = 2, `B` = 1, `T` = 16, and with
`H` = 4, `B` = 2, `T` = 16, the small space, into `out/ra4-f1-trial`. The
second configuration was sampled (`RA4_F1B_SAMPLE="--top 4 --random 4"`)
to try that path.
- The dry run timed one nest for each configuration and wrote the day.
- Day one, the check (2 and 3 shards), and day two with both ranks. L holds,
  216 of 216 points checked on the interpreter. At the sampled
  configuration M is "inconclusive: swept on a sample", its cells marked,
  and N leaves it out.
- The refusals, each seen: a day that exists, a check that exists, an
  uncommitted tree, a running `presk`, and `presk rank --frozen` on the
  trial's `-dirty` days.

**The dry run** (`make ra4-f1-dry`, 2026-09-25 19:13–19:36, presk
`c75ed5b`, a clean tree; the A725 cores more than 99.7 % idle by `sar`)
swept `Linear.apply/%3` in each configuration, the nest the model predicts
longest in both. Every output was bit-identical to point 0 natively.

| | points | swept in | measured over predicted | outside the kernels | estimate |
|---|---|---|---|---|---|
| F1a | 1 029 | 245.6 s | 1.346 | 2.8 ms a point | 7 min 32 s |
| F1b | 1 563 | 1 105.5 s | 1.345 | 3.4 ms a point | 44 min 29 s |

The two ratios agree to the third digit, inside the development set's range
(1.16–1.60) and above the A725's 1.154. A day was estimated at 52 min,
well within the 8 hours, so both configurations were swept whole.

**The days.** Both ran on `c75ed5b`, a clean tree descending from the tag,
day two on day one's date (`RA4_SAME_DATE=1`, the user's decision of
2026-09-25, D72). Every output was bit-identical to point 0 natively. The
A725 cores, which share the L3, were more than 99.6 % idle in every `sar`
sample of both days.

| | day one | day two |
|---|---|---|
| run | 2026-09-25 20:01–20:50 | 22:35–23:23, the same date, after the check |
| F1a | 3 305 points, 457.0 s | 3 305 points, 457.1 s |
| F1b | 5 959 points, 2 443.6 s | 5 959 points, 2 446.5 s |
| the day | 48 min 22 s | 48 min 31 s |

- **Against the estimate.** Day one took 48 min 22 s against 52 min
  estimated, 7 % less. The estimate was 1 % short at F1a and 9 % long at
  F1b, whose `ein`s run nearer their predictions than `Linear.apply/%3`
  does: measured over predicted, 1.154 and 1.130 there against 1.350 (1.300
  and 1.429 at F1a).
- **The check** of statement L (`make ra4-f1-check`, 20:50–22:34, 1 h 44 min
  against about 1 h 40 estimated) ran 100 shards on every core: F1b's 80 in
  1 135–1 480 s each, F1a's 20 in 844–965 s. None of day one's 9 264 points
  differed, natively or on the kernel interpreter.

Archived read-only in `~/presk-data/ra4-f1-0.8.0`, with one checksum list
per piece and a README. F1 is complete: the directory is read-only.

| piece | `SHA256SUMS` | presk-project `data/` |
|---|---|---|
| day one and the dry run | `.day1` `dbb001fc…`, `.dry` `90c6e583…` | `ra4-f1-day1-0.8.0.tar.xz` `4827b59e…` |
| the check | `.check` `6042e8e4…` | `ra4-f1-check-0.8.0.tar.xz` `2d530ff5…` |
| day two and the rankings | `.day2` `d27e61cd…` | `ra4-f1-day2-0.8.0.tar.xz` `b4450b1a…` |

**The rankings** (`make ra4-f1-rank`: `presk rank --frozen
ra4-model-0.8.0` of the two days with the check, M and O at F1a into
`out/ra4-f1a.md` and at F1b into `out/ra4-f1b.md`). These figures are
reported here; step 7 decides the statements.
- **L**: all 9 286 points of day one, 0.6.0's baselines included, checked
  on the interpreter; none differing.
- **M's nests are below 90 % at both configurations.**

  | | scores `ein` (`%13`) | values `ein` (`%23`) |
  |---|---|---|
  | F1a, B=1, T=384, head 64 | 87.4 % of 7 657 | 88.5 % of 9 120 |
  | F1b, B=2, T=512, head 128 | 85.4 % of 23 347 | 88.8 % of 15 291 |
  | F2, the A725, B=16 (D72) | 92.7 % of 13 488 | 91.7 % of 15 826 |
  | the development set (D71) | 89.3 % of 12 925 | 89.6 % of 14 868 |

  Reaching more than 90 % would have taken about 200 and 140 more right
  pairs at F1a, and about 1 080 and 180 at F1b.
- **The misses lie within u.** Of the pairs the model predicts apart by
  more than u = 0.132, 99.3–99.7 % are right in M's nests; 39–46 % of the
  distinguished pairs are undecided, and ties are at most 0.2 %. Spearman
  is 0.970–0.982.
- **Where they are**, in M's nests: tile pairs are the largest class, 89.2–90.5 %
  right, about 10 % wrong on 4 300–11 000 pairs a nest. Orders are 47–71 % right
  and make up most of the rest; at F1b's scores `ein`, whose sum runs
  over a head of 128, the orders the model decides (about 120 pairs) are
  only 44.6 % right, the one class of a contraction nest where decided pairs
  are mostly wrong. The pack is
  40–67 %, k-blocks 67–71 %, splits and lanes 99–100 %.
- **O**, the model's pick against the best measured point: 1.03, 1.02 and
  1.00 at F1a (`Linear.apply/%3`, `%13`, `%23`), 1.00, 1.05 and 1.00 at
  F1b. The model's candidates within `1 + u` (11–18 points a nest) hold a
  point within 1.00 of the best, to two decimals, in every contraction nest.
- **N**: 1 013× and 1 016× at F1a, on the two days, just above 10³; 2 880×
  and 2 933× at F1b. With the capture that runs the entry, the ratios are
  276–299× and 1 107–1 136×.
- **Every nest**: 87.6 % of 76 830 distinguished pairs right, 99.2 % of the
  pairs predicted apart by more than u. By class: tiles 90.7 %, splits
  99.1 %, lanes 100 %, orders 60.6 %, the pack 47.9 %, k-blocks 64.3 %.
- **Against 0.6.0's default layout**, the best point measured is 27–41×
  faster on `Linear.apply/%3` and 42–132× on the `ein`s.
- **Not yet computed**: statement Q, which `presk rank` does not decide
  before step 7.

## D74 — The statements of 0.8.0: L, O and Q hold, M and N fail (2026-09-26)

Plan 0.8.0 step 7 (§3.6). Nothing is measured in this step. Both fresh
sweeps were ranked from their archived days, under the frozen model, with
every statement. No defect was recorded under the freeze (D72, D73), and
this record ends it (§3.4). `make ra4-dev-frozen` stays as the check that
the frozen model's development report comes back byte for byte.

**Statement Q's rule, fixed before it was computed on the fresh days** (the
user's decision, 2026-09-26). `presk rank` did not decide Q. Its "top-k
regret (k)" column held the quality half, and two counts of the cost
existed: D71's development figure, and question 72's provisional answer.
- **The candidates**: the points of a nest predicted within `1 + u` of the
  model's best.
- **Quality**: the best candidate, at the mean of the two days' medians,
  within 1.05× of the best point measured.
- **Cost, question 72's count**: statement N's static side (the capture of
  shapes and the costing of every point, the whole configuration's, charged
  whole to each nest) plus the candidates' measured time, over the
  exhaustive sweep of the nest's points. It is computed on each day, and
  both days must be at or below 10 %.
- **Beside it, D71's count**: the candidates' share of the nest's timed
  launches (`experiments/ra4_ablation.py`, `hybrid()`).
- **Where**: every contraction nest (`ein`, `dot`) of every configuration
  swept whole on both days.
- **Why question 72's count.** It is what §3.6 says, "the exhaustive
  sweep's time", the wall time N divides by, and what question 72 answered
  before any data. It counts what a chooser pays. D71's count leaves out
  the static side and the compiling of each point, and at B = 1 those are
  most of the cost: on the development set the static side alone is 3.0–4.6 %
  of a nest's sweep at B = 1, while the candidates' timed launches are
  0.12–0.65 % of the nest's.
- **What is estimated.** A sweep records each point's timed launches, but
  not its warm-up launch, its compiling or its comparison of bits, and no
  time per nest. A point's measured time is taken as its `R + 1` launches at
  its median, plus the configuration's time a point outside the kernels:
  the sweep's time less its points' launches, divided by its points, 0.6.0's
  included, as the dry run counts it (D72). Only its spread over the points
  is assumed, even across the configuration; the nests' points and 0.6.0's
  add up to the sweep's time exactly. On the fresh days that time is 1.9–4.9 ms a point,
  and 23–25 ms at B = 16 in `f64` on the A725. Charging each nest the whole
  static side only makes the chooser dearer.

**In `presk rank`.** A Q row closes the statements' table, and a table per
contraction nest follows it: candidates, regret, the cost on each day with
its three parts, D71's share and a verdict. Both appear only for days that
record the capture of shapes (`static_capture_seconds`, D65), as the
decision-cost table's two extra columns do. `--statement-q` decides Q on
earlier days too, costing their static side with the capture that runs the
entry. So `make ra4-dev` (52 142 lines and 0.7.0's two reports) and `make
ra4-dev-frozen` come back byte for byte. A unit test prices a nest by hand
(5.9 % of an 85 s sweep, 5.21 % of its launches) and checks that Q fails
when the best point is not a candidate, and that the row is absent on an
earlier day unless asked for.

**The days are the archived ones.** Every `out/...` line of
`~/presk-data/ra4-f2-0.8.0` and `~/presk-data/ra4-f1-0.8.0`'s
`SHA256SUMS.{dry,day1,check,day2}` passes `sha256sum -c` from the presk
tree, and `out/ra4-f2` and `out/ra4-f1` hold no other file. Ranked from the
presk tree, where `--frozen` checks every day's commit against the tag:

```
presk rank out/ra4-f2/day1 out/ra4-f2/day2 --frozen ra4-model-0.8.0 --check out/ra4-f2/check --out docs/ra4-f2-2026-09-25.md
presk rank out/ra4-f1/day1 out/ra4-f1/day2 --frozen ra4-model-0.8.0 --check out/ra4-f1/check --at D=1024,H=16,B=1,T=384 --out docs/ra4-f1a-2026-09-25.md
presk rank out/ra4-f1/day1 out/ra4-f1/day2 --frozen ra4-model-0.8.0 --check out/ra4-f1/check --at D=1024,H=8,B=2,T=512 --out docs/ra4-f1b-2026-09-25.md
```

Each report is the archived one (`out/ra4-f2.md`, `out/ra4-f1a.md`,
`out/ra4-f1b.md`) but for Q's heading, row and table. F1 has two reports,
since `presk rank` decides M and O at one configuration a run (D73). Plan
step 7's `docs/ra4-f1-<date>.md` is these two, which differ only in M's
and O's rows.

**The verdicts** (plan §3.6).

| | F1 (X925: F1a, F1b) | F2 (A725: B = 1, 16; `f32`, `f64`) | verdict |
|---|---|---|---|
| L | 9 286 of 9 286 points of day one checked on the interpreter | 16 368 of 16 368 | **holds**: no output differing, natively or on the interpreter, on any day or shard |
| M | F1a 87.4 % of 7 657, 88.5 % of 9 120; F1b 85.4 % of 23 347, 88.8 % of 15 291 | 92.7 % of 13 488, 91.7 % of 15 826 | **fails**: four of the six nests at or below 90 % |
| N | F1a 1 013×, 1 016×; F1b 2 880×, 2 933× | B = 16: 2 168–2 734×; B = 1: 249×, 252× in `f32`, 218×, 219× in `f64` | **fails**: F2 at B = 1 on both days |
| O | F1a 1.03, 1.02, 1.00; F1b 1.00, 1.05, 1.00 | 1.00, 1.01, 1.00 | **holds** |
| Q | 6 nests: regret 1.00, cost at most 0.6 % | 12 nests: regret at most 1.03, cost at most 1.8 % | **holds** |

O's and M's nests are `Linear.apply/%3`, the scores `ein` (`%13`) and the
values `ein` (`%23`). No configuration was swept on a sample, so D72's
question whether O counts on a sample does not arise. P is retired (§3.6);
its row reads "not measured". With the capture that runs the entry, N's
ratios would be 276–299× at F1a, 1 107–1 136× at F1b, 1 395–1 546× and
161–173× on F2.

**Q, per configuration** (the three contraction nests of each; the larger
day's cost):

| configuration | candidates | regret | cost (question 72) | D71's share |
|---|---|---|---|---|
| F1a, B=1, T=384 | 12–18 | 1.00 | 0.3–0.6 % | 0.10–0.16 % |
| F1b, B=2, T=512 | 11–13 | 1.00 | 0.1–0.2 % | 0.05–0.06 % |
| F2, B=1, `f32` | 5–15 | 1.00 | 1.2–1.5 % | 0.05–0.16 % |
| F2, B=1, `f64` | 13–19 | 1.00–1.01 | 1.5–1.8 % | 0.21–0.31 % |
| F2, B=16, `f32` | 13–19 | 1.00 | 0.2 % | 0.10–0.12 % |
| F2, B=16, `f64` | 19–28 | 1.00–1.03 | 0.4–0.6 % | 0.20–0.44 % |

The static side is most of the cost where the sweep is short. The scores
`ein` at B = 1 in `f64` on the A725 is the dearest nest: 0.77 s of static
side and 0.15 s of candidates, against 50.4 s swept.

**Per class, every nest**: right, and right among the decided pairs.

| class | F1 | F2 | development set (D71) |
|---|---|---|---|
| tile | 90.7 % (99.5 %) | 92.4 % (98.7 %) | 89.1 % (97.6 %) |
| split | 99.1 % (99.8 %) | 98.1 % (99.0 %) | 97.7 % (98.5 %) |
| lanes | 100 % (100 %) | 98.7 % (98.7 %) | 100 % |
| order | 60.6 % (88.1 %) | 72.6 % (96.3 %) | 67.0 % (92.3 %) |
| k-block | 64.3 % (—) | 53.4 % (—) | 65.9 % |
| pack | 47.9 % (—) | 62.9 % (—) | 57.8 % |
| all | 87.6 % of 76 830 (99.2 %) | 90.4 % of 134 022 (98.8 %) | 88.5 % of 137 307 (98.0 %) |

K-block and pack pairs almost all lie within u (—: none or one decided).

**What failed, and why.**

*M.* The model transferred to the other machine and fell short on the new
shapes. F2, 0.7.0's shapes on the A725, ranks better than the development
set did (92.7 % and 91.7 % against 89.3 % and 89.6 %); F1 ranks worse, by
0.8–3.9 points. Reaching more than 90 % would have taken 200 and 134 more
right pairs at F1a, 1 064 and 183 at F1b. The misses in M's nests, exactly:

| nest | tile | order | pack | k-block | split | wrong | decided wrong |
|---|---|---|---|---|---|---|---|
| F1a `%13` | 447 | 410 | 106 | — | 2 | 965 | 21 |
| F1a `%23` | 525 | 407 | — | 87 | 26 | 1 045 | 31 |
| F1b `%13` | 1 178 | 1 455 | 492 | 258 | 15 | 3 398 | 88 |
| F1b `%23` | 673 | 878 | — | 149 | 12 | 1 712 | 32 |
| F2 `%13` | 471 | 265 | 243 | — | 10 | 989 | 1 |
| F2 `%23` | 555 | 527 | — | 199 | 28 | 1 309 | 17 |

- **The misses lie within u.** 97–98 % of them on F1, and 99 % on F2, are
  pairs the model predicts within `1 + u` of each other. The pairs it decides are 99.3–99.7 % right
  on F1. Within u it is right on 69–73 % of the pairs on F1, 78–81 % on F2
  and 72–75 % on the development set, and those pairs are 39–46 % of the
  distinguished ones on F1. M is decided by how the model orders what it
  predicts to be close: tiles about 10 % wrong on the largest counts, orders
  47–71 % right, the pack 40–67 % and k-blocks 67–71 %.
- **The one decided failure: the orders at F1b's scores `ein`.** The model
  decides 121 order pairs there and 54 are right. All 67 wrong ones are
  one-row tiles, `[1, 8]` and `[1, 16]`, packed, four lanes. Every decided
  order pair of a two- or four-row tile is right (39 of 39). In 66 of the 67
  the two points have loads within 5 % and work within 1 %, and the one the
  model predicts slower has more than three times the L2 term. It is always
  the order with `I` outside `J` (`IJ`, `IJK`, `IKJ`, `KIJ`, against `JI`,
  `JIK`, `JKI`, `KJI`), which reads the whole packed operand once a row
  where the other keeps a panel of it. The model predicts it 15–23 %
  slower, and it measures 1–16 % faster. Under the p-norm, with `q` = 1.06 (nearly a sum), the L2 time is
  added to the load time, and the measurement says the one-row kernel does
  not pay it on top of its loads. The packed operand of one head is 256 KiB
  at F1b and 96 KiB at F1a, against 64 KiB of L1. At F1a's scores `ein`, 8
  of the 30 decided order pairs are wrong, on the same tiles. In F2 and the
  development set it is 64 KiB, and their decided order pairs at M's
  configuration are all right (84 on F2, about 100 on the development set). The values `ein` decides 308–319 order pairs at F1, 98–99 % right.

*N.* F2 at B = 1 on the A725, as plan §7 foresaw. The sweep takes 143–170 s
and the static side 0.57–0.78 s. The capture of shapes is 0.03–0.05 s of
that, and costing 3 305 points is the rest, 0.16–0.22 ms a point on these
cores. 10³ would need a static side of 0.14–0.17 s, costing four to five
times faster. At F1a, B = 1 on the X925, the ratio clears 10³ by 1–2 %.

**Statement D, reported, deciding nothing.** The development set under
the frozen profile. `make ra4-dev-frozen` regenerates
`docs/ra4-dev-model-0.8.0.md` byte for byte, so M, O and N are D71's, quoted.
Q, which that report does not hold, is computed by `presk rank
--statement-q` on the same re-priced days (`out/ra4-dev`, ranked from
there as `ra4_dev.sh` ranks them, with the archived check):

| | development set | fresh |
|---|---|---|
| M | 89.3 % of 12 925, 89.6 % of 14 868 | F1 85.4–88.8 %, F2 91.7–92.7 % |
| O | 1.07, 1.00, 1.01 | at most 1.05 |
| N | 75–717× (0.7.0's static side, the entry run) | 218× to 2 933× |
| Q | 12 of 13 contraction nests; the scores `ein` at B = 1 in `f32` 1.08×; cost at most 5.2 % (0.7.0's static side), D71's share at most 1.11 % | 18 of 18 |

The development figure foresaw F1's M within four points, and it
understated F2's. It showed Q's one miss, which the fresh sweeps do not
repeat.

**Recorded for after 0.8.0**:
- how the L2 term combines with the load term at one-row tiles, which the
  decided misses at F1's scores `ein` point to, where the packed operand
  exceeds L1;
- the order of pairs within u, which decides M: 69–81 % right there;
- the costing of a point, most of the static side at B = 1 (N).

## D75 — The frozen model's form becomes the calibration's default (2026-09-26)

Plan 0.8.0 step 8. The user's decision, on the agent's recommendation, taken
after the freeze ended (D74). Until now `presk bench --calibrate --space
calib` and `presk refit` defaulted to the load term per group under the
maximum, and a new profile to no rule of the planned cost. That form ranks
worse than the frozen one on the calibration and on the development set
(D71: `u` 0.138 against 0.132 under the rules kept, statement M's nests
81.9 % and 85.7 % against 89.3 % and 89.6 %), while the manual calls the
frozen model 0.8.0's. The version ships the model it describes as its
default:

- `bench --calibrate --space calib`: `--combine pnorm` by default, the load
  term per group as before, and a new profile costed under `lines,overhead`.
  A profile that already has samples keeps their rules, as before, and a
  `--rules` that differs from them is refused, as before.
- `presk refit`: `--combine pnorm` by default. `--recost` costs the samples
  under the profile's own rules unless `--rules` names others, where it used
  `none`; `--rules` without `--recost` is refused by the parser (exit code 2,
  as before).
- 0.7.0's form stays one set of flags away: `--load-term none --combine max
  --rules none`.

Nothing the freeze covered changes: the planned cost, `LayoutFit`, the
combination and the calibration space are as tagged, and only which form a
command picks when none is named moves. `make calibrate-ra4`,
`calibrate-ra4-small` and `ra4-ablation` name their forms, so they do not
depend on the defaults. Checks: the definitive calibration, re-costed under
`lines,overhead` and refitted with the defaults, gives the frozen profile
back byte for byte, and so does the frozen profile re-costed under its own
rules; `make ra4-dev` and `ra4-dev-frozen` byte for byte; tutorial step 17,
now without flags, gives the same form of output as with them.

## D76 — Version 0.9.0 is a C export; the user's five decisions, and the target refused until it exists (2026-09-26)

Plan 0.9.0 step 0 (`presk-specs/implementation/presk-plan-0.9.0.md`,
presk-specs !15, merged as `f351468`). The plan was written from the design
note `presk-project/docs/presk-c-export-2026-09-26.md`. The session that
closed 0.8.0 corrected that note the same day: its section numbers now point
at the documents that hold them (spec §8.4 and §8.5, the language
reference's §7.3 and §7.5); the stages the export skips are named by their
codes (the lowering, `E5xxx`, then regularity and effects, `E3xxx` and
`E4xxx`); a residual bound per kernel, which nothing derives, is no longer
promised; and decisions 10–13 were added (floating-point contraction, lanes
and `bf16`, the thread split, the transcendentals).

**The user's five decisions** (2026-09-26, each as the plan recommended):

0. **The version.** A C export as 0.9.0. The GPU and the matrix unit, which
   D63 put "at 0.9.0 at the earliest", come after it. The export is smaller,
   runs on the machine already in use, and gives the GPU work a backend seam
   to follow.
1. **Where it lives.** A crate `presk-export-c` behind `presk export
   --target c`. Its input is the lowered kernels and a recording of the host
   program, and it is written to become the first backend at spec §8.5's
   seam when `.pskk` exists. The `.pskk` binary and `presk build` are left
   to a later version.
2. **What is exported.** One entry, at bound dimensions and constants, under
   one realization or realization plan, with optional weights. Any entry the
   native backend runs qualifies: a forward pass, a loss with its gradient,
   one optimizer step. `sample` comes in a later step of the version. The
   training loop, llm.c's other half, is not in 0.9.0.
3. **Exactness.** The C program computes the bits of `presk run --backend
   native` under the same realization and plan, on every kernel. It is built
   without floating-point contraction and uses the native backend's
   transcendental functions. Those are the `libm` crate's, a port of musl's
   C library, so the export ships musl's sources for the twelve functions
   the kernels call (`exp`, `log`, `tanh`, `sin`, `cos`, `erf` and their `f`
   forms). The alternative, the system's `libm` and a tolerance band, gives
   up bit identity on almost every real model, since softmax calls `exp`.
4. **The host by recording.** The host program is a recording of the kernel
   backend's own run at the entry's dimensions, on D65's capture of shapes,
   which runs no kernel. It is not a second interpreter of PSK-Core. The C
   therefore launches exactly the kernels, layouts, types and shapes the
   native backend launches; calls become C functions, a `stack` fold a loop.

Steps 7 (integer weights) and 8 (the SCoP mode) stay in the version, also
the user's decision. The plan's acceptance is five statements, C1–C5 (§3.8):
bit identity with the native backend (C1), the conformance band of the exact
reference (C2), compilation without warnings on GCC and Clang and a
cross-compilation to x86-64 (C3), the exported program's wall time reported
(C4), and the tokens of `presk serve` (C5).

**The plan checked against the code.** Its facts hold: the 41 primitive
cases of `presk-kernel/tests/conformance.rs`; the four nests of
`presk-cranelift/tests/layout.rs` (`linear`, `scores`, `values`,
`normalize`); the twelve `libm` symbols the Cranelift backend binds
(`presk-cranelift/src/lib.rs`); the kernel interpreter calling the standard
library for `exp`, `log`, `tanh`, `sin` and `cos` and the `libm` crate for
`erf` (`presk-kernel/src/exec.rs`); the ten entries of
`tests/plan/entries.txt`. Two passages did not, and are amended on the
step's branch (presk-specs):

- **Dropout is not a draw outside training.** §3.1 refused "an entry that
  `uses Rand`" before step 6. The course GPT's `SelfAttention.apply` and
  `GPT.apply` use `Rand` for dropout and `split_key` only. Two of the ten
  entries, and the course GPT itself, would have stayed out of steps 3–5,
  whose checks name the course GPT (step 4's layer loop, step 5's C4).
  Outside a training run a dropout mask is all ones whatever its key: the
  `language` chapter says so, and the kernel backend's host builds it so
  (`DropoutMask` in `presk-kernel/src/host.rs`). Such an entry draws
  nothing, and its keys reach no output. The refusal is now "an entry whose
  run calls `sample`"; §3.3 drops the keys that reach only `split_key` and
  such masks, so that no ChaCha20 is needed before step 6; step 3's check
  names every entry but `next_token`. The header's decision 2 already said
  `sample`, not `Rand`.
- **`loss` is a forward pass.** §3.8 counted "three of the ten
  [that] differentiate or step an optimizer (`mse_grad`, `loss`,
  `adamw.step`)". MiniGPT's `loss` is the training objective,
  `mean(cross_entropy(…))`, with no gradient. One entry differentiates and
  one steps an optimizer.

**A gradient through attention in the test set** (the user's agreement on
review, 2026-09-26). The only gradient among the ten entries was
`mse_grad`, an MLP's, so C1 would never have seen the adjoints of attention
(the softmax, the scores and values `ein`s, `split_heads`, `merge_heads`) or
of a stacked block, which are llm.c's backward pass. `tests/plan/entries.txt`
gains an eleventh entry, `loss_grad` of `tests/plan/minigpt_grad.psk`: the
gradient of `train_minigpt.psk`'s loss, at the tiny MiniGPT's constants.

- **Its head is drawn, not zero.** On the example's model, the head is
  `init = zeros`, so on the parameters `presk run` synthesizes every
  gradient but the head's is exactly zero: 1 024 non-zero values of 8 768.
  The adjoint kernels would run on zeros, and a bit-identity test on them
  would say little. The entry's model is MiniGPT built from the example's
  own `Block` and `LayerNorm`, with its head `xavier_uniform`. Every one of
  its 21 gradients is then non-zero (`tok` in the 256 values of the 16
  tokens drawn).
- **Its embeddings are written out.** A tree `params Model` whose module
  uses the imported `m.Embedding` leaves that module's `T` unbound (`E7001`,
  "unbound dimension T" in `Embedding.init`), though `params m.MiniGPT`
  binds it (`train_minigpt.psk`). The two tables are declared in the entry's
  module instead, as the course GPT declares them. That binding is a
  limitation of input synthesis, not examined further here.
- **Its gradient crosses the conformance band, natively.** The band is per
  element, `|x − x̂| ≤ ε·(|x| + a)`. Against the exact reference at 256 bits,
  the native backend's gradient lies outside it on 34 of its 8 768 elements
  in `f64`, 121 in `f32` and 99 in `bf16`, and the kernel interpreter's on
  36 in `f64`. They are elements where large terms cancel. The key bias's
  gradient is exactly zero, since softmax ignores a shift of every score of
  a row: the reference gives 1e-78, the machine executors about 1e-17. D58
  saw the same in `bf16` on `stack_prefix_grad`; this is the first time in
  `f64`. Statement C2 said "as the native backend's do" and was falsified
  by "an output outside the band". It is amended to "an output outside the
  band where the native backend's is inside it", which keeps what C2 asks
  of the C program, and the user accepted it. The entry is not added to
  the kernel conformance suite, whose assertion it would fail for this
  reason. The definition itself is question 75.

**Two sentences corrected.** `presk-cranelift`'s crate documentation and the
`native` chapter said the Cranelift backend's `libm` functions were "the same
definitions the kernel interpreter uses". That holds for `erf` only (decision
13 of the design note, decision 3 above). The chapter's next paragraph
already said the two executors differ in the last digits of those functions.
Both now say which library each executor calls.

**The refusal.** `presk export --target c` is parsed. The program is then
checked as for the PyTorch export (`typed_front`), and its diagnostics come
first, with exit code 1. Then the target is refused with `E8002`, "the C
export is not implemented in 0.9.0-dev", and exit code 2, before anything is
written. `E8001` was already the PyTorch export's, so the C export's codes
start at `E8002`. The exit code is the `cli` chapter's 2, "not implemented
in this version". A diagnostic otherwise exits 1, which stays for what is
wrong with the program. The test runs the binary
(`crates/presk-cli/tests/export_c.rs`, the CLI's first test), since the
refusal depends on no program and `tests/diag` holds programs. Nothing else
is implemented. The crate `presk-export-c` is not created, since its first
content is step 1's kernel printer. The flags of §3.1 (`--entry`,
`--realize`, `--plan`, `--weights`, `--threads`) are not added either, since
nothing would read them before step 3.

**Bookkeeping.** Version `0.9.0-dev`; the CHANGELOG's "Unreleased" section;
spec questions 73 (the C export's place in the spec) and 74 (bit identity
across executors and conformance) drafted from plan §6, and 75 (how
conformance judges an output that cancels, from the gradient above, drafted
at the user's request); the manual's `cli`
(the target and its refusal), `diagnostics` (`E8002`) and `status` (the
version's steps and statements). `dependencies.md` waits for musl, in step 2.
The tutorial still quotes `presk --help` as 0.8.0 prints it, with the version
in its first line; step 9 refreshes its outputs. The native backend's code is
unchanged. `make plan-check` passes, and `make ra4-dev-frozen` gives back
`docs/ra4-dev-model-0.8.0.md` byte for byte. On the eleven entries of
`tests/plan/entries.txt`, `main`'s release binary and this branch's print
the same kernels (`presk dump --stage kernel`) and the same native outputs
in `f64`, `f32` and `bf16` (`presk run --backend native --json --digits
20`), byte for byte.

Alternatives considered:

- Exit code 1, since `E8002` is a diagnostic. But exit code 1 then stops
  telling a program's errors from the tool's limits, and the `cli` chapter
  already reserves 2 for "not implemented in this version".
- Creating `presk-export-c` now, holding only the refusal. The crate would
  hold no code of its own until step 1, and a test of it would not test the
  command line.
- Keeping §3.1's refusal of every `Rand` entry and moving the course GPT's
  checks to step 6. That would leave steps 3–5 without the one model their
  checks are about, to avoid computing keys that no output reads.

## D77 — The kernel printer, checked by running its C inside the kernel backend; two facts about GCC (2026-09-26)

Plan 0.9.0 step 1 (§3.2, §3.5, §3.8). The crate `presk-export-c` holds the
kernel printer. It depends on `presk-kernel` only. The CLI does not change:
`presk export --target c` stays refused with `E8002` until step 3 (D76).

**What the printer writes.** `kernel_fn(k, name)` prints a kernel as one C
function. The kernel's textual form, as `presk dump --stage kernel` prints
it, is the comment above it, with one more line saying which loop the range
bounds.

- **The signature** is `static void name(const T *restrict in0, …, T
  *restrict out, int64_t lo, int64_t hi)`. `T` is `double`, `float` or
  `uint16_t` for a real buffer and `int64_t` for an index buffer. §3.2 said
  `size_t lo, hi`. The loop variables are `int64_t`, as the native
  backend's are, since an index may be loaded or computed by `SetL`.
  Comparing one with a `size_t` bound is a `-Wsign-compare` warning under
  `-Wextra`, which C3 forbids. The range is the native backend's `i64`.
  The plan is amended.
- **Locals.** Loop, real, integer and vector variables are declared at the
  top, at zero, as the native backend starts them. A parameter or a variable
  that is never read gets a `(void)`, so that `-Wall -Wextra` reports
  nothing.
- **Expressions keep the kernel's tree.** A parenthesis is dropped only
  where C's precedence and left associativity give the same tree back. So
  `a - (b - c)` keeps its parenthesis and `(a - b) - c` prints as `a - b -
  c`, and a negative operand on the right is parenthesized (`x * (-0.5)`,
  `-(-x)`). The compiler then rounds each operation where the kernel does.
  Unit tests pin the cases.
- **Constants.** A real is printed as Rust's shortest representation that
  reads back as the same number, and C compilers round decimal constants
  correctly. In an `f32` kernel the constant is first rounded to `f32` and
  printed with `f`, as `f32const(x as f32)` is. Infinities are `HUGE_VAL`
  and `HUGE_VALF`.
- **Offsets are multiplied out.** An affine index, a sum of literal multiples
  of loop and integer variables, becomes one coefficient per variable and
  one constant: `%1[i2, (i1 * 8) + 4]` over `[4, 8]` is `in1[i2 * 8 + i1 * 8
  + 4]`. Any other index, such as a loaded one, is a term times its stride.
  This is integer arithmetic, so no bit moves. Signed overflow is undefined
  in C where Cranelift wraps, and no index comes near 2⁶³.
- **Lanes.** Each `VSet` and `VStore` becomes `for (int l = 0; l < L; l++)
  …` over local arrays `vN[L]`. The native backend computes every lane
  before it stores any. So a `VStore` whose value reads the output goes
  through a temporary `t[L]`, since a lane loop could otherwise read a lane
  it has already written. The lowering produces no such store today; a
  hand-written kernel tests it.
- **`bf16`.** A load widens with `presk_bf16_to_f32` and a store rounds with
  `presk_f32_to_bf16`. Both are in `presk_rt.h`, the second a copy of
  `presk_rt::f32_to_bf16` (ties to even, a NaN kept and made quiet).
  Arithmetic is in `float`.
- **Conversions.** A real becomes an integer through `presk_f64_to_i64`:
  rounded toward zero, saturating, and 0 for a NaN, as Cranelift's
  `fcvt_to_sint_sat` is. C's own cast is undefined out of range.
- **The range.** `ranged(k)` is the native backend's rule: the loop at the
  depth of the kernel's split, or the outermost when it is parallel. It is
  copied, not shared, so that the native backend's code does not change.
  That loop runs `for (iN = lo; iN < hi; iN++)`, and the loops above it
  run in full. A kernel with no such loop leaves `lo` and `hi` unused.
- **Functions.** `sqrt` and `fabs` (and their `f` forms) are C's, correctly
  rounded and exact. `exp`, `log`, `tanh`, `sin`, `cos` and `erf` are calls
  to `presk_exp`, `presk_expf`, …, which `presk_rt.h` declares. Step 2
  defines them with musl's sources. `calls(k)` lists what a kernel calls.
- **Refusals.** A kernel reading integer weights (`BufKind::Quant`) is
  refused with a message, until step 7. So are forms the lowering never
  produces and on which the native backend and the kernel interpreter
  disagree: a store to an input, a real load from an index buffer, a real
  store to an index output.
- `entry_fn` wraps a kernel in the native backend's calling convention,
  `void name(void *const *bufs, int64_t lo, int64_t hi)`, for a loader that
  knows no kernel's signature.

**The check is an executor, not a harness.** The plan said "a small
generated harness". Instead, the crate's tests hold a `KernelExec` over the
printed C. The kernels of a set of runs are printed and deduplicated by
their text. They are compiled 48 to a file, in parallel, with the export's
flags (`-std=c11 -O2 -ffp-contract=off`), `-Wall -Wextra -Werror` and
`-fPIC`, then linked into one shared library with `-lm`, loaded with
`dlopen`, and called through `entry_fn`. A `KernelBackend` over this
executor runs an entry or a primitive with its kernels in C, and its raw
bits are compared with the native backend's on the same point. The reasons:

- **The backend itself drives the C.** The lowering, the inputs converted
  at the kernel's edge, the splits checked against the executor's workers
  (condition 5; the executor states four, as `layout.rs`'s native runs do)
  and the outputs chained from kernel to kernel are the ones the native
  backend has. The existing tests carry over with the executor swapped:
  `layout.rs`'s plan per point, and the conformance suite's primitive
  calls.
- **Nothing is written only to be thrown away.** A harness needs a file
  format for inputs and outputs and a generated `main` per batch. Step 2's
  `.psk.d` reader and step 3's host replace both.
- **It stays useful.** When step 3's host disagrees with the native
  backend, running the entry on this executor tells a kernel from the host.
  Step 2's check is this one with `presk_libm.c` in place of the stand-ins
  below.

The cost:

- **A dev-dependency.** The `libc` crate, for `dlopen`, `dlsym` and
  `dlerror`, is a dev-dependency of `presk-export-c`. It is already in
  `Cargo.lock` through `cranelift-jit` and `gmp-mpfr-sys`, so nothing new is
  downloaded or built. `libloading`, the usual wrapper, is not in the lock.
- **Unsafe code**, in the tests only.
- **A C compiler for `make test`**: `cc`, or the one `CC` names. One was
  already needed to build `rug`'s GMP and MPFR.
- **The C runs in the test's process**, so a crash in a kernel would take the
  test with it. The kernels run the same loops as the native backend on
  the same buffers, so an access out of bounds would crash the native
  backend first.

Both are recorded in `dependencies.md` and `presk-tooling.md`.

**Every range is cut three ways.** A kernel whose loop may be divided is
called three ways:

- over its whole range;
- over the pieces the native backend gives its workers (the split's count,
  or the executor's four);
- one iteration at a time.

The pieces are called last first. In the last two cuts, each piece first
runs alone on an output filled with sentinels (signalling NaNs, which
arithmetic and the `bf16` rounding make quiet, or an index no shape
reaches), and the elements it changes must be changed by no other piece.
A range that bounded the wrong loop, or none, would recompute elements,
give the same bits, and be a race between threads under step 5's OpenMP.
Checked by mutation: a ranged loop started at 0 instead of `lo` fails at
once ("the ranges (1, 2) and (0, 1) both write element 0").

**The transcendentals are compared too.** The prompt, following the plan,
said that kernels calling a transcendental would be compiled but not
compared until step 2. Here the executor's library defines `presk_exp`, …
as stand-ins that call, through pointers the loader sets, the `libm`
crate's functions, which are the ones the native backend calls. So those
kernels are compared as well, and are bit-identical: 30 of the primitives'
123 distinct kernels, and the 11 of the layouts' (the softmax of
`normalize`). This checks the printer (the call, its type, its argument),
not the functions. Step 2's comparison of `presk_libm.c` with the `libm`
crate is unchanged.

**Results** (falicon, GCC 13.3):

- **The 41 primitive cases** of `presk-kernel/tests/conformance.rs`, in
  `f64`, `f32` and `bf16`, scalar and at each type's natural width (two
  `f64` lanes, four `f32`, `bf16` scalar), under the three cuts: 738 runs,
  123 distinct kernels, all bit-identical.
- **The four nests of `presk-cranelift/tests/layout.rs`** in the three
  types: every point of each space, 1 491, and `layout.rs`'s thread splits
  (depths 0–2 over 2 and 4 workers on every fifth point), 1 618 run and 200
  refused by condition 8 as there. Each run under the three cuts: 2 000
  distinct kernels, all bit-identical.
- **Hand-written kernels** for forms the lowering does not produce: the
  overlapping vector store above, `SetI`, `SetL`, `If`, `Select` on
  integers, a saturating `ToI` (−10³⁰⁰ and a NaN), an index output, −∞, a
  negative constant and a double negation. All bit-identical.
- **Time.** The crate's tests take 11 s in `make test`: the native runs
  3.4 s, compiling 2 000 kernels 6 s on twenty cores, the C runs 1.5 s. An
  `#[ignore]`d test states the splits on every point instead of every
  fifth: 7 950 splits and 3 978 kernels, bit-identical, 26 s (`cargo test
  -p presk-export-c -- --ignored`). `make test` keeps the first. It checks
  every layout; only the splits are the sample `layout.rs` takes.
- **Mutations**, one at a time, then reverted. Built with
  `-ffp-contract=fast`, the first `f64` dot of the primitives moves. So it
  does built as GCC's GNU mode with no contraction flag (`-std=gnu11
  -O2`). A `bf16` store that truncates instead of rounding moves the first
  `bf16` add.

**Two facts the plan got wrong** (found in step 0's session, checked again
here):

- **GCC does not implement `#pragma STDC FP_CONTRACT OFF`.** It ignores the
  pragma and warns about it under `-Wall` ("ignoring '#pragma STDC
  FP_CONTRACT'", `-Wunknown-pragmas`), which C3 forbids. The pragma is in
  `presk_rt.h`, which every file includes first, under `#if
  defined(__clang__)`. On GCC the flag holds. §3.5 rule 1 is amended. The
  same warning meets §3.6's OpenMP pragmas: GCC warns about `#pragma omp
  parallel` without `-fopenmp`. So "without OpenMP, the pragmas are
  ignored" needs them under `#ifdef _OPENMP`, and §3.6 is amended for step
  5.
- **`-std=c11` already turns contraction off on GCC.** GCC's default for
  `-ffp-contract` is `off` in the ISO modes and `fast` in the GNU modes; its
  default mode is `gnu17`. At `-O2`, `a + b*c` gives `fmadd` by default and
  under `-std=gnu11`, `fmul` and `fadd` under `-std=c11`, and `fmadd` again
  under `-std=c11 -ffp-contract=fast`. (`gcc -Q --help=common` says `fast`
  in every mode; the assembly is what counts.) The flag stays in the
  Makefile, since it holds when a user builds in a GNU mode, and Clang reads
  it. But §3.5 rule 1's test, "the bits must move under GCC's default", must
  build in a GNU mode, without `-std=c11`, or it finds nothing. This step's
  check shows the bits moving so. The test itself stays step 2's, and the
  plan is amended.

**Left for later steps.**

- **`static` kernels in another file.** §3.1 puts the kernels in
  `kernels.c` as `static` functions, which `model.c` calls. Step 3 either
  includes `kernels.c` in `model.c` or drops `static`.
- **`restrict` on the output.** It says no input aliases the output. The
  native backend gives every launch a fresh output buffer, and step 3's
  arena must never hand a kernel an input as its output.

Alternatives considered:

- **The plan's generated harness**, a C program per batch of kernels whose
  `main` reads inputs from files and writes outputs. No dependency, and a
  crash stays in a child process. But it needs a file format and a driver
  that steps 2 and 3 replace. It also tests kernels one at a time on
  recorded inputs, not inside the backend.
- **`libloading`**, the usual wrapper over `dlopen`. It is not in the lock,
  and three functions of `libc` suffice for a test.
- **Compiling each kernel on first use** inside the executor. That is one
  compiler run per kernel, about 2 000 of them, where the batch takes 6 s.
- **Every parenthesis printed.** Simpler, and C would build the same tree,
  but the result is unreadable, and the plan asks for readable C.
- **Loop variables scoped in their `for`**, as llama2.c writes them. That
  breaks down on `SetL`, which assigns a loop variable outside any loop
  over it. Declaring every variable at the top keeps the native backend's
  semantics with no case analysis.

## D78 — The runtime, and musl 1.1.22's transcendental functions, bit-identical to the `libm` crate (2026-09-26)

Plan 0.9.0 step 2 (§3.4, §3.5 rules 1–3, §7). `presk-export-c` gains the C
that every exported program holds beside its own, `RUNTIME`: `presk_rt.h`,
`presk_rt.c` and `presk_libm.c`. The CLI does not change: `presk export
--target c` stays refused with `E8002` until step 3.

**The sources of `presk_libm.c`.** The plan said "musl's C sources". That
needs a version, since musl changed the functions the native backend calls.

- **The `libm` crate is musl before 1.1.23.** The native backend calls the
  crate (0.2.16, MIT) for `exp`, `log`, `tanh`, `sin`, `cos` and `erf`,
  and their `f` forms. Its files keep musl's headers ("origin: FreeBSD
  /usr/src/lib/msun/src/e_exp.c"). Read side by side with musl 1.1.22's
  files, each of the twelve and each helper they call follows the same
  operations in the same order: `expm1` and `expm1f` for `tanh`, `scalbn`
  and `scalbnf`, the `__rem_pio2` family, and the `sin` and `cos` kernels.
  On aarch64 the crate's arch module replaces only `fma`, `rint` and
  `sqrt`, and none of these calls them.
- **musl 1.1.23 rewrote four of them.** It replaced `expf` and `logf`
  (committed in 2017) and `exp` and `log` (2018) with ARM's table-based
  routines. Current musl's four differ from the crate on 97 311, 98 261,
  83 916 and 85 639 of 10⁶ inputs (`exp` and `expf` on [−20, 20], `log`
  and `logf` on [0, 4]; measured in C against this step's `presk_libm.c`,
  which is the crate's). Shipping them would have broken C1 on every
  softmax.
- **Two later fixes touch the other files.** In 2020 musl fixed
  `__rem_pio2` and `__rem_pio2f` for directed rounding modes. The crate
  does not have that fix, and 1.1.22 does not either. In 2021 musl fixed
  `expm1f`'s overflow threshold (commit 964104f9, released in 1.2.3). The
  crate has that fix, so `expm1f.c` is taken at that commit. `tanhf` calls
  `expm1f` only below |2x| = 20, far under the threshold, so the choice
  changes no output. It follows the crate.
- **The choice.** Every file is musl 1.1.22's (tag `v1.1.22`, 2019-04-09),
  the last release before the rewrite, except `expm1f.c`, taken at 964104f9.
  The check below decides, and it passes.
- **The crate's departures from musl** found in the reading change no
  value. `expf`'s underflow check raises its flag through `0x1p-126`
  where musl uses `0x1p-149` (the crate's comment asks why). The crate's
  `cos` returns 1 for a tiny argument through `x as i32 == 0` where musl
  forces an evaluation, and it forces evaluations with `read_volatile`.
  These raise or do not raise flags. They do not change a result.

**The edits to musl's files**, none of which changes an operation. They are
listed at the top of `presk_libm.c`:

- **Names.** The twelve are `presk_exp`, `presk_expf`, …, as `presk_rt.h`
  declares them. The helpers are `static` and named `presk_…`: `scalbn` is
  `presk_scalbn`, `__sin` is `presk_k_sin`, `__sindf` is `presk_k_sindf`,
  `__rem_pio2` is `presk_rem_pio2`, and `erf.c`'s `erfc1` is `presk_erfc1`
  (`erff.c`'s is `presk_erfc1f`). Nothing collides with the system's libm,
  which a program links for `sqrt`. The object needs nothing from it on
  aarch64, where GCC inlines `floor` and `fabs`, both exact.
- **Constants moved.** The files' constants share names (`P1`, `ln2_hi`,
  `pp0`, …), so each file's are moved from file scope into the function
  that reads them. `erf.c`'s and `erff.c`'s table is divided among `erfc1`,
  `erfc2` and `erf`, which read disjoint parts of it.
- **Left out.** `erfc` and `erfcf`, which nothing calls here, and musl's
  internal declarations. Of `libm.h`, the macros the files use are copied.
- **`M_PI_2`,** which `<math.h>` does not define under `-std=c11`, is
  defined with musl's value when it is missing.
- **Two edits for `-Wall -Wextra -Werror` under GCC 13** (statement C3):
  - `FORCE_EVAL` reads back the volatile it writes, `(void)__x`, since GCC
    reports it as set but not used.
  - `__rem_pio2_large`'s `fq[20]` starts at zero, as the crate's does. GCC
    reports `fq[0]` as maybe used uninitialized, although the function
    writes `fq[0..jz]` before it reads them.
- **Licences.** musl's MIT notice and its paragraph on the math code head
  the file, and each file's notice (Sun's, for the functions from FreeBSD's
  msun) heads its section, as musl keeps them. Both are permissive, so they
  are compatible with the compiler's Apache-2.0.

The file was assembled by a script from musl's git (renames outside comments,
the moves, the two edits), then read. The script is not kept: the file's
header lists the edits, and `git diff` against musl's files shows them.

**The comparison** (`tests/libm.rs`, §3.5 rule 2). `presk_libm.c` is
compiled with the export's flags and `-Wall -Wextra -Werror`, loaded with
`dlopen`, and each function is called on 10⁶ inputs beside the crate's.
Their bits are compared, NaNs included.

- **The inputs** of each function:
  - the special values, ±0, ±∞, NaNs quiet and signalling with payloads,
    the least and greatest subnormals, the least normal and the greatest
    finite value, each with its neighbours;
  - the thresholds of its branches, stated as musl states them, on the
    high word of an `f64` or the bits of an `f32`, each with its
    neighbours (overflow, underflow, the ends of each approximation, and
    `expm1`'s thresholds at half the argument for `tanh`);
  - multiples of ln 2 / 2 for `exp`, every power of two and the
    neighbourhood of 1 for `log`;
  - for `sin` and `cos`: multiples of π/4 up to 5 000, multiples of π/2
    up to the medium range's end (2²⁰·π/2 in `f64`, 2²⁸·π/2 in `f32`),
    every power of two, and the double nearest a multiple of π/2
    (6 381 956 970 095 103·2⁷⁹⁷), where the large reduction,
    `__rem_pio2_large`, must be right to the last bit;
  - then, to 10⁶: 40 % any finite value, 5 % any bits, and the rest
    uniform over the function's domain.
- **Results.** 12 × 10⁶ inputs, none differs, in 0.4 s. So the 10⁶ run in
  `make test`, and no sample was needed.
- **Every `f32`.** An `#[ignore]`d test runs `expf`, `logf`, `tanhf`,
  `sinf`, `cosf` and `erff` on all 2³² `f32`s, on twenty threads: none
  differs, 46 s (`cargo test -p presk-export-c --test libm --
  --ignored`).
- **Mutations.** Built with contraction (`-std=gnu11`, or
  `-ffp-contract=fast`), ten of the twelve move: `exp` on 332 inputs,
  `expf` on 286, `log` on 5 092, `logf` on 5 088, `tanh` on 704, `tanhf`
  on 587, `sin` on 7 623, `cos` on 5 585, `erf` on 17 303 and `erff` on
  19 618. `sinf` and `cosf` do not move, since their kernels compute in
  `f64` and round to `f32` once. Scaling the tail of `__rem_pio2f`'s
  medium reduction by 1.001 moves `sinf` and `cosf` on about 248 000 inputs
  each, the first at the medium range's end. A one-ulp change to a
  high-order coefficient of `__sindf` moves nothing: the kernel computes in
  `f64` with room to spare. So the test tests the functions, and the
  contraction rule holds for them too.
- **For the record, the system's libm.** glibc 2.39 differs from the crate
  on 2.9–9.8 % of each `f64` function's inputs and on 1.2–9.8 % of each
  `f32` function's, over the ranges above: `exp` 9.8 %, `log` 8.4 %,
  `tanh` 7.7 %, `sin` and `cos` 3.1 %, `erf` 2.9 %; `expf` 9.8 %, `logf`
  8.6 %, `tanhf` 7.7 %, `erff` 2.3 %, `sinf` 1.4 %, `cosf` 1.2 %.
  That is the plan's alternative of a tolerance band, which gives up bit
  identity.

**§7's fallback is not used.** No function differs, so the stand-ins that
step 1's executor called go. `tests/cexec` now compiles `presk_libm.c` into
the library with the kernels. Step 1's check runs so, bit-identical: 738
primitive runs (180 of them calling `presk_libm.c`), and the 1 491 layouts
and 1 618 splits of the four nests (the softmax of `normalize` included).
The `#[ignore]`d test of every split of every point is too: 7 950 splits and
3 978 kernels, 27 s. The `libm` crate stays a dev-dependency, for the
comparison.

**The contraction test** (§3.5 rule 1, D77). `contraction_moves_bits_in_gccs_gnu_mode`
builds the primitives' kernels and `presk_libm.c` as GCC builds by default,
`-std=gnu11 -O2` with no `-ffp-contract`, and runs each primitive once
against the native backend. 14 of the 246 runs move: `dot`, `dot_vjp_l`,
`dot_vjp_r` and attention's `ein` in `f64`, scalar and with lanes, and
`dot`, `dot_vjp_r` and `ein` in `f32`. None of them is a run through
`presk_libm.c`, whose six-element inputs here miss the inputs where the
functions move. On aarch64 the test asserts that some run moves, or it
would not test the rule. Elsewhere it reports.

**The runtime** (`presk_rt.c`, §3.3, §3.4).

- **Conversions.** `presk_convert(dst, to, src, from, n)` converts a buffer
  between the real machine types as the native backend converts a buffer
  at a kernel's edge. A buffer already of the kernel's type is copied as it
  is (`kernel_input` lends it). Otherwise it goes through `f64`: widening
  is exact, `f64 → f32` rounds to nearest, and `→ bf16` rounds to `f32`
  and then to `bf16`, as `Buf::of` does (`f32_to_bf16(x as f32)`). That is
  two roundings, and they differ from one where the first lands on a tie.
  The header gains `presk_f64_to_bf16`, which does both.
  - Checked on 200 000 reals in the nine pairs of types, bit for bit
    against `Buf::of(to, b.to_f64())`. The reals include NaN payloads,
    signalling NaNs, the subnormals of each type, the ties of each
    rounding, and 2 364 values where rounding through `f32` differs from
    rounding once.
- **The `.psk.d` reader.** The C parses no Presk type. The export (step 3)
  states the sample type as the file's header writes it, and its leaves in
  order, each a `presk_leaf { bound, n }` (`bound` 0 for reals).
  `presk_open` checks the magic, the version, the type byte for byte, and
  a length of exactly `count` samples of the leaves.
  - A shorter file is `E9003` and a longer one `E9005`, as `presk-data`'s
    `with_layout` judges a `params` file. That is stricter than its
    `Reader::open` on a closed type, which accepts a longer file. A file of
    other dimensions is thus never read at the wrong offsets.
  - `presk_read` reads the next sample. Each real leaf is read as `f64` and
    stored in the type the program asks for that leaf, converted as above,
    once, at load. Each index leaf goes into `int64_t`, and each index is
    checked below its bound (`E9004`). `presk-data` checks bounds when it
    writes, not when it reads, but in C an index past its bound would make
    a gather read outside its buffer.
  - `presk_load` reads a file of one sample: a checkpoint, or an entry's
    inputs.
  - Errors carry `presk-data`'s codes, `E9002` to `E9005`, and messages in
    its words. `presk_error()` gives the text. No code is added.
- **The writer.** `presk_write` writes `count` samples as `presk-data`
  writes them. Reals of any type are written as `f64`, which is exact.
  Indices are written in their bound's width, each checked, and a
  negative one refused. Nothing is left of a file whose writing fails.
  Every integer is read and written byte by byte, little-endian whatever the
  machine.
- **The reader and writer checked** (`tests/runtime.rs`), on four sample
  types with one to three samples: every index width (1, 2, 4 and 8 bytes),
  a scalar, an empty leaf, `idx[1]`, leaves larger than the reader's 32 KiB
  pieces, and a parameter tree.
  - Files written by `presk-data` read in C into `f64`, `f32` and `bf16`
    give `Buf::of` of `presk-data`'s values, bit for bit.
  - Files written in C from `f64`, `f32` and `bf16` buffers are, byte for
    byte, the files `presk-data` writes from the same reals, and
    `presk-data` reads them back: 24 round trips.
  - Fifteen errors, each with its code. `presk-data`'s reader gives the
    same code on the four file-level cases it checks.
- **The arena.** `presk_arena_init` allocates one block, aligned to 64
  bytes. `presk_alloc` bumps it, each size rounded up to 64.
  `presk_mark` and `presk_release` free a function's temporaries.
  `presk_release_keep` keeps its result: it moves the result down to the
  mark with `memmove`, which copies bits. The block's `peak` is recorded,
  so that step 3 can compare it with the size it computes.
  - `presk_try_alloc` returns `NULL` when the block is exhausted.
    `presk_alloc` then stops the program with a message, since the export
    sized the block from the recording and exhaustion is its fault.

**Time.** The crate's tests take 13 s in `make test`: the kernels 10.5 s,
the functions 0.4 s, the runtime 0.2 s. The contraction test adds a
compilation of the primitives' kernels.

**Left for step 3.**

- **The type as a C string.** The export prints the sample type into the C
  as a string literal. Its non-ASCII characters (`ℝ`) must be written as
  octal escapes: a `\x` escape would take the digits that follow it.
- **`presk_convert` in the host.** The host calls it where a buffer
  reaches a kernel of another type, as `kernel_input` converts it.
- **The block's size** comes from the recording's peak of live bytes, each
  buffer rounded up to 64 bytes as `presk_alloc` rounds it.

Alternatives considered:

- **FreeBSD's msun,** the origin of musl's files. musl's are its
  descendants with musl's own edits (`double_t` temporaries, cleanups,
  fixes), and they are the files the crate ports, so they are the nearer
  source. Not tried.
- **A C transcription of the crate's Rust.** It would be a new text to
  maintain, where musl's is published and the crate says it ports it.
- **Current musl with the four old files put back.** Among the files used
  here, current musl differs from 1.1.22 only by the 2020 `__rem_pio2`
  fix, which the crate lacks, and the 2021 `expm1f` fix, which it has
  (`scalbn`, `scalbnf` and the rest are unchanged). So it is the same file
  with one more change to undo, stated less simply.
- **The system's libm and a band,** the plan's fallback. It gives up bit
  identity on almost every model (above).
- **`#pragma GCC diagnostic`** around `__rem_pio2_large`, instead of
  zeroing `fq`. It changes no code, but it ties the file to GCC's
  warning names, which Clang reads as unknown. Zeroing is what the crate
  does.
- **Parsing the Presk type in C**, so that the reader derives the leaves
  itself. It would be a second parser of types to keep in step with
  `presk-syntax`, where the export already knows the layout.

## D79 — The host, straight-line: a symbolic mirror of the kernel backend's host, recorded on the native run (2026-09-26)

Plan 0.9.0 step 3 (§3.1, §3.3, §3.4, §3.8). `presk export FILE --target c
--entry E [--dim] [--const] [--realize] [--weights] [--seed] [--no-exact]
-o DIR` writes the entry as a C program that computes the bits of `presk run
--backend native`. Its host is straight-line; its structure is step 4's.

**The recorder** (`presk-kernel/src/record.rs`).

- **Why a mirror.** The kernel backend's values carry no identity. A `let`
  reads its operand with `env.get(v).cloned()`, `slice_tree` copies a slice
  of a stacked tree, and `merge_axes` clones its data. A recorder that
  watched launches could not tell which buffer a launch reads, least of all
  on the shape capture's zeros. So the recorder keeps a symbolic value
  (`Sym`) beside each value the host holds. A tensor is a `View`: a buffer
  of the recording (an input, a launch's output, a fill, a copy) at an
  element offset, with its shape. The recorder follows calls, `let`s,
  tuples, records, `Proj`, `Field`, `SetField`, `SetElem`, closures and
  their `Apply`, the fold's slices (views at offsets) and its scan's stack.
  For each primitive it follows the host's decomposition into launches: a
  parameter tree by a scalar and a tree by a tree, leaf by leaf in
  `collect_leaves` order; a fusion group over a tree, the scalars riding
  along; a pack before its contraction; `sum_squares` leaf by leaf, each
  result read back and added in `f64`; a unary function of a host scalar,
  launched on a one-element buffer and read back.
- **The hooks** are an `Option<Box<Recorder>>` on `KernelBackend`, `None`
  by default, and thirteen `if let Some(r) = self.recorder…` calls: the
  entry's arguments (`Backend::call`), a function entered and left, a `let`
  ended (`let_end`, `fused`, `closure`), a `Call`'s and an `Apply`'s
  arguments, a fold begun, each iteration begun and ended, the fold ended,
  and each launch (the kernel's index, whether it is a pack's copy, the
  output's shape). The launch hook runs after the launch and reads only the
  kernel list. Nothing the backend computes changes. The other edits are a
  type annotation on `call`'s argument vector and, in `launch`, a flag read
  off the key (`pack|`).
- **It decides nothing.** Each launch the mirror expects is consumed from
  the backend's own event queue. A launch the mirror did not expect, one it
  expected that did not happen, or an operand of the wrong kind is an
  internal error (code `internal`, exit code 2), never a guess. Control flow
  is static (no conditional in PSK-Core, a fold's count is a dimension), so
  one run records the host for every input of these shapes.
- **Conversions at a kernel's edge.** Where a view's buffer is not of the
  type the kernel reads (a mixed realization), the recording converts it
  into a new buffer through `f64`, as `launch` does with `Buf::of(k.dtype,
  b.to_f64())`.
- **One liberty, which changes no value.** The host's `with_data` makes a
  new buffer of a slice, of a stack, of a slice written back, and turns a
  `bf16` source into an `f64` buffer. The recording keeps a slice as a view
  of the `bf16` buffer, and gives a stack and the rest `with_data`'s type.
  The values are equal, and a kernel reading them converts to its own type
  either way, so its inputs have the same bits.

**Recorded on the native run, not on the shape capture.** The plan and the
prompt said to record on the capture of shapes (`NoRun`, D65), with the
recorder switched on for the entry's call only, since the capture runs the
initializers of `params` arguments first. The export records the native
backend's own run instead:

- **The same backend.** Its workers, lanes, plan and kernel cache are the
  ones the C copies, so no second backend needs to be kept in agreement
  with the native one.
- **The outputs are needed anyway.** `conformance/` needs the native
  outputs, so the entry runs natively in any case; the recording costs
  that one run.
- **No initializer runs on it.** The arguments are made as `presk run` makes
  them, in the interpreter, before the backend exists, and the recorder is
  set just before the entry's call.

The course GPT at `B = 1`, `T = 256` runs natively in under a second. The
shape capture stays `presk plan`'s.

**Dead launches.** A gradient's host runs the forward pass twice: once for
the primal result, which `vjp` discards, and once inside the reverse pass.
The C program copies it: 125 of `loss_grad`'s 767 launches and 15 of
`mse_grad`'s 46 compute values the result does not depend on. The export
counts them (`host::live_launches`), and the README and its summary line say
so. Removing them changes no bit, and is left to the structured host or
later, so that step 3's host is the native backend's, launch for launch.

**Scalars fixed at export.** A `.psk.d` file holds tensors only, and the
test set's entries take scalar arguments: `eps` (`f64`, the norms), `t` and
`total` (`nat`, `adamw.step`), `temperature` (`next_token`). They are fixed
at export as `presk run` synthesizes them (the parameter's default, or
`optimizer_input`'s `t = 3`, `total = 10`), like the dimensions, and the
README lists them. They reach the C as literals: a scalar made a one-element
buffer (`presk_fill`, rounding as `Buf::of` does) or an operand of the
host's `double` arithmetic. A key argument is dropped, and so is a key that
reaches only `split_key` and dropout masks, which outside a training run are
fills of ones (§3.3, amended in step 0). The alternatives were a command-line
argument per scalar or a scalar file, each a second input channel the
native run does not have.

**The files.**

- `model.h` declares the entry, `presk_<entry>(presk_arena *, const
  presk_<entry>_in *, presk_<entry>_out *)`. Its inputs and outputs are
  structs of typed pointers, one per tensor leaf, named after the argument
  and the leaf's path. Presk identifiers become C ones by a mangling:
  ASCII letters, digits and `_` kept, Greek letters spelled out (`θ` is
  `theta`), a hat `_hat`, anything else `_`, runs collapsed, a C keyword
  suffixed. So `LayerNorm{8}.apply` is `presk_LayerNorm_8_apply`, and
  `θ.l1.w` is `theta_l1_w`. The header also defines the arena's size.
- `model.c` holds the function, straight-line. `kernels.c` holds its
  kernels, deduplicated by text, each `static` under its PSK-Kernel text,
  and `model.c` includes it: D77 left the choice between that and dropping
  `static` behind a header. Including keeps one translation unit, in which
  the compiler may inline each call, and needs no header.
- `main.c` is apart from `model.c`, where plan §3.1 put `main`: a program
  embedding the entry links `model.c` without it.
- The `Makefile` keeps the flags bit identity rests on (`STRICT = -std=c11
  -ffp-contract=off`) apart from `CFLAGS = -O2 -Wall -Wextra`, which a user
  may change.
- `README.md` says what was exported (entry, dimensions, constants,
  realization, seed, scalars fixed, the files' types, launches, dead
  launches, arena, the commit of the program's repository with `-dirty`)
  and which flags must not change.
- `conformance/check.sh` builds, runs, and compares the outputs with the
  native backend's by `cmp`.

**The conformance files.**

- **Closed types.** The inputs and outputs are `.psk.d` files of closed
  types: the tuple of the leaves' types with literal dimensions and no axis
  names (`(ℝ[2, 4], ℝ[2, 4])`), or the one leaf's type. `presk data`
  reads them without the program. An output of type `(params M, (params M,
  params M))`, `adamw.step`'s, could not have been read otherwise, since
  `presk-data` handles `params` files only through a layout the program
  gives.
- **The weights.** The first argument, when it is the parameters of a
  module, is the weights, in its checkpoint form: `params M`, `M` the
  module's name, as `presk train` writes one. `load_params` never compares
  the type's text, but `presk_open` compares it byte for byte. Without
  `--weights`, the export writes the parameters `presk run` synthesizes to
  `conformance/weights.psk.d` under that type. With `--weights`, the C
  expects the header of the checkpoint given, read at export, and the
  script names the checkpoint's absolute path. A stack or a composition of
  modules, which `presk train` does not train, is an input like the others.
- **What the files hold.** The inputs are the exact values rounded once to
  `f64`, which is what the native backend rounds to its type. The outputs
  are `f64`, exact for every type. The type's text is printed into the C
  with octal escapes (D78).

**The arena** holds the inputs and every buffer the host makes, each rounded
to 64 bytes; nothing is released during a call. The export computes it as
a sum, not as a peak of live bytes: a straight-line host frees nothing, so
the peak is the total. Examples: `mse_grad`'s 14 848 bytes; the course
GPT's at `B = 1`, `T = 256`, 352 MB in `f64`, 176 MB in `f32`, 88 MB in
`bf16`. Release per call is step 4's.

**The codes** (`presk-cli`'s tests run the binary on each):

- `E8002` changes meaning: it refused the whole target until this step. It
  now refuses an option the export implements in a later step (`--plan`
  and `--threads`, step 5), with exit code 2, "not implemented in this
  version".
- The refusals of an entry, exit code 1:
  - `E8003`: unbound dimensions, or a shape left open;
  - `E8004`: an initializer, a host primitive the C runtime does not
    provide (`M.init`);
  - `E8005`: `sample`, until step 6;
  - `E8006`: a key in a kernel's operand or in the result;
  - `E8007`: integer weights, until step 7, naming the argument;
  - `E8008`: a scalar in the result, which a `.psk.d` cannot hold.
  - No test entry reaches `E8004`, `E8006` or `E8008`. The tests use
    `Linear{4,4}.init`, `tests/diag/export_c_key.psk` and
    `tests/diag/export_c_scalar.psk`.
- The PyTorch export's options are refused with `--target c` (`--mode`,
  `--shapes`, `--stack`, `--conformance-entry`, `--train`, `--serve`, and
  `--opt` and `--profile`, which rewrite the program only for the PyTorch
  conformance data). This target's options are refused with `--target
  pytorch`. `--mode` and `--stack` lost their clap defaults so that giving
  them can be told apart. These usage errors have exit code 2 and no code.
- The options are checked before the program, since they do not depend on
  it. Step 0 had the program's diagnostics first, when the only answer was
  the refusal. A program with errors still reports them with exit code 1
  and writes nothing.

**`--no-exact`** was not in the prompt. The course's trained checkpoint
fixes `T = 256` (the position table's size), where the exact reference at
256 bits would take hours. The option leaves `exact.psk.d` out, and the
check script skips the distance when it is absent.

**C1 and C2: how each is checked.**

- **C1** is the program's own `conformance/check.sh`: build, run, and `cmp`
  the outputs with `native.psk.d`.
- **C2** is reported by `presk-cli`'s tests: the elements outside the band
  of `presk-kernel/tests/conformance.rs` (`tol`) are counted for the C
  program and for the native backend. The test asserts that the C
  program's are no more than the native backend's, which C1 already
  implies.
- **The script's distance.** With `presk` on the `PATH`, the script also
  prints `presk data diff`'s ℓ2 distance to `exact.psk.d`.
- **Why C1 carries C2.** Now that no function differs from the `libm` crate
  (D78), bits identical to the native backend's have its band status, so C2
  needs no second measurement.

**The zero-output problem.** On the parameters `presk run` synthesizes,
`MiniGPT.apply`'s logits are all zero, and so are the course GPT's `GPT.apply`
at any size, since both heads are `init = zeros`. C1 on them compares zeros.
So every entry whose first argument is a module's parameters runs twice in
the test: on the synthesized parameters, and with every value an initializer
set to exactly zero (the head, the biases, the shifts) drawn from a fixed
sequence, through `--weights`. With them drawn, every output of the forward
entries is non-zero. The gradients keep the zeros their structure gives
(the rows of the tokens not drawn, the key bias's, D76), and `adamw.step`
the moments of the leaves whose gradient and state, inputs synthesized by
the initializers and not drawn, are zero.

**Results** (falicon, GCC 13.3):

- **The test set** (`make test`): every entry of `tests/plan/entries.txt`
  but `next_token`, in `f64`, `f32` and `bf16`, on the synthesized
  parameters and with their zeros drawn: 60 programs. MiniGPT's four
  entries also run under a mixed realization, `tests/plan/mixed-realization.json`:
  `f32` with seven sites in `f64` or `bf16`, 69 conversions at kernel
  edges in `MiniGPT.apply` and 343 in `loss_grad`. So 68 programs in all.
  - Every one is bit-identical to the native backend and built by its own
    `Makefile` without a warning.
  - They take 4.8 s on twenty cores, and `make test` gains about 7 s with
    the mutations below. `make test` takes 2 min 12 s in all, compilation
    included.
- **C2.** The elements outside the band are the native backend's, element
  for element (the counts below hold for both):
  - `loss_grad`: 34, 121 and 99 of 8 768 in `f64`, `f32` and `bf16`, as D76
    measured; with its zeros drawn 40, 93 and 73.
  - Forward passes cross it too, in `f32` and `bf16`:
    `SelfAttention.apply` 1 and 4 of 256 on the synthesized parameters, 2
    and 8 with its zeros drawn; with their zeros drawn, `MiniGPT.apply` 5
    and 35 of 1 024 and the small course GPT 7 and 26 of 1 472. Under the
    mixed realization, with their zeros drawn, `MiniGPT.apply` 26 and
    `loss_grad` 136.
  - These are logits whose sums cancel. MiniGPT's five in `f32` are
    6·10⁻⁴ to 3·10⁻³ against a median of 0.17, off by about 10⁻⁷, a few
    units in the last place of the terms, which the band's absolute floor,
    2⁻²⁰ in `f32`, does not cover. It is question 75's case in a forward
    pass, which D76 saw only in a gradient.
  - Every other entry is inside the band.
- **The course GPT at its constants** (`make c-conformance`, an
  `#[ignore]`d test, 26 s): bit-identical in the three types at `B = 1`,
  `T = 8`, on its synthesized parameters (zero logits) and with its zeros
  drawn. C2 there: 0, 18 and 63 of 736 outside, the native backend's. With
  the course's trained checkpoint (`out/course_probe_f32/final.psk.d`, 10.8
  million reals) at `B = 1`, `T = 256`, `--no-exact`: bit-identical in the
  three types, every one of the 23 552 logits non-zero. Its program builds
  and runs in about a second on one thread.
- **Mutations** (`presk-export-c/tests/host.rs`, 2.3 s). Three entries,
  with their zeros drawn: `LayerNorm{8}.apply` in `f32`, `mse_grad` in
  `f64`, `SelfAttention.apply` in `bf16`. Each is recorded, exported, built
  and run as a subprocess, and its outputs are compared with the native
  backend's. The arena is filled with NaNs first, so that a buffer no
  launch wrote is caught even where its value would be zero.
  - Unmutated, all three agree.
  - **Every live launch dropped** (54 programs) changes the outputs.
  - **Two live launches of one type and size writing each other's buffer**
    (52 pairs) changes the outputs. On the synthesized parameters two of
    them went unseen: `x·w` and `+ b` with `b` zero, where skipping the
    addition of a zero bias changes nothing. Hence the drawn zeros.
  - **A `bf16` input rounded once.** `x[0] = 1 + 2⁻⁸ + 2⁻³⁰` rounds to
    `f32` on a `bf16` tie, which ties to even give 1, where one rounding
    gives `1 + 2⁻⁷`. The reader patched to round once changes the output.
    No weight of the course checkpoint is such a case: it was trained in
    `f32`, so its values are `f32`s already.
- **The refusals** `E8002`–`E8008` and the option errors: each code and
  message asserted, nothing written.
- **The native backend unchanged.** On the eleven entries of
  `tests/plan/entries.txt`, in `f64`, `f32` and `bf16`, `main`'s release
  binary (`10bef70`) and this branch's print the same kernels (`presk dump
  --stage kernel`) and the same native outputs (`presk run --backend native
  --json --digits 20`), byte for byte: 66 comparisons. `make plan-check`
  passes, and `make ra4-dev-frozen` gives back
  `docs/ra4-dev-model-0.8.0.md` byte for byte.

**Left for later steps.** The structure: a C function per Presk function,
the fold as a loop, names from the source, release per call, and the dead
launches (step 4). `--plan` and OpenMP splits (step 5). `sample` and keys
(step 6). Integer weights (step 7). `make c-examples` (plan §5) is not
added: `make c-conformance` exports and builds every entry.

Alternatives considered:

- **Tags in the data.** Filling each buffer with its identity, and letting
  the no-run executor write tags into outputs. The host rounds and converts
  data (`Buf::of`, the `bf16` realization), and does scalar arithmetic on
  it, so tags would not survive. The mirror is exact by construction.
- **Syms passed at every call site in `prim`.** A shadow calling
  convention, the backend pushing each leaf's symbol before each recursive
  call. More hooks, scattered through the primitives. The mirror keeps the
  hooks to structure and launches, and checks itself against the launches.
- **Recording on the shape capture**, as planned (above).
- **Values with identity** (an `Arc` per buffer in `presk-rt`). It would
  change the runtime's values for every backend, for a recorder.
- **Scalars from the command line.** A second input channel that the native
  run does not have.
- **`params` outputs as `params` files.** They cannot be read without the
  program.

**On review (the user, 2026-09-26), three decisions, each as the agent
recommended.**

- **The dead launches stay in the C program**, in step 4 too.
  - C4 compares the program's time with the native backend's, so the two
    must do the same work.
  - Launch-for-launch identity lets a disagreement be traced launch by
    launch.
  - The waste is the native backend's as well. Its fix, dead-code
    elimination in PSK-Core after automatic differentiation, is recorded in
    `docs/kernels-roadmap.md` ("Work a gradient does and discards"), for a
    step of its own with a measurement. It has two parts: within a function
    it is local, and across calls a `.vjp` whose primal no caller reads needs
    a specialized copy. In `loss_grad`, 89 of the 125 dead launches are
    launched in forward functions and 36 in the `.vjp` functions and the
    fold's backward body.
- **Question 75 amended**, not a new question. Two facts join it.
  - **The examples pass the band partly on trivial outputs.** On the
    synthesized parameters, `MiniGPT.apply`'s logits are zero and `loss` is
    log V, so `examples_conform` passed there on zeros. Its own report
    shows `MiniGPT.apply`'s gap as exactly 0 in every type.
  - **Two conforming executors leave the band on different elements.** With
    the values the initializers set to zero drawn, `MiniGPT.apply` leaves
    the `f32` band on 4 of 1 024 logits on the kernel interpreter and on 5
    natively. The native backend alone crosses on two of them (logits 218
    and 503, at 1.51 and 1.32 times the band) and the interpreter alone on
    one (462, 1.57), since their `exp` differ in the last bits.
  - **What follows for the provisional answer.** Holding one backend to
    another's gap element by element works only between executors that
    compute the same bits, as C2 does after C1. It is a check of a copy,
    not a definition, and the first-order bound (question 39) is the
    remaining candidate for the definition.
  - **The conformance suite follows the provisional answer.** It gains
    `examples_with_their_zeros_drawn`, which asserts the band on every
    example in `f64` and on every one but `MiniGPT.apply` in `f32` and
    `bf16`. It reports `MiniGPT.apply`'s gaps there, 4 and 35 elements
    outside, at up to 3.7 times the band, and checks that no drawn run's
    outputs are all zero.
- **`tests/plan/mixed-realization.json` stays** beside the plans, where
  step 5's plans for the same entries will sit. CLAUDE.md's description of
  `tests/plan/` now names it.

## D80 — The host, structured: the recording's tree, a C function per distinct recording, folds as loops, results by destination, the arena at its peak (2026-09-26)

Plan 0.9.0 step 4 (§3.3). `presk export --target c` now writes the host as
C functions, one per distinct recording of a Presk function, each fold a
loop over its layers, buffers named after the source, and an arena
released per call and sized at the program's peak. Step 3's straight-line
host stays, as its oracle (`host::export_flat`, `--straight-line`). The
bits are the same: on every program of the test set, through both hosts,
and the native backend's.

**The structure recorded** (`presk-kernel/src/record.rs`). Step 3's
recording was flat: `Recording.ops` had no call structure, and calls, folds,
closures and applications existed only as hooks. The recording now keeps a
tree beside the list.

- **A call** (`Call`): the PSK-Core function, its dimensions at the call in
  the order it declares them (`Linear.apply` at `In = 16, Out = 64`), its
  parameters' source names, the arguments it received (a closure's captured
  values first), its result, its body as nodes, the caller's `let` that made
  it, and whether it was an application of a closure. The entry is
  `calls[0]`. An iteration of a fold is a call like any other.
- **A fold** (`Fold`): its `let`, its count, `index`, `reverse`, `scan`, its
  stacked values, extras and initial accumulator, its iterations in the
  order they ran with their indices, its value, and the operations of its
  scan's stack.
- **Nodes** (`Node::Op`, `Call`, `Fold`): every operation is in exactly one
  body or one fold's stack, and a depth-first walk gives the flat list
  back. The straight-line host reads the list, which did not change.
- **Names**: each buffer carries the number and source name of the `let`
  that made it (`q`, `h0`).
- **The hooks**: `enter` also passes the function's name, dimensions and
  parameter names (computed only when the recorder is on); a new
  `let_begin` gives the `let` a call or a fold is made under; `fold_begin`
  passes `reverse` and the count; `fused` passes the root `let`'s name. The
  parameter names come from `presk_core::Param`, which the recorder lacked
  (the prompt's fact), since `presk-export-c` depends on `presk-kernel`
  only.
- **The native backend unchanged.** On the eleven entries of
  `tests/plan/entries.txt` in `f64`, `f32` and `bf16`, `main`'s release
  binary (`8f1e3e0`) and this branch's print the same kernels (`presk dump
  --stage kernel`) and the same native outputs (`presk run --backend native
  --json --digits 20`), byte for byte: 66 comparisons. `make plan-check`
  passes, and `make ra4-dev-frozen` gives back
  `docs/ra4-dev-model-0.8.0.md` byte for byte.

**The canonical form** (`presk-export-c/src/structure.rs`, `Form`). Module
functions are shared across instances (`Linear.apply{In Out}` is one
PSK-Core function, called at D→D in attention and D→4D in the MLP), and a
call's recording depends on its dimensions, its arguments' types and the
constants it reads. So each call is put in a canonical form, built bottom
up, and calls whose forms are equal share one C function.

- **Parameters by position**: one per leaf of the arguments. A tensor is a
  pointer, of its type; a host scalar computed at run time is a `double`;
  the index of a fold's iteration, in the body of a fold that takes it, is
  `int64_t i`; a constant (`eps`, a `nat`, an `f64` literal) and a key are
  not passed. A parameter the body never reads is left out of the C
  signature.
- **Buffers by first definition**, the function's own and those it
  allocates for its callees' results; **outputs** in the order the result
  reads them.
- **Kernels by their C text** (the `k<n>` of `kernels.c`, deduplicated by
  text as in step 3), **callees by their own forms**, sites and names from
  the source, which are the same at every call of one function.
- **Constants are folded into the operations that read them.** The recorder
  resolves every constant where it is used (a scalar made a buffer, `f64`
  arithmetic, a slice at a static index), so a constant nothing reads does
  not split a function, and two calls that read different constants record
  different operations and become two functions. This is the prompt's
  "static scalars equal across calls can be constants of the C function; if
  they differ, the recordings differ", without having to compare the
  arguments themselves.
- **Names**: `presk_<Function>`; with its dimensions when the function has
  forms at several (`presk_Linear_apply_In16_Out64`); numbered when it has
  several forms at the same dimensions, the one called most keeping the
  plain name, and its comment naming the others. On the test set: `mse_grad`
  10 functions, `MiniGPT.apply` 8, `loss_grad` 21, the course GPT 7.

**Results by destination.** A function's result is written into buffers
its caller allocates and passes (`out`, `out_0_w`, …): the caller allocates
before the call, the callee writes, and a callee's temporaries are released
when it returns (`presk_mark` at entry, `presk_release` at return). The
entry's outputs are allocated before its own mark, so that they outlive it.

- **Why not `presk_release_keep`.** It moves one block, and a result is a
  tree of leaves: some are views of the callee's arguments, outside its
  frame; some are views at an offset in a buffer of its frame. Moving the
  leaves would copy every result at every return. Writing it where the
  caller wants it copies nothing, and is how llama2.c's functions take
  their output buffer.
- **A leaf that is a view of an argument is not an output**: the caller has
  it already, and reads it where it is. **A buffer an output is a view of**
  is passed whole.
- **A scalar computed at run time** is returned through a `double *`.
- **A fold's value returned as it is** (the entry or a function returning
  the fold) is left in the loop's alternating buffers, and copied into the
  caller's buffer after the loop (`memcpy`): the one copy the structured
  host adds, and exact. No entry of the test set does it; a program of
  `tests/export_c/fold_value.psk` does, as the entry and through a call.

**Folds as loops**, for the three kinds the test set has: the forward stack
(MiniGPT, L = 2; the course GPT, L = 6), automatic differentiation's scan
(`scan: true`), and the backward fold (`index: true, reverse: true`).

- **Runs.** A fold's iterations are cut into runs whose forms are equal up
  to offsets that are affine in the iteration's index: the law is fitted on
  the run's first two iterations and checked on the others. A run of two or
  more is a loop, `for (int64_t l = …)` (`int64_t`, not the plan's
  `size_t`, for `-Wsign-compare`, D77), calling one function; a run of one
  is a call. Reverse folds run `l` from `L − 1` down to 0.
- **Slices** of stacked parameters are the loop's own arithmetic, `w + l *
  chunk`, as in llama2.c. The course GPT's `GPT.apply` at its constants is a
  loop over its six blocks calling `presk_Block_apply(arena,
  theta_blocks_attn_wq_w + l * 147456, …, s_0, mask, s_0_buf[l % 2])`.
- **The index.** The backward body writes the layer's gradient into the
  stacked cotangent with `set_slice` at `i`, which copies the whole
  cotangent every iteration (O(L²) bytes). The native backend does the same,
  and so does the C: its body takes `int64_t i`, and the slice written back
  is at `out_1_attn_wq_w + i * 256`. Only a fold that takes its index may
  have offsets that move with it.
- **The accumulator** is a variable per leaf, named after the body's
  accumulator parameter (`x` in MiniGPT, `s_0` in the course GPT, whose
  accumulator is `(h, key)`: the key is dropped and the loop carries `h`).
  Each output of the body gets two buffers, alternating (`buf[l % 2]`), so
  that an iteration never writes the buffer it reads. A run cannot loop, and
  its iterations become calls, when a leaf changes type from one iteration
  to the next, or is set from another leaf, which the alternation could
  overwrite before it is read.
- **The scan's stack** is allocated before the loop, and each iteration
  writes its slot at its start: the accumulator before the iteration, as
  the recording's stack copies it once the last iteration has returned. The
  copy reads what no later operation of the loop changes, so moving it into
  the loop keeps the bits, and the accumulators need not outlive their
  iteration.
- **Iterations that differ: peeled, not unrolled** (the prompt's
  recommendation, the plan said "unrolled"). Under
  `tests/plan/mixed-realization.json` MiniGPT's first block reads `h0` from
  an `f64` site, so it records four more conversions at kernels' edges (34
  launches and 33 copies against 34 and 29). At L = 4 iteration 0 is a call
  of its own function, `presk_Block_apply_2`, and iterations 1 to 3 a loop
  over `presk_Block_apply`; at L = 2 both iterations are calls. The rule is
  the runs': the plan's unrolling is what happens when every iteration
  differs from the next. The README and a comment above each fold say how
  it was emitted and why.
- **A fold of one iteration is a call**, not a loop of one: the test set's
  `GPT.apply` at L = 1.

**Scalars.** Host scalars computed at run time exist only in `adamw.step`
(the gradient norm's sum of squares, read back from kernels), where they
become a one-element tensor in the same function. No source program makes
one cross a call: the type checker allows no unary function of an `f64`,
and that is the only other way the host reads a scalar back. The C
functions take them anyway, as `double` parameters, and return them through
`double *`; the choice costs little, since a scalar is canonicalized like a
buffer, and it is tested on a recording written by hand in which one goes
into a call and comes back out of it. The alternative was to refuse, a
limit no program would meet today.

**Names from the source.** Parameters are the Presk parameter's name and the
leaf's path (`theta_attn_wq_w`, `x`, `mask`); outputs `out`, or `out_<path>`
when there are several; a buffer is its `let`'s source name (`q`, `h0`,
`y_hat`) or `t<let>`; a conversion at a kernel's edge `<what it converts>_
<type>` (`t9_f64`); a callee's outputs, in the caller, the caller's `let`
(`t8_0_w`); the accumulator as above, its stack `<acc>_stack`, its buffers
`<acc>_buf` (a loop) or `<acc>_at<i>` (an iteration apart). No local shadows
a kernel (`k<n>`), a variable (`s<n>`), the runtime (`presk_…`) or the
function's own words (`arena`, `mark`, `l`, `i`), nor, in the entry, its
structs `in` and `out`.

**The arena at its peak.** Each function's frame is what it allocates, and
at each call its callee's frame on top; a fold's alternating buffers are
allocated before its loop. The export computes the peak by walking the
structure (`Structured::arena`): the inputs, the entry's outputs, and the
entry's frame. The tests read the arena's peak back from the running
program (`arena.peak`) and find it equal to the size the export computed,
on seven programs. Against step 3's sum:

| entry | released per call | released nothing (step 3) |
|---|---|---|
| `mse_grad` (`f64`) | 9 600 B | 14 848 B |
| `MiniGPT.apply` (`f64`, L = 2) | 113 920 B | 238 208 B |
| `loss_grad` (`f64`) | 744 896 B | 2 606 656 B |
| course GPT, `T = 256`, `f64` / `f32` / `bf16` | 117.1 / 58.6 / 29.3 MB | 352.4 / 176.2 / 88.1 MB |

Of the course GPT's arenas, 86.5, 43.3 and 21.6 MB are the weights, so the
host's own part falls from 265.8 to 30.6 MB in `f64`, 8.7 times. The release
is per call, not per buffer: a function's buffers live until it returns,
and releasing each at its last use, which would lower a long function's
peak further, is not done.

**The straight-line host stays** as the oracle: `host::export_flat` in the
library, and `presk export … --straight-line` in the CLI, exposed because
the tests compare the two hosts through the binary, and because it is where
to go when a structured program is suspected. The test set's check exports
every program both ways.

**The checks.**

- **C1 through both hosts.** The 68 programs of `make test`
  (`presk-cli/tests/export_c.rs`) and the 9 of `make c-conformance` (the
  course GPT, the trained checkpoint at `T = 256` included) are each
  exported twice, structured and `--straight-line`: every program is
  bit-identical to the native backend, and the two hosts' output files are
  the same bytes.
- **Step 3's mutations stay**, on the straight-line host: each of 57 live
  launches dropped, 52 swaps, the `bf16` input rounded once.
- **Structural mutations**, on MiniGPT's forward pass at L = 3 and on
  `loss_grad` (its forward fold, scan and backward fold), the arena filled
  with NaNs, each run as a program:
  - **a slice off by a layer**: each stacked leaf a loop slices, `w + (l +
    1) · chunk` (the last iteration reads past the tensor): 64 programs;
  - **a loop bound off by one**, each loop one iteration less and one more:
    8 programs;
  - **a release before a buffer's last use**: in each function whose first
    buffer is read after a later buffer is written, its frame released
    right after that first buffer is written, so the later buffer takes its
    place: 10 programs.

  Every one changes the outputs. Two exclusions, both of reads that cannot
  change anything: the backward body's slice of `mlp.l2.b` is read by no
  live operation (only by the primal a `vjp` recomputes and discards), and
  an operand of `sum_to` is read for its shape, never loaded, so a buffer
  is counted as read only where its kernel loads it.
- **`make test`'s added time**, warm, `cargo test --workspace` on `main`
  and on this branch in turn on an otherwise idle falicon: 125.6 s against
  133.1 s, 7.5 s more. `presk-cli/tests/export_c.rs` goes from 5.0 to
  7.7 s (every program exported and built twice) and
  `presk-export-c/tests/host.rs` from 2.3 to 7.3 s (the structural
  mutations, the arena's peaks and the programs of this step).

**Left for later.** Release at a buffer's last use. The dead launches stay,
as decided on step 3's review; their removal is in
`docs/kernels-roadmap.md`. `--plan` and OpenMP (step 5).

## D81 — Plans, threads and portability: the plan exported, a divided launch an OpenMP region, its team capped by `OMP_NUM_THREADS`, C3 through zig, C4 measured (2026-09-26)

Plan 0.9.0 step 5 (§3.6, §3.8 C3 and C4). `presk export --target c` takes a
realization plan and the native executor's worker count, and a launch the
native backend divides among threads is an OpenMP region of the exported
host. The bits do not move: C1 holds under the plans `presk plan` gives, on
one thread, on every thread and without OpenMP, and the program compiles
without a warning on GCC with and without OpenMP, on Clang and for x86-64.

**The plan.** `--plan @P.json` is read as `presk run --plan` reads it
(`Plan::parse`, exclusive with `--realize`), checked against the program
(`check_plan`, `E7002`), and the native run the export records is the
plan's: the program's kernels are the plan's tiles, orders, lanes and packs,
and its splits are the program's threads. The README names the plan.
`E8002`, which refused the target in steps 0–2 and then `--plan` and
`--threads`, is no longer emitted; the code is not reused.

**`--threads N`** is the native executor's worker count, as for `presk run`
(it sets `PRESK_THREADS`). It is what condition 5 checks a plan's split
against (`N` or `N/2` workers), and what the executor's default divides a
kernel among. The prompt's command for C4's one-thread side, `presk bench
--threads 1 --plan @P`, is therefore refused on a plan that splits among
ten (`E7002`, condition 5): the native backend cannot run that plan on one
worker. C4 runs the same plan with every split stated as none, which lowers
the same kernels.

**The split belongs to the launch** (the prompt's fact). A kernel's C text
records the depth its range divides but not its workers: a kernel under the
executor's default (`Threads::Default`) and the same kernel stated on one
worker (`Threads::One`) print alike, and kernels are deduplicated by text.
So the kernel table gives each backend kernel its workers beside its C
function and its range, resolved by the native backend's rule copied as
`presk_export_c::workers` beside `ranged` (a plan's split among its
workers, at most one per iteration; the executor's default over its
outermost parallel loop among every worker when there are two or more, the
loop has two iterations and the kernel's work reaches `PARALLEL_MIN_WORK`;
one worker otherwise). The structured host's launch statement carries the
workers, so they are part of the canonical form: two calls whose launches
differ only in how they are divided are two functions. No program of the
test set has two such calls (the plan keys a layout by nest name, and a
function at one set of dimensions launches the same backend kernels), so a
test makes one: MiniGPT's three blocks, one of the block's kernels copied
and launched by blocks 1 and 2 divided among three workers where block 0
runs it on one. Both print as one `k<n>`; the first block becomes a
function of its own and the other two a loop, and the program keeps the
native bits.

**A divided launch is one OpenMP region** (§3.6):

```c
#ifdef _OPENMP
#pragma omp parallel num_threads(presk_team(10))
    {
        int64_t t = omp_get_thread_num(), w = omp_get_num_threads();
        k7(x, mu, t4, 256 * t / w, 256 * (t + 1) / w);
    }
#else
    k7(x, mu, t4, 0, 256);
#endif
```

Each thread calls the kernel on its piece of the divided loop, the pieces
the native backend computes (`n·t/w` to `n·(t+1)/w`), from the team it
actually gets, so a smaller team still covers the loop; the loops above
run in full. `t` and `w` are reserved names of the structured host (no
buffer takes them). The `Makefile` has `OPENMP = -fopenmp` when a launch is
divided, empty otherwise, and `make OPENMP=` builds without OpenMP.

**`OMP_NUM_THREADS` and `num_threads(p)`** (the prompt's fact: with
`num_threads(10)`, `OMP_NUM_THREADS=3` gives a team of ten). Plan §3.6 said
"the plan's p is the default `num_threads`, and `OMP_NUM_THREADS` overrides
it". Two ways: the program reads the variable itself, or the rule changes.
The rule changed. `presk_team(p)` in `presk_rt.h` is `min(p,
omp_get_max_threads())`: `omp_get_max_threads()` is the count OpenMP would
use, set by `OMP_NUM_THREADS` (its first element when it is a list, which a
program parsing the variable would have to handle) or by
`omp_set_num_threads`, and the cores by default. So the variable lowers a
team and never raises it above the export's workers, `OMP_NUM_THREADS=1`
runs the program on one thread, and on a machine with fewer cores than the
export's `--threads` the team is the cores, never more. The native backend
cannot raise a plan's workers either. Checked: `OMP_NUM_THREADS=3` on the
course GPT's plan (splits among ten) gives the native bits.

**Timing.** `main.c` takes `-r R`: the entry called `R` times, the arena
released to its mark between calls (the entry allocates its outputs), each
call timed with C11's `timespec_get`, the median (`sorted[R/2]`, as `presk
bench` takes it) and the best printed on standard error, the last call's
outputs written. `presk bench` gains `--plan-only` (with `--plan`: the plan's
row alone, since at the course GPT's `B = 16` on one thread the six uniform
rows take most of half an hour) and `--weights` (the entry's first argument
from a checkpoint, as `presk run --weights`), so that both sides run the
same plan on the same weights. `bert.sh` reads `--native-only`'s rows,
which do not change.

**C1 under plans and on any team.** Each program below is exported
structured and straight-line (the two output files the same bytes), built
by its own `Makefile`, run by its own `conformance/check.sh`, then run the
other ways: a program that divides a launch on one thread
(`OMP_NUM_THREADS=1`) and rebuilt without OpenMP, one that divides none
rebuilt with it; every way, the native bits.

- **`make test`** (`presk-cli/tests/export_c.rs`): the 68 programs of step
  3's matrix, which divide nothing at the test set's sizes; MiniGPT's
  forward pass under `tests/export_c/minigpt-splits.json`, a plan stating
  splits at depths 0, 1 and 2 among two and four workers with `--threads 4`
  (25 of 79 launches divided), and at `D = 64`, `B = 4`, `T = 32`, where
  the native backend's own rule divides the 4 launches whose work reaches
  its threshold among `--threads 4`; each on the synthesized parameters and
  with the zeros drawn.
- **`make test`** (`presk-export-c/tests/host.rs`): every legal depth. Five
  recordings (LayerNorm `f32`, `mse_grad` `f64`, the attention `bf16`,
  MiniGPT's forward pass at `L = 3` `f32` and `loss_grad` `f64`), each
  kernel divided at its first, second, third and fourth legal depth in
  turn (cyclically, among 3, 4, 2 and 5 workers, so pieces are uneven and
  sometimes fewer than the workers): 20 variants, both hosts, 40 programs
  and 4 212 OpenMP regions, each run on one thread, three, every thread and
  without OpenMP. The recording's native outputs stay the program's, since
  the bits do not depend on the split (§3.5, rule 4, checked kernel by
  kernel in step 1).
- **`make c-conformance`**: every entry of the test set under the plan
  `presk plan` gives with the frozen profile, in `f64`, `f32` and `bf16` and
  MiniGPT's under the mixed realization, on the synthesized parameters and
  with the zeros drawn: 68 programs. The course GPT at `B = 16`, `T = 256`
  with the trained checkpoint, under its plan in the three types (201 of 228
  launches divided among ten threads) and under the default layout in `f32`
  (105 of 222 launches divided by the native backend's rule among up to
  sixteen, the batch), beside step 4's nine.

Two facts about `presk plan` the plan family met, both of `presk plan`
(0.7.0), not of the export, and both left for later:

- **A plan `presk plan` writes may be refused by `presk run`.** It states
  one layout per nest name, chosen at the nest's first shape (`ra4.rs`: "the
  first shape a nest launches at decides it"). The MLP's `Linear.apply/%3`
  runs at D→4D, then at 4D→D: at `D = 4`, `B = 2`, the tile `[2, 16]` chosen
  for the first does not divide the second (condition 3), and `presk run
  --backend native --plan` refuses the plan in `f64` and `f32` (and in
  `bf16`, whose plan is `f32`'s, below), `main`'s binary as this branch's. The export refuses it too, with the same
  diagnostic and nothing written, which the test asserts; it then checks C1
  with that nest left to its default layout (`mse`, `mse_grad`). A fix
  would choose, for a nest launched at several shapes, a layout legal at
  every one.
- **In `bf16` `presk plan` states no layout, silently.** The frozen model is
  calibrated in `f32` and `f64` alone (its groups), a nest it cannot price
  gets no layout and no line, and the plan written is empty. So C1 "under a
  plan from `presk plan`" would be the default layout in `bf16`. The test
  takes the `f32` plan instead, each nest's lanes one and its softmax passes
  the default, which `bf16` refuses (conditions 4 and 7): the same tiles,
  orders, k-blocks, packs and splits, in `bf16`.

**The split checked by mutation** (`host.rs`). MiniGPT's forward pass and
the attention in `bf16`, every kernel divided at its first legal depth
among four workers, run on three threads, the arena filled with NaNs:

| mutation | the outputs | the check of the pieces |
|---|---|---|
| each piece one element short (a gap) | differ | fails |
| the last thread's piece dropped (a gap) | differ | fails |
| each piece one element into the one before (an overlap) | the native bits | fails |
| every thread the whole range (an overlap) | the native bits | fails |

A gap leaves NaNs, which reach the outputs, as the prompt asked. An overlap
cannot: every element it computes twice is computed from the same inputs by
the same loops, so both writes are the same bits (rule 4), and only a race
on an output a kernel accumulates in could show, which is not certain. So
the tests add a check of the pieces to `model.c`: each thread records the
range it calls its kernel on, and after the region the pieces must cover
the loop exactly once, or the program stops. Unmutated, the check passes on
one thread, three and every one.

**C3.** GCC 13.3, `-std=c11 -O2 -ffp-contract=off -Wall -Wextra`: every
program of `make test` and of `make c-conformance`, with `-fopenmp` and
without, no warning. Clang and x86-64 (the prompt's decision, taken here):
**zig 0.16.0 from PyPI**, its `zig cc` (Clang 21.1.0), in `.venv-zig` (`make
zig`; `make c-conformance` installs it), rather than asking for the
distribution's `clang` and `gcc-x86-64-linux-gnu`, which need root on each
machine and follow the distribution's versions; zig gives both from one
pinned version without root. With `PRESK_C3=1` (`make c-conformance`) every
program the tests export (the 68, the 68 under plans, the 4 divided, the
course GPT's 13, each both hosts) is also built by `zig cc` for aarch64,
linked and run, its outputs the native backend's bits, and for
`x86_64-linux-musl`, linked (static) and not run: no x86-64 machine or
emulator is at hand. zig ships no OpenMP runtime, so with `-fopenmp` the
two files that print OpenMP regions (`main.c`, `model.c`) are compiled for
both targets and not linked, against GCC's `omp.h` read as a system header
(a directory holding it alone, since GCC's other headers are not Clang's).
None warns. Two facts of the run: `presk_rt.h`'s OpenMP part and the region
compile clean on Clang 21; and many `zig cc` at once, each compiling
several files, deadlock on zig's cache (each process holding one object's
lock and waiting for another's; the identical `presk_rt.c` and
`presk_libm.c` of every program share entries): the tests run one file per
`zig cc`, link the objects apart, and make the first program's runs alone,
which fills the cache.

**C4** (reported, decides nothing; `make c-speed`, `experiments/c_speed.sh`,
on an otherwise idle falicon). The course GPT's forward pass at `B = 16`,
`T = 256`, `f32`, the trained checkpoint, under the plan the frozen profile
gives; seven calls a side, wall time in ms, the C program built by its
`Makefile` (GCC 13.3, `-O2`, OpenMP) and timed by its `-r` loop, the native
backend by `presk bench --plan-only --weights`:

| threads | native median (best) | C median (best) | native / C |
|---|---|---|---|
| every thread (the plan's ten workers, unpinned) | 1 094 (1 086) | 290 (275) | 3.8 |
| one thread (core 5, an X925) | 3 132 (3 129) | 1 700 (1 700) | 1.8 |

The threads buy the native backend 2.9 and the C program 5.9. On one
thread the two run the same kernels, so the 1.8 is code generation (GCC's
`-O2` against Cranelift's) and the host (the native host copies each
stacked layer's slice and each view; the C program slices by pointer). On
ten, the native backend spawns a thread per piece at each of the 201
divided launches, where the C program's team is libgomp's pool. Neither is
measured apart here. The native side's one-thread row runs the plan with
its splits stated as none (condition 5, above).

**The native backend unchanged.** On the eleven entries of
`tests/plan/entries.txt` in `f64`, `f32` and `bf16`, `main`'s release binary
(`b137957`) and this branch's print the same kernels (`presk dump --stage
kernel`) and the same native outputs (`presk run --backend native --json
--digits 20`): 66 comparisons. `make plan-check` passes, and `make
ra4-dev-frozen` gives back `docs/ra4-dev-model-0.8.0.md` byte for byte.
Nothing in `presk-kernel` or `presk-cranelift` changed.

**`make test`'s added time**, warm, `cargo test --workspace` on `main` and
on this branch in turn on an otherwise idle falicon: 133.4 s against
144.6 s, 11.2 s more. `presk-cli/tests/export_c.rs` goes from 7.8 to 15.0 s
(every program built twice, and the divided programs) and
`presk-export-c/tests/host.rs` from 7.3 to 13.5 s (every legal depth, the
mutations of the split).

**Left for later.** `presk plan`'s two facts above (fixed after the review,
D82). Timing the native
backend's host apart from its kernels, which C4 would need to say where
its 1.8 and its 3.8 come from.

## D82 — `presk plan`: one layout every shape of a nest takes, and a type the profile cannot price refused (2026-09-26)

Found in 0.9.0 step 5 (D81, "Two facts about `presk plan`") and fixed on the
user's decision after its review, before step 6. `presk plan` (0.7.0 step 5)
had two defects, neither of the C export.

**One layout per nest, chosen at its first shape.** A plan states one layout
per nest name, and `presk plan` took the best prediction at the first shape
the nest was launched at ("the first shape a nest launches at decides it").
A module function launched at two sizes shares its nests: the MLP's
`Linear.apply/%3` runs at D→4D, then at 4D→D. At `D = 4`, `B = 2` the tile
`[2, 16]` chosen for the first does not divide the second (condition 3), and
`presk run --plan` refused the plan `presk plan` had written, in `f64` and
`f32`.

- **The rule now.** A nest launched at one shape takes that shape's best,
  as before. A nest launched at several takes the layout the model prices
  at every one of them whose predicted time, each shape's weighed by the
  times the entry launches it, is least; a tie goes to the order of the
  first shape's predictions. The objective is the entry's predicted time
  for that nest, which is what the choice is for.
- **The counts.** The capture recorded each nest's first launch per shape;
  it now also counts the others (`Launch::count`, `presk-kernel`). Nothing
  else reads it. The MLP's gradient launches each shape of `Linear.apply/%3`
  three times (the forward pass and the one the gradient recomputes); the
  course GPT at `L = 6` launches its D→D shape 24 times and the two others
  6 each.
- **The table** keeps its line per nest and shape. A nest whose shapes do
  not all choose the layout it takes gets one more line, `= NEST at N
  shapes, counts: layout; time over them, time at each shape's own best`,
  the price of one layout per name. The last line counts nests and, when
  they differ, shapes: `-- 23 nests at 37 shapes`. An entry whose nests
  each run at one shape prints what it printed (the tutorial's attention:
  the same table and the same plan, byte for byte but the times).
- **What changes.** On the test set, the MLP's plans in `f64` and `f32` (the
  tile `[2, 4]`, which both shapes take); every other plan is the one 0.8.0
  wrote. MiniGPT's and the course GPT's `Linear.apply/%3` keep the first
  shape's layout, which the weighing also picks (at the course GPT's `B =
  16`: 227.8 ms over its 36 launches, against 216.4 ms at each shape's own
  best). So the plan C4 was measured under (D81) is unchanged, and so are
  its times.

**A type the profile cannot price.** The layout model has a group per type,
lanes and workers, and the frozen profile's are in `f32` and `f64`. A nest
none of whose points a group prices got no line and no layout, silently, so
`presk plan --realize bf16` wrote an empty plan, which runs the default
layout.

- **No nest priced**: `presk plan` refuses, exit code 2, and writes nothing:
  "the profile's layout model prices no layout of `E` in bf16: P was
  calibrated in f32 and f64 only, and a plan stating no layout would be the
  default one".
- **Some nests unpriced** (a mixed realization with `bf16` or `f64` sites
  under a profile of `f32` groups): they are left to the default layout, as
  before, and a last line names them. A nest that the model cannot price at
  one of its shapes gets no layout at any, since the default suits them
  all. MiniGPT's plan under `tests/plan/mixed-realization.json` is the one
  0.8.0 wrote.
- **Calibrating `bf16`** is not done: it is a timed calibration on a quiet
  machine, and nothing in 0.9.0 needs it. The C export's tests keep
  checking `bf16` under the `f32` plan made one (D81).

**Checks.** Two tests in `presk-cli` (`ra4.rs`), on a synthetic profile of
`f32` groups: the MLP's gradient, where the first shape's own best does not
lower at the second (so the test tests the rule), gets a layout every nest
lowers under at every shape, and each single-shape nest keeps its best; `bf16`
is refused with nothing written, and under the mixed realization the sites
of other types have no layout. In `make c-conformance` the test set's plans
now run as written (the fallback of D81 that left the refused nest to its
default is gone): 68 programs, the MLP's under their new plans, all
bit-identical to the native backend. The native backend unchanged: `main`'s
binary and this branch's, 66 comparisons byte for byte; `make plan-check`,
`make ra4-dev-frozen` pass. `presk rank`, `reprice` and `autotune` do not
use the plan's rule, so no RA4 report moves.

The reference's "Status in 0.8.0" of PSK-Realize says the first shape
decides, which was 0.8.0's behaviour; its 0.9.0 status (plan 0.9.0 step 9)
says otherwise.

## D83 — Randomness and generation: `sample` on the program's stream, its `exp` the `libm` crate's, `f64` arguments at run time, `--serve` and `generate` (2026-09-26)

Plan 0.9.0 step 6. The plan's step read "ChaCha20 and presk's key functions
(`split_key`, `sample`) in the runtime, with the interpreter's definitions",
and a driver over a sliding window. Facts checked before the step, and what
was decided on them.

**`sample` takes no key.** `serve_minigpt.psk`'s `next_token(θ, ctx,
temperature : f64 = 0.8)` calls `sample(last / real(temperature))`, and the
source says the compiler draws from a stream seeded by the run's seed. So
the program needs no `split_key` and no key in the runtime: keys stay
dropped (§3.3), an initializer stays refused (`E8004`), and `E8006` is
reworded without "before step 6". The course GPT has no generation
function: C5 is on `serve_minigpt.psk` alone.

**The stream.** The native `sample` (the kernel backend's host, not a
kernel) draws from `ChaCha20Rng::seed_from_u64(seed ^ 0x5a5a_5a5a)`, one per
backend, created with it; `presk serve` uses one backend for a generation,
so step `k`'s draw continues step `k − 1`'s stream. The runtime gains
`presk_rng` and its functions (`presk_rt.c`), written from the algorithm,
not from the crates' code:

- Bernstein's ChaCha20, twenty rounds, a 64-bit block counter in words
  12–13 and a 64-bit stream number (0) in words 14–15, not RFC 7539's 32-bit
  counter and 96-bit nonce;
- read as `rand_core`'s `BlockRng` reads `rand_chacha`'s core: four blocks
  (64 words) at a time, `next_u64` as words `i` and `i + 1`, the low one
  first, and the case of a `u64` straddling two refills, which never happens
  from a fresh generator read by `next_u64` alone but does for `next_u32`
  mixed in;
- `seed_from_u64`, `rand_core`'s default: eight outputs of PCG32 (multiplier
  6364136223846793005, increment 11634580027462260723, XSH RR), each
  little-endian;
- `presk_uniform01`, the top 53 bits over 2⁵³, and `presk_rng_init(seed)`,
  the backends' salt.

Checked against the crates in `make test` (`presk-export-c/tests/rng.rs`,
0.2 s): 10⁶ `next_u64` draws over 10³ seeds (special values, the backends'
seeds, drawn ones; 32 000 refills), `presk_uniform01` and `presk_rng_init`
on the same seeds; 10³ drawn 32-byte seeds read by a drawn mix of words and
pairs, over a hundred pairs straddling two refills.

**The stream is the program's state.** A C function that draws takes
`presk_rng *rng` after the arena, and so does one that calls it (the
structured host's analysis, as for the arena); the entry too,
`presk_next_token(arena, rng, in, out)`. The caller carries it from call to
call, as `presk serve` carries the backend's. `main` seeds it with `-s
SEED`, by default the seed the program was exported with, the one
`conformance/native.psk.d` was computed with; under `-r R` it seeds it again
before each call, so that every timed call computes the same outputs.

**The draw, and its `exp`.** `presk_sample` is the native arithmetic: per row
of the last axis, the logits as `f64` (a `bf16` or `f32` widened), `m` their
largest from `−∞` (a NaN passed over, as `f64::max` passes over it; which
zero is kept does not matter, `exp(−0) = exp(+0)`), `e_i = exp(x_i − m)`,
`z` their sum in order, `u` a draw, and the first `i` with `u < Σ_{j≤i}
e_j / z`, the last otherwise. The native `exp` there was Rust's `f64::exp`,
the platform's C library (glibc 2.39 on falicon), not the `libm` crate the
kernels call (D78). Two ways to bit identity: the C calls the system's
`exp`, or the native `sample` calls `libm::exp` and the C `presk_exp`.

- **Decided: `libm::exp`.** The exported program then depends on no C
  library, as its kernels do not (D78); C1 holds on any machine, the x86-64
  musl build of C3 included, where the system's `exp` would be musl's
  current one against glibc's natively; and the native backend's draws no
  longer depend on the platform either. `presk-kernel` already depended on
  the crate (for `erf`). The kernel interpreter shares the host, and moves
  with it; the reference interpreter samples from its exact softmax and is
  unchanged.
- **The cost**: a token moves where glibc's `exp` and musl's differ in the
  last bit (1.2–9.8 % of inputs, D78) and the draw falls between the two
  cumulative sums, about 2⁻⁵³ per token. Measured: `make serve-examples`
  gives the same tokens (greedy `33 33 33 33 0 0 0 0`, sampled `19 55 1 63
  57 1 54 29`); 540 generations of 32 sampled tokens (three types, three
  temperatures, sixty seeds), `main`'s binary against this branch's, all
  identical; and `next_token`'s native outputs among the 66 comparisons.
- **Checked**: `presk_sample` against the native backend's `sample`
  (`KernelBackend::prim`), row by row from the same stream, on 12 028 rows
  in `f64`, `f32` and `bf16`: logits drawn at scales from 10⁻³ to 10⁴,
  ties, `±∞`, NaNs, rows of `−∞`, vocabularies of 1 to 1 000, four seeds.

**`f64` arguments at run time.** D79 fixed every scalar argument at export,
so a sampled program would have had one temperature. An `f64` argument of
the entry is now a host scalar computed at run time in the recording
(`Scalar::Var`, the value recorded kept in `Recording::scalar_args`) and a
`double` of the program's `_in` struct, which `main` sets to the value
`presk run` synthesizes (so `check.sh` still compares with the native
backend) and a caller sets as it likes. Naturals, integers and booleans stay
fixed: they may shape the program. On the test set this moves `eps` of
`LayerNorm{8}.apply` and `RMSNorm{8}.apply` and the temperature; an `eps`
a callee receives as a literal (MiniGPT's norms) stays folded.

**The greedy copy and `--serve`.** At temperature 0 `presk serve` calls a
copy of the function with every `sample` an `argmax` (a kernel), with
temperature 1 (`fns.greedy`, which `prepare_serve` appends to the Core
program); no `--entry` names it. Decided: `presk export --target c --serve`,
the PyTorch export's option for its generation loop (`serve.py`), now both
targets'. It resolves the generation function as `presk serve` does (the
file's `inference serve`, or `--entry`), at `B = 1` (another `--dim B` is
refused, exit code 2) and the window `--dim T=N`; records the function as
written, whose one call `conformance/` checks, then its greedy copy on the
same backend, so that they share kernels. The program holds both entries on
the same `_in` and `_out` structs, `presk_next_token` and
`presk_next_token_greedy`; the structured host builds both from one set of
functions (`presk_MiniGPT_apply` is one), each its own root
(`Structured::others`), the arena the larger of the two peaks. `generate.c`
is `presk serve`'s loop: the last `T` tokens in, a token out, appended;
greedy at temperature 0 (the copy, at temperature 1), else the entry, every
draw from one stream seeded once; options named as `presk serve`'s
(`--weights`, `--prompt`, `--tokens` 16, `--temperature` the function's
default, `--seed` the export's, `--json`); its output `presk serve`'s, the
`--json` document byte for byte (`serde_json`'s pretty layout, its keys
sorted); `presk serve`'s refusals (a negative temperature, an empty prompt,
one shorter than the window, a token outside the vocabulary).

- **A model whose parameters do not fix the context's length** is served
  over the whole context, one token longer at every step, which a C program
  of one shape cannot run: `E8009`, tested on `tests/export_c/serve_untied.psk`.
  The rule is `presk serve`'s own (the initializer's dimensions include the
  context's `seq` dimension), so `generate` slides a window exactly where
  `presk serve` does. Found: with `dim T` declared at file level, a model
  none of whose parameters depends on `T` counts as tied, since its
  initializer takes the file's dimensions, and `presk serve` slides a
  window over it; left as `presk serve` has it.

**`E8005` retired**, as `E8002` was: not emitted, the code not reused.

**Checks.**

- **C1 on `next_token`**: in `make test`'s matrix since it samples no
  longer refused, in `f64`, `f32`, `bf16` and the mixed realization, on
  synthesized parameters and with their zeros drawn (76 programs, 8 of them
  `next_token`'s); under `presk plan`'s plans and with the trained checkpoint
  (`out/ckpt/final.psk.d`) at `B = 2`, `T = 8` in `make c-conformance`.
- **C5 in `make test`**: `serve_minigpt.psk` at the tiny constants, its zeros
  drawn (the example's head is `init = zeros`, so its logits are otherwise
  zero and every token a uniform draw), a prompt of twelve tokens, sixteen
  generated, greedy and sampled at temperatures 0.3, 0.8 and 1.5 and seeds
  0, 1, 2 and 7, in the three types, through both hosts: 6 programs, 30
  documents, each `presk serve --backend native`'s byte for byte; the
  settings generate different tokens.
- **C5 in `make c-conformance`**: the same with the trained checkpoint of
  `make train-examples`, in the three types through both hosts. Its logits
  are about 10⁻⁴, nearly flat, so its sampled tokens hardly depend on the
  temperature (at 0.01 one token of sixteen differs from 0.8), but they are
  the stream's. `generate.c` goes through C3 (Clang and x86-64 through
  zig), and Clang's `generate` gives `presk serve`'s documents.
- **Mutations of `generate.c`**, each changing the tokens: a draw skipped at
  the second step (the sampled tokens move, the greedy ones stay), the
  stream seeded again at each step (the sampled tokens collapse to one or
  two), and the window one token behind (every setting moves).
- **The native backend unchanged, but for `sample`'s `exp`**: on the eleven
  entries in the three types, `main`'s release binary (`20c1c13`) and this
  branch's print the same kernels and the same native outputs, `next_token`'s
  sampled tokens included: 66 comparisons. `make plan-check` passes, `make
  ra4-dev-frozen` gives back `docs/ra4-dev-model-0.8.0.md` byte for byte,
  `make serve-examples` gives the same tokens. `make test` takes 148.6 s
  against `main`'s 145.3 s (one run each).

## D84 — Integer weights in the C export: quantized when read, a struct of integers and scales, the host's copies followed (2026-09-27)

Plan 0.9.0 step 7. The plan's step read "`int8` and `int4` realizations,
their packed buffers and scales in the checkpoint reader and the kernels;
check: C1 on the quantised entries of `tests/quant`". Facts checked before
the step, and what was decided on them.

**No checkpoint holds integers.** A checkpoint holds `f64` reals; the
native backend quantizes the weights each time it loads them
(`Realization::quantize_args`, `QTensor::quantize`), the scheme being the
realization's and the scales data (D44, question 57). So the C program
quantizes as it reads, and the plan's "packed buffers and scales in the
checkpoint reader" is wrong. `presk_quantize` (`presk_rt.c`) is a C
transcription of `QTensor::quantize`: per channel (each index of the axes
other than the one the scales are constant along) the largest magnitude in
`f64`, a NaN passed over as `f64::max` passes over it; the scale
`f32(m/qmax)` (`absmax`) or the smallest power of two with `m/s ≤ qmax`
(`pow2`), 0 for an all-zero channel; each weight `w/s` rounded to the
nearest integer, ties to even, clamped to `±qmax`, 0 for a NaN (which C
would leave undefined); `int4` two to a byte, the even element in the low
nibble. `main` and `generate` read a quantized leaf's reals into memory of
their own (`malloc`), quantize them into the arena and free them; the
arena counts the integers and the scales.

- **Checked** against presk-rt (`presk-export-c/tests/runtime.rs`, 0.4 s in
  `make test`): 4 027 392 weights in `int8` and `int4`, `absmax` and `pow2`,
  in ten shapes (the scales' axis first, in the middle and last, of odd
  lengths), channels of nine kinds: weights of every exponent, ordinary
  ones, ties `(n + ½)·s` on the channel's own scale (426 083 ties), zeros of
  both signs (20 773 channels), subnormals, huge values and infinities,
  NaNs among weights, reals whose scale falls below every `f32`, and a
  largest magnitude of exactly `qmax·2ʲ`, where `pow2`'s loops stop on an
  equality. Integers and scales bit for bit. `presk_dequantize`, the copy a
  kernel that reads reals gets, on 6 025 989 elements into the three types,
  bit for bit with `Buf::of(to, b.to_f64())`. Mutations of the C (ties
  away from zero, NaNs not passed over in the maximum, `pow2` as `absmax`,
  the halving loop stopping on an equality, the nibbles swapped) each fail
  it; removing the NaN guard does not on aarch64, whose conversion of a
  NaN to `int` gives 0, what the guard makes defined.

**`pow2` never ended on a tiny channel.** presk-rt started its two loops at
`2^⌈log2(m/qmax)⌉`, which `powi` makes 0 when `m/qmax` is at most 2⁻¹⁰²⁴
(a channel whose largest weight is at most about 7·10⁻³⁰⁷ in `int8`); the
second loop then doubled 0 for ever. The loops find the smallest power of
two with `m/s ≤ qmax` from any finite power of two, so they now start at 1
(at most about 2 100 steps), in presk-rt and in the C. Wherever the old
start ended, the scale is the same: 399 256 scales of maxima of every
exponent compared (`presk-rt/tests/quant.rs`), and 744 of those drawn that
never ended before now give the scale 0 and zeros. Only the native
backend's quantization moves, and only where it hung.

**The conformance weights are the unquantized parameters.** `presk export`
saved `args[0]` after `quantize_args`, as `f32(q·s)`. Quantizing those again
need not give the same scales, `s′ = f32(f32(qmax·s)/qmax)`: on 2·10⁵
random scales 0.8 % moved in `int8` and 10.7 % in `int4`. The export now
writes the arguments as read, before a realization quantizes them;
`--weights` names a checkpoint of reals already.

**Integer weights in the program: a struct.** The step had to choose
between two pointers per leaf and one value. Decided: one value,
`presk_q8` or `presk_q4` (`presk_rt.h`), `{ const int8_t *q; const float
*s; }`, as llama2.c's `runq.c` has `QuantizedTensor`. A leaf of the `_in`
struct, a parameter of a host function and a slice passed to a callee are
one C value each, its integers and scales travelling together; a view at
element `e` is the two pointers moved, `(presk_q8){w.q + e, w.s + e/k}`
(`e/2` bytes for `int4`), `e` a whole number of channels (and of bytes),
which the export checks.

**The kernels.** An integer input is `const int8_t *` or `const uint8_t *`,
and its scales come after the output, `const float *restrict sc<i>`, in
input order, as `NativeExec::run` passes them. A load is `f32(q)·s`, one
`float` multiplication (`(float)in1[e] * sc1[c]`, `presk_int4` for a
nibble), widened in a `double` kernel; a `bf16` kernel computes in `float`
and uses it as it is. The scale's index is the element's with the scales'
strides (`BufSpec::scale_strides`), zero along the axis they are constant
along. A vector load is the same lane by lane; its scales move with the
lanes along the last axis, or stay, when the last axis is the one they are
constant along.

- **Checked** as step 1's kernels are, run inside the kernel backend over
  `dlopen`ed C (`presk-export-c/tests/kernels.rs`): the 41 primitive cases
  with every real operand of rank one or more quantized (the host marks an
  integer input on every primitive), in `int8` and `int4`, the scales
  constant along the first axis or the last, in `f64`, `f32` and `bf16`,
  scalar and at each type's natural width: 960 cases, 2 880 runs; and the
  four nests of `layout.rs` with an operand quantized along its contracted
  axis (the keys of `scores`, which a pack copies, so the pack kernel loads
  the integers), every third point of each space with the splits of every
  fifth: 500 points and 568 splits in each width, each over three
  partitions of its range. All bit-identical to the native backend.
  Every point and every split is an `#[ignore]`d test: 1 491 points and
  7 950 splits in each width, bit-identical, 63 s.

**The host.** The recorder's buffers gain a type of integer weights,
`Store::Quant { bits, k, inner }`, and the axis their scales are constant
along (`RecBuf::quant`). What the host does with them, mirrored:

- a kernel the host marks takes the integers as they are, the view
  starting on a channel;
- any other (a fusion group reading a norm's `γ` and `β`, which
  `tensor_prim_kinds` does not mark) gets the host's conversion at its edge,
  `Buf::of(dtype, b.to_f64())`: a dequantizing copy (`presk_dequantize`);
- a slice (`slice_tree`) stays integers where `QBuf::slice_leading` keeps
  them, the scales at the element offset over `k`; where it gives up (the
  scales constant along the sliced axis, or an `int4` slice of odd length,
  which would start inside a byte), `with_data` makes a copy of its values
  in **`f64`**, which the kernel then converts to its type. In a fold the
  copy is made while the host feeds the iteration: the recording keeps it
  apart (`Fold::slices`), and the structured host dequantizes each slice
  into one buffer before each call (`Feed::Dequant`). MiniGPT at `D = 9`,
  `H = 3` in `int4`: attention's matrices, 81 per layer, reach their `dot`
  as copies; the MLP's, 324, stay integers.
- A view that reshapes integer weights (`merge_axes`, `split_axes`,
  `reshape_like`) keeps them with their old scales' axis under a new shape,
  which need not be where the scales are constant any more. None occurs on
  the test set: the recorder refuses it. **`E8007` is narrowed**, no longer
  retired: integer weights in the entry's result, under a reshape, or drawn
  from by `sample`.

**Checks** (`presk-cli/tests/export_c.rs`, through the binary, both hosts,
each program built with and without OpenMP, `-Wall -Wextra` silent):

- **C1 in `make test`**, 42 programs: `tests/quant/odd.psk` (rows of 13, an
  `int4` row every other one starting mid-byte) in `int8` and `int4`,
  `absmax` and `pow2`, in the three types; MiniGPT's forward pass with its
  head, embeddings and every matrix quantized, in `int8` `absmax` and
  `int4` `pow2`, in the three types; `next_token` in `int8` `f32` and
  `int4` `bf16`; `loss_grad`, whose scan and backward fold slice the
  integers layer by layer, and `adamw.step`, whose fusion groups read their
  values, in `int8` `f32`; `D = 9`, `H = 3` in `int4`, `f32` and `bf16`; the biases
  and the norms' parameters in `int8` (a bias's `add` loads `int8/0[64]`,
  LayerNorm's fusion group reads copies); the plan stating splits at three
  depths, its realization quantizing the matrices. MiniGPT's programs on
  synthesized parameters and with their zeros drawn. All bit-identical to
  the native backend; C2 outside the band only where the native backend is
  (MiniGPT in `f32` 6 of 1 024 with its zeros drawn in `int8`, 8 in `int4`;
  in `bf16` 26 to 35; `loss_grad` 102 and 104 of 8 768, question 75's
  cancellation).
- **C5 in `make test`**: `--serve` under a realization quantizing the
  matrices, `generate` quantizing them as `presk serve` does: `int8` `f32`
  through both hosts and `int4` `bf16`, 15 documents, each `presk serve`'s
  byte for byte.
- **The check checked**: three mutations of an exported program, each of
  which fails C1: a scale one channel off (the head's first channel takes
  its neighbour's), the two nibbles of a byte swapped, and the scale kept
  in `f64` when quantizing (`w/(m/qmax)`). The last changes nothing on
  MiniGPT's drawn weights, a few thousand: an integer moves only when
  `w/s` lies within about 10⁻⁵ of a half. So its checkpoint gets a tie
  `3.5·s` in each column of the head, which presk-rt rounds to even and the
  mutation moves to one side.
- **With trained checkpoints** (`make c-conformance`, 3 min 43 s against
  step 6's 3 min 19 s): the tiny MiniGPT's (`make train-examples`) forward
  pass and `next_token` in `int8` `absmax` (`f64`, `f32`, `bf16`) and
  `int4` `pow2` (`f32`), C5 under `--serve` in each, the course GPT at
  `B = 1`, `T = 256` in `int8`, `f32` and `bf16`, and MiniGPT under the
  plan `presk plan` gives its quantized realization with the frozen profile
  (`presk plan` keeps the `quant` map): 12 programs and 4 generation loops,
  all the native backend's bits and `presk serve`'s documents; with
  `PRESK_C3=1`, each built by Clang and for x86-64 without a warning, the
  Clang programs giving the native bits too.
- **The native backend unchanged**: on the eleven entries in the three
  types, `main`'s release binary (`685b40e`) and this branch's print the
  same kernels and the same native outputs, 66 comparisons. `make
  plan-check` passes, `make ra4-dev-frozen` gives back
  `docs/ra4-dev-model-0.8.0.md` byte for byte, and `make serve-examples`
  gives the same tokens (greedy `33 33 33 33 0 0 0 0`, sampled `19 55 1 63
  57 1 54 29`). `make test` takes 158.6 s against `main`'s 147.8 s (warm runs,
  `main` built in a worktree with its own target directory; the branch's
  three runs took 158.6 to 161.9 s).

**A stale build, found on the way.** `make test` first failed in
`presk-core`'s AD test: `canonicalize` of its repository root, not found.
Its binary had been compiled on 2026-09-26 from a worktree of `main` in a
session's scratch directory, built into this repository's `target/`, and
the worktree was gone; cargo judged the binary fresh, presk-core's sources
being the same. The workspace's crates were cleaned and rebuilt. A
worktree built for a comparison takes its own `CARGO_TARGET_DIR`.

## D85 — The SCoP mode: each nest at point zero in arrays of literal dimensions, its static control checked with Polly (2026-09-27)

Plan 0.9.0 step 8 (§3.7). The plan's step read "`--mode scop` emits each nest
of the entry at its default layout (one worker, one lane, 0.6.0's kernel);
each nest is a standalone C function between `#pragma scop` and `#pragma
endscop`, with a driver that times it; check: every nest of the test set's
entries is emitted, compiles standalone, and is accepted by a polyhedral
front end (`pet`, or Clang with Polly) as one SCoP per nest". Facts checked
before the step, and what was decided on them.

**The nests are RA4's.** `presk export --target c --mode scop --entry E`
captures the entry's launches from their shapes, running no kernel
(`ra4::capture`, `Capture::Shapes`, D65), as `presk plan` does, and lowers
each at point zero (`ra4::point_zero`: the nest's default layout, one lane,
split none, no pack; `plan::lower_launch`), one kernel each. A captured
launch is a nest at one shape in one type, so the unit is `(nest, shape,
type)`: one C function each, named after the nest (`Linear.apply/%3` is
`Linear_apply_3`, `Linear_apply_3_2` at its second shape). A nest launched
at several shapes is several functions because a SCoP has literal bounds and
RA4 prices a nest per shape. The capture records fusion groups and tensor
primitives only; the kernels the host launches itself (`positions`,
`causal_mask`, `one_hot`, fills) are not nests, no plan stating their layout,
and are left out: `MiniGPT.apply` is 23 nests at 26 shapes, the test set 383
shapes in each type, the course GPT 28.

**The form.** The kernel printer gained a SCoP form (`Printer` with `scop`,
in `presk-export-c/src/lib.rs`; `scop::scop_fn`), which changes how the
kernel is written, not what it computes:

- **Arrays of literal dimensions, one index per axis** (`const float
  in0[restrict 2][8][16]`, `in0[i0][i1][i2]`), as PolyBench/C writes its
  kernels, where the program's kernels address a buffer at its linear
  offset. The kernel IR keeps one index per axis; the linear offset is a
  choice of the program's pointers. Each axis's index is multiplied out as
  an offset is; a buffer of rank 0 is `[1]`, read at `[0]`. Graphite
  accepted both forms on a GEMM (the prompt's check); Polly then needs no
  delinearization, and pet and PLUTO read the accesses as written.
- **No range and no OpenMP**: every loop from 0 to its literal bound.
- **A branch on a computed value that only assigns scalars is a selection**
  of each value: `if (x > m) m = x;` is `m = x > m ? x : m;`, and an
  argmax's two assignments share a condition computed first, `c0 = x >= m;
  m = c0 ? x : m; n1 = c0 ? k : n1;`. The same bits (the unselected value
  is the variable itself), and control that depends on no data, which is
  what a static-control part asks. A branch guarding anything else would
  stay a branch and make the nest not static control; point zero has none.
- **The locals** are declared, at zero, before `#pragma scop`, so that the
  region between the pragmas is the loops alone.
- **Refused**: a kernel with lanes (point zero has one) and integer weights
  (below).

**The pragmas' warning.** GCC 13.3 and Clang (21.1 through zig, 23.1.2)
warn about `#pragma scop` under `-Wall`, which C3 forbids. Each file ignores
`-Wunknown-pragmas` around its function (`#pragma GCC diagnostic push`,
`ignored "-Wunknown-pragmas"`, `pop`), which GCC and Clang both read
(checked on the three): the file builds without a warning with any flags,
and another unknown pragma elsewhere still warns. Not `-Wno-unknown-pragmas`
in the Makefile, which a user building a file by hand does not get and which
hides every other unknown pragma; not a guard (`#ifdef`) around the
pragmas, which pet, running Clang's preprocessor, would not see.

**Calls a tool can take for operations.** A call a polyhedral tool knows
nothing of is a side effect. The transcendental functions are declared
`__attribute__((const))` in each file that calls them (musl's functions read
and write no memory and set no `errno`), not in `presk_rt.h`, where it would
let a compiler move or merge the calls of every exported program. The
inline conversions of `presk_rt.h` (`presk_bf16_to_f32`, `presk_f32_to_bf16`,
`presk_f64_to_i64`, `presk_f32_to_i64`, `presk_f64_to_bf16`) are declared
`const` there (`PRESK_CONST`, under `__GNUC__`): Polly runs before inlining
(below), so in `bf16` every load and store was an unknown call, and Clang
ignores an attribute declared after the definition ("attribute declaration
must precede definition", checked). A compiler sees their bodies anyway, so
the programs' code does not change; C1 in `make test` covers every program
again.

**Integer weights are refused** (exit code 2, nothing written): a nest
reading them loads `f32(q)·s` from packed integers (two to a byte in
`int4`, a row possibly starting inside a byte) and their scales, which the
arrays of this mode do not print. The layouts RA4 ranks are on reals.

**The options.** `--mode scop` is `--target c`'s (the PyTorch target
refuses it; `--mode readable` and `graph` with `--target c` are refused as
the PyTorch export's modes). `--entry`, `--dim`, `--const`, `--realize`
(a mixed realization gives each nest its site's type) and `--seed` apply.
Refused, exit code 2, nothing written: `--plan` (every nest is at point
zero, before any plan), `--threads` (one worker, no OpenMP),
`--straight-line` and `--serve` (no host), `--weights` (the shapes do not
depend on them, and the driver draws its inputs), `--no-exact` (no exact
reference is computed), and a quantized realization. No code is added:
option refusals are usage, as `--serve` with `B ≠ 1` is.

**The driver.** `driver.c` builds `scop`: each nest (or those named) called
`R` times (5; `make time`, 11) on inputs it draws, the median (`sorted[R/2]`)
and the best wall time in µs; with `-c conformance/native.bin`, each called
once and its output compared with the native backend's byte for byte. The
inputs are drawn with `splitmix64`, seeded per nest by the export's seed and
the nest's place (`scop::Draw`, the same numbers in Rust and C): a real
uniformly in [-1, 1), exact in `f64` (the top 53 bits over 2⁵³, twice, less
one), rounded to the nest's type as `Buf::of` rounds (`bf16` through
`float`); an index below the least length of an axis it indexes, or of a
loop it is compared with (a cross-entropy's gradient compares its labels
with the classes), so that every gather and scatter stays in its array, and
0 when it does neither (`scop::index_bounds`). The capture's own inputs are
zeros, and a captured operand carries no bound, so neither could serve. The
output starts at zeros before each call, as the native backend allocates
it: a nest that accumulates reads it. The export runs each nest natively at
point zero (`NativeExec`, one thread) on the same draws and writes the
outputs, in order, to `conformance/native.bin`.

**Static control** (`scop::static_control`). A nest is a static-control
part when no loop variable is assigned and every access indexes each axis
by an affine function of the loop variables; every loop already runs from 0
to a literal bound, and the selections leave no data-dependent control. The
nests that are not index an axis by a value loaded from an index array: a
gather (`embed`, `table[ids[b][t]][d]`; `cross_entropy`,
`logits[b][t][y[b][t]]`) or a scatter (`embed_vjp`, `out[ids[b][t]][d] +=
…`, one reason though it reads what it writes). On the test set, 366 of the
383 nests of each type are static control, the 17 others gathers and
scatters; the course GPT, 26 of 28. `nests.json` and the README give each
nest's loops (those of two iterations or more, below) and the reason.

**The front end: Polly, from LLVM's release.** None of the plan's front
ends is on falicon (clang, pet, PLUTO, iscc and PPCG missing; Ubuntu 24.04
packages neither pet nor PLUTO; `clang-18` and `libpolly-18-dev` need the
user's root). LLVM's release for Linux on aarch64,
`LLVM-23.1.2-Linux-ARM64.tar.xz` (1.8 GB from GitHub, 43 s here), links Polly
into its `clang`: `-mllvm -polly` works alone, and loading `LLVMPolly.so`
as well aborts ("Option 'polly-dependences-computeout' registered more than
once"). `make llvm` downloads it, checks its SHA-256 (pinned), keeps
`bin/clang` and `lib/clang/23/include` in `.llvm` (413 MB) and removes the
rest, without root: 1 min 44 s. It is a new tool, with its row in
`dependencies.md` and a section in `presk-tooling.md`. The flags
(`scop::POLLY_FLAGS`, the Makefile's `POLLYFLAGS`, `make polly`), each
found necessary on the test set:

- **`-mllvm -polly-position=early`.** At Polly's default position, before
  the vectorizer, `-O2` has rewritten the nests' outer loops: 149 of
  `loss_grad`'s 157 `f32` nests gave no SCoP ("Failed to derive an affine
  function from the loop bounds"), and a loop of four iterations was fully
  unrolled first. At its early position Polly runs after its own
  canonicalization, before LLVM's loop passes; `-fno-unroll-loops` is then
  not needed.
- **`-mllvm -polly-optimizer=none`.** Polly's scheduler tiles, and the AST
  gains loops; without it the AST is the nest's own schedule.
- **`-mllvm -polly-process-unprofitable`**, so that a nest of one loop or
  little work is still a SCoP (Graphite drops those, below), and **`-mllvm
  -polly-detect-full-functions`**, so that a SCoP may be the whole function:
  a nest of no loop (a scalar `sqrt`, 3 of `loss_grad`'s) is a SCoP of one
  statement only so.
- **`-fno-math-errno`.** Under ISO C's `math_errno`, `sqrtf` is a call that
  may write `errno` ("This function call cannot be handled"): 2 of
  `loss_grad`'s nests. Without it, `sqrt` is the operation, correctly
  rounded either way, so no bit moves.
- **`-mllvm -polly-code-generation=ast`**: the AST and no code. Polly 23.1.2
  crashes generating the code of a SCoP that is a whole function of no loop
  (`polly::simplifyRegion`, exit 139), after printing its AST, which a first
  count read without looking at the exit code.
- **`-mllvm -polly-print-ast`**, read per function (`scop::polly_scops`): a
  file's other functions may have SCoPs of their own (`presk_f32_to_bf16`
  in a `bf16` file).

**The criterion**: a nest is static control if and only if Polly finds
exactly one SCoP in its function, whose AST has as many loops as the kernel
has loops of two iterations or more. isl prints no `for` for a loop of one
iteration (the course GPT's batch loop at `B = 1`), so those are not
counted. Checked (`presk-cli/tests/export_scop.rs`, `#[ignore]`d, `make
c-conformance`):

| | nests | static control, one SCoP of their loops | the others, no such SCoP | the others, allowed non-affine accesses |
|---|---|---|---|---|
| the test set, `f64`, `f32`, `bf16` | 1 149 | 1 098 of 1 098 | 51 of 51 | one SCoP of their loops, 51 of 51 |
| the course GPT, `B = 1` (three types), `B = 16` (`f32`) | 112 | 104 of 104 | 8 of 8 | 8 of 8 |

The rule and Polly agree on every nest, and Clang 23 compiles every file
without a warning under `-Wall -Wextra -Werror`. With
`-polly-allow-nonaffine` Polly takes the gathers and scatters whole,
over-approximating the loaded index by the whole axis: the loaded index is
the one obstacle.

**Not Graphite.** GCC 13.3's Graphite is on the machine but is not a front
end the plan names. On the test set's `f32` files (`-O2 -fno-inline
-fno-tree-loop-distribute-patterns -fgraphite-identity`, one SCoP in the
function's section holding every loop): 301 of the 366 static-control nests
pass; it rejects nests of one loop (a heuristic) and of none, cuts a softmax
(2 of its 6 loops), counts a `sum_to`'s loops as GCC's passes left them (7
for 6), and takes 14 of the 17 gathers and scatters whole. Its verdict
follows GCC's pipeline, not the nest. **Not pet** either: it would be built
from source against Clang's C++ libraries (the release ships them; not
tried). The files keep pet's convention, the pragmas, for pet, PPCG and
PLUTO, which read the region between them; none was run.

**The bits.** The SCoP form is not `kernel_fn`'s, so its bits are checked
as step 1's are: every primitive case of the conformance suite at point
zero in the three types, 123 distinct kernels (114 static control, 3
selecting two values: argmax), each in its nest file, compiled with the
export's flags and `-Werror`, loaded with `dlopen` and run on the driver's
draws against the native backend (`presk-export-c/tests/kernels.rs`, 0.4 s).
Through the binary (`make test`): every nest of the test set's eleven
entries in the three types, 1 149, each file compiled on its own by its
Makefile with GCC and `-Werror` (no warning), run once against
`conformance/native.bin` (bit for bit) and once timed; 3 s. The check
checked: built in GCC's GNU mode, which contracts `a + b*c`, 7 of `loss`'s
28 nests move, every dot among them; a softmax's maximum made a minimum
moves that nest. The course GPT's nests too, at `B = 1` in the three types
and at `B = 16` in `f32` (`make c-conformance`, 11 s with Polly).

**Checks.** `make test` takes 161.0 and 168.0 s against `main`'s 157.7 and
161.2 s (warm, two runs of each in turn, `main` built in a worktree with its
own target directory): 3 to 7 s more, the test set through the binary 3 s of
it (its exports and builds in parallel), the primitives in the SCoP form
0.4 s. The native backend is unchanged: nothing in
`presk-kernel` or `presk-cranelift` changed, and on the eleven entries in
the three types `main`'s release binary (`7a12849`) and this branch's print
the same kernels and the same native outputs (66 comparisons). `make
plan-check` passes; `make ra4-dev-frozen` gives back
`docs/ra4-dev-model-0.8.0.md` byte for byte; `make c-conformance` passes
in 4 min 7 s, the SCoP mode's tests 19 s of it (step 7's took 3 min 43 s).

**Left for later.** The comparison itself (plan §8): a polyhedral tool's
transformation of these nests against RA4's ranking of their layouts, with
the driver's times. pet, PPCG and PLUTO, not run. Integer weights in this
mode. The host's own kernels (positions, masks), which no plan decides.

Alternatives considered:

- **The program's form** (`kernel_fn`, linear offsets, a range). No second
  printer and no second bits check, but a linearized access asks a tool to
  delinearize, and a range is not a literal bound.
- **One function per nest over all its shapes**, a shape parameter: bounds
  would no longer be literal, and RA4 prices a nest per shape.
- **The branch kept as a branch**: Polly takes it as a non-affine subregion,
  but a source-level tool (pet, PLUTO's clan) sees data-dependent control.
- **Graphite as the front end**, installed; and **pet**, which the plan
  names first: above.
- **The `const` attribute in `presk_rt.h` for the transcendental functions
  too**: one place, but every exported program's calls could then be moved
  or merged by its compiler.

## D86 — Version 0.9.0 closed: C1, C2, C3 and C5 hold, C4 reported; the documentation, tutorial step 18, and what is left (2026-09-27)

Plan 0.9.0 step 9. Nothing is measured anew for the statements in this
step: they were checked step by step (D77–D85), and the targets that check
them were run again on this step's binary (below). The step writes the
version down, adds the tutorial's step, and records what 0.9.0 leaves.

**The statements** (plan §3.8; the manual's `experiments` chapter §9).

| # | Verdict | On what |
|---|---|---|
| C1 | **holds** | every entry of `tests/plan/entries.txt` in `f64`, `f32` and `bf16`, on synthesized parameters and with their zeros drawn, MiniGPT's five under a mixed realization, the test set under `presk plan`'s plans (the frozen profile), a plan stating splits, the native backend's own threshold at `D = 64`, both hosts, with OpenMP and without, on one thread and every thread (`make test`); the course GPT at its constants with its zeros drawn and with its trained checkpoint at `T = 256`, and at `B = 16` under its plans (201 of 228 launches among ten threads) and the default layout (`make c-conformance`); 42 programs with integer weights and the trained checkpoints in `int8` and `int4` (D84); every nest of the test set at point zero in the SCoP form, and 123 primitive kernels in it (D85). No transcendental function differs from the `libm` crate (D78), so the plan's exception for one is not used. |
| C2 | **holds**, as a copy | the outputs are the native backend's, so they leave the band on exactly its elements: 34, 121 and 99 of `loss_grad`'s 8 768 in the three types, `SelfAttention.apply` 1 and 4 of 256 in `f32` and `bf16`, and with the zeros drawn MiniGPT 5 and 35 of 1 024, the course GPT 18 and 63 of 736 (D76, D79, question 75). |
| C3 | **holds** | GCC 13.3 with and without `-fopenmp` on every program; Clang 21.1 and `x86_64-linux-musl` through `zig cc` 0.16.0, the aarch64 program linked, run and bit-identical, `generate` giving `presk serve`'s tokens, the x86-64 one linked, the OpenMP builds compiled only (zig has no OpenMP runtime); the SCoP mode's files on Clang 23.1.2 (D81, D83, D85). |
| C4 | reported | the course GPT's forward pass at `B = 16`, `T = 256`, `f32`, trained checkpoint, the frozen profile's plan, median of seven: every thread, native 1 094 ms and C 290 ms; one thread, 3 132 and 1 700 ms (`make c-speed`, D81). Not measured again: it decides nothing and needs a quiet machine. |
| C5 | **holds** | `generate`'s `--json` is `presk serve --backend native`'s byte for byte, greedy and at temperatures 0.3, 0.8 and 1.5 with four seeds, in the three types, through both hosts, on `serve_minigpt.psk` with its zeros drawn (`make test`) and on the trained checkpoint (`make c-conformance`), in `int8` and `int4` too (D83, D84). |

**The documentation** (decided in this step).
- **A chapter of its own, `export_c`.** The `native` chapter's section "A
  C program" had grown a paragraph per step to about 285 lines, a third of
  the chapter, organized by step rather than by subject. It becomes a
  chapter (`crates/presk-manual/manual/export_c.md`, module `export_c`),
  reorganized: how a program is made, the host's structure, plans and
  threads, sampling and generation, integer weights, why the bits are the
  native backend's, what is checked, portability, the band, speed, the
  SCoP mode, and what the export does not do. `native` keeps a paragraph
  that points to it; `cli` keeps the command, its flags and its directory.
  Alternative: keep the section in `native`, as the plan allowed; rejected
  because the export has its own command, its own exactness rules and its
  own checks, and the section was the longest of the chapter.
- **`experiments` §9.** Every version's statements are answered there
  (§4–§8), so 0.9.0's are too, though they test no hypothesis of the
  research programme: a table of the five with what was asked and
  measured, how, what makes C1 hold, why C2 became a check of a copy, and
  C4's figures.
- **`cli` and `diagnostics` checked.** `presk --help` and `presk export
  --help` against `cli`; the codes the code emits (`E8001`, `E8003`,
  `E8004`, `E8006`–`E8009`) against `diagnostics`, whose list of the test
  programs lacked `tests/diag/export_c_quant.psk` (step 7's).

**Tutorial step 18, "A C program".** The tutorial's attention, `attend` of
`tutorial/03_attention.psk`:
- exported at step 3's dimensions (`B = 1`, `T = 3`, `f64`): 7 launches of
  7 kernels, `scores` and the entry, an arena of 1 472 bytes. The chapter
  quotes a kernel under its PSK-Kernel text and the entry's function, both
  taken from the files, then `make -C out/tut_c check` with `PRESK` set, so
  that the script also prints the distance to the exact reference
  (3.5·10⁻¹⁶), and the program run by hand, its output read by `presk data
  info` (step 3's result in `f64`);
- under step 17's plan at `B = 16`, `T = 256` in `f32`: 5 of 8 launches
  divided among up to 20 threads, checked on every thread and with
  `OMP_NUM_THREADS=1`, and timed with `-r 11` (2.4 ms, 8.6 ms on one
  thread). **What the chapter quotes about threads**: the plan's workers are
  those of step 17's calibration, unpinned on this machine's twenty cores,
  so the count is the machine's; the chapter quotes it as run, as step 17
  does, says where it comes from, and shows the region of the values `ein`,
  whose batch loop of 16 caps its team at 16. It does not time the native
  backend beside the program: `make tutorial` runs the debug binary, whose
  host is slower than a release build's, and C4 is measured on the course
  GPT with the release binary. `--no-exact` is not needed: the exact
  reference takes 5 s at this size;
- in the SCoP mode at `B = 16`, `T = 256` in `f32`: 6 nests, all static
  control, the softmax's nest quoted, `make check` bit for bit. Polly, run
  once on these nests (one SCoP each), is named and not run by `make
  tutorial`, since LLVM's release is 1.8 GB to download.
`make tutorial` covers steps 0 to 18 and needs a C compiler; every output
step 18 quotes is from its run, and the other steps' quoted outputs agree
with it but for timings and what depends on them (step 17's fit, its
predictions and its rank; both `ein`s chose order `JI` where the chapter
quotes `IJ`, a choice within `u`, as in 0.8.0's second run; the same
tiles, packs, splits and verdicts).

**Two strings of the binary.** The export's summary line said "1
functions and 0 loops": it counts the functions besides the entry's, and
now says so, "1 function besides the entry's and 0 loops", with the
plural right. `presk --help` named the PyTorch export only; it now says
"PyTorch and C exports". No test holds either string.

**Bookkeeping.** CHANGELOG 0.9.0 dated, its steps newest first as the
released versions' are, step 1's sentence on the version and the questions
moved to step 0, where they belong, and a paragraph on what the version
shows. Questions 73–75 copied to `presk-specs` as
`implementation/presk-spec-questions-0.9.0.md`. Version `0.9.0` in
`Cargo.toml` and `Cargo.lock`; the exported READMEs and headers take it
from `CARGO_PKG_VERSION`, and no snapshot holds it. In `presk-specs`, the
language reference's status sections of PSK-Realize and PSK-Kernel are
brought to 0.9.0 (the first said a nest's layout was chosen at its first
shape, which D82 changed; the second said nothing of the C export) and its
open questions to 75; the report, the slides and the thesis summary say
0.9.0.

**Left for later.**
- **The training loop in C**, llm.c's other half: the `training run`, its
  data stream and its checkpoints around the gradient and optimizer
  entries 0.9.0 exports (plan §8).
- **`.pskk`, `presk build` and the bundle** (questions 35, 36 and 73): the
  export prints C at §8.5's seam without the binary form.
- **The work a gradient does and discards**: dead launches kept in the C so
  that C4 compares the same work (`docs/kernels-roadmap.md`, "Work a
  gradient does and discards").
- **Reshaping integer weights in the native host**: a view that merges or
  splits a quantized leaf's axes keeps its old scales' axis under the new
  shape (D84); the export refuses it (`E8007`), and nothing of the test set
  does it.
- **The polyhedral comparison**: pet, PPCG and PLUTO not run; a tool's
  transformation of the SCoP mode's nests against RA4's ranking of their
  layouts (plan §8); integer weights and the host's own kernels in that
  mode.
- **The CUDA export** at the same seam (one thread per output, reductions
  in order, no fused multiply-add, musl's functions as device code, so that
  C1's bit identity holds on the GPU; the host recorded as here), decided
  by the user on 2026-09-26 for after the C export (the design note
  `presk-project/docs/presk-c-export-2026-09-26.md`, last section); then
  the native GPU backend and the matrix unit.
- **C4's shares**: whether the C program's lead on one thread is GCC's
  code or the host's copies, and on ten the thread spawned per piece, is
  not measured (`docs/kernels-roadmap.md`, "What the C export's speed
  says").

**Checks**, on falicon on 2026-09-27 with this step's binary: `make test`
(207 tests pass), `make check-examples` (only `W0001`), `cargo doc -p
presk-manual` (no warning); `make plan-check` (11 of 11 entries, the same
kernels); `make ra4-dev-frozen` (`docs/ra4-dev-model-0.8.0.md` byte for
byte); `make c-conformance` (18 tests, 4 min 9 s); `make tutorial`, every
command of steps 0 to 18, in 1 min 15 s.
