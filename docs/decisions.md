# Architecture decisions

One paragraph per decision, newest last. The plans these come from are
`presk-specs/implementation/presk-plan-0.1.0.md` through
`presk-plan-0.5.0.md`.

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
