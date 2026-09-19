# Architecture decisions

One paragraph per decision, newest last. The plan these come from is
`presk-specs/implementation/presk-plan-0.1.0.md`.

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
