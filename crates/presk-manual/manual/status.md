# Status

Version 0.1.0 was built in eight steps (plan:
`presk-specs/implementation/presk-plan-0.1.0.md`); all are done and the
acceptance criteria below hold. `CHANGELOG.md` lists the contents of the
version.

| Step | Contents | Status |
|---|---|---|
| 0 | Workspace scaffold, CLI surface, Makefile, examples copied from `presk-specs`, decision and spec-question logs | done (2026-09-18) |
| 1 | Lexer (Unicode identifiers, ASCII synonyms, layout newlines), parser, AST, diagnostics rendering, `presk dump --stage tokens\|ast`, syntax errors in `presk check` | done (2026-09-18) |
| 2 | Dimension expressions as linear forms, the Presburger-fragment solver (equalities, bounds, divisibility, non-zero divisor obligations), elaboration: `import` resolution, `const` evaluation, `static assert`, module instantiation, `--const` overrides, `presk dump --stage elab` | done (2026-09-19) |
| 3 | Shape typing: named axes, `...` prefixes, broadcasting, `idx`, `params`, `ein`, call argument forms, configuration-block schemas; `presk check` complete, `--summary` | done (2026-09-19) |
| 4 | PSK-Core IR (ANF over exact real tensors), lowering with explicit parameter trees, `stack{L}` unrolling, `presk dump --stage core` | done (2026-09-19) |
| 5 | Regularity classes (`C∞ < C⁰⁺ < Disc`, per argument) and effect checking; `grad` legality with a named culprit | done (2026-09-19) |
| 6 | Automatic differentiation (reverse mode, source to source on PSK-Core) | done (2026-09-19) |
| 7 | Reference interpreter in arbitrary precision (`rug`/MPFR), `presk run` | done (2026-09-19) |
| 7b | Readable PyTorch export and generated conformance test, `presk export` | done (2026-09-19) |
| 8 | Hardening, `--json` for `check`, `CHANGELOG.md`, README, spec questions sent to `presk-specs`, tag `v0.1.0` | done (2026-09-19) |
| — | Tutorial chapter (`tutorial/`, `make tutorial`); PyTorch conformance verified with `make conformance` | done (2026-09-19) |

## Acceptance criteria of 0.1.0

- The five example programs (`tests/examples/{mlp,layernorm}/*.psk`,
  `tests/examples/minigpt/*.psk`) pass `presk check`.
  *Done: all five pass the complete `presk check` (shapes, regularity,
  effects) with only `W0001` implicit-dimension warnings (`make check-examples`);
  since 0.2.0 step 2 the `training run` of `train_minigpt.psk` is checked
  against `examples/minigpt/data/tokens.psk.d`.*
- `mlp.psk` (`mse`, `mse_grad`), `layernorm.psk` (`LayerNorm.apply`) and the
  `minigpt.psk` forward pass with reduced constants evaluate with `presk run`.
  *Done (`make run-examples`); golden outputs in `tests/run/`, and the AD
  gradient of `mse` agrees with finite differences at 512 bits to 2⁻¹⁰⁰.*
- The three model files export to PyTorch and the generated code matches the
  golden snapshots. *Done: the five files export, the generated Python
  byte-compiles, and `test_conformance.py` is generated with the
  interpreter's data (to be run where PyTorch is installed).*

## Version 0.2.0

Training end to end, planned in
`presk-specs/implementation/presk-plan-0.2.0.md` and built the same day
(2026-09-19); the acceptance criteria of the plan hold (`make train-examples`,
`make conformance`, `make tutorial`). E-graph rewriting and Cranelift code
generation open 0.3.0.

| Step | Contents | Status |
|---|---|---|
| 0 | Version `0.2.0-dev`, CHANGELOG "Unreleased", spec questions 28–34 logged | done (2026-09-19) |
| 1 | The `.psk.d` container (crate `presk-data`), `presk data pack \| lm-windows \| info \| export`, the pipeline over sample indices, `E9xxx` | done (2026-09-19) |
| 2 | The `data` pipeline typed against the objective (sample type = batch type minus `batch`, `B` bound by `batch(B = n)`, the file's type matched and its dimensions bound); `TrainingRun` record in the typed program; `--summary`; `W0002`; `tests/diag/train_*.psk` | done (2026-09-19) |
| 3 | The optimizer lowered to PSK-Core (`adamw.init`, `adamw.lr`, `adamw.step`; AdamW as PyTorch defines it, three schedules, clipping); `sum_squares`; scalar-over-tree arithmetic in the interpreter; `presk run --entry adamw.step`; one-step conformance against `torch.optim.AdamW` (`make conformance`) | done (2026-09-19) |
| 4 | `stack{L}` as a `fold` in PSK-Core (`Rhs::Fold`, scan/reverse/index forms, `set_slice`), the interpreter and AD (backward fold with a generated `fold_bwd` body); `--unroll-stack` for 0.1.0's form; golden outputs unchanged; finite differences through a stacked module (`tests/data/stack_grad.psk`) | done (2026-09-19) |
| 5 | `presk train` (crate `presk-train`): the loop over the interpreter with the objective's `vjp`, the PSK-Core optimizer and the data pipeline; `--steps`, `--epochs`, `--data`, `--ckpt`, `--log-every`, `--json`; checkpoints as `params M` files; regression converges, tiny MiniGPT 20 steps golden; `presk run` on a training file points to `presk train` | done (2026-09-19) |
| 6 | PyTorch export: `--train` writes `pskd.py`, `train.py` and the data file; `--conformance-entry train:K` replays `K` interpreter steps in PyTorch (20 steps of the tiny MiniGPT: worst relative gap 1.3e-10); `--mode graph` writes `export.py` producing `model.pt2` and `model.onnx`, tested under ONNX Runtime; `make venv` gains onnx, onnxscript, onnxruntime | done (2026-09-19) |
| 7 | Manual chapters `training`, `data` and `experiments` (`make experiment`), tutorial step 11 (train a model), README, CHANGELOG 0.2.0, spec questions copied to `presk-specs`, version 0.2.0; tag `v0.2.0` by the user | done (2026-09-19) |
| 8 | Optional Cranelift spike | not done (open for 0.3.0) |

## Version 0.3.0

Native backend first, then e-graph rewriting, planned in
`presk-specs/implementation/presk-plan-0.3.0.md` and built the same day
(2026-09-19); the verification list of the plan (§5) holds, see below.

| Step | Contents | Status |
|---|---|---|
| 0 | Version `0.3.0-dev`, CHANGELOG "Unreleased", spec questions 35–42, decision D21 (the `Backend` seam) | done (2026-09-19) |
| 1 | Crate `presk-rt`: runtime values (exact, f64, f32, i64 buffers with named axes; trees), the `Backend` trait and `Prepared` entries, `InterpBackend`, parameter I/O, `run_with`; `presk run --backend interp` and `presk train --backend interp` over the trait, goldens unchanged | done (2026-09-19) |
| 2 | PSK-Kernel v0 (crate `presk-kernel`): one kernel per tensor primitive at concrete shapes, specialized on first use; loop-level IR with a textual form (`presk dump --stage kernel`); kernel-IR interpreter in `f64`/`f32`; the host program over PSK-Core bodies; `--backend kernel`; the conformance suite (41 primitive cases, 11 example entries, both realizations) with measured gaps | done (2026-09-19) |
| 3 | Cranelift backend (crate `presk-cranelift`): every kernel JIT-compiled on first use, `libm` symbols, `--backend native` on `run`/`train`; the native conformance suite; `presk bench` and the first performance table (tiny MiniGPT forward 27× the reference, 20 training steps in 0.4 s) | done (2026-09-19) |
| 4 | Training natively: `presk train --backend native --realize f64\|f32` measured against the reference (regression 200 steps: 4.4e-16 in f64, 1.6e-7 in f32, a test); `train:K` exports the native `θ_K` too and the experiments chapter's realization table gains presk's `f64`/`f32` points; `make train-examples` extended; `V=64 D=64 H=4 L=4` trains 500 steps in 159 s with the loss decreasing | done (2026-09-19) |
| 5 | `inference serve` resolved in the checker (model, weights path, generation function by signature, `W0003`, `--summary`); crate `presk-serve` (the loop over `Backend`, greedy copy `sample → argmax`, sliding window when the parameters fix `T`); `presk serve`; `presk export --serve` (`serve.py`, `pskd.read_params` with the generated layout); greedy tokens identical on the reference, native f64/f32 and `serve.py` float64/float32; manual chapter `serve`; question 43 | done (2026-09-19) |
| 6 | Crate `presk-opt` over `egg`: one e-graph per PSK-Core function body with the types as class analysis, rules exact over ℝ (algebra, reductions, the adjoint cleanup: `sum_to`/`broadcast_like` identities, `zeros_like`, `field`/`set_field` chains into records, `proj` of tuples), extraction by operation counts, ANF write-back (dead code disappears); `--opt` on `run`/`train`/`serve`/`bench`/`export`, `presk dump --stage core-opt` with a report; the eight run goldens and the 20-step training golden unchanged, a property test against the interpreter at 128 bits, native outputs identical with and without `--opt`; `Linear.apply.vjp` 14 → 7 lets; manual chapter `optimization`; D24, question 44 | done (2026-09-19) |
| 4 | Crate `presk-sched`: fusion groups over PSK-Core bodies (elementwise chains of one result type whose interior values are read only inside, bounded expression duplication), mapped over the leaves of a parameter tree; one loop nest per group in PSK-Kernel; `presk dump --stage sched`; invariants tested on the corpus and the examples. `adamw.step` 414 → 111 launches and 1.54 → 1.32 ms, the forward 108 → 81 launches and no time change (it is contraction-bound), a whole training step 1515 → 884 (42 %, short of the plan's half: the forward and backward chains are cut by reductions); every golden byte-identical. Manual chapter `scheduling`; D30 | done (2026-09-20) |
| 5 | Threads: a kernel whose outermost loop is parallel (one loop nest, every store indexed by that loop's variable) is split across workers by the Cranelift backend, which now compiles the loop over a caller-given range; only kernels above four million weighted operations are split; `--threads N` and `PRESK_THREADS`. Exact by construction and tested bit for bit at 1, 2, 5 and 12 workers in both realizations. One training step of the corpus GPT 4.00 s → 0.99 s at `D=64`, 289 s → 58.3 s at the course's constants. Vectorization deferred with its reason (a contraction's inner sum reassociates); D31 | done (2026-09-20) |
| 6 | Contractions blocked: `dot`/`matmul` hold eight columns of the result in registers across the sum, with a row-wise fallback for other widths; both exact (each output accumulates `k` in order), tested bit for bit against each other and leaving the conformance gaps unchanged. 1024³ `f32` matmul 3.7 → 9.9 GFLOP/s on one thread, 47–56 on twelve (≈ two thirds of scalar peak); a course-scale GPT step 289 s → 39.8 s across steps 4–6; D32 | done (2026-09-20) |
| 7 (0.4.0) | A rewrite the cost model arbitrates: moving a scalar across a contraction (exact over ℝ), the e-graph proposing every operand. The plan's `(q·k)·v` reassociation has nowhere to fire — zero adjacent contractions in the corpus GPT, nonlinearities separate them all. Attention divides a quarter as many elements; the roofline prices the forms apart (10 205 → 9 489 µs) where counts cannot. 0.710 → 0.626 (counts) → 0.617 (profile) seconds a step over forty steps, identical losses; the profile's extraction costs 2.54 s against 0.41 s. D33 | done (2026-09-20) |
| 8 (0.4.0) | The decisive experiment against PyTorch on the course GPT, answered by measurement: **A fails** (94× a step, 82× with `--opt`, against 3× asked), **B fails** (1.14× against 1.30×), **C holds** (predicted 1.087× where 1.14× was measured, 5 % off). Diagnosed: PyTorch's CPU matmul reaches Apple's matrix coprocessor at 1309–1454 GFLOP/s where presk's scalar blocked kernel does 47–56; the rest is presk's non-contraction work. Every transformation this version shipped is exact, so 7.3× came without changing a number. `experiments` chapter §3 | done (2026-09-20) |
| 9 (0.4.0) | Coverage measured on the MLPerf architectures: BERT-large checks whole (`coverage/bert.psk`), the Llama/GPT-J/Mixtral decoder checks with two substitutions (`coverage/llama.psk`), both blocked on the same missing axis split; convolution blocks four benchmarks, segmented reductions two, top-*k* with a runtime parameter gather one, and non-maximum suppression is outside the language. Manual chapter `coverage`, `make coverage` | done (2026-09-20) |
| 7 | Cost model v0.1: work and traffic of every kernel from its IR (`presk_kernel::cost`) and of every PSK-Core node from its type (`presk_opt::model`); `presk bench --calibrate FILE.json` times every kernel natively (`TimedExec`), fits the roofline `λ + max(W/π, Q/β)` by relative least squares and writes the profile with Spearman correlations (counts 0.15, work 0.91, roofline 0.95 on the tiny MiniGPT in `f64`); `--opt --profile` extracts by predicted seconds; goldens unchanged under the profile; bench with and without the profile within the noise; manual section in `optimization`; D25 | done (2026-09-19) |
| 8 | Version 0.3.0; tutorial step 12 (native run and training, `--opt`, `--calibrate`, `--profile`, in `make tutorial`); README, CHANGELOG 0.3.0, spec questions 35–44 copied to `presk-specs`; tag `v0.3.0` by the user | done (2026-09-19) |

### Verification of 0.3.0 (plan §5)

- `make test`: the whole workspace, including the kernel and native
  conformance suites, the e-graph oracle tests (the run goldens and the
  training golden under `--opt`, with and without a profile) and the
  property test of the rewriting against the interpreter at 128 bits.
- `make run-examples`: the interpreter, unchanged; `presk run … --backend
  native --realize f32` on the tiny MiniGPT forward pass (gap 0 on the
  argmax of the example, `native` chapter).
- `presk train … --backend native` on the tiny MiniGPT, `presk serve … --backend
  native` on its checkpoint (`make train-examples`, `make serve-examples`):
  greedy tokens identical on every backend.
- `presk bench` on forward, loss and optimizer step (`make bench`); `presk
  train --backend native --opt --profile` on the same goldens, timed within
  the noise (`make opt-examples`, `make calibrate`).
- `make conformance`: the PyTorch side plus the native K-step line
  (`conformance/native_{f64,f32}.json`); `make experiment`: the realization
  table with presk's `f64`/`f32` points; `make tutorial`: steps 0–12.

Not done in 0.3.0, carried forward: `ein` reassociation and fusion rules
(the `(q·k)·v` decision), a fusion pass making `Q` non-additive, the
`.pskk`/`.pskb` bundle and the Zig runtime (questions 35–36), the precision
solver behind `--realize`, the native StableHLO emission of `--mode graph`.

## Version 0.4.0

Real models, real speed, planned in
`presk-specs/implementation/presk-plan-0.4.0.md` and built on 2026-09-20.
Steps 0 to 9 are in the table above (the 0.4.0 rows); step 10 is this
documentation, the tutorial's step 13 and the version itself.

### Verification of 0.4.0

- `make test`: the whole workspace, including the fusion invariants on the
  corpus and the examples, the bit-equality of threaded and sequential
  kernels at 1, 2, 5 and 12 workers, the bit-equality of the blocked and
  row-wise contractions, the gradient through `merge` and through `dropout`
  against finite differences at 512 bits, and the e-graph oracle.
- `make corpus` and `make corpus-gpt`: the ladder on the French civil code;
  `make coverage`: the MLPerf transcriptions; `make sched` and `make
  calibrate`: the launches fusion removes and the fitted profile.
- `make experiment-gpt`: the course GPT trained by `presk train` and by the
  exported `train.py`, the comparison of `experiments` §3.
- `make tutorial`: steps 0 to 14, whose quoted outputs are produced by the
  commands themselves.

### What 0.4.0 does not have

Vectorized kernels and a matrix-coprocessor or GPU path — deferred with their
reason, since both reassociate a sum and so need a declared realization
(D31, D32); `docs/kernels-roadmap.md` records the target. Convolution, and
with it WaveNet and four MLPerf benchmarks: step 9 established it is a
type-system extension, not a primitive. Multi-output kernels and reduction
fusion, which is why the forward pass fuses as little as it does. An
extraction that knows the partition (question 49). The axis `split` that
would finish the transformer-decoder family (question 45). Per-layer keys
inside a `stack{L}` fold (question 48).

## Out of 0.1.0

E-graph rewriting, cost models, code generation (backend to be chosen in
0.2), PSK-Sched and PSK-Kernel, the `.pskk`/`.pskb`/`.pskt` formats, the Zig
runtime, Lean 4 certification, execution of a `training run`, `presk export
--mode graph` and `--precision=solved`.

## Version 0.5.0 (in progress)

Precision as a resource, plan
`presk-specs/implementation/presk-plan-0.5.0.md`. The subject is RA3 of the
research programme: given a declared error budget, which machine types and
which reduction orders should the backends use, and how far from the exact
denotation does that put the execution? Both previous plans deferred it for
the same reason — the solver could not be evaluated on programs that take
minutes a step — and 0.4.0 removed that reason.

Two tracks, in one version because each is the other's precondition. The
realization becomes an object rather than a flag, which is what decisions D31
and D32 of 0.4.0 needed and could not write down; and one vectorized GEMM
ships under that contract, because the `native` chapter records that `f32` is
no faster than `f64` here, so without vector lanes a precision solver's
objective is flat and its result untestable.

| Step | Contents | Status |
|---|---|---|
| 0 | Version `0.5.0-dev`, crate `presk-precision` scaffolded, decisions D34 and D35, spec questions 50–55 | done (2026-09-20) |
| 1 | The axis `split` (question 45's open half) and per-layer keys inside a fold (question 48's residue) | done (2026-09-20); grouped-query attention now exact, rotary positions blocked on `concat`, not on the split |
| 2 | The realization as an object: per-group storage and accumulator types, a declared reduction order, `bf16` storage, conformance per realization | done (2026-09-20); `f64` and `f32` unchanged digit for digit |
| 3 | The magnitude envelope: `presk calibrate --precision`, and its stability measured between disjoint sample sets | done (2026-09-20); disjoint slices agree within 1.24× |
| 4 | First-order error propagation over PSK-Core, `presk check --precision` | done (2026-09-20); sound everywhere, vacuous on a transformer |
| 5 | Sensitivity by differentiation, `dump --stage sensitivity` | done (2026-09-20); half the folklore reproduced, half contradicted |
| 6 | The solver: propose by sensitivity, verify by **measurement**, tighten, report the gap; `presk solve … --out R.json`, run with `--realize @R.json` | done (2026-09-20); 2.67× less traffic on the corpus GPT, accepted first time |
| 7 | The vectorized GEMM under the contract | **started, not landed** (2026-09-20): the declared order reaches the lowering and the cache key; neither the interleaved contraction nor the vector lanes are emitted |
| 8 | The ε-to-quality curve, `make curve` | done for the ε axis (2026-09-20); the quality axis needs a training run per point and was not affordable |
| 9 | The four statements answered by measurement; the MLPerf foil written up against BERT-large | done (2026-09-20); **D holds, E and F fail, G is not askable** |
| 10 | Documentation, tutorial step 15, CHANGELOG, version `0.5.0`, tag | done (2026-09-20) |

### What 0.5.0 will be judged on

Four falsifiable statements, in the form 0.4.0 used, where two of three
failed and the version was better for having said in advance what would count
as failure. On the corpus GPT at course scale: the solved realization's
**measured** gap is inside a `1e-3` budget (D); it is at least 1.5× faster a
step than all-`f64` (E); the **propagated** bound exceeds the measured gap by
at most 10³ on at least half the conformance entries (F); and the vectorized
GEMM's measured gap falls inside the bound its declared reduction order
predicts (G). E is the statement most at risk, and step 7 is sequenced before
step 8 to give it a chance.

### 0.5.0 step 1, in detail

The step closed two language debts and found that one of them was mislabelled.

The axis `split` is in the checker, the IR, the lowering, differentiation,
the reference interpreter, the kernel host, the cost model and the PyTorch
export (`.unflatten`). It is a pure retag, so it launches no kernel; its
adjoint is `reshape_like`, checked against central finite differences at 512
bits in `tests/data/split_grad.psk` on a model that treats the two halves
differently, which a transposed or column-major reading would not survive.
The three backends agree to every digit and the generated PyTorch passes its
conformance test.

Per-layer randomness needed **nothing new**. A fold hands every iteration the
same extras, so the key is carried in the accumulator instead;
`tests/data/stack_keys.psk` measures that four independent masks at rate ½
zero about fifteen sixteenths of a tensor where one reused mask zeroes about
half. The index route was implemented far enough to establish that it could
not be differentiated (`E6001`), and the builtin written for it was removed
(D37).

Two findings worth carrying forward. Splits chain and merges do not, because
a split names its halves and a merge names nothing (D36) — which is what lets
`coverage/llama.psk` express grouped-query attention exactly, with neither
`split_heads` nor `merge_heads`. And rotary positions, which 0.4.0 attributed
to the missing split, are not blocked by it: the pairing checks, and what is
missing is `concat` — a way to build a tensor along an axis from computed
parts. The decoder family is one construct from exact, and it is not a shape
construct.


### 0.5.0 step 2, in detail

`presk_rt::Realize`, a flag with two values, became `presk_rt::Realization`,
a record: a storage and an accumulator type per site, a declared reduction
order, and the magnitude envelope any bound about it assumed. It travels
through `presk-kernel`, `presk-cranelift`, `presk-serve`, `presk-train` and
the CLI, which now takes `--realize f64|f32|bf16|@FILE`.

A site is a fusion group's root, `"<function>/%<let>"`. The kernel host
resolves the realization at each site as it walks a body, and a buffer that
crosses from a site of one type to a kernel of another is converted at the
kernel's edge — the one place a mixed assignment's conversion cost actually
lives. Measured on `mse`: `f64` gives 0.935893833554, uniform `bf16` gives
0.9375, and narrowing the single site `mse/%5` gives 0.937812805176, which
is neither.

`bf16` is storage only (D38), widened by a shift and narrowed by one round to
nearest even, written once in Rust and again as Cranelift integer
instructions. The two executors agree digit for digit on a `bf16` gradient,
which is what makes the conformance number a realization gap rather than a
difference between two programs.

**The check that mattered** was that nothing moved: every `f64` and `f32`
entry of the conformance table is what 0.4.0 measured, to the last digit, and
every golden is unchanged. The suite now reports three columns, and the
`bf16` one is new evidence rather than a regression: worst 3.3e-2, with an
absolute floor in the band because a result near a cancellation has unbounded
relative error at any precision.

### 0.5.0 step 3, in detail

`presk calibrate --precision` runs an entry on the reference interpreter and
records the largest `|x|` at each site, with the program, the entry, the
sample count, the precision and the day in the file. The recorder is a field
on the interpreter rather than a separate pass, so the magnitudes are those
of the exact denotation and not of a rounded run.

The step's real content is the check. `make precision` calibrates three
disjoint slices of the civil code on the corpus GPT and compares them: 101
sites, 77 compared, 24 identically zero in both, **none in only one**, worst
ratio **1.241×** between adjacent slices and **1.128×** between far-apart
ones. So the magnitudes are a property of the program within about a
quarter, not of the sample drawn — which is the assumption the whole error
analysis rests on, now measured rather than assumed (D35). The worst
disagreement is at the input of a normalization, the widest value in a
transformer block and the one most exposed to an unusual batch.

Two limits stated in the `precision` chapter: the measurement is at reduced
constants, because a course-scale run on the exact interpreter is not
affordable, and it is taken at a fresh initialization, so it says nothing yet
about how the envelope moves during training.

### 0.5.0 step 4, in detail

`presk check --precision ENVELOPE.json` propagates an entry's rounding error
under a realization and prints the bound; `--sites N` prints the widest `N`
sites, which is the per-value table the plan asked a separate dump stage for
— one surface rather than two, since `check` already loads the envelope.

The rules are in the `precision` chapter. Three of them carry the analysis: a
quotient propagates in *relative* terms, which is why a normaliser costs
nothing; a residual adds rather than multiplies; a softmax is 1-Lipschitz in
the ∞ norm. The reduction order is an argument, and a test asserts what that
means — blocking leaves the bound untouched, eight interleaved partials
change it.

**The result is that the analysis is sound and, on a real model, vacuous.**
On every conformance entry in every realization the bound exceeds the
measured gap, which a test asserts over 24 combinations. The median ratio is
1.8·10³ and ten of twenty-one are within 10³; on a normaliser it is 27. On
the corpus GPT it is 10²⁰, which is a relative bound of 10²⁰ on a loss of
4.5 and says nothing.

The cause is located: a contraction of `k` terms multiplies its input error
by `k`, worst-case correct, while committed errors grow like `√k`. Four
contractions in series carry `k⁴`. The tightening that closes it is each
sum's condition number measured into the envelope, named as future work
(D39). Two rules were wrong before the soundness test found them — a mean
was multiplying error by the batch size, and the absolute quotient rule was
pairing the largest numerator with the smallest divisor — which is what the
test is for.

### 0.5.0 step 5, in detail

`presk dump --stage sensitivity --entry L` prints `‖∂L/∂v‖₁` at every site
and summed per module. It costs one reverse sweep: differentiation already
computes every intermediate's adjoint and throws all but the parameters'
away, so the step only keeps the correspondence (D40).

Two results. **A ranking at initialization is vacuous** — a zero-initialized
output head stops every upstream gradient, so `--weights` is required for the
number to mean anything, exactly as the envelope needs a trained model to
describe one. And **half the mixed-precision folklore is reproduced**: the
output head dominates at 5.86 against 2.46 for the whole block stack, while
the token embedding table scores 0.28, below the down-projection that
practice narrows without hesitation. Keeping embeddings wide is therefore not
explained by the loss's sensitivity to them.

### 0.5.0 step 6, in detail

`presk solve` turns a budget into a realization: propose greedily by bytes
saved per unit of sensitivity-weighted error, verify, and while the
verification fails widen the worst site one step of the menu.

**It verifies by measurement rather than by the bound** (D41), because step 4
showed the bound is vacuous on a transformer and verifying against it would
accept everything. Measuring the candidate against the exact reference is the
stronger check, and one only this compiler can make. Two deviations from the
plan follow: the budget comes from `--budget` rather than the program's
`precision` block, whose plumbing through the typed configuration is not
done, and the result is written to a file that `--realize @FILE` runs, rather
than a `--realize solved` mode — the two compose and the file is what the
next steps want anyway.

Measured on the corpus GPT at `D = 64`, trained, budget 10⁻³: 2.67× less
traffic than all-`f64`, **zero** widening rounds, measured gap 1.05e-7
against a budget of 4.5e-3, and the solved realization runs natively. On a
program small enough for exhaustive search the greedy proposal is exactly
optimal at 10⁻³ and 1.057× off at 10⁻⁶.

One bug worth recording because it inverted the whole search: reading a
*missing* sensitivity as zero makes a site look maximally safe to narrow.
With it, the solver narrowed everything, measurement rejected the result, and
103 widening rounds recovered 1.06×. Treating the absence as unknown gives
2.67× accepted on the first measurement.

### 0.5.0 step 7, where it stands

Two things came out of starting it, and only one of them is code.

**The finding.** D31 refused vectorization because vectorizing a
contraction's inner sum reassociates it. That is true and it is not the only
way to vectorize a contraction. The blocked form of 0.4.0 already holds eight
accumulators, and they are eight *different output columns*, each still
summing `k` in increasing order. Putting those in vector lanes is bit-exact:
the lanes are independent columns and the inner sum is untouched. So the
largest single factor the kernels roadmap expects does **not** need a new
realization, which is a better position than the plan assumed (D42). The
contract remains necessary for the matrix coprocessor and the GPU, where a
reduction genuinely crosses lanes.

**The code.** `Stmt::SetL` assigns a loop variable in PSK-Kernel and in both
executors, which is what an unrolled index needs; on it the interleaved
contraction is emitted, and the declared order is part of the kernel cache
key. It reaches the kernel and reassociates, which a test asserts: the
embedding MLP at `D = 32` in `f32` gives 1.5494239330292 sequentially and
1.5494240522385 with eight partials, and the two executors agree with each
other to the last digit under each order.

`presk bench --native-only` skips the exact interpreter's row, which at 1024³
is what stopped a large benchmark from finishing at all. With it, on a 1024³
matmul over twelve cores: `f64` 102.4 ms, `f32` 87.3, `bf16` 96.9, and `f32`
with `Interleaved(8)` **239.0**.

**The reassociating order is 2.7× slower**, structurally: it cannot use the
blocked form, whose eight accumulators are eight output columns, so it falls
back to the naive per-element contraction and loses the cache behaviour that
was the whole gain of 0.4.0 step 6. Breaking the accumulator's dependency
chain buys far less than blocking does. So for a CPU contraction there is no
reason to reassociate at all — the exact form is both faster and exact, and
is also the shape vector lanes want. The order stays implemented, declared
and measured, because the coprocessor and the GPU will need it and a contract
one cannot exercise is not a contract; it is nothing's default.

What remains is the vector lanes, which need a vector form in the kernel IR.
Statement G as the plan worded it is about a vectorized kernel that does not
exist yet, and stays unanswered.

Statement G of the plan is therefore **not yet answerable**, and the version
cannot be closed until this step is.

### 0.5.0 step 8, in detail

`make curve` sweeps the budget on the corpus GPT: one program, one parameter
varying, which is the whole point — the sweep the specification says nobody
can do cleanly, because elsewhere each precision level is a different
pipeline and the comparison is between artefacts.

Every point is inside its budget by measurement against the exact reference,
and traffic rises monotonically as the budget tightens: 4.00× less at 10⁻¹,
2.67× at 10⁻³, 1.69× at 10⁻⁷, with a knee between 10⁻² and 10⁻³. The
`precision` chapter has the table.

Two things it does not hide. The measured gap is not monotone, because the
greedy proposal changes shape between budgets and the constraint is on being
inside rather than tight. And the propagated bound at these constants is
7.15e11, vacuous, which is why the measured columns are the result — exactly
as the plan said they would have to be.

**What is missing is the quality axis.** A validation loss per point needs a
training run per point, and that is the one experiment this version has not
been able to afford. The curve is therefore an ε-to-*traffic* curve with
measured error, not yet an ε-to-quality one.

## What 0.5.0 has, and what it does not

**Has.** A realization that is an object rather than a flag: per-site storage
and accumulator types, a declared reduction order, and the envelope any bound
about it assumed. A magnitude envelope measured on the exact interpreter,
stable to 1.24× across disjoint samples. A first-order error propagation that
is sound everywhere it can be checked. A sensitivity analysis costing one
reverse sweep. A solver that turns a declared budget into an assignment and
**verifies it by measurement against the exact denotation** — 1.05e-7 against
a budget of 4.52e-3 on the corpus GPT. And a one-parameter sweep of that
budget, every point verified, four times less traffic at 10⁻¹ and 1.69× at
10⁻⁷.

**Does not have.** Speed. The solved realization runs at 0.96× all-`f64`,
because the objective is traffic and these kernels are bound by arithmetic
(statement E). The vectorized kernel is deferred to 0.6.0: vectorizing across
output columns turns out to be bit-exact (D42), and emitting it needs a
vector form in PSK-Kernel that this version did not write. Statement G is
about that kernel and needs restating. The curve has no *quality* axis, which
needs a training run per point. The bound is vacuous on a network, and the
tightening is named but not built.

**The shape of the result is 0.4.0's.** There, the discipline was shown
affordable and the speed was not there. Here, precision is shown declarable
and verifiable and the speed is still not there. The next version's first
item is the same as this one's last: make a narrower type buy time, not only
bytes.