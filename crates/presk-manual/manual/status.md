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
| 4 | Crate `presk-realize` (named `presk-sched` until 0.6.0): fusion groups over PSK-Core bodies (elementwise chains of one result type whose interior values are read only inside, bounded expression duplication), mapped over the leaves of a parameter tree; one loop nest per group in PSK-Kernel; `presk dump --stage sched`; invariants tested on the corpus and the examples. `adamw.step` 414 → 111 launches and 1.54 → 1.32 ms, the forward 108 → 81 launches and no time change (it is contraction-bound), a whole training step 1515 → 884 (42 %, short of the plan's half: the forward and backward chains are cut by reductions); every golden byte-identical. Manual chapter `scheduling`; D30 | done (2026-09-20) |
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
  `make coverage`: the MLPerf transcriptions; `make realize` and `make
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

## Version 0.5.0

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
cannot be closed until this step is. *(0.5.0 closed with G not askable, step
9; 0.6.0 retired it for H, D43.)*

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

## Version 0.6.0

Make it pay, plan `presk-specs/implementation/presk-plan-0.6.0.md`, built
from 2026-09-20 to 2026-09-22. Two versions had ended with the same
sentence, and this one existed to change it:
0.4.0 showed the exactness discipline affordable and the speed not there,
0.5.0 showed precision declarable, solved and verified and the speed still
not there. A budget that buys nothing is a guarantee nobody will pay for.

0.5.0 had established which three things must become true, so the plan was
unusually determined: arithmetic width must buy throughput, which means
vector lanes and which D42 showed are **bit-exact** in the shape the blocked
contraction already has; the objective must be time and not traffic; and the
menu must be wide enough for the choice to matter, which means integer
quantisation.

| Step | Contents | Status |
|---|---|---|
| 0 | Version `0.6.0-dev`, statement G retired for H (D43), spec questions 57–59 | done (2026-09-20) |
| 1 | The vector form in PSK-Kernel: vector variables, contiguous load, broadcast, lane-wise arithmetic | done (2026-09-20); **bit-identical to scalar at 2 and 4 lanes** |
| 2 | Cranelift vector emission (packed panels **not** written) | done (2026-09-20); **H holds on both paths, I fails at 1.15–1.25×** |
| 3 | The machine profile fitted per dtype | done (2026-09-20); vector `f32` π is **2.5×** the `f64` π |
| 4 | Time as the solver's objective (conversion costs **not** in the model) | done (2026-09-20); **J fails at 1.06×**, but the sign turned from 0.96× |
| 5 | `int8` and `int4` as realizations with per-channel scales (D44): 5a the `quant` map, scales, packing, the dequantised path and `presk quantize`; 5b the integer kernels; 5c `int4` kernels; 5d the solver's menu and statement K | 5a done (2026-09-21): the gap splits into representation and arithmetic; `int8` 2.5·10⁻⁶ relative, `int4` 1.1·10⁻⁴ on a barely trained GPT. 5b done (2026-09-21): `int8` kernels on both executors, **bit-identical** to the dequantised path; 4.4× on a 16 MB contraction, 1.10× on the GPT at `D = 256`; `presk bench` no longer times the conversion of its inputs (D46). 5c done (2026-09-21): `int4` kernels, bit-identical as well, odd starts included; 5.2× on the 16 MB contraction, level with `int8`. 5d done (2026-09-21): `presk solve --quant`, the gap table, **statement K holds** (`int8` 3.2·10⁻⁶ against 10⁻²); a checkpoint of other dimensions is refused (D48) |
| 6 | The condition number, and the answer about the deterministic bound | done (2026-09-21): `Σ\|xᵢwᵢ\|`, `√Σ(xᵢwᵢ)²` and normwise factors recorded per sum; four rules (`presk check --precision --model`, `--measure`). Median bound/gap on small entries 1.8·10³ → **18**; on the trained corpus GPT 10³³ → **10¹⁰** in `f32`: **the condition number does not rescue the bound**. The root-sum-square estimate is within 3.7–10× on uniform realizations, 2.4·10³× on a mixed one. Recommended: question 56 (c), 58 yes to propose, not to discharge (D51) |
| 7 | The quality axis of the ε curve; J at course scale | done (2026-09-21): **the quality axis is flat** within 1.2·10⁻³ against a seed spread of 0.13; **J fails at 1.13×** (all-`f32` 1.87×) because the solved realization covers only the loss's sites (question 60); D48's re-measurements done |
| 7b | A budget on the training step (§3.7), and J again | done (2026-09-21): `presk calibrate --step` and `presk solve --step`; the budget binds **one step's update** against the exact step (question 60 answered: the update, not the gradient — AdamW multiplies an `f32` gradient error by 340); the solver chooses per function, measured alone; a **storage floor** `u·‖θ‖/‖θ′ − θ‖` no realization can go below (question 61). The recipe is 2.9·10⁻² from the exact step, all-`f32` 2.2·10⁻⁴. **J′ holds at 1.99×** at course scale, the solved step inside its 10⁻³ budget by measurement (D52) |
| 8 | The MLPerf comparison, with numbers | done (2026-09-22): the foil rests on what holds, each result stated as what it is (D53). **Both rules on one model** (`make foil`): MLPerf's accuracy rule admits all eight realizations of the corpus GPT on its held-out tenth, at 99.9 % as at 99 % (seven predictions of 118 016 at most), where a budget separates them over twelve orders of magnitude from two windows and attributes every quantised gap to the weights; a check takes 0.1 s, a pass over the held-out set 4 s, the exact reference 7.5 s once. **BERT-large runs** at MLPerf's constants, on its initialization (`make bert`): the exact reference of one SQuAD-shaped feature in 1 h 57 min; on its logits `int8` weights are 5.7·10⁻² from it (outside 10⁻²) and 2.2× faster than `f32` at batch 1 and at batch 20 — the first type narrower than `f32` to buy time on a model; at batch 1 a pass runs on one core; a dataset pass would take 9.3 h in `int8` and 20.8 h in `f32` per realization. The exact interpreter shares tuples and records (four blocks 40 → 13.5 GB, byte-identical). K restated: 1.46·10⁻⁴ on held-out windows. New entries `squad` (`coverage/bert.psk`) and `accuracy` (`05_gpt.psk`); question 62 |
| 9 | Documentation, tutorial step 16, version `0.6.0`, tag | done (2026-09-22): the manual brought to 0.6.0 as implemented — this chapter closed, `experiments` §6 and a new closing section, `cli`, `native`, `training`, `serve`, `optimization`, `precision`, `language`, `install`, `data`, the index; tutorial step 16, "A budget on the training step"; `make tutorial` runs every command of the chapter, steps 0 to 16, and every quoted output was regenerated with it; `presk --help` takes its version from the workspace; CHANGELOG 0.6.0; spec questions 57–62 copied to `presk-specs`; the plan's §5 verification, below. Tag `v0.6.0` by the user |

### What 0.6.0 was judged on

Four statements fixed before the version began, in the form the last three
versions used, and a fifth added at step 7b. The vector kernel is
**bit-identical** to the scalar blocked one (H, replacing 0.5.0's G, which
asked the wrong question). A 1024³ `f32` matmul reaches at least 3× the
scalar figure of 0.4.0 (I). The solved realization at a 10⁻³ budget is at
least 1.3× faster per training step than all-`f64` (J — **0.5.0's failing
E, restated**, and the statement this whole version existed to turn). The
same with the budget binding the training step's update, the solver free
over the whole step (J′, step 7b). And an `int8` realization stays inside a
10⁻² budget by measurement (K).

**The answers** (`experiments` chapter §6). **H holds** on both executors,
integer weights included. **I fails** — 1.15–1.25× from lanes, measured
before D46 took the conversion of the inputs out of `bench`; on falicon after
it, 1.1–1.2× in `f64` and nothing measurable in `f32` — and is retired: the
factor it asks for is the matrix coprocessor's. **J fails**: 1.06× at `D =
64`, and 1.13× at course scale in step 7, where all-`f32` is 1.87×, because
the solved realization narrowed only the loss's sites and the reverse sweep
and the optimizer stayed in `f64` (D50, question 60). **J′ holds**: 1.99× per
step at course scale, 2.02× re-measured at the end of the version, inside
the 10⁻³ budget by measurement (D52). **K holds** by a factor of three
thousand, `int8` weights at 3.2·10⁻⁶ of the exact loss on text — a margin
step 8 revised: on held-out windows of another checkpoint the same scheme is
1.46·10⁻⁴, inside 10⁻² by seventy, and on BERT-large's logits 5.7·10⁻²,
outside it (D53).

### Verification of 0.6.0 (plan §5)

Run on falicon (aarch64, twenty cores, 119 GB) on 2026-09-22, with the
binary of this step.

- `make test`: 119 tests pass before this step's edits and 120 after them
  (one new, for the envelope's date), and none fails; `make check-examples`
  gives only `W0001`; `cargo doc -p presk-manual` warns about nothing.
- `make conformance` (eleven PyTorch tests in ten exports), `make
  experiment` (the realization table of the `experiments` chapter's §2, again
  to its last digit) and `make tutorial` (every command of steps 0 to 16)
  pass, with PyTorch 2.14 in a `.venv` made with `pip`, since this machine has
  no `uv`.
- `make precision` reproduces its numbers: disjoint slices agree within
  1.241× and 1.128× as in 0.5.0, and every measured gap and bound/gap ratio
  of D51's table comes out again. §5's `presk solve … --budget 1e-3` is its
  last solve (the envelope is `out/gpt_env.json`, with `--weights`): 2.29×
  less traffic, 8.4·10⁻⁷ from the exact reference, no widening round.
- `make curve`: its seven solves and its eight training runs (all-`f64` and
  the seven budgets) redone from scratch, every realization and every
  checkpoint byte-identical to D50's, so the table is D50's to the digit
  except its bound column. That column printed `cond`, the rule's name:
  `curve.sh` read the second field of `presk check`'s bound line, whose
  format step 6 changed. It now takes the number, 2.58·10⁻¹³ in `f64` under
  the conditioned rule, where step 7's table had the `k` rule's 8.7·10¹⁹
  from an envelope made before step 6. The two seed runs D50 compares with
  are made by no target (they were trained by hand in step 7) and were not
  re-run.
- §5's `presk bench out/gemm.psk --entry gemm --native-only --repeat 5`,
  four runs: `f64` 78–85 ms and 68–75 in two lanes, `f32` 46–53 and 46–59 in
  four, `bf16` 40 — statement I's figures on this machine (`native`
  chapter).
- `make coverage` passes; `make quant-table` and `make step-table` give the
  tables of steps 5d and 7b byte for byte (K at 3.23·10⁻⁶), and `make foil`
  gives D53's to the digit except the wall times.
- **Too long, and what ran instead.** §5's `presk train corpus/nlp/05_gpt.psk
  --backend native --realize @out/solved.json` is one epoch at the course's
  constants, 288 steps of about 50 s: two steps of it ran (110 s), and `make
  j-step` measured the step solved at 10⁻³ again at course scale — **2.02×**
  (all-`f64` 61.1 s a step, solved 30.3 s, all-`f32` 30.8 s, the recipe 35.2
  s), the update gaps identical to D52's to four digits. `make bert` at 24
  blocks takes about five hours and 76 GB; it ran at two blocks instead
  (`L=2`, into `out/bert_L2.md`, leaving step 8's table alone): the exact
  reference in 10 min and 11 GB, `int8` weights 1.09·10⁻² from it — all of it
  the weights' rounding, the split coming out byte for byte as in step 8 —
  and 2.4× faster than `f32` at batch 1, 1.9× at batch 20. `make step-train`,
  two 600-step runs of about 35 min each, was not re-run: it is native
  training only, which step 8 did not touch, and the curve redoes ten runs of
  the same kind.
- 0.4.0's list: `run-`, `train-`, `serve-` and `opt-examples`, `serve-pt`,
  `calibrate`, `realize`, `bench`, `corpus` and `corpus-gpt` pass, and `presk
  serve` with every matrix of the tiny MiniGPT in `int8` gives the same greedy
  tokens on the reference and natively. **`make experiment-gpt` fails**: the
  exported `train.py` calls the objective without the dropout key the corpus
  GPT's `loss` has taken since 0.5.0 step 1 (`loss() missing 1 required
  positional argument`), so it has failed since then; the export also runs
  that dropout with `training=False`. The re-measurement of PyTorch that
  follows this version needs it fixed first.
- **Goldens.** The golden outputs of `tests/run` are byte-identical to
  v0.4.0's. The seven insta snapshots changed since then differ from it only
  in byte offsets, the examples' comments (translated to English on
  2026-09-22) and insta's metadata, which a script checked line by line; the
  test sources changed since v0.4.0 only follow renamed APIs, and no
  expected value moved. **`--realize f64` against 0.4.0**: v0.4.0 built from
  its tag, and thirty outputs compared between the two binaries — six
  entries run natively, 20, 200 and 100 training steps (logs and
  checkpoints) and greedy serving, each in `f64` and in `f32`; 12 steps of
  0.4.0's corpus GPT in `f64`; two runs of the reference interpreter — are
  byte-identical, all thirty.
- **Found and fixed.** The day an envelope records (`measured`) was a
  constant in the source, `2026-09-21`, whatever the day of measurement; it
  is now today's date in UTC, with a unit test of the calendar arithmetic.
  `make curve`'s bound column, above. And a quoted calibration in the
  `precision` chapter had one magnitude wrong since 0.5.0 (v0.5.0 prints the
  same number HEAD does); it is corrected.

## What 0.6.0 has, and what it does not

**Has.** Vector lanes in PSK-Kernel and in the Cranelift emission, across
the output columns of the blocked contraction and therefore bit-identical to
the scalar kernels (H), on by default. A machine profile fitted per machine
type and lane width, and a solver whose objective is the roofline's
predicted time instead of traffic. Integer weights: `int8` and `int4` per
channel, stored as integers and dequantised on load, with kernels
bit-identical to the dequantised path; `presk quantize`, which splits a
quantised gap into the weights' rounding and the arithmetic, a split only an
exact reference can make; `presk solve --quant`, and statement K. Each sum's
condition number measured into the envelope, four propagation rules, and the
answer to a question open since 0.5.0: the condition number does not rescue
the deterministic bound, so a budget is discharged by measurement and a
statistical model proposes (D51). The ε curve's quality axis, flat at this
scale. A budget in a training run with a meaning — one optimizer step's
update against the exact step — printed with its storage floor and the
sampling noise, a solver over the whole step, and statement J′: 1.99× over
all-`f64` at course scale, inside the budget. MLPerf's accuracy rule and a
budget measured on the same eight realizations of one model. BERT-large
inference run at MLPerf's own constants, exact denotation included, where
`int8` weights are 2.2× faster than `f32`. And an exact interpreter that
shares its parameter trees instead of cloning them, which is what let it
reach BERT-large at all.

**Does not have.** A realization both inside a tight budget and faster than
FP32: the solved training step is level with all-`f32`, and `int8` weights
on BERT-large's logits are inside 10⁻¹, not 10⁻². Packed panels, the matrix
coprocessor and the GPU (`docs/kernels-roadmap.md`), which is why I fails.
Activation quantisation (W8A8), group scales for `int4`, GPTQ or AWQ.
Conversion costs at kernel edges in the time model, and lanes chosen per
kernel: they are on everywhere, and on BERT-large at batch 20 scalar `f32`
code is 15 % faster. The program's `precision` clause read by any command:
the budget is `--budget`. An exact
reference for a course-scale training step (question 62). And a test of the research programme's central
hypothesis, RA4 — that a static cost model orders the schedule variants of a
kernel as well as autotuning does — which no version has run yet.

**The shape of the result.** 0.4.0 showed the exactness discipline
affordable and the speed not there; 0.5.0 showed precision declarable and
verifiable and the speed still not there. 0.6.0 is the first version to end
differently: a declared budget buys time — 1.99× on a training step inside
10⁻³, and 2.2× for `int8` weights on BERT-large inside 10⁻¹ — each measured
against the exact denotation, or against `f64` where the exact one cannot
reach. Against `f32`, which is
what practitioners and MLPerf compare with, the solved step is level. The
recommendation of 2026-09-22 in the plan names what follows: re-measure
PyTorch on this machine, then a 0.7.0 on RA4.

## Since 0.6.0

- **`make experiment-gpt` runs again** (2026-09-23, D54). The exported
  `train.py` calls a keyed objective with `None` for the key and the new
  `training=True` switch, so its `dropout` drops as `presk train`'s does,
  with masks PyTorch draws; the switch rides beside `generator` on every
  `uses Rand` function and through every call, and is off everywhere else
  (the expectation). The stacked loop over a tuple accumulator is exported
  correctly. Both sides print the median step time, the first step
  excluded, and take a thread count; the target runs the same steps,
  batches and threads on both sides, `f32` on both.
- **PyTorch re-measured on falicon** (2026-09-23, D55), which the 94× of the
  `experiments` chapter no longer stood for: one training step of the course
  GPT takes **29.96 s** in presk `f32` against **1.34 s** in PyTorch 2.14
  `float32`, twenty threads and dropout on both sides, three repetitions —
  a factor of **22**, of which 14 is the contraction: a 1024³ `f32` product
  runs at 710 GFLOP/s under PyTorch here (its own OpenBLAS 0.3.34 and
  oneDNN 3.12 over Arm Compute Library 53.2, within 4 % of each other) and
  at 49.6 under presk. Neither side moved for a reason inside the compiler:
  presk's matmul rate is where 0.4.0 left it and PyTorch's halved for want
  of a matrix coprocessor, which these cores do not have (`sve sve2 i8mm
  bf16`, no `sme`). The step solved at 10⁻³ is level with all-`f32`, as
  0.6.0 said. `experiments` §3 carries the tables.
- **PSK-Sched is PSK-Realize** (2026-09-23), as the language reference
  renamed it: crate `presk-realize`, `presk dump --stage realize`, `make
  realize` and the manual chapter `realize`, with `sched` kept as an alias
  of the stage and of the target. The IR name and the CLI only: the
  compiler's realization files still say "site" where the reference says
  *nest*, and the plans, decisions and past CHANGELOG entries are dated
  records and keep the old name.

## Version 0.7.0

Ranking, not speed, plan `presk-specs/implementation/presk-plan-0.7.0.md`,
built from 2026-09-23 to 2026-09-24.
Three versions have measured precision and speed and none has measured the
research programme's central hypothesis, RA4: that on regular tensor
kernels a static cost model orders more than 90 % of the pairs of variants
autotuning distinguishes, at a decision cost lower by orders of magnitude.
0.7.0 tests it on the CPU backend, with attention as the case study: a
layout space per nest whose every point is exact by construction, measured
exhaustively, against a cost model that reads the layout. It does not
promise to close the 22× against PyTorch.

| Step | Contents | Status |
|---|---|---|
| 0 | The layout decisions, their domains, the plan object and well-formedness conditions 7–9 in the language reference (draft v0.2); the defaults checked against 0.6.0's kernels (D56: the `ein`s are the tile `(1, 1)`, and the row-wise `dot` rounds its partial sums to `bf16` on a `bf16` nest); version `0.7.0-dev`, spec questions 63–69, the X925 cores recorded | done (2026-09-23) |
| 1 | `Layout`, `Plan` and `--plan`: a plan file (a realization, the profile it was chosen against, a layout per nest) on `run`, `bench`, `train` and `dump`; the plan checked against the program before a run and each nest's layout at its launch, the reference's conditions 3, 4, 7 and 9 included (`E7002`); the lanes lowered as a plan states them, every other decision refused unless it is 0.6.0's; the layout in the kernel cache's key; `dump --stage realize` lists the plan; `make plan-check`, and the kernels of ten entries byte-identical to 0.6.0's in three types | done (2026-09-23) |
| 2 | One contraction builder for `dot` and `ein`: the contraction nest recognized once for both, and lowered in the form, register tile, k-block, loop order and lanes its layout states (the pack waits for step 3); every `f64`/`f32` `dot` kernel text-identical to 0.6.0's, the `ein`s less an exact multiplication by one; the `bf16` odd-width `dot` accumulating in `f32` at last (D56); lanes on an NT nest need the pack, a condition added to the reference; every point of three nests, 1 142 in all, bit-identical to the default on both executors (D58) | done (2026-09-23) |
| 3 | Pack, two-pass softmax, the split at depth d: the pack as two launches, a copy to an NN panel then the contraction; softmax in two passes; `Kernel::parallel_at(depth)` after the reference's definition, the Cranelift kernel ranged at that depth with its enclosing loops in full, the threshold bypassed for a stated split, conditions 5 and 8 checked against the machine and the kernel; every decision of the reference now lowered; every point of four nests, the pack included, bit-identical, and 1 618 split points too, 200 refused by condition 8; one worker against many at depths 0–2 and at batch one (D59) | done (2026-09-23) |
| 4 | The cost model and its calibration: the machine description read from the cores the process runs on; one worker's work and traffic from L2, L3 and memory by footprint, and its register pressure (`planned_cost`); `λ`, spawn, `β` per level and `π` per type, lanes and workers fitted by coordinate descent on a GEMM at three sizes in two types, pinned (`bench --space calib`, `make calibrate-ra4`); u = 0.185 over 576 samples, Spearman 0.980 (D60) | done (2026-09-23) |
| 5 | `presk plan`, `presk autotune`, `presk rank`: the entry run once natively, the first launch of every nest at every shape captured with its buffers, each nest lowered again under every point of its space (splits included, condition 8 checked); `plan` costs the points and writes a plan, `autotune` times each on the captured buffers and checks it bit-identical to point 0 natively and on the kernel interpreter, a JSON line per point with the model's terms, and `rank` applies §3.6's rule to two days; `experiments/ra4.sh`, pinned; `make ra4-small` in 55 s, every point exact. At §3.4's shapes the space is 3 305 points a type at B = 1 and 4 857 at B = 16; the scores nest's full sweep takes 18 s and 348 s, 57 and 360 times its static choice, where N asks 10³ (D61) | done (2026-09-23) |
| 6 | The measurement, and the numbers: one night on falicon, two pinned sweeps of the course GPT's attention (B = 1 and 16, f32 and f64, 16 324 points each), the GEMM at 1024³ and a BERT-shaped sample; a 20-core sweep; statement L checked apart on the kernel interpreter in 130 shards. **L holds** (17 879 points, none differing), **M fails** (47.9 % and 43.9 % of the attention `ein`s' distinguished pairs; 37–46 % of them tied), **N fails** (582–717× at B = 16, 75–89× at B = 1), **O fails** (top-1 regret 2.18–3.01), **P holds** (5.49×, 460 GFLOP/s). The model reads lanes and threads (93–97 % right) and cannot see a register tile: a load weighs nothing and traffic starts at L2, so tiles, orders and k-blocks tie. Its candidates within `1 + u` always hold the best point, at 2–15 % of the space (D62) | done (2026-09-24) |
| 7 | Documentation, tutorial step 17, version `0.7.0`, tag | done (2026-09-24): the manual brought to 0.7.0 as implemented — `realize` with the plan file and a table of the layout decisions, `native` with a packed and split kernel and the GEMM's best point, `optimization` with what `u` is for and what the layout model cannot see, `cli` checked against `presk --help`, `experiments` §7, this chapter closed; tutorial step 17, "A plan for attention" (`plan`, `run --plan`, `autotune --space small`, `rank` on the tutorial's attention), in `make tutorial`, whose quoted outputs were checked against a run (step 16's `bf16` rows moved with D58); the register-load term recorded as a design in `docs/kernels-roadmap.md`; CHANGELOG 0.7.0; spec questions 63–70 copied to `presk-specs`; version `0.7.0`. Tag `v0.7.0` by the user |

### What 0.7.0 was judged on

Five statements. Every point of the layout space, on every nest of the
attention entry and of the GEMM, is **bit-identical** to the scalar
sequential unsplit kernel, natively and on the kernel interpreter (L). On
each of the two attention `ein`s at batch 16 in `f32`, the static model
ranks correctly more than 90 % of the close pairs the measurement
distinguishes, its constants fitted on the GEMM alone (M, the thesis).
Choosing the plan statically costs at least 10³ times less wall time than
the exhaustive sweep (N). The model's pick is within 1.25× of the best
measured point on every contraction nest (O). And the best measured point
of the GEMM space at 1024³ `f32` is at least 3× the 0.6.0 default (P).
Step 6 decided them (D62, `docs/ra4-2026-09-24.md`): L and P hold; M, N
and O fail. The static model does not rank the attention's layouts as the
hypothesis says, and the reason is named: it has no term for the loads a
register tile saves.

**The answers** (`experiments` chapter §7). **L holds**: 17 879 points of the
course GPT's attention, the GEMM and a BERT-shaped sample, none differing from
the scalar sequential kernel natively or on the kernel interpreter. **M
fails**: the scores and values `ein`s at batch 16 order 47.9 % and 43.9 % of
their distinguished close pairs right, 37–46 % of them ties; lanes and splits
are 93–97 % right, tiles 28 %, orders 30 %, k-blocks 1 %. **N fails**: the
static choice is 582–717 times cheaper than the sweep at batch 16 and 75–89
at batch 1. **O fails**: the model's pick is 2.18–3.01 times slower than the
best point. **P holds**: the best GEMM point is 5.49× 0.6.0's, 460 GFLOP/s on
ten cores. The model's candidates within `1 + u` hold the best point or one
within 1.08× of it, at 2.4–15 % of the space. The user decided on 2026-09-24
that the version closes on these results, and that the load term the failure
names goes to a later version, tested on a fresh sweep
(`docs/kernels-roadmap.md`).

### Verification of 0.7.0

Run on falicon (aarch64, twenty cores) on 2026-09-24, with the binary of
this step.

- `make test` (140 tests pass), `make check-examples` (only `W0001`) and
  `cargo doc -p presk-manual` (no warning).
- `make plan-check`: a plan stating the default layout gives the same kernels,
  byte for byte, on the ten entries of `tests/plan/entries.txt`.
- `make ra4-small`: the daily subset on the tutorial's attention, `B` = 1 and
  16 in `f32` and `f64`, two runs, 312 points a run, every one bit-identical
  natively and, on the first run, on the kernel interpreter; in 50 s. Its
  report left M, N and O "not measured", because `experiments/ra4.sh` ranked
  with `presk rank`'s default entry, `SelfAttention.apply`, where the small
  sweep is on `attend`; the script now passes its entry, and the report
  decides them at that size (M inconclusive below 10³ pairs, N fails at
  11–25×, O holds at 1.11).
- `make tutorial`: every command of steps 0 to 17 passes, twice, with PyTorch
  2.14 in a `.venv` made with `pip`. Every quoted output was compared with
  the first run. They agree but for timings and what depends on them, and
  for step 16's `bf16` and recipe rows, which moved with D58's `bf16` `dot`
  (the recipe's update gap 8.49·10⁻³ → 8.66·10⁻³); every choice of the
  solver, and the final distance to the `f64` run, are unchanged. The second
  run gave step 17 the same plan and the same verdicts.
- `make conformance`: eleven PyTorch tests in ten exports pass.
- Plan §5's `presk plan` on the course GPT's attention at `B = 16`, `T = 256`
  with the pinned profile of step 4 costs 4 857 points in 1.5 s, and `presk
  run --plan` with the plan it writes prints the default's output, byte for
  byte (26 MB).
- The measurement itself, `make calibrate-ra4`, `make ra4`, `ra4-check`,
  `ra4-20` and `ra4-day2`, is step 6's (D62), and was not run again. Plan §5's
  `make gemm-space` is part of `make ra4`, which sweeps the GEMM. Every golden
  of `tests/run` is checked by `make test`; `--realize f64` against 0.4.0's
  binary was not compared again in this version, whose default `f64` and
  `f32` kernels are 0.6.0's, the `ein`s less an exact multiplication by one
  (D57, D58).

## What 0.7.0 has, and what it does not

**Has.** A layout space per loop nest, the language reference's (§6.3):
the contraction's form, register tile, k-block, loop order and lanes, a pack
of an NT operand, a thread split at any depth up to 2, and a softmax in two
passes, each with its domain and its well-formedness conditions, and every
point exact by construction — asserted over every point of four nests in the
tests and 17 879 in the measurement. One contraction builder for `dot` and
attention's `ein`s, whose default is 0.6.0's kernel, and a `bf16` `dot` that
no longer keeps its partial sums in `bf16`. A realization plan beside the
realization (`--plan @P.json` on `run`, `bench`, `train` and `dump`), refused
with `E7002` where it is ill-formed, never ignored. A cost model that reads a
layout — one worker's work, traffic by cache level from footprints, register
pressure, an uncertainty `u` — with the machine read from the cores the
process runs on, calibrated on a GEMM (`bench --calibrate --space calib`).
`presk plan`, `presk autotune` and `presk rank`: a static choice, an
exhaustive measurement on captured inputs with bit-for-bit checks, and a
report that decides the statements. And the first test of RA4, with its
answer.

**Does not have.** A static model that ranks layouts as RA4 asks: it cannot
see a register tile, a loop order or a k-block, because it counts no load
from L1 into registers (`optimization` chapter), and the term that would is a
design, not code (`docs/kernels-roadmap.md`). A chooser that prunes
statically and measures the candidates, which the data support. Packing both
operands, as BLAS libraries do, the matrix coprocessor and the GPU, with
memory placement, the third block of the reference's table; the pack is two
launches meanwhile (question 66). A plan for a training step: the derivatives
of `dot` are not contraction nests. A solver over types and layouts at once.
The comparison with MDH and ATF on the shared space. And a second day of
measurement: day two ran four hours after day one, in the same night.

**The shape of the result.** 0.4.0 to 0.6.0 measured precision and speed;
0.7.0 measured the thesis. The part of it about exactness holds at a scale
the earlier versions never tried: a space of variants, every one bit-identical,
whose best points are 2.6 to 77 times faster than what 0.6.0 emitted. The part
about a static model does not hold as stated: the model is right on what it
reads and blind to what it does not, and it is a good pruner rather than a
chooser. The next version knows which term to add, and that it must be tested
on a sweep it was not designed on.

## Version 0.8.0

RA4 re-tested, plan `presk-specs/implementation/presk-plan-0.8.0.md`, built
from 2026-09-24 to 2026-09-26.
0.7.0's static model could not see a register tile, a loop order or a
k-block: a load weighed nothing, and its traffic terms never bound under a
maximum. 0.8.0 changes the whole model — a load term, a combination of the
terms other than the maximum, a calibration that identifies the traffic
constants — designed on 0.7.0's measurements alone, which are its
development set and decide nothing. The model was then frozen in a tagged
commit, `ra4-model-0.8.0`, before any new measurement, and tested on two
fresh sweeps: new shapes on the ten X925 cores, and 0.7.0's shapes on the
ten A725 cores. The GPU and the matrix unit come later; the version does not
promise to close the 22× against PyTorch.

| Step | Contents | Status |
|---|---|---|
| 0 | 0.7.0's raw measurements, profiles and reports archived read-only outside the repository with a checksum list, the development set (`presk rank` regenerates both reports from it byte for byte); the roadmap's request for "a footprint analysis that starts at L1" corrected, since the model counts the bytes from L2 into L1 as `Q_L2` and the maximum is what hides them; version `0.8.0-dev`, spec questions 71–72, the A725 cores recorded as `RA4_CORES_SMALL` (D63) | done (2026-09-24) |
| 1 | `presk reprice`: a recorded day priced again under another profile, every point lowered again from its layout, its times and other fields copied as written; `make ra4-dev` re-prices 0.7.0's measurements with 0.7.0's profiles and gives back all 52 142 lines and both reports byte for byte (step 0's check). The static side 2.1–4.1× cheaper, every prediction unchanged: a layout lowered once for all its splits, the traffic's footprints shared between the cache levels (D64), and the launches captured from the entry's shapes, on zero inputs without running a kernel, which statement N's static side now is, the entry-run ratio printed beside it (D65, decided before any fresh measurement); 1.51 → 0.54 s at batch 16 in `f32`, 1.76 → 0.45 s at BERT-large's attention. `presk rank --frozen TAG`, and `-dirty` days | done (2026-09-24) |
| 2 | The load term: one worker's register loads and stores (`kernel_loads`), the spill moved from work to loads, `(N_ld + S)/ρ_g` in the maximum with `ρ` fitted per group, once for the machine, or not at all (`bench --calibrate --space calib --load-term`); a profile without `ρ` predicts as 0.7.0's did, to the last bit; the count checked on the six tiles of plan §1, a pack and a spilling `[4, 16]`, `ρ` recovered within 10 % from a synthetic fit; on the development set the count's Spearman with the measured times is 0.953 where 0.7.0's prediction reached 0.437 (D66) | done (2026-09-24) |
| 3 | The combination (`--combine max\|sum\|pnorm`, the p-norm's `q` searched on its own, each form recovered from synthetic samples), `presk refit` and `--compare`, the calibration space of plan §3.3 (300 layouts a shape, the GEMM at 256–2048 and two thin products, `experiments/matmul.psk`); measured on the X925 cores: 3 536 samples, the load term takes `u` from 0.226 to 0.133, the three combinations fit alike and the p-norm settles at `q` = 1; not met: `β_mem` above `β_L3` (the thin products are not memory-bound here), one-lane `ρ` unidentified, and 3 h 13 min where an hour was asked, most of it an input conversion now removed (D67); decided by the user: the thin products kept and `β_mem` recorded as not identifiable here (a 2-row product is not memory-bound either), the one-worker layouts left out at 2048 | done (2026-09-24) |
| 4 | Development on 0.7.0's measurements, and the freeze (three weeks at most). Done so far: plan §3.3 amended on the user's decisions; `bench --min-workers`, the one-worker layouts left out at 2048, and a calibration that starts from an empty profile; the definitive calibration on the X925 cores, 3 336 samples in 44.9 min, `u` 0.142 with the load term (0.241 without), the three combinations alike, `β_mem` above `β_L3` as expected, archived read-only (D68); the ablation on the development set (`make ra4-ablation`): statement M's nests go from 47.9 % and 43.9 % right under 0.7.0's model to 84.6 % and 84.4 % under a load rate per group and the p-norm, the form chosen, with O at 1.07, 1.00 and 1.01 (D69); the calibration re-costed from its layouts, its times kept (`presk refit --recost`), which gives the definitive calibration back byte for byte; three rules of the planned cost tried on it and on the development set, cache lines and loop overhead kept (M's nests 89.3 % and 89.6 %), buffers sharing a level not (D70); the nine forms again under those rules, the chosen form still leading; the frozen model's profile archived read-only, its development report in `docs/ra4-dev-model-0.8.0.md`, checked by `make ra4-dev-frozen`, statement D (D71); the tag `ra4-model-0.8.0` on the commit that holds all of it, from which the freeze holds | done (2026-09-25) |
| 5 | F2: 0.7.0's shapes on the A725 cores, two days. Done so far: the targets, `make calibrate-ra4-small` (the frozen form and space, a profile of the A725's own), `ra4-f2-dry`, `ra4-f2`, `ra4-f2-check`, `ra4-f2-day2` and `ra4-f2-rank` (`presk rank --frozen`); checks before a fresh day, which refuse an uncommitted tree, a commit not descending from the tag, day two on day one's calendar date and a running `presk`; a dry run that sizes a day from one nest's sweep and the model's predictions, 5 % short on 0.7.0's day one (D72); `presk rank` leaves M inconclusive on a configuration swept on a sample (§3.5); all of it tried on a tiny configuration. The A725 calibrated in the frozen form (u 0.168, the X925's 0.132), archived read-only; F2 measured twice, on one date by the user's decision, every point bit-identical natively and on the kernel interpreter, archived read-only; ranked by `presk rank --frozen`: M's nests 92.7 % and 91.7 %, O at most 1.01, N 2 168–2 734× at B = 16 but 218–252× at B = 1 (D72; the statements are decided in step 7) | done (2026-09-25) |
| 6 | F1: new shapes on the X925 cores, two days. The targets `ra4-f1-dry`, `ra4-f1`, `ra4-f1-check`, `ra4-f1-day2` and `ra4-f1-rank`, priced by the frozen model's profile itself; two configurations in one day, BERT-large's attention (F1a) and a head of 128 (F1b), in `f32` over the whole space; a dry run per configuration and a day that keeps a configuration whole while it fits in 8 hours; two ranks of the same days, M and O at F1a into `out/ra4-f1a.md` and at F1b into `out/ra4-f1b.md`; all of it tried on a tiny configuration (D73). The dry run estimated 52 min a day, both configurations whole; F1 measured twice on one date, 48 min a day, every point bit-identical natively and on the kernel interpreter, archived read-only; ranked by `presk rank --frozen`: M's nests 87.4 % and 88.5 % at F1a, 85.4 % and 88.8 % at F1b, below 90 %, the misses within u; O at most 1.05; N 1 013–1 016× at F1a, 2 880–2 933× at F1b (D73; the statements are decided in step 7) | done (2026-09-25) |
| 7 | The statements, on the archived days under the frozen model. `presk rank` decides Q: the candidates within `1 + u` of the model's best, their best within 1.05× of the best measured, and their cost counted as question 72 asks, the static side and the candidates' measured time over the nest's exhaustive sweep, D71's share of the kernel time beside it (`--statement-q` for 0.7.0's days); the reports `docs/ra4-f2-2026-09-25.md`, `docs/ra4-f1a-2026-09-25.md` and `docs/ra4-f1b-2026-09-25.md`. **L holds**: every point of both sweeps bit-identical, natively and on the kernel interpreter. **M fails**: 92.7 % and 91.7 % on F2, but 87.4 % and 88.5 % at F1a, 85.4 % and 88.8 % at F1b; 97–98 % of the misses lie within u, and the one decided failure is the orders of one-row tiles at F1b's scores `ein`, where the L2 term is added to the loads and the machine does not pay it. **N fails**: 218–252× on F2 at B = 1, where costing the points is most of the static side; 1 013× and more everywhere else. **O holds**: at most 1.05. **Q holds**: on all 18 contraction nests the candidates hold a point within 1.03× of the best, at most 1.8 % of the sweep. D, reported: M 89.3 % and 89.6 %, Q on 12 of 13 nests. The freeze ends (D74) | done (2026-09-26) |
| 8 | Documentation, version `0.8.0`, tag | done (2026-09-26): the manual brought to 0.8.0 as implemented — `optimization` with the A725's calibration and what the fresh sweeps found, `cli` checked against `presk --help` and each command's help (the plan's pick under both models, the calibration example in the frozen form, the commands that ranked the fresh sweeps), `experiments` §8 with the statements of 0.8.0 and their verdicts, `realize` and `native` with the frozen model's pick, this chapter closed; the frozen model's form made the default of `bench --calibrate --space calib` and `presk refit` (D75); tutorial step 17 re-run under those defaults, its quoted outputs refreshed, and its `presk --help` quote, stale since 0.7.0; what D74 recorded for after 0.8.0 in `docs/kernels-roadmap.md`, beside D71's; CHANGELOG 0.8.0; spec questions 71–72 copied to `presk-specs`; version `0.8.0`. Tag `v0.8.0` by the user |

### What 0.8.0 was judged on

Five statements, on measurements the frozen model never saw. Every point of
both fresh sweeps is **bit-identical** to the scalar sequential unsplit
kernel, natively and on the kernel interpreter (L). On each attention `ein`
of BERT-large's attention (`D = 1024`, `H = 16`, `T = 384`, `B = 1`), of a
head dimension of 128 (`D = 1024`, `H = 8`, `T = 512`, `B = 2`) and of the
course GPT's attention at batch 16 on the A725 cores, all in `f32`, the model
ranks correctly more than 90 % of the close pairs the measurement
distinguishes, ties counted as not right (M, the thesis). Choosing statically
costs at least 10³ times less wall time than the exhaustive sweep, on every
fresh configuration and day (N). The model's pick is within 1.25× of the
best measured point on every contraction nest of those configurations (O).
And measuring only the points the model predicts within `1 + u` of its best
finds a point within 1.05× of the best measured, at no more than 10 % of the
sweep's time (Q, not RA4's claim: the hybrid chooser 0.7.0's data suggested).
The same metrics on the development set are reported beside them and decide
nothing (D). 0.7.0's statement P is retired: it was about the space, which
0.8.0 does not change. Step 7 decided them (D74; `docs/ra4-f2-2026-09-25.md`,
`docs/ra4-f1a-2026-09-25.md`, `docs/ra4-f1b-2026-09-25.md`): L, O and Q
hold; M and N fail.

**The answers** (`experiments` chapter §8). **L holds**: every point of both
fresh sweeps, 25 588 a day, bit-identical to the scalar sequential kernel,
natively and on the kernel interpreter. **M fails**: on the A725 cores, at
0.7.0's shapes, the two attention `ein`s rank 92.7 % and 91.7 % of their
distinguished close pairs right; at BERT-large's attention on the X925
cores, 87.4 % and 88.5 %; at a head of 128, 85.4 % and 88.8 %. The model
carried over to another core and not to new shapes. Ties fell from 34 % to
0.1 %, and the pairs it decides are about 99 % right: 97–99 % of the misses are
pairs it predicts within `1 + u`. The one decided failure is the loop order
of one-row tiles at the head of 128, where the p-norm adds the L2 term to
the loads and the machine does not pay it. **N fails**: 1 013–2 933 times
cheaper than the sweep everywhere but at batch one on the A725 cores,
218–252 times, where costing 3 305 points is most of the static side.
**O holds**: the model's pick is within 1.05× of the best point on every
contraction nest M names. **Q holds**: on all eighteen contraction nests,
measuring only the model's candidates finds a point within 1.03× of the
best, at most 1.8 % of the sweep's time. **D**, reported: 89.3 % and 89.6 %
on the development set, which foresaw F1 within four points.

### Verification of 0.8.0

Run on falicon (aarch64, twenty cores) on 2026-09-26, with the binary of
this step. Step 8 measures nothing new and changes nothing the freeze
covered.

- `make test` (153 tests pass), `make check-examples` (only `W0001`) and
  `cargo doc -p presk-manual` (no warning).
- `make plan-check`: a plan stating the default layout gives the same
  kernels, byte for byte, on the ten entries of `tests/plan/entries.txt`.
- `make ra4-dev`: 0.7.0's days re-priced under 0.7.0's profiles give back
  all 52 142 lines and both of 0.7.0's reports, byte for byte. `make
  ra4-dev-frozen`: the development set under the frozen profile gives back
  `docs/ra4-dev-model-0.8.0.md`, byte for byte.
- `make tutorial`: every command of steps 0 to 17 passes, three times,
  with PyTorch 2.14 in a `.venv` made with `pip`. Every quoted output was
  compared with the first run. They agree but for timings and what depends
  on them, and for `presk --help`, which now lists `reprice` and `refit`,
  and step 17, run again with the layout model calibrated in the frozen
  form. The first two runs named that form by flags. The third ran under the
  new defaults (D75), with no flag, and calibrated in the same form. All
  three gave step 17 the same tiles, packs and splits and the same verdicts;
  the second chose the other loop order on both `ein`s, a choice within
  `u`.
- D75's check: the definitive calibration, re-costed under `lines,overhead`
  and refitted with the new defaults, gives the frozen profile back byte for
  byte, and so does the frozen profile re-costed under its own rules.
- `make conformance`: eleven PyTorch tests in ten exports pass.
- The measurements themselves, `make calibrate-ra4` (step 4, D68), `make
  calibrate-ra4-small`, the dry runs, days and checks of F2 and F1 (steps 5
  and 6, D72, D73), and their ranking (step 7, D74), were not run again.
- Plan §5's `make ra4-small`, priced by the frozen profile (`PROFILE=` the
  archived one): the tutorial's attention at `B` = 1 and 16 in `f32` and
  `f64`, two runs, 332 points of day one checked on the interpreter, every
  one bit-identical, in 50 s. O holds at 1.07; M is inconclusive below 10³
  pairs (92.1 % of 165 on the scores); N and Q fail at this size, where
  choosing costs more than sweeping a four-point nest.

## What 0.8.0 has, and what it does not

**Has.** A layout cost model that reads a register tile: one worker's
register loads and stores against a rate per group, beside its work and its
traffic from each cache level, combined by a p-norm whose exponent is
fitted, with footprints in whole cache lines and the loops' own overhead in
the work (`presk bench --calibrate --space calib`, `--load-term`,
`--combine`, `--rules`). A calibration on a GEMM widened to five tiles,
sizes to 2048 and two thin products, run as one measurement per machine
(`make calibrate-ra4`, `make calibrate-ra4-small`). Tools that judge a
model without measuring: `presk reprice` prices a recorded day again under
another profile, `presk refit` fits a profile in another form or costs its
samples again under other rules, and `make ra4-dev`, `ra4-dev-frozen` and
`ra4-ablation` put a candidate against 0.7.0's measurements in minutes. A
static side that runs no kernel: `presk plan` captures the entry's launches
from its shapes. A discipline for testing a model: development data apart,
a freeze recorded by a tag, `presk rank --frozen`, fresh sweeps refused on
an uncommitted tree or a commit that does not descend from the tag, every
day and check archived read-only with checksums. `presk rank` deciding the
hybrid chooser's statement Q. And the second test of RA4, on two sweeps the
model never saw, with its answer.

**Does not have.** A static model that ranks more than 90 % of the close
pairs on shapes it was not designed on: it orders 85–89 % there, and the
pairs it misses are those it predicts within its own uncertainty. A term for
how an iteration's loads fall across cache lines, which is what the pack
buys; a combination of the L2 and load terms that fits a one-row tile; a
static side cheap enough for 10³ at batch one on the small cores; and an
identified memory bandwidth, which no GEMM this compiler emits reaches on
falicon (`docs/kernels-roadmap.md`). A chooser that measures the model's
candidates as a command: `presk autotune --top N` measures a fixed number,
and statement Q was computed by `presk rank` on exhaustive sweeps. Packing
both operands, the matrix coprocessor and the GPU, memory placement, a plan
for a training step, a solver over types and layouts at once, and the
comparison with MDH and ATF, as in 0.7.0.

**The shape of the result.** 0.7.0 measured the thesis and failed it for a
named cause. 0.8.0 added what the cause named, froze the model, and measured
again on data it had never seen. The static model now ranks what 0.7.0's
tied, carries over to another core above RA4's threshold, and falls a few
points short of it on new shapes; its pick is within 5 % of the best point,
and pruning by it and measuring the rest finds a point within 3 % of the
best at under 2 % of a sweep's cost. What RA4 asks, a static model that
decides alone above 90 %, is not shown; what the data support is a
static model as the pruner of a short measurement.

## Version 0.9.0

A C export, plan `presk-specs/implementation/presk-plan-0.9.0.md`, built
from 2026-09-26 to 2026-09-27. `presk export --target c` writes one C
program per exported entry, in the style of llama2.c and StrasGPT. Its
kernels come from PSK-Kernel, with the fusion, the realization and the
layout plan already applied. Its host comes from a recording of the native
backend's own run at the entry's dimensions, structured as a C function per
Presk function and a loop per layer fold. The program runs without the
compiler and computes the bits the native backend computes. Any entry the
native backend runs can be exported, a gradient, one optimizer step and a
generation function with `presk serve`'s loop included, under a plan, on
OpenMP threads, with integer weights; the training loop cannot. A SCoP
mode, `--mode scop`, writes each nest at point zero for polyhedral tools.
The GPU and the matrix unit come after it. The `export_c` chapter describes
the export, tutorial step 18 runs it; until step 3 `presk export --target
c` was refused (`E8002`).

| Step | Contents | Status |
|---|---|---|
| 0 | The user's five decisions recorded: the C export as 0.9.0 before the GPU, the crate `presk-export-c` behind `presk export --target c`, any entry the native backend runs but not the training loop, bit identity with the native backend with musl's functions shipped, the host by recording; steps 7 and 8 kept. `--target c` parsed and refused (`E8002`, exit code 2, nothing written). The plan checked against the code and amended: the course GPT, whose `Rand` is dropout only, is exported from step 3; `loss` is a forward pass. A gradient through attention added to the test set, `loss_grad` of `tests/plan/minigpt_grad.psk` (MiniGPT with its head drawn, since the example's zero head leaves every other gradient zero), and C2 judged against the native backend's own gap, which crosses the band on 34–121 of its 8 768 elements where terms cancel. Version `0.9.0-dev`, spec questions 73–75, 75 on an output that cancels (D76) | done (2026-09-26) |
| 1 | The kernel printer, crate `presk-export-c`: a PSK-Kernel kernel as one C function with its text as its comment, literal bounds, offsets multiplied out, lanes as loops over a local array, `bf16` as `uint16_t` rounded as `presk_rt` rounds, a split's `[lo, hi)` on the loop at its depth, `presk_exp`, … declared for step 2; integer weights refused until step 7. Checked by running the C inside the kernel backend, through an executor over the compiled C loaded with `dlopen`: compiled with `-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror`, bit-identical to the native backend on the conformance suite's 41 primitives in `f64`, `f32` and `bf16`, scalar and with lanes (738 runs), on the 1 491 layouts and 1 618 thread splits of the four nests of attention, and on hand-written kernels. Every range is cut three ways, and no two pieces write the same element. The transcendentals call the `libm` crate's functions through stand-ins. In `make test`, 11 s. `--target c` is still refused. Two facts of the plan corrected: GCC ignores `#pragma STDC FP_CONTRACT OFF` and warns about it, and `-std=c11` alone already stops GCC from contracting (D77) | done (2026-09-26) |
| 2 | The runtime and the transcendental functions, in `presk-export-c`. `c/presk_libm.c` holds `exp`, `log`, `tanh`, `sin`, `cos`, `erf` and their `f` forms from musl 1.1.22, the last release before musl replaced `exp`, `expf`, `log` and `logf` with ARM's routines, which the `libm` crate of the native backend does not port and which differ from it on 8–10 % of inputs; renamed `presk_…`, their helpers static, no operation changed. Bit-identical to the `libm` crate on 10⁶ inputs per function, special values, subnormals, branch thresholds and range-reduction boundaries included (0.4 s in `make test`), and each `f32` function on every `f32` (an `#[ignore]`d test, 46 s). `c/presk_rt.c`: the conversion of a buffer between machine types as the native backend converts one (`f64 → f32 → bf16`, two roundings), the `.psk.d` reader and writer, which take the type as text and its leaves from the export and parse no Presk type, and the arena. Files written by `presk-data` read in C, files written in C equal byte for byte to `presk-data`'s, each error with `presk-data`'s code. Step 1's check runs with `presk_libm.c` in place of the stand-ins, bit-identical; built in GCC's GNU mode without `-ffp-contract=off`, 14 of its 246 primitive runs move (D78) | done (2026-09-26) |
| 3 | The host, straight-line. A recorder on the kernel backend (`presk-kernel/src/record.rs`): the kernel backend's values carry no identity, so it mirrors the host symbolically, each tensor a view of a buffer at an offset, following calls, `let`s, tuples, records, closures, the `stack` fold's slices and scans and each primitive's decomposition into launches, and takes every launch from the backend itself. Its hooks do nothing when it is off: on the eleven entries of `make plan-check`, in the three types, the kernels and native outputs are `main`'s byte for byte, and `make plan-check` and `make ra4-dev-frozen` pass. The export records the native run itself, whose outputs `conformance/` needs anyway, and writes `model.c` (one straight-line function), `kernels.c`, `main.c`, the runtime, a Makefile, a README and `conformance/` (weights as a `params M` checkpoint, inputs and outputs as `.psk.d` files of closed types, a script that builds, runs and compares with `cmp`). Scalar arguments are fixed at export as `presk run` synthesizes them, keys dropped. Refusals `E8002`–`E8008`. C1 on the test set but `next_token`, in `f64`, `f32` and `bf16`, on the synthesized parameters and with their zeros drawn, and MiniGPT's four entries under a mixed realization: 68 programs, all bit-identical, built without a warning, 4.8 s in `make test`; the course GPT at its constants with its zeros drawn and with the trained checkpoint at `T = 256`, bit-identical in the three types (`make c-conformance`). C2 reported: the elements outside the band are the native backend's. Dropping any live launch, swapping two launches' buffers and rounding a `bf16` input once each change the outputs. On review: the conformance suite runs the examples with their zeros drawn too, question 75 amended, the work a gradient discards recorded in `docs/kernels-roadmap.md` (D79) | done (2026-09-26) |
| 4 | The host, structured. The recorder keeps the tree of calls and folds beside its operations, with the dimensions and the parameters' names. One C function per distinct recording of a Presk function, found by a canonical form (arguments by position, buffers by first definition, kernels by their text, callees by their forms, constants folded into the operations that read them), named with its dimensions when it has several. Results written into buffers the caller allocates, each function releasing what it allocated. The three kinds of fold as loops (the forward stack, the scan, the backward fold from the last layer), stacked parameters sliced as `w + l * chunk`, the accumulator alternating between two buffers; an iteration that differs from the next runs apart (the first block under a mixed realization). Buffers named after the source. The arena at the program's peak, read back from the running program: the course GPT at `T = 256`, 117, 59 and 29 MB in the three types, where releasing nothing took 352, 176 and 88 MB. Runtime scalars as `double` parameters. The straight-line host kept as the oracle (`--straight-line`): the 68 programs and the course GPT's 9 exported both ways, bit-identical to the native backend and to each other. A slice off by a layer (64 programs), a loop bound off by one (8) and a release before a buffer's last use (10) are each caught. The native backend's kernels and outputs unchanged (D80) | done (2026-09-26) |
| 5 | Plans, threads and portability. `--plan @P.json` exported, the native run recorded under it; `--threads N`, the native executor's workers. A launch the native backend divides among threads is one OpenMP region, each thread calling the kernel on the native backend's piece; the team capped at `omp_get_max_threads()`, since a `num_threads` clause overrides `OMP_NUM_THREADS` (so the variable lowers it, and `=1` gives one thread); without OpenMP one call. The split is part of the launch and of the canonical form, the kernel's text lacking its workers. `main.c -r R` times the entry; `presk bench --plan-only --weights`. C1 on every program built with OpenMP and without, and on one thread when a launch is divided: the test set, a plan stating splits and the native backend's own rule through the binary (`make test`); every legal depth of five recordings, 40 programs, on one thread, three, every thread and without OpenMP; the test set under `presk plan`'s plans in the three types and the course GPT at `B = 16`, `T = 256` under its plans (201 of 228 launches divided among ten threads) and the default layout (`make c-conformance`). A gap between pieces changes the outputs; an overlap cannot (the same bits twice) and a check of the pieces catches it. C3: GCC with and without OpenMP, Clang 21.1 (linked, run, the native bits) and x86-64 (linked) through zig 0.16.0 from PyPI, no warning. C4 at `B = 16`: every thread, native 1 094 ms, C 290 ms; one thread, 3 132 and 1 700 ms. Found in `presk plan`: a plan refused by `presk run` when a nest runs at two shapes (the MLP), and no layout at all in `bf16`. `make test` 11 s longer (D81) | done (2026-09-26) |
| 5b | After step 5's review: `presk plan` fixed. A nest launched at several shapes gets the layout the model prices at every shape whose predicted time, weighed by each shape's launches, is least, where the first shape decided and the MLP's plan was refused by `presk run`; a type the profile cannot price (`bf16` under the frozen profile) refused, where it wrote an empty plan. Only the MLP's plans change on the test set; the tutorial's and the course GPT's are the same (D82) | done (2026-09-26) |
| 6 | Randomness and generation. `sample` takes no key: it draws from the backend's stream, `ChaCha20Rng::seed_from_u64(seed ^ 0x5a5a5a5a)`, so the plan's key functions were not needed, and keys stay dropped. The recorder records `sample` as a host operation (`E8005` retired); the runtime has the stream, `presk_rng` (Bernstein's ChaCha20, `rand_core`'s block reading and `seed_from_u64`), and `presk_sample`; the stream is the program's state, taken by the functions that draw and seeded by `main -s` (the export's seed by default). Checked against the crates: 10⁶ draws over 10³ seeds, reads straddling two refills, and the native `sample` on 12 028 rows in the three types. The native `sample` now calls the `libm` crate's `exp`, the kernels', not the platform's: the C program depends on no C library; 540 generations of 32 sampled tokens are the ones glibc's `exp` gave. An entry's `f64` argument is a `double` of its `_in` struct, set at run time. `presk export --target c --serve`: the generation function and its greedy copy (every `sample` an `argmax`) recorded on one backend, sharing their functions, and `generate.c`, `presk serve`'s loop, whose output, `--json` included, is `presk serve --backend native`'s byte for byte; a model whose parameters do not fix the context's length refused (`E8009`). C1 on `next_token` (8 more programs in `make test`; with the trained checkpoint in `make c-conformance`). C5 in the three types through both hosts, greedy and sampled at three temperatures and four seeds, on the example with its zeros drawn (`make test`) and with the trained checkpoint (`make c-conformance`); C3 on `generate.c`. A draw skipped, the stream seeded again at each step and the window one token behind each change the tokens (D83) | done (2026-09-26) |
| 7 | Integer weights. No checkpoint holds integers: the native backend quantizes the weights each time it loads them, so the C program reads their reals and quantizes them as it reads them (`presk_quantize`, a transcription of `QTensor::quantize`), bit-identical to presk-rt on 4 027 392 weights of both widths and schemes, ties, zero channels, subnormals, huge values and NaNs among them. presk-rt's `pow2` scale, which looped for ever on a channel whose largest weight is below about 10⁻³⁰⁶, fixed, its scales otherwise unchanged. The conformance weights are the parameters as read, no longer `f32(q·s)`, whose re-quantized scales moved (0.8 % in `int8`, 10.7 % in `int4`). A leaf of integer weights is a `presk_q8` or `presk_q4` of its integers and scales; the kernels take the integers and their scales after the output, `(float)q * s[c]`; the recorder follows the host's conversions at a kernel's edge and its slices, integers where `QBuf::slice_leading` keeps them and `f64` copies where it gives up (`int4` layers of odd length). `E8007` narrowed to integer weights in the result, under a reshape, or sampled. Step 1's check on integer inputs: 2 880 primitive runs and the four nests' layouts, bit-identical. C1 through the binary on 42 programs (`odd.psk` in both widths and schemes, MiniGPT's forward pass, `next_token`, gradient and optimizer step with every matrix quantized, `D = 9`, `H = 3` in `int4`, biases and norms, a plan with splits), C5 under `--serve` (15 documents), and with trained checkpoints (`make c-conformance`); three mutations of the program each change its outputs. The native backend's kernels and outputs are `main`'s (66 comparisons); `make test` 158.6 s against `main`'s 147.8 s (D84) | done (2026-09-27) |
| 8 | The SCoP mode, `presk export --target c --mode scop`. Each nest of an entry at point zero (its default layout on one lane and one worker, no pack), captured from its shapes as `presk plan` captures the nests RA4 ranks, one C function per nest, shape and type, each in its own file between `#pragma scop` and `#pragma endscop`: arrays of literal dimensions, one index per axis; literal bounds, no range, no OpenMP; a branch on a computed value that only assigns scalars printed as a selection; the transcendental functions declared `const`, and `presk_rt.h`'s conversions `const`; the pragmas' warning ignored around the function. The host's own kernels are not nests and are left out. A driver times each nest on drawn inputs and checks it against the native backend's outputs (`make check`); `make polly` gives Polly's view. `--plan`, `--threads`, `--straight-line`, `--serve`, `--weights`, `--no-exact` and a quantized realization refused. Static control, by the rule (no loop variable assigned, every access affine in the loop variables): 366 of the test set's 383 nests in each type, the 17 others gathers and scatters. The front end is Polly, in LLVM 23.1.2's release (`make llvm`, `.llvm`, no root), none of the plan's being on the machine: each of the 1 098 static-control nests of the three types is one SCoP of the kernel's loops, none of the 51 others is (all are, allowed non-affine accesses); the course GPT's 28 nests alike. Graphite, not chosen, follows GCC's passes rather than the nests. Every primitive at point zero in the SCoP form (123 kernels) and every nest of the test set in the three types bit-identical to the native backend; built in GCC's GNU mode, or with a softmax's maximum made a minimum, not. The native backend's kernels and outputs are `main`'s (66 comparisons); `make test` 3 to 7 s longer (D85) | done (2026-09-27) |
| 9 | Documentation, tutorial step 18, version `0.9.0`, tag | done (2026-09-27): the manual brought to 0.9.0 as implemented — a chapter of its own for the export, `export_c`, made from the `native` chapter's section, which had grown a paragraph per step, and reorganized (how a program is made, its structure, threads, sampling, integer weights, the exactness rules, what is checked, portability, the band, speed, the SCoP mode, what it does not do), `native` pointing to it; `experiments` §9 with the statements of 0.9.0 and their verdicts; `cli` and `diagnostics` checked (`presk --help`, `presk export --help`, `E8001`–`E8009` against the code and their tests); this chapter closed; `index`, `install`, the crate's chapter list and the README; tutorial step 18, "A C program": the tutorial's attention exported at step 3's dimensions, built, checked against the native backend and the exact reference and run by hand, then under step 17's plan on OpenMP threads and on one, and in the SCoP mode, every quoted output from `make tutorial`, which covers steps 0 to 18; the export's summary line says "1 function besides the entry's", where it said "1 functions", and `presk --help` names the C export; CHANGELOG 0.9.0; spec questions 73–75 copied to `presk-specs`; version `0.9.0`; D86. Tag `v0.9.0` by the user |

### What 0.9.0 was judged on

Five statements, on the eleven entries `make plan-check` compares and the
course GPT at its course constants, fixed before the version began. An
entry was judged from the step that could export it, and every entry of
the test set was by step 6.

| # | Statement | Falsified if |
|---|---|---|
| C1 | every exported entry, in `f64`, `f32` and `bf16`, under the default layout and under a plan from `presk plan`, computes outputs **bit-identical** to `presk run --backend native` with the same realization, plan, dimensions, seed and weights | any output bit differs (step 2 found no function differing from the `libm` crate, so there is no exception) |
| C2 | the same outputs lie within the realization's conformance band of the exact reference, as the native backend's do | an output outside the band where the native backend's is inside it: the band is per element, and the native backend's gradient of `minigpt_grad.psk` crosses it where large terms cancel |
| C3 | the exported C compiles without warnings (`-Wall -Wextra -std=c11`) with GCC and Clang on aarch64, and cross-compiles for x86-64 | a warning or an error on any of them |
| C4 | (reported, decides nothing) the exported program's wall time against the native backend's, one thread and every thread, on the course GPT's forward at `B = 16` | — |
| C5 | the tokens the exported `next_token` generates over a sliding window are those `presk serve` generates, greedy and sampled with the same seed | any token differs |

**The answers** (`experiments` chapter §9). **C1 holds**: every entry of the
test set in the three types, on synthesized parameters and with their zeros
drawn, under a mixed realization and under `presk plan`'s plans, through
the structured host and the straight-line one, built with OpenMP and
without and run on one thread and every thread, and the course GPT at `T =
256` with its trained checkpoint and at `B = 16` under its plans, computes
the native backend's outputs bit for bit; so do 42 programs with integer
weights and every nest in the SCoP form. No transcendental function differs
from the `libm` crate, so C1 has no exception. **C2 holds**, as a copy: the
outputs are the native backend's, outside the band on exactly its
elements, which cancel (question 75). **C3 holds**: no warning on GCC 13.3
with and without OpenMP, on Clang 21.1 and for x86-64 through zig, whose
aarch64 program runs and gives the native bits, and on Clang 23.1.2 for the
SCoP mode's files. **C4**, reported: on the course GPT at `B = 16` in
`f32`, 290 ms against the native backend's 1 094 ms on every thread, 1 700
against 3 132 ms on one. **C5 holds**: `generate` prints `presk serve
--backend native`'s document byte for byte, greedy and sampled, in the
three types, with integer weights too, on the example with its zeros drawn
and on the trained checkpoint.

### Verification of 0.9.0

Run on falicon (aarch64, twenty cores, GCC 13.3) on 2026-09-27, with the
binary of step 9. Step 9 changes two strings of the binary and measures
nothing new: the statements' checks were run again as they stand.

- `make test` (207 tests pass, among them the C export's
  programs through the binary, both hosts, with OpenMP and without, and the
  SCoP mode's nests), `make check-examples` (only `W0001`) and `cargo doc -p
  presk-manual` (no warning).
- `make c-conformance`: the test set under `presk plan`'s plans, the course
  GPT at its constants and at `B = 16`, the trained checkpoints with integer
  weights, C5 on the trained MiniGPT, Clang and x86-64 through zig, and
  Polly on every nest: 18 tests in 4 min 9 s.
- `make plan-check`: a plan stating the default layout gives the same
  kernels, byte for byte, on the eleven entries. `make ra4-dev-frozen`:
  `docs/ra4-dev-model-0.8.0.md`, byte for byte.
- `make tutorial`: every command of steps 0 to 18 passes, with PyTorch in
  `.venv`, in 1 min 15 s. Step 18's quoted outputs are from that run; the
  other steps' agree with it but for timings and what depends on them, as
  in 0.8.0: step 17's fit, predictions and rank moved, and both `ein`s
  chose the other loop order, within `u`, with the same tiles, packs,
  splits and verdicts.
- Polly on the tutorial's six nests (step 18): one SCoP each.
- C4 was not measured again (`make c-speed`, a quiet machine): it decides
  nothing, and nothing it times changed since step 5.

## What 0.9.0 has, and what it does not

**Has.** A way out of the compiler that keeps its numbers: `presk export
--target c` writes an entry, any entry the native backend runs, as a C
program of a few files that builds with any C11 compiler and computes the
native backend's bits (`export_c` chapter). The kernels are the native
backend's, under its realization and plan; the host is a recording of the
native run, structured as C functions and loops over layers, in an arena
sized at the program's peak; a divided launch is an OpenMP region, the same
bits on any team; `sample` draws from the program's own ChaCha20 stream,
and `--serve` exports `presk serve`'s loop; integer weights are quantized
as they are read. Its exactness rests on three rules: no floating-point
contraction, musl 1.1.22's transcendental functions shipped (the `libm`
crate's), and every conversion and host operation done in the native
backend's order. Every program carries its own check, `make check`, against
the native backend's outputs. A SCoP mode writes each nest at point zero
for polyhedral tools, its static control checked with Polly. And tutorial
step 18.

**Does not have.** The training loop in C (llm.c's other half), `.pskk`,
`presk build` and the bundle; the CUDA export decided for after it, the
GPU and the matrix unit. A gradient's discarded work removed (the C copies
it). Reshaping integer weights in the native host. The polyhedral
comparison with RA4 (pet, PPCG and PLUTO not run). A measure of where C4's
lead comes from (`docs/kernels-roadmap.md`, "What the C export's speed
says").

**The shape of the result.** 0.9.0 tested no hypothesis of the research
programme: it gave the compiler a portable output with a proof obligation,
and met it. An exported program computes, bit for bit, what the native
backend computes, which the exact reference has measured; so a budget's
verdict on the native backend holds for the program, on GCC or Clang, on
one thread or twenty (on x86-64 the program builds, and was not run). On
the course GPT it also runs
1.8 to 3.8 times faster than the native backend, with the same kernels:
the gap is the native backend's to close.
