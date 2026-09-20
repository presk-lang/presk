# Changelog

## Unreleased — 0.5.0-dev

Precision as a resource
(`presk-specs/implementation/presk-plan-0.5.0.md`): RA3 of the research
programme, which the two previous plans deferred for one stated reason — the
solver could not be evaluated on programs that take minutes a step — and
which 0.4.0 made reachable by getting a course-scale step to 39.8 s. The
version turns `--realize` from a two-valued flag into a solved object
carrying a storage and an accumulator type per fusion group and a declared
reduction order, adds a measured magnitude envelope, a first-order error
propagation that exploits known conditioning, a sensitivity analysis by the
existing differentiation, and the solver that turns `budget(rel_error = ε)`
into an assignment minimising predicted time. One vectorized GEMM ships under
that contract, the first item of `docs/kernels-roadmap.md`, because without
vector lanes precision buys no measurable time and without the contract the
vector kernel reassociates a sum it is not allowed to reassociate. The
primary deliverable is the ε-to-quality curve the roadmap asks for.

**What it shows.** A budget declared in a program becomes an assignment whose
distance from the exact denotation is *measured*, not asserted: 1.05e-7
against a budget of 4.52e-3 on the corpus GPT (statement D). A sweep of that
budget is a clean one-parameter curve, four times less traffic at 1e-1 and
1.69× at 1e-7, every point verified. **What it does not show** is that any of
it is faster: the solved realization runs at 0.96× the speed of all-`f64`
(statement E), because the solver minimises traffic and these kernels are
bound by arithmetic. **The vectorized kernel is deferred to 0.6.0**, as 0.4.0
deferred vectorization before it and for a reason now better understood:
vectorizing across output columns is bit-exact (D42) and needs a vector form
in PSK-Kernel, which this version did not write. Statement G is about that
kernel, is not askable, and needs restating.

- Step 9: the four statements answered by measurement, and the MLPerf
  comparison. **D holds** — a declared budget becomes an assignment whose
  measured distance from the exact denotation is 1.05e-7 against 4.52e-3,
  and no other compiler can run that check. **E fails**: the solved
  realization is 5.63 ms against all-`f64`'s 5.39, **0.96×**, because the
  solver minimises traffic and these kernels are not traffic-bound —
  optimising the right proxy in the wrong regime. **F fails narrowly**, 10 of
  21 entries within 1e3, median 1.8e3. **G is not askable**: it is about a
  vectorized GEMM that does not exist and, by D42, should be built under a
  sequential order where it would have no gap to measure. So the claim that
  precision can be declared and verified is demonstrated; the claim that it
  pays is not.
- Step 8 (ε axis done): `make curve` sweeps the budget on the corpus GPT —
  one program, one parameter varying, which is the sweep the specification
  says cannot be done cleanly elsewhere. **Every point is inside its budget
  by measurement**, and traffic rises monotonically as the budget tightens:
  4.00× less than all-`f64` at 1e-1, 2.67× at 1e-3, 1.69× at 1e-7, with a
  knee between 1e-2 and 1e-3. The measured gap is not monotone, and the
  propagated bound at these constants is 7.15e11 — vacuous, which is why the
  measured columns are the result. **The quality axis is missing**: a
  validation loss per point needs a training run per point, which this
  version has not been able to afford.
- Step 7 (partly landed): `Stmt::SetL` assigns a loop variable in PSK-Kernel
  and both executors, and on it the interleaved contraction is emitted; the
  declared order is part of the kernel cache key and reaches the kernel,
  which a test asserts — the MLP at `D = 32` in `f32` gives 1.5494239330292
  sequentially and 1.5494240522385 with eight partials, both executors
  agreeing to the last digit. `presk bench --native-only` makes a large
  benchmark affordable. **The measurement is a negative result and a useful
  one**: on a 1024³ matmul the reassociating order is **2.7× slower** (239.0
  ms against 87.3), because it cannot use the blocked form and falls back to
  the naive contraction. For a CPU contraction there is no reason to
  reassociate: the exact form is faster *and* exact. The order stays
  implemented and measured for the coprocessor and the GPU. **The finding is worth more than the code**
  (D42): the blocked form's eight accumulators are eight different output
  *columns*, each still summing in order, so putting them in vector lanes is
  **bit-exact**. Vectorization does not need a new realization after all,
  which is a better position than D31 assumed; the contract is still needed
  for the coprocessor and the GPU, where a reduction crosses lanes.
- Step 6: the solver. `presk solve --precision ENV.json --budget ε [--weights
  ckpt] --out R.json` turns a budget into a realization, which `--realize
  @R.json` then runs. Propose greedily by bytes saved per unit of
  sensitivity-weighted error, **verify by measurement against the exact
  reference** rather than by the propagated bound (D41) — the bound is
  vacuous on a transformer, and measuring is the stronger check and one only
  this compiler can make — then widen the worst site one step at a time. The
  objective is traffic, not time, because `bf16` is slower than `f32` on a
  compute-bound kernel. On the corpus GPT at `D = 64`, trained, budget 1e-3:
  **2.67× less traffic**, zero widening rounds, measured gap 1.05e-7 against
  a budget of 4.5e-3. On a small program the greedy proposal is exactly
  optimal at 1e-3 and 1.057× off at 1e-6.
- Step 5: sensitivity by differentiation. `presk dump --stage sensitivity`
  prints `‖∂L/∂v‖₁` at every site and per module, for one reverse sweep: the
  adjoints already exist, so the step keeps the correspondence between a
  forward value and its adjoint rather than differentiating again (D40). Two
  findings. A ranking at initialization is vacuous, because a zero-
  initialized head stops every upstream gradient, so `--weights` is needed.
  And half the mixed-precision folklore holds: the output head dominates at
  5.86 against 2.46 for the whole block stack, while the token embedding
  table scores 0.28 — below the down-projection practice narrows freely — so
  keeping embeddings wide is not explained by sensitivity.
- Step 4: first-order error propagation over PSK-Core.
  `presk check --precision ENVELOPE.json [--sites N]` bounds an entry's
  distance from its denotation under a realization. Quotients, roots and
  logarithms propagate in *relative* terms, which is why a normaliser costs
  nothing; a residual adds; a softmax is 1-Lipschitz in the ∞ norm; and a
  contraction's constant is a function of the declared reduction order, so
  blocking leaves a bound untouched where interleaved partials change it.
  **Sound and, on a real model, vacuous** (D39): the bound exceeds the
  measured gap on every conformance entry in every realization, median ratio
  1.8e3 and 27 on a normaliser, but 1e20 on the corpus GPT — a contraction of
  `k` terms multiplies error by `k` where committed errors grow like `√k`.
  The tightening is named: each sum's condition number, measured. New
  `presk-precision::propagate`, new tests for soundness and for the reduction
  order's effect, `make precision` now bounds as well as calibrates.
- Step 3: the magnitude envelope. `presk calibrate --precision` records the
  largest `|x|` at each site on the reference interpreter, since an absolute
  error means nothing without a scale and a relative one is unbounded near a
  zero. Static intervals cannot supply that scale for a network, so it is
  measured and every bound quoting it quotes its sample count (D35). The
  step's content is the check that the envelope is a property of the program
  rather than of the sample: on the corpus GPT, disjoint slices of the civil
  code agree within **1.241×** and far-apart ones within **1.128×**, over 77
  compared sites with none present in only one. New manual chapter
  `precision`, new target `make precision`.
- Step 2: the realization becomes an object. `Realize`, a flag with two
  values, is now `Realization`: a storage and an accumulator type per site, a
  declared reduction order, and the magnitude envelope any bound assumed
  (D34). `--realize` takes `f64`, `f32`, `bf16` or `@FILE`. A site is a
  fusion group's root, and a buffer crossing between sites of different types
  is converted at the kernel's edge. `bf16` is storage only — a shift in, one
  round to nearest even out, written once in Rust and again as Cranelift
  integer instructions, with the two executors agreeing digit for digit on a
  gradient (D38). The reduction order is the field D31 and D32 both needed
  and had nowhere to write down: `blocked` is bit-identical to `sequential`,
  `interleaved` reassociates. **Nothing moved**: every `f64` and `f32` entry
  of the conformance table is what 0.4.0 measured, to the last digit, and the
  suite now reports a third column, `bf16`, worst 3.3e-2.
- Step 1: the axis `split`, the declared inverse of `merge` — one axis read
  as two, row-major, a pure retag whose adjoint is `reshape_like`, checked
  against finite differences at 512 bits and identical on all three backends.
  Unlike `merge` it names both halves, so **splits chain where merges cannot**
  (D36), and that is what lets `coverage/llama.psk` transcribe grouped-query
  attention exactly — with neither `split_heads` nor `merge_heads`, and with
  no transpose, because `ein` contracts by name. Per-layer randomness inside
  a `stack{L}` fold needed nothing new: the key rides in the accumulator,
  which is already threaded, and the index route was dropped after
  establishing that AD refuses to differentiate an indexed fold (D37). The
  corpus GPT gains the course's dropout at its three sites. Two corrections
  to 0.4.0's account: rotary positions were never blocked on the split but on
  `concat`, a way to build a tensor along an axis, and question 51's
  provisional answer was wrong.
- Step 0: version `0.5.0-dev`; crate `presk-precision` scaffolded, with the
  two quantities it must not confuse written into its own documentation — a
  bound guarantees and a sensitivity ranks; decisions D34 (the realization is
  an object, not a flag) and D35 (a bound is conditional on a measured
  envelope, and says so); spec questions 50 to 55.

## 0.4.0 — 2026-09-20

Real models, real speed
(`presk-specs/implementation/presk-plan-0.4.0.md`): a corpus of models from
an NLP course as the forcing function, then the scheduling and backend work
it forces. One training step of the course's GPT went from 289 s to 39.8 s,
seven times faster, **without changing a single printed digit of the loss** —
every transformation in this version is exact over ℝ and reassociates
nothing. Measured against PyTorch the tool is still 94× slower a step, and
the `experiments` chapter says where that goes: about twenty-six of it is
PyTorch reaching a matrix coprocessor that presk does not emit for, and the
reason it does not is that both routes there reassociate a sum, which in this
system is a different realization owing a declared order and a measured gap.
Steps in reverse order:

- Step 1 (in progress): `corpus/nlp/`, the ladder of the NLP course on the
  words and text of the Code civil (`data/prepare.py`, `make corpus`,
  `EXPECTED.md`). Rung 1 `01_bigram_count.psk`: the counting bigram in closed
  form, as one `ein` over two `one_hot` tensors — 2.343771504 on all 67652
  pairs, identical on the reference interpreter and the native backend, and
  equal to the NumPy reference. Rung 2 `02_bigram_neural.psk`: the same
  function fitted by AdamW, 3960 steps in 3.6 s natively, batch losses
  straddling the closed-form value, which is the course's own sanity check.
- Step 10: version 0.4.0. Tutorial step 13 "A model that is not a toy" walks
  the corpus — text to dataset, the counting bigram in closed form on 67 652
  real pairs, the same model fitted, the MLP that needed `merge`, and the
  course GPT checking at its own constants — with every quoted output
  produced by `make tutorial`. New manual chapters `scheduling`, `corpus` and
  `coverage`; `status` closed with the verification list and with what this
  version does not have; README and the version itself. Spec questions 45–49
  copied to `presk-specs/implementation/presk-spec-questions-0.4.0.md`, and
  the plan carries a status note listing every deviation. WaveNet and dilated
  convolution are deferred with their reason: step 9 established convolution
  is a type-system extension, not a primitive.
- Step 9: coverage of the shape fragment, measured on the MLPerf
  architectures — the roadmap's own metric for the type system, and the only
  thing this project takes from MLCommons (D27). `coverage/bert.psk` is
  BERT-large as the suite fixes it, 24 post-normalized blocks with the three
  embedding tables, a padding mask that broadcasts by name and the SQuAD span
  head: it checks whole. `coverage/llama.psk` is the decoder Llama, GPT-J and
  Mixtral share, with `RMSNorm`, SwiGLU and `silu` written from `sigmoid`: it
  checks with two substitutions, rotary positions and grouped-query
  attention, both blocked on the same missing construct — splitting one axis
  into two, whose three workarounds are closed (`split` unresolved, `concat`
  not implemented, range slicing not syntax). Of seven model families, two
  are expressible and they are the transformer families; convolution blocks
  four benchmarks, segmented reductions two, top-*k* with a runtime parameter
  gather one, and non-maximum suppression is outside a shape language
  altogether. Manual chapter `coverage`, `make coverage`.
- Step 8: the decisive experiment, answered by measurement. The course GPT
  (`D=384`, 6 heads, 6 blocks, context 256, batch 16, character-level Code
  civil, `f32`, twelve cores) trained by `presk train` and by the `train.py`
  the export writes. Of the three statements this version set: **A fails**
  (94× PyTorch a step, 82× with `--opt`, against the 3× asked), **B fails**
  (`--opt --profile` is worth 1.14×, against 1.30× asked), **C holds** (the
  profile predicted 1.087× where 1.14× was measured, 5 % off against the
  30 % band). One step is 266 GFLOP: PyTorch 0.405 s at 656 GFLOP/s, presk
  37.9 s at 7.0.
  The gap is diagnosed rather than lamented: PyTorch's CPU matmul reaches
  Apple's matrix coprocessor through the platform BLAS at 1309–1454 GFLOP/s
  where presk's scalar blocked kernel does 47–56, a factor of twenty-six that
  is instruction-set access and not compiler cleverness; the remaining three
  is presk's own non-contraction work. Recorded in the `experiments` chapter
  with what it says about the thesis: this version bought 7.3× and every
  transformation it shipped is exact — the losses are identical to every
  digit printed — so the discipline is affordable, and the next order of
  magnitude needs a new declared realization rather than a faster loop
  (D31, D32).
- Step 7: a rewrite the cost model arbitrates. The rule this version's plan
  named, reassociating `(q·k)·v`, has nowhere to fire: counted on the corpus
  GPT after differentiation, no contraction takes another contraction as an
  operand, because a transformer puts a softmax, a GELU or a normalization
  between every pair. The rule that does fire moves a scalar across a
  contraction — `(Σ aᵢbᵢ)/s = Σ(aᵢ/s)bᵢ`, exact over ℝ — and the e-graph
  proposes scaling every operand so the cost can choose. Attention divides
  6 291 456 scores by `√hd` where it could divide 1 572 864 queries; the
  calibrated roofline prices the two forms apart (`SelfAttention.apply`
  10 205 µs against 9 489 µs) while operation counts price them the same.
  Measured over forty training steps of the corpus GPT: 0.710 s a step
  without rewriting, 0.626 with counts, 0.617 with the profile, all three
  giving identical losses. The profile's predictions rank far better than
  counts (Spearman 0.96 against 0.59) but its decisions barely do, and its
  extraction costs 2.54 s against 0.41 s, so it takes some 230 steps to repay
  itself — a cost model can rank kernels well and still not change which
  program is extracted. D33.
- Step 6: contractions in a blocked form. `dot` and `matmul` no longer walk a
  column of the right operand per output element: eight columns of the result
  are held in scalar accumulators across the whole contraction, and the inner
  loop loads one element of the left operand and a contiguous run of eight
  from the right, storing nothing. A width that is not a multiple of eight
  takes a row-wise form that sweeps a whole row and handles any width. Both
  are **exact** — each output still accumulates `k` in increasing order, only
  the memory the loop touches changes — and a test asserts that the two forms
  agree bit for bit in `f64` and `f32`, with the conformance gaps unchanged to
  every digit. A 1024³ `f32` matmul goes from 3.7 to 9.9 GFLOP/s on one thread
  and reaches 47–56 GFLOP/s on twelve, about two thirds of this machine's
  scalar peak. **One training step of the corpus GPT at the course's own
  constants now takes 39.8 s, against 289 s before this version's scheduling
  and backend work — seven times faster, with the loss unchanged to the last
  digit.** D32.
- Step 5: threads. A kernel whose outermost loop is parallel — the body is
  one loop nest and every store indexes the output with that loop's variable,
  so the iterations write disjoint elements — is split across workers by the
  Cranelift backend; a reduction over the outer axis and a scatter such as
  `embed_vjp` fail that test and stay sequential. Compiled kernels now take
  the half-open range of their outermost loop, so one function serves the
  sequential call and every worker. Only kernels above four million weighted
  operations are split, since spawning costs tens of microseconds.
  `--threads N` on `run`, `train` and `bench`, defaulting to the machine's
  cores; `PRESK_THREADS` does the same for a library caller.
  The split is *exact*: each output element is computed by the same inner
  loops in the same order, so one worker and twelve give bit-identical
  buffers in `f64` and `f32`, asserted element by element. Measured on twelve
  cores, one training step of the corpus GPT: 4.00 s → 0.99 s at `D=64`, and
  **289 s → 58.3 s at the course's own constants**, a five-fold gain for an
  unchanged loss. Vectorization is not in this version, and the reason is
  recorded: the kernels that would vectorize exactly are not the bottleneck,
  and vectorizing a contraction's inner sum reassociates it, which is a new
  realization needing its own declared order and measured gap. D31.
- Step 4: PSK-Sched v0, fusion groups. New crate `presk-sched` partitions
  each PSK-Core body into connected sets of elementwise nodes of one result
  type whose interior values are read only inside the set; PSK-Kernel lowers
  a group to a single loop nest, so the intermediates stay in registers and
  only the result is written. A group whose type is a parameter tree is
  mapped over the leaves, one launch per leaf for the whole chain, which is
  where most of the saving is. A value read twice inside a group is computed
  twice rather than stored, bounded at sixty-four expression nodes. The
  partition is structural, so it is computed once per function and serves
  every shape, realization and backend. `presk dump --stage sched`; the
  safety invariants are tested on the corpus and the examples.
  A whole training step of the corpus GPT goes from 1515 launches to 884,
  42 % removed (`presk train` reports it); the plan asked for half, and the
  gap is the forward and backward passes, whose chains reductions cut.
  On the corpus GPT the optimizer step goes from 414 launches to 111 and,
  on the tiny MiniGPT, from 1.54 ms to 1.32 ms — about 15 %, outside the
  ±4 % spread, and the first time a pass in this project has moved the clock
  outside the noise. The saving is traffic, not launch overhead: 303 launches
  at the measured λ of 50 ns would be 1 % of that time. The forward pass
  loses a quarter of its launches and no time, being contraction-bound, and
  every golden output is byte-identical since fusion reassociates nothing.
  Not done, and recorded: multi-output kernels, reduction fusion, and an
  extraction that knows about the partition (question 49). Manual chapter
  `scheduling`; D30.
- Step 3: `dropout`, with its key in the signature. `dropout(rate, key, x)`
  multiplies `x` by an inverted-dropout mask drawn from `key` and nothing
  else, and `split_key(key) → (key, key)` gives independent keys so a
  function can drop more than once. The key is explicit for a correctness
  reason, not a stylistic one: differentiation recomputes the forward pass,
  so a mask drawn from a hidden stream would differ between the forward and
  the backward and the gradient would be silently wrong. Being a function of
  the key, the mask recomputes identically, and `dropout` needs no
  differentiation rule of its own — it lowers to `x · dropout_mask(rate,
  key, x)`, whose adjoint is the adjoint of a product. Checked against
  central finite differences at 512 bits *while dropping*
  (`tests/data/dropout_grad.psk`), with the same key giving the same mask and
  another key another one.
  An objective may take a per-step key between `θ` and the batch (`fn loss(θ
  : params M, key : key, batch : …)`), which `presk train` derives from the
  run's seed as the step's child key (`Key::child`, constant time);
  `tests/train/dropout_reg.psk` trains reproducibly from its seed. Outside a
  training run the mask is all ones, the expectation of the random function,
  so `run`, `serve` and the export evaluate the model's mean: the seam gains
  `Backend::set_training`, and the export emits `F.dropout(x, p,
  training=False)` because a Presk key cannot drive PyTorch's generator.
  Answers spec questions 33 and 48, except for one part recorded there: a
  `stack{L}` fold passes the same extras to every iteration, so the blocks of
  a stacked module share one key.
- Step 2: text at scale. `presk data chars` builds a character vocabulary
  from a UTF-8 text (every distinct character, sorted, as a JSON array of
  tokens) and `presk data encode` turns a text into the token stream
  `lm-windows` cuts into windows; a character the vocabulary does not hold is
  named with its code point, and a sub-word vocabulary is refused rather than
  guessed at (`presk_data::{char_vocab, encode, decode_text}`, round-trip
  test on accents, apostrophes and newlines). The Code civil becomes 1182121
  tokens over 92 characters and 4617 non-overlapping windows of 256 in a
  2.4 MB file. Corpus rung 5 `05_gpt.psk`, the course GPT, passes `presk
  check` at the course's own constants (`D=384, H=6, L=6, T=256`) with the
  data binding `T`; a training step takes 289 s at those constants (1.87 GB
  resident, about 0.9 GFLOP/s) and 4.2 s reduced to `D=64, H=2, L=2`, which
  is the wall steps 4 to 6 exist to break. Resident memory is identical at 3 and at 12 steps, because
  the reader already seeks to each sample as the batch asks for it: the
  streaming the plan asked for was already there, and only overlapping
  windows would need lazy windowing (`make corpus-gpt`). `presk data bpe` is
  deferred: the course's GPT is character-level, so nothing in the corpus
  needs sub-word tokens yet.
- New builtin `merge(axis a, axis b, x)`, the answer to spec question 45: two
  adjacent axes become one of length `|a|·|b|`, row-major. Adjacency is
  required (joining distant axes would also transpose) and at least one length
  must be constant (`E1007` otherwise, two `tests/diag/shape_merge_*.psk`
  fixtures); the joined axis is unnamed and is named by the signature it is
  passed to. Since row-major buffers make it metadata-only, it launches no
  kernel on any backend and costs nothing in the model; its adjoint is the
  retag back (`reshape_like`), checked against central finite differences at
  512 bits (`tests/data/merge_grad.psk`). Exported as `.flatten(k, k+1)`, with
  the conformance test passing.
- `embed` now carries the name of its table's second axis into the result
  (`param tok : ℝ[vocab: V, emb: E]` gives `ℝ[…, emb: E]`), so the new axis
  can be referred to by name; an unnamed table column still gives an unnamed
  axis, so nothing that existed changes.
- Corpus rung 3 `03_mlp.psk`: the embedding MLP of the course (Bengio 2003),
  three characters of context through `merge`, 2112 AdamW steps in 35 s
  natively, loss 3.71 → 1.73, well below the bigram's 2.34.
- New builtin `one_hot(t : idx[V][S]) → ℝ[S, V]`, the indicator vectors whose
  contraction with a table `embed` denotes: typing, lowering, regularity
  (`Disc`, like every index argument), differentiation (no adjoint flows
  through it), the reference interpreter, a PSK-Kernel v0 kernel, the
  PyTorch export (`F.one_hot`) and the cost model. Identical results on the
  interpreter, the kernel interpreter and the Cranelift backend.
- `presk run --data FILE.psk.d [--samples N]`: the entry's last argument is
  taken from a data file instead of synthesized, with the `batch` axis of
  that argument bound to the number of samples read, so an objective can be
  evaluated on its own dataset (`presk_train::run_on_data`;
  `presk_interp::run::resolve_entry` is now public).

## 0.3.0 — 2026-09-19

Native backend first, then e-graph rewriting
(`presk-specs/implementation/presk-plan-0.3.0.md`): a runtime seam
(`Backend`), PSK-Kernel v0 and its interpreter, a Cranelift JIT behind it
(`--backend native --realize f64|f32`, the tiny MiniGPT forward 27× the
reference, twenty training steps 34 s → 0.4 s), training and serving on it
with their gaps to the exact reference measured, `inference serve` and
`presk serve`, the e-graph optimizer `presk-opt` with rules exact over ℝ and
the interpreter as oracle, and a cost model whose roofline constants are
fitted on measured kernels. Steps in reverse order:

- Step 8: version 0.3.0; tutorial step 12 "Run it natively, then let the
  compiler rewrite" (`make tutorial` runs its commands: bench, native
  training with the same log as the reference, `--opt` with the same
  gradient, `--calibrate`, `--profile`); README, this changelog, `status`
  closed with the acceptance checks of the plan; spec questions 35–44 copied
  to `presk-specs/implementation/presk-spec-questions-0.3.0.md`; the
  kernel and native backends report their run-time errors as `E7001` (no
  separate `E10xxx` range was needed).
- Step 7: cost model v0.1, calibrated — `presk_kernel::cost` computes the
  work (weighted operation count of the loop nest) and traffic (bytes read
  and written) of every kernel from its IR and `TimedExec` times every
  launch; `presk bench --calibrate FILE.json [--realize f64|f32]` fits the
  roofline `λ + max(W/π, Q/β)` on the measured kernels of an entry (relative
  least squares on a refined log grid), merges with an existing profile and
  writes it with Spearman's rank correlation of three predictors (operation
  counts 0.15, work 0.91, fitted roofline 0.95 on the tiny MiniGPT's three
  entries in `f64`; 0.37 / 0.77 / 0.89 in `f32`); `presk_opt::model` derives
  the same work and traffic for a PSK-Core node from its type, the run's
  `--dim`s bound by name and 64 for the other dimensions; `--opt --profile
  FILE.json` on `run`/`train`/`serve`/`bench`/`export`/`dump --stage
  core-opt` extracts by predicted seconds (the report then prints
  microseconds). The run goldens are unchanged under a profile; the three
  MiniGPT entries time the same with and without the profile, within the
  ±4 % run-to-run spread. `make calibrate`. Manual: `optimization` gains
  "The cost model, calibrated"; D25.
- Step 6: crate `presk-opt` — e-graph rewriting of PSK-Core with `egg`:
  one e-graph per function body, the PSK-Core type of every class as an
  analysis (rules fire only between compatible types, the rewritten program
  is typed), rules exact over ℝ in three families (algebra: commutativity,
  associativity, neutral and absorbing elements, `neg`, `pow`, `exp`/`log`,
  `sqrt`, distributivity; reductions: scalar factors out of `sum`/`mean`;
  structure: `sum_to`/`broadcast_like` identities between equal types,
  `zeros_like` in sums, `field`/`set_field` chains and their sums into a
  single `record`, `proj` of tuples), extraction by operation counts,
  write-back in ANF (dead definitions, such as the forward recomputation in
  adjoints, disappear); `--opt` on `presk run`, `train`, `serve`, `bench`
  and the conformance path of `export`; `presk dump --stage core-opt` with a
  report (functions rewritten, lets and cost before and after, stop
  reason); a function that hits a limit is extracted from the partial
  e-graph and reported (D24). Oracle: the eight `tests/run` goldens and the
  20-step MiniGPT training golden are unchanged under `--opt`, a property
  test compares random PSK-Core expressions before and after rewriting on
  the interpreter at 128 bits (2⁻¹⁰⁰), and the native backend gives the
  same forward, gradient, training and serving outputs with and without
  `--opt` (`make opt-examples`). `Linear.apply.vjp` goes from 14 to 7 lets,
  `mlp.psk` from 89 to 71; no measurable speed change on the tiny shapes.
  `PrimOp` and `Lit` gain `Eq`/`Hash`; `presk dump --stage core|core-after-ad`
  honour `--const`. Manual chapter `optimization`; question 44 (axis names
  in the equality of values).
- Step 5: serving — the `inference serve` block resolved by the checker
  (model, `weights` path, the generation function found by its signature
  `(θ : params M, ctx : idx[V][batch: B, seq: T], …) -> idx[V][batch: B]`,
  `W0003` without one, `--summary`); crate `presk-serve`: the generation
  loop over the `Backend` seam, sampling as written or greedy through a copy
  of the function with `sample` replaced by `argmax`, a sliding window of
  `T` tokens when the parameters fix the context length (`pos : ℝ[T, D]`,
  question 43); `presk serve`; `presk export --serve` writes `serve.py` and a
  `pskd.read_params` that decodes a `params M` checkpoint with the parameter
  tree's layout generated into the script; on the tiny MiniGPT the greedy
  tokens are identical on the reference, natively in `f64` and `f32`, and in
  `serve.py` (`make serve-examples`, `make serve-pt`); manual chapter `serve`.
- Step 4: training natively — `presk train --backend native --realize
  f64|f32` compared with the reference (regression, 200 steps: 4.4e-16 in
  `f64`, 1.6e-7 in `f32`; test in `presk-cranelift`); `presk export
  --conformance-entry train:K` also runs the `K` steps natively and writes
  `conformance/native_{f64,f32}.json`; `experiments/exact_vs_almost.py` and
  the `experiments` chapter measure presk's own realizations next to
  PyTorch's (tiny MiniGPT, 20 steps: 3.7e-7 / 6.7e-16 against 9.6e-7 /
  8.9e-16); `make train-examples` trains natively too; `V=64 D=64 H=4 L=4`
  trains 500 steps in 159 s.
- Step 3: crate `presk-cranelift` — every PSK-Kernel v0 kernel compiled to
  machine code by Cranelift's JIT on first use (loops as blocks, `libm`
  symbols for the transcendental functions), behind the kernel backend's
  host program; `--backend native` on `presk run` and `presk train`; the
  conformance suite passes natively in both realizations; `presk bench`
  times an entry on every backend (tiny MiniGPT forward: 16 ms on the
  reference at 64 bits, 0.59 ms native; 20 training steps 34 s → 0.4 s).
- Step 2: crate `presk-kernel` — PSK-Kernel v0: every tensor primitive of
  PSK-Core lowered to a loop-nest kernel at concrete shapes and one machine
  type (specialized on first use, cached), a textual form (`presk dump
  --stage kernel`), an interpreter of the kernel IR in `f64` or `f32`, and
  the host program evaluating PSK-Core bodies over the kernels (structural
  primitives, initializers and sampling on the host with the reference's
  keys); `--backend kernel` on `presk run` and `presk train`; the conformance
  suite against the interpreter (41 primitive cases, 11 example entries,
  gaps ≤ 2.6e-14 in f64 and ≤ 5.3e-6 in f32).
- Step 1: crate `presk-rt` — runtime values (buffers of exact reals, `f64`,
  `f32` or `i64` with named axes; tuples and records), the `Backend` trait
  (`prepare` an entry at bound dimensions and a realization, `call` it),
  `InterpBackend` (the reference interpreter behind the trait, exact values
  passing through untouched), the parameter-tree I/O moved from
  `presk-train`; `presk run` and `presk train` run over the trait
  (`--backend interp`, the only backend so far); the interpreter's input
  synthesis is exposed as `prepare_run`.
- Step 0: version `0.3.0-dev`; spec questions 35–42; decision D21.

## 0.2.0 — 2026-09-19

Training end to end (`presk-specs/implementation/presk-plan-0.2.0.md`): a
data format, the optimizer as PSK-Core, `stack{L}` as a fold, `presk train`
on the interpreter, the training run and the spec's graph mode in the PyTorch
export, conformance of one optimizer step and of 20 training steps against
PyTorch. Steps in reverse order:

- Step 7: manual chapters `training`, `data` and `experiments` (`make
  experiment`: a silent broadcasting bug of plain PyTorch that the types rule
  out; the gap of float32/float64 training to the exact reference, 3e-7 and
  4e-16 after ten steps), tutorial step 11 (train a model), README, this
  changelog; interpreter errors name the function and
  the binding (`in \`RMSNorm.apply\` (%8): …`); the constant `pi`; an
  unconstrained dimension argument of a call is left unbound instead of
  failing.

- Step 1: the `.psk.d` data container (crate `presk-data`): samples of one
  Presk type, `f64` reals injected exactly, `idx` at the width of their
  bound; `presk data pack | lm-windows | info | export`; the training
  pipeline (`shuffle`, `batch`, `take`) over sample indices; error codes
  `E9xxx`; a standalone type parser in `presk-syntax`.
- Step 6: PyTorch export of the training run: `--train` writes `pskd.py` (a
  `.psk.d` `Dataset`), `train.py` (the run as a PyTorch loop: AdamW with the
  block's hyperparameters, the schedule, clipping, checkpoints) and the data
  file; `--conformance-entry train:K` replays `K` interpreter steps in PyTorch
  on the same batches and compares the parameters (tiny MiniGPT, 20 steps:
  worst relative gap 1.3e-10); `--mode graph` (the spec's tool-level mode,
  light form): `export.py` saves `model.pt2` and `model.onnx`, tested under
  ONNX Runtime; `make venv` adds onnx, onnxscript, onnxruntime.
- Step 5: `presk train` (crate `presk-train`): the training loop on the
  interpreter — initializer, data pipeline, objective and its `vjp`, the
  PSK-Core optimizer, schedule horizon, log (`--json`), checkpoints as
  `params M` files (`final.psk.d`); `--steps`, `--epochs`, `--data`, `--ckpt`,
  `--log-every`; `tests/train/mlp_regression.psk` converges; the tiny MiniGPT's
  20-step log is a golden; `presk run` on a training file exits 2 pointing to
  `presk train` (exit code 3 retired).
- Step 4: `stack{L}` lowered to a `fold` in PSK-Core (program size independent
  of `L`), interpreted as a loop and differentiated as a backward fold over the
  scanned accumulators; new primitive `set_slice`; `presk run|dump
  --unroll-stack` keeps 0.1.0's unrolled form; golden outputs unchanged; the
  gradient through a stacked module checked against finite differences.
- Step 3: the optimizer as PSK-Core (`adamw.init`, `adamw.lr`, `adamw.step`
  over `params M`): AdamW as PyTorch defines it, `constant`/`linear`/`cosine`
  schedules, global-norm clipping, decoupled weight decay; new primitive
  `sum_squares`; scalar-against-tree arithmetic in the interpreter;
  `presk run --entry adamw.step`; `--conformance-entry adamw.step` generates
  a test against `torch.optim.AdamW` (passes at rtol 1e-9); the block's name
  is the algorithm (`adamw` only, `E2010` otherwise).
- Step 2: the `training run` block typed against its objective: sample
  type = batch type without the `batch` axis, `batch(B = n)` binds `B`, the
  data file's type is matched when the file exists (its dimensions bind the
  objective's), `W0002` when it is missing; the `TrainingRun` record in the
  typed program, printed by `presk check --summary`; `tests/diag/train_*.psk`.
  The example `train_minigpt.psk` now has its data
  (`examples/minigpt/data/tokens.psk.d`, 510 windows of a 64-token stream);
  a file whose index bound is below the program's is accepted.
- Step 0: version `0.2.0-dev`; spec questions 28–34.

## 0.1.0 — 2026-09-19

The first version of the Presk compiler: the front-end, an arbitrary-precision
reference interpreter and a readable PyTorch export, built in eight steps
(`presk-specs/implementation/presk-plan-0.1.0.md`). The language is the one of
`presk-specs/examples/*.psk`, which are the acceptance tests and pass every
command below.

### `presk check`

- Lexer with Unicode identifiers (`θ`, `x̂`), ASCII synonyms of the five
  symbols (`Real`, `\`, `@`, `><`, `<<`), newline layout; recursive-descent
  parser with recovery; diagnostics rendered by ariadne, phrased in source
  terms, with codes (`E0xxx` syntax, `E1xxx` elaboration, `E2xxx` typing,
  `E3xxx` regularity, `E4xxx` effects, `E5xxx` lowering, `E6xxx`
  differentiation, `E7xxx` run, `E8xxx` export, `W0001`).
- Elaboration: imports, exact constants with `--const` overrides (reaching
  imported files), `dim` bounds, type aliases, generic modules with structural
  instances, parameter trees, `static assert` decided for all values in ℕ by
  an in-house procedure for the Presburger fragment (property-tested).
- Shape typing of function bodies: named axes, `...` prefixes, broadcasting,
  `idx`, `params`, `ein` with the `seq`/`seq2` relabeling rule, the four call
  argument forms, module members with implicit or explicit θ, `grad`, a
  builtin table; configuration blocks against schemas.
- Regularity classes per parameter (`C∞ < C⁰⁺ < Disc`) with the culprit
  named on `grad` errors; effects against `uses`.
- `--summary` (types, `let`s, classes, effects), `--json`.

### `presk run`

- PSK-Core: a dimension-polymorphic ANF IR with explicit parameter trees,
  unrolled `stack`, lifted lambdas, per-module initializers, structural
  verification.
- Reverse-mode automatic differentiation as a source-to-source
  transformation (`f.vjp`, `f.grad`), with structural adjoints of parameter
  trees; validated against central finite differences at 512 bits.
- Interpreter over MPFR (`rug`, `--bits`), tensors with named axes,
  reproducible randomness from `--seed`; `--entry`, `--dim`, `--const`,
  `--digits`, `--json`, `--verbose`. Golden outputs in `tests/run/`.

### `presk export --target pytorch --mode readable`

- One `nn.Module` per module, `forward` for `apply`, shapes as comments or
  `--shapes=check` asserts, `stack` as `nn.ModuleList`, `torch.func` for
  parameter trees and `grad`, a generated `test_conformance.py` fed with the
  interpreter's data (`--conformance-entry`). Verified against PyTorch 2.14
  in float64 (`make venv`, `make conformance`): `mse_grad`,
  `LayerNorm{8}.apply`, `MiniGPT.apply` and the training `loss` agree with
  the MPFR reference within rtol 1e-9.

### Also

- A tutorial chapter in the manual, with its programs in `tutorial/` and
  `make tutorial` running every command of it (found and fixed: file stems
  that are not Python identifiers, a stacked call nested in an expression,
  `dim` bounds not enforced by `presk run`).
- `presk dump --stage tokens|ast|elab|core|core-after-ad`.
- A rustdoc user manual (`make doc`), architecture decisions
  (`docs/decisions.md`), the questions sent to the specification
  (`docs/spec-questions.md`), and the tooling rationale
  (`presk-specs/implementation/presk-tooling.md`).

### Not in 0.1.0

E-graph rewriting and cost models, code generation, PSK-Sched/PSK-Kernel, the
`.pskk`/`.pskb`/`.pskt` formats, the Zig runtime, Lean 4 certification,
execution of `training run`, `--mode graph`, second-order differentiation,
`vjp`/`jvp`/`concat`/`dropout`/`clamp`/`max`/`min`.
