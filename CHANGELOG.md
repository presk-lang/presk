# Changelog

## 0.17.0 — 2026-10-29

The native backend at GCC on its own kernels, integers in the solver's
menu, the training step per stored value, and real weights
(`presk-specs/implementation/presk-plan-0.17.0.md`, D173–D182; the user's
decisions of 2026-10-28).
- The compiler's own code generator, and a second native backend through
  LLVM's `clang`, compute the native bits. On the course GPT the step takes
  0.994× the exported C program's time on Cranelift and 0.871× on LLVM
  (NB2, NB3).
- `presk solve --menu f64,f32,bf16,int8,int4` and `presk solve --step
  --grain value`, behind their flags: without them every solve is
  0.16.1's.
- A slice that keeps its axis, `concat`, `max`, `min` and `clamp`;
  `.psk.d` version 2; Llama 3.2 1B from its published weights.
- QM2, TS1 and TS2 fail; the other statements hold or are reported
  (`docs/rows-0.17.0.md`, the manual's `status`).

- Version 0.17.0 opened (D173, plan `presk-plan-0.17.0.md`, presk-specs
  !102). The native backend is to reach GCC on its own kernels: each
  kernel's share of the gap, the register tile unrolled in the lowering,
  and an LLVM backend at the C export's seam. `int8` and `int4` join the
  solver's menu beside the floats. The training step is budgeted per
  stored value. The language gains `concat`, a slice that keeps its axis,
  `min`, `max` and `clamp`, so that RoPE and Llama 3.2 1B's real weights
  run. Version `0.17.0-dev`; nothing in the compiler has changed yet.
- Each kernel's share of the native backend's gap to the C program (D174,
  `make kernel-shares`, `docs/kernel-shares-2026-10-28.md`). On ten X925
  the kernels are 1.64× GCC's on the forward pass and 1.90× on the step,
  and on one core clang's build of the same C is 1.32× faster than GCC's.
  `presk bench --calibrate --plan-only` times the plan's kernels alone,
  without timing them again under the other uniform realizations.
- The native backend's code generator (D175, the `native` chapter's "The
  code generator"). It changes no operation and no order, so X1's native
  outputs are `v0.16.1`'s byte for byte.
  - An output element its loop does not move lives in a variable.
  - Addresses are built by loop depth.
  - Loop headers branch to the exit first. Cranelift 0.131's elaborator
    hoisted nothing out of a loop whose branch named the body first.
  - Innermost loops are unrolled by four, by two or not at all, as the
    registers allow.
  - The lane-wise loops of a one-lane kernel run over four or two lanes,
    as GCC does to the exported C.
  - Loads of an input at affine indices outside a condition may move.

  On the course GPT under the plan of record (ten X925, steering figures),
  the kernels take 331.2 → 240.8 ms on the forward pass (GCC 202.1) and
  1 312.3 → 776.5 ms on the step (GCC 690.5).
- The LLVM backend, `--backend llvm` (D176, crate `presk-llvm`, the
  `native` chapter's "The LLVM backend").
  - Each kernel is printed as C by the C export's printer, with the native
    backend's lanes.
  - It is compiled by `make llvm`'s `clang -O3 -mcpu=native` with the
    export's other flags, kept in a cache (`PRESK_LLVM_CACHE`), and loaded
    with `dlopen`.
  - It runs on the native backend's host, pool and splits.
  - The bits are the native backend's.

  `presk bench --llvm` times the native rows, and `--calibrate`, on it, and
  `make llvm-conformance` checks the C export's programs against it. On the
  course GPT (ten X925, steering figures) its kernels take 201.6 ms on the
  forward pass and 683.6 ms on the step, against 202.1 and 690.5 ms for
  GCC's build of the exported C.
- One menu, `presk solve --menu f64,f32,bf16,int8,int4` (D177, the
  `precision` chapter's "One menu"). A parameter of rank two or more is
  offered `int8` and `int4`, under `absmax` and `pow2` scales, in the solve
  itself rather than in `--quant`'s pass after it.
  - Its term is `s · max |ŵ − w|`, the largest move of its values
    quantized, the form of a float type's term at rest (D155).
  - Its scale is constant along the axis its readers contract, else −2.
  - Every move of the solver runs over one list of rungs, and an integer
    decision is written in the realization's `quant` map.

  Without integer rungs nothing moves: X1, `make quant-table`, `make
  step-table` and `make precision`'s solve are `v0.16.1`'s. `presk explain
  --menu` gives an integer decision its rung and its axis. On the trained
  course GPT at 10⁻³ the default solve takes no integer, each `int8` term
  being about the budget; under `--narrow measured` it reaches 4.37×, and
  4.39× with `--margin 1`, as `--quant` does narrowed (steering figures).
- The training step by stored value, `presk solve --step --grain value`
  (D178, the `precision` chapter's "By stored value"): D52's method at the
  grain of the step's stored values, the optimizer's state kept in the
  default type, the realization in `storage`, which `presk train
  --realize` applies under either `--remat`. `--grain function`, J′'s
  solver, stays the default, and its outputs are `v0.16.1`'s. `presk
  calibrate --step` prices a realization of stored values by role. At
  10⁻³ on the course GPT at D = 64 the per-value realization is inside
  its budget by measurement and predicted within 0.11 % of J′'s.
- The language of plan 0.17.0 step 6 (D179, presk-specs !103, the
  reference at draft v0.25).
  - **A slice**, `e[a: lo..hi]`, which keeps its axis at the length
    `hi − lo`, on real and on index tensors; its bounds are dimensions
    `presk check` places inside the axis (`E2014`, `E1007`).
  - **`concat(axis a, x, y)`**, both operands carrying the axis.
  - **`max`, `min`, `clamp`**, element by element, a tie to the second
    operand, value and derivative.

  They go through every stage: the checker, PSK-Core and its adjoints
  (`slice_vjp`), the interpreter, the kernels (a `concat` and a slice are
  copies, `max` and `min` a comparison and a selection), the native, C
  and LLVM backends, the PyTorch export, the precision analysis and the
  printers. The specification's examples `rope/` and `concat/` run with
  the same bits on the kernel interpreter, the native backend and the LLVM
  backend in `f64`, `f32` and `bf16`, and join the test set
  (`tests/plan/entries.txt`). `coverage/llama.psk` takes RoPE and drops
  its learned position table: the Llama decoder checks with no
  substitution. The PyTorch export binds a dimension from an axis of
  length `T + 1`, passes a function the file's dimensions its arguments do
  not give, and its conformance test makes its tensors in `float64`.
- Real weights (D180, plan 0.17.0 step 7).
  - **`.psk.d` version 2**: each real leaf's encoding, `f64`, `f32` or
    `bf16`, in the header, read exactly. A file of `f64` leaves is still
    version 1, the same bytes. Every reader takes both: `presk-data`, the C
    runtime, `pskd.py` and the experiments' scripts.
  - **`coverage/llama3.psk`**: Llama 3.2 at 1B's configuration, every weight
    read whole.
  - **`experiments/llama/`**:
    - `hf_to_pskd.py` converts a Hugging Face checkpoint, reading
      `safetensors` by hand, with no permutation;
    - `tokens.py` is the tokenizer, kept outside;
    - `rw2.py` compares against `transformers` in `float64`;
    - `make llama3` runs the statements.
  - **`presk export --serve --dim T=n`** exports a model whose parameters do
    not fix the context's length (`E8009` without `--dim`). The context is
    padded after its last token, whose logits copies of the generation
    function read through an indicator (`presk_serve::at_position`).
    `generate.c` gives `presk serve`'s tokens over the whole, growing
    context.
  - **Three defects fixed**:
    - `presk serve` took every model as tied to the context's length, since
      an initializer's dimensions are all of its file's. It now reads the
      parameter tree.
    - The reference interpreter gave a parameter its argument's axis
      names, not its declared ones, as the kernel host does.
    - A checkpoint's parameters were first drawn in exact arithmetic and
      then replaced: about 80 bytes a number at 256 bits, so `presk run`
      on Llama 3.2 1B passed 88 GB and was OOM-killed. They are now
      described, and the checkpoint is read against a tree of zeros of
      their shape, built by the kernel backend with no executor. This holds
      for `presk run`, `export`, `bench`, `quantize`, `solve` and
      `calibrate`. The same run takes 21.7 GB, and X1's outputs do not
      move. The checkpoint's numbers are moved into the tree, not copied.
  - **`presk run --result-only`** prints the result alone, without the
    inputs that `--json` and `--verbose` print. A checkpoint of 1.2·10⁹
    numbers is then neither printed nor held in exact arithmetic to be
    printed.
  - **On `meta-llama/Llama-3.2-1B`'s published weights** (the
    `experiments` chapter, §25): presk's `f64` run is 6.2·10⁻¹⁵ from
    `transformers`' in `float64` (RW2 holds); `presk serve` and
    `generate.c` give the same 32 greedy tokens from four prompts in `f32`
    and `int8` (RW4 holds); against the exact reference at 128 bits `f32`
    is 2–4·10⁻⁶ off and changes no decision, `bf16` and `int8` 1–5·10⁻²
    and one of 64, `int4` with one scale per output column 0.4–0.6 and
    6 to 12 of 16 (RW3).
- The version's rows (D181, plan 0.17.0 step 8, `docs/rows-0.17.0.md`, the
  `experiments` chapter's §26), the timed ones alone by D127's protocol.
  - **The native code.** On the course GPT under the plan of record, on
    ten X925, Cranelift with the new code generator takes 1.19× the C
    program's time on the forward pass and 0.994× on the step (NB2
    holds; T4 was 1.63× and 1.67× at 0.16.0). The LLVM backend takes
    0.995× and 0.871× (NB3 holds). A kernel compiles through `clang` in
    37–48 ms, once (NB4).
  - **One menu.** Every one-menu solve is inside its budget by
    measurement (QM1). On V6 and the course logits at ε\* it gives
    0.16.1's realizations byte for byte, certified on the Code civil and
    Y1 on the Code pénal (QM3). On BERT-large's two blocks at 10⁻³ it
    takes no integer (2.00×) where `--quant` quantizes all ten matrices
    (4.93×): QM2 fails.
  - **The step by stored value** takes 7.80 s at course scale, 1.92×
    faster than all-`f64` but slower than J′'s 6.99 s (TS2 fails), most of
    it the optimizer's state kept in `f64`. The conversions take 3.1 % of
    its step and 12.5 % of H6's at D = 64 (TS3).
  - `experiments/pytorch_baseline.sh` takes `LLVM=1`, and
    `kernel_shares.sh` takes `BENCH=--llvm`; the default outputs do not
    change.

## 0.16.1 — 2026-10-23

A patch of the solver, after the certificate failed on a second text
(`presk-specs/implementation/presk-plan-0.16.1.md`, D168–D172; the user's
decisions of 2026-10-22).
- `presk solve` no longer offers a value of one element, nor a value whose
  first narrowing saves less than 10⁻³ of the entry's traffic
  (`--min-gain`).
- The rows those rules touch were solved again, and judged on three texts
  the model never saw.

- Two rules in the solver of stored values and parameters (0.16.1, D169,
  `presk-specs/implementation/presk-plan-0.16.1.md`). A value of one element
  is not offered. A value whose first narrowing, `f64` to `f32`, saves less
  than θ of the entry's all-`f64` traffic is not offered either
  (`presk solve --min-gain θ`, 10⁻³ by default). Both keep `f64`, and the
  nests that read them round them at their edge. `presk solve` names them
  in a line `rules`, and `presk explain` gives them their rule as their
  reason. On the course logits at ε\* the rules keep LayerNorm's ε, the head
  width and its root, LayerNorm's mean and the twelve vector parameters at
  `f64`, which 0.16.0's decisions had narrowed for 0.24 % of the traffic.
- The rows the rules touch, solved again (D170, `docs/rows-0.16.1.md`):
  the curves, V6, the course logits and `make precision`, on 0.16.0's
  envelopes, judged on the Code civil and the Code pénal. Every ratio of
  record moves by at most 0.03×. The certificate rows have no `bf16` left:
  V6 measures all-`f32`'s gap, the course logits half all-`f32`'s on the
  Code civil and 2.24·10⁻⁵ on the Code pénal. New script
  `experiments/rows_0161.py`; `margins_course.sh` takes `CURVE`.
- The certificate on a third and a fourth text (D171,
  `docs/further-texts-2026-10-23.md`, `make further-texts`): the Code du
  travail and the Code de procédure pénale, their typographic characters
  mapped (`second_corpus.py map`), 0.16.1's realizations and 0.16.0's
  beside them. With 0.16.1's realizations Y1 holds on both texts at both
  scales, where 0.16.0's fail it. Y2 fails at the course constants on both,
  at `f32`'s level (0.49× and 0.495×, and 2.07×). New script
  `experiments/further_texts.sh`.
- The certificate on a second text (D168, `docs/second-corpus-2026-10-22.md`,
  `make second-corpus`): the Code pénal from the course's container, on the
  Code civil's vocabulary (32 characters deleted), its first 461 windows
  through the Code civil's models, realizations and ε\*(10⁻³). At D = 64
  every certified realization stays inside ε\* and flips nothing, and every
  gap is within 0.54× to 1.31× (Y1 and Y2 hold). At the course constants one
  stretch of window 261 takes four certified realizations above ε\*, the
  logits solved at ε\* to 1.90 ε\*. LayerNorm's ε in `bf16` carries it. Two
  near-ties flip one or two decisions, and seven gaps move more than 2× (Y1
  and Y2 fail). New scripts `experiments/second_corpus.{sh,py}`; no code
  changes.

## 0.16.0 — 2026-10-21

The solver's defaults after D154–D156, `dropout` outside a training run as
the identity of the language, and every row of record solved and timed
again (`presk-specs/implementation/presk-plan-0.16.0.md`, D157–D167; the
user's decisions of 2026-10-19).
- `presk solve` charges each site the rounding its values take
  (`--rounding measured`), and prices one sample's activations.
- The kernel interpreter computes the native backend's bits, so the
  solver's measurement is the native gap.
- An entry compiled outside a training run has no dropout mask.
- `presk plan --mode training` plans a training run's own program.
- Three flags spend the budget out of sample, and none became a default.

Measured on every row of record (`docs/rows-0.16.0.md`):
- the measured term buys nothing on the logits once the masks are gone;
- the norms' gains of a model at its initialization go to `bf16` for
  nothing;
- the C program is 5 % faster without masks (T2 0.93×);
- the decoder block under its realization is faster than all-`f32` in
  every process.

- Version 0.16.0 (D167): R0 holds on the version's tree; `make test`,
  `check-examples`, `coverage`, `plan-check` and `c-conformance` pass; the
  manual, the report, the reference, the slides, the summary, a status note
  and the poster are at 0.16.0.
- `presk plan --mode inference|training` (D164): the plan of a training run
  is planned on the program it runs, its masks kept; since D160 the plan of a
  gradient that drops, planned on the inference reading, named nests `presk
  train` fuses into others, which it refused (`E7002`).
- The rows of record solved again at 0.16.0 (plan 0.16.0 step 5). The Llama 2
  decoder's block by D106's whole protocol (D162): the same 31 of 32 stored
  values, 1.99×; RMSNorm's three gains, ones at initialization, in `bf16`;
  the gap 2.544·10⁻⁵, the native one (D124's 2.639·10⁻⁵ was the kernel
  interpreter's `std` `exp`); the times within 1 % of D124's
  (`docs/compile-times-2026-10-19-llama.md`). The rows solved and not timed
  (D163, `docs/rows-0.16.0.md`): every one holds what it held, the curve at
  10⁻³ 2.04× (the masks gone, one sample's traffic), V6 1.97×, the quant and
  step tables byte for byte. D154–D156 run again (D165): the measured
  rounding term buys nothing on V6 once the masks are gone (1.96×, as the
  unit term), the tools' searches 2.00×, the solver's measurement the native
  gap to the digit; their report scripts count the space they read. The
  timings taken again alone (D166): Table 1 (`docs/compile-times-2026-10-20.md`
  and `-bert.md`), V5, Table 4 in five processes (T2 0.93×, T4 1.63× and
  1.67×) and H6 (1.02×).
- Three ways to spend the budget, judged out of sample (D161, `make spend`):
  `presk solve --estimate rss` (the proposal's estimate a root-sum-square,
  aimed at the budget over `--margin`), `--margin K` (2.5; the narrowings'
  threshold) and `--narrow heldout [--heldout M]` (early stopping on the
  samples after the envelope's). Every form holds out of sample on the
  logits at D = 64; none keeps 1.5× of the budget at 10⁻⁶ and none of the new
  ones keeps its realization from 4 to 16 windows, so they stay flags and the
  defaults are final.
- Outside a training run `dropout` is the identity of the language (D160,
  the compiler's question 78): an entry compiled outside a training run is
  its inference reading, with no `dropout_mask` and no product by it, so
  that no realization types a mask and the solver offers none. A training
  run compiles the program as written; under a realization solved outside
  training each mask takes its operand's type. The outputs under `f64` and
  `f32` are the same bits, the training path is byte-identical, and the C
  export loses its mask fills. `presk dump --mode training` prints the
  program as written.
- The kernel interpreter takes the `libm` crate's `exp`, `log`, `tanh`,
  `sin` and `cos`, as the native backend does (D159): the two executors
  now compute the same bits everywhere, and the solver, which measures on
  the kernel interpreter, measures what a native run computes. On U1's
  entries 48 of 186 kernel/native outputs differed and none does now.
- `presk solve --rounding measured` is the default (D158, the user's
  decision after D155 and D156); `--rounding unit` is the solver of
  0.12.0–0.15.0. The traffic cost is one sample's: `presk calibrate --data`
  records each site's elements for one sample (`per_sample`), and the
  solver prices the activations at them, the same at any number of
  samples. On the logits at D = 64 the default gives 2.12× at ε\*(10⁻³)
  and 1.36× at 10⁻⁶ from 1, 2 and 16 windows; at four windows one
  one-element decision differs at ε\*.
- Version 0.16.0 opened (D157, plan `presk-plan-0.16.0.md`, presk-specs
  !97): the solver's defaults after D154–D156, `dropout` outside a
  training run as the identity of the language (question 78 answered: an
  entry's mode is part of its meaning), the kernel interpreter on the
  `libm` crate, the traffic cost per sample, three ways to spend the
  budget judged out of sample, and every row of record solved again.
  Version `0.16.0-dev`.
- The certificate off-sample against the size of the calibration sample
  (D156, `docs/certificate-off-sample-2026-10-19.md`, `make off-sample`).
  The envelope of the logits at D = 64 is calibrated on the first 1 to 32
  held-out windows and on all 461, the solver run on each in three forms
  at ε\*(10⁻³) and 10⁻⁶, and every realization judged on the windows it was
  not calibrated on. No code changes.
  - *The ratio.* The solver of record's realization is 1.5× to 3.8×
    further from the exact logits out of sample than in, 2.3× at the four
    windows of record and still 1.9× at 32. Its decisions are the same from
    4 to 32 windows.
  - *The narrowing.* `--narrow measured` fails out of sample at 10⁻⁶ from
    every sample size: it needs a held-out check.
  - *The certificate.* No realization inside ε\* flips a decision out of
    sample.
- The estimate's rounding term measured on the envelope, and a narrowing
  step decided by measurement (D155, `docs/solver-rounding-2026-10-19.md`,
  `make solver-rounding`). Both are off by default; every row of record was
  solved without them.
  - *The envelope* records each site's largest rounding error per storage
    type, `max |fl_d(x) − x|` over the sampled values, and `presk calibrate
    --precision` and `presk explain` print it beside the magnitude.
  - *`presk solve --rounding measured`* charges a site that rounding
    instead of `u(d) · max|x|`, so that a value exact in a type costs
    nothing there. On the logits at ε\*(10⁻³) it takes the solver from 1.97×
    to 2.14× in one sweep and one measurement, holding on the 461 held-out
    windows.
  - *`--narrow measured`* narrows each decision one step after a passing
    measurement and keeps it while the measured gap fits. It finds D154's
    searches' 2.17× in 79 measurements, and overfits at 10⁻⁶ as they did.
- A precision-tuning tool on the exported C (D154,
  `docs/precision-tuning-2026-10-19.md`, `make precision-tuning`). CRAFT's
  variable mode ran on the `f64` C program of the logits at D = 64, with a
  script retyping its declarations (`experiments/tuning/retype.py`), under the
  paper's criterion: the logits within ε\*(10⁻³) of the exact outputs the
  export ships. It finds `double`/`float` programs that pass and hold on the
  461 held-out windows, at most 1.95× less traffic as a realization against
  the solver's 1.97×: it has no `bf16`, and the parameters are its inputs.
  Precimonious's and CRAFT's searches, reimplemented on presk's realizations,
  find 2.17× in 70 to 2 858 trials, holding out of sample. Most of what they
  add are dropout masks of ones in `bf16`, which the solver's estimate charges
  as if they rounded. At 10⁻⁶ they overfit their two windows, where the
  solver's realization holds. New scripts, `experiments/tuning/`, the tools
  fetched into `.tuning`; no code changes.
- PyTorch compiled and in mixed precision, beside eager (D153,
  `docs/pytorch-modes-2026-10-18.md`, `make pytorch-modes`): the exported
  course GPT run by `torch.compile` with its defaults and by AMP in
  `bfloat16`, by D126's protocol on ten X925 cores. Compiled, PyTorch's
  training step is 0.57× eager's, so the C step is 1.80× it where it is
  0.98× eager's (T2), and 2.07× like for like, neither side drawing dropout
  masks. AMP's `bfloat16` is 4.0·10⁻² from the exact logits at D = 64 and
  flips 505 of 118 016 decisions, as the compiler's all-`bf16` does. It
  costs 1.4·10⁻³ of validation loss, above the curve's realizations
  (1.3·10⁻⁴) and inside the spread of one dropout draw. A new script,
  `experiments/pytorch_modes.{sh,py}`; no code changes.

## 0.15.0 — 2026-10-18

`presk memory`: what a program needs in memory, counted from the program
and its realization rather than estimated
(`presk-specs/implementation/presk-plan-0.15.0.md`, D147–D152; the user's
request of 2026-10-18). The host is recorded as the C export records it,
each buffer live from its first write to its last read. The report gives:
- the parameters at their stored types;
- the peak of live bytes and each call's share of it;
- the C program's arena beside it, which is 1.3 to 3.1 times the peak;
- a training step's terms under both `--remat`s;
- the same as polynomials in the dimensions;
- an overlay of the Network tab.

Judged on six statements fixed before the work:
- nothing else moves (M0, holds: X1 by its version and commit lines
  alone, the tables byte for byte, the tutorial but for its measurements,
  `presk --help`'s line and section 13; `explain.json` by its `format` (4)
  and its `memory` object);
- one computation per number (M1, holds): the arena is the C export's,
  and a straight-line C program placing each buffer by its life computes
  the native bits on the 285 programs of `make c-conformance` and on the
  decoder block at the 7B width;
- a training step's terms sum to its peak (M2, holds);
- the polynomials count what a recording counts at three more sizes (M3,
  holds, on 14 programs);
- the page's panel counts the compiler's numbers and answers under 200 ms
  with the overlay on (M4, holds);
- the numbers at scale (M5, stated). The course GPT at its course size
  needs 3.76 GB for a training step storing its forward pass, 3.24 GB of
  it kept for the backward, and 0.92 GB recomputing it. The decoder block
  at the 7B width peaks at 57 MB of activations beside 1.86 GB of
  parameters, and its C arena is 3.06 times that peak.

- Plan 0.15.0, `presk memory` (D147): what a program needs in memory,
  computed from the program and its realization (the parameters in their
  stored types, the peak of live bytes and each box's share of it, a
  training step's terms), as numbers, as a polynomial in the `dim`s and as
  an overlay of the Network tab; version `0.15.0-dev`. Found: the C
  export's arena is a stack of frames, 1.3 to 2.5 times the peak of live
  bytes on the course GPT, so the report prints the peak with the arena
  beside it; and no entry is a training step, which `presk train` makes
  from three calls on one backend.
- `presk memory` (plan 0.15.0 step 1, D148): what one call of an entry
  needs in memory, from the program and its realization. It prints the
  parameters at their stored types, the inputs, the peak of live bytes
  (each buffer from its first write to its last read, over the host the C
  export records) and its operation, the outputs, the C program's arena
  beside them (`presk export`'s own, 1.3 to 2.5 times the peak on the
  course GPT), and each call's share of the peak and its own peak, by the
  network diagram's boxes. `presk explain` prints the same in a section
  13, *Memory*, and `explain.json` is format 4, with a `memory` object;
  the page refuses format 3. The hidden `presk export --target c
  --straight-line --placed` puts every buffer at an offset by its life,
  and `make c-conformance` checks that it computes the native backend's
  bits on every program it exports, with the arena `presk memory` prints
  equal to the export's (M1 holds).
- `presk memory --step` (plan 0.15.0 step 2, D149): the training run's
  step as `presk train` makes it, from its three calls on one backend
  (`loss.value_vjp`, `adamw.lr`, `adamw.step`), the parameters and AdamW's
  moments held across them, under `--remat store` and `recompute` side by
  side: the parameters, the moments, the gradient, what is kept for the
  backward, each call's phase and the step's peak, whose terms sum to it
  (M2 holds). `presk explain` prints it in section 13 when its entry is
  the training run's objective.
- `presk memory --symbolic` (plan 0.15.0 step 3, D150): the inputs, the
  outputs and the peak as polynomials in the entry's dimensions, the
  `const`s as their numbers, each buffer's bytes interpolated exactly from
  recordings on a grid (every axis a linear form in the dimensions) and
  checked off it; the peak the largest of the candidates no other
  dominates; written by the printer of the denotation. The course GPT's
  forward pass at `D = 64` peaks at 4 608·B·T + 8·T² until `T = 75`, then
  at its attention. M3 holds.
- The memory overlay of the Network tab (plan 0.15.0 step 4, D151): each
  box coloured by its share of the peak of live bytes, its share on its
  header, and a panel with a slider per dimension and a stored type per
  parameter group, the page counting again from each buffer's polynomial
  (M4 holds); `presk dump --stage network --memory` for the SVG and the
  TikZ. An executor that runs no kernel no longer draws its parameters'
  random values: the kernel listing of the decoder block at the 7B width
  takes 2.7 s, where it took 21.7 s.

## 0.14.1 — 2026-10-17

The exploded view of the network diagram, by a compositional layout
(`presk-specs/implementation/presk-plan-0.14.1.md`, D146; the user's
choice of 2026-10-17). Each box's graph is laid out once, in its own
frame, with its children folded, and an opening composes the frames, so
that any set of boxes may be open: an open box takes its composed size,
the items after it in its layer move right and the layers below move
down, nothing being reordered. The page opens and folds a box without
touching its siblings, "expand all" opens every box, and `presk dump
--stage network --level all` draws the exploded view. The graph, its
numbers and the provenance table do not change. Judged on six statements
fixed before the work:
- nothing else moves (E0, holds but in one part: X1, the tables and the
  tutorial, by their version lines and measurements; `explain.json` by
  its `format` and its positions; but the entry alone and a box alone,
  which were to keep 0.14.0's positions, move by the fix of a defect
  below);
- the openings compose: a box with no child open is drawn as its frame,
  translated, in every opening, the layers and their order kept, no two
  siblings overlapping, every child inside its box, every arrow meeting an
  open box at its port (E1, holds, on 36 diagrams in every chain opening,
  the levels, the exploded one and 20 random sets);
- any set of boxes opens in the page, and a click on any box of the
  exploded view lights the columns (E2, holds);
- one rule: the page's composition is the compiler's to the tenth of a
  point, and the SVG and the TikZ draw it (E3, holds);
- 0.14.0's N1 to N4 still hold, the report's two figures on A4 at 0.74
  and 0.68 (E4, holds);
- with every box open, the page answers within P3's 200 ms on the decoder
  block at the 7B width: the exploded opening drawn in at most 4.0 ms
  (E5, holds).

- `explain.json` format 3: the `network` object's `frames` (one per box,
  each child's layer and rectangle, each arrow's places in the layers it
  crosses), `default` an open set, and `views` the compiler's composition
  of the default and the exploded openings, which the page checks itself
  against. The page refuses formats 1 and 2.
- A defect of 0.14.0's layout, fixed: an arrow that crossed several layers
  kept one place for all of them, and could run through a box (1 183
  cases on the 36 diagrams). An arrow now has a place in each layer,
  leaves its source straight down to its layer's bottom, crosses each
  layer straight down and bends between layers. The report's figure 9.2
  (`Block` alone) moves with it, from 693 to 705 pt wide.
- The page writes the open set into the address (`net=`) at each opening.
- The Network tab lights the box that holds a selection. A selection
  inside a folded box, as the lesson's traced sites at the default
  opening, lit nothing drawn and left the whole diagram dimmed; it now
  lights that box's header, and the diagram is dimmed only around
  something lit. The diagram's contrast is raised in both themes: its
  labels darker, the arrows at 0.95 and at least 1.6 wide, a box's title
  at 12 px bold (narrowed when its box is too narrow), a box without
  parameters bordered in the muted ink, the dimmed at 0.35. The page
  alone changes: the JSON, `network.svg` and the TikZ are as at 0.14.0.

## 0.14.0 — 2026-10-17

The network diagram: the program drawn as boxes and arrows in a second
tab of the pipeline view, read off the provenance table
(`presk-specs/implementation/presk-plan-0.14.0.md`, D140–D145; the
reference at draft v0.19). A box is a call, opening in place to its
callee's graph at that call, a stack of layers one box; an operation node
a loop nest; an arrow a value; each an element of the table, so that a
click selects it in the columns. The boxes carry the printer's formulas,
their parameters and the cost model's W and Q, the arrows their shapes,
types and bytes. The compiler lays it out, once per opening a reader can
reach, and the page, `network.svg`, `presk dump --stage network` and a
TikZ figure draw the same positions. Nothing a program computes changes.
Judged on six statements fixed before the work:
- nothing else moves (N0, holds: X1, the tables and the tutorial, every
  difference a version line or a measurement; `explain.json` by its
  `format` and `network` alone);
- every box, node and arrow is an element of the table, and a click on it
  lights its thread in the columns (N1, holds, on 36 diagrams and in a
  browser);
- the numbers are exact: the parameters, each arrow's elements, every
  opening's W and Q (N2, holds);
- a box's formula is the printer's (N3, holds);
- one set of positions, the course GPT at level 2 on one A4 page (N4,
  holds, its labels printing small);
- the page answers within P3's 200 ms on the decoder block at the 7B
  width (N5, holds: at most 20 ms).

- Plan 0.14.0, the network diagram in the pipeline view (D140): the
  program as boxes and arrows in a second tab of the page, drawn from the
  provenance table (a crate `presk-diagram`, JSON format 2, the Network
  tab, `network.svg`, `presk dump --stage network`, a TikZ figure);
  version `0.14.0-dev`. Found: one PSK-Core function serves every
  instance of its module, so a box's own element is its call; the
  envelope keeps a site's largest count of elements over its calls; and
  the launch log holds no call above a launch, which W and Q per box
  need, so it gains the path of calls (the plan's question 1, the user's
  answer).
- The network diagram (D141, plan 0.14.0 step 1): crate `presk-diagram`
  draws the program the kernels run as boxes and arrows read off the
  provenance table, with the printer's formulas, each arrow's shape, type
  and bytes, each box's parameters and the cost model's W and Q over the
  launches inside its call, laid out top to bottom once for every opening
  a reader can reach. `explain.json` is format 2, with `network`;
  `presk explain --format json|html` also writes `network.svg`. The kernel
  backend's launch log records the calls above each launch
  (`Launched::calls`). Found: `explain.json` under the native target is
  not the same from one run to the next, since a kernel that calls a
  `libm` function holds its address in its machine code (0.13.0 step 5).
- The Network tab (D142, plan 0.14.0 step 2): `explain.html` draws the
  program as boxes and arrows beside the columns, a box opening in place,
  a click selecting in the columns; "labels", "keys", "backward"; the
  lesson records the tab. A gradient is drawn as a value, not as a
  parameter.
- The native code's bytes before linking (D145, the user's answer on
  presk !100): `explain.json` keeps each kernel's machine code as
  Cranelift emitted it, the fields the linker fills with a `libm` address
  zero and named (`relocs`), and is the same from one run to the next; the
  code a run executes is unchanged.
- The exports (D143, plan 0.14.0 step 3): `presk dump --stage network`
  (the outline, `--format json|svg|tikz`, `--level N`, `--standalone`);
  `make network-figure` writes the course GPT at level 2 into the
  manual's new chapter `explain_network`, and level 1 and `Block` alone
  (`--box`) into the report's two figures, each readable on paper; the
  SVG and the TikZ draw the JSON's positions (N4).

## 0.13.0 — 2026-10-17

The pipeline view: one compilation drawn as columns, one per stage, from
the source to the machine code, every element linked to what it was made
from (`presk-specs/implementation/presk-plan-0.13.0.md`, D133–D139; the
reference at draft v0.18). A provenance table, filled by the stages as
they run with the IRs unchanged, makes the links; `presk explain` reads
its traces off it, writes it as JSON, and draws it as a page with each
element's equations in the reference's notation and each kernel's machine
code by line. Nothing a program computes changes, but for one refusal:
`--opt` with a realization that types sites. Judged on six statements
fixed before the work:
- nothing else moves (P0, holds: X1, the tables and the tutorial, every
  difference a version line or a measurement);
- `presk explain`'s traces read off the table print what 0.12.0 printed,
  every difference named (P1, holds: two rows corrected);
- every element of every stage has an origin (P2, holds);
- the page answers a select or a fold in under 200 ms on the Llama 2
  decoder block at the 7B width (P3, holds: 7 and 44 ms);
- every byte of every kernel's machine code lies in one range of one
  line (P4, holds, 88 % of the bytes with Cranelift's own line);
- the printer writes the reference's equations (P5, holds, checked by
  the user).

The entries below since 0.12.0 also hold the paper's measurements on
0.12.0 (D126–D132), which came before the plan.

- The user's decisions after 0.12.0 (D126). A parameter's sensitivity stays
  the exact 1-norm of its leaf of the cotangent, and `--quant` stays a pass
  on top of the float assignment; one menu for both kinds of type is a
  design for later (`docs/precision-menu-roadmap.md`). Every timing of
  record runs each row in several processes, at least three, and reports
  the median across them and the fastest, since a process may run its
  kernels in a slower mode for its whole life (`docs/kernels-roadmap.md`).
  No code changes.
- The course GPT's timings across five processes, by that protocol (D127,
  `docs/retime-2026-10-15.md`). The C training step is 0.98× PyTorch's (T2
  holds). The native backend is 1.52× and 1.60× the C program (T4 fails).
  The solved step is 1.75× all-`f64` (J′ holds). Each figure is the median
  of five processes and within 0.01 of its one-process figure, and no
  process ran in a slow mode on these rows.
  `experiments/retime_processes.py` combines the rounds.
- Where the measurement changes the solver's decision (D128,
  `docs/widening-2026-10-16.md`, `make widening`): an unshifted softmax
  (overflow), the two-pass standard deviation (a mean whose sensitivity is
  zero) and a sum of equal terms (correlated roundings) make the solver
  widen; the softmax primitive, the one-pass formula and noisy terms do
  not. The widening reaches the misjudged value last. Fixed: an entry
  whose first argument is a tensor no longer has it offered as a
  parameter, which no executor runs and every measurement refused.
- PyTorch's own gap to the exact model (D130,
  `docs/pytorch-gap-2026-10-16.md`, `make pytorch-gap`): the course GPT at
  D = 64 exported and run by PyTorch eager in `float32` is 2.95·10⁻⁶ from
  the exact logits on the 461 held-out windows, flips no decision and lies
  158 times inside ε\*(10⁻³); the compiler's all-`f32` is 3.12·10⁻⁶.
- The course GPT trained at its course constants (600 steps in `f64`, 4 h)
  and taken through the margins (D129, `docs/margins-course-2026-10-16.md`):
  validation loss 2.3307, accuracy 0.3037, ε\*(10⁻³) = 4.96·10⁻⁴; the
  curve's realization at 10⁻³ (1.98×, eight parameters in `bf16`) and the
  one solved on the logits (1.97×) flip none of 118 016 decisions, and
  statement S holds. New scripts `experiments/curve_course.sh`,
  `margins_course.sh`, `margins_rechunk.py`, `margins_loss.py`,
  `val_loss.sh`.
- A contraction's bound proved in Gappa (D131, `docs/gappa-dot-2026-10-16.md`,
  `make gappa-dot`): one element of `[8, K]·[K, 8]` in `f32` at K = 8 to 128,
  sequential and `interleaved(n)`; the proof stays 2.6 to 3.1 times `cond`
  under the sequential order, flat in K, 3.3 to 6.0 times interleaved.
  `experiments/gappa_dot.{psk,py,sh}`; Gappa 1.6.0 built from source on
  falicon (the manual's dependencies).
- The curve and the envelope-stability figure on the trained weights (D132,
  `docs/envelope-state-2026-10-16.md`): calibrated on the trained model and
  two held-out windows instead of the initialization and random tokens,
  the curve narrows more (D = 64: 3.17× to 1.15×; course: 2.68× to 1.03×),
  never widens, and its 10⁻³ realization flips 30 and 29 decisions on the
  logits, outside ε\*; statement S holds under both envelopes; the trained
  model's stability factor is 1.25×; the quality axis stays flat (every
  realization trains to within 1.3·10⁻⁴ of all-`f64`).
  `experiments/curve_trained.sh` (`QUALITY=1` for the quality axis),
  `curve_course_trained.sh`.
- Plan 0.13.0, the pipeline view (D133): a provenance table every stage
  appends to, `presk explain --format json`, the page, the math printer
  and the native column; version `0.13.0-dev`. Found under `--opt`: the
  optimizer renumbers every `let`, so a realization that types sites
  types nothing it names (refused from step 1), and `presk explain`'s §12
  names no kernel for the sites it traces (rebuilt over the table in step
  1).
- The provenance table (D134, crate `presk-provenance`): every element of
  every stage of one compilation, from a span of the source to a function
  of `kernels.c`, and for each one what it was made from and how (the
  lowering's rule, the derivative rule an adjoint is made by, unchanged or
  rewritten by the optimizer, the fusion group, the kernel the nests
  share). The IRs do not change: the lowering, differentiation, the
  optimizer, the kernel backend and the C export return their links beside
  their values, and `presk_driver::provenance` puts them together.
  `presk explain` §12 reads it: under `--opt` it follows each site to the
  `let` the optimizer made of it, and an `add` a contraction's kernel
  computes as its epilogue is no longer said to be computed by the host.
  `--opt` with a realization that types sites, or a plan's layouts, is
  refused (`E7002`), each site named with its new name; `presk explain`
  refuses `--opt` with `--budget`.
- `presk explain --format json` (D135): `explain.json` beside `explain.md`,
  the document's facts and the provenance table as values: the document's
  sections (from which the Markdown is rendered again byte for byte), each
  stage's elements with their facts (the envelope's magnitude, the
  sensitivity, the type and its term, the layout, the kernel's text, the C
  function's lines), the links and the sources. Deterministic without
  `--measure`; its schema is in the manual's `explain` chapter.
- The pipeline view (D136): `presk explain --format html` writes
  `explain.html`, one offline file with the JSON inside it. The stages are
  columns, every element is linked by an arrow to what it was made from,
  and an unchanged element passes through. Functions fold, with §12's sites
  open and lit; selecting lights a thread through every stage; hovering
  gives the facts and the reasons; a lesson of captioned selections lives
  in the page's address.
- The printer of the denotation (D137): `presk dump --stage elab
  --format latex|unicode` prints each function of a program as the
  equations of the language reference's chapter 3, whose new section 3.10
  states the notation (presk-specs !85, draft v0.18): an operation by its
  name, `ein` as its sum at an index named by its axes, a stack as its
  recurrence. `--stage core-after-ad` prints PSK-Core after differentiation,
  one equation per binding, each generated binding with what it was made
  for. `--module`, `--function`, `--standalone`; `make math-check` compiles
  the documents. The pipeline view shows the same formulas on its cards,
  with a derivative's rule and a rewrite's identity on the links, a site's
  term of the estimate and a contraction nest's order of summation;
  KaTeX 0.19.0 is inlined in the page.
- The native column (D138): `presk explain --format json|html` with the
  native target writes each kernel's machine code, compiled as a run
  compiles it, every byte with the line of the kernel it compiles (from
  Cranelift's `SourceLoc`s; the register allocator's moves, to which
  Cranelift gives none, take the line before them), and Cranelift's text
  of the instructions. The page's last column shows it. A run's code does
  not change.

## 0.12.0 — 2026-10-15

A parameter's floating-point storage type at rest, `params` (`f32` or
`bf16`, beside `quant`): rounded once when the entry is given it, read in
place by a nest of its type and widened in the loads of a wider one, with
no copy at a kernel's edge, on the three executors and in the C program
(`presk-specs/implementation/presk-plan-0.12.0.md`, D119–D125; the
reference at draft v0.17). The solver chooses it, by path, with the
functions' results, which it now ranks, and takes an entry whose result
is a tensor under a budget on its largest elementwise gap
(`--budget-abs`). Judged on six statements fixed before the work:
- no output of a realization that types no parameter changes outside the
  solver (V1, holds, every difference named);
- the three executors agree on typed parameters (V2, holds);
- no typed parameter is converted at a kernel's edge (V3, holds, at the
  7B width too);
- the solves again, the weights counted (V4, reported: the Llama 2
  decoder block 31 of 32 stored values and 13 parameters at 1.99×, every
  gap inside its budget);
- the decoder block under its solved realization no slower than all-`f32`
  at batch 1 (V5, holds: as fast, and 3.5 to 4 times faster than under
  0.11.0's);
- the course GPT's logits solved at ε\*(10⁻³) (V6, reported: 7.5·10⁻⁵,
  no decision changed).

Found by D120: a `bf16` nest rounds its wider operands to `bf16`, where
the reference read them in `f32`. The user decided on 2026-10-15 to amend
the reference to what the executors do (its question W19, closed); no
output moves.

- Version 0.12.0 opened (D119, 2026-10-13): a parameter's floating-point
  storage type at rest, the functions' results offered by the solver, and
  a budget on an entry whose result is a tensor; version `0.12.0-dev`
  until step 6.
  `experiments/pytorch_decoder.sh` times further realizations of the
  decoder block under their plans (`REALIZATIONS`, `ROWS=realizations`),
  with the host by phase at each batch size: V5's baseline, 0.11.0's
  solved realization, measured.
- A parameter's own storage type in a realization record, `params` (D120,
  plan 0.12.0 step 1; the reference's chapter 4 at draft v0.17): a path of
  the parameter tree to `f32` or `bf16`, beside `quant`. Parsed, written
  only when not empty, in `is_uniform` and `label` (`+Nparams`); checked
  where the parameters are loaded (`Realization::check_params`, `E7001`:
  a type other than `f32` or `bf16`, a path `quant` names too, a path that
  is not a parameter). `presk train` and the steps of `calibrate --step`
  and `solve --step` refuse it.
- Every executor and the C export run `params` (D121, D122, plan 0.12.0
  steps 2 and 3): a parameter is stored in its type once, when it is
  loaded (`Realization::store_params`, `stored_params`, `realize_args`;
  `quantize_args` and `quantized_params` renamed), and a kernel of a wider
  type reads it from an input marked `BufKind::Wide`, its loads widening
  each element exactly, natively in lanes, with no copy at the kernel's
  edge. The C program keeps the weight in its type in the arena. `presk
  explain` lists a given realization's typed and quantised parameters.
  Statement V2 holds on the three executors (each gives the bits of the
  parameters rounded once; C1 on 48 programs, C5), V3 as a test.
- `presk solve` offers the functions' results and the parameters, and takes
  an entry whose result is a tensor (D123, plan 0.12.0 step 4):
  - *results:* the interpreter records the 1-norm of a `.vjp`'s cotangent
    argument, the adjoint of the function's result, so a result is ranked
    and offered like any stored value (`presk dump --stage sensitivity`
    lists them);
  - *parameters:* decided by path into the realization's `params`, ranked
    by their cotangent's 1-norm per leaf, charged at the narrowest of their
    type and the types of the nests that read them, which
    `presk_precision::solve::param_reads` follows from the entry; their
    bytes at rest are in the traffic; `--quant` stays a pass on top;
  - *tensor entries:* `--budget-abs EPS` beside `--budget`, on the largest
    elementwise gap; one sweep seeded with ones, read per output element,
    ranks; the measurement decides;
  - *the bound* charges the parameter tree at its narrowest type;
  - *`presk explain`'s §6.3* lists the parameters among the decisions and,
    under a given realization, every stored value with its type, the
    default's included, and every parameter.
- The measurements of 0.12.0 (D124, plan 0.12.0 step 5), alone on
  falicon, on the version's code:
  - *V1 holds:* no output of a realization that types no parameter
    changes, outside the solver; every difference is named;
  - *V2 and V3 hold* as tests, and V3 at the decoder's 7B width: under
    its solved realization, no parameter byte is converted at an edge;
  - *V4:* the solver narrows the results and types every parameter. The
    tutorial goes from 7 to 11 stored values; the course GPT from 29 to 33
    at 2.12× with the weights counted; the curve runs 2.78× to 1.08×;
    BERT-large's two blocks get 28 stored values and 23 parameters at
    2.00×; the decoder gets 31 of 32 and 13 parameters at 1.99×, in 71.7
    GB. Every gap is inside its budget.
  - *V5 holds:* the decoder block's forward pass under its solved
    realization is no slower than all-`f32` at batch 1, and 3.5 to 4
    times faster than under 0.11.0's;
  - *V6:* the course GPT's logits solved at 4.67·10⁻⁴ absolute, 1.97×,
    7.5·10⁻⁵ measured, no decision changed;
  - `experiments/curve.sh` trains under each solved realization less its
    `params`, since a training run keeps its master weights in the default
    type;
  - the tutorial quotes the new solve; the solves' times are in
    `docs/compile-times-2026-10-13.md` (BERT-large) and
    `docs/compile-times-2026-10-14-llama.md` (the decoder).
- The checks of record and the documents (D125, plan 0.12.0 step 6):
  version `0.12.0`; `make test`, `check-examples`, `coverage`,
  `plan-check` and `c-conformance` pass on the version's tree; the
  manual at 0.12.0 (its figures, `training`'s refusal of `params`); the
  `explain` snapshots and the manual's `explain_loss` at the version;
  `cargo doc`'s two warnings, links to private items since 0.11.0, fixed.
- ε from the margins (D118): `experiments/margins.sh`, `make margins`, and
  `experiments/margins.psk`, the course GPT's `logits` and `decisions`
  imported from `corpus/nlp/05_gpt.psk`, which is unchanged.
  - *What it runs:* the exact logits at 128 bits on the 461 held-out
    windows of `make curve`'s split; the margins and ε\*(δ) with the
    Clopper–Pearson correction; then each of the foil's and the curve's
    fourteen realizations natively, its gap on the logits, its flips
    against the exact decisions and the margins' bound. The report is
    `docs/epsilon-2026-10-12.md`, the tables `docs/epsilon/`.
  - *What it finds:* ε\*(10⁻³) = 4.7·10⁻⁴. Five of the curve's
    realizations are within it, statement D's changing one decision of
    118 016 where its margins allow 70, and every realization changes at
    most as many as its margins allow.
  - *No compiler change.*
- `presk run --json` prints, as `inputs`, the arguments the entry ran on:
  with `--data` the data file's samples, with `--weights` the checkpoint's
  parameters as read, where it printed the ones it had synthesized (found
  by D118; `tests/run_json_inputs.rs`).

## 0.11.0 — 2026-10-10

`presk explain FILE --entry E … --out DIR` writes one compilation of one
entry as a document: every stage's representation from the source to the C
program, every decision with what decided it, and on request what a run
measures (`presk-specs/implementation/presk-plan-0.11.0.md`, D109–D117).
It is Markdown, its PDF a make target. Beneath it, a driver,
`presk-driver`, takes the orchestration out of the command-line tool, so
that the commands and the document read the same values; it fixes D108's
two CLI defects. The solver offers only the values the program stores,
since the other sites' types change no bit. Judged on six statements fixed
before the work: no output changes but those named (X1, holds: 37 files of
X1's runs, each named); D108's findings fixed (X2, holds); the document's
figures are the commands', and two runs give the same bytes (X3, holds);
the kernel listing from the shapes, the course GPT's in under 10 s (X4,
holds: 0.18 s); the decoder block at the 7B width explained from D106's
files in under 5 minutes and 40 GB (X5, holds: 40 s, 10.9 GB); the solver's
sites before and after (X6, reported: the course GPT 69 → 29 sites,
2.09× → 2.14×, the same measured gap).

- Step 0 (D109): the plan, the statements X1–X6, the facts; version
  `0.11.0-dev`.
- Step 1 (D110): the crate `presk-driver`, the pipeline as values that every
  command reports through; `presk check --precision` binds a module entry's
  own arguments (D108, finding 1: the bound on `LayerNorm{8}.apply` was 93×
  its rule's), and `presk calibrate --precision` refuses `--samples` without
  `--data` (finding 2). Every other output is byte-identical. The launch
  capture, `presk plan`'s choice and the C export moved into the driver too.
- Step 2 (D111): `presk dump --stage kernel` lists an entry's kernels from
  its shapes, nothing computed (the course GPT's forward pass at `B = 16`,
  `T = 256` in 0.18 s, where the kernel interpreter ran it for minutes),
  the same listing; the driver gives a point's terms per kernel, the
  solver's breakdown per site (`presk_precision::solve::breakdown`) and the
  sensitivities' sweep as a program.
- Step 2b (D112): `presk solve` offers the values the program stores, and
  nothing else: a parameter's field, a projection, a call's or a fold's
  result and a fused member change no bit when typed (the reference's
  chapter 4), and are no longer offered, counted or priced; their rounding
  is charged where it happens. `presk check --precision` charges a fused
  member at its root's type. The course GPT's solve at 10⁻³: 29 sites, not
  69, at 2.14×, the same measured gap.
- Step 3 (D113): `presk explain FILE --entry E … --out DIR`, one
  compilation as a document (crate `presk-explain`): every stage's
  representation, the solve with each site's role and term of the estimate,
  the bound, fusion, the layout plan, the kernels under their sites, the
  host and the targets, each section with the commands that print it;
  deterministic, and every line of the commands it quotes found in it
  verbatim (X3), what they print being one function each in
  `presk_driver::text`. `presk solve --profile` no longer calls the
  exhaustive search's time bytes, nor prints its ratio as 0.000x.
- Step 4 (D114): `presk explain` gives every decision its reason: each
  site's type from the solver's moves (`presk_precision::solve::Move`,
  narrowed, kept by the budget, kept for no saving, widened after a
  measurement), where each fusion group ends and why
  (`presk_core::fusion::stops`), each nest's chosen layout beside its
  runners-up and its best split with every prediction's terms, the
  optimizer's rewrites before and after, the static assertions and how
  they were decided. No output of the CLI changes.
- Step 5 (D115): `presk explain --measure` runs the entry on this machine
  (the native output against the exact reference, a call, each kernel
  beside its prediction, the host by phase) and says it did not reserve
  it; `--site` traces a site through every stage, by default the largest
  shares of the estimate and the slowest predicted nests, and an index of
  every site goes to `listings/`. `presk bench`'s three measurements are
  `presk_driver::bench`'s, its output unchanged.
- Step 6 (D116): `make explain` and `make explain-pdf` (pandoc and
  LuaLaTeX, TeX Live 2026, the DejaVu fonts; `crates/presk-explain/pandoc/
  header.tex`); a table's columns sized by its cells; the manual's chapter
  `explain_loss`, the tutorial's document, held current by a test (`make
  explain-manual`); the tutorial's step 7 shows `presk explain`. X5 holds:
  the Llama 2 decoder block at the 7B width, D106's envelope and
  realization given, no exact reference, in 40 s and 10.9 GB.
- Step 7 (D117): three of D110's defects fixed: `presk dump --stage elab`
  and `--stage core` bind `--const`, which they ignored; a missing
  `--weights` file is named once in the message; the PyTorch export's
  conformance reports a PSK-Core verification failure as every command
  does (exit 2), where it exited 1 without a word. `presk check --precision
  --measure` says "the kernel interpreter against the exact reference",
  which it runs, where it said "native" (the same bits, U5). Version
  `0.11.0`, before the checks of record, which pass on the version's tree:
  `make test`, `check-examples`, `coverage`, `plan-check`, `c-conformance`
  (14 of 14), `cargo doc`; X1 on `out/x1/x1.sh` and `make tutorial`, every
  change named. The paper's runs at 0.11.0: `make precision` 29 sites,
  2.14×, the same gap; `make quant-table` byte for byte; the curve 2.87× to
  1.09× less traffic, every budget inside, the quality axis within
  1.7·10⁻⁴ (`precision` chapter); BERT-large at two blocks 25 sites at 2.03×, measured 1.05·10⁻⁵ against
  6.3·10⁻³ (D101: 63 sites, 2.00×, 5.1·10⁻⁷; over the stored values the
  greedy order types the query `bf16` and the layer norm's mean `f32`;
  `docs/compile-times-2026-10-09.md`); the
  decoder at the 7B width 24 of 25 sites at 1.96×, measured 1.99·10⁻⁵
  against 1.1·10⁻² (D106: 52 of 56, 1.99×, 4.3·10⁻⁶; the softmax and
  RMSNorm's mean now `f32`), in 14 h 03 min and 71.7 GB,
  `docs/compile-times-2026-10-09-llama.md`.
- Step 8: the manual's `precision`, `coverage`, `experiments` and `status`
  chapters at 0.11.0; `CLAUDE.md`; presk-specs' plan marked done, its
  reference, report and tooling at 0.11.0.

Since the tag `v0.10.1`, before 0.11.0:

- One nest's bound proved in Gappa beside the compiler's bound and the
  measured gap (D108): `make gappa` (`experiments/gappa.sh`,
  `experiments/gappa_layernorm.py`, `experiments/gappa_layernorm.psk`) on
  the examples' layer norm in `f32` at `D = 8, 16, 32, 64`, the report
  `docs/gappa-2026-10-09.md`, the statements, answers and proofs under
  `docs/gappa/`. At `D = 64` on one row: measured 2.4·10⁻⁷, `cond`
  1.3·10⁻⁵, proved 3.3·10⁻⁵; on eight rows 4.1·10⁻⁷, 1.4·10⁻⁵, 6.9·10⁻⁵.
  The compiler's bound meets the proof on one row's envelope and is below
  it on a sample's (its quotient and square-root rules take a value's
  error as proportional to its magnitude). Gappa is an optional tool
  (Debian's package), documented in the manual's `dependencies`. Four
  defects of the CLI found and left: `presk check --precision` and `presk
  solve` do not bind a module entry's dimension arguments (the mean's
  length falls back to 1 024 terms on `LayerNorm{8}.apply`); `calibrate
  --samples` without `--data` draws nothing; the entry's arguments are not
  sites of the envelope; one goal per element of a row.
- PyTorch eager on the Llama 2 decoder block at the 7B width (D107):
  `experiments/pytorch_decoder.sh`, `make pytorch-decoder`, the forward
  pass at 384 tokens, batch 1 and 20, by the PyTorch baseline's protocol on
  the ten X925 cores, the native backend in `f32` and with `int8` weights
  on every matrix, under the default layouts and the frozen model's plans,
  then its kernels alone and its host by phase; the report
  `docs/pytorch-decoder-0.10.1.md`. Under the plan the native forward pass
  takes 13.6× PyTorch's at batch 1 and 14.4× at batch 20 (`int8` 6.8× and
  8.3×), two thirds of its kernels' time the attention's output
  projection, which the layout space does not tile.
  `experiments/pytorch_baseline.py` builds the decoder with `--model
  llama` (its own seeded draw and token ids); the course GPT's path is
  unchanged.

## 0.10.1 — 2026-10-07

A patch of 0.10.0 that changes how much the exact interpreter holds and no
value it computes (`presk-specs/implementation/presk-plan-0.10.1.md`,
D105). At the tag, `presk run --bits 128` held every parameter twice at 64
bytes a real, and `presk solve` kept every partial cotangent of its reverse
sweep: on one block of the Llama 2 decoder at the 7B width it was killed at
108 GB (D104). 0.10.1 holds each value once and frees it after its last
use; the solve on that block fits, and is measured (D106). Judged on five
statements fixed before the work: no output changes, byte for byte against
`v0.10.0` (U1, holds); the exact run one copy per parameter and the solve
at most two plus its transients (U2, holds); the exact reference no slower,
within 3 % (U3, holds on an X925 core: the course GPT +0.9 %, BERT −2.6 %);
the decoder's solve measured (U4); and the native backend equal to the
kernel interpreter on a quantised parameter read through `split` (U5,
holds). Every number the paper took at `v0.10.0` stands at 0.10.1.

- The exact interpreter frees a value after its last use and holds each
  parameter once (steps 1–3, D105): a value is removed from its
  environment after the `let` that reads it last, and moved into a call
  there; tensors share their elements copy-on-write
  (`presk_interp::Data`), in the interpreter and across the seam to the
  other executors (`presk_rt::Buf::Exact`); a stacked layer a fold reads is
  a view; a stack is built by moving its layers; zeros are shared, and a
  sum with them is the other operand itself where MPFR gives it.
  `presk run --bits 128` holds 64 bytes a parameter where it held 128, and
  `presk solve` needs 148 bytes a parameter of a block, and keeps the
  cotangent of the parameters only under `--quant`. Every output is the
  same, byte for byte: 558 outputs of the test set on the three executors,
  the goldens, the examples, `make coverage`, `precision`, `quant-table`,
  `step-table`, BERT at two blocks, D104's native logits, `make tutorial`
  but for its times. `Interp::live` counts the reals bound, and a test
  bounds what a chain and its gradient hold.
- The native backend and the C export compute a quantised parameter read
  through `split` right (step 6, D105): they found an element's scale by the
  stored axis's number in the kernel's view, and read past the scales
  (D104's defect: the Llama decoder's `wo`, NaN logits natively). A
  kernel's integer input is `int8/o..p` in `presk dump --stage kernel`
  when the axis the scales are constant along is read as several
  (`BufKind::Quant { bits, over, span }`); `tests/quant/split.psk` is
  D104's two-line case.
- The Llama 2 decoder's budget solved (step 8, D106,
  `docs/llama-7b-solve-2026-10-07.md`): `presk solve --budget 1e-3` on its
  next-token loss at the 7B width, one block, 384 tokens, in 14 h 07 min and
  71.7 GB (`calibrate` 8 h 23 min, the reference 1 h 54 min and 29.3 GB):
  52 of 56 sites narrowed, 1.99× less traffic than all-`f64`, measured
  4.3·10⁻⁶ against 1.1·10⁻², the bound 6.2·10⁵ vacuous. The whole
  quantisation scheme natively at one block, `int8` 5.3·10⁻² and `int4`
  0.76 from the exact logits, the kernel interpreter's gaps, `int8` on
  every matrix 1.72× and 2.34× faster than `f32`; the exact logits at two
  blocks (41.4 GB) and their native gaps; the split at one block, the
  weights' rounding the whole of the quantised gap. The solve's row of the
  setup table is `docs/compile-times-2026-10-05-llama.md` (key
  `llama-7b-L1`). `experiments/llama.sh` runs the whole scheme natively, its
  split at one block (`SPLIT_L`, default 1) and its bench on every matrix,
  and its table still reads D104's files.
- The manual at 0.10.1: the `language` chapter's interpreter memory,
  `coverage`'s *The decoder, run* with the solve, the whole scheme, two
  blocks and the split at one block; `status`, `precision`, `experiments`
  §10, `native` (the integer weights read through `split`).

Since the tag `v0.10.0`, before 0.10.1:

- The ε-to-quality curve measured again at the tag `v0.10.0` (D102,
  2026-10-03; `make curve`, then `make foil`), its solver columns having
  been of the program before step 7b and of `bf16`'s former unit roundoff:
  every budget inside, the quality axis flat within 2.0·10⁻⁴; the loosest
  budgets narrow less to `bf16` (21 sites at 10⁻¹ where 31, 2.69× less
  traffic where 2.95×); statement D's realization at 10⁻³ measures
  2.12·10⁻⁵ with 2.00× less traffic (D93: 1.89·10⁻⁶, 2.08×), and 2.5·10⁻⁷
  on the foil's held-out windows. The manual's figures that quoted D93's
  curve follow, with two stale passages the language reference's check of
  0.10.0 found (the estimate's ratios in `precision`, the `bf16` launch
  counts in `language`).
- The paper's run at the tag (D103, 2026-10-03): every output of D93's
  chain reproduced at `v0.10.0` to every digit but two. J′ holds at
  **1.75×** under `--remat store` (1.95× in D93), its gaps the same, the
  policy moving neither ratio; BERT-large's `bf16` logits are 5.3·10⁻²
  from the exact ones at 24 blocks (1.1·10⁻¹ in D53), moved since 0.7.0's
  step 2 (D58, bisected). The manual's J′ and BERT figures follow.
- The second model of the paper's RQ5 (D104, 2026-10-04; `make llama`,
  `experiments/llama.sh`): the Llama 2 decoder of `coverage/llama.psk` at
  the 7B configuration's width (`G = 32`), one of its 32 blocks, on its
  initialization and 384 random tokens, by BERT-large's protocol. The exact
  logits in 1 h 55 min and 87.6 GB; `f64` 1.3·10⁻¹⁴, `f32` 6.2·10⁻⁶,
  `bf16` 2.5·10⁻², `int8` weights 4.9·10⁻² (5.3·10⁻² with `wo`, on the
  kernel interpreter), `int4` 0.71 from them; the split on no block, the
  weights' rounding the whole of the quantised gap; `int8` weights 1.59×
  faster than `f32` at batch 1 and 1.86× at batch 20. No solve: it does not
  fit in 119 GB at one block. `coverage/llama.psk` gains an entry `loss`
  and draws its head as the other matrices (it was `zeros`);
  `experiments/llama_ids.py`; `make coverage` runs the decoder at tiny
  constants; `compile_times.sh` gains the key `llama-7b-L<N>` and keeps each
  command's peak memory. Found on the way: the native backend computed a
  quantised parameter read through `split` wrongly (fixed in 0.10.1,
  above).

## 0.10.0 — 2026-10-02

Make a training step pay (`presk-specs/implementation/presk-plan-0.10.0.md`),
from the PyTorch baseline below (D87): on the course GPT at `B = 16`, `T =
256`, `f32`, a training step in C under a plan takes 12× PyTorch's and
`presk train` 22×, and four fifths of the C gradient is the derivatives of
`dot`, outside the layout space. 0.10.0 takes the causes in order. The
derivatives of `dot` become contraction nests of the layout space, their
default layout today's kernel, and the frozen model of 0.8.0 ranks them,
nests it has never seen. A gradient stops recomputing its forward pass: the
dead code after differentiation removed, then a forward pass that returns
its residuals and a backward pass that reads them (`--remat store`, the
default), today's recomputation kept as `--remat recompute`. The native
host is measured phase by phase, then fixed: values sharing their
buffers, views, buffers reused, dropout masks drawn on the workers, a pool
of workers on `std`. The bias of
a linear layer becomes its product's epilogue, and the C export's compiler
flags are measured. Every change is exact. Judged on five statements, fixed
before the version began: every change bit-identical to 0.9.0's outputs in
`f64`, `f32` and `bf16`, and C1 kept (T1); the course GPT's training step
in C under the plans within 2× of PyTorch's on the ten X925 cores (T2);
0.8.0's statements O and Q on the derivative nests, RA4's third fresh test
(T3); the native backend within 1.3× of the C program on the forward pass
and on the step (T4); the gradient's peak memory under both policies
reported (T5); and, with step 7b, the language's four decisions of
2026-09-29 changing no value (T6). The CUDA export comes after it.

**What it shows.** A training step can be made fast without moving a bit.
On the course GPT at `B = 16`, `T = 256`, `f32`, on ten X925 cores, the
C program's training step under the frozen profile's plans takes 995.4 ms
against PyTorch's 1 008.6, 0.99× where it was 12× at 0.9.0 (T2), and every
change on the way was a layout the space proves bit-identical or the same
operations in the same order (T1); the two gradient defects found on the
way (3a, 3f) were fixed and what they moved measured again (D93). The
derivatives of `dot` run at 633 GFLOP/s under the plan, as the products do.
The frozen model of 0.8.0 picks within 1.19× of the best on derivative
nests it had never seen, and fails Q on the cost of deciding (T3). The
native backend's host is gone, and what is left between it and the C
program is the kernels' code, Cranelift's against GCC's: 1.52× on the
forward pass and 1.60× on the step, over the 1.3 asked (T4 fails, as
decision 2 accepted). Storing the residuals costs 2.75 times the
recomputing gradient's memory, 2.49 GB against 0.91 (T5). The language's
four decisions of 2026-09-29 changed no value (T6), and the reference's own
gap, measured, is 21 orders of magnitude below what any gap is read at
(D98).

- The compiler's own time measured (D101, on 2026-10-02, after step 8, for
  the paper's setup table). `make compile-times`
  (`experiments/compile_times.sh`, the report
  `docs/compile-times-2026-10-02.md`): `presk check`, `presk calibrate
  --precision`, `presk solve --budget 1e-3` and the exact reference on one
  sample, wall seconds on one X925 core, on four models. The front end is
  milliseconds (7 ms on the course GPT); the rest is the reference
  interpreter at 128 bits: on the course GPT at its course constants, 178.6
  s for the reference on one window, 1 497.7 s to calibrate on two, 2 702.4
  s to solve; on BERT-large's two blocks, on their SQuAD loss, 616.9 s,
  5 086.2 s and 10 463.8 s. `presk solve` now refuses an entry that does
  not return a scalar `ℝ[]` before computing anything, as `presk dump
  --stage sensitivity` does: on BERT-large's `squad`, which returns its
  logits, it ran the reverse sweep's forward half for 39 minutes before
  failing. `coverage/bert.psk` gained `squad_loss`, BERT's SQuAD
  fine-tuning loss as `run_squad.py` computes it, and
  `experiments/bert_features.py` an answer's span (`--span`), on which
  BERT's row is measured.
- Step 8: documentation, the final measurement, version 0.10.0 (D100).
  Seven code items, one commit each:
  - **Recursion refused** (`E2016`, the reference's requirement "No
    recursion"): a function that calls itself, directly or through others,
    and a module that includes itself through its `use` members, by the
    checker once every call is resolved, one diagnostic per cycle naming
    its functions or modules; both overflowed the stack of the analyses,
    and `presk dump --stage elab` refuses the module form too.
  - **W4's field**: `inference serve`'s `generate = name` names the
    generation function; a name that is no top-level function of the
    file, or whose signature does not fit, is `E2010`; without it, the
    first that fits. `presk serve` and both exports' serves take it;
    `--entry` overrides.
  - **A stack of no layer**: the parameters of `stack{0}(M)`, every leaf of
    `M` with a leading axis of length 0, built from their type on the
    interpreter and the kernel backend, and the scan of a fold of count 0
    (a gradient through such a stack) likewise; the C export follows. Both
    panicked.
  - **A mask printed as truth values**: `presk run` prints a value of type
    `bool[S]` as `bool[…]` of `true` and `false`, and in `--json` as JSON
    booleans, from the entry's signature; at run time and in a `.psk.d` a
    mask stays an index tensor of bound 2.
  - **A declared accumulation type the kernels do not honour**, other than
    the storage's arithmetic type, refused by the kernel and native
    backends before anything runs (`E7002`); condition 7, which read it,
    admitted spills whose bits differed from the default's.
  - **`bf16`'s unit roundoff is 2⁻⁸**, not 2⁻⁹ (the reference's chapter 4):
    every `bf16` bound doubles, and the solver keeps a few sites in `f32`
    it gave to `bf16` (`make precision`: two at 10⁻³, the same measured
    gap); the `precision` chapter's curve, not measured again, is dated.
  - **The gap is +∞ at a non-finite difference**: `presk solve`,
    `quantize` and `check --precision --measure` left such an element out,
    so a realization that overflowed was measured on the rest and could be
    accepted (`tests/data/overflow.psk`: `main` accepted `f32`, now
    `f64`).
  Against `main`'s binary, 154 outputs and training runs identical, a
  mask's printing aside; C1, `plan-check`, the archived rankings byte for
  byte. The final measurement on the version's tree
  (`docs/pytorch-baseline-0.10.0.md`): T2 0.99×, T4 1.52× and 1.60×, T5
  2.49 and 0.91 GB. `docs/spec-questions.md` marked from the reference's
  chapter 8; the reconciliations' documentation items; the manual at
  0.10.0 (`experiments` §10, `status` closed); spec questions 76–77 copied
  to presk-specs; the tutorial refreshed. The findings the user recorded
  for a later plan on 2026-10-02 (the reference's chapters 3, 4, 5, and
  δ_p's two) are in plan §8. Version `0.10.0`.
- Step 7b: the language's four decisions of 2026-09-29 (the reference,
  draft v0.6), in the compiler (D99). Four premises of the plan did not
  hold and were amended at the step.
  - **Refused at the dimensions where an operation is undefined**
    (`E7003`, a code of the run's range). Before an entry runs, on the
    dimensions it is given: a `mean` over no element, `mean axis a` with
    `a` of length 0, a selection outside its axis, `embed` reading indices
    below 0, `cross_entropy`, `argmax` and `sample` over a last axis of
    length 0, and their derivatives; through the functions the entry calls
    at their instantiations, the closures it applies and every `fold` of
    count at least 1; and a parameter whose type has no value (`idx[0][S]`
    with an element). In `presk run`, both exports, `bench`, `plan`,
    `autotune`, `dump`, `calibrate`, `quantize`, `solve`, and in every
    backend's `prepare`, so in `presk train` and at each of `presk serve`'s
    context lengths. A division by zero and a mask that empties a row stay
    errors of the run. An unmasked softmax over an empty axis computes
    nothing: the interpreter panicked there and the kernels read out of
    bounds. A `tests/diag/run_*.psk` per condition.
  - **`sample(key, logits)`**. Every draw from its key, as the reference
    defines it. The three private streams of `sample` are gone (the
    interpreter's, the kernel backend's, the C program's), and with them
    the backends' seed (`Interp::new`, `InterpBackend::new`,
    `KernelBackend::new`, `native_backend`); PSK-Core keeps no `let`
    because it may draw. The interpreter compares the key's draw exactly.
    `presk serve` gives token `i` the child `i` of the key `--seed` names,
    to a generation function that takes a key as its second parameter; one
    without a key is served as before. The interpreter and the native
    backend now draw the same tokens. The C runtime gains a key
    (`presk_key`, `presk_key_from_seed`, `presk_key_split`,
    `presk_key_child`, `presk-interp`'s `Key` byte for byte); an entry's
    key that `sample` reads is a field of its inputs, derived by `main`
    from `-s SEED`; `generate.c` gives each token its child; a key the
    program does not hold reaching `sample` is refused (`E8006`). The
    PyTorch export takes the key and draws with PyTorch's own generator.
  - **`bool[S]`, `causal_mask`, `softmax(axis a, x, where = m)`**. A tensor
    of truth values is a type (`bool` followed by axes); `causal_mask`
    gives one, and `softmax` reads one with `where`, the masked softmax on
    the interpreter, both executors in three passes and in two, the C
    export, the SCoP mode (selections only, still static control) and the
    PyTorch export (`masked_fill`). Its bits are those of the scores with
    `−∞` added at the refused positions. A mask is an index tensor of
    bound 2 at run time, synthesized all true, stored `idx[2][…]` in a
    `.psk.d` (a file's type may write `bool[…]`). No tensor holds `−∞`:
    `causal_mask` made the only ones, and the envelope's special case for
    them is gone.
  - **The programs.** The course GPT, the tutorial's attention and
    sampling and the Llama transcription rewritten; the examples
    synchronized with presk-specs (`train_minigpt.psk` and
    `serve_minigpt.psk` write `m.B`, `m.T`: a function's dimensions now
    include those its signature mentions from an imported file). The
    archived days of 0.7.0's and 0.8.0's sweeps name the old attention's
    nests, so it is frozen in `experiments/ra4_attention.psk`, which
    `make ra4-dev-frozen`, `ra4-f1` and `ra4-f2` read; four plans
    renumbered.
  - **T6 holds**: against `main`'s binary, 147 outputs and training runs
    byte for byte (the eleven entries on three executors in three types
    and under plans, the course GPT's forward pass and gradient, the
    training runs), only `next_token`'s draws moving; C1; `make
    check-examples`; the archived rankings and the development set's
    report byte for byte. Tutorial steps 17 and 18 refreshed.
- The reference's own gap measured (D98; the language reference's W16, on
  2026-10-02, beside step 7b). `make delta-p` (`experiments/delta_p.sh` and
  `delta_p.py`, the report `docs/delta-p-2026-10-02.md`): the reference
  interpreter at p bits against itself at a higher precision, on the course
  GPT at `D = 64` on trained weights and eight windows of the Code civil,
  its loss, logits and gradient each in the form of its budget's gap. δ_p is
  five to ten units of roundoff at p bits: 3·10⁻³⁸ at 128 bits, 4–9·10⁻⁷⁷
  at 256, 4–8·10⁻¹⁵⁴ at 512, every ratio δ_p/δ_2p near 2ᵖ, 21 orders of
  magnitude below binary64's resolution, at which every measured gap is
  read; the training step's all-`f64` row reads the same at 128, 256, 512
  and 1000 bits. No compiler code changes. Two defects of the interpreter
  found and known, left for a later plan since both move the reference's
  bits (plan §8): the gradient through `gelu` never ends at `--bits` ≥ 1016
  (`Interp::pi` sums Machin's series to a threshold that underflows in
  `f64`, where MPFR's constant is in the trait), and `dropout`'s scale
  1/(1 − r) is computed in binary64 (1.4·10⁻¹⁷ off the equation, at every
  p).
- Step 7's follow-up, the user's answers on !47 (D97's addendum). The C
  export always inlines every kernel (`PRESK_KERNEL`, `always_inline`) and
  compiles `model.c` on GCC without scheduling before register allocation
  (`#pragma GCC optimize ("no-schedule-insns")`, skipped by Clang): a
  product's inner loop under a 4×16 tile is at the edge of the vector
  registers, and wherever GCC ran short, out of line or after that
  scheduling, a sum lived on the stack and the product took about 30 %
  longer; which kernels GCC inlined moved with the program's other
  kernels. Now the course GPT's gradient takes 822–827 ms under the plans
  with or without their epilogues, where it took 844 or 959, the threads
  waiting as the native ones; no bit moves. libgomp's idle threads sleep
  after about 0.1 ms, the native pool's after about 2 ms (measured;
  `pool.rs`'s comment corrected): the README names `GOMP_SPINCOUNT=8M` and
  `OMP_WAIT_POLICY=active`, and `make pytorch-baseline` times every C
  program both ways (`C_WAIT`), T4 on the second
  (`docs/pytorch-baseline-0.10.0-step-7-followup.md`): T2 1.01×; T4 1.47×
  on the forward pass and 1.57× on the step, the kernels' code
  Cranelift's against GCC's, judged at step 8. `kernel_time.py` reads both
  kernel declarations.
- Step 7: the bias as its product's epilogue, and the C flags measured
  (D97). A ninth layout decision, `"epilogue": true`, on a contraction nest
  whose value one `add` alone reads, launched by itself and of the
  product's type (`presk_core::fusion::epilogue`): the product's `let`
  launches nothing, and the `add`'s launches the product's kernel under its
  layout (`dot+add`), which adds the operand to each element once its sum
  is done. Exact: in `f64` and `f32` the operand is added in the register
  before the store, in `bf16` the stored sum is read back and added to, under
  a spill in the last block of `K`, in the row-wise form by one more sweep.
  Unstated, the two nests as before, byte for byte. It is no point of a
  nest's space, so `presk autotune`'s sweeps and every archived ranking stay
  as they were; `presk plan` states it wherever it applies, over the layout
  it chose, and names them; a plan stating it where it does not apply is
  refused (`E7002`); the C export prints it. Spec question 77. Checked:
  every point of the products of `tests/plan/epilogue.psk` and of the
  derivative nests given an operand (7 488) against the product and the
  `add`, both executors, three types; whole programs and mixed
  realizations planned; T1 against `main`'s binary on 184 outputs; C1 under
  plans stating it. `make pytorch-baseline` takes `CFLAGS` and
  `BASELINE_OUT`. Measured (`docs/pytorch-baseline-0.10.0-step-7.md`): the
  36 bias launches of a forward pass, about 10 ms on each side; **T2
  1.02×**; T4 1.16× on the forward pass and 1.305× on the step, at its
  threshold: the C gradient took 113 ms less, 100 of them from GCC inlining
  the program's other kernels otherwise (built `-fno-inline`, `main`'s and
  the branch's take the same time). The flags (`…-o3-native.md`,
  `…-o3.md`): `-O3` −4 % to +2 %, `-mcpu=native` nothing, C1 under both;
  `-O2` kept. Found: the C program's OpenMP workers sleep between regions
  (`OMP_WAIT_POLICY=active`: −18 % on the forward pass, −16 % on the
  gradient); condition 7 admits a spill under a declared `bf16`
  accumulation, which the kernels do not honour.
- Step 6: the native host fixed (D96), in the order step 5 gave, each
  change measured apart, pinned, against `main`'s binary of the same day
  (`make host-profile CONFIGS=10`; `PRESK` may name another build). Values
  share their buffers: a machine buffer's elements are `presk_rt::Elems`,
  behind a reference count, so reading a value, passing it to a call, a fold
  or a kernel, putting it in a tuple and projecting it out copy nothing,
  and a write to shared elements copies them first (the gradient's copies
  5.2 s → 10 ms). Views: a stacked layer's slice is a view of its stack, a
  scan's stack its parts, laid out in one buffer only if read whole, a slice
  set or a gradient scattered its parts in their places (the slices → 0.4
  ms). Buffers reused: a free list by length (`presk_rt::FreeList`), a
  launch's output taken from it and not zeroed, since every kernel writes
  its output before reading it (a debug build poisons it, and `make test`
  passes); `zeros_like` one shared buffer; glibc's setting not shipped (the
  gradient's 3.3 million fresh pages a call → none). Dropout masks off the
  host: a training mask drawn in pieces on the workers, each from its place
  in the key's ChaCha20 stream, the bits of one stream in order; outside
  training ones, one buffer per length (the training step's 0.95 s → 0.08).
  A pool of workers on `std` (`presk_cranelift::Pool`): a divided launch's
  pieces on persistent workers, the first on the calling thread, a waiting
  worker spinning about 2 ms while the workers have a core each (the
  forward pass's threads 38 → 0.6 ms); unpinned, the kernels take 326 ms
  where they took 755, level with pinned, so the workers are not pinned to
  cores. The entry's arguments already in their type shared (`adamw.step` 44
  → 16 ms). Keys left out (0.2 %). Pinned, the forward pass 661 → 332 ms,
  the gradient 8.2 → 1.32 s, the training step 7.8 → 1.43 s
  (`docs/host-profile-0.10.0-step-6.md`). `make pytorch-baseline` times the
  native `loss_grad` and `adamw.step` and judges T4
  (`docs/pytorch-baseline-0.10.0-step-6.md`): **T4 holds**, the native
  forward pass 1.16× the C program's, the step like for like 1.17×, `presk
  train`'s step 1.26× beside; what is left is the kernels' code,
  Cranelift's against GCC's. T2 1.13×. T1 against `main`'s binary on 114
  outputs and training runs, C1, `plan-check`.
- Step 5: the native host measured (D95). `presk bench --host-profile`
  times the native backend's host by phase, the kernels apart
  (`presk_kernel::profile`): finding a kernel (its site, its fusion groups,
  its layout, its key, lowering), buffers (allocated, converted, freed),
  slices and views, copies (the entry's arguments, a launch's operands out
  of the environment and again as tensors, a call's arguments, tuples and
  fields, a fold's arguments and accumulators), the executor's dispatch
  and a divided launch's threads (`KernelExec::set_profiling`, `overhead`),
  the dropout masks and the other primitives the host computes, and the
  interpreter by difference; `--repeat` calls with the timers and as many
  without, alternated, compared bit for bit, and the minor page faults of a
  call. `presk train --host-profile` gives a step's table. Only the clock
  is read: the same bits with the timers and without on both executors, a
  training run's weights the same bytes. `make host-profile`
  (`experiments/host_profile.sh`, `.py`; kept as
  `docs/host-profile-0.10.0-step-5.md`): the course GPT under the frozen
  profile's plans, ten X925 cores pinned, every core unpinned, and pinned
  with glibc keeping its pages. Pinned, the forward pass takes 664 ms: its
  kernels' code 341, the copies of values 213 (6.2 GB a call), the
  threads' start and join 43, the dropout masks of ones 32, the outputs'
  allocation 27, finding a kernel 2. The gradient takes 8.3 s: its kernels
  1.45 s, the copies 5.3 s (73 GB a call, most of it the stored residuals
  carried in tuples), the slices of stacked layers 0.69, the frees 0.53,
  the threads 0.13, finding a kernel 0.01; it touches 3.25 million fresh
  pages a call, and with glibc keeping its pages it takes 5.0 s, its
  kernels' code 1.32 s, 1.33× the C program's (the forward pass's 1.27×).
  The training step, 8.0 s, also draws its dropout masks on the host, 0.89
  s. Recommended order for step 6, for review: values sharing their
  buffers, views, buffer reuse, dropout masks off the host, the pool, the
  entry's arguments, keys.
- Step 4b: the fusion groups kept only where a realization can see them
  (D94; the user's answers of 2026-09-30 on presk !40). A group computes
  its members in its root site's arithmetic type and stores only the root,
  so its partition decides a value's bits under a `bf16` site and under a
  site typed apart from its function (a `storage` entry), and nowhere else
  (`Realization::groups_visible`); the user's decision said `bf16`, and
  a realization mixing `f32` and `f64` by site, as the curve's at 10⁻⁶
  and 10⁻⁷, needs the groups too (a stored gradient's value otherwise
  departs from its function's). `presk_core::ad::Options { remat, groups }`:
  without the groups the dead-code pass holds nothing and the split shares
  every residual with the function's computation. The commands decide by
  the realizations they run (`bench` keeps them unless it times its plan
  alone, `solve --step` always, `calibrate --step` when a realization
  would or it writes an envelope). In `f32` and `f64` the course GPT's
  gradient launches 1 012 kernels storing, none dead (1 182, 10), 1 548
  recomputing (1 616), and its C arena is 2.49 GB (3.04; 0.90 recomputing,
  1.02); `bf16` is step 4's program. `W0004` names a `.vjp` a realization
  types where the program stores its gradient, unless the realization
  names its `.fwd` or `.bwd` too. T1: the same outputs as `main` under both
  policies, 168 comparisons; the training logs; C1, with a gradient's
  `bf16` plan dropping the layouts of nests `bf16` computes inside a group.
  T2 again (kept as `docs/pytorch-baseline-0.10.0-step-4b.md`): the C
  training step under the plan takes 1.13 s against PyTorch's 1.03 s on the
  ten X925 cores, 1.10× (1.35× at step 4), and on every core unpinned 0.85×;
  `presk train` under the plan 8.1 s (9.7 s). The C gradient under its plan
  takes 205 ms less than `main`'s, holding (1 107–1 118 ms against
  1 310–1 326), the same bits.
- The results defect 3f moved, measured again (D93; the user's answer of
  2026-09-30 on presk !40), under `--remat recompute` so that only the
  gradient changed since D48–D55. `presk solve` and `presk calibrate` take
  `--remat`; a step's envelope records the policy it was calibrated under,
  and `solve --step` refuses one of the other policy; the experiment
  scripts and the Makefile take `REMAT`. The course GPT at `D = 64` trains
  to 2.757 in 600 steps where the wrong gradient reached 2.995. On those
  weights K holds at 2.73·10⁻⁵ (3.23·10⁻⁶) and 3.90·10⁻⁵ held out
  (1.46·10⁻⁴); D at 1.89·10⁻⁶ (4.48·10⁻⁶); the curve's quality axis is flat
  within 1.9·10⁻⁴ against a seed spread of 0.092 (1.2·10⁻³, 0.13); in the
  step table AdamW multiplies an `f32` gradient's error by 545 (340), the
  recipe is 4.4·10⁻² from the exact step (2.9·10⁻²), and the solver again
  takes sixteen functions `f32`; J′ holds at 1.95× (1.99×), inside its
  budget; a run under the solved step ends 5.5·10⁻⁶ from the `f64` run's
  weights (5.0·10⁻³). In the foil, MLPerf's rule at 99.9 % now refuses
  `int4` weights, 136 predictions lost of 28 612, where their gap is
  4.0·10⁻⁴. The condition number buys 28 orders and not the last nine.
  D48–D55 stay as written; the manual's `precision`, `experiments` and
  `coverage` chapters, the tutorial and question 60 take the new numbers.
- Step 4: residuals, `--remat store|recompute` (D92). First, defect 3f of
  the reference's chapter 5: a scanned fold stacked the first key of its
  accumulators for every layer, so the backward fold of a stacked module
  whose key rides the accumulator gave every layer layer 0's dropout masks,
  a wrong gradient under training, on the interpreter and natively alike
  (the course GPT's training since 0.5.0 step 1; `presk run` and the
  exports, which drop nothing, never saw it). A leaf with no axis to stack
  along is now stacked as the sequence of its values (`tests/data/stack_dropout_grad.psk`,
  against finite differences on both executors). Then a gradient keeps
  its forward pass's values (`presk_core::ad::store`, `--remat store`, the
  default of `presk run`, `export`, `train`, `bench`, `plan`, `autotune`,
  `reprice` and `dump`): each `.vjp` is split into `f.fwd`, which runs
  `f`'s body once and returns its value and the residuals its sweep reads,
  and `f.bwd`, which runs the sweep on them and calls each callee's `.bwd`;
  a fold stacks its layers' residuals through the scan it had. `f.fwd`'s
  bindings are `f`'s and `f.bwd`'s the `.vjp`'s, so a sweep's nests keep
  their numbers (`Linear.apply.bwd/%11`), and what a realization or a plan
  states of `f/%k` it states of `f.fwd/%k`. Exact in `f64`, `f32` and
  `bf16`: where the fusion groups of `f` and of its `.vjp`'s copy differ, a
  residual is computed apart as the copy computed it (in the course GPT,
  the normalizations' activations, and, because of the launches step 3
  holds for `bf16`'s groups, attention's core and the feed-forward's first
  product again). The course GPT's gradient launches 1 182 kernels
  (1 616), its `dot` and `ein` 150.1 GFLOP (311.9), its arena 3.04 GB
  (1.02, T5). T1: the same outputs as `main` under both policies on the
  three executors in three types, the same training logs but for a
  realization written by function for `.vjp`s, C1. Two copies of the
  native host that storing made costly go: a slice converted its whole
  stack, a projection or a field copied its whole tuple or record. T2
  judged, and holds (kept as `docs/pytorch-baseline-0.10.0-step-4.md`): the
  C training step under the plan takes 1.36 s against PyTorch's 1.01 s on
  the ten X925 cores, 1.35× (1.86× at step 3); `presk train` under the plan
  9.7 s (9.9 s). The launches held for `bf16`'s groups now cost about
  15 % of the C gradient (3.2 % at step 3). A plan written for a gradient
  before names `.vjp` nests and is refused (`E7002`); T3's nests are the
  `.bwd`s'.
- Step 3: the dead code after differentiation; one call a training step
  (D91). A lambda applied in a differentiated function is differentiated as
  itself: the `.vjp` looked the lambda up under the wrong variable, so with
  one lambda in scope the gradient was refused (`E6001`), and with several
  it was another lambda's, without a diagnostic (defect 3a of the
  reference's chapter 5; `tests/data/closure_grad.psk`, against finite
  differences on the interpreter and natively). A pass after
  differentiation (`presk_core::ad::prune`), which every command that
  differentiates runs, removes what a gradient computes and never reads:
  forward values its sweep does not read (the primal loss, a callee's
  value only its own `.vjp` needs), cotangents no caller reads (a `.vjp`
  returns the cotangents its callers read; a backward fold keeps the
  accumulators someone reads), `sum_to` and `broadcast_like` between equal
  types, and values read for their shape only (a template reads another
  value of its type the function computes anyway: the forward feed-forward
  a block's `dropout_mask` read for its shape goes). Names and binding
  numbers stay, so nests keep their names. Every fusion group of what the
  result reads is kept, since a `bf16` group holds its intermediates in
  `f32`; what would change one is held (68 launches of the course GPT's
  gradient). PSK-Realize's partition moves into PSK-Core
  (`presk_core::fusion`, re-exported by `presk-realize`). `presk train`
  calls the objective's `value_vjp` once a step, the value then every
  cotangent, where it called the objective and then its `.vjp`; the value
  is computed as the objective computes it, and a realization or a plan
  stating the `.vjp`'s or the objective's sites states `value_vjp`'s. The
  course GPT's gradient launches 1 616 kernels, 12 of them dead, where it
  launched 2 220, 386 dead; its C program's arena 1.02 GB (1.22). T1:
  `main`'s binary and this one give the same outputs on the eleven entries
  of `tests/plan/entries.txt` and the course GPT's gradient on the three
  executors in `f64`, `f32` and `bf16`, the same training logs, and C1
  holds. A plan written before for a gradient names nests that are gone
  and is refused (`E7002`). T2's second reading (kept as
  `docs/pytorch-baseline-0.10.0-step-3.md`): the C training step under the
  plan takes 1.88 s against PyTorch's 1.01 s on the ten X925 cores,
  1.86×, where it took 2.76×; `presk train` under the plan 9.9 s (12.8 s).
- Step 2: the frozen model on the derivatives of `dot`, T3, RA4's third
  fresh test (D90). `presk plan` and `presk autotune` take several nests
  (`--nest` repeated, or names separated by commas), recorded in
  `sweep.json`, so that one sweep holds a gradient's four derivative nests.
  A nest launched at several shapes keeps every shape's points in its file,
  where each launch overwrote the last; no archived sweep had such a nest.
  `presk rank` counts `dot_vjp_l` and `dot_vjp_r` among the contraction
  nests of statements O and Q, and names a nest at several shapes with its
  operands; every earlier report comes back byte for byte. Targets
  `ra4-t3-dry`, `ra4-t3`, `ra4-t3-day2`, `ra4-t3-rank` and `ra4-t3-check`.
  Measured on 2026-09-29 on the ten X925 cores: the course GPT's gradient
  at `B = 16`, `T = 256`, `f32`, its four derivative nests at eight shapes,
  11 028 points, swept whole twice (3 h 05 min each) and checked on the
  kernel interpreter between the days (7 h 06 min), priced by the frozen
  profile unchanged, archived read-only in `~/presk-data/ra4-t3-0.10.0`;
  the report is `docs/ra4-t3-2026-09-29.md`. **T3 fails.** O holds: the
  pick within 1.00–1.19× of the best on every nest and shape. Q's
  candidates hold the best within 1.01× everywhere, but its cost, the whole
  gradient's static side (its shapes captured, 3.9–5.0 s, and the points
  costed, 1.1 s) charged to each nest, is 16.4–19.9 % and 9.9–11.9 % of the
  head's two nests' sweeps, of 30 and 51 s; 0.2–1.7 % on the blocks'. L
  holds. 88.5 % of 92 279 close pairs are ranked right, attention's figure,
  98.6 % of those the model decides. Over one gradient, the plan's layouts
  take the derivative launches within 1.05× of each shape's best, 24×
  faster than 0.9.0's derivative kernels. The model is not refitted.
- Step 1: the derivatives of `dot` as contraction nests (D89).
  `dot_vjp_l` is the builder's NT form. `dot_vjp_r` sums over every leading
  axis of its operands as one summed slot over several axes, iterated as one
  loop per axis, so no operand is reshaped; a k-block divides the innermost.
  Their layouts are accepted by a plan (`E7002` checks them as any
  contraction's) and stated by `presk plan`, so the plans it gives for a
  gradient, and `presk train`'s, tile them. At the default layout their
  kernels are 0.9.0's byte for byte, and on the eleven entries of
  `tests/plan/entries.txt` in three types `main`'s binary and this one give
  the same kernels, outputs and exported C programs. Every point of every
  derivative nest of three gradients is bit-identical to the default on
  both executors (4 280 points), and C1 holds under the frozen profile's
  plans, which lay out every derivative kernel of `mse_grad` and
  `loss_grad`. The pack of `dot_vjp_l` copies a matrix that carries no
  outer axis. T2's first reading (`make pytorch-baseline`, kept as
  `docs/pytorch-baseline-0.10.0-step-1.md`): on the ten X925 cores the C
  training step under the plan takes 2.85 s against PyTorch's 1.03 s,
  2.76×, where it took 12.2 s, 12.1×. The derivatives take 0.36 s of the
  gradient's 2.72 s of kernels at 485 GFLOP/s, where they took 9.7 s at
  18. `presk train` under the plan takes 12.8 s, where it took 22.3 s. The
  forward products the gradient recomputes are now half its kernel time
  (steps 3 and 4).
- Step 0: bookkeeping (D88). The user's five decisions of 2026-09-28, taken
  as recommended when the plan was merged: 0.10.0 before the CUDA export,
  steps 0–2 before PLDI's deadline of 12 November; a gradient keeps its
  forward pass's residuals by default, recomputation kept as an option; in
  the kernels, the bias as an epilogue and the C flags measured, no fusion;
  the native pool written on `std`, no `rayon`; the thresholds 2× (T2) and
  1.3× (T4). The baseline's second run of record kept as
  `docs/pytorch-baseline-2026-09-28.md`, the version's starting point.
  Spec question 76, store or recompute; it cites spec §5's `checkpoint e`,
  an annotation the plan did not cite and the compiler does not parse, and
  plan §6 is amended to cite it. `experiments/pytorch_baseline.py` titles
  its report with the version it times, where it wrote 0.9.0. Version
  `0.10.0-dev`. Nothing else is implemented, and the native backend's code
  is unchanged.

**The PyTorch baseline at 0.9.0** (D87; the manual's `experiments` §3,
"Three sides at 0.9.0"). A measurement, not a version. The course GPT at
`B = 16`, `T = 256` in `f32` is timed on three sides: the native backend,
the C program the export writes, and PyTorch 2.14 running the exported
model. There are two workloads, the forward pass and one training step,
each under the default layout and under the frozen profile's plans. The
row of record is the ten X925 cores pinned, with ten threads on every side;
every core unpinned is the second row. Each figure is the median of seven.
- **The forward pass.** The C program takes 281–285 ms under the plan
  against PyTorch's 153–156 (1.8–1.9×; 1.13× on every core); the native
  backend takes 697–710 ms.
- **A training step.** C takes 12.2 s (the gradient plus `adamw.step`),
  `presk train` 22.3–23.0 s and `train.py` 1.0 s. At the default layout on
  every core the native step is still 22× PyTorch's, as on 23 September.
- **Where the time goes.** Four fifths of the C gradient is the
  derivatives of `dot`, at 18 GFLOP/s, which the layout space does not
  reach. Each `.vjp` recomputes its callee's forward pass, so a block's
  linear products run six times a gradient. On the forward pass four
  fifths of the native backend's lead over C is its host, Cranelift's code
  1.3× GCC's. The contraction rate, 14 of D55's 22, is now within 1.3–2×
  of PyTorch's libraries.
- **The counts.** 96.93 GFLOP a forward pass and 290.78 a step, exactly as
  PyTorch's counter gives them; the 266 of 0.4.0 and D55 was 8.5 % low.
- `make pytorch-baseline` (`experiments/pytorch_baseline.sh` and `.py`)
  writes `out/pytorch-baseline/report.md`: the times and rates, the
  distances to the native backend's `f64` logits, the linear products
  alone on each side, the native kernels' share of a call, and the C
  program's time kernel by kernel. The last comes from
  `experiments/kernel_time.py` and `.c`, a copy of `model.c` with every
  launch timed, the kernels untouched.
- `experiments/gpt_grad.psk`: the course GPT's gradient, which exports to C
  at its constants (2 220 launches, an arena of 1.22 GB, C1 holds).
- `presk bench --warmup N`: calls before the timed ones, not timed.
  `presk bench --calibrate` prints `-- a call:`, the kernels' share of a
  call.

## 0.9.0 — 2026-09-27

A C export (`presk-specs/implementation/presk-plan-0.9.0.md`), in the style
of llama2.c and StrasGPT: one C program per exported entry, which runs
without the compiler and computes the bits the native backend computes. The
kernels come from PSK-Kernel, where the fusion, the realization and the
layout plan are already applied, printed one C function each. The host comes
from a recording of the native backend's own run at the entry's dimensions,
with one C function per Presk function and the layer fold as a loop. Any
entry the native backend runs can be exported, gradients, one optimizer
step and a generation function with `presk serve`'s loop included, under a
plan, on OpenMP threads, with integer weights; the training loop cannot.
Bit identity rests on building without floating-point contraction and on
shipping musl's transcendental functions, which the native backend calls
through the `libm` crate. A SCoP mode writes each nest at point zero, for
polyhedral tools. Judged on five statements: bit identity with `presk run
--backend native` (C1), the exact reference's band (C2), compilation
without warnings on GCC and Clang and a cross-compilation to x86-64 (C3),
the wall time reported (C4), and `presk serve`'s tokens (C5). The GPU and
the matrix unit come after it.

**What it shows.** An entry can leave the compiler without leaving its
numbers behind. Every exported entry of the test set, in `f64`, `f32` and
`bf16`, under the default layout, a mixed realization and `presk plan`'s
plans, through both hosts, with OpenMP and without, on one thread and on
every thread, computes the native backend's outputs bit for bit, and so do
the course GPT at its constants with its trained checkpoint, 42 programs
with integer weights and every nest in the SCoP form (C1). The outputs are
therefore outside the exact reference's band exactly where the native
backend's are, on elements that cancel (C2, question 75). The C compiles
without a warning on GCC with and without OpenMP, on Clang, and for x86-64
(C3), and the exported generation loop prints `presk serve`'s tokens byte
for byte, greedy and sampled (C5). The course GPT's forward pass at `B =
16` takes 290 ms in C against 1 094 ms natively on ten threads, 1 700
against 3 132 ms on one (C4, reported). Getting there took three facts the
plan did not have: current musl is not the `libm` crate (musl 1.1.22 is),
`-std=c11` alone stops GCC from contracting, and the native backend's own
`sample` called the platform's `exp` (D78, D77, D83).

- Step 9: documentation, tutorial step 18, version 0.9.0 (D86). The manual
  brought to 0.9.0 as implemented: a chapter of its own for the export,
  `export_c`, made from the `native` chapter's section, which had grown a
  paragraph per step, and reorganized (how a program is made, its
  structure, threads, sampling, integer weights, the exactness rules, what
  is checked, portability, the band, speed, the SCoP mode, and what the
  export does not do); `experiments` §9, the statements of 0.9.0 and their
  verdicts; `cli` and `diagnostics` checked against `presk --help`, `presk
  export --help` and the codes the code emits; `status` closed; `index`,
  `install`, the manual crate's chapter list and the README. Tutorial step
  18, "A C program": the tutorial's attention exported at step 3's
  dimensions, built, checked against the native backend and the exact
  reference and run by hand, then under step 17's plan, on OpenMP threads
  and on one, and in the SCoP mode; `make tutorial` covers steps 0 to 18
  and needs a C compiler. The export's summary line says "1 function
  besides the entry's", where it said "1 functions", and `presk --help`
  names the C export. Questions 73–75 copied to `presk-specs` as
  `presk-spec-questions-0.9.0.md`. Version `0.9.0`.
- Step 8: the SCoP mode (D85). `presk export --target c --mode scop --entry
  E` writes each nest of the entry at point zero (its default layout on one
  lane and one worker, no pack, 0.6.0's kernel), captured from the entry's
  shapes as `presk plan` captures the nests RA4 ranks, one C function per
  nest, shape and type, each in a file of its own between `#pragma scop` and
  `#pragma endscop`: arrays of literal dimensions with one index per axis,
  as PolyBench/C writes its kernels; every loop to a literal bound, no range,
  no OpenMP; a branch on a computed value that only assigns scalars (a
  softmax's running maximum, an argmax) printed as a selection of each
  value; the transcendental functions declared `const` in the file, and
  `presk_rt.h`'s inline conversions `const` (`PRESK_CONST`); the pragmas'
  `-Wunknown-pragmas` warning ignored around the function, so that each file
  builds without a warning on GCC and Clang. A driver (`scop`) times each
  nest on inputs it draws (`splitmix64`, indices below the axes they index)
  and checks each against the native backend's outputs on the same inputs
  (`conformance/native.bin`, `make check`); `make polly POLLY=CLANG` gives
  Polly's view of each nest; `nests.json` and the README list the nests,
  their loops and whether each is static control. `--plan`, `--threads`,
  `--straight-line`, `--serve`, `--weights`, `--no-exact` and a quantized
  realization are refused. A nest is a static-control part when no loop
  variable is assigned and every access is affine in the loop variables: on
  the test set 366 of the 383 nests of each type, the others being gathers
  and scatters (an embedding, its gradient, a cross-entropy's label). The
  front end is Polly, from LLVM 23.1.2's release (`make llvm`, into `.llvm`,
  its checksum pinned, without root), which links it into its clang: at its
  early position, whole functions allowed as SCoPs, no rescheduling, the
  analysis only, it takes each of the 1 098 static-control nests of the
  three types as one SCoP of the kernel's loops and none of the 51 others
  (all of them once allowed non-affine accesses); the course GPT's 28 nests
  alike (`make c-conformance`). None of the plan's front ends was on the
  machine: Polly was chosen over GCC's Graphite, whose verdict follows GCC's
  passes rather than the nests, and pet, which would have to be built
  against Clang's libraries. Every primitive at point zero in the SCoP form
  (123 kernels) and every nest of the test set in the three types are
  bit-identical to the native backend; built in GCC's GNU mode, or with a
  softmax's maximum made a minimum, they are not. The native backend's
  kernels and outputs are `main`'s (66 comparisons); `make test` takes 3 to
  7 s more.
- Step 7: integer weights (D84). `presk export --target c` exports a
  realization that quantizes weights (`int8`, `int4`, `absmax` or `pow2`).
  No checkpoint holds integers: the native backend quantizes the weights
  each time it loads them, so the C program reads their reals and quantizes
  them as it reads them (`presk_quantize` in the runtime, a transcription of
  `QTensor::quantize`), bit-identical to presk-rt on 4 027 392 weights of
  both widths and schemes, ties at `n + ½`, zero channels, subnormals, huge
  values and NaNs among them. The conformance weights are the parameters as
  read, where they were saved after quantization as `f32(q·s)`, whose
  re-quantized scales need not be the same. A leaf of integer weights is a
  `presk_q8` or `presk_q4`, its integers and scales, as llama2.c's
  `QuantizedTensor`; a kernel takes the integers and, after its output,
  their scales, and loads `(float)q * s[c]`; the recorder follows the host's
  conversions at a kernel's edge (a norm's `γ` read by its fusion group) and
  its slices, integers where `QBuf::slice_leading` keeps them, `f64` copies
  where it gives up (an `int4` layer of odd length), which the structured
  host dequantizes before each call of the fold's body. `E8007` narrowed to
  integer weights in the result, under a reshape, or sampled
  (`tests/diag/export_c_quant.psk`). presk-rt's `pow2` scale, which looped
  for ever on a channel whose largest weight is below about 10⁻³⁰⁶, now
  starts its loops at 1, its scales otherwise unchanged. Step 1's check on
  integer inputs (2 880 primitive runs, the four nests' layouts); C1 on 42
  programs through the binary (`odd.psk` in both widths and schemes,
  MiniGPT's forward pass, `next_token`, gradient and optimizer step with
  every matrix quantized, at
  `D = 9`, `H = 3` in `int4`, with its biases and norms quantized, under a
  plan with splits), C5 under `--serve`, and the trained checkpoints in
  `make c-conformance`; a scale one channel off, the two nibbles of a byte
  swapped and the scale kept in `f64` when quantizing each change the
  outputs. The native backend's kernels and outputs are `main`'s (66
  comparisons); `make test` takes 11 s more.
- Step 6: randomness and generation (D83). `sample` takes no key: it draws
  from the backend's stream, seeded by the run's seed, so the plan's key
  functions were not needed and keys stay dropped. The recorder records it
  as a host operation (`E8005` retired), and the runtime gains the stream,
  `presk_rng` (Bernstein's ChaCha20 read as `rand_chacha`'s `ChaCha20Rng`,
  seeded by `rand_core`'s `seed_from_u64`), and `presk_sample`; checked
  against the crates on 10⁶ draws over 10³ seeds and against the native
  `sample` on 12 028 rows in the three types. The stream is the program's
  state: a function that draws takes it after the arena, `main` seeds it
  with `-s SEED` (the export's seed by default). The native `sample` now
  calls the `libm` crate's `exp`, the kernels' own, where it called the
  platform's (glibc): the C program depends on no C library, and 540
  generations of 32 sampled tokens are unchanged. An entry's `f64` argument
  (a temperature, an `eps`) is a `double` of its `_in` struct, set at run
  time, where it was fixed at export. `presk export --target c --serve`
  exports the generation function of the file's `inference serve` at one
  sequence and the window `--dim T=N`, with its greedy copy (every `sample`
  an `argmax`) recorded on the same backend and sharing its functions, and
  `generate.c`, `presk serve`'s loop, whose output, `--json` included, is
  `presk serve --backend native`'s byte for byte (C5); a model whose
  parameters do not fix the context's length is refused (`E8009`). C1 on
  `next_token` (`make test`, and with the trained checkpoint in `make
  c-conformance`); C5 in the three types through both hosts, greedy and
  sampled at three temperatures and four seeds, on the example with its
  zeros drawn (`make test`) and on the trained checkpoint (`make
  c-conformance`); C3 on `generate.c`. A draw skipped, the stream seeded
  again at each step and the window one token behind each change the
  tokens. The native backend's kernels and outputs are `main`'s (66
  comparisons); `make test` takes 3 s more.
- `presk plan` fixed after step 5's review (D82). A nest launched at
  several shapes gets the layout the model prices at every one whose
  predicted time, each shape's weighed by its launches (the capture now
  counts them), is least; the first shape decided, and the MLP's plan stated
  a tile its second shape refused. A type the profile was not calibrated in
  is refused when no nest can be priced (exit code 2, nothing written),
  where an empty plan was written, and its nests are named when some can.
  On the test set only the MLP's plans change.
- Step 5: plans, threads and portability (D81). `presk export --target c`
  takes `--plan @P.json`, checked as `presk run` checks it, and records the
  native run under it, and `--threads N`, the native executor's workers;
  `E8002` is no longer emitted. A launch the native backend divides among
  threads, a plan's split or a kernel above the executor's threshold, is one
  OpenMP region whose threads each call the kernel on the native backend's
  piece. A `num_threads` clause overrides `OMP_NUM_THREADS`, so the team is
  capped at `omp_get_max_threads()`: the variable lowers it, and
  `OMP_NUM_THREADS=1` gives one thread. Without OpenMP each region is one
  call, and the `Makefile` builds with `-fopenmp` only when a launch is
  divided. The split is part of the launch and of the canonical form, since
  a kernel's text lacks its workers. `main.c` takes `-r R` and prints the
  median and best of `R` calls; `presk bench` gains `--plan-only` and
  `--weights`. C1 on every program built with OpenMP and without and, when a
  launch is divided, on one thread: under a plan stating splits and at a size
  the native backend divides by its own rule (`make test`); every kernel of
  five recordings divided at each legal depth in turn, 40 programs (`make
  test`); the test set under `presk plan`'s plans in the three types and the
  course GPT at `B = 16`, `T = 256` under its plans, 201 of 228 launches
  among ten threads (`make c-conformance`). A gap between pieces changes the
  outputs; an overlap computes the same bits twice, and a check of the
  pieces in the tests catches it. C3: no warning on GCC with and without
  OpenMP, on Clang 21.1 (linked, run, the native bits) and for x86-64
  (linked), both through zig 0.16.0 from PyPI (`make zig`, `.venv-zig`). C4
  (`make c-speed`): at `B = 16`, every thread, native 1 094 ms and C 290 ms;
  one thread, 3 132 and 1 700 ms. Found in `presk plan`: the plan it writes
  for the MLP is refused by `presk run` (a layout chosen at one shape of a
  nest does not divide another), and in `bf16` it states no layout. `make
  test` takes 11 s more.
- Step 4: the host, structured (D80). The recorder keeps the structure
  beside its list of operations: each call with its function, dimensions,
  parameters' source names, arguments and result, each fold with its
  iterations and its scan's stack, every operation in one body; each buffer
  carries its `let`'s source name. The export puts each call in a canonical
  form (arguments by position, buffers by first definition, kernels by their
  text, callees by their forms, constants folded into the operations that
  read them) and writes one C function per distinct form, named after the
  Presk function and, when it has several, its dimensions
  (`presk_Linear_apply_In16_Out64`). A result is written into buffers the
  caller allocates, and each function releases what it allocated when it
  returns, so nothing is moved. A fold is a loop over the runs of its
  iterations that are equal up to offsets affine in the index: the forward
  stack, the scan (its stack written slot by slot) and the backward fold
  (its body taking the index), the stacked parameters sliced as `w + l *
  chunk`, the accumulator alternating between two buffers. An iteration that
  differs from the next runs apart, as its own function, the first block
  under a mixed realization. The course GPT's `GPT.apply` is a loop over its
  six blocks. Buffers are named after the source. The arena is sized at the
  program's peak, which the tests read back from the running program: the
  course GPT at `T = 256` takes 117, 59 and 29 MB in `f64`, `f32` and
  `bf16`, where releasing nothing took 352, 176 and 88 MB. Runtime scalars
  are `double` parameters. The straight-line host stays as the oracle,
  `--straight-line`: the 68 programs of `make test` and the course GPT's 9
  are exported both ways, bit-identical to the native backend and to each
  other. New mutations of the structure are each caught: a slice off by a
  layer, a loop bound off by one, a release before a buffer's last use. A
  test program returns a fold's value, `tests/export_c/fold_value.psk`. The
  native backend's kernels and outputs are unchanged, byte for byte.
- Step 3: the host, straight-line (D79). `presk export FILE --target c
  --entry E [--dim] [--const] [--realize] [--weights] [--seed] [--no-exact]
  -o DIR` writes the entry as a C program: `model.c`, one straight-line
  function performing every operation of the native backend's host in
  order, every buffer in one arena; `kernels.c`; `main.c`, which reads the
  weights and inputs and writes the outputs as `.psk.d` files; the runtime;
  a Makefile with the flags bit identity rests on; a README; and
  `conformance/`, the inputs, the native backend's and the exact
  reference's outputs, and a script that builds, runs and compares with
  `cmp`. The host is recorded by `presk-kernel`'s new recorder, a symbolic
  mirror of the kernel backend's host: its values carry no identity, so
  each tensor is followed as a view of a buffer at an offset through calls,
  `let`s, tuples, records, closures, the `stack` fold and each primitive's
  launches, and every launch is the backend's own. It records the native
  run itself, switched on for the entry's call only. Scalar arguments are
  fixed at export as `presk run` synthesizes them, keys dropped; a
  checkpoint's header is what the C reader expects byte for byte. New codes
  `E8003`–`E8008` (unbound dimensions, an initializer, `sample`, a key, integer
  weights, a scalar result); `E8002` now refuses `--plan` and `--threads`
  until step 5. `presk-cli` depends on `presk-export-c`. C1 holds on the
  test set but `next_token` in `f64`, `f32` and `bf16`, on the synthesized
  parameters and with their zeros drawn (MiniGPT's logits are zero
  otherwise), and under a mixed realization: 68 programs, 4.8 s in `make
  test`; and on the course GPT at its constants, with the course's trained
  checkpoint at `T = 256` (`make c-conformance`). C2 is reported: the
  elements outside the band are the native backend's, question 75's
  cancellations, now seen in forward passes too. Dropping a live launch,
  swapping two launches' buffers and rounding a `bf16` input once are each
  caught. The native backend's kernels and outputs are unchanged, byte for
  byte, on the eleven entries of `make plan-check` in the three types.
  On review: the kernel conformance suite also runs the examples with their
  zeros drawn, asserting the band where it holds and reporting
  `MiniGPT.apply`'s cancelling logits in `f32` and `bf16`; question 75 gains
  the forward-pass case and the finding that two conforming executors leave
  the band on different elements; the work a gradient discards is recorded
  in the kernels roadmap for a step of its own.
- Step 2: the runtime and the transcendental functions (D78). `presk-export-c`
  gains the C every exported program holds beside its own. `c/presk_libm.c`
  is `exp`, `log`, `tanh`, `sin`, `cos`, `erf` and their `f` forms with the
  helpers they call, from musl 1.1.22: the `libm` crate the native backend
  calls ports musl as it was before 1.1.23, which replaced `exp`, `expf`,
  `log` and `logf` with ARM's routines, and current musl's four differ from
  the crate on 8–10 % of inputs. The files are renamed `presk_…`, their
  helpers made static and their constants moved into the functions that read
  them; no operation changes. Each function is bit-identical to the crate on
  10⁶ inputs, special values, subnormals, branch thresholds and the range
  reductions' boundaries included, in 0.4 s of `make test`; an `#[ignore]`d
  test runs each `f32` function on every `f32`, bit-identical, in 46 s.
  `c/presk_rt.c` converts a buffer between machine types as the native
  backend does (to `bf16` through `f32`, two roundings), reads and writes
  `.psk.d` files from the type as text and its leaves as the export states
  them, with `presk-data`'s error codes, and holds the arena. Files written
  by `presk-data` are read in C bit for bit, and files written in C are
  `presk-data`'s byte for byte. Step 1's check runs with `presk_libm.c` in
  place of the stand-ins, bit-identical; built in GCC's GNU mode without
  `-ffp-contract=off`, 14 of its 246 primitive runs move, as the contraction
  rule says they must. `presk-data` becomes a dev-dependency of the crate.
  `--target c` is still refused, and the native backend's code is unchanged.
- Step 1: the kernel printer (D77). A new crate, `presk-export-c`, prints a
  PSK-Kernel kernel as one C function, its textual form as the comment above
  it: literal loop bounds, offsets multiplied out, lanes as loops over a
  local array, `bf16` as `uint16_t` rounded as `presk_rt` rounds, a split's
  `[lo, hi)` on the loop at its depth, and `presk_exp`, … declared for step
  2. Integer weights are refused until step 7. The check runs the C inside
  the kernel backend: an executor over the compiled C, loaded with `dlopen`
  (the `libc` crate, a dev-dependency already in the lock), compiled with
  `-std=c11 -O2 -ffp-contract=off -Wall -Wextra -Werror`. The C is
  bit-identical to the native backend on the conformance suite's 41
  primitives in the three types, scalar and with lanes, on the 1 491
  layouts and 1 618 thread splits of the four nests of attention, and on
  hand-written kernels. Every range is cut three ways, and no two pieces
  write the same element. The transcendentals are compared too, through
  the `libm` crate's functions standing in for step 2's. `make test` gains
  11 s and now needs a C compiler, which `rug` already did. Two facts of
  the plan corrected: GCC ignores `#pragma STDC FP_CONTRACT OFF` and warns
  about it, so the pragma is for Clang only; and `-std=c11` alone already
  stops GCC from contracting, so the contraction test of step 2 must build
  in a GNU mode. `--target c` is still refused.
- Step 0: bookkeeping (D76). The user's five decisions of 2026-09-26: the C
  export as 0.9.0, before the GPU; the crate `presk-export-c` behind `presk
  export --target c`; any entry the native backend runs, not the training
  loop; bit identity with the native backend, musl's functions shipped; the
  host by recording. Steps 7 (integer weights) and 8 (the SCoP mode) kept.
  `presk export --target c` is parsed, the program checked, then refused
  with `E8002` and exit code 2 before anything is written, tested by
  running the binary. The plan checked against the code and amended in two
  places: the course GPT uses `Rand` for dropout only, whose masks are all
  ones outside a training run, so it is exported from step 3 and not
  refused until step 6; and MiniGPT's `loss` is a forward pass, not a
  gradient. The test set gains a gradient through attention, `loss_grad` of
  `tests/plan/minigpt_grad.psk`, an eleventh entry of `make plan-check`: the
  gradient of MiniGPT's loss with the output head drawn, since the
  example's zero head leaves every other gradient zero. The native
  backend's own gradient crosses the element-wise band on 34–121 of its
  8 768 elements where terms cancel, so C2 is judged against the native
  backend's gap. The Cranelift crate's documentation and the `native` chapter
  said its `libm` was the kernel interpreter's; that holds for `erf` only.
  Version `0.9.0-dev`; spec questions 73–74 from the plan, and 75, how
  conformance judges an output that cancels. Nothing else is implemented, and
  the native backend's code is unchanged.

## 0.8.0 — 2026-09-26

RA4 re-tested (`presk-specs/implementation/presk-plan-0.8.0.md`). 0.7.0's
static model ordered lanes and thread splits and could not see a register
tile, a loop order or a k-block. It gave a load no weight, and its traffic
terms never bound under a maximum (D62). 0.8.0 changed the whole model: a
load term, a combination that lets a term that does not bind still separate
two points, and a calibration that identifies the traffic constants. It was
designed on 0.7.0's measurements alone, which became the development set and
decide nothing, then frozen in a tagged commit, `ra4-model-0.8.0`, before
any new measurement. The test was two fresh sweeps, each measured twice:
new shapes on the ten X925 cores (BERT-large's attention swept whole, and a
head dimension of 128), and 0.7.0's shapes on the ten A725 cores with their
own calibration. Statements L, M, N and O as 0.7.0 posed them, and Q for
the hybrid chooser, which measures only the model's candidates. The GPU and
the matrix unit come later. The version does not promise to close the 22×
against PyTorch.

**What it shows.** The model the failure named works, and does not quite
reach the hypothesis. Every point of both fresh sweeps, 25 588 a day, is
bit-identical to the scalar sequential kernel, natively and on the kernel
interpreter (L). The frozen model ranks 92.7 % and 91.7 % of the attention
`ein`s' distinguished close pairs right at 0.7.0's shapes on the A725 cores,
where RA4 asks more than 90 % and 0.7.0's model ranked 47.9 % and 43.9 %;
but 87.4 %, 88.5 %, 85.4 % and 88.8 % at BERT-large's attention and at a
head of 128 on the X925 cores, so M fails. Ties fell from 34 % to 0.1 %, the
pairs the model decides are about 99 % right, and 97–99 % of the misses are
pairs it predicts within its own uncertainty. Its pick is within 1.05× of
the best point on every contraction nest M names (O holds; 0.7.0's was
2.18–3.01×). Measuring only its candidates within `1 + u` finds a point
within 1.03× of the best on all eighteen contraction nests, at most 1.8 % of
the sweep's time, the static side included (Q holds). The static choice is
1 013–2 933 times cheaper than the sweep, but 218–252 times at batch one on
the A725 cores, where costing the points is most of it (N fails). The
development figures, 89.3 % and 89.6 %, foresaw the new shapes within four
points (D74).

- Step 8: documentation, version 0.8.0. The manual brought to 0.8.0 as
  implemented: `optimization` gains the A725's calibration and what the
  fresh sweeps found, what the model ranks by decision on each sweep and
  where it is wrong, and its forward-looking sentences are made past;
  `experiments` §8, the statements of 0.8.0 and their verdicts, every number
  from D72–D74 and the committed reports; `cli` checked against `presk
  --help` and each command's help (the plan's pick under 0.7.0's model and
  the frozen one, the `presk rank` commands that ranked the fresh
  sweeps); `realize` and `native` with
  the frozen model's pick (445 GFLOP/s at 1024³ on the development data,
  where 0.7.0's picked 75); `status` closed, with the answers, the
  verification and what the version has and has not; the index, `install`,
  the chapter summaries, README and CLAUDE.md follow. The frozen model's
  form becomes the default of `presk bench --calibrate --space calib` and
  `presk refit`: the load term per group, the p-norm, and for a new profile
  the rules `lines` and `overhead`; `refit --recost` costs under the
  profile's own rules unless `--rules` says otherwise, and `--rules` needs
  `--recost` (D75, the user's decision). The definitive calibration,
  re-costed under `lines,overhead` and refitted with the defaults, gives
  the frozen profile back byte for byte. Tutorial step 17 run again with
  the layout model calibrated under those defaults, its quoted outputs
  refreshed: 300 layouts
  a size, `u` = 0.094, tiles 98 % right where 0.7.0's model ordered 39 %,
  O holding at 1.15, and Q, new in the report, failing at this toy size.
  Every other quoted output of steps 0–16 agrees with the run but for
  timings, and `presk --help`, stale since 0.7.0, now lists `reprice` and
  `refit`. `docs/kernels-roadmap.md` records what 0.8.0 built of its design
  and what D71 and D74 left for later: the distinct lines an iteration's
  loads touch, which would see the pack; why the maximum forms rank
  k-blocks better; how the L2 term combines with the load term at one-row
  tiles, where the packed operand exceeds L1; the order of the pairs within
  `u`, which decides M; and the costing of a point, most of N's static side
  at batch one. Spec questions 71 and 72 copied to `presk-specs`
  (`presk-spec-questions-0.8.0.md`). Version 0.8.0. The CHANGELOG gains
  steps 4 to 7, which it lacked. Nothing is measured, and nothing the
  freeze covered changes: `make ra4-dev` and `make ra4-dev-frozen` give back
  their reports byte for byte. Verification on falicon: `make test` (153
  tests), `check-examples`, `plan-check`, `ra4-dev`, `ra4-dev-frozen`,
  `ra4-small` under the frozen profile (every point exact on both
  executors), `tutorial` (twice), `conformance` and `cargo doc -p
  presk-manual` pass.
- Step 7: the statements (D74). Nothing measured: both fresh sweeps ranked
  from their archived days under the frozen model. `presk rank` decides Q,
  its cost counted as question 72 answers it before any data: the whole
  configuration's static side, charged to each contraction nest, plus the
  candidates' measured time, over the nest's exhaustive sweep, on each day,
  with D71's share of the timed launches beside it; a table per nest
  follows the statements. Days recorded before the capture of shapes need
  `--statement-q`, so 0.7.0's reports come back byte for byte. The reports
  are `docs/ra4-f2-2026-09-25.md`, `docs/ra4-f1a-2026-09-25.md` and
  `docs/ra4-f1b-2026-09-25.md`. **L holds**, **M fails** (F1 85.4–88.8 %,
  F2 92.7 % and 91.7 %), **N fails** (218–252× at batch one on the A725
  cores), **O holds** (at most 1.05), **Q holds** (regret at most 1.03, cost
  at most 1.8 %). The one class of pairs the model decides and gets wrong
  is the loop order of one-row tiles at the head of 128, where the p-norm
  adds the L2 term to the loads and the machine does not pay it. No defect
  was recorded under the freeze, and this step ended it.
- Step 6: F1, new shapes on the X925 cores (D73). The targets `ra4-f1-dry`,
  `ra4-f1`, `ra4-f1-check`, `ra4-f1-day2` and `ra4-f1-rank`, priced by the
  frozen profile itself: two configurations in one day, BERT-large's
  attention (F1a: `D = 1024`, `H = 16`, `T = 384`, `B = 1`) and a head of 128
  (F1b: `D = 1024`, `H = 8`, `T = 512`, `B = 2`), in `f32` over the whole
  space; a dry run per configuration, and a day that keeps a configuration
  whole while it fits in 8 hours; two ranks of the same days, since `presk
  rank` decides M and O at one configuration. Measured on 2026-09-25, twice,
  on one date (the user's decision): 9 264 points in 48 minutes a day, the
  dry run's estimate 52; every point bit-identical natively and, in 100
  shards, on the kernel interpreter. Every piece archived read-only in
  `~/presk-data/ra4-f1-0.8.0` with checksums, and in presk-project `data/`.
- Step 5: F2, 0.7.0's shapes on the A725 cores (D72). `make
  calibrate-ra4-small` (the frozen form and space, a profile of the A725's
  own), `ra4-f2-dry`, `ra4-f2`, `ra4-f2-check`, `ra4-f2-day2` and
  `ra4-f2-rank` (`presk rank --frozen`). `experiments/ra4_fresh.sh` refuses
  a fresh day that would not count before it is measured: a day that
  exists, an uncommitted tree, a commit that does not descend from the tag,
  day two on day one's date unless `RA4_SAME_DATE=1`, a running `presk`. A
  dry run sizes a day from one nest's sweep and the model's predictions
  (`experiments/ra4_dry.sh`, `ra4_dry.py`), 5 % short on 0.7.0's day one.
  `presk rank` leaves M inconclusive on a configuration swept on a sample,
  decided before any fresh data. The A725 calibrated in 101 minutes, `u` =
  0.168 against the X925's 0.132; F2 measured twice on 2026-09-25, 16 324
  points in 1 h 45 min a day, every point bit-identical natively and, in 88
  shards, on the kernel interpreter; everything archived read-only in
  `~/presk-data/ra4-0.8.0-a725-calibration` and `~/presk-data/ra4-f2-0.8.0`.
- Step 4: development on 0.7.0's measurements, and the freeze (D68–D71).
  `bench --min-workers` and a calibration that starts from an empty
  profile. The definitive calibration on the X925 cores: 3 336 samples in
  44.9 min, `u` 0.142 with the load term and 0.241 without, archived
  read-only (D68). The ablation on the development set (`make
  ra4-ablation`): statement M's nests go from 47.9 % and 43.9 % right under
  0.7.0's model to 84.6 % and 84.4 % under a load rate per group and the
  p-norm, the form chosen (D69). `presk refit --recost` costs a
  calibration's samples again from their layouts, its times kept, and gives
  the definitive calibration back byte for byte. Three rules of the planned
  cost, each named by a profile and off by default so that older profiles
  price bit for bit: cache lines and loop overhead kept (M's nests 89.3 %
  and 89.6 %), buffers sharing a level not (D70). The nine forms again under
  those rules, the chosen one still leading. The frozen profile archived
  read-only in `~/presk-data/ra4-model-0.8.0`, its development report in
  `docs/ra4-dev-model-0.8.0.md` and checked by `make ra4-dev-frozen`,
  statement D, `bench --rules`, `make calibrate-ra4` in the frozen form by
  default, and the tag `ra4-model-0.8.0` on 2026-09-25 (D71).
- Step 3: the combination and the wider calibration (D67). The layout model's
  terms combine as the maximum (0.7.0's), the sum of the in-core and memory
  terms, or a p-norm with its `q` fitted (`--combine`). Each form is
  recovered from synthetic samples. `presk refit` fits a profile's samples
  again in any form, and `--compare` prints all nine. The calibration times
  300 layouts a shape (five tiles, ten orders and k-blocks, both lane
  widths, 1, 5 and 10 workers) on the GEMM at 256 to 2048 and two thin
  products (`experiments/matmul.psk`). Measured on the X925 cores (3 536
  samples), the load term takes `u` from 0.226 to 0.133, the three
  combinations fit alike, and the p-norm settles at the sum. Not met:
  `β_mem` comes out above `β_L3` because the thin products are not bound by
  memory on this machine; `ρ` at one lane is not identified; and the
  calibration took 3 h 13 min, two thirds of it converting exact inputs at
  every call, which is now done once.

- Step 2: the load term (D66). `planned_cost` counts one worker's register
  loads and stores, every load and store once whatever its lanes, and the
  spilled registers' accesses move from work to loads. The layout model
  gains `(N_ld + S)/ρ_g` in its maximum, with `ρ` fitted per group, once for
  the machine, or not at all (`presk bench --calibrate --space calib
  --load-term group|machine|none`). A profile without `ρ`, 0.7.0's among
  them, predicts exactly as before, and `make ra4-dev` still reproduces
  both of 0.7.0's reports. The count is checked on the six tiles of plan
  §1, a pack and a spilling tile. A synthetic fit recovers `ρ` within 10 %.
  On the development set the count's rank correlation with the measured
  times is 0.953, where 0.7.0's prediction reached 0.437; that is the data
  the term was designed on, not its test. A calibration drops earlier
  samples that have no load count.
- Step 1: `presk reprice`, a cheaper static side, `presk rank --frozen` (D64).
  `presk reprice` prices a recorded day again under another profile: every
  point lowered again from its layout and costed, its times and every other
  field copied as written, nothing measured. `make ra4-dev` re-prices the
  development set with 0.7.0's profiles and ranks it in half a minute: all
  52 142 lines come back byte for byte, and both of 0.7.0's reports too. That
  is step 0's check. `presk plan` and `presk autotune` lower a layout once
  for all its thread splits and share the footprints between the three
  cache levels' traffic, and cost the attention's space 1.7–2.2× faster
  with every prediction unchanged. `presk plan` and `presk reprice` capture
  the entry's launches from its shapes (D65): the host program on zero
  inputs, parameter trees built by their initializers, no kernel compiled or
  run, nothing computed in exact arithmetic, and the same launches as a
  native run on ten entries in three types. That capture and the costing are
  now statement N's static side, decided before any fresh measurement.
  `presk autotune` records the capture's time, and `presk rank` prints the
  ratio with the capture that runs the entry beside N's. The static side
  goes from 1.51 to 0.54 s at batch 16 in `f32`, and from 1.76 to 0.45 s at
  BERT-large's attention. `presk rank --frozen TAG` refuses a day measured
  on a commit that does not descend from the tag, or on a modified working
  tree, which `experiments/ra4.sh` now marks `-dirty` in `day.txt`.
- Step 0: bookkeeping, and the development set kept (D63). 0.7.0's raw
  measurements (`out/ra4`, `out/ra4-20`, `out/ra4/check`), its two profiles
  and its two reports, 1 605 files and 76 MB that no commit holds, copied
  read-only to `~/presk-data/ra4-0.7.0` on falicon with a checksum list;
  `presk rank` regenerates both committed reports from there byte for byte.
  The step's own check, 0.7.0's report regenerated by re-pricing every point
  with 0.7.0's profile, needs `presk reprice` and is step 1's. A sentence of
  `docs/kernels-roadmap.md` corrected: it asked for "a footprint analysis
  that starts at L1", but the model has counted the bytes from L2 into L1 as
  `Q_L2` since 0.7.0 step 4, and they differ between loop orders; what hides
  the orders is the maximum. Version `0.8.0-dev`; spec questions 71–72; the
  A725 cores recorded as `RA4_CORES_SMALL` (`0-4,10-14`).

## 0.7.0 — 2026-09-24

Ranking, not speed (`presk-specs/implementation/presk-plan-0.7.0.md`).
Three versions have measured precision and speed; none has measured the
thesis. RA4's hypothesis is that on regular tensor kernels a static cost
model orders more than 90 % of the pairs of variants autotuning
distinguishes, at a decision cost lower by orders of magnitude. 0.7.0 tests
it on the CPU backend that exists, with attention as the case study: a
layout space over each nest whose every point is exact by construction and
asserted bit-identical to the scalar sequential kernel, a plan object that
carries the layout beside the realization, a cost model that reads the
layout (reuse-aware traffic, threads, register pressure, an uncertainty),
calibrated on a GEMM the attention nests never enter, and an exhaustive
sweep pinned to the ten X925 cores and repeated (four hours later, where
the plan asked for a second day). Statements L to P. The version does not
promise to close the 22× against PyTorch.

**What it shows.** Exactness survives a space of variants: 17 879 points of
the course GPT's attention, a GEMM and a BERT-shaped sample — register
tiles, k-blocks, loop orders, lanes, packs, thread splits at any depth — are
every one bit-identical to the scalar sequential kernel, natively and on the
kernel interpreter (L). The space pays: its best points run the attention's
`ein`s 9.2× and 16.5× faster than 0.6.0 at batch 16 and 47× and 77× at batch
1, and a 1024³ `f32` product at 460 GFLOP/s, 5.49× 0.6.0's point (P). **What
it does not show** is RA4. The static model orders 47.9 % and 43.9 % of the
attention `ein`s' distinguished close pairs right where the hypothesis asks
more than 90 % (M), chooses 582–717 times faster than the sweep at batch 16
where it asks 10³ (N), and picks points 2.18–3.01 times slower than the best
(O). The cause is named: the model gives a load no weight and counts traffic
from L2, so it ranks lanes and thread splits (93–97 %) and cannot see a
register tile. Its candidates within `1 + u` still hold the best point, or
one within 1.08× of it, at 2.4–15 % of the space: a pruner, not a chooser.
The version closes on these results; the load term is a design for a later
one, to be tested on a fresh sweep.

- Step 7: documentation, tutorial step 17, version 0.7.0. The manual brought
  to 0.7.0 as implemented: `realize` gains a table of the layout decisions
  (domains, defaults, what each changes), their well-formedness and how a
  plan is chosen, and loses a paragraph that said this version lowers few
  of them; `native` a packed, tiled and split kernel as `dump --stage kernel`
  prints it, the GEMM's best point beside 0.6.0's and the model's pick, and
  what the split at a depth buys at batch one; `optimization` what `u` is for
  and what the layout model cannot see (D62); `cli` checked against `presk
  --help` and each command's (`--seed` added to the synopses of `plan` and
  `autotune`, a section name corrected); `experiments` §7, the statements of
  0.7.0, and a closing section that no longer says RA4 is untested or quotes
  the 94×; `status` closed with L to P, the verification and what the version
  has and has not; the index, `install` and the chapter summaries. Tutorial
  step 17, "A plan for attention": the layout model calibrated on a GEMM,
  then `presk plan`, `presk run --plan` (the same output), `presk autotune
  --space small` twice and `presk rank` on the tutorial's attention, in
  `make tutorial`, which now runs steps 0 to 17. Its quoted outputs were
  compared with a run: they agree but for timings, `presk --help` (the three
  commands of 0.7.0), and step 16's `bf16` and recipe rows, which moved with
  D58's `bf16` `dot` (the recipe 8.49·10⁻³ → 8.66·10⁻³ from the exact step;
  every choice of the solver unchanged). The register-load term is recorded
  as a design in `docs/kernels-roadmap.md`, with the evidence for it (thirty
  points of the values `ein` that the model predicts equal, ordered by their
  loads per vector multiply-add from 6.9–7.3 ms down to 2.5–2.9 ms), what it
  will not fix alone, and how to test it on a sweep it was not designed on.
  Spec questions 63–70 copied to `presk-specs`
  (`presk-spec-questions-0.7.0.md`), 63, 64, 66, 67 and 68 marked as in the
  reference, 65 and 69 given the measurement's numbers. Version 0.7.0; `presk
  --help` lists `plan`, `autotune` and `rank` without their step numbers and
  names layout plans among the compiler's parts. Found and fixed: `make
  ra4-small` ranked its sweep with `presk rank`'s default entry,
  `SelfAttention.apply`, so its report left M, N and O "not measured";
  `experiments/ra4.sh` now passes its entry. Verification on falicon: `make
  test` (140 tests), `check-examples`, `plan-check`, `ra4-small` (every point
  exact on both executors), `tutorial` (twice), `conformance` and `cargo doc
  -p presk-manual` pass; §5's `presk plan` on the course GPT's attention and
  `presk run --plan` give the default's output byte for byte. The measurement
  itself is step 6's and was not run again.
- Step 6: the measurement (D62). One night on falicon ran four things:
  - two pinned sweeps of the course GPT's attention (B = 1 and 16, `f32`
    and `f64`, 16 324 points each), the GEMM at 1024³ and a BERT-shaped
    sample, four hours apart;
  - a 20-core sweep with its own calibration;
  - statement L's check on the kernel interpreter, in 130 shards on every
    core.

  **L holds**: 17 879 points, none differing. **M fails**: the attention
  `ein`s order 47.9 % and 43.9 % of their distinguished close pairs right,
  37–46 % of them tied. **N fails**: 582–717× at B = 16 and 75–89× at
  B = 1. **O fails**: top-1 regret 2.18–3.01. **P holds**: the best GEMM
  point is 5.49× 0.6.0's, 460 GFLOP/s on ten cores.

  The model is right on lanes and threads, and blind to register tiles,
  loop orders and k-blocks: a load weighs nothing and its traffic starts
  at L2, so every point is predicted compute-bound. Its candidates within
  `1 + u` always hold the best point, at 2–15 % of the space.

  The tools gained what the measurement needed:
  - `autotune --top/--random` (a sample), `--repeat 0 --shard K/N` (the
    check split over processes) and 0.6.0's point beside every space;
  - `rank --check` and the statements decided in the report;
  - `experiments/ra4_check.sh`, `make ra4-check` and `make ra4-20`.

  `rank` counts a tie as not right and takes `presk plan`'s pick (question
  70).
- Step 5: `presk plan`, `presk autotune`, `presk rank` (D61). The entry runs
  once natively, and the first launch of every nest at every shape is
  captured with its input buffers. Each nest is lowered again under every
  point of its space, the thread splits included and checked by condition 8.
  `presk plan` costs the points with the profile's layout model and writes a
  plan. `presk autotune` times every point on the captured buffers and
  checks its output bit-identical to point 0's (the default layout, scalar,
  one worker), natively and with `--interp` on the kernel interpreter. It
  writes a JSON line per point with the prediction's terms. `presk rank`
  applies plan §3.6's rule to two days: close pairs, distinguished pairs,
  rank accuracy overall and on decided pairs, Spearman, top-1 and top-k
  regret, per class, and the decision cost. `experiments/ra4.sh` runs them
  pinned. `make ra4-small` takes 55 s on the tutorial's attention, every
  point exact. At plan §3.4's shapes the space is 3 305 points per type at
  B = 1 and 4 857 at B = 16. The scores nest's full sweep takes 18 s and
  348 s, 57 and 360 times its static choice, where statement N asks 10³.
  Checking every B = 16 point on the interpreter would take about 30 hours,
  so `make ra4` checks B = 1 there by default (`RA4_INTERP`).
- Step 4: the layout cost model and its calibration (D60). `planned_cost`
  reads one worker's part of a kernel: its work, its traffic from L2, L3 and
  memory (footprints from the affine indices against each cache), and its
  register pressure. The machine is read from the cores the process runs on
  (pinned to an X925: 2 MiB of L2, where `cpu0`, an A725, has 512 KiB). The
  model `λ + spawn·[p>1] + max(W/π_g, Q_L2/β_L2, p·Q_L3/β_L3, p·Q_mem/β_mem)`
  is fitted by coordinate descent into the profile's `layout`, with `π` per
  (type, lanes, workers). `presk bench --calibrate FILE --space calib` times
  a GEMM under 96 layouts; `make calibrate-ra4` runs it at three sizes in
  `f32` and `f64`, pinned. On falicon, u = 0.185 over 576 samples, Spearman
  0.980. In those samples the best 1024³ `f32` layout runs at 405 GFLOP/s
  against 77 for 0.6.0's default (a preview; step 6 decides P).
- Step 3: the pack, two softmax passes and a split at a depth (D59). Every
  decision of the reference's table is now lowered, and `E7002` means an
  ill-formed layout. A pack is two launches, a copy of operand 1 to an NN
  panel, then the contraction on it; the softmax may take two passes, each
  exponential computed once. A plan's split divides the loop at a depth over
  its workers, whatever the work: `Kernel::parallel_at` follows the
  reference's definition, the Cranelift kernel is compiled with that loop
  ranged and the enclosing ones in full, and conditions 5 and 8 are checked
  against the machine and the kernel. The default split is 0.6.0's.
  Every point of four nests (1 491, the pack included) is bit-identical to the
  default on both executors, and 1 618 split points natively; one worker and
  many agree at depths 0–2 and at batch 1. The reference's example plan runs
  the course attention with the default's output.
- Step 2: one contraction builder for `dot` and `ein` (D58). The contraction
  nest is recognized once for both, and lowered in the form (row-wise or
  tiled), register tile, k-block, loop order and lanes its plan states; only
  the pack, two softmax passes and thread splits are still refused. The
  default is 0.6.0's kernel: every `f64` and `f32` `dot` text-identical, the
  attention `ein`s less an exact multiplication by one. A `bf16` `dot` whose
  width is not a multiple of eight is tiled `[1, 1]` and accumulates in
  `f32`, 2.4·10⁻² → 2.7·10⁻³ from the exact product at 1024 terms (D56).
  Every point of three nests, 1 142 in all, each its own kernel, is
  bit-identical to the default in `f64`, `f32` and `bf16` on both executors
  (`tests/layout.rs`). Lanes on an NT contraction need the pack, a condition
  added to the reference. The conformance suite asserts its bands on the
  examples in every realization and on the test programs in `f64` and `f32`,
  reporting their `bf16` gaps. `stack_prefix_grad` had crossed the `bf16`
  band on 6 input seeds of 20 with 0.6.0's kernels and passed at seed 0 by
  luck; the user decided.
- Step 1: `Layout`, `Plan` and `--plan` (D57). A realization plan
  (`{ "realization", "profile", "layout" }`, language reference §6.6) is
  read by `--plan @P.json` on `run`, `bench`, `train` and `dump`, exclusive
  with `--realize`, which is now the plan with the default layout. The kernel
  backend carries the plan and reads each nest's layout at its launch: the
  lanes are lowered as stated, every other decision is accepted at the value
  0.6.0's kernel has and refused otherwise, as is any thread split, and a
  layout ill-formed at the nest's shapes (the reference's conditions 3, 4, 7
  and 9) is refused too — all with the new `E7002`, never ignored. The layout
  is part of the kernel cache's key. `dump --stage realize --plan` lists the
  plan per nest; `bench --plan` adds a row for the plan's layout. `make
  plan-check`: a default plan gives byte-identical `dump --stage kernel` on
  ten entries, identical also to `v0.6.0`'s in `f64`, `f32` and `bf16`. The
  backend's `lanes` field became its default layout's. Found on the way: the
  `realize` chapter still said "named PSK-Realize until 0.6.0" and "`--stage
  realize` is an alias of `--stage realize`", both from the rename; fixed.
- Step 0: definitions and bookkeeping (D56). The language reference (draft
  v0.2) gives the layout decisions their domains — contraction form,
  register tile, k-block, loop order, lanes, thread split at a depth, pack,
  softmax passes — with the exactness rule, the plan object, well-formedness
  conditions 7 to 9, the parallel loop at any depth and the tile loops of
  the kernel grammar. Checking that every default reproduces 0.6.0's kernel
  found that it did not as the plan had it: an `ein` is the tile `(1, 1)`
  with one lane, not `(1, 8)`; the row-wise `dot` is taken whenever J is
  not a multiple of eight, and in `bf16` it keeps its partial sums in `bf16`
  (seven times further from the exact product than the tiled form, at 1024
  terms), which the realization does not declare; `embed`'s derivative and
  `sum_to` do the same; and no kernel reads the realization's `accum`. The
  reference, the plan's §3 and the manual's `native` chapter say so. Version
  `0.7.0-dev`; spec questions 63–69; the X925 cores recorded as
  `RA4_CORES`.

Before the plan, on 2026-09-23:

- **PSK-Sched is renamed PSK-Realize**, as the language reference renamed it
  on 2026-09-23: crate `presk-realize`, `presk dump --stage realize`, `make
  realize`, manual chapter `realize`, with `sched` kept as an alias of the
  stage and of the target. The IR name and the CLI only: the realization
  files keep "site" for the precision vocabulary, and the dated records
  (plans, decisions, spec questions, past entries here) keep the old name.
- **PyTorch re-measured on falicon** (D55): one training step of the course
  GPT is 29.96 s in presk `f32` against 1.34 s in PyTorch 2.14 `float32`,
  twenty threads and dropout on both sides, three repetitions — **22×**,
  where the `experiments` chapter quoted 0.4.0's 94× from another machine.
  Fourteen of the twenty-two is the contraction: a 1024³ `f32` product runs
  at 710 GFLOP/s under PyTorch (its own OpenBLAS 0.3.34, oneDNN 3.12 over
  Arm Compute Library 53.2) and at 49.6 under presk. Neither side moved for
  a reason inside the compiler.
- **The PyTorch export's `training` switch** (D54): every `uses Rand`
  function takes `training=False` beside `generator=None`, both passed on
  through every call to a `uses Rand` function; `dropout` drops under the
  switch and is the expectation without it. `train.py` calls a keyed
  objective with `None` for the key and `training=True`, so it drops as
  `presk train` does, with masks PyTorch draws. `make experiment-gpt` runs
  again (it had failed since 0.5.0 step 1), with the same steps, batches and
  threads on both sides.
- The stacked loop over a tuple accumulator is exported as `block((h, k),
  mask)`; it splatted the pattern into two arguments.
- `presk train` prints the median step time, the first step excluded, on
  standard error, and `--json` carries `ms` per step and `step_ms`;
  `train.py` prints the same, takes `--threads` (default: the machine's
  cores) and prints its thread count.

## 0.6.0 — 2026-09-22

Make it pay (`presk-specs/implementation/presk-plan-0.6.0.md`). Two versions
have ended with the same sentence: 0.4.0 showed the exactness discipline
affordable and the speed not there, 0.5.0 showed precision declarable, solved
and verified and the speed still not there — statement D holds by
measurement, statement E fails at 0.96×. A budget that buys nothing is a
guarantee nobody will pay for, and the MLPerf comparison needs one
realization that is both inside a budget *and* faster.

0.5.0 established which three things must become true: arithmetic width must
buy throughput (vector lanes, which D42 showed are **bit-exact** in the shape
the blocked contraction already has), the objective must be time and not
traffic, and the menu must be wide enough for the choice to matter (int8/int4
with per-channel scales). The version also settles whether the condition
number rescues the deterministic bound, and gives the ε curve its quality
axis.

**What it shows.** A narrower type buys time, twice, each time inside a
declared budget measured against the exact denotation. A training step
solved under a 10⁻³ budget on its update is **1.99×** faster than all-`f64`
at course scale (statement J′), once the budget was given the whole step as
its scope — a budget on the loss alone had given 1.13× (J). And `int8`
weights run BERT-large inference at MLPerf's own constants **2.2×** faster
than `f32`, 5.7·10⁻² from the exact logits: inside a budget of 10⁻¹, not
10⁻². Vector lanes are bit-identical to the scalar kernels (H); `int8`
weights stay inside a 10⁻² budget on the corpus GPT's loss (K); a quantised
gap splits into the weights' rounding and the arithmetic, which only an
exact reference can separate; and MLPerf's accuracy rule admits all eight
realizations of the corpus GPT that a budget separates over twelve orders of
magnitude. **What it does not show** is a realization inside a budget and
faster than FP32, MLPerf's reference: the solved step is `f32` in every
function and level with all-`f32` (30.0 against 31.0 s). Lanes buy little on
a matmul where 3× was asked (I, retired: the factor is an instruction set's).
The condition number does not rescue the deterministic bound, so a budget is
discharged by measurement (question 56, option (c)); the quality axis is
flat at this scale; and the 94× against PyTorch is still 0.4.0's, measured
on another machine.

- Step 9: documentation, tutorial step 16, version 0.6.0. The manual brought
  to 0.6.0 as implemented: `status` closed with H, I, J, J′ and K and with
  what the version has and has not; `experiments` §6, the statements of
  0.6.0, and a rewritten "What these experiments do not show"; `cli`,
  `native`, `training`, `serve`, `optimization`, `precision`, `language`,
  `install`, `data` and the index corrected where they still described
  0.3.0 to 0.5.0 (vectorization "not in this version", `solve` minimising
  traffic, `train` without its native flags, per-layer keys before D37, lanes
  said to make every run faster). Tutorial step 16, "A budget on the training
  step": `presk calibrate --step`, `presk solve --step` and `presk train
  --realize` on the tutorial's tower, where `bf16` loses the step to its
  storage rounding, the recipe is 8.5·10⁻³ from the exact step, and the
  solved step's run ends four thousand times closer to the `f64` run than a
  change of seed. `make tutorial` now runs every command of the chapter,
  steps 0 to 16 — step 0's, step 9's and step 15's were outside it — and
  every quoted output was regenerated: step 0 still quoted the `presk --help`
  of 0.2.0, step 12 a calibration from before fusion, step 15 a `traffic`
  line `solve` no longer prints. `presk --help` takes its version from the
  workspace (it said 0.4.0), and two stale help lines are fixed. Spec
  questions 57–62 copied to `presk-specs` (`presk-spec-questions-0.6.0.md`).
  The plan's §5 verification, on falicon: `make test` (120 tests),
  `check-examples`, `conformance`, `precision`, `curve` (its eight training
  runs redone from scratch, byte-identical), `coverage`, `experiment` and
  `tutorial` pass; `quant-table`, `step-table` and `foil` give their tables
  again; J′ re-measured at 2.02×; `--realize f64` byte-identical to 0.4.0's
  on thirty outputs; every golden of 0.1.0–0.5.0 unchanged. Too long to run: one
  course-scale epoch (two steps ran, and `make j-step`) and `make bert` at 24
  blocks (run at two). `make experiment-gpt` fails, and has since 0.5.0 step
  1: the exported `train.py` omits the dropout key the corpus GPT's objective
  takes. Also found, and fixed: the day an envelope records was a constant
  in the source (it is now the day of measurement, in UTC, with a test), and
  `make curve`'s bound column printed the rule's name instead of the bound.
  The 0.5.0 heading of this file said "Unreleased" although `v0.5.0` is
  tagged.
- Step 8: the MLPerf comparison, with numbers (D53). The foil rests on what
  holds, each result stated as what it is. **Both rules on one model**
  (`make foil`, `experiments/foil.sh`): MLPerf's rule — task accuracy within
  99 % or 99.9 % of FP32 over the whole validation set — and a budget, applied
  to the same eight realizations of the corpus GPT, on D50's all-`f64`
  checkpoint and its held-out tenth (461 windows, 118 016 predictions; a new
  entry of `05_gpt.psk`, `accuracy`, scored in `f64` as MLPerf's harness
  scores from outside). The accuracy rule admits all eight, at 99.9 % as at
  99 %, seven predictions being the largest change; the budget separates them
  over twelve orders of magnitude from two windows and attributes every
  quantised gap to the weights. A check takes 0.1 s, a pass 4 s, the exact
  reference 7.5 s once. K restated: on held-out windows `int8` weights are
  1.46·10⁻⁴ from the exact loss, not 3.23·10⁻⁶ — still inside 10⁻² by
  seventy. **BERT-large runs** (`make bert`, `experiments/bert.sh`): inference
  at MLPerf's constants on its initialization and SQuAD-shaped features
  (`experiments/bert_features.py`, a new entry `squad`); the exact reference
  of one feature in 1 h 57 min and 44 GB; on the logits, `int8` weights 5.7·10⁻²
  from it (outside 10⁻²) and 2.2× faster than `f32` at batch 1 and 20, the first type
  narrower than `f32` to buy time on a model; at batch 1 one core does all the
  work. The exact interpreter **shares tuples and records** instead of
  cloning them at every use, and its backend consumes its arguments: four
  full-width blocks 40 → 13.5 GB, byte-identical, every golden unchanged. The
  tiny BERT joins the native conformance suite; `make coverage` runs it. The
  foil's 0.5.0 sentences that overclaimed are withdrawn; question 62, new.
- Step 7b: a budget on the training step, and J again (D52). A budget in a
  `training run` binds **one optimizer step's update**,
  `‖θ̂′ − θ′‖₂ / ‖θ′ − θ‖₂`, against the exact step from the same state.
  `presk calibrate --step` builds the state (a checkpoint plus `--warmup`
  steps in `f64`), records the envelope of the exact step (the `vjp`'s and
  `adamw.step`'s sites), and prints per `--realize` the update gap, the
  **storage floor** (the exact `θ′` merely rounded to its storage type), the
  gradient gap and the sampling noise between two batches; `recipe` is mixed
  precision as practitioners use it, and `--reference f64` measures where the
  exact step is not affordable. `presk solve --step` measures every function
  of the step alone in `f32` and `bf16`, narrows greedily by predicted time
  per unit of gap under a root-sum-square, verifies and widens; a
  realization gains a `functions` map (a type per function). At `D = 64`: the
  `f32` update is 2.2·10⁻⁴ from the exact one while its gradient is 6.4·10⁻⁷
  — AdamW multiplies the error by 340, so **question 60 is answered with the
  update, not the gradient**; the recipe is 2.9·10⁻², outside 10⁻²; the
  solver takes `f32` for all sixteen functions and `bf16` nowhere, the profile
  predicting it slower. **J′ holds at course scale: 1.99×** per step (`f64`
  59.7 s, solved 30.0 s, all-`f32` 31.0 s, recipe 34.6 s), the solved step
  6.7·10⁻⁵ from `f64` there and 2.2·10⁻⁴ from the exact step at `D = 64`.
  Trained from scratch for 600 steps under the solved step, the GPT ends
  2.9·10⁻⁴ from all-`f64`'s validation loss and 5.0·10⁻³ from its weights,
  where two seeds end 1.22 apart. The storage floor grows as the learning rate falls,
  so a relative update budget is stated at a state (question 61, new). New
  `presk data diff`; `make step-table`, `make j-step`, `make step-train`.
- Step 6: the condition number, and the answer about the bound (D51).
  `presk calibrate` records at every sum and contraction `Σ|xᵢwᵢ|`,
  `√Σ(xᵢwᵢ)²` and each operand's normwise factor (by re-evaluating it on
  `|operands|`, `operands²` and ones), prints the worst condition numbers, and
  takes `--weights`; the envelope records its weights and data slice. The
  propagation has four rules for sums — `k` (0.5.0's), `normwise` (sound),
  `cond` (the condition number, now the default, sound in every case tested)
  and `rss` (an estimate) — chosen by `presk check --precision --model`, and
  `--measure` prints the gap measured natively on the envelope's inputs
  beside each. Median bound/gap on the conformance entries 1.8·10³ → 18; on
  the trained corpus GPT 10³³ → 10¹⁰ in `f32`: **the condition number does
  not rescue the bound**. The root-sum-square estimate is 3.7–10× the
  measured gap in `f64`/`f32`/`bf16`, 2.4·10³× on a mixed realization.
  Recommended to the specification: question 56 (c), a budget is discharged
  by measurement; question 58 yes, a statistical model to propose and not to
  discharge. `presk solve` runs on the envelope's data slice; `make
  precision` measures the GPT on trained weights and text.
- Step 7: the quality axis, and J at course scale (D50). `experiments/
  curve.sh` gains a validation-loss column: each of the seven solved
  realizations trains the corpus GPT (`D = 64`) from scratch, 600 steps on
  90 % of the Code civil, evaluated in `f64` on the other 10 %. **The axis is
  flat**: every run within 1.2·10⁻³ of all-`f64` (3.0263), no trend, while
  changing the seed moves it by 0.13 — no knee in quality at this scale. The
  solver columns are re-measured on trained weights at `T = 256` (D48): all
  inside, 3.22× to 1.20× less traffic. **J fails at course scale: 1.13×** per
  training step (`f64` 56.9 s, solved 50.2 s), while all-`f32` is 1.87×; the
  solved realization covers the loss's sites, and the reverse sweep and the
  optimizer, most of a step, stay at its `f64` default — question 60. The
  sensitivity ranking re-measured on trained weights **reverses 0.5.0's
  finding**: per element the token embedding is the most sensitive
  parameter and the head the second, twenty times the down-projection, as
  the folklore says. Statement D re-measured: 4.48·10⁻⁶ against a 10⁻³
  budget. New `presk run --weights` and `--from`; the curve script reads
  `solve`'s `cost` line again.
- Step 5d: which weights to quantise, and statement K (D49). `presk solve
  --quant [--scheme pow2]` offers every matrix parameter — by *declared*
  rank, so a stacked bias is not one — as `int8` or `int4` on top of the
  float assignment, priced by the gradient the reverse sweep already
  computes and accepted by measurement. The signed first-order prediction
  `Σ ∂L/∂w·(ŵ − w)` came within ten per cent of the measured gap where the
  bound `Σ|g|·|δ|` was 230× too large and quantised nothing; the proposal
  uses the root-sum-square. At 10⁻³ it quantises all nine matrices (five
  `int8`, four `int4`, 11× fewer weight bytes), measured 3.11·10⁻³. `presk
  quantize` gained `--data`, `--list`, and float realizations;
  `make quant-table` and `make gpt-trained`. **Statement K holds**: `int8`
  weights 3.23·10⁻⁶ from the exact loss on text, against a 10⁻² budget;
  `int4` 8.32·10⁻⁵, `bf16` everywhere 4.27·10⁻⁴.
- **A checkpoint of other dimensions was read from shifted offsets (D48).**
  A `params` file records no shapes, and the reader only checked it was not
  too short: the corpus GPT trained at `T = 256` and opened at `T = 32` —
  which every "trained weights" measurement since 0.5.0 step 5 did — read
  every leaf after `pos` from the wrong place, and `solve` ignored load
  errors. Now refused (`E9005`). 0.5.0's statement D number, its sensitivity
  ranking (and the finding against the folklore), the ε curve and 0.6.0 step
  4's assignments were measured on those weights; they are to be redone on
  properly trained ones in step 7. `make precision` and the curve now run at
  `T = 256`.
- Step 5c: `int4` kernels (D47). Both executors load a nibble as the
  packing stores it; the JIT reads four lanes as one 32-bit word, shifted by
  the first element's parity, each lane's nibble moved to the top of its
  `i32` by a per-lane multiplication and shifted back arithmetically. `int4`
  buffers carry four bytes of padding for that word, and a layer's slice
  stays quantised when it starts on a byte. The bit-identity test covers
  `int4` in both schemes, and a program of odd widths (`tests/quant/odd.psk`)
  makes half its rows start mid-byte. Measured: the 16 MB contraction 52.9 →
  10.1 ms (5.2×, level with `int8`); the GPT forward level at `D = 256`,
  0.82× at `D = 64`.
- Step 5b: integer kernels (D45). Integer weights cross the runtime seam as
  `Buf::Quant` — integers and scales, shared rather than copied — and stay
  integers through a `stack{L}` fold's slicing. The lowering is unchanged:
  the host marks a kernel input that receives them (`BufKind::Quant`, part of
  the cache key), and both executors load a marked input as `f32(q)·s`, the
  JIT in four lanes at once (four bytes, sign-extended twice, converted,
  times four scales). Running on the integers is **bit-identical** to running
  on their dequantised values, on both executors, scalar and in lanes, in
  `f32` and `f64`, which a test asserts; and with `pow2` scales, applying the
  scale after the sum is bit-identical to applying it on load, which another
  asserts. Measured (lanes on): a 64 × 1024 × 4096 contraction 47.6 → 10.7 ms
  (4.4×), the GPT forward at `D = 256` 28.6 → 26.0 ms (1.10×), at `D = 64`
  unchanged. `int4` is still read as values (5c).
- **`presk bench` timed the loading of its inputs (D46).** Its inputs come
  from the reference interpreter as exact reals, and every native row rounded
  them to its machine type on every call, inside the timing; now each row
  rounds (and quantises) once, outside it. On this machine that conversion
  was most of a 1024³ matmul — `f64` 183 → 68 ms — so every `bench` figure
  before this entry carries a constant on both sides and its ratios are
  diluted, statements I and J included. Calibrated profiles are unaffected:
  they time kernels, and the conversion was outside every kernel.
- Step 5a: integer weights, the reference path (D44). A realization gains a
  `quant` map: parameters by path (`blocks.attn.wq.w`), each with `bits` (8
  or 4), the axis it is contracted over (`over`, a name or a position from
  the end, since module parameters are written without axis names), and a
  `scheme` (`absmax`, or `pow2` for exact dequantisation). Scales are
  computed from the weights at load and never written into the realization
  (question 57). `run`, `run --data` and `serve` run on the dequantised
  parameters, `train` refuses a quantised realization, and a misspelt path is
  an error rather than a no-op. New `presk quantize` splits the gap of the
  output into its **representation** part (exact on `ŵ` against exact on `w`)
  and its **arithmetic** part (the machine run on `ŵ` against exact on `ŵ`),
  which only an exact reference can separate. On the corpus GPT at `D = 64`
  after 12 steps, blocks and head quantised: `int8` 2.52·10⁻⁶ relative
  (arithmetic 1.81·10⁻⁷), `int4` 1.06·10⁻⁴ (arithmetic 4.4·10⁻⁸), `pow2`
  `int8` 4.40·10⁻⁶ — rounding the weights dominates by one to three orders.
  No integer kernels yet (5b): the numbers are right and the bytes are not
  saved at run time.
- Vector lanes are **on by default** on the native backend, because they are
  bit-exact: no accuracy to trade, no realization to declare, no gap to
  measure, and a transformation that is free and exact has no reason to be
  opt-in. It does not close statement J — it raises both sides, and the
  solved realization is still 1.06× — but on the development machine it
  made every native run faster for nothing (not everywhere: on falicon,
  BERT-large at batch 20 is 15 % slower in lanes in `f32`, step 8). Turning it on exposed a bug: the lane width was taken from the
  *realization's default*, so a mixed assignment sent a two-lane request to a
  `bf16` kernel that cannot take one. The width belongs to each kernel's own
  type.
- Step 4: time as the solver's objective. `presk solve --profile` prices each
  site with the roofline's own prediction, the dtype in `Q` and not in `W`,
  and `W` from the site's elements and the weight of its primitive, which the
  envelope now records. Under traffic the solver reported 2.67× less; under
  time it reports 1.19× faster and the **relaxation floor rises above the
  chosen cost** — narrowing everything would be slower, which the traffic
  objective could not express. **Statement J fails at 1.06×** against 1.3×,
  but the sign has turned: 0.5.0's traffic-solved assignment was 0.96×,
  *slower* than doing nothing. Conversion costs at kernel edges are **not**
  in the model, and the solved realization is not run with lanes; the uniform
  rows show both are worth more than the assignment is.
- Step 3: the machine profile fitted per realization. One profile now holds
  the constants of every machine type and lane width on that machine, because
  a narrow type's `π` is not a wide type's — which is the whole reason lanes
  exist — and a cost model meant to compare types cannot have one `π`. Files
  written before this step still load and still mean what they said.
  Measured on the corpus GPT: `f64` π 6.36e9, `f32` 8.03e9, **`f32` with four
  lanes 1.59e10**, two and a half times the `f64` figure. A bug of step 1 was
  caught by this: the cost model counted a vector statement as *one*
  element's work, which made the fitted π of a vector kernel come out below
  its scalar form's — the opposite of what the hardware does.
- Step 2: Cranelift vector emission. The lane count is the machine type's —
  four `f32` or two `f64`, since Cranelift's vector types are 128 bits — and
  anything else is refused rather than quietly rounded. **Statement H holds
  on both paths**, asserted as an equality: the emitted vector kernel is
  bit-identical to the emitted scalar one. **Statement I fails**: on a 1024³
  matmul the lanes give 1.25× in `f64` (104.1 → 83.3 ms) and 1.15× in `f32`
  (87.2 → 75.8), where it asked for 3×. The gain is real and free, the same
  bits faster, but the blocked form already ran eight independent accumulator
  chains so most of the parallelism was there. **Packed panels are not
  written** and are the first thing to try before concluding anything about
  the ceiling.
- Step 1: the vector form in PSK-Kernel. Vector variables of `lanes` reals,
  a contiguous load, a broadcast and lane-wise arithmetic, executed by the
  kernel interpreter; the blocked contraction emits them when asked.
  **Statement H holds on the interpreter**: a vector kernel is
  *bit-identical* to the scalar one at two and four lanes in `f64` and `f32`,
  which a test asserts as an equality rather than a tolerance. A lane is a
  different output column, never a partial sum of one, so every output still
  accumulates in increasing order — a layout change, not a reassociation, and
  it owes no declared order and no gap. The Cranelift backend refuses a
  vector kernel with a message; emitting them is step 2.
- Step 0: version `0.6.0-dev`; **statement G retired in favour of H** (D43),
  because G asked the wrong question — the kernel it describes is exact, so
  the claim to make is bit-identity, which is stronger than any tolerance;
  spec questions 57 to 59.

## 0.5.0 — 2026-09-20

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
