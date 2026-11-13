# Every Presk program of the repository

Every `.psk` file of the compiler's repository is under this directory,
with the files the programs read beside them (D198, plan 0.17.2 step 1).
Each directory keeps the place its programs' `import "…"` and
`stream "…"` lines rely on.

- `make check-examples` runs `presk check` on every program here but the
  diagnostics (`tests/diag/`, `tutorial/errors/`) and the parser's ASCII
  spellings (`tests/ascii/`). It takes them by wildcard, not from a list.
- The test `the_manual_quotes_the_examples`
  (`crates/presk-cli/tests/manual_quotes.rs`, D199) holds every listing of
  Presk in the manual equal to the program, part or declaration here that
  the line `<!-- psk: … -->` before it names, and the diagnostics it quotes
  (those of `tutorial/errors/` among them) equal to what `presk check`
  prints; `make manual-quotes` writes them again.
- The test `every_program_is_in_examples`
  (`crates/presk-print-math/tests/listings.rs`) fails on a `.psk` file
  outside this directory, and on a file here that this index does not name
  between backquotes, by its path from here or by a pattern with `*`.
- Paths in commands, scripts and the manual are written from the
  repository's root (`presk check examples/spec/mlp/mlp.psk`).

## `spec/`: the language's programs

The specification's example programs, the syntax's source of truth, and the
compiler's acceptance tests. They came from presk-specs' `examples/` at its
commit `68a9dc3` (2026-10-28), byte for byte; their history up to it is
there. From D198 on they change only by a merge request here paired with one
in presk-specs that states the language change. presk-specs' report prints
them from here, from the sibling checkout `../presk` or from `PRESK_DIR`
(plan 0.17.2 step 3).

- `spec/README.md`: the index of the specification's examples.
- `spec/mlp/mlp.psk`, `spec/mlp/README.md`: the multilayer perceptron.
- `spec/layernorm/layernorm.psk`, `spec/layernorm/README.md`: the layer
  norm and RMSNorm.
- `spec/minigpt/minigpt.psk`, `spec/minigpt/train_minigpt.psk`,
  `spec/minigpt/serve_minigpt.psk`, `spec/minigpt/README.md`: MiniGPT, its
  training run and its serving; `spec/minigpt/data/tokens.txt` and
  `spec/minigpt/data/tokens.psk.d` the training stream;
  `spec/minigpt/target/jetson-agx.pskt` a target description the compiler
  does not read.
- `spec/concat/concat.psk`, `spec/concat/README.md`: slices that keep their
  axis, `concat`, `min`, `max`, `clamp`.
- `spec/rope/rope.psk`, `spec/rope/README.md`: rotary position embeddings.

Read by most crates' tests (the parser's snapshots, typing, AD, lowering,
the interpreter's golden outputs in `tests/run/`, the backends, the exports,
the explain documents), by `make run-examples`, `export-examples`,
`conformance`, `train-examples`, `serve-examples`, `opt-examples`, `bench`
and `math-check`, and by `tests/plan/entries.txt`.

## `tutorial/`: the manual's tutorial chapter

`tutorial/01_linear.psk`, `tutorial/02_norm.psk`, `tutorial/03_attention.psk`,
`tutorial/04_grad.psk`, `tutorial/05_sample.psk`, `tutorial/06_tower.psk`,
`tutorial/07_train.psk`, and `tutorial/data/pairs.txt` with
`tutorial/data/pairs.psk.d`, the training pairs of its step 11. The chapter
(`crates/presk-manual/manual/tutorial.md`) quotes them and `make tutorial`
runs its commands. `tutorial/errors/bad_axis.psk`, `tutorial/errors/bad_ret.psk`,
`tutorial/errors/bad_unbound.psk`, `tutorial/errors/bad_syntax.psk` (step 2),
`tutorial/errors/bad_argmax.psk` (step 4) and `tutorial/errors/bad_rand.psk`
(step 5) are the programs whose diagnostics the chapter prints (D199). Also read by `make explain` and `explain-manual` (the
chapter `explain_loss`), `math-check`, RA4's small sweep, and the tests of
the CLI, the explain documents, the printer of the denotation and the
backends.

## `corpus/nlp/`: the corpus of an NLP course

The ladder of the course on the Code civil (D26: benchmark material, not the
specification's): `corpus/nlp/01_bigram_count.psk`,
`corpus/nlp/02_bigram_neural.psk`, `corpus/nlp/03_mlp.psk`,
`corpus/nlp/05_gpt.psk` (the course GPT), with `corpus/nlp/README.md`,
`corpus/nlp/EXPECTED.md` (what each rung must give) and
`corpus/nlp/data/prepare.py`, which fetches the text and writes the rest of
`corpus/nlp/data/` (`make corpus`); `corpus/nlp/data/civil_mots.txt` is
committed. Read by `make corpus`, `corpus-gpt`, `gpt-trained`, `precision`,
`realize`, `network-figure`, `math-check`, most scripts of `experiments/`,
and the tests of the backends, the exports, the memory count, the explain
documents and `tests/plan/entries.txt`.

Generated there and git-ignored, never committed: `corpus/nlp/data/*.md`
(the texts as fetched and mapped), `corpus/nlp/data/*.txt` (token streams,
vocabularies, pairs), `corpus/nlp/data/*.json` (the vocabulary) and
`corpus/nlp/data/*.psk.d` (the windows).

## `coverage/`: the MLPerf architectures

`coverage/bert.psk` (BERT-large), `coverage/llama.psk` (the Llama 2
decoder) and `coverage/llama3.psk` (Llama 3.2 1B). Read by `make coverage`,
`bert`, `llama`, `llama3`, `llama3-budget`, `pytorch-decoder`, the scripts
of `experiments/` and `experiments/llama/`, and the tests of the native
backend, the parameters, the memory count, the explain documents and the C
export's serving.

## `experiments/`: the experiments' programs

The scripts that run them stay in `../experiments/`.
- `experiments/gemm.psk`, `experiments/matmul.psk`: a square product and a
  product of any shape, the layout cost model's calibration (`make
  calibrate-ra4`) and its recost (`presk refit --recost`).
- `experiments/ra4_attention.psk`: the course GPT's attention as the RA4
  sweeps measured it, frozen at plan 0.10.0 step 7b (D99; `make ra4-f1`,
  `ra4-f2`).
- `experiments/gpt_grad.psk`: the gradient of the course GPT's training loss
  as an entry (D87, `make pytorch-baseline`; T3, J′, the host profile, the
  step's plans).
- `experiments/margins.psk`: the course GPT's logits as an entry (D118; `make
  margins` and the runs on its rows).
- `experiments/widening.psk`: three pairs of small programs (D128, `make
  widening`).
- `experiments/gappa_layernorm.psk`, `experiments/gappa_dot.psk`: the layer
  norm and the contraction Gappa proves bounds on (`make gappa`, D108; `make
  gappa-dot`, D131).
- `experiments/gemm_1024.psk`: the 1024³ product `make realize` times at one
  thread and at all cores (D199; the Makefile wrote it into `out/gemm.psk`
  before).
- `experiments/delta_p_entries.psk-part`: not a program, the two entries
  `experiments/delta_p.sh` appends to a copy of the course GPT (D98, D199;
  a heredoc of the script before).

## `tests/`: the test suites' programs

- `tests/diag/*.psk`, with `tests/diag/helpers/*.psk`: one erroneous program
  per diagnostic, each with an insta snapshot of its report (the parser's,
  the types', the analyses' tests); `tests/diag/run_*.psk` and
  `tests/diag/export_c_*.psk` name in their header the commands that
  `crates/presk-cli/tests/run_diag.rs` and `export_c.rs` run.
- `tests/data/`: the data tests' token stream `tests/data/tokens.txt` and
  windows `tests/data/windows.psk.d`, a valid training program
  `tests/data/train_ok.psk`, and small programs checked against finite
  differences or bit for bit: `tests/data/closure_grad.psk`,
  `tests/data/concat_grad.psk`, `tests/data/defined.psk`,
  `tests/data/dropout_grad.psk`, `tests/data/gemm_exact.psk`,
  `tests/data/logits.psk`, `tests/data/masked_softmax.psk`,
  `tests/data/masks.psk`, `tests/data/merge_grad.psk`,
  `tests/data/overflow.psk`, `tests/data/serve_generate.psk`,
  `tests/data/split_grad.psk`, `tests/data/stack_dropout_grad.psk`,
  `tests/data/stack_grad.psk`, `tests/data/stack_keys.psk`,
  `tests/data/stack_none.psk`, `tests/data/stack_prefix_grad.psk`,
  `tests/data/terms.psk`.
- `tests/plan/`: realization plans (`tests/plan/default.json`,
  `tests/plan/attention.json`, `tests/plan/tutorial-typed.json`,
  `tests/plan/two_dots.json`, and `tests/plan/mixed-realization.json`, under
  which the C export's tests run MiniGPT), the entries `make plan-check`
  and the C export's tests compare (`tests/plan/entries.txt`), and the
  programs they need: `tests/plan/minigpt_grad.psk`,
  `tests/plan/contractions.psk`, `tests/plan/epilogue.psk`,
  `tests/plan/two_dots.psk`.
- `tests/train/`: regressions for `presk train`,
  `tests/train/mlp_regression.psk`, `tests/train/dropout_reg.psk` and its
  gradient `tests/train/dropout_reg_grad.psk`, on `tests/train/reg.txt` and
  `tests/train/reg.psk.d`.
- `tests/quant/odd.psk`, `tests/quant/split.psk`: odd widths for the `int4`
  kernels, and a quantised parameter read through `split`.
- `tests/export_c/`: what the C export's tests need beyond the test set,
  `tests/export_c/fold_value.psk`, `tests/export_c/serve_untied.psk` and a
  plan of splits for MiniGPT, `tests/export_c/minigpt-splits.json`.
- `tests/ascii/mlp_ascii.psk`, `tests/ascii/symbols.psk`,
  `tests/ascii/symbols_unicode.psk`: ASCII spellings the parser must read as
  their Unicode forms (`crates/presk-syntax/tests/snapshots.rs`).
- `tests/analysis_ok_relu.psk`: a program the analyses accept
  (`crates/presk-core/tests/analysis.rs`).
- `tests/precision/heads.psk`: a parameter per way its readers contract it
  (`crates/presk-precision/tests/menu.rs`; a string in the test before
  D199).

One test builds its program instead of reading it: `chain_source(n)` in
`crates/presk-interp/tests/run.rs` writes a chain of `n` links for each `n`
it measures the peak of, a generator rather than a program.

## `snippets/`: the documents' listings

Programs whose parts a document prints, when no other program here holds
the listing. A part is the lines between a line `-- quote: NAME` and the
next `-- end quote`, or a declaration the document names; the lines outside
it declare what it presupposes, so that each file is a program `make
check-examples` checks.

- `snippets/spec/types.psk` (the parts `shapes`, `axes`, `idx`) and
  `snippets/spec/modules.psk` (`module`, `composition`, `transformer`): the
  listings of the specification's sections 3 and 4, which presk-specs'
  report prints from here (`report/chapters/02-spec.tex`, plan 0.17.2
  step 3). A part changed changes the report: like `spec/`, they change
  only by a merge request here paired with one in presk-specs.
- `snippets/tutorial/linear_ascii.psk`: the tutorial's `Linear` in the ASCII
  spelling (step 1, the declaration `Linear`).
- `snippets/language/split_key.psk` (the declaration `block`) and
  `snippets/language/stack_key.psk` (the part `acc`): the `language`
  chapter's two keys split from one, and the key carried through
  `stack{L}`.

## Where each file was before D198

The records (the decisions before D198, the reports of `docs/`, the
envelopes' JSON, the CHANGELOG's released sections, the manual's `status`
chapter up to 0.17.1) keep the paths of their version.

| Before (v0.17.1) | Since D198 |
|---|---|
| `tests/examples/` (a copy of presk-specs' `examples/`, `make sync-examples`) | `examples/spec/` |
| `tutorial/` | `examples/tutorial/` |
| `corpus/nlp/` | `examples/corpus/nlp/` |
| `coverage/` | `examples/coverage/` |
| `experiments/*.psk` | `examples/experiments/*.psk` |
| `tests/diag/`, `tests/data/`, `tests/plan/`, `tests/train/`, `tests/quant/`, `tests/export_c/` | `examples/tests/…` under the same names |
| `tests/analysis_ok_relu.psk` | `examples/tests/analysis_ok_relu.psk` |
| `crates/presk-syntax/tests/ascii/` | `examples/tests/ascii/` |
