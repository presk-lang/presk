# Presk compiler — user manual

**Version 0.6.0** (plans 0.1.0 to 0.6.0 done; [`status`]). This
manual describes the `presk` tool as it is implemented, no more: every
chapter was updated at each step of each plan, and features that exist only
in the language specification are marked *not implemented*. The reference for the language itself is the
specification repository `presk-specs` (`report/chapters/02-spec.tex`); this
manual documents what the compiler accepts and does.

Presk is a domain-specific language for describing, training and running
neural networks. A Presk program denotes a mathematical function over tensors
of real numbers: the language has no machine number types (`float32`,
`bfloat16`…), and the precision of an execution is a compilation choice made
under a declared error budget. Shapes, with named and symbolic axes, are part
of the types; every function carries a regularity class that decides whether
it may be differentiated; randomness is an explicit effect.

## Chapters

- [`install`] — building the tool from source, requirements.
- [`tutorial`] — a guided tour, from a first module to a PyTorch export;
  start here. Its programs are in `tutorial/`, its commands in `make tutorial`.
- [`cli`] — the `presk` command line.
- [`language`] — the syntax accepted by this version.
- [`training`] — `presk train`: the training loop on any backend, checkpoints, a precision budget on the step.
- [`serve`] — `inference serve`, `presk serve`: the generation loop on any backend, `serve.py` in the export.
- [`data`] — `.psk.d` data files, `presk data`, the training data pipeline.
- [`native`] — the runtime seam, PSK-Kernel v0 (`--backend kernel`) and its vector form, the realization (`--realize f64|f32|bf16|@FILE`), the Cranelift backend, threads, the conformance suite.
- [`scheduling`] — fusion groups: what shares a loop nest, and the launches and traffic it removes.
- [`optimization`] — `--opt`: e-graph rewriting of PSK-Core, exact over ℝ, checked against the interpreter; the cost model calibrated per machine type.
- [`corpus`] — models from an NLP course, transcribed as a forcing function.
- [`coverage`] — what the shape fragment can say, measured on the MLPerf architectures; BERT-large run at MLPerf's constants.
- [`precision`] — precision as a resource: the envelope, the bound and the condition number, sensitivities, the solver, integer weights, a budget on a training step, and what a budget promises.
- [`experiments`] — exact versus almost, the statements of 0.4.0, 0.5.0 and 0.6.0 answered by measurement, MLPerf's rule beside a budget, and what the experiments do not show.
- [`diagnostics`] — error reports and codes.
- [`dependencies`] — every tool and library, what it is and why.
- [`status`] — what is implemented, step by step.

## Where things are

| Item | Location |
|---|---|
| Compiler sources | `crates/` of the `presk` repository (Rust workspace) |
| Example programs (acceptance tests) | `tests/examples/` (copied from `presk-specs/examples/`) |
| Tutorial programs | `tutorial/*.psk`, run by `make tutorial` |
| The NLP-course corpus | `corpus/nlp/`, run by `make corpus` and `make corpus-gpt` |
| MLPerf transcriptions | `coverage/`, checked by `make coverage` |
| Implementation plans | `presk-specs/implementation/presk-plan-0.1.0.md` … `presk-plan-0.6.0.md` |
| Experiment scripts | `experiments/*.sh`, run by `make curve`, `quant-table`, `step-table`, `j-step`, `step-train`, `foil` and `bert` |
| Architecture decisions | `docs/decisions.md` |
| Open questions sent to the specification | `docs/spec-questions.md`, copied at each version to `presk-specs/implementation/presk-spec-questions-0.X.0.md` |
| This manual | `crates/presk-manual/manual/*.md`, rendered by `cargo doc -p presk-manual --open` (or `make doc`) |
