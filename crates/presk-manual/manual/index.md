# Presk compiler — user manual

**Version 0.2.0.** This manual describes the `presk` tool as it is
implemented, no more: every chapter was updated at each step of the 0.1.0
plan, and features that exist only in the language specification are marked
*not implemented*. The reference for the language itself is the
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
- [`training`] — `presk train`: the training loop on the interpreter, checkpoints.
- [`data`] — `.psk.d` data files, `presk data`, the training data pipeline.
- [`experiments`] — exact versus almost: what the tool can show today, with numbers, and what it cannot yet.
- [`diagnostics`] — error reports and codes.
- [`dependencies`] — every tool and library, what it is and why.
- [`status`] — what is implemented, step by step.

## Where things are

| Item | Location |
|---|---|
| Compiler sources | `crates/` of the `presk` repository (Rust workspace) |
| Example programs (acceptance tests) | `tests/examples/` (copied from `presk-specs/examples/`) |
| Tutorial programs | `tutorial/*.psk`, run by `make tutorial` |
| Implementation plans | `presk-specs/implementation/presk-plan-0.1.0.md`, `presk-plan-0.2.0.md` |
| Architecture decisions | `docs/decisions.md` |
| Open questions sent to the specification | `docs/spec-questions.md` |
| This manual | `crates/presk-manual/manual/*.md`, rendered by `cargo doc -p presk-manual --open` (or `make doc`) |
