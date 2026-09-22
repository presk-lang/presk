#![doc = include_str!("../manual/index.md")]
#![allow(rustdoc::redundant_explicit_links)]

/// Building and running the tool.
#[doc = include_str!("../manual/install.md")]
pub mod install {}

/// A guided tour: write programs, run them, read the errors, export to
/// PyTorch, test everything (`make tutorial` runs every command).
#[doc = include_str!("../manual/tutorial.md")]
pub mod tutorial {}

/// The `presk` command line: subcommands, flags, exit codes.
#[doc = include_str!("../manual/cli.md")]
pub mod cli {}

/// The Presk language as accepted by this version: lexical structure,
/// declarations, types, expressions, configuration blocks.
#[doc = include_str!("../manual/language.md")]
pub mod language {}

/// Training: `presk train` on any backend, the loop, checkpoints, and what a
/// precision budget binds in a training run.
#[doc = include_str!("../manual/training.md")]
pub mod training {}

/// Native execution: the `Backend` seam, PSK-Kernel v0 and its vector form,
/// realizations, the Cranelift backend, threads, the conformance suite.
#[doc = include_str!("../manual/native.md")]
pub mod native {}

/// The corpus: models from an NLP course, used as a forcing function.
#[doc = include_str!("../manual/corpus.md")]
pub mod corpus {}

/// Coverage: what the shape fragment can say, measured on the MLPerf
/// architectures — which families are expressible and what the rest cost —
/// and BERT-large run at MLPerf's constants.
#[doc = include_str!("../manual/coverage.md")]
pub mod coverage {}

/// Precision as a resource: the magnitude envelope, the bound and the
/// condition number, sensitivities, the solver, integer weights, a budget on
/// the training step, and what a budget does and does not promise.
#[doc = include_str!("../manual/precision.md")]
pub mod precision {}

/// Experiments: shapes as types, the realization gap of training against the
/// exact reference, the statements of 0.4.0 to 0.6.0 answered by measurement,
/// and MLPerf's acceptance rule beside a budget.
#[doc = include_str!("../manual/experiments.md")]
pub mod experiments {}

/// Scheduling: fusion groups, what shares a loop nest, and what that buys.
#[doc = include_str!("../manual/scheduling.md")]
pub mod scheduling {}

/// Optimization: e-graph rewriting of PSK-Core (`--opt`, `presk dump --stage
/// core-opt`), the rules, the oracle, what changes.
#[doc = include_str!("../manual/optimization.md")]
pub mod optimization {}

/// Serving: the `inference serve` block, `presk serve` on any backend,
/// `serve.py` in the export.
#[doc = include_str!("../manual/serve.md")]
pub mod serve {}

/// Data files: the `.psk.d` container, `presk data`, the training pipeline.
#[doc = include_str!("../manual/data.md")]
pub mod data {}

/// Diagnostics: how errors are reported and the list of codes.
#[doc = include_str!("../manual/diagnostics.md")]
pub mod diagnostics {}

/// Every external tool and library, what it is and why it is used.
#[doc = include_str!("../manual/dependencies.md")]
pub mod dependencies {}

/// What each version and step implements; what is not there yet.
#[doc = include_str!("../manual/status.md")]
pub mod status {}
