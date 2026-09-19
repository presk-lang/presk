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

/// Training on the reference interpreter: `presk train`, the loop, checkpoints.
#[doc = include_str!("../manual/training.md")]
pub mod training {}

/// Native execution: the `Backend` seam, PSK-Kernel v0, realizations, the
/// conformance suite and the measured gaps.
#[doc = include_str!("../manual/native.md")]
pub mod native {}

/// Experiments: shapes as types against silent broadcasting, and the
/// realization gap of float32/float64 training against the exact reference.
#[doc = include_str!("../manual/experiments.md")]
pub mod experiments {}

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
