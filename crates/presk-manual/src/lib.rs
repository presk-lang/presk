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

/// Diagnostics: how errors are reported and the list of codes.
#[doc = include_str!("../manual/diagnostics.md")]
pub mod diagnostics {}

/// Every external tool and library, what it is and why it is used.
#[doc = include_str!("../manual/dependencies.md")]
pub mod dependencies {}

/// What each version and step implements; what is not there yet.
#[doc = include_str!("../manual/status.md")]
pub mod status {}
