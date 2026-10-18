//! The pipeline as values (plan 0.11.0, D109): what every command of the
//! `presk` tool and `presk explain` share.
//!
//! Until 0.11.0 each command of `presk-cli` parsed, elaborated, lowered,
//! differentiated, bound its entry's dimensions and loaded its realization
//! itself, and printed as it went. This crate holds those steps once. A step
//! never prints: it returns its value, and reports its diagnostics and its
//! messages to a [`Report`] its caller gives, at the point where the command
//! printed them before, so that the command's output is unchanged. The
//! command-line tool reports to [`Stderr`]; `presk explain` collects the same
//! reports into its document.
//!
//! A step that cannot go on returns [`Stop`], the exit code the command
//! returns: 1 when diagnostics were reported, 2 for a usage error or an
//! internal one.

pub mod ad;
pub mod backend;
pub mod bench;
pub mod entry;
pub mod export_c;
pub mod front;
pub mod gap;
pub mod layout;
pub mod memory;
pub mod opt;
pub mod plan;
pub mod precision;
pub mod provenance;
pub mod report;
pub mod text;

pub use report::{Collect, Report, Stderr, Stop};

/// A step's value, or the exit code its command returns.
pub type Result<T> = std::result::Result<T, Stop>;
