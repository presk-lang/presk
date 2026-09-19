//! Source map, spans, diagnostics, lexer, AST and parser for Presk.
//!
//! Step 1 of the 0.1.0 plan (presk-specs/implementation/presk-plan-0.1.0.md).
//! The syntax accepted is the one of `presk-specs/examples/*.psk` and of the
//! specification chapter; where the two disagree or the grammar is only
//! sketched, the choice is recorded in `docs/spec-questions.md`.

pub mod ast;
pub mod diag;
pub mod lexer;
pub mod parser;
pub mod source;
pub mod token;

pub use diag::{Diagnostic, Severity};
pub use parser::parse_file;
pub use source::{FileId, SourceMap, Span};
