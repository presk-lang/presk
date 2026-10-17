//! The printer of the denotation (plan 0.13.0 step 4): a program's
//! functions as the equations of the language reference's chapter 3, in
//! its notation and with its macros (`\code`, `\den`), or the same formulas
//! for a terminal.
//!
//! One renderer of expressions ([`m`]), two entries:
//! - [`typed`], the typed program: the author's names, a module's function
//!   once, `ein` as its sum, a stack as its recurrence, the operations by
//!   their names (`presk dump --stage elab --format latex|unicode`);
//! - [`core`], PSK-Core after differentiation: one equation per binding,
//!   named as `presk dump --stage core` names them, the generated
//!   functions' bindings with what the provenance table says they were made
//!   for (`presk dump --stage core-after-ad --format latex|unicode`).
//!
//! It reads no realization, no plan and no machine: what it prints is over
//! the reals. A form it does not write is printed as its source text and
//! reported (the listing's gaps).

pub mod core;
pub mod m;
pub mod render;
pub mod rules;
pub mod typed;

pub use m::{Format, M};
pub use render::{Select, aligned, listing};
pub use typed::{Display, Equation, Listing, typed_program};
