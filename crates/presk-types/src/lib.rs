//! Dimensions, the Presburger fragment, elaboration and shape typing.
//!
//! Steps 2–3 of the 0.1.0 plan (presk-specs/implementation/presk-plan-0.1.0.md).
//! Step 2 (this crate today): [`dim`] linear forms, [`presburger`] decision
//! procedure, [`elab`] resolution of imports, constants, `static assert`s,
//! dimensions, type aliases, module definitions and instances, function
//! signatures. Function bodies stay syntax trees until step 3.

pub mod check;
pub mod dim;
pub mod elab;
pub mod presburger;
pub mod ty;

pub use dim::{DimExpr, DimVar};
pub use check::{TypedProgram, check};
pub use elab::{ElabProgram, elaborate};
pub use ty::{ModRef, ParamsTree, Shape, Ty};
