//! PSK-Core: typed functional IR in ANF over exact real tensors, with
//! lowering from the typed program, structural verification and a pretty
//! printer. Regularity/effect analyses (step 5) and automatic
//! differentiation (step 6) operate on this IR.
//!
//! Step 4 of the 0.1.0 plan (presk-specs/implementation/presk-plan-0.1.0.md).

pub mod ad;
pub mod analysis;
pub mod fusion;
pub mod ir;
pub mod lower;
pub mod pretty;
pub mod verify;

pub use analysis::{Analysis, analyze};
pub use ir::{Func, FuncId, PrimOp, Program, Rhs, VarId};
pub use lower::lower;
pub use pretty::pretty;
pub use verify::verify;
