//! Reference interpreter: arbitrary-precision evaluation of PSK-Core.
//!
//! Step 7 of the 0.1.0 plan (presk-specs/implementation/presk-plan-0.1.0.md).
//! Values are tensors with named axes over [`real::Real`] numbers (MPFR
//! through `rug`, precision in bits chosen per run); functions are called with
//! their dimensions instantiated; randomness comes from splittable ChaCha
//! keys so that every run is reproducible from `--seed`.

pub mod eval;
pub mod key;
pub mod prims;
pub mod real;
pub mod run;
pub mod value;

pub use eval::{Interp, RunError};
pub use real::Real;
pub use run::{RunOptions, RunOutput, run};
pub use value::{Tensor, Value};
