//! PSK-Kernel v0 (plan 0.3.0, step 2): loop-level kernels for the tensor
//! primitives of PSK-Core at concrete shapes and one machine type, their
//! interpreter, and the host program that runs PSK-Core bodies over them
//! ([`KernelBackend`], the first native-arithmetic realization).

pub mod cost;
pub mod exec;
pub mod host;
pub mod ir;
pub mod lower;
pub mod plan;
pub mod record;

pub use host::{InterpExec, KernelBackend, KernelExec, Launch, LaunchOp};
pub use ir::{BufKind, Dtype, Kernel, Threads};
