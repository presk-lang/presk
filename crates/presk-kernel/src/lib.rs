//! PSK-Kernel v0 (plan 0.3.0, step 2): loop-level kernels for the tensor
//! primitives of PSK-Core at concrete shapes and one machine type, their
//! interpreter, and the host program that runs PSK-Core bodies over them
//! ([`KernelBackend`], the first native-arithmetic realization).

pub mod cost;
pub mod exec;
pub mod host;
pub mod ir;
pub mod lower;
pub mod memory;
pub mod plan;
pub mod profile;
pub mod record;

pub use host::{EpilogueAt, InterpExec, KernelBackend, KernelExec, Launch, LaunchOp, Launched};
pub use profile::{HostProfile, Overhead, Phase};
pub use ir::{BufKind, Dtype, Kernel, Threads};
