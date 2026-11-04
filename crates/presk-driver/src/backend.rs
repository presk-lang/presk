//! The executor a command runs an entry on.

use presk_rt::{Backend, InterpBackend, Plan};
use presk_types::elab::ElabProgram;

use crate::Report;

/// The four executors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// The reference interpreter (MPFR at `bits`).
    Interp,
    /// PSK-Kernel v0 interpreted in the realization's machine type.
    Kernel,
    /// PSK-Kernel v0 compiled to machine code by Cranelift (in process).
    Native,
    /// PSK-Kernel v0 printed as C and compiled by LLVM's `clang -O3`, on the
    /// native backend's host (plan 0.17.0 step 3, D176).
    Llvm,
}

/// Runs `f` with the chosen backend over `core`, under `plan`. The
/// reference interpreter computes at `bits` whatever the plan says; the
/// kernel backends take its realization and its layout. Where the native
/// backend is unavailable, the kernel interpreter runs, and the report
/// says so.
pub fn with_backend<R>(kind: BackendKind, core: &presk_core::Program, prog: &ElabProgram, bits: u32, plan: &Plan, report: &mut dyn Report, f: impl FnOnce(&mut dyn Backend) -> R) -> R {
    let realize = &plan.realization;
    let kernel = || {
        let mut b = presk_kernel::KernelBackend::new(core, prog, realize);
        b.set_plan(plan.clone());
        b
    };
    match kind {
        BackendKind::Interp => f(&mut InterpBackend::new(core, prog, bits)),
        BackendKind::Kernel => f(&mut kernel()),
        BackendKind::Native => match presk_cranelift::native_backend(core, prog, realize) {
            Ok(mut b) => {
                b.set_plan(plan.clone());
                f(&mut b)
            }
            Err(e) => {
                report.line(&format!("presk: the native backend is unavailable: {}", e.message));
                f(&mut kernel())
            }
        },
        // The commands check `presk_llvm::available` before they get here
        // and refuse without the compiler; a library caller that did not is
        // told, and runs on the kernel interpreter as for `Native`.
        BackendKind::Llvm => match presk_llvm::llvm_backend(core, prog, realize) {
            Ok(mut b) => {
                b.set_plan(plan.clone());
                f(&mut b)
            }
            Err(e) => {
                report.line(&format!("presk: the LLVM backend is unavailable: {}", e.message));
                f(&mut kernel())
            }
        },
    }
}
