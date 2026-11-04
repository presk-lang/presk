//! Timing an entry (plan 0.11.0 step 5, moved from `presk-cli`): a call
//! timed on any backend, each kernel of a call timed natively, and the
//! host's time by phase. Each returns its times as a value, its error as the
//! message `presk bench: ` prefixes; `presk bench` prints them as before, and
//! `presk explain --measure` in its measured section.

use presk_core::FuncId;
use presk_interp::eval::Dims;
use presk_kernel::Kernel;
use presk_kernel::profile::HostProfile;
use presk_rt::{Backend, Plan, Realization, Value};
use presk_types::elab::ElabProgram;

/// Why a call was not timed: the backend could not prepare the entry
/// (`presk bench` leaves the row out without a word), or a call failed.
#[derive(Debug)]
pub enum TimeError {
    Prepare(String),
    Run(String),
}

/// `warmup` calls of `func` on `b`, then `repeat` timed: the median and the
/// best, in milliseconds. The inputs are built by the reference interpreter,
/// as exact reals; rounding them to the machine type, and quantising the
/// weights of a quantised realization, is loading a checkpoint, not a step,
/// so both happen once, outside the timing (0.6.0 step 5b).
#[allow(clippy::too_many_arguments)]
pub fn time_call(b: &mut dyn Backend, func: FuncId, fdims: &Dims, realize: &Realization, args: &[Value], warmup: usize, repeat: usize) -> Result<(f64, f64), TimeError> {
    let p = b.prepare(func, fdims, realize).map_err(|e| TimeError::Prepare(e.message))?;
    let args: Vec<Value> = if b.name() == "interp" {
        let mut a = args.to_vec();
        realize.store_params(&mut a).map_err(|e| TimeError::Run(e.message))?;
        a
    } else {
        realize.realize_args(args).map_err(|e| TimeError::Run(e.message))?
    };
    for _ in 0..warmup {
        b.call(&p, args.clone()).map_err(|e| TimeError::Run(e.message))?;
    }
    let mut samples = Vec::new();
    for _ in 0..repeat.max(1) {
        let t0 = std::time::Instant::now();
        b.call(&p, args.clone()).map_err(|e| TimeError::Run(e.message))?;
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((samples[samples.len() / 2], samples[0]))
}

/// Every kernel of a call, timed natively under a plan.
pub struct KernelTimes {
    /// The backend's kernels, in the order of their first launch.
    pub kernels: Vec<Kernel>,
    /// Each kernel's median launch, in seconds, `None` if it never ran.
    pub medians: Vec<Option<f64>>,
    /// Each kernel's launches over the `repeat` calls.
    pub launches: Vec<usize>,
    pub repeat: usize,
    /// The realization's label.
    pub label: String,
}

impl KernelTimes {
    /// The launches of one call.
    pub fn per_call_launches(&self) -> usize {
        self.launches.iter().sum::<usize>() / self.repeat.max(1)
    }

    /// The kernels' share of a call: each kernel's median launch times its
    /// launches a call, so that a first launch's compilation is not counted.
    pub fn per_call_seconds(&self) -> f64 {
        self.medians.iter().zip(&self.launches).map(|(m, c)| m.unwrap_or(0.0) * *c as f64).sum::<f64>() / self.repeat.max(1) as f64
    }
}

/// The kernels of `repeat` calls of `func`, each launch timed on the native
/// executor under `plan` (`presk bench --calibrate`'s measurement). `args`
/// are passed as they are.
#[allow(clippy::too_many_arguments)]
pub fn kernel_times(core: &presk_core::Program, prog: &ElabProgram, func: FuncId, fdims: &Dims, args: &[Value], plan: &Plan, repeat: usize) -> Result<KernelTimes, String> {
    let native = presk_cranelift::NativeExec::new().map_err(|e| format!("native backend unavailable: {}", e.message))?;
    kernel_times_with(Box::new(native), core, prog, func, fdims, args, plan, repeat)
}

/// [`kernel_times`] on another executor: the LLVM backend's
/// (`presk bench --llvm --calibrate`, plan 0.17.0 step 3).
#[allow(clippy::too_many_arguments)]
pub fn kernel_times_with(exec: Box<dyn presk_kernel::KernelExec>, core: &presk_core::Program, prog: &ElabProgram, func: FuncId, fdims: &Dims, args: &[Value], plan: &Plan, repeat: usize) -> Result<KernelTimes, String> {
    let realize = &plan.realization;
    let (timed, samples) = presk_kernel::cost::TimedExec::new(exec);
    let mut backend = presk_kernel::KernelBackend::with_exec(core, prog, realize, Box::new(timed));
    backend.set_plan(plan.clone());
    let p = Backend::prepare(&mut backend, func, fdims, realize).map_err(|e| e.message)?;
    for _ in 0..repeat {
        Backend::call(&mut backend, &p, args.to_vec()).map_err(|e| e.message)?;
    }
    let medians = presk_kernel::cost::median_per_kernel(&samples.borrow(), backend.kernels.len());
    let mut launches = vec![0usize; backend.kernels.len()];
    for (i, _) in samples.borrow().iter() {
        if let Some(c) = launches.get_mut(*i) {
            *c += 1;
        }
    }
    Ok(KernelTimes { label: backend.realize.label(), kernels: std::mem::take(&mut backend.kernels), medians, launches, repeat })
}

/// The host's time by phase (`presk bench --host-profile`, 0.10.0 step 5,
/// D95): calls timed by phase and calls not, alternated, after a warm-up.
pub struct HostProfiled {
    /// Each call without the timers, in seconds.
    pub plain: Vec<f64>,
    /// Each call with them: its phases and its time.
    pub timed: Vec<(HostProfile, f64)>,
    /// Minor page faults of each call without the timers.
    pub plain_faults: Vec<u64>,
    /// The outputs of a call with the timers and of one without are the
    /// same bits.
    pub same: bool,
    pub workers: usize,
    pub warmup: usize,
}

/// The host of `func` by phase, on the native backend under `plan` (its
/// layouts when `planned`), `repeat` calls of each kind.
#[allow(clippy::too_many_arguments)]
pub fn host_profile(core: &presk_core::Program, prog: &ElabProgram, func: FuncId, fdims: &Dims, args: &[Value], plan: &Plan, planned: bool, repeat: usize, warmup: usize) -> Result<HostProfiled, String> {
    use presk_kernel::profile::{Phase, minor_faults, same_bits, value_bytes};
    let realize = &plan.realization;
    let mut b = presk_cranelift::native_backend(core, prog, realize).map_err(|e| format!("native backend unavailable: {}", e.message))?;
    if planned {
        b.set_plan(plan.clone());
    }
    let p = Backend::prepare(&mut b, func, fdims, realize).map_err(|e| e.message)?;
    let args = realize.realize_args(args).map_err(|e| e.message)?;
    let profile = HostProfile::shared();
    // A call, timed; `keep`: its result is returned rather than freed (the
    // comparison, untimed).
    #[allow(clippy::type_complexity)]
    let call = |b: &mut presk_kernel::KernelBackend, timed: bool, keep: bool| -> Result<(f64, Option<HostProfile>, Option<Value>, Option<u64>), String> {
        b.set_host_profile(timed.then(|| profile.clone()));
        let f0 = minor_faults();
        let t0 = std::time::Instant::now();
        let a = args.clone();
        let copy = t0.elapsed().as_secs_f64();
        let r = Backend::call(b, &p, a).map_err(|e| e.message)?;
        if keep {
            b.set_host_profile(None);
            return Ok((0.0, None, Some(r), None));
        }
        let t1 = std::time::Instant::now();
        drop(r);
        let drop_s = t1.elapsed().as_secs_f64();
        let total = t0.elapsed().as_secs_f64();
        let faults = f0.zip(minor_faults()).map(|(a, b)| b - a);
        b.set_host_profile(None);
        let snap = timed.then(|| {
            let mut s = std::mem::take(&mut *profile.borrow_mut());
            // The caller's copy of the arguments, and the result freed.
            s.add(Phase::Entry, copy, args.iter().map(value_bytes).sum());
            s.add(Phase::Free, drop_s, 0);
            s.faults = faults;
            s
        });
        Ok((total, snap, None, faults))
    };
    for _ in 0..warmup.max(1) {
        call(&mut b, false, false)?;
    }
    let (mut plain, mut timed, mut plain_faults) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..repeat.max(1) {
        for on in [false, true] {
            match call(&mut b, on, false)? {
                (s, Some(snap), _, _) => timed.push((snap, s)),
                (s, None, _, f) => {
                    plain.push(s);
                    plain_faults.extend(f);
                }
            }
        }
    }
    // One call of each kind, outside the timing, their outputs compared.
    let same = match (call(&mut b, false, true), call(&mut b, true, true)) {
        (Ok((_, _, Some(x), _)), Ok((_, _, Some(y), _))) => same_bits(&x, &y),
        _ => false,
    };
    let workers = presk_kernel::KernelExec::workers(&*b.exec).unwrap_or(1);
    Ok(HostProfiled { plain, timed, plain_faults, same, workers, warmup: warmup.max(1) })
}
