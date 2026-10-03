//! The workers a divided launch runs on (plan 0.10.0 step 6, decision 3: on
//! `std`, not a crate).
//!
//! Until step 6 a divided launch started a scoped thread for each of its
//! pieces and joined them: about 210 µs a launch of ten threads on the
//! X925 cores, 43 ms of the course GPT's forward pass (D95); and the
//! scheduler placed each new thread afresh. A [`Pool`] keeps its workers for
//! the executor's life. A launch posts piece `t` to worker `t` (`t ≥ 1`),
//! runs piece 0 on the calling thread, as OpenMP's team does, and returns
//! when every piece has: the same pieces the scoped threads ran, each
//! computed by the same code, so the same bits.
//!
//! A worker waits for its next piece spinning a while, then parked; the
//! launch wakes it. The calling thread waits for the others the same way.
//! A parked worker takes 0.2 to 1.2 ms to wake on these cores, more than
//! most launches' pieces, and the host often runs undivided kernels for
//! longer than a tenth of a millisecond between two divided launches: so a
//! waiting thread spins about two milliseconds while the process's workers
//! have a core each, and a few microseconds when they do not (several pools
//! at once, the tests'), as libgomp throttles its spinning when its threads
//! outnumber the cores. Measured on the X925 cores (D97's addendum), a
//! launch after a gap of 2.0 ms finds the workers spinning, after 2.5 ms
//! parked. That is far longer than libgomp spins by default (300 000 spins,
//! about 0.1 ms there), which the exported C program's team does unless
//! told otherwise; `GOMP_SPINCOUNT=8M` makes it spin as long as this pool.

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle, Thread};

/// How many times a waiting thread checks before it parks, while the
/// process's workers have a core each: about two milliseconds on the X925
/// cores (`spin_loop` takes 7 ns there, a check about 8).
const SPIN_LONG: u32 = 1 << 18;
/// The same when they outnumber the cores: a few microseconds.
const SPIN_SHORT: u32 = 1 << 10;

/// The workers of every pool alive in the process.
static WORKERS: AtomicUsize = AtomicUsize::new(0);

/// How many times a waiting thread checks before it parks, now.
fn spin() -> u32 {
    static CORES: OnceLock<usize> = OnceLock::new();
    let cores = *CORES.get_or_init(|| thread::available_parallelism().map_or(1, |n| n.get()));
    if WORKERS.load(Ordering::Relaxed) < cores { SPIN_LONG } else { SPIN_SHORT }
}

/// A piece of work, its lifetime erased: [`Pool::run`] does not return
/// before every worker is done with it.
type Job = *const (dyn Fn(usize) + Sync);

struct Slot {
    /// The number of the last piece posted to this worker.
    posted: AtomicU64,
    /// The job and the piece, written before `posted`.
    job: UnsafeCell<(Job, usize)>,
}

struct Shared {
    slots: Vec<Slot>,
    /// Pieces of the current launch not yet done by the workers.
    left: AtomicUsize,
    /// The thread that waits for them, written before the pieces are posted.
    caller: UnsafeCell<Option<Thread>>,
    panicked: AtomicBool,
    stop: AtomicBool,
}

// SAFETY: `job` and `caller` are written by the launching thread before the
// `Release` store that posts a piece, and read by a worker after the
// `Acquire` load that sees it; the launching thread writes them again only
// after every worker has signalled, through `left`, that it is done.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// A pool of `n` workers beside the calling thread.
pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    posts: u64,
}

impl Pool {
    pub fn new(n: usize) -> Pool {
        let noop: &(dyn Fn(usize) + Sync) = &|_| {};
        let noop: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(noop) };
        let shared = Arc::new(Shared {
            slots: (0..n).map(|_| Slot { posted: AtomicU64::new(0), job: UnsafeCell::new((noop, 0)) }).collect(),
            left: AtomicUsize::new(0),
            caller: UnsafeCell::new(None),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
        });
        let workers = (0..n)
            .map(|w| {
                let shared = Arc::clone(&shared);
                thread::Builder::new().name(format!("presk-worker-{}", w + 1)).spawn(move || work(&shared, w)).expect("a worker thread")
            })
            .collect();
        WORKERS.fetch_add(n, Ordering::Relaxed);
        Pool { shared, workers, posts: 0 }
    }

    /// The workers beside the calling thread.
    pub fn len(&self) -> usize {
        self.workers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.workers.is_empty()
    }

    /// `f(0)`, …, `f(n − 1)`: piece 0 on the calling thread, piece `t` on
    /// worker `t`, the pieces beyond the workers on the calling thread after
    /// its own; returns when all are done. A piece that panics makes this
    /// panic, after every other piece is done.
    pub fn run(&mut self, n: usize, f: &(dyn Fn(usize) + Sync)) {
        let posted = n.saturating_sub(1).min(self.workers.len());
        if posted == 0 {
            (0..n).for_each(f);
            return;
        }
        // SAFETY: the job is read by the workers only until they signal it
        // done, and this function waits for that before returning.
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f) };
        let sh = &*self.shared;
        unsafe { *sh.caller.get() = Some(thread::current()) };
        sh.left.store(posted, Ordering::Relaxed);
        self.posts += 1;
        for t in 1..=posted {
            let slot = &sh.slots[t - 1];
            unsafe { *slot.job.get() = (job, t) };
            slot.posted.store(self.posts, Ordering::Release);
            self.workers[t - 1].thread().unpark();
        }
        let mine = catch_unwind(AssertUnwindSafe(|| {
            f(0);
            (posted + 1..n).for_each(f);
        }));
        let (mut spins, budget) = (0, spin());
        while sh.left.load(Ordering::Acquire) != 0 {
            if spins < budget {
                spins += 1;
                std::hint::spin_loop();
            } else {
                thread::park();
            }
        }
        if let Err(e) = mine {
            resume_unwind(e);
        }
        if sh.panicked.swap(false, Ordering::Relaxed) {
            panic!("a piece of a divided launch panicked on a worker");
        }
    }
}

fn work(sh: &Shared, w: usize) {
    let slot = &sh.slots[w];
    let mut seen = 0;
    loop {
        let (mut spins, budget) = (0, spin());
        let now = loop {
            let p = slot.posted.load(Ordering::Acquire);
            if p != seen {
                break p;
            }
            if sh.stop.load(Ordering::Acquire) {
                return;
            }
            if spins < budget {
                spins += 1;
                std::hint::spin_loop();
            } else {
                thread::park();
            }
        };
        seen = now;
        let (job, piece) = unsafe { *slot.job.get() };
        // SAFETY: the launch that posted the piece waits for it (above).
        if catch_unwind(AssertUnwindSafe(|| unsafe { (*job)(piece) })).is_err() {
            sh.panicked.store(true, Ordering::Relaxed);
        }
        let caller = unsafe { (*sh.caller.get()).clone() };
        if sh.left.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(c) = caller
        {
            c.unpark();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        WORKERS.fetch_sub(self.workers.len(), Ordering::Relaxed);
        self.shared.stop.store(true, Ordering::Release);
        for w in &self.workers {
            w.thread().unpark();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}
