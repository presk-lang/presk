//! The pool of workers a divided launch runs on (plan 0.10.0 step 6, D96):
//! each piece runs once, every launch waits for all of its pieces, a worker
//! parked is woken, and a piece that panics reaches the caller.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use presk_cranelift::Pool;

#[test]
fn every_piece_runs_once_at_every_launch() {
    let mut pool = Pool::new(9);
    assert_eq!(pool.len(), 9);
    for launch in 0..20_000usize {
        // Fewer pieces than workers, as many, more.
        let n = 1 + launch % 13;
        let ran: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        pool.run(n, &|t| {
            ran[t].fetch_add(launch + 1, Ordering::Relaxed);
        });
        // Every piece done when `run` returns, and done once.
        assert!(ran.iter().all(|r| r.load(Ordering::Relaxed) == launch + 1), "launch {launch} of {n} pieces");
    }
}

#[test]
fn a_parked_worker_is_woken() {
    let mut pool = Pool::new(3);
    let sum = AtomicUsize::new(0);
    for _ in 0..3 {
        // Long enough for every worker to stop spinning and park.
        std::thread::sleep(Duration::from_millis(50));
        pool.run(4, &|t| {
            sum.fetch_add(t, Ordering::Relaxed);
        });
    }
    assert_eq!(sum.load(Ordering::Relaxed), 3 * (1 + 2 + 3));
}

#[test]
fn a_piece_that_panics_reaches_the_caller() {
    let mut pool = Pool::new(2);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.run(3, &|t| assert!(t != 2, "piece 2"))));
    assert!(r.is_err());
    // The pool still runs.
    let n = AtomicUsize::new(0);
    pool.run(3, &|_| {
        n.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(n.load(Ordering::Relaxed), 3);
    drop(pool);
}
