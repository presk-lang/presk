//! Precision as a resource: what a realization is, what it costs, and what it
//! promises.
//!
//! Every other crate of this workspace works on the *denotation*: a Presk
//! program is a function over exact reals, and `presk-core`, `presk-opt` and
//! `presk-sched` transform it under identities that hold over ℝ. This crate
//! works on the other side of that line. It answers a different question:
//! given a program and a declared error budget, which machine types and which
//! reduction orders should the backends use, and how far from the denotation
//! does that put the execution?
//!
//! The version-0.5.0 plan
//! (`presk-specs/implementation/presk-plan-0.5.0.md`) calls this RA3 of the
//! research programme, *precision as a resource*. Three ingredients the rest
//! of the compiler already has make it attackable here and nowhere else:
//!
//! - the reference interpreter is an **exact oracle**, so the distance
//!   between any realization and the denotation is a measurable quantity
//!   rather than a modelled one;
//! - reverse-mode differentiation over PSK-Core gives a **sensitivity**
//!   analysis, which is how a budget is allocated between operations instead
//!   of by folklore;
//! - the calibrated roofline of `presk-opt` gives the **time** side of the
//!   bi-criteria choice, so that error can circulate as a resource in the
//!   same way as time.
//!
//! # The two quantities, which must never be confused
//!
//! A *bound* (`propagate`) is sound and pessimistic: it over-estimates the
//! distance to the denotation and it guarantees something. A *sensitivity*
//! (`sensitivity`) is a first-order estimate of how much an output moves when
//! a value is perturbed: it guarantees nothing and it ranks well. The solver
//! **proposes with the sensitivity and verifies by measurement** against the
//! exact reference (D41), because the bound, though sound, is vacuous on a
//! transformer — still ten orders above the error once each sum's condition
//! number is measured (D51). The root-sum-square reading of `propagate`
//! (`Model::Statistical`) is an estimate like the sensitivity, and is printed
//! as one. Code in this crate that mixes bounds and estimates is wrong.
//!
//! # What a bound is conditional on
//!
//! Static interval arithmetic over a neural network is hopeless: weights are
//! unbounded and the intervals explode with depth. The magnitudes here are
//! therefore measured, on the reference interpreter, over a stated number of
//! calibration batches (`envelope`). Every bound this crate prints is
//! conditional on the envelope it used, and carries it (decision D35). That
//! conditionality is a property of the result, not a caveat to be dropped in
//! the presentation layer.
//!
//! # Status
//!
//! The realization lives in `presk-rt`, at the seam every backend already
//! crosses. The magnitude envelope, the propagation (four rules for sums since
//! 0.6.0 step 6), the sensitivities, the solver — over an entry's sites, and
//! since 0.6.0 step 7b over a training step's functions — and the choice of
//! quantised parameters are here.

/// The realization: per fusion group a storage and an accumulator type, plus
/// the reduction order a contraction is allowed to use.
///
/// This is the object that replaces `presk_rt::Realize`, the two-valued flag
/// of versions 0.1.0 to 0.4.0. Decisions D31 and D32 of 0.4.0 both refused a
/// transformation — vectorizing a sum, and calling a vendor BLAS — on the
/// ground that a reassociated sum is a *different realization* owing a
/// declared order and a measured gap, and then had nowhere to declare it.
/// Here is where it is declared (plan 0.5.0 §3.1, decision D34); step 2.
pub mod realization {}

pub mod envelope;

pub mod propagate;

pub mod sensitivity;

/// Which parameters to store as integers (0.6.0 step 5d, D44).
pub mod quant;

pub mod solve;
