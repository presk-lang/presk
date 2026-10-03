//! PSK-Realize v0: **fusion groups** over PSK-Core bodies (0.4.0 plan, §3.4).
//!
//! The representation was named PSK-Sched until 0.6.0: "schedule" suggests an
//! ordering in time, and what a plan at this level mostly fixes is arithmetic
//! (language reference, chapter 6, 2026-09-23).
//!
//! One kernel per primitive makes every intermediate a round trip through
//! memory. A group is a connected set of elementwise nodes of the same result
//! type whose intermediates are used nowhere else; lowered to a single loop
//! nest it reads its external inputs once and writes one output, so the
//! cost-model document's rule applies:
//!
//! ```text
//! Q(G) = external inputs of G + external outputs of G
//! ```
//!
//! The partition is structural — it depends on the body, not on the shapes or
//! the realization — so it is computed once per function and reused by every
//! backend. v0 fuses elementwise chains only: a reduction ends a group,
//! because its loop nest is not the nest of its consumers.

pub use presk_core::fusion::{Group, Plan, Src, Step, plan};
use presk_core::ir::Func;
use presk_core::PrimOp;

/// Whether a primitive is an elementwise map over its operands, and so can
/// share a loop nest with its neighbours ([`PrimOp::elementwise`], in
/// PSK-Core since 0.10.0 step 3, whose dead-code pass needs it too).
pub fn elementwise(op: &PrimOp) -> bool {
    op.elementwise()
}

/// One line per group, for `presk dump --stage sched`.
pub fn dump(func: &Func, plan: &Plan) -> String {
    let mut out = format!("fn {} — {} lets, {} groups, {} launches saved\n", func.name, func.body.lets.len(), plan.groups.len(), plan.saved());
    for l in &func.body.lets {
        if let Some(g) = plan.groups.get(&l.var) {
            out.push_str(&format!("  %{} = {} [{} inputs, {} steps]\n", g.root.0, g.signature(), g.inputs.len(), g.steps.len()));
        }
    }
    out
}
