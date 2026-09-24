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

use std::collections::{HashMap, HashSet};

use presk_core::ir::{Func, Let, Rhs};
use presk_core::{PrimOp, VarId};
use presk_types::ty::Ty;

/// Where a step reads an operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Src {
    /// The group's `i`-th external input (a buffer the kernel reads).
    Input(usize),
    /// The value computed by an earlier step, which never reaches memory.
    Step(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Step {
    pub op: PrimOp,
    pub args: Vec<Src>,
}

/// A fused group: its steps in evaluation order, the last one producing the
/// value the group's root `let` binds.
#[derive(Clone, Debug)]
pub struct Group {
    pub root: VarId,
    /// The `let`s the group subsumes, in body order; the root is last.
    pub members: Vec<VarId>,
    /// The values the group reads from outside, in kernel-operand order.
    pub inputs: Vec<VarId>,
    pub steps: Vec<Step>,
    /// The result type shared by every member.
    pub ty: Ty,
}

impl Group {
    /// Identifies the computation, for the kernel cache. Shapes are added by
    /// the caller, which knows them.
    pub fn signature(&self) -> String {
        let step = |s: &Step| {
            let args: Vec<String> = s.args.iter().map(|a| match a {
                Src::Input(i) => format!("i{i}"),
                Src::Step(k) => format!("s{k}"),
            }).collect();
            format!("{}({})", s.op.name(), args.join(","))
        };
        self.steps.iter().map(step).collect::<Vec<_>>().join(";")
    }
}

/// The groups of one function, keyed by the `let` that binds their result,
/// and the `let`s they subsume (which the backend must not evaluate).
#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub groups: HashMap<VarId, Group>,
    pub interior: HashSet<VarId>,
}

impl Plan {
    /// Launches saved: every interior `let` is one kernel that does not run.
    pub fn saved(&self) -> usize {
        self.interior.len()
    }
}

/// The largest expression tree a group may emit. A value read twice inside a
/// group is computed twice rather than stored, which trades arithmetic for a
/// round trip through memory; this keeps that trade bounded.
const MAX_TREE: usize = 64;

/// Nodes of the tree the group would emit, counting a value once per read.
fn tree_size(members: &HashSet<VarId>, by_var: &HashMap<VarId, &Let>, root: VarId) -> usize {
    fn go(v: VarId, members: &HashSet<VarId>, by_var: &HashMap<VarId, &Let>, budget: &mut usize) {
        if *budget == 0 {
            return;
        }
        *budget -= 1;
        if !members.contains(&v) {
            return;
        }
        let Some(Rhs::Prim { args, .. }) = by_var.get(&v).map(|c| &c.rhs) else { return };
        for a in args {
            go(*a, members, by_var, budget);
        }
    }
    let mut budget = MAX_TREE + 1;
    go(root, members, by_var, &mut budget);
    MAX_TREE + 1 - budget
}

/// Whether a primitive is an elementwise map over its operands, and so can
/// share a loop nest with its neighbours. `real` and `pi` are excluded: they
/// lift a hyperparameter, and the host produces them without a kernel.
pub fn elementwise(op: &PrimOp) -> bool {
    use PrimOp::*;
    // `pow n` expands to `n − 1` products of its argument, so a big exponent
    // would duplicate the fused subtree; those stay kernels of their own.
    matches!(op, Add | Sub | Mul | Div | Neg | Exp | Log | Tanh | Sqrt | Gelu | Relu | Abs | Sigmoid | Sin | Cos | GeluDeriv | Heaviside | Sign) || matches!(op, Pow(n) if *n <= 2)
}

fn fusible(l: &Let) -> Option<(&PrimOp, &Vec<VarId>)> {
    match &l.rhs {
        // A tensor, or a parameter tree: the backend maps a group over the
        // leaves of a tree, which is how the optimizer's long chain fuses.
        Rhs::Prim { op, args } if elementwise(op) && matches!(l.ty, Ty::Real(_) | Ty::Params(_)) => Some((op, args)),
        _ => None,
    }
}

fn uses_of(rhs: &Rhs) -> Vec<VarId> {
    match rhs {
        Rhs::Lit(_) => vec![],
        Rhs::Prim { args, .. } | Rhs::Tuple(args) | Rhs::Call { args, .. } | Rhs::Closure { captured: args, .. } => args.clone(),
        Rhs::Proj(x, _) | Rhs::Field(x, _) | Rhs::Grad(x) => vec![*x],
        Rhs::Record(fields) => fields.iter().map(|(_, v)| *v).collect(),
        Rhs::Apply { fun, args } => {
            let mut v = vec![*fun];
            v.extend(args);
            v
        }
        Rhs::Fold { stacked, init, extra, .. } => {
            let mut v = stacked.clone();
            v.push(*init);
            v.extend(extra);
            v
        }
    }
}

/// Partition a function body into fusion groups, greedily and backwards: each
/// elementwise `let` becomes the root of a group that absorbs every producer
/// which is elementwise, of the same result type, and used nowhere else.
pub fn plan(func: &Func) -> Plan {
    let body = &func.body;
    let mut uses: HashMap<VarId, usize> = HashMap::new();
    for l in &body.lets {
        for v in uses_of(&l.rhs) {
            *uses.entry(v).or_default() += 1;
        }
    }
    *uses.entry(body.result).or_default() += 1;
    let by_var: HashMap<VarId, &Let> = body.lets.iter().map(|l| (l.var, l)).collect();
    // Who reads each value: a producer may join a group only if all of them do.
    let mut consumers: HashMap<VarId, Vec<VarId>> = HashMap::new();
    for l in &body.lets {
        for v in uses_of(&l.rhs) {
            consumers.entry(v).or_default().push(l.var);
        }
    }

    let mut claimed: HashSet<VarId> = HashSet::new();
    let mut plan = Plan::default();
    // Backwards, so a group forms at its last node and grows into its producers.
    for l in body.lets.iter().rev() {
        if claimed.contains(&l.var) || fusible(l).is_none() {
            continue;
        }
        // Grow the group backwards. A producer joins when every one of its
        // consumers is already inside — then its value never has to reach
        // memory — and when it is elementwise and of the group's own type.
        // A value consumed twice inside appears twice in the expression tree,
        // so the tree is bounded: past `MAX_TREE` nodes the group stops.
        let mut members: HashSet<VarId> = HashSet::from([l.var]);
        loop {
            let mut grew = false;
            let candidates: Vec<VarId> = members.iter().flat_map(|v| by_var.get(v).map(|c| uses_of(&c.rhs)).unwrap_or_default()).collect();
            for a in candidates {
                if members.contains(&a) || claimed.contains(&a) || a == body.result {
                    continue;
                }
                let Some(p) = by_var.get(&a) else { continue };
                if fusible(p).is_none() || p.ty != l.ty {
                    continue;
                }
                // Every consumer of `a` must be in the group, or `a` escapes.
                if consumers.get(&a).is_none_or(|cs| !cs.iter().all(|c| members.contains(c))) {
                    continue;
                }
                members.insert(a);
                if tree_size(&members, &by_var, l.var) > MAX_TREE {
                    members.remove(&a);
                    continue;
                }
                grew = true;
            }
            if !grew {
                break;
            }
        }
        if members.len() < 2 {
            continue;
        }
        let ordered: Vec<VarId> = body.lets.iter().filter(|m| members.contains(&m.var)).map(|m| m.var).collect();
        let index: HashMap<VarId, usize> = ordered.iter().enumerate().map(|(i, v)| (*v, i)).collect();
        let mut inputs: Vec<VarId> = Vec::new();
        let mut steps: Vec<Step> = Vec::new();
        for v in &ordered {
            let (op, args) = fusible(by_var[v]).expect("members are fusible");
            let srcs = args
                .iter()
                .map(|a| match index.get(a) {
                    Some(k) => Src::Step(*k),
                    None => Src::Input(match inputs.iter().position(|x| x == a) {
                        Some(k) => k,
                        None => {
                            inputs.push(*a);
                            inputs.len() - 1
                        }
                    }),
                })
                .collect();
            steps.push(Step { op: op.clone(), args: srcs });
        }
        for m in &ordered {
            claimed.insert(*m);
            if *m != l.var {
                plan.interior.insert(*m);
            }
        }
        plan.groups.insert(l.var, Group { root: l.var, members: ordered, inputs, steps, ty: l.ty.clone() });
    }
    plan
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
