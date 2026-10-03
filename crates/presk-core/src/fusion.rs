//! PSK-Realize's fusion groups over a PSK-Core body: the partition of its
//! elementwise `let`s into groups, each one loop nest (the `presk-realize`
//! crate describes them, and re-exports this module).
//!
//! It lives in PSK-Core since 0.10.0 step 3, because the dead-code pass after
//! automatic differentiation ([`crate::ad::prune`]) must keep it: a value
//! inside a group stays in the kernel's arithmetic type (`f32` for `bf16`),
//! and a value outside every group reaches memory in its storage type, so
//! removing a reader of a value can change what a kernel computes.

use std::collections::{HashMap, HashSet};

use presk_types::ty::Ty;

use crate::ir::{Func, Let, PrimOp, Rhs, VarId};

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

fn fusible(l: &Let) -> Option<(&PrimOp, &Vec<VarId>)> {
    match &l.rhs {
        // A tensor, or a parameter tree: the backend maps a group over the
        // leaves of a tree, which is how the optimizer's long chain fuses.
        Rhs::Prim { op, args } if op.elementwise() && matches!(l.ty, Ty::Real(_) | Ty::Params(_)) => Some((op, args)),
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

/// A contraction whose value one `add` alone reads, the `add` launched by
/// itself (plan 0.10.0 step 7): the product's kernel can add the `add`'s
/// other operand to each output element once its sum is done, its epilogue,
/// and the `add` then launches nothing. Whether a plan asks for it, whether
/// the two sites have one type and whether the operand fits the product's
/// output are the backend's to check; this is the structure alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Epilogue {
    /// The product's `let`.
    pub product: VarId,
    /// The `add`'s `let`.
    pub add: VarId,
    /// The `add`'s other operand.
    pub other: VarId,
    /// The product is the `add`'s first operand.
    pub first: bool,
}

/// A primitive that may be a contraction nest: whether it is one at its
/// operands' shapes is the lowering's (`Contraction::of`).
fn may_contract(op: &PrimOp, args: &[VarId]) -> bool {
    args.len() == 2 && matches!(op, PrimOp::Dot | PrimOp::Matmul | PrimOp::DotVjpL | PrimOp::DotVjpR | PrimOp::Ein { .. })
}

/// The epilogue the product at `var` can take, given its function's fusion
/// groups `plan`; else why it can take none, phrased for a diagnostic.
pub fn epilogue(func: &Func, plan: &Plan, var: VarId) -> Result<Epilogue, String> {
    let body = &func.body;
    let at = |v: VarId| body.lets.iter().find(|l| l.var == v);
    let Some(p) = at(var) else { return Err(format!("`{}` has no binding %{}", func.name, var.0)) };
    match &p.rhs {
        Rhs::Prim { op, args } if may_contract(op, args) && matches!(p.ty, Ty::Real(_)) => {}
        _ => return Err(format!("%{} is not a product: an epilogue is a decision of a contraction nest", var.0)),
    }
    if body.result == var {
        return Err(format!("%{} is the result of `{}`, which no `add` reads", var.0, func.name));
    }
    let readers: Vec<&Let> = body.lets.iter().filter(|l| uses_of(&l.rhs).contains(&var)).collect();
    let [a] = readers.as_slice() else {
        return Err(format!("the value of %{} is read by {} bindings, and an epilogue needs it read by one `add` alone", var.0, readers.len()));
    };
    let (first, other) = match &a.rhs {
        Rhs::Prim { op: PrimOp::Add, args } if args.len() == 2 && matches!(a.ty, Ty::Real(_)) && (args[0] == var) != (args[1] == var) => {
            if args[0] == var { (true, args[1]) } else { (false, args[0]) }
        }
        _ => return Err(format!("the value of %{} is read by %{}, which is not an `add` of it and another tensor", var.0, a.var.0)),
    };
    if plan.interior.contains(&a.var) || plan.groups.contains_key(&a.var) {
        return Err(format!("the `add` that reads %{}, %{}, is a member of a fusion group, which launches it with its other members", var.0, a.var.0));
    }
    Ok(Epilogue { product: var, add: a.var, other, first })
}

/// Every epilogue of a function's products, by the product's `let`.
pub fn epilogues(func: &Func, plan: &Plan) -> HashMap<VarId, Epilogue> {
    func.body.lets.iter().filter_map(|l| epilogue(func, plan, l.var).ok()).map(|e| (e.product, e)).collect()
}
