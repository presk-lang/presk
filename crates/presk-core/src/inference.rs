//! The inference reading of a program (the reference's requirement 3.30,
//! the compiler's question 78, plan 0.16.0 step 3, D160): every
//! `dropout(r, k, x)` read as `x`.
//!
//! An evaluation is in one of two modes. A training run computes its
//! objective as written, `dropout` drawing its mask from its key; every
//! other evaluation (an entry evaluated by a tool, an inference serve, an
//! exported program) computes the function's inference reading, in which
//! every mask is the tensor of ones of its type. Until 0.16.0 the program
//! was the same in both modes and the executors filled the mask with ones
//! at launch, so that a realization typed the mask and the solver offered
//! it as a stored value. The mode is part of what an entry denotes, so the
//! lowering specializes the program to it: in the inference mode
//! [`reading`] removes each `dropout_mask` and the product by it, and with
//! them the `let`s that only they read, a key's `split_key` among them.
//! What remains computes the same bits, `x · 1` being `x`.
//!
//! The pass runs on the lowered program, before differentiation, so that
//! no `.vjp`, `.fwd` or `.bwd` ever holds a mask either. A `let` keeps its
//! number and a function its name, as `ad::prune` keeps them, since a site
//! is named `<function>/%<let>`.

use std::collections::{HashMap, HashSet};

use crate::ir::{PrimOp, Program, Rhs, VarId};

/// The mode a program is lowered for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// The program as written, its masks drawn when a training run turns
    /// the executors' training on: `presk train`, the `--step` paths,
    /// `presk check`.
    #[default]
    Training,
    /// The inference reading: no `dropout_mask`, no product by one.
    Inference,
}

/// What [`reading`] removed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Removed {
    /// The masks, with their products: one per `dropout`.
    pub masks: usize,
    /// The `let`s only they read, transitively.
    pub dead: usize,
    /// Per function, the `let`s removed (masks, products and dead ones).
    pub lets: HashMap<String, Vec<VarId>>,
}

/// Rewrite `prog` to its inference reading, in place.
///
/// A `dropout` lowers to `%m = dropout_mask r, k, x` and `%y = mul x, %m`
/// (`lower/expr.rs`). Every `%y` read becomes a read of `x`, the two `let`s
/// go, and so does every `let` that only they read (a `let` read by
/// nothing before the pass stays: removing what the program wrote unread is
/// not this pass's business). A mask read by anything but its product is
/// left as it is.
pub fn reading(prog: &mut Program) -> Removed {
    let mut removed = Removed::default();
    for f in &mut prog.funcs {
        let reads = |lets: &[crate::ir::Let], result: VarId| -> HashMap<VarId, usize> {
            let mut n: HashMap<VarId, usize> = HashMap::new();
            for l in lets {
                for v in l.rhs.vars() {
                    *n.entry(v).or_default() += 1;
                }
            }
            *n.entry(result).or_default() += 1;
            n
        };
        let before = reads(&f.body.lets, f.body.result);
        let masks: HashSet<VarId> = f.body.lets.iter().filter(|l| matches!(&l.rhs, Rhs::Prim { op: PrimOp::DropoutMask, .. })).map(|l| l.var).collect();
        if masks.is_empty() {
            continue;
        }
        // `%y = mul x, %m` (either order), `%m` read by it alone.
        let mut subst: HashMap<VarId, VarId> = HashMap::new();
        let mut gone: HashSet<VarId> = HashSet::new();
        for l in &f.body.lets {
            if let Rhs::Prim { op: PrimOp::Mul, args } = &l.rhs
                && args.len() == 2
            {
                let (x, m) = if masks.contains(&args[1]) { (args[0], args[1]) } else if masks.contains(&args[0]) { (args[1], args[0]) } else { continue };
                if before.get(&m) == Some(&1) && !gone.contains(&m) {
                    subst.insert(l.var, x);
                    gone.insert(m);
                    gone.insert(l.var);
                    removed.masks += 1;
                }
            }
        }
        // A product's operand may itself be a product removed (a dropout of
        // a dropout): follow the chain.
        let resolve = |mut v: VarId| {
            while let Some(w) = subst.get(&v) {
                v = *w;
            }
            v
        };
        let mut lets: Vec<crate::ir::Let> = f.body.lets.iter().filter(|l| !gone.contains(&l.var)).cloned().collect();
        for l in &mut lets {
            l.rhs = l.rhs.map_vars(resolve);
        }
        let result = resolve(f.body.result);
        // The `let`s read before and by nothing now.
        loop {
            let now = reads(&lets, result);
            let dead: Vec<VarId> = lets.iter().filter(|l| before.get(&l.var).copied().unwrap_or(0) > 0 && now.get(&l.var).copied().unwrap_or(0) == 0).map(|l| l.var).collect();
            if dead.is_empty() {
                break;
            }
            removed.dead += dead.len();
            gone.extend(dead.iter().copied());
            lets.retain(|l| !dead.contains(&l.var));
        }
        let mut names: Vec<VarId> = gone.into_iter().collect();
        names.sort_by_key(|v| v.0);
        removed.lets.insert(f.name.clone(), names);
        f.body.lets = lets;
        f.body.result = result;
    }
    removed
}

/// A `dropout` of a training run's program: its mask, the product by it,
/// and what the product multiplies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    /// The function, by name.
    pub func: String,
    pub mask: VarId,
    pub product: VarId,
    /// The product's other operand: a `let` of the function, or one of its
    /// parameters.
    pub operand: Operand,
}

/// What a mask's product multiplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    Let(VarId),
    Param(VarId),
}

/// Every `dropout` of a program as written (the pattern [`reading`]
/// removes), function by function.
pub fn masks(prog: &Program) -> Vec<Mask> {
    let mut out = Vec::new();
    for f in &prog.funcs {
        let masks: HashSet<VarId> = f.body.lets.iter().filter(|l| matches!(&l.rhs, Rhs::Prim { op: PrimOp::DropoutMask, .. })).map(|l| l.var).collect();
        let lets: HashSet<VarId> = f.body.lets.iter().map(|l| l.var).collect();
        for l in &f.body.lets {
            if let Rhs::Prim { op: PrimOp::Mul, args } = &l.rhs
                && args.len() == 2
            {
                let (x, m) = if masks.contains(&args[1]) { (args[0], args[1]) } else if masks.contains(&args[0]) { (args[1], args[0]) } else { continue };
                let operand = if lets.contains(&x) { Operand::Let(x) } else { Operand::Param(x) };
                out.push(Mask { func: f.name.clone(), mask: m, product: l.var, operand });
            }
        }
    }
    out
}
