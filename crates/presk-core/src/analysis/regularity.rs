//! Regularity classes: for every variable, the worst class of the function
//! that computes it from each input parameter, with the span of the
//! primitive responsible (the *witness*). `grad` of a function value is
//! legal only if its result is at least `C⁰⁺` in the differentiated
//! parameter.

use std::collections::{BTreeMap, HashMap};
use std::fmt;

use presk_syntax::{Diagnostic, Span};
use presk_types::elab::ElabProgram;
use presk_types::ty::Ty;

use crate::ir::{Func, FuncId, PrimOp, Program, Rhs, VarId};

/// The three-point lattice `C∞ < C⁰⁺ < Disc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// Infinitely differentiable.
    Smooth,
    /// Continuous, smooth outside a set of measure zero (`relu`, `abs`).
    PiecewiseSmooth,
    /// Discontinuous (`argmax`, comparisons, indexing by an index argument).
    Disc,
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Class::Smooth => "C∞",
            Class::PiecewiseSmooth => "C⁰⁺",
            Class::Disc => "Disc",
        })
    }
}

/// Class of a function's result with respect to one parameter, with the
/// primitive that lowered it (name and span) when the class is not `C∞`.
#[derive(Clone, Debug)]
pub struct ParamClass {
    pub class: Class,
    pub witness: Option<(String, Span)>,
}

/// Dependencies of a variable: per input parameter index, the class and witness.
type Deps = BTreeMap<usize, ParamClass>;

fn join(a: &Deps, b: &Deps) -> Deps {
    let mut out = a.clone();
    for (k, v) in b {
        match out.get(k) {
            Some(cur) if cur.class >= v.class => {}
            _ => {
                out.insert(*k, v.clone());
            }
        }
    }
    out
}

/// Raise every dependency to at least `class`, attributing it to `witness`.
fn raise(deps: &Deps, class: Class, witness: (String, Span)) -> Deps {
    deps.iter()
        .map(|(k, v)| {
            let pc = if v.class >= class { v.clone() } else { ParamClass { class, witness: Some(witness.clone()) } };
            (*k, pc)
        })
        .collect()
}

/// A function value: the function and the dependencies of its captured arguments.
#[derive(Clone, Debug)]
struct FnValue {
    f: FuncId,
    captured: Vec<Deps>,
}

struct Ctx<'a> {
    prog: &'a Program,
    elab: &'a ElabProgram,
    diags: &'a mut Vec<Diagnostic>,
    summaries: HashMap<FuncId, Vec<ParamClass>>,
}

pub fn analyze(prog: &Program, elab: &ElabProgram, diags: &mut Vec<Diagnostic>) -> HashMap<FuncId, Vec<ParamClass>> {
    let mut ctx = Ctx { prog, elab, diags, summaries: HashMap::new() };
    for i in 0..prog.funcs.len() {
        ctx.summary(FuncId(i as u32));
    }
    ctx.summaries
}

/// Class of primitive `op` in its `arg`-th argument, given the argument's type.
fn prim_class(op: &PrimOp, arg: usize, arg_ty: &Ty) -> Class {
    match op {
        PrimOp::Argmax | PrimOp::Sample => Class::Disc,
        PrimOp::Embed | PrimOp::CrossEntropy => {
            if arg == 1 { Class::Disc } else { Class::Smooth }
        }
        PrimOp::Relu | PrimOp::Abs => Class::PiecewiseSmooth,
        PrimOp::Real => {
            if matches!(arg_ty, Ty::Idx { .. }) { Class::Disc } else { Class::Smooth }
        }
        _ => Class::Smooth,
    }
}

impl Ctx<'_> {
    fn summary(&mut self, id: FuncId) -> Vec<ParamClass> {
        if let Some(s) = self.summaries.get(&id) {
            return s.clone();
        }
        let f = self.prog.func(id);
        let (deps, _) = self.body(f);
        let result = deps.get(&f.body.result).cloned().unwrap_or_default();
        let summary: Vec<ParamClass> = (0..f.params.len()).map(|i| result.get(&i).cloned().unwrap_or(ParamClass { class: Class::Smooth, witness: None })).collect();
        self.summaries.insert(id, summary.clone());
        summary
    }

    /// Dependencies of every variable of the body, and the function values.
    fn body(&mut self, f: &Func) -> (HashMap<VarId, Deps>, HashMap<VarId, FnValue>) {
        let mut deps: HashMap<VarId, Deps> = HashMap::new();
        let mut fns: HashMap<VarId, FnValue> = HashMap::new();
        let mut types: HashMap<VarId, Ty> = HashMap::new();
        for (i, p) in f.params.iter().enumerate() {
            deps.insert(p.var, BTreeMap::from([(i, ParamClass { class: Class::Smooth, witness: None })]));
            types.insert(p.var, p.ty.clone());
        }
        let get = |deps: &HashMap<VarId, Deps>, v: &VarId| deps.get(v).cloned().unwrap_or_default();
        for l in &f.body.lets {
            types.insert(l.var, l.ty.clone());
            let d = match &l.rhs {
                Rhs::Lit(_) => Deps::new(),
                Rhs::Prim { op, args } => {
                    let mut acc = Deps::new();
                    for (i, a) in args.iter().enumerate() {
                        let ty = types.get(a).cloned().unwrap_or(Ty::Nat);
                        let c = prim_class(op, i, &ty);
                        let raised = raise(&get(&deps, a), c, (op.name(), l.span));
                        acc = join(&acc, &raised);
                    }
                    acc
                }
                Rhs::Call { f: callee, args, .. } => {
                    let s = self.summary(*callee);
                    let mut acc = Deps::new();
                    for (i, a) in args.iter().enumerate() {
                        let pc = s.get(i).cloned().unwrap_or(ParamClass { class: Class::Smooth, witness: None });
                        let w = pc.witness.clone().unwrap_or((self.prog.func(*callee).name.clone(), l.span));
                        acc = join(&acc, &raise(&get(&deps, a), pc.class, w));
                    }
                    acc
                }
                Rhs::Fold { body, stacked, init, extra, index, .. } => {
                    // The body's parameters after the optional index: stacked slices, the
                    // accumulator, the extras; the accumulator feeds back into itself.
                    let s = self.summary(*body);
                    let off = usize::from(*index);
                    let mut acc = Deps::new();
                    let inputs: Vec<&VarId> = stacked.iter().chain(std::iter::once(init)).chain(extra.iter()).collect();
                    for (i, a) in inputs.iter().enumerate() {
                        let pc = s.get(off + i).cloned().unwrap_or(ParamClass { class: Class::Smooth, witness: None });
                        let w = pc.witness.clone().unwrap_or((self.prog.func(*body).name.clone(), l.span));
                        acc = join(&acc, &raise(&get(&deps, a), pc.class, w));
                    }
                    acc
                }
                Rhs::Tuple(vs) => vs.iter().fold(Deps::new(), |acc, v| join(&acc, &get(&deps, v))),
                Rhs::Record(fields) => fields.iter().fold(Deps::new(), |acc, (_, v)| join(&acc, &get(&deps, v))),
                Rhs::Proj(v, _) | Rhs::Field(v, _) => get(&deps, v),
                Rhs::Closure { f: lam, captured } => {
                    fns.insert(l.var, FnValue { f: *lam, captured: captured.iter().map(|c| get(&deps, c)).collect() });
                    captured.iter().fold(Deps::new(), |acc, v| join(&acc, &get(&deps, v)))
                }
                Rhs::Grad(v) => {
                    if let Some(fv) = fns.get(v).cloned() {
                        self.check_grad(&fv, l.span, f);
                        fns.insert(l.var, fv);
                    }
                    get(&deps, v)
                }
                Rhs::Apply { fun, args } => {
                    let mut acc = get(&deps, fun);
                    if let Some(fv) = fns.get(fun).cloned() {
                        let s = self.summary(fv.f);
                        let n = fv.captured.len();
                        for (i, a) in args.iter().enumerate() {
                            let pc = s.get(n + i).cloned().unwrap_or(ParamClass { class: Class::Smooth, witness: None });
                            let w = pc.witness.clone().unwrap_or((self.prog.func(fv.f).name.clone(), l.span));
                            acc = join(&acc, &raise(&get(&deps, a), pc.class, w));
                        }
                    } else {
                        for a in args {
                            acc = join(&acc, &get(&deps, a));
                        }
                    }
                    acc
                }
            };
            deps.insert(l.var, d);
        }
        (deps, fns)
    }

    /// `grad` of a function value: its result must be at least `C⁰⁺` in the
    /// differentiated (non-captured) parameters.
    fn check_grad(&mut self, fv: &FnValue, grad_span: Span, in_fn: &Func) {
        let s = self.summary(fv.f);
        let lam = self.prog.func(fv.f);
        let n = fv.captured.len();
        for (i, pc) in s.iter().enumerate().skip(n) {
            if pc.class == Class::Disc {
                let pname = lam.params.get(i).map(|p| p.name.clone()).unwrap_or_default();
                let (culprit, wspan) = pc.witness.clone().unwrap_or(("a discontinuous operation".into(), grad_span));
                let help = match culprit.as_str() {
                    "argmax" => "use `softmax` for a smooth surrogate, or an explicit straight-through relaxation",
                    "sample" => "sampling is not differentiable: differentiate the logits and sample outside the differentiated function",
                    "embed" | "cross_entropy" => "the gradient with respect to an index does not exist: differentiate with respect to the table or the logits instead",
                    "real" => "the numeric value of an index is piecewise constant: no gradient flows through `real(t)`",
                    _ => "the differentiated path must be at least C⁰⁺ (piecewise smooth)",
                };
                let mut d = Diagnostic::error("E3001", format!("cannot differentiate through `{culprit}`: the function is discontinuous (Disc) in `{pname}`"))
                    .label(grad_span, format!("`grad` here, with respect to `{pname}`"));
                if wspan != grad_span {
                    d = d.label(wspan, format!("`{culprit}` makes the result discontinuous in `{pname}`"));
                }
                d = d.help(help).note(format!("in `{}`; `grad` requires the class C⁰⁺ or better", in_fn.name));
                self.diags.push(d);
                let _ = self.elab;
            }
        }
    }
}
