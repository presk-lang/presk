//! From a PSK-Core function body to an e-graph and back: the ANF `let`s
//! become nodes (variables are the classes of their definitions, so common
//! subexpressions merge on entry), the rules run under limits, the cheapest
//! term of the result's class is extracted and written back as fresh ANF.
//! Unused definitions are not in the result's term and disappear.

use std::collections::HashMap;
use std::time::Duration;

use egg::{EGraph, Extractor, Id, Runner, StopReason};
use presk_core::ir::{Block, Func, Let, Lit, Rhs};
use presk_core::{PrimOp, VarId};
use crate::analysis::{Ana, Modules};
use crate::cost::{Cost, Model};
use crate::lang::{Node, Opaque, ParamKey, Table, node_of};
use crate::model::DimGuess;
use crate::rules::rules;
use crate::{Options, Outcome};

/// Rewrite one function; `None` when the body is left as it was (a hit
/// limit before any extraction, or a term whose type is unknown).
pub fn rewrite_func(func: &Func, modules: &Modules, opts: &Options, guess: &DimGuess) -> (Option<Block>, Outcome) {
    let limits = &opts.limits;
    let model = || match &opts.profile {
        Some(profile) => Model::Roofline { profile, guess },
        None => Model::Counts,
    };
    // ----- build -----
    let mut egraph: EGraph<Node, Ana> = EGraph::new(Ana { modules: modules.clone(), table: Table::default() });
    let mut class: HashMap<VarId, Id> = HashMap::new();
    // The original definition of a class: its source name and span.
    let mut orig: HashMap<Id, (Option<String>, presk_syntax::Span)> = HashMap::new();
    let mut param_of: HashMap<Id, VarId> = HashMap::new();
    for (i, p) in func.params.iter().enumerate() {
        let id = egraph.add(Node::Param(ParamKey(i as u32)));
        egraph[id].data.ty = Some(p.ty.clone());
        class.insert(p.var, id);
        param_of.insert(id, p.var);
    }
    for l in &func.body.lets {
        if matches!(l.rhs, Rhs::Grad(_)) {
            return (None, Outcome::Skipped("contains `grad`".into()));
        }
        let mut table = std::mem::take(&mut egraph.analysis.table);
        let lookup = |v: VarId| class[&v];
        let node = node_of(&l.rhs, &mut table, &lookup, &mut |leaf| egraph.add(leaf));
        egraph.analysis.table = table;
        let id = egraph.add(node);
        if egraph[id].data.ty.is_none() {
            egraph[id].data.ty = Some(l.ty.clone());
        }
        class.insert(l.var, id);
        let e = orig.entry(id).or_insert((None, l.span));
        if e.0.is_none() && l.name.is_some() {
            e.0 = l.name.clone();
        }
    }
    egraph.rebuild();
    let root = class[&func.body.result];
    let lets_before = func.body.lets.len();
    let cost_before = {
        let ex = Extractor::new(&egraph, Cost { egraph: &egraph, model: model() });
        ex.find_best_cost(root)
    };

    // ----- saturate -----
    let runner = Runner::<Node, Ana, ()>::new(Ana { modules: modules.clone(), table: Table::default() })
        .with_egraph(egraph)
        .with_iter_limit(limits.iterations)
        .with_node_limit(limits.nodes)
        .with_time_limit(Duration::from_millis(limits.time_ms))
        .with_explanations_disabled()
        .run(&rules());
    let stop = match runner.stop_reason.clone() {
        Some(StopReason::Saturated) => "saturated".to_string(),
        Some(StopReason::IterationLimit(n)) => format!("iteration limit {n}"),
        Some(StopReason::NodeLimit(n)) => format!("node limit {n}"),
        Some(StopReason::TimeLimit(t)) => format!("time limit {t:.2} s"),
        Some(StopReason::Other(s)) => s,
        None => "not run".to_string(),
    };
    let iterations = runner.iterations.len();
    let egraph = runner.egraph;
    let root = egraph.find(root);

    // ----- extract and write back -----
    let extractor = Extractor::new(&egraph, Cost { egraph: &egraph, model: model() });
    let cost_after = extractor.find_best_cost(root);
    let orig: HashMap<Id, (Option<String>, presk_syntax::Span)> = orig.into_iter().map(|(id, v)| (egraph.find(id), v)).collect();
    let param_of: HashMap<Id, VarId> = param_of.into_iter().map(|(id, v)| (egraph.find(id), v)).collect();
    let next_var = func.params.iter().map(|p| p.var.0).chain(func.body.lets.iter().map(|l| l.var.0)).max().map_or(0, |m| m + 1);
    let mut w = Writer { egraph: &egraph, extractor: &extractor, orig: &orig, param_of: &param_of, span: func.span, lets: Vec::new(), memo: HashMap::new(), next_var };
    let result = match w.emit(root) {
        Ok(v) => v,
        Err(why) => return (None, Outcome::Skipped(why)),
    };
    let block = Block { lets: w.lets, result };
    let lets_after = block.lets.len();
    let outcome = Outcome::Rewritten { lets_before, lets_after, cost_before, cost_after, stop, iterations, classes: egraph.number_of_classes(), nodes: egraph.total_number_of_nodes() };
    (Some(block), outcome)
}

struct Writer<'e, 'g> {
    egraph: &'g EGraph<Node, Ana>,
    extractor: &'e Extractor<'g, Cost<'g>, Node, Ana>,
    orig: &'e HashMap<Id, (Option<String>, presk_syntax::Span)>,
    param_of: &'e HashMap<Id, VarId>,
    span: presk_syntax::Span,
    lets: Vec<Let>,
    memo: HashMap<Id, VarId>,
    next_var: u32,
}

impl Writer<'_, '_> {
    fn nat(&self, id: Id) -> Result<u64, String> {
        self.egraph[id].nodes.iter().find_map(|n| if let Node::Nat(k) = n { Some(*k) } else { None }).ok_or_else(|| "a natural attribute is missing".into())
    }

    fn sym(&self, id: Id) -> Result<String, String> {
        self.egraph[id].nodes.iter().find_map(|n| if let Node::Sym(s) = n { Some(s.to_string()) } else { None }).ok_or_else(|| "a field name is missing".into())
    }

    fn names(&self, id: Id) -> Result<Vec<String>, String> {
        self.egraph[id].nodes.iter().find_map(|n| if let Node::Names(k) = n { Some(self.egraph.analysis.table.get_names(*k).to_vec()) } else { None }).ok_or_else(|| "record field names are missing".into())
    }

    fn emit(&mut self, id: Id) -> Result<VarId, String> {
        let id = self.egraph.find(id);
        if let Some(v) = self.memo.get(&id) {
            return Ok(*v);
        }
        if let Some(p) = self.param_of.get(&id) {
            self.memo.insert(id, *p);
            return Ok(*p);
        }
        let node = self.extractor.find_best_node(id).clone();
        let args = |w: &mut Self, ids: &[Id]| ids.iter().map(|x| w.emit(*x)).collect::<Result<Vec<_>, _>>();
        let prim = |op: PrimOp, a: Vec<VarId>| Rhs::Prim { op, args: a };
        let rhs = match &node {
            Node::Nat(k) => Rhs::Lit(Lit::Nat(*k)),
            Node::Add([a, b]) => prim(PrimOp::Add, args(self, &[*a, *b])?),
            Node::Sub([a, b]) => prim(PrimOp::Sub, args(self, &[*a, *b])?),
            Node::Mul([a, b]) => prim(PrimOp::Mul, args(self, &[*a, *b])?),
            Node::Div([a, b]) => prim(PrimOp::Div, args(self, &[*a, *b])?),
            Node::Neg(x) => prim(PrimOp::Neg, args(self, &[*x])?),
            Node::Exp(x) => prim(PrimOp::Exp, args(self, &[*x])?),
            Node::Log(x) => prim(PrimOp::Log, args(self, &[*x])?),
            Node::Sqrt(x) => prim(PrimOp::Sqrt, args(self, &[*x])?),
            Node::Pow([n, x]) => prim(PrimOp::Pow(self.nat(*n)?), args(self, &[*x])?),
            Node::Real(x) => prim(PrimOp::Real, args(self, &[*x])?),
            Node::ZerosLike(x) => prim(PrimOp::ZerosLike, args(self, &[*x])?),
            Node::SumTo([x, t]) => prim(PrimOp::SumTo, args(self, &[*x, *t])?),
            Node::BroadcastLike([x, t]) => prim(PrimOp::BroadcastLike, args(self, &[*x, *t])?),
            Node::SumAll(x) => prim(PrimOp::SumAll, args(self, &[*x])?),
            Node::MeanAll(x) => prim(PrimOp::MeanAll, args(self, &[*x])?),
            Node::Field([n, x]) => Rhs::Field(self.emit(*x)?, self.sym(*n)?),
            Node::SetField([n, r, v]) => prim(PrimOp::SetField { name: self.sym(*n)? }, args(self, &[*r, *v])?),
            Node::Proj([i, x]) => Rhs::Proj(self.emit(*x)?, self.nat(*i)? as usize),
            Node::Tuple(xs) => Rhs::Tuple(args(self, xs)?),
            Node::Record(ch) => {
                let names = self.names(ch[0])?;
                let vals = args(self, &ch[1..])?;
                Rhs::Record(names.into_iter().zip(vals).collect())
            }
            Node::Param(_) | Node::Names(_) | Node::Sym(_) => return Err("an attribute leaf is used as a value".into()),
            Node::Other(k, ch) => {
                let a = args(self, ch)?;
                match self.egraph.analysis.table.get_op(*k).clone() {
                    Opaque::Lit(l) => Rhs::Lit(l),
                    Opaque::Prim(op) => Rhs::Prim { op, args: a },
                    Opaque::Call { f, dim_args } => Rhs::Call { f, dim_args, args: a },
                    Opaque::Closure(f) => Rhs::Closure { f, captured: a },
                    Opaque::Apply => Rhs::Apply { fun: a[0], args: a[1..].to_vec() },
                    Opaque::Fold { body, dim_args, n, stacked, index, reverse, scan } => Rhs::Fold { body, dim_args, n, stacked: a[..stacked].to_vec(), init: a[stacked], extra: a[stacked + 1..].to_vec(), index, reverse, scan },
                }
            }
        };
        let Some(ty) = self.egraph[id].data.ty.clone() else {
            return Err(format!("the type of `{node}` is unknown"));
        };
        let (name, span) = self.orig.get(&id).cloned().unwrap_or((None, self.span));
        let var = VarId(self.next_var);
        self.next_var += 1;
        self.lets.push(Let { var, name, ty, rhs, span });
        self.memo.insert(id, var);
        Ok(var)
    }
}
