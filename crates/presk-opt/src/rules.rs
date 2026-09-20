//! The rewrite rules, all exact over ℝ. Three families: algebra
//! (commutativity, associativity, neutral and absorbing elements, `neg`,
//! `pow`, `exp`/`log`, `sqrt`, distributivity), reductions (a scalar factor
//! out of `sum`/`mean`), and structure (the adjoint primitives of AD:
//! `sum_to`/`broadcast_like` between equal types, `zeros_like` in sums,
//! `field`/`set_field`/`record` chains, `proj` of tuples).
//!
//! Extraction is over trees (`egg`'s extractor does not see sharing), so
//! rules that only move an operation around — a negation into a product, a
//! square into a product — are left out: they would let the extractor pick,
//! among equal-cost forms, one that duplicates a shared subterm.
//!
//! Every rule that builds a new elementwise operation is guarded by the types
//! of its operands: equal types, or a real scalar against a tensor or a tree.
//! Under that guard broadcasting is trivial, so the algebra of ℝ applies to
//! every element and the axis names of the result are those of the tensor
//! operand whatever the order. Rules on mixed shapes (`x : ℝ[B, D]` with
//! `b : ℝ[D]`) are left out on purpose: reassociating them could change the
//! alignment and is not always well typed.

use egg::{Applier, EGraph, Id, PatternAst, Rewrite, Subst, Symbol, Var, rewrite};
use presk_types::ty::Ty;

use crate::analysis::{Ana, Data, Konst, field_names, is_real_scalar, is_tensorish, ty_of};
use crate::lang::{Node, Opaque};

type G = EGraph<Node, Ana>;
type Rw = Rewrite<Node, Ana>;

fn var(s: &str) -> Var {
    s.parse().unwrap()
}

/// Both real tensors or trees, of equal types or one a real scalar.
fn compat(a: &str, b: &str) -> impl Fn(&mut G, Id, &Subst) -> bool {
    let (a, b) = (var(a), var(b));
    move |eg, _, subst| match (ty_of(eg, subst[a]), ty_of(eg, subst[b])) {
        (Some(x), Some(y)) => is_tensorish(x) && is_tensorish(y) && (x == y || is_real_scalar(x) || is_real_scalar(y)),
        _ => false,
    }
}

fn compat3(a: &str, b: &str, c: &str) -> impl Fn(&mut G, Id, &Subst) -> bool {
    let (ab, bc, ac) = (compat(a, b), compat(b, c), compat(a, c));
    move |eg, id, s| ab(eg, id, s) && bc(eg, id, s) && ac(eg, id, s)
}

/// Exactly the same type (names included).
fn same_ty(a: &str, b: &str) -> impl Fn(&mut G, Id, &Subst) -> bool {
    let (a, b) = (var(a), var(b));
    move |eg, _, subst| match (ty_of(eg, subst[a]), ty_of(eg, subst[b])) {
        (Some(x), Some(y)) => is_tensorish(x) && x == y,
        _ => false,
    }
}

fn tensorish(a: &str) -> impl Fn(&mut G, Id, &Subst) -> bool {
    let a = var(a);
    move |eg, _, subst| ty_of(eg, subst[a]).is_some_and(is_tensorish)
}

fn real_tensor(a: &str) -> impl Fn(&mut G, Id, &Subst) -> bool {
    let a = var(a);
    move |eg, _, subst| matches!(ty_of(eg, subst[a]), Some(Ty::Real(_)))
}

fn scalar(a: &str) -> impl Fn(&mut G, Id, &Subst) -> bool {
    let a = var(a);
    move |eg, _, subst| ty_of(eg, subst[a]).is_some_and(is_real_scalar)
}

/// A real scalar constant equal to `k`.
fn konst(a: &str, k: Konst) -> impl Fn(&mut G, Id, &Subst) -> bool {
    let a = var(a);
    move |eg, _, subst| {
        let d: &Data = &eg[subst[a]].data;
        d.konst == Some(k) && d.ty.as_ref().is_some_and(is_real_scalar)
    }
}

fn distinct(a: &str, b: &str) -> impl Fn(&mut G, Id, &Subst) -> bool {
    let (a, b) = (var(a), var(b));
    move |eg, _, subst| eg.find(subst[a]) != eg.find(subst[b])
}

fn both(p: impl Fn(&mut G, Id, &Subst) -> bool, q: impl Fn(&mut G, Id, &Subst) -> bool) -> impl Fn(&mut G, Id, &Subst) -> bool {
    move |eg, id, s| p(eg, id, s) && q(eg, id, s)
}

/// The tree type `ty` has fields; `Field`-typing then works on it.
fn has_fields(a: &str) -> impl Fn(&mut G, Id, &Subst) -> bool {
    let a = var(a);
    move |eg, _, subst| ty_of(eg, subst[a]).is_some_and(|t| field_names(&eg.analysis.modules, t).is_some())
}

fn sym_of(eg: &G, id: Id) -> Option<Symbol> {
    eg[id].nodes.iter().find_map(|n| if let Node::Sym(s) = n { Some(*s) } else { None })
}

fn nat_of(eg: &G, id: Id) -> Option<u64> {
    eg[id].nodes.iter().find_map(|n| if let Node::Nat(k) = n { Some(*k) } else { None })
}

fn record_of(eg: &G, id: Id) -> Option<(Vec<String>, Vec<Id>)> {
    eg[id].nodes.iter().find_map(|n| {
        if let Node::Record(ch) = n {
            let Node::Names(k) = eg[ch[0]].nodes.first()? else { return None };
            Some((eg.analysis.table.get_names(*k).to_vec(), ch[1..].to_vec()))
        } else {
            None
        }
    })
}

/// Add a record node typed like `ty` and union it with `eclass`.
fn add_record(eg: &mut G, eclass: Id, names: Vec<String>, comps: Vec<Id>, ty: Ty) -> Vec<Id> {
    let key = eg.analysis.table.names(names);
    let mut ch = vec![eg.add(Node::Names(key))];
    ch.extend(comps);
    let id = eg.add(Node::Record(ch.into_boxed_slice()));
    if eg[id].data.ty.is_none() {
        eg[id].data.ty = Some(ty);
    }
    if eg.union(eclass, id) { vec![id] } else { vec![] }
}

/// A contraction: scaling one of its operands scales its result, because a
/// sum of products is linear in each operand.
fn contraction(op: &presk_core::PrimOp) -> bool {
    matches!(op, presk_core::PrimOp::Dot | presk_core::PrimOp::Matmul | presk_core::PrimOp::Ein { .. })
}

/// `(op (contraction a b) s)` with `s` a real scalar becomes
/// `(contraction (op a s) b)` and `(contraction a (op b s))`: exact over ℝ,
/// since `(Σ aᵢbᵢ) · s = Σ (aᵢ · s) bᵢ`, and cheaper whenever an operand holds
/// fewer elements than the result — the attention scores of a transformer are
/// `T/HD` times larger than the queries they come from. Which form wins is a
/// question about *sizes*, so only a calibrated profile can answer it;
/// operation counts see one division either way.
struct ScaleIntoContraction {
    divide: bool,
}

impl Applier<Node, Ana> for ScaleIntoContraction {
    fn apply_one(&self, eg: &mut G, eclass: Id, subst: &Subst, _: Option<&PatternAst<Node>>, _: Symbol) -> Vec<Id> {
        let (x, s) = (subst[var("?x")], subst[var("?s")]);
        if !ty_of(eg, s).is_some_and(is_real_scalar) {
            return vec![];
        }
        let Some(xty) = ty_of(eg, x).cloned() else { return vec![] };
        let nodes: Vec<(crate::lang::OpKey, Vec<Id>)> = eg[x]
            .nodes
            .iter()
            .filter_map(|n| match n {
                Node::Other(k, ch) if matches!(eg.analysis.table.get_op(*k), Opaque::Prim(op) if contraction(op)) => Some((*k, ch.clone())),
                _ => None,
            })
            .collect();
        let mut out = Vec::new();
        for (k, ch) in nodes {
            for i in 0..ch.len() {
                let Some(aty) = ty_of(eg, ch[i]).cloned() else { continue };
                // Only a real tensor can carry the scalar.
                if !matches!(aty, Ty::Real(_)) {
                    continue;
                }
                let scaled = if self.divide { eg.add(Node::Div([ch[i], s])) } else { eg.add(Node::Mul([ch[i], s])) };
                if eg[scaled].data.ty.is_none() {
                    eg[scaled].data.ty = Some(aty);
                }
                let mut nch = ch.clone();
                nch[i] = scaled;
                let moved = eg.add(Node::Other(k, nch));
                // The result's shape is unchanged: only an operand was scaled.
                if eg[moved].data.ty.is_none() {
                    eg[moved].data.ty = Some(xty.clone());
                }
                if eg.union(eclass, moved) {
                    out.push(moved);
                }
            }
        }
        out
    }

    fn vars(&self) -> Vec<Var> {
        vec![var("?x"), var("?s")]
    }
}

/// `(field ?n ?r)` where `?r` is a record: its component.
struct FieldRecord;
impl Applier<Node, Ana> for FieldRecord {
    fn apply_one(&self, eg: &mut G, eclass: Id, subst: &Subst, _: Option<&PatternAst<Node>>, _: Symbol) -> Vec<Id> {
        let (Some(name), Some((names, comps))) = (sym_of(eg, subst[var("?n")]), record_of(eg, subst[var("?r")])) else { return vec![] };
        let Some(i) = names.iter().position(|m| m == name.as_str()) else { return vec![] };
        if eg.union(eclass, comps[i]) { vec![comps[i]] } else { vec![] }
    }
    fn vars(&self) -> Vec<Var> {
        vec![var("?n"), var("?r")]
    }
}

/// `(set_field ?n ?r ?v)` where `?r` is a record: the record with one component replaced.
struct SetRecord;
impl Applier<Node, Ana> for SetRecord {
    fn apply_one(&self, eg: &mut G, eclass: Id, subst: &Subst, _: Option<&PatternAst<Node>>, _: Symbol) -> Vec<Id> {
        let (Some(name), Some((names, mut comps))) = (sym_of(eg, subst[var("?n")]), record_of(eg, subst[var("?r")])) else { return vec![] };
        let Some(i) = names.iter().position(|m| m == name.as_str()) else { return vec![] };
        let Some(ty) = ty_of(eg, subst[var("?r")]).cloned() else { return vec![] };
        comps[i] = subst[var("?v")];
        add_record(eg, eclass, names, comps, ty)
    }
    fn vars(&self) -> Vec<Var> {
        vec![var("?n"), var("?r"), var("?v")]
    }
}

/// `(set_field ?n (zeros_like ?t) ?v)`: the record with `?v` in field `?n`
/// and `zeros_like (field m ?t)` in every other field `m`.
struct SetZerosRecord;
impl Applier<Node, Ana> for SetZerosRecord {
    fn apply_one(&self, eg: &mut G, eclass: Id, subst: &Subst, _: Option<&PatternAst<Node>>, _: Symbol) -> Vec<Id> {
        let Some(name) = sym_of(eg, subst[var("?n")]) else { return vec![] };
        let t = subst[var("?t")];
        let Some(ty) = ty_of(eg, t).cloned() else { return vec![] };
        let Some(names) = field_names(&eg.analysis.modules, &ty) else { return vec![] };
        if !names.iter().any(|m| m == name.as_str()) {
            return vec![];
        }
        let mut comps = Vec::with_capacity(names.len());
        for m in &names {
            if m == name.as_str() {
                comps.push(subst[var("?v")]);
            } else {
                let sym = eg.add(Node::Sym(Symbol::from(m.as_str())));
                let field = eg.add(Node::Field([sym, t]));
                if eg[field].data.ty.is_none() {
                    return vec![];
                }
                comps.push(eg.add(Node::ZerosLike(field)));
            }
        }
        add_record(eg, eclass, names, comps, ty)
    }
    fn vars(&self) -> Vec<Var> {
        vec![var("?n"), var("?t"), var("?v")]
    }
}

/// `(proj ?i ?t)` where `?t` is a tuple: its component.
struct ProjTuple;
impl Applier<Node, Ana> for ProjTuple {
    fn apply_one(&self, eg: &mut G, eclass: Id, subst: &Subst, _: Option<&PatternAst<Node>>, _: Symbol) -> Vec<Id> {
        let Some(i) = nat_of(eg, subst[var("?i")]) else { return vec![] };
        let comp = eg[subst[var("?t")]].nodes.iter().find_map(|n| if let Node::Tuple(ch) = n { ch.get(i as usize).copied() } else { None });
        match comp {
            Some(c) if eg.union(eclass, c) => vec![c],
            _ => vec![],
        }
    }
    fn vars(&self) -> Vec<Var> {
        vec![var("?i"), var("?t")]
    }
}

/// `(pow ?n (pow ?m ?x))` → `(pow n·m ?x)`.
struct PowPow;
impl Applier<Node, Ana> for PowPow {
    fn apply_one(&self, eg: &mut G, eclass: Id, subst: &Subst, _: Option<&PatternAst<Node>>, _: Symbol) -> Vec<Id> {
        let (Some(n), Some(m)) = (nat_of(eg, subst[var("?n")]), nat_of(eg, subst[var("?m")])) else { return vec![] };
        let Some(nm) = n.checked_mul(m) else { return vec![] };
        let k = eg.add(Node::Nat(nm));
        let id = eg.add(Node::Pow([k, subst[var("?x")]]));
        if eg.union(eclass, id) { vec![id] } else { vec![] }
    }
    fn vars(&self) -> Vec<Var> {
        vec![var("?n"), var("?m"), var("?x")]
    }
}

pub fn rules() -> Vec<Rw> {
    let mut rs: Vec<Rw> = vec![
        // ----- algebra -----
        rewrite!("add-comm"; "(add ?a ?b)" => "(add ?b ?a)" if compat("?a", "?b")),
        rewrite!("mul-comm"; "(mul ?a ?b)" => "(mul ?b ?a)" if compat("?a", "?b")),
        rewrite!("add-assoc"; "(add (add ?a ?b) ?c)" => "(add ?a (add ?b ?c))" if compat3("?a", "?b", "?c")),
        rewrite!("add-assoc-r"; "(add ?a (add ?b ?c))" => "(add (add ?a ?b) ?c)" if compat3("?a", "?b", "?c")),
        rewrite!("mul-assoc"; "(mul (mul ?a ?b) ?c)" => "(mul ?a (mul ?b ?c))" if compat3("?a", "?b", "?c")),
        rewrite!("mul-assoc-r"; "(mul ?a (mul ?b ?c))" => "(mul (mul ?a ?b) ?c)" if compat3("?a", "?b", "?c")),
        rewrite!("add-zero"; "(add ?x ?z)" => "?x" if both(konst("?z", Konst::Zero), tensorish("?x"))),
        rewrite!("zero-add"; "(add ?z ?x)" => "?x" if both(konst("?z", Konst::Zero), tensorish("?x"))),
        rewrite!("sub-zero"; "(sub ?x ?z)" => "?x" if both(konst("?z", Konst::Zero), tensorish("?x"))),
        rewrite!("sub-self"; "(sub ?x ?x)" => "(zeros_like ?x)" if tensorish("?x")),
        rewrite!("mul-one"; "(mul ?x ?o)" => "?x" if both(konst("?o", Konst::One), tensorish("?x"))),
        rewrite!("one-mul"; "(mul ?o ?x)" => "?x" if both(konst("?o", Konst::One), tensorish("?x"))),
        rewrite!("div-one"; "(div ?x ?o)" => "?x" if both(konst("?o", Konst::One), tensorish("?x"))),
        rewrite!("mul-zero"; "(mul ?x ?z)" => "(zeros_like ?x)" if both(konst("?z", Konst::Zero), tensorish("?x"))),
        rewrite!("zero-mul"; "(mul ?z ?x)" => "(zeros_like ?x)" if both(konst("?z", Konst::Zero), tensorish("?x"))),
        rewrite!("neg-neg"; "(neg (neg ?x))" => "?x"),
        rewrite!("neg-zeros"; "(neg (zeros_like ?x))" => "(zeros_like ?x)"),
        rewrite!("sub-as-add-neg"; "(sub ?a ?b)" => "(add ?a (neg ?b))" if compat("?a", "?b")),
        rewrite!("add-neg-as-sub"; "(add ?a (neg ?b))" => "(sub ?a ?b)" if compat("?a", "?b")),
        rewrite!("pow-one"; "(pow 1 ?x)" => "?x"),
        rewrite!("mul-self"; "(mul ?x ?x)" => "(pow 2 ?x)" if tensorish("?x")),
        rewrite!("log-exp"; "(log (exp ?x))" => "?x"),
        rewrite!("exp-add"; "(mul (exp ?a) (exp ?b))" => "(exp (add ?a ?b))" if compat("?a", "?b")),
        rewrite!("exp-add-r"; "(exp (add ?a ?b))" => "(mul (exp ?a) (exp ?b))" if compat("?a", "?b")),
        // On the domain of the left-hand side (x ≥ 0).
        rewrite!("sqrt-sq"; "(mul (sqrt ?x) (sqrt ?x))" => "?x"),
        rewrite!("sqrt-pow"; "(pow 2 (sqrt ?x))" => "?x"),
        rewrite!("distrib"; "(mul ?a (add ?b ?c))" => "(add (mul ?a ?b) (mul ?a ?c))" if compat3("?a", "?b", "?c")),
        rewrite!("factor"; "(add (mul ?a ?b) (mul ?a ?c))" => "(mul ?a (add ?b ?c))" if compat3("?a", "?b", "?c")),
        rewrite!("distrib-sub"; "(mul ?a (sub ?b ?c))" => "(sub (mul ?a ?b) (mul ?a ?c))" if compat3("?a", "?b", "?c")),
        rewrite!("factor-sub"; "(sub (mul ?a ?b) (mul ?a ?c))" => "(mul ?a (sub ?b ?c))" if compat3("?a", "?b", "?c")),
        rewrite!("mul-div"; "(mul ?a (div ?b ?c))" => "(div (mul ?a ?b) ?c)" if compat3("?a", "?b", "?c")),
        rewrite!("div-div"; "(div (div ?a ?b) ?c)" => "(div ?a (mul ?b ?c))" if compat3("?a", "?b", "?c")),
        // ----- reductions -----
        rewrite!("sum-scale"; "(sum (mul ?s ?x))" => "(mul ?s (sum ?x))" if both(scalar("?s"), real_tensor("?x"))),
        rewrite!("sum-scale-r"; "(sum (mul ?x ?s))" => "(mul ?s (sum ?x))" if both(scalar("?s"), real_tensor("?x"))),
        rewrite!("mean-scale"; "(mean (mul ?s ?x))" => "(mul ?s (mean ?x))" if both(scalar("?s"), real_tensor("?x"))),
        rewrite!("mean-scale-r"; "(mean (mul ?x ?s))" => "(mul ?s (mean ?x))" if both(scalar("?s"), real_tensor("?x"))),
        rewrite!("sum-neg"; "(sum (neg ?x))" => "(neg (sum ?x))" if real_tensor("?x")),
        rewrite!("mean-neg"; "(mean (neg ?x))" => "(neg (mean ?x))" if real_tensor("?x")),
        // ----- structure -----
        rewrite!("sum_to-id"; "(sum_to ?x ?t)" => "?x" if same_ty("?x", "?t")),
        rewrite!("broadcast_like-id"; "(broadcast_like ?x ?t)" => "?x" if same_ty("?x", "?t")),
        rewrite!("sum_to-zeros"; "(sum_to (zeros_like ?x) ?t)" => "(zeros_like ?t)"),
        rewrite!("add-zeros_like"; "(add ?x (zeros_like ?t))" => "?x" if same_ty("?x", "?t")),
        rewrite!("zeros_like-add"; "(add (zeros_like ?t) ?x)" => "?x" if same_ty("?x", "?t")),
        rewrite!("sub-zeros_like"; "(sub ?x (zeros_like ?t))" => "?x" if same_ty("?x", "?t")),
        rewrite!("field-set-same"; "(field ?n (set_field ?n ?r ?v))" => "?v"),
        rewrite!("field-set-other"; "(field ?n (set_field ?m ?r ?v))" => "(field ?n ?r)" if distinct("?n", "?m")),
        rewrite!("field-zeros"; "(field ?n (zeros_like ?t))" => "(zeros_like (field ?n ?t))" if has_fields("?t")),
        rewrite!("set-set-same"; "(set_field ?n (set_field ?n ?r ?u) ?v)" => "(set_field ?n ?r ?v)"),
        rewrite!("add-set-set"; "(add (set_field ?n ?r ?a) (set_field ?n ?s ?b))" => "(set_field ?n (add ?r ?s) (add ?a ?b))" if same_ty("?r", "?s")),
        rewrite!("add-set"; "(add (set_field ?n ?r ?a) ?s)" => "(set_field ?n (add ?r ?s) (add ?a (field ?n ?s)))" if both(same_ty("?r", "?s"), has_fields("?s"))),
        rewrite!("set-add"; "(add ?s (set_field ?n ?r ?a))" => "(set_field ?n (add ?s ?r) (add (field ?n ?s) ?a))" if both(same_ty("?r", "?s"), has_fields("?s"))),
    ];
    rs.push(Rewrite::new("div-into-contraction", "(div ?x ?s)".parse::<egg::Pattern<Node>>().unwrap(), ScaleIntoContraction { divide: true }).unwrap());
    rs.push(Rewrite::new("mul-into-contraction", "(mul ?x ?s)".parse::<egg::Pattern<Node>>().unwrap(), ScaleIntoContraction { divide: false }).unwrap());
    rs.push(Rewrite::new("field-record", "(field ?n ?r)".parse::<egg::Pattern<Node>>().unwrap(), FieldRecord).unwrap());
    rs.push(Rewrite::new("set-record", "(set_field ?n ?r ?v)".parse::<egg::Pattern<Node>>().unwrap(), SetRecord).unwrap());
    rs.push(Rewrite::new("set-zeros-record", "(set_field ?n (zeros_like ?t) ?v)".parse::<egg::Pattern<Node>>().unwrap(), SetZerosRecord).unwrap());
    rs.push(Rewrite::new("proj-tuple", "(proj ?i ?t)".parse::<egg::Pattern<Node>>().unwrap(), ProjTuple).unwrap());
    rs.push(Rewrite::new("pow-pow", "(pow ?n (pow ?m ?x))".parse::<egg::Pattern<Node>>().unwrap(), PowPow).unwrap());
    rs
}
