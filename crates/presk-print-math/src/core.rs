//! The second entry: PSK-Core after differentiation (plan 0.13.0 step 4b).
//!
//! Each function is a system of equations in the order of its bindings,
//! one per binding, named as `presk dump --stage core` names them (the
//! source's name when the binding has one, else `%n`), each primitive in
//! the notation of the reference's chapter 3: the arithmetic as arithmetic,
//! the other primitives, those that differentiation writes among them, by
//! their names (`\code{dot\_vjp\_r}\ x, c`). A `fold` is its recurrence.
//! The bindings of the functions differentiation generated carry what the
//! provenance table says they were made for: the forward pass's copy of a
//! binding, or a contribution to an adjoint by the rule of its primitive
//! (the reference's table `core-adjoint`, which [`adjoint_rule`] writes).

use std::collections::{BTreeMap, HashMap, HashSet};

use presk_core::ir::{Func, Let, Lit, PrimOp, Program, Rhs, VarId};
use presk_provenance::{How, Link, Stage};
use presk_types::dim::{DimExpr, rat};
use presk_types::elab::ElabProgram;
use presk_types::ty::{Shape, Ty};

use crate::m::{B, F, Format, M, R, T, escape};
use crate::render::{PREAMBLE, Select};
use crate::typed::Equation;

/// What a binding of a generated function was made for.
#[derive(Clone, Debug)]
pub struct Note {
    /// `adjoint`, `primal`, `accumulate`, `zero`, `seed`.
    pub what: String,
    /// The primal binding it serves, by its name.
    pub of: Option<String>,
    /// Or the function whose parameter it serves.
    pub of_fn: Option<String>,
    /// The rule: the primitive's name, `call`, `fold`, or a primal's mode.
    pub rule: Option<String>,
}

/// One function of PSK-Core as equations.
#[derive(Clone, Debug)]
pub struct CoreDisplay {
    pub name: String,
    /// `generated (bwd) from loss`, from the table.
    pub origin: Option<(String, String)>,
    pub dims: Vec<M>,
    pub of: Vec<M>,
    pub effects: Vec<String>,
    /// The equations, each with its site's variable (none for a fold's
    /// steps and the result) and its note.
    pub eqs: Vec<(Equation, Option<VarId>, Option<Note>)>,
}

/// Every function of a program, in its order.
pub fn core_program(prog: &Program, elab: &ElabProgram, links: &[Link]) -> Vec<CoreDisplay> {
    let made: HashMap<&str, &Link> = links.iter().filter(|l| l.child.stage == Stage::Ad).map(|l| (l.child.key.as_str(), l)).collect();
    prog.funcs.iter().map(|f| function(f, prog, elab, &made)).collect()
}

fn function(f: &Func, prog: &Program, elab: &ElabProgram, made: &HashMap<&str, &Link>) -> CoreDisplay {
    let origin = made.get(f.name.as_str()).and_then(|l| match &l.how {
        How::Generated { kind } => Some((kind.clone(), l.parents.first().map(|p| p.key.clone()).unwrap_or_default())),
        _ => None,
    });
    let mut c = Ctx { f, prog, elab, used: HashSet::new() };
    c.used.extend(f.params.iter().map(|p| p.name.clone()));
    c.used.extend(f.body.lets.iter().filter_map(|l| l.name.clone()));
    let mut eqs = Vec::new();
    for l in &f.body.lets {
        let note = made.get(format!("{}/%{}", f.name, l.var.0).as_str()).and_then(|link| note(link, prog));
        let mut steps = c.equation(l);
        let last = steps.pop().expect("an equation per binding");
        for s in steps {
            eqs.push((s, None, None));
        }
        eqs.push((last, Some(l.var), note));
    }
    let args = f.params.iter().map(|p| M::name(&p.name)).collect();
    eqs.push((Equation { lhs: M::apply(M::den(M::code(&f.name)), args), rhs: M::name(c.var(f.body.result)), cond: None }, None, None));
    CoreDisplay {
        name: f.name.clone(),
        origin,
        dims: f.dims.iter().map(|d| M::name(elab.dim_name(*d))).collect(),
        of: f.params.iter().map(|p| M::rel(R::Has, M::name(&p.name), ty(&p.ty, elab))).collect(),
        effects: f.effects.clone(),
        eqs,
    }
}

/// The note of a binding, from the link that made it.
fn note(l: &Link, prog: &Program) -> Option<Note> {
    let of = || {
        l.parents.first().and_then(|p| {
            let (fname, v) = p.key.rsplit_once("/%")?;
            let f = prog.funcs.iter().find(|f| f.name == fname)?;
            let v: u32 = v.parse().ok()?;
            Some(var_name(f, VarId(v)))
        })
    };
    // A parameter's adjoint names its function.
    let of_fn = l.parents.first().filter(|p| !p.key.contains("/%")).map(|p| p.key.clone());
    let n = |what: &str, rule: Option<String>| Some(Note { what: what.into(), of: of(), of_fn: of_fn.clone(), rule });
    match &l.how {
        How::Adjoint { rule } => n("adjoint", Some(rule.clone())),
        How::Primal { mode } => n("primal", Some(mode.clone())),
        How::Accumulate => n("accumulate", None),
        How::Zero => n("zero", None),
        How::Seed => Some(Note { what: "seed".into(), of: None, of_fn: None, rule: None }),
        _ => None,
    }
}

/// The name `presk dump --stage core` gives a variable.
pub fn var_name(f: &Func, v: VarId) -> String {
    if let Some(p) = f.params.iter().find(|p| p.var == v) {
        return p.name.clone();
    }
    if let Some(n) = f.body.lets.iter().find(|l| l.var == v).and_then(|l| l.name.clone()) {
        return n;
    }
    format!("%{}", v.0)
}

/// A dimension expression.
pub fn dim(e: &DimExpr, elab: &ElabProgram) -> M {
    if let Some(n) = e.as_nat() {
        return M::num(n.to_string());
    }
    let mut acc: Option<M> = None;
    for (v, c) in &e.coeffs {
        let n = M::name(elab.dim_name(*v));
        let neg = *c < rat(0);
        let a = if neg { -c.clone() } else { c.clone() };
        let term = if a.is_integer() {
            if a == rat(1) { n } else { M::bin(B::DimMul, M::num(a.numer().to_string()), n) }
        } else {
            let num = if a.numer() == &1.into() { n } else { M::bin(B::DimMul, M::num(a.numer().to_string()), n) };
            M::bin(B::Div, num, M::num(a.denom().to_string()))
        };
        acc = Some(match (acc, neg) {
            (None, false) => term,
            (None, true) => M::Neg(Box::new(term)),
            (Some(x), false) => M::bin(B::Add, x, term),
            (Some(x), true) => M::bin(B::Sub, x, term),
        });
    }
    let k = &e.constant;
    if *k != rat(0) {
        let neg = *k < rat(0);
        let t = M::num(if neg { (-k.clone()).to_string() } else { k.to_string() });
        acc = Some(match (acc, neg) {
            (None, false) => t,
            (None, true) => M::Neg(Box::new(t)),
            (Some(x), false) => M::bin(B::Add, x, t),
            (Some(x), true) => M::bin(B::Sub, x, t),
        });
    }
    acc.unwrap_or_else(|| M::num("0"))
}

/// A polynomial in the dimensions (plan 0.15.0 step 3): its terms, each a
/// rational coefficient and the dimensions' names with their powers, the
/// highest degree first as given.
pub fn poly(terms: &[(presk_types::dim::Rational, Vec<(String, u32)>)]) -> M {
    let mut acc: Option<M> = None;
    for (c, vars) in terms {
        if *c == rat(0) {
            continue;
        }
        let neg = *c < rat(0);
        let a = if neg { -c.clone() } else { c.clone() };
        let factors: Vec<M> = vars.iter().filter(|(_, e)| *e > 0).map(|(n, e)| if *e == 1 { M::name(n.clone()) } else { M::Pow(Box::new(M::name(n.clone())), Box::new(M::num(e.to_string()))) }).collect();
        // The coefficient first, then the factors, left to right.
        let mut parts: Vec<M> = Vec::new();
        if a.numer() != &1.into() || factors.is_empty() {
            parts.push(M::num(a.numer().to_string()));
        }
        parts.extend(factors);
        let mut term = parts.remove(0);
        for f in parts {
            term = M::bin(B::DimMul, term, f);
        }
        if !a.is_integer() {
            term = M::bin(B::Div, term, M::num(a.denom().to_string()));
        }
        acc = Some(match (acc, neg) {
            (None, false) => term,
            (None, true) => M::Neg(Box::new(term)),
            (Some(x), false) => M::bin(B::Add, x, term),
            (Some(x), true) => M::bin(B::Sub, x, term),
        });
    }
    acc.unwrap_or_else(|| M::num("0"))
}

fn shape(s: &Shape, elab: &ElabProgram) -> Vec<M> {
    let mut v = Vec::new();
    if s.prefix.is_some() {
        v.push(M::Ellipsis);
    }
    for a in &s.axes {
        let len = dim(&a.len, elab);
        v.push(match &a.name {
            Some(n) => M::AxisLen(n.clone(), Box::new(len)),
            None => len,
        });
    }
    v
}

/// A type of PSK-Core.
pub fn ty(t: &Ty, elab: &ElabProgram) -> M {
    let n = elab.names();
    match t {
        Ty::Real(s) => M::Tensor(T::Real, None, shape(s, elab)),
        Ty::Idx { bound, shape: s } => M::Tensor(T::Idx, Some(Box::new(dim(bound, elab))), shape(s, elab)),
        Ty::Truth(s) => M::Tensor(T::Bool, None, shape(s, elab)),
        Ty::Tuple(ts) => M::Tuple(ts.iter().map(|t| ty(t, elab)).collect()),
        Ty::Params(r) => M::op("params", vec![], vec![M::code(r.display(&n))]),
        other => M::code(other.display(&n)),
    }
}

struct Ctx<'a> {
    f: &'a Func,
    prog: &'a Program,
    elab: &'a ElabProgram,
    used: HashSet<String>,
}

impl Ctx<'_> {
    fn var(&self, v: VarId) -> String {
        var_name(self.f, v)
    }

    fn v(&self, v: VarId) -> M {
        M::name(self.var(v))
    }

    fn ty_of(&self, v: VarId) -> Option<&Ty> {
        self.f.params.iter().find(|p| p.var == v).map(|p| &p.ty).or_else(|| self.f.body.lets.iter().find(|l| l.var == v).map(|l| &l.ty))
    }

    fn fresh(&mut self) -> String {
        let mut primes = String::new();
        loop {
            for c in ["y", "z", "u"] {
                let n = format!("{c}{primes}");
                if !self.used.contains(&n) {
                    self.used.insert(n.clone());
                    return n;
                }
            }
            primes.push('\'');
        }
    }

    fn callee(&self, f: presk_core::FuncId, dim_args: &BTreeMap<presk_types::dim::DimVar, DimExpr>) -> M {
        let name = M::code(&self.prog.func(f).name);
        if dim_args.is_empty() {
            name
        } else {
            M::Braced(Box::new(name), dim_args.iter().map(|(k, e)| M::rel(R::Eq, M::name(self.elab.dim_name(*k)), dim(e, self.elab))).collect())
        }
    }

    /// The equations of a binding: one, or a fold's recurrence and its
    /// value.
    fn equation(&mut self, l: &Let) -> Vec<Equation> {
        let lhs = self.v(l.var);
        let eq = |lhs: M, rhs: M| Equation { lhs, rhs, cond: None };
        let vs = |c: &Self, xs: &[VarId]| xs.iter().map(|x| c.v(*x)).collect::<Vec<_>>();
        match &l.rhs {
            Rhs::Lit(Lit::Nat(k)) => vec![eq(lhs, M::num(k.to_string()))],
            Rhs::Lit(Lit::Float(t)) => vec![eq(lhs, M::num(t))],
            Rhs::Prim { op: PrimOp::Ein { operands, output }, args } => {
                let terms: Vec<M> = output.iter().map(M::code).collect();
                let contracted: Vec<String> = {
                    let mut seen = Vec::new();
                    for o in operands {
                        for n in o {
                            if !output.contains(n) && !seen.contains(n) {
                                seen.push(n.clone());
                            }
                        }
                    }
                    seen
                };
                let factors: Vec<M> = operands.iter().zip(args).map(|(o, x)| M::index(self.v(*x), o.iter().map(M::code).collect())).collect();
                let prod = if factors.len() == 1 { factors.into_iter().next().unwrap() } else { M::Prod(factors) };
                let rhs = if contracted.is_empty() { prod } else { M::Sum(contracted.iter().map(M::code).collect(), Box::new(prod)) };
                vec![eq(M::index(lhs, terms), rhs)]
            }
            Rhs::Prim { op, args } => {
                let rhs = self.prim(op, args);
                vec![eq(lhs, rhs)]
            }
            Rhs::Call { f, dim_args, args } => vec![eq(lhs, M::apply(M::den(self.callee(*f, dim_args)), vs(self, args)))],
            Rhs::Tuple(xs) => vec![eq(lhs, M::Tuple(vs(self, xs)))],
            Rhs::Proj(x, i) => vec![eq(lhs, M::Field(Box::new(self.v(*x)), i.to_string()))],
            Rhs::Record(fs) => vec![eq(lhs, M::Record(fs.iter().map(|(n, x)| (n.clone(), self.v(*x))).collect()))],
            Rhs::Field(x, n) => vec![eq(lhs, M::Field(Box::new(self.v(*x)), n.clone()))],
            Rhs::Closure { f, captured } => {
                // `closure f[z₁, …]`: the function of the parameters it
                // leaves (chapter 3's value of a closure).
                let rest = self.prog.func(*f).params.len().saturating_sub(captured.len());
                let ps: Vec<M> = if rest == 1 { vec![M::name("v")] } else { (1..=rest).map(|k| M::subscript(M::name("v"), M::num(k.to_string()))).collect() };
                let mut a = vs(self, captured);
                a.extend(ps.iter().cloned());
                vec![eq(lhs, M::MapsTo(ps, Box::new(M::apply(M::den(M::code(&self.prog.func(*f).name)), a))))]
            }
            Rhs::Grad(x) => vec![eq(lhs, M::Nabla(Box::new(self.v(*x))))],
            Rhs::Apply { fun, args } => vec![eq(lhs, M::apply(self.v(*fun), vs(self, args)))],
            Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, scan } => {
                let y = self.fresh();
                let ell = M::name("ℓ");
                let count = dim(n, self.elab);
                let last = match n.as_nat() {
                    Some(k) => M::num(k.saturating_sub(1).to_string()),
                    None => M::bin(B::Sub, count.clone(), M::num("1")),
                };
                let at = if *reverse { M::bin(B::Sub, last.clone(), ell.clone()) } else { ell.clone() };
                let mut a = Vec::new();
                if *index {
                    a.push(at.clone());
                }
                a.extend(stacked.iter().map(|s| M::index(self.v(*s), vec![at.clone()])));
                a.push(M::subscript(M::name(&y), ell.clone()));
                a.extend(vs(self, extra));
                let range = M::rel(R::Lt, M::rel(R::Le, M::num("0"), ell.clone()), count.clone());
                let step = Equation { lhs: M::subscript(M::name(&y), M::bin(B::Add, ell, M::num("1"))), rhs: M::apply(M::den(self.callee(*body, dim_args)), a), cond: Some(range) };
                let end = M::subscript(M::name(&y), count.clone());
                let value = if *scan { M::Tuple(vec![end, M::Tuple(vec![M::subscript(M::name(&y), M::num("0")), M::Ellipsis, M::subscript(M::name(&y), last)])]) } else { end };
                vec![eq(M::subscript(M::name(&y), M::num("0")), self.v(*init)), step, eq(lhs, value)]
            }
        }
    }

    /// A primitive in chapter 3's notation.
    fn prim(&self, op: &PrimOp, args: &[VarId]) -> M {
        use PrimOp::*;
        let a: Vec<M> = args.iter().map(|x| self.v(*x)).collect();
        let x = |i: usize| a.get(i).cloned().unwrap_or(M::Ellipsis);
        let named = |n: &str| M::op(n, vec![], a.clone());
        let with = |n: &str, attrs: Vec<M>| M::op(n, vec![attrs], a.clone());
        match op {
            Add => M::bin(B::Add, x(0), x(1)),
            Sub => M::bin(B::Sub, x(0), x(1)),
            Mul => M::bin(B::Mul, x(0), x(1)),
            Div => M::bin(B::Div, x(0), x(1)),
            Neg => M::Neg(Box::new(x(0))),
            Pow(n) => M::Pow(Box::new(x(0)), Box::new(M::num(n.to_string()))),
            Exp => M::Fun(F::Exp, Box::new(x(0))),
            Log => M::Fun(F::Ln, Box::new(x(0))),
            Tanh => M::Fun(F::Tanh, Box::new(x(0))),
            Sin => M::Fun(F::Sin, Box::new(x(0))),
            Cos => M::Fun(F::Cos, Box::new(x(0))),
            Sqrt => M::Fun(F::Sqrt, Box::new(x(0))),
            Abs => M::Fun(F::Abs, Box::new(x(0))),
            Gelu => named("gelu"),
            Relu => named("relu"),
            Sigmoid => named("sigmoid"),
            // The injection of a hyperparameter is exact: the number
            // itself; of an index tensor, `real t`.
            Real => match args.first().and_then(|v| self.ty_of(*v)) {
                Some(t) if t.is_tensor() => named("real"),
                _ => x(0),
            },
            Pi => M::name("π"),
            Dot | Matmul => named("dot"),
            Ein { .. } => unreachable!("written at an index by the caller"),
            SumAll => named("sum"),
            MeanAll => named("mean"),
            Sum { axis } => with("sum axis", vec![M::code(axis)]),
            Mean { axis } => with("mean axis", vec![M::code(axis)]),
            Softmax { axis } => with("softmax axis", vec![M::code(axis)]),
            Embed => named("embed"),
            OneHot => named("one_hot"),
            CrossEntropy => named("cross_entropy"),
            Argmax => named("argmax"),
            Sample => named("sample"),
            IndexConst { axis, index } => with("index", vec![M::AxisLen(axis.clone(), Box::new(M::num(index.to_string())))]),
            MergeAxes { a: p, b: q } => with("merge", vec![M::code(p), M::code(q)]),
            SplitAxes { a: p, n1, l1, n2, l2 } => M::op("split", vec![vec![M::code(p)], vec![M::code("into")], vec![M::AxisLen(n1.clone(), Box::new(dim(l1, self.elab))), M::AxisLen(n2.clone(), Box::new(dim(l2, self.elab)))]], a.clone()),
            ReshapeLike => named("reshape_like"),
            SplitHeads { h, hd } => with("split_heads", vec![M::code(format!("{h}x{hd}"))]),
            MergeHeads { h, hd } => with("merge_heads", vec![M::code(format!("{h}x{hd}"))]),
            Positions { axis, len } => with("positions", vec![M::AxisLen(axis.clone(), Box::new(dim(len, self.elab)))]),
            CausalMask { axis, len } => with("causal_mask", vec![M::AxisLen(axis.clone(), Box::new(dim(len, self.elab)))]),
            Zeros { .. } => named("zeros"),
            Ones { .. } => named("ones"),
            Normal { std, .. } => with("normal", vec![M::code(format!("std={std}"))]),
            XavierUniform { .. } => named("xavier_uniform"),
            DropoutMask => named("dropout_mask"),
            SplitKey { n } => with("split_key", vec![M::num(n.to_string())]),
            SliceParams { index } => with("slice_params", vec![M::num(index.to_string())]),
            StackParams { n } => with("stack_params", vec![M::num(n.to_string())]),
            ZerosLike => named("zeros_like"),
            SumSquares => named("sum_squares"),
            SetField { name } => with("set_field", vec![M::code(name)]),
            SetElem { index } => with("set_elem", vec![M::num(index.to_string())]),
            SumTo => named("sum_to"),
            BroadcastLike => named("broadcast_like"),
            MeanVjp { axis: None } => named("mean_vjp"),
            MeanVjp { axis: Some(ax) } => with("mean_vjp axis", vec![M::code(ax)]),
            GeluDeriv => named("gelu'"),
            Heaviside => named("heaviside"),
            Sign => named("sign"),
            DotVjpL => named("dot_vjp_l"),
            DotVjpR => named("dot_vjp_r"),
            EmbedVjp => named("embed_vjp"),
            CrossEntropyVjp => named("cross_entropy_vjp"),
            ScatterIndex { axis, index } => with("scatter_index", vec![M::AxisLen(axis.clone(), Box::new(M::num(index.to_string())))]),
            ScatterParams { index } => with("scatter_params", vec![M::num(index.to_string())]),
            SetSlice => named("set_slice"),
        }
    }
}

/// The equation of one binding, for the page: a fold's recurrence as its
/// value's equation only.
pub fn equation(f: &Func, l: &Let, prog: &Program, elab: &ElabProgram) -> Equation {
    let mut c = Ctx { f, prog, elab, used: HashSet::new() };
    c.used.extend(f.params.iter().map(|p| p.name.clone()));
    c.used.extend(f.body.lets.iter().filter_map(|l| l.name.clone()));
    c.equation(l).pop().expect("an equation per binding")
}

/// The contributions the rule of a primitive writes, as the reference's
/// table `core-adjoint` writes them: `y` the binding, `c` its cotangent.
pub fn adjoint_rule(rule: &str) -> Option<M> {
    let x = |n: &str| M::name(n);
    let c = || M::name("c");
    // `x += r`, as the table writes it.
    let inc = |to: M, r: M| M::rel(R::AddTo, to, r);
    let op = |n: &str, args: Vec<M>| M::op(n, vec![], args);
    Some(match rule {
        "add" => M::Seq(vec![inc(x("x_1"), op("sum_to", vec![c(), x("x_1")])), inc(x("x_2"), op("sum_to", vec![c(), x("x_2")]))]),
        "sub" => M::Seq(vec![inc(x("x_1"), op("sum_to", vec![c(), x("x_1")])), inc(x("x_2"), op("sum_to", vec![op("neg", vec![c()]), x("x_2")]))]),
        "mul" => M::Seq(vec![inc(x("x_1"), op("sum_to", vec![op("mul", vec![c(), x("x_2")]), x("x_1")])), inc(x("x_2"), op("sum_to", vec![op("mul", vec![c(), x("x_1")]), x("x_2")]))]),
        "div" => M::Prod(vec![
            inc(x("x_1"), op("sum_to", vec![op("div", vec![c(), x("x_2")]), x("x_1")])),
            inc(x("x_2"), op("sum_to", vec![op("neg", vec![op("div", vec![op("mul", vec![c(), x("y")]), x("x_2")])]), x("x_2")])),
        ]),
        "neg" => inc(x("x"), op("neg", vec![c()])),
        "pow" => inc(x("x"), op("mul", vec![c(), op("mul", vec![op("real", vec![x("n")]), M::op("pow", vec![vec![M::bin(B::Sub, x("n"), M::num("1"))]], vec![x("x")])])])),
        "exp" => inc(x("x"), op("mul", vec![c(), x("y")])),
        "log" => inc(x("x"), op("div", vec![c(), x("x")])),
        "sqrt" => inc(x("x"), op("div", vec![c(), op("mul", vec![M::code("2"), x("y")])])),
        "tanh" => inc(x("x"), op("mul", vec![c(), op("sub", vec![M::code("1"), op("mul", vec![x("y"), x("y")])])])),
        "sigmoid" => inc(x("x"), op("mul", vec![c(), op("mul", vec![x("y"), op("sub", vec![M::code("1"), x("y")])])])),
        "sin" => inc(x("x"), op("mul", vec![c(), op("cos", vec![x("x")])])),
        "cos" => inc(x("x"), op("mul", vec![c(), op("neg", vec![op("sin", vec![x("x")])])])),
        "gelu" => inc(x("x"), op("mul", vec![c(), op("gelu'", vec![x("x")])])),
        "relu" => inc(x("x"), op("mul", vec![c(), op("heaviside", vec![x("x")])])),
        "abs" => inc(x("x"), op("mul", vec![c(), op("sign", vec![x("x")])])),
        "dot" | "matmul" => M::Seq(vec![inc(x("x"), op("dot_vjp_l", vec![c(), x("w")])), inc(x("w"), op("dot_vjp_r", vec![x("x"), c()]))]),
        "sum" => inc(x("x"), op("broadcast_like", vec![c(), x("x")])),
        "mean" => inc(x("x"), op("mean_vjp", vec![c(), x("x")])),
        "softmax" => inc(x("x"), op("mul", vec![x("y"), op("sub", vec![c(), op("broadcast_like", vec![M::op("sum axis", vec![vec![x("a")]], vec![op("mul", vec![c(), x("y")])]), c()])])])),
        "embed" => inc(x("w"), op("embed_vjp", vec![x("w"), x("t"), c()])),
        "cross_entropy" => inc(x("l"), op("cross_entropy_vjp", vec![x("l"), x("t"), c()])),
        "index" => inc(x("x"), M::op("scatter_index", vec![vec![M::AxisLen("a".into(), Box::new(x("i")))]], vec![x("x"), c()])),
        "merge" | "split" | "reshape_like" => inc(x("x"), op("reshape_like", vec![c(), x("x")])),
        "split_heads" => inc(x("x"), M::op("merge_heads", vec![vec![M::Cat(vec![x("H"), M::code("x"), x("HD")])]], vec![c()])),
        "merge_heads" => inc(x("x"), M::op("split_heads", vec![vec![M::Cat(vec![x("H"), M::code("x"), x("HD")])]], vec![c()])),
        "slice_params" => inc(x("s"), M::op("scatter_params", vec![vec![x("i")]], vec![x("s"), c()])),
        "sum_to" => inc(x("x"), op("broadcast_like", vec![c(), x("x")])),
        "broadcast_like" => inc(x("x"), op("sum_to", vec![c(), x("x")])),
        "ein" => M::Raw {
            latex: r"x_j \mathrel{{+}{=}} \code{ein[}o'_j\code{](}\bar a \code{ * } o_1 \code{ * } \dots \code{ * } o_n\code{)}\ c, x_1, \dots, x_n \quad \text{for every } j\text{, without } o_j \text{ and } x_j".into(),
            text: "x_j += ein[o'_j](ā * o_1 * … * o_n) c, x_1, …, x_n   for every j, without o_j and x_j".into(),
        },
        "checkpoint" => inc(x("x"), c()),
        "stack_params" => M::Raw { latex: r"x_j \mathrel{{+}{=}} \code{slice\_params}\ (j-1)\ c \quad \text{for every } j".into(), text: "x_j += slice_params (j−1) c   for every j".into() },
        "set_field" => M::Seq(vec![inc(x("r"), M::op("set_field", vec![vec![x("n")]], vec![c(), op("zeros_like", vec![x("v")])])), inc(x("v"), M::Field(Box::new(c()), "n".into()))]),
        "set_elem" => M::Seq(vec![inc(x("t"), M::op("set_elem", vec![vec![x("i")]], vec![c(), op("zeros_like", vec![x("v")])])), inc(x("v"), M::Field(Box::new(c()), "i".into()))]),
        "mean_vjp" => inc(M::name("c'"), op("mean", vec![c()])),
        _ => return None,
    })
}

/// The listing of a program after differentiation.
pub fn listing(ds: &[CoreDisplay], f: Format, sel: &Select, main: &str, standalone: bool) -> String {
    let latex = f == Format::Latex;
    let mut out = String::new();
    if latex {
        if standalone {
            out.push_str(PREAMBLE);
        }
        out.push_str(&format!("% presk dump --stage core-after-ad --format latex: {main}, PSK-Core after differentiation,\n% one equation per binding, in the notation of the language reference's chapter 3.\n"));
    } else {
        out.push_str(&format!("-- presk dump --stage core-after-ad --format unicode: {main}, PSK-Core after differentiation,\n-- one equation per binding, in the notation of the language reference's chapter 3\n"));
    }
    for d in ds {
        if sel.function.as_deref().is_some_and(|n| n != d.name) || sel.module.as_deref().is_some_and(|m| !d.name.starts_with(&format!("{m}."))) {
            continue;
        }
        let join = |xs: &[M]| {
            if latex {
                xs.iter().map(|x| format!("${}$", x.render(f))).collect::<Vec<_>>().join(", ")
            } else {
                xs.iter().map(|x| x.render(f)).collect::<Vec<_>>().join(",  ")
            }
        };
        let origin = d.origin.as_ref().map(|(k, p)| if latex { format!(" Generated by differentiation (\\texttt{{{}}}) from ${}$.", escape(k), M::code(p).render(f)) } else { format!("  -- generated ({k}) from {p}") }).unwrap_or_default();
        let uses = if d.effects.is_empty() { String::new() } else { format!(", uses {}", d.effects.join(", ")) };
        if latex {
            out.push_str(&format!("\n\\paragraph{{Function ${}$.}}{origin}", M::code(&d.name).render(f)));
            if !d.of.is_empty() {
                out.push_str(&format!(" Of {}", join(&d.of)));
                if !d.dims.is_empty() {
                    out.push_str(&format!(", dimensions {}", join(&d.dims)));
                }
                out.push_str(&format!("{uses}."));
            }
            // `alignat*{2}`: the note follows its equation, where `align*`
            // would push it to the margin.
            out.push_str("\n\\begin{alignat*}{2}\n");
            let n = d.eqs.len();
            for (i, (e, _, note)) in d.eqs.iter().enumerate() {
                out.push_str(&format!("{} &= {}", e.lhs.render(f), e.rhs.render(f)));
                if let Some(c) = &e.cond {
                    out.push_str(&format!(" &\\qquad& {}", c.render(f)));
                } else if let Some(nt) = note {
                    out.push_str(&format!(" &\\qquad& {}", note_text(nt, f)));
                }
                out.push_str(if i + 1 < n { " \\\\\n" } else { "\n" });
            }
            out.push_str("\\end{alignat*}\n");
        } else {
            out.push_str(&format!("\n{}", d.name));
            if !d.of.is_empty() {
                out.push_str(&format!(", of {}", join(&d.of)));
            }
            if !d.dims.is_empty() {
                out.push_str(&format!("; dimensions {}", join(&d.dims)));
            }
            out.push_str(&format!("{uses}{origin}\n"));
            let lines: Vec<(String, Option<String>)> = d
                .eqs
                .iter()
                .map(|(e, _, note)| {
                    let s = format!("{} = {}", e.lhs.render(f), e.rhs.render(f));
                    let tail = e.cond.as_ref().map(|c| format!("({})", c.render(f))).or_else(|| note.as_ref().map(|n| format!("-- {}", note_text(n, f))));
                    (s, tail)
                })
                .collect();
            let width = lines.iter().map(|(s, _)| s.chars().count()).max().unwrap_or(0).min(60);
            for (s, tail) in lines {
                match tail {
                    Some(t) => out.push_str(&format!("    {s:<width$}    {t}\n")),
                    None => out.push_str(&format!("    {s}\n")),
                }
            }
        }
    }
    if latex && standalone {
        out.push_str("\n\\end{document}\n");
    }
    out
}

/// A note in words, the names in mathematics.
pub fn note_text(n: &Note, f: Format) -> String {
    let latex = f == Format::Latex;
    let name = |s: &str| if latex { format!("${}$", M::name(s).render(f)) } else { s.to_string() };
    let code = |s: &str| if latex { format!("\\texttt{{{}}}", escape(s)) } else { s.to_string() };
    let of = match (&n.of, &n.of_fn) {
        (Some(v), _) => name(v),
        (None, Some(f)) => format!("a parameter of {}", code(f)),
        (None, None) => String::new(),
    };
    let words = match n.what.as_str() {
        "adjoint" => format!("adjoint of {of} ({})", code(n.rule.as_deref().unwrap_or(""))),
        "primal" => format!("primal {of}, {}", n.rule.as_deref().unwrap_or("")),
        "accumulate" => format!("accumulates the adjoint of {of}"),
        "zero" => format!("zero adjoint of {of}"),
        "seed" => "the cotangent 1".to_string(),
        other => other.to_string(),
    };
    if latex { format!("\\text{{\\footnotesize {words}}}") } else { words }
}
