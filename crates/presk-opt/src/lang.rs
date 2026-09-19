//! The e-graph language: one node per PSK-Core right-hand side. The
//! operations the rewrite rules mention are explicit variants (so that rules
//! can be written as patterns); everything else — the other primitives,
//! calls, closures, folds, literals — is an *opaque* node whose header is
//! interned in a [`Table`] and referenced by key. Attributes that rules need
//! to see (a `pow` exponent, a field name, a projection index) are leaf
//! children rather than payload, so that patterns can bind them.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::str::FromStr;

use egg::{Id, Symbol, define_language};
use presk_core::ir::{Lit, Rhs};
use presk_core::{FuncId, PrimOp};
use presk_types::dim::{DimExpr, DimVar};

macro_rules! key {
    ($name:ident, $prefix:literal) => {
        /// A key into the [`Table`]; prints as a prefixed number so that the
        /// e-graph and its patterns stay textual.
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
        pub struct $name(pub u32);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}{}", $prefix, self.0)
            }
        }

        impl FromStr for $name {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, String> {
                s.strip_prefix($prefix).and_then(|n| n.parse().ok()).map($name).ok_or_else(|| format!("not a {} key: {s}", $prefix))
            }
        }
    };
}

key!(OpKey, "o");
key!(ParamKey, "p");
key!(NamesKey, "r");

// Variants (the macro takes no doc comments): `Nat` is a natural number — a
// `nat` literal, a `pow` exponent or a projection index; `Pow` is `(pow n x)`;
// `Field` is `(field name x)`; `SetField` is `(set_field name record value)`;
// `Proj` is `(proj i x)`; `Record` is `(record names x₁ … xₙ)` with a `Names`
// leaf first; `Param` is a function parameter; `Names` the interned field
// names of a record; `Sym` a field name; `Other` any other right-hand side,
// with the interned header and the argument children.
define_language! {
    pub enum Node {
        Nat(u64),
        "add" = Add([Id; 2]),
        "sub" = Sub([Id; 2]),
        "mul" = Mul([Id; 2]),
        "div" = Div([Id; 2]),
        "neg" = Neg(Id),
        "exp" = Exp(Id),
        "log" = Log(Id),
        "sqrt" = Sqrt(Id),
        "pow" = Pow([Id; 2]),
        "real" = Real(Id),
        "zeros_like" = ZerosLike(Id),
        "sum_to" = SumTo([Id; 2]),
        "broadcast_like" = BroadcastLike([Id; 2]),
        "sum" = SumAll(Id),
        "mean" = MeanAll(Id),
        "field" = Field([Id; 2]),
        "set_field" = SetField([Id; 3]),
        "proj" = Proj([Id; 2]),
        "tuple" = Tuple(Box<[Id]>),
        "record" = Record(Box<[Id]>),
        Param(ParamKey),
        Names(NamesKey),
        Sym(Symbol),
        Other(OpKey, Vec<Id>),
    }
}

/// The header of an opaque node.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Opaque {
    Lit(Lit),
    Prim(PrimOp),
    Call { f: FuncId, dim_args: BTreeMap<DimVar, DimExpr> },
    Closure(FuncId),
    /// Children: the function value, then the arguments.
    Apply,
    /// Children: the stacked values, the initial accumulator, the extra arguments.
    Fold { body: FuncId, dim_args: BTreeMap<DimVar, DimExpr>, n: DimExpr, stacked: usize, index: bool, reverse: bool, scan: bool },
}

/// The interned headers and field-name lists of one e-graph.
#[derive(Default, Debug)]
pub struct Table {
    ops: Vec<Opaque>,
    op_index: HashMap<Opaque, OpKey>,
    names: Vec<Vec<String>>,
    names_index: HashMap<Vec<String>, NamesKey>,
}

impl Table {
    pub fn op(&mut self, o: Opaque) -> OpKey {
        if let Some(k) = self.op_index.get(&o) {
            return *k;
        }
        let k = OpKey(self.ops.len() as u32);
        self.ops.push(o.clone());
        self.op_index.insert(o, k);
        k
    }

    pub fn get_op(&self, k: OpKey) -> &Opaque {
        &self.ops[k.0 as usize]
    }

    pub fn names(&mut self, n: Vec<String>) -> NamesKey {
        if let Some(k) = self.names_index.get(&n) {
            return *k;
        }
        let k = NamesKey(self.names.len() as u32);
        self.names.push(n.clone());
        self.names_index.insert(n, k);
        k
    }

    pub fn get_names(&self, k: NamesKey) -> &[String] {
        &self.names[k.0 as usize]
    }
}

/// The node of a right-hand side, given the class of every variable and a
/// way to add attribute leaves (exponents, field names, projection indices).
pub fn node_of(rhs: &Rhs, table: &mut Table, class: &dyn Fn(presk_core::VarId) -> Id, leaf: &mut dyn FnMut(Node) -> Id) -> Node {
    let ids = |xs: &[presk_core::VarId]| xs.iter().map(|v| class(*v)).collect::<Vec<Id>>();
    match rhs {
        Rhs::Lit(Lit::Nat(k)) => Node::Nat(*k),
        Rhs::Lit(l) => Node::Other(table.op(Opaque::Lit(l.clone())), vec![]),
        Rhs::Prim { op, args } => {
            let a = ids(args);
            match (op, a.as_slice()) {
                (PrimOp::Add, [x, y]) => Node::Add([*x, *y]),
                (PrimOp::Sub, [x, y]) => Node::Sub([*x, *y]),
                (PrimOp::Mul, [x, y]) => Node::Mul([*x, *y]),
                (PrimOp::Div, [x, y]) => Node::Div([*x, *y]),
                (PrimOp::Neg, [x]) => Node::Neg(*x),
                (PrimOp::Exp, [x]) => Node::Exp(*x),
                (PrimOp::Log, [x]) => Node::Log(*x),
                (PrimOp::Sqrt, [x]) => Node::Sqrt(*x),
                (PrimOp::Pow(n), [x]) => Node::Pow([leaf(Node::Nat(*n)), *x]),
                (PrimOp::Real, [x]) => Node::Real(*x),
                (PrimOp::ZerosLike, [x]) => Node::ZerosLike(*x),
                (PrimOp::SumTo, [x, t]) => Node::SumTo([*x, *t]),
                (PrimOp::BroadcastLike, [x, t]) => Node::BroadcastLike([*x, *t]),
                (PrimOp::SumAll, [x]) => Node::SumAll(*x),
                (PrimOp::MeanAll, [x]) => Node::MeanAll(*x),
                (PrimOp::SetField { name }, [r, v]) => Node::SetField([leaf(Node::Sym(Symbol::from(name.as_str()))), *r, *v]),
                _ => Node::Other(table.op(Opaque::Prim(op.clone())), a),
            }
        }
        Rhs::Call { f, dim_args, args } => Node::Other(table.op(Opaque::Call { f: *f, dim_args: dim_args.clone() }), ids(args)),
        Rhs::Tuple(xs) => Node::Tuple(ids(xs).into_boxed_slice()),
        Rhs::Proj(x, i) => Node::Proj([leaf(Node::Nat(*i as u64)), class(*x)]),
        Rhs::Record(fields) => {
            let key = table.names(fields.iter().map(|(n, _)| n.clone()).collect());
            let mut ch = vec![leaf(Node::Names(key))];
            ch.extend(fields.iter().map(|(_, v)| class(*v)));
            Node::Record(ch.into_boxed_slice())
        }
        Rhs::Field(x, name) => Node::Field([leaf(Node::Sym(Symbol::from(name.as_str()))), class(*x)]),
        Rhs::Closure { f, captured } => Node::Other(table.op(Opaque::Closure(*f)), ids(captured)),
        Rhs::Grad(_) => unreachable!("`grad` is eliminated by AD before optimization"),
        Rhs::Apply { fun, args } => {
            let mut ch = vec![class(*fun)];
            ch.extend(ids(args));
            Node::Other(table.op(Opaque::Apply), ch)
        }
        Rhs::Fold { body, dim_args, n, stacked, init, extra, index, reverse, scan } => {
            let mut ch = ids(stacked);
            ch.push(class(*init));
            ch.extend(ids(extra));
            Node::Other(table.op(Opaque::Fold { body: *body, dim_args: dim_args.clone(), n: n.clone(), stacked: stacked.len(), index: *index, reverse: *reverse, scan: *scan }), ch)
        }
    }
}
