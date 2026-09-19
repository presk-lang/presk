//! `M.init(key) -> params M`: the initializer of a module's parameter tree,
//! built from the `init = …` clauses and the sub-modules' initializers.

use std::collections::BTreeMap;

use presk_syntax::ast::{Arg, Expr, Lit};
use presk_syntax::{Diagnostic, Span};
use presk_types::dim::DimExpr;
use presk_types::ty::{ModRef, ModuleId, Ty};

use super::{Builder, Lowerer};
use crate::ir::{Block, Lit as CLit, Param, PrimOp, Rhs, VarId};

impl Lowerer<'_> {
    /// The dimensions an initializer is generic in: the module's type
    /// parameters and the `dim`s of its file (parameters may mention them).
    pub(crate) fn init_dims(&self, mid: ModuleId) -> Vec<presk_types::dim::DimVar> {
        let m = self.elab.module(mid);
        let mut dims: Vec<_> = m.typarams.clone();
        dims.extend(self.elab.units[m.unit].dims.values().copied());
        dims.sort();
        dims.dedup();
        dims
    }

    pub(crate) fn lower_init(&mut self, mid: ModuleId) {
        let m = self.elab.module(mid).clone();
        let core_id = self.init_ids[&mid];
        let mut b = Builder::new(core_id);
        let key = b.fresh();
        let self_ref = ModRef::Inst { def: mid, args: m.typarams.iter().map(|v| DimExpr::var(*v)).collect() };
        let n = m.params.len() + m.uses.len();
        let keys = self.split_keys(&mut b, key, n, m.span);
        let mut fields = Vec::new();
        let mut i = 0;
        for p in &m.params {
            let op = match &p.init {
                None => PrimOp::Zeros { ty: p.ty.clone() },
                Some(Expr::Var { path }) => match path.dotted().as_str() {
                    "zeros" => PrimOp::Zeros { ty: p.ty.clone() },
                    "ones" => PrimOp::Ones { ty: p.ty.clone() },
                    "xavier_uniform" => PrimOp::XavierUniform { ty: p.ty.clone() },
                    _ => PrimOp::Zeros { ty: p.ty.clone() },
                },
                Some(Expr::Call { args, .. }) => {
                    let std = match args.first() {
                        Some(Arg::Named { value: Expr::Lit { lit: Lit::Float(f), .. }, .. }) => f.clone(),
                        _ => "1.0".into(),
                    };
                    PrimOp::Normal { ty: p.ty.clone(), std }
                }
                Some(other) => {
                    self.error(Diagnostic::error("E5001", "unsupported initializer").label(other.span(), "not lowered"));
                    PrimOp::Zeros { ty: p.ty.clone() }
                }
            };
            let args = if op.uses_rand() { vec![keys[i]] } else { vec![] };
            let v = b.emit(Some(p.name.clone()), p.ty.clone(), Rhs::Prim { op, args }, p.span);
            fields.push((p.name.clone(), v));
            i += 1;
        }
        for u in &m.uses {
            let v = self.init_of(&mut b, &u.module, keys[i], u.span);
            fields.push((u.name.clone(), v));
            i += 1;
        }
        let ty = Ty::Params(self_ref);
        let result = b.emit(None, ty.clone(), Rhs::Record(fields), m.span);
        let dims = self.init_dims(mid);
        let func = &mut self.funcs[core_id.0 as usize];
        func.dims = dims;
        func.params = vec![Param { var: key, name: "key".into(), ty: Ty::Key }];
        func.ret = ty;
        func.effects = vec!["Rand".into()];
        func.body = Block { lets: b.lets, result };
    }

    fn split_keys(&mut self, b: &mut Builder, key: VarId, n: usize, span: Span) -> Vec<VarId> {
        if n == 0 {
            return vec![];
        }
        let tuple = b.emit(None, Ty::Tuple(vec![Ty::Key; n]), Rhs::Prim { op: PrimOp::SplitKey { n }, args: vec![key] }, span);
        (0..n).map(|i| b.emit(None, Ty::Key, Rhs::Proj(tuple, i), span)).collect()
    }

    /// The parameter tree of a module reference, initialized from `key`.
    pub(crate) fn init_of(&mut self, b: &mut Builder, r: &ModRef, key: VarId, span: Span) -> VarId {
        match r {
            ModRef::Inst { def, args } => {
                let f = self.init_ids[def];
                let mut dim_args: BTreeMap<_, _> = self.elab.module(*def).typarams.iter().copied().zip(args.iter().cloned()).collect();
                // The file's dimensions pass through unchanged.
                for d in self.init_dims(*def) {
                    dim_args.entry(d).or_insert_with(|| DimExpr::var(d));
                }
                b.emit(None, Ty::Params(r.clone()), Rhs::Call { f, dim_args, args: vec![key] }, span)
            }
            ModRef::Stack { n, inner } => {
                let Some(count) = n.as_nat() else {
                    self.error(Diagnostic::error("E5002", "`stack` with a symbolic count is not supported in 0.1.0").label(span, format!("the count `{}` must be a constant", n.display(&|v| self.elab.dim_name(v)))));
                    return b.emit(None, Ty::Params(r.clone()), Rhs::Lit(CLit::Nat(0)), span);
                };
                let keys = self.split_keys(b, key, count as usize, span);
                let parts: Vec<VarId> = keys.iter().map(|k| self.init_of(b, inner, *k, span)).collect();
                b.emit(None, Ty::Params(r.clone()), Rhs::Prim { op: PrimOp::StackParams { n: count }, args: parts }, span)
            }
            ModRef::Compose(parts) => {
                let keys = self.split_keys(b, key, parts.len(), span);
                let vs: Vec<(String, VarId)> = parts.iter().zip(keys).enumerate().map(|(i, (p, k))| (format!("_{i}"), self.init_of(b, p, k, span))).collect();
                b.emit(None, Ty::Params(r.clone()), Rhs::Record(vs), span)
            }
        }
    }
}
