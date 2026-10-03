//! Module definitions and function signatures.

use presk_syntax::Diagnostic;
use presk_syntax::ast::{self, Member};

use super::{DimOrigin, Elaborator, FnId, FnSig, Locals, ModuleDef, ParamDef, ParamSig, Scope, UnitId, UseDef};
use crate::dim::DimVar;
use crate::ty::{ModuleId, Ty};

impl Elaborator<'_> {
    fn typarams(&mut self, tps: &[ast::TyParam]) -> (Vec<DimVar>, Locals) {
        let mut vars = Vec::new();
        let mut locals = Locals::default();
        for tp in tps {
            if tp.kind.name != "dim" {
                self.error(
                    Diagnostic::error("E1010", format!("kind `{}` is not supported", tp.kind.name))
                        .label(tp.kind.span, "only `dim` type parameters exist in 0.1.0")
                        .note("`Shape` parameters (`{S : Shape}`) are a specification feature not implemented yet"),
                );
                continue;
            }
            let v = self.new_dim(&tp.name.name, DimOrigin::TyParam, tp.name.span);
            if locals.dims.insert(tp.name.name.clone(), v).is_some() {
                self.duplicate("type parameter", &tp.name);
            }
            vars.push(v);
        }
        (vars, locals)
    }

    /// Elaborate a module body: type parameters, `param`s, `use`s, member fns.
    pub(crate) fn module_decl(&mut self, unit: UnitId, mid: ModuleId, m: &ast::ModuleDecl) {
        let (typarams, locals) = self.typarams(&m.typarams);
        self.prog.modules[mid.0 as usize].typarams = typarams;
        let mut params = Vec::new();
        let mut uses = Vec::new();
        let mut fns = Vec::new();
        let mut seen: Vec<&ast::Ident> = Vec::new();
        for member in &m.members {
            match member {
                Member::Param { names, ty, init, span } => {
                    let scope = Scope { unit, locals: vec![&locals] };
                    let Some(t) = self.ty(&scope, ty) else { continue };
                    if !matches!(t, Ty::Real(_)) {
                        self.error(Diagnostic::error("E1009", "a `param` must be a real tensor").label(ty.span(), format!("this type is `{}`", t.display(&self.prog.names()))));
                        continue;
                    }
                    for n in names {
                        if seen.iter().any(|s| s.name == n.name) {
                            self.duplicate("member", n);
                            continue;
                        }
                        seen.push(n);
                        params.push(ParamDef { name: n.name.clone(), ty: t.clone(), init: init.clone(), span: *span });
                    }
                }
                Member::Use { names, module, span } => {
                    let scope = Scope { unit, locals: vec![&locals] };
                    let Some(r) = self.mod_expr(&scope, module) else { continue };
                    for n in names {
                        if seen.iter().any(|s| s.name == n.name) {
                            self.duplicate("member", n);
                            continue;
                        }
                        seen.push(n);
                        uses.push(UseDef { name: n.name.clone(), module: r.clone(), span: *span });
                    }
                }
                Member::Fn(_) => {}
            }
        }
        // Member functions see the module's type parameters.
        for member in &m.members {
            if let Member::Fn(f) = member {
                if seen.iter().any(|s| s.name == f.name.name) {
                    self.duplicate("member", &f.name);
                    continue;
                }
                seen.push(&f.name);
                fns.push(self.fn_decl(unit, Some(mid), &locals, f));
            }
        }
        let def: &mut ModuleDef = &mut self.prog.modules[mid.0 as usize];
        def.params = params;
        def.uses = uses;
        def.fns = fns;
    }

    /// Elaborate a function signature; the body stays syntax.
    pub(crate) fn fn_decl(&mut self, unit: UnitId, owner: Option<ModuleId>, outer: &Locals, f: &ast::FnDecl) -> FnId {
        let (typarams, mut locals) = self.typarams(&f.typarams);
        let prefix = self.new_prefix();
        locals.prefix = Some(prefix);
        let mut params = Vec::new();
        for p in &f.params {
            let ty = match &p.ty {
                None => None,
                Some(t) => {
                    // Parameters see the type parameters, the module's, and
                    // the module-typed parameters declared before them.
                    let ty = {
                        let mut implicit_locals = Locals::default();
                        let scope = Scope { unit, locals: vec![&locals, outer] };
                        let r = self.ty_with(&scope, t, &mut Some(&mut implicit_locals));
                        // Merge the implicit dimensions found in this parameter type.
                        for (name, v) in implicit_locals.dims {
                            locals.dims.entry(name).or_insert(v);
                        }
                        r
                    };
                    if let Some(Ty::Module(m)) = &ty {
                        locals.modules.insert(p.name.name.clone(), m.clone());
                    }
                    ty
                }
            };
            params.push(ParamSig { name: p.name.name.clone(), ty, default: p.default.clone(), axis: p.axis, span: p.span });
        }
        let ret = match &f.ret {
            None => None,
            Some(t) => {
                let mut implicit_locals = Locals::default();
                let scope = Scope { unit, locals: vec![&locals, outer] };
                let r = self.ty_with(&scope, t, &mut Some(&mut implicit_locals));
                for (name, v) in implicit_locals.dims {
                    locals.dims.entry(name).or_insert(v);
                }
                r
            }
        };
        let implicit_dims: Vec<DimVar> = locals.dims.values().copied().filter(|v| self.prog.dims[v.0 as usize].origin == DimOrigin::Implicit).collect();
        let prefix = uses_prefix(&sig_types(&params, &ret), prefix);
        let sig = FnSig {
            name: f.name.name.clone(),
            unit,
            owner,
            typarams,
            implicit_dims,
            params,
            ret,
            effects: f.effects.iter().map(|e| e.name.clone()).collect(),
            body: f.body.clone(),
            span: f.span,
            prefix,
        };
        self.prog.fns.push(sig);
        FnId(self.prog.fns.len() as u32 - 1)
    }
}

fn sig_types<'a>(params: &'a [ParamSig], ret: &'a Option<Ty>) -> Vec<&'a Ty> {
    params.iter().filter_map(|p| p.ty.as_ref()).chain(ret.iter()).collect()
}

fn uses_prefix(tys: &[&Ty], p: crate::ty::PrefixVar) -> Option<crate::ty::PrefixVar> {
    fn has(t: &Ty, p: crate::ty::PrefixVar) -> bool {
        match t {
            Ty::Real(s) | Ty::Idx { shape: s, .. } | Ty::Truth(s) => s.prefix == Some(p),
            Ty::Tuple(ts) => ts.iter().any(|t| has(t, p)),
            Ty::Fn { params, ret } => params.iter().any(|t| has(t, p)) || has(ret, p),
            _ => false,
        }
    }
    tys.iter().any(|t| has(t, p)).then_some(p)
}
