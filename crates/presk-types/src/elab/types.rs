//! Syntactic types to shape-ready types; module expressions to module references.

use presk_syntax::Diagnostic;
use presk_syntax::ast::{self, AxisSpec, ModExpr};

use super::{Elaborator, Locals, ModuleInst, Scope};
use crate::ty::{Axis, ModRef, ModuleId, PrefixVar, Shape, Ty};

impl Elaborator<'_> {
    /// Convert a type. `implicit` is the function-level locals that receive
    /// implicit dimensions, or `None` where undeclared names are errors.
    pub(crate) fn ty(&mut self, scope: &Scope, t: &ast::Ty) -> Option<Ty> {
        self.ty_with(scope, t, &mut None)
    }

    pub(crate) fn ty_with(&mut self, scope: &Scope, t: &ast::Ty, implicit: &mut Option<&mut Locals>) -> Option<Ty> {
        match t {
            ast::Ty::Real { axes, .. } => Some(Ty::Real(self.shape(scope, axes, implicit)?)),
            ast::Ty::Idx { bound, axes, .. } => {
                let bound = self.dim_expr(scope, bound, implicit)?;
                Some(Ty::Idx { bound, shape: self.shape(scope, axes, implicit)? })
            }
            ast::Ty::Bool { axes, .. } => Some(Ty::Truth(self.shape(scope, axes, implicit)?)),
            // `param T` is a declaration qualifier; as a type it is `T`.
            ast::Ty::Param { inner, .. } => self.ty_with(scope, inner, implicit),
            ast::Ty::Params { module, span } => {
                let m = self.module_value(scope, module)?;
                let _ = span;
                Some(Ty::Params(m))
            }
            ast::Ty::Tuple { elems, .. } => {
                let mut ts = Vec::new();
                for e in elems {
                    ts.push(self.ty_with(scope, e, implicit)?);
                }
                Some(Ty::Tuple(ts))
            }
            ast::Ty::Named { path, args, span } => {
                if path.segs.len() == 1 && args.is_empty() {
                    match path.segs[0].name.as_str() {
                        "nat" => return Some(Ty::Nat),
                        "int" => return Some(Ty::Int),
                        "rat" => return Some(Ty::Rat),
                        "f64" => return Some(Ty::F64),
                        "bool" => return Some(Ty::Bool),
                        "key" => return Some(Ty::Key),
                        _ => {}
                    }
                    // A local module-typed parameter used as a type? Not a type.
                    if let Some(t) = self.lookup_alias(scope, path) {
                        return Some(t);
                    }
                }
                if let Some(mid) = self.lookup_module(scope, path) {
                    let m = self.instantiate(scope, mid, args, *span, implicit)?;
                    return Some(Ty::Module(m));
                }
                self.error(Diagnostic::error("E1001", format!("unresolved type `{}`", path.dotted())).label(path.span, "not a scalar type, a type alias or a module in scope"));
                None
            }
        }
    }

    fn lookup_alias(&self, scope: &Scope, path: &ast::Path) -> Option<Ty> {
        let unit = if path.segs.len() == 2 { *self.prog.units[scope.unit].imports.get(&path.segs[0].name)? } else { scope.unit };
        self.prog.units[unit].types.get(&path.segs.last()?.name).cloned()
    }

    pub(crate) fn lookup_module(&self, scope: &Scope, path: &ast::Path) -> Option<ModuleId> {
        let unit = if path.segs.len() == 2 { *self.prog.units[scope.unit].imports.get(&path.segs[0].name)? } else { scope.unit };
        self.prog.units[unit].modules.get(&path.segs.last()?.name).copied()
    }

    /// A module used as a value: a module-typed parameter in scope
    /// (`params net`) or a module definition without arguments (`params m.MiniGPT`).
    fn module_value(&mut self, scope: &Scope, path: &ast::Path) -> Option<ModRef> {
        if path.segs.len() == 1 {
            for l in &scope.locals {
                if let Some(m) = l.modules.get(&path.segs[0].name) {
                    return Some(m.clone());
                }
            }
        }
        if let Some(mid) = self.lookup_module(scope, path) {
            return self.instantiate(scope, mid, &[], path.span, &mut None);
        }
        self.error(Diagnostic::error("E1009", format!("`{}` is not a module", path.dotted())).label(path.span, "`params` expects a module or a module-typed parameter"));
        None
    }

    fn shape(&mut self, scope: &Scope, axes: &[AxisSpec], implicit: &mut Option<&mut Locals>) -> Option<Shape> {
        let mut shape = Shape::default();
        for (i, a) in axes.iter().enumerate() {
            match a {
                AxisSpec::Ellipsis { span } => {
                    if i != 0 {
                        self.error(Diagnostic::error("E1010", "`...` must be the first axis").label(*span, "only a leading prefix of axes may be left open"));
                        return None;
                    }
                    shape.prefix = Some(self.current_prefix(scope));
                }
                AxisSpec::Axis { name, dim } => {
                    let len = self.dim_expr(scope, dim, implicit)?;
                    shape.axes.push(Axis { name: name.as_ref().map(|n| n.name.clone()), len });
                }
            }
        }
        Some(shape)
    }

    /// `M{args}`: check the arity; the instance is also recorded for the dump.
    pub(crate) fn instantiate(&mut self, scope: &Scope, mid: ModuleId, args: &[ast::Expr], span: presk_syntax::Span, implicit: &mut Option<&mut Locals>) -> Option<ModRef> {
        let def = &self.prog.modules[mid.0 as usize];
        let (name, arity) = (def.name.clone(), def.typarams.len());
        if args.len() != arity {
            self.error(
                Diagnostic::error("E1003", format!("`{name}` takes {arity} dimension argument{}, {} given", if arity == 1 { "" } else { "s" }, args.len()))
                    .label(span, "wrong number of dimension arguments"),
            );
            return None;
        }
        let mut dims = Vec::new();
        for a in args {
            dims.push(self.dim_expr(scope, a, implicit)?);
        }
        self.intern_inst(ModuleInst { def: mid, args: dims.clone() });
        Some(ModRef::Inst { def: mid, args: dims })
    }

    /// `Linear{D, D}`, `ff.MLP{D, 4*D}`, `stack{L}(Block)`, `A ∘ B`.
    pub(crate) fn mod_expr(&mut self, scope: &Scope, m: &ModExpr) -> Option<ModRef> {
        match m {
            ModExpr::Compose { parts, .. } => {
                let mut out = Vec::new();
                for p in parts {
                    out.push(self.mod_expr(scope, p)?);
                }
                Some(ModRef::Compose(out))
            }
            ModExpr::App { path, dim_args, mod_args, span } => {
                if path.is_single("stack") {
                    if dim_args.len() != 1 || mod_args.len() != 1 {
                        self.error(Diagnostic::error("E1003", "`stack` takes one dimension `{N}` and one module `(M)`").label(*span, "expected `stack{N}(M)`"));
                        return None;
                    }
                    let n = self.dim_expr(scope, &dim_args[0], &mut None)?;
                    let inner = self.mod_expr(scope, &mod_args[0])?;
                    return Some(ModRef::Stack { n, inner: Box::new(inner) });
                }
                if !mod_args.is_empty() {
                    self.error(Diagnostic::error("E1009", format!("`{}` takes no module arguments", path.dotted())).label(*span, "only `stack{N}(M)` applies a module to a module"));
                    return None;
                }
                if path.segs.len() == 1 {
                    for l in &scope.locals {
                        if let Some(m) = l.modules.get(&path.segs[0].name) {
                            return Some(m.clone());
                        }
                    }
                }
                let Some(mid) = self.lookup_module(scope, path) else {
                    self.error(Diagnostic::error("E1001", format!("unresolved module `{}`", path.dotted())).label(path.span, "no module of this name in scope"));
                    return None;
                };
                self.instantiate(scope, mid, dim_args, *span, &mut None)
            }
        }
    }
}

impl Elaborator<'_> {
    /// The prefix variable of the signature being elaborated: every `...` of
    /// one signature is the same prefix. Outside a signature (type aliases,
    /// module parameters) each `...` gets its own.
    fn current_prefix(&mut self, scope: &Scope) -> PrefixVar {
        for l in &scope.locals {
            if let Some(p) = l.prefix {
                return p;
            }
        }
        self.new_prefix()
    }

    pub(crate) fn new_prefix(&mut self) -> PrefixVar {
        self.prog.prefixes += 1;
        PrefixVar(self.prog.prefixes - 1)
    }
}
