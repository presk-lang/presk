//! Elaboration: from the syntax trees of a file and its imports to a program
//! whose names are resolved, constants evaluated, `static assert`s
//! discharged, module definitions and instances known, and signature types
//! converted to shape-ready [`Ty`]s. Function bodies stay syntax trees; they
//! are typed in step 3.
//!
//! Error codes `E1xxx`, warning `W0001` (implicit dimension).

mod dims;
mod dump;
mod modules;
mod types;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use presk_syntax::ast::{self, ConfigItem, Program};
use presk_syntax::{Diagnostic, FileId, SourceMap, Span, parse_file};

use crate::dim::{DimExpr, DimVar};
use crate::presburger::Facts;
use crate::ty::{InstId, ModRef, ModuleId, ParamsTree, Ty};

pub use dump::dump;

pub type UnitId = usize;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FnId(pub u32);

#[derive(Clone, Debug)]
pub struct ConstDef {
    pub value: u64,
    pub span: Span,
    /// Set by `--const NAME=N`.
    pub overridden: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimOrigin {
    /// `dim NAME`
    Decl,
    /// `{NAME : dim}` of a module or function
    TyParam,
    /// An undeclared name used as a dimension in a function signature (W0001).
    Implicit,
}

#[derive(Clone, Debug)]
pub struct DimInfo {
    pub name: String,
    pub origin: DimOrigin,
    pub span: Span,
}

/// One source file.
pub struct Unit {
    pub file: FileId,
    pub path: PathBuf,
    pub imports: IndexMap<String, UnitId>,
    pub consts: IndexMap<String, ConstDef>,
    pub dims: IndexMap<String, DimVar>,
    pub types: IndexMap<String, Ty>,
    pub modules: IndexMap<String, ModuleId>,
    pub fns: IndexMap<String, FnId>,
    pub models: IndexMap<String, ModRef>,
    pub effects: IndexMap<String, Span>,
    pub optimizers: IndexMap<String, Vec<ConfigItem>>,
    pub training: Vec<(Vec<ConfigItem>, Span)>,
    pub inference: Vec<(Vec<ConfigItem>, Span)>,
    /// `static assert`s, with whether they were discharged.
    pub asserts: Vec<(Span, bool)>,
    pub ast: Program,
}

#[derive(Clone, Debug)]
pub struct ParamDef {
    pub name: String,
    pub ty: Ty,
    pub init: Option<ast::Expr>,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct UseDef {
    pub name: String,
    pub module: ModRef,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct ModuleDef {
    pub name: String,
    pub unit: UnitId,
    pub typarams: Vec<DimVar>,
    pub params: Vec<ParamDef>,
    pub uses: Vec<UseDef>,
    pub fns: Vec<FnId>,
    pub span: Span,
}

/// A module applied to dimension arguments; interned, so equal instances
/// share an id.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ModuleInst {
    pub def: ModuleId,
    pub args: Vec<DimExpr>,
}

#[derive(Clone, Debug)]
pub struct ParamSig {
    pub name: String,
    pub ty: Option<Ty>,
    pub default: Option<ast::Expr>,
    pub axis: bool,
    pub span: Span,
}

#[derive(Clone, Debug)]
pub struct FnSig {
    pub name: String,
    pub unit: UnitId,
    /// The module this function belongs to, for members.
    pub owner: Option<ModuleId>,
    pub typarams: Vec<DimVar>,
    /// Dimensions introduced implicitly by the signature (W0001).
    pub implicit_dims: Vec<DimVar>,
    pub params: Vec<ParamSig>,
    pub ret: Option<Ty>,
    pub effects: Vec<String>,
    pub body: ast::Expr,
    pub span: Span,
    /// The `...` of this signature, if any parameter or the result uses one.
    pub prefix: Option<crate::ty::PrefixVar>,
}

pub struct ElabProgram {
    /// Load order: imports come before the files that import them; the main
    /// file is last.
    pub units: Vec<Unit>,
    pub main: UnitId,
    pub modules: Vec<ModuleDef>,
    pub insts: Vec<ModuleInst>,
    pub fns: Vec<FnSig>,
    pub dims: Vec<DimInfo>,
    /// Bounds of `dim` declarations and discharged symbolic assertions.
    pub facts: Facts,
    /// Number of prefix variables allocated (`...` of signatures).
    pub prefixes: u32,
}

impl ElabProgram {
    pub fn dim_name(&self, v: DimVar) -> String {
        self.dims.get(v.0 as usize).map(|d| d.name.clone()).unwrap_or_else(|| format!("?{}", v.0))
    }

    pub fn inst_name(&self, i: InstId) -> String {
        let inst = &self.insts[i.0 as usize];
        ModRef::Inst { def: inst.def, args: inst.args.clone() }.display(&self.names())
    }

    pub fn names(&self) -> crate::ty::Names<'_> {
        crate::ty::Names { dim: Box::new(|v| self.dim_name(v)), module: Box::new(|m| self.modules[m.0 as usize].name.clone()) }
    }

    pub fn inst(&self, id: InstId) -> &ModuleInst {
        &self.insts[id.0 as usize]
    }

    pub fn module(&self, id: ModuleId) -> &ModuleDef {
        &self.modules[id.0 as usize]
    }

    pub fn function(&self, id: FnId) -> &FnSig {
        &self.fns[id.0 as usize]
    }

    /// The substitution of a module's type parameters by an instance's arguments.
    pub fn inst_subst(&self, def: ModuleId, args: &[DimExpr]) -> BTreeMap<DimVar, DimExpr> {
        self.module(def).typarams.iter().copied().zip(args.iter().cloned()).collect()
    }

    /// The parameter tree denoted by `params M`.
    pub fn params_tree(&self, m: &ModRef) -> ParamsTree {
        self.params_tree_under(m, &BTreeMap::new())
    }

    /// `params_tree` with an outer substitution applied to the instance
    /// arguments (a `use l1 : Linear{D, hidden}` inside `MLP{D hidden : dim}`
    /// is instantiated with MLP's own arguments).
    pub fn params_tree_under(&self, m: &ModRef, outer: &BTreeMap<DimVar, DimExpr>) -> ParamsTree {
        match m {
            ModRef::Inst { def, args } => {
                let def = self.module(*def);
                let map: BTreeMap<DimVar, DimExpr> = def.typarams.iter().copied().zip(args.iter().map(|a| a.subst(outer))).collect();
                let mut fields = Vec::new();
                for p in &def.params {
                    fields.push((p.name.clone(), ParamsTree::Leaf(p.ty.subst(&map))));
                }
                for u in &def.uses {
                    fields.push((u.name.clone(), self.params_tree_under(&u.module, &map)));
                }
                ParamsTree::Node(fields)
            }
            ModRef::Stack { n, inner } => ParamsTree::Stacked { n: n.subst(outer), inner: Box::new(self.params_tree_under(inner, outer)) },
            ModRef::Compose(parts) => ParamsTree::Node(parts.iter().enumerate().map(|(i, p)| (format!("_{i}"), self.params_tree_under(p, outer))).collect()),
        }
    }
}

/// Options of an elaboration.
#[derive(Default)]
pub struct Options {
    /// `--const NAME=N` overrides, applied to the main file.
    pub const_overrides: Vec<(String, u64)>,
}

/// Elaborate `file` (already in the source map) and everything it imports.
/// Diagnostics of all files are returned; the program is returned even with
/// errors, with the failing declarations omitted.
pub fn elaborate(sm: &mut SourceMap, file: FileId, main_ast: Program, opts: &Options) -> (ElabProgram, Vec<Diagnostic>) {
    let mut e = Elaborator {
        sm,
        prog: ElabProgram { units: vec![], main: 0, modules: vec![], insts: vec![], fns: vec![], dims: vec![], facts: Facts::default(), prefixes: 0 },
        diags: vec![],
        loading: vec![],
        opts,
    };
    let path = PathBuf::from(e.sm.file(file).name.clone());
    let main = e.unit(file, path, main_ast, true);
    e.prog.main = main;
    (e.prog, e.diags)
}

pub(crate) struct Elaborator<'a> {
    pub(crate) sm: &'a mut SourceMap,
    pub(crate) prog: ElabProgram,
    pub(crate) diags: Vec<Diagnostic>,
    /// Files being elaborated, for cycle detection.
    loading: Vec<PathBuf>,
    opts: &'a Options,
}

/// Where a name in a dimension or type position is looked up.
pub(crate) struct Scope<'s> {
    pub(crate) unit: UnitId,
    /// Innermost first: function then module type parameters and module-typed
    /// parameters (`net : MLP{…}` bound for `params net`).
    pub(crate) locals: Vec<&'s Locals>,
}

#[derive(Default)]
pub(crate) struct Locals {
    pub(crate) dims: IndexMap<String, DimVar>,
    pub(crate) modules: IndexMap<String, ModRef>,
    /// The `...` of the enclosing signature.
    pub(crate) prefix: Option<crate::ty::PrefixVar>,
}

impl<'a> Elaborator<'a> {
    pub(crate) fn error(&mut self, d: Diagnostic) {
        self.diags.push(d);
    }

    pub(crate) fn new_dim(&mut self, name: &str, origin: DimOrigin, span: Span) -> DimVar {
        let v = DimVar(self.prog.dims.len() as u32);
        self.prog.dims.push(DimInfo { name: name.to_string(), origin, span });
        v
    }

    pub(crate) fn intern_inst(&mut self, inst: ModuleInst) -> InstId {
        if let Some(i) = self.prog.insts.iter().position(|x| *x == inst) {
            return InstId(i as u32);
        }
        self.prog.insts.push(inst);
        InstId(self.prog.insts.len() as u32 - 1)
    }

    /// Elaborate one file into a unit and return its id.
    fn unit(&mut self, file: FileId, path: PathBuf, ast: Program, is_main: bool) -> UnitId {
        let canon = path.canonicalize().unwrap_or_else(|_| path.clone());
        self.loading.push(canon.clone());
        let mut unit = Unit {
            file,
            path: path.clone(),
            imports: IndexMap::new(),
            consts: IndexMap::new(),
            dims: IndexMap::new(),
            types: IndexMap::new(),
            modules: IndexMap::new(),
            fns: IndexMap::new(),
            models: IndexMap::new(),
            effects: IndexMap::new(),
            optimizers: IndexMap::new(),
            training: vec![],
            inference: vec![],
            asserts: vec![],
            ast,
        };
        // 1. imports
        let decls = unit.ast.decls.clone();
        for d in &decls {
            if let ast::Decl::Import { path: rel, alias, span } = d {
                if let Some(id) = self.import(&path, rel, *span)
                    && unit.imports.insert(alias.name.clone(), id).is_some()
                {
                    self.duplicate("import alias", alias);
                }
            }
        }
        // The unit must exist (with its imports) before the rest is resolved.
        let id = self.prog.units.len();
        self.prog.units.push(unit);
        self.declare_all(id, &decls, is_main);
        self.loading.pop();
        id
    }

    fn import(&mut self, from: &Path, rel: &str, span: Span) -> Option<UnitId> {
        let base = from.parent().map(Path::to_path_buf).unwrap_or_default();
        let target = base.join(rel);
        let canon = match target.canonicalize() {
            Ok(c) => c,
            Err(err) => {
                self.error(Diagnostic::error("E1002", format!("cannot import `{rel}`: {err}")).label(span, format!("resolved to `{}`", target.display())));
                return None;
            }
        };
        if self.loading.contains(&canon) {
            self.error(Diagnostic::error("E1002", format!("import cycle through `{rel}`")).label(span, "this file is already being imported"));
            return None;
        }
        if let Some(i) = self.prog.units.iter().position(|u| u.path.canonicalize().map(|c| c == canon).unwrap_or(false)) {
            return Some(i);
        }
        let file = match self.sm.load(&target) {
            Ok(f) => f,
            Err(err) => {
                self.error(Diagnostic::error("E1002", format!("cannot read `{rel}`: {err}")).label(span, "imported here"));
                return None;
            }
        };
        let (ast, diags) = parse_file(self.sm, file);
        let failed = diags.iter().any(|d| d.is_error());
        self.diags.extend(diags);
        if failed {
            return None;
        }
        Some(self.unit(file, target, ast, false))
    }

    pub(crate) fn duplicate(&mut self, what: &str, name: &ast::Ident) {
        self.error(Diagnostic::error("E1008", format!("duplicate {what} `{}`", name.name)).label(name.span, "already defined in this file"));
    }

    /// Declarations of a unit, in dependency order: consts, dims, type
    /// aliases, effects, modules (registered first, then elaborated), fns,
    /// models, config blocks, and finally the static assertions.
    fn declare_all(&mut self, id: UnitId, decls: &[ast::Decl], is_main: bool) {
        for d in decls {
            match d {
                ast::Decl::Const { name, ty, value, span } => self.const_decl(id, name, ty.as_ref(), value, *span, is_main),
                ast::Decl::Dim { name, bound, span } => self.dim_decl(id, name, bound.as_ref(), *span),
                _ => {}
            }
        }
        if is_main {
            for (name, _) in &self.opts.const_overrides {
                if !self.prog.units.iter().any(|u| u.consts.contains_key(name)) {
                    self.error(Diagnostic::error("E1011", format!("`--const {name}` names no `const` of this file or its imports")));
                }
            }
        }
        for d in decls {
            match d {
                ast::Decl::TypeAlias { name, ty, span: _ } => {
                    let scope = Scope { unit: id, locals: vec![] };
                    if let Some(t) = self.ty(&scope, ty) {
                        if self.prog.units[id].types.insert(name.name.clone(), t).is_some() {
                            self.duplicate("type", name);
                        }
                    }
                }
                ast::Decl::Effect { name, span } => {
                    if self.prog.units[id].effects.insert(name.name.clone(), *span).is_some() {
                        self.duplicate("effect", name);
                    }
                }
                _ => {}
            }
        }
        // Register module names before elaborating bodies (forward references in `use`).
        for d in decls {
            if let ast::Decl::Module(m) = d {
                let mid = ModuleId(self.prog.modules.len() as u32);
                self.prog.modules.push(ModuleDef { name: m.name.name.clone(), unit: id, typarams: vec![], params: vec![], uses: vec![], fns: vec![], span: m.span });
                if self.prog.units[id].modules.insert(m.name.name.clone(), mid).is_some() {
                    self.duplicate("module", &m.name);
                }
            }
        }
        for d in decls {
            if let ast::Decl::Module(m) = d {
                let mid = self.prog.units[id].modules[&m.name.name];
                self.module_decl(id, mid, m);
            }
        }
        for d in decls {
            if let ast::Decl::Fn(f) = d {
                let fid = self.fn_decl(id, None, &Locals::default(), f);
                if self.prog.units[id].fns.insert(f.name.name.clone(), fid).is_some() {
                    self.duplicate("function", &f.name);
                }
            }
        }
        for d in decls {
            match d {
                ast::Decl::Model { name, body, .. } => {
                    let scope = Scope { unit: id, locals: vec![] };
                    if let Some(m) = self.mod_expr(&scope, body) {
                        if self.prog.units[id].models.insert(name.name.clone(), m).is_some() {
                            self.duplicate("model", name);
                        }
                    }
                }
                ast::Decl::Optimizer { name, items, .. } => {
                    if self.prog.units[id].optimizers.insert(name.name.clone(), items.clone()).is_some() {
                        self.duplicate("optimizer", name);
                    }
                }
                ast::Decl::Training { items, span } => self.prog.units[id].training.push((items.clone(), *span)),
                ast::Decl::Inference { items, span } => self.prog.units[id].inference.push((items.clone(), *span)),
                _ => {}
            }
        }
        for d in decls {
            if let ast::Decl::StaticAssert { cond, span } = d {
                let ok = self.static_assert(id, cond, *span);
                self.prog.units[id].asserts.push((*span, ok));
            }
        }
    }
}
