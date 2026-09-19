//! Declarations: program, import, const, static assert, dim, type, module,
//! fn, effect, model; configuration blocks are in `config.rs`.

use super::{PResult, ParseError, Parser};
use crate::ast::{Decl, FnDecl, Ident, Member, ModExpr, ModuleDecl, Param, Path, TyParam};
use crate::token::{Kw, Op, TokKind};

impl Parser {
    pub(crate) fn program(&mut self) -> Vec<Decl> {
        let mut decls = Vec::new();
        loop {
            self.skip_newlines();
            if self.at(&TokKind::Eof) {
                break;
            }
            match self.decl() {
                Ok(d) => decls.push(d),
                Err(ParseError) => self.recover_decl(),
            }
        }
        decls
    }

    /// Skip to the next line that starts, at column 0, with a declaration keyword.
    fn recover_decl(&mut self) {
        loop {
            self.skip_newlines();
            let t = self.toks[self.pos].clone();
            if matches!(t.kind, TokKind::Eof) {
                return;
            }
            if self.column(t.span) == 0 && self.starts_decl(&t.kind) {
                return;
            }
            self.pos += 1;
            while !self.at_newline() && !matches!(self.toks[self.pos].kind, TokKind::Eof) {
                self.pos += 1;
            }
        }
    }

    fn starts_decl(&self, kind: &TokKind) -> bool {
        match kind {
            TokKind::Kw(k) => matches!(
                k,
                Kw::Import | Kw::Const | Kw::Static | Kw::Dim | Kw::Type | Kw::Module | Kw::Fn | Kw::Optimizer | Kw::Training | Kw::Inference | Kw::Effect
            ),
            TokKind::Ident(s) => s == "model",
            _ => false,
        }
    }

    fn decl(&mut self) -> PResult<Decl> {
        let start = self.peek().span;
        match self.peek_kind() {
            TokKind::Kw(Kw::Import) => {
                self.bump();
                let path = match self.peek_kind() {
                    TokKind::Str(s) => {
                        self.bump();
                        s
                    }
                    _ => return Err(self.unexpected("a file path in quotes")),
                };
                self.expect_kw(Kw::As)?;
                let alias = self.expect_ident("an alias for the imported file")?;
                self.end_of_line()?;
                Ok(Decl::Import { path, alias, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Const) => {
                self.bump();
                let name = self.expect_ident("the name of the constant")?;
                let ty = if self.eat_op(Op::Colon).is_some() { Some(self.ty()?) } else { None };
                self.expect_op(Op::Eq)?;
                let value = self.expr()?;
                self.end_of_line()?;
                Ok(Decl::Const { name, ty, value, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Static) => {
                self.bump();
                self.expect_kw(Kw::Assert)?;
                let cond = self.expr()?;
                self.end_of_line()?;
                Ok(Decl::StaticAssert { cond, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Dim) => {
                self.bump();
                let name = self.expect_ident("the name of the dimension")?;
                let bound = if self.eat_op(Op::Le).is_some() { Some(self.expr()?) } else { None };
                self.end_of_line()?;
                Ok(Decl::Dim { name, bound, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Type) => {
                self.bump();
                let name = self.expect_ident("the name of the type")?;
                self.expect_op(Op::Eq)?;
                let ty = self.ty()?;
                self.end_of_line()?;
                Ok(Decl::TypeAlias { name, ty, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Module) => Ok(Decl::Module(self.module()?)),
            TokKind::Kw(Kw::Fn) => Ok(Decl::Fn(self.fn_decl()?)),
            TokKind::Kw(Kw::Optimizer) => {
                self.bump();
                let name = self.expect_ident("the name of the optimizer")?;
                let items = self.config_block()?;
                Ok(Decl::Optimizer { name, items, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Training) => {
                self.bump();
                self.expect_word("run")?;
                let items = self.config_block()?;
                Ok(Decl::Training { items, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Inference) => {
                self.bump();
                self.expect_word("serve")?;
                let items = self.config_block()?;
                Ok(Decl::Inference { items, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Effect) => {
                self.bump();
                let name = self.expect_ident("the name of the effect")?;
                self.end_of_line()?;
                Ok(Decl::Effect { name, span: start.to(self.prev_span()) })
            }
            TokKind::Ident(s) if s == "model" => {
                self.bump();
                let name = self.expect_ident("the name of the model")?;
                self.expect_op(Op::Eq)?;
                let body = self.mod_expr()?;
                self.end_of_line()?;
                Ok(Decl::Model { name, body, span: start.to(self.prev_span()) })
            }
            _ => Err(self.unexpected("a declaration (`module`, `fn`, `const`, `dim`, `type`, `import`, `static assert`, `optimizer`, `training run`, `inference serve`)")),
        }
    }

    /// A declaration or member must end at the end of its line.
    pub(crate) fn end_of_line(&mut self) -> PResult<()> {
        if self.at_newline() || matches!(self.toks[self.pos].kind, TokKind::Eof) || self.at_op(Op::RBrace) {
            Ok(())
        } else {
            Err(self.unexpected("end of line"))
        }
    }

    fn module(&mut self) -> PResult<ModuleDecl> {
        let start = self.expect_kw(Kw::Module)?.span;
        let name = self.expect_ident("the name of the module")?;
        let typarams = self.typarams_opt()?;
        self.expect_op(Op::LBrace)?;
        let members = self.lines(|p| p.members())?;
        self.expect_op(Op::RBrace)?;
        Ok(ModuleDecl { name, typarams, members, span: start.to(self.prev_span()) })
    }

    fn members(&mut self) -> PResult<Vec<Member>> {
        let mut members = Vec::new();
        loop {
            self.skip_newlines();
            let start = self.peek().span;
            match self.peek_kind() {
                TokKind::Op(Op::RBrace) => return Ok(members),
                TokKind::Eof => return Err(self.unexpected("`}` closing the module")),
                TokKind::Kw(Kw::Param) => {
                    self.bump();
                    let names = self.ident_list("a parameter name")?;
                    self.expect_op(Op::Colon)?;
                    let ty = self.ty()?;
                    let init = if self.eat_kw(Kw::Init).is_some() {
                        self.expect_op(Op::Eq)?;
                        Some(self.expr()?)
                    } else {
                        None
                    };
                    self.end_of_line()?;
                    members.push(Member::Param { names, ty, init, span: start.to(self.prev_span()) });
                }
                TokKind::Kw(Kw::Use) => {
                    self.bump();
                    let names = self.ident_list("a name for the used module")?;
                    self.expect_op(Op::Colon)?;
                    let module = self.mod_expr()?;
                    self.end_of_line()?;
                    members.push(Member::Use { names, module, span: start.to(self.prev_span()) });
                }
                TokKind::Kw(Kw::Fn) => members.push(Member::Fn(self.fn_decl()?)),
                _ => return Err(self.unexpected("`param`, `use`, `fn` or `}`")),
            }
        }
    }

    /// `a, b, c`
    fn ident_list(&mut self, what: &str) -> PResult<Vec<Ident>> {
        let mut names = vec![self.expect_ident(what)?];
        while self.eat_op(Op::Comma).is_some() {
            names.push(self.expect_ident(what)?);
        }
        Ok(names)
    }

    /// `{In Out : dim, S : Shape}`
    fn typarams_opt(&mut self) -> PResult<Vec<TyParam>> {
        // A module body also starts with `{`; type parameters start with a name.
        if !self.at_op(Op::LBrace) || !matches!(self.toks[self.pos + 1].kind, TokKind::Ident(_)) {
            return Ok(vec![]);
        }
        self.free(|p| {
            p.bump();
            let mut out = Vec::new();
            while !p.at_op(Op::RBrace) {
                let mut names = vec![p.expect_ident("a type parameter name")?];
                while let TokKind::Ident(_) = p.peek_kind() {
                    names.push(p.expect_ident("a type parameter name")?);
                }
                p.expect_op(Op::Colon)?;
                let kind = match p.peek_kind() {
                    TokKind::Kw(Kw::Dim) => {
                        let t = p.bump();
                        Ident { name: "dim".into(), span: t.span }
                    }
                    TokKind::Ident(_) => p.expect_ident("a kind")?,
                    _ => return Err(p.unexpected("a kind (`dim` or `Shape`)")),
                };
                for name in names {
                    out.push(TyParam { name, kind: kind.clone() });
                }
                if p.eat_op(Op::Comma).is_none() {
                    break;
                }
            }
            p.expect_op(Op::RBrace)?;
            Ok(out)
        })
    }

    /// `fn name[{typarams}](params) [-> ty] [uses effects] = body`
    fn fn_decl(&mut self) -> PResult<FnDecl> {
        let start = self.expect_kw(Kw::Fn)?.span;
        let name = self.expect_ident("the name of the function")?;
        let typarams = self.typarams_opt()?;
        let params = self.free(|p| {
            p.expect_op(Op::LParen)?;
            let mut params = Vec::new();
            while !p.at_op(Op::RParen) {
                params.push(p.param()?);
                if p.eat_op(Op::Comma).is_none() {
                    break;
                }
            }
            p.expect_op(Op::RParen)?;
            Ok::<_, ParseError>(params)
        })?;
        // The header may continue on the next line (`-> params net =`).
        self.skip_newlines();
        let ret = if self.eat_op(Op::Arrow).is_some() { Some(self.ty()?) } else { None };
        self.skip_newlines();
        let mut effects = Vec::new();
        if self.eat_kw(Kw::Uses).is_some() {
            effects = self.ident_list("an effect name")?;
        }
        self.skip_newlines();
        self.expect_op(Op::Eq)?;
        self.skip_newlines();
        let body = self.expr()?;
        self.end_of_line()?;
        Ok(FnDecl { name, typarams, params, ret, effects, body, span: start.to(self.prev_span()) })
    }

    /// `[axis] name [: ty] [= default]`
    fn param(&mut self) -> PResult<Param> {
        let start = self.peek().span;
        let axis = self.eat_kw(Kw::Axis).is_some();
        let name = self.expect_ident("a parameter name")?;
        let ty = if self.eat_op(Op::Colon).is_some() { Some(self.ty()?) } else { None };
        let default = if self.eat_op(Op::Eq).is_some() { Some(self.expr()?) } else { None };
        Ok(Param { name, ty, default, axis, span: start.to(self.prev_span()) })
    }

    /// `path[{dims}][(modexprs)] (∘ …)*`
    pub(crate) fn mod_expr(&mut self) -> PResult<ModExpr> {
        let first = self.mod_app()?;
        if !self.at_op(Op::Compose) {
            return Ok(first);
        }
        let start = first.span();
        let mut parts = vec![first];
        while self.eat_op(Op::Compose).is_some() {
            parts.push(self.mod_app()?);
        }
        Ok(ModExpr::Compose { parts, span: start.to(self.prev_span()) })
    }

    fn mod_app(&mut self) -> PResult<ModExpr> {
        let path = self.path()?;
        let dim_args = if self.at_op(Op::LBrace) { self.dim_args()? } else { vec![] };
        let mod_args = if self.at_op(Op::LParen) {
            self.free(|p| {
                p.bump();
                let mut args = Vec::new();
                while !p.at_op(Op::RParen) {
                    args.push(p.mod_expr()?);
                    if p.eat_op(Op::Comma).is_none() {
                        break;
                    }
                }
                p.expect_op(Op::RParen)?;
                Ok::<_, ParseError>(args)
            })?
        } else {
            vec![]
        };
        Ok(ModExpr::App { span: path.span.to(self.prev_span()), path, dim_args, mod_args })
    }

    /// `{expr, expr}` dimension arguments.
    pub(crate) fn dim_args(&mut self) -> PResult<Vec<crate::ast::Expr>> {
        self.free(|p| {
            p.expect_op(Op::LBrace)?;
            let mut args = Vec::new();
            while !p.at_op(Op::RBrace) {
                args.push(p.expr()?);
                if p.eat_op(Op::Comma).is_none() {
                    break;
                }
            }
            p.expect_op(Op::RBrace)?;
            Ok(args)
        })
    }

    /// `a.b.c`
    pub(crate) fn path(&mut self) -> PResult<Path> {
        let first = self.expect_ident("a name")?;
        let mut path = Path::single(first);
        while self.at_op(Op::Period) && matches!(self.peek_nth(1).kind, TokKind::Ident(_)) {
            self.bump();
            let seg = self.expect_ident("a name")?;
            path.span = path.span.to(seg.span);
            path.segs.push(seg);
        }
        Ok(path)
    }
}
