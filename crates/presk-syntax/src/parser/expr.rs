//! Expressions: Pratt precedence climbing, postfix forms, `let`, `λ`, `ein`,
//! call arguments and index selectors.

use super::{PResult, ParseError, Parser};
use crate::ast::{Arg, BinOp, Expr, Ident, IndexSel, Lit, Pat, Path, UnOp};
use crate::token::{Kw, Op, TokKind};

/// Binary operators with their precedence (higher binds tighter). `^` is
/// handled in `unary` (right-associative, above unary minus).
fn binop(op: Op) -> Option<(BinOp, u8)> {
    Some(match op {
        Op::Pipe => (BinOp::Pipe, 1),
        Op::Bar => (BinOp::Alt, 2),
        Op::Tensor => (BinOp::Tensor, 3),
        Op::Compose => (BinOp::Compose, 3),
        Op::EqEq => (BinOp::Eq, 4),
        Op::Le => (BinOp::Le, 4),
        Op::Ge => (BinOp::Ge, 4),
        Op::Lt => (BinOp::Lt, 4),
        Op::Gt => (BinOp::Gt, 4),
        Op::Plus => (BinOp::Add, 5),
        Op::Minus => (BinOp::Sub, 5),
        Op::Star => (BinOp::Mul, 6),
        Op::Slash => (BinOp::Div, 6),
        Op::Percent => (BinOp::Rem, 6),
        Op::Dot => (BinOp::Dot, 6),
        _ => return None,
    })
}

impl Parser {
    pub(crate) fn expr(&mut self) -> PResult<Expr> {
        match self.peek_kind() {
            TokKind::Kw(Kw::Let) => self.let_expr(),
            TokKind::Kw(Kw::Lambda) => self.lambda(),
            _ => self.binary(0),
        }
    }

    /// `let pat = init NEWLINE body`
    fn let_expr(&mut self) -> PResult<Expr> {
        let start = self.expect_kw(Kw::Let)?.span;
        let pat = self.pat()?;
        self.expect_op(Op::Eq)?;
        let init = self.binary_or_lambda()?;
        if self.free > 0 {
            return Err(self.error_at(
                init.span(),
                "`let` is not allowed inside parentheses or brackets",
                "the body of a `let` must start on the next line",
            ));
        }
        if !self.at_newline() {
            return Err(self.unexpected("end of line (the body of a `let` starts on the next line)"));
        }
        self.skip_newlines();
        if self.at_op(Op::RBrace) || self.at(&TokKind::Eof) {
            return Err(self.error_at(start.to(init.span()), "`let` without a body", "this binding is never used: an expression must follow on the next line"));
        }
        let body = self.expr()?;
        Ok(Expr::Let { pat, init: Box::new(init), span: start.to(body.span()), body: Box::new(body) })
    }

    fn binary_or_lambda(&mut self) -> PResult<Expr> {
        if self.at_kw(Kw::Lambda) { self.lambda() } else { self.binary(0) }
    }

    /// `λ pat. body`
    fn lambda(&mut self) -> PResult<Expr> {
        let start = self.expect_kw(Kw::Lambda)?.span;
        let pat = self.pat()?;
        self.expect_op(Op::Period)?;
        let body = self.expr()?;
        Ok(Expr::Lambda { pat, span: start.to(body.span()), body: Box::new(body) })
    }

    pub(crate) fn pat(&mut self) -> PResult<Pat> {
        let start = self.peek().span;
        if self.at_op(Op::LParen) {
            return self.free(|p| {
                p.bump();
                let mut elems = Vec::new();
                while !p.at_op(Op::RParen) {
                    elems.push(p.pat()?);
                    if p.eat_op(Op::Comma).is_none() {
                        break;
                    }
                }
                p.expect_op(Op::RParen)?;
                Ok(Pat::Tuple { elems, span: start.to(p.prev_span()) })
            });
        }
        Ok(Pat::Ident(self.expect_ident("a pattern (a name or a tuple of names)")?))
    }

    /// In lines mode a binary operator may start the next line, provided it
    /// cannot begin an expression itself (so `-` never continues a line).
    fn peek_binop(&mut self) -> Option<(BinOp, u8)> {
        if self.free == 0 && self.at_newline() {
            let mut i = self.pos;
            while matches!(self.toks[i].kind, TokKind::Newline) {
                i += 1;
            }
            if let TokKind::Op(op) = self.toks[i].kind
                && op != Op::Minus
                && let Some(b) = binop(op)
            {
                self.pos = i;
                return Some(b);
            }
            return None;
        }
        match self.peek_kind() {
            TokKind::Op(op) => binop(op),
            _ => None,
        }
    }

    fn binary(&mut self, min_prec: u8) -> PResult<Expr> {
        let mut lhs = self.unary()?;
        while let Some((op, prec)) = self.peek_binop() {
            if prec < min_prec {
                break;
            }
            self.bump();
            if self.free == 0 {
                self.skip_newlines();
            }
            let rhs = self.binary(prec + 1)?;
            let span = lhs.span().to(rhs.span());
            lhs = Expr::Binary { op, lhs: Box::new(lhs), rhs: Box::new(rhs), span };
        }
        Ok(lhs)
    }

    fn unary(&mut self) -> PResult<Expr> {
        if let Some(t) = self.eat_op(Op::Minus) {
            let e = self.unary()?;
            return Ok(Expr::Unary { op: UnOp::Neg, span: t.span.to(e.span()), expr: Box::new(e) });
        }
        let base = self.postfix()?;
        if self.eat_op(Op::Caret).is_some() {
            let rhs = self.unary()?;
            let span = base.span().to(rhs.span());
            return Ok(Expr::Binary { op: BinOp::Pow, lhs: Box::new(base), rhs: Box::new(rhs), span });
        }
        Ok(base)
    }

    fn postfix(&mut self) -> PResult<Expr> {
        let mut e = self.primary()?;
        loop {
            match self.peek_kind() {
                TokKind::Op(Op::LParen) => {
                    let args = self.call_args()?;
                    e = Expr::Call { span: e.span().to(self.prev_span()), callee: Box::new(e), args };
                }
                TokKind::Op(Op::LBracket) => {
                    let sels = self.index_sels()?;
                    e = Expr::Index { span: e.span().to(self.prev_span()), base: Box::new(e), sels };
                }
                TokKind::Op(Op::LBrace) if matches!(e, Expr::Var { .. }) => {
                    let args = self.dim_args()?;
                    e = Expr::Inst { span: e.span().to(self.prev_span()), base: Box::new(e), args };
                }
                TokKind::Op(Op::Period) if matches!(self.peek_nth(1).kind, TokKind::Ident(_)) => {
                    self.bump();
                    let field = self.expect_ident("a field name")?;
                    e = Expr::Field { span: e.span().to(field.span), base: Box::new(e), field };
                }
                _ => return Ok(e),
            }
        }
    }

    fn primary(&mut self) -> PResult<Expr> {
        let t = self.peek().clone();
        match t.kind {
            TokKind::Nat(s) => {
                self.bump();
                Ok(Expr::Lit { lit: Lit::Nat(s), span: t.span })
            }
            TokKind::Float(s) => {
                self.bump();
                Ok(Expr::Lit { lit: Lit::Float(s), span: t.span })
            }
            TokKind::Quantity { num, unit } => {
                self.bump();
                Ok(Expr::Lit { lit: Lit::Quantity { num, unit }, span: t.span })
            }
            TokKind::Str(s) => {
                self.bump();
                Ok(Expr::Lit { lit: Lit::Str(s), span: t.span })
            }
            TokKind::Kw(Kw::Ein) => {
                self.bump();
                let axes = self.free(|p| {
                    p.expect_op(Op::LBracket)?;
                    let mut axes = Vec::new();
                    while !p.at_op(Op::RBracket) {
                        axes.push(p.expect_ident("an axis name")?);
                        if p.eat_op(Op::Comma).is_none() {
                            break;
                        }
                    }
                    p.expect_op(Op::RBracket)?;
                    Ok::<_, ParseError>(axes)
                })?;
                let body = self.free(|p| {
                    p.expect_op(Op::LParen)?;
                    let e = p.expr()?;
                    p.expect_op(Op::RParen)?;
                    Ok::<_, ParseError>(e)
                })?;
                Ok(Expr::Ein { axes, body: Box::new(body), span: t.span.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Let) => self.let_expr(),
            TokKind::Kw(Kw::Lambda) => self.lambda(),
            TokKind::Ident(ref s) if s == "stream" && matches!(self.peek_nth(1).kind, TokKind::Str(_)) => {
                // `stream "path"`: juxtaposition sugar for `stream("path")`.
                let callee = self.expect_ident("stream")?;
                let arg = self.primary()?;
                Ok(Expr::Call { span: callee.span.to(arg.span()), callee: Box::new(Expr::Var { path: Path::single(callee) }), args: vec![Arg::Pos(arg)] })
            }
            TokKind::Ident(_) => Ok(Expr::Var { path: self.path()? }),
            TokKind::Op(Op::LParen) => self.free(|p| {
                p.bump();
                if p.eat_op(Op::RParen).is_some() {
                    return Ok(Expr::Tuple { elems: vec![], span: t.span.to(p.prev_span()) });
                }
                let first = p.expr()?;
                if p.eat_op(Op::Colon).is_some() {
                    let ty = p.ty()?;
                    p.expect_op(Op::RParen)?;
                    return Ok(Expr::Ascribe { expr: Box::new(first), ty, span: t.span.to(p.prev_span()) });
                }
                if p.at_op(Op::Comma) {
                    let mut elems = vec![first];
                    while p.eat_op(Op::Comma).is_some() {
                        if p.at_op(Op::RParen) {
                            break;
                        }
                        elems.push(p.expr()?);
                    }
                    p.expect_op(Op::RParen)?;
                    return Ok(Expr::Tuple { elems, span: t.span.to(p.prev_span()) });
                }
                p.expect_op(Op::RParen)?;
                Ok(first)
            }),
            _ => Err(self.unexpected("an expression")),
        }
    }

    /// `(arg, …)` with the four argument forms.
    fn call_args(&mut self) -> PResult<Vec<Arg>> {
        self.free(|p| {
            p.expect_op(Op::LParen)?;
            let mut args = Vec::new();
            while !p.at_op(Op::RParen) {
                args.push(p.call_arg()?);
                if p.eat_op(Op::Comma).is_none() {
                    break;
                }
            }
            p.expect_op(Op::RParen)?;
            Ok(args)
        })
    }

    fn call_arg(&mut self) -> PResult<Arg> {
        if self.eat_kw(Kw::Axis).is_some() {
            return Ok(Arg::Axis(self.expect_ident("an axis name after `axis`")?));
        }
        if let TokKind::Ident(_) = self.peek_kind() {
            match self.peek_nth(1).kind {
                TokKind::Op(Op::Eq) => {
                    let name = self.expect_ident("a name")?;
                    self.bump(); // `=`
                    let value = self.expr()?;
                    return Ok(Arg::Named { name, value });
                }
                TokKind::Op(Op::Colon) => {
                    // `name: expr` is a named axis, except `name : ℝ` and the
                    // other type keywords, which ascribe (`sqrt(HD : ℝ)`).
                    let is_ascription = matches!(self.peek_nth(2).kind, TokKind::Kw(Kw::Real | Kw::Idx | Kw::Param | Kw::Params));
                    if !is_ascription {
                        let name = self.expect_ident("a name")?;
                        self.bump(); // `:`
                        let value = self.expr()?;
                        return Ok(Arg::NamedAxis { name, value });
                    }
                    let name = self.expect_ident("a name")?;
                    self.bump(); // `:`
                    let ty = self.ty()?;
                    let span = name.span.to(ty.span());
                    return Ok(Arg::Pos(Expr::Ascribe { expr: Box::new(Expr::Var { path: Path::single(name) }), ty, span }));
                }
                _ => {}
            }
        }
        Ok(Arg::Pos(self.expr()?))
    }

    /// `[sel, …]` where sel is `axis: expr` or `expr`.
    fn index_sels(&mut self) -> PResult<Vec<IndexSel>> {
        self.free(|p| {
            p.expect_op(Op::LBracket)?;
            let mut sels = Vec::new();
            while !p.at_op(Op::RBracket) {
                if matches!(p.peek_kind(), TokKind::Ident(_)) && p.peek_nth(1).kind == TokKind::Op(Op::Colon) {
                    let axis: Ident = p.expect_ident("an axis name")?;
                    p.bump(); // `:`
                    let value = p.expr()?;
                    sels.push(IndexSel::Named { axis, value });
                } else {
                    sels.push(IndexSel::Pos(p.expr()?));
                }
                if p.eat_op(Op::Comma).is_none() {
                    break;
                }
            }
            p.expect_op(Op::RBracket)?;
            Ok(sels)
        })
    }
}
