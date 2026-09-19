//! Types and axis lists.

use super::{PResult, ParseError, Parser};
use crate::ast::{AxisSpec, Ty};
use crate::token::{Kw, Op, TokKind};

impl Parser {
    pub(crate) fn ty(&mut self) -> PResult<Ty> {
        let start = self.peek().span;
        match self.peek_kind() {
            TokKind::Kw(Kw::Real) => {
                self.bump();
                let axes = if self.at_op(Op::LBracket) { self.axes()? } else { vec![] };
                Ok(Ty::Real { axes, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Idx) => {
                self.bump();
                let bound = self.free(|p| {
                    p.expect_op(Op::LBracket)?;
                    let e = p.expr()?;
                    p.expect_op(Op::RBracket)?;
                    Ok::<_, ParseError>(e)
                })?;
                let axes = if self.at_op(Op::LBracket) { self.axes()? } else { vec![] };
                Ok(Ty::Idx { bound: Box::new(bound), axes, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Param) => {
                self.bump();
                let inner = Box::new(self.ty()?);
                Ok(Ty::Param { inner, span: start.to(self.prev_span()) })
            }
            TokKind::Kw(Kw::Params) => {
                self.bump();
                let module = self.path()?;
                Ok(Ty::Params { module, span: start.to(self.prev_span()) })
            }
            TokKind::Op(Op::LParen) => self.free(|p| {
                p.bump();
                let mut elems = Vec::new();
                while !p.at_op(Op::RParen) {
                    elems.push(p.ty()?);
                    if p.eat_op(Op::Comma).is_none() {
                        break;
                    }
                }
                p.expect_op(Op::RParen)?;
                Ok(Ty::Tuple { elems, span: start.to(p.prev_span()) })
            }),
            TokKind::Ident(_) => {
                let path = self.path()?;
                let args = if self.at_op(Op::LBrace) { self.dim_args()? } else { vec![] };
                Ok(Ty::Named { path, args, span: start.to(self.prev_span()) })
            }
            _ => Err(self.unexpected("a type")),
        }
    }

    /// `[axis, axis, …]` where axis is `...` or `[name:] dimexpr`.
    fn axes(&mut self) -> PResult<Vec<AxisSpec>> {
        self.free(|p| {
            p.expect_op(Op::LBracket)?;
            let mut axes = Vec::new();
            while !p.at_op(Op::RBracket) {
                if let Some(t) = p.eat_op(Op::Ellipsis) {
                    axes.push(AxisSpec::Ellipsis { span: t.span });
                } else {
                    let name = if matches!(p.peek_kind(), TokKind::Ident(_)) && p.peek_nth(1).kind == TokKind::Op(Op::Colon) {
                        let id = p.expect_ident("an axis name")?;
                        p.bump(); // `:`
                        Some(id)
                    } else {
                        None
                    };
                    let dim = p.expr()?;
                    axes.push(AxisSpec::Axis { name, dim });
                }
                if p.eat_op(Op::Comma).is_none() {
                    break;
                }
            }
            p.expect_op(Op::RBracket)?;
            Ok(axes)
        })
    }
}
