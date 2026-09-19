//! Configuration blocks: `optimizer NAME { … }`, `training run { … }`,
//! `inference serve { … }`. Items are `name = expr`, `name : ty = expr`,
//! `name : expr` and `checkpoint every N steps to "path"`.

use super::{PResult, Parser};
use crate::ast::ConfigItem;
use crate::token::{Op, TokKind};

impl Parser {
    pub(crate) fn config_block(&mut self) -> PResult<Vec<ConfigItem>> {
        self.expect_op(Op::LBrace)?;
        let items = self.lines(|p| {
            let mut items = Vec::new();
            loop {
                p.skip_newlines();
                if p.at_op(Op::RBrace) {
                    return Ok(items);
                }
                if p.at(&TokKind::Eof) {
                    return Err(p.unexpected("`}` closing the block"));
                }
                items.push(p.config_item()?);
                p.end_of_line()?;
            }
        })?;
        self.expect_op(Op::RBrace)?;
        Ok(items)
    }

    fn config_item(&mut self) -> PResult<ConfigItem> {
        let start = self.peek().span;
        if self.at_word("checkpoint") {
            self.bump();
            self.expect_word("every")?;
            let every = self.expr()?;
            self.expect_word("steps")?;
            self.expect_word("to")?;
            let to = match self.peek_kind() {
                TokKind::Str(s) => {
                    self.bump();
                    s
                }
                _ => return Err(self.unexpected("a path in quotes")),
            };
            return Ok(ConfigItem::Checkpoint { every, to, span: start.to(self.prev_span()) });
        }
        // Field names may coincide with keywords (`optimizer = adamw`).
        let name = match self.peek_kind() {
            TokKind::Kw(kw) => {
                let t = self.bump();
                crate::ast::Ident { name: kw.text().to_string(), span: t.span }
            }
            _ => self.expect_ident("a field name")?,
        };
        if self.eat_op(Op::Eq).is_some() {
            let value = self.expr()?;
            return Ok(ConfigItem::Field { name, ty: None, value, span: start.to(self.prev_span()) });
        }
        self.expect_op(Op::Colon)?;
        // `name : ty = expr` or `name : expr` — try a type first, keep it only
        // if an `=` follows.
        let save = (self.pos, self.diags.len());
        if let Ok(ty) = self.ty() {
            if self.eat_op(Op::Eq).is_some() {
                let value = self.expr()?;
                return Ok(ConfigItem::Field { name, ty: Some(ty), value, span: start.to(self.prev_span()) });
            }
        }
        self.pos = save.0;
        self.diags.truncate(save.1);
        let value = self.expr()?;
        Ok(ConfigItem::Field { name, ty: None, value, span: start.to(self.prev_span()) })
    }
}
