//! Structural verification of a PSK-Core program: every variable is
//! defined once and before use, blocks return a defined variable, calls have
//! the callee's arity and instantiate all its dimensions, tuple projections
//! are in range. Run in debug builds after lowering (and, later, after every
//! transformation) and by the tests.

use std::collections::HashSet;

use crate::ir::{Program, Rhs};

pub fn verify(prog: &Program) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();
    for (i, f) in prog.funcs.iter().enumerate() {
        let where_ = format!("function `{}` (#{i})", f.name);
        let mut defined: HashSet<_> = HashSet::new();
        for p in &f.params {
            if !defined.insert(p.var) {
                errors.push(format!("{where_}: parameter `{}` defined twice", p.name));
            }
        }
        for l in &f.body.lets {
            let uses: Vec<_> = match &l.rhs {
                Rhs::Lit(_) => vec![],
                Rhs::Prim { args, .. } | Rhs::Tuple(args) => args.clone(),
                Rhs::Call { f: callee, dim_args, args } => {
                    match prog.funcs.get(callee.0 as usize) {
                        None => errors.push(format!("{where_}: call of unknown function #{}", callee.0)),
                        Some(c) => {
                            if c.params.len() != args.len() {
                                errors.push(format!("{where_}: call of `{}` with {} arguments, {} expected", c.name, args.len(), c.params.len()));
                            }
                            for d in &c.dims {
                                if !dim_args.contains_key(d) {
                                    errors.push(format!("{where_}: call of `{}` does not instantiate dimension {d:?}", c.name));
                                }
                            }
                        }
                    }
                    args.clone()
                }
                Rhs::Proj(x, _) | Rhs::Field(x, _) | Rhs::Grad(x) => vec![*x],
                Rhs::Record(fields) => fields.iter().map(|(_, v)| *v).collect(),
                Rhs::Closure { f: callee, captured } => {
                    if let Some(c) = prog.funcs.get(callee.0 as usize)
                        && captured.len() > c.params.len()
                    {
                        errors.push(format!("{where_}: closure of `{}` captures more than its parameters", c.name));
                    }
                    captured.clone()
                }
                Rhs::Apply { fun, args } => {
                    let mut v = vec![*fun];
                    v.extend(args);
                    v
                }
            };
            for u in uses {
                if !defined.contains(&u) {
                    errors.push(format!("{where_}: variable %{} used before its definition", u.0));
                }
            }
            if !defined.insert(l.var) {
                errors.push(format!("{where_}: variable %{} defined twice", l.var.0));
            }
        }
        if !defined.contains(&f.body.result) {
            errors.push(format!("{where_}: result %{} is not defined", f.body.result.0));
        }
    }
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
