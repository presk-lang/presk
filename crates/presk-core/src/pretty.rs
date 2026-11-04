//! Readable rendering of PSK-Core, for `presk dump --stage core` and the
//! snapshot tests. Variables print as their source name when they have one,
//! else `%n`.

use presk_types::elab::ElabProgram;
use presk_types::ty::Names;

use crate::ir::{Func, Lit, PrimOp, Program, Rhs, VarId};

pub fn pretty(prog: &Program, elab: &ElabProgram) -> String {
    let mut out = String::new();
    for f in &prog.funcs {
        out.push_str(&pretty_func(f, prog, elab));
        out.push('\n');
    }
    out
}

fn var_name(f: &Func, v: VarId) -> String {
    for p in &f.params {
        if p.var == v {
            return p.name.clone();
        }
    }
    for l in &f.body.lets {
        if l.var == v {
            if let Some(n) = &l.name {
                return n.clone();
            }
        }
    }
    format!("%{}", v.0)
}

pub fn pretty_func(f: &Func, prog: &Program, elab: &ElabProgram) -> String {
    let n: Names = elab.names();
    let mut s = String::new();
    let dims: Vec<String> = f.dims.iter().map(|d| elab.dim_name(*d)).collect();
    let params: Vec<String> = f.params.iter().map(|p| format!("{} : {}", p.name, p.ty.display(&n))).collect();
    let eff = if f.effects.is_empty() { String::new() } else { format!(" uses {}", f.effects.join(", ")) };
    s.push_str(&format!("fn {}{}({}) -> {}{} {{\n", f.name, if dims.is_empty() { String::new() } else { format!("{{{}}}", dims.join(" ")) }, params.join(", "), f.ret.display(&n), eff));
    for l in &f.body.lets {
        s.push_str(&format!("  {}\n", pretty_let(f, l, prog, elab)));
    }
    s.push_str(&format!("  return {}\n}}\n", var_name(f, f.body.result)));
    s
}

/// One `let` of `f`, as [`pretty_func`] prints it without its indentation:
/// `%3 : ℝ[..., Out] = dot x, w`. A site's line, for `presk explain`.
pub fn pretty_let(f: &Func, l: &crate::ir::Let, prog: &Program, elab: &ElabProgram) -> String {
    let n: Names = elab.names();
    let v = |x: &VarId| var_name(f, *x);
    let args = |xs: &[VarId]| xs.iter().map(v).collect::<Vec<_>>().join(", ");
    let rhs = match &l.rhs {
        Rhs::Lit(Lit::Nat(k)) => format!("{k}"),
        Rhs::Lit(Lit::Float(t)) => t.clone(),
        // A slice's bounds may be dimensions: printed with their names.
        Rhs::Prim { op: op @ (PrimOp::Slice { axis, lo, hi } | PrimOp::SliceVjp { axis, lo, hi }), args: a } => {
            let word = if matches!(op, PrimOp::Slice { .. }) { "slice" } else { "slice_vjp" };
            format!("{word} {axis}: {}..{} {}", lo.display(&*n.dim), hi.display(&*n.dim), args(a))
        }
        Rhs::Prim { op, args: a } => if a.is_empty() { op.name() } else { format!("{} {}", op.name(), args(a)) },
        Rhs::Call { f: callee, dim_args, args: a } => {
            let d: Vec<String> = dim_args.iter().map(|(k, e)| format!("{} = {}", elab.dim_name(*k), e.display(&*n.dim))).collect();
            format!("call {}{}({})", prog.func(*callee).name, if d.is_empty() { String::new() } else { format!("{{{}}}", d.join(", ")) }, args(a))
        }
        Rhs::Tuple(a) => format!("({})", args(a)),
        Rhs::Proj(x, i) => format!("{}.{i}", v(x)),
        Rhs::Record(fields) => format!("{{{}}}", fields.iter().map(|(k, x)| format!("{k} = {}", v(x))).collect::<Vec<_>>().join(", ")),
        Rhs::Field(x, name) => format!("{}.{name}", v(x)),
        Rhs::Closure { f: callee, captured } => format!("closure {}[{}]", prog.func(*callee).name, args(captured)),
        Rhs::Grad(x) => format!("grad {}", v(x)),
        Rhs::Apply { fun, args: a } => format!("apply {}({})", v(fun), args(a)),
        Rhs::Fold { body, dim_args, n: count, stacked, init, extra, index, reverse, scan } => {
            let d: Vec<String> = dim_args.iter().map(|(k, e)| format!("{} = {}", elab.dim_name(*k), e.display(&*n.dim))).collect();
            let mut flags = String::new();
            if *index {
                flags.push_str(" index");
            }
            if *reverse {
                flags.push_str(" reverse");
            }
            if *scan {
                flags.push_str(" scan");
            }
            format!("fold{flags} {} {}{}(stacked {}; acc {}{})", count.display(&*n.dim), prog.func(*body).name, if d.is_empty() { String::new() } else { format!("{{{}}}", d.join(", ")) }, args(stacked), v(init), if extra.is_empty() { String::new() } else { format!("; {}", args(extra)) })
        }
    };
    format!("{} : {} = {}", var_name(f, l.var), l.ty.display(&n), rhs)
}
