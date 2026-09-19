//! Readable rendering of PSK-Core, for `presk dump --stage core` and the
//! snapshot tests. Variables print as their source name when they have one,
//! else `%n`.

use presk_types::elab::ElabProgram;
use presk_types::ty::Names;

use crate::ir::{Func, Lit, Program, Rhs, VarId};

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
        let v = |x: &VarId| var_name(f, *x);
        let args = |xs: &[VarId]| xs.iter().map(v).collect::<Vec<_>>().join(", ");
        let rhs = match &l.rhs {
            Rhs::Lit(Lit::Nat(k)) => format!("{k}"),
            Rhs::Lit(Lit::Float(t)) => t.clone(),
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
        };
        s.push_str(&format!("  {} : {} = {}\n", var_name(f, l.var), l.ty.display(&n), rhs));
    }
    s.push_str(&format!("  return {}\n}}\n", var_name(f, f.body.result)));
    s
}
