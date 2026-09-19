//! `presk dump --stage elab`: a readable listing of the elaborated program.

use super::{DimOrigin, ElabProgram};
use crate::ty::{InstId, ModRef};

pub fn dump(p: &ElabProgram) -> String {
    let n = p.names();
    let mut out = String::new();
    for (uid, u) in p.units.iter().enumerate() {
        out.push_str(&format!("== {}{}\n", u.path.display(), if uid == p.main { " (main)" } else { "" }));
        for (alias, target) in &u.imports {
            out.push_str(&format!("import {alias} = {}\n", p.units[*target].path.display()));
        }
        for (name, c) in &u.consts {
            out.push_str(&format!("const {name} = {}{}\n", c.value, if c.overridden { "  (--const)" } else { "" }));
        }
        for (name, v) in &u.dims {
            let bound = p.facts.constraints.iter().find_map(|c| match c {
                crate::presburger::Constraint::Le(l, r) if *l == crate::dim::DimExpr::var(*v) => Some(format!(" <= {}", r.display(&*n.dim))),
                _ => None,
            });
            out.push_str(&format!("dim {name}{}\n", bound.unwrap_or_default()));
        }
        for (name, t) in &u.types {
            out.push_str(&format!("type {name} = {}\n", t.display(&n)));
        }
        for (name, _) in &u.effects {
            out.push_str(&format!("effect {name}\n"));
        }
        for (_, mid) in &u.modules {
            let m = p.module(*mid);
            let tps: Vec<String> = m.typarams.iter().map(|v| p.dim_name(*v)).collect();
            out.push_str(&format!("module {}{} {{\n", m.name, if tps.is_empty() { String::new() } else { format!("{{{} : dim}}", tps.join(" ")) }));
            for pd in &m.params {
                out.push_str(&format!("  param {} : {}{}\n", pd.name, pd.ty.display(&n), if pd.init.is_some() { "  init" } else { "" }));
            }
            for ud in &m.uses {
                out.push_str(&format!("  use {} : {}\n", ud.name, ud.module.display(&n)));
            }
            for fid in &m.fns {
                out.push_str(&format!("  {}\n", fn_line(p, *fid)));
            }
            out.push_str("}\n");
        }
        for (_, fid) in &u.fns {
            out.push_str(&format!("{}\n", fn_line(p, *fid)));
        }
        for (name, m) in &u.models {
            out.push_str(&format!("model {name} = {}\n", m.display(&n)));
        }
        for (name, items) in &u.optimizers {
            out.push_str(&format!("optimizer {name} ({} fields)\n", items.len()));
        }
        for (items, _) in &u.training {
            out.push_str(&format!("training run ({} items)\n", items.len()));
        }
        for (items, _) in &u.inference {
            out.push_str(&format!("inference serve ({} items)\n", items.len()));
        }
        for (i, (_, ok)) in u.asserts.iter().enumerate() {
            out.push_str(&format!("static assert #{}: {}\n", i + 1, if *ok { "discharged" } else { "FAILED" }));
        }
    }
    let mut implicit: Vec<String> = p.dims.iter().filter(|d| d.origin == DimOrigin::Implicit).map(|d| d.name.clone()).collect();
    implicit.dedup();
    if !implicit.is_empty() {
        out.push_str(&format!("== implicit dimensions: {}\n", implicit.join(", ")));
    }
    if !p.insts.is_empty() {
        out.push_str("== instances\n");
        let mut seen = Vec::new();
        for i in 0..p.insts.len() {
            let id = InstId(i as u32);
            let mut block = format!("{}:\n", p.inst_name(id));
            let inst = p.inst(id);
            for (path, ty) in p.params_tree(&ModRef::Inst { def: inst.def, args: inst.args.clone() }).leaves() {
                block.push_str(&format!("  {path} : {}\n", ty.display(&n)));
            }
            // Generic instances of different functions render alike; print once.
            if !seen.contains(&block) {
                out.push_str(&block);
                seen.push(block);
            }
        }
    }
    out
}

fn fn_line(p: &ElabProgram, fid: super::FnId) -> String {
    let n = p.names();
    let f = p.function(fid);
    let tps: Vec<String> = f.typarams.iter().map(|v| p.dim_name(*v)).collect();
    let params: Vec<String> = f
        .params
        .iter()
        .map(|q| {
            let ty = match &q.ty {
                Some(t) => format!(" : {}", t.display(&n)),
                None => String::new(),
            };
            format!("{}{}{}{}", if q.axis { "axis " } else { "" }, q.name, ty, if q.default.is_some() { " = …" } else { "" })
        })
        .collect();
    let ret = match &f.ret {
        Some(t) => format!(" -> {}", t.display(&n)),
        None => String::new(),
    };
    let eff = if f.effects.is_empty() { String::new() } else { format!(" uses {}", f.effects.join(", ")) };
    let implicit = if f.implicit_dims.is_empty() {
        String::new()
    } else {
        format!("   -- implicit dims: {}", f.implicit_dims.iter().map(|v| p.dim_name(*v)).collect::<Vec<_>>().join(", "))
    };
    format!("fn {}{}({}){}{}{}", f.name, if tps.is_empty() { String::new() } else { format!("{{{} : dim}}", tps.join(" ")) }, params.join(", "), ret, eff, implicit)
}
