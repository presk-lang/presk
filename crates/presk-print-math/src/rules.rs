//! The rules the page writes on its arrows and its cards, in the
//! reference's words: the identity of each rewrite rule of the optimizer
//! (tables `core-algebra` and `core-structure`), and a contraction nest's
//! sum with the order its terms are added in (definition
//! `contraction-nest`, the reduction order of table `realize-decisions`).

use presk_core::ir::{Func, Let, PrimOp, Rhs};
use presk_types::elab::ElabProgram;
use presk_types::ty::Ty;

use crate::core::{dim, var_name};
use crate::m::{B, Format, M};

/// The identity of a rewrite rule of `presk-opt`, as the reference's
/// tables write it, read left to right as the optimizer applies it.
pub fn rewrite(rule: &str) -> Option<&'static str> {
    Some(match rule {
        "add-comm" => r"a + b = b + a",
        "mul-comm" => r"a\,b = b\,a",
        "add-assoc" => r"(a + b) + c = a + (b + c)",
        "add-assoc-r" => r"a + (b + c) = (a + b) + c",
        "mul-assoc" => r"(a\,b)\,c = a\,(b\,c)",
        "mul-assoc-r" => r"a\,(b\,c) = (a\,b)\,c",
        "add-zero" => r"x + 0 = x",
        "zero-add" => r"0 + x = x",
        "sub-zero" => r"x - 0 = x",
        "sub-self" => r"x - x = 0_x",
        "mul-one" => r"x \cdot 1 = x",
        "one-mul" => r"1 \cdot x = x",
        "div-one" => r"x / 1 = x",
        "mul-zero" => r"x \cdot 0 = 0_x",
        "zero-mul" => r"0 \cdot x = 0_x",
        "neg-neg" => r"-(-x) = x",
        "neg-zeros" => r"-0_x = 0_x",
        "sub-as-add-neg" => r"a - b = a + (-b)",
        "add-neg-as-sub" => r"a + (-b) = a - b",
        "pow-one" => r"x^1 = x",
        "mul-self" => r"x\,x = x^2",
        "log-exp" => r"\log(\exp x) = x",
        "exp-add" => r"\exp a\,\exp b = \exp(a + b)",
        "exp-add-r" => r"\exp(a + b) = \exp a\,\exp b",
        "sqrt-sq" => r"\sqrt{x}\,\sqrt{x} = x",
        "sqrt-pow" => r"(\sqrt{x})^2 = x",
        "distrib" => r"a\,(b + c) = a\,b + a\,c",
        "factor" => r"a\,b + a\,c = a\,(b + c)",
        "distrib-sub" => r"a\,(b - c) = a\,b - a\,c",
        "factor-sub" => r"a\,b - a\,c = a\,(b - c)",
        "mul-div" => r"a\,(b / c) = (a\,b) / c",
        "div-div" => r"(a / b) / c = a / (b\,c)",
        "sum-scale" => r"\textstyle\sum (s\,x) = s \sum x",
        "sum-scale-r" => r"\textstyle\sum (x\,s) = s \sum x",
        "mean-scale" => r"\code{mean}\ (s\,x) = s\ \code{mean}\ x",
        "mean-scale-r" => r"\code{mean}\ (x\,s) = s\ \code{mean}\ x",
        "sum-neg" => r"\textstyle\sum (-x) = -\sum x",
        "mean-neg" => r"\code{mean}\ (-x) = -\code{mean}\ x",
        "sum_to-id" => r"\code{sum\_to}\ x, t = x",
        "broadcast_like-id" => r"\code{broadcast\_like}\ x, t = x",
        "sum_to-zeros" => r"\code{sum\_to}\ 0_x, t = 0_t",
        "add-zeros_like" => r"x + 0_t = x",
        "zeros_like-add" => r"0_t + x = x",
        "sub-zeros_like" => r"x - 0_t = x",
        "field-set-same" => r"(r[n := v]).n = v",
        "field-set-other" => r"(r[m := v]).n = r.n",
        "field-zeros" => r"(0_t).n = 0_{t.n}",
        "set-set-same" => r"(r[n := u])[n := v] = r[n := v]",
        "add-set-set" => r"r[n := a] + s[n := b] = (r + s)[n := a + b]",
        "add-set" => r"r[n := a] + s = (r + s)[n := a + s.n]",
        "set-add" => r"s + r[n := a] = (s + r)[n := s.n + a]",
        _ => return None,
    })
}

/// The sum a contraction nest computes, at an index, in chapter 3's
/// notation (definition `den-contract`): `dot`, `matmul`, `ein`, and the
/// two derivatives of `dot` (table `core-prims`). `None` for any other
/// binding.
pub fn contraction(f: &Func, l: &Let, elab: &ElabProgram) -> Option<M> {
    let Rhs::Prim { op, args } = &l.rhs else { return None };
    let v = |i: usize| M::name(var_name(f, args[i]));
    let ty = |i: usize| f.params.iter().find(|p| p.var == args[i]).map(|p| &p.ty).or_else(|| f.body.lets.iter().find(|x| x.var == args[i]).map(|x| &x.ty));
    // The length of an axis of an operand, from its type.
    let len = |i: usize, axis: isize| -> M {
        match ty(i).and_then(|t| t.shape()) {
            Some(s) if !s.axes.is_empty() => {
                let k = if axis < 0 { s.axes.len() as isize + axis } else { axis };
                s.axes.get(k as usize).map(|a| dim(&a.len, elab)).unwrap_or(M::name("K"))
            }
            _ => M::name("K"),
        }
    };
    // `∑_{k=0}^{n-1}`.
    let sum = |k: &str, n: M, body: M| {
        let to = match &n {
            M::Num(t) if t.parse::<u64>().is_ok_and(|v| v > 0) => M::num((t.parse::<u64>().unwrap() - 1).to_string()),
            _ => M::bin(B::Sub, n, M::num("1")),
        };
        M::SumTo { var: k.to_string(), to: Box::new(to), body: Box::new(body) }
    };
    let i = M::name("i");
    let j = M::name("j");
    let k = M::name("k");
    let prefixed = |t: Option<&Ty>| t.and_then(|t| t.shape()).is_some_and(|s| s.prefix.is_some() || s.axes.len() > 2);
    let outer = |mut xs: Vec<M>, on: bool| {
        if on {
            xs.insert(0, M::Ellipsis);
        }
        xs
    };
    Some(match op {
        PrimOp::Dot | PrimOp::Matmul => {
            let o = prefixed(ty(0));
            let lhs = M::index(M::name(var_name(f, l.var)), outer(vec![i.clone(), j.clone()], o));
            let body = M::Prod(vec![M::index(v(0), outer(vec![i, k.clone()], o)), M::index(v(1), vec![k, j])]);
            M::rel(crate::m::R::Eq, lhs, sum("k", len(0, -1), body))
        }
        PrimOp::DotVjpL => {
            // (c, w): Σ_n c[…, i, n] w[j, n]
            let o = prefixed(ty(0));
            let n = M::name("n");
            let lhs = M::index(M::name(var_name(f, l.var)), outer(vec![i.clone(), j.clone()], o));
            let body = M::Prod(vec![M::index(v(0), outer(vec![i, n.clone()], o)), M::index(v(1), vec![j, n])]);
            M::rel(crate::m::R::Eq, lhs, sum("n", len(0, -1), body))
        }
        PrimOp::DotVjpR => {
            // (a, c): Σ_k a[k, i] c[k, j], k over every leading axis
            let lhs = M::index(M::name(var_name(f, l.var)), vec![i.clone(), j.clone()]);
            let body = M::Prod(vec![M::index(v(0), vec![k.clone(), i]), M::index(v(1), vec![k, j])]);
            M::rel(crate::m::R::Eq, lhs, M::Sum(vec![M::name("k")], Box::new(body)))
        }
        PrimOp::Ein { operands, output } => {
            let mut contracted: Vec<&String> = Vec::new();
            for o in operands {
                for n in o {
                    if !output.contains(n) && !contracted.contains(&n) {
                        contracted.push(n);
                    }
                }
            }
            let lhs = M::index(M::name(var_name(f, l.var)), output.iter().map(M::code).collect());
            let body = M::Prod(operands.iter().zip(args).map(|(o, x)| M::index(M::name(var_name(f, *x)), o.iter().map(M::code).collect())).collect());
            if contracted.is_empty() { M::rel(crate::m::R::Eq, lhs, body) } else { M::rel(crate::m::R::Eq, lhs, M::Sum(contracted.iter().map(|c| M::code(c.as_str())).collect(), Box::new(body))) }
        }
        _ => return None,
    })
}

/// The order in which a contraction's terms `t_k` are added: in
/// increasing `k` (`partials` none: the reduction orders `sequential` and
/// `blocked`, bit-identical), or in `n` partial sums, one per residue of
/// `k` modulo `n`, each in increasing `k`, added in increasing residue
/// (`interleaved(n)`).
pub fn order(partials: Option<u64>, f: Format) -> String {
    let latex = f == Format::Latex;
    match partials {
        None => {
            if latex {
                r"(\cdots((t_0 + t_1) + t_2) + \cdots) + t_{K-1}".into()
            } else {
                "(⋯((t₀ + t₁) + t₂) + ⋯) + t_(K−1)".into()
            }
        }
        Some(n) => {
            if latex {
                format!(r"\textstyle\sum_{{r=0}}^{{{}}} \bigl(\sum_{{q}} t_{{q \cdot {n} + r}}\bigr)", n.saturating_sub(1))
            } else {
                format!("∑_(r=0..{}) (∑_q t_(q·{n}+r))", n.saturating_sub(1))
            }
        }
    }
}
