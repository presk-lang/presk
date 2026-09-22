//! `presk run`: entry-point resolution, input synthesis, execution and
//! rendering of the result.

use std::collections::BTreeMap;

use presk_core::ir::Origin;
use presk_core::{FuncId, Program};
use presk_syntax::ast;
use presk_types::dim::DimVar;
use presk_types::elab::ElabProgram;
use presk_types::ty::{ModuleId, Ty};
use rand_chacha::ChaCha20Rng;
use rug::Float;

use crate::eval::{Dims, Interp, Result, RunError, err};
use crate::key::{Key, normal, uniform01};
use crate::real::Real;
use crate::value::{Tensor, Value};

pub struct RunOptions {
    pub entry: String,
    /// `--dim NAME=N`
    pub dims: Vec<(String, u64)>,
    pub seed: u64,
    pub bits: u32,
    pub digits: usize,
}

pub struct RunOutput {
    pub entry: String,
    pub func: FuncId,
    pub dims: Vec<(String, u64)>,
    pub inputs: Vec<(String, Value<Float>)>,
    pub result: Value<Float>,
}

/// Resolve `--entry`, synthesize the inputs, run.
pub fn run(prog: &Program, elab: &ElabProgram, opts: &RunOptions) -> Result<RunOutput> {
    let (func, dims, inputs) = prepare_run(prog, elab, opts)?;
    let mut interp: Interp<Float> = Interp::new(prog, elab, opts.bits, opts.seed);
    let args: Vec<Value<Float>> = inputs.iter().map(|(_, v)| v.clone()).collect();
    let result = interp.call(func, dims.clone(), args)?;
    let dims_named: Vec<(String, u64)> = dims.iter().map(|(v, n)| (elab.dim_name(*v), *n)).collect();
    Ok(RunOutput { entry: opts.entry.clone(), func, dims: dims_named, inputs, result })
}

/// The synthesized inputs of an entry, by parameter name.
pub type Inputs = Vec<(String, Value<Float>)>;

/// An entry resolved and its inputs synthesized: the function, its bound
/// dimensions and the named inputs.
pub type PreparedRun = (FuncId, Dims, Inputs);

/// Resolve `--entry`, bind the dimensions and synthesize the inputs (the
/// reference's job, whatever backend then runs the entry).
pub fn prepare_run(prog: &Program, elab: &ElabProgram, opts: &RunOptions) -> Result<PreparedRun> {
    let (func, mut dims, module_dims) = resolve_entry(prog, elab, &opts.entry)?;
    let f = prog.func(func);
    // Dimensions from --dim, by name among the function's dimensions. Names
    // that are not dimensions of the function become, in order, the axes of
    // the open prefix `...` of its parameters (`--dim batch=3` for `x : ℝ[..., D]`).
    let has_prefix = f.params.iter().any(|p| p.ty.shape().is_some_and(|s| s.prefix.is_some()));
    let mut prefix: Vec<crate::value::Axis> = Vec::new();
    for (name, n) in &opts.dims {
        let v = f.dims.iter().find(|d| elab.dim_name(**d) == *name).copied();
        match v {
            Some(v) => {
                dims.insert(v, *n);
            }
            None if has_prefix => prefix.push(crate::value::Axis { name: Some(name.clone()), len: *n as usize }),
            None => {
                let known: Vec<String> = f.dims.iter().map(|d| elab.dim_name(*d)).collect();
                return err(format!("`--dim {name}`: `{}` has no dimension named `{name}` (its dimensions: {})", f.name, if known.is_empty() { "none".into() } else { known.join(", ") }));
            }
        }
    }
    let unbound: Vec<String> = f.dims.iter().filter(|d| !dims.contains_key(d)).map(|d| elab.dim_name(*d)).collect();
    if !unbound.is_empty() {
        return err(format!("unbound dimension{} {}: give {}", if unbound.len() == 1 { "" } else { "s" }, unbound.join(", "), unbound.iter().map(|n| format!("`--dim {n}=N`")).collect::<Vec<_>>().join(" ")));
    }
    // The file-level facts (`dim T <= 1024`) must hold for the values given.
    {
        let by_name: BTreeMap<String, u64> = dims.iter().map(|(v, n)| (elab.dim_name(*v), *n)).collect();
        for c in &elab.facts.constraints {
            let a: BTreeMap<DimVar, presk_types::dim::Rational> =
                c.vars().iter().filter_map(|v| by_name.get(&elab.dim_name(*v)).map(|n| (*v, presk_types::dim::rat(*n as i64)))).collect();
            if c.holds(&a) == Some(false) {
                let name = |v: DimVar| elab.dim_name(v);
                let text = match c {
                    presk_types::presburger::Constraint::Eq(l, r) => format!("{} == {}", l.display(&name), r.display(&name)),
                    presk_types::presburger::Constraint::Le(l, r) => format!("{} <= {}", l.display(&name), r.display(&name)),
                    presk_types::presburger::Constraint::Divides(k, e) => format!("{} % {k} == 0", e.display(&name)),
                };
                return err(format!("`dim` bound violated: `{text}` is false for {}", presk_types::presburger::display_assignment(&a, c, &name)));
            }
        }
    }
    let mut interp: Interp<Float> = Interp::new(prog, elab, opts.bits, opts.seed);
    let master = Key::from_seed(opts.seed);
    let keys = master.split(f.params.len());
    let mut inputs = Vec::new();
    let sig = match f.origin {
        Origin::Source(id) => Some(elab.function(id)),
        _ => None,
    };
    for (i, p) in f.params.iter().enumerate() {
        let default = sig.and_then(|s| s.params.iter().find(|q| q.name == p.name)).and_then(|q| q.default.clone());
        let mut v = synthesize(&mut interp, &p.ty, &dims, &keys[i], default.as_ref(), &prefix).map_err(|e| RunError { message: format!("cannot synthesize `{}`: {}", p.name, e.message), span: e.span })?;
        if matches!(f.origin, Origin::Optimizer { .. }) {
            v = optimizer_input(&p.name, v);
        }
        inputs.push((p.name.clone(), v));
    }
    let _ = module_dims;
    Ok((func, dims, inputs))
}

/// Inputs of an optimizer entry (`adamw.step`) made meaningful: the second
/// moment `v` of the state `s` is non-negative (squared draws), the step
/// `t` is 3 and the horizon `total` 10, so that the schedule is exercised
/// inside its warm-up.
fn optimizer_input<R: Real>(name: &str, v: Value<R>) -> Value<R> {
    match (name, v) {
        ("s", Value::Tuple(parts)) if parts.len() == 2 => {
            let second = parts[1].map_leaves(&mut |leaf| match leaf {
                Value::Real(t) => Value::Real(t.map(|x: &R| x.mul(x))),
                other => other.clone(),
            });
            Value::tuple(vec![parts[0].clone(), second])
        }
        ("t", Value::Nat(_)) => Value::Nat(3),
        ("total", Value::Nat(_)) => Value::Nat(10),
        (_, v) => v,
    }
}

/// `fn`, `Module.fn`, `Module{8, 4}.fn`: the Core function and the module's
/// dimension arguments.
pub fn resolve_entry(prog: &Program, elab: &ElabProgram, entry: &str) -> Result<(FuncId, Dims, Vec<u64>)> {
    // Optimizer functions (`adamw.step`) are found by their exact name.
    if let Some(id) = prog.find(entry)
        && matches!(prog.func(id).origin, Origin::Optimizer { .. })
    {
        return Ok((id, Dims::new(), vec![]));
    }
    let (module_part, fn_name) = match entry.rsplit_once('.') {
        Some((m, f)) => (Some(m), f),
        None => (None, entry),
    };
    let mut dims = Dims::new();
    let mut margs = Vec::new();
    let core_name = match module_part {
        None => fn_name.to_string(),
        Some(m) => {
            let (mname, args) = match m.split_once('{') {
                Some((n, rest)) => (n, rest.trim_end_matches('}').split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse::<u64>()).collect::<std::result::Result<Vec<_>, _>>().map_err(|_| RunError { message: format!("`{entry}`: module arguments must be natural literals"), span: None })?),
                None => (m, vec![]),
            };
            // The module definition, in the main unit or its imports.
            let main = &elab.units[elab.main];
            let mid: Option<ModuleId> = main.modules.get(mname).copied().or_else(|| main.imports.values().find_map(|u| elab.units[*u].modules.get(mname).copied()));
            let Some(mid) = mid else {
                return err(format!("`{entry}`: no module named `{mname}`"));
            };
            let def = elab.module(mid);
            if def.typarams.len() != args.len() {
                return err(format!("`{entry}`: `{}` takes {} dimension argument{}, {} given (write `{}{{…}}.{fn_name}`)", def.name, def.typarams.len(), if def.typarams.len() == 1 { "" } else { "s" }, args.len(), def.name));
            }
            for (v, n) in def.typarams.iter().zip(&args) {
                dims.insert(*v, *n);
            }
            margs = args;
            format!("{}.{fn_name}", def.name)
        }
    };
    let Some(id) = prog.find(&core_name) else {
        let candidates: Vec<String> = prog.funcs.iter().filter(|f| matches!(f.origin, Origin::Source(_))).map(|f| f.name.clone()).collect();
        return err(format!("`{entry}`: no function `{core_name}`; entries: {}", candidates.join(", ")));
    };
    Ok((id, dims, margs))
}

/// A value of the given type, deterministic from `key`.
fn synthesize(interp: &mut Interp<Float>, ty: &Ty, dims: &Dims, key: &Key, default: Option<&ast::Expr>, prefix: &[crate::value::Axis]) -> Result<Value<Float>> {
    let prec = interp.prec;
    match ty {
        Ty::Real(s) => {
            let axes = interp.axes_of_with(s, dims, prefix)?;
            let mut rng: ChaCha20Rng = key.rng();
            let n: usize = axes.iter().map(|a| a.len).product();
            let data = (0..n).map(|_| Float::from_f64(normal(&mut rng), prec)).collect();
            Ok(Value::Real(Tensor { axes, data }))
        }
        Ty::Idx { bound, shape } => {
            let axes = interp.axes_of_with(shape, dims, prefix)?;
            let b = interp.dim(bound, dims)?;
            let mut rng = key.rng();
            let n: usize = axes.iter().map(|a| a.len).product();
            let data = (0..n).map(|_| (uniform01(&mut rng) * b as f64) as i64).collect();
            Ok(Value::Idx { bound: b, t: Tensor { axes, data } })
        }
        Ty::Nat => Ok(Value::Nat(match default {
            Some(ast::Expr::Lit { lit: ast::Lit::Nat(n), .. }) => n.parse().unwrap_or(1),
            _ => 1,
        })),
        Ty::Int => Ok(Value::Int(0)),
        Ty::Rat | Ty::F64 => Ok(Value::F64(match default {
            Some(ast::Expr::Lit { lit: ast::Lit::Float(f), .. }) => f.parse().unwrap_or(1.0),
            Some(ast::Expr::Lit { lit: ast::Lit::Nat(n), .. }) => n.parse().unwrap_or(1.0),
            _ => 1.0,
        })),
        Ty::Bool => Ok(Value::Bool(false)),
        Ty::Key => Ok(Value::Key(key.clone())),
        Ty::Tuple(ts) => {
            let keys = key.split(ts.len());
            Ok(Value::tuple(ts.iter().zip(&keys).map(|(t, k)| synthesize(interp, t, dims, k, None, prefix)).collect::<Result<_>>()?))
        }
        Ty::Params(r) => {
            // The module's initializer, with the instance's dimensions.
            let (def, args) = match r {
                presk_types::ty::ModRef::Inst { def, args } => (*def, args.clone()),
                _ => return err("cannot synthesize the parameters of a stacked or composed module directly"),
            };
            let init = interp.prog.find(&format!("{}.init", interp.elab.module(def).name)).ok_or_else(|| RunError { message: "no initializer".into(), span: None })?;
            // The instance's arguments; the initializer's other dimensions (its
            // file's `dim`s) are bound by name from the entry's dimensions, since
            // another file's `B` and `T` are distinct variables with the same name.
            let mut idims = dims.clone();
            for (v, e) in interp.elab.module(def).typarams.iter().zip(&args) {
                idims.insert(*v, interp.dim(e, dims)?);
            }
            for d in &interp.prog.func(init).dims {
                if !idims.contains_key(d) {
                    let name = interp.elab.dim_name(*d);
                    if let Some((_, n)) = dims.iter().find(|(v, _)| interp.elab.dim_name(**v) == name) {
                        idims.insert(*d, *n);
                    }
                }
            }
            interp.call(init, idims, vec![Value::Key(key.clone())])
        }
        Ty::Module(_) | Ty::Fn { .. } | Ty::Infer(_) => err(format!("cannot synthesize a value of type `{}`", ty.display(&interp.elab.names()))),
    }
}

// ----- rendering -----------------------------------------------------------

pub fn render(v: &Value<Float>, digits: usize, indent: usize) -> String {
    let pad = " ".repeat(indent);
    match v {
        Value::Real(t) => render_tensor(t, &|x: &Float| x.to_string_sig(digits)),
        Value::Idx { t, .. } => render_tensor(t, &|x: &i64| x.to_string()),
        Value::Nat(n) => n.to_string(),
        Value::Int(n) => n.to_string(),
        Value::F64(x) => format!("{x}"),
        Value::Bool(b) => b.to_string(),
        Value::Key(k) => format!("key {}…", k.short()),
        Value::Tuple(vs) => format!("({})", vs.iter().map(|v| render(v, digits, indent)).collect::<Vec<_>>().join(", ")),
        Value::Record(fs) => {
            let mut s = String::from("{\n");
            for (n, v) in fs.iter() {
                s.push_str(&format!("{pad}  {n} = {}\n", render(v, digits, indent + 2)));
            }
            s.push_str(&format!("{pad}}}"));
            s
        }
        Value::Closure { .. } => "<function>".into(),
    }
}

fn render_tensor<T: Clone>(t: &Tensor<T>, f: &dyn Fn(&T) -> String) -> String {
    fn go<T: Clone>(t: &Tensor<T>, f: &dyn Fn(&T) -> String, level: usize, offset: usize) -> String {
        if level == t.rank() {
            return f(&t.data[offset]);
        }
        let stride = t.strides()[level];
        let items: Vec<String> = (0..t.axes[level].len).map(|i| go(t, f, level + 1, offset + i * stride)).collect();
        format!("[{}]", items.join(", "))
    }
    let axes = crate::prims::show_axes(&t.axes);
    if t.rank() == 0 { f(&t.data[0]) } else { format!("{axes} {}", go(t, f, 0, 0)) }
}

/// Type-like header of a value for the output: `ℝ[batch: 2, model: 4]`.
pub fn describe(v: &Value<Float>) -> String {
    match v {
        Value::Real(t) => format!("ℝ{}", crate::prims::show_axes(&t.axes)),
        Value::Idx { bound, t } => format!("idx[{bound}]{}", crate::prims::show_axes(&t.axes)),
        Value::Tuple(vs) => format!("({})", vs.iter().map(describe).collect::<Vec<_>>().join(", ")),
        Value::Record(_) => "params".into(),
        other => other.kind().into(),
    }
}

pub fn to_json(v: &Value<Float>, digits: usize) -> serde_json::Value {
    use serde_json::{Map, Value as J, json};
    fn tensor_json<T: Clone>(t: &Tensor<T>, f: &dyn Fn(&T) -> J, level: usize, offset: usize) -> J {
        if level == t.rank() {
            return f(&t.data[offset]);
        }
        let stride = t.strides()[level];
        J::Array((0..t.axes[level].len).map(|i| tensor_json(t, f, level + 1, offset + i * stride)).collect())
    }
    match v {
        Value::Real(t) => json!({"axes": t.axes.iter().map(|a| json!({"name": a.name, "len": a.len})).collect::<Vec<_>>(), "data": tensor_json(t, &|x: &Float| J::String(x.to_string_sig(digits)), 0, 0)}),
        Value::Idx { bound, t } => json!({"bound": bound, "axes": t.axes.iter().map(|a| json!({"name": a.name, "len": a.len})).collect::<Vec<_>>(), "data": tensor_json(t, &|x: &i64| json!(x), 0, 0)}),
        Value::Nat(n) => json!(n),
        Value::Int(n) => json!(n),
        Value::F64(x) => json!(x),
        Value::Bool(b) => json!(b),
        Value::Key(k) => json!(format!("key {}", k.short())),
        Value::Tuple(vs) => J::Array(vs.iter().map(|v| to_json(v, digits)).collect()),
        Value::Record(fs) => {
            let mut m = Map::new();
            for (n, v) in fs.iter() {
                m.insert(n.clone(), to_json(v, digits));
            }
            J::Object(m)
        }
        Value::Closure { .. } => json!("<function>"),
    }
}

#[allow(dead_code)]
fn _unused(_: &BTreeMap<DimVar, u64>) {}
