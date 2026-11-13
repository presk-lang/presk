//! `presk serve`: the generation loop of an `inference serve` over a
//! [`Backend`] (plan 0.3.0 step 5, §3.6: minimal).
//!
//! ```text
//! θ = load(weights, shaped like M.init(key))
//! ctx = prompt
//! k = key(seed)
//! repeat N times, i = 1, 2, …:
//!   next = generate(θ, child(k, i), ctx[-window:], temperature)   -- the file's generation function
//!   ctx = ctx ++ [next]
//! ```
//!
//! The generation function is the file's function of signature
//! `(θ : params M, [key : key,] ctx : idx[V][batch: B, seq: T], … defaults)
//! -> idx[V][batch: B]` (resolved by the checker, or named with `--entry`).
//! When it takes a key, token `i` is given the child `i` of the serve's key,
//! the key `--seed` names, as a training run gives step `t` the child `t` of
//! its own (the reference's definition "Generation", plan 0.10.0 step 7b);
//! its `sample` draws from that key, so every backend draws alike. Two forms
//! of the loop:
//!
//! - **sampling** (`temperature > 0`): the function is called as written;
//! - **greedy** (`temperature = 0`): a copy of the function with every
//!   `sample(key, x)` replaced by `argmax(x)` is called (with temperature
//!   `1`, so the division is harmless). Every backend agrees on this path up
//!   to ties.
//!
//! When the model's parameters depend on the context length (`pos : ℝ[T, D]`,
//! question 43), `T` is fixed by the checkpoint: it is given by `--dim T=N`
//! (default: the prompt's length) and the function sees the last `T` tokens
//! of the context, a sliding window. Otherwise the whole context is passed
//! and grows, checked against the file's `dim T <= …` bound at every step.
//!
//! The loop's time is split in two (plan 0.17.2 step 4, D200). Compiling is
//! preparing each context length the loop has not seen, and, inside the
//! calls, lowering and compiling each kernel at its first launch, which the
//! backend counts ([`Backend::compile_stats`]); computing is the calls' time
//! less that. The initializer's call and the weights' reading are neither.

use std::collections::BTreeMap;
use std::path::PathBuf;

use presk_core::ir::{Let, Origin, Param, PrimOp, Rhs, VarId};
use presk_core::{FuncId, Program};
use presk_interp::eval::Dims;
use presk_interp::key::Key;
use presk_rt::{Backend, Buf, Realization, Result, RtError, Value, axes_of, err, load_params};
use presk_syntax::ast::{Expr, Lit};
use presk_types::check::TypedProgram;
use presk_types::dim::DimVar;
use presk_types::elab::ElabProgram;
use presk_types::ty::{Axis, ModRef, Shape, Ty};

/// Options of `presk serve`.
#[derive(Clone, Debug)]
pub struct ServeOptions {
    /// The weights file (default: the block's `weights`).
    pub weights: Option<PathBuf>,
    pub prompt: Vec<u64>,
    /// Tokens to generate.
    pub tokens: u64,
    pub seed: u64,
    pub bits: u32,
    pub realize: Realization,
    /// `None`: the generation function's default; `Some(0.0)`: greedy.
    pub temperature: Option<f64>,
    /// `--dim NAME=N`: dimensions of the model not fixed by the prompt (`T`
    /// of a positional embedding).
    pub dims: BTreeMap<String, u64>,
}

/// The functions of a serve, resolved on the Core program.
#[derive(Clone, Debug)]
pub struct ServeFns {
    pub model: String,
    pub init: FuncId,
    /// The generation function as written.
    pub generate: FuncId,
    /// Its greedy copy (`sample` → `argmax`).
    pub greedy: FuncId,
    /// Whether the generation function takes a key before the context.
    pub key: bool,
    /// Whether the generation function takes a temperature after the context.
    pub temperature: bool,
    /// The temperature's default, when it is a literal.
    pub default_temperature: Option<f64>,
    /// Names of the `batch` and `seq` dimensions of the context type.
    pub batch_dim: String,
    pub seq_dim: String,
    /// The context type (`idx[V][batch: B, seq: T]`).
    pub ctx_ty: Ty,
    /// Whether the model's parameters fix the context's length: a leaf of
    /// its parameter tree has `seq_dim` in its type (`pos : ℝ[T, D]`). Its
    /// initializer's dimensions do not say: every initializer is generic in
    /// every dimension of its file (found in plan 0.17.0 step 7, D180).
    pub tied: bool,
    /// The block's weights file.
    pub weights: Option<PathBuf>,
}

/// What a serve produced.
#[derive(Clone, Debug)]
pub struct ServeOutput {
    pub model: String,
    pub weights: PathBuf,
    pub prompt: Vec<u64>,
    pub generated: Vec<u64>,
    /// The context length passed to the model at each step, when fixed by
    /// the parameters (`None`: the whole context, growing).
    pub window: Option<u64>,
    pub greedy: bool,
    pub vocab: u64,
    /// The loop's seconds compiling: the lengths prepared and the kernels
    /// lowered and compiled inside the calls.
    pub compile_s: f64,
    /// The calls' seconds less their compiling.
    pub compute_s: f64,
    /// The context lengths prepared, one per length first seen.
    pub compilations: u64,
    /// The kernels compiled inside the loop.
    pub kernels: u64,
}

fn te(code: &'static str, message: impl Into<String>) -> RtError {
    RtError { code, message: message.into(), span: None }
}

/// Resolves the serve: the model, the generation function (the checker's, or
/// `entry`), its greedy copy appended to `core`.
pub fn prepare_serve(core: &mut Program, elab: &ElabProgram, typed: &TypedProgram, entry: Option<&str>) -> Result<ServeFns> {
    let Some(serve) = typed.inference.first() else {
        return err("E7001", "no `inference serve` in this file");
    };
    let ModRef::Inst { def, .. } = &serve.model else {
        return err("E7001", "the model of an `inference serve` must be a plain module");
    };
    let model = elab.module(*def).name.clone();
    let init = core.find(&format!("{model}.init")).ok_or_else(|| te("E7001", "no initializer"))?;
    let fid = match entry {
        Some(name) => *elab.units[elab.main].fns.get(name).ok_or_else(|| te("E7001", format!("`--entry {name}`: no top-level function `{name}` in this file")))?,
        None => serve.generate.ok_or_else(|| te("E7001", "this `inference serve` has no generation function (W0003): name one with `--entry FN`"))?,
    };
    let sig = elab.function(fid);
    // The signature, checked again for an `--entry`: the context is the
    // third parameter when a key comes second.
    let key = matches!(sig.params.get(1).and_then(|p| p.ty.as_ref()), Some(Ty::Key));
    let c = if key { 2 } else { 1 };
    let (Some(Ty::Idx { shape, .. }), Some(Ty::Params(_))) = (sig.params.get(c).and_then(|p| p.ty.as_ref()), sig.params.first().and_then(|p| p.ty.as_ref())) else {
        return err("E7001", format!("`{}` is not a generation function: expected `(θ : params {model}, [key : key,] ctx : idx[V][batch: B, seq: T], …) -> idx[V][batch: B]`", sig.name));
    };
    let dim_name = |n: &str| -> Result<String> {
        let a = shape.axes.iter().find(|a| a.name.as_deref() == Some(n)).ok_or_else(|| te("E7001", format!("`{}`: the context has no axis named `{n}`", sig.name)))?;
        let vars: Vec<DimVar> = a.len.vars().collect();
        if vars.len() == 1 {
            Ok(elab.dim_name(vars[0]))
        } else {
            err("E7001", format!("`{}`: the `{n}` axis of the context must be a single dimension", sig.name))
        }
    };
    let (batch_dim, seq_dim) = (dim_name("batch")?, dim_name("seq")?);
    let seq_var = shape.axes.iter().find(|a| a.name.as_deref() == Some("seq")).and_then(|a| a.len.vars().next());
    let tied = elab.params_tree(&serve.model).leaves().iter().any(|(_, t)| t.shape().is_some_and(|sh| sh.axes.iter().any(|a| a.len.vars().any(|v| Some(v) == seq_var))));
    let temperature = sig.params.len() > c + 1;
    if sig.params.len() > c + 2 || (temperature && !matches!(sig.params[c + 1].ty, Some(Ty::F64) | Some(Ty::Rat))) {
        return err("E7001", format!("`{}`: after the context, at most one `f64` parameter (the temperature) is supported", sig.name));
    }
    let default_temperature = sig.params.get(c + 1).and_then(|p| p.default.as_ref()).and_then(|e| match e {
        Expr::Lit { lit: Lit::Float(f), .. } | Expr::Lit { lit: Lit::Nat(f), .. } => f.parse().ok(),
        _ => None,
    });
    let generate = core.funcs.iter().position(|f| matches!(f.origin, Origin::Source(id) if id == fid)).map(|i| FuncId(i as u32)).ok_or_else(|| te("E7001", "generation function not lowered"))?;
    // The greedy copy: every `sample(key, x)` becomes an `argmax(x)`; the
    // copy keeps the function's parameters, its key read by nothing.
    let mut g = core.func(generate).clone();
    g.name = format!("{}.greedy", g.name);
    g.origin = Origin::Lambda { parent: generate };
    g.effects.retain(|e| e != "Rand");
    for l in &mut g.body.lets {
        if let Rhs::Prim { op: PrimOp::Sample, args } = &l.rhs {
            l.rhs = Rhs::Prim { op: PrimOp::Argmax, args: vec![args[1]] };
        }
    }
    core.funcs.push(g);
    let greedy = FuncId(core.funcs.len() as u32 - 1);
    presk_core::verify(core).map_err(|e| te("E7001", format!("internal error in the greedy copy: {}", e.join("; "))))?;
    Ok(ServeFns { model, init, generate, greedy, key, temperature, default_temperature, batch_dim, seq_dim, ctx_ty: sig.params[c].ty.clone().unwrap(), tied, weights: serve.weights_path.clone() })
}

/// The generation function and its greedy copy, each reading the logits at
/// a position its caller gives rather than at the last one (plan 0.17.0
/// step 7, D180): `name.at` and `name.greedy.at`, appended to `core`.
///
/// A model whose parameters do not fix the context's length is served
/// over the whole context, which grows. A C program has one shape, so the C
/// export runs these copies at a length `n` (`--dim T=n`), on the context
/// padded after its last token, and reads that token's logits. Each copy
/// takes one more argument, last:
/// - `at : ℝ[seq: T]`, 1 at the position to read and 0 elsewhere;
/// - its selection `x[seq: -1]` becomes `sum(axis seq, x · at)`, which is
///   exactly `x` at that position: a product by 0 or by 1 is exact, and so
///   is adding zeros.
///
/// Causality makes the logits at the last real position those of the
/// context alone. The function must select the last position itself, once.
pub fn at_position(core: &mut Program, fns: &ServeFns) -> Result<(FuncId, FuncId)> {
    let ctx = if fns.key { 2 } else { 1 };
    let mut made = Vec::new();
    for f in [fns.generate, fns.greedy] {
        let mut g = core.func(f).clone();
        let seq_len = match g.params.get(ctx).map(|p| &p.ty) {
            Some(Ty::Idx { shape, .. }) => shape.axes.iter().find(|a| a.name.as_deref() == Some("seq")).map(|a| a.len.clone()),
            _ => None,
        }
        .ok_or_else(|| te("E8009", format!("`{}`: the context has no axis `seq`", g.name)))?;
        let picks: Vec<usize> = g.body.lets.iter().enumerate().filter(|(_, l)| matches!(&l.rhs, Rhs::Prim { op: PrimOp::IndexConst { axis, index: -1 }, .. } if axis == "seq")).map(|(i, _)| i).collect();
        let [at] = picks.as_slice() else {
            return err("E8009", format!("`{}` selects the last position `[seq: -1]` {} times in its own body: a C program of one length reads the logits at the context's last token, which needs that selection, once", g.name, picks.len()));
        };
        let at = *at;
        let mut ty: BTreeMap<VarId, Ty> = g.params.iter().map(|p| (p.var, p.ty.clone())).collect();
        ty.extend(g.body.lets.iter().map(|l| (l.var, l.ty.clone())));
        let next = ty.keys().map(|v| v.0).max().unwrap_or(0) + 1;
        let (pos, prod) = (VarId(next), VarId(next + 1));
        let Rhs::Prim { args, .. } = &g.body.lets[at].rhs else { unreachable!("found above") };
        let x = args[0];
        let x_ty = ty.get(&x).cloned().ok_or_else(|| te("E7001", "the selected value has no type"))?;
        let span = g.body.lets[at].span;
        g.params.push(Param { var: pos, name: "at".into(), ty: Ty::Real(Shape { prefix: None, axes: vec![Axis { name: Some("seq".into()), len: seq_len }] }) });
        g.body.lets[at].rhs = Rhs::Prim { op: PrimOp::Sum { axis: "seq".into() }, args: vec![prod] };
        g.body.lets.insert(at, Let { var: prod, name: None, ty: x_ty, rhs: Rhs::Prim { op: PrimOp::Mul, args: vec![x, pos] }, span });
        g.name = format!("{}.at", g.name);
        g.origin = Origin::Lambda { parent: f };
        core.funcs.push(g);
        made.push(FuncId(core.funcs.len() as u32 - 1));
    }
    presk_core::verify(core).map_err(|e| te("E7001", format!("internal error in the copies at a position: {}", e.join("; "))))?;
    Ok((made[0], made[1]))
}

/// Runs the generation loop over `backend`.
pub fn serve(backend: &mut dyn Backend, core: &Program, elab: &ElabProgram, fns: &ServeFns, opts: &ServeOptions, mut on_token: impl FnMut(u64, u64)) -> Result<ServeOutput> {
    let weights = opts.weights.clone().or_else(|| fns.weights.clone()).ok_or_else(|| te("E9003", "no weights: the block has no `weights` field, give --weights FILE.psk.d"))?;
    if !weights.is_file() {
        return err("E9003", format!("{}: No such file (train first, or give --weights)", weights.display()));
    }
    let temperature = opts.temperature.or(fns.default_temperature).unwrap_or(1.0);
    if temperature.is_nan() || temperature < 0.0 {
        return err("E7001", format!("temperature {temperature}: must be ≥ 0 (0 is greedy)"));
    }
    let greedy = temperature == 0.0;
    // Is the model tied to the context length?
    let tied = fns.tied;
    let prompt_len = opts.prompt.len() as u64;
    if prompt_len == 0 {
        return err("E7001", "empty prompt: give at least one token");
    }
    let window = if tied { Some(opts.dims.get(&fns.seq_dim).copied().unwrap_or(prompt_len)) } else { None };
    if let Some(w) = window
        && prompt_len < w
    {
        return err("E7001", format!("the prompt has {prompt_len} token{} but the model's parameters fix the context length {} = {w}: give at least {w} tokens (or --dim {}=N for a checkpoint trained with another length)", if prompt_len == 1 { "" } else { "s" }, fns.seq_dim, fns.seq_dim));
    }
    // Dimensions by name for every function called.
    let names_for = |ctx_len: u64| -> BTreeMap<String, u64> {
        let mut m = opts.dims.clone();
        m.insert(fns.batch_dim.clone(), 1);
        m.insert(fns.seq_dim.clone(), ctx_len);
        m
    };
    let dims_of = |f: FuncId, names: &BTreeMap<String, u64>| -> Result<Dims> {
        let mut d = Dims::new();
        for v in &core.func(f).dims {
            let name = elab.dim_name(*v);
            match names.get(&name) {
                Some(n) => {
                    d.insert(*v, *n);
                }
                None => return err("E7001", format!("dimension `{name}` of `{}` is not fixed by the prompt: give `--dim {name}=N`", core.func(f).name)),
            }
        }
        check_facts(elab, &d)?;
        Ok(d)
    };
    // θ, shaped by the initializer at the window (or the prompt) length.
    let init_names = names_for(window.unwrap_or(prompt_len));
    let init = backend.prepare(fns.init, &dims_of(fns.init, &init_names)?, &opts.realize)?;
    let template = backend.call(&init, vec![Value::Key(Key::from_seed(opts.seed))])?;
    // Scales are data derived from the realization and these weights, so
    // they are computed here, at load (D44, question 57).
    let theta = opts.realize.stored_params(&load_params(&weights, &template)?)?;
    // The context type's vocabulary.
    let Ty::Idx { bound, shape } = &fns.ctx_ty else { return err("E7001", "the context is not an index tensor") };
    let f = if greedy { fns.greedy } else { fns.generate };
    // The serve's key: token `step` is given its child `step`.
    let serve_key = Key::from_seed(opts.seed);
    let mut tokens = opts.prompt.clone();
    let mut generated = Vec::new();
    let mut prepared: BTreeMap<u64, presk_rt::Prepared> = BTreeMap::new();
    let mut vocab = 0;
    let (kernels0, _) = backend.compile_stats();
    let (mut compile_s, mut compute_s) = (0.0, 0.0);
    for step in 1..=opts.tokens {
        let ctx: &[u64] = match window {
            Some(w) => &tokens[tokens.len() - w as usize..],
            None => &tokens,
        };
        let len = ctx.len() as u64;
        let names = names_for(len);
        let dims = dims_of(f, &names)?;
        let p = match prepared.get(&len) {
            Some(p) => p,
            None => {
                let t = std::time::Instant::now();
                let p = backend.prepare(f, &dims, &opts.realize)?;
                compile_s += t.elapsed().as_secs_f64();
                prepared.entry(len).or_insert(p)
            }
        };
        let v = bound.subst(&dims.iter().map(|(v, n)| (*v, presk_types::dim::DimExpr::constant(presk_types::dim::rat(*n as i64)))).collect()).as_nat().ok_or_else(|| te("E7001", "the context's vocabulary is not fixed"))?;
        vocab = v;
        if let Some(t) = tokens.iter().find(|t| **t >= v) {
            return err("E7001", format!("token {t} is not below the vocabulary {v}"));
        }
        let axes = axes_of(shape, &dims)?;
        let ctx_v = Value::Idx { bound: v, t: presk_rt::Tensor { axes, data: Buf::I64(ctx.iter().map(|t| *t as i64).collect()) } };
        let mut args = vec![theta.clone()];
        if fns.key {
            args.push(Value::Key(serve_key.child(step)));
        }
        args.push(ctx_v);
        if fns.temperature {
            args.push(Value::F64(if greedy { 1.0 } else { temperature }));
        }
        let (_, before) = backend.compile_stats();
        let t = std::time::Instant::now();
        let out = backend.call(p, args)?;
        let call_s = t.elapsed().as_secs_f64();
        let compiled = backend.compile_stats().1 - before;
        compile_s += compiled;
        compute_s += (call_s - compiled).max(0.0);
        let next = match out {
            Value::Idx { t, .. } if t.numel() == 1 => t.data.to_f64()[0] as u64,
            other => return err("E7001", format!("the generation function returned a {}, expected one token", other.kind())),
        };
        on_token(step, next);
        generated.push(next);
        tokens.push(next);
    }
    let compilations = prepared.len() as u64;
    let kernels = backend.compile_stats().0 - kernels0;
    Ok(ServeOutput { model: fns.model.clone(), weights, prompt: opts.prompt.clone(), generated, window, greedy, vocab, compile_s, compute_s, compilations, kernels })
}

/// The file-level facts (`dim T <= 1024`) must hold for the dimensions bound.
fn check_facts(elab: &ElabProgram, dims: &Dims) -> Result<()> {
    use presk_types::presburger::Constraint;
    let by_name: BTreeMap<String, u64> = dims.iter().map(|(v, n)| (elab.dim_name(*v), *n)).collect();
    for c in &elab.facts.constraints {
        let a: BTreeMap<DimVar, presk_types::dim::Rational> = c.vars().iter().filter_map(|v| by_name.get(&elab.dim_name(*v)).map(|n| (*v, presk_types::dim::rat(*n as i64)))).collect();
        if c.holds(&a) == Some(false) {
            let name = |v: DimVar| elab.dim_name(v);
            let text = match c {
                Constraint::Eq(l, r) => format!("{} == {}", l.display(&name), r.display(&name)),
                Constraint::Le(l, r) => format!("{} <= {}", l.display(&name), r.display(&name)),
                Constraint::Divides(k, e) => format!("{} % {k} == 0", e.display(&name)),
            };
            return err("E7001", format!("`dim` bound violated: `{text}` is false for {} (the context cannot grow further)", presk_types::presburger::display_assignment(&a, c, &name)));
        }
    }
    Ok(())
}
