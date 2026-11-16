//! The bound in every format (plan 0.18.0 step 5, D208): a realization that
//! stores a value in a format no backend computes in, rounds a nest's
//! operations, products or additions to formats of its own, sums integers
//! counted in their unit or stores an affine value is bounded with each
//! format's term, and the bound is above the gap the kernel interpreter's
//! emulation measures against the exact reference.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Once;

use presk_format::Format;
use presk_interp::{Interp, RunOptions};
use presk_kernel::KernelBackend;
use presk_precision::envelope::Envelope;
use presk_precision::propagate::{Analysis, Model, propagate_with};
use presk_rt::{Dtype, Realization, ReductionOrder, Scale, Value};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, ElabProgram, Options};
use rug::Float;

type IValue = presk_interp::Value<Float>;

fn chdir_root() {
    static ONCE: Once = Once::new();
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    ONCE.call_once(|| std::env::set_current_dir(root).unwrap());
}

fn prepare(rel: &str) -> (ElabProgram, presk_core::Program) {
    chdir_root();
    let mut sm = SourceMap::new();
    let id = sm.load(Path::new(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &Options::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty(), "{}", render_all(&diags, &sm, false));
    (prog, core)
}

fn gap(reference: &IValue, got: &Value) -> f64 {
    match (reference, got) {
        (IValue::Real(t), Value::Real(g)) => {
            let gd = g.data.to_f64();
            t.data.iter().zip(&gd).map(|(x, y)| (x.to_f64() - y).abs()).map(|d| if d.is_finite() { d } else { f64::INFINITY }).fold(0.0, f64::max)
        }
        (IValue::Tuple(xs), Value::Tuple(ys)) => xs.iter().zip(ys).map(|(x, y)| gap(x, y)).fold(0.0, f64::max),
        _ => 0.0,
    }
}

/// An entry's envelope, measured exactly at seed 0, its exact value, and
/// what a realization `make` builds from that envelope measures and is
/// bounded at by each rule.
struct Case {
    measured: f64,
    analyses: Vec<Analysis>,
}

fn case(rel: &str, entry: &str, dims: &[(&str, u64)], make: &dyn Fn(&Envelope) -> Realization) -> Case {
    let (prog, core) = prepare(rel);
    let opts = RunOptions { entry: entry.into(), dims: dims.iter().map(|(n, v)| (n.to_string(), *v)).collect(), seed: 0, bits: 256, digits: 17 };
    let (func, fdims, inputs) = presk_interp::run::prepare_run(&core, &prog, &opts).unwrap();
    let args: Vec<IValue> = inputs.iter().map(|(_, v)| v.clone()).collect();
    let mut interp: Interp<Float> = Interp::new(&core, &prog, 256);
    interp.magnitudes = Some(Default::default());
    interp.minima = Some(Default::default());
    interp.ranges = Some(Default::default());
    interp.terms = Some(Default::default());
    let reference = interp.call(func, fdims.clone(), args.clone()).unwrap();
    let mut env = Envelope::new(rel, entry, 256, "2026-11-14");
    env.observe(&interp.magnitudes.take().unwrap());
    env.observe_minima(&interp.minima.take().unwrap());
    env.observe_range(&interp.ranges.take().unwrap());
    env.observe_terms(&interp.terms.take().unwrap());
    let r = make(&env);
    let mut backend = KernelBackend::new(&core, &prog, &r);
    let rargs = r.realize_args(&args.iter().map(|v| Value::from_interp(v).unwrap()).collect::<Vec<_>>()).unwrap();
    let p = presk_rt::Backend::prepare(&mut backend, func, &fdims, &r).unwrap_or_else(|e| panic!("{rel} {entry}: {}", e.message));
    let got = presk_rt::Backend::call(&mut backend, &p, rargs).unwrap_or_else(|e| panic!("{rel} {entry}: {}", e.message));
    presk_precision::propagate::bounded_formats(&r).unwrap();
    let dmap: BTreeMap<_, _> = fdims.iter().map(|(v, n)| (*v, *n)).collect();
    let analyses = Model::ALL.iter().map(|m| propagate_with(&core, &prog, func, &dmap, &env, &r, *m)).collect();
    Case { measured: gap(&reference, &got), analyses }
}

fn f(name: &str) -> Format {
    Format::parse(name).unwrap()
}

const E4M3: &str = "float(p=4,emin=-6,emax=8,specials=none,underflow=+0)";

/// Every site of `r`'s program a realization names here: the ones the
/// envelope saw.
fn sites(env: &Envelope) -> Vec<String> {
    env.sites.keys().cloned().collect()
}

/// The realizations of a systolic tile and of the specification's MLP and
/// layer norm in binary16 and E4M3, their formats stated per site: each
/// rule's bound is above the gap, and finite.
#[test]
fn a_bound_in_every_format_is_above_the_gap() {
    let f16 = Dtype::of_name("f16").unwrap();
    let e4m3 = Dtype::of_name(E4M3).unwrap();
    // Every site computing in `arith`, its sums' products and additions in
    // `product` and `accum`, under `order`.
    let computing = |default: Dtype, arith: &str, product: Option<&str>, accum: Option<&str>, order: ReductionOrder| {
        let (arith, product, accum) = (arith.to_string(), product.map(str::to_string), accum.map(str::to_string));
        move |env: &Envelope| {
            let mut r = Realization::uniform(default);
            for s in sites(env) {
                r.arith.insert(s.clone(), f(&arith));
                if let Some(p) = &product {
                    r.product.insert(s.clone(), f(p));
                }
                if let Some(a) = &accum {
                    r.accum.insert(s, f(a));
                }
            }
            r.order = order;
            r
        }
    };
    type Make = Box<dyn Fn(&Envelope) -> Realization>;
    type Row<'a> = (&'a str, &'a str, &'a [(&'a str, u64)], &'a str, Make);
    let cases: Vec<Row> = vec![
        ("examples/tests/formats/tile.psk", "tile", &[], "the tile's realization", Box::new(|_: &Envelope| Realization::parse("@examples/tests/formats/e4m3-f16-chunked.json").unwrap())),
        ("examples/tests/formats/tile.psk", "tile", &[], "f16", Box::new(move |_: &Envelope| Realization::uniform(f16))),
        ("examples/tests/formats/tile.psk", "tile", &[], "E4M3, f16 products, f32 sums", Box::new(computing(e4m3, "f32", Some("f16"), Some("f32"), ReductionOrder::Sequential))),
        ("examples/tests/formats/tile.psk", "tile", &[], "f16 in f16, chunked 8", Box::new(computing(f16, "f16", None, None, ReductionOrder::Chunked { size: 8 }))),
        ("examples/spec/mlp/mlp.psk", "mse", &[("D", 4), ("B", 2)], "f16 in f16", Box::new(computing(f16, "f16", None, None, ReductionOrder::Sequential))),
        ("examples/spec/mlp/mlp.psk", "mse", &[("D", 4), ("B", 2)], "f32 in bf16", Box::new(computing(Dtype::F32, "bf16", None, None, ReductionOrder::Sequential))),
        ("examples/spec/layernorm/layernorm.psk", "LayerNorm{8}.apply", &[("batch", 3)], "f16", Box::new(move |_: &Envelope| Realization::uniform(f16))),
        ("examples/spec/layernorm/layernorm.psk", "LayerNorm{8}.apply", &[("batch", 3)], "f16 in f16", Box::new(computing(f16, "f16", None, None, ReductionOrder::Sequential))),
    ];
    let mut report = String::from("\n");
    for (rel, entry, dims, what, make) in cases {
        let c = case(rel, entry, dims, &make);
        for (m, a) in Model::ALL.iter().zip(&c.analyses) {
            let b = a.output.err;
            assert!(b.is_finite() && a.unbounded.is_empty(), "{rel} {entry} under {what}: the {} bound is {b:e}, unbounded at {:?}", m.name(), a.unbounded);
            assert!(!m.is_bound() || b >= c.measured, "{rel} {entry} under {what}: measured {:.3e} over the {} bound {b:.3e}", c.measured, m.name());
        }
        report.push_str(&format!("{entry:<22} {what:<30} measured {:.3e}, bound {:.3e} (cond)\n", c.measured, c.analyses[2].output.err));
    }
    eprintln!("{report}");
}

/// A dense layer on integers: its activation in eight unsigned bits by
/// `minmax`, its weights in eight bits with a scale per column, summed in a
/// wrapping `i32` counted in the scales' product. The sum is exact, since
/// 16 terms of at most 255·127 stay inside `i32`, so the bound is the
/// activation's and the weights' half steps carried through it, and the
/// stores'; above the gap. In a wrapping `i16` the count may wrap: unbounded.
#[test]
fn an_integer_sum_is_exact_under_its_precondition() {
    let make = |accum: &'static str| {
        move |env: &Envelope| {
            let mut r = Realization::parse("@examples/tests/formats/w8a8.json").unwrap();
            for (site, s) in r.scaled.iter_mut() {
                if let Scale::Scheme(scheme) = s {
                    let range = env.range.get(site).map(|r| (r[0], r[1]));
                    *s = Scale::Static(scheme.resolve(env.get(site).unwrap(), range).unwrap());
                }
            }
            r.accum.insert("Dense.apply/%4".into(), f(accum));
            r
        }
    };
    let c = case("examples/tests/formats/w8a8.psk", "layer", &[], &make("i32"));
    let a = &c.analyses[2];
    assert!(a.unbounded.is_empty(), "{:?}", a.unbounded);
    assert!(a.output.err >= c.measured && a.output.err.is_finite(), "measured {:.3e}, bound {:.3e}", c.measured, a.output.err);
    // The sum's own term is nothing: the bound in a wider integer, or in
    // sixteenths of the unit, is the same.
    let wide = case("examples/tests/formats/w8a8.psk", "layer", &[], &make("i48"));
    assert_eq!(wide.analyses[2].output.err, a.output.err);
    let wide = case("examples/tests/formats/w8a8.psk", "layer", &[], &make("fixed(i32,frac=4)"));
    assert_eq!(wide.analyses[2].output.err, a.output.err);
    let narrow = case("examples/tests/formats/w8a8.psk", "layer", &[], &make("i16"));
    let a16 = &narrow.analyses[2];
    assert!(a16.output.err.is_infinite() && a16.unbounded == ["Dense.apply/%4"], "{:?} {:e}", a16.unbounded, a16.output.err);
    eprintln!("w8a8: measured {:.3e}, bound {:.3e}", c.measured, a.output.err);
}
