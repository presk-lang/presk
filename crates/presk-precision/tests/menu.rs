//! One menu (plan 0.17.0 step 4, D177): a parameter of rank two or more is
//! offered `int8` and `int4` beside its float types, in the solve itself.
//! Without integer rungs nothing moves (QM0); with them every move of the
//! solver is today's, over one list of rungs.

use std::collections::BTreeMap;

use presk_precision::envelope::{Envelope, Rounding};
use presk_precision::sensitivity::Sensitivity;
use presk_precision::solve::{self, Budget, IntRung, Menu, Move, Narrow, Options, ParamSite, Role, Roles, Rule, Rung};
use presk_rt::{AxisRef, Dtype, Quant, QuantScheme, Realization};

const U32: f64 = f32::EPSILON as f64 / 2.0;
const U16: f64 = 1.0 / 256.0;

/// Sites of magnitude 1 and sensitivity 1, with these elements, their
/// rounding the unit term's.
fn envelope(sites: &[(&str, u64)]) -> (Envelope, Sensitivity) {
    let rounds = [Dtype::F64.unit_roundoff(), U32, U16];
    let mut e = Envelope::new("p.psk", "f", 128, "2026-10-28");
    e.observe(&sites.iter().map(|(s, _)| (s.to_string(), 1.0)).collect());
    e.observe_elems(&sites.iter().map(|(s, n)| (s.to_string(), *n)).collect());
    e.observe_rounding(&sites.iter().map(|(s, _)| (s.to_string(), rounds)).collect());
    let sens = Sensitivity { sites: sites.iter().map(|(s, _)| (s.to_string(), 1.0)).collect() };
    (e, sens)
}

fn rung(bits: u8, scheme: QuantScheme, round: f64, bytes: f64) -> IntRung {
    IntRung { bits, scheme, axis: -2, contracted: true, round, bytes }
}

/// A parameter of `elems` elements, magnitude and sensitivity 1, its float
/// roundings the unit term's, read where `fields` say.
fn param(path: &str, elems: f64, fields: &[&str], ints: Vec<IntRung>) -> ParamSite {
    ParamSite { path: path.into(), elems, mag: 1.0, sens: 1.0, fields: fields.iter().map(|f| f.to_string()).collect(), round: Some(Rounding::of([0.0, U32, U16])), ints }
}

/// A matrix of 1000 elements: `int8` stores 1100 bytes, `int4` 600, each
/// a smaller move under `absmax` than under `pow2`.
fn matrix() -> Vec<IntRung> {
    vec![
        rung(8, QuantScheme::Absmax, 2e-3, 1100.0),
        rung(8, QuantScheme::Pow2, 3e-3, 1100.0),
        rung(4, QuantScheme::Absmax, 3e-2, 600.0),
        rung(4, QuantScheme::Pow2, 5e-2, 600.0),
    ]
}

fn one_menu() -> Menu {
    Menu::parse("f64,f32,bf16,int8,int4").unwrap()
}

fn solve(env: &Envelope, sens: &Sensitivity, roles: &Roles, menu: &Menu, budget: f64, options: Options, verify: Option<&mut dyn FnMut(&Realization) -> f64>) -> solve::Solution {
    solve::solve_with(env, sens, roles, "f/%1", Budget::Absolute(budget), menu, Dtype::F64, &solve::Traffic, verify, options)
}

fn quant(bits: u8, scheme: QuantScheme) -> Quant {
    Quant { bits, over: AxisRef::Pos(-2), scheme }
}

#[test]
fn a_menu_is_parsed_in_one_order() {
    assert_eq!(Menu::parse("f64,f32,bf16").unwrap(), Menu::default());
    assert_eq!(Menu::parse("bf16, f64,f32,f32").unwrap(), Menu::default());
    let m = one_menu();
    assert_eq!(m.1, [(8, QuantScheme::Absmax), (8, QuantScheme::Pow2), (4, QuantScheme::Absmax), (4, QuantScheme::Pow2)]);
    assert_eq!(m.rungs().map(|r| r.name()).collect::<Vec<_>>(), ["f64", "f32", "bf16", "int8", "int8/pow2", "int4", "int4/pow2"]);
    assert_eq!(Menu::parse("int4/pow2,f64,int8/absmax").unwrap().1, [(8, QuantScheme::Absmax), (4, QuantScheme::Pow2)]);
    assert!(Menu::parse("int8,int4").is_err(), "a menu needs a float type");
    assert!(Menu::parse("f64,f16").is_err());
    // A rung's name is its float type's, so that no output moves (QM0).
    assert_eq!(Rung::from(Dtype::Bf16).to_string(), Dtype::Bf16.to_string());
}

/// QM0 in the solver: a parameter priced in integers and a menu without
/// them solve as a parameter without integers, decision for decision, move
/// for move, through the measurement's widening and the narrowing step.
#[test]
fn without_integer_rungs_nothing_moves() {
    let (env, sens) = envelope(&[("f/%1", 300), ("f/%2", 200)]);
    let with = Roles::default().with_params(vec![param("θ.w", 1000.0, &[], matrix())]);
    let without = Roles::default().with_params(vec![param("θ.w", 1000.0, &[], Vec::new())]);
    for (budget, narrow) in [(1.0, Narrow::Off), (1e-2, Narrow::Off), (1e-3, Narrow::Measured), (1e-6, Narrow::Off)] {
        let options = Options { narrow, margin: 1.0, ..Options::default() };
        let run = |roles: &Roles| {
            let mut m = |r: &Realization| if r.storage.values().chain(r.params.values()).any(|d| *d == Dtype::Bf16) { 1.0 } else { 0.0 };
            solve(&env, &sens, roles, &Menu::default(), budget, options, Some(&mut m))
        };
        let (a, b) = (run(&with), run(&without));
        assert_eq!(a.realization, b.realization, "at {budget}");
        assert!(a.realization.quant.is_empty());
        assert_eq!(a.moves, b.moves, "at {budget}");
        assert_eq!((a.bytes, a.proposed_bytes, a.relaxation_bytes, a.estimated_error, a.rounds), (b.bytes, b.proposed_bytes, b.relaxation_bytes, b.estimated_error, b.rounds));
        let costs = |roles: &Roles, r: &Realization| solve::breakdown(&env, &sens, roles, r, &Menu::default(), Dtype::F64, &solve::Traffic).into_iter().map(|c| (c.name, c.rung, c.err, c.cost, c.bytes, c.int.is_none())).collect::<Vec<_>>();
        assert_eq!(costs(&with, &a.realization), costs(&without, &b.realization));
    }
}

/// The proposal takes a rung where it saves the most per unit of term,
/// floats and integers in one list: `int8` before `bf16` on the matrix,
/// whose `int8` moves less than `bf16` rounds and saves more.
#[test]
fn the_proposal_takes_integers_where_they_save_most() {
    let (env, sens) = envelope(&[("f/%1", 300)]);
    let roles = Roles::default().with_params(vec![param("θ.w", 1000.0, &[], matrix())]);
    let menu = one_menu();
    let s = solve(&env, &sens, &roles, &menu, 1.0, Options::default(), None);
    assert_eq!(s.realization.quant.get("θ.w"), Some(&quant(4, QuantScheme::Absmax)), "absmax, the smaller move of the two int4");
    assert_eq!(s.realization.params.get("θ.w"), None, "integers replace a float type at rest");
    assert_eq!(s.realization.storage.get("f/%1"), Some(&Dtype::Bf16));
    assert_eq!(s.bytes, 600.0 + 600.0);
    let s = solve(&env, &sens, &roles, &menu, 1e-2, Options::default(), None);
    assert_eq!(s.realization.quant.get("θ.w"), Some(&quant(8, QuantScheme::Absmax)));
    assert_eq!(s.realization.storage.get("f/%1"), Some(&Dtype::Bf16));
    assert_eq!(s.bytes, 1100.0 + 600.0);
    // The int8 under pow2 saves no byte more than absmax's: never tried
    // once absmax is taken, and bf16, wider, neither; both int4 are, and
    // past the budget.
    let tried: Vec<(Rung, bool)> = s.moves.iter().filter_map(|m| if let Move::Tried { site, to, taken, .. } = m { (site == "θ.w").then_some((*to, *taken)) } else { None }).collect();
    let int = |bits, scheme| Rung::Int { bits, scheme };
    assert_eq!(tried, [(Rung::Float(Dtype::F32), true), (int(8, QuantScheme::Absmax), true), (int(4, QuantScheme::Absmax), false), (int(4, QuantScheme::Pow2), false)]);
    let s = solve(&env, &sens, &roles, &menu, 2.5e-3, Options::default(), None);
    assert_eq!(s.realization.quant.get("θ.w"), Some(&quant(8, QuantScheme::Absmax)));
    assert_eq!(s.realization.storage.get("f/%1"), Some(&Dtype::F32));
    let s = solve(&env, &sens, &roles, &menu, 1e-3, Options::default(), None);
    assert!(s.realization.quant.is_empty());
    assert_eq!(s.realization.params.get("θ.w"), Some(&Dtype::F32));
    // The relaxation floor is every value at its narrowest rung.
    assert_eq!(s.relaxation_bytes, 600.0 + 600.0);
}

/// A parameter's term in integers is the larger of their move at rest and
/// the rounding the narrowest nest reading it gives it; in a float type,
/// the narrowest of its type and the readers', as before.
#[test]
fn the_term_of_integers_is_the_larger_of_their_move_and_the_readers_rounding() {
    let (env, sens) = envelope(&[("f/%1", 300)]);
    let roles = Roles(BTreeMap::from([("f/%9".to_string(), Role::Follows(vec!["f/%1".into()]))]), vec![param("θ.w", 1000.0, &["f/%9"], matrix())]);
    let menu = one_menu();
    let term = |nest: Dtype, q: Option<Quant>| {
        let mut r = Realization::uniform(Dtype::F64);
        r.storage.insert("f/%1".into(), nest);
        match q {
            Some(q) => {
                r.quant.insert("θ.w".into(), q);
            }
            None => {
                r.params.insert("θ.w".into(), Dtype::F32);
            }
        }
        let b = solve::breakdown(&env, &sens, &roles, &r, &menu, Dtype::F64, &solve::Traffic);
        b.into_iter().find(|c| c.name == "θ.w").unwrap()
    };
    let c = term(Dtype::Bf16, Some(quant(8, QuantScheme::Absmax)));
    assert_eq!(c.err, U16, "the bf16 nest rounds it more than int8 moves it");
    assert_eq!((c.rung, c.bytes), (Rung::Int { bits: 8, scheme: QuantScheme::Absmax }, 1100.0));
    assert_eq!(c.int.as_ref().map(|q| (q.axis, q.contracted)), Some((-2, true)));
    assert_eq!(term(Dtype::F32, Some(quant(8, QuantScheme::Absmax))).err, 2e-3);
    assert_eq!(term(Dtype::Bf16, Some(quant(4, QuantScheme::Absmax))).err, 3e-2);
    assert_eq!(term(Dtype::Bf16, None).err, U16, "f32 at rest, bf16 at the nest's edge");
    // A quant entry the solver did not price, a `--quant` realization's, is
    // read as the float type it replaced.
    let c = term(Dtype::F32, Some(quant(4, QuantScheme::Absmax)));
    assert!(c.int.is_some());
    let unpriced = Roles(roles.0.clone(), vec![param("θ.w", 1000.0, &["f/%9"], Vec::new())]);
    let mut r = Realization::uniform(Dtype::F64);
    r.quant.insert("θ.w".into(), quant(4, QuantScheme::Absmax));
    let c = solve::breakdown(&env, &sens, &unpriced, &r, &Menu::default(), Dtype::F64, &solve::Traffic).into_iter().find(|c| c.name == "θ.w").unwrap();
    assert_eq!((c.rung, c.bytes, c.int.is_none()), (Rung::Float(Dtype::F64), 8000.0, true));
}

/// Rule 2 reads the first float step, `f32`, whatever the integers would
/// save: a parameter under θ·B is offered no rung, one above it all.
#[test]
fn rule_2_on_either_side_of_its_threshold_with_integer_rungs() {
    let theta = Options { min_gain: 1e-2, ..Options::default() };
    let small = |n: f64| vec![rung(8, QuantScheme::Absmax, 1e-3, n + 20.0), rung(4, QuantScheme::Absmax, 1e-2, n / 2.0 + 20.0)];
    let menu = Menu::parse("f64,f32,bf16,int8/absmax,int4/absmax").unwrap();
    // B = 8·(1000 + 20); f32 saves 80 < θ·B = 81.6.
    let (env, sens) = envelope(&[("f/%1", 1000)]);
    let roles = Roles::default().with_params(vec![param("θ.s", 20.0, &[], small(20.0))]);
    let s = solve(&env, &sens, &roles, &menu, 1.0, theta, None);
    let e = &s.offers.excluded;
    assert_eq!(e.iter().map(|e| (e.name.as_str(), e.rule)).collect::<Vec<_>>(), [("θ.s", Rule::MinGain)]);
    assert_eq!((e[0].gain, e[0].forgone), (80.0, 160.0 - 30.0), "its most is its int4");
    assert!(s.realization.quant.is_empty() && s.realization.params.is_empty());
    // 21 elements: 84 ≥ 81.68, offered, and int4 at a loose budget.
    let roles = Roles::default().with_params(vec![param("θ.s", 21.0, &[], small(21.0))]);
    let s = solve(&env, &sens, &roles, &menu, 1.0, theta, None);
    assert!(s.offers.excluded.is_empty());
    assert_eq!(s.realization.quant.get("θ.s"), Some(&quant(4, QuantScheme::Absmax)));
}

/// The measurement widens one rung at a time: int4 to int8, the smaller
/// move of the two, and int8 to bf16.
#[test]
fn the_measurement_widens_integers_one_rung_at_a_time() {
    let (env, sens) = envelope(&[("f/%1", 300)]);
    let roles = Roles::default().with_params(vec![param("θ.w", 1000.0, &[], matrix())]);
    let menu = one_menu();
    let mut no_int4 = |r: &Realization| if r.quant.values().any(|q| q.bits == 4) { 2.0 } else { 0.0 };
    let s = solve(&env, &sens, &roles, &menu, 1.0, Options::default(), Some(&mut no_int4));
    assert_eq!(s.rounds, 1);
    assert_eq!(s.realization.quant.get("θ.w"), Some(&quant(8, QuantScheme::Absmax)));
    let mut no_int = |r: &Realization| if r.quant.is_empty() { 0.0 } else { 2.0 };
    let s = solve(&env, &sens, &roles, &menu, 1.0, Options::default(), Some(&mut no_int));
    let widened: Vec<(&str, String, String)> = s.moves.iter().filter_map(|m| if let Move::Widened { site, from, to, .. } = m { Some((site.as_str(), from.name(), to.name())) } else { None }).collect();
    // The second round widens the nest, whose share (bf16's) is then the
    // larger; the third the matrix again.
    assert_eq!(widened, [("θ.w", "int4".into(), "int8".into()), ("f/%1", "bf16".into(), "f32".into()), ("θ.w", "int8".into(), "bf16".into())]);
    assert_eq!(s.realization.params.get("θ.w"), Some(&Dtype::Bf16));
    assert_eq!(s.measured, Some(0.0));
}

/// The narrowing step goes down the same list, one rung a pass: bf16, then
/// int8 under absmax, then int4.
#[test]
fn the_narrowing_step_narrows_into_integers() {
    let (env, sens) = envelope(&[("f/%1", 300)]);
    let roles = Roles::default().with_params(vec![param("θ.w", 1000.0, &[], matrix())]);
    let mut zero = |_: &Realization| 0.0;
    let s = solve(&env, &sens, &roles, &one_menu(), 1e-3, Options { narrow: Narrow::Measured, margin: 1.0, ..Options::default() }, Some(&mut zero));
    let steps: Vec<(usize, String, String)> = s.moves.iter().filter_map(|m| if let Move::Narrowed { pass, site, from, to, kept: true, .. } = m { (site == "θ.w").then(|| (*pass, from.name(), to.name())) } else { None }).collect();
    assert_eq!(steps, [(1, "f32".into(), "bf16".into()), (2, "bf16".into(), "int8".into()), (3, "int8".into(), "int4".into())]);
    assert_eq!(s.realization.quant.get("θ.w"), Some(&quant(4, QuantScheme::Absmax)));
}

/// The exhaustive search runs over every rung of every value offered: a
/// nest's three float types, the matrix's seven rungs; and the greedy
/// proposal finds its optimum on this program at every budget.
#[test]
fn the_exhaustive_search_runs_over_the_rungs() {
    let (env, sens) = envelope(&[("f/%1", 300)]);
    let roles = Roles::default().with_params(vec![param("θ.w", 1000.0, &[], matrix())]);
    let menu = one_menu();
    let search = |budget: f64, states: u64| solve::exhaustive_with(&env, &sens, &roles, "f/%1", Budget::Absolute(budget), &menu, Dtype::F64, &solve::Traffic, states, Options::default());
    assert!(search(1.0, 20).is_none(), "3 × 7 = 21 states");
    for (budget, bytes) in [(1.0, 1200.0), (1e-2, 1700.0), (2.5e-3, 2300.0), (1e-3, 5200.0)] {
        let (r, b) = search(budget, 21).unwrap();
        assert_eq!(b, bytes, "at {budget}");
        let s = solve(&env, &sens, &roles, &menu, budget, Options::default(), None);
        assert_eq!((s.proposed_bytes, &s.realization), (b, &r), "at {budget}");
    }
}

/// The axis a parameter's readers contract, from the program: a right
/// operand of `·` its first, a left operand its last, an `ein` operand the
/// label the output drops; a gathered table none, nor a matrix both
/// gathered and contracted.
#[test]
fn the_scale_s_axis_is_the_one_every_reader_contracts() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/tests/precision/heads.psk");
    let mut sm = presk_syntax::SourceMap::new();
    let id = sm.load(&path).unwrap();
    let (program, diags) = presk_syntax::parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", presk_syntax::diag::render_all(&diags, &sm, false));
    let (prog, diags) = presk_types::elab::elaborate(&mut sm, id, program, &presk_types::elab::Options::default());
    assert!(diags.iter().all(|d| !d.is_error()), "{}", presk_syntax::diag::render_all(&diags, &sm, false));
    let (typed, diags) = presk_types::check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", presk_syntax::diag::render_all(&diags, &sm, false));
    let (core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty(), "{}", presk_syntax::diag::render_all(&diags, &sm, false));
    let (entry, _, _) = presk_interp::run::resolve_entry(&core, &prog, "Heads.apply").unwrap();
    let axes = solve::contracted_axes(&core, entry);
    assert_eq!(axes, BTreeMap::from([("w".to_string(), -2), ("m".to_string(), -1), ("p".to_string(), -1)]));
}
