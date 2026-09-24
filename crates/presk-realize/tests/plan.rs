//! The invariants that make a fusion group safe to launch as one kernel: an
//! interior value is read only inside its own group, so it never has to reach
//! memory; the members share a loop nest; and the steps are in order.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use presk_core::ir::Rhs;
use presk_core::VarId;
use presk_realize::{Src, plan};
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, Options};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn core_of(rel: &str, consts: &[(&str, u64)]) -> presk_core::Program {
    let mut sm = SourceMap::new();
    let id = sm.load(&repo_root().join(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let opts = Options { const_overrides: consts.iter().map(|(n, v)| (n.to_string(), *v)).collect() };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (mut core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty());
    let diags = presk_core::ad::transform(&mut core);
    assert!(diags.is_empty());
    core
}

#[test]
fn groups_are_safe_on_the_corpus_and_the_examples() {
    let programs = [
        ("corpus/nlp/05_gpt.psk", &[("V", 92), ("D", 64), ("H", 2), ("L", 2)][..]),
        ("corpus/nlp/03_mlp.psk", &[][..]),
        ("tests/examples/minigpt/train_minigpt.psk", &[("V", 64), ("D", 16), ("H", 2), ("L", 2)][..]),
        ("tests/data/dropout_grad.psk", &[][..]),
    ];
    let mut total_groups = 0;
    for (rel, consts) in programs {
        let core = core_of(rel, consts);
        for f in &core.funcs {
            let p = plan(f);
            total_groups += p.groups.len();
            let by_var: std::collections::HashMap<VarId, &presk_core::ir::Let> = f.body.lets.iter().map(|l| (l.var, l)).collect();
            let mut seen: HashSet<VarId> = HashSet::new();
            for g in p.groups.values() {
                assert!(g.members.len() >= 2, "{rel} {}: a group of one is not a group", f.name);
                assert_eq!(*g.members.last().unwrap(), g.root, "{rel} {}: the root is the last member", f.name);
                let members: HashSet<VarId> = g.members.iter().copied().collect();
                for m in &g.members {
                    assert!(seen.insert(*m), "{rel} {}: %{} is in two groups", f.name, m.0);
                    assert_eq!(by_var[m].ty, g.ty, "{rel} {}: members share the group's type", f.name);
                    if *m == g.root {
                        continue;
                    }
                    // An interior value never reaches memory, so everything
                    // that reads it must be inside the group.
                    assert!(p.interior.contains(m), "{rel} {}: %{} is a member but not interior", f.name, m.0);
                    assert_ne!(*m, f.body.result, "{rel} {}: the result cannot be interior", f.name);
                    for l in &f.body.lets {
                        let reads = match &l.rhs {
                            Rhs::Prim { args, .. } => args.contains(m),
                            _ => false,
                        };
                        let other_kind = !matches!(l.rhs, Rhs::Prim { .. });
                        assert!(!reads || members.contains(&l.var), "{rel} {}: %{} escapes to %{}", f.name, m.0, l.var.0);
                        assert!(!(other_kind && rhs_reads(&l.rhs, *m)), "{rel} {}: %{} escapes to a non-primitive", f.name, m.0);
                    }
                }
                // A step may only read inputs and strictly earlier steps.
                for (i, s) in g.steps.iter().enumerate() {
                    for a in &s.args {
                        if let Src::Step(k) = a {
                            assert!(*k < i, "{rel} {}: step {i} reads step {k}", f.name);
                        }
                        if let Src::Input(k) = a {
                            assert!(*k < g.inputs.len(), "{rel} {}: step {i} reads input {k}", f.name);
                        }
                    }
                }
                assert_eq!(g.steps.len(), g.members.len(), "one step per member");
            }
        }
    }
    assert!(total_groups > 10, "the corpus should fuse something: {total_groups} groups");
}

fn rhs_reads(rhs: &Rhs, v: VarId) -> bool {
    match rhs {
        Rhs::Lit(_) => false,
        Rhs::Prim { args, .. } | Rhs::Tuple(args) | Rhs::Call { args, .. } | Rhs::Closure { captured: args, .. } => args.contains(&v),
        Rhs::Proj(x, _) | Rhs::Field(x, _) | Rhs::Grad(x) => *x == v,
        Rhs::Record(fields) => fields.iter().any(|(_, x)| *x == v),
        Rhs::Apply { fun, args } => *fun == v || args.contains(&v),
        Rhs::Fold { stacked, init, extra, .. } => stacked.contains(&v) || *init == v || extra.contains(&v),
    }
}
