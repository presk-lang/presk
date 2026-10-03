//! The dead code of a gradient goes after automatic differentiation (plan
//! 0.10.0 step 3, `presk_core::ad::prune`). What survives must be computed as
//! before, and a fusion group is part of how: a value inside a group stays
//! in the kernel's arithmetic type, `f32` for `bf16`, where a value that
//! reaches memory is rounded to its storage type. So removing a reader of a
//! value must not bring it into a group, nor a group lose or gain a member:
//! every function the pass rewrites has, over the `let`s its result reads,
//! the groups it had without the pass.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use presk_core::VarId;
use presk_realize::plan;
use presk_syntax::diag::render_all;
use presk_syntax::{SourceMap, parse_file};
use presk_types::check;
use presk_types::elab::{self, Options};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

/// The program after AD with the pass, and without it.
fn both(rel: &str, consts: &[(&str, u64)]) -> (presk_core::Program, presk_core::Program) {
    let mut sm = SourceMap::new();
    let id = sm.load(&repo_root().join(rel)).unwrap();
    let (program, diags) = parse_file(&sm, id);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let opts = Options { const_overrides: consts.iter().map(|(n, v)| (n.to_string(), *v)).collect() };
    let (prog, diags) = elab::elaborate(&mut sm, id, program, &opts);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (typed, diags) = check::check(&sm, &prog);
    assert!(diags.iter().all(|d| !d.is_error()), "{}", render_all(&diags, &sm, false));
    let (core, diags) = presk_core::lower(&prog, &typed);
    assert!(diags.is_empty());
    let (mut pruned, mut whole) = (core.clone(), core);
    // Recomputing, as step 3 did: storing (step 4) splits the `.vjp`s.
    assert!(presk_core::ad::transform_with(&mut pruned, true).is_empty());
    assert!(presk_core::ad::transform_with(&mut whole, false).is_empty());
    presk_core::verify(&pruned).unwrap_or_else(|e| panic!("{rel}: {}", e.join("; ")));
    (pruned, whole)
}

const TINY: &[(&str, u64)] = &[("V", 64), ("D", 16), ("H", 2), ("L", 2)];

#[test]
fn pruning_keeps_every_fusion_group() {
    let programs: &[(&str, &[(&str, u64)])] = &[
        ("tests/examples/mlp/mlp.psk", &[]),
        ("tests/plan/minigpt_grad.psk", TINY),
        ("experiments/gpt_grad.psk", &[]),
        ("tests/data/stack_grad.psk", &[]),
        ("tests/data/stack_prefix_grad.psk", &[]),
        ("tests/data/dropout_grad.psk", &[]),
        ("tests/data/stack_keys.psk", &[]),
        ("tests/data/merge_grad.psk", &[]),
        ("tests/data/split_grad.psk", &[]),
        ("tests/data/closure_grad.psk", &[]),
    ];
    let (mut functions, mut groups, mut removed) = (0, 0, 0);
    let mut bad: Vec<String> = vec![];
    for (rel, consts) in programs {
        let (pruned, whole) = both(rel, consts);
        assert_eq!(pruned.funcs.len(), whole.funcs.len(), "{rel}: the pass adds and removes no function");
        for (f, w) in pruned.funcs.iter().zip(&whole.funcs) {
            assert_eq!(f.name, w.name, "{rel}: functions keep their names and places");
            if f.body.lets.len() == w.body.lets.len() {
                continue;
            }
            functions += 1;
            removed += w.body.lets.len() - f.body.lets.len();
            // What the result reads, directly or not: the values that must
            // be computed as before. A `let` the pass holds, read by nothing,
            // is there only as a reader of another.
            let mut kept: HashSet<VarId> = HashSet::from([f.body.result]);
            for l in f.body.lets.iter().rev() {
                if kept.contains(&l.var) {
                    kept.extend(l.rhs.vars());
                }
            }
            let (p, q) = (plan(f), plan(w));
            for g in p.groups.values().filter(|g| kept.contains(&g.root)) {
                match q.groups.get(&g.root) {
                    None => bad.push(format!("{rel} {}: %{} roots a group it did not root", f.name, g.root.0)),
                    Some(b) if b.members != g.members => bad.push(format!("{rel} {}: the group of %{} gained or lost a member", f.name, g.root.0)),
                    _ => groups += 1,
                }
            }
            for g in q.groups.values().filter(|g| kept.contains(&g.root)) {
                if !p.groups.contains_key(&g.root) {
                    bad.push(format!("{rel} {}: the group of %{} is no longer one", f.name, g.root.0));
                }
            }
            for v in &kept {
                if p.interior.contains(v) != q.interior.contains(v) {
                    bad.push(format!("{rel} {}: %{} moved into or out of a group", f.name, v.0));
                }
            }
        }
    }
    eprintln!("{functions} functions rewritten, {removed} lets removed, {groups} groups kept");
    assert!(bad.is_empty(), "{} changes:\n{}", bad.len(), bad.join("\n"));
    assert!(functions >= 50 && groups >= 40, "the programs exercise the pass");
}
