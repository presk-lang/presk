//! How `presk run` prints a value whose static type is `bool[S]` (D99's
//! addendum, plan 0.10.0 step 8): `bool[…]` of `true` and `false`, in its
//! text and in `--json` (JSON booleans, no `bound`), a result's leaves and
//! the inputs `--verbose` and `--json` show, from the entry's signature, on
//! every executor. At run time a mask stays an index tensor of bound 2.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap()
}

fn run(args: &[&str]) -> String {
    let o = Command::new(env!("CARGO_BIN_EXE_presk")).current_dir(root()).arg("run").arg("tests/data/masks.psk").args(args).env("NO_COLOR", "1").output().expect("running presk");
    assert!(o.status.success(), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).to_string()
}

#[test]
fn a_mask_is_printed_as_truth_values() {
    let mut text = String::new();
    for args in [
        &["--entry", "causal", "--dim", "T=3"][..],
        &["--entry", "causal", "--dim", "T=2", "--json"],
        &["--entry", "masked", "--dim", "T=2", "--verbose", "--digits", "6"],
        &["--entry", "masked", "--dim", "T=2", "--json", "--digits", "6"],
    ] {
        let out = run(args);
        for backend in ["kernel", "native"] {
            let mut b = args.to_vec();
            b.extend(["--backend", backend, "--realize", "f64"]);
            // The same printing natively: the mask's leaves, and the reals
            // to the six digits printed.
            assert_eq!(run(&b).replace(" at 64 bits", " at 256 bits").replace("\"bits\": 64", "\"bits\": 256"), out, "{b:?}");
        }
        text.push_str(&format!("$ presk run tests/data/masks.psk {}\n{out}\n", args.join(" ")));
    }
    insta::assert_snapshot!(text);
}
