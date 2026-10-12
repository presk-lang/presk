//! The role of every site of a program, as `presk solve` takes them since
//! plan 0.11.0 step 2b (D112): `cargo run -p presk-driver --example roles --
//! FILE.psk [NAME=N …]`, the pairs being the file's `--const`s. One line per
//! site: its name, `decision`, `carried`, or `follows` and the sites it
//! follows.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((file, consts)) = args.split_first() else {
        eprintln!("usage: roles FILE.psk [NAME=N …]");
        std::process::exit(2);
    };
    let mut report = presk_driver::Stderr { color: false };
    let Ok((_, _, mut core)) = presk_driver::front::core(std::path::Path::new(file), consts, false, &mut report) else {
        std::process::exit(1);
    };
    // As `presk solve` differentiates.
    presk_driver::ad::differentiate(&mut core, presk_core::ad::Remat::Store, true);
    for (site, role) in presk_precision::solve::Roles::of(&core).0 {
        match role {
            presk_precision::solve::Role::Decision => println!("{site}\tdecision"),
            presk_precision::solve::Role::Carried => println!("{site}\tcarried"),
            presk_precision::solve::Role::Follows(l) => println!("{site}\tfollows\t{}", l.join(" ")),
        }
    }
}
