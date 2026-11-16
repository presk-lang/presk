# Contributing

## Building and testing

```sh
make build            # cargo build --workspace (GMP and MPFR compiled once; m4 and a C compiler)
make test             # cargo test --workspace
make check-examples   # presk check on the example programs
make doc              # the manual (rustdoc of crates/presk-manual)
```

The manual's `install` chapter has the requirements and the layout of the
repository; its `tutorial` chapter is the place to start. Two tests of the
CLI plan under the frozen layout model of the authors' measurement machine
(`~/presk-data/ra4-model-0.8.0/profile.json`); elsewhere, point
`RA4_FROZEN` at the copy the repository carries,
`crates/presk-explain/tests/data/x925-frozen.json` (an absolute path), as
the CI does. The CI (`.github/workflows/ci.yml`) runs the build, the tests
and the example programs on Linux and macOS; it measures nothing.

## Issues and pull requests

Issues are welcome: a program the compiler mishandles, a diagnostic that
misleads, a number of the manual that does not reproduce at the stated
scale. Give the version (`presk --version`), the program, and the command.

Pull requests are read and, when taken, applied to the development
repository, where they are tested against every example and record; the
change then appears in the next release's commit, credited in its section
of `CHANGELOG.md`. A pull request is not merged into the public `main`
directly, since `main` is replaced by each export. By submitting a
contribution you agree that it is licensed under the Apache License,
Version 2.0, like the rest of the compiler (`LICENSE`, `NOTICE`).

## The export, for the maintainers

The export lives in the development repository (`make publish`), with its
two lists — the forbidden patterns and the private spellings it substitutes
in the records — which name what a public tree must not contain. It
appends the versions not yet public to the public `main` and tags them,
audits every public tree against the forbidden patterns and prints the
`git push` commands; it never pushes. The same tags and rules give the same
commits on any machine. Commits already public are never rebuilt, except by
a deliberate rebuild of the whole chain for a change of the filter, which
is then force-pushed. The crates are published to crates.io from the
public checkout at each version, `cargo publish --workspace`, in the
workspace's dependency order; the package `presk` (in `crates/presk-cli`)
is the binary, `cargo install presk`.
