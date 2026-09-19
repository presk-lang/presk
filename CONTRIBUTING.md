# Contributing

## What this repository is

`github.com/presk-lang/presk` is the public copy of the Presk compiler. It
holds **one commit per version**, each the tree of a tagged version of the
development repository, exported by a script of that repository: the
tagged tree without the coding agents' instructions, without the
development records under `docs/` (the files the manual names and the
design documents stay) and without the export itself, with the licence and
citation files laid over it. The intermediate commits of the development
are not here, by design; the history of a version is its section of
`CHANGELOG.md`, which is written as the version is made and names the
decisions each number comes from.

The development happens in a private repository, with merge requests per
step and the discipline the manual describes (every change exact unless
declared, every number from a record). The public `main` only ever
receives release commits.

## How the code was written

The compiler was written by its author with AI coding assistants (Claude
Code, OpenAI Codex, Mistral Vibe and Google's Antigravity), under the
author's direction and review: the language's design and every decision of
record are the author's; the assistants wrote code, tests and documents
from them. Every claim about the compiler's behaviour is checked by the
test suite (`make test`, the snapshots under `crates/*/tests/snapshots/`)
and by the experiments' scripts, whose outputs are the records
`CHANGELOG.md` cites.

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
