# Tools and libraries

Every external tool and library the compiler relies on, what it is, and why
it was chosen. The rationale in full is in the specification repository,
`presk-specs/implementation/presk-tooling.md`; this page tracks what is
*actually in use*, with versions from `Cargo.lock`. Adding a dependency means
adding a row here and a section there.

## Build tools

| Tool | What it is | Why |
|---|---|---|
| **Rust** (rustc/cargo ≥ 1.85, edition 2024; developed with 1.92) | The implementation language and its build system. | Chosen by the specification (`presk-specs/report/chapters/08-implementation.tex`): pattern matching and traits for deep IRs, memory safety, the e-graph ecosystem (`egg`) for the later optimizer, `serde` for the serialized IR, LLVM/Cranelift bindings for code generation. Cargo builds and tests every crate; nothing else drives it. |
| **make** | GNU make, driving the `Makefile`. | A thin orchestrator over cargo for the recipes that are not a single cargo command (`check-examples`, `sync-examples`, `doc`). Preferred to `just` because it is installed everywhere the project is developed. The specification's rule: cargo for the compiler, a thin orchestrator on top, never a foreign build system driving cargo. |
| **rustdoc** (`cargo doc`) | Rust's documentation generator. | Renders this manual (crate `presk-manual`, Markdown chapters included with `#[doc = include_str!]`) and the API documentation of the crates, in one tool, with search and navigation, no extra dependency. |
| **cargo-insta** (optional CLI) | Reviewer for `insta` snapshots. | `cargo insta review` shows the diff of every changed snapshot and accepts or rejects it; without it, `INSTA_UPDATE=always cargo test` rewrites snapshots blindly. |
| **cranelift-codegen**, **-frontend**, **-module**, **-jit**, **-native** 0.131 | The Cranelift code generator (from the Wasmtime project): IR, a builder for it, linking, an in-process JIT, host ISA detection. | The native backend compiles each kernel to machine code at run time. Pure Rust, fast compilation, no LLVM installation; version 0.131 is the newest that builds with rustc 1.92 (Cranelift's minimum Rust version follows stable closely). |
| **libm** 0.2 | Pure-Rust port of the C math library. | The kernel interpreter needs `erf`/`erff` for the exact GELU and its derivative in the kernels' machine type; Rust's standard library has no `erf`. Also the functions the Cranelift backend will call. |
| **uv** (optional) | Python package and environment manager. | `make venv` creates `.venv` for the conformance tests in seconds; any way of installing PyTorch into `.venv` works. |
| **PyTorch** 2.14, **NumPy**, **pytest** (optional, in `.venv`) | The framework the export targets; its array dependency; the test runner. | Needed only to *run* the generated `test_conformance.py` and `train.py` (`make conformance`); the compiler never imports them. PyTorch's exact `F.gelu` (erf-based) and `torch.func` (`functional_call`, `grad`) are what the export relies on. |
| **onnx**, **onnxscript**, **onnxruntime** (optional, in `.venv`) | The ONNX graph format and its Python API; the library `torch.onnx.export(dynamo=True)` uses to build ONNX graphs; the reference runtime for ONNX models. | `--mode graph` writes `model.onnx` through PyTorch's exporter (which needs `onnxscript`), and the generated `test_onnx` runs it under ONNX Runtime against the interpreter; none of the three is imported by the compiler. |

## Libraries in use

| Crate | Version | Used by | What it is | Why |
|---|---|---|---|---|
| **clap** | 4.6 | `presk-cli` | Command-line argument parser (derive API). | The standard for Rust CLIs: the whole `presk` interface (`check`, `run`, `export`, `dump`, their flags, `--help`) is a set of annotated enums; value validation and help text come for free. |
| **ariadne** | 0.5 | `presk-syntax` | Renderer of source-located diagnostics (labels, notes, help, colors). | Diagnostics must be phrased in source terms and point at several places at once (a `grad` site and the `argmax` three calls away, possibly in another file). ariadne renders multi-label, multi-file reports, has an ASCII/no-color mode for stable snapshots, and byte-offset spans. Our own `Diagnostic` type is renderer-independent; ariadne is only the printer. |
| **unicode-ident** | 1 | `presk-syntax` | Unicode XID_Start/XID_Continue tables. | Presk identifiers are Unicode (`θ`, `γ`, `x̂`); the lexer needs the official identifier character classes rather than `is_alphanumeric`. |
| **unicode-normalization** | 0.1 | `presk-syntax` | NFC/NFD normalization. | `x̂` may be written precomposed or as `x` + combining circumflex; identifier names are NFC-normalized so both spellings denote the same name. |
| **num-bigint**, **num-rational**, **num-integer**, **num-traits** | 0.4 / 0.4 / 0.1 / 0.2 | `presk-types` | Arbitrary-precision integers and rationals, integer utilities (gcd/lcm), numeric traits. | Dimension expressions are linear forms with *rational* coefficients (`D / 12`), decided exactly; constants are evaluated exactly (`768 / 7` must be rejected, not rounded). Big integers avoid overflow questions in the Presburger procedure. |
| **indexmap** | 2 | `presk-types`, `presk-core` | Hash map that preserves insertion order. | Declarations, module members and parameter trees must keep their source order (dumps, parameter records, deterministic output) while being looked up by name. |
| **insta** (dev) | 1.48 | `presk-syntax`, `presk-types`, `presk-core` | Snapshot testing (`assert_snapshot!`, `glob!`). | The compiler's outputs are texts — tokens, ASTs, elaborated programs, PSK-Core, rendered diagnostics. Snapshots make every change to them visible in review, and the `tests/diag/` corpus is one glob. |
| **proptest** (dev) | 1.11 | `presk-types` | Property-based testing. | The decision procedure must be *sound*: random linear forms, facts and assignments check that a `Valid` answer is never contradicted and that a counterexample really is one. |
| **rug** (GMP + MPFR) | 1.30 (gmp-mpfr-sys 1.7) | `presk-interp` | Safe Rust bindings to the GNU arbitrary-precision libraries: `rug::Float` is an MPFR floating-point number with a chosen precision in bits and *correctly rounded* operations (`+ − × ÷`, `sqrt`, `exp`, `log`, `tanh`, `sin`, `cos`, `erf`, …). Features `integer`, `float`, `std` only (no MPC). | The reference interpreter must approximate the exact denotation over ℝ closely enough to be the oracle of everything else (conformance suites, ε-budget validation, AD versus finite differences at 512 bits). At 256 bits the per-operation error is ~2⁻²⁵⁶, and correct rounding makes results identical across machines, so golden outputs are compared byte for byte. MPFR is the specification's choice; `rug` is its mature binding and provides `erf`, needed for the exact GELU `x·Φ(x)`, which pure-Rust alternatives lack. Cost: `gmp-mpfr-sys` compiles GMP and MPFR from bundled C sources once (about four minutes here; needs `m4` and a C compiler); the `use-system-libs` feature can link Homebrew's `mpfr` instead. The interpreter is generic over a `Real` trait so that another implementation could replace it. |
| **rand_chacha**, **rand_core** | 0.9 | `presk-interp` | The ChaCha20 pseudo-random generator, seedable and portable; the traits it implements. | Input synthesis for `presk run --seed`, initializers and `sample` must be reproducible across machines and Rust versions; ChaCha gives a fixed stream for a fixed seed. Keys are split into child keys by drawing 32 bytes per child from the parent's stream. |
| **egg** | 0.11 | `presk-opt` | E-graphs and equality saturation: a data structure that holds every rewrite of a term at once (classes of equivalent nodes), a rewrite engine over patterns, class analyses, and extraction of the cheapest term by a cost function. | The specification's rewriting layer *is* equality saturation ("the algorithmic heart of PSK-Core rewriting"): exact identities over ℝ applied without an order, the cost deciding at the end. `egg` is the reference implementation of the technique in Rust, small (no C dependency), with the analysis hook the pass uses to carry PSK-Core types on classes and the runner limits that bound saturation. `egglog` (its Datalog successor) was considered and deferred: the rule set is a few dozen patterns and needs none of its relational features yet. |
| **serde**, **serde_json** | 1.0 | `presk-interp`, `presk-cli`, `presk-opt`, `presk-rt` | Serialization framework and its JSON format. | `presk run --json`, the conformance test's `inputs.json`/`expected.json`, and since 0.3.0 step 7 the machine profile of `presk bench --calibrate` (constants, samples, correlations); since 0.5.0 step 2 the realization object that `--realize @FILE` reads and the precision solver writes (`presk-rt`); since 0.8.0 step 1 `serde_json`'s `raw_value` feature in `presk-cli`, so that `presk reprice` copies a recorded point's fields as written (D64); later the serialized `.pskk` IR of the specification. |

## Libraries planned for the next step

| Crate | Step | What it is | Why |
|---|---|---|---|
| (none new) | 0.3.0 step 8 | Documentation, tutorial step 12 and the tag need no library. | |

## Not used, and why

| Candidate | Reason |
|---|---|
| **z3** (SMT solver) | The shape fragment is tiny (linear arithmetic + divisibility over ℕ); an in-house procedure of a few hundred lines is sound by construction, needs no C++ system dependency, and reports counterexamples in source terms (decision D4). |
| **logos** (lexer generator) | The Unicode identifier rules, the `·`-is-never-an-identifier exception, the layout-significant newlines and the `5ms` quantity literals are simpler by hand (D7). |
| **just** | Not installed on the development machines; `make` is (D2). |
| **miette** | Its derive-oriented error types fit application errors better than a compiler's diagnostic bag; ariadne renders our own `Diagnostic` value. |
| **astro-float**, **dashu** (pure-Rust arbitrary precision) | Kept as a fallback behind the `Real` trait if `rug` cannot be built somewhere; they lack `erf` and MPFR's correct-rounding guarantees for transcendental functions (D5). |
