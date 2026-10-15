# `presk explain`: one compilation as a document

`presk explain` compiles one entry of a program under the parameters of one
compilation, a budget or a realization, a machine profile, the targets, and
writes what the compiler did as a document. Every stage's representation is
shown, then the decisions taken there and the commands that print the same
thing (0.11.0, D113). It is for a reader who wants to see the whole path from
the source to the C program on one example, and for a reviewer who wants
each figure of a result traced to the command that prints it.

```sh
presk explain tutorial/07_train.psk --entry loss --dim B=4 --budget 1e-3 \
    --profile ~/presk-data/ra4-model-0.8.0/profile.json --threads 10 \
    --target native,c,pytorch --out out/explain/loss
```

On the tutorial's `loss` this takes 0.05 s and writes `out/explain/loss/`:

| File | What |
|---|---|
| `explain.md` | the document, in Markdown |
| `listings/` | the listings too long for the body, which names each one |
| `artifacts/envelope.json` | the envelope, as `presk calibrate --precision --out` writes it |
| `artifacts/realization.json` | the solved realization, as `presk solve --out` writes it |
| `artifacts/plan.json` | the layout plan, as `presk plan --out` writes it |
| `artifacts/c/` | the C program, as `presk export --target c` writes it |
| `artifacts/pytorch/` | the PyTorch modules, as `presk export --target pytorch` writes them |
| `artifacts/SHA256SUMS` | the checksum of every file under `artifacts/` |

## The parameters

The flags are the commands' own, with the same names and defaults.

| Flag | Meaning |
|---|---|
| `--entry E` | the entry: a top-level function or `Module{args}.fn` |
| `--dim N=v`, `--const N=v` | as for `presk run` |
| `--budget ε` | solve for a realization under a relative budget, as `presk solve` |
| `--realize R` | a realization given: `f64`, `f32`, `bf16` or `@FILE.json` |
| `--plan @P.json` | a plan given, its realization with it |
| `--profile P.json` | a calibrated profile: the solver prices time, and the layouts are planned as `presk plan` plans them |
| `--threads N` | the workers a split is counted on, and the native executor's |
| `--remat store\|recompute` | what a gradient does with its forward values (0.10.0 step 4) |
| `--opt` | the e-graph optimizer; refused with a profile or a plan, since `presk plan` plans the program the optimizer has not rewritten |
| `--weights W` | trained weights in place of the initial ones |
| `--seed S`, `--bits B` | the inputs' seed and the exact reference's precision (128 by default) |
| `--envelope ENV` | an envelope given in place of measuring one |
| `--no-exact` | no exact reference: no envelope measured, no gap, no conformance outputs |
| `--target native,c,pytorch` | the targets, `native` by default |
| `--measure` | also run it on this machine: §11 |
| `--site S` | a site to trace through every stage (§12), the option repeated |
| `--out DIR` | where the document and its files go |

`--budget`, `--realize` and `--plan` exclude one another. Without any of
them every site is in `f64`; without a profile nothing is planned and every
nest runs its default layout. Each input given in place of being computed
(`--envelope`, `--realize @`, `--plan @`, the profile, the weights) is
listed in the document's header with its path and its SHA-256, and
`--no-exact` is listed as what was not computed. That is how a document on a
model at full size can read the results of a run that took hours instead of
paying for the exact reference again.

## The document

| § | Section | What it shows |
|---|---|---|
| | The parameters | the table above's values, the compiler's version, the artifacts and their provenance |
| 0 | Summary | the outcome, each figure labelled a *fact*, a *decision*, a *parameter*, a *bound*, an *estimate* or a *measurement* |
| 1 | The source | the entry's file and its imports, numbered; the diagnostics |
| 2 | Elaboration | `presk dump --stage elab`; the static assertions and how each was decided; the types, regularity and effects of `presk check --summary`; the entry's parameters and their sizes |
| 3 | PSK-Core | the functions the entry reaches, and how a site is named |
| 4 | Differentiation | the derivatives the entry reaches; the precision analysis's own reverse sweep, in `listings/` |
| 5 | The e-graph optimizer | under `--opt`, its report, and each function the entry reaches that it rewrote, before and after |
| 6 | Precision | 6.1 the envelope; 6.2 the sensitivities; 6.3 the solve and, per decision (the stored values, the functions' results among them, and the parameters by path, since 0.12.0), its type, term of the estimate and why it has that type, then the values rounded elsewhere or carried; or the realization given: every value the program stores with the type the record gives it or the default's, and every parameter with its type (since 0.12.0); 6.4 every rule's bound beside the measured gap |
| 7 | Fusion | the groups of the functions the entry reaches, the launches they save, and why each group ends where it does |
| 8 | The machine and the layout plan | the profile's machine and layout model; the plan as `presk plan` prints it; for each nest the chosen point, its runners-up and the best split, each prediction in its terms; every point's prediction in `listings/` |
| 9 | The kernels and the host | the kernels, each under the sites that launch it; the host as the C export prints it |
| 10 | Targets | the native backend; the C program, its files and its first contraction as C; the PyTorch modules |
| 11 | Measured | with `--measure`: the realization natively against the exact reference, a call timed, each kernel's median launch beside its prediction, the host by phase |
| 12 | Sites through every stage | each site's line, magnitude, sensitivity, type and why, fusion, nest and layout, kernel, C function and time; an index of every site and the sections that name it, in `listings/` |
| | Appendix | every command behind the sections, in order |

**The site is the thread.** A site is named `<function>/%<n>`, `n` the
number of its variable in PSK-Core (§3). The envelope, the sensitivities and
the realization are keyed by it, a nest is named by the site it computes,
and the host launches each kernel under it, so a site can be followed from
the source to the C.

**The roles of §6.3.** The solver's table gives every site it offers a
role (0.11.0 step 2b, D112; `precision` chapter): a *decision* is a value
the program stores, whose type the solver chooses; a value that *follows*
another is rounded where that one computes it, and its term is charged at
that type; a *carried* value is another site's, with no term and no cost.
The terms sum to the solver's estimate.

## The reasons

Each decision of the compiler is printed with what decided it (0.11.0
step 4, D114):

- **A site's type (§6.3).** The solver tries every narrowing in order of
  what it saves per unit of error it adds, and takes one when the estimate
  stays under the budget; then it measures, and widens the largest share by
  one type while the gap exceeds the budget. It records each move
  (`presk_precision::solve::Move`), and the document gives each decision its
  own: *narrowed to `f32` (move 4): saves 2.6·10⁻⁸ s, adds 2.0·10⁻⁷ to the
  estimate*; *kept: `f32` would add 1.2·10⁻⁶ to the estimate, which stood at
  1.8·10⁻⁷: past the budget*; *kept: no narrower type saves anything under
  the machine*; *widened from `f32` to `f64` at round 1, the gap over the
  budget*.
- **Where a fusion group ends (§7).** Each element-wise value that reaches
  memory, a group's root or a value alone, with what reads it
  (`presk_core::fusion::stops`): a reader that is not element-wise (a
  reduction, a contraction, a call), a reader of another shape, readers in
  different groups, the expression's size limit, or the function's result.
- **A nest's layout (§8).** The chosen point, its two runners-up and the
  best point that splits, each prediction split into its terms (a launch,
  `spawn`, work, register loads, the traffic from L2, L3 and memory) and
  the term that bounds it. A split pays `spawn`, about 100 µs on the X925
  profile, so on a small program no nest splits. The twelve nests of the
  most predicted time are in the body, every nest in
  `listings/plan-reasons.txt`.
- **The optimizer's rewrites (§5).** Each function its report says it
  rewrote (its lets or its cost changed), before and after. Every function
  it visits has its variables numbered anew, so under `--opt` the sites of
  the later sections are the rewritten program's.
- **The static assertions (§2),** each with the decision procedure that
  discharged it.

## What it guarantees

- **The figures are the commands'.** Where the document quotes what a
  command prints, it calls the function the command prints with
  (`presk_driver::text`), so every line `presk calibrate --precision`,
  `presk dump --stage sensitivity`, `presk solve`, `presk check
  --precision`, `presk plan` and `presk dump --stage kernel` print is found
  in it verbatim (X3, `crates/presk-cli/tests/explain.rs`). The exceptions
  are what holds a time or a path: `presk plan`'s seconds, which the
  document leaves out, and the envelope's path.
- **Deterministic.** The same parameters give the same bytes: no time, no
  date and no absolute path in the body, the compiler's version in its
  header only. A file under the working directory is named relative to it,
  one under the home directory with `~`, and the document's own directory
  is `$DOC`. The document is therefore a snapshot of every stage at once:
  `crates/presk-explain/tests/documents.rs` holds the documents of the MLP,
  the layer norm, the MiniGPT and the tutorial's `loss`, and a change to a
  stage shows in them as a diff.
- **Built from values.** The document is made from the driver's values
  (`presk-driver`), never from a command's printed text.

## The measured section and the traces

**`--measure` (§11)** runs the entry on this machine (0.11.0 step 5,
D115):
- the exact reference, and the realization natively under the plan, with
  their gap; it is the gap the solver measured on the kernel interpreter,
  which computes the native backend's bits (statement U5), and the section
  says so;
- a call timed, 21 calls after 3 of warm-up (`presk bench`'s
  measurement, `presk_driver::bench::time_call`);
- each kernel's median launch over 21 calls beside the profile's
  prediction (`presk bench --calibrate`'s, `bench::kernel_times`); a
  launch is timed with its dispatch, so on a small program the ratio is
  large;
- the host by phase, as `presk bench --host-profile` prints it
  (`bench::host_profile`).

The section is the document's only one that changes from run to run.
`presk explain` does not reserve the machine: the section gives the load
average before and after, and says that its times are not a record.

**The traces (§12)** follow a site through every stage: its PSK-Core line,
its envelope, its sensitivity, its type with its role and reason, its
fusion (a member of a group, a group's end or alone, and why), the nest
that computes it and its layout beside the runner-up, its kernel, the C
function that launches it and, with `--measure`, its time. `--site` names
the sites; by default they are the three largest shares of the solver's
estimate (the three largest bounds without a solve) and the three nests of
the most predicted time a call. `listings/site-index.tsv` lists every site
of the functions the entry reaches and the sections that name it.

## The PDF

The Markdown is the document of record; the PDF is made from it by a make
target, so the compiler has no TeX dependency (0.11.0 step 6, D116):

```sh
make explain                          # the tutorial's loss into out/explain/loss
make explain-pdf                      # out/explain/loss/explain.pdf, 19 pages
make explain-pdf EXPLAIN=out/explain/other   # any document
```

It runs pandoc 3.1.3 with LuaLaTeX from TeX Live 2026 (`TEXLIVE=` its
`bin` directory), the DejaVu fonts, whose monospace face has every
character the listings print, and `crates/presk-explain/pandoc/header.tex`:
fvextra's verbatim, which breaks a long listing's lines, and tables in a
smaller size. A table's columns take their widths from its longest cells,
which the renderer writes as the separator's dashes. GitLab renders the
Markdown as it is. The `dependencies` chapter lists the tools.

The manual's chapter `explain_loss` is the tutorial's document, as `make
explain` writes it; a test holds it current, and `make explain-manual`
writes it again.

## At a model's size

The document reads what costs hours from files instead of computing it
again. On the Llama 2 decoder block at the 7B configuration's width and 384
tokens, the envelope and solved realization of `make compile-times`' run
given, the frozen profile, no exact reference (statement X5, D116; 43 s and
12.4 GB at 0.12.0):

```sh
presk explain coverage/llama.psk --entry loss --const G=32 --const L=1 --dim T=384 --dim B=1 \
    --realize @out/llama/compile_times_T384/llama-7b-L1_solved.json \
    --envelope out/llama/compile_times_T384/llama-7b-L1_env.json \
    --profile ~/presk-data/ra4-model-0.8.0/profile.json --threads 10 --no-exact \
    --out out/explain/llama-7b-L1
```

The entry reads its batch from a data file, which the envelope names; the
document's shapes are those of one sequence, `--dim B=1`, the envelope's
one sample. At 0.12.0 that realization is D124's: 31 of the 32 stored
values and the 13 parameters typed. §6.3 lists every stored value with
its type, the one at the default (RMSNorm's square, `RMSNorm.apply/%2`)
among them, and every parameter. D106's realization, solved before 0.11.0
step 2b, typed sites whose type changes no bit too: §6.3 gives each its
role, a follower or a carried value, and says so.

## What is not there yet

Cranelift's IR is not shown.
