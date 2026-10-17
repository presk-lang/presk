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
| `--opt` | the e-graph optimizer; refused with a profile or a plan, since `presk plan` plans the program the optimizer has not rewritten; refused with `--budget` and with a realization that types sites, whose sites name the program before it (since 0.13.0 step 1, D134) |
| `--weights W` | trained weights in place of the initial ones |
| `--seed S`, `--bits B` | the inputs' seed and the exact reference's precision (128 by default) |
| `--envelope ENV` | an envelope given in place of measuring one |
| `--no-exact` | no exact reference: no envelope measured, no gap, no conformance outputs |
| `--target native,c,pytorch` | the targets, `native` by default |
| `--measure` | also run it on this machine: §11 |
| `--site S` | a site to trace through every stage (§12), the option repeated |
| `--format md\|json\|html` | `json`: `explain.json` beside `explain.md`, the document's facts and the provenance table as values (below; since 0.13.0 step 2), and `network.svg`, the network diagram (below; since 0.14.0 step 1); `html`: `explain.html` too, the pipeline view (below; since 0.13.0 step 3) |
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
its envelope, its sensitivity, its type with its role and reason, under
`--opt` the `let` the optimizer made of it, its fusion (a member of a
group, a group's end or alone, and why), the nest that computes it and its
layout beside the runner-up, its kernel (as the epilogue of a contraction's
kernel, when it is one), the C function that launches it and, with
`--measure`, its time. `--site` names the sites; by default they are the
three largest shares of the solver's estimate (the three largest bounds
without a solve) and the three nests of the most predicted time a call.
`listings/site-index.tsv` lists every site of the functions the entry
reaches and the sections that name it.

Since 0.13.0 (step 1, D134) the traces are read off **the provenance
table** of the compilation, `presk_provenance::Table`: every element of
every stage (a span of the source; a function, an instance or a typed
expression of the elaborated program; a `let` of PSK-Core, of the program
after differentiation and of the program after the optimizer; a site the
precision stages name; a nest; a layout; a kernel; a launch; a function of
`kernels.c`), and for each element a stage made, the elements it was made
from and how: the lowering's rule, the derivative rule of the operation an
adjoint differentiates, `unchanged` or `rewritten` by the optimizer, the
fusion group, the kernel the nests share. The IRs do not change: each
stage returns its links beside its value, and `presk-driver` puts them
together (`presk_driver::provenance`). An element with no child at the
next stage says why (the optimizer removed it, the host computes it, a
group the call never launches). Until 0.12.0 the traces followed a site by
its number and a kernel by its printed text: under `--opt`, which numbers
every `let` anew, they named no fusion and no kernel, and an `add` a
contraction's kernel computes as its epilogue was said to be computed by
the host (D133, D134).

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

## The JSON (`--format json`)

`--format json` writes `explain.json` beside `explain.md` (0.13.0 step 2,
D135), for a program that reads a compilation rather than a person: the
pipeline view's page reads it. It holds the document's facts as values and
the provenance table of §12, and it is deterministic without `--measure`,
as the Markdown is: no time, no absolute path. Its fields, in layout
`format` 2 (0.14.0 step 1, D141, which added `network`; format 1 is
0.13.0's):

| Field | What it holds |
|---|---|
| `format` | `2`, the layout's version; a reader refuses one it does not know |
| `compiler` | the compiler's version |
| `program` | `file`, `entry`, `dims` (`[name, value]` pairs), `consts`, `realization` (its label), `opt`, `planned` (a profile planned the layouts), `differentiated` (the program has generated functions) |
| `traced` | the sites §12 traces: `--site`'s, or the default ones |
| `sources` | each file read: `id` (the file number the source stage's keys use), `path`, `text` |
| `stages` | one per stage that has elements, in the compiler's order (source, elab, core, ad, precision, opt, fusion, layout, kernel, launch, c): `stage`, `title`, `section` (the document's section that shows it) and `elements` |
| `stages[].elements[]` | `key` (the element's key in its stage: a span `file:start-end`, `fn NAME`, a site `f/%n`, a nest, `k<j>`, `kernels.c k<n>`, `native k<j>`), `label` (the line the document prints for it), `group` (the function it belongs to, which the page folds by), `facts`, and `ends`, why it has no child at the next stage |
| `facts` | by stage: a `let`'s `span` and source `name`; a precision site's `magnitude`, `elements`, `condition`, `sensitivity`, `type`, `term`, `rounded_in`, `bound` (the conditioned bound's error at the site) and, for a decision, `why`; a layout's `shape`, `predicted_us`, `runner_up`, `runner_up_us` and, for a contraction nest, `order` (the realization's reduction order); a kernel's `text` and `dtype`; a C function's `text`; a kernel's machine code, under the native target (0.13.0 step 5, D138): `bytes` (its size), `hex` (its bytes, before they are linked, since 0.14.0, D145), `ranges` (`[start, end, line]`, the bytes each line of the kernel's text compiles to, in the code's order, every byte in one range) and `given` (`[start, end]`, the bytes Cranelift gave no line, which `ranges` gives to the line before them), `disasm`, Cranelift's text of the instructions, and `relocs` (`[start, end, symbol]`, the fields the linker fills with a `libm` function's address, zero in `hex`; D145). And `tex`, the element's formula in LaTeX (0.13.0 step 4, D137): a function's display and an expression's formula on the elaborated program, a `let`'s equation on PSK-Core and the stages after it, a site's term of the solver's estimate, a contraction nest's sum, with `order_tex`, the order its terms are added in |
| `rules` | the rules the links name, in LaTeX: `adjoint`, the contributions of each primitive's derivative as the reference's table `core-adjoint` writes them, by the rule a link names; `rewrite`, the identity of each rewrite rule of the optimizer (tables `core-algebra`, `core-structure`) |
| `precision` | under `--budget`: `estimate` (the solver's first-order estimate, the sum of the terms), `budget` (absolute), `measured` (the gap measured against the exact reference) and `bound` (the conditioned rule's, on the output), or `null` |
| `links` | every link: `child` and `parents` as `stage:key`, `how` (the reason's name, from the `presk_provenance::How` list) and its fields: `rule`, `kind`, `mode`, `role`, `rules`, `members`, `stop`, `point`, `points`, `shared_by`, `count`, `lines` |
| `network` | the network diagram (below), or `null` when the entry does not resolve |
| `document` | the document itself: `title`, `subtitle`, `header` and `sections` (`number`, `title`, `purpose`, `blocks`, `commands`, `sections`), each block one of `prose`, `table` (`head`, `rows`), `code` (`lang`, `text`, `name`), `note` (`title`, `text`), `attach` (`name`, `text`) |

The document's strings are those the Markdown prints, and the Markdown is
rendered again from them byte for byte (a test of `presk-explain`). On the
Llama 2 decoder block at the 7B width (X5's parameters, `--no-exact`) it is
1.1 MB.

## The network diagram (format 2's `network`)

Since 0.14.0 step 1 (D141), `explain.json` holds the program drawn as
boxes and arrows, and `presk explain` writes its default opening as
`network.svg` beside it. The diagram is a view of the provenance table,
not a second model: every node and every arrow names an element of the
table, and the drawing adds positions and numbers to them. Crate
`presk-diagram` reads the program the kernels run (the table's stage `ad`,
or `opt` under `--opt`):

- a **call box** for each call of a function, named by the field of the
  parameters it is given (`ln1`, `attn`), its instance from their type
  (`LayerNorm{64}`); its element is the call's `let`, and `function` names
  its callee's, which every instance shares;
- a **stack** box for a `fold` over stacked parameters, `stack{L}` once ×L,
  which opens to one layer;
- an **operation node** for each fusion element of the function, a group
  (`add · mul`) or a primitive alone, and a **host** node for a primitive
  the host computes (`split_key`, a dropout's mask outside training);
- a **parameter** node for each parameter an operation reads (`θ.w`); the
  parameters a box is given are its port;
- the function's **inputs** and its **result**;
- an **arrow** for each value that flows, through the projections and
  tuples the host carries, named by the value's site, or by the entry's
  function for the entry's inputs (the table has no element for a
  function's parameters).

A box's graph is its callee's, drawn at its call: one PSK-Core function
serves every instance of its module, its types in the module's dimensions
(`ℝ[..., Out]`), so an arrow inside a box takes the type under the call's
dimension arguments, the open prefix `...` bound to the leading axes of the
caller's argument. Under `grad`, the generated functions are boxes like any
other (`MLP.apply.fwd`, `MLP.apply.bwd`), each naming the function it was
generated from.

The numbers, each from where the compiler keeps it (statement N2):

| On the drawing | From |
|---|---|
| a box's formula | the printer's display of its function (`render::aligned`, the elaborated column's `tex`), and the same in Unicode |
| an operation's formula | its root `let`'s equation, the string its column shows, and a contraction's sum and order |
| an arrow's type, elements and bytes | the `let`'s type at the call, at the entry's dimensions; the realized type (`Realization::dtype_at`); elements × the type's bytes |
| a box's parameters | its subtree of the entry's parameters, each leaf in its stored type (a quantized leaf's integers and `f32` scales); inside a stack, a layer's |
| a box's work W and bytes Q | the cost model's (`kernel_cost`) over the launches inside its call: the kernel backend's launch log records, since this step, the calls above each launch (`Launched::calls`) |

W and Q are over every run of a node; `numbers`, the line the drawing
writes, divides them by `runs`, the layers of the stacks the node is
inside, so that a layer shows a layer's. A test (`presk-explain`'s
`tests/network.rs`) checks, on `make plan-check`'s entries, the coverage
models, the six snapshot documents and the course GPT, with and without
the optimizer, that every element the diagram names is in the table
(statement N1), that the parameters sum to the entry's, that each arrow's
elements are the envelope's at the sites the envelope holds (the largest
over the instances, which is what the envelope keeps), and that in every
opening the work and the bytes of the nodes drawn sum to the entry's
launches' (N2).

The layout is layered, top to bottom, computed in Rust once for every
opening a reader can reach: the entry alone, then each box opened with the
boxes it is inside, a box opened folding its siblings. The default opening
is the entry and its first box. Sizes come from the strings' lengths, no
text being measured. The `network` object:

| Field | What it holds |
|---|---|
| `entry`, `stage` | the entry, and the table's stage the graph is read from |
| `default` | the view the page opens on |
| `params`, `work`, `bytes` | the entry's: `{count, bytes, types}`, and W and Q over its launches (`null` when the kernels were not listed) |
| `nodes[]` | `id`, `kind` (`root`, `call`, `stack`, `group`, `op`, `host`, `param`, `input`, `output`), `parent` (the box it is drawn in), `element` and `function` (`stage:key`), `callee`, `label`, `title` and `numbers` (the lines the drawing writes), `instance`, `stack` (L), `runs`, `path` (in the entry's parameters), `tex` and `text` (the formula in LaTeX and in Unicode), `sum` and `order` (a contraction's, in LaTeX), `params`, `work`, `bytes`, `generated` (`{kind, of}`), `members` (a group's sites), `index` (an input's place), `children`, `depth`, `key` (it carries keys only) |
| `edges[]` | `id`, `from`, `to`, `parent`, `arg` (its place among `to`'s arguments), `site`, `ty` (in the entry's dimension names), `at` (at their values), `dtype`, `elems`, `bytes`, `key` |
| `views[]` | `open` (the boxes opened, outermost first), `width`, `height`, `rects` (`[id, x, y, w, h]`, in points) and `routes` (`[id, x0, y0, x1, y1, …]`) of every node and arrow drawn |

`presk dump FILE --stage network --entry E` (0.14.0 step 3, D143; `cli`
chapter) prints the same diagram without writing a document: its outline,
its JSON, or one opening as SVG or TikZ (`--level N`), or one box opened by
itself, without what is around it (`--box NAME`). Every format draws
the positions of the one layout (statement N4, a test of `presk-explain`):
the page, `network.svg`, the SVG and the TikZ. The TikZ's boxes carry the
printer's LaTeX, each formula shrunk to its box when it is wider, with the
reference's macros provided where the document has none. `make
network-figure` writes the course GPT at level 2 into the chapter
`explain_network` of this manual (741 × 1 569 pt, to be read on a screen),
and the report's two figures as TikZ (presk-specs `report/figures/`), each
readable on paper, as the user decided on 2026-10-17: level 1, the GPT
with its stack of blocks folded (`network-gpt.tex`, 647 × 899 pt, on an A4
page at 0.74 of its size), and `Block` opened by itself
(`network-gpt-block.tex`, `--box blocks`, 693 × 808 pt, at 0.69), their
9 pt labels printing at about 6 pt.

## The pipeline view (`--format html`)

`--format html` also writes `explain.html` (0.13.0 step 3, D136): the
stages of the compilation as columns, every element linked by an arrow to
what it was made from, read off the JSON above, which the page carries
inside it. It is one file, with no framework, no build step and nothing
fetched; it opens offline in any browser.

- **Columns**, in the order the compiler runs the stages: the source, the
  elaborated program, PSK-Core, what differentiation made, the precision
  sites, what the optimizer changed, the fusion nests, the layouts, the
  kernels, the launches, the functions of `kernels.c`, the kernels'
  machine code (below).
- **A column is a decision taken.** Without `--opt` there is no optimized
  column; without a `grad` there is no column for differentiation; without
  a profile, no layout column. An element a stage left unchanged is not
  drawn again: its arrow passes through the column, dashed.
- **Arrows** are coloured by the reason (the lowering, differentiation,
  the optimizer, fusion), and from the precision column on by the realized
  type (`f64`, `f32`, `bf16`).
- **Folding.** Each column shows one card per function, folded. The
  functions of §12's traced sites are open, and their threads are lit. A
  function opens and folds by its header in every column at once. The
  precision column lists a function's decisions and folds the sites that
  follow or are carried.
- **Selecting** an element lights its origins and what it became, through
  every stage, and dims the rest; shift adds to the selection, `Esc`
  clears it.
- **Hovering** shows the element's facts and the reason of each of its
  links, in words, and its mathematics (below).
- **The mathematics** (0.13.0 step 4c, D137), the printer of the
  denotation's, in the notation of the language reference's chapter 3
  (section 3.10): the elaborated program's cards are its functions'
  displays and its expressions' formulas, PSK-Core's cards and those after
  it each `let`'s equation, a contraction nest's card its sum. Hovering
  adds, for the source, what the function it defines denotes; for a `let`
  differentiation made, the rule of its primitive's derivative as the
  reference's table `core-adjoint` writes it; for a `let` the optimizer
  rewrote, the identity of each rule that applied in its function; for a
  precision site, its term of the solver's estimate (its sensitivity, the
  unit roundoff of its type, its magnitude) and the estimate against the
  budget, the gap measured and the bound; for a contraction nest, the
  order its terms are added in under the realization. "∑ math" turns the
  cards back to text. KaTeX 0.19.0 typesets them, inlined in the page with
  its fonts (`dependencies` chapter), and only a formula that is shown is
  typeset, once.
- **A lesson** is a sequence of selections with captions, held in the
  page's address (`#lesson=…&step=…`): `←` and `→` step through it, and "add
  step" appends the current selection. Without one, the lesson is §12's
  traced sites.
- **Measured quantities** are overlays on the cards: a layout's predicted
  time, and a kernel's measured one under `--measure`.

`#bench` in the address times a select of each element and a fold and an
open of each function by the page's own clock, and prints the slowest
(statement P3). In headless Firefox 158 on falicon, on the Llama 2 decoder
block at the 7B width (1.8 MB with KaTeX and the machine code, 473
elements, 237 arrows drawn, the formulas on), the first drawing took 19
ms, a select at most 7.0 ms, a fold or an open at most 44.0 ms (0.13.0
step 5); on the course GPT at `D = 64`, 31, 6.0 and 33.0 ms.

## The Network tab

Since 0.14.0 step 2 (D142), the page has two tabs. **Pipeline** is the
columns above; **Network** draws the program as boxes and arrows from the
JSON's `network` (above), at the positions `presk-diagram` computed: the
page lays nothing out. It opens on the default opening, the entry and its
first box opened.

- **A box** is a call: its name and instance, its parameters with their
  stored type, the cost model's W and Q (a box inside a stack shows one
  run's, "each of L runs"), and folded, its function's display typeset in
  it, scaled to fit. A stack is one box ×L, its border doubled.
- **Opening.** A box's header opens it in place, to its callee's graph at
  that call, and folds it again; a stack opens to one layer, ℓ. Opening a
  box folds its siblings. "default" goes back to the default opening.
- **Operation nodes** are the fusion groups (`add · mul`) and the
  primitives alone; dashed, a primitive the host computes. Parameter nodes
  are the leaves an operation reads (`θ.w`).
- **Arrows** are coloured by their realized type (a parameter's by its
  stored type) and their width grows with their bytes. Hovering an arrow
  shows its site, its type and its bytes; "labels" writes them on every
  arrow. "keys" draws the random keys and their `split_key`s, dashed,
  which are hidden by default. Under `grad`, "backward" draws the `.bwd`
  boxes beside their `.fwd`, each naming the function it was generated
  from, hidden by default.
- **Hovering** a node shows its numbers (W/Q, its intensity), its formula
  and, for a group, each member's equation, the values that never reach
  memory; then its element's facts, as in the columns.
- **A click** selects the node's elements in the columns, a box's call and
  its callee's function, and switches to the Pipeline tab with their
  threads lit. With shift it stays on the diagram, where the nodes and
  arrows of the thread light. A selection made in the columns lights the
  diagram too.
- **The lesson** records the tab and the opened boxes with each step.

The address takes `tab=network`, `net=` (the opened boxes' ids, outermost
first), `labels`, `keys`, `backward`, `theme=light|dark` and `plain`
(nothing selected).

Under `#bench`, after P3's timings, the page draws every opening and
selects every node of the default one by its own clock (statement N5), and
clicks each box of the default opening, checking that its elements light
in the columns (N1). In headless Firefox on falicon, three runs each: on
the course GPT at `D = 64`, 15 openings drawn in at most 20.0 ms and 21
selects in at most 1.0 ms, the columns' select at most 7.0 ms and fold at
most 34.0 ms; on the Llama 2 decoder block at the 7B width (1.9 MB), 13 openings in at
most 20.0 ms and 19 selects in at most 2.0 ms, the columns' select at
most 7.0 ms and fold at most 45.0 ms, as at 0.13.0 (7.0 and 44.0). Every
box clicked lit its elements.

## The native column

Under `--format json` or `html` with the native target, each kernel of the
listing is compiled as `presk run --backend native` compiles it, and its
machine code is the last column (0.13.0 step 5, D138). Every byte lies in
one range of one line of the kernel's printed text (statement P4): each
statement's instructions carry the line as a Cranelift `SourceLoc`, the
kernel's header holding the prologue and its closing brace the return. A
byte Cranelift gives no line, a move the register allocator inserts or a
branch at a block's edge, is given the line before it, and the page shows
it in italics. The lines change no byte: a test compiles every kernel of
the test set with them and without them and finds the same code.
Hovering a kernel's machine code shows each range with its line of the
kernel and its bytes, then Cranelift's text of the instructions, register
allocated.

The bytes are those Cranelift emitted, before the JIT links them (0.14.0,
D145, the user's answer on presk !100). Linked, a kernel that calls a
`libm` function (`exp`, `erff`) holds that function's address, which the
loader places differently in each process, so that `explain.json` differed
from one run to the next. Before linking the field is zero; the page
underlines it and names the function whose address it receives (`relocs`).
The kernel a run executes is unchanged: the same bytes, linked.

On the test set's 1 164 kernels Cranelift gives a line to 88 % of the
bytes; on the decoder block at the 7B width, to 65 %, its 4 × 16 tiles in
lanes moving sixteen accumulators at each block's edge. A constant keeps
the line of the instruction that made it: a kernel's zero is made once,
for its header, so the zeroing of its accumulators shows on line 1.

## What is not there yet

Cranelift's IR (CLIF) is not shown: the machine code is, and Cranelift's
text of it after register allocation.
