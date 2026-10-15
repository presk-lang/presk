# The C export: `presk export --target c`

Version 0.9.0 lets an entry leave the compiler as a C program, in the style
of llama2.c and StrasGPT: a few files that a C compiler builds on any
machine, that run without presk, and that compute the bits `presk run
--backend native` computes with the same realization, plan, dimensions,
seed and weights (plan `presk-specs/implementation/presk-plan-0.9.0.md`,
statement C1). Any entry the native backend runs can be exported: a forward
pass, a loss and its gradient, one optimizer step, a generation function
with `presk serve`'s loop around it, under a realization plan, on threads,
with integer weights. The training loop cannot. A second mode writes each
loop nest of an entry on its own, as the input of a polyhedral tool.

The `cli` chapter has the commands, their flags and the directory they
write; tutorial step 18 exports the tutorial's attention and checks it.
This chapter says how a program is made, why its bits are the native
backend's, what is checked, and what the export does not do. The five
statements it was judged on, and their verdicts, are in the `experiments`
chapter (§9): C1, C2, C3 and C5 hold, and C4, a measure of speed, decides
nothing.

## How a program is made

An exported program is the two halves of a native run of the entry:

- **The kernels** are the native backend's, the fusion, the realization and
  the layout already applied, each printed as one C function with its
  PSK-Kernel text as the comment above it (0.9.0 step 1, D77): literal loop
  bounds, offsets multiplied out, lanes as loops over a local array, `bf16`
  as `uint16_t` computed in `float` and rounded as the runtime rounds, a
  thread split's `[lo, hi)` on the loop at its depth.
- **The host** is a recording of the native backend's own run of the entry
  (step 3, D79). The kernel backend's values carry no identity (a `let`
  reads a clone of its operand, a slice of a stacked tree is a copy), so the
  recorder (`presk-kernel`'s `record`) keeps a symbolic copy of the host's
  values beside them: each tensor is a view of a buffer at an offset, and
  the recorder follows calls, `let`s, tuples, records, closures, the `stack`
  fold's slices and each primitive's decomposition into launches. Every
  launch it records is the backend's own (kernel, types, layout, shape); it
  decides none. Its hooks do nothing when it is off, and the native
  backend's kernels and outputs are the same with and without them.

The export runs the entry natively once, with the recorder on for that call
only, and the same run gives the native outputs that `conformance/` holds
beside the inputs and the exact reference's outputs.

## The host's structure

The recording keeps, beside its list of operations, the tree it came from:
each call with its function, dimensions, parameters' names, arguments and
result, each fold with its iterations (step 4, D80). The export puts each
call in a canonical form, its arguments by position, its buffers by first
definition, its kernels by their C text, its callees by their own forms, and
writes one C function per distinct form: a module function called at two
dimensions (`Linear.apply` at D→D and D→4D) is two functions, one called
fifty times is one. A result is written into buffers the caller allocates,
so a function releases everything it allocated when it returns and nothing
is moved.

A fold's iterations that record the same operations up to offsets that move
with the layer make a loop, with each stacked parameter sliced as `w + l *
chunk`, as llama2.c's `forward` does; an iteration that differs from the
next (the first block under a mixed realization, which reads an `f64` site)
runs apart, as a call of its own function. A gradient's scan and backward
fold are loops too, the backward one from the last layer down.

The arena is the program's peak, computed from the structure and equal to
the peak the program reaches: the course GPT at `T = 256` takes 117, 59 and
29 MB in `f64`, `f32` and `bf16`, of which 86, 43 and 22 MB are its weights,
where releasing nothing took 352, 176 and 88 MB.

The straight-line host of step 3 stays, as the structured one's oracle
(`presk export --straight-line`): one function performing every operation
in order, nothing released. The program performs what the native backend
performs: until 0.10.0 step 3 a gradient's host also ran the forward pass
for the primal result it discards, and the C program copied it (125 of
`loss_grad`'s 767 launches); that dead code now goes after differentiation
(the `language` chapter, "Dead code after differentiation"), and
`loss_grad` records 583 launches, 4 of them dead.

## Plans and threads

`presk export --plan @P.json` records the native run under the plan, so the
program's kernels are the plan's tiles, orders, k-blocks, lanes and packs
(step 5, D81). A launch the native backend divides among threads (a plan's
split, or a kernel above the executor's threshold of work, among
`--threads` workers) is an OpenMP region whose threads each call the kernel
on their piece of the divided loop, the native backend's pieces. The team is
the launch's workers at most, fewer when `OMP_NUM_THREADS` says so: a
`num_threads` clause alone would override the variable, so the program caps
its team at `omp_get_max_threads()`, and `OMP_NUM_THREADS=1` runs it on one
thread. Built without OpenMP, each region is one call over the whole loop.
The split belongs to the launch, not to the kernel's text, and is part of a
function's canonical form. The course GPT at `B = 16`, `T = 256` under the
plan the frozen profile gives divides 201 of its 228 launches among ten
threads.

## Sampling and generation

`sample(key, logits)` draws from its key (0.10.0 step 7b, D99; until then
it took no key, and the native backend and the program each drew from a
stream of their own, step 6, D83). The recorder records `sample` as a host
operation, and the key it reads must be an argument of the entry: the
program holds it as a `presk_key` of the entry's `_in` struct, and a key
the program does not hold (split by `split_key`, or stacked by a fold)
reaching `sample` is refused (`E8006`). The runtime's keys are
`presk-interp`'s: the key a seed names (`presk_key_from_seed`, the first 32
bytes of `seed_from_u64`'s stream, PCG32), a key's split
(`presk_key_split`) and its child (`presk_key_child`), each read from
Bernstein's ChaCha20 (a 64-bit block counter and a 64-bit stream number,
not RFC 7539's layout) as `rand_core`'s block generator reads it (four
blocks at a time, a 64-bit draw as two words, the low one first). A
function that draws takes the entry's keys after the arena, and `main` sets
each key field to the key `presk run --seed` synthesizes for that argument,
from the export's seed or `-s`; a key nothing draws from is no field.

A draw is the native backend's, in `f64`: the row's largest logit, `exp(x −
m)` of each, their sum in order, and the first token whose running sum of
`e_i / z` exceeds `u`, the key's uniform draw of the row's rank, the top 53
bits of its 64-bit word over 2⁵³. The native
backend's `exp` there is the `libm` crate's, which the kernels call, since
step 6; it was Rust's standard library's, the platform's C library (glibc on
falicon). So the C program calls `presk_exp`, depends on no C library, and
draws the native tokens on any machine, the x86-64 musl build included. On
540 generations of 32 sampled tokens (three types, three temperatures, sixty
seeds), the tokens are the ones the standard library's `exp` gave.

An entry's `f64` argument (a temperature, an `eps`) is a `double` of its
`_in` struct, not a constant of the program: `main` sets it to the value
`presk run` synthesizes. `presk export --target c --serve` exports a
generation function (`presk serve`'s, or `--entry`), its key, when it takes
one, given to token `i` as the child `i` of the key the seed names, at one sequence and the
window `T` its parameters fix, and its greedy copy, every `sample` an
`argmax`, which `presk serve` calls at temperature 0: both are recorded on
one backend and share their functions (`presk_MiniGPT_apply` is one), and
`generate.c` runs `presk serve`'s loop over them. A model whose parameters
do not fix the context's length is refused (`E8009`), since `presk serve`
then passes a context that grows.

## Integer weights

A realization's `quant` map makes a parameter integers and a scale per
channel, and the native backend makes them each time it loads the weights,
from the checkpoint's reals (the `precision` chapter). The C program does
the same (step 7, D84): `main` reads the reals and quantizes them
(`presk_quantize` in `presk_rt.c`, a transcription of presk-rt's
`QTensor::quantize`: per channel the largest magnitude, its scale
`f32(m/qmax)` or the smallest power of two above `m/qmax`, each weight
divided by the scale and rounded to even), into a `presk_q8` or a
`presk_q4`, its integers and scales, as llama2.c's `runq.c` has
`QuantizedTensor`.

A kernel the native backend hands the integers takes them, the scales after
its output, and loads an element as `(float)q * s[c]`, one `float`
multiplication, widened in a `double` kernel; a kernel that reads reals
(LayerNorm's fusion group reading a quantized `γ`) gets a copy of their
values, as the native backend converts them at its edge. A stacked matrix's
layer is a view of its integers, the scales at the element offset over the
channel's length; where the native backend gives up keeping a slice
integers (the scales constant along the stacked axis, an `int4` layer of odd
length, which would start inside a byte) its slice is a copy of its values
in `f64`, and so it is in the program, made before each call of the fold's
body. presk-rt's `pow2` scale, which looped for ever on a channel whose
largest weight is below about 10⁻³⁰⁶, ends since step 7; its scales are
otherwise unchanged.

## Why the bits are the native backend's

Rule by rule:

| Where | The native backend | The C program |
|---|---|---|
| each kernel | Cranelift's code of the loop nest, every product rounded before it is added | the same loop nest in C, built with `-std=c11 -ffp-contract=off` (never `-ffast-math`), so the compiler contracts nothing (D77) |
| `exp`, `log`, `tanh`, `sin`, `cos`, `erf` | the `libm` crate | `presk_libm.c`, musl 1.1.22's, bit-identical to the crate (D78) |
| an input or a weight | the exact value rounded to `f64`, then to the site's type, `bf16` through `f32` | read as `f64` from the `.psk.d` file and rounded the same way (`presk_convert`) |
| a buffer of another type at a kernel's edge (a mixed realization) | converted through `f64` | the same conversion, into a new buffer |
| a weight a realization quantizes | quantized from the checkpoint's reals when loaded; loaded by a kernel as `f32(q)·s`, or converted at a kernel's edge | `presk_quantize` from the same reals, the same arithmetic; `(float)q * s[c]` in the kernel, `presk_dequantize` at an edge |
| a weight a realization stores in its own type (`params`, 0.12.0) | stored once in its type when loaded; read by a kernel of its type as it is, by a wider one through loads that widen it | the same type in the arena, converted as the `.psk.d` is read; `presk_bf16_to_f32` or a cast to `double` in the kernel's load |
| a fill (`zeros_like`, a dropout mask outside training) and a scalar made a tensor | `Buf::of`, rounding from `f64` | `presk_fill`, the same rounding |
| the host's arithmetic on hyperparameters, a scalar read back from a kernel (the gradient norm's sum of squares in `adamw.step`), an `f64` argument | `f64` in Rust | `double` in C, the same operations in the same order |
| `sample` | the uniform draws of its key, `presk-interp`'s `Key` (ChaCha20), one per row; each row in `f64` with the `libm` crate's `exp` | `presk_sample` on the key the caller passes (`presk_key`, the same bytes, derived from the seed as `presk run --seed` derives them); `presk_exp` |
| a thread split | pieces that write disjoint elements, a scoped thread per piece at each launch | the same pieces, an OpenMP team (a pool) of at most the same workers, or one thread: each element computed by the same loops in the same order |

Two of these rules corrected the plan. GCC ignores `#pragma STDC
FP_CONTRACT OFF` and warns about it, so the pragma is for Clang only, and
`-std=c11` alone already stops GCC from contracting: GCC's GNU modes
(`-std=gnu11`, its default) contract, and there 14 of 246 primitive runs
move (D77, D78). And current musl is not the `libm` crate: musl 1.1.23
replaced `exp`, `expf`, `log` and `logf` with ARM's routines, which differ
from the crate on 8–10 % of inputs (glibc 2.39's on 1.2–9.8 %), so the
export ships musl 1.1.22's, the last release before, renamed `presk_…` with
no operation changed (D78).

## What is checked

**The kernels** (step 1, `presk-export-c/tests/kernels.rs`): an executor
over the compiled C, loaded with `dlopen`, runs inside the kernel backend,
bit-identical to the native backend on the conformance suite's 41 primitives
in `f64`, `f32` and `bf16`, scalar and in lanes, on the 1 491 layouts and
1 618 thread splits of the four nests of attention, and on hand-written
kernels, every range cut three ways. Each transcendental function is
bit-identical to the `libm` crate on 10⁶ inputs, and each `f32` one on every
`f32` (an `#[ignore]`d test). The runtime reads and writes `.psk.d` files as
`presk-data` does, byte for byte.

**Programs through the binary** (`crates/presk-cli/tests/export_c.rs`, in
`make test`; `make c-conformance` adds the course GPT and the trained
checkpoints). Every entry of `tests/plan/entries.txt` is exported in
`f64`, `f32` and `bf16`, on the parameters `presk run` synthesizes and with
the ones an initializer sets to zero drawn (`MiniGPT.apply`'s logits are
zero otherwise, its head being `init = zeros`), and MiniGPT's five entries
under a mixed realization: 76 programs, each built by its own `Makefile`
without a warning and run by its own `conformance/check.sh`, all
bit-identical to the native backend. The course GPT at its constants is
bit-identical in the three types with its zeros drawn (`B = 1`, `T = 8`)
and with the course's trained checkpoint (`B = 1`, `T = 256`). Each
program is exported twice, structured and straight-line, and the two give
the same output file.

**Under plans and on threads.** Every program is built with OpenMP and
without it, and a program that divides a launch is run on one thread too:
the same bits every way. MiniGPT under a plan stating splits at three
depths, and at a size where the native backend's own rule divides kernels
(`make test`); the test set under the plan `presk plan` gives with the
frozen profile, in the three types, and the course GPT at `B = 16`, `T =
256` under its plans and under the default layout (`make c-conformance`).
In `presk-export-c/tests/host.rs`, five recordings with every kernel
divided at each of its legal depths in turn (40 programs, 4 212 OpenMP
regions), run on one thread, three, every thread and without OpenMP. At the
test set's sizes the plans state tiles, lanes and packs but no split, so
the course GPT at `B = 16` and those tests are what reach threads.

**Integer weights.** The kernel printer's check covers integer inputs: the
41 primitives with every real operand quantized in `int8` and `int4`, the
scales constant along the first axis or the last, in the three types,
scalar and in lanes (2 880 runs), and the four nests' layouts with an
operand quantized (500 points and 568 splits in each width).
`presk_quantize` is bit-identical to presk-rt on 4 027 392 weights of both
widths and schemes (ties at `n + ½`, zero channels, subnormals, huge
values, NaNs among them). Through the binary (`make test`, 42 programs):
`tests/quant/odd.psk` in both widths and schemes and the three types;
MiniGPT's forward pass and `next_token` with its head, embeddings and every
matrix quantized, and its gradient (`loss_grad`) and an optimizer step; at
`D = 9`, `H = 3` in `int4`, where attention's layers are copied to `f64`;
with its biases and norms quantized; under a plan stating splits; and
`generate` under `--serve`. With the trained checkpoints (`make
c-conformance`): the tiny MiniGPT, its `next_token` and C5, the course GPT
at `T = 256` in `int8`, and MiniGPT under the plan `presk plan` gives its
quantized realization.

**Generation** (statement C5). `generate` against `presk serve --backend
native`, the same `--json` document byte for byte: in `make test`,
`serve_minigpt.psk` with its zeros drawn, in the three types, through both
hosts, greedy and sampled at three temperatures and four seeds; in `make
c-conformance`, the same with the trained checkpoint of `make
train-examples`, and `next_token` with it at the test set's dimensions
(C1). The generator is checked against the crates
(`presk-export-c/tests/rng.rs`: 10⁶ draws over 10³ seeds, reads of words
and pairs that straddle two refills; the keys of 10³ seeds, their splits and
their children against `presk-interp`'s, byte for byte; and `presk_sample`
against the native `sample` from 4 800 keys, on rows in the three types,
ties, infinities and NaNs among them).

**The check is checked**, by mutations that each change the outputs, on an
arena filled with NaNs (`presk-export-c/tests/host.rs`):

- of the straight-line host: dropping any live launch, making two launches
  write each other's buffer, reading a `bf16` input with one rounding
  instead of two;
- of the structured host, on MiniGPT's forward pass and its gradient: a
  slice of a stacked parameter off by a layer, a loop bound off by one, a
  function's buffers released before one is read for the last time;
- of the threads: a split whose pieces leave a gap. One whose pieces
  overlap cannot change the outputs, since the elements computed twice are
  the same bits both times, and a check of the pieces added to the program
  catches it;
- of `generate.c`: each token given the key of the token before, every
  token given the first one's (the greedy tokens stay, 0.10.0 step 7b; a
  draw skipped and the stream seeded again at each step until then), the
  window one token behind;
- of integer weights: a scale one channel off, the two nibbles of a byte
  swapped, and the scale kept in `f64` when quantizing (on a checkpoint
  with a tie `w/s = 3.5` in each column of the head: drawn weights meet a
  tie too rarely).

**The native backend unchanged.** Every step compared the native backend's
kernels and outputs on the eleven entries in the three types with the
previous `main`'s, byte for byte (66 comparisons), and `make plan-check`
and `make ra4-dev-frozen` pass. The one change to what the native backend
computes is `sample`'s `exp` (step 6), whose tokens did not move.

## Portability

Statement C3. The C compiles without a warning under `-Wall -Wextra
-std=c11` on GCC 13.3, with OpenMP and without it (`make test`), and on
Clang 21.1 and for x86-64, both `zig cc` from zig 0.16.0 (`make
c-conformance`, which installs it into `.venv-zig`; `dependencies` chapter),
`generate.c` of a `--serve` export included, whose Clang build gives `presk
serve`'s tokens. The Clang program on aarch64 is linked and run and gives
the native bits; the x86-64 one (`x86_64-linux-musl`, static) is linked,
not run, no emulator being at hand. zig has no OpenMP runtime, so there the
OpenMP build is compiled and not linked. No program warns.

## The conformance band

Statement C2. The C program's outputs are the native backend's, so they
leave the exact reference's band on exactly the elements the native
backend's do. On the test set those are the gradient of `minigpt_grad.psk`
(34, 121 and 99 elements of 8 768 in `f64`, `f32` and `bf16`, where large
terms cancel; spec question 75), and a few outputs of the forward passes in
`f32` and `bf16` (`SelfAttention.apply` 1 and 4 of 256; with the zero
parameters drawn, MiniGPT 5 and 35 of 1 024, the course GPT 18 and 63 of
736). They are sums that cancel: MiniGPT's five logits in `f32` are 6·10⁻⁴
to 3·10⁻³ against a median of 0.17, and miss by about 10⁻⁷, a few units in
the last place of the terms, which the band's absolute floor (2⁻²⁰ in
`f32`) does not cover. The kernel interpreter, another conforming executor,
leaves it on some of the same elements and on others (question 75). So
checking the band element by element checks a copy, which is what C2
became: an output outside the band where the native backend's is inside it.

## Speed

Statement C4, reported, deciding nothing (`make c-speed`, measured once in
step 5). The course GPT's forward pass at `B = 16`, `T = 256` in `f32` with
the trained checkpoint, under the plan the frozen profile gives, median of
seven calls:

| threads | native | C program |
|---|---|---|
| every thread (the plan's ten workers) | 1 094 ms | 290 ms |
| one thread (an X925 core) | 3 132 ms | 1 700 ms |

On one thread both run the same kernels; the difference is the code
generator (GCC `-O2` against Cranelift) and the host (the native host copies
each stacked layer's slice and each view, where the C program slices by
pointer). On ten, the native backend also spawns a thread per piece at each
divided launch, where the C program's team is OpenMP's pool. Neither share
is measured apart. The native side is `presk bench --plan-only --weights`;
it cannot run a plan that splits among ten on one worker (condition 5), so
its one-thread row runs the plan with every split stated as none, the same
kernels.

**The shares, and PyTorch** (after 0.9.0, `make pytorch-baseline`, D87;
`experiments` chapter §3). On the ten X925 cores pinned, the same forward
pass under the same plan takes 281–285 ms in C, 697–710 ms natively and
153–156 ms under PyTorch (the model `presk export --target pytorch` writes,
in eval mode), two runs. Timed launch by launch, the C program's kernels
take 268–274 ms and its host 7–17. The native backend's kernels take
362 ms, counting each launch's thread starts (`presk bench --calibrate`), so
Cranelift's code is 1.3× GCC's on the same kernels and the native host
takes the other 335–348 ms. The host is about four fifths of the native
backend's lead, and not by first touches of memory: its page faults do not
grow with the calls. On every core unpinned the C program is 1.13×
PyTorch's time. A gradient exports too: the course GPT's `loss_grad`
(`experiments/gpt_grad.psk`) at these constants records 2 220 launches, 386
of them dead, into an arena of 1.22 GB (since 0.10.0 step 3, which removes
the dead code after differentiation, 1 616 launches, 12 dead, 1.02 GB;
since step 4, which keeps the forward pass's residuals, 1 182 launches, 10
dead, 3.04 GB, and `--remat recompute` the step-3 program; since step 4b,
which keeps the fusion groups only where a realization can see them, in
`f32` 1 012 launches, none dead, 2.49 GB, the `bf16` program step 4's), and
C1 holds. With `adamw.step`
the program runs a training step in 12.2 s under the plan, against
PyTorch's 1.0 s. Four fifths of its time is the derivatives of `dot`,
which no plan reaches.

**The flags, and what else moves the program's speed** (0.10.0 step 7,
D97). The Makefile's `STRICT` holds the flags bit identity rests on and
`CFLAGS` the rest, `-O2` and the warnings, which `make CFLAGS=…`
replaces. Built `-O3` or `-O3 -mcpu=native`, every program of the
baseline gives the native backend's bits: `-O3` reassociates no sum
without `-ffast-math`, and `-mcpu=native`'s fused multiply-adds stay
unused under `-ffp-contract=off`. Neither pays: on the course GPT `-O3`
moves the forward pass and the gradient by −4 % to +2 %, its sign the
program's, and `-mcpu=native` adds nothing; `-O2` stays the default. Two
other things moved the program more, and the follow-up of step 7 settled
both (D97's addendum).

- **Every kernel is always inlined** into the host's functions
  (`PRESK_KERNEL`, a static inline function with GCC's `always_inline`, in
  `kernels.c`), and on GCC `model.c` is compiled without the scheduling
  that runs before register allocation (a `#pragma GCC optimize
  ("no-schedule-insns")` at its head, which Clang does not read). Under a
  4×16 tile a product's inner loop holds sixteen sums, four rows and four
  scalars, at the edge of the 32 vector registers. Left to itself, GCC
  inlined some kernels and not others, a choice that moved with the
  program's other kernels, and where the allocator ran short (out of line
  always, inlined after the scheduling hoisted the sixteen products ahead
  of their sums) one sum lived on the stack, read and written at every
  iteration: the product took about 60 ms where it takes 45. So the
  course GPT's gradient took 1 124 or 1 011 ms with the same products, by
  whether the bias `add`s were kernels of their own; it now takes about
  the same either way, 822 and 827 ms with its threads waiting as below,
  where today's programs took 959 and 844. No bit moves: C1 holds on
  every program, and Clang and GCC compile the files without a warning.
- **How the OpenMP threads wait.** libgomp lets an idle worker spin
  300 000 times, about 0.1 ms on the X925 cores, then sleep, and a sleeping
  worker takes a few hundred microseconds to wake; the host often runs
  undivided kernels for longer than that between two regions. The native
  backend's workers spin about 2 ms. `GOMP_SPINCOUNT=8M` makes libgomp's
  spin as long (measured: an empty region costs 2 µs after a gap of 2.0
  ms, 0.2 ms after 2.2), and `OMP_WAIT_POLICY=active`, which every OpenMP
  runtime reads, keeps them spinning: the forward pass then takes 222 ms
  instead of 268 and the gradient 827 instead of 969, the same bits. The
  program leaves the setting to its runner, and its README says so; `make
  pytorch-baseline` times every C program both ways, and T4 compares the
  native backend with the second.

## The SCoP mode

`presk export --target c --mode scop` (step 8, D85; the `cli` chapter)
writes each nest of an entry at point zero, as `presk plan` captures the
nests, as a standalone C function between `#pragma scop` and `#pragma
endscop`: the input of a polyhedral tool, and of a comparison of such a
tool with RA4's ranking of the same nests, which comes after 0.9.0. The
kernel is the native backend's at point zero, printed in another form:

- arrays of literal dimensions, one index per axis (`in0[i0][i1][i2]`), as
  PolyBench/C writes its kernels, where the other programs address a buffer
  at its linear offset; no range, no OpenMP;
- a branch on a computed value that only assigns scalars (a softmax's
  running maximum, an argmax) as a selection of each value, `m = x > m ? x :
  m`, its condition computed once when it selects two;
- the transcendental functions declared `const` in the file, and
  `presk_rt.h`'s conversions `const` (`PRESK_CONST`), which changes no code
  of the other programs, the compiler seeing their bodies already;
- the warning `#pragma scop` gives under `-Wall` ignored around the function.

The bits are checked as the kernel printer's are: every primitive case at
point zero in the three types, 123 distinct kernels, each in its file, run
through `dlopen` on the inputs the driver draws, against the native backend
(`presk-export-c/tests/kernels.rs`). Through the binary, every nest of the
test set's eleven entries in the three types (383 nests a type) is built
with GCC without a warning and run by the driver against the native
backend's outputs, bit for bit; built in GCC's GNU mode, which contracts,
the dots move, and a softmax's maximum made a minimum moves its nest.

**Static control, checked with Polly.** A nest is a static-control part when
no loop variable is assigned and every access indexes each axis by an
affine function of the loop variables. On the test set 366 nests of each
type are; the 17 others are gathers and scatters (an embedding, its
gradient, a cross-entropy's label), an index loaded from an index array.
Polly, in LLVM 23.1.2's clang (`make llvm`), agrees nest by nest: each of the
1 098 static-control nests of the three types is one SCoP, the whole
function, whose AST has the nest's loops, and none of the 51 others is,
unless Polly is allowed non-affine accesses, which it then over-approximates
by the whole axis. The course GPT's 28 nests at its constants too (`B = 1`
in the three types, `B = 16` in `f32`): 26 SCoPs, 2 gathers. Polly runs at
its early position, before LLVM's loop passes rewrite the loops, with the
whole function allowed as a SCoP, no rescheduling, and only the analysis
(`make c-conformance`). GCC's Graphite, which the machine has, is not the
check: on the same files its verdict follows GCC's passes rather than the
nests (it rejects nests of one loop, cuts a softmax, and takes 14 of the
17 gathers and scatters whole). pet, PPCG and PLUTO, which read the region
between the pragmas, were not run: none was on the machine.

## What the export does not do

- **The training loop**, llm.c's other half: the `training run` block, its
  data stream and its checkpoints around the gradient and optimizer entries
  0.9.0 exports (plan 0.9.0 §8).
- **`.pskk` and `presk build`**: the export prints C at the seam of spec
  §8.5 without the binary form the specification gives it (questions 35,
  36 and 73).
- **Work a gradient does and discards.** Since 0.10.0 step 3 the dead code
  of a gradient goes after differentiation, and since step 4 a gradient
  keeps its forward pass's residuals rather than recomputing it (`language`
  chapter, "Residuals"). What is left: the values `bf16`'s fusion groups
  make the gradient compute apart from the forward pass, and the few dead
  launches the dead-code pass keeps so that no group changes (10 of the
  course GPT's gradient, four of them the zeros a fold stacks first). The
  program copies them, so that it performs the native backend's work
  (`docs/kernels-roadmap.md`, "Work a gradient does and discards").
- **Reshaping integer weights**: an entry that merges or splits the axes of
  a quantized leaf is refused (`E8007`). The native host's view keeps the
  leaf's old scales' axis under the new shape, where the scales need not be
  constant any more (D84); no program of the test set does it, and it is
  left to fix in the native host first.
- **The polyhedral comparison**: a tool's transformation of the SCoP mode's
  nests against RA4's ranking of their layouts, with the driver's times;
  pet, PPCG and PLUTO not run; integer weights and the host's own kernels
  (positions, masks) not in that mode.
- **A GPU** and the matrix unit, which come after 0.9.0 (plan 0.9.0,
  decision 0).
- **Where C4's ratios come from**, the code generator or the host, is not
  measured.
