> PyTorch compiled and in mixed precision, beside eager (presk D153). It was run at the user's request. Version 0.15.0 (presk `f407639`, the tag's tree and the two scripts; the addendum without dropout at `caf42c1`), PyTorch 2.14.0+cpu, on falicon, 2026-10-18 19:52 to 20:35, nothing else on the machine. The tables come from `experiments/pytorch_modes.sh` (`make pytorch-modes`): `out/pytorch-modes-0.15.0/report.md`, `distances.md`, `modes-vs-eager.md`, `identity.txt`, `gap/margins.md`, `quality/quality.md`, `attribution/`. The outputs are in presk-project `data/pytorch-modes-0.15.0.tar.xz`.

# PyTorch compiled and in mixed precision, beside eager

T2 and T4 (D127: the C program's training step at 0.98× of PyTorch's) compare with PyTorch **eager** in `float32`. Eager is not what a PyTorch user who cares about speed runs. `torch.compile` captures the graph and fuses it, and AMP in `bfloat16` is the mixed-precision recipe. This run measures both, on the same model, the same inputs and the same cores, by D126's protocol.

**In one paragraph.** On ten X925 cores, `torch.compile` with its defaults takes PyTorch's training step from 987.6 ms to 558.2 ms (0.57×), and its forward pass from 151.4 ms to 122.4 ms (0.81×). **The compiler's C step (1 003.5 ms, D127) is 1.80× the compiled step**, where it is 0.98× eager. With AMP added, PyTorch's step is 466.2 ms, so the C step is 2.15× it. Like for like, PyTorch's step drawing no dropout masks as the C step draws none, the C step is 1.46× eager's (685.0 ms) and 2.07× the compiled step's (483.8 ms). The compiled graph computes the products as eager does, through ATen's `mm` and `bmm`, about two thirds of its step. It wins on the rest: the elementwise operations fused, and the dropout masks drawn inside its kernels, which take a quarter of eager's step. In accuracy, the compiled `float32` model is as close to the exact logits as eager's (3.3·10⁻⁶ against 2.9·10⁻⁶ at D = 64, no flip). AMP's `bfloat16` is 4.0·10⁻² from them and flips 505 of 118 016 decisions, as the compiler's all-`bf16` does (5.0·10⁻², 480): the recipe certifies nothing at the margins' budget. Trained 600 steps at D = 64, AMP's validation loss is eager's +1.4·10⁻³ on average over three seeds. That is above the curve's realizations (within 1.3·10⁻⁴ of all-`f64`) and inside the spread a different dropout mask alone makes (about 10⁻²).

## What ran

**The model** is `presk export corpus/nlp/05_gpt.psk --target pytorch --mode readable --train`, the course GPT at its course constants (D = 384, H = 6, L = 6), unchanged.
- *The export* is D127's byte for byte (`_05_gpt.py`, `pskd.py`, `train.py`), and so is the D = 64 export D130 ran (`identity.txt`).
- *The driver* is `experiments/pytorch_modes.py`. It loads the module and presk's checkpoint as `pytorch_baseline.py` does and runs four modes:

| mode | what |
|---|---|
| `eager` | the module as exported, `float32` |
| `compile` | `torch.compile` with its defaults: the backend inductor, the default mode, on the CPU |
| `amp` | `torch.autocast("cpu", dtype=torch.bfloat16)` around the forward pass and the loss; the parameters, the gradient and AdamW in `float32` (the recipe as PyTorch documents it) |
| `compile+amp` | both, the autocast region inside the compiled function |

**The forward pass** is `GPT.apply` in eval mode under `torch.no_grad()`, at B = 16, T = 256. It runs on the checkpoint `out/course_probe_f32/final.psk.d` and the token indices of the C export's conformance inputs, as in D127. The first call is timed apart (under `compile` it compiles), then seven calls, the median and the best.

**The training step** is `train.py`'s loop: the same schedule, AdamW constants and `clip_grad_norm_`, imported from the exported `train.py`. The model is initialized from seed 0 and the batches come from its data file in its order.
- The objective calls the model directly. The exported `loss` instead calls it through `torch.func.functional_call` on a `GPT(T)` it builds at every step, 22 ms of parameter draws a step (`pytorch_baseline.py`'s log). So the exported `train.py` is timed beside the modes as the control, D127's own row.
- `compile` compiles the objective, the model and its cross-entropy together, as a model that returns its loss is compiled.
- The first step, which compiles, is timed apart; the median of the seven after it is reported.

**The protocol** is D126's:
- five processes a row, each row of a round a process of its own, the five rounds one after the other;
- each round after a quiet machine (one-minute load under 1.5, `driver.log`);
- the median of the five processes' medians, and the fastest.

The rows of record are on the ten X925 cores pinned (5–9, 15–19, `taskset`, ten threads), as D127's. Twenty threads unpinned are reported beside them.

**Every compile is cold.** Each process compiles with inductor's caches off (`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`), so its first call or step pays the whole compile.

**The controls**, in each round:
- `pytorch_baseline.py forward` and the exported `train.py`, D127's own PyTorch rows;
- on ten threads, the compiler's step under the plans: the C program's `loss_grad` and `adamw.step` (under OpenMP's default and with `GOMP_SPINCOUNT=8M`), and the native backend's (`presk bench`).

**The addendum without dropout** was decided after the rounds, when the profile showed where eager's step goes (below). The step was run in `eager` and `compile` with the model called with `training=False` (its dropout off), the rest unchanged. It took five rounds of its own, after the five, by the same protocol (`nodrop-round1..5`). The C step it compares with draws no masks either: `experiments/gpt_grad.psk`'s export drops keys, so its dropout is the expectation.

**The configuration** (`torch-config.txt`, `inductor-config.txt`):
- *torch:* 2.14.0+cpu, built with GCC 13.3, OpenMP, oneDNN 3.12.0, `BLAS_INFO=open` (OpenBLAS), CPU capability SVE128.
- *inductor:* the vector ISA it picks is `asimd` (NEON, 128 bits); `cpp.cxx` g++ (13.3.0); `cpp_wrapper` False; `max_autotune` False; `freezing` False; `fallback_random` False, so a compiled graph draws its own random numbers; `epilogue_fusion` True.
- *dynamo:* `cache_size_limit` 8; `automatic_dynamic_shapes` True.

**What dynamo captures:**
- *The graphs.* `torch._dynamo.explain` finds one graph and no graph break on the forward pass (251 operations) and on the objective (252), in both compiled modes (`explain-*.txt`). Each timed process reports one frame, one graph and no graph break (its log's `dynamo:` line).
- *What inductor does with the products.* It sends every product to ATen's `mm` or `bmm`, 111 and 36 calls a step as in eager, and fuses the rest into C++ kernels.

## The time

### One training step, ten X925 pinned (ms)

| mode | median | fastest | max ÷ min | ÷ eager | the first step (compiles) |
|---|---|---|---|---|---|
| eager, the exported `train.py` (D127's row) | 1 022.0 | 1 006.0 | 1.024 | 1.03 | — |
| eager | **987.6** | 974.8 | 1.031 | 1.00 | 0.94 s |
| `torch.compile` | **558.2** | 541.8 | 1.049 | **0.57** | 8.3 s |
| AMP `bfloat16` | **827.3** | 813.4 | 1.023 | 0.84 | 0.74 s |
| `torch.compile` + AMP `bfloat16` | **466.2** | 461.4 | 1.028 | 0.47 | 9.7 s |
| eager, without dropout | **685.0** | 682.2 | 1.010 | 0.69 | 0.61 s |
| `torch.compile`, without dropout | **483.8** | 475.6 | 1.021 | 0.49 | 7.2 s |

The first step's time is the median across the five processes. The five processes of each row are in `report.md`.

### The forward pass, ten X925 pinned (ms)

| mode | median | fastest | max ÷ min | ÷ eager | the first call (compiles) |
|---|---|---|---|---|---|
| eager, `pytorch_baseline.py` (D127's row) | 147.3 | 137.8 | 1.149 | 0.97 | — |
| eager | **151.4** | 144.9 | 1.087 | 1.00 | 0.16 s |
| `torch.compile` | **122.4** | 119.0 | 1.040 | **0.81** | 3.7 s |
| AMP `bfloat16` | **107.2** | 102.4 | 1.089 | 0.71 | 0.12 s |
| `torch.compile` + AMP `bfloat16` | **88.4** | 84.0 | 1.141 | 0.58 | 4.4 s |

### Twenty threads, unpinned (ms, the medians)

| mode | forward | ÷ eager | step | ÷ eager |
|---|---|---|---|---|
| eager, D127's row | 250.4 | 0.99 | 1 370.8 | 1.01 |
| eager | 254.1 | 1.00 | 1 350.6 | 1.00 |
| `torch.compile` | 199.2 | 0.78 | 837.1 | 0.62 |
| AMP `bfloat16` | 181.0 | 0.71 | 1 141.4 | 0.85 |
| `torch.compile` + AMP `bfloat16` | 134.2 | 0.53 | 669.6 | 0.50 |

**The controls reproduce D127**, each within D127's own five processes or just below their fastest:

| row | here | D127 (its range) |
|---|---|---|
| PyTorch's step, the exported `train.py` | 1 022.0 | 1 026.9 (1 010.1–1 033.0) |
| PyTorch's forward pass, `pytorch_baseline.py` | 147.3 | 151.2 (142.1–157.6) |
| the C step under the plans, OpenMP's default | 995.9 | 1 003.5 (991.7–1 009.1) |
| the C step, `GOMP_SPINCOUNT=8M` | 836.7 | 846.9 (839.6–855.2) |
| the native step under the plans | 1 346.9 | 1 354.3 (1 350.3–1 356.9) |

**The compiler's rows are D127's, not timed again for the record.**
- *The C programs.* The C programs this run exports (`loss_grad`, `adamw.step` and `GPT.apply` under the plans, ten threads, and the `f64` forward pass) are D127's, byte for byte, but for the lines that name presk's version and the plan file's path (`identity.txt`). Their plans are D127's byte for byte, and their native outputs (`native.psk.d`) are D127's bit for bit.
- *The ratios.* The controls above are the same programs timed again in this session, 0.5 % to 1.2 % faster than D127's medians. So the ratios below read D127's figures of record; this session's controls give ratios within 0.02 of them (the C step ÷ the compiled step 1.78 against 1.80).

### The compiler's step against each mode's, ten X925 pinned

The C step is D127's 1 003.5 ms (both runtimes at their default waiting, as T2 reads it). Beside it: the same with its workers spinning, 846.9 ms, and the native backend's 1 354.3 ms.

| PyTorch's step | ms | **C ÷ it** | fastest ÷ fastest | C, spinning, ÷ it | native ÷ it |
|---|---|---|---|---|---|
| eager, the exported `train.py` (T2's row) | 1 022.0 | **0.98** | 0.99 | 0.83 | 1.33 |
| eager | 987.6 | 1.02 | 1.02 | 0.86 | 1.37 |
| `torch.compile` | 558.2 | **1.80** | 1.83 | 1.52 | 2.43 |
| AMP `bfloat16` | 827.3 | 1.21 | 1.22 | 1.02 | 1.64 |
| `torch.compile` + AMP `bfloat16` | 466.2 | **2.15** | 2.15 | 1.82 | 2.90 |
| eager, without dropout | 685.0 | **1.46** | 1.45 | 1.24 | 1.98 |
| `torch.compile`, without dropout | 483.8 | **2.07** | 2.09 | 1.75 | 2.80 |

**The forward pass.** The C program's forward pass under the plan is D127's 274.2 ms (226.0 spinning), the native one 343.4 ms. The C program is then:
- 1.81× PyTorch's eager (1.49× spinning);
- 2.24× its compiled forward (1.85×);
- 3.10× compiled with AMP (2.56×).

On twenty threads unpinned, the C step under the plan (D127: 1 100.1 ms, 872.2 spinning) is 1.31× the compiled step (837.1), 1.04× spinning.

### Where a step's time goes

One step under `torch.profiler`, the third of a process, on ten threads pinned (`attribution/profile-*.txt`). This is an attribution, not a timing of record; the profiler adds its own cost.

| op, self CPU ms | eager | eager, no dropout | compile | compile, no dropout |
|---|---|---|---|---|
| `aten::mm` (111 calls) | 294.5 | 284.0 | 277.0 | 287.3 |
| `aten::bmm` (36 calls) | 103.5 | 80.7 | 79.8 | 62.5 |
| `aten::bernoulli_`, the dropout masks (19 calls) | **253.0** | — | — | — |
| the compiled graph's own kernels | — | — | 150.9 | 117.1 |
| the rest | 352 | 319 | 22 | 19 |
| the step | 1 003 | 683 | 530 | 486 |

What the table shows:
- **The products cost the same in every mode.** `torch.compile` on the CPU sends them to ATen's `mm` and `bmm`, as eager does: OpenBLAS or oneDNN, 350 to 400 ms a step, about two thirds of the compiled step.
- **Two things make the compiled step faster.** It fuses the elementwise operations around the products into a few C++ kernels: layer norm written as means and a square root, as the export writes it; GELU; softmax with its mask; the residuals; the biases. In eager these are separate passes over memory (about 320 ms). It also draws the dropout masks inside those kernels, with its own random numbers (`fallback_random` False). In eager, `bernoulli_` takes a quarter of the step.
- **Without dropout, eager pays no masks** and its step drops from 987.6 to 685.0 ms. The compiled step drops from 558.2 to 483.8.

## The identity of the modes

**At course scale**, the forward pass's logits are set against the native backend's `f64` ones (`distances.md`, `modes-vs-eager.md`). The inputs are the conformance inputs, the checkpoint the probe's, whose logits are small (largest |y| about 4.5·10⁻³), so few positions' argmax means much there.

| side | ‖Δ‖₂ / ‖y‖₂ | max \|Δ\| / max \|y\| | argmax moved |
|---|---|---|---|
| native `f32`, and the C program | 3.26·10⁻⁷ | 1.49·10⁻⁶ | 1 of 4 096 |
| PyTorch eager | 1.62·10⁻⁷ | 6.08·10⁻⁷ | 0 |
| PyTorch `torch.compile` | 1.65·10⁻⁷ | 6.14·10⁻⁷ | 0 |
| PyTorch AMP `bfloat16` | 1.78·10⁻³ | 4.83·10⁻³ | 650 |
| PyTorch `torch.compile` + AMP | 1.79·10⁻³ | 4.67·10⁻³ | 666 |

Three things follow from it:
- **The compiled graph is not eager's, bit for bit.** Its logits differ from eager's by at most 2.3·10⁻⁹ (5·10⁻⁷ of the largest), since its fused kernels reduce in their own order.
- **Its distance to the `f64` logits is eager's.**
- **Eager on twenty threads gives eager's ten-thread logits bit for bit.** AMP's are `bfloat16` logits; the recipe's gap is three to four orders of magnitude the `float32` modes'.

**At D = 64, against the exact logits** (`gap/margins.md`). This is `make margins`' model, `out/curve_train_f64/final.psk.d`, on the 461 held-out windows (118 016 positions), PyTorch on ten threads pinned, counted by `experiments/margins.py report` as every realization is. ε\*(10⁻³) = 4.67·10⁻⁴.

| run | gap on the logits | relative | flips | bound #{m < 2ε} | certified at 10⁻², 10⁻³, 10⁻⁴ | accuracy |
|---|---|---|---|---|---|---|
| PyTorch eager `float32` (D130's row, bit for bit) | 2.948·10⁻⁶ | 5.66·10⁻⁷ | 0 | 0 | yes, yes, yes | 0.242442 |
| PyTorch `torch.compile` `float32` | 3.320·10⁻⁶ | 6.37·10⁻⁷ | 0 | 0 | yes, yes, yes | 0.242442 |
| **PyTorch AMP `bfloat16`** | **4.028·10⁻²** | 7.73·10⁻³ | **505** | 9 036 | no, no, no | 0.242365 |
| PyTorch `torch.compile` + AMP | 4.028·10⁻² | 7.73·10⁻³ | 491 | 9 036 | no, no, no | 0.242391 |
| presk all-`f32` | 3.115·10⁻⁶ | 5.98·10⁻⁷ | 0 | 0 | yes, yes, yes | 0.242442 |
| presk all-`bf16` | 5.033·10⁻² | 9.66·10⁻³ | 480 | 13 661 | no, no, no | 0.242425 |
| presk, solved at 10⁻³ on the loss (D124) | 2.516·10⁻⁴ | 4.83·10⁻⁵ | 0 | 51 | yes, yes, no | 0.242442 |
| presk, solved at 10⁻³ on the trained envelope (D132) | 7.785·10⁻³ | 1.49·10⁻³ | 30 | 1 692 | no, no, no | 0.242433 |
| presk, solved at 4.67·10⁻⁴ on the logits (V6) | 7.525·10⁻⁵ | 1.44·10⁻⁵ | 0 | 18 | yes, yes, no | 0.242442 |

What it says:
- **This run's eager row is D130's, bit for bit**, every gap and decision (`identity.txt`).
- **The compiled `float32` run is as close as eager's and as presk's all-`f32`**, and decides as the exact model does at every position.
- **AMP's `bfloat16` is 86 times ε\*(10⁻³) from the exact logits.**
  - It flips 505 decisions (491 compiled) and certifies at no δ: the compiler's all-`bf16` in kind, a little closer (4.0·10⁻² against 5.0·10⁻², 505 flips against 480).
  - The recipe rounds every product's operands and the softmax to `bfloat16` whatever the budget. The realizations the solver certifies keep the gap at 2.5·10⁻⁴ and 7.5·10⁻⁵ with no flip.

## The quality at D = 64

The exported D = 64 model is trained from scratch for 600 steps in each mode, on `out/civil_train_T256.psk.d` (90 % of the Code civil, 4 155 windows of 256). The loop is `train.py`'s: its batch of 16, AdamW, `cosine(warmup = 100)` over 600 steps, `clip_grad_norm_` 1.0, the shuffle seeded 7. This is the curve's protocol (`out/curve_train_f64.log`: 600 steps of 16, the third epoch's start at step 519).
- *The batch.* The prompt's B = 2 is the solver's dimension in `curve_trained.sh` (`DIMS`), not the training batch; the curve trains at the training run's `batch(B = 16)`.
- *The seeds.* Seeds 0, 1 and 2 for PyTorch's initialization. Each seed is the same in every mode, and so is the batch order.
- *The masks.* Eager and AMP draw the same dropout masks from the global generator: seed 2's AMP run ends 4·10⁻⁶ from eager's. The compiled modes draw their own (`fallback_random` False), so `torch.compile` − eager measures the masks as much as the compiler.
- *The evaluation.* Each final model's validation loss and next-character accuracy in `float64` on the 461 held-out windows, the weights cast up (`pytorch_modes.py evaluate`). The same code on presk's all-`f64` runs of the curve gives `make curve`'s 2.7736101 to every digit, which checks it.

| mode | seed 0 | seed 1 | seed 2 | mean | − eager, seed by seed | mean − eager | accuracy, mean |
|---|---|---|---|---|---|---|---|
| eager `float32` | 2.7858141 | 2.7529430 | 2.7459793 | 2.7615788 | — | — | 0.235338 |
| **AMP `bfloat16`** | 2.7898181 | 2.7532509 | 2.7459753 | 2.7630148 | +4.0·10⁻³, +3.1·10⁻⁴, −4·10⁻⁶ | **+1.4·10⁻³** | 0.234008 |
| `torch.compile` | 2.7980323 | 2.7747306 | 2.7439028 | 2.7722219 | +1.2·10⁻², +2.2·10⁻², −2.1·10⁻³ | +1.1·10⁻² | 0.230418 |
| `torch.compile` + AMP | 2.7959961 | 2.7753112 | 2.7434111 | 2.7715728 | +1.0·10⁻², +2.2·10⁻², −2.6·10⁻³ | +1.0·10⁻² | 0.230477 |
| presk all-`f64` (the curve's three seeds) | 2.7736101 | 2.7583979 | 2.6812643 | 2.7377574 | — | — | 0.247769 |

The seeds' spread, the largest loss less the smallest over a mode's three seeds: PyTorch eager 0.040, AMP 0.044, `torch.compile` 0.054, presk's 0.092 (`make step-train`'s, D93).

**What it says.**
- **AMP's quality cost is above the curve's realizations and inside the spread.** Paired with eager by seed and by mask, AMP's validation loss is +1.4·10⁻³ on average (+4.0·10⁻³ at worst), and its accuracy −1.3·10⁻³.
  - *Against the curve:* every realization the curve trained came within 1.3·10⁻⁴ of all-`f64` (D132's addendum). AMP's cost is ten times that.
  - *Against the spread:* the masks alone move the loss by about 10⁻² (`torch.compile` − eager, the same seed and batches), and PyTorch's seeds by 0.040, so the cost is lost in both.
  - *Read as:* the recipe costs more quality than the solver's realizations, and less than a different dropout draw.
- **`torch.compile`'s quality is eager's, within the masks' spread**, as it should be: the difference has the masks' size and both signs.
- **PyTorch's losses are not presk's.** The two initializations differ (PyTorch's `xavier_uniform_` and its generator, presk's own key), so their absolute losses are not comparable across frameworks. They overlap: PyTorch eager 2.746–2.786, presk 2.681–2.774.

## What it says

**For T2 and T4.** On this CPU, `torch.compile` with its defaults makes PyTorch's training step of the course GPT 1.77× faster than eager, and its forward pass 1.24×.
- **Against compiled PyTorch, the compiler's C step is 1.80× slower**, where it is 0.98× eager's.
- *T2's statement* (the C step at most 2× PyTorch's) would still hold against the compiled step as `train.py` runs it. It would fail against compile + AMP (2.15×, another precision), and like for like without masks (2.07×).
- *The eager comparison already favoured the C program.* The C step draws no dropout masks and eager PyTorch's spends a quarter of its time drawing them. Without masks, the C step is 1.46× eager.
- *The gap is not in the products*, which compiled PyTorch computes through the same libraries as eager. It is in the elementwise passes, which inductor fuses, and in the products' kernels themselves: the C program's are GCC's code from the export's nests, against OpenBLAS and oneDNN (D97, D100).
- **In short:** the C step is level with eager PyTorch and behind compiled PyTorch, 1.8× with dropout and 2.1× like for like.

**For the recipe.**
- *AMP's gap.* The recipe PyTorch documents for mixed precision, AMP in `bfloat16`, is 4.0·10⁻² from the exact logits at D = 64, flips 505 decisions and certifies nothing at ε\*(10⁻³). That is the compiler's all-`bf16` in kind, where the solver's certified realizations keep the gap 160 to 540 times smaller with no flip.
- *AMP's speed.* AMP buys PyTorch 0.84× its step and 0.71× its forward pass in eager, and 0.84× its compiled step (466.2 against 558.2).
- *AMP's quality.* Its quality cost (+1.4·10⁻³) is ten times the solver's realizations' and inside the spread of one dropout draw.

## What it does not say

- **One machine, one model, one PyTorch.**
  - *The setting:* the X925 cores of one machine, the course GPT at one size, PyTorch 2.14.0+cpu.
  - *Inductor:* on the CPU, with its defaults. It was not tried in the `max-autotune` mode, with `freezing`, with the C++ wrapper, or with a compiled optimizer step.
  - *No GPU*, where `torch.compile` and AMP are made for.
- **The eager model is the export's.** It is the readable module `presk export` writes: layer norm as means and a square root, attention as `einsum`, mask and softmax. It is not a hand-written PyTorch model with `nn.LayerNorm` and `F.scaled_dot_product_attention`, which eager would run faster. The compiled rows fuse what such a model's library calls would fuse, so they bound it from below. That model is not measured.
- **The compile times are cold.** Every process compiles with the caches off: 3.7 s for the forward pass, 8.3 s for the step. A user's second run reads inductor's cache.
- **AMP's quality at D = 64 only**, three seeds, 600 steps, and its gap on the trained D = 64 model only. At course scale AMP's distance is on the probe checkpoint, whose logits are near zero.
- **The compiler's rows were not re-timed as rows of record.** They are D127's, with their programs shown identical and timed again here as controls, within 1.2 %.

## Files

- `experiments/pytorch_modes.sh` (`make pytorch-modes`) and `experiments/pytorch_modes.py`: the run, its tables and its report. Copies of nothing; no script of record changed.
- `out/pytorch-modes-0.15.0/`:
  - `report.md`: every row of every round;
  - `identity.txt`: the plans, the exports and the C programs against D127's and D130's;
  - `distances.md`, `modes-vs-eager.md`: the identity at course scale;
  - `gap/margins.md`: the gap at D = 64;
  - `quality/quality.md` and the training logs;
  - `attribution/`: the profiles;
  - `explain-*.txt`: dynamo's graphs;
  - `torch-config.txt` and `inductor-config.txt`;
  - `driver.log`: the load at each round's start.
- Archived as presk-project `data/pytorch-modes-0.15.0.tar.xz`, without what the scripts write again: the exported modules, the trained models (`.pt`), PyTorch's logits (`.npy`), the C binaries and `.psk.d` files. Their checksums are in `EXCLUDED.sha256`.
