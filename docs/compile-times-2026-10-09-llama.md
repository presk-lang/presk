> The time the compiler's commands take per model, measured from 2026-10-09 to 2026-10-10 by `make
> compile-times`' script (`experiments/compile_times.sh`, run by `experiments/llama.sh`'s `solve` stage)
> on presk `bb78cb8` (`presk 0.11.0`),
> on aarch64, 20 cores. Wall seconds of one command each, start-up included: `presk check`;
> `presk calibrate --precision` (the magnitude envelope of the entry on 1 sample(s) at 128 bits);
> `presk solve --budget 1e-3` (a realization proposed and verified by measurement against the reference
> at 128 bits); and the exact reference alone, `presk run --bits 128` on one sample. The first column is
> the model's key, which the paper's `tables/setup.py` reads.

# Compile and solve times, 2026-10-09: the Llama 2 decoder

| model | constants | presk check (s) | presk solve (s) | presk calibrate (s) | exact reference (s) |
|---|---|---|---|---|---|
| llama-7b-L1 | the Llama 2 decoder at the 7B configuration's width, L = 1 of 32, T = 384, its next-token loss | 0.007 | 50572.436 | 29497.365 | 6600.795 |

The outputs of every command are under `out/llama/compile_times_T384/` (not committed; archived as
`presk-project/data/llama-7b-0.11.0.tar.xz`). A time is of the
command as the paper's reader would run it, the front end, the JIT and the reading of the data
included; `presk solve` includes the reference run it verifies against. The row is `out/llama/compile_times_L1_T384.md`'s, the script's own report
(decision D117), D106's row measured again at 0.11.0 (`docs/compile-times-2026-10-05-llama.md`).

- **How it ran.** `STAGES=solve L=1 T=384 ./experiments/llama.sh`, from 22:44 CEST on 2026-10-09 to
  22:48 on 2026-10-10, alone on the machine, in a `systemd-run` scope capped at 105 GB, in plan
  0.11.0's step 7 (D117), on presk `bb78cb8`, version 0.11.0's tree. The model, its constants, its
  seed, its entry and its one sequence are D106's (`out/llama/ids_loss_T384.psk.d`, the same bytes).
  A first start of the same stage, at 22:18, was stopped four minutes into `presk calibrate`, to
  run the solve alone on D106's envelope; the user chose the whole protocol instead, so that this
  row's four times are 0.11.0's.
- **The same envelope and the same reference.** The envelope is D106's but for its date, and the
  exact loss is the same, 11.1761830595: 0.11.0 changes neither.
- **The solve differs, as 0.11.0's step 2b makes it** (D112: the solver offers the values the program
  stores). 25 stored values, 24 away from `f64` (21 in `f32`, 3 in `bf16`): 2.9583·10⁸ bytes against
  5.7909·10⁸ all-`f64`, **1.96×** less traffic. The estimate is 6.897·10⁻³ against the budget's
  1.118·10⁻², and the measured gap **1.993·10⁻⁵**, with no widening round. D106's solve listed 56
  sites, 52 of them narrowed (50 in `f32`, 2 in `bf16`), at 1.99×, and measured 4.324·10⁻⁶; read
  under 0.11.0's roles, its realization narrows 22 of the 25 stored values, at 1.84× (D112). Three
  stored values moved: the softmax `Attention.apply/%18` (4 718 592 elements) and RMSNorm's mean
  `RMSNorm.apply/%3` (384), `f64` in D106, are `f32`, and the square root of the head width
  `Attention.apply/%16` (one element) is `bf16` where it was `f32`. The one stored value left in
  `f64` is RMSNorm's square `RMSNorm.apply/%2`, which its mean reads (1 572 864 elements), as in
  D106. The 31 sites no longer offered change no bit when typed. A function's result is not offered,
  so the head, the attention's output projection, the feed-forward block's three products, the
  norm's output and the block's second residual stay in `f64`, as in D106.
- **The bound** (`presk check --precision` on the solved realization, `out/llama/bound_L1_T384.txt`):
  `cond` 1.1872·10⁶ (D106's realization: 6.2390·10⁵), vacuous, as the discharge is the
  measurement's.
- **Times and memory.** `presk calibrate` 29 497.4 s (D106: 30 193.4), `presk solve` 50 572.4 s
  (50 811.6) and the reference 6 600.8 s (6 828.2), within 3.5 % of D106's; `/usr/bin/time -v`'s
  peaks 32.0, 71.7 and 29.3 GB, D106's (GB of 2³⁰ bytes).
- **The cores.** Every command is one thread, unpinned; sampled once, `presk calibrate` was on core
  16, an X925.
