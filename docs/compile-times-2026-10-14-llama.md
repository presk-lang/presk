> The time the compiler's commands take per model, measured from 2026-10-14 to 2026-10-15 by `make
> compile-times`' script (`experiments/compile_times.sh`, run by `experiments/llama.sh`'s `solve` stage)
> on presk `7334e31` (`presk 0.12.0-dev`),
> on aarch64, 20 cores. Wall seconds of one command each, start-up included: `presk check`;
> `presk calibrate --precision` (the magnitude envelope of the entry on 1 sample(s) at 128 bits);
> `presk solve --budget 1e-3` (a realization proposed and verified by measurement against the reference
> at 128 bits); and the exact reference alone, `presk run --bits 128` on one sample. The first column is
> the model's key, which the paper's `tables/setup.py` reads.

# Compile and solve times, 2026-10-14: the Llama 2 decoder

| model | constants | presk check (s) | presk solve (s) | presk calibrate (s) | exact reference (s) |
|---|---|---|---|---|---|
| llama-7b-L1 | the Llama 2 decoder at the 7B configuration's width, L = 1 of 32, T = 384, its next-token loss | 0.005 | 48698.455 | 30030.703 | 6813.521 |

The outputs of every command are under `out/llama/compile_times_T384/` (not committed; archived as
`presk-project/data/llama-7b-0.12.0.tar.xz`). A time is of the
command as the paper's reader would run it, the front end, the JIT and the reading of the data
included; `presk solve` includes the reference run it verifies against. The row is `out/llama/compile_times_L1_T384.md`'s, the script's own report
(decision D124), D106's and D117's row measured again at 0.12.0 (`docs/compile-times-2026-10-05-llama.md`,
`docs/compile-times-2026-10-09-llama.md`).

- **How it ran.** `STAGES=solve L=1 T=384 ./experiments/llama.sh`, from 04:35 CEST on 2026-10-14 to
  04:21 on 2026-10-15 (23 h 46 min; D117: 24 h 05 min), alone on the machine, in a `systemd-run`
  scope capped at 105 GB, in plan 0.12.0's step 5 (D124), on presk `7334e31`, version 0.12.0's
  code, after BERT-large's two blocks. The model, its constants, its seed, its entry and its one
  sequence are D106's (`out/llama/ids_loss_T384.psk.d`, the same bytes).
- **The same envelope and the same reference.** The envelope is D106's but for its date, and the
  exact loss is the same, 11.1761830595: 0.12.0 changes neither.
- **The solve differs, as 0.12.0's step 4 makes it** (D123: the solver offers the functions'
  results and the parameters). 32 stored values, 31 away from `f64` (28 in `f32`, 3 in `bf16`), and
  all 13 parameters in `f32`: 2.2621·10⁹ bytes against 4.5116·10⁹ all-`f64`, the parameters' bytes
  at rest counted (1.8644·10⁹ against 3.7288·10⁹), **1.99×** less traffic; 1.97× on the stored
  values alone (D117: 1.96×). The estimate is 7.595·10⁻³ against the budget's 1.118·10⁻², and the
  measured gap **2.639·10⁻⁵**, with no widening round (D117: 1.993·10⁻⁵). The seven stored values
  added are results, all in `f32`: the head `Llama.apply/%14` (12 288 000 elements, the block's
  largest value), the attention's output projection `Attention.apply/%23`, the feed-forward
  product `Dense.apply/%3`, SiLU's product `silu/%2`, the norm's output `RMSNorm.apply/%10`, the
  block's second residual `Block.apply/%12` and the loss `loss/%6`. The 24 that D117 narrowed keep
  its types; the one stored value left in `f64` is RMSNorm's square `RMSNorm.apply/%2`, which its
  mean reads, as in D106 and D117.
- **The bound** (`presk check --precision` on the solved realization, `out/llama/bound_L1_T384.txt`):
  `cond` 1.1901·10⁶ (D117: 1.1872·10⁶), vacuous, as the discharge is the measurement's.
- **Times and memory.** `presk calibrate` 30 030.7 s (D117: 29 497.4), `presk solve` 48 698.5 s
  (50 572.4) and the reference 6 813.5 s (6 600.8), within 4 % of D117's; `/usr/bin/time -v`'s
  peaks 32.0, 71.7 and 29.3 GB, D117's (GB of 2³⁰ bytes). So the exact reference the solve now
  runs before its sweep, the results' 1-norms and the parameters' leaf norms cost no memory at 384
  tokens, as step 4's gate found at 16 (D123).
- **The cores.** Every command is one thread, unpinned, and was not sampled.
