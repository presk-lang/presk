> The time the compiler's commands take per model, measured from 2026-10-19 to 2026-10-20 by `make
> compile-times`' script (`experiments/compile_times.sh`, run by `experiments/llama.sh`'s `solve` stage)
> on presk `22dd345` (`presk 0.16.0-dev`),
> on aarch64, 20 cores. Wall seconds of one command each, start-up included: `presk check`;
> `presk calibrate --precision` (the magnitude envelope of the entry on 1 sample(s) at 128 bits);
> `presk solve --budget 1e-3` (a realization proposed and verified by measurement against the reference
> at 128 bits); and the exact reference alone, `presk run --bits 128` on one sample. The first column is
> the model's key.

# Compile and solve times, 2026-10-19: the Llama 2 decoder

| model | constants | presk check (s) | presk solve (s) | presk calibrate (s) | exact reference (s) |
|---|---|---|---|---|---|
| llama-7b-L1 | the Llama 2 decoder at the 7B configuration's width, L = 1 of 32, T = 384, its next-token loss | 0.007 | 48888.016 | 29830.869 | 6806.989 |

The outputs of every command are under `out/llama-0.16.0/compile_times_T384/` (not committed; archived as
`presk-project/data/llama-7b-0.16.0.tar.xz`). A time is of the
command as a reader would run it, the front end, the JIT and the reading of the data
included; `presk solve` includes the reference run it verifies against. The row is `out/llama-0.16.0/compile_times_L1_T384.md`'s, the script's own report
(decision D162), D124's row measured again at 0.16.0 (`docs/compile-times-2026-10-14-llama.md`).

- **How it ran.** `STAGES=solve L=1 T=384 LLAMA_OUT=out/llama-0.16.0 ./experiments/llama.sh`
  (`out/llama-0.16.0/run.sh`), from 19:58 CEST on 2026-10-19 to 19:43 on 2026-10-20 (23 h 45 min;
  D124: 23 h 46 min), alone on the machine, in a `systemd-run` scope capped at 105 GB, in plan
  0.16.0's step 5 (D162), on presk `22dd345`, the tree of steps 1 to 4 (D158–D161). The model, its
  constants, its seed, its entry and its one sequence are D106's (`ids_loss_T384.psk.d`, the same
  bytes as `out/llama/`'s).
- **The same envelope and the same reference.** The envelope is D124's but for its date and the
  largest rounding error of each site's values, which `presk calibrate` records since D155 (65
  sites, 5 exact in `bf16`). One sample, so its counts are its `elems` (D158). The exact loss is
  the same, 11.1761830595.
- **The solve, as 0.16.0's defaults make it.** The same 32 stored values and the same 31 away from
  `f64` (28 in `f32`, 3 in `bf16`), each in D124's type. All 13 parameters are typed, but now **10
  in `f32` and 3 in `bf16`**. The three are RMSNorm's gains `blocks.n1.γ`, `blocks.n2.γ` and
  `norm.γ`, declared `init = ones`. The decoder is solved at its initialization, and the measured
  rounding term (D155, the default since D158) finds a one exact in `bf16`. The bytes at rest are
  1.8644·10⁹ against 3.7288·10⁹ (three vectors of 4 096 elements move 24 KB), and the traffic
  2.2621·10⁹ against 4.5116·10⁹ all-`f64`, **1.99×**, as D124. The estimate is 3.553·10⁻³
  (D124: 7.595·10⁻³), the measured term replacing s·u·|v|. The measured gap is **2.544·10⁻⁵**
  (D124: 2.639·10⁻⁵), with no widening round.
- **Why the gap moved: the executor, not the realization.** The solver measures on the kernel
  interpreter, which computes the native backend's bits since D159. Natively, D124's realization
  and this one both give the loss 11.1762084961, a gap of 2.544·10⁻⁵, under both presk
  `454bb9f` (0.15.0's tree) and `22dd345`. So D124's 2.639·10⁻⁵ was the interpreter's
  `exp` (Rust's `std`); natively the gap was 2.544·10⁻⁵ at 0.15.0 already. The gains in `bf16`
  change no bit, since x · 1 is exact.
- **The bound** (`presk check --precision` on the solved realization, `out/llama-0.16.0/bound_L1_T384.txt`):
  `cond` 1.5304·10⁸ (D124: 1.1901·10⁶), vacuous, as the discharge is the measurement's. The bound
  charges a parameter its type's unit roundoff whatever its values, so the three gains in `bf16`
  raise it 129 times. With the gains put back in `f32` it is 1.1901·10⁶ again, D124's to the digit.
- **Times and memory.** `presk calibrate` 29 830.9 s (D124: 30 030.7), `presk solve` 48 888.0 s
  (48 698.5) and the reference 6 807.0 s (6 813.5), within 1 % of D124's; `/usr/bin/time -v`'s
  peaks 32.0, 71.7 and 29.3 GB, D124's (GB of 2³⁰ bytes).
- **The cores.** Every command is one thread, unpinned, and was not sampled.
- **`presk explain` on the realization** (`--no-exact --target native`, the frozen X925 profile, ten
  threads): 61 s and 14.5 GB, its document in `out/llama-0.16.0/explain/llama-7b-L1/`, archived with
  the rest.
