> The time the compiler's commands take per model, measured from 2026-10-05 to 2026-10-06 by `make
> compile-times`' script (`experiments/compile_times.sh`, run by `experiments/llama.sh`'s `solve` stage)
> on presk `4610f5e` (`presk 0.10.1-dev`), on aarch64, 20 cores. Wall seconds of one command each,
> start-up included: `presk check`; `presk calibrate --precision` (the magnitude envelope of the entry
> on 1 sample at 128 bits); `presk solve --budget 1e-3` (a realization proposed and verified by
> measurement against the reference at 128 bits); and the exact reference alone, `presk run --bits 128`
> on one sample. The first column is the model's key, which the paper's `tables/setup.py` reads.

# Compile and solve times, 2026-10-05: the Llama 2 decoder

| model | constants | presk check (s) | presk solve (s) | presk calibrate (s) | exact reference (s) |
|---|---|---|---|---|---|
| llama-7b-L1 | the Llama 2 decoder at the 7B configuration's width, L = 1 of 32, T = 384, its next-token loss | 0.006 | 50811.591 | 30193.361 | 6828.241 |

The outputs of every command are under `out/llama/compile_times_T384/` (not committed; archived as
`presk-project/data/llama-7b-0.10.1.tar.xz`). A time is of the command as the paper's reader would run
it, the front end, the JIT and the reading of the data included; `presk solve` includes the reference
run it verifies against. The row is `out/llama/compile_times_L1_T384.md`'s, the script's own report
(decision D106; the report `docs/llama-7b-solve-2026-10-07.md`).

- **How it ran.** `STAGES=solve L=1 T=384 ./experiments/llama.sh`, as a stage of
  `out/llama/run-0.10.1.sh` (plan 0.10.1, step 8), from 20:45 CEST on 2026-10-05 to 21:09 on
  2026-10-06, alone on the machine, in a `systemd-run` scope capped at 108 GB with no swap. The model is
  `coverage/llama.psk` with `--const G=32 --const L=1` (`D = 4096`, `H = 32`, `FF = 11008`, `V =
  32000`), on its initialization (`--seed 0`), its entry `loss` on one sequence of 384 random token
  ids and their next tokens (`out/llama/ids_loss_T384.psk.d`). One sample where D101's rows have two:
  the envelope of the one sequence the reference measures.
- **The compiler is 0.10.1's.** At `v0.10.0` the solve did not fit in the machine at one block of this
  width, even on 16 tokens (D104); 0.10.1 holds each parameter once in the exact interpreter (D105),
  and computes every value 0.10.0 does, bit for bit. The reference's time is within 3 % of 0.10.0's on
  the course GPT and 2.6 % faster on BERT (D105's U3), so this row stands beside D101's.
- **Memory**, `/usr/bin/time -v`'s peaks: `presk calibrate` 32.0 GB, `presk solve` 71.7 GB, the
  reference 29.3 GB (GB of 2³⁰ bytes).
- **The solve** put 52 of its 56 narrowable sites away from `f64` (50 in `f32`, 2 in `bf16`), 1.99× less
  traffic than all-`f64`, and measured 4.324·10⁻⁶ against the budget's 1.118·10⁻², with no widening
  round.
- **The cores.** Every command is one thread and the script does not pin it; the core was not
  sampled. D101 found the scheduler keeping such a command on an X925 core.
