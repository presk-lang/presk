# The experiments' scripts

Every table and figure the manual's `experiments` and `precision` chapters
report, and every statement the `status` chapter judges, was produced by a
script of this directory, run by a `make` target of the repository's
`Makefile` (its head lists them with what each takes). The scripts are the
record of how a number was taken; the manual says what it means.

What a reader should know before running them:

- **The machine.** The timings of record were taken on the authors'
  measurement machine, named `falicon` in the scripts and the records: an
  Arm server with ten Cortex-X925 and ten Cortex-A725 cores (the manual's
  `experiments` chapter describes it; `RA4_CORES` in the `Makefile` pins
  to its X925 cores). A script that pins cores or expects that machine's
  memory says so; on another machine, override the variables it names.
  Everything else — the bit-exactness checks, the solves, the certificate's
  rows — reproduces anywhere the compiler builds, at the scales the manual
  states, since nothing but the timings depends on the machine.
- **The archives of measurements.** The outputs of record (the sweeps,
  the exact logits, the calibrations, the field's logits) are kept by the
  authors outside this repository, under `~/presk-data/…` and in archives
  whose names the scripts and the `CHANGELOG` cite (`presk-project
  data/…`). A script that reads one of them states the path; without the
  archive it recomputes what it can, or stops with the name of what it
  lacks. The archives are to be published as release assets.
- **Third-party tools.** Some scripts install a tool into an ignored
  directory of the checkout (`.venv`, `.llvm`, `.gappa`, `.tuning`,
  `.gcc`, `.llamacpp`): PyTorch and the conformance tests, LLVM with
  Polly, Gappa, CRAFT, GCC 16, llama.cpp. Each `make` target that needs
  one says so in the `Makefile`'s head; nothing of them is in this
  repository.
- **Nothing here is timed in CI.** The timings are medians over several
  processes on a quiet machine (the manual, "Timing in several
  processes"); the CI builds and tests, and measures nothing.
