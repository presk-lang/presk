# Test data of `presk-explain`

- `x925-frozen.json`: the frozen layout model of the X925 cores
  (`~/presk-data/ra4-model-0.8.0/profile.json`, SHA-256 `cfc9237f…`,
  presk D71, tag `ra4-model-0.8.0`), with its calibration samples
  (`layout.samples`) and the roofline's kernels (`kernels`) left out: only a
  refit reads them, and they are 1.4 MB. Its constants, and so every
  prediction and every plan, are the frozen model's; the documents made
  under either file differ only in the profile's path and checksum
  (checked 2026-10-09).
