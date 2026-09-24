# Provenance

These files are a verbatim copy of `examples/` in the `presk-specs` repository
(`https://github.com/presk-lang/presk-specs`), which is the syntax
source of truth for Presk. They are the acceptance tests of the compiler and
must not be edited here: change them in `presk-specs`, then run
`make sync-examples` (uses `PRESK_SPECS_DIR`, default `../presk-specs`).

Commit: 432e195

The tree is copied, not symlinked, because `import` paths are relative
(`../layernorm/layernorm.psk`) and the repository must stay self-contained on
every machine. `minigpt/target/jetson-agx.pskt` is copied for completeness;
0.1.0 does not read it.
