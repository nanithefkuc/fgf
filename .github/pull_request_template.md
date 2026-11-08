<!--
Thank you for contributing to fgf. Keep the subject to `fgf: short verb
phrase`, one crate per PR, and `just validate` green before opening.
-->

## Summary

What changed and why.

## Checks

- [ ] `just validate` is green (lint, deps, doc, coverage, unsafe-check).
- [ ] `CHANGELOG.md` entry for every user-visible change.

## Kernel and acceptance evidence

Complete this section for every PR that changes a kernel entry, a dispatch
threshold, or a file under `src/kernel/x86/`.

- Tiers touched and their `just isa-accept <tier>` results, per acceptance
  host (Willow Cove, Golden Cove), including every downgrade iteration:
  - Willow Cove (v4x):
  - Golden Cove (v3_gfni_crypto):
- For PRs touching files on the GFNI, AVX-512, or dispatch coverage lists
  (`COV_IGNORE_GFNI` / `COV_IGNORE_AVX512` / `COV_IGNORE_DISPATCH` in
  `crate.just`), the `just cover-native` group figures for the touched lists
  (wrapper-line coverage; kernel bodies are evidenced by receipts), from an
  acceptance host that executes both ISAs:
  - GFNI list:
  - AVX-512 list:
  - Dispatch list:
- For performance-motivated changes: the paired before/after measurement
  record and its `BENCHMARKS.md` pointer.
