# Changelog

Notable changes follow [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/). Entries link to public change
records; those records hold decisions, implementation details, and migration
notes. Historical records preserve the supplied changelog without implying
fresh verification.

## [Unreleased]

### Changed

- Use Issue/PR-backed release summaries and preserve historical details in
  linked release records. [#79](https://github.com/nanithefkuc/fgf/issues/79).
- Consolidate comparison packages under `benches/compare/`, trim repository
  instructions, and retire stale benchmark tooling and ISA descriptions.
  [#101](https://github.com/nanithefkuc/fgf/issues/101).

## [4.0.0] - 2026-11-06

### Added

- Stable API inventory and release checks. [#80](https://github.com/nanithefkuc/fgf/issues/80).

### Changed

- **Breaking:** Consolidate public paths, seal trait plumbing, and use `FieldBuffer` for byte operations; rename elementwise overwrite APIs and revise map construction and errors. Migrate with the [3.x guide](https://github.com/nanithefkuc/fgf/wiki/Migrating-from-3.x). [#80](https://github.com/nanithefkuc/fgf/issues/80).

### Removed

- **Breaking:** Remove prepared field maps and their row APIs; use compact per-element maps. Retire the 3.x benchmark publication tooling. [#80](https://github.com/nanithefkuc/fgf/issues/80).

### Fixed

- Preserve source bytes in unit-coefficient scatter and validate binary relations and map source rows at compile time; update macOS CI. [#80](https://github.com/nanithefkuc/fgf/issues/80).

Release PR: [#78](https://github.com/nanithefkuc/fgf/pull/78).

## [3.0.2] - 2026-10-20

### Added

- Reciprocal/division rows, scalar dot products, prepared field maps, mixed map composition, and missing row-operation siblings. [#81](https://github.com/nanithefkuc/fgf/issues/81).

### Changed

- **Breaking:** Cap extension degrees at 64 and reserve `Embedding` for proper inclusions; migrate equal-degree maps to `Isomorphism`. [#81](https://github.com/nanithefkuc/fgf/issues/81).

### Removed

- Remove NEON and Wasm SIMD acceleration; those targets use portable scalar kernels. [#81](https://github.com/nanithefkuc/fgf/issues/81).

### Fixed

- Correct prime-field and quadratic multiplication paths while preserving raw-input contracts. [#81](https://github.com/nanithefkuc/fgf/issues/81).

Release PR: [#67](https://github.com/nanithefkuc/fgf/pull/67).

## [3.0.1] - 2026-10-19

### Added

- Generic finite fields, extension specifications, characteristic-independent maps, residue/index conversions, and consumer/release acceptance checks. [#82](https://github.com/nanithefkuc/fgf/issues/82).

### Changed

- **Breaking:** Unify fields under `FiniteField` and `Elem`, seal field traits, revise residue lane widths, and replace custom towers with extension specifications. Migrate with the [2.x guide](https://github.com/nanithefkuc/fgf/wiki/Migrating-from-2.x). [#82](https://github.com/nanithefkuc/fgf/issues/82).

### Fixed

- Reject invalid offset-gather geometry before mutation on 32-bit targets and use portable dispatch on i686. [#82](https://github.com/nanithefkuc/fgf/issues/82).

## [3.0.0] - 2026-10-07 [YANKED]

### Changed

- **Breaking:** Introduce the binary field/element model, recursive towers, ordered bases, and embeddings. This accidental, yanked release is unsupported; migrate to 3.0.1 using the [2.x guide](https://github.com/nanithefkuc/fgf/wiki/Migrating-from-2.x). [#83](https://github.com/nanithefkuc/fgf/issues/83).

## [2.1.0] - 2026-10-02

### Added

- Runnable educational examples and AVX-512 quadratic-extension operations. [#84](https://github.com/nanithefkuc/fgf/issues/84).

### Changed

- Refresh benchmark publication and byte-field matrix and prime-field kernels. [#84](https://github.com/nanithefkuc/fgf/issues/84).

### Fixed

- Validate staged matrix-provider sources and correct benchmark alignment and coverage configuration. [#84](https://github.com/nanithefkuc/fgf/issues/84).

Release PR: [#37](https://github.com/nanithefkuc/fgf/pull/37).

## [2.0.0] - 2026-09-30

### Added

- AVX-512 byte-field, tower, prime-field, XOR, and population-count operations. [#85](https://github.com/nanithefkuc/fgf/issues/85).

### Changed

- **Breaking:** Move to `simdispatch = "=0.2.0"`; consumers pinning both crates must upgrade together. Reorganize unstable kernel paths; migrate direct callers using the release record. [#85](https://github.com/nanithefkuc/fgf/issues/85).

### Removed

- Remove deferred and rejected unstable kernel variants. [#85](https://github.com/nanithefkuc/fgf/issues/85).

### Fixed

- Correct prime-field unit-coefficient offset matrices and prepared overwrite-gather routing; refine alignment handling. [#85](https://github.com/nanithefkuc/fgf/issues/85).

## [1.2.2] - 2026-09-30

### Changed

- Extend vector broadcast addition/subtraction across binary fields. **Breaking:** Direct unstable SSE broadcast callers migrate to the SSE2 entry. [#86](https://github.com/nanithefkuc/fgf/issues/86).

### Fixed

- Correct prime-field unit-coefficient offset-matrix results. [#86](https://github.com/nanithefkuc/fgf/issues/86).

## [1.2.1] - 2026-09-24

### Fixed

- Dispatch binary-field broadcast addition/subtraction through vector kernels instead of the portable loop. [#87](https://github.com/nanithefkuc/fgf/issues/87).

Release PR: [#23](https://github.com/nanithefkuc/fgf/pull/23).

## [1.2.0] - 2026-09-24

### Added

- In-place elementwise multiplication, broadcast addition/subtraction, and field-family competitor benchmark commands. [#88](https://github.com/nanithefkuc/fgf/issues/88).

### Changed

- Use safe capability-token entries for ARM/Wasm kernels and improve benchmark and Miri workflows; direct unstable callers migrate to token-bearing entries. [#88](https://github.com/nanithefkuc/fgf/issues/88).

### Removed

- Retire superseded comparison and attribution harnesses. [#88](https://github.com/nanithefkuc/fgf/issues/88).

## [1.1.1] - 2026-09-18

### Changed

- **Breaking:** Make `internals` a visibility-only facade; direct unstable callers migrate to `fgf::internals` paths. [#89](https://github.com/nanithefkuc/fgf/issues/89).

## [1.1.0] - 2026-09-16

### Fixed

- Correct wrapped-subtraction reduction in AVX2 Goldilocks products. [#90](https://github.com/nanithefkuc/fgf/issues/90).

### Added

- AVX2 quadratic-extension kernels, an elementwise routing query, and a klauspost competitor harness. [#90](https://github.com/nanithefkuc/fgf/issues/90).

### Changed

- Prefetch destinations in fused overwrite kernels. [#90](https://github.com/nanithefkuc/fgf/issues/90).

## [1.0.0] - 2026-09-14

### Added

- First stable release, `alloc`-gated prepared collections, row addition, prepared bit ranges, and binary negation. [#91](https://github.com/nanithefkuc/fgf/issues/91).

### Changed

- **Breaking:** Require Rust 1.93; split prepared plans, normalize operation names/argument order, privatize element storage, and revise value/byte contracts. Migrate using the linked release record. [#91](https://github.com/nanithefkuc/fgf/issues/91).

### Fixed

- Correct empty geometry, prepared-matrix bounds, wide population counts, and overwrite semantics. [#91](https://github.com/nanithefkuc/fgf/issues/91).

## [0.7.1] - 2026-09-09

### Changed

- Pin backend selection to the published `simdispatch` release. [#92](https://github.com/nanithefkuc/fgf/issues/92).

## [0.7.0] - 2026-08-24

### Added

- Native scalar and bit-packed GF(2), coverage gating, and expanded public-contract tests. [#93](https://github.com/nanithefkuc/fgf/issues/93).

### Changed

- Improve offset gathers, binary gathers, bit reductions, and quadratic-extension kernels. [#93](https://github.com/nanithefkuc/fgf/issues/93).

## [0.6.0] - 2026-08-23

### Added

- Quadratic Mersenne31 extension arithmetic and checked operations; existing field encodings are unchanged. [#94](https://github.com/nanithefkuc/fgf/issues/94).

## [0.5.0] - 2026-08-23

### Added

- Mersenne31 and Goldilocks scalar arithmetic and packed x86 operations. [#95](https://github.com/nanithefkuc/fgf/issues/95).

### Changed

- Route addition/subtraction by field and improve prime-field kernels; existing binary encodings are unchanged. [#95](https://github.com/nanithefkuc/fgf/issues/95).

### Fixed

- Use fused prime-field overwrite multiplication instead of copy-then-scale. [#95](https://github.com/nanithefkuc/fgf/issues/95).

## [0.4.0] - 2026-08-22

### Added

- Reed–Solomon byte-field representation, overwrite gather/matrix, and offset-addressed matrix operations. [#96](https://github.com/nanithefkuc/fgf/issues/96).

### Changed

- **Breaking:** Rename `Gf8`/`gf8` to `Gf8B`/`gf8b`; update imports. Existing field bytes are unchanged. [#96](https://github.com/nanithefkuc/fgf/issues/96).

## [0.3.0] - 2026-08-08

### Changed

- **Breaking:** Rename crate `fff` to `fgf` and move backend selection to `simdispatch`; update dependencies/imports and replace `FFF_BACKEND` with `SIMD_BACKEND`. [#97](https://github.com/nanithefkuc/fgf/issues/97).

## [0.2.0] - 2026-07-31

### Added

- Wider binary-field SIMD coverage and unstable direct-kernel access. [#98](https://github.com/nanithefkuc/fgf/issues/98).

### Changed

- **Breaking:** Extend backend reporting and elementwise capability results; update consumers matching the backend set. Refine kernel routing, stores, alignment, and tails. [#98](https://github.com/nanithefkuc/fgf/issues/98).

### Fixed

- Repair Wasm tests and portable all-target builds. [#98](https://github.com/nanithefkuc/fgf/issues/98).

### Removed

- Remove crates.io publishing configuration; distribution uses Git. [#98](https://github.com/nanithefkuc/fgf/issues/98).

## [0.1.1] - 2026-07-29

### Added

- Prepared operations, packing conversions, backend queries, uniform element APIs, and release documentation. [#99](https://github.com/nanithefkuc/fgf/issues/99).

### Changed

- Seal kernel contracts, normalize geometry, and retain vectorized prepared operations. [#99](https://github.com/nanithefkuc/fgf/issues/99).

## [0.1.0] - 2026-07-29

### Added

- Initial public release. [#100](https://github.com/nanithefkuc/fgf/issues/100).

[Unreleased]: https://github.com/nanithefkuc/fgf/compare/v4.0.0...HEAD
[4.0.0]: https://github.com/nanithefkuc/fgf/compare/v3.0.2...v4.0.0
[3.0.2]: https://github.com/nanithefkuc/fgf/compare/v3.0.1...v3.0.2
[3.0.1]: https://github.com/nanithefkuc/fgf/compare/v3.0.0...v3.0.1
[3.0.0]: https://github.com/nanithefkuc/fgf/compare/v2.1.0...v3.0.0
[1.0.0]: https://github.com/nanithefkuc/fgf/compare/v0.7.1...v1.0.0
[0.7.1]: https://github.com/nanithefkuc/fgf/compare/v0.7.0...v0.7.1
[0.6.0]: https://github.com/nanithefkuc/fgf/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/nanithefkuc/fgf/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/nanithefkuc/fgf/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/nanithefkuc/fgf/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/nanithefkuc/fgf/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/nanithefkuc/fgf/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nanithefkuc/fgf/releases/tag/v0.1.0

[1.1.0]: https://github.com/nanithefkuc/fgf/compare/v1.0.1...v1.1.0
[1.1.1]: https://github.com/nanithefkuc/fgf/compare/v1.1.0...v1.1.1
[1.2.0]: https://github.com/nanithefkuc/fgf/compare/v1.1.1...v1.2.0
[1.2.1]: https://github.com/nanithefkuc/fgf/compare/v1.2.0...v1.2.1
[1.2.2]: https://github.com/nanithefkuc/fgf/compare/v1.2.1...v1.2.2
[2.0.0]: https://github.com/nanithefkuc/fgf/compare/v1.2.2...v2.0.0
[2.1.0]: https://github.com/nanithefkuc/fgf/compare/v2.0.0...v2.1.0
