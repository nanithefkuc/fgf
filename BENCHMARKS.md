# Benchmarks

One campaign refreshes every family, both released-v2 and current-v3 workloads,
and all wired competitors. Generic prime residues compare the current public
API across backend requests; they have no released-v2 counterpart.

Cells use **Willow Cove / Golden Cove** order. Both hosts run the complete
campaign; unavailable operations or competitor measurements remain `-`.
[Field labels](benchmarks/labels.md) identify the exact presentations;
[comparison rules](benchmarks/comparison/README.md) define matched work and ratios.

## Results

| Field presentation | Released v2 | Current v3 | Comparison | Competitors |
| --- | --- | --- | --- | --- |
| Byte polynomial fields | [gf8](benchmarks/v2/gf8.md) | [gf8](benchmarks/v3/gf8.md) | [v3/v2](benchmarks/comparison/gf8.md) | Intel ISA-L, klauspost/reedsolomon |
| Degree-16 extensions | [gf16](benchmarks/v2/gf16.md) | [gf16](benchmarks/v3/gf16.md) | [v3/v2](benchmarks/comparison/gf16.md) | GF-Complete, reed-solomon-erasure, reed-solomon-simd, Leopard-RS 1.x |
| Wider binary extensions | [wide binary](benchmarks/v2/wide-binary.md) | [wide binary](benchmarks/v3/wide-binary.md) | [v3/v2](benchmarks/comparison/wide-binary.md) | - |
| Bit-packed GF(2) | [gf1](benchmarks/v2/gf1.md) | [gf1](benchmarks/v3/gf1.md) | [v3/v2](benchmarks/comparison/gf1.md) | - |
| Mersenne specializations | [m31, qm31](benchmarks/v2/mersenne31.md) | [m31, qm31](benchmarks/v3/mersenne31.md) | [v3/v2](benchmarks/comparison/mersenne31.md) | Plonky3 |
| Goldilocks specialization | [gld](benchmarks/v2/goldilocks.md) | [gld](benchmarks/v3/goldilocks.md) | [v3/v2](benchmarks/comparison/goldilocks.md) | Plonky3 |
| Generic prime residues | - | [Residues](benchmarks/v3/residue.md) | [AVX2 and scalar](benchmarks/comparison/residue.md) | - |
| Field maps (inclusions, isomorphisms, Frobenius) | - | - | - | - |

Absolute pages contain public-operation timings and competitor comparisons.
The residue tables put row lengths in columns, with expandable operation panels
instead of appending separate tuning campaigns. Version and backend factors
live only in the comparison pages; values below one remain visible.

## Campaign

| Setting | Willow Cove | Golden Cove |
| --- | --- | --- |
| Campaign date (UTC) | 2026-10-13 | 2026-10-13 |
| CPU | Intel Core i5-1135G7 | Intel Core i7-12700K |
| Availability | Measured | Measured |
| OS | Arch Linux 7.2.7-arch1-1 | CachyOS Linux 7.2.6-1-cachyos |
| Isolated CPU | 3 | 8; isolated set 8–11 |
| Governor | performance | performance |
| SMT | Siblings offline | Enabled; sibling 9 isolated and online |
| Requested process backend | v4x | v3_gfni_crypto |
| Rust | 1.98.1 | 1.98.1 |
| Execution | SCHED_IDLE, pinned CPU; RAYON_NUM_THREADS=1, GOMAXPROCS=1 | SCHED_IDLE, pinned CPU; RAYON_NUM_THREADS=1, GOMAXPROCS=1 |

| Source and method | Value |
| --- | --- |
| Released baseline | v2.1.0, `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Current library | Working tree based on v3.0.1, revision `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Current source archive SHA-256 | `778804696da94a736b05ca29dd1c686c2c7d42813d6e166ad5441c77ef5e84f9` |
| Baseline source archive SHA-256 | `9c36b80d2f5fb5c585f581854f6e4ba6a4e4fb564746077ac5903d2452aa6c42` |
| Self build | Portable x86-64 target, all features, thin LTO, one codegen unit; published dependency pins |
| Competitor build | Native host builds; versions and packing widths in family pages |
| Sampling | Median of five per-run medians; no historical timing inputs |
| Version order | Both versions in shuffled order within every family round |
| Residue order | Same artifact per host under routed, v3, scalar requests; rotating order within every round |
| Residue placements | Median over offsets 0, 16, 32 within each run |
| Self calibration | Target 100 µs; at most 64 samples, 250 ms deadline |
| Controls | Unchanged byte-XOR at process entry/exit; duplicate fgf competitor arms |
| Number format | Uniform total digit count per table, at least three significant figures; round-half-up; ratios keep five digits |

## Caveats

- A process backend is not an operation-level claim. Field caps and residue
  operation routes are retained in the evidence; short rows can use narrower
  or portable paths.
- Allocation, fixture generation, reusable preparation, and representation
  conversion are excluded unless the panel explicitly includes them.
- Throughput numerators belong to each panel. Self multi-row panels and native
  competitor panels do not use interchangeable byte counts.
- Scalar dependency chains include zero coefficients. Plonky3 scalar panels
  instead perform indexed multiply-accumulate operations.
- Native GF(2^16) libraries keep their own encodings; validated field maps are
  outside timing. The custom `gf16d` presentation is v3-only and scalar-routed.
- Backend comparisons measure public calls under downgrade requests, not
  isolated instruction bodies or a separate released residue implementation.
- Differences comparable to control or round variability do not establish a win.
- Golden Cove has no live AVX-512 capability. Its routed residue calls use
  AVX2 or scalar; a routed-versus-AVX2 factor on that host is not an
  AVX-512 comparison.
- Short-call and native competitor results can vary across rounds. Per-case
  duplicate-fgf controls remain in the GF(2^16) pages; their variability
  also limits interpretation of small differences.

## Reproduce

Run the complete campaign on each host from an isolated checkout outside
umbrella dependency patches, then collect both host directories before
publishing:

```sh
just doctor
just bench-prepare-v2 /tmp/fgf-v2-bench
export RUSTUP_TOOLCHAIN=1.98.1 RAYON_NUM_THREADS=1 GOMAXPROCS=1
export HOST=willow FEC_GOLDEN_CORE=3 SIMD_BACKEND=v4x
# Golden Cove: HOST=golden FEC_GOLDEN_CORE=8 SIMD_BACKEND=v3_gfni_crypto
mkdir -p "/tmp/fgf-records/$HOST"
{
  date -u; uname -a; lscpu; rustc -Vv; cargo -V
  cat "/sys/devices/system/cpu/cpu$FEC_GOLDEN_CORE/cpufreq/scaling_governor"
  cat "/sys/devices/system/cpu/cpu$FEC_GOLDEN_CORE/topology/thread_siblings_list"
} > "/tmp/fgf-records/$HOST/environment.txt"
just bench-paired /tmp/fgf-v2-bench "/tmp/fgf-records/$HOST"
# Supply metadata.json and collect both complete host directories.
just bench-publish /tmp/fgf-records benchmarks
just bench-publish /tmp/fgf-records benchmarks --verify
```

Use this manifest shape, replacing the current revision with the measured
checkout's revision and retaining source/artifact digests with the raw records:

```json
{
  "v2_revision": "c20465ba22d078e0fb115fb6a99fa77027992c90",
  "v3_revision": "<measured-revision>",
  "toolchain": "1.98.1",
  "rounds": 5,
  "host_order": ["willow", "golden"],
  "families": ["gf", "gdl", "m31", "gf2", "prime", "maps"],
  "hosts": {
    "willow": {"cpu": "Intel Core i5-1135G7", "core": 3, "requested_backend": "v4x"},
    "golden": {"cpu": "Intel Core i7-12700K", "core": 8, "requested_backend": "v3_gfni_crypto"}
  }
}
```

`families` declares the campaign inventory the publisher verifies for
completeness. Trees recorded before the maps suite carry no key and verify
as the legacy five-family inventory.

The runner records realized order and completion under a host-wide lock.
Raw logs, source/build digests, and cell-level `evidence.json` stay in
git-ignored `bench-records/`. The publisher rejects incomplete available-host
rounds, changed geometries, missing competitors or controls, invalid values,
and backend mismatches.
