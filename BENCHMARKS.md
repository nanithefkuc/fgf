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
  "hosts": {
    "willow": {"cpu": "Intel Core i5-1135G7", "core": 3, "requested_backend": "v4x"},
    "golden": {"cpu": "Intel Core i7-12700K", "core": 8, "requested_backend": "v3_gfni_crypto"}
  }
}
```

The runner records realized order and completion under a host-wide lock.
Raw logs, source/build digests, and cell-level `evidence.json` stay in
git-ignored `bench-records/`. The publisher rejects incomplete available-host
rounds, changed geometries, missing competitors or controls, invalid values,
and backend mismatches.

## Targeted Mersenne regression measurements

| Request | Public operation | Work unit | Before (ns) | After (ns) | Before / after |
| --- | --- | --- | ---: | ---: | ---: |
| v3_gfni_crypto | M31 mul_into | 64 KiB region | 4018.250 | 3046.250 | 1.3191 |
| v3_gfni_crypto | M31 mul_add | 64 KiB region | 5603.625 | 4637.688 | 1.2083 |
| v3_gfni_crypto | M31 mul_elementwise | 64 KiB region | 5151.125 | 3694.594 | 1.3942 |
| v3_gfni_crypto | QM31 indexed acc += a*b | element | 16.195 | 10.737 | 1.5083 |
| v3_gfni_crypto | QM31 mul_elementwise control | 64 KiB region | 7480.812 | 7479.719 | 1.0001 |
| v4x | M31 mul_into | 64 KiB region | 2420.875 | 2197.594 | 1.1016 |
| v4x | M31 mul_add | 64 KiB region | 3715.312 | 3359.469 | 1.1059 |
| v4x | M31 mul_elementwise | 64 KiB region | 3503.531 | 3208.688 | 1.0919 |
| v4x | QM31 indexed acc += a*b | element | 16.196 | 10.715 | 1.5115 |
| v4x | QM31 mul_elementwise control | 64 KiB region | 5113.750 | 5114.344 | 0.9999 |
| v2 | M31 mul_into | 64 KiB region | 7880.344 | 5911.250 | 1.3331 |
| v2 | M31 mul_add | 64 KiB region | 11101.500 | 9185.875 | 1.2085 |
| v2 | M31 mul_elementwise | 64 KiB region | 10203.781 | 7217.906 | 1.4137 |
| v2 | QM31 indexed acc += a*b | element | 16.149 | 10.737 | 1.5041 |
| v2 | QM31 mul_elementwise control | 64 KiB region | 32184.188 | 32265.031 | 0.9975 |

- Tracking, candidate decisions, source identities, and reproduction:
  [Issue #47](https://github.com/nanithefkuc/fgf/issues/47).
- Host architecture: Willow Cove i5-1135G7; Linux 7.2.7, Rust 1.98.1, isolated core
  3, performance governor, SCHED_IDLE; actual affinity is recorded by the harness.
- Requests resolve M31 and QM31 to `v3`, `v4x`, and `v2`, respectively.
- Both versions run interleaved in each of five rounds; aggregation is the
  median of per-run medians. Ratios use unrounded values.
- Each version runs `just bench-refresh-round m31` with the requested
  `SIMD_BACKEND` and `FEC_GOLDEN_CORE=3`, under `chrt --idle 0 taskset -c 3`.
- Self artifacts use the portable all-feature bench profile; comparator
  artifacts use native-host code generation. Both dependency locks match.
- Native fixtures use page-aligned canonical lanes. Setup, allocation, fixture
  generation, conversion, and preparation remain outside timing; required
  operand checks and normalization remain inside the public calls.
- The native harness validates outputs against Plonky3 before timing. Unchanged
  QM31 elementwise and byte-XOR controls accompany the changed operations.
- These targeted results do not replace the published family snapshots and
  are not Golden Cove measurements. Full Willow Cove and Golden Cove campaigns gate merging.

