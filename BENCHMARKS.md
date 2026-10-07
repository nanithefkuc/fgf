# Benchmarks

Released v2 and v3 absolute measurements occupy separate directories. The
comparison directory contains matched v3 speed factors, not absolute results.
The [canonical labels](benchmarks/labels.md) identify field presentations
independently of public type names.

Every paired cell uses **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)** order. Units appear in table
headers; `-` marks an unavailable measurement. The
[comparison protocol](benchmarks/comparison/README.md) defines matching and ratios.

| Family | Released v2 | v3 | Comparison | Competitors |
| --- | --- | --- | --- | --- |
| `gf8b`, `gf8d` | [v2](benchmarks/v2/gf8.md) | [v3](benchmarks/v3/gf8.md) | [Factors](benchmarks/comparison/gf8.md) | Intel ISA-L, klauspost/reedsolomon |
| `gf16b`, custom `gf16d` | [v2](benchmarks/v2/gf16.md) | [v3](benchmarks/v3/gf16.md) | [Factors](benchmarks/comparison/gf16.md) | GF-Complete, reed-solomon-erasure, reed-solomon-simd, Leopard-RS 1.x |
| `gf32b`, `gf64b`, `fp16`, `fp32`, `fp64` | [v2](benchmarks/v2/wide-binary.md) | [v3](benchmarks/v3/wide-binary.md) | [Factors](benchmarks/comparison/wide-binary.md) | - |
| Bit-packed `gf1` | [v2](benchmarks/v2/gf1.md) | [v3](benchmarks/v3/gf1.md) | [Factors](benchmarks/comparison/gf1.md) | - |
| `m31`, `qm31` | [v2](benchmarks/v2/mersenne31.md) | [v3](benchmarks/v3/mersenne31.md) | [Factors](benchmarks/comparison/mersenne31.md) | Plonky3 |
| `gld` | [v2](benchmarks/v2/goldilocks.md) | [v3](benchmarks/v3/goldilocks.md) | [Factors](benchmarks/comparison/goldilocks.md) | Plonky3 |

## Sources and hosts

| Setting | Value |
| --- | --- |
| Campaign date (UTC) | 2026-10-06 |
| Released baseline | v2.1.0, `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| v3 revision | `4fa978b27d7b7b11c83291e212f47eccb5e7e8f3` |
| Baseline adapter | Benchmark-only API adaptation; released library source unchanged |
| Measurement adapters | Matched benchmark-only timing/logging and canonical input/alignment fixtures; v3 library change is a documentation link |
| CPU | Intel Core i5-1135G7 / Intel Core i7-12700K |
| OS | Arch Linux 7.2.7-arch1-1 / CachyOS Linux 7.2.6-1-cachyos |
| Isolated CPU | 3 / 8 |
| Governor | `performance` / `powersave` |
| SMT | Willow Cove siblings offline; Golden Cove CPU 8 has online sibling 9 |
| Requested process backend | `v4x` / `v3_gfni_crypto` |
| Rust | 1.98.1 |
| Dependencies | `simdispatch` 0.2.0, `archmage` 0.9.29 |
| Portable self executables | Identical binaries on both hosts; default x86-64 target, all features, thin LTO, one codegen unit; published dependencies, outside umbrella patches |
| Native competitors | Built on each host; versions and execution modes appear in family pages |
| Execution | `SCHED_IDLE`, pinned CPU, `RAYON_NUM_THREADS=1`, `GOMAXPROCS=1` |

Process selection is not an operation-level capability claim. Prime-field
routes and the scalar custom `gf16d` route are recorded separately. GF(2^16)
scatter and gather retain their implemented narrower kernel routes.

## Sampling and geometry

Five rounds interleave both versions independently for each family. Competitor
arms interleave within their harness. Each result aggregates the five per-run
medians; no historical values feed the tables. Controls retain the same rounds
and aggregation rather than serving as cross-host calibration.

Self-suite batches calibrate toward a minimum elapsed duration before collecting
bounded repeated samples. Scalar chains use wrapping coefficient streams,
including zero; they do not represent a nonzero-only dependency chain. The
Plonky3 scalar panel instead performs indexed `acc += a * b` operations.

| Protocol | Value |
| --- | --- |
| Self batch calibration | Target 100 µs; minimum 32 or 512 calls according to panel; maximum 1,048,576 calls |
| Self repeated sampling | At most 64 samples, 250 ms deadline |
| Scalar batch | 1,048,576 operations |
| Self controls | Identical 256 KiB byte-XOR operation at process start and end |
| Native GF(2^16) checks | All 65,536 mapped values, all 256 basis products, and every measured geometry checked before timing |
| Setup excluded | Allocation, deterministic input construction, caller-owned coefficient preparation, representation conversion, and native map construction; one-shot API internal preparation remains timed |
| Self single-row numerator | Buffer bytes |
| Self scatter/gather/matrix numerator | Row bytes × sources × destinations, unless the panel explicitly states a source-byte numerator |
| ISA-L/Go and native GF(2^16) numerator | Row bytes × sources; destination count does not multiply the numerator |
| Plonky3 packed numerator | Two region extents; 131,072 logical bytes for the 64 KiB-region workload |
| GF(2) numerator | Packed bytes; row-pair latency is per compared pair |

Native GF(2^16) arms retain their own encodings. Validated field maps convert
inputs and coefficients outside timing; native encodings are not relabeled
`gf16d`. That label denotes the custom v3 presentation and has no invented v2
counterpart.

## Number format

A table uses a uniform total digit count, including leading zeros, sufficient
for at least three significant figures at its recorded resolution. Rounding is
round-half-up; integer parts remain intact. Ratios use five total digits.
Throughput and speed factors derive from unrounded host-local latency
aggregates, not displayed cells.

## Reproduction

Prepare the released baseline and run the paired campaign from each v3 checkout:

```sh
just doctor
just bench-prepare-v2 /tmp/fgf-v2-bench
export RUSTUP_TOOLCHAIN=1.98.1 RAYON_NUM_THREADS=1 GOMAXPROCS=1
export FEC_GOLDEN_CORE=3 SIMD_BACKEND=v4x # Willow Cove (i5-1135G7)
# Golden Cove (i7-12700K): FEC_GOLDEN_CORE=8 SIMD_BACKEND=v3_gfni_crypto
just bench-paired /tmp/fgf-v2-bench /tmp/fgf-paired-records
```

The measured campaign also supplied `FGF_SELF_ARTIFACTS` pointing at the shared
portable `v2/` and `v3/` executable directories. Without that setting, the
recipe builds the self executables locally. A host lock prevents overlapping
campaigns, and completion markers and shuffled-order records accompany the logs.

Collect both host directories beneath one records root with the campaign
metadata, then publish and verify without overwriting:

```sh
just bench-publish <records-root> benchmarks
just bench-publish <records-root> benchmarks --verify
```

`metadata.json` declares `v2_revision`, `v3_revision`, `toolchain`, `rounds`
(five), and `host_order` (private identifiers ordered Willow Cove, Golden Cove).
Its `hosts` object uses that insertion order and gives each host's CPU model,
integer `core`, and string
`requested_backend`. Environment captures and source/build checksums accompany
the manifest; host directories contain the runner's original logs and markers.

The publisher rejects incomplete rounds, changed workload sets, malformed
records, invalid timing values, missing controls, and backend mismatches. Raw
logs, build/source checksums, environment captures, and the cell-level evidence
index remain in git-ignored `bench-records/`.
