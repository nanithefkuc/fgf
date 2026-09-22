# Benchmarks

These measurements cover the public operation shapes on two x86 hosts. Both
hosts ran the same source tree with the `v3_gfni_crypto` backend selected.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Linux 7.2.4, Arch Linux | 1.98.0 | 3 |
| Golden Cove | Intel Core i7-12700K | Linux 7.2.3, CachyOS | 1.98.1 | 8, isolated |

The custom harness warms each operation, then reports the median per-iteration
time from up to 64 batches of 32 iterations. Inputs are deterministic and
coefficient preparation is outside the timed region unless the label says
one-shot. Throughput is logical bytes processed, not total cache traffic.

Run from the crate root:

```sh
FEC_GOLDEN_CORE=<cpu> just bench kernels
FEC_GOLDEN_CORE=<cpu> just bench compare
```

Results below are one complete pinned run on each host. Small nanosecond-scale
cases are timer-resolution-sensitive; the larger row and matrix cases are the
headline measurements.

## Scalar multiplication

The scalar harness executes a fixed batch of element multiplications. Results
are nanoseconds per operation.

| Field | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `Gf8B` | 0.37 ns | 0.34 ns |
| `Gf8D` | 0.36 ns | 0.34 ns |
| `Gf16` | 1.11 ns | 1.33 ns |

## Single-row packed operations

These rows use 64 KiB, 32-byte-aligned buffers. Throughput is GiB/s.

| Field | Operation | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | `mul_add` | 74.80 | 64.59 |
| `Gf8B` | `mul_into` | 63.51 | 60.25 |
| `Gf8D` | `mul_add` | 75.45 | 64.79 |
| `Gf8D` | `mul_into` | 63.58 | 60.43 |
| `Gf16` | `mul_add` | 71.05 | 30.92 |
| `Gf16` | `mul_into` | 61.16 | 31.48 |

The broader 256 KiB operation panel uses ordinary `Vec<u8>` buffers.

| Field | Operation | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | `add_assign` / XOR | 43.50 | 51.08 |
| `Gf8B` | `mul_add` | 44.38 | 49.60 |
| `Gf8B` | `mul_assign` | 61.45 | 67.63 |
| `Gf8B` | `mul_elementwise` | 29.53 | 42.98 |
| `Gf16` | `mul_add` | 41.70 | 41.77 |
| `Gf16` | `mul_assign` | 56.01 | 54.58 |
| `Gf16` | `mul_elementwise` | 25.51 | 29.77 |

At 64 KiB, prepared and one-shot `mul_add` converge because coefficient
preparation is small relative to the row loop.

| Field | Form | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | one-shot | 49.91 | 49.38 |
| `Gf8B` | prepared | 48.91 | 48.79 |
| `Gf16` | one-shot | 45.35 | 41.07 |
| `Gf16` | prepared | 45.38 | 41.10 |

`QuadMersenne31` dispatches to AVX2 above measured row thresholds — one
full vector (32 bytes) for the multiplies, two vectors (64 bytes) for add
and sub, because the mixed vector-plus-tail rows between them lose to the
scalar loop. The interleaved campaign that set both thresholds compares the
public scalar dispatch against the direct AVX2 entries per row length;
ratios below are scalar divided by vector, so above 1.00 the vector kernel
wins. Lunar Lake, one criterion median of two complete pinned runs with a
duplicated control at parity; reproduce with
`FEC_GOLDEN_CORE=<cpu> just bench-save qm31 prime_ntt` then
`FEC_GOLDEN_CORE=<cpu> just bench qm31 prime_ntt`.

| Bytes | `add_assign` | `sub_assign` | `mul_add` | `mul_into` | `mul_assign` | `mul_elementwise` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.34 | 0.35 | 0.40 | 0.57 | 1.19 | 0.47 |
| 16 | 0.35 | 0.37 | 0.62 | 0.82 | 1.03 | 0.68 |
| 24 | 0.40 | 0.40 | 0.86 | 0.93 | 0.82 | 0.93 |
| 32 | 1.50 | 1.42 | 3.55 | 3.72 | 1.66 | 2.63 |
| 40 | 0.50 | 0.57 | 1.07 | 1.14 | 1.68 | 0.98 |
| 64 | 2.63 | 2.40 | 4.10 | 4.56 | 2.66 | 2.69 |
| 128 | 4.37 | 4.35 | 3.94 | 4.51 | 4.00 | 2.73 |
| 256 | 5.07 | 5.59 | 3.91 | 4.56 | 4.32 | 2.46 |
| 2048 | 5.61 | 5.62 | 3.86 | 4.50 | 4.34 | 2.58 |

`vector_elementwise_min_bytes::<QuadMersenne31>()` reports the multiply
threshold so consumers choose schedules by row length instead of paying the
vector entry over the scalar body on sub-vector rows.

## In-place elementwise multiply and broadcast scalar add/sub (prime fields)

256 KiB panel buffers, ordinary `Vec<u8>` geometry. `add_assign` and
`mul_elementwise` are the unchanged three-slice controls beside the new
in-place and broadcast forms. Throughput is GiB/s.

| Field | Operation | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Mersenne31` | `add_assign` | 32.01 | 35.91 |
| `Mersenne31` | `add_assign_scalar` | 51.98 | 52.92 |
| `Mersenne31` | `sub_assign_scalar` | 44.06 | 43.51 |
| `Mersenne31` | `mul_elementwise` | 21.12 | 22.86 |
| `Mersenne31` | `mul_elementwise_assign` | 21.83 | 23.53 |
| `Goldilocks` | `add_assign` | 17.77 | 18.39 |
| `Goldilocks` | `add_assign_scalar` | 24.87 | 23.90 |
| `Goldilocks` | `sub_assign_scalar` | 26.75 | 25.48 |
| `Goldilocks` | `mul_elementwise` | 11.63 | 10.97 |
| `Goldilocks` | `mul_elementwise_assign` | 11.64 | 10.99 |
| `QuadMersenne31` | `add_assign` | 28.72 | 33.00 |
| `QuadMersenne31` | `add_assign_scalar` | 50.15 | 50.81 |
| `QuadMersenne31` | `sub_assign_scalar` | 43.04 | 43.14 |
| `QuadMersenne31` | `mul_elementwise` | 6.72 | 5.84 |
| `QuadMersenne31` | `mul_elementwise_assign` | 6.76 | 5.90 |

- Lunar Lake pairing host: Intel Core Ultra 7 258V, CPU 3 pinned
  (`FEC_GOLDEN_CORE=3`), `v3_gfni_crypto` backend, Linux 7.2.6, rustc 1.98.0,
  one complete `FEC_GOLDEN_CORE=3 just bench kernels` run, median
  per-iteration throughput.
- Golden Cove pairing host: Intel Core i7-12700K, CPU 8 pinned and isolated
  (`FEC_GOLDEN_CORE=8`), `v3_gfni_crypto` backend, Linux CachyOS, rustc
  1.98.1, one complete `FEC_GOLDEN_CORE=8 just bench kernels` run, median
  per-iteration throughput.

## Scatter, gather, and matrix

Each row is 64 KiB. Scatter writes eight rows from one source, gather combines
eight sources into one row, and matrix combines eight sources into eight rows.
Throughput is GiB/s over the harness's logical operation volume.

| Field | Operation | Geometry | Lunar Lake | Golden Cove |
| --- | --- | --- | ---: | ---: |
| `Gf8B` | `mul_add_scatter` | 1 source × 8 rows | 68.53 | 61.07 |
| `Gf16` | `mul_add_scatter` | 1 source × 8 rows | 48.95 | 47.97 |
| `Gf8B` | `mul_add_gather` | 8 sources × 1 row | 64.83 | 69.34 |
| `Gf16` | `mul_add_gather` | 8 sources × 1 row | 68.46 | 77.62 |
| `Gf8B` | `mul_add_matrix` | 8 sources × 8 rows | 124.79 | 114.13 |
| `Gf16` | `mul_add_matrix` | 8 sources × 8 rows | 65.90 | 55.57 |

Overwrite matrix results use 64 KiB rows and ten sources. Throughput counts
source bytes once per call.

| Field | Output rows | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | 2 | 70.08 | 61.88 |
| `Gf8B` | 4 | 39.62 | 36.12 |
| `Gf8B` | 6 | 25.02 | 22.73 |
| `Gf8D` | 2 | 71.65 | 64.94 |
| `Gf8D` | 4 | 43.07 | 39.82 |
| `Gf8D` | 6 | 26.72 | 24.76 |

## Wider binary fields

These are 256 KiB `mul_add` operations. Throughput is GiB/s.

| Field | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `Gf16` polynomial tower | 42.28 | 36.40 |
| `Gf32` polynomial tower | 31.44 | 30.69 |
| `Gf64` polynomial tower | 19.32 | 16.88 |
| `FanPaar16` | 19.09 | 17.03 |
| `FanPaar32` | 7.54 | 6.31 |
| `FanPaar64` | 2.54 | 1.91 |

## Bit-packed GF(2)

The input size is 256 KiB, representing 2,097,152 field elements. Throughput is
GiB/s over packed bytes.

| Operation | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `bits::xor_assign` | 57.42 | 66.06 |
| `bits::and_into` | 31.80 | 48.90 |
| `bits::weight` | 21.24 | 15.66 |
| `bits::dot_product` | 36.73 | 41.45 |
| `bits::xor_range` | 62.84 | 73.91 |
| `bits::xor_range_with` | 64.23 | 73.98 |

For 256 calls over 16-byte rows and the bit range `61..69`, preparing
`bits::XorRange` reduces the per-call cost from 7.08 ns to 2.13 ns on Lunar
Lake and from 5.34 ns to 2.07 ns on Golden Cove.

| Shape | `fgf` `Gf8D` | `reed-solomon-erasure` 6 | Intel ISA-L | `klauspost/reedsolomon` |
| --- | ---: | ---: | ---: | ---: |
| 64 KiB `dst = c * src` | 82.82/64.72 | 70.07/60.37 | 91.10/42.18 | 59.96/56.78 |
| 64 KiB `dst ^= c * src` | 80.42/65.07 | 64.86/50.15 | 79.37/65.35 | 60.85/49.91 |
| 4 KiB, 16 sources, one row | 108.80/82.48 | 61.47/50.28 | 135.33/105.05 | 64.32/50.78 |
| 16 KiB, 16 sources, one row | 94.23/86.54 | 63.30/53.85 | 87.82/95.33 | 69.24/57.81 |
| 4 KiB, 10 sources, 2 rows | 103.10/75.54 | - | 103.10/74.51 | 87.09/75.69 |
| 16 KiB, 10 sources, 2 rows | 85.68/72.04 | - | 84.21/68.55 | 91.81/85.15 |
| 64 KiB, 10 sources, 2 rows | 79.56/73.18 | - | 79.47/67.85 | 86.94/89.44 |
| 4 KiB, 10 sources, 4 rows | 50.73/40.71 | - | 42.48/32.97 | 47.80/40.63 |
| 16 KiB, 10 sources, 4 rows | 47.89/40.81 | - | 33.88/29.82 | 51.69/45.08 |
| 64 KiB, 10 sources, 4 rows | 46.28/39.82 | - | 35.10/31.82 | 52.07/45.49 |
| 4 KiB, 10 sources, 6 rows | 32.72/26.79 | - | 30.76/22.97 | 33.35/28.19 |
| 16 KiB, 10 sources, 6 rows | 29.94/26.08 | - | 25.26/22.05 | 36.49/30.23 |
| 64 KiB, 10 sources, 6 rows | 28.79/26.00 | - | 29.28/23.74 | 36.84/30.45 |

Cells: **Lunar Lake / Golden Cove**, median GiB/s over source bytes.
`-` means unmeasured. Each cell is a median per host over five pinned runs of
the harness in the setup tables below (one run for `just bench compare`). The
harnesses interleave the two arms inside one process, but the aggregation
differs per column, so quotients across columns are not paired benchmark
ratios.

**Lunar Lake setup:** CPU 3, `v3_gfni_crypto`, rustc 1.98.0.
Run commands below from the crate root with `FEC_GOLDEN_CORE=3`.

| Library | Command | Aggregation | Library-specific setup |
| --- | --- | --- | --- |
| `fgf` `Gf8D` | `just bench-klauspost` | Median of five per-run medians | Competitor-harness fixtures, distinct from the fgf-only tables above. |
| `reed-solomon-erasure` 6 | `just bench compare` | One run's medians | `simd-accel`; gathers include zero-fill followed by `mul_slice_xor`. |
| Intel ISA-L 2.32.0 | `just bench-isal` | Median of five per-run medians | System library, runtime dispatch. |
| `klauspost/reedsolomon` v1.14.2 | `just bench-klauspost` | Median of five per-run medians | Go 1.27.1 C archive, `GOMAXPROCS=1`; `WithCustomMatrix`, `Encode` for overwrite and `EncodeIdx` for single-source accumulation. |

**Golden Cove setup:** CPU 8, isolated, `v3_gfni_crypto`, rustc 1.98.1.
Run commands below from the crate root with `FEC_GOLDEN_CORE=8`.

| Library | Command | Aggregation | Library-specific setup |
| --- | --- | --- | --- |
| `fgf` `Gf8D` | `just bench-klauspost` | Median of five per-run medians | Competitor-harness fixtures, distinct from the fgf-only tables above. |
| `reed-solomon-erasure` 6 | `just bench compare` | One run's medians | `simd-accel`; gathers include zero-fill followed by `mul_slice_xor`. |
| Intel ISA-L 2.32.0 | `just bench-isal` | Median of five per-run medians | System library, runtime dispatch. |
| `klauspost/reedsolomon` v1.14.2 | `just bench-klauspost` | Median of five per-run medians | Go 1.27.1 C archive, `GOMAXPROCS=1`; `WithCustomMatrix`, `Encode` for overwrite and `EncodeIdx` for single-source accumulation. |
