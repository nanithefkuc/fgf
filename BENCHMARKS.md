# Benchmarks

These measurements cover the public operation shapes on two x86 hosts. Both
hosts ran the same source tree with the `v3_gfni_crypto` backend selected.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Linux 7.2.4, Arch Linux | 1.98.0 | 1 |
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

## Competitor matrix

| Shape | `fgf` `Gf8D` | `reed-solomon-erasure` 6 | Intel ISA-L | `klauspost/reedsolomon` |
| --- | ---: | ---: | ---: | ---: |
| 64 KiB `dst = c * src` | 85.97/60.43 | 69.04/- | 90.42/- | 60.91/- |
| 64 KiB `dst ^= c * src` | 79.27/64.79 | 64.25/- | 79.16/- | 59.09/- |
| 4 KiB, 16 sources, one row | 106.33/83.61 | 61.53/50.32 | 135.03/- | 63.45/- |
| 16 KiB, 16 sources, one row | 95.00/85.69 | 65.21/53.68 | 91.23/- | 69.81/- |
| 4 KiB, 10 sources, 2 rows | 103.94/- | - | 103.94/- | 86.90/- |
| 16 KiB, 10 sources, 2 rows | 90.99/- | - | 83.02/- | 92.59/- |
| 64 KiB, 10 sources, 2 rows | 80.37/64.94 | - | 82.06/- | 88.09/- |
| 4 KiB, 10 sources, 4 rows | 51.97/- | - | 45.25/- | 49.41/- |
| 16 KiB, 10 sources, 4 rows | 48.80/- | - | 34.48/- | 52.84/- |
| 64 KiB, 10 sources, 4 rows | 47.20/39.82 | - | 36.76/- | 53.39/- |
| 4 KiB, 10 sources, 6 rows | 34.34/- | - | 31.29/- | 34.06/- |
| 16 KiB, 10 sources, 6 rows | 30.74/- | - | 25.58/- | 37.41/- |
| 64 KiB, 10 sources, 6 rows | 29.50/24.76 | - | 30.60/- | 37.45/- |

Cells: **Lunar Lake / Golden Cove**, median GiB/s over source bytes.
`-` means unmeasured. Different runs and fixtures back the cells; their
quotients are not paired benchmark ratios.

**Lunar Lake setup:** CPU 1, `v3_gfni_crypto`, rustc 1.98.0.
Run commands below from the crate root with `FEC_GOLDEN_CORE=1`.

| Library | Command | Aggregation | Library-specific setup |
| --- | --- | --- | --- |
| `fgf` `Gf8D` | `just bench-klauspost` | Median of five per-run medians | Competitor-harness fixtures, distinct from the fgf-only tables above. |
| `reed-solomon-erasure` 6 | `just bench compare` | One run's medians | `simd-accel`; gathers include zero-fill followed by `mul_slice_xor`. |
| Intel ISA-L 2.32.0 | `just bench-isal` | Median of five per-run medians | System library, runtime dispatch. |
| `klauspost/reedsolomon` v1.14.2 | `just bench-klauspost` | Median of five per-run medians | Go 1.27.1 C archive, `GOMAXPROCS=1`; `WithCustomMatrix`, `Encode` for overwrite and `EncodeIdx` for single-source accumulation. |

**Golden Cove sources:**

- **`fgf`:** single-row packed table for single-source operations; overwrite
  matrix table for encodes; the two-host `just bench compare` run for gathers.
- **`reed-solomon-erasure`:** the two-host `just bench compare` run for gathers.
- **ISA-L and klauspost:** unmeasured.
