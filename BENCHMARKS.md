# Benchmarks

These measurements cover the public operation shapes on two x86 hosts. Both
hosts ran the same source tree with the `v3_gfni_crypto` backend selected.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Linux 7.2.6, Arch Linux | 1.98.0 | 3 |
| Golden Cove | Intel Core i7-12700K | Linux 7.2.6, CachyOS | 1.98.1 | 8, isolated |

The custom harness warms each operation, then reports the median per-iteration
time from up to 64 batches of 32 iterations. Inputs are deterministic and
coefficient preparation is outside the timed region unless the label says
one-shot. Throughput is logical bytes processed, not total cache traffic.

Run from the crate root:

```sh
FEC_GOLDEN_CORE=<cpu> just bench-gf-comp
FEC_GOLDEN_CORE=<cpu> just bench-gdl-comp
FEC_GOLDEN_CORE=<cpu> just bench-m31-comp
FEC_GOLDEN_CORE=<cpu> just bench-gf2-comp
```

Each `bench-<field>-comp` recipe runs five complete rounds interleaving that
field family's self suite with every competitor harness wired for the family,
shuffling the unit order within each round and printing the realized order.
The granular entry points (`just bench kernels`, `just bench compare`, `just
bench prime_ntt`) serve self-benchmark reruns; the competitor harnesses run
only through the `-comp` recipes.
Results below are one campaign's medians per field family on each host. Small
nanosecond-scale cases are timer-resolution-sensitive; the larger row and matrix
cases are the headline measurements.

## Scalar multiplication

The scalar harness executes a fixed batch of element multiplications. Results
are nanoseconds per operation.

| Field | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `Gf8B` | 0.38 ns | 0.34 ns |
| `Gf8D` | 0.38 ns | 0.34 ns |
| `Gf16` | 1.18 ns | 1.31 ns |

## Single-row packed operations

These rows use 64 KiB, 32-byte-aligned buffers. Throughput is GiB/s.

| Field | Operation | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | `mul_add` | 69.70 | 64.58 |
| `Gf8B` | `mul_into` | 75.76 | 64.60 |
| `Gf8D` | `mul_add` | 70.51 | 64.75 |
| `Gf8D` | `mul_into` | 75.50 | 64.65 |
| `Gf16` | `mul_add` | 63.81 | 56.56 |
| `Gf16` | `mul_into` | 72.39 | 63.87 |

The broader 256 KiB operation panel uses ordinary `Vec<u8>` buffers.

| Field | Operation | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | `add_assign` / XOR | 43.51 | 51.11 |
| `Gf8B` | `mul_add` | 44.25 | 48.98 |
| `Gf8B` | `mul_assign` | 59.97 | 67.72 |
| `Gf8B` | `mul_elementwise` | 31.43 | 39.27 |
| `Gf16` | `mul_add` | 41.09 | 41.33 |
| `Gf16` | `mul_assign` | 53.58 | 55.15 |
| `Gf16` | `mul_elementwise` | 27.42 | 27.97 |

At 64 KiB, prepared and one-shot `mul_add` converge because coefficient
preparation is small relative to the row loop.

| Field | Form | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | one-shot | 51.19 | 47.95 |
| `Gf8B` | prepared | 48.40 | 48.42 |
| `Gf16` | one-shot | 44.49 | 40.79 |
| `Gf16` | prepared | 45.27 | 40.79 |

`QuadMersenne31` dispatches to AVX2 above measured row thresholds — one
full vector (32 bytes) for the multiplies, two vectors (64 bytes) for add
and sub, because the mixed vector-plus-tail rows between them lose to the
scalar loop. The interleaved campaign that set both thresholds compares the
public scalar dispatch against the direct AVX2 entries per row length;
ratios below are scalar divided by vector, so above 1.00 the vector kernel
wins. Two complete pinned criterion runs per host, with the duplicated
scalar control pooled into the scalar estimate; reproduce with
`FEC_GOLDEN_CORE=<cpu> just bench-save prime_ntt` then
`FEC_GOLDEN_CORE=<cpu> just bench prime_ntt`.

| Bytes | `add_assign` | `sub_assign` | `mul_add` | `mul_into` | `mul_assign` | `mul_elementwise` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.37/0.35 | 0.38/0.35 | 0.45/0.46 | 0.64/0.56 | 1.23/1.15 | 0.54/0.50 |
| 16 | 0.38/0.42 | 0.60/0.45 | 0.66/0.65 | 0.87/0.81 | 1.04/0.93 | 0.77/0.72 |
| 24 | 0.44/0.54 | 0.44/0.58 | 0.88/0.88 | 0.99/0.94 | 0.84/0.81 | 0.98/0.91 |
| 32 | 1.65/2.23 | 1.53/1.99 | 1.86/1.85 | 2.10/1.94 | 1.10/1.01 | 1.92/1.36 |
| 40 | 0.54/0.75 | 0.64/0.83 | 0.68/0.67 | 1.15/1.09 | 1.49/1.28 | 0.75/0.68 |
| 64 | 1.50/1.66 | 1.30/1.30 | 1.43/1.28 | 1.49/1.38 | 1.10/0.96 | 1.21/1.08 |
| 128 | 1.38/1.25 | 1.34/0.63 | 0.98/1.14 | 1.19/1.10 | 1.09/1.09 | 1.05/1.04 |
| 256 | 1.18/1.09 | 1.21/1.11 | 0.98/1.04 | 1.03/1.00 | 1.01/1.02 | 1.03/1.02 |
| 2048 | 1.02/1.00 | 1.00/1.03 | 0.97/0.96 | 0.95/0.96 | 0.96/0.97 | 1.00/1.00 |

`vector_elementwise_min_bytes::<QuadMersenne31>()` reports the multiply
threshold so consumers choose schedules by row length instead of paying the
vector entry over the scalar body on sub-vector rows.

## In-place elementwise multiply and broadcast scalar add/sub (prime fields)

256 KiB panel buffers, ordinary `Vec<u8>` geometry. `add_assign` and
`mul_elementwise` are the unchanged three-slice controls beside the new
in-place and broadcast forms. Throughput is GiB/s.

| Field | Operation | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Mersenne31` | `add_assign` | 39.47 | 34.27 |
| `Mersenne31` | `add_assign_scalar` | 57.31 | 51.76 |
| `Mersenne31` | `sub_assign_scalar` | 47.29 | 43.50 |
| `Mersenne31` | `mul_elementwise` | 24.83 | 21.30 |
| `Mersenne31` | `mul_elementwise_assign` | 25.46 | 21.73 |
| `Goldilocks` | `add_assign` | 20.08 | 18.39 |
| `Goldilocks` | `add_assign_scalar` | 26.54 | 23.91 |
| `Goldilocks` | `sub_assign_scalar` | 28.79 | 25.48 |
| `Goldilocks` | `mul_elementwise` | 13.05 | 10.52 |
| `Goldilocks` | `mul_elementwise_assign` | 13.06 | 10.51 |
| `QuadMersenne31` | `add_assign` | 34.16 | 33.01 |
| `QuadMersenne31` | `add_assign_scalar` | 53.73 | 50.82 |
| `QuadMersenne31` | `sub_assign_scalar` | 47.13 | 43.15 |
| `QuadMersenne31` | `mul_elementwise` | 7.20 | 5.73 |
| `QuadMersenne31` | `mul_elementwise_assign` | 7.27 | 5.80 |

- Lunar Lake pairing host: Intel Core Ultra 7 258V, CPU 3 pinned
  (`FEC_GOLDEN_CORE=3`), `v3_gfni_crypto` backend, Linux 7.2.6, rustc 1.98.0,
  one complete `just bench kernels` run, median per-iteration throughput.
- Golden Cove pairing host: Intel Core i7-12700K, CPU 8 pinned and isolated
  (`FEC_GOLDEN_CORE=8`), `v3_gfni_crypto` backend, Linux CachyOS, rustc
  1.98.1, one complete `just bench kernels` run, median per-iteration
  throughput.

## Broadcast scalar add/sub (binary fields)

256 KiB panel buffers, ordinary `Vec<u8>` geometry. `add_assign` is the
unchanged control beside the broadcast forms. Throughput is GiB/s.

| Field | Operation | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | `add_assign` | 45.32 | 51.20 |
| `Gf8B` | `add_assign_scalar` | 63.07 | 67.46 |
| `Gf8B` | `sub_assign_scalar` | 63.14 | 67.54 |
| `Gf16` | `add_assign` | 44.85 | 51.16 |
| `Gf16` | `add_assign_scalar` | 62.98 | 67.46 |
| `Gf16` | `sub_assign_scalar` | 63.02 | 67.46 |

- Lunar Lake pairing host: Intel Core Ultra 7 258V, CPU 3 pinned
  (`FEC_GOLDEN_CORE=3`), `v3_gfni_crypto` backend, Linux 7.2.6, rustc 1.98.0,
  one complete `just bench kernels` run, median per-iteration throughput.
- Golden Cove pairing host: Intel Core i7-12700K, CPU 8 pinned and isolated
  (`FEC_GOLDEN_CORE=8`), `v3_gfni_crypto` backend, Linux CachyOS, rustc
  1.98.1, one complete `just bench kernels` run, median per-iteration
  throughput.

## Scatter, gather, and matrix

Each row is 64 KiB. Scatter writes eight rows from one source, gather combines
eight sources into one row, and matrix combines eight sources into eight rows.
Throughput is GiB/s over the harness's logical operation volume.

| Field | Operation | Geometry | Lunar Lake | Golden Cove |
| --- | --- | --- | ---: | ---: |
| `Gf8B` | `mul_add_scatter` | 1 source × 8 rows | 67.24 | 71.43 |
| `Gf16` | `mul_add_scatter` | 1 source × 8 rows | 56.50 | 52.66 |
| `Gf8B` | `mul_add_gather` | 8 sources × 1 row | 61.88 | 69.26 |
| `Gf16` | `mul_add_gather` | 8 sources × 1 row | 65.45 | 77.86 |
| `Gf8B` | `mul_add_matrix` | 8 sources × 8 rows | 119.14 | 114.29 |
| `Gf16` | `mul_add_matrix` | 8 sources × 8 rows | 63.44 | 55.60 |

Overwrite matrix results use 64 KiB rows and ten sources. Throughput counts
source bytes once per call.

| Field | Output rows | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | 2 | 66.74 | 62.24 |
| `Gf8B` | 4 | 35.82 | 34.75 |
| `Gf8B` | 6 | 22.75 | 22.36 |
| `Gf8D` | 2 | 68.16 | 65.16 |
| `Gf8D` | 4 | 37.82 | 38.26 |
| `Gf8D` | 6 | 24.24 | 24.20 |

## Wider binary fields

These are 256 KiB `mul_add` operations. Throughput is GiB/s.

| Field | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `Gf16` polynomial tower | 40.68 | 41.73 |
| `Gf32` polynomial tower | 30.25 | 31.39 |
| `Gf64` polynomial tower | 18.53 | 17.01 |
| `FanPaar16` | 18.40 | 17.22 |
| `FanPaar32` | 7.22 | 6.35 |
| `FanPaar64` | 2.46 | 1.91 |

## Bit-packed GF(2)

The input size is 256 KiB, representing 2,097,152 field elements. Throughput is
GiB/s over packed bytes.

| Operation | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `bits::xor_assign` | 39.81 | 49.42 |
| `bits::and_into` | 32.62 | 47.52 |
| `bits::weight` | 21.26 | 15.67 |
| `bits::dot_product` | 37.17 | 41.49 |
| `bits::xor_range` | 61.79 | 74.52 |
| `bits::xor_range_with` | 64.28 | 74.57 |

For 256 calls over 16-byte rows and the bit range `61..69`, the per-call cost
of the one-shot and prepared forms:

| Form | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `xor_range` one-shot | 7.15 ns | 5.36 ns |
| `XorRange` prepared | 2.09 ns | 2.07 ns |

## Self comparisons

`fgf` against `fgf`: prepared forms against one-shot, blocked multi-row kernels
against per-row loops, and the direct-kernel entries behind dispatch
thresholds. Cells are **Lunar Lake / Golden Cove**; a ratio is the first form's
throughput divided by the second, so above 1.00 the first form is faster.
Every figure comes from the `kernels` and `compare` panels of the field-family
runs.

### Preparation crossover (`mul_add` one-shot ÷ prepared)

| Row bytes | `Gf8B` | `Gf16` |
| ---: | ---: | ---: |
| 16 | 1.01/1.01 | 1.10/1.15 |
| 32 | 1.04/1.00 | 1.08/1.11 |
| 64 | 1.04/0.97 | 1.09/1.02 |
| 128 | 1.03/1.00 | 1.08/1.10 |
| 256 | 1.03/1.01 | 1.06/1.08 |
| 512 | 0.99/1.01 | 1.05/1.08 |
| 1024 | 1.01/1.01 | 1.02/1.04 |
| 2048 | 1.01/1.01 | 1.01/1.04 |
| 4096 | 1.00/1.00 | 1.01/1.00 |
| 16384 | 1.00/1.01 | 1.07/1.00 |
| 65536 | 1.00/1.00 | 1.00/1.00 |

### Small-row fusion (`mul_into` fused ÷ copy+scale)

| Row bytes | `Gf16` |
| ---: | ---: |
| 64 | 0.85/0.87 |
| 256 | 1.11/0.95 |
| 1024 | 1.47/1.57 |
| 16384 | 1.38/1.43 |

### Row-interleaved addition (`add_assign_rows` ÷ flat `add_assign`)

| Row bytes | Rows | `Gf8B` | `Gf16` | `Mersenne31` |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 1 | 1.09/0.98 | 0.96/0.96 | 0.75/1.00 |
| 64 | 2 | 1.12/0.95 | 0.96/0.99 | 1.14/1.00 |
| 64 | 4 | 0.88/0.96 | 0.91/0.87 | 0.80/1.00 |
| 64 | 8 | 0.80/0.94 | 0.77/0.91 | 0.93/1.00 |
| 64 | 16 | 0.85/0.95 | 0.81/0.93 | 1.00/1.00 |
| 64 | 32 | 1.02/1.01 | 1.04/1.03 | 0.98/1.00 |
| 1024 | 1 | 0.85/0.95 | 1.02/0.94 | 1.00/1.00 |
| 1024 | 2 | 1.03/1.01 | 1.04/1.03 | 0.98/1.02 |
| 1024 | 4 | 1.01/0.98 | 0.99/0.99 | 0.99/0.99 |
| 1024 | 8 | 0.96/1.00 | 0.94/0.98 | 1.02/1.01 |
| 1024 | 16 | 0.98/0.99 | 0.96/0.97 | 1.00/1.00 |
| 1024 | 32 | 1.03/1.01 | 1.05/1.02 | 1.00/0.96 |
| 65536 | 1 | 0.99/1.01 | 1.01/1.01 | 0.97/1.05 |
| 65536 | 2 | 1.03/1.00 | 1.01/1.00 | 0.99/0.99 |
| 65536 | 4 | 0.97/1.00 | 0.98/1.00 | 0.98/1.04 |
| 65536 | 8 | 0.98/1.00 | 0.99/1.00 | 0.99/1.01 |
| 65536 | 16 | 0.97/1.00 | 0.98/1.00 | 1.01/1.00 |
| 65536 | 32 | 1.00/1.00 | 0.98/1.00 | 1.00/1.00 |

### Blocked multi-row ÷ unblocked AXPY

| Rows | `Gf8B` scatter | `Gf8B` matrix | `Gf8B` gather | `Gf16` scatter | `Gf16` matrix | `Gf16` gather |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.78/1.44 | 2.43/2.13 | 0.99/1.19 | 1.25/1.36 | 1.27/1.11 | 1.12/1.27 |
| 4 | 1.57/1.49 | 2.81/2.37 | 1.13/1.32 | 1.39/1.59 | 1.55/1.29 | 1.26/1.48 |
| 8 | 1.49/1.49 | 2.79/2.41 | 1.14/1.13 | 1.32/1.40 | 1.53/1.33 | 1.26/1.48 |
| 16 | 1.48/1.21 | 2.74/3.29 | 1.28/1.36 | 1.37/1.12 | 1.53/1.74 | 1.50/1.74 |

### Blocked ÷ AXPY, GF(2^16) direct kernels

Direct kernel calls per backend tier with dispatch bypassed: the blocked
multi-row entry against repeated single-row `mul_add`. Matrix shapes exist for
the `avx2` tier only.

| Shape | `ssse3` | `avx2` | `gfni` |
| --- | ---: | ---: | ---: |
| gather 2 sources × 4 KiB | 0.85/0.84 | 0.85/0.79 | 0.99/1.07 |
| gather 4 sources × 4 KiB | 0.89/0.89 | 0.84/0.82 | 1.34/1.32 |
| gather 8 sources × 4 KiB | 0.81/0.92 | 0.89/0.84 | 1.47/1.38 |
| gather 16 sources × 4 KiB | 0.79/0.91 | 0.80/0.79 | 1.78/1.39 |
| gather 2 sources × 16 KiB | 0.74/0.83 | 0.81/0.78 | 1.03/1.08 |
| gather 4 sources × 16 KiB | 0.75/0.87 | 0.73/0.78 | 1.21/1.34 |
| gather 8 sources × 16 KiB | 0.75/0.89 | 0.72/0.78 | 1.17/1.32 |
| gather 16 sources × 16 KiB | 0.75/0.89 | 0.70/0.77 | 1.09/1.28 |
| gather 2 sources × 64 KiB | 0.75/0.83 | 0.74/0.77 | 1.10/1.20 |
| gather 4 sources × 64 KiB | 0.74/0.89 | 0.70/0.78 | 1.23/1.44 |
| gather 8 sources × 64 KiB | 0.75/0.87 | 0.70/0.76 | 1.25/1.43 |
| gather 16 sources × 64 KiB | 0.76/0.87 | 0.69/0.74 | 1.26/1.51 |
| matrix 2 sources × 4 rows × 4 KiB | - | 1.08/1.01 | - |
| matrix 4 sources × 4 rows × 4 KiB | - | 1.05/1.07 | - |
| matrix 8 sources × 4 rows × 4 KiB | - | 1.02/1.05 | - |
| matrix 16 sources × 4 rows × 4 KiB | - | 1.01/1.06 | - |
| matrix 2 sources × 4 rows × 16 KiB | - | 0.89/0.97 | - |
| matrix 4 sources × 4 rows × 16 KiB | - | 0.87/0.99 | - |
| matrix 8 sources × 4 rows × 16 KiB | - | 0.93/1.00 | - |
| matrix 16 sources × 4 rows × 16 KiB | - | 0.93/1.02 | - |
| matrix 2 sources × 4 rows × 64 KiB | - | 0.88/0.95 | - |
| matrix 4 sources × 4 rows × 64 KiB | - | 0.88/0.97 | - |
| matrix 8 sources × 4 rows × 64 KiB | - | 0.89/0.94 | - |
| matrix 16 sources × 4 rows × 64 KiB | - | 0.90/0.96 | - |

### Store policy over large destinations

`mul_into` over destinations that straddle the non-temporal store threshold,
with the read-back control and `mul_add` beside it. Throughput is GiB/s.

| Size | `Gf8B` `mul_into` | `Gf8B` `mul_into` + read | `Gf8B` `mul_add` | `Gf16` `mul_into` |
| --- | ---: | ---: | ---: | ---: |
| 1 MiB | 35.51/26.90 | 23.26/21.07 | 40.53/24.83 | 37.91/23.25 |
| 8 MiB | 35.06/41.76 | 17.32/16.59 | 17.58/23.85 | 34.05/41.69 |
| 32 MiB | 28.11/28.46 | 14.83/14.39 | 16.42/16.23 | 25.89/27.98 |

### Scatter destination skew (16-byte skew ÷ aligned)

| Row bytes | `Gf8B` | `Gf16` |
| ---: | ---: | ---: |
| 65536 | 0.99/1.00 | 1.06/1.05 |
| 262144 | 0.99/0.99 | 1.03/1.01 |

### Network-size payloads

Throughput is GiB/s at consumer payload lengths.

| Payload bytes | `xor` | `mul_add` | `mul_assign` | scatter 4 rows | scatter 16 rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 14.23/16.59 | 12.47/14.45 | 16.44/19.07 | 11.65/14.45 | 14.42/18.87 |
| 256 | 47.39/42.15 | 39.94/35.32 | 49.22/57.80 | 33.35/33.68 | 50.09/53.28 |
| 512 | 51.03/69.99 | 52.08/60.55 | 77.85/109.78 | 38.08/48.21 | 66.40/82.09 |
| 1152 | 94.84/83.53 | 80.97/73.20 | 109.69/137.88 | 91.98/96.64 | 105.74/104.65 |
| 1168 | 65.55/76.84 | 62.83/71.33 | 74.06/89.25 | 70.68/77.18 | 74.44/80.46 |
| 1184 | 93.10/84.82 | 78.41/74.29 | 113.46/138.38 | 93.23/95.69 | 103.53/103.48 |
| 1200 | 64.21/77.91 | 63.19/67.48 | 76.42/90.31 | 72.58/77.41 | 73.87/79.75 |
| 1216 | 96.38/85.07 | 80.89/73.96 | 113.25/138.32 | 93.64/94.68 | 103.03/104.12 |
| 1232 | 62.98/77.46 | 60.59/69.94 | 72.56/89.55 | 70.92/77.62 | 75.68/81.00 |
| 1248 | 95.37/86.10 | 80.68/74.84 | 113.39/138.78 | 97.49/93.51 | 107.85/104.31 |
| 1400 | 66.12/78.28 | 63.41/59.86 | 73.20/86.03 | 56.71/59.29 | 61.95/62.69 |

### Six-row shuffle table forms

The six-row overwrite kernel with raw coefficient terms against prepacked
scale tables, 64 KiB rows and ten sources. Throughput is GiB/s.

| Form | `Gf8B` | `Gf8D` |
| --- | ---: | ---: |
| raw terms | 8.94/9.30 | 8.97/9.38 |
| packed tables | 14.26/14.60 | 14.38/14.64 |

## Competitor comparison

The matrix compares `fgf` `Gf8D` against Intel ISA-L and
`klauspost/reedsolomon` over the harness shapes. Cells are **Lunar Lake /
Golden Cove**, median GiB/s over source bytes; each cell is a median per host
over five pinned runs of the harness in the setup tables below. The harnesses
interleave the two arms inside one process, so quotients across columns are
not paired benchmark ratios.

| Shape | `fgf` `Gf8D` | Intel ISA-L | `klauspost/reedsolomon` |
| --- | ---: | ---: | ---: |
| 64 KiB `dst = c * src` | 74.82/64.68 | 79.91/45.27 | 52.70/56.71 |
| 64 KiB `dst ^= c * src` | 67.43/65.00 | 67.70/65.27 | 49.05/49.63 |
| 4 KiB, 16 sources, one row | 87.23/82.42 | 97.66/98.61 | 44.82/50.60 |
| 16 KiB, 16 sources, one row | 79.15/85.50 | 74.19/90.09 | 52.11/57.01 |
| 4 KiB, 10 sources, 2 rows | 85.58/75.60 | 79.38/70.27 | 72.04/75.90 |
| 16 KiB, 10 sources, 2 rows | 75.36/72.39 | 69.46/65.67 | 73.72/85.19 |
| 64 KiB, 10 sources, 2 rows | 67.12/70.92 | 62.59/64.47 | 65.70/87.54 |
| 4 KiB, 10 sources, 4 rows | 42.43/40.65 | 37.80/35.27 | 39.57/40.60 |
| 16 KiB, 10 sources, 4 rows | 40.39/41.07 | 32.03/32.13 | 40.04/45.12 |
| 64 KiB, 10 sources, 4 rows | 38.82/39.97 | 27.93/29.90 | 43.38/45.88 |
| 4 KiB, 10 sources, 6 rows | 27.59/26.81 | 28.10/24.66 | 28.14/28.16 |
| 16 KiB, 10 sources, 6 rows | 24.65/26.25 | 25.93/24.02 | 29.90/29.90 |
| 64 KiB, 10 sources, 6 rows | 24.13/25.59 | 22.90/21.93 | 30.62/30.46 |

### Prime fields against Plonky3

The matrix compares `fgf` `Mersenne31`, `Goldilocks`, and `QuadMersenne31`
against Plonky3 — `p3-mersenne-31`, `p3-goldilocks`, and the packed
quadratic extension `Complex<Mersenne31>` (`X² + 1`) — over the harness
shapes. Cells are **Lunar Lake / Golden Cove**, median over five pinned
runs of the harness in the setup tables below. Region throughput is GiB/s
over one operand pair at 64 KiB regions; scalar rows are nanoseconds per
element multiply.

| Field | Shape | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | ---: | ---: |
| `Mersenne31` | `dst = c * src` | 68.20/58.45 | 80.91/72.89 |
| `Mersenne31` | `dst += c * src` | 47.40/42.45 | 63.84/57.05 |
| `Mersenne31` | `dst = a * b` | 52.74/47.19 | 75.28/69.39 |
| `Mersenne31` | `dst += v` | 114.66/107.83 | 219.71/152.83 |
| `Mersenne31` | `dst -= v` | 91.88/86.45 | 214.97/152.75 |
| `Goldilocks` | `dst = c * src` | 27.05/23.78 | 30.24/27.13 |
| `Goldilocks` | `dst += c * src` | 18.97/16.98 | 22.97/20.44 |
| `Goldilocks` | `dst = a * b` | 26.54/23.22 | 30.56/26.41 |
| `Goldilocks` | `dst += v` | 54.24/49.53 | 172.70/147.61 |
| `Goldilocks` | `dst -= v` | 56.30/52.71 | 167.99/147.81 |
| `QuadMersenne31` | `dst = c * src` | 29.80/25.88 | 41.14/36.12 |
| `QuadMersenne31` | `dst += c * src` | 24.07/20.81 | 36.08/31.73 |
| `QuadMersenne31` | `dst = a * b` | 14.40/12.26 | 29.62/26.21 |
| `QuadMersenne31` | `dst += v` | 115.09/107.84 | 216.21/152.75 |
| `QuadMersenne31` | `dst -= v` | 94.17/86.43 | 215.41/152.83 |

Scalar multiplication over the same fixture stream, nanoseconds per
operation.

| Field | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | ---: | ---: |
| `Mersenne31` | 1.50/1.55 | 0.85/0.80 |
| `Goldilocks` | 1.63/1.64 | 1.14/1.11 |
| `QuadMersenne31` | 3.31/4.37 | 2.87/2.91 |

Across both hosts the ordering is identical: Plonky3 leads every recorded
shape. The lead is widest on the broadcast add/sub and elementwise product
shapes and narrowest on the Goldilocks scale shapes and the quadratic
scalar multiply. The fgf-to-Plonky3 ratios agree closely between the two
hosts, so the comparison's conclusions do not depend on which host runs
them. Golden Cove's isolated core yields tighter paired bands and smaller
run-to-run spread than Lunar Lake.

- Each library runs on its own native layout and layout conversion is
  excluded from every timed region, so the cells compare kernels rather
  than encodings.
- Only the AVX2 tier is measured: Plonky3 selects its packed kernels with
  `cfg(target_feature)`, and the target hosts carry no AVX-512.
- The harnesses interleave the two arms inside one process, so quotients
  across columns are not paired benchmark ratios.

### Competitor setup

Commands, aggregation, and library-specific setup for the competitor
matrices.

**Lunar Lake setup:** CPU 3, `v3_gfni_crypto`, rustc 1.98.0.
Run commands below from the crate root with `FEC_GOLDEN_CORE=3`.

| Library | Command | Aggregation | Library-specific setup |
| --- | --- | --- | --- |
| `fgf` `Gf8D` | `just bench-gf-comp` | Median of five per-run medians | Competitor-harness fixtures, distinct from the fgf-only tables above. |
| Intel ISA-L 2.32.0 | `just bench-gf-comp` | Median of five per-run medians | System library, runtime dispatch. |
| `klauspost/reedsolomon` v1.14.2 | `just bench-gf-comp` | Median of five per-run medians | Go 1.27.1 C archive, `GOMAXPROCS=1`; `WithCustomMatrix`, `Encode` for overwrite and `EncodeIdx` for single-source accumulation. |
| `fgf` prime fields | `just bench-gdl-comp`, `just bench-m31-comp` | Median of five per-run medians | Runtime dispatch; competitor-harness fixtures, distinct from the fgf-only tables above. |
| Plonky3 `p3-mersenne-31`, `p3-goldilocks` 0.7.0 | `just bench-gdl-comp`, `just bench-m31-comp` | Median of five per-run medians | Pinned `=0.7.0`; compile-time AVX2 kernels, so the harness package builds with `-C target-cpu=native` and the numbers are host-build-specific. |

**Golden Cove setup:** CPU 8, isolated, `v3_gfni_crypto`, rustc 1.98.1.
Run commands below from the crate root with `FEC_GOLDEN_CORE=8`.

| Library | Command | Aggregation | Library-specific setup |
| --- | --- | --- | --- |
| `fgf` `Gf8D` | `just bench-gf-comp` | Median of five per-run medians | Competitor-harness fixtures, distinct from the fgf-only tables above. |
| Intel ISA-L 2.32.0 | `just bench-gf-comp` | Median of five per-run medians | System library, runtime dispatch. |
| `klauspost/reedsolomon` v1.14.2 | `just bench-gf-comp` | Median of five per-run medians | Go 1.27.1 C archive, `GOMAXPROCS=1`; `WithCustomMatrix`, `Encode` for overwrite and `EncodeIdx` for single-source accumulation. |
| `fgf` prime fields | `just bench-gdl-comp`, `just bench-m31-comp` | Median of five per-run medians | Runtime dispatch; competitor-harness fixtures, distinct from the fgf-only tables above. |
| Plonky3 `p3-mersenne-31`, `p3-goldilocks` 0.7.0 | `just bench-gdl-comp`, `just bench-m31-comp` | Median of five per-run medians | Pinned `=0.7.0`; compile-time AVX2 kernels, so the harness package builds with `-C target-cpu=native` and the numbers are host-build-specific. |
