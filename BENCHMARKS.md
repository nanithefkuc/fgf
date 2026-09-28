# Benchmarks

Public API timings. Every cell is **Tiger Lake / Golden Cove**, each the median of five per-run medians; paired ratio cells take the median of per-round ratios. Measurements that are unavailable are denoted with `-`.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Tiger Lake | Intel Core i5-1135G7 | Arch Linux, Linux 7.2.7 | 1.98.1 | 3, isolated |
| Golden Cove | Intel Core i7-12700K | CachyOS, Linux 7.2.6 | 1.98.1 | 8, isolated |

| Setting | Value |
| --- | --- |
| Crate | `fgf` 1.2.1 working tree, source fingerprint `72b26d7a4d26401ae4e162cd137bceddf3ee2d77bc3ca5e4d640bb95339e8b36` |
| Dependencies | `simdispatch` 0.2.0, `archmage` 0.9.29, Criterion 0.8.2; ISA-L 2.32.0, klauspost/reedsolomon v1.14.2 (Go 1.27.1), Plonky3 0.7.0 |
| Build | `--all-features`; the competitor harness packages build `fgf` with `simd512` added, so both hosts measure their requested tier |
| Threads | `RAYON_NUM_THREADS=1`, `GOMAXPROCS=1`; affinity verified per process with `taskset -pc` |
| Resolved backend | `V4x` on Tiger Lake, `v3_gfni_crypto` on Golden Cove, reported by every harness banner |
| Aggregation | Median of five per-run medians, three significant figures; one shuffled `-comp` campaign per field family per host |

The measured tree is a working snapshot rather than a release commit, and `just test` passed on both hosts before timing. Tiger Lake runs under `bench-run` (real-time priority, frequency-pinned 3.0 GHz core, turbostat logging); Golden Cove runs pinned to the isolated core 8 under the session's `SCHED_IDLE` policy.

## Scalar multiplication

The scalar harness executes a fixed batch of element multiplications.

| Operation | Tiger Lake (ns/op) | Golden Cove (ns/op) |
| --- | ---: | ---: |
| `Gf8B` | 0.70 / 0.34 |
| `Gf8D` | 0.70 / 0.34 |
| `Gf16` | 2.43 / 1.34 |

## Single-row packed operations

64 KiB, 64-byte-aligned buffers.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `Gf8B` | `mul_add` | 40.6 / 64.7 |
| `Gf8B` | `mul_into` | 39.6 / 64.7 |
| `Gf8D` | `mul_add` | 40.6 / 64.8 |
| `Gf8D` | `mul_into` | 39.7 / 64.7 |
| `Gf16` | `mul_add` | 28 / 55.6 |
| `Gf16` | `mul_into` | 37.2 / 63.7 |

### One-shot versus prepared `mul_add` at 64 KiB

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `Gf8B` | one-shot | 37.4 / 48.9 |
| `Gf8B` | prepared | 37.4 / 48.9 |
| `Gf16` | one-shot | 24.1 / 41.5 |
| `Gf16` | prepared | 24.1 / 41.5 |

## In-place elementwise multiply (binary fields, 256 KiB)

Ordinary `Vec<u8>` buffers.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `Gf8B` | `add_assign` / XOR | 25 / 51.1 |
| `Gf8B` | `mul_add` | 37.9 / 48.9 |
| `Gf8B` | `mul_assign` | 51.5 / 67.7 |
| `Gf8B` | `mul_elementwise` | 25.2 / 42.3 |
| `Gf16` | `mul_add` | 23.9 / 41.7 |
| `Gf16` | `mul_assign` | 41 / 54.6 |
| `Gf16` | `mul_elementwise` | 15 / 31 |

## Broadcast scalar add/sub and in-place elementwise multiply (prime fields, 256 KiB)

`add_assign` and `mul_elementwise` are the unchanged controls beside the broadcast and in-place forms.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `Mersenne31` | `add_assign` | 20 / 35.8 |
| `Mersenne31` | `add_assign_scalar` | 29.5 / 53 |
| `Mersenne31` | `sub_assign_scalar` | 24.5 / 43.5 |
| `Mersenne31` | `mul_elementwise` | 10.6 / 22.6 |
| `Mersenne31` | `mul_elementwise_assign` | 11.2 / 23.6 |
| `Goldilocks` | `add_assign` | 10.3 / 18.6 |
| `Goldilocks` | `add_assign_scalar` | 13 / 23.9 |
| `Goldilocks` | `sub_assign_scalar` | 13.7 / 25.4 |
| `Goldilocks` | `mul_elementwise` | 6.3 / 11 |
| `Goldilocks` | `mul_elementwise_assign` | 6.31 / 11.3 |
| `QuadMersenne31` | `add_assign` | 18.5 / 32.6 |
| `QuadMersenne31` | `add_assign_scalar` | 27.3 / 53.5 |
| `QuadMersenne31` | `sub_assign_scalar` | 22.6 / 43.3 |
| `QuadMersenne31` | `mul_elementwise` | 3.14 / 5.8 |
| `QuadMersenne31` | `mul_elementwise_assign` | 3.29 / 5.9 |

## Broadcast scalar add/sub (binary fields, 256 KiB)

`add_assign` is the unchanged control beside the broadcast forms.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `Gf8B` | `add_assign` | 25 / 51.1 |
| `Gf8B` | `add_assign_scalar` | 41.5 / 67.6 |
| `Gf8B` | `sub_assign_scalar` | 41.5 / 67.5 |
| `Gf16` | `add_assign` | 25 / 51.1 |
| `Gf16` | `add_assign_scalar` | 41.5 / 67.5 |
| `Gf16` | `sub_assign_scalar` | 41.5 / 67.5 |

## Scatter, gather, and matrix

Each row is 64 KiB. Scatter writes n rows from one source, gather combines n sources into one row, and matrix combines n sources into n rows. Throughput counts the harness's logical operation volume.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `Gf8B` | `mul_add_scatter` | 1 × 2 rows | 45.5 / 70 |
| `Gf16` | `mul_add_scatter` | 1 × 2 rows | 30.6 / 56.2 |
| `Gf8B` | `mul_add_gather` | 2 × 1 row | 44.5 / 72.4 |
| `Gf16` | `mul_add_gather` | 2 × 1 row | 33.8 / 67.1 |
| `Gf8B` | `mul_add_matrix` | 2 × 2 rows | 78.5 / 103 |
| `Gf16` | `mul_add_matrix` | 2 × 2 rows | 23.7 / 47.8 |
| `Gf8B` | `mul_add_scatter` | 1 × 4 rows | 48.3 / 72.3 |
| `Gf16` | `mul_add_scatter` | 1 × 4 rows | 26.1 / 65.5 |
| `Gf8B` | `mul_add_gather` | 4 × 1 row | 43 / 79.7 |
| `Gf16` | `mul_add_gather` | 4 × 1 row | 39.6 / 77.7 |
| `Gf8B` | `mul_add_matrix` | 4 × 4 rows | 110 / 115 |
| `Gf16` | `mul_add_matrix` | 4 × 4 rows | 25.3 / 55.3 |
| `Gf8B` | `mul_add_scatter` | 1 × 8 rows | 48.3 / 72.3 |
| `Gf16` | `mul_add_scatter` | 1 × 8 rows | 27.1 / 65.7 |
| `Gf8B` | `mul_add_gather` | 8 × 1 row | 43.6 / 69.3 |
| `Gf16` | `mul_add_gather` | 8 × 1 row | 38.6 / 77.8 |
| `Gf8B` | `mul_add_matrix` | 8 × 8 rows | 109 / 115 |
| `Gf16` | `mul_add_matrix` | 8 × 8 rows | 25.3 / 54.9 |
| `Gf8B` | `mul_add_scatter` | 1 × 16 rows | 45.1 / 46.5 |
| `Gf16` | `mul_add_scatter` | 1 × 16 rows | 27 / 36.9 |
| `Gf8B` | `mul_add_gather` | 16 × 1 row | 43.6 / 65 |
| `Gf16` | `mul_add_gather` | 16 × 1 row | 40 / 73.3 |
| `Gf8B` | `mul_add_matrix` | 16 × 16 rows | 105 / 114 |
| `Gf16` | `mul_add_matrix` | 16 × 16 rows | 25.2 / 55.5 |

## Overwrite matrix

64 KiB rows and ten sources; `_with` and `_at` run `Gf8B` prepared-coefficient and scattered-row forms. Throughput counts source bytes once per call.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `Gf8B` | 2 rows | `mul_add_matrix` | 47.3 / 50.3 |
| `Gf8B` | 2 rows | `mul_into_matrix` | 57.2 / 59.7 |
| `Gf8D` | 2 rows | `mul_add_matrix` | 47 / 51.8 |
| `Gf8D` | 2 rows | `mul_into_matrix` | 56.4 / 63.7 |
| `Gf8B` | 2 rows | `mul_add_matrix_with` | 53.7 / 56.2 |
| `Gf8B` | 2 rows | `mul_into_matrix_with` | 56.7 / 60.1 |
| `Gf8B` | 2 rows | `mul_add_matrix_at` | 57.4 / 56.9 |
| `Gf8B` | 4 rows | `mul_add_matrix` | 26.2 / 27.2 |
| `Gf8B` | 4 rows | `mul_into_matrix` | 32.9 / 33.9 |
| `Gf8D` | 4 rows | `mul_add_matrix` | 26.6 / 29.8 |
| `Gf8D` | 4 rows | `mul_into_matrix` | 32.4 / 37.5 |
| `Gf8B` | 4 rows | `mul_add_matrix_with` | 30.4 / 32 |
| `Gf8B` | 4 rows | `mul_into_matrix_with` | 32.1 / 34.4 |
| `Gf8B` | 4 rows | `mul_add_matrix_at` | 32.2 / 32.1 |
| `Gf8B` | 6 rows | `mul_add_matrix` | 16.8 / 17.5 |
| `Gf8B` | 6 rows | `mul_into_matrix` | 20.7 / 22 |
| `Gf8D` | 6 rows | `mul_add_matrix` | 16.6 / 18.7 |
| `Gf8D` | 6 rows | `mul_into_matrix` | 20.6 / 24.2 |
| `Gf8B` | 6 rows | `mul_add_matrix_with` | 19.3 / 20.5 |
| `Gf8B` | 6 rows | `mul_into_matrix_with` | 20.2 / 22.2 |
| `Gf8B` | 6 rows | `mul_add_matrix_at` | 20.7 / 20.4 |

### Scattered-row fallback through dispatched `mul_add`, 1.2 line

Baseline: `mul_add_matrix_at` falling back to the element-wise scalar loop on every field and tier without a blocked scattered kernel. Candidate: the same fallback as one dispatched `mul_add` per row and term. Ten canonical sources rebuild the erased slots 1, 4, 7, 9, 11, 13 in order inside a codeword buffer. Cells are baseline time ÷ candidate time, medians of five paired rounds, and span the 2-, 4-, and 6-row shapes; above 1.00 favours the candidate. `Gf8B` and `Gf8D` keep their blocked GFNI kernel in both builds and are the unchanged-path control.

| Field | Row bytes | Tiger Lake |
| --- | ---: | ---: |
| `Gf8B` | 1024 | 0.991–1.01 |
| `Gf8B` | 16384 | 0.978–0.996 |
| `Gf8B` | 65536 | 1.00 |
| `Gf8D` | 1024 | 1.00–1.01 |
| `Gf8D` | 16384 | 0.998–1.01 |
| `Gf8D` | 65536 | 0.997–1.00 |
| `Gf16` | 1024 | - |
| `Gf16` | 16384 | 77–80 |
| `Gf16` | 65536 | 72 |
| `Gf32` | 1024 | 83–90 |
| `Gf32` | 16384 | - |
| `Gf32` | 65536 | 108–109 |
| `Gf64` | 1024 | 67 |
| `Gf64` | 16384 | 106 |
| `Gf64` | 65536 | 106 |
| `FanPaar8` | 1024 | 2.76 |
| `FanPaar8` | 16384 | 2.76–2.77 |
| `FanPaar8` | 65536 | 2.75 |
| `FanPaar16` | 1024 | - |
| `FanPaar16` | 16384 | 116–117 |
| `FanPaar16` | 65536 | 113–114 |
| `FanPaar32` | 1024 | - |
| `FanPaar32` | 16384 | 83 |
| `FanPaar32` | 65536 | 80 |
| `FanPaar64` | 1024 | 40–41 |
| `FanPaar64` | 16384 | 48–49 |
| `FanPaar64` | 65536 | 48 |
| `Mersenne31` | 1024 | 4.86–4.93 |
| `Mersenne31` | 16384 | 4.74–4.75 |
| `Mersenne31` | 65536 | 4.69–4.71 |
| `Goldilocks` | 1024 | - |
| `Goldilocks` | 16384 | 4.52–4.61 |
| `Goldilocks` | 65536 | 4.53–4.56 |
| `QuadMersenne31` | 1024 | 6.10–6.14 |
| `QuadMersenne31` | 16384 | 6.21–6.23 |
| `QuadMersenne31` | 65536 | 6.21–6.22 |

- One pinned `bench-run` session on CPU 3 of the Tiger Lake host at a validated 3.0 GHz pin: every loaded `Bzy_MHz` sample 3000, `CoreThr` 0, live thermal bit 0. Both binaries reported `v3_gfni_crypto` in every run, the highest tier the 1.2 line dispatches: it has no `simd512` feature. Rust 1.98.1, `fgf` 1.2 default features, `codegen-units = 1`.
- Five rounds alternate baseline–candidate–baseline and candidate–baseline–candidate; each round's ratio divides the mean of its baseline arms by the mean of its candidate arms. Each arm is the median of 31 batches of 8 calls after warm-up.
- The control is the contiguous `mul_add_matrix` of the same terms, unchanged between the builds, timed beside every cell. Per-round control ratios stayed within 0.975–1.05, except `Gf16`, `FanPaar16`, `FanPaar32`, and `Goldilocks` at 1024 bytes and `Gf32` at 16384 bytes, whose rounds reached 0.993–1.344; those rows are `-`.
- The baseline returns wrong prime-field results for unit coefficients; the fixture's coefficients are pseudorandom and include no unit coefficient, so both builds perform the same arithmetic volume.
- Baseline source fingerprint `12f7148f2b4cf691853da5172fd8554395d89195c23e5ee938d2484ea22504a2`, candidate `b810adc3f0637569617d9412b5f419ba759d497d8feafd07dcabd98c64f022fb`: sha256 over the ordered `sha256sum` output of `src/` and `benches/`. The two differ only in the fallback body.

## Gather dot products

Sixteen sources into one destination row; the raw row uses raw coefficients, the prepared rows the `CoeffVec` `_with` form.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `Gf8B` | 4 KiB × 16 sources | overwrite prepared | 71.7 / 83.8 |
| `Gf8D` | 4 KiB × 16 sources | overwrite prepared | 83.6 / 82 |
| `Gf8D` | 4 KiB × 16 sources | overwrite raw terms | 80.5 / 105 |
| `Gf8B` | 16 KiB × 16 sources | overwrite prepared | 81.4 / 88.1 |
| `Gf8D` | 16 KiB × 16 sources | overwrite prepared | 96.1 / 85.4 |
| `Gf8D` | 16 KiB × 16 sources | overwrite raw terms | 92 / 112 |

## Wider binary fields

These are 256 KiB `mul_add` operations.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `Gf16` polynomial tower | 23.9 / 41.7 |
| `Gf32` polynomial tower | 17.1 / 30.6 |
| `Gf64` polynomial tower | 9.07 / 17 |
| `FanPaar16` | 9.97 / 17.1 |
| `FanPaar32` | 3.71 / 6.33 |
| `FanPaar64` | 1.36 / 2.2 |

## Bit-packed GF(2)

The input size is 256 KiB, representing 2,097,152 field elements. Throughput counts packed bytes.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| `bits::xor_assign` | 23.6 / 51.7 |
| `bits::and_into` | 19.3 / 46.7 |
| `bits::weight` | 7.28 / 15.7 |
| `bits::dot_product` | 18.3 / 43.5 |
| `bits::xor_range` | 37.6 / 74.3 |
| `bits::xor_range_with` | 37.6 / 75 |

## Store policy over large destinations

`mul_into` over destinations that straddle the non-temporal store threshold, with the read-back control and `mul_add` beside it.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| 1 MiB | `Gf8B` `mul_into` | 18.9 / 26.8 |
| 1 MiB | `Gf8B` `mul_into` + read | 13.8 / 20.8 |
| 1 MiB | `Gf8B` `mul_add` | 18.9 / 24.8 |
| 1 MiB | `Gf16` `mul_into` | 18.4 / 23.1 |
| 8 MiB | `Gf8B` `mul_into` | 18 / 41.8 |
| 8 MiB | `Gf8B` `mul_into` + read | 7.63 / 16.5 |
| 8 MiB | `Gf8B` `mul_add` | 8.36 / 23.9 |
| 8 MiB | `Gf16` `mul_into` | 18.1 / 41.8 |
| 32 MiB | `Gf8B` `mul_into` | 9.37 / 27.5 |
| 32 MiB | `Gf8B` `mul_into` + read | 5.91 / 14.2 |
| 32 MiB | `Gf8B` `mul_add` | 6.27 / 16.4 |
| 32 MiB | `Gf16` `mul_into` | 9.29 / 27.9 |

## In-place scale over increasing destinations

Ordinary 64-byte-aligned buffers from 4 KiB to 4 MiB.

| Operation | Tiger Lake (GiB/s) | Golden Cove (GiB/s) |
| --- | ---: | ---: |
| 4 KiB | `Gf8B` `mul_assign` | 129 / 175 |
| 4 KiB | `Gf8D` `mul_assign` | 128 / 172 |
| 16 KiB | `Gf8B` `mul_assign` | 137 / 187 |
| 16 KiB | `Gf8D` `mul_assign` | 138 / 186 |
| 64 KiB | `Gf8B` `mul_assign` | 51.3 / 76.3 |
| 64 KiB | `Gf8D` `mul_assign` | 51.3 / 68.9 |
| 256 KiB | `Gf8B` `mul_assign` | 51.5 / 76.6 |
| 256 KiB | `Gf8D` `mul_assign` | 51.5 / 69.1 |
| 1 MiB | `Gf8B` `mul_assign` | 48.1 / 52.8 |
| 1 MiB | `Gf8D` `mul_assign` | 49.1 / 50.6 |
| 4 MiB | `Gf8D` `mul_into` | 18.1 / 41.7 |

## Scatter destination skew (16-byte skew ÷ aligned)

| Operation | Tiger Lake | Golden Cove |
| --- | ---: | ---: |
| 64 KiB rows | `Gf8B` | 1 / 1 |
| 256 KiB rows | `Gf8B` | 1 / 0.99 |
| 64 KiB rows | `Gf16` | 1.09 / 1.05 |
| 256 KiB rows | `Gf16` | 1.05 / 1.01 |

### Preparation crossover (`mul_add` one-shot ÷ prepared)

| Row bytes | `Gf8B` | `Gf16` |
| --- | ---: | ---: |
| `16` | 0.981 / 0.957 | 0.903 / 0.869 |
| `32` | 0.979 / 0.963 | 0.913 / 0.89 |
| `64` | 0.977 / 1.02 | 0.916 / 0.933 |
| `128` | 0.991 / 1 | 0.915 / 0.904 |
| `256` | 0.987 / 1.01 | 0.924 / 0.925 |
| `512` | 0.989 / 1 | 0.954 / 0.954 |
| `1024` | 0.989 / 0.989 | 0.968 / 0.969 |
| `2048` | 1.01 / 0.99 | 0.981 / 0.954 |
| `4096` | 1 / 1 | 0.992 / 0.994 |
| `16384` | 0.999 / 0.986 | 1 / 0.999 |
| `65536` | 0.999 / 0.998 | 0.999 / 1 |

### Small-row fusion (`mul_into` fused ÷ copy+scale)

| Operation | Tiger Lake | Golden Cove |
| --- | ---: | ---: |
| 64 B × 16 rows × 8 | 0.794 / 0.851 |
| 256 B × 16 rows × 8 | 0.872 / 0.963 |
| 1024 B × 16 rows × 8 | 1.25 / 1.57 |
| 16384 B × 16 rows × 8 | 1.45 / 1.41 |

### Row-interleaved addition (`add_assign_rows` ÷ flat `add_assign`)

| Row bytes | Rows | `Gf8B` | `Gf16` | `Mersenne31` |
| --- | ---: | ---: | ---: | ---: |
| 64 | 1 | 0.85 / 0.985 | 0.64 / 0.864 | 0.75 / 0.926 |
| 64 | 2 | 0.896 / 0.95 | 0.705 / 0.941 | 0.816 / 0.908 |
| 64 | 4 | 0.874 / 0.952 | 0.705 / 0.862 | 0.844 / 0.934 |
| 64 | 8 | 0.864 / 0.959 | 0.793 / 0.871 | 0.894 / 0.972 |
| 64 | 16 | 0.829 / 0.957 | 0.824 / 0.931 | 0.966 / 0.986 |
| 64 | 32 | 0.992 / 1.02 | 0.963 / 1.02 | 1.03 / 0.99 |
| 1024 | 1 | 0.831 / 0.955 | 0.824 / 0.939 | 0.969 / 0.981 |
| 1024 | 2 | 0.992 / 1.02 | 0.963 / 1.02 | 0.973 / 0.991 |
| 1024 | 4 | 0.969 / 0.978 | 0.962 / 0.983 | 0.99 / 0.993 |
| 1024 | 8 | 0.974 / 0.968 | 0.959 / 1.03 | 0.995 / 0.993 |
| 1024 | 16 | 0.986 / 0.98 | 0.977 / 0.966 | 0.998 / 0.999 |
| 1024 | 32 | 0.991 / 1 | 0.994 / 1 | 0.995 / 0.998 |
| 65536 | 1 | 0.999 / 1 | 0.997 / 1 | 0.998 / 1 |
| 65536 | 2 | 1 / 1 | 0.999 / 1 | 1 / 1 |
| 65536 | 4 | 1 / 1 | 0.999 / 1 | 0.998 / 1 |
| 65536 | 8 | 1 / 1 | 1 / 1 | 1 / 1.02 |
| 65536 | 16 | 1 / 1 | 1 / 1 | 1 / 1 |
| 65536 | 32 | 1 / 1 | 1 / 1 | 1 / 0.998 |

### Network-size payloads

| Payload bytes | `xor` (GiB/s) | `mul_add` (GiB/s) | `mul_assign` (GiB/s) | scatter 4 rows (GiB/s) | scatter 16 rows (GiB/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 B | 7.95 / 14.2 | 3.66 / 13.5 | 6.51 / 16.3 | 7.34 / 14.7 | 9.37 / 19.2 |
| 256 B | 23.7 / 47.7 | 14 / 34.7 | 21.9 / 57.4 | 23.8 / 37.2 | 26.6 / 60.9 |
| 512 B | 36.8 / 71.3 | 27.1 / 59.4 | 52.4 / 105 | 35.4 / 48.2 | 43.5 / 82.3 |
| 1152 B | 48.5 / 84.2 | 46.6 / 73 | 81.4 / 137 | 57 / 97.7 | 85.5 / 105 |
| 1168 B | 46.2 / 77 | 46.7 / 70.6 | 80.8 / 90.7 | 52.6 / 77.5 | 56.5 / 80.6 |
| 1184 B | 47.8 / 85 | 49.1 / 74 | 86.1 / 137 | 61.4 / 95.7 | 64 / 104 |
| 1200 B | 46 / 78.1 | 48.2 / 67.1 | 79 / 91.5 | 54.6 / 77.5 | 55.1 / 79.5 |
| 1216 B | 47.8 / 85.1 | 48.8 / 73.8 | 81.8 / 137 | 63.9 / 94.8 | 84.2 / 104 |
| 1232 B | 45.9 / 77.8 | 48.4 / 69.7 | 82.7 / 90.7 | 54.6 / 77.7 | 56.8 / 81.3 |
| 1248 B | 48.3 / 85.7 | 50.8 / 74.4 | 84.2 / 138 | 61.3 / 93.9 | 63.3 / 104 |
| 1400 B | 47.1 / 77.7 | 40.2 / 59.5 | 68.7 / 86.6 | 41 / 60.1 | 44 / 63 |

### Blocked multi-row ÷ unblocked AXPY (dispatched forms)

A ratio above 1.00 means the blocked form is faster.

| Operation | Tiger Lake | Golden Cove |
| --- | ---: | ---: |
| 2 rows | `Gf8B` scatter | 1.22 / 1.44 |
| 2 rows | `Gf8B` matrix | 2.11 / 2.12 |
| 2 rows | `Gf8B` gather | 1.24 / 1.19 |
| 2 rows | `Gf16` scatter | 1.3 / 1.37 |
| 2 rows | `Gf16` matrix | 0.981 / 1.11 |
| 2 rows | `Gf16` gather | 1.4 / 1.27 |
| 4 rows | `Gf8B` scatter | 1.3 / 1.49 |
| 4 rows | `Gf8B` matrix | 2.89 / 2.4 |
| 4 rows | `Gf8B` gather | 1.26 / 1.34 |
| 4 rows | `Gf16` scatter | 1.21 / 1.59 |
| 4 rows | `Gf16` matrix | 1.05 / 1.3 |
| 4 rows | `Gf16` gather | 1.67 / 1.48 |
| 8 rows | `Gf8B` scatter | 1.3 / 1.49 |
| 8 rows | `Gf8B` matrix | 2.86 / 2.4 |
| 8 rows | `Gf8B` gather | 1.25 / 1.14 |
| 8 rows | `Gf16` scatter | 1.21 / 1.59 |
| 8 rows | `Gf16` matrix | 1.05 / 1.3 |
| 8 rows | `Gf16` gather | 1.68 / 1.48 |
| 16 rows | `Gf8B` scatter | 1.28 / 1.26 |
| 16 rows | `Gf8B` matrix | 2.98 / 3.28 |
| 16 rows | `Gf8B` gather | 1.23 / 1.36 |
| 16 rows | `Gf16` scatter | 1.21 / 1.13 |
| 16 rows | `Gf16` matrix | 1.06 / 1.76 |
| 16 rows | `Gf16` gather | 1.67 / 1.75 |

## AVX-512 alignment peel floors (peel ÷ no peel, rows at 16 mod 64)

Scatter is four rows, gather sixteen sources, matrix ten sources into four rows. Below 1.00 the peel wins. The adopted floors are `SCATTER_PEEL_MIN` 512, `GATHER_PEEL_MIN` 3072, and `MATRIX_PEEL_MIN` 8192 in `src/kernel/x86/gf8/`.

| Operation | Tiger Lake | Golden Cove |
| --- | ---: | ---: |
| 256 | 0.86 / 0.81 | 0.94 / 0.92 | 0.91 / 0.91 |
| 512 | 0.74 / 0.61 | 0.90 / 0.89 | 0.92 / 0.90 |
| 1024 | 0.78 / 0.59 | 0.88 / 0.86 | 0.88 / 0.88 |
| 2048 | 0.89 / 0.96 | 0.86 / 0.86 | 0.90 / 0.88 |
| 3072 | 0.90 / 0.99 | 0.76 / 0.82 | 0.87 / 0.93 |
| 4096 | 0.91 / 0.97 | 0.83 / 0.82 | 0.86 / 0.93 |
| 8192 | 0.97 / 0.97 | 0.91 / 0.82 | 0.88 / 0.94 |
| 16384 | 1.00 / 1.00 | 0.95 / 0.82 | 0.94 / 0.94 |

## Competitor comparison

The matrix compares `fgf` `Gf8D` against Intel ISA-L and klauspost/reedsolomon over the harness shapes. Throughput counts source bytes. The trio harness interleaves all three arms inside one process, rotating through all six arm orders, so quotients across columns share the same timed session.

| Shape | `fgf` (GiB/s) | Intel ISA-L (GiB/s) | `klauspost/reedsolomon` (GiB/s) |
| --- | ---: | ---: | ---: |
| 64 KiB `dst = c * src` | 40.1 / 63.7 | 18.1 / 42.1 | 33.7 / 56.2 |
| 64 KiB `dst ^= c * src` | 40.9 / 63.5 | 41.2 / 63.8 | 29 / 48.2 |
| 4 KiB, 16 sources, one row | 82.8 / 102 | 83 / 95.8 | 30.7 / 49.9 |
| 16 KiB, 16 sources, one row | 89.8 / 105 | 74.9 / 89.9 | 35.7 / 57.2 |
| 4 KiB, 10 sources, 2 rows | 65.1 / 73.8 | 64.9 / 69.7 | 53.2 / 74.9 |
| 16 KiB, 10 sources, 2 rows | 63.2 / 71.6 | 52.8 / 65.2 | 67.1 / 85.1 |
| 64 KiB, 10 sources, 2 rows | 58.4 / 72.1 | 54 / 63.8 | 73.4 / 86.6 |
| 4 KiB, 10 sources, 4 rows | 34.5 / 40.4 | 35.6 / 35 | 31.7 / 40.6 |
| 16 KiB, 10 sources, 4 rows | 35.5 / 40.9 | 36.6 / 29.8 | 38.4 / 44.8 |
| 64 KiB, 10 sources, 4 rows | 33.2 / 39.9 | 35.5 / 31.7 | 40.8 / 45.4 |
| 4 KiB, 10 sources, 6 rows | 22.3 / 26.6 | 24.2 / 24.5 | 22.3 / 28 |
| 16 KiB, 10 sources, 6 rows | 22.6 / 26.1 | 25 / 22 | 25.8 / 30.3 |
| 64 KiB, 10 sources, 6 rows | 20.8 / 25.7 | 23.6 / 23.6 | 26.7 / 30.5 |

### Mersenne31 (scalar multiply)

| Shape | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | ---: | ---: |
| `dst = a * b` | 4.20 / 1.55 | 2.09 / 0.80 |

### Mersenne31 (region operations)

| Shape | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | ---: | ---: |
| `dst = c * src` | 33.3 / 58.5 | 65.4 / 72.7 |
| `dst += c * src` | 24.7 / 42.6 | 49.8 / 57 |
| `dst = a * b` | 27.4 / 47.2 | 59.8 / 69.4 |
| `dst += v` | 62.5 / 108 | 103 / 153 |
| `dst -= v` | 49.9 / 86.3 | 103 / 153 |

### Goldilocks (scalar multiply)

| Shape | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | ---: | ---: |
| `dst = a * b` | 3.75 / 1.64 | 2.60 / 1.10 |

### Goldilocks (region operations)

| Shape | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | ---: | ---: |
| `dst = c * src` | 14.4 / 23.8 | 27.4 / 27.1 |
| `dst += c * src` | 10.8 / 17 | 22.3 / 20.6 |
| `dst = a * b` | 13.9 / 23.3 | 26.9 / 26.4 |
| `dst += v` | 58.6 / 49.4 | 103 / 148 |
| `dst -= v` | 58.7 / 52.7 | 103 / 147 |

### QuadMersenne31 (scalar multiply)

| Shape | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | ---: | ---: |
| `dst = a * b` | 8.97 / 4.42 | 6.98 / 2.91 |

### QuadMersenne31 (region operations)

| Shape | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | ---: | ---: |
| `dst = c * src` | 14.8 / 25.9 | 30.6 / 36.1 |
| `dst += c * src` | 12 / 20.8 | 27 / 31.8 |
| `dst = a * b` | 7.23 / 12.3 | 23.8 / 26.4 |
| `dst += v` | 62.5 / 108 | 103 / 153 |
| `dst -= v` | 49.9 / 86.4 | 103 / 153 |

Region throughput counts one operand pair over 64 KiB regions. The harnesses interleave the two arms inside one process, so quotients across columns are not paired benchmark ratios.

- The Tiger Lake prime-field cells aggregate five dedicated rounds of the recipe's own units (`just bench kernels --m31|--gdl`, then the pinned Plonky3 harness) run under `bench-run`, because the family campaign aborted before its rounds completed.

- The Plonky3 fixture builds with a snapshot-only alignment patch on Tiger Lake: its AVX-512 packed extension element is 128 bytes wide with 64-byte natural alignment, which the harness's self-alignment guard rejects. Golden Cove's narrower packing needed no patch; the patch allocates outside every timed region.

## QuadMersenne31 dispatch thresholds

Ratios are the pooled duplicated scalar control divided by the direct AVX2 candidate, midpoint estimates over two complete pinned criterion executions per host; above 1.00 the vector kernel wins. `vector_elementwise_min_bytes::<QuadMersenne31>()` reports the multiply threshold so consumers choose schedules by row length.

| Bytes | add_assign | sub_assign | mul_add | mul_into | mul_assign | mul_elementwise |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.39 / 0.34 | 0.40 / 0.36 | 0.62 / 0.46 | 0.66 / 0.55 | 1.09 / 1.14 | 0.54 / 0.49 |
| 16 | 0.50 / 0.33 | 0.53 / 0.46 | 0.99 / 0.65 | 0.82 / 0.81 | 0.82 / 0.93 | 0.85 / 0.71 |
| 24 | 0.62 / 0.53 | 0.65 / 0.57 | 1.33 / 0.88 | 1.28 / 0.94 | 0.75 / 0.83 | 1.21 / 0.91 |
| 32 | 2.72 / 2.22 | 2.59 / 2.01 | 2.39 / 1.85 | 2.18 / 1.93 | 1.10 / 1.01 | 1.61 / 1.32 |
| 40 | 0.91 / 0.75 | 1.02 / 0.82 | 0.88 / 0.67 | 1.24 / 1.09 | 1.13 / 1.27 | 0.81 / 0.67 |
| 64 | 1.81 / 1.62 | 1.61 / 1.28 | 1.89 / 1.27 | 1.62 / 1.36 | 1.20 / 0.97 | 1.19 / 1.06 |
| 128 | 0.83 / 1.28 | 1.41 / 0.87 | 1.40 / 1.13 | 1.28 / 1.11 | 1.25 / 1.09 | 1.10 / 1.03 |
| 256 | 1.16 / 1.13 | 1.21 / 1.12 | 1.15 / 1.03 | 1.12 / 1.01 | 1.12 / 1.03 | 1.09 / 1.01 |
| 2048 | 1.06 / 1.08 | 1.03 / 1.00 | 0.98 / 0.95 | 0.97 / 0.95 | 0.98 / 0.97 | 1.02 / 1.00 |

## Reproduction

Set `FEC_GOLDEN_CORE` to the host's pinned core and run from the crate root:
```sh
FEC_GOLDEN_CORE=<cpu> just bench-gf-comp
FEC_GOLDEN_CORE=<cpu> just bench-gdl-comp
FEC_GOLDEN_CORE=<cpu> just bench-m31-comp
FEC_GOLDEN_CORE=<cpu> just bench-gf2-comp
```

Each `bench-<field>-comp` recipe runs five complete rounds interleaving that field family's self suite with every competitor harness wired for the family, shuffling the unit order within each round and printing the realized order. `prime_ntt` runs through `just bench-save prime_ntt` then `just bench prime_ntt` inside `bench-m31-comp`.

| Detail | Value |
| --- | --- |
| Source of each number | Median of the per-round medians printed by the custom harnesses; `median.point_estimate` over criterion runs for `prime_ntt` |
| Samples | Up to 64 batches of 32 iterations per printed median; Criterion 10 samples per case |
| Warm-up / measurement | Custom harnesses warm 16 iterations then spend 250 ms per row; Criterion 300 ms warm-up, 1 s measurement |
| Competitor versions | ISA-L 2.32.0 (system, runtime dispatch), klauspost/reedsolomon v1.14.2 (Go 1.27.1 C archive, `GOMAXPROCS=1`), Plonky3 0.7.0 (pinned `=0.7.0`, `-C target-cpu=native` builds) |

## Coverage

Every public panel completed five rounds on both hosts; no cell is `-`. The legend above still applies to reruns that drop a host or a family.
