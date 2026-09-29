# Benchmarks

Public API timings. Host cells list **Tiger Lake / Golden Cove**; the hosts and measurement runs are not directly comparable. Ratio tables define their operands locally. Each timing is the median of five per-run medians, and paired ratios take the median of per-round ratios. Unavailable measurements are `-`.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Tiger Lake | Intel Core i5-1135G7 | Arch Linux, Linux 7.2.7 | 1.98.1 | 3 |
| Golden Cove | Intel Core i7-12700K | CachyOS, Linux 7.2.6 | 1.98.1 | 8, isolated |

| Setting | Value |
| --- | --- |
| Crate | `fgf` 1.2.1 working tree |
| Dependencies | `simdispatch` 0.2.0, `archmage` 0.9.29, Criterion 0.8.2; ISA-L 2.32.0, klauspost/reedsolomon v1.14.2 (Go 1.27.1), Plonky3 0.7.0 |
| Build | `--all-features`; the competitor harness packages build `fgf` with `simd512` added, so both hosts measure their requested tier |
| Threads | `RAYON_NUM_THREADS=1`, `GOMAXPROCS=1`; affinity verified per process with `taskset -pc` |
| Resolved backend | `V4x` on Tiger Lake, `v3_gfni_crypto` on Golden Cove, reported by every harness banner |
| Aggregation | Median of five per-run medians, three significant figures; ratio tables describe their own comparison procedure |
| Source snapshots | `cca41d9954b1c83f44c6994668252715481dd7998159604d1f28903649ce5207` for Tiger Lake `Gf16` packed operations at 256 KiB and matrices, `Gf8B` packed `mul_add`, `mul_into`, `mul_elementwise_assign` at 256 KiB, and the `Gf8B` matrix at 8 × 4 rows; `72b26d7a4d26401ae4e162cd137bceddf3ee2d77bc3ca5e4d640bb95339e8b36` for the other results |
| Tiger Lake packed and matrix runs | `FEC_GOLDEN_CORE=3 just bench kernels --gf` for the entries named above, pinned with `taskset`, without a real-time policy or frequency pin; the other Tiger Lake results ran under `bench-run` with pinned frequency |
| Golden Cove runs | Pinned to the isolated core under `SCHED_IDLE`; no new Golden Cove measurements were available for the additional packed and prepared-matrix operations |

The Tiger Lake `Gf16` packed entries at 256 KiB and the matrix entries use AVX-512 GFNI, while scatter and gather retain narrow GFNI. Golden Cove resolves `v3_gfni_crypto`.

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

## Packed operations (binary fields, 256 KiB)

Ordinary `Vec<u8>` buffers.

| Field | Operation | Tiger Lake / Golden Cove (GiB/s) |
| --- | --- | ---: |
| `Gf8B` | `add_assign` / XOR | 25 / 51.1 |
| `Gf8B` | `mul_add` | 38.0 / 48.9 |
| `Gf8B` | `mul_assign` | 51.5 / 67.7 |
| `Gf8B` | `mul_elementwise` | 25.2 / 42.3 |
| `Gf8B` | `mul_into` | 38.2 / - |
| `Gf8B` | `mul_elementwise_assign` | 31.2 / - |
| `Gf16` | `mul_add` | 31.7 / 41.7 |
| `Gf16` | `mul_into` | 33.7 / - |
| `Gf16` | `mul_assign` | 48.5 / 54.6 |
| `Gf16` | `mul_elementwise` | 23.0 / 31 |
| `Gf16` | `mul_elementwise_assign` | 27.7 / - |

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

Each row is 64 KiB. Scatter writes n rows from one source, gather combines n sources into one row, and matrix combines eight sources into n rows. Geometry lists sources before destinations. Throughput counts the harness's logical operation volume.

| Field | Operation | Geometry | Tiger Lake / Golden Cove (GiB/s) |
| --- | --- | --- | ---: |
| `Gf8B` | `mul_add_scatter` | 1 × 2 rows | 45.5 / 70 |
| `Gf16` | `mul_add_scatter` | 1 × 2 rows | 30.6 / 56.2 |
| `Gf8B` | `mul_add_gather` | 2 × 1 row | 44.5 / 72.4 |
| `Gf16` | `mul_add_gather` | 2 × 1 row | 33.8 / 67.1 |
| `Gf8B` | `mul_add_matrix` | 8 × 2 rows | 78.5 / 103 |
| `Gf16` | `mul_add_matrix` | 8 × 2 rows | 43.5 / 47.8 |
| `Gf16` | `mul_add_matrix_with` | 8 × 2 rows | 46.8 / - |
| `Gf8B` | `mul_add_scatter` | 1 × 4 rows | 48.3 / 72.3 |
| `Gf16` | `mul_add_scatter` | 1 × 4 rows | 26.1 / 65.5 |
| `Gf8B` | `mul_add_gather` | 4 × 1 row | 43 / 79.7 |
| `Gf16` | `mul_add_gather` | 4 × 1 row | 39.6 / 77.7 |
| `Gf8B` | `mul_add_matrix` | 8 × 4 rows | 110.2 / 115 |
| `Gf16` | `mul_add_matrix` | 8 × 4 rows | 50.0 / 55.3 |
| `Gf16` | `mul_add_matrix_with` | 8 × 4 rows | 49.2 / - |
| `Gf8B` | `mul_add_scatter` | 1 × 8 rows | 48.3 / 72.3 |
| `Gf16` | `mul_add_scatter` | 1 × 8 rows | 27.1 / 65.7 |
| `Gf8B` | `mul_add_gather` | 8 × 1 row | 43.6 / 69.3 |
| `Gf16` | `mul_add_gather` | 8 × 1 row | 38.6 / 77.8 |
| `Gf8B` | `mul_add_matrix` | 8 × 8 rows | 109 / 115 |
| `Gf16` | `mul_add_matrix` | 8 × 8 rows | 50.0 / 54.9 |
| `Gf16` | `mul_add_matrix_with` | 8 × 8 rows | 49.4 / - |
| `Gf8B` | `mul_add_scatter` | 1 × 16 rows | 45.1 / 46.5 |
| `Gf16` | `mul_add_scatter` | 1 × 16 rows | 27 / 36.9 |
| `Gf8B` | `mul_add_gather` | 16 × 1 row | 43.6 / 65 |
| `Gf16` | `mul_add_gather` | 16 × 1 row | 40 / 73.3 |
| `Gf8B` | `mul_add_matrix` | 8 × 16 rows | 105 / 114 |
| `Gf16` | `mul_add_matrix` | 8 × 16 rows | 49.7 / 55.5 |
| `Gf16` | `mul_add_matrix_with` | 8 × 16 rows | 49.1 / - |

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

The payload table above is the allocator-backed baseline snapshot; row pitch
and buffer-base alignment both change across its lengths. A matched-offset
comparison fixes each source and destination base at the listed residue
modulo the cache-line width. Ratios are baseline time ÷ candidate time
(above 1.00 favors the candidate).

| Operation | Row bytes | Base offset mod 64 | Tiger Lake | Golden Cove |
| --- | ---: | ---: | ---: | ---: |
| `Gf8B` `mul_assign` | 1152 | 16 | 0.975 | 1.689 |
| `Gf8B` `mul_assign` | 1168 | 0 | 1.000 | 1.019 |
| `Gf8B` `mul_assign` | 1168 | 16 | 0.987 | 1.611 |
| `Gf8B` scatter, 4 rows | 1152 | 16 | 0.996 | 1.648 |
| `Gf8B` scatter, 16 rows | 1152 | 16 | 1.026 | 1.716 |
| `Gf8B` scatter, 16 rows | 1168 | 0 | 1.051 | 1.115 |
| `Gf8B` scatter, 16 rows | 1168 | 16 | 1.061 | 1.148 |
| `Gf8B` scatter, 16 rows | 1184 | 16 | 1.009 | 1.709 |
| `Gf8B` scatter, 16 rows | 1200 | 16 | 1.003 | 1.141 |

- Baseline: `7dbbb85`; candidate: source fingerprint `5bac3c4f14d071507aa473dad527a05d0ed22f4baf27732481b1d3ec0bba5104` over the ordered `sha256sum` output of `src/kernel/x86/gf8/{gfni,scatter,gf8d512_sg}.rs` and `benches/kernels.rs`. The diagnostic harness is identical in both builds.
- Run `FEC_GOLDEN_CORE=<cpu> just bench kernels --network-diagnostic` to build each checkout, then run its pinned benchmark binary in baseline–candidate–baseline order for five rounds. Compute each round's average bracketing baseline time divided by candidate time; the cells are medians of those paired ratios. The baseline checkout needs the candidate's `benches/kernels.rs` diagnostic harness copied in before building. Each binary reports its resolved backend.
- Both builds use `--all-features`, Rust 1.98.1, `taskset` on the environment table's core, and the same warm inputs at fixed offsets. Each cell is a median of fixed-count repeated batches after warm-up. The unchanged bracketing baseline is the drift control; rows with unstable controls are excluded from this table. `V4x` executes on Tiger Lake and `v3_gfni_crypto` on Golden Cove.

### v3 half-lane peels and overwrite-gather dispatch

Baseline `c976532b87cd0866` (the shipped tree of the AVX-512 section);
candidates `0bcadaaeb3d49196` (the five-round campaign below, three rounds
for the Tiger Lake matrix and gather rows), `c5072e891ee05aaf` (the
confirmation round below), and `125d18b87e83d73a` (the final sources,
identical to the confirmed build in every benchmarked file — the deltas
are record-pointer comments and a test fixture). Fingerprints are sha256
over sorted `src/` and `benches/`. Cells are baseline time ÷ candidate
time, medians of per-round ratios; above 1.00 favours the candidate.

| Surface | Shape | Tiger Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `xor` | 1152–1248 B, base 16 mod 32 | - | 1.28–1.31 |
| `xor` | 1152–1248 B, base 0 mod 32 | - | 0.98 |
| `Gf8B mul_add` | 1152–1248 B, base 16 mod 32 | - | 1.08–1.17 |
| `Gf8B mul_add` | 1152–1248 B, base 0 mod 32 | - | 0.90 / 0.98–1.00 |
| `Gf8B mul_add` | 4 KiB, page-aligned | - | 1.03 |
| `Gf16 mul_add` (v3) | 1.5–64 KiB, base 16 mod 64 | - | 1.06–1.36 |
| `Gf16 mul_into` (v3) | 1.5–64 KiB, base 16 mod 64 | - | 1.16–1.32 |
| `Gf16 mul_assign` (v3) | 1.5–64 KiB, base 16 mod 64 | - | 1.09–1.40 |
| `Gf16 mul_into` (V4x peel) | 3.5–16 KiB, base 16 mod 64 | 1.05–1.10 | - |
| `Gf8D` prepared overwrite gather | 4 KiB × 16 sources | 0.98 | 1.63 |
| `Gf8D` prepared overwrite gather | 16 KiB × 16 sources | 0.99 | 1.15 |
| `Gf8B` overwrite gather, raw and prepared | 4–16 KiB × 16 sources | 1.05–1.14 | 1.00 |
| `Gf16` AVX-512 matrix, selected and prepared | 8 sources × 2–16 rows | 1.14–1.18 | - |

- The aligned `Gf8B mul_add` reading splits by build: the campaign build
  read 0.90 at 1152–1248 B and 1.03 at 4 KiB; the confirmation build read
  0.98–1.00 at both — an entry-layout effect in the class the byte-XOR
  length table records, not a stable cost of the peel.
- Controls: `mul_assign` 0.98–1.02, scatter 1.00, `Gf8D` gather on `V4x`
  0.98–1.00, `Gf8B` gather on Golden Cove 1.00, `Gf8B` matrix 0.98–1.03.
- The `Gf16` two-row gather on `V4x` reads 0.94 across every measured
  build — the shortest dispatched gather shape — while the same surface's
  matrix rows gain the 1.14–1.18 above.
- Unchanged-path cells moved between builds — the `Gf16` two-row scatter
  read 1.01 in one build and 0.84 in the next although its kernels are
  identical — the code-layout class the recipe-suite table already
  records. Every cell above reproduces across both measured builds except
  the Tiger Lake sweep shapes below the V4x peel floor, which read
  0.93–0.97 in one build and at parity in the other.
- Rows come from `kernels --network-diagnostic` (network shapes),
  `peel_probe` (sweep and gather), and `kernels --gf` (matrix), each
  pinned to the environment table's core with the round order rotated.

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

## AVX-512 alignment peel floors

An offset-skew ratio does not isolate peeling: both operands can follow the
same path. This comparison holds the `Gf8D` source and destination bases at
16 mod 64 in both binaries. Gather has sixteen sources; matrix has ten
sources and four destination rows. The Tiger Lake ratio is no-peel time ÷
peel time, so values above 1.00 favor peeling. Golden Cove does not execute
these V4x kernels; its column checks baseline ÷ lower-floor candidate on
the unchanged v3 backend.

| Operation | Row bytes | Tiger Lake: no peel ÷ peel | Golden Cove: unchanged-backend control |
| --- | ---: | ---: | ---: |
| gather | 256 | 0.599 | 0.977 |
| gather | 512 | 0.640 | - |
| gather | 1024 | 0.737 | - |
| gather | 2048 | 0.864 | 0.998 |
| gather | 3072 | 1.352 | - |
| gather | 4096 | 1.552 | 1.004 |
| gather | 8192 | 1.781 | - |
| gather | 16384 | 1.893 | - |
| matrix | 256 | 0.298 | 0.993 |
| matrix | 512 | 0.418 | - |
| matrix | 1024 | 0.592 | - |
| matrix | 2048 | 0.758 | - |
| matrix | 3072 | 0.832 | - |
| matrix | 4096 | 0.926 | 0.990 |
| matrix | 8192 | 1.087 | 0.995 |
| matrix | 16384 | 1.162 | - |

- Baseline source: `2ed0840`; copy the revised `benches/compare.rs` into the
  baseline checkout. For the Tiger Lake variants, set `GATHER_PEEL_MIN` and
  `MATRIX_PEEL_MIN` together to `usize::MAX` (no peel) or 256 (peel) in
  `src/kernel/x86/gf8/`. Only these constants differ between variants. The
  production floors remain 3072 for gather and 8192 for matrix. This sweep
  does not set the scatter floor.
- Run `FEC_GOLDEN_CORE=<cpu> just bench compare --peel-diagnostic` for each
  build; then run the pinned binaries no-peel–peel–no-peel for five rounds.
  Each round divides the mean of its bracketing control medians by the
  candidate median; the table takes the median ratio across rounds. The
  fixed-offset control at 0 mod 64 stays on the same path in both builds.
  The Golden Cove control interleaves production and lower-floor builds.
- Builds use `--all-features`, Rust 1.98.1 and the environment table's
  isolated cores. The benchmark warms each fixture and takes the median of
  repeated fixed-size batches; each binary prints its resolved backend.

## AVX-512 byte, popcount, and prime-field kernels

Tiger Lake runs `V4x` in both builds and measures the new kernels. Golden Cove resolves `v3_gfni_crypto`, so its columns are an unchanged-path control that carries the same dispatch code: every operation sits at 1.00 within noise, and the byte-XOR length table shows the same short-buffer dispatch cost as Tiger Lake. The baseline tree dispatches byte XOR, `bits::weight`, `Mersenne31`, and `Goldilocks` to their AVX2 or portable paths on `V4x`, and runs `Gf16` `mul_add` without a destination peel; the candidate tree adds the 64-byte kernels and the peel. "Before ÷ after time" is the median of per-round baseline ÷ candidate ratios, so values above 1.00 favor the candidate.

### Public-path before and after

| Field | Operation | Bytes | Base offset mod 64 | Before (GiB/s) | After (GiB/s) | Tiger Lake before ÷ after | Golden Cove before ÷ after |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Gf8B` | `add_assign` | 4096 | 0 | 42.6 | 88.1 | 2.05 | 0.99 |
| `Gf16` | `add_assign` | 4096 | 0 | 41.2 | 77.6 | 1.88 | 1.01 |
| GF(2) bits | `bits::xor_assign` | 4096 | 0 | 43.2 | 82.8 | 1.92 | 0.98 |
| `Gf8B` | `add_assign` | 4096 | 16 | 29.1 | 84.2 | 2.89 | 1.00 |
| `Gf16` | `add_assign` | 4096 | 16 | 28.6 | 75.3 | 2.65 | 1.01 |
| GF(2) bits | `bits::xor_assign` | 4096 | 16 | 31.8 | 79.8 | 2.51 | 1.00 |
| `Gf8B` | `add_assign` | 65536 | 0 | 28.6 | 40.6 | 1.42 | 1.00 |
| `Gf16` | `add_assign` | 65536 | 0 | 28.5 | 40.6 | 1.42 | 1.00 |
| GF(2) bits | `bits::xor_assign` | 65536 | 0 | 28.7 | 40.7 | 1.42 | 1.00 |
| `Gf8B` | `add_assign` | 65536 | 16 | 22.2 | 40.5 | 1.83 | 1.00 |
| `Gf16` | `add_assign` | 65536 | 16 | 22.1 | 40.6 | 1.84 | 1.01 |
| GF(2) bits | `bits::xor_assign` | 65536 | 16 | 22.2 | 40.7 | 1.83 | 1.00 |
| `Gf8B` | `add_assign` | 262144 | 0 | 38.3 | 41.4 | 1.08 | 1.00 |
| `Gf16` | `add_assign` | 262144 | 0 | 38.3 | 41.4 | 1.08 | 1.00 |
| GF(2) bits | `bits::xor_assign` | 262144 | 0 | 38.3 | 41.4 | 1.08 | 1.00 |
| `Gf8B` | `add_assign` | 262144 | 16 | 21.8 | 40.9 | 1.87 | 1.00 |
| `Gf16` | `add_assign` | 262144 | 16 | 21.8 | 40.9 | 1.87 | 1.00 |
| GF(2) bits | `bits::xor_assign` | 262144 | 16 | 21.8 | 40.9 | 1.86 | 1.00 |
| GF(2) bits | `bits::weight` | 4096 | 0 | 7.19 | 131 | 18.29 | 1.00 |
| GF(2) bits | `bits::weight` | 262144 | 0 | 7.28 | 128 | 17.60 | 1.00 |
| GF(2) bits | `bits::weight` | 8388608 | 0 | 7.14 | 38.7 | 5.46 | 1.00 |
| `Mersenne31` | `mul_into` | 4096 | 0 | 16.8 | 22.7 | 1.36 | 1.02 |
| `Mersenne31` | `mul_add` | 4096 | 0 | 11.4 | 16.3 | 1.43 | 1.00 |
| `Mersenne31` | `mul_assign` | 4096 | 0 | 16.9 | 23 | 1.36 | 1.00 |
| `Mersenne31` | `add_assign` | 4096 | 0 | 22.1 | 30 | 1.35 | 1.00 |
| `Mersenne31` | `sub_assign` | 4096 | 0 | 19 | 25.5 | 1.34 | 1.00 |
| `Mersenne31` | `mul_elementwise` | 4096 | 0 | 13.1 | 18.1 | 1.38 | 0.98 |
| `Mersenne31` | `add_assign_scalar` | 4096 | 0 | 30.5 | 41 | 1.34 | 1.00 |
| `Goldilocks` | `mul_into` | 4096 | 0 | 6.7 | 11.7 | 1.75 | 1.00 |
| `Goldilocks` | `mul_add` | 4096 | 0 | 4.51 | 9.47 | 2.10 | 0.99 |
| `Goldilocks` | `mul_assign` | 4096 | 0 | 6.72 | 11.8 | 1.75 | 0.98 |
| `Goldilocks` | `add_assign` | 4096 | 0 | 9.95 | 35.3 | 3.55 | 0.98 |
| `Goldilocks` | `sub_assign` | 4096 | 0 | 10 | 44.2 | 4.41 | 1.00 |
| `Goldilocks` | `mul_elementwise` | 4096 | 0 | 6.03 | 11.2 | 1.85 | 1.00 |
| `Goldilocks` | `add_assign_scalar` | 4096 | 0 | 13.2 | 46.3 | 3.50 | 1.00 |
| `Mersenne31` | `mul_into` | 65536 | 0 | 16.4 | 23.2 | 1.42 | 1.00 |
| `Mersenne31` | `mul_add` | 65536 | 0 | 11.5 | 16.7 | 1.46 | 1.00 |
| `Mersenne31` | `mul_assign` | 65536 | 0 | 17.3 | 23.6 | 1.37 | 1.01 |
| `Mersenne31` | `add_assign` | 65536 | 0 | 22.6 | 31.5 | 1.39 | 1.00 |
| `Mersenne31` | `sub_assign` | 65536 | 0 | 19.2 | 26.4 | 1.37 | 1.00 |
| `Mersenne31` | `mul_elementwise` | 65536 | 0 | 13.3 | 18.4 | 1.39 | 1.00 |
| `Mersenne31` | `add_assign_scalar` | 65536 | 0 | 31.4 | 43.3 | 1.38 | 1.00 |
| `Goldilocks` | `mul_into` | 65536 | 0 | 6.85 | 12 | 1.74 | 0.98 |
| `Goldilocks` | `mul_add` | 65536 | 0 | 4.64 | 9.51 | 2.05 | 1.00 |
| `Goldilocks` | `mul_assign` | 65536 | 0 | 6.85 | 12 | 1.75 | 0.98 |
| `Goldilocks` | `add_assign` | 65536 | 0 | 10.2 | 36.6 | 3.58 | 1.00 |
| `Goldilocks` | `sub_assign` | 65536 | 0 | 10.2 | 40.3 | 3.93 | 1.00 |
| `Goldilocks` | `mul_elementwise` | 65536 | 0 | 6.13 | 11.4 | 1.87 | 1.00 |
| `Goldilocks` | `add_assign_scalar` | 65536 | 0 | 13.6 | 48.8 | 3.59 | 1.00 |
| `Mersenne31` | `mul_into` | 262144 | 0 | 16.3 | 23.3 | 1.43 | 1.00 |
| `Mersenne31` | `mul_add` | 262144 | 0 | 11.3 | 16.7 | 1.48 | 1.00 |
| `Mersenne31` | `mul_assign` | 262144 | 0 | 17.3 | 23.7 | 1.37 | 1.01 |
| `Mersenne31` | `add_assign` | 262144 | 0 | 22.7 | 31.7 | 1.39 | 0.99 |
| `Mersenne31` | `sub_assign` | 262144 | 0 | 19.4 | 26.5 | 1.36 | 1.00 |
| `Mersenne31` | `mul_elementwise` | 262144 | 0 | 13.4 | 18.5 | 1.38 | 1.00 |
| `Mersenne31` | `add_assign_scalar` | 262144 | 0 | 31.7 | 43.5 | 1.38 | 1.00 |
| `Goldilocks` | `mul_into` | 262144 | 0 | 6.86 | 12 | 1.75 | 0.99 |
| `Goldilocks` | `mul_add` | 262144 | 0 | 4.46 | 9.13 | 2.05 | 1.00 |
| `Goldilocks` | `mul_assign` | 262144 | 0 | 6.86 | 12 | 1.75 | 0.98 |
| `Goldilocks` | `add_assign` | 262144 | 0 | 10.2 | 37.3 | 3.63 | 1.00 |
| `Goldilocks` | `sub_assign` | 262144 | 0 | 10.3 | 40.6 | 3.93 | 1.01 |
| `Goldilocks` | `mul_elementwise` | 262144 | 0 | 6.13 | 11.5 | 1.87 | 1.00 |
| `Goldilocks` | `add_assign_scalar` | 262144 | 0 | 13.6 | 49.2 | 3.61 | 1.00 |
| `Gf16` | `mul_add` | 4096 | 0 | 52.7 | 55.9 | 1.06 | 1.00 |
| `Gf16` | `mul_add` | 4096 | 16 | 37.8 | 51.1 | 1.35 | 1.00 |
| `Gf16` | `mul_add` | 65536 | 0 | 40.2 | 40.2 | 1.00 | 0.98 |
| `Gf16` | `mul_add` | 65536 | 16 | 27.7 | 40.1 | 1.44 | 1.00 |
| `Gf16` | `mul_add` | 262144 | 0 | 40.8 | 40.7 | 1.00 | 0.99 |
| `Gf16` | `mul_add` | 262144 | 16 | 27.9 | 40.7 | 1.46 | 0.99 |

### Byte XOR by length (`Gf8B` `add_assign`)

| Bytes | Base offset mod 64 | Tiger Lake before (ns) | Tiger Lake after (ns) | Tiger Lake ratio | Golden Cove before (ns) | Golden Cove after (ns) | Golden Cove ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 0 | 6.35 | 7.02 | 0.90 | 3.68 | 3.89 | 0.89 |
| 32 | 16 | 6.35 | 7.02 | 0.90 | 3.68 | 3.68 | 1.00 |
| 64 | 0 | 6.68 | 7.02 | 0.95 | 3.52 | 3.68 | 0.96 |
| 64 | 16 | 6.68 | 7.03 | 0.95 | 3.68 | 3.68 | 1.00 |
| 128 | 0 | 7.35 | 7.7 | 0.96 | 3.89 | 4.09 | 0.95 |
| 128 | 16 | 7.36 | 7.7 | 0.96 | 4.09 | 4.1 | 0.95 |
| 256 | 0 | 8.7 | 7.37 | 1.18 | 4.7 | 4.92 | 0.96 |
| 256 | 16 | 8.7 | 7.68 | 1.18 | 4.91 | 5.32 | 0.92 |
| 512 | 0 | 11.4 | 9.11 | 1.26 | 6.36 | 6.55 | 0.97 |
| 512 | 16 | 13 | 11 | 1.18 | 7.77 | 7.77 | 1.00 |
| 768 | 0 | 14.1 | 11.2 | 1.26 | 8.11 | 8.18 | 0.99 |
| 768 | 16 | 23.8 | 12.4 | 1.92 | 14.7 | 14.7 | 1.00 |
| 1024 | 0 | 17.3 | 12.4 | 1.40 | 9.68 | 9.82 | 0.99 |
| 1024 | 16 | 24.4 | 13.7 | 1.78 | 12.7 | 15.2 | 0.88 |
| 1536 | 0 | 23.4 | 16.6 | 1.41 | 13 | 13.2 | 0.99 |
| 1536 | 16 | 33.9 | 16.4 | 2.04 | 20.6 | 19.4 | 1.03 |
| 2048 | 0 | 30.4 | 19.8 | 1.53 | 18.3 | 18.6 | 0.98 |
| 2048 | 16 | 48.4 | 19.9 | 2.43 | 27.7 | 27.2 | 1.02 |
| 4096 | 0 | 90.7 | 47 | 1.93 | 30.4 | 30.6 | 0.99 |
| 4096 | 16 | 134 | 47.9 | 2.82 | 57.2 | 57.6 | 0.99 |

### Short `Gf16` `mul_add` rows

| Row bytes | Base offset mod 64 | Tiger Lake before (ns) | Tiger Lake after (ns) | Tiger Lake ratio | Golden Cove before (ns) | Golden Cove after (ns) | Golden Cove ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 16 | 19.1 | 19.2 | 1.00 | 10.8 | 10.8 | 1.00 |
| 16 | 0 | 19.1 | 19.2 | 1.00 | 10.8 | 10.8 | 1.00 |
| 64 | 16 | 18.9 | 16.2 | 1.17 | 11.4 | 11.4 | 1.00 |
| 64 | 0 | 18.7 | 16.1 | 1.16 | 11.4 | 11.4 | 1.00 |
| 256 | 16 | 21.5 | 18.7 | 1.14 | 13.6 | 13.5 | 1.01 |
| 256 | 0 | 21.4 | 18.8 | 1.13 | 13.4 | 13.4 | 1.00 |
| 1024 | 16 | 35.6 | 35.6 | 1.00 | 24.9 | 24.7 | 1.01 |
| 1024 | 0 | 29.6 | 27.5 | 1.09 | 20.9 | 20.9 | 1.00 |
| 2048 | 16 | 44.7 | 42.9 | 1.04 | 40.1 | 40 | 1.00 |
| 2048 | 0 | 40.5 | 37.7 | 1.07 | 35.2 | 35.1 | 1.00 |

### Threshold variants

Each row changes one constant in the candidate tree and compares it with the shipped value in the same session. Values above 1.00 favor the shipped value.

| Change from the shipped value | Bytes | Base offset mod 64 | Shipped ÷ variant time |
| --- | ---: | ---: | ---: |
| `XOR512_MIN` 256 → 64 | 128 | 0 | 1.10 |
| `XOR512_MIN` 256 → 64 | 128 | 16 | 1.10 |
| `XOR512_MIN` 256 → 64 | 256 | 0 | 1.00 |
| `XOR512_MIN` 256 → 64 | 256 | 16 | 1.00 |
| `XOR512_MIN` 256 → 64 | 512 | 0 | 1.12 |
| `XOR512_MIN` 256 → 64 | 512 | 16 | 1.01 |
| `XOR512_MIN` 256 → 64 | 1024 | 0 | 1.03 |
| `XOR512_MIN` 256 → 64 | 1024 | 16 | 1.00 |
| `XOR512_MIN` 256 → 64 | 2048 | 0 | 0.97 |
| `XOR512_MIN` 256 → 64 | 2048 | 16 | 0.95 |
| `XOR512_MIN` 256 → 64 | 4096 | 0 | 0.99 |
| `XOR512_MIN` 256 → 64 | 4096 | 16 | 1.00 |
| `XOR_PEEL_MIN` 512 → no peel | 128 | 0 | 0.96 |
| `XOR_PEEL_MIN` 512 → no peel | 128 | 16 | 0.96 |
| `XOR_PEEL_MIN` 512 → no peel | 256 | 0 | 1.00 |
| `XOR_PEEL_MIN` 512 → no peel | 256 | 16 | 1.05 |
| `XOR_PEEL_MIN` 512 → no peel | 512 | 0 | 1.03 |
| `XOR_PEEL_MIN` 512 → no peel | 512 | 16 | 0.87 |
| `XOR_PEEL_MIN` 512 → no peel | 1024 | 0 | 1.01 |
| `XOR_PEEL_MIN` 512 → no peel | 1024 | 16 | 0.73 |
| `XOR_PEEL_MIN` 512 → no peel | 2048 | 0 | 0.98 |
| `XOR_PEEL_MIN` 512 → no peel | 2048 | 16 | 0.54 |
| `XOR_PEEL_MIN` 512 → no peel | 4096 | 0 | 1.09 |
| `XOR_PEEL_MIN` 512 → no peel | 4096 | 16 | 0.61 |
| `XOR_PEEL_MIN` 512 → 0 | 128 | 0 | 0.96 |
| `XOR_PEEL_MIN` 512 → 0 | 128 | 16 | 0.96 |
| `XOR_PEEL_MIN` 512 → 0 | 256 | 0 | 0.85 |
| `XOR_PEEL_MIN` 512 → 0 | 256 | 16 | 0.73 |
| `XOR_PEEL_MIN` 512 → 0 | 512 | 0 | 0.96 |
| `XOR_PEEL_MIN` 512 → 0 | 512 | 16 | 0.93 |
| `XOR_PEEL_MIN` 512 → 0 | 1024 | 0 | 0.85 |
| `XOR_PEEL_MIN` 512 → 0 | 1024 | 16 | 0.93 |
| `XOR_PEEL_MIN` 512 → 0 | 2048 | 0 | 0.86 |
| `XOR_PEEL_MIN` 512 → 0 | 2048 | 16 | 0.82 |
| `XOR_PEEL_MIN` 512 → 0 | 4096 | 0 | 0.99 |
| `XOR_PEEL_MIN` 512 → 0 | 4096 | 16 | 0.98 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → no peel | 1536 | 16 | 1.02 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → no peel | 2048 | 16 | 1.07 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → no peel | 3072 | 16 | 1.04 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → no peel | 3584 | 16 | 0.93 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → no peel | 4096 | 16 | 0.77 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → no peel | 8192 | 16 | 0.77 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → 128 | 1536 | 16 | 0.88 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → 128 | 2048 | 16 | 1.02 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → 128 | 3072 | 16 | 1.11 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → 128 | 3584 | 16 | 1.01 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → 128 | 4096 | 16 | 1.01 |
| `Gf16` `MUL_ADD_PEEL_MIN` 3584 → 128 | 8192 | 16 | 1.00 |

### Recipe-suite cells slower in every round

`just bench kernels --gf`, interleaved baseline and candidate. The table lists every cell whose five per-round ratios all fall below 0.97; the unchanged `Gf32`, `Gf64`, and Fan-Paar cells stay within 0.97–1.02.

| Panel | Field | Shape | Case | Before (ns) | After (ns) | Before ÷ after time |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `add_assign_rows` | `Gf8B` | 16 rows × 64 B | add_assign per row | 89.6 | 113 | 0.79 |
| `add_assign_rows` | `Gf16` | 16 rows × 64 B | add_assign per row | 94.2 | 118 | 0.80 |
| `add_assign_rows` | `Gf8B` | 8 rows × 64 B | add_assign_rows | 13.6 | 16 | 0.85 |
| `add_assign_rows` | `Gf8B` | 32 rows × 64 B | add_assign per row | 183 | 215 | 0.85 |
| `add_assign_rows` | `Gf16` | 32 rows × 64 B | add_assign per row | 193 | 225 | 0.86 |
| `add_assign_rows` | `Gf8B` | 8 rows × 64 B | add_assign per row | 46.6 | 54 | 0.86 |
| `add_assign_rows` | `Gf16` | 4 rows × 64 B | add_assign per row | 25.6 | 29.5 | 0.87 |
| `add_assign_rows` | `Gf8B` | 4 rows × 64 B | add_assign per row | 25.2 | 28.5 | 0.89 |
| `add_assign_rows` | `Gf16` | 2 rows × 64 B | add_assign per row | 14.6 | 16.2 | 0.90 |
| `add_assign_rows` | `Gf8B` | 2 rows × 64 B | add_assign per row | 14.6 | 15.8 | 0.92 |
| `add_assign_rows` | `Gf8B` | 2 rows × 64 B | flat add_assign | 8.12 | 8.75 | 0.93 |
| `add_assign_rows` | `Gf16` | 2 rows × 64 B | flat add_assign | 8.44 | 9.03 | 0.94 |
| bit-packed GF(2) | GF(2) bits | 4 KiB buffers | `bits::dot_product` | 188 | 211 | 0.89 |
| bit-packed GF(2) | GF(2) bits | 256 row pairs, 16-byte rows | `bits::xor_range_with`, bits 61..69 | 1060 | 1150 | 0.92 |
| bit-packed GF(2) | GF(2) bits | 256 KiB buffers | `bits::dot_product` | 12800 | 13400 | 0.95 |
| network payloads | `Gf8B` | payload 64 B | `add_assign` | 7.72 | 8.06 | 0.95 |
| blocked vs AXPY (dispatch bypassed) | `Gf16` | 4 sources × 4 rows × 4 KiB | direct `mul_add_matrix_avx2` | 5760 | 6020 | 0.96 |

- Host: Tiger Lake row of the environment table, CPU 3 isolated, `performance` governor at a fixed 3.0 GHz, `SCHED_FIFO` priority 99 under `bench-run`; turbostat recorded no thermal throttling.
- Source fingerprints (`sha256sum` over sorted `src/` and `benches/`): baseline `0091d06c0867a266cb14568d9e1d4f8af7b25841d313e33aedececcb49c347dd`, candidate `40ecbfc1bd002543cb262b4feeb018095b49f7023e7e29599fdee912e1c46690`.
- The public-path, length, and short-row tables come from a throwaway harness that calls `fgf::ops` and `fgf::bits` on buffers placed at a stated offset from a 64-byte boundary. Both builds link the same harness source with `simd512`. Before any timing, each binary checks every candidate path against an independent oracle: byte-wise XOR, exact integer arithmetic for the prime fields, and per-byte `count_ones` for `bits::weight`.
- The public-path, length, and short-row tables use seven rounds with shuffled build order. Every row warms 16 iterations and reports the median of 15 fixed-size batches. An unchanged `Gf8B` `mul_add` control at 64 KiB stays within 0.94–1.06 in every campaign.
- The threshold rows use seven shuffled rounds per variant set in one session. The recipe-suite table uses five shuffled rounds of the full `kernels` bench binary.
- Golden Cove control: isolated core 8 under `SCHED_IDLE`, `powersave` governor, seven shuffled rounds for the harness tables and five for the recipe suite; every banner reported `v3_gfni_crypto`. Its baseline tree is a reconstruction that reverts the four candidate changes (`dc6dc3453599d909a183ab461b4f5a2fe76ea2fb96249e043a60309a4c156809`) and its candidate is the shipped tree (`c976532b87cd08666447da5707ee90ecbe0521c0ddafa0ba9d52263a61898204`); the two trees differ only in `src/` and `benches/` content covered by the fingerprint method above.
- Golden Cove recipe suite: two cells fall below 0.97 in every round — the `Gf32` polynomial-tower `mul_add` at 256 KiB (0.93) and the direct blocked AVX2 matrix at 2 sources × 4 rows × 4 KiB (0.97). Neither path is touched by the candidate; the `Gf32` cell is the same class of code-layout effect as the Tiger Lake `bits::dot_product` cells. The Golden Cove threshold variants were not measured; the floors only change behaviour on `V4x`.

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

### Mersenne31 with `simd512` on Tiger Lake

One pinned invocation of the prime harness built with
`--features fgf/simd512`, the feature set the family recipes now build;
the Plonky3 arm is the unchanged AVX2 body.

| Shape | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | ---: | ---: |
| `dst = c * src` | 48.0 | 65.4 |
| `dst += c * src` | 33.5 | 49.8 |
| `dst = a * b` | 37.2 | 59.9 |
| `dst += v` | 87.6 | 102.9 |
| `dst -= v` | 70.3 | 102.9 |

- The Mersenne31 panel above was recorded with `fgf` resolving
  `v3_gfni_crypto`: the pre-change harness builds no `simd512`, and a
  forced-`V3` rerun of the `simd512` build reproduces those cells, which
  attributes the difference to the build's feature set. The Goldilocks
  and QuadMersenne31 panels carry the same qualifier.

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

The competitor panels ran on both hosts; the additional packed and prepared-matrix entries ran on Tiger Lake only. The peel diagnostic marks V4x measurements unavailable on Golden Cove; its v3 control cells are labelled separately. The AVX-512 byte, popcount, and prime-field section ran on both hosts; its Golden Cove columns are unchanged-path controls, and its threshold variants ran on Tiger Lake only.
