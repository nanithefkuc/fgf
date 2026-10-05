# Mersenne31 and `QuadMersenne31`: v2 versus v3

Same-session comparison of the v2 line and v3. Every result cell reports
Tiger Lake / Golden Cove; hosts and shared method:
[BENCHMARKS.md](../../BENCHMARKS.md). The v3 pages one level up hold the
v3 snapshot and its competitor comparisons.

## Setup

| Setting | Value |
| --- | --- |
| v2 | Revision `b844fa6` (2.1.0 line, all features) |
| v3 | Revision `b19d97b` (3.0.0, all features) |
| Build | Portable x86-64, rustc 1.98.1, thin LTO, one codegen unit; identical binaries on both hosts |
| Resolved backend | `v4x` / `v3_gfni_crypto`, checked on every run |
| Protocol | Five rounds; each round runs every suite and shuffles the version order |
| Aggregation | Median of five per-run medians |
| Ratio | v3 throughput / v2 throughput; above one means v3 is faster |
| Harness rows | Latency per call; the ratio is v2 latency / v3 latency |

## Results

### add_assign_rows — m31 (prime control, flat default)

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| flat add_assign (1 rows × 64 B) | 6.76 / 11.92 | 6.84 / 12.39 | 1.01 / 1.04 |
| add_assign_rows (1 rows × 64 B) | 5.25 / 11.03 | 5.43 / 11.03 | 1.03 / 1.00 |
| add_assign per row (1 rows × 64 B) | 5.68 / 10.66 | 5.68 / 10.72 | 1.00 / 1.01 |
| flat add_assign (2 rows × 64 B) | 12.27 / 18.70 | 11.88 / 17.83 | 0.97 / 0.95 |
| add_assign_rows (2 rows × 64 B) | 9.68 / 17.66 | 9.83 / 17.91 | 1.02 / 1.01 |
| add_assign per row (2 rows × 64 B) | 7.09 / 12.89 | 7.09 / 12.89 | 1.00 / 1.00 |
| flat add_assign (4 rows × 64 B) | 19.77 / 25.77 | 19.17 / 26.40 | 0.97 / 1.02 |
| add_assign_rows (4 rows × 64 B) | 16.77 / 24.69 | 16.80 / 24.30 | 1.00 / 0.98 |
| add_assign per row (4 rows × 64 B) | 8.07 / 14.34 | 8.07 / 14.34 | 1.00 / 1.00 |
| flat add_assign (8 rows × 64 B) | 23.30 / 30.22 | 25.30 / 30.28 | 1.09 / 1.00 |
| add_assign_rows (8 rows × 64 B) | 19.41 / 29.18 | 21.16 / 29.46 | 1.09 / 1.01 |
| add_assign per row (8 rows × 64 B) | 8.10 / 15.20 | 8.32 / 15.20 | 1.03 / 1.00 |
| flat add_assign (16 rows × 64 B) | 24.61 / 33.87 | 30.25 / 33.95 | 1.23 / 1.00 |
| add_assign_rows (16 rows × 64 B) | 25.86 / 33.28 | 27.82 / 33.39 | 1.08 / 1.00 |
| add_assign per row (16 rows × 64 B) | 8.60 / 15.65 | 9.00 / 15.65 | 1.05 / 1.00 |
| flat add_assign (32 rows × 64 B) | 27.41 / 36.97 | 27.46 / 36.95 | 1.00 / 1.00 |
| add_assign_rows (32 rows × 64 B) | 25.84 / 36.53 | 26.46 / 36.68 | 1.02 / 1.00 |
| add_assign per row (32 rows × 64 B) | 8.79 / 15.29 | 8.72 / 15.28 | 0.99 / 1.00 |
| flat add_assign (1 rows × 1024 B) | 24.63 / 34.17 | 29.92 / 34.29 | 1.21 / 1.00 |
| add_assign_rows (1 rows × 1024 B) | 25.80 / 33.32 | 27.67 / 33.46 | 1.07 / 1.00 |
| add_assign per row (1 rows × 1024 B) | 23.90 / 34.21 | 28.26 / 34.41 | 1.18 / 1.01 |
| flat add_assign (2 rows × 1024 B) | 27.46 / 36.97 | 27.38 / 36.97 | 1.00 / 1.00 |
| add_assign_rows (2 rows × 1024 B) | 25.92 / 36.53 | 26.46 / 36.64 | 1.02 / 1.00 |
| add_assign per row (2 rows × 1024 B) | 28.97 / 35.59 | 29.08 / 35.92 | 1.00 / 1.01 |
| flat add_assign (4 rows × 1024 B) | 26.14 / 36.45 | 26.30 / 36.69 | 1.01 / 1.01 |
| add_assign_rows (4 rows × 1024 B) | 25.70 / 35.69 | 25.89 / 35.68 | 1.01 / 1.00 |
| add_assign per row (4 rows × 1024 B) | 26.08 / 33.75 | 26.03 / 34.16 | 1.00 / 1.01 |
| flat add_assign (8 rows × 1024 B) | 30.62 / 36.38 | 30.72 / 37.33 | 1.00 / 1.03 |
| add_assign_rows (8 rows × 1024 B) | 30.22 / 37.24 | 30.31 / 37.14 | 1.00 / 1.00 |
| add_assign per row (8 rows × 1024 B) | 30.49 / 33.28 | 30.49 / 33.43 | 1.00 / 1.00 |
| flat add_assign (16 rows × 1024 B) | 31.20 / 38.49 | 31.46 / 38.42 | 1.01 / 1.00 |
| add_assign_rows (16 rows × 1024 B) | 31.00 / 38.44 | 31.18 / 38.31 | 1.01 / 1.00 |
| add_assign per row (16 rows × 1024 B) | 30.26 / 32.94 | 30.30 / 33.13 | 1.00 / 1.01 |
| flat add_assign (32 rows × 1024 B) | 29.20 / 36.77 | 29.34 / 37.29 | 1.00 / 1.01 |
| add_assign_rows (32 rows × 1024 B) | 29.13 / 36.67 | 29.26 / 36.90 | 1.00 / 1.01 |
| add_assign per row (32 rows × 1024 B) | 26.92 / 32.67 | 27.34 / 32.89 | 1.02 / 1.01 |
| flat add_assign (1 rows × 65536 B) | 27.58 / 37.14 | 27.58 / 37.31 | 1.00 / 1.00 |
| add_assign_rows (1 rows × 65536 B) | 27.57 / 37.18 | 27.62 / 37.41 | 1.00 / 1.01 |
| add_assign per row (1 rows × 65536 B) | 27.62 / 37.46 | 27.65 / 37.37 | 1.00 / 1.00 |
| flat add_assign (2 rows × 65536 B) | 27.65 / 36.30 | 27.63 / 37.26 | 1.00 / 1.03 |
| add_assign_rows (2 rows × 65536 B) | 27.69 / 37.28 | 27.64 / 37.34 | 1.00 / 1.00 |
| add_assign per row (2 rows × 65536 B) | 27.63 / 37.27 | 27.59 / 37.22 | 1.00 / 1.00 |
| flat add_assign (4 rows × 65536 B) | 27.68 / 36.29 | 27.74 / 37.41 | 1.00 / 1.03 |
| add_assign_rows (4 rows × 65536 B) | 27.69 / 37.15 | 27.70 / 37.34 | 1.00 / 1.01 |
| add_assign per row (4 rows × 65536 B) | 27.55 / 36.24 | 27.61 / 36.40 | 1.00 / 1.00 |
| flat add_assign (8 rows × 65536 B) | 27.08 / 32.77 | 26.13 / 34.07 | 0.96 / 1.04 |
| add_assign_rows (8 rows × 65536 B) | 27.04 / 32.75 | 26.13 / 34.25 | 0.97 / 1.05 |
| add_assign per row (8 rows × 65536 B) | 26.89 / 32.78 | 26.00 / 33.71 | 0.97 / 1.03 |
| flat add_assign (16 rows × 65536 B) | 18.12 / 23.44 | 17.87 / 23.40 | 0.99 / 1.00 |
| add_assign_rows (16 rows × 65536 B) | 18.15 / 23.37 | 17.86 / 23.39 | 0.98 / 1.00 |
| add_assign per row (16 rows × 65536 B) | 18.09 / 23.61 | 17.85 / 23.35 | 0.99 / 0.99 |
| flat add_assign (32 rows × 65536 B) | 17.86 / 22.24 | 17.76 / 22.07 | 0.99 / 0.99 |
| add_assign_rows (32 rows × 65536 B) | 17.89 / 22.25 | 17.77 / 22.33 | 0.99 / 1.00 |
| add_assign per row (32 rows × 65536 B) | 18.23 / 22.51 | 17.76 / 22.34 | 0.97 / 0.99 |

### buffer 4 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| add_assign m31 | 26.18 / 36.01 | 26.28 / 36.46 | 1.00 / 1.01 |
| add_assign_scalar m31 | 40.31 / 51.64 | 45.08 / 61.10 | 1.12 / 1.18 |
| sub_assign_scalar m31 | 32.53 / 41.46 | 45.55 / 60.61 | 1.40 / 1.46 |
| elementwise m31 | 24.63 / 32.06 | 14.95 / 19.66 | 0.61 / 0.61 |
| elementwise_assign m31 | 25.40 / 33.26 | 15.23 / 20.79 | 0.60 / 0.63 |
| add_assign qm31 | 19.57 / 38.81 | 19.73 / 35.80 | 1.01 / 0.92 |
| add_assign_scalar qm31 | 30.03 / 53.10 | 33.95 / 60.82 | 1.13 / 1.15 |
| sub_assign_scalar qm31 | 23.89 / 41.20 | 33.79 / 57.88 | 1.41 / 1.40 |
| elementwise qm31 | 12.53 / 14.83 | 11.14 / 13.67 | 0.89 / 0.92 |
| elementwise_assign qm31 | 8.42 / 15.28 | 7.56 / 13.77 | 0.90 / 0.90 |

### buffer 256 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| add_assign m31 | 27.66 / 37.22 | 27.65 / 37.40 | 1.00 / 1.00 |
| add_assign_scalar m31 | 43.57 / 54.52 | 49.34 / 61.60 | 1.13 / 1.13 |
| sub_assign_scalar m31 | 34.64 / 43.61 | 49.48 / 61.54 | 1.43 / 1.41 |
| elementwise m31 | 19.36 / 29.42 | 13.04 / 19.16 | 0.67 / 0.65 |
| elementwise_assign m31 | 21.27 / 34.57 | 13.95 / 20.27 | 0.66 / 0.59 |
| add_assign qm31 | 18.16 / 35.73 | 18.46 / 34.88 | 1.02 / 0.98 |
| add_assign_scalar qm31 | 28.87 / 54.43 | 32.35 / 61.70 | 1.12 / 1.13 |
| sub_assign_scalar qm31 | 23.27 / 43.58 | 32.11 / 61.56 | 1.38 / 1.41 |
| elementwise qm31 | 11.15 / 14.44 | 9.94 / 13.20 | 0.89 / 0.91 |
| elementwise_assign qm31 | 7.32 / 15.24 | 6.48 / 13.69 | 0.89 / 0.90 |

### buffer 8 MiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| add_assign m31 | 6.20 / 6.34 | 6.25 / 12.01 | 1.01 / 1.89 |
| add_assign_scalar m31 | 25.48 / 30.41 | 26.07 / 30.91 | 1.02 / 1.02 |
| sub_assign_scalar m31 | 23.95 / 31.03 | 26.12 / 30.78 | 1.09 / 0.99 |
| elementwise m31 | 4.54 / 5.21 | 4.05 / 4.63 | 0.89 / 0.89 |
| elementwise_assign m31 | 5.14 / 5.37 | 4.70 / 9.34 | 0.91 / 1.74 |
| add_assign qm31 | 3.18 / 6.35 | 3.52 / 12.37 | 1.11 / 1.95 |
| add_assign_scalar qm31 | 20.35 / 33.91 | 21.78 / 34.01 | 1.07 / 1.00 |
| sub_assign_scalar qm31 | 17.89 / 35.16 | 21.73 / 34.00 | 1.21 / 0.97 |
| elementwise qm31 | 3.85 / 4.30 | 3.81 / 4.19 | 0.99 / 0.97 |
| elementwise_assign qm31 | 2.50 / 4.56 | 2.44 / 7.70 | 0.98 / 1.69 |

## Caveats

- v3 makes every prime-field kernel output canonical for any accepted input lane; v2 assumed canonical inputs on its multiply paths. The Mersenne31 and `QuadMersenne31` elementwise rows carry that contract.
- Rows whose code is instruction-identical between the two revisions can still differ by a few percent from binary layout; the same-session protocol bounds, but does not remove, that effect.
- Case labels are the benchmark harness's own labels; field suffixes (`gf8`, `gf16`, `m31`, `qm31`, `gld`) name the measured field.
