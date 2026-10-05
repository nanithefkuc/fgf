# GF(2^8): v2 versus v3

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

### row 16 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 1.36 / 3.46 | 1.35 / 3.46 | 0.99 / 1.00 |
| mul_add prepared gf8 | 1.37 / 3.61 | 1.37 / 3.61 | 1.00 / 1.00 |

### row 32 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 2.71 / 6.86 | 2.48 / 6.91 | 0.92 / 1.01 |
| mul_add prepared gf8 | 2.76 / 7.22 | 2.50 / 7.22 | 0.91 / 1.00 |

### row 64 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 4.94 / 13.15 | 4.93 / 13.72 | 1.00 / 1.04 |
| mul_add prepared gf8 | 4.98 / 13.72 | 4.99 / 13.72 | 1.00 / 1.00 |

### row 128 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 10.8 / 19.8 | 10.8 / 23.0 | 1.01 / 1.16 |
| mul_add prepared gf8 | 10.8 / 20.6 | 10.9 / 24.0 | 1.01 / 1.16 |

### row 256 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 18.4 / 38.5 | 18.3 / 33.2 | 1.00 / 0.86 |
| mul_add prepared gf8 | 18.3 / 39.7 | 18.3 / 39.9 | 1.00 / 1.01 |

### row 512 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 26.0 / 58.0 | 26.5 / 58.5 | 1.02 / 1.01 |
| mul_add prepared gf8 | 25.8 / 56.1 | 26.7 / 59.6 | 1.03 / 1.06 |

### row 1024 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 50.0 / 71.5 | 44.5 / 71.8 | 0.89 / 1.00 |
| mul_add prepared gf8 | 50.7 / 72.2 | 44.2 / 72.2 | 0.87 / 1.00 |

### row 2048 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 68.6 / 79.8 | 68.0 / 80.3 | 0.99 / 1.01 |
| mul_add prepared gf8 | 68.7 / 80.2 | 68.7 / 80.7 | 1.00 / 1.01 |

### row 4096 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 67.0 / 83.1 | 67.6 / 83.2 | 1.01 / 1.00 |
| mul_add prepared gf8 | 67.4 / 83.3 | 68.0 / 83.5 | 1.01 / 1.00 |

### row 16384 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 75.2 / 88.2 | 75.0 / 88.2 | 1.00 / 1.00 |
| mul_add prepared gf8 | 75.0 / 88.7 | 75.4 / 88.3 | 1.01 / 1.00 |

### row 65536 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add one-shot gf8 | 37.5 / 58.3 | 37.4 / 58.3 | 1.00 / 1.00 |
| mul_add prepared gf8 | 37.5 / 59.1 | 37.5 / 58.3 | 1.00 / 0.99 |

### add_assign_rows — gf8

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| flat add_assign (1 rows × 64 B) | 6.55 / 13.72 | 6.60 / 13.62 | 1.01 / 0.99 |
| add_assign_rows (1 rows × 64 B) | 6.55 / 12.63 | 6.34 / 12.98 | 0.97 / 1.03 |
| add_assign per row (1 rows × 64 B) | 6.15 / 11.77 | 6.00 / 11.77 | 0.98 / 1.00 |
| flat add_assign (2 rows × 64 B) | 13.20 / 25.10 | 13.15 / 27.05 | 1.00 / 1.08 |
| add_assign_rows (2 rows × 64 B) | 12.72 / 23.99 | 12.15 / 25.43 | 0.96 / 1.06 |
| add_assign per row (2 rows × 64 B) | 6.72 / 13.15 | 6.82 / 12.51 | 1.01 / 0.95 |
| flat add_assign (4 rows × 64 B) | 25.26 / 43.35 | 25.35 / 43.85 | 1.00 / 1.01 |
| add_assign_rows (4 rows × 64 B) | 23.92 / 43.60 | 23.69 / 39.74 | 0.99 / 0.91 |
| add_assign per row (4 rows × 64 B) | 7.19 / 12.13 | 7.41 / 13.34 | 1.03 / 1.10 |
| flat add_assign (8 rows × 64 B) | 26.82 / 58.91 | 28.31 / 58.69 | 1.06 / 1.00 |
| add_assign_rows (8 rows × 64 B) | 22.34 / 57.80 | 24.26 / 57.80 | 1.09 / 1.00 |
| add_assign per row (8 rows × 64 B) | 7.45 / 12.67 | 7.73 / 13.37 | 1.04 / 1.06 |
| flat add_assign (16 rows × 64 B) | 49.06 / 83.15 | 48.83 / 83.61 | 1.00 / 1.01 |
| add_assign_rows (16 rows × 64 B) | 43.60 / 80.52 | 45.35 / 80.10 | 1.04 / 0.99 |
| add_assign per row (16 rows × 64 B) | 7.09 / 13.69 | 7.41 / 14.64 | 1.05 / 1.07 |
| flat add_assign (32 rows × 64 B) | 73.27 / 93.33 | 73.18 / 92.90 | 1.00 / 1.00 |
| add_assign_rows (32 rows × 64 B) | 66.49 / 90.83 | 66.13 / 90.16 | 0.99 / 0.99 |
| add_assign per row (32 rows × 64 B) | 7.41 / 14.62 | 7.74 / 13.97 | 1.04 / 0.96 |
| flat add_assign (1 rows × 1024 B) | 50.28 / 83.61 | 50.53 / 83.61 | 1.00 / 1.00 |
| add_assign_rows (1 rows × 1024 B) | 43.60 / 80.31 | 45.48 / 80.10 | 1.04 / 1.00 |
| add_assign per row (1 rows × 1024 B) | 42.04 / 85.97 | 45.14 / 85.72 | 1.07 / 1.00 |
| flat add_assign (2 rows × 1024 B) | 73.36 / 93.33 | 73.36 / 92.90 | 1.00 / 1.00 |
| add_assign_rows (2 rows × 1024 B) | 65.42 / 90.83 | 66.06 / 90.16 | 1.01 / 0.99 |
| add_assign per row (2 rows × 1024 B) | 50.36 / 86.09 | 51.03 / 85.97 | 1.01 / 1.00 |
| flat add_assign (4 rows × 1024 B) | 44.08 / 96.88 | 44.10 / 96.80 | 1.00 / 1.00 |
| add_assign_rows (4 rows × 1024 B) | 42.34 / 96.27 | 42.50 / 95.89 | 1.00 / 1.00 |
| add_assign per row (4 rows × 1024 B) | 43.67 / 86.57 | 43.80 / 86.45 | 1.00 / 1.00 |
| flat add_assign (8 rows × 1024 B) | 81.57 / 134.14 | 80.23 / 133.63 | 0.98 / 1.00 |
| add_assign_rows (8 rows × 1024 B) | 79.45 / 130.63 | 77.88 / 128.50 | 0.98 / 0.98 |
| add_assign per row (8 rows × 1024 B) | 55.35 / 103.06 | 55.35 / 102.80 | 1.00 / 1.00 |
| flat add_assign (16 rows × 1024 B) | 84.32 / 137.23 | 83.87 / 137.89 | 0.99 / 1.00 |
| add_assign_rows (16 rows × 1024 B) | 83.61 / 133.30 | 82.54 / 132.18 | 0.99 / 0.99 |
| add_assign per row (16 rows × 1024 B) | 53.29 / 100.76 | 53.87 / 100.74 | 1.01 / 1.00 |
| flat add_assign (32 rows × 1024 B) | 34.89 / 66.75 | 34.89 / 66.68 | 1.00 / 1.00 |
| add_assign_rows (32 rows × 1024 B) | 34.72 / 66.57 | 34.71 / 66.37 | 1.00 / 1.00 |
| add_assign per row (32 rows × 1024 B) | 31.69 / 62.83 | 30.94 / 62.74 | 0.98 / 1.00 |
| flat add_assign (1 rows × 65536 B) | 31.76 / 63.29 | 31.72 / 63.28 | 1.00 / 1.00 |
| add_assign_rows (1 rows × 65536 B) | 31.78 / 63.31 | 31.73 / 63.20 | 1.00 / 1.00 |
| add_assign per row (1 rows × 65536 B) | 31.78 / 63.43 | 31.74 / 63.38 | 1.00 / 1.00 |
| flat add_assign (2 rows × 65536 B) | 31.93 / 63.57 | 31.92 / 63.56 | 1.00 / 1.00 |
| add_assign_rows (2 rows × 65536 B) | 31.92 / 63.55 | 31.92 / 63.52 | 1.00 / 1.00 |
| add_assign per row (2 rows × 65536 B) | 31.87 / 63.24 | 31.85 / 63.27 | 1.00 / 1.00 |
| flat add_assign (4 rows × 65536 B) | 32.02 / 63.70 | 32.02 / 63.70 | 1.00 / 1.00 |
| add_assign_rows (4 rows × 65536 B) | 32.02 / 63.69 | 32.01 / 63.69 | 1.00 / 1.00 |
| add_assign per row (4 rows × 65536 B) | 31.93 / 63.39 | 31.90 / 63.41 | 1.00 / 1.00 |
| flat add_assign (8 rows × 65536 B) | 39.16 / 48.45 | 35.66 / 47.08 | 0.91 / 0.97 |
| add_assign_rows (8 rows × 65536 B) | 40.01 / 48.58 | 35.64 / 47.13 | 0.89 / 0.97 |
| add_assign per row (8 rows × 65536 B) | 39.85 / 48.65 | 35.53 / 47.25 | 0.89 / 0.97 |
| flat add_assign (16 rows × 65536 B) | 18.91 / 26.77 | 18.86 / 26.67 | 1.00 / 1.00 |
| add_assign_rows (16 rows × 65536 B) | 18.91 / 26.77 | 18.86 / 26.78 | 1.00 / 1.00 |
| add_assign per row (16 rows × 65536 B) | 18.86 / 26.73 | 18.86 / 26.80 | 1.00 / 1.00 |
| flat add_assign (32 rows × 65536 B) | 18.43 / 25.79 | 18.37 / 25.79 | 1.00 / 1.00 |
| add_assign_rows (32 rows × 65536 B) | 18.42 / 25.79 | 18.38 / 25.79 | 1.00 / 1.00 |
| add_assign per row (32 rows × 65536 B) | 18.40 / 25.74 | 18.36 / 25.75 | 1.00 / 1.00 |

### large destinations — mul_into store policy

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| 1 MiB mul_into gf8 | 18.81 / 26.41 | 18.88 / 26.42 | 1.00 / 1.00 |
| 1 MiB mul_into+read gf8 | 13.67 / 20.95 | 13.84 / 20.86 | 1.01 / 1.00 |
| 1 MiB mul_add gf8 | 18.80 / 25.09 | 18.89 / 25.12 | 1.00 / 1.00 |
| 8 MiB mul_into gf8 | 18.19 / 41.76 | 18.20 / 41.76 | 1.00 / 1.00 |
| 8 MiB mul_into+read gf8 | 7.69 / 16.55 | 7.73 / 16.58 | 1.01 / 1.00 |
| 8 MiB mul_add gf8 | 8.13 / 25.00 | 8.31 / 25.76 | 1.02 / 1.03 |
| 32 MiB mul_into gf8 | 9.34 / 28.26 | 9.38 / 28.31 | 1.00 / 1.00 |
| 32 MiB mul_into+read gf8 | 5.93 / 14.21 | 5.94 / 14.26 | 1.00 / 1.00 |
| 32 MiB mul_add gf8 | 6.21 / 16.57 | 6.24 / 16.57 | 1.00 / 1.00 |

### destination alignment — multi-row scatter

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| 64 KiB rows, skew 0 gf8 | 48.5 / 71.9 | 48.5 / 72.2 | 1.00 / 1.00 |
| 64 KiB rows, skew 16 gf8 | 48.3 / 71.7 | 48.3 / 72.1 | 1.00 / 1.00 |
| 256 KiB rows, skew 0 gf8 | 29.3 / 31.2 | 29.6 / 31.4 | 1.01 / 1.01 |
| 256 KiB rows, skew 16 gf8 | 29.5 / 23.8 | 29.5 / 23.8 | 1.00 / 1.00 |

### gather 2 sources x 4 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.46 / 7.33 | 3.48 / 7.36 | 1.01 / 1.00 |
| gather AXPY ssse3 | 4.96 / 8.85 | 4.97 / 8.90 | 1.00 / 1.01 |
| gather blocked gfni | 28.96 / 55.35 | 31.67 / 57.98 | 1.09 / 1.05 |
| gather AXPY gfni | 31.72 / 57.51 | 31.67 / 57.42 | 1.00 / 1.00 |

### gather 4 sources x 4 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.98 / 7.86 | 3.99 / 7.88 | 1.00 / 1.00 |
| gather AXPY ssse3 | 5.18 / 8.91 | 5.19 / 8.93 | 1.00 / 1.00 |
| gather blocked gfni | 40.43 / 75.41 | 40.39 / 75.27 | 1.00 / 1.00 |
| gather AXPY gfni | 34.38 / 61.23 | 34.29 / 61.78 | 1.00 / 1.01 |

### gather 8 sources x 4 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 4.06 / 8.04 | 4.09 / 8.07 | 1.01 / 1.00 |
| gather AXPY ssse3 | 5.15 / 8.96 | 5.14 / 8.85 | 1.00 / 0.99 |
| gather blocked gfni | 39.35 / 75.68 | 39.23 / 75.54 | 1.00 / 1.00 |
| gather AXPY gfni | 28.80 / 57.70 | 28.73 / 58.14 | 1.00 / 1.01 |

### gather 16 sources x 4 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.92 / 7.96 | 3.96 / 7.98 | 1.01 / 1.00 |
| gather AXPY ssse3 | 5.22 / 8.94 | 5.22 / 8.95 | 1.00 / 1.00 |
| gather blocked gfni | 40.61 / 72.66 | 39.06 / 72.76 | 0.96 / 1.00 |
| gather AXPY gfni | 24.55 / 56.71 | 24.36 / 57.21 | 0.99 / 1.01 |

### gather 2 sources x 16 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.46 / 7.54 | 3.48 / 7.55 | 1.01 / 1.00 |
| gather AXPY ssse3 | 5.18 / 9.06 | 5.18 / 9.06 | 1.00 / 1.00 |
| gather blocked gfni | 31.73 / 61.09 | 34.97 / 63.21 | 1.10 / 1.03 |
| gather AXPY gfni | 33.70 / 58.57 | 33.19 / 59.55 | 0.98 / 1.02 |

### gather 4 sources x 16 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.81 / 7.98 | 3.85 / 7.99 | 1.01 / 1.00 |
| gather AXPY ssse3 | 5.23 / 9.08 | 5.24 / 9.16 | 1.00 / 1.01 |
| gather blocked gfni | 39.33 / 72.71 | 39.30 / 71.49 | 1.00 / 0.98 |
| gather AXPY gfni | 25.74 / 55.34 | 25.69 / 55.75 | 1.00 / 1.01 |

### gather 8 sources x 16 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.99 / 8.15 | 4.04 / 8.16 | 1.01 / 1.00 |
| gather AXPY ssse3 | 5.29 / 9.19 | 5.29 / 9.26 | 1.00 / 1.01 |
| gather blocked gfni | 38.67 / 73.95 | 38.65 / 74.53 | 1.00 / 1.01 |
| gather AXPY gfni | 24.65 / 56.79 | 24.63 / 58.02 | 1.00 / 1.02 |

### gather 16 sources x 16 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.98 / 8.15 | 4.02 / 8.16 | 1.01 / 1.00 |
| gather AXPY ssse3 | 5.38 / 9.21 | 5.39 / 9.22 | 1.00 / 1.00 |
| gather blocked gfni | 40.61 / 75.25 | 40.61 / 75.62 | 1.00 / 1.00 |
| gather AXPY gfni | 24.49 / 58.28 | 24.41 / 58.74 | 1.00 / 1.01 |

### gather 2 sources x 64 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.46 / 7.62 | 3.44 / 7.63 | 0.99 / 1.00 |
| gather AXPY ssse3 | 5.06 / 9.12 | 5.06 / 9.12 | 1.00 / 1.00 |
| gather blocked gfni | 27.88 / 51.52 | 30.50 / 54.82 | 1.09 / 1.06 |
| gather AXPY gfni | 25.92 / 51.34 | 25.91 / 52.14 | 1.00 / 1.02 |

### gather 4 sources x 64 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.71 / 8.08 | 3.77 / 8.09 | 1.02 / 1.00 |
| gather AXPY ssse3 | 5.13 / 9.14 | 5.13 / 9.14 | 1.00 / 1.00 |
| gather blocked gfni | 33.44 / 72.52 | 33.45 / 72.08 | 1.00 / 0.99 |
| gather AXPY gfni | 23.72 / 52.26 | 23.72 / 51.57 | 1.00 / 0.99 |

### gather 8 sources x 64 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.57 / 7.97 | 3.57 / 7.99 | 1.00 / 1.00 |
| gather AXPY ssse3 | 5.24 / 9.15 | 5.24 / 9.15 | 1.00 / 1.00 |
| gather blocked gfni | 33.60 / 72.70 | 33.59 / 72.54 | 1.00 / 1.00 |
| gather AXPY gfni | 22.85 / 52.10 | 22.83 / 52.06 | 1.00 / 1.00 |

### gather 16 sources x 64 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| gather blocked ssse3 | 3.56 / 7.98 | 3.58 / 7.98 | 1.01 / 1.00 |
| gather AXPY ssse3 | 5.37 / 9.15 | 5.36 / 9.15 | 1.00 / 1.00 |
| gather blocked gfni | 35.38 / 73.15 | 34.89 / 73.22 | 0.99 / 1.00 |
| gather AXPY gfni | 22.77 / 51.10 | 22.59 / 51.44 | 0.99 / 1.01 |

### payload 64 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 6.60 / 12.63 | 6.58 / 13.62 | 1.00 / 1.08 |
| mul_add | 5.36 / 13.15 | 5.40 / 12.98 | 1.01 / 0.99 |
| mul_assign | 8.44 / 19.07 | 7.95 / 17.66 | 0.94 / 0.93 |
| scatter (4 rows) | 8.76 / 18.08 | 8.73 / 18.12 | 1.00 / 1.00 |
| scatter (16 rows) | 13.12 / 27.54 | 13.09 / 27.59 | 1.00 / 1.00 |

### payload 256 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 25.3 / 43.6 | 26.0 / 43.6 | 1.03 / 1.00 |
| mul_add | 17.7 / 38.7 | 17.8 / 38.5 | 1.00 / 0.99 |
| mul_assign | 25.2 / 67.5 | 20.9 / 67.5 | 0.83 / 1.00 |
| scatter (4 rows) | 30.2 / 56.0 | 30.1 / 56.2 | 1.00 / 1.00 |
| scatter (16 rows) | 37.8 / 58.9 | 37.6 / 57.7 | 0.99 / 0.98 |

### payload 512 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 26.9 / 69.0 | 27.9 / 68.7 | 1.04 / 1.00 |
| mul_add | 25.8 / 57.4 | 26.2 / 58.0 | 1.01 / 1.01 |
| mul_assign | 41.8 / 85.7 | 38.4 / 80.7 | 0.92 / 0.94 |
| scatter (4 rows) | 36.1 / 65.3 | 34.3 / 65.2 | 0.95 / 1.00 |
| scatter (16 rows) | 43.3 / 82.4 | 41.0 / 80.8 | 0.95 / 0.98 |

### payload 1152 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 53.9 / 83.5 | 53.9 / 83.3 | 1.00 / 1.00 |
| mul_add | 53.2 / 72.7 | 53.1 / 72.6 | 1.00 / 1.00 |
| mul_assign | 81.5 / 137.9 | 83.9 / 133.1 | 1.03 / 0.97 |
| scatter (4 rows) | 63.7 / 95.0 | 63.5 / 95.2 | 1.00 / 1.00 |
| scatter (16 rows) | 71.4 / 107.7 | 71.5 / 107.7 | 1.00 / 1.00 |

### payload 1168 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 59.8 / 98.0 | 59.7 / 98.0 | 1.00 / 1.00 |
| mul_add | 53.9 / 78.0 | 53.8 / 80.2 | 1.00 / 1.03 |
| mul_assign | 82.7 / 136.5 | 82.9 / 133.9 | 1.00 / 0.98 |
| scatter (4 rows) | 57.3 / 89.3 | 57.0 / 89.5 | 0.99 / 1.00 |
| scatter (16 rows) | 61.1 / 99.5 | 61.8 / 99.5 | 1.01 / 1.00 |

### payload 1184 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 65.1 / 84.8 | 62.8 / 84.6 | 0.96 / 1.00 |
| mul_add | 57.1 / 73.8 | 55.7 / 73.7 | 0.98 / 1.00 |
| mul_assign | 90.9 / 136.8 | 90.9 / 133.2 | 1.00 / 0.97 |
| scatter (4 rows) | 64.4 / 97.1 | 63.8 / 97.1 | 0.99 / 1.00 |
| scatter (16 rows) | 68.9 / 109.5 | 68.4 / 109.5 | 0.99 / 1.00 |

### payload 1200 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 64.9 / 99.1 | 65.0 / 98.0 | 1.00 / 0.99 |
| mul_add | 61.3 / 80.2 | 61.5 / 80.2 | 1.00 / 1.00 |
| mul_assign | 87.0 / 137.0 | 81.1 / 133.4 | 0.93 / 0.97 |
| scatter (4 rows) | 54.8 / 90.4 | 55.1 / 90.2 | 1.00 / 1.00 |
| scatter (16 rows) | 61.4 / 99.4 | 60.4 / 99.6 | 0.98 / 1.00 |

### payload 1216 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 56.0 / 84.5 | 56.0 / 84.3 | 1.00 / 1.00 |
| mul_add | 54.4 / 73.2 | 54.4 / 73.5 | 1.00 / 1.00 |
| mul_assign | 86.1 / 137.3 | 86.3 / 133.2 | 1.00 / 0.97 |
| scatter (4 rows) | 65.6 / 98.2 | 65.6 / 98.3 | 1.00 / 1.00 |
| scatter (16 rows) | 74.2 / 110.5 | 74.0 / 110.6 | 1.00 / 1.00 |

### payload 1232 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 66.3 / 98.2 | 61.4 / 97.9 | 0.93 / 1.00 |
| mul_add | 54.7 / 81.0 | 54.8 / 81.0 | 1.00 / 1.00 |
| mul_assign | 89.3 / 136.5 | 85.2 / 132.6 | 0.95 / 0.97 |
| scatter (4 rows) | 58.4 / 91.7 | 58.9 / 91.6 | 1.01 / 1.00 |
| scatter (16 rows) | 64.2 / 100.2 | 63.4 / 100.2 | 0.99 / 1.00 |

### payload 1248 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 64.0 / 85.7 | 64.0 / 85.7 | 1.00 / 1.00 |
| mul_add | 58.4 / 73.8 | 57.8 / 73.9 | 0.99 / 1.00 |
| mul_assign | 93.2 / 138.3 | 89.0 / 135.2 | 0.95 / 0.98 |
| scatter (4 rows) | 64.5 / 99.6 | 64.7 / 99.6 | 1.00 / 1.00 |
| scatter (16 rows) | 68.3 / 111.5 | 68.6 / 111.4 | 1.01 / 1.00 |

### payload 1400 B

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor | 60.1 / 101.0 | 60.6 / 100.8 | 1.01 / 1.00 |
| mul_add | 47.0 / 69.8 | 47.6 / 69.2 | 1.01 / 0.99 |
| mul_assign | 67.1 / 123.1 | 70.6 / 121.3 | 1.05 / 0.99 |
| scatter (4 rows) | 44.2 / 61.3 | 40.4 / 61.3 | 0.91 / 1.00 |
| scatter (16 rows) | 45.4 / 68.0 | 45.4 / 67.8 | 1.00 / 1.00 |

### buffer 4 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor gf8 | 51.3 / 83.1 | 49.0 / 82.8 | 0.96 / 1.00 |
| mul_add gf8 | 64.0 / 84.4 | 66.2 / 84.2 | 1.03 / 1.00 |
| mul_into gf8 | 81.4 / 163.8 | 79.6 / 163.6 | 0.98 / 1.00 |
| mul_assign gf8 | 125.3 / 177.2 | 124.4 / 177.2 | 0.99 / 1.00 |
| elementwise gf8 | 54.8 / 66.5 | 55.6 / 66.8 | 1.02 / 1.00 |
| elementwise_assign gf8 | 59.5 / 67.2 | 57.4 / 67.5 | 0.96 / 1.01 |
| add_assign_scalar gf8 | 63.1 / 113.1 | 63.3 / 113.1 | 1.00 / 1.00 |
| sub_assign_scalar gf8 | 63.2 / 113.2 | 63.3 / 113.2 | 1.00 / 1.00 |

### buffer 256 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor gf8 | 32.0 / 63.7 | 32.0 / 63.7 | 1.00 / 1.00 |
| mul_add gf8 | 38.0 / 58.5 | 37.9 / 58.5 | 1.00 / 1.00 |
| mul_into gf8 | 38.2 / 64.8 | 38.2 / 64.8 | 1.00 / 1.00 |
| mul_assign gf8 | 51.5 / 76.7 | 51.5 / 76.7 | 1.00 / 1.00 |
| elementwise gf8 | 24.2 / 40.2 | 23.5 / 42.0 | 0.97 / 1.04 |
| elementwise_assign gf8 | 28.1 / 63.3 | 27.0 / 63.2 | 0.96 / 1.00 |
| add_assign_scalar gf8 | 51.9 / 76.6 | 51.9 / 76.6 | 1.00 / 1.00 |
| sub_assign_scalar gf8 | 51.9 / 76.6 | 51.9 / 76.6 | 1.00 / 1.00 |

### buffer 8 MiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| xor gf8 | 7.76 / 14.78 | 7.65 / 14.69 | 0.99 / 0.99 |
| mul_add gf8 | 8.50 / 20.12 | 8.62 / 20.14 | 1.01 / 1.00 |
| mul_into gf8 | 5.23 / 33.31 | 5.24 / 33.31 | 1.00 / 1.00 |
| mul_assign gf8 | 30.61 / 29.11 | 30.72 / 29.10 | 1.00 / 1.00 |
| elementwise gf8 | 5.37 / 7.41 | 5.29 / 7.40 | 0.99 / 1.00 |
| elementwise_assign gf8 | 6.97 / 7.79 | 6.90 / 14.07 | 0.99 / 1.81 |
| add_assign_scalar gf8 | 29.14 / 28.94 | 29.19 / 28.84 | 1.00 / 1.00 |
| sub_assign_scalar gf8 | 29.15 / 29.10 | 29.18 / 29.11 | 1.00 / 1.00 |

### scatter/matrix — 2 rows x 64 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| scatter gf8 | 45.5 / 69.8 | 45.5 / 67.1 | 1.00 / 0.96 |
| scatter prepared gf8 | 45.5 / 69.8 | 45.5 / 67.1 | 1.00 / 0.96 |
| scatter (unblocked) gf8 | 37.4 / 58.0 | 37.4 / 58.0 | 1.00 / 1.00 |
| matrix (selected) gf8 | 90.1 / 109.9 | 90.9 / 109.6 | 1.01 / 1.00 |
| matrix (unblocked AXPY) gf8 | 38.2 / 62.0 | 38.1 / 60.7 | 1.00 / 0.98 |
| matrix prepared gf8 | 90.4 / 109.8 | 91.3 / 109.9 | 1.01 / 1.00 |
| gather (selected) gf8 | 44.6 / 67.4 | 44.5 / 67.3 | 1.00 / 1.00 |
| gather (unblocked) gf8 | 36.0 / 61.0 | 36.0 / 60.7 | 1.00 / 1.00 |
| gather prepared gf8 | 44.5 / 66.1 | 44.5 / 66.0 | 1.00 / 1.00 |

### scatter/matrix — 4 rows x 64 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| scatter gf8 | 48.4 / 72.3 | 48.4 / 72.0 | 1.00 / 1.00 |
| scatter prepared gf8 | 48.4 / 72.3 | 48.4 / 72.0 | 1.00 / 1.00 |
| scatter (unblocked) gf8 | 37.5 / 58.1 | 37.4 / 58.1 | 1.00 / 1.00 |
| matrix (selected) gf8 | 135.0 / 151.8 | 135.7 / 152.2 | 1.01 / 1.00 |
| matrix (unblocked AXPY) gf8 | 38.5 / 61.6 | 38.4 / 61.0 | 1.00 / 0.99 |
| matrix prepared gf8 | 134.6 / 153.2 | 135.3 / 153.8 | 1.01 / 1.00 |
| gather (selected) gf8 | 43.9 / 75.0 | 43.8 / 74.9 | 1.00 / 1.00 |
| gather (unblocked) gf8 | 34.2 / 60.5 | 34.2 / 60.7 | 1.00 / 1.00 |
| gather prepared gf8 | 44.0 / 76.0 | 43.9 / 75.7 | 1.00 / 1.00 |

### scatter/matrix — 8 rows x 64 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| scatter gf8 | 48.5 / 72.3 | 48.5 / 72.2 | 1.00 / 1.00 |
| scatter prepared gf8 | 48.5 / 72.3 | 48.5 / 72.2 | 1.00 / 1.00 |
| scatter (unblocked) gf8 | 37.5 / 58.1 | 37.4 / 57.4 | 1.00 / 0.99 |
| matrix (selected) gf8 | 130.3 / 151.9 | 130.5 / 152.5 | 1.00 / 1.00 |
| matrix (unblocked AXPY) gf8 | 38.4 / 59.0 | 38.4 / 59.4 | 1.00 / 1.01 |
| matrix prepared gf8 | 129.8 / 152.7 | 130.4 / 154.3 | 1.00 / 1.01 |
| gather (selected) gf8 | 45.1 / 66.6 | 45.0 / 66.5 | 1.00 / 1.00 |
| gather (unblocked) gf8 | 35.0 / 60.2 | 35.0 / 60.8 | 1.00 / 1.01 |
| gather prepared gf8 | 45.3 / 67.2 | 45.1 / 67.2 | 1.00 / 1.00 |

### scatter/matrix — 16 rows x 64 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| scatter gf8 | 45.2 / 50.2 | 46.3 / 47.5 | 1.02 / 0.94 |
| scatter prepared gf8 | 45.5 / 50.2 | 46.2 / 47.3 | 1.02 / 0.94 |
| scatter (unblocked) gf8 | 35.3 / 45.0 | 35.6 / 43.6 | 1.01 / 0.97 |
| matrix (selected) gf8 | 122.3 / 151.5 | 123.8 / 151.9 | 1.01 / 1.00 |
| matrix (unblocked AXPY) gf8 | 35.7 / 43.5 | 35.4 / 42.2 | 0.99 / 0.97 |
| matrix prepared gf8 | 122.4 / 151.3 | 124.6 / 151.6 | 1.02 / 1.00 |
| gather (selected) gf8 | 44.9 / 65.7 | 45.0 / 65.6 | 1.00 / 1.00 |
| gather (unblocked) gf8 | 36.2 / 57.0 | 37.0 / 56.9 | 1.02 / 1.00 |
| gather prepared gf8 | 44.9 / 65.7 | 45.0 / 65.3 | 1.00 / 0.99 |

### Comparison suite

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| w8d mul_add off=0 | 97.9 / 96.2 | 98.3 / 89.1 | 1.00 / 0.93 |
| w8d mul_add off=16 | 97.5 / 98.9 | 97.5 / 97.4 | 1.00 / 0.98 |
| w8d mul_add off=32 | 97.8 / 97.2 | 98.0 / 96.7 | 1.00 / 1.00 |
| w8d mul_add off=48 | 97.6 / 99.0 | 98.0 / 97.0 | 1.00 / 0.98 |
| scatter 4x16K raw | 48.8 / 73.4 | 48.8 / 73.4 | 1.00 / 1.00 |
| scatter 4x16K prepared | 48.8 / 73.9 | 48.8 / 73.8 | 1.00 / 1.00 |
| w8 mul_add (dst ^= c*src) | 40.6 / 64.7 | 40.6 / 64.7 | 1.00 / 1.00 |
| w8 mul_into (dst = c*src) | 39.6 / 64.7 | 39.6 / 64.6 | 1.00 / 1.00 |
| w8d mul_add (dst ^= c*src) | 40.6 / 64.7 | 40.6 / 64.7 | 1.00 / 1.00 |
| w8d mul_into (dst = c*src) | 39.7 / 64.7 | 39.7 / 64.7 | 1.00 / 1.00 |
| w8 assign 4096B | 123.2 / 169.8 | 123.9 / 169.8 | 1.01 / 1.00 |
| w8d assign 4096B | 123.8 / 169.1 | 125.2 / 169.8 | 1.01 / 1.00 |
| w8 assign 16384B | 132.4 / 186.1 | 136.1 / 186.8 | 1.03 / 1.00 |
| w8d assign 16384B | 132.5 / 186.1 | 136.5 / 186.7 | 1.03 / 1.00 |
| w8 assign 65536B | 51.4 / 76.3 | 51.4 / 76.4 | 1.00 / 1.00 |
| w8d assign 65536B | 51.3 / 70.4 | 51.4 / 70.5 | 1.00 / 1.00 |
| w8 assign 262144B | 51.5 / 76.6 | 51.5 / 76.6 | 1.00 / 1.00 |
| w8d assign 262144B | 51.5 / 70.9 | 51.5 / 70.9 | 1.00 / 1.00 |
| w8 assign 1048576B | 49.2 / 53.1 | 48.7 / 55.5 | 0.99 / 1.04 |
| w8d assign 1048576B | 48.3 / 51.4 | 49.4 / 52.8 | 1.02 / 1.03 |
| w8d mul_into 4MiB (dst = c*src) | 18.2 / 41.8 | 18.2 / 41.7 | 1.00 / 1.00 |
| w8d elementwise (dst = a*b) | 29.1 / 49.5 | 29.1 / 49.5 | 1.00 / 1.00 |
| w8d elementwise assign (dst *= src) | 40.8 / 56.6 | 40.8 / 56.6 | 1.00 / 1.00 |
| w8 elementwise (dst = a*b) | 29.4 / 54.3 | 29.4 / 54.3 | 1.00 / 1.00 |
| w8 elementwise assign (dst *= src) | 40.7 / 65.8 | 40.7 / 65.7 | 1.00 / 1.00 |

### Comparison suite

| Case | v2 (Mops/s) | v3 (Mops/s) | v3 / v2 |
| --- | --- | --- | --- |
| w8 scalar mul | 1427 / 2919 | 1653 / 2923 | 1.16 / 1.00 |
| w8d scalar mul | 1427 / 2918 | 1427 / 2917 | 1.00 / 1.00 |

### 4096 B x 16 sources overwrite dot product (raw)

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| fgf Gf8D mul_into_gather | 91.5 / 107.6 | 91.2 / 108.1 | 1.00 / 1.00 |
| fgf Gf8D mul_add_gather_with | 97.5 / 91.2 | 98.2 / 91.3 | 1.01 / 1.00 |

### 4096 B x 16 sources overwrite dot product

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| fgf Gf8B mul_into_gather_with | 91.6 / 87.8 | 91.1 / 87.8 | 1.00 / 1.00 |
| fgf Gf8D mul_into_gather_with | 89.0 / 106.2 | 89.4 / 108.0 | 1.00 / 1.02 |

### 16384 B x 16 sources overwrite dot product (raw)

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| fgf Gf8D mul_into_gather | 101.9 / 100.0 | 101.7 / 100.2 | 1.00 / 1.00 |
| fgf Gf8D mul_add_gather_with | 100.1 / 93.1 | 99.2 / 92.8 | 0.99 / 1.00 |

### 16384 B x 16 sources overwrite dot product

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| fgf Gf8B mul_into_gather_with | 101.7 / 89.3 | 101.8 / 89.3 | 1.00 / 1.00 |
| fgf Gf8D mul_into_gather_with | 101.2 / 99.6 | 101.0 / 100.3 | 1.00 / 1.01 |

### 65536 B x 10 sources -> 2 rows encode

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| fgf Gf8B fill + mul_add_matrix | 56.5 / 50.9 | 56.6 / 50.8 | 1.00 / 1.00 |
| fgf Gf8B mul_into_matrix | 73.0 / 67.1 | 73.0 / 66.9 | 1.00 / 1.00 |
| fgf Gf8D fill + mul_add_matrix | 56.7 / 51.9 | 56.6 / 52.1 | 1.00 / 1.00 |
| fgf Gf8D mul_into_matrix | 75.6 / 68.9 | 75.4 / 69.2 | 1.00 / 1.00 |
| fgf Gf8B mul_into_matrix_with | 75.8 / 68.1 | 75.8 / 68.3 | 1.00 / 1.00 |
| fgf Gf8B mul_add_matrix_with | 72.8 / 56.9 | 73.1 / 56.8 | 1.00 / 1.00 |
| fgf Gf8B mul_add_matrix_at | 73.3 / 57.2 | 73.2 / 57.1 | 1.00 / 1.00 |

### 65536 B x 10 sources -> 4 rows encode

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| fgf Gf8B fill + mul_add_matrix | 31.1 / 26.9 | 30.9 / 26.9 | 0.99 / 1.00 |
| fgf Gf8B mul_into_matrix | 42.2 / 35.1 | 42.0 / 35.2 | 0.99 / 1.00 |
| fgf Gf8D fill + mul_add_matrix | 31.3 / 33.4 | 31.1 / 33.5 | 0.99 / 1.00 |
| fgf Gf8D mul_into_matrix | 42.4 / 38.0 | 42.1 / 37.9 | 0.99 / 1.00 |
| fgf Gf8B mul_into_matrix_with | 42.5 / 35.4 | 42.5 / 35.3 | 1.00 / 1.00 |
| fgf Gf8B mul_add_matrix_with | 40.1 / 30.8 | 40.3 / 30.8 | 1.00 / 1.00 |
| fgf Gf8B mul_add_matrix_at | 40.2 / 30.9 | 40.3 / 30.9 | 1.00 / 1.00 |

### 65536 B x 10 sources -> 6 rows encode

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| fgf Gf8B fill + mul_add_matrix | 19.3 / 19.1 | 19.2 / 19.2 | 1.00 / 1.00 |
| fgf Gf8B mul_into_matrix | 25.8 / 25.7 | 25.7 / 25.7 | 1.00 / 1.00 |
| fgf Gf8D fill + mul_add_matrix | 19.5 / 22.7 | 19.6 / 22.7 | 1.00 / 1.00 |
| fgf Gf8D mul_into_matrix | 26.0 / 28.5 | 26.0 / 28.5 | 1.00 / 1.00 |
| fgf Gf8B mul_into_matrix_with | 26.1 / 26.3 | 26.2 / 26.2 | 1.00 / 1.00 |
| fgf Gf8B mul_add_matrix_with | 25.4 / 22.3 | 25.2 / 22.4 | 0.99 / 1.00 |
| fgf Gf8B mul_add_matrix_at | 24.5 / 22.5 | 24.6 / 22.4 | 1.00 / 1.00 |

### Gf8D peel floor sweep (row bytes, base offset mod 64)

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| scatter 4x256 off=16 | 32.4 / 40.6 | 28.7 / 40.4 | 0.89 / 1.00 |
| gather 16x256 off=16 | 59.5 / 75.0 | 53.9 / 75.1 | 0.90 / 1.00 |
| matrix 10->4x256 off=16 | 23.7 / 29.8 | 23.7 / 29.8 | 1.00 / 1.00 |
| scatter 4x512 off=16 | 38.0 / 70.2 | 38.0 / 70.3 | 1.00 / 1.00 |
| gather 16x512 off=16 | 74.8 / 87.9 | 74.5 / 87.7 | 1.00 / 1.00 |
| matrix 10->4x512 off=16 | 27.3 / 37.4 | 27.2 / 37.6 | 0.99 / 1.01 |
| scatter 4x1024 off=16 | 61.2 / 102.7 | 61.5 / 102.8 | 1.01 / 1.00 |
| gather 16x1024 off=16 | 89.2 / 94.0 | 84.6 / 93.9 | 0.95 / 1.00 |
| matrix 10->4x1024 off=16 | 32.8 / 42.8 | 32.5 / 42.7 | 0.99 / 1.00 |
| scatter 4x2048 off=16 | 88.5 / 128.6 | 88.0 / 128.8 | 0.99 / 1.00 |
| gather 16x2048 off=16 | 93.6 / 101.1 | 94.0 / 100.5 | 1.00 / 0.99 |
| matrix 10->4x2048 off=16 | 36.0 / 44.8 | 36.2 / 44.7 | 1.01 / 1.00 |
| scatter 4x3072 off=16 | 92.3 / 117.6 | 91.4 / 117.8 | 0.99 / 1.00 |
| gather 16x3072 off=16 | 60.9 / 78.0 | 61.7 / 78.3 | 1.01 / 1.00 |
| matrix 10->4x3072 off=16 | 36.4 / 44.9 | 36.6 / 44.3 | 1.01 / 0.99 |
| scatter 4x4096 off=16 | 84.5 / 110.7 | 84.1 / 110.7 | 0.99 / 1.00 |
| gather 16x4096 off=16 | 65.0 / 74.7 | 64.7 / 74.6 | 1.00 / 1.00 |
| matrix 10->4x4096 off=16 | 33.4 / 41.9 | 33.8 / 41.7 | 1.01 / 1.00 |
| scatter 4x8192 off=16 | 89.2 / 116.1 | 89.0 / 116.3 | 1.00 / 1.00 |
| gather 16x8192 off=16 | 76.7 / 75.2 | 76.8 / 75.0 | 1.00 / 1.00 |
| matrix 10->4x8192 off=16 | 36.8 / 44.3 | 36.7 / 44.4 | 1.00 / 1.00 |
| scatter 4x16384 off=16 | 48.4 / 73.5 | 48.4 / 73.7 | 1.00 / 1.00 |
| gather 16x16384 off=16 | 84.7 / 75.7 | 82.6 / 75.5 | 0.98 / 1.00 |
| matrix 10->4x16384 off=16 | 39.7 / 45.2 | 39.8 / 45.1 | 1.00 / 1.00 |
| scatter 4x256 off=0 | 33.0 / 49.6 | 32.8 / 49.6 | 0.99 / 1.00 |
| gather 16x256 off=0 | 76.2 / 87.3 | 76.4 / 85.9 | 1.00 / 0.98 |
| matrix 10->4x256 off=0 | 25.8 / 30.1 | 25.9 / 30.2 | 1.01 / 1.01 |
| scatter 4x512 off=0 | 54.7 / 83.6 | 54.6 / 83.4 | 1.00 / 1.00 |
| gather 16x512 off=0 | 101.8 / 107.5 | 101.9 / 107.7 | 1.00 / 1.00 |
| matrix 10->4x512 off=0 | 29.1 / 37.6 | 29.3 / 37.6 | 1.01 / 1.00 |
| scatter 4x1024 off=0 | 81.2 / 115.0 | 81.2 / 115.0 | 1.00 / 1.00 |
| gather 16x1024 off=0 | 111.2 / 117.0 | 111.8 / 116.4 | 1.01 / 0.99 |
| matrix 10->4x1024 off=0 | 35.1 / 42.9 | 35.3 / 43.1 | 1.01 / 1.00 |
| scatter 4x2048 off=0 | 95.6 / 136.0 | 94.6 / 136.0 | 0.99 / 1.00 |
| gather 16x2048 off=0 | 125.2 / 122.1 | 124.8 / 121.9 | 1.00 / 1.00 |
| matrix 10->4x2048 off=0 | 39.5 / 45.5 | 39.5 / 45.6 | 1.00 / 1.00 |
| scatter 4x3072 off=0 | 99.1 / 122.1 | 98.2 / 120.6 | 0.99 / 0.99 |
| gather 16x3072 off=0 | 98.6 / 97.7 | 99.7 / 98.3 | 1.01 / 1.01 |
| matrix 10->4x3072 off=0 | 40.9 / 46.6 | 40.8 / 46.5 | 1.00 / 1.00 |
| scatter 4x4096 off=0 | 91.2 / 113.0 | 90.9 / 112.7 | 1.00 / 1.00 |
| gather 16x4096 off=0 | 89.5 / 92.7 | 89.2 / 92.6 | 1.00 / 1.00 |
| matrix 10->4x4096 off=0 | 40.5 / 45.4 | 40.8 / 45.5 | 1.01 / 1.00 |
| scatter 4x8192 off=0 | 93.5 / 120.0 | 93.4 / 120.1 | 1.00 / 1.00 |
| gather 16x8192 off=0 | 93.7 / 93.4 | 93.3 / 93.3 | 1.00 / 1.00 |
| matrix 10->4x8192 off=0 | 41.6 / 47.4 | 41.6 / 47.5 | 1.00 / 1.00 |
| scatter 4x16384 off=0 | 48.9 / 73.7 | 48.9 / 73.7 | 1.00 / 1.00 |
| gather 16x16384 off=0 | 94.1 / 93.9 | 92.3 / 93.9 | 0.98 / 1.00 |
| matrix 10->4x16384 off=0 | 42.4 / 48.4 | 42.5 / 48.3 | 1.00 / 1.00 |

## Caveats

- v3 makes every prime-field kernel output canonical for any accepted input lane; v2 assumed canonical inputs on its multiply paths. The Mersenne31 and `QuadMersenne31` elementwise rows carry that contract.
- Rows whose code is instruction-identical between the two revisions can still differ by a few percent from binary layout; the same-session protocol bounds, but does not remove, that effect.
- Case labels are the benchmark harness's own labels; field suffixes (`gf8`, `gf16`, `m31`, `qm31`, `gld`) name the measured field.
