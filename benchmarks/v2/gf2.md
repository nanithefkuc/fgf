# Bit-packed GF(2): v2 versus v3

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

### 4 KiB buffers (32768 elements)

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| bits::xor_assign packed | 43.1 / 103.1 | 42.8 / 103.0 | 0.99 / 1.00 |
| bits::and_into packed | 22.6 / 66.3 | 22.4 / 66.8 | 0.99 / 1.01 |
| bits::weight packed | 116.3 / 14.7 | 106.0 / 14.7 | 0.91 / 1.00 |
| bits::dot_product packed | 20.3 / 35.3 | 18.2 / 38.3 | 0.90 / 1.09 |
| bits::xor_range 5/8 packed | 53.6 / 77.3 | 54.0 / 77.0 | 1.01 / 1.00 |
| bits::XorRange 5/8 packed | 58.0 / 86.8 | 58.0 / 86.6 | 1.00 / 1.00 |
| ops::add_assign gf8 control | 43.1 / 103.1 | 42.9 / 103.0 | 1.00 / 1.00 |

### 256 KiB buffers (2097152 elements)

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| bits::xor_assign packed | 31.9 / 66.1 | 32.0 / 66.0 | 1.00 / 1.00 |
| bits::and_into packed | 18.6 / 44.9 | 18.6 / 47.2 | 1.00 / 1.05 |
| bits::weight packed | 67.4 / 15.0 | 67.8 / 15.0 | 1.01 / 1.00 |
| bits::dot_product packed | 19.1 / 36.9 | 18.2 / 43.4 | 0.95 / 1.18 |
| bits::xor_range 5/8 packed | 51.0 / 73.4 | 51.0 / 73.5 | 1.00 / 1.00 |
| bits::XorRange 5/8 packed | 51.0 / 73.5 | 51.0 / 73.5 | 1.00 / 1.00 |
| ops::add_assign gf8 control | 31.9 / 66.0 | 32.0 / 66.0 | 1.00 / 1.00 |

### 8 MiB buffers (67108864 elements)

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| bits::xor_assign packed | 8.03 / 15.18 | 8.23 / 15.27 | 1.02 / 1.01 |
| bits::and_into packed | 4.20 / 8.05 | 4.23 / 7.99 | 1.01 / 0.99 |
| bits::weight packed | 34.37 / 15.01 | 34.03 / 15.01 | 0.99 / 1.00 |
| bits::dot_product packed | 10.99 / 34.01 | 10.81 / 34.17 | 0.98 / 1.00 |
| bits::xor_range 5/8 packed | 16.27 / 19.53 | 16.26 / 19.00 | 1.00 / 0.97 |
| bits::XorRange 5/8 packed | 16.33 / 19.53 | 16.28 / 19.01 | 1.00 / 0.97 |
| ops::add_assign gf8 control | 7.85 / 15.14 | 7.98 / 15.27 | 1.02 / 1.01 |

### 64 MiB buffers (536870912 elements)

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| bits::xor_assign packed | 6.12 / 14.89 | 6.13 / 14.73 | 1.00 / 0.99 |
| bits::and_into packed | 4.65 / 10.95 | 4.64 / 10.91 | 1.00 / 1.00 |
| bits::weight packed | 17.85 / 14.45 | 17.98 / 14.45 | 1.01 / 1.00 |
| bits::dot_product packed | 8.82 / 18.91 | 8.73 / 19.41 | 0.99 / 1.03 |
| bits::xor_range 5/8 packed | 9.76 / 24.14 | 9.76 / 23.99 | 1.00 / 0.99 |
| bits::XorRange 5/8 packed | 9.84 / 24.52 | 9.85 / 24.25 | 1.00 / 0.99 |
| ops::add_assign gf8 control | 6.04 / 14.61 | 6.07 / 14.48 | 1.00 / 0.99 |

### 256 row pairs, 16-byte rows, bits 61..69

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| bits::xor_range 8-bit row | 1.25 / 2.78 | 1.28 / 2.76 | 1.02 / 0.99 |
| bits::XorRange 8-bit row | 3.60 / 7.18 | 3.58 / 7.18 | 0.99 / 1.00 |

## Caveats

- v3 makes every prime-field kernel output canonical for any accepted input lane; v2 assumed canonical inputs on its multiply paths. The Mersenne31 and `QuadMersenne31` elementwise rows carry that contract.
- Rows whose code is instruction-identical between the two revisions can still differ by a few percent from binary layout; the same-session protocol bounds, but does not remove, that effect.
- Case labels are the benchmark harness's own labels; field suffixes (`gf8`, `gf16`, `m31`, `qm31`, `gld`) name the measured field.
