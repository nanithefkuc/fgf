# Goldilocks: v2 versus v3

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

### buffer 4 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| add_assign gld | 29.6 / 22.4 | 29.6 / 22.4 | 1.00 / 1.00 |
| add_assign_scalar gld | 44.5 / 39.3 | 44.8 / 41.0 | 1.01 / 1.04 |
| sub_assign_scalar gld | 59.9 / 40.6 | 59.8 / 39.2 | 1.00 / 0.97 |
| elementwise gld | 11.0 / 11.1 | 11.0 / 11.1 | 1.00 / 1.00 |
| elementwise_assign gld | 11.2 / 11.2 | 11.2 / 11.2 | 1.00 / 1.00 |

### buffer 256 KiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| add_assign gld | 28.47 / 22.51 | 28.41 / 22.54 | 1.00 / 1.00 |
| add_assign_scalar gld | 49.20 / 41.46 | 49.18 / 41.51 | 1.00 / 1.00 |
| sub_assign_scalar gld | 51.76 / 41.08 | 51.76 / 41.03 | 1.00 / 1.00 |
| elementwise gld | 9.66 / 10.61 | 9.39 / 10.62 | 0.97 / 1.00 |
| elementwise_assign gld | 10.45 / 11.16 | 10.13 / 10.99 | 0.97 / 0.98 |

### buffer 8 MiB

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| add_assign gld | 5.43 / 5.14 | 5.42 / 10.18 | 1.00 / 1.98 |
| add_assign_scalar gld | 27.46 / 35.66 | 27.47 / 35.64 | 1.00 / 1.00 |
| sub_assign_scalar gld | 28.74 / 35.76 | 28.73 / 35.74 | 1.00 / 1.00 |
| elementwise gld | 3.54 / 3.48 | 3.57 / 3.50 | 1.01 / 1.01 |
| elementwise_assign gld | 3.82 / 3.57 | 3.82 / 6.37 | 1.00 / 1.78 |

## Caveats

- v3 makes every prime-field kernel output canonical for any accepted input lane; v2 assumed canonical inputs on its multiply paths. The Mersenne31 and `QuadMersenne31` elementwise rows carry that contract.
- Rows whose code is instruction-identical between the two revisions can still differ by a few percent from binary layout; the same-session protocol bounds, but does not remove, that effect.
- Case labels are the benchmark harness's own labels; field suffixes (`gf8`, `gf16`, `m31`, `qm31`, `gld`) name the measured field.
