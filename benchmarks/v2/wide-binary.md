# Wider binary fields: v2 versus v3

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

### Tier 3 field cost — 262144 bytes

| Case | v2 (GiB/s) | v3 (GiB/s) | v3 / v2 |
| --- | --- | --- | --- |
| mul_add polynomial tower gf32 | 15.59 / 30.67 | 15.69 / 30.72 | 1.01 / 1.00 |
| mul_add polynomial tower gf64 | 8.66 / 16.77 | 8.69 / 16.59 | 1.00 / 0.99 |
| mul_add canonical Fan-Paar fp16 | 9.50 / 17.08 | 9.70 / 17.06 | 1.02 / 1.00 |
| mul_add canonical Fan-Paar fp32 | 3.30 / 6.30 | 3.32 / 6.30 | 1.01 / 1.00 |
| mul_add canonical Fan-Paar fp64 | 1.25 / 1.92 | 1.25 / 2.21 | 1.00 / 1.15 |

## Caveats

- v3 makes every prime-field kernel output canonical for any accepted input lane; v2 assumed canonical inputs on its multiply paths. The Mersenne31 and `QuadMersenne31` elementwise rows carry that contract.
- Rows whose code is instruction-identical between the two revisions can still differ by a few percent from binary layout; the same-session protocol bounds, but does not remove, that effect.
- Case labels are the benchmark harness's own labels; field suffixes (`gf8`, `gf16`, `m31`, `qm31`, `gld`) name the measured field.
