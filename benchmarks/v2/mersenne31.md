# Mersenne fields: `m31`, `qm31` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/mersenne31.md); the other snapshot at [v3](../v3/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `v3_gfni_crypto` |
| Resolved field backends | `m31`: `v4x` / `v3`; `qm31`: `v4x` / `v3` |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |
| Competitors | Willow Cove (i5-1135G7): Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16}; Golden Cove (i7-12700K): Plonky3 0.7.0; native packing widths {'m31': 8, 'gld': 4, 'qm31': 8} |

## Self-timings

### Packed operations

`m31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 29.5 / 33.4 | 28.4 / 34.4 | 7.90 / 19.8 |
| add_assign_scalar | 40.5 / 50.5 | 41.3 / 50.1 | 24.7 / 20.9 |
| mul_elementwise | 23.1 / 31.9 | 20.3 / 28.8 | 5.45 / 17.1 |
| mul_elementwise_assign | 25.1 / 31.7 | 23.0 / 32.7 | 7.92 / 21.3 |
| sub_assign_scalar | 32.6 / 39.6 | 33.9 / 41.7 | 22.8 / 23.9 |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 20.0 / 32.0 | 18.4 / 31.6 | 7.88 / 22.4 |
| add_assign_scalar | 29.7 / 49.6 | 26.0 / 48.5 | 19.1 / 25.8 |
| mul_elementwise | 11.6 / 13.9 | 11.8 / 13.6 | 5.28 / 13.6 |
| mul_elementwise_assign | 7.26 / 14.5 | 7.39 / 14.3 | 6.61 / 13.1 |
| sub_assign_scalar | 23.6 / 41.0 | 21.2 / 41.3 | 16.7 / 31.7 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 24.7 / 32.3 |
| 1 KiB | 16 | 30.5 / 35.7 |
| 1 KiB | 2 | 26.3 / 35.1 |
| 1 KiB | 32 | 29.9 / 33.8 |
| 1 KiB | 4 | 28.9 / 33.1 |
| 1 KiB | 8 | 29.9 / 34.9 |
| 64 B | 1 | 5.57 / 11.6 |
| 64 B | 16 | 24.7 / 32.3 |
| 64 B | 2 | 6.36 / 10.5 |
| 64 B | 32 | 26.3 / 35.1 |
| 64 B | 4 | 17.9 / 24.5 |
| 64 B | 8 | 26.2 / 31.4 |
| 64 KiB | 1 | 28.2 / 34.2 |
| 64 KiB | 16 | 18.7 / 21.6 |
| 64 KiB | 2 | 29.6 / 36.9 |
| 64 KiB | 32 | 17.9 / 20.7 |
| 64 KiB | 4 | 26.1 / 35.3 |
| 64 KiB | 8 | 25.5 / 31.5 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 26.4 / 33.1 |
| add_assign per row | 1 KiB | 16 | 28.2 / 31.5 |
| add_assign per row | 1 KiB | 2 | 28.6 / 33.9 |
| add_assign per row | 1 KiB | 32 | 27.9 / 30.3 |
| add_assign per row | 1 KiB | 4 | 28.9 / 31.6 |
| add_assign per row | 1 KiB | 8 | 29.2 / 31.7 |
| add_assign per row | 64 B | 1 | 6.14 / 11.2 |
| add_assign per row | 64 B | 16 | 8.70 / 15.1 |
| add_assign per row | 64 B | 2 | 6.25 / 9.70 |
| add_assign per row | 64 B | 32 | 8.55 / 14.7 |
| add_assign per row | 64 B | 4 | 8.29 / 14.1 |
| add_assign per row | 64 B | 8 | 8.49 / 14.8 |
| add_assign per row | 64 KiB | 1 | 28.2 / 33.0 |
| add_assign per row | 64 KiB | 16 | 18.6 / 21.7 |
| add_assign per row | 64 KiB | 2 | 29.6 / 36.9 |
| add_assign per row | 64 KiB | 32 | 17.9 / 20.6 |
| add_assign per row | 64 KiB | 4 | 26.0 / 35.1 |
| add_assign per row | 64 KiB | 8 | 25.3 / 31.8 |
| add_assign | 1 KiB | 1 | 26.5 / 33.2 |
| add_assign | 1 KiB | 16 | 30.6 / 35.9 |
| add_assign | 1 KiB | 2 | 27.7 / 35.3 |
| add_assign | 1 KiB | 32 | 29.8 / 33.9 |
| add_assign | 1 KiB | 4 | 29.7 / 33.5 |
| add_assign | 1 KiB | 8 | 30.3 / 35.1 |
| add_assign | 64 B | 1 | 7.11 / 11.6 |
| add_assign | 64 B | 16 | 26.5 / 32.7 |
| add_assign | 64 B | 2 | 6.31 / 10.5 |
| add_assign | 64 B | 32 | 27.8 / 35.5 |
| add_assign | 64 B | 4 | 21.0 / 25.5 |
| add_assign | 64 B | 8 | 30.2 / 32.3 |
| add_assign | 64 KiB | 1 | 28.2 / 32.8 |
| add_assign | 64 KiB | 16 | 18.7 / 21.7 |
| add_assign | 64 KiB | 2 | 29.7 / 36.9 |
| add_assign | 64 KiB | 32 | 17.9 / 20.7 |
| add_assign | 64 KiB | 4 | 26.1 / 35.3 |
| add_assign | 64 KiB | 8 | 25.5 / 31.8 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 3.742 / 1.714 | 2.570 / 1.159 |
| m31 | scalar a * b | 4.219 / 1.620 | 2.105 / 0.837 |
| qm31 | scalar a * b | 9.004 / 4.578 | 6.975 / 3.039 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.3 | 22.5 / 19.6 |
| gld | dst += v | 97.4 / 78.4 | 103 / 141 |
| gld | dst -= v | 103 / 77.6 | 103 / 142 |
| gld | dst = a * b | 23.8 / 22.3 | 26.9 / 25.4 |
| gld | dst = c * src | 25.1 / 22.3 | 27.4 / 26.0 |
| m31 | dst += c * src | 55.6 / 58.7 | 49.8 / 54.7 |
| m31 | dst += v | 87.5 / 103 | 103 / 147 |
| m31 | dst -= v | 70.3 / 82.9 | 103 / 147 |
| m31 | dst = a * b | 59.6 / 66.2 | 59.9 / 66.5 |
| m31 | dst = c * src | 73.3 / 73.7 | 65.4 / 69.9 |
| qm31 | dst += c * src | 22.9 / 27.2 | 27.0 / 30.5 |
| qm31 | dst += v | 62.6 / 103 | 103 / 147 |
| qm31 | dst -= v | 49.9 / 82.9 | 103 / 147 |
| qm31 | dst = a * b | 27.0 / 29.8 | 24.1 / 25.1 |
| qm31 | dst = c * src | 32.1 / 36.1 | 30.5 / 34.6 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
