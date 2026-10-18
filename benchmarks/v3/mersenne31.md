# Mersenne fields: `m31`, `qm31` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/mersenne31.md); the other snapshot at [v2](../v2/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| Revision | `388cbb4a43d5d20e8a663e89c3b44380c7c4f8b7` |
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
| add_assign | 29.6 / 33.8 | 28.2 / 34.3 | 7.87 / 20.5 |
| add_assign_scalar | 45.2 / 54.8 | 44.7 / 52.5 | 25.1 / 21.0 |
| mul_elementwise | 24.1 / 28.6 | 20.3 / 28.7 | 5.45 / 18.9 |
| mul_elementwise_assign | 24.9 / 25.8 | 24.9 / 27.1 | 8.29 / 22.9 |
| sub_assign_scalar | 45.8 / 54.8 | 44.9 / 52.5 | 25.0 / 21.1 |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 19.0 / 32.2 | 18.4 / 31.6 | 7.75 / 24.2 |
| add_assign_scalar | 33.1 / 53.4 | 28.4 / 52.2 | 20.1 / 24.5 |
| mul_elementwise | 14.9 / 16.9 | 14.0 / 17.1 | 5.32 / 16.2 |
| mul_elementwise_assign | 7.96 / 17.2 | 8.14 / 17.5 | 7.18 / 17.3 |
| sub_assign_scalar | 33.5 / 53.5 | 28.2 / 50.8 | 20.0 / 24.5 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 25.7 / 32.4 |
| 1 KiB | 16 | 30.4 / 35.8 |
| 1 KiB | 2 | 26.3 / 35.0 |
| 1 KiB | 32 | 29.7 / 34.2 |
| 1 KiB | 4 | 28.9 / 33.4 |
| 1 KiB | 8 | 29.8 / 34.9 |
| 64 B | 1 | 5.52 / 11.6 |
| 64 B | 16 | 25.6 / 32.3 |
| 64 B | 2 | 9.80 / 18.5 |
| 64 B | 32 | 26.3 / 35.1 |
| 64 B | 4 | 16.9 / 24.4 |
| 64 B | 8 | 26.4 / 29.0 |
| 64 KiB | 1 | 27.9 / 34.3 |
| 64 KiB | 16 | 18.6 / 21.7 |
| 64 KiB | 2 | 29.6 / 36.9 |
| 64 KiB | 32 | 18.1 / 20.7 |
| 64 KiB | 4 | 26.1 / 35.1 |
| 64 KiB | 8 | 25.8 / 31.9 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 26.6 / 33.0 |
| add_assign per row | 1 KiB | 16 | 29.0 / 31.4 |
| add_assign per row | 1 KiB | 2 | 28.6 / 33.9 |
| add_assign per row | 1 KiB | 32 | 27.9 / 30.6 |
| add_assign per row | 1 KiB | 4 | 29.0 / 31.7 |
| add_assign per row | 1 KiB | 8 | 28.9 / 31.7 |
| add_assign per row | 64 B | 1 | 6.13 / 11.2 |
| add_assign per row | 64 B | 16 | 8.29 / 15.1 |
| add_assign per row | 64 B | 2 | 7.42 / 13.0 |
| add_assign per row | 64 B | 32 | 8.47 / 14.7 |
| add_assign per row | 64 B | 4 | 8.10 / 14.1 |
| add_assign per row | 64 B | 8 | 8.80 / 14.8 |
| add_assign per row | 64 KiB | 1 | 28.0 / 34.4 |
| add_assign per row | 64 KiB | 16 | 18.5 / 21.6 |
| add_assign per row | 64 KiB | 2 | 29.5 / 36.8 |
| add_assign per row | 64 KiB | 32 | 18.0 / 20.6 |
| add_assign per row | 64 KiB | 4 | 26.0 / 34.9 |
| add_assign per row | 64 KiB | 8 | 25.7 / 31.7 |
| add_assign | 1 KiB | 1 | 30.1 / 33.0 |
| add_assign | 1 KiB | 16 | 30.6 / 35.9 |
| add_assign | 1 KiB | 2 | 27.2 / 35.3 |
| add_assign | 1 KiB | 32 | 29.9 / 34.2 |
| add_assign | 1 KiB | 4 | 29.3 / 33.9 |
| add_assign | 1 KiB | 8 | 30.0 / 35.2 |
| add_assign | 64 B | 1 | 7.11 / 12.7 |
| add_assign | 64 B | 16 | 30.1 / 33.1 |
| add_assign | 64 B | 2 | 13.2 / 19.2 |
| add_assign | 64 B | 32 | 27.3 / 35.2 |
| add_assign | 64 B | 4 | 20.1 / 26.1 |
| add_assign | 64 B | 8 | 30.3 / 30.0 |
| add_assign | 64 KiB | 1 | 28.0 / 34.4 |
| add_assign | 64 KiB | 16 | 18.6 / 21.7 |
| add_assign | 64 KiB | 2 | 29.6 / 37.0 |
| add_assign | 64 KiB | 32 | 18.1 / 20.7 |
| add_assign | 64 KiB | 4 | 26.1 / 35.1 |
| add_assign | 64 KiB | 8 | 25.8 / 31.7 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 3.741 / 1.702 | 2.570 / 1.152 |
| m31 | scalar a * b | 4.181 / 1.611 | 2.088 / 0.880 |
| qm31 | scalar a * b | 4.507 / 2.406 | 6.908 / 3.035 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.3 | 22.5 / 19.8 |
| gld | dst += v | 97.5 / 78.3 | 103 / 142 |
| gld | dst -= v | 103 / 77.7 | 103 / 142 |
| gld | dst = a * b | 23.8 / 22.4 | 26.9 / 25.3 |
| gld | dst = c * src | 25.1 / 22.8 | 27.5 / 26.0 |
| m31 | dst += c * src | 48.4 / 48.5 | 49.8 / 54.7 |
| m31 | dst += v | 98.4 / 117 | 103 / 147 |
| m31 | dst -= v | 98.4 / 116 | 103 / 147 |
| m31 | dst = a * b | 55.7 / 58.6 | 59.8 / 66.6 |
| m31 | dst = c * src | 70.5 / 66.2 | 65.4 / 69.8 |
| qm31 | dst += c * src | 26.8 / 28.8 | 27.0 / 30.5 |
| qm31 | dst += v | 68.2 / 117 | 103 / 147 |
| qm31 | dst -= v | 67.8 / 117 | 103 / 147 |
| qm31 | dst = a * b | 34.8 / 36.3 | 24.1 / 25.1 |
| qm31 | dst = c * src | 37.6 / 42.1 | 30.5 / 34.7 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
