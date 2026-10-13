# Mersenne fields: `m31`, `qm31` (v2)

Paired-run snapshot for v2. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/mersenne31.md); the other snapshot at [v3](../v3/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| Revision | `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Toolchain | 1.98.1 |
| Resolved process backend | `v4x` / `-` |
| Resolved field backends | `m31`: `v4x` / -; `qm31`: `v4x` / - |
| Protocol | Five rounds per host; each round runs both versions in a shuffled order; competitor arms interleave inside their harness process. |
| Aggregation | Median of five per-run medians. |
| Throughput | Logical bytes divided by the aggregate latency per call; numerators are stated per panel. |
| Competitors | Willow Cove (i5-1135G7): Plonky3 0.7.0; native packing widths {'m31': 16, 'gld': 8, 'qm31': 16} |

## Self-timings

### Packed operations

`m31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 29.4 / - | 28.0 / - | 8.14 / - |
| add_assign_scalar | 40.4 / - | 41.2 / - | 24.7 / - |
| mul_elementwise | 23.0 / - | 20.2 / - | 5.47 / - |
| mul_elementwise_assign | 25.1 / - | 22.7 / - | 7.92 / - |
| sub_assign_scalar | 32.7 / - | 34.1 / - | 23.0 / - |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 19.0 / - | 18.3 / - | 7.77 / - |
| add_assign_scalar | 29.8 / - | 25.8 / - | 19.1 / - |
| mul_elementwise | 11.5 / - | 11.7 / - | 5.31 / - |
| mul_elementwise_assign | 7.24 / - | 7.37 / - | 6.56 / - |
| sub_assign_scalar | 23.6 / - | 21.1 / - | 16.8 / - |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 28.3 / - |
| 1 KiB | 16 | 30.3 / - |
| 1 KiB | 2 | 26.9 / - |
| 1 KiB | 32 | 29.4 / - |
| 1 KiB | 4 | 29.1 / - |
| 1 KiB | 8 | 29.8 / - |
| 64 B | 1 | 5.57 / - |
| 64 B | 16 | 28.3 / - |
| 64 B | 2 | 10.3 / - |
| 64 B | 32 | 26.9 / - |
| 64 B | 4 | 18.0 / - |
| 64 B | 8 | 25.3 / - |
| 64 KiB | 1 | 27.9 / - |
| 64 KiB | 16 | 18.5 / - |
| 64 KiB | 2 | 29.6 / - |
| 64 KiB | 32 | 17.9 / - |
| 64 KiB | 4 | 26.1 / - |
| 64 KiB | 8 | 25.6 / - |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 28.1 / - |
| add_assign per row | 1 KiB | 16 | 29.1 / - |
| add_assign per row | 1 KiB | 2 | 30.0 / - |
| add_assign per row | 1 KiB | 32 | 28.4 / - |
| add_assign per row | 1 KiB | 4 | 29.9 / - |
| add_assign per row | 1 KiB | 8 | 30.0 / - |
| add_assign per row | 64 B | 1 | 6.14 / - |
| add_assign per row | 64 B | 16 | 8.98 / - |
| add_assign per row | 64 B | 2 | 7.42 / - |
| add_assign per row | 64 B | 32 | 8.43 / - |
| add_assign per row | 64 B | 4 | 8.19 / - |
| add_assign per row | 64 B | 8 | 8.80 / - |
| add_assign per row | 64 KiB | 1 | 27.9 / - |
| add_assign per row | 64 KiB | 16 | 18.5 / - |
| add_assign per row | 64 KiB | 2 | 29.5 / - |
| add_assign per row | 64 KiB | 32 | 18.0 / - |
| add_assign per row | 64 KiB | 4 | 26.0 / - |
| add_assign per row | 64 KiB | 8 | 25.5 / - |
| add_assign | 1 KiB | 1 | 30.3 / - |
| add_assign | 1 KiB | 16 | 30.4 / - |
| add_assign | 1 KiB | 2 | 27.8 / - |
| add_assign | 1 KiB | 32 | 29.5 / - |
| add_assign | 1 KiB | 4 | 29.6 / - |
| add_assign | 1 KiB | 8 | 30.2 / - |
| add_assign | 64 B | 1 | 7.41 / - |
| add_assign | 64 B | 16 | 30.2 / - |
| add_assign | 64 B | 2 | 13.2 / - |
| add_assign | 64 B | 32 | 27.9 / - |
| add_assign | 64 B | 4 | 21.0 / - |
| add_assign | 64 B | 8 | 27.8 / - |
| add_assign | 64 KiB | 1 | 27.9 / - |
| add_assign | 64 KiB | 16 | 18.5 / - |
| add_assign | 64 KiB | 2 | 29.6 / - |
| add_assign | 64 KiB | 32 | 18.0 / - |
| add_assign | 64 KiB | 4 | 26.1 / - |
| add_assign | 64 KiB | 8 | 25.6 / - |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 3.74 / - | 2.57 / - |
| m31 | scalar a * b | 4.20 / - | 2.10 / - |
| qm31 | scalar a * b | 9.00 / - | 6.96 / - |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / - | 22.5 / - |
| gld | dst += v | 97.5 / - | 103 / - |
| gld | dst -= v | 103 / - | 103 / - |
| gld | dst = a * b | 23.8 / - | 26.9 / - |
| gld | dst = c * src | 25.1 / - | 27.4 / - |
| m31 | dst += c * src | 55.6 / - | 49.8 / - |
| m31 | dst += v | 87.6 / - | 103 / - |
| m31 | dst -= v | 70.3 / - | 103 / - |
| m31 | dst = a * b | 59.5 / - | 59.9 / - |
| m31 | dst = c * src | 73.1 / - | 65.4 / - |
| qm31 | dst += c * src | 22.9 / - | 27.0 / - |
| qm31 | dst += v | 62.5 / - | 103 / - |
| qm31 | dst -= v | 49.9 / - | 103 / - |
| qm31 | dst = a * b | 27.0 / - | 24.1 / - |
| qm31 | dst = c * src | 32.1 / - | 30.6 / - |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
