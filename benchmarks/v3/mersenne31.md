# Mersenne fields: `m31`, `qm31` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/mersenne31.md); the other snapshot at [v2](../v2/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
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
| add_assign | 29.3 / - | 28.2 / - | 8.23 / - |
| add_assign_scalar | 45.3 / - | 44.7 / - | 25.1 / - |
| mul_elementwise | 14.5 / - | 13.5 / - | 5.30 / - |
| mul_elementwise_assign | 15.2 / - | 15.0 / - | 7.61 / - |
| sub_assign_scalar | 45.7 / - | 44.9 / - | 25.2 / - |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 19.0 / - | 18.6 / - | 7.92 / - |
| add_assign_scalar | 33.5 / - | 28.4 / - | 20.0 / - |
| mul_elementwise | 10.3 / - | 10.3 / - | 5.20 / - |
| mul_elementwise_assign | 6.73 / - | 6.85 / - | 6.26 / - |
| sub_assign_scalar | 32.9 / - | 28.2 / - | 20.0 / - |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 26.0 / - |
| 1 KiB | 16 | 30.4 / - |
| 1 KiB | 2 | 26.2 / - |
| 1 KiB | 32 | 29.5 / - |
| 1 KiB | 4 | 29.3 / - |
| 1 KiB | 8 | 29.8 / - |
| 64 B | 1 | 5.77 / - |
| 64 B | 16 | 26.0 / - |
| 64 B | 2 | 10.4 / - |
| 64 B | 32 | 26.2 / - |
| 64 B | 4 | 17.9 / - |
| 64 B | 8 | 25.1 / - |
| 64 KiB | 1 | 27.9 / - |
| 64 KiB | 16 | 18.7 / - |
| 64 KiB | 2 | 29.6 / - |
| 64 KiB | 32 | 17.9 / - |
| 64 KiB | 4 | 26.1 / - |
| 64 KiB | 8 | 25.3 / - |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 27.1 / - |
| add_assign per row | 1 KiB | 16 | 28.9 / - |
| add_assign per row | 1 KiB | 2 | 28.7 / - |
| add_assign per row | 1 KiB | 32 | 28.3 / - |
| add_assign per row | 1 KiB | 4 | 28.7 / - |
| add_assign per row | 1 KiB | 8 | 29.0 / - |
| add_assign per row | 64 B | 1 | 6.14 / - |
| add_assign per row | 64 B | 16 | 8.70 / - |
| add_assign per row | 64 B | 2 | 7.42 / - |
| add_assign per row | 64 B | 32 | 8.71 / - |
| add_assign per row | 64 B | 4 | 8.29 / - |
| add_assign per row | 64 B | 8 | 8.80 / - |
| add_assign per row | 64 KiB | 1 | 28.0 / - |
| add_assign per row | 64 KiB | 16 | 18.6 / - |
| add_assign per row | 64 KiB | 2 | 29.5 / - |
| add_assign per row | 64 KiB | 32 | 17.9 / - |
| add_assign per row | 64 KiB | 4 | 26.0 / - |
| add_assign per row | 64 KiB | 8 | 25.2 / - |
| add_assign | 1 KiB | 1 | 28.4 / - |
| add_assign | 1 KiB | 16 | 30.5 / - |
| add_assign | 1 KiB | 2 | 27.3 / - |
| add_assign | 1 KiB | 32 | 29.7 / - |
| add_assign | 1 KiB | 4 | 29.3 / - |
| add_assign | 1 KiB | 8 | 30.1 / - |
| add_assign | 64 B | 1 | 7.41 / - |
| add_assign | 64 B | 16 | 28.4 / - |
| add_assign | 64 B | 2 | 12.3 / - |
| add_assign | 64 B | 32 | 27.3 / - |
| add_assign | 64 B | 4 | 21.6 / - |
| add_assign | 64 B | 8 | 29.9 / - |
| add_assign | 64 KiB | 1 | 28.1 / - |
| add_assign | 64 KiB | 16 | 18.7 / - |
| add_assign | 64 KiB | 2 | 29.7 / - |
| add_assign | 64 KiB | 32 | 17.9 / - |
| add_assign | 64 KiB | 4 | 26.1 / - |
| add_assign | 64 KiB | 8 | 25.3 / - |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 4.48 / - | 2.56 / - |
| m31 | scalar a * b | 4.18 / - | 2.09 / - |
| qm31 | scalar a * b | 16.1 / - | 7.05 / - |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / - | 22.5 / - |
| gld | dst += v | 97.5 / - | 103 / - |
| gld | dst -= v | 103 / - | 103 / - |
| gld | dst = a * b | 23.8 / - | 26.9 / - |
| gld | dst = c * src | 25.1 / - | 27.5 / - |
| m31 | dst += c * src | 32.9 / - | 49.8 / - |
| m31 | dst += v | 98.5 / - | 103 / - |
| m31 | dst -= v | 98.5 / - | 103 / - |
| m31 | dst = a * b | 34.8 / - | 59.9 / - |
| m31 | dst = c * src | 50.4 / - | 65.4 / - |
| qm31 | dst += c * src | 21.4 / - | 27.0 / - |
| qm31 | dst += v | 68.1 / - | 103 / - |
| qm31 | dst -= v | 67.7 / - | 103 / - |
| qm31 | dst = a * b | 23.9 / - | 24.1 / - |
| qm31 | dst = c * src | 29.2 / - | 30.6 / - |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
