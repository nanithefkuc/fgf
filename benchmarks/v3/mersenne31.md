# Mersenne fields: `m31`, `qm31` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/mersenne31.md); the other snapshot at [v2](../v2/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| Revision | `358c31f92e3febd885a4b9d2cdb4858c632cb4fc` |
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
| add_assign | 29.8 / 33.6 | 28.5 / 34.4 | 7.99 / 19.8 |
| add_assign_scalar | 45.5 / 54.8 | 44.8 / 52.0 | 25.1 / 21.1 |
| mul_elementwise | 24.1 / 27.8 | 22.8 / 28.9 | 5.39 / 17.4 |
| mul_elementwise_assign | 18.6 / 25.9 | 18.0 / 24.8 | 7.85 / 22.5 |
| sub_assign_scalar | 45.5 / 54.8 | 45.0 / 52.0 | 25.1 / 21.1 |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 20.0 / 32.3 | 18.5 / 31.7 | 7.88 / 22.2 |
| add_assign_scalar | 33.4 / 53.4 | 28.5 / 50.8 | 20.5 / 24.5 |
| mul_elementwise | 12.5 / 17.1 | 12.3 / 16.7 | 5.26 / 15.5 |
| mul_elementwise_assign | 8.18 / 17.3 | 8.39 / 16.8 | 6.98 / 16.8 |
| sub_assign_scalar | 33.5 / 53.5 | 28.4 / 50.8 | 20.5 / 24.5 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 27.7 / 32.3 |
| 1 KiB | 16 | 30.7 / 35.7 |
| 1 KiB | 2 | 26.9 / 35.1 |
| 1 KiB | 32 | 30.1 / 34.0 |
| 1 KiB | 4 | 29.2 / 33.2 |
| 1 KiB | 8 | 30.3 / 35.0 |
| 64 B | 1 | 5.78 / 11.6 |
| 64 B | 16 | 27.7 / 32.3 |
| 64 B | 2 | 6.37 / 10.5 |
| 64 B | 32 | 26.7 / 35.1 |
| 64 B | 4 | 17.8 / 24.5 |
| 64 B | 8 | 26.2 / 31.9 |
| 64 KiB | 1 | 28.2 / 34.0 |
| 64 KiB | 16 | 18.6 / 21.9 |
| 64 KiB | 2 | 29.6 / 36.9 |
| 64 KiB | 32 | 17.9 / 20.6 |
| 64 KiB | 4 | 26.1 / 35.2 |
| 64 KiB | 8 | 25.8 / 30.8 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 28.2 / 33.0 |
| add_assign per row | 1 KiB | 16 | 29.2 / 31.4 |
| add_assign per row | 1 KiB | 2 | 30.1 / 33.7 |
| add_assign per row | 1 KiB | 32 | 28.1 / 30.6 |
| add_assign per row | 1 KiB | 4 | 29.5 / 31.7 |
| add_assign per row | 1 KiB | 8 | 29.6 / 31.8 |
| add_assign per row | 64 B | 1 | 6.14 / 11.1 |
| add_assign per row | 64 B | 16 | 9.01 / 15.2 |
| add_assign per row | 64 B | 2 | 6.04 / 9.47 |
| add_assign per row | 64 B | 32 | 8.52 / 14.7 |
| add_assign per row | 64 B | 4 | 8.29 / 14.1 |
| add_assign per row | 64 B | 8 | 8.54 / 14.7 |
| add_assign per row | 64 KiB | 1 | 28.3 / 34.2 |
| add_assign per row | 64 KiB | 16 | 18.5 / 21.9 |
| add_assign per row | 64 KiB | 2 | 29.6 / 35.9 |
| add_assign per row | 64 KiB | 32 | 17.9 / 20.8 |
| add_assign per row | 64 KiB | 4 | 26.0 / 33.8 |
| add_assign per row | 64 KiB | 8 | 25.7 / 32.1 |
| add_assign | 1 KiB | 1 | 29.5 / 33.1 |
| add_assign | 1 KiB | 16 | 30.9 / 35.8 |
| add_assign | 1 KiB | 2 | 27.8 / 35.3 |
| add_assign | 1 KiB | 32 | 30.2 / 33.8 |
| add_assign | 1 KiB | 4 | 29.9 / 33.6 |
| add_assign | 1 KiB | 8 | 30.6 / 34.2 |
| add_assign | 64 B | 1 | 7.41 / 11.2 |
| add_assign | 64 B | 16 | 29.4 / 33.1 |
| add_assign | 64 B | 2 | 6.37 / 10.5 |
| add_assign | 64 B | 32 | 27.8 / 35.3 |
| add_assign | 64 B | 4 | 21.6 / 26.3 |
| add_assign | 64 B | 8 | 30.8 / 32.4 |
| add_assign | 64 KiB | 1 | 28.3 / 33.1 |
| add_assign | 64 KiB | 16 | 18.6 / 21.8 |
| add_assign | 64 KiB | 2 | 29.6 / 36.9 |
| add_assign | 64 KiB | 32 | 17.9 / 20.6 |
| add_assign | 64 KiB | 4 | 26.1 / 35.3 |
| add_assign | 64 KiB | 8 | 25.8 / 31.2 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 4.485 / 2.149 | 2.563 / 1.162 |
| m31 | scalar a * b | 4.201 / 1.611 | 2.098 / 0.881 |
| qm31 | scalar a * b | 4.524 / 2.417 | 6.959 / 3.053 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 16.3 | 22.5 / 19.6 |
| gld | dst += v | 97.4 / 78.1 | 103 / 141 |
| gld | dst -= v | 103 / 77.6 | 103 / 142 |
| gld | dst = a * b | 23.8 / 22.4 | 26.9 / 25.3 |
| gld | dst = c * src | 25.1 / 22.8 | 27.4 / 26.0 |
| m31 | dst += c * src | 38.1 / 48.2 | 49.8 / 54.7 |
| m31 | dst += v | 98.5 / 117 | 103 / 147 |
| m31 | dst -= v | 98.5 / 116 | 103 / 147 |
| m31 | dst = a * b | 51.4 / 58.6 | 59.9 / 66.5 |
| m31 | dst = c * src | 55.6 / 66.6 | 65.4 / 69.9 |
| qm31 | dst += c * src | 23.4 / 28.8 | 27.0 / 30.4 |
| qm31 | dst += v | 68.2 / 117 | 103 / 147 |
| qm31 | dst -= v | 67.8 / 117 | 103 / 147 |
| qm31 | dst = a * b | 28.2 / 36.3 | 24.0 / 25.2 |
| qm31 | dst = c * src | 33.2 / 42.1 | 30.6 / 34.7 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
