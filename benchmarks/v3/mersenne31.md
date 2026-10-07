# Mersenne fields: `m31`, `qm31` (v3)

Paired-run snapshot for v3. Every result cell reports Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); shared hosts, toolchain, sampling, and number format: [BENCHMARKS.md](../../BENCHMARKS.md). Canonical field labels: [labels.md](../labels.md). The paired comparison lives at [v2 versus v3](../comparison/mersenne31.md); the other snapshot at [v2](../v2/mersenne31.md).

## Setup

| Setting | Value |
| --- | --- |
| Campaign | `just bench-paired` (family round `m31`) |
| Revision | `4fa978b27d7b7b11c83291e212f47eccb5e7e8f3` |
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
| add_assign | 29.8 / 35.1 | 28.5 / 35.7 | 8.20 / 21.4 |
| add_assign_scalar | 45.2 / 57.7 | 44.8 / 54.4 | 25.2 / 22.0 |
| mul_elementwise | 14.5 / 19.6 | 14.1 / 19.7 | 5.33 / 16.6 |
| mul_elementwise_assign | 15.2 / 20.4 | 15.1 / 20.5 | 7.60 / 19.6 |
| sub_assign_scalar | 45.6 / 57.1 | 45.0 / 56.5 | 25.1 / 22.0 |

`qm31`, ordinary `Vec<u8>` buffers.

| Operation | 4 KiB (GiB/s) | 256 KiB (GiB/s) | 8 MiB (GiB/s) |
| --- | --- | --- | --- |
| add_assign | 19.9 / 33.6 | 18.5 / 33.0 | 8.03 / 24.0 |
| add_assign_scalar | 33.4 / 55.7 | 28.5 / 53.0 | 20.2 / 25.4 |
| mul_elementwise | 10.4 / 13.4 | 10.6 / 13.5 | 5.24 / 12.8 |
| mul_elementwise_assign | 6.74 / 13.7 | 6.87 / 13.7 | 6.25 / 12.5 |
| sub_assign_scalar | 33.0 / 55.7 | 28.4 / 53.0 | 20.1 / 25.4 |

### Row-wise addition

`add_assign_rows` interleaves the row streams; flat `add_assign` and the literal per-row loop appear under compositions.

| Row bytes | Rows | add_assign_rows (GiB/s) |
| --- | --- | --- |
| 1 KiB | 1 | 26.9 / 33.6 |
| 1 KiB | 16 | 30.7 / 37.3 |
| 1 KiB | 2 | 26.4 / 36.5 |
| 1 KiB | 32 | 30.0 / 35.2 |
| 1 KiB | 4 | 29.0 / 34.8 |
| 1 KiB | 8 | 30.1 / 36.5 |
| 64 B | 1 | 5.52 / 12.1 |
| 64 B | 16 | 27.1 / 33.6 |
| 64 B | 2 | 6.37 / 19.0 |
| 64 B | 32 | 26.4 / 36.5 |
| 64 B | 4 | 16.6 / 25.2 |
| 64 B | 8 | 24.5 / 30.1 |
| 64 KiB | 1 | 28.2 / 34.4 |
| 64 KiB | 16 | 18.5 / 22.5 |
| 64 KiB | 2 | 29.6 / 38.5 |
| 64 KiB | 32 | 18.1 / 21.5 |
| 64 KiB | 4 | 26.1 / 36.7 |
| 64 KiB | 8 | 25.5 / 33.0 |

### Row-wise addition compositions

Public call compositions: flat `add_assign` and the literal per-row loop over the same geometry.

| Operation | Row bytes | Rows | Throughput (GiB/s) |
| --- | --- | --- | --- |
| add_assign per row | 1 KiB | 1 | 28.2 / 34.5 |
| add_assign per row | 1 KiB | 16 | 29.6 / 32.8 |
| add_assign per row | 1 KiB | 2 | 30.1 / 35.2 |
| add_assign per row | 1 KiB | 32 | 28.5 / 31.8 |
| add_assign per row | 1 KiB | 4 | 29.8 / 32.9 |
| add_assign per row | 1 KiB | 8 | 30.0 / 33.0 |
| add_assign per row | 64 B | 1 | 6.14 / 11.6 |
| add_assign per row | 64 B | 16 | 9.02 / 15.8 |
| add_assign per row | 64 B | 2 | 6.04 / 13.5 |
| add_assign per row | 64 B | 32 | 8.53 / 15.3 |
| add_assign per row | 64 B | 4 | 8.29 / 14.7 |
| add_assign per row | 64 B | 8 | 8.80 / 15.4 |
| add_assign per row | 64 KiB | 1 | 28.3 / 35.4 |
| add_assign per row | 64 KiB | 16 | 18.5 / 22.5 |
| add_assign per row | 64 KiB | 2 | 29.6 / 38.3 |
| add_assign per row | 64 KiB | 32 | 18.1 / 21.7 |
| add_assign per row | 64 KiB | 4 | 26.0 / 35.1 |
| add_assign per row | 64 KiB | 8 | 25.4 / 32.9 |
| add_assign | 1 KiB | 1 | 29.1 / 34.7 |
| add_assign | 1 KiB | 16 | 30.9 / 36.2 |
| add_assign | 1 KiB | 2 | 27.9 / 36.6 |
| add_assign | 1 KiB | 32 | 30.2 / 35.5 |
| add_assign | 1 KiB | 4 | 29.8 / 35.1 |
| add_assign | 1 KiB | 8 | 30.6 / 35.7 |
| add_assign | 64 B | 1 | 7.41 / 13.2 |
| add_assign | 64 B | 16 | 29.1 / 34.2 |
| add_assign | 64 B | 2 | 6.37 / 20.2 |
| add_assign | 64 B | 32 | 27.9 / 36.6 |
| add_assign | 64 B | 4 | 20.4 / 26.6 |
| add_assign | 64 B | 8 | 27.4 / 30.8 |
| add_assign | 64 KiB | 1 | 28.3 / 35.6 |
| add_assign | 64 KiB | 16 | 18.5 / 22.6 |
| add_assign | 64 KiB | 2 | 29.7 / 38.5 |
| add_assign | 64 KiB | 32 | 18.1 / 21.6 |
| add_assign | 64 KiB | 4 | 26.1 / 36.5 |
| add_assign | 64 KiB | 8 | 25.6 / 33.0 |

## Competitors

Page-aligned regions on each library's native layout; conversions are excluded. Packed throughput counts two region byte extents per call. Scalar rows measure indexed products with accumulation. Arms alternate within one process, and outputs are validated before timing.

### Scalar

| Field | Operation | `fgf` (ns/op) | Plonky3 (ns/op) |
| --- | --- | --- | --- |
| gld | scalar a * b | 4.484 / 2.046 | 2.610 / 1.110 |
| m31 | scalar a * b | 4.192 / 1.547 | 2.043 / 0.803 |
| qm31 | scalar a * b | 9.691 / 4.959 | 6.953 / 2.911 |

### Packed

| Field | Operation | `fgf` (GiB/s) | Plonky3 (GiB/s) |
| --- | --- | --- | --- |
| gld | dst += c * src | 19.7 / 17.0 | 22.5 / 20.6 |
| gld | dst += v | 97.5 / 81.6 | 103 / 149 |
| gld | dst -= v | 103 / 81.1 | 103 / 147 |
| gld | dst = a * b | 23.8 / 23.3 | 26.9 / 26.5 |
| gld | dst = c * src | 25.1 / 23.8 | 27.4 / 27.1 |
| m31 | dst += c * src | 32.9 / 37.6 | 49.8 / 57.0 |
| m31 | dst += v | 98.4 / 122 | 103 / 153 |
| m31 | dst -= v | 98.5 / 121 | 103 / 153 |
| m31 | dst = a * b | 34.9 / 41.2 | 59.9 / 69.3 |
| m31 | dst = c * src | 50.4 / 53.6 | 65.4 / 72.7 |
| qm31 | dst += c * src | 21.4 / 25.3 | 27.0 / 31.8 |
| qm31 | dst += v | 68.2 / 122 | 103 / 153 |
| qm31 | dst -= v | 67.8 / 121 | 103 / 153 |
| qm31 | dst = a * b | 23.9 / 28.5 | 24.0 / 26.2 |
| qm31 | dst = c * src | 29.2 / 34.2 | 30.6 / 36.1 |


## Caveats

- The unchanged byte-XOR control bracketed every self-suite process; its values are retained in the raw evidence.
- The prime harness control row precedes the first field heading; control values are retained in the raw evidence.
