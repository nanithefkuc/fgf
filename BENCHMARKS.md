# Benchmarks

Public API performance and matched competitor comparisons for `fgf`, one page
per field family. Every result cell reports **Tiger Lake / Golden Cove**. Units
live in the column headers, and `-` marks an unavailable measurement.

| Fields | Page | Campaign | Competitors |
| --- | --- | --- | --- |
| `Gf8B`, `Gf8D` | [GF(2^8)](benchmarks/gf8.md) | `bench-gf-comp` | Intel ISA-L, klauspost/reedsolomon |
| `Gf16` | [GF(2^16)](benchmarks/gf16.md) | `bench-gf-comp`, `bench-gf16-comp` | GF-Complete, `reed-solomon-erasure`, `reed-solomon-simd`, Leopard-RS 1.x |
| `Gf32`, `Gf64`, `FanPaar16`–`FanPaar64` | [Wider binary fields](benchmarks/wide-binary.md) | `bench-gf-comp` | - |
| `Gf2` bit-packed | [GF(2)](benchmarks/gf2.md) | `bench-gf2-comp` | - |
| `Mersenne31`, `QuadMersenne31` | [Mersenne31](benchmarks/mersenne31.md) | `bench-m31-comp` | Plonky3 |
| `Goldilocks` | [Goldilocks](benchmarks/goldilocks.md) | `bench-gdl-comp` | Plonky3 |

## Hosts

The hosts are separate machines, not interchangeable measurement controls.

| Setting | Value |
| --- | --- |
| CPU | Intel Core i5-1135G7 / Intel Core i7-12700K |
| OS | Arch Linux, Linux 7.2.7-arch1-1 / CachyOS, Linux 7.2.6-1-cachyos |
| Isolated CPU | 3 / 8 |
| Governor | `performance` / `powersave` |

## Method

| Setting | Value |
| --- | --- |
| Source | `fgf` 2.1.0, identical measurement sources on both hosts |
| Toolchain | Rust 1.98.1 |
| Dependencies | `simdispatch` 0.2.0, `archmage` 0.9.29 |
| Build | Self suites use `--all-features`; each field page states its competitor build |
| Execution | `SCHED_IDLE` on the isolated CPU; `RAYON_NUM_THREADS=1`, `GOMAXPROCS=1` |
| Sampling | Every cell is the median of five per-run medians. Self-suite warm batches use 32 calls; scalar batches contain 1,048,576 operations. A field page states any deviation. |
| Competitor protocol | Arms interleave within each harness process and are validated before timing; duplicate-operation controls check measurement bias |
| Setup | Allocation, input generation, coefficient preparation, and layout conversion occur outside timed regions. Buffers are warm; alignment is stated per table. |
| Throughput | Single-row cases count buffer bytes; accumulating scatter, gather, and matrix cases count row bytes × sources × destinations. A field page states any other numerator. |

## Number format

Every value in one table carries the same number of digits, counting leading
zeros, so paired cells align down a column. A table takes the fewest digits
that leave its smallest value at least three significant figures, limited by
the resolution the harness records. Integer parts are never rounded, so a
value larger than the table's digit count prints in full.

## Reproduction

Run every campaign on each host's isolated CPU:

```sh
export FEC_GOLDEN_CORE=<cpu> RAYON_NUM_THREADS=1 GOMAXPROCS=1
chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just bench-gf-comp
chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just bench-gdl-comp
chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just bench-m31-comp
chrt --idle 0 taskset -c "$FEC_GOLDEN_CORE" just bench-gf2-comp
RUSTUP_TOOLCHAIN=1.98.1 just bench-gf16-comp
```
