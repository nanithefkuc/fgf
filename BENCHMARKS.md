# Benchmarks

These measurements cover the public operation shapes on two x86 hosts. Both
hosts ran the same source tree with the `v3_gfni_crypto` backend selected.

## Environment

| Host | CPU | Operating system | Rust | Pinned CPU |
| --- | --- | --- | --- | ---: |
| Lunar Lake | Intel Core Ultra 7 258V | Linux 7.2.4, Arch Linux | 1.98.0 | 1 |
| Golden Cove | Intel Core i7-12700K | Linux 7.2.3, CachyOS | 1.98.1 | 8, isolated |

The custom harness warms each operation, then reports the median per-iteration
time from up to 64 batches of 32 iterations. Inputs are deterministic and
coefficient preparation is outside the timed region unless the label says
one-shot. Throughput is logical bytes processed, not total cache traffic.

Run from the crate root:

```sh
FEC_GOLDEN_CORE=<cpu> just bench kernels
FEC_GOLDEN_CORE=<cpu> just bench compare
```

Results below are one complete pinned run on each host. Small nanosecond-scale
cases are timer-resolution-sensitive; the larger row and matrix cases are the
headline measurements.

## Scalar multiplication

The scalar harness executes a fixed batch of element multiplications. Results
are nanoseconds per operation.

| Field | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `Gf8B` | 0.37 ns | 0.34 ns |
| `Gf8D` | 0.36 ns | 0.34 ns |
| `Gf16` | 1.11 ns | 1.33 ns |

## Single-row packed operations

These rows use 64 KiB, 32-byte-aligned buffers. Throughput is GiB/s.

| Field | Operation | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | `mul_add` | 74.80 | 64.59 |
| `Gf8B` | `mul_into` | 63.51 | 60.25 |
| `Gf8D` | `mul_add` | 75.45 | 64.79 |
| `Gf8D` | `mul_into` | 63.58 | 60.43 |
| `Gf16` | `mul_add` | 71.05 | 30.92 |
| `Gf16` | `mul_into` | 61.16 | 31.48 |

The broader 256 KiB operation panel uses ordinary `Vec<u8>` buffers.

| Field | Operation | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | `add_assign` / XOR | 43.50 | 51.08 |
| `Gf8B` | `mul_add` | 44.38 | 49.60 |
| `Gf8B` | `mul_assign` | 61.45 | 67.63 |
| `Gf8B` | `mul_elementwise` | 29.53 | 42.98 |
| `Gf16` | `mul_add` | 41.70 | 41.77 |
| `Gf16` | `mul_assign` | 56.01 | 54.58 |
| `Gf16` | `mul_elementwise` | 25.51 | 29.77 |

At 64 KiB, prepared and one-shot `mul_add` converge because coefficient
preparation is small relative to the row loop.

| Field | Form | Lunar Lake | Golden Cove |
| --- | --- | ---: | ---: |
| `Gf8B` | one-shot | 49.91 | 49.38 |
| `Gf8B` | prepared | 48.91 | 48.79 |
| `Gf16` | one-shot | 45.35 | 41.07 |
| `Gf16` | prepared | 45.38 | 41.10 |

## Scatter, gather, and matrix

Each row is 64 KiB. Scatter writes eight rows from one source, gather combines
eight sources into one row, and matrix combines eight sources into eight rows.
Throughput is GiB/s over the harness's logical operation volume.

| Field | Operation | Geometry | Lunar Lake | Golden Cove |
| --- | --- | --- | ---: | ---: |
| `Gf8B` | `mul_add_scatter` | 1 source × 8 rows | 68.53 | 61.07 |
| `Gf16` | `mul_add_scatter` | 1 source × 8 rows | 48.95 | 47.97 |
| `Gf8B` | `mul_add_gather` | 8 sources × 1 row | 64.83 | 69.34 |
| `Gf16` | `mul_add_gather` | 8 sources × 1 row | 68.46 | 77.62 |
| `Gf8B` | `mul_add_matrix` | 8 sources × 8 rows | 124.79 | 114.13 |
| `Gf16` | `mul_add_matrix` | 8 sources × 8 rows | 65.90 | 55.57 |

Overwrite matrix results use 64 KiB rows and ten sources. Throughput counts
source bytes once per call, matching the reconstruction comparison below.

| Field | Output rows | Lunar Lake | Golden Cove |
| --- | ---: | ---: | ---: |
| `Gf8B` | 2 | 70.08 | 61.88 |
| `Gf8B` | 4 | 39.62 | 36.12 |
| `Gf8B` | 6 | 25.02 | 22.73 |
| `Gf8D` | 2 | 71.65 | 64.94 |
| `Gf8D` | 4 | 43.07 | 39.82 |
| `Gf8D` | 6 | 26.72 | 24.76 |

## Wider binary fields

These are 256 KiB `mul_add` operations. Throughput is GiB/s.

| Field | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `Gf16` polynomial tower | 42.28 | 36.40 |
| `Gf32` polynomial tower | 31.44 | 30.69 |
| `Gf64` polynomial tower | 19.32 | 16.88 |
| `FanPaar16` | 19.09 | 17.03 |
| `FanPaar32` | 7.54 | 6.31 |
| `FanPaar64` | 2.54 | 1.91 |

## Bit-packed GF(2)

The input size is 256 KiB, representing 2,097,152 field elements. Throughput is
GiB/s over packed bytes.

| Operation | Lunar Lake | Golden Cove |
| --- | ---: | ---: |
| `bits::xor_assign` | 57.42 | 66.06 |
| `bits::and_into` | 31.80 | 48.90 |
| `bits::weight` | 21.24 | 15.66 |
| `bits::dot_product` | 36.73 | 41.45 |
| `bits::xor_range` | 62.84 | 73.91 |
| `bits::xor_range_with` | 64.23 | 73.98 |

For 256 calls over 16-byte rows and the bit range `61..69`, preparing
`bits::XorRange` reduces the per-call cost from 7.08 ns to 2.13 ns on Lunar
Lake and from 5.34 ns to 2.07 ns on Golden Cove.

## Reconstruction comparison

`benches/compare.rs` compares compatible GF(2^8) overwrite gathers with
`reed-solomon-erasure` 6 using its `simd-accel` feature. Every case uses 16
dense nonzero coefficients. Throughput is GiB/s over source bytes.

| Row length | Implementation | Lunar Lake | Golden Cove |
| ---: | --- | ---: | ---: |
| 4 KiB | `fgf` `Gf8B` `mul_into_gather_with` | 98.29 | 84.42 |
| 4 KiB | `fgf` `Gf8D` `mul_into_gather_with` | 100.88 | 83.61 |
| 4 KiB | `reed-solomon-erasure` zero + `mul_slice_xor` | 52.71 | 50.32 |
| 16 KiB | `fgf` `Gf8B` `mul_into_gather_with` | 80.92 | 88.26 |
| 16 KiB | `fgf` `Gf8D` `mul_into_gather_with` | 86.15 | 85.69 |
| 16 KiB | `reed-solomon-erasure` zero + `mul_slice_xor` | 56.41 | 53.68 |

## ISA-L comparison

`bench-isal/` is a separate, unpublished package in this repository that
links the system Intel ISA-L and times it against `fgf` inside one process.
Both libraries implement GF(2^8) under the same `0x11D` polynomial, so every
arm is validated byte-for-byte against the other library's output before any
timing is taken; a disagreement aborts the run.

```sh
FEC_GOLDEN_CORE=<cpu> just bench-isal
```

The two arms of a shape are interleaved sample by sample within one process,
with the order alternating every round, so neither implementation owns the
warm side of a drift. Fixtures are page-aligned: with cache-line alignment
alone, whichever destination the allocator placed at the source's page offset
paid a 4 KiB store-to-load aliasing penalty, which moved single-source
throughput by a factor of four and the control off 1.00x.

Each row also carries the tenth and ninetieth percentile of its per-round
paired ratios, and the first row is a control: one `fgf` body timed against
itself over two destinations. The control is what says how much of a ratio
is real. It does not read a flat 1.00 — over five runs it spans 0.91 to
1.04, so a ratio inside that band is parity, not a result.

Lunar Lake, ISA-L 2.32.0, `fgf` backend `v3_gfni_crypto`, rustc 1.98.0, five
pinned runs. Columns are the range of the five per-run medians, in GiB/s
over source bytes, and the ratio column is the range of the five per-run
ratios; above 1.00 means `fgf` is faster.

| Shape | `fgf` | ISA-L | Ratio |
| --- | ---: | ---: | ---: |
| control: `mul_into` against itself | 60.37-66.78 | 64.11-66.78 | 0.91-1.04 |
| 64 KiB `dst = c * src` | 60.73-67.00 | 86.45-92.34 | 0.66-0.73 |
| 64 KiB `dst ^= c * src` | 76.49-80.63 | 78.35-79.47 | 0.96-1.03 |
| 4 KiB, 16 sources, one row | 111.18-114.30 | 134.14-138.40 | 0.83 |
| 16 KiB, 16 sources, one row | 96.16-98.76 | 91.61-94.04 | 1.04-1.07 |
| 4 KiB, 10 sources, 2 rows | 107.15-109.93 | 102.27-105.96 | 1.04-1.05 |
| 16 KiB, 10 sources, 2 rows | 91.15-94.95 | 86.06-88.92 | 1.06-1.10 |
| 64 KiB, 10 sources, 2 rows | 81.95-84.44 | 81.94-84.26 | 1.00 |
| 4 KiB, 10 sources, 4 rows | 52.47-55.45 | 43.75-45.96 | 1.20-1.21 |
| 16 KiB, 10 sources, 4 rows | 50.16-51.48 | 35.03-36.00 | 1.40-1.47 |
| 64 KiB, 10 sources, 4 rows | 48.08-49.23 | 36.36-37.49 | 1.31-1.32 |
| 4 KiB, 10 sources, 6 rows | 34.74-36.16 | 31.40-32.66 | 1.10-1.13 |
| 16 KiB, 10 sources, 6 rows | 31.39-32.54 | 26.12-27.26 | 1.17-1.23 |
| 64 KiB, 10 sources, 6 rows | 30.04-30.62 | 30.49-31.23 | 0.98-0.99 |

The libraries are at parity on the erasure-encode shapes: `fgf` leads at two
and four and six output rows below 64 KiB rows, widest at four, where
ISA-L's blocked dot-product family has no four-row specialization, and the
two are inside the control band at 64 KiB rows. ISA-L leads the 4 KiB
gather and the single-source overwrite; the overwrite gap is a store-policy
difference, not a kernel one, and the measurement that settles it is under
"Crossover and dispatch decisions" below. The `0x11B` field is not
comparable here: ISA-L implements `0x11D` only.

This record covers the Lunar Lake host. The Golden Cove column of the tables
above was measured in an earlier session and this section was not part of
that run.

## Crossover and dispatch decisions

### Non-temporal store threshold

A fused overwrite writes a destination it never reads, so an ordinary store
pays a read-for-ownership fetch of every line it is about to replace whole.
`vmovntdq` skips that fetch; the price is that the destination leaves cache.
`kernel::x86::NT_STORE_MIN` is the size at which that trade turns, and this
is the measurement that sets it, re-taken with page-aligned fixtures after
the aliasing artifact described above was found.

Lunar Lake, `v3_gfni_crypto`, `Gf8D` `ops::mul_into`, two interleaved rounds
of a 2 MiB build against a 16 KiB build, best of the two per arm, GiB/s.
"Read back" sums the destination after writing it, which is what an encoder
feeding a checksum or a network write does.

| Buffer | Write only, 2 MiB | Write only, 16 KiB | Read back, 2 MiB | Read back, 16 KiB |
| ---: | ---: | ---: | ---: | ---: |
| 16 KiB | 214.91 | 63.05 | 80.73 | 7.21 |
| 64 KiB | 66.92 | 78.05 | 41.44 | 9.51 |
| 256 KiB | 49.46 | 79.27 | 30.95 | 20.04 |
| 1 MiB | 46.37 | 77.11 | 28.58 | 19.67 |
| 2 MiB | 79.14 | 78.90 | 21.11 | 20.70 |
| 8 MiB | 35.50 | 35.93 | 18.48 | 18.48 |

The 2 MiB and 8 MiB rows are the control: both builds take the streaming
path there and agree to within a percent. Streaming early wins the
write-only column from 64 KiB up and loses the read-back column by four
times at 64 KiB and eleven times at 16 KiB. The threshold stays at 2 MiB,
which is where the read-back case breaks even.

This is also the whole of the single-source gap against ISA-L: `gf_vect_mul`
streams unconditionally, which is why it requires a 32-byte-aligned
destination and why it reaches 92 GiB/s where `fgf` reaches 67. Building
`fgf` with the threshold at 16 KiB moves that arm to 83 GiB/s, a 0.91 ratio
— and costs every consumer that reads its output back.

### Destination write prefetch: rejected

If the read-for-ownership fetch were a latency problem rather than a
bandwidth one, a write prefetch would hide it without evicting anything.
It is a bandwidth problem, and the prefetch buys nothing.

A `_MM_HINT_ET0` prefetch of the destination was added to both fused
overwrite bodies and swept over distances of 128, 256, 512 and 1024 bytes
by one, two and four lines per 128-byte tile. Against an unchanged 66.6 to
66.9 baseline at 64 KiB, every variant landed between 61.7 and 66.3: the
best matched the baseline and none beat it. Built without `prfchw` in the
target features, where the hint lowers to `prefetcht0` and fetches the line
for reading, the same code ran at 52 to 53 — the read prefetch adds traffic
and still leaves the ownership upgrade at the store.

The arithmetic agrees with the result. An ordinary-store overwrite moves
three streams per byte — source read, ownership read, destination write —
and a streaming one moves two; 92 over 67 is within a percent of that
three-to-two. A prefetch reorders the ownership read, it does not remove
it. The change was reverted; no prefetch is present in the kernels.

## Named competitors

| Library | License | Current comparison status |
| --- | --- | --- |
| `reed-solomon-erasure` 6 | MIT / Apache-2.0 | Measured in-process above for compatible GF(2^8) operations. |
| Intel ISA-L | BSD-3-Clause | Measured in-process above through `bench-isal/`, which links the system library. |
| `catid/leopard` | BSD-2-Clause | Its codec-level transforms do not expose the same operation-level contract. |

The `reed-solomon-erasure` and ISA-L comparisons are reported numerically.
`catid/leopard` requires a different API and is not presented as a matched
baseline.
