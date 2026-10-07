# v2 versus v3 comparisons

These pages compare matched public workloads on released v2 and v3, grouped
by [canonical field label](../labels.md). Absolute timings and competitor
panels live separately under [v2](../v2/) and [v3](../v3/). Shared hosts,
builds, and reproduction are recorded in [BENCHMARKS.md](../../BENCHMARKS.md).

| Family | Comparison |
| --- | --- |
| Byte polynomial fields | [gf8b, gf8d](gf8.md) |
| Degree-16 towers | [gf16b, gf16d](gf16.md) |
| Wider towers | [gf32b, gf64b, fp16, fp32, fp64](wide-binary.md) |
| Bit-packed field | [gf1](gf1.md) |
| Mersenne fields | [m31, qm31](mersenne31.md) |
| Goldilocks | [gld](goldilocks.md) |

## Matched workloads

A pair fixes field presentation, semantic operation, preparation mode,
coefficient schedule, row bytes, sources, destinations, pitch, alignment,
base offsets, and timed scope. API spelling differences are adapted only
in benchmark code. Released library bodies are not changed to construct
the baseline. Prime-field fixtures use canonical lanes accepted by both
versions; v3's broader raw-lane acceptance is not attributed to a v2-valid
input benchmark.

The frozen released-v2 benchmark adaptation is materialized by
`just bench-prepare-v2 <new-source-root>`. The adaptation preserves the
measurement fixtures and protocols while calling released public APIs.
Its patch cannot change library source. A private test implementation does
not substitute for an absent public version counterpart.

## Interleaving and aggregation

Each host runs both versions in five complete rounds during one session.
Every family round records its version order; competitor arms interleave
inside their own process. Timed operations warm their production state,
and setup, allocation, fixture generation, preparation, and representation
conversion remain outside warm application timings unless named as part
of the operation.

Each host's case aggregate is the median of five per-run medians. Result
cells are Willow Cove (i5-1135G7) / Golden Cove (i7-12700K); the hosts are independent measurements.
A speed factor never compares one host with the other or combines different
sessions. An interrupted campaign requires both versions to be remeasured
on that host.

## Metrics

Tables use latency for scalar and short prepared-operation cases, and bulk
throughput for packed regions. The **v3 speed factor** is `v2 / v3` for
latency and `v3 / v2` for throughput. Above one means less elapsed time for
v3 on the matched workload. Ratios are computed from unrounded host-local
aggregates, not the displayed cells.

Throughput numerators belong to each panel. Multi-row self panels and native
competitor panels do not share an assumed byte count. Bit-range XOR reports
its touched range while retaining its documented full-buffer numerator;
short-range latency is per row pair, not per batch.

## Controls and availability

The unchanged byte-XOR control runs at process entry and exit. Duplicate
fgf arms in competitor harnesses measure scheduling and interleaving bias.
The raw evidence retains controls, individual round values, and variability.
Differences comparable to control variability are unresolved, not proof of
a faster implementation.

`-` means no public version counterpart or no native primitive for that panel.
Missing host records, failed correctness, missing required arms, mismatched
geometry, and incomplete runs reject publication rather than producing partial
results. No speed factor is computed from an unavailable pair.
The custom `gf16d` fixture is v3-only and scalar-routed;
its existence does not imply a public v2 API or host-tier SIMD execution.

## Evidence generation

`just bench-publish <raw-records> <output-root>` parses unrounded records,
checks round completeness and case identities, and generates snapshots
and comparison tables together. `--verify` regenerates and checks their
bytes without overwriting. The raw record retains source and artifact
digests, environment, realized order, correctness output, backend banners,
controls, and normalized case evidence. Previous published numbers are
not an input to generation.
