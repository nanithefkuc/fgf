# Benchmarks

Public results use the complete **2026-10-19** two-host campaign over the
3.0.1 source; prepared field maps come from a separate campaign at revision
`f10e59d`, and row compositions from a separate paired base/candidate
campaign whose revisions its page lists. Library changes absent from those
measured sources have no figures here. Choose a workload or field family
below, then use its operation links to find a figure.
Cells use **Willow Cove / Golden Cove** order. Higher **GiB/s** is better;
lower **ns/op** is better. A dash means no measurement.

## Find a benchmark

| Workload | Results |
| --- | --- |
| Public calls across representative fields | [Public operations](benchmarks/v3/public-api.md) — arithmetic, buffers, preparation, queries, and encodings |
| Embeddings, isomorphisms, restrictions, conversion loops | [Field maps](benchmarks/v3/maps.md) |
| Native GF16 encodings and matched competitors | [GF16 presentations](benchmarks/v3/gf16-presentations.md) |
| Prepared maps, Frobenius, and bit-packed map rows | [Prepared field maps](benchmarks/v3/field-maps.md) |
| Prime, residue, and component scatter, gather, matrix, and row addition | [Row compositions](benchmarks/v3/compositions.md) — paired base/candidate factors and routing boundaries |

| Field family | Size sweeps and competitors |
| --- | --- |
| Byte polynomial fields | [gf8](benchmarks/v3/gf8.md) — Intel ISA-L, klauspost/reedsolomon |
| Degree-16 extensions | [gf16](benchmarks/v3/gf16.md) — GF-Complete, reed-solomon-erasure, reed-solomon-simd, Leopard-RS 1.x |
| Wider binary extensions | [Wide binary](benchmarks/v3/wide-binary.md) |
| Bit-packed GF(2) | [gf1](benchmarks/v3/gf1.md) |
| Mersenne specializations | [m31, qm31](benchmarks/v3/mersenne31.md) — Plonky3 |
| Goldilocks specialization | [gld](benchmarks/v3/goldilocks.md) — Plonky3 |
| Generic prime residues | [Residues](benchmarks/v3/residue.md) |

### Reading the results

- **Public operations** group results by task and operation, with fields down
  the rows. Exact callable paths are expandable, not repeated in every cell.
- **Geometry** uses named counts. Shared counts appear above a table;
  varying counts occupy columns. Counted bytes are the throughput numerator,
  not necessarily the size of one source or destination row.
- **Charts** summarize every packed operation at the explicitly labelled
  256 KiB size, with a shared zero-based scale for both hosts. Exact figures
  and other sizes remain in the tables.
- **Measurement details and controls** are expandable. Timer-floor warnings
  remain beside affected latency figures.
- **Field definitions** are in [labels.md](benchmarks/labels.md).

The surfaces inventory contains 2240 registered cases over 167 compiler-derived
stable owned callables and 45 representative field presentations. The GF16
harness adds seven presentations, 110 registered case groups, and 376 timed
arm measurements per round. These are representative classes, not every
possible generic parameter combination.

Public pages exclude released-v2 snapshots and version/backend-factor tables.
Their samples remain in raw evidence; the paired capture protocol is unchanged.

## Mersenne31 contract note

v3 packed Mersenne31 operations accept every raw lane bit pattern and
return canonical lanes; v2 kernels assume canonical inputs. The wider
input contract costs a high-bit screen on the multiply paths: the
AVX-512 entries screen lanes for bit 31, multiplying lanes already
inside the redundant range as loaded. A lane outside it triggers
canonicalized recomputation of its register group for the in-place
entries and of the whole call for the out-of-place entries. The AVX2
entries screen each input vector on load. M31 parity is not reached:
the remaining native M31 gap is an accepted measured tradeoff,
attributed by two independent assembly reviews to the raw-lane contract
rather than to dispatch or bounds-check overhead. Golden Cove runs the
v3/AVX2 route and never executes the AVX-512 screen; its M31 rows carry
the AVX2 per-vector screen cost. A tile-amortized AVX2 screen remains an
unevaluated follow-up. The snapshots below record the screened revision;
the targeted factors for both hosts follow.


## Campaign

<details>
<summary>Hosts, source archives, toolchain, and sampling</summary>


| Setting | Willow Cove | Golden Cove |
| --- | --- | --- |
| Campaign date (UTC) | 2026-10-19 | 2026-10-19 |
| CPU | Intel Core i5-1135G7 | Intel Core i7-12700K |
| Availability | Measured | Measured |
| OS | Arch Linux 7.2.7-arch1-1 | CachyOS Linux 7.2.6-1-cachyos |
| Isolated CPU | 3 | 8; isolated set 8–11 |
| Governor | performance | performance |
| SMT | Siblings offline | Enabled; sibling 9 isolated and online |
| Requested process backend | v4x | v3_gfni_crypto |
| Rust | 1.98.1 | 1.98.1 |
| Go | 1.27.1-X:nodwarf5 | 1.27.1-X:nodwarf5 |
| GCC | 16.2.1 | 16.2.1 |
| System ISA-L | 2.32.0 | 2.32.0 |
| Execution | SCHED_IDLE, pinned CPU; RAYON_NUM_THREADS=1, GOMAXPROCS=1 | SCHED_IDLE, pinned CPU; RAYON_NUM_THREADS=1, GOMAXPROCS=1 |

| Source and method | Value |
| --- | --- |
| Released baseline | v2.1.0, `c20465ba22d078e0fb115fb6a99fa77027992c90` |
| Current library | Measured working-tree archive; base revision `63b8d168a7e4b81043fafe4db1ffb9918dc1c231` |
| Current source archive SHA-256 | `8482fccf8654aa0772550d9e6a4bedaeffef2194126f41cd77ff9533bb2d60ac` |
| Baseline source archive SHA-256 | `d40af5503b4a02363ebab510ae57673e4dbd52c18b872757fa17fc74937d6179` |
| Self build | Portable x86-64 target, all features, thin LTO, one codegen unit; published dependency pins |
| Competitor build | Native host builds; versions and packing widths in family pages |
| Sampling | Median of five per-run medians; no historical timing inputs |
| Version order | Both versions in shuffled order within every family round |
| Residue order | Same artifact per host under routed, v3, scalar requests; rotating order within every round |
| Residue placements | Median over offsets 0, 16, 32 within each run |
| Surfaces records | One `round-N-v3-surfaces.log` per round: public-API and GF16 presentation units run in shuffled order; inventories match across rounds and hosts |
| Surfaces inventory gate | Pinned nightly rustdoc JSON independently declares stable owned callables; every registered fixture executes under `--check` |
| Self calibration | Target 100 µs; at most 64 samples, 250 ms deadline |
| Public-API calibration | Whole batches target 100 µs; at most 24 samples, 250 ms deadline; a black-boxed timer-floor control accompanies latency records |
| Single-body calibration | Scalar-chain, preparation, and conversion batches target 100 µs (full) / 15 µs (smoke); at most 1200 samples; 500 ms (full) / 50 ms (smoke) deadline. The scalar chain measures 4096-call chunks and reports nanoseconds per multiplication; its `batch_reps` counts primitive calls |
| Native presentation sampling | Per arm: 32 warmup calls, then whole-batch doubling toward a 100 µs (full) / 15 µs (smoke) batch; the batch floor is 1 fixed / 16 cycling16 calls, capped at 2^20; the duplicate fgf control reuses the fgf batch size. Arm orders follow a Williams balanced Latin square (4 orders for 4 arms, 10 for 5): within one cycle every arm takes every position and directly follows every other arm equally often. Sampling stops only at a cycle boundary, after at least 30 rounds (full) / one cycle (smoke) and the 500 ms (full) / 50 ms (smoke) budget, or at 1200 rounds; both bounds round up to whole cycles, so the last cycle may overshoot. Each sample divides a timed batch by its batch size; every `GF16_SURFACE` record carries its `batch_reps` |
| Run modes | Each surfaces and GF16 process header is followed by a `BENCHMARK_MODE` banner (`full`, `smoke`, `check`; `inventory` only on the `surfaces` artifact). Only `full` rounds enter publication; the publisher rejects smoke, missing mode markers, and incomplete rounds |
| Controls | Unchanged byte-XOR at process entry/exit; duplicate fgf competitor arms |
| Number format | Uniform total digit count per table, at least three significant figures; round-half-up; ratios keep five digits |

</details>

## Caveats

- A process backend is not an operation-level claim. Field caps and residue
  operation routes are retained in the evidence; short rows can use narrower
  or portable paths.
- Allocation, fixture generation, reusable preparation, and representation
  conversion are excluded unless the panel explicitly includes them.
- Throughput numerators belong to each panel. Self multi-row panels and native
  competitor panels do not use interchangeable byte counts.
- Binary-field scalar dependency chains cycle through nonzero coefficients.
  Plonky3 scalar panels perform indexed multiply-accumulate operations.
- Native GF(2^16) libraries keep their own encodings; validated field maps are
  outside timing. The custom `gf16d` presentation is v3-only and scalar-routed.
- Public-API surfaces and GF16 presentations are current-only: no version
  factors are published. Map work uses scalar public calls and caller loops,
  not a bulk-map API; map records report the scalar route.
- Retained backend samples measure public calls under downgrade requests,
  not isolated instruction bodies or a released residue implementation.
- Differences comparable to control or round variability do not establish a win.
- Golden Cove self rows of 8 MiB and larger exceed its caches. The byte-XOR
  process control uses 256 KiB and cannot detect placement bias at those sizes;
  large-buffer figures do not establish small cross-run performance differences.
- Golden Cove has no live AVX-512 capability. Its routed residue calls use
  AVX2 or scalar; a routed-versus-AVX2 factor on that host is not an
  AVX-512 comparison.
- Short-call and native competitor results can vary across rounds. Per-case
  duplicate-fgf controls remain in the GF(2^16) pages; their variability
  also limits interpretation of small differences.

## Reproduce

The retained campaign is `bench-records/fgf-301-20261019T181500Z/`.
Regeneration uses its captured compiler inventory, not a newer checkout's JSON:

```sh
just bench-publish bench-records/fgf-301-20261019T181500Z benchmarks \
  --rustdoc bench-records/fgf-301-20261019T181500Z/fgf-rustdoc-final.json
just bench-publish bench-records/fgf-301-20261019T181500Z benchmarks \
  --rustdoc bench-records/fgf-301-20261019T181500Z/fgf-rustdoc-final.json --verify
```

### Capture a new campaign


Run the complete campaign on each host from an isolated checkout outside
umbrella dependency patches, then collect both host directories before
publishing:

```sh
just doctor
just bench-prepare-v2 /tmp/fgf-v2-bench
export RUSTUP_TOOLCHAIN=1.98.1 RAYON_NUM_THREADS=1 GOMAXPROCS=1
export HOST=willow FEC_GOLDEN_CORE=3 SIMD_BACKEND=v4x
# Golden Cove: HOST=golden FEC_GOLDEN_CORE=8 SIMD_BACKEND=v3_gfni_crypto
mkdir -p "/tmp/fgf-records/$HOST"
{
  date -u; uname -srvmo; lscpu; rustc -Vv; cargo -V
  cat "/sys/devices/system/cpu/cpu$FEC_GOLDEN_CORE/cpufreq/scaling_governor"
  cat "/sys/devices/system/cpu/cpu$FEC_GOLDEN_CORE/topology/thread_siblings_list"
} > "/tmp/fgf-records/$HOST/environment.txt"
just bench-paired /tmp/fgf-v2-bench "/tmp/fgf-records/$HOST"
# Supply metadata.json and collect both complete host directories.
just benchmark-api-json
python3 benches/check_surfaces.py --manifest benches/surface_manifest.json \
    --rustdoc target/doc/fgf.json --records /tmp/fgf-records
just bench-publish /tmp/fgf-records benchmarks
just bench-publish /tmp/fgf-records benchmarks --verify
```

The compiler-derived callable inventory comes from the measured source through
`just benchmark-api-json`. The campaign manifest pins its `rustdoc_json_sha256`;
the captured JSON remains with the raw evidence. The supplemental
`benches/surface_manifest.json` covers dependency-owned operators and
non-callable observations, not a second list of crate-owned functions.
The publisher accepts `--surface-manifest` and `--rustdoc` overrides.
Captured `surfaces --check` and `gf16 --presentations --check` stdout are
validated against the same compiler inventory:

```sh
python3 benches/check_surfaces.py --manifest benches/surface_manifest.json \
    --rustdoc target/doc/fgf.json --check-log surfaces-check.log --check-log presentations-check.log
```

Use this manifest shape, replacing the current revision with the measured
checkout's revision and retaining source/artifact digests with the raw records:

```json
{
  "v2_revision": "c20465ba22d078e0fb115fb6a99fa77027992c90",
  "v3_revision": "<measured-revision>",
  "rustdoc_json_sha256": "<captured-rustdoc-JSON-sha256>",
  "toolchain": "1.98.1",
  "rounds": 5,
  "host_order": ["willow", "golden"],
  "families": ["gf", "gdl", "m31", "gf2", "prime", "maps", "surfaces"],
  "hosts": {
    "willow": {"cpu": "Intel Core i5-1135G7", "core": 3, "requested_backend": "v4x"},
    "golden": {"cpu": "Intel Core i7-12700K", "core": 8, "requested_backend": "v3_gfni_crypto"}
  }
}
```

`families` declares the campaign inventory the publisher verifies for
completeness. Trees recorded before the maps suite carry no key and verify
as the legacy five-family inventory.

The runner records realized order and completion under a host-wide lock.
Raw logs, source/build digests, and cell-level `evidence.json` stay in
git-ignored `bench-records/`. The publisher rejects incomplete available-host
rounds, changed geometries, missing competitors or controls, invalid values,
and backend mismatches.
