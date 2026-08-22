# Benchmarks

`fgf` uses small custom benchmark binaries rather than a statistical harness.
They print throughput directly so operation shape, row size, and backend remain
visible beside each result. The one Criterion harness is `benches/affine.rs`,
which decided the `0x11D` GFNI affine adoption; its command and numbers are
under "Crossover and dispatch decisions" below.

## Reproduce

```sh
cargo bench --bench kernels
cargo bench --bench compare
cargo bench --features internals --bench dot_product
cargo bench --features internals --bench dot_product -- --smoke
cargo bench --features internals --bench dot_product -- --tails
```

Pin a weaker backend to measure a dispatch crossover (commands recorded with
the historical `FFF_BACKEND`; the override is now `SIMD_BACKEND`, owned by
[`simdispatch`](https://github.com/nanithefkuc/simdispatch), with the tiers
renamed: `avx2`→`v3`, `ssse3`→`v2`):

```sh
SIMD_BACKEND=v3  cargo bench --bench kernels
SIMD_BACKEND=v2 cargo bench --bench kernels
SIMD_BACKEND=scalar cargo bench --bench kernels
```

Non-x86 targets need a runner. The `aarch64` and `wasm32` kernels were
measured with these:

```sh
# aarch64, on an adb-connected arm64 device (cargo-ndk):
cargo ndk -t arm64-v8a build --release --bench kernels
adb push target/aarch64-linux-android/release/deps/kernels-<hash> /data/local/tmp/
adb shell 'cd /data/local/tmp && taskset 80 ./kernels-<hash>'

# wasm32, under Node's WASI shim:
RUSTFLAGS="-C target-feature=+simd128" \
  cargo build --release --target wasm32-wasip1 --bench kernels
node wasi-run.mjs target/wasm32-wasip1/release/deps/kernels-<hash>.wasm
```

`wasi-run.mjs` is the whole shim (the same one works as
`CARGO_TARGET_WASM32_WASIP1_RUNNER` for `cargo test --target wasm32-wasip1`):

```js
import { WASI } from "node:wasi";
import { argv, env, exit } from "node:process";
import { readFile } from "node:fs/promises";

const [wasmPath, ...rest] = argv.slice(2);
const wasi = new WASI({ version: "preview1", args: [wasmPath, ...rest], env,
                        preopens: { "/": "/" }, returnOnExit: true });
const module = await WebAssembly.compile(await readFile(wasmPath));
exit(wasi.start(await WebAssembly.instantiate(module, wasi.getImportObject())));
```

Both runners are far noisier than a pinned desktop CPU, so a single pair of
runs proves nothing:

- On a phone, `taskset` to **one** big core, not the cluster: migration
  between cores produces a bimodal distribution (two clean clusters ~1.5x
  apart), and a whole run can land in either mode.
- Under Node, tiering and GC shift results by up to 40% run to run.
- Interleave the two binaries (`base, new, base, new, …`), take the **maximum**
  of at least three runs per key, and always include an unchanged operation as
  a control. Every number below was accepted only with its control at 1.00x.

Record the CPU, operating system, Rust version, selected backend, row size, row
count, and source count with any quoted result. `backend_for::<F>()` matters
for the wider towers and Fan–Paar family: GF(2^32)/GF(2^64) use GFNI on
x86 GFNI hosts, the canonical Fan–Paar GF(2^16)/32/64 use AVX2/SSSE3 on x86,
but FanPaar8, SSSE3 for the wider Fan–Paar fields, and every shuffle-only
backend still report the portable kernels even when the process-wide backend
is `avx512` or `gfni`.

## Interpreting the shapes

- `mul_add` is the single-row AXPY baseline.
- `mul_add_scatter` tests source-load sharing across destination rows.
- `mul_add_gather` and `mul_add_matrix` test whether destination tiles stay in
  registers across sources.
- `_with` operations separate coefficient preparation from byte-loop cost.
- `mul_elementwise` has no broadcast coefficient. It vectorizes on
  AVX-512/GFNI (native `GF2P8MULB`) and, via a branchless shift/reduce vector
  multiply, on AVX2/SSSE3, NEON, and Wasm `simd128`. Measured on a Core Ultra
  7 258V (Linux, rustc 1.93), 256 KiB buffers: gf8 0.78 → 7.97 GiB/s (AVX2)
  and 2.62 GiB/s (SSSE3); gf16 0.43 → 4.39 GiB/s (AVX2) and 1.48 GiB/s
  (SSSE3). The wider fields still use the scalar reference.

`dot_product` freezes the direct GF(2^8) N-to-1 baseline and the public
overwrite composition. It interleaves the raw AXPY-tail control, raw
fused-tail candidate, public accumulating operation, timed zero-then-gather,
and public `dot_product`. Every body is validated and allocation-free in the
timed region. The full run covers dense nontrivial coefficients,
1/2/3/4/8/12/16/24/32 sources, SIMD boundaries through 513 B, and 1–64 KiB
rows; `--smoke` keeps the same identifiers on a focused boundary grid,
`--tails` isolates 16/32/64/96/128-byte rows across source counts
1/2/3/4/8/16/32, and `--affine` interleaves native GFNI with the prepared
`Gf8B` affine-map prototype. Fixtures are 32-byte-aligned and hot-cache.
Controlled misalignment, displaced-cache, and coefficient-distribution
variants remain separate additions.

### Native GFNI source-fused short rows (2026-08-22)

The pre-change N-to-1 gather sent every row below 128 bytes through repeated
single-source AXPY. `gather_gfni` now selects a source-fused body only for
rows of exactly 32, 64, or 96 bytes with at least three sources. The
`gather_gfni_axpy_tail` preserves the old body for an interleaved control in
the same binary. Core Ultra 7 258V, Linux, rustc 1.93, backend
`v3_gfni_crypto`, 32-byte-aligned hot fixtures, dense nontrivial coefficients:

| Row | 4 sources | 16 sources |
| ---: | ---: | ---: |
| 32 B | 1.50x | 1.49x |
| 64 B | 1.41x | 1.11x |
| 96 B | 1.49x | 1.48x |

Ratios are fused divided by the interleaved AXPY-tail control. A fused 16-byte
prototype lost at low source counts and did not produce a stable 16-source win.
Fusing a remainder after the 128-byte main body was neutral-to-slower.
Production therefore keeps AXPY for one source, 16-byte/sub-lane and compound
scalar remainders, and every row at or above 128 bytes. Interleaved controls at
16/31/95/97/127/128/129 B, 4 KiB, and 4 KiB + 64 B remain at parity.

### Overwrite accumulator policy (2026-08-22)

The overwrite candidate shared the GFNI gather loop but seeded each destination
accumulator from the first source, removing the destination load and its first
XOR. The control timed `fill(0)` plus the production gather in the same process.
On the pinned short-row panel the native candidate was usually 0.84–1.04x the
control, with no coherent winning source/size region; at 128 B x 16 it was
substantially slower. The prototype was deleted.

The public `dot_product` contract remains useful independently of that rejected
kernel. It ignores the initial destination, zeros empty/all-zero input, maps one
source to `mul_into`, and otherwise composes zeroing with the field's measured
gather. `dot_product_with` consumes a prepared plan. Against public
zero-then-`mul_add_gather`, it is generally within 0–5% from eight sources
upward on the pinned 16–128 B panel; smaller multi-source calls pay up to about
10% for validation and the all-zero shortcut. Both forms are proven
allocation-free in `tests/zero_alloc.rs`. Consumer-level measurement, not this
microbenchmark, decides whether callers migrate.

### Prepared `Gf8B` affine gather prototype (2026-08-22)

The `--affine` panel substitutes prepared `VGF2P8AFFINEQB` maps for native
`GF2P8MULB` inside the same generic 128-byte gather body and the same measured
short-row fusion rule. Map lookup happens before the prepared sample; a
separate one-shot sample rewrites a preallocated factor slice before gathering.
Overwrite times `fill(0)` on both sides. The scalar map model and the hardware
gather each cover all 65,536 coefficient/input products. Generated assembly
broadcasts each map directly from the prepared slice into
`VGF2P8AFFINEQB`; the main loop does not spill maps.

Core Ultra 7 258V, Linux, rustc 1.93, backend `v3_gfni_crypto`,
32-byte-aligned hot fixtures, dense nontrivial coefficients. Ratios are native
time divided by prepared-affine time, so values above 1 favor affine:

| Row | 4 sources | 8 sources | 16 sources |
| ---: | ---: | ---: | ---: |
| 32 B | 0.72x | 0.77x | 1.02x |
| 64 B | 0.74x | 0.69x | 1.01x |
| 96 B | 0.78x | 0.72x | 1.05x |
| 128 B | 0.62x | 0.58x | 0.57x |
| 256 B | 0.74x | 0.65x | 0.72x |
| 1 KiB | 0.92x | 0.90x | 0.92x |
| 4 KiB | 1.04x | 0.94x | 1.03x |
| 16 KiB | 0.97x | 1.05x | 1.06x |

A repeated pinned run preserved the pattern. Prepared affine is substantially
slower through 1 KiB in the source-count region the prototype targeted. At
4 KiB the sign changes with source count; at 16 KiB it wins 3–6% from eight
sources upward, but the neighboring three/four-source shapes remain
neutral-to-slower. Overwrite follows the same topology: 128 B x 8–16 is
0.55–0.56x, while 16 KiB x 8–32 is 1.03–1.04x. Preparation does not erase the
long-row wins, but deepens the short-row losses.

Production therefore remains entirely on native `GF2P8MULB`: there is no
single multiply policy for the next tile sweep, and the modest long-row region
does not justify a new source-count/length dispatch without alignment,
cache-displacement, counter, and second-host evidence. The affine body and map
bank remain available only through `internals` so that exact cross-host rerun
stays reproducible; no public operation dispatches to them.

### GFNI gather tile and page-topology sweep (2026-08-22)

The `--tiles` panel instantiates the native gather body with one through four
32-byte accumulator lanes. Four lanes/128 bytes is the exact production
control, including its short-row policy; narrower candidates retain the same
single-source AXPY remainder. Fixtures cover 128 B through 16 KiB and
4/8/16/32 sources. The focused topology panel independently controls source
and destination misalignment, equal page offsets at 0 and 4032 bytes, and
64-byte-staggered source page offsets. All comparisons are interleaved and
differentially validated before timing.

Ratios are production time divided by candidate time:

| Shape/layout | 32 B tile | 64 B tile | 96 B tile | split 128 B |
| --- | ---: | ---: | ---: | ---: |
| 128 B x 16, aligned | 0.45x | 0.52x | 0.52x | — |
| 1 KiB x 16, aligned | 0.53x | 0.73x | 0.75x | — |
| 4 KiB x 16, aligned | 0.69x | 0.88x | 0.94–0.97x | 0.92x |
| 4 KiB x 16, page offset 0 | 0.78x | 1.07x | 1.09x | 1.02x |
| 16 KiB x 4, aligned | 0.72x | 0.90x | 0.95x | 0.95x |
| 16 KiB x 8, aligned | 0.76x | 0.94x | 1.02x | 1.01x |
| 16 KiB x 16, aligned | 0.86x | 1.01x | 1.02x | 1.04x |
| 16 KiB x 32, aligned | 0.80x | 0.95x | 0.96x | 1.01x |

The 4 KiB x 16 result is page topology, not a 32-byte-tile advantage. The
96-byte tile loses 3–6% with allocator-aligned fixtures but wins 4–9% when all
streams start at fixed 32-byte-aligned page offsets. Source misalignment moves
it back to parity; destination misalignment makes it lose up to 10%. The
32-byte candidate loses every controlled 4 KiB x 16 layout, by 16–31%.
Consequently the earlier external 1.73x result cannot be attributed to tile
width and is removed as an optimization target until both implementations run
inside one page-controlled harness.

`perf stat -d` over 1,048,576 iterations of page-zero 4 KiB x 16 explains why
96 bytes can win that one layout:

| Variant | Core cycles | Instructions | IPC | Backend bound |
| --- | ---: | ---: | ---: | ---: |
| production 128 B | 2.64B | 8.44B | 3.2 | 49.8% |
| 96 B | 2.45B | 10.26B | 4.2 | 32.7% |
| split 128 B | 2.61B | 10.17B | 3.9 | 42.7% |

L1D misses were only 1.5–2.6 thousand and the detailed 96/128-byte runs each
reported about 220 dTLB misses, so cache/TLB misses do not explain the cycle
gap. A counter-triggered even/odd source-chain split reduced backend pressure
but paid enough extra instructions and branches to lose on ordinary 4 KiB
layouts; its wins remained page-offset-dependent. Generated main loops for all
tile widths keep vector state in registers.

Production therefore keeps the single static 128-byte tile. A 96-byte or split
long-row dispatch is rejected: gains are small, source-count-dependent, and
reverse under neighboring alignment/page layouts. No second-host gate is
needed because no production policy changes. The tile and split bodies remain
available only through `internals` so the controlled sweep and counter workload
stay reproducible.

Small GF(2^16) rows are sensitive to coefficient preparation because a shuffle
backend builds four nibble tables per coefficient. Use `Coeff` or `Plan` when a
coding matrix is reused. Large rows amortize the same setup in the byte loop.
GF(2^8) preparation is free at every size — its 256-entry table bank is static
— and GFNI preparation is two broadcast words, so the effect is a shuffle-
backend GF(2^16) concern only.

`bench_preparation_crossover` in `benches/kernels.rs` measures exactly where
that stops mattering, one-shot against prepared over 16 B … 64 KiB rows. Run
it rather than quoting a remembered threshold; on a hybrid CPU pin the process
(`taskset -c <p-core>`) or the two core types will produce two answers.

`bench_blocked_vs_axpy` measures the blocked multi-row GF(2^16) kernels
against repeated single-row AXPY by calling both directly, bypassing dispatch.
It needs the `internals` feature (`cargo bench --bench kernels --features
internals`) and is the harness to rerun before changing which shape a backend
dispatches to.

`bench_small_row_shapes` measures the GF(2^16) multi-row shapes at row lengths
where a coefficient's four nibble tables are still a visible share of the work,
with a coefficient set that is one-fifth zeros and one-fifth ones so the
zero/one specializations are exercised. It also times fused `mul_into` against
the copy-then-scale it replaces.

`bench_large_destination` measures `mul_into` at 1, 8 and 32 MiB. Past 2 MiB
the x86 kernels store the destination non-temporally, which skips the
read-for-ownership fetch of lines they overwrite whole; the cost is that the
destination is not left cached, so the section also times an
encode-then-read-back loop, and `mul_add` as a control the change cannot touch.
Rerun it before moving that threshold, and pin the process: the effect is
entirely memory-side, so an unpinned run on a hybrid CPU reports the wrong
size at which it starts paying.

`bench_destination_alignment` measures multi-row scatter with the destination
at 32-byte residues 0 and 16. The GFNI and AVX2 kernels peel a row group's
lead-in so the rest of the pass is aligned; a single allocator-fresh buffer
cannot show that effect, because its residue never varies. Read the two skews
against each other, and keep an unchanged field (GF(2^8) here, when only the
GF(2^16) kernels changed) as a drift control.

## aarch64 and wasm32 kernel numbers (2026-07-31)

Snapdragon 8 Gen 3 (`arm64-v8a`, Android, rustc 1.93, backend `neon`), one big
core, maximum of four interleaved runs, GF(2^8) buffers as control (1.00x):

| Shape | Before | After |
| --- | --- | --- |
| gf16 `mul_add`, 4 KiB … 8 MiB | 2.09–2.17 GiB/s | 2.30–2.41 GiB/s (+11%) |
| gf16 `mul_assign`, ≤ 256 KiB | 2.30–2.35 GiB/s | 2.39–2.45 GiB/s (+4%) |
| gf16 `mul_into`, 16 KiB rows | copy-then-scale | fused, +6% over copy+scale |

The gf16 two-lane unroll is what moves `mul_add`; it wins at every size from
4 KiB to 8 MiB on a pinned core. Measured on the *cluster* instead, the same
binary reports anything from 0.65x to 1.11x — that spread is core migration,
not the kernel.

### PMULL against the nibble shuffle

`Backend::NeonAes` (was `Pmull`) and `Backend::Neon` differ by dispatch alone,
so `SIMD_BACKEND=neon_aes` against `SIMD_BACKEND=neon` in **one binary** is
the cleanest A/B this crate has (recorded with the historical
`FFF_BACKEND=pmull`/`neon`). Same device, one pinned big core, max of three
runs each:

| Shape | `neon` | `pmull` | |
| --- | --- | --- | --- |
| gf8 `mul_elementwise`, 4 KiB … 8 MiB | 0.83 GiB/s | 1.28–1.30 GiB/s | **1.55x** |
| gf16 `mul_elementwise`, 4 KiB … 8 MiB | 0.39–0.41 GiB/s | 0.36 GiB/s | 0.88–0.92x |
| gf8 `mul_add` fixed coefficient | 9.0–9.2 GiB/s | 1.17–1.23 GiB/s | 0.13x |
| gf16 `mul_add` fixed coefficient | 2.4 GiB/s | 0.62 GiB/s | 0.26x |

Only the first line survived into dispatch. `PMULL` pays when it replaces
eight bit-serial rounds; against `vqtbl1q_u8`'s five instructions it cannot
carry its twenty-instruction reduction network, and against the tower's
three-multiply identity it merely ties. The losing kernels were deleted, with
the numbers kept in `src/kernel/aarch64/`.

Everything else in the run matches within 3% — except rows of 16–128 B, where
dispatch-identical code varied by up to 1.36x between the two runs. Treat
anything measured on rows that small as noise unless it survives many runs.

Node 26 WASI (`wasm32-wasip1`, `simd128`, rustc 1.93), maximum of four
interleaved runs, GF(2^8) `xor`/`elementwise` as control (1.00x):

| Shape | Before | After |
| --- | --- | --- |
| gf16 `mul_add`, 4 KiB … 8 MiB | 4.5–5.4 GiB/s | 5.6–6.1 GiB/s (+13–24%) |
| gf16 `mul_assign`, 4 KiB … 8 MiB | 5.0–5.2 GiB/s | 6.0–6.2 GiB/s (+14–25%) |
| gf16 `scatter`, 16 rows, ≥ 256 B | 4.4–5.1 GiB/s | 9.8–11.1 GiB/s (~2.1x) |
| gf16 `gather`, 8 sources, ≥ 256 B | 4.5–5.4 GiB/s | 7.5–9.1 GiB/s (~1.7x) |
| gf16 `matrix`, 16x8, ≥ 256 B | 4.8–5.0 GiB/s | 6.2–7.9 GiB/s (1.3–1.6x) |

The multi-row factors come from hoisting the four-table derivation out of the
per-`(term, row)` loop and skipping zero/one coefficients; at 64 B rows the
result is inside Node's noise band. A two-lane unroll of the *GF(2^8)* wasm
kernels measured 1.00x and was therefore not kept: two swizzles per lane leave
no latency to hide, unlike GF(2^16)'s eight.

## Crossover and dispatch decisions

Several kernels are wired one way rather than another because of a
measurement. The code at each site states the decision and the reason; the
numbers behind it are here, so that re-deriving them is a matter of rerunning
a benchmark rather than trusting a comment.

Unless noted otherwise: Core Ultra 7 258V, Linux, rustc 1.93, one core,
maximum of interleaved runs with an unaffected operation as a 1.00x control.

### Non-temporal stores in `mul_into` (`kernel::x86::NT_STORE_MIN`)

`mul_into` never reads its destination, so an ordinary store pays a
read-for-ownership fetch of every line it fully overwrites. `vmovntdq` skips
it, at the price of evicting the destination. GF(2^8) `mul_into`, ordinary
stores → non-temporal:

| Destination | Write-only | Encode then read back |
| --- | --- | --- |
| 1 MiB | 32.3 → 61.2 GiB/s | 21.6 → 17.9 GiB/s |
| 2 MiB | 21.3 → 66.0 | 16.5 → 16.9 |
| 4 MiB | 21.5 → 37.8 | 14.5 → 16.1 |
| 16 MiB | 16.1 → 34.4 | 10.7 → 12.9 |
| 64 MiB | 14.2 → 23.0 | — |

2 MiB (this host's L3 is 12 MiB) is where the read-back workload stops losing
while the write-only one is already ~3x, which is why the threshold sits
there. The forced `avx2` and `ssse3` arms hit the same ~14 GiB/s ceiling at
16 MiB, so every backend is store-bound at that size, not multiply-bound.
`mul_assign` reads its destination anyway and measures 22.0 GiB/s either way
at 64 MiB, 0.5x at 256 KiB — it keeps ordinary stores.

**Not taken:** GF(2^16) on SSSE3. Eight `PSHUFB` per 16 bytes hold that loop
to ~5.9 GiB/s, well under the host's write bandwidth, and 16-byte
non-temporal stores from a slow loop flush write-combining buffers before a
line fills: 5.41/5.44 GiB/s ordinary against 4.84/4.79 non-temporal at
32 MiB. The GF(2^8) SSSE3 kernel does reach the ceiling (19.7 GiB/s) and does
use them.

### Destination alignment peel (`kernel::x86::peel_to_align`)

A 32-byte `vmovdqu` at an odd multiple of 32 straddles two cache lines, and a
multi-row body issues one load and one store per row per vector. On GF(2^8)
64 KiB rows the aligned form runs ~1.4x the misaligned one. Peeling at most
31 bytes per row group buys the aligned body for the rest of the pass. The
2 KiB floor comes from the original all-scalar GF(2^8) peel, which won from
about 2 KiB up and lost badly below 1 KiB.

Generalized to the GF(2^16) scatter kernels, normalized against untouched
GF(2^8) rows as a control, 8 rows, misaligned destination:

| Kernel | 64 KiB rows | 256 KiB rows |
| --- | --- | --- |
| gf16 scatter, GFNI | 1.30–1.36x | 1.23–1.30x |
| gf16 scatter, AVX2 | 1.03–1.11x | 1.03–1.11x |

An already-aligned destination is unchanged.

**Not taken:** the GF(2^16) matrix kernel (0.96–1.03x at 64 KiB, 0.93–1.01x
at 256 KiB — its row tile is stored once per term block, not per source
window) and the SSSE3 scatter (1–3% slower; 16-byte accesses never straddle a
line at the alignment allocators already give).

### Blocking against repeated AXPY

Register-blocked multi-row kernels are not universally better than dispatching
to repeated single-row AXPY, so dispatch picks per `(field, backend, shape)`.
Across 2–16 sources and 4–64 KiB rows, blocked against AXPY:

| Shape | Result | Wired to |
| --- | --- | --- |
| gf16 gather, GFNI | 1.03–1.59x | blocked |
| gf16 gather, SSSE3 | 1.5–1.8x | blocked |
| gf16 gather, AVX2 | 0.84–1.01x | AXPY |
| gf16 matrix, AVX2 | 0.95–1.20x | AXPY |
| gf8 matrix, AVX2 | +20% | blocked |

AVX2 loses on GF(2^16) because it has enough width but not enough registers to
retain several four-table coefficient sets, and because a gather's
coefficients are one-to-one with its sources — a source's nibble split feeds
exactly one coefficient, so there is nothing to share. SSSE3's smaller table
vectors fit. The AVX2 matrix wins only at or below ~8 KiB rows, which is not
enough to justify a row-length branch in dispatch. Before its broadcasts were
hoisted out of the byte loop, the GFNI gather ran at 0.29–0.47x, which is why
dispatch previously avoided it.

### Zero-coefficient skipping in the blocked GF(2^8) matrix kernel

Not done: coefficients have to reach a general-purpose register to be tested,
which stops each factor broadcast from folding into a memory-operand
`vpbroadcastb`. Over eight terms and 64 KiB rows the check cost ~9%. Sparsity
is handled in the scatter shape instead, which drops zero rows before grouping
and outside any loop.

### GFNI affine multiply for `Gf8D` (`kernel::x86::gf8::mul_*_affine`)

`GF2P8MULB` multiplies only in the AES field, so `Gf8D` (`0x11D`) uses
`VGF2P8AFFINEQB`, which applies an arbitrary 8×8 GF(2) linear map per byte
lane: fixed-coefficient multiplication in any GF(2^8), one instruction per 32
lanes. The maps are const-derived from `gf8d::Elem::mul` (see
`kernel::tables::affine_8d`), never copied from another library, and the
kernels are the `GF2P8MULB` loops with only the multiply instruction
substituted. The candidate ran against the AVX2 nibble shuffle a GFNI host
would otherwise dispatch to, with the native `Gf8B` `GF2P8MULB` loop as the
single-instruction control:

```sh
cargo bench --features internals --bench affine
```

Criterion 0.8, 100 samples per point (20 at 4 MiB), candidates interleaved
per size. GiB/s, mean of the estimate interval; ratio is affine ÷ shuffle.

`mul_add` (`dst ^= c * src`):

| Size | Shuffle AVX2 | Affine GFNI | Native `0x11B` | Ratio |
| --- | --- | --- | --- | --- |
| 64 B | 23.40 | 21.61 | 20.29 | 0.92 |
| 256 B | 43.31 | 53.02 | 57.24 | 1.22 |
| 1 KiB | 55.04 | 75.95 | 78.33 | 1.38 |
| 4 KiB | 56.99 | 118.00 | 120.17 | 2.07 |
| 16 KiB | 64.42 | 129.94 | 120.54 | 2.02 |
| 64 KiB | 54.08 | 59.36 | 61.45 | 1.10 |
| 256 KiB | 43.50 | 48.33 | 50.33 | 1.11 |
| 1 MiB | 34.35 | 36.20 | 36.11 | 1.05 |

`mul_assign` (`dst *= c`, single stream, in place):

| Size | Shuffle AVX2 | Affine GFNI | Native `0x11B` | Ratio |
| --- | --- | --- | --- | --- |
| 64 B | 21.51 | 21.75 | 22.03 | 1.01 |
| 256 B | 50.04 | 56.03 | 51.78 | 1.12 |
| 1 KiB | 69.08 | 84.73 | 86.06 | 1.23 |
| 4 KiB | 64.54 | 185.91 | 191.38 | 2.88 |
| 16 KiB | 72.32 | 81.08 | 80.96 | 1.12 |
| 64 KiB | 62.64 | 61.17 | 62.23 | 0.98 |
| 256 KiB | 59.55 | 50.23 | 51.11 | 0.84 |
| 1 MiB | 54.57 | 46.87 | 47.52 | 0.86 |

`mul_into` (`dst = c * src`, fused out-of-place):

| Size | Shuffle AVX2 | Affine GFNI | Native `0x11B` | Ratio |
| --- | --- | --- | --- | --- |
| 64 B | 24.82 | 23.63 | 25.95 | 0.95 |
| 256 B | 46.31 | 64.00 | 74.38 | 1.38 |
| 1 KiB | 63.86 | 81.16 | 77.14 | 1.27 |
| 4 KiB | 66.01 | 79.51 | 77.83 | 1.20 |
| 16 KiB | 75.87 | 79.69 | 79.97 | 1.05 |
| 64 KiB | 55.71 | 58.23 | 58.71 | 1.05 |
| 256 KiB | 42.99 | 55.99 | 55.78 | 1.30 |
| 1 MiB | 38.34 | 46.23 | 48.01 | 1.21 |
| 4 MiB | 36.60 | 35.72 | 34.49 | 0.98 |

Composed scatter (per-row `mul_add`), ratio only: 4×16 KiB 1.29, 16×16 KiB
1.02, 4×64 KiB 1.14, 16×64 KiB 1.20.

Decision: affine for `mul_add` and `mul_into` at every size, and for
`mul_assign` below 64 KiB; the register-blocked multi-row shapes build on the
affine `mul_add` (see the next section). The affine/native column holds
0.96–1.08 throughout — the map
reaches native-multiply speed, and the 64 B loss is a sub-nanosecond
small-buffer effect the native loop shares. `mul_assign` at and past 64 KiB
is the one measured exception: in-place scaling is single-stream, both
single-instruction forms fall ~15% behind the shuffle there (affine/native
0.98–0.99, so the cause is the host's store path, not the map), and dispatch
keeps the shuffle for it. At 4 MiB `mul_into` both candidates are
non-temporal-store-bound and even.

### Register-blocked multi-row for `Gf8D` (`scatter_affine`/`gather_affine`/`matrix_affine`)

The multi-row shapes were first composed as a per-row affine `mul_add`. Holding
a destination tile in registers across sources (gather) or terms (matrix), and
one source load across a row group (scatter), removes the redundant destination
traffic — the same blocking `Gf8B`'s GFNI kernels use. Both run
`VGF2P8AFFINEQB`; the strategy seam (`kernel::x86::gf8::Blocked`) monomorphizes
the `GF2P8MULB` and affine forms from one body, so `Gf8B` is byte-identical and
unchanged. `cargo bench --features internals --bench affine`, blocked ÷ per-row
affine:

| Shape | Small (4 KiB rows) | Mid (16 KiB) | Large (64 KiB) | 256 KiB |
| --- | --- | --- | --- | --- |
| gather, 4 sources | 1.74 | 0.99 | 1.25 | 1.36 |
| gather, 16 sources | 1.57 | 0.97 | 1.58 | 1.18 |
| scatter, 4 rows | — | 1.09 | 1.61 | — |
| scatter, 16 rows | — | 1.01 | 1.46 | — |
| matrix, 4 rows × 8 terms | 1.86 | 1.99 | 2.07 | — |
| matrix, 16 rows × 16 terms | 1.38 | 2.12 | 2.44 | — |

Decision: dispatch `Gf8D` scatter/gather/matrix (and the scattered matrix) to
the blocked affine kernels on a GFNI host. Matrix wins everywhere (1.38–2.44×,
largest where the term count is high and the destination reread dominates);
gather and scatter win at every size but the two 16 KiB gather points, which
are a wash (0.97–0.99) — a single L2-resident tile leaves nothing for blocking
to save there. Non-GFNI backends keep the per-row shuffle composition.

### `Gf8D` elementwise (`elementwise_avx2::<0x1d>`)

Two varying operands have no fixed coefficient, so `GF2P8MULB` is out even on a
GFNI host; the branchless eight-round shift/reduce vector multiply, with the
`0x11D` reduction byte threaded as a const generic, replaces the scalar
reference. 7.1–7.9 GiB/s against the reference's ~1.1 GiB/s — 6.2–7.1× from
64 B up, the win flat across sizes because the loop is compute-bound.

## Comparative benchmark

`benches/compare.rs` compares compatible GF(2^8) operations against
`reed-solomon-erasure` with its `simd-accel` feature. It is a development
comparison, not a claim that the crates expose identical abstractions. Run it
on the same machine and toolchain before quoting a ratio.

Historical measurements from local development are intentionally not copied
into the crate landing page: results without their original CPU and command
line are not reproducible evidence.
