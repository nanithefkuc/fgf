//! Routing thresholds and geometry selectors.
//!
//! Every constant and inline gate that chooses between kernels — by
//! destination length, row shape, tier, or disposition — lives here, so the
//! routing surface is one module rather than a fact spread over the
//! families. Kernel tuning parameters that shape one body's inner loops
//! (tile widths, unroll counts, prefetch distances, block sizes) stay with
//! their kernels.
//!
//! A threshold's doc names the routes it selects between. Each value set by
//! a paired measurement cites the record cell that measured it; a value
//! without a link is an uncited threshold whose record link is added by the
//! measurement that owns it.
//!
//! # Resolution sites
//!
//! Backend policy resolves exactly once per process, in the engine's
//! `ResolvedBackend` next to the `LazyLock` in [`crate::kernel`]: tier
//! selection, the per-family strategy enums, and the capability token.
//! Every other place that selects a kernel reads an already-resolved
//! strategy or proof, never the process backend:
//!
//! - `crate::kernel`'s strategy accessors hand each family its cached
//!   strategy enum, and its proof accessors narrow the cached token.
//! - `crate::kernel::bytes` matches the cached `X86Proof` for the
//!   field-independent byte kernels.
//! - `crate::kernel::gf8` folds its representation's const descriptor facts
//!   (polynomial-root presentation) into the cached byte-field strategy.
//! - `crate::kernel::tower` folds each row's pinned-instance descriptor
//!   facts into the cached tower strategies; a prepared coefficient carries
//!   the coefficient-dependent choice its `prepare` made.
//! - `crate::kernel::mersenne31` and `crate::kernel::goldilocks` match the
//!   cached prime-lane strategy through one shared generator.
//! - `crate::kernel::quad_mersenne31` matches the cached `QuadMersenne31`
//!   strategy against the vector floors in this module.
//! - `crate::kernel::residue` matches the selectors in this module, which
//!   fold call geometry into the cached residue strategy.
//! - The public `backend` queries on the dispatch contracts map a cached
//!   strategy to its `Backend` label; they answer reports and never run on
//!   a row operation's behalf.

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
use crate::kernel::{ResidueStrategy, residue_strategy};

// ---------------------------------------------------------------------------
// Byte and bit-buffer thresholds.
// ---------------------------------------------------------------------------

/// Buffers at most this long skip the dispatched SIMD XOR for the inline
/// portable one: below a single vector the out-of-line SIMD entry boundary
/// costs more than the body saves, and short GF(2) rows (eight elements per
/// byte) sit almost entirely under it.
pub(super) const XOR_INLINE_MAX: usize = 31;

/// Shortest destination a fused overwrite serves with non-temporal stores.
///
/// `mul_into` writes a destination it never reads, so an ordinary store
/// pays a read-for-ownership fetch of every line it is about to overwrite
/// completely. `vmovntdq` skips that fetch and the allocation, at the
/// price of evicting the destination from cache.
///
/// Non-temporal stores evict the written destination from cache, so a
/// subsequent read pays to fetch it again. `mul_add` and `mul_assign` read
/// their destination and keep ordinary stores.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(super) const NT_STORE_MIN: usize = 2 << 20;

/// Shortest 32-byte-lane destination that peels a 16-byte half-lane head.
///
/// A 256-bit store starting 16 bytes into a cache line splits it, and a
/// body of split stores increases line traffic; one narrow head puts every
/// remaining lane on a boundary. The paired measurement is the 3.0.2
/// record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/gf8.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(super) const HALF_LANE_PEEL_MIN: usize = 512;

/// Shortest row a 32-byte alignment peel serves.
///
/// The peel runs the narrow single-row kernels, which descend to 128-bit
/// lanes and then scalar code; below this floor the peel costs more than
/// the split-line loads it removes. The crossover is the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/gf8.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(super) const ALIGN_PEEL_MIN: usize = 16 * 128;

/// Shortest destination the 64-byte byte XOR serves.
///
/// Shorter destinations run [`crate::kernel::x86::bytes::xor_avx2`], which
/// covers them in one out-of-line call.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
pub(super) const XOR_AVX512_MIN: usize = 256;

/// Shortest XOR destination that peels its head to a 64-byte boundary.
///
/// The peel is one out-of-line AVX2 call, which short buffers do not
/// repay; from this floor head bytes up to the destination's first 64-byte
/// boundary run there so every 64-byte access on the destination stays
/// within one cache line.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
pub(super) const XOR_PEEL_MIN: usize = 512;

// ---------------------------------------------------------------------------
// GF(2^8) thresholds.
// ---------------------------------------------------------------------------

/// Shortest destination the nibble-shuffle in-place scale serves on a
/// non-AES representation.
///
/// The affine in-place GFNI form serves shorter buffers; the native AES
/// multiply stays on the GFNI body at every size.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(super) const GF8_NIBBLE_MUL_ASSIGN_MIN: usize = 65_536;

/// Shortest matrix row using chunk-resolved matrix coefficients.
///
/// Shorter rows read coefficients directly in each tile or lane. The
/// crossover is a measured parameter; public matrix shapes appear in the
/// 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/gf8.md>.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
pub(super) const MATRIX_RESOLVE_MIN: usize = 512;

/// Shortest matrix row that peels its head to a 64-byte boundary.
///
/// The peel runs one sub-lane AXPY per row per term, so it repays only
/// once the tile body is long enough.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
pub(super) const MATRIX_PEEL_MIN: usize = 8192;

/// Shortest scatter row that peels its head to a 64-byte boundary.
///
/// Below this the scalar/AVX2 head costs more than the split-line loads it
/// removes.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
pub(super) const SCATTER_PEEL_MIN: usize = 512;

/// Shortest gather destination that peels its head to a 64-byte boundary.
///
/// The peel runs one sub-lane AXPY per source, so it repays only once the
/// tile body is long enough.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
pub(super) const GATHER_PEEL_MIN: usize = 3072;

// ---------------------------------------------------------------------------
// GF(2^16) thresholds.
// ---------------------------------------------------------------------------

/// Shortest GF(2^16) row that peels its head to a 64-byte boundary.
///
/// Head bytes run the 32-byte body, which short rows do not repay. The
/// paired measurement is the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/gf16.md>.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
pub(super) const MUL_ADD_PEEL_MIN: usize = 3584;

// ---------------------------------------------------------------------------
// QuadMersenne31 thresholds.
// ---------------------------------------------------------------------------

/// Row length in bytes at or above which the vector multiplies take over;
/// one full vector of four complex elements.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(super) const QUAD_VECTOR_MUL_MIN_BYTES: usize = 32;

/// Row length in bytes at or above which the vector add and sub take over;
/// the mixed vector-plus-tail rows below this stay scalar.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub(super) const QUAD_VECTOR_ADD_MIN_BYTES: usize = 64;

// ---------------------------------------------------------------------------
// Residue selectors.
// ---------------------------------------------------------------------------

/// Whether this row's prepared fixed-coefficient path can vectorize:
/// byte rows at `P <= 127`, any four-byte row, eight-byte odd rows.
/// This predicate only names the classes the kernels cover. The routing
/// evidence is the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
/// The benchmark routing report calls this instead of copying the class list.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn routes_vector(bytes: usize, p: u64) -> bool {
    (bytes == 1 && p <= 127) || bytes == 4 || (bytes == 8 && p % 2 == 1)
}

/// Shortest AVX2 overwrite row, in lanes, for this width class.
/// The paired measurement is the 3.0.2 record at <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
#[allow(clippy::match_same_arms)]
pub const fn vector_floor_into_lanes(bytes: usize) -> usize {
    match bytes {
        1 => 64,
        8 => 8,
        _ => 8,
    }
}

/// Shortest AVX2 in-place-scale row, in lanes, for this width class.
/// The paired measurement is the 3.0.2 record at <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn vector_floor_assign_lanes(bytes: usize) -> usize {
    match bytes {
        1 => 64,
        8 => 16,
        _ => 8,
    }
}

/// Shortest AVX2 accumulation row, in lanes, for this width class.
/// The paired measurement is the 3.0.2 record at <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
#[allow(clippy::match_same_arms)]
pub const fn vector_floor_add_lanes(bytes: usize) -> usize {
    match bytes {
        1 => 8,
        8 => 32,
        _ => 8,
    }
}

/// The single-row and elementwise operation the routing floor selects.
/// `Assign` is the in-place scale; `Elementwise` and `ElementwiseAssign`
/// are the raw-word element products. Dispatch matches on this instead of
/// threading positional flags through every selector call.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorOp {
    /// Overwrite: `mul_into`.
    Into,
    /// Accumulation: `mul_add`.
    Add,
    /// In-place scale: `mul_assign`.
    Assign,
    /// Raw-word product: `mul_elementwise`.
    Elementwise,
    /// In-place raw-word product: `mul_elementwise_assign`.
    ElementwiseAssign,
}

/// Shortest AVX2 elementwise row, in lanes, for this operation.
/// The paired measurement is the 3.0.2 record at <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn vector_floor_elementwise_lanes(op: VectorOp) -> usize {
    match op {
        VectorOp::Elementwise => 16,
        VectorOp::ElementwiseAssign => 8,
        VectorOp::Into | VectorOp::Add | VectorOp::Assign => {
            panic!("elementwise lanes have no single-row operation")
        }
    }
}

/// Shortest AVX2 row, in bytes, for this width class and operation.
/// The paired measurement is the 3.0.2 record at <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn avx2_floor_op_len(bytes: usize, op: VectorOp) -> usize {
    let lanes = match op {
        VectorOp::Add => vector_floor_add_lanes(bytes),
        VectorOp::Assign => vector_floor_assign_lanes(bytes),
        VectorOp::Into => vector_floor_into_lanes(bytes),
        VectorOp::Elementwise | VectorOp::ElementwiseAssign => vector_floor_elementwise_lanes(op),
    };
    lanes * bytes
}

/// Shortest row the AVX-512 single-row kernels serve, in bytes, for this
/// width class and operation. The paired measurement is the 3.0.2 record
/// at <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd512", target_arch = "x86_64"))]
#[must_use]
pub const fn avx512_floor_op_len(bytes: usize, op: VectorOp) -> usize {
    match (bytes, op) {
        (4, VectorOp::Assign) => 1024,
        (8, VectorOp::Assign) => 256,
        _ => 64,
    }
}

/// Select the single-row kernel class for this geometry and operation.
///
/// AVX-512 requires `simd512` and a capable process backend. Shorter rows
/// retain their AVX2 or portable route. Elementwise vectors require u32
/// lanes. Coefficient zero/unit shortcuts apply before this selector.
/// Routing evidence is the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
#[inline]
pub fn single_row_backend(bytes: usize, p: u64, len: usize, op: VectorOp) -> ResidueStrategy {
    if !routes_vector(bytes, p)
        || (matches!(op, VectorOp::Elementwise | VectorOp::ElementwiseAssign) && bytes != 4)
    {
        return ResidueStrategy::Scalar;
    }
    let tier = residue_strategy();
    if tier == ResidueStrategy::Scalar {
        return ResidueStrategy::Scalar;
    }
    #[cfg(feature = "simd512")]
    {
        if tier == ResidueStrategy::Avx512 && len >= avx512_floor_op_len(bytes, op) {
            return ResidueStrategy::Avx512;
        }
    }
    if len >= avx2_floor_op_len(bytes, op) {
        ResidueStrategy::Avx2
    } else {
        ResidueStrategy::Scalar
    }
}

/// Sources per delayed group: `K = floor(2^32 / P)`, the largest count
/// whose 64-bit products stay below `2^32 * P`. Four-byte rows hold
/// `P < 2^32`, so the quotient fits the delay depth. The benchmark
/// routing report calls this instead of recomputing the depth.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub const fn delay_depth<const P: u64>() -> u64 {
    ((1u128 << 32) / P as u128) as u64
}

/// Fewest sources routed into the delayed tile, set by the paired
/// measurement in the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub const DELAYED_MIN_SOURCES: usize = 8;

/// Smallest delay depth routed for overwrite, set by the paired measurement
/// in the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub const DELAYED_INTO_MIN_DEPTH: u64 = 4;

/// Smallest delay depth routed for accumulation, set by the paired
/// measurement in the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub const DELAYED_ADD_MIN_DEPTH: u64 = 4;

/// Output rows one delayed tile covers.
///
/// The delayed kernels accumulate a fixed four-output group; the paired
/// measurement is the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub const DELAYED_TILE_ROWS: usize = 4;

/// Shortest row a delayed tile serves, in bytes, set by the paired
/// measurement in the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub const DELAYED_MIN_ROW_BYTES: usize = 256;

/// Whether the delayed tile covers this call: exactly
/// `DELAYED_TILE_ROWS` outputs, rows of at least
/// `DELAYED_MIN_ROW_BYTES` bytes, a depth at or above the disposition's
/// floor in [`DELAYED_INTO_MIN_DEPTH`] or [`DELAYED_ADD_MIN_DEPTH`], and at
/// least [`DELAYED_MIN_SOURCES`] sources, with the accumulation composition
/// check at the call site. Shallower depths, fewer sources, and every
/// other shape stay on the repeated single-row path. The routing evidence
/// is the 3.0.2 record at
/// <https://github.com/nanithefkuc/fgf/blob/v3.0.2/benchmarks/v3/residue.md>.
/// The benchmark routing report calls this instead of copying the floors.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
pub const fn routes_delayed(
    nrows: usize,
    k: u64,
    row_len: usize,
    nterms: usize,
    add: bool,
) -> bool {
    let floor = if add {
        DELAYED_ADD_MIN_DEPTH
    } else {
        DELAYED_INTO_MIN_DEPTH
    };
    nrows == DELAYED_TILE_ROWS
        && k >= floor
        && row_len >= DELAYED_MIN_ROW_BYTES
        && nterms >= DELAYED_MIN_SOURCES
}

/// Select the delayed-matrix kernel class for this geometry and
/// disposition.
///
/// `Scalar` selects the repeated single-row composition instead of a
/// delayed tile. Accumulation additionally checks byte-preserving zero
/// coefficients at the call site. The benchmark report uses this same
/// selector.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[must_use]
#[inline]
pub fn delayed_matrix_backend(
    nrows: usize,
    k: u64,
    row_len: usize,
    nterms: usize,
    add: bool,
) -> ResidueStrategy {
    if routes_delayed(nrows, k, row_len, nterms, add) {
        residue_strategy()
    } else {
        ResidueStrategy::Scalar
    }
}
