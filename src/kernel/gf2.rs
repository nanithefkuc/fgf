//! Bit-packed GF(2) kernels over `&[u8]` buffers.
//!
//! Layout: element `i` is bit `i % 8` (LSB-first) of byte `i / 8`; words are
//! the little-endian `u64` assembly of those bytes, so element `i` is bit
//! `i % 64` of word `i / 64`. Bits past the logical length are padding and
//! are zero on every output.
//!
//! # Backend policy
//!
//! XOR of whole buffers is byte-local and layout-identical to the
//! field-independent dispatched byte XOR, so `xor` here reuses that
//! dispatched kernel — no second arithmetic core for one shape. Everything
//! else is the
//! portable `u64` word loop, and that is the *final* kernel for these shapes
//! today, not a placeholder: AND/XOR are memory-bandwidth-bound and the word
//! loop already saturates them under autovectorization, so a hand-written
//! vector body has to beat memory before it earns a line here. The
//! compute-bound shapes (`weight`, `parity`) gain `VPOPCNTQ`/NEON `CNT`
//! bodies only under an AVX-512-tier story that `FGF_TIERS` does not expose
//! yet. The dispatch seam for both is this module: backend-specific arms
//! attach beside the portable bodies, never in [`crate::bits`].
//!
//! Preconditions (length pairing, `from <= to <= bits`, buffer coverage) are
//! checked by [`crate::bits`]; only `debug_assert`s remain here.

/// `dst ^= src` over whole buffers.
///
/// Reuses the field-independent dispatched byte XOR — bit-packed GF(2)
/// addition over equal-length buffers is byte XOR regardless of packing.
pub(crate) fn xor(dst: &mut [u8], src: &[u8]) {
    super::xor_impl(dst, src);
}

/// `dst = a & b`, elementwise GF(2) multiplication.
pub(crate) fn and_into(dst: &mut [u8], a: &[u8], b: &[u8]) {
    debug_assert!(dst.len() == a.len() && a.len() == b.len());

    let mut dst_chunks = dst.chunks_exact_mut(8);
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for ((d, x), y) in dst_chunks
        .by_ref()
        .zip(a_chunks.by_ref())
        .zip(b_chunks.by_ref())
    {
        let mixed =
            u64::from_ne_bytes(x.try_into().unwrap()) & u64::from_ne_bytes(y.try_into().unwrap());
        d.copy_from_slice(&mixed.to_ne_bytes());
    }
    for ((d, &x), &y) in dst_chunks
        .into_remainder()
        .iter_mut()
        .zip(a_chunks.remainder())
        .zip(b_chunks.remainder())
    {
        *d = x & y;
    }
}

/// `dst &= src`, in-place elementwise multiplication.
pub(crate) fn and_assign(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());

    let mut dst_chunks = dst.chunks_exact_mut(8);
    let mut src_chunks = src.chunks_exact(8);
    for (d, s) in dst_chunks.by_ref().zip(src_chunks.by_ref()) {
        let mixed =
            u64::from_ne_bytes(d.try_into().unwrap()) & u64::from_ne_bytes(s.try_into().unwrap());
        d.copy_from_slice(&mixed.to_ne_bytes());
    }
    for (d, &s) in dst_chunks
        .into_remainder()
        .iter_mut()
        .zip(src_chunks.remainder())
    {
        *d &= s;
    }
}

/// `dst &= !mask`: clears every bit of `dst` where `mask` has a 1.
pub(crate) fn andnot_assign(dst: &mut [u8], mask: &[u8]) {
    debug_assert_eq!(dst.len(), mask.len());

    let mut dst_chunks = dst.chunks_exact_mut(8);
    let mut mask_chunks = mask.chunks_exact(8);
    for (d, m) in dst_chunks.by_ref().zip(mask_chunks.by_ref()) {
        let mixed =
            u64::from_ne_bytes(d.try_into().unwrap()) & !u64::from_ne_bytes(m.try_into().unwrap());
        d.copy_from_slice(&mixed.to_ne_bytes());
    }
    for (d, &m) in dst_chunks
        .into_remainder()
        .iter_mut()
        .zip(mask_chunks.remainder())
    {
        *d &= !m;
    }
}

/// The byte window of a bit range `[from, to)`: derived once, applied to
/// many buffer pairs by [`xor_window`]. `end == 0` marks the empty range.
///
/// The window is byte-granular — `start` holds the low-masked `lead` bits,
/// `end - 1` the `trail` bits, and everything between is fully live — so
/// it never runs past a buffer that covers the range, and sub-word ranges
/// (the panel-elimination shape) are one or two masked byte ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Window {
    /// First byte of the window.
    pub(crate) start: usize,
    /// One past the last byte of the window; `0` for an empty range.
    pub(crate) end: usize,
    /// Live-bit mask of byte `start`.
    pub(crate) lead: u8,
    /// Live-bit mask of byte `end - 1` (folded into `lead` when the
    /// window is one byte wide).
    pub(crate) trail: u8,
}

/// Derives the [`Window`] of bits `[from, to)`. An empty or inverted range
/// yields the empty window.
pub(crate) fn window(from: usize, to: usize) -> Window {
    if from >= to {
        return Window {
            start: 0,
            end: 0,
            lead: 0,
            trail: 0,
        };
    }
    let start = from / 8;
    let last = (to - 1) / 8;
    let span = to - last * 8; // 1..=8 live bits in the last byte.
    let trail = if span == 8 { 0xFF } else { (1u8 << span) - 1 };
    let mut lead = 0xFFu8 << (from - start * 8);
    if start == last {
        lead &= trail;
    }
    Window {
        start,
        end: last + 1,
        lead,
        trail,
    }
}

/// `dst ^= src` over a derived [`Window`]: the two masked end bytes as
/// scalars, the fully-live interior through the dispatched whole-buffer
/// byte XOR (whose short-buffer path inlines). Both buffers must cover
/// the window — the checked surface guarantees it.
pub(crate) fn xor_window(dst: &mut [u8], src: &[u8], w: &Window) {
    if w.start >= w.end {
        return;
    }
    debug_assert!(w.end <= dst.len() && w.end <= src.len());
    dst[w.start] ^= src[w.start] & w.lead;
    let last = w.end - 1;
    if last > w.start {
        dst[last] ^= src[last] & w.trail;
        let mid = w.start + 1..last;
        if !mid.is_empty() {
            super::xor_impl(&mut dst[mid.clone()], &src[mid]);
        }
    }
}

/// Zeros bits `[from, to)` of `dst`.
///
/// Interior words are fully live and become one bulk fill; only the two
/// end words are masked scalars.
pub(crate) fn clear_range(dst: &mut [u8], from: usize, to: usize) {
    if from >= to {
        return;
    }
    let (first, last) = (from / 64, (to - 1) / 64);
    if first == last {
        read_modify_word(
            dst,
            first,
            mask_between(from - first * 64, to - first * 64),
            false,
        );
        return;
    }
    read_modify_word(dst, first, u64::MAX << (from - first * 64), false);
    dst[(first + 1) * 8..last * 8].fill(0);
    read_modify_word(dst, last, low_mask(to - last * 64), false);
}

/// Sets bits `[from, to)` of `dst` to one.
///
/// Interior words are fully live and become one bulk fill; only the two
/// end words are masked scalars.
pub(crate) fn set_range(dst: &mut [u8], from: usize, to: usize) {
    if from >= to {
        return;
    }
    let (first, last) = (from / 64, (to - 1) / 64);
    if first == last {
        read_modify_word(
            dst,
            first,
            mask_between(from - first * 64, to - first * 64),
            true,
        );
        return;
    }
    read_modify_word(dst, first, u64::MAX << (from - first * 64), true);
    dst[(first + 1) * 8..last * 8].fill(u8::MAX);
    read_modify_word(dst, last, low_mask(to - last * 64), true);
}

/// Hamming weight of the live bits `[0, bits)`.
///
/// The fold runs eight independent popcount accumulators: a single
/// `total += word.count_ones()` chain pins the loop to `popcnt` latency,
/// while split accumulators saturate its throughput (measured 1.1–1.9x
/// across cache tiers on the reference host; see BENCHMARKS.md).
pub(crate) fn weight(buf: &[u8], bits: usize) -> u32 {
    let words = bits / 64;
    let mut w = buf.chunks_exact(8);
    let mut acc = [0u32; 8];
    for _ in 0..words / 8 {
        for lane in &mut acc {
            *lane += next_word(&mut w).count_ones();
        }
    }
    let mut total = acc.iter().sum::<u32>();
    for _ in words / 8 * 8..words {
        total += next_word(&mut w).count_ones();
    }
    let rem = bits % 64;
    if rem != 0 {
        total += partial_word(buf, bits / 64, low_mask(rem)).count_ones();
    }
    total
}

/// Parity of the GF(2) inner product over the live bits: 0 or 1.
///
/// Four independent XOR accumulators, folded once at the end: a single
/// accumulator serializes every AND-XOR on one register and the loop runs
/// at dependency latency instead of port throughput (measured 1.2–1.4x
/// across cache tiers on the reference host; see BENCHMARKS.md).
pub(crate) fn parity(a: &[u8], b: &[u8], bits: usize) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let words = bits / 64;
    let mut wa = a.chunks_exact(8);
    let mut wb = b.chunks_exact(8);
    let (mut a0, mut a1, mut a2, mut a3) = (0u64, 0u64, 0u64, 0u64);
    for _ in 0..words / 4 {
        a0 ^= next_word(&mut wa) & next_word(&mut wb);
        a1 ^= next_word(&mut wa) & next_word(&mut wb);
        a2 ^= next_word(&mut wa) & next_word(&mut wb);
        a3 ^= next_word(&mut wa) & next_word(&mut wb);
    }
    let mut acc = a0 ^ a1 ^ a2 ^ a3;
    for _ in words / 4 * 4..words {
        acc ^= next_word(&mut wa) & next_word(&mut wb);
    }
    let rem = bits % 64;
    if rem != 0 {
        let live = low_mask(rem);
        acc ^= partial_word(a, bits / 64, live) & partial_word(b, bits / 64, live);
    }
    acc.count_ones() & 1
}

pub(crate) fn xor_gather(dst: &mut [u8], srcs: &[&[u8]], mut selector: u64) {
    debug_assert!(srcs.len() >= 64 || selector >> srcs.len() == 0);
    while selector != 0 {
        let index = selector.trailing_zeros() as usize;
        xor(dst, srcs[index]);
        selector &= selector - 1;
    }
}

/// The next whole little-endian word of a chunk walk.
#[inline]
fn next_word(words: &mut core::slice::ChunksExact<'_, u8>) -> u64 {
    u64::from_ne_bytes(words.next().unwrap().try_into().unwrap())
}

/// Clears (`set == false`) or sets (`set == true`) the live bits of word `w`
/// of `dst`; a word running past the end of the buffer falls back to masked
/// bytes.
#[inline]
fn read_modify_word(dst: &mut [u8], w: usize, live: u64, set: bool) {
    let offset = 8 * w;
    if offset + 8 <= dst.len() {
        let mut d = u64::from_le_bytes(dst[offset..offset + 8].try_into().unwrap());
        d = if set { d | live } else { d & !live };
        dst[offset..offset + 8].copy_from_slice(&d.to_le_bytes());
    } else {
        #[allow(clippy::cast_possible_truncation)] // extracting the low byte
        for b in 0..dst.len() - offset {
            let byte_live = (live >> (8 * b)) as u8;
            if set {
                dst[offset + b] |= byte_live;
            } else {
                dst[offset + b] &= !byte_live;
            }
        }
    }
}

/// The little-endian word at byte offset `8 * start`, assembled from however
/// many bytes remain (fewer than eight near the end of the buffer) and
/// masked to `live`.
fn partial_word(buf: &[u8], start: usize, live: u64) -> u64 {
    let mut word = 0u64;
    for (shift, &byte) in buf[8 * start..].iter().take(8).enumerate() {
        word |= u64::from(byte) << (8 * shift);
    }
    word & live
}

/// The low `n` bits set, `0 <= n <= 64`.
#[inline]
fn low_mask(n: usize) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// Bits `[lo, hi)` set within a single word, `0 <= lo < hi <= 64`.
#[inline]
fn mask_between(lo: usize, hi: usize) -> u64 {
    (u64::MAX << lo) & low_mask(hi)
}
