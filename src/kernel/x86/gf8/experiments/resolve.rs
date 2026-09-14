//! Pricing per-call coefficient resolution on the resolved matrix path.
//!
//! Measurement scaffolding. Resolution is moved out of the timed region,
//! chunked at selectable widths, or run alone with the tile loops removed,
//! so a pair of these prices resolution, the destination pass structure, or
//! the scratch policy on its own.

use super::super::rows::{RESOLVE_CHUNK, rows_body, rows_resolved_chunked};
use super::super::{Affine8D, factor_word};
use super::rows_external_chunked;
use crate::field::gf8d;

/// Experimental one-row overwrite matrix running the production body over
/// coefficients resolved *outside* the call.
///
/// The body, tile width, lane count and store policy are the production
/// path's; only the resolution moves out of the timed region. Paired against
/// [`mul_into_matrix1_tile_8d`](super::tile::mul_into_matrix1_tile_8d) at the same length this prices per-call
/// coefficient resolution, with nothing else varying.
///
/// `maps` holds one `VGF2P8AFFINEQB` map qword per source, in source order.
/// The sub-32-byte remainder is *not* handled: `row_len` must be a 32-byte
/// multiple, matching what the pairing needs and what ISA-L's explicit
/// kernels contract.
///
/// # Panics
/// Panics unless `row_len` is a 32-byte multiple, `rows` holds one
/// `row_len`-byte row, `maps` covers every source, every source spans
/// `row_len` bytes, and `lanes` is 3 or 4.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix1_external_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    maps: &[u64],
    srcs: &[&[u8]],
    lanes: usize,
) {
    assert_eq!(
        row_len % 32,
        0,
        "matrix_overwrite1_external: needs a 32-byte-multiple row"
    );
    assert!(
        row_len <= rows.len(),
        "matrix_overwrite1_external: rows buffer does not hold one row of {row_len} bytes"
    );
    assert_eq!(
        maps.len(),
        srcs.len(),
        "matrix_overwrite1_external: one resolved map per source"
    );
    for src in srcs {
        assert_eq!(src.len(), row_len);
    }
    // The body reads `[u64; 1]` rows; a flat map slice has that layout.
    let (chunks, rest) = maps.as_chunks::<1>();
    debug_assert!(rest.is_empty());
    let ptr = rows.as_mut_ptr();
    // One in-bounds row of `row_len` bytes, as asserted above; the map and
    // source lists cover the same term count.
    match lanes {
        3 => rows_body::<Affine8D, 1, 3, true>([ptr], row_len, chunks, srcs),
        4 => rows_body::<Affine8D, 1, 4, true>([ptr], row_len, chunks, srcs),
        _ => panic!("matrix_overwrite1_external: lanes must be 3 or 4"),
    }
}

/// Resolve a one-row group's coefficients exactly as the production path
/// does — chunked at the production resolve chunk, scratch sized to the occupied prefix —
/// touch no destination, and return a checksum of the resolved state.
///
/// This is the resolution half of the resolved one-row path with the tile
/// loops removed, for pricing resolution on its own. The checksum exists so
/// the work cannot be optimized away; its value is not a contract. Pure book
/// keeping over the term list: no CPU feature is involved, so no token is
/// taken.
///
/// The `MaybeUninit` staging arrays are the residue this function keeps.
///
/// # Panics
/// Panics unless every term supplies at least one coefficient.
#[must_use]
#[allow(unsafe_code)]
pub fn resolve_probe_8d(terms: &[(&[gf8d::Elem], &[u8])]) -> u64 {
    for (coeffs, _) in terms {
        assert!(
            !coeffs.is_empty(),
            "resolve_probe: term needs one coefficient"
        );
    }
    let count = terms.len();
    let mut sink = 0u64;
    let mut start = 0;
    loop {
        let taken = (count - start).min(RESOLVE_CHUNK);
        // The arrays live in the loop body's scope so the slices below
        // reference storage that outlives their consumption; only the
        // occupied prefix is written, and the checksum reads exactly that
        // prefix.
        // SAFETY:
        // INITIALIZATION
        // SINCE: `MaybeUninit` has no validity invariant, so an array of it
        //        may be assumed initialized, and exactly `taken` slots are
        //        written below before the prefix is reinterpreted.
        // THUS: forming the uninitialized staging arrays reads no invalid
        //       value.
        let mut maps: [core::mem::MaybeUninit<[u64; 1]>; RESOLVE_CHUNK] =
            unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        let mut srcs: [core::mem::MaybeUninit<&[u8]>; RESOLVE_CHUNK] =
            unsafe { core::mem::MaybeUninit::uninit().assume_init() };
        for offset in 0..taken {
            let (coeffs, source) = terms[start + offset];
            maps[offset].write([factor_word::<Affine8D>(coeffs[0])]);
            srcs[offset].write(source);
        }
        // SAFETY:
        // INITIALIZATION
        // SINCE: the slices cover only the `taken` initialized slots.
        // THUS: the reinterpretation exposes no unwritten slot.
        let (maps, srcs): (&[[u64; 1]], &[&[u8]]) = unsafe {
            (
                core::slice::from_raw_parts(maps.as_ptr().cast::<[u64; 1]>(), taken),
                core::slice::from_raw_parts(srcs.as_ptr().cast::<&[u8]>(), taken),
            )
        };
        for (words, src) in maps.iter().zip(srcs) {
            sink ^= words[0] ^ src.as_ptr() as u64;
        }
        start += taken;
        if start >= count {
            return sink;
        }
    }
}

/// Experimental one-row overwrite matrix on the production resolved path with
/// the *shipped* scratch policy replaced by the original full-array
/// initialization.
///
/// Identical to [`mul_into_matrix1_tile_8d`](super::tile::mul_into_matrix1_tile_8d) in resolution order, tile width,
/// remainder and checks; only the scratch initialization differs, so the pair
/// prices that policy alone. Benchmark evidence only; production dispatch is
/// unchanged.
///
/// # Panics
/// Panics unless `rows` holds one `row_len`-byte row, every term supplies at
/// least one coefficient over a `row_len`-byte source, and `lanes` is 3 or 4.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix1_fullinit_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    terms: &[(&[gf8d::Elem], &[u8])],
    lanes: usize,
) {
    assert!(
        row_len <= rows.len(),
        "matrix_overwrite1_fullinit: rows buffer does not hold one row of {row_len} bytes"
    );
    for (coeffs, src) in terms {
        assert_eq!(src.len(), row_len);
        assert!(
            !coeffs.is_empty(),
            "matrix_overwrite1_fullinit: term needs one coefficient"
        );
    }
    let ptr = rows.as_mut_ptr();
    // One in-bounds row of `row_len` bytes, as asserted above; every term
    // covers one coefficient over a `row_len`-byte source.
    match lanes {
        3 => rows_resolved_chunked::<
            Affine8D,
            [(&[gf8d::Elem], &[u8])],
            1,
            3,
            true,
            RESOLVE_CHUNK,
            true,
        >([ptr], row_len, 0, terms),
        4 => rows_resolved_chunked::<
            Affine8D,
            [(&[gf8d::Elem], &[u8])],
            1,
            4,
            true,
            RESOLVE_CHUNK,
            true,
        >([ptr], row_len, 0, terms),
        _ => panic!("matrix_overwrite1_fullinit: lanes must be 3 or 4"),
    }
}

/// Experimental one-row overwrite matrix on the production resolved path with
/// a selectable resolve-chunk width.
///
/// Same resolution, tile width, remainder and checks as
/// [`mul_into_matrix1_tile_8d`](super::tile::mul_into_matrix1_tile_8d); only the chunk width (and therefore the
/// destination pass count above 32 terms) differs. `chunk` 32 is production
/// by construction. Benchmark evidence only; production dispatch is unchanged.
///
/// # Panics
/// Panics unless `rows` holds one `row_len`-byte row, every term supplies at
/// least one coefficient over a `row_len`-byte source, `lanes` is 3 or 4, and
/// `chunk` is 32, 64 or 96.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix1_chunk_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    terms: &[(&[gf8d::Elem], &[u8])],
    lanes: usize,
    chunk: usize,
) {
    assert!(
        row_len <= rows.len(),
        "matrix_overwrite1_chunk: rows buffer does not hold one row of {row_len} bytes"
    );
    for (coeffs, src) in terms {
        assert_eq!(src.len(), row_len);
        assert!(
            !coeffs.is_empty(),
            "matrix_overwrite1_chunk: term needs one coefficient"
        );
    }
    let ptr = rows.as_mut_ptr();
    // One in-bounds row of `row_len` bytes, as asserted above; every term
    // covers one coefficient over a `row_len`-byte source.
    match (lanes, chunk) {
        (3, 32) => {
            rows_resolved_chunked::<Affine8D, [(&[gf8d::Elem], &[u8])], 1, 3, true, 32, false>(
                [ptr],
                row_len,
                0,
                terms,
            );
        }
        (4, 32) => {
            rows_resolved_chunked::<Affine8D, [(&[gf8d::Elem], &[u8])], 1, 4, true, 32, false>(
                [ptr],
                row_len,
                0,
                terms,
            );
        }
        (3, 64) => {
            rows_resolved_chunked::<Affine8D, [(&[gf8d::Elem], &[u8])], 1, 3, true, 64, false>(
                [ptr],
                row_len,
                0,
                terms,
            );
        }
        (4, 64) => {
            rows_resolved_chunked::<Affine8D, [(&[gf8d::Elem], &[u8])], 1, 4, true, 64, false>(
                [ptr],
                row_len,
                0,
                terms,
            );
        }
        (3, 96) => {
            rows_resolved_chunked::<Affine8D, [(&[gf8d::Elem], &[u8])], 1, 3, true, 96, false>(
                [ptr],
                row_len,
                0,
                terms,
            );
        }
        (4, 96) => {
            rows_resolved_chunked::<Affine8D, [(&[gf8d::Elem], &[u8])], 1, 4, true, 96, false>(
                [ptr],
                row_len,
                0,
                terms,
            );
        }
        _ => panic!("matrix_overwrite1_chunk: lanes must be 3 or 4, chunk 32/64/96"),
    }
}

/// Experimental one-row overwrite matrix running the production body over
/// externally resolved map words, in chunks of `chunk` terms.
///
/// [`mul_into_matrix1_external_8d`] is the `chunk == 0` single-pass case; a
/// finite chunk adds the destination pass structure of the resolved path
/// without its per-call resolution, so resolved-chunked versus
/// external-chunked at the same chunk prices resolution alone, and
/// external-chunked versus external single-pass prices the passes alone.
/// Benchmark evidence only; production dispatch is unchanged.
///
/// # Panics
/// Panics unless `row_len` is a 32-byte multiple, `rows` holds one
/// `row_len`-byte row, `maps` covers every source, every source spans
/// `row_len` bytes, `lanes` is 3 or 4, and `chunk` is 0, 32, 64 or 96.
#[allow(clippy::used_underscore_binding)]
#[archmage::arcane(import_intrinsics)]
pub fn mul_into_matrix1_external_chunk_8d(
    _token: archmage::X64V3GfniCryptoToken,
    rows: &mut [u8],
    row_len: usize,
    maps: &[u64],
    srcs: &[&[u8]],
    lanes: usize,
    chunk: usize,
) {
    assert_eq!(
        row_len % 32,
        0,
        "matrix_overwrite1_external_chunk: needs a 32-byte-multiple row"
    );
    assert!(
        row_len <= rows.len(),
        "matrix_overwrite1_external_chunk: rows buffer does not hold one row of {row_len} bytes"
    );
    assert_eq!(
        maps.len(),
        srcs.len(),
        "matrix_overwrite1_external_chunk: one resolved map per source"
    );
    for src in srcs {
        assert_eq!(src.len(), row_len);
    }
    assert!(
        matches!(chunk, 0 | 32 | 64 | 96),
        "matrix_overwrite1_external_chunk: chunk must be 0, 32, 64 or 96"
    );
    assert!(
        matches!(lanes, 3 | 4),
        "matrix_overwrite1_external_chunk: lanes must be 3 or 4"
    );
    if maps.is_empty() {
        // The empty overwrite sum is zero, as the unchunked external body's
        // single `rows_body` call produces.
        rows[..row_len].fill(0);
        return;
    }
    let (words, rest) = maps.as_chunks::<1>();
    debug_assert!(rest.is_empty());
    let ptr = rows.as_mut_ptr();
    // One in-bounds row of `row_len` bytes, as asserted above; the map and
    // source lists cover the same term count.
    match lanes {
        3 => rows_external_chunked::<1, 3>([ptr], row_len, words, srcs, chunk),
        4 => rows_external_chunked::<1, 4>([ptr], row_len, words, srcs, chunk),
        _ => panic!("matrix_overwrite1_external_chunk: lanes must be 3 or 4"),
    }
}
