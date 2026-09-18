//! Measurement variants for the GF(2^8) x86 kernels.
//!
//! Nothing here is reachable from production dispatch: these variants are
//! exposed through the internals facade to price one decision
//! against the production body it is paired with. Variants are grouped by
//! what they measure — `tile` for tile width and store policy, `shuffle` for
//! the nibble alternative to the GFNI multiply, `resolve` for per-call
//! coefficient resolution, `grouped` for the multi-row crossed panel, and
//! `gather` for the gather tile and remainder controls.
//!
//! Every entry is a safe [`archmage`] capability-token function. The
//! externally-resolved chunk loop below is shared by `resolve` and `grouped`.

mod gather;
mod grouped;
mod resolve;
mod shuffle;
mod tile;

pub use gather::*;
pub use grouped::*;
pub use resolve::*;
pub use shuffle::*;
pub use tile::*;

use super::Affine8D;
use super::rows::rows_body;

/// Run the production tile body over externally resolved map words, in chunks
/// of `chunk` terms (`0` folds every term into one pass).
///
/// The first chunk overwrites; later chunks accumulate, exactly as the
/// resolved path's chunk loop. The rows are runtime-offset windows of one
/// region — the offset-addressed-row residue — bounded by the caller's row
/// geometry.
#[archmage::rite(v3_gfni_crypto)]
fn rows_external_chunked<const ROWS: usize, const LANES: usize>(
    ptrs: [*mut u8; ROWS],
    vector_len: usize,
    maps: &[[u64; ROWS]],
    srcs: &[&[u8]],
    chunk: usize,
) {
    let terms = maps.len();
    let step = if chunk == 0 { terms.max(1) } else { chunk };
    let mut start = 0;
    let mut first = true;
    while start < terms {
        let taken = (terms - start).min(step);
        if first {
            rows_body::<Affine8D, ROWS, LANES, true>(
                ptrs,
                vector_len,
                &maps[start..start + taken],
                &srcs[start..start + taken],
            );
        } else {
            rows_body::<Affine8D, ROWS, LANES, false>(
                ptrs,
                vector_len,
                &maps[start..start + taken],
                &srcs[start..start + taken],
            );
        }
        first = false;
        start += taken;
    }
}
