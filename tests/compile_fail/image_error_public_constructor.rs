// expect: constructor is not visible
// expect-code: E0423
//! `ImageError` is obtained from a rejected image sequence and compared
//! by value; the unit cannot be constructed outside the crate.
use fgf::maps::ImageError;

pub fn probe() -> ImageError {
    ImageError
}
