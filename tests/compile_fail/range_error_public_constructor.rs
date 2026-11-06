// expect: unit struct `RangeError` is private
//! `RangeError` is obtained from a rejected conversion and compared by
//! value; the unit cannot be constructed outside the crate.
pub fn probe() -> fgf::RangeError {
    fgf::RangeError
}
