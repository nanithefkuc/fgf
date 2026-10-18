//! Byte-lane inner product: fold two packed rows into one element.
//!
//! The two-row reduction to go with [`mul_elementwise`](crate::ops::mul_elementwise):
//! where that call varies the product per lane, this one sums it into a
//! single [`Elem`]. The name keeps the short scalar-noun spelling
//! (`dot_product`, like `pack` and `mul_elementwise`, overwrites without an
//! `_into` suffix) and stays distinct from bit-packed
//! [`bits::dot_product`](crate::bits::dot_product), which folds an AND
//! parity over live bits rather than a field sum over lanes.

use crate::field::{Elem, FieldElem};
use crate::kernel::FieldKernels;

use super::{check_pair, check_width};

/// The field inner product `sum(a[i] * b[i])` of two packed rows.
///
/// Returns [`Elem::ZERO`] on empty inputs. Prime-field lanes accept any raw
/// bit pattern a decode accepts, matching the packed-operation contract;
/// outputs are always the canonical element.
///
/// # Panics
///
/// Panics if `a` and `b` differ in length or hold a partial trailing
/// element.
#[must_use]
pub fn dot_product<F: FieldKernels>(a: &[u8], b: &[u8]) -> Elem<F> {
    check_pair::<F>("dot_product", "a", a.len(), "b", b.len());
    check_width::<F>("dot_product", a.len());
    let mut acc = Elem::<F>::ZERO;
    for (x, y) in a.chunks_exact(F::BYTES).zip(b.chunks_exact(F::BYTES)) {
        acc = acc.add(F::decode(x).mul(F::decode(y)));
    }
    acc
}
