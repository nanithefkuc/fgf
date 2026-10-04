//! Pointwise products and prepared linear combinations over prime fields.
//!
//! Run with `just example prime_vectors`. This is the packed arithmetic used
//! on evaluation vectors; transforms and polynomial interpolation are outside
//! fgf. Packed prime operations accept any lane bit pattern and emit
//! canonical lanes.

#![forbid(unsafe_code)]

use fgf::field::FieldElem;
use fgf::{
    Elem, FieldKernels, Goldilocks, Mersenne31, QuadMersenne31, backend_for, mersenne31, ops,
};

fn main() {
    // from_raw reduces the stored lane; noncanonical aliases such as the
    // modulus itself land on the same element as zero.
    let m = [0, 1, mersenne31::MODULUS + 7, u32::MAX].map(Elem::<Mersenne31>::from_raw);
    combine::<Mersenne31>(
        m,
        [2, 3, 5, 8].map(Elem::<Mersenne31>::from_raw),
        Elem::<Mersenne31>::from_raw(13),
    );

    let g = [0, 1, fgf::goldilocks::MODULUS + 7, u64::MAX].map(Elem::<Goldilocks>::from_raw);
    combine::<Goldilocks>(
        g,
        [2, 3, 5, 8].map(Elem::<Goldilocks>::from_raw),
        Elem::<Goldilocks>::from_raw(13),
    );

    // QuadMersenne31 has two canonical u32 limbs per element, in [re, im]
    // order. A raw byte stream is not automatically a valid evaluation vector.
    let q = [
        (0, 0),
        (1, 2),
        (mersenne31::MODULUS + 7, 11),
        (u32::MAX, u32::MAX),
    ]
    .map(|(re, im)| Elem::<QuadMersenne31>::from_raw(re, im));
    combine::<QuadMersenne31>(
        q,
        [(2, 1), (3, 4), (5, 6), (8, 9)].map(|(re, im)| Elem::<QuadMersenne31>::from_raw(re, im)),
        Elem::<QuadMersenne31>::from_raw(13, 17),
    );
}

fn combine<F: FieldKernels>(a: [Elem<F>; 4], b: [Elem<F>; 4], challenge: Elem<F>) {
    let packed_a = ops::pack_to_vec::<F>(&a);
    let packed_b = ops::pack_to_vec::<F>(&b);
    let mut output = vec![0u8; packed_a.len()];
    let prepared = ops::Coeff::<F>::new(challenge);

    // output[i] = a[i]*b[i] + challenge*a[i]. Output storage and prepared
    // state can be retained for subsequent vectors with the same geometry.
    ops::mul_elementwise::<F>(&mut output, &packed_a, &packed_b);
    ops::mul_add_with::<F>(&mut output, &prepared, &packed_a);
    let mut elements = [Elem::<F>::ZERO; 4];
    ops::unpack::<F>(&mut elements, &output);
    let expected = std::array::from_fn(|i| a[i].mul(b[i]).add(challenge.mul(a[i])));
    assert_eq!(elements, expected);

    // Prime-field subtraction is modular subtraction, not bytewise XOR.
    ops::sub_assign::<F>(&mut output, &packed_a);
    ops::unpack::<F>(&mut elements, &output);
    assert_eq!(elements, std::array::from_fn(|i| expected[i].sub(a[i])));
    println!(
        "{} ({:?}): a*b + challenge*a - a = {elements:?}",
        F::NAME,
        backend_for::<F>()
    );
}
