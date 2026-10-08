//! A generic prime through the packed path.
//!
//! Run with `just example generic_prime`. The modulus `1_000_000_007` is a
//! non-special prime: no fold identities, plain widened modular arithmetic,
//! the default residue presentation at degree one. The packed operations
//! accept any lane bit pattern and emit canonical eight-byte lanes, and the
//! row reports the backend its kernels actually use.

#![forbid(unsafe_code)]

use fgf::{Elem, Field, FiniteField, backend_for, ops};

type Prime = FiniteField<1_000_000_007, 1>;

fn main() {
    // from_raw reduces the stored lane; the modulus itself and the full word
    // land on their ordinary residues.
    let a = [0, 1, 2, u64::MAX].map(Elem::<Prime>::from_raw);
    let b = [3, 5, 7, 1_000_000_007].map(Elem::<Prime>::from_raw);
    let challenge = Elem::<Prime>::from_residue(123_456_789).expect("in range");

    let packed_a = ops::pack_to_vec::<Prime>(&a);
    let packed_b = ops::pack_to_vec::<Prime>(&b);
    let mut output = vec![0u8; packed_a.len()];

    // output[i] = a[i]*b[i] + challenge*a[i], with the coefficient prepared
    // once and reused.
    ops::mul_elementwise::<Prime>(&mut output, &packed_a, &packed_b);
    let prepared = ops::Coeff::<Prime>::new(challenge);
    ops::mul_add_with::<Prime>(&mut output, &prepared, &packed_a);

    let mut elements = [Elem::<Prime>::ZERO; 4];
    ops::unpack::<Prime>(&mut elements, &output);
    let expected = std::array::from_fn(|i| a[i].mul(b[i]).add(challenge.mul(a[i])));
    assert_eq!(elements, expected);

    // Prime-field subtraction is modular subtraction, not bytewise XOR, and
    // the broadcast forms follow the same modular rule.
    ops::sub_assign::<Prime>(&mut output, &packed_a);
    ops::sub_assign_scalar::<Prime>(&mut output, challenge);
    ops::unpack::<Prime>(&mut elements, &output);
    let expected = std::array::from_fn(|i| expected[i].sub(a[i]).sub(challenge));
    assert_eq!(elements, expected);

    println!(
        "{} ({:?}): packed elementwise, prepared AXPY, and broadcast add/sub agree",
        Prime::NAME,
        backend_for::<Prime>()
    );
}
