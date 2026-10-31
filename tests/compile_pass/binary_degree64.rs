//! Degree-64 relations over GF(2), compiled with the default const-eval lint.

use fgf::{
    Elem, Extension, ExtensionSpec, Field, FieldElem, FiniteField, Packed, Polynomial, Relation,
    ops,
};

type Gf2 = FiniteField<2, 1, Polynomial<3>>;

struct Spec<const LOW: u64>;
impl<const LOW: u64> ExtensionSpec<64> for Spec<LOW> {
    type Base = Gf2;
    type Storage = Packed<u64>;
    const NAME: &'static str = "binary degree-64 relation";
    const RELATION: Relation = {
        let mut coefficients = [0; 64];
        let mut i = 0;
        while i < 64 {
            coefficients[i] = (LOW >> i) & 1;
            i += 1;
        }
        Relation::from_binary_words::<Gf2>(&coefficients)
    };
}

type Row<const LOW: u64> = FiniteField<2, 64, Extension<Spec<LOW>, 64>>;
const _: () = <Row<0x1b> as Field>::VALID;
const _: () = <Row<0xa1c6_7ae0_5c18_5c83> as Field>::VALID;

// Carryless multiplication and long division supply a reference independent
// of the descriptor's component arithmetic and relation validator.
fn reference(left: u64, right: u64, low: u64) -> u64 {
    let mut product = 0u128;
    for bit in 0..64 {
        if right & (1u64 << bit) != 0 {
            product ^= u128::from(left) << bit;
        }
    }
    let modulus = (1u128 << 64) | u128::from(low);
    for bit in (64..128).rev() {
        if product & (1u128 << bit) != 0 {
            product ^= modulus << (bit - 64);
        }
    }
    u64::try_from(product).unwrap()
}

fn exercise_row<const LOW: u64>() {
    let words = [0, u64::MAX, 1 << 63, 0x1234_5678_9abc_def0];
    let coefficient = (1 << 63) | 7;
    let mut source = [0; 32];
    let mut expected = [0; 32];
    for (i, word) in words.into_iter().enumerate() {
        source[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
        expected[i * 8..i * 8 + 8]
            .copy_from_slice(&reference(word, coefficient, LOW).to_le_bytes());
    }
    let mut destination = [0xa5; 32];
    let coefficient = Elem::<Row<LOW>>::from_raw(coefficient);
    ops::mul_into::<Row<LOW>>(&mut destination, coefficient, &source);
    assert_eq!(destination, expected);
    ops::mul_add::<Row<LOW>>(&mut destination, coefficient, &source);
    assert_eq!(destination, [0; 32]);
    let prepared = ops::Coeff::<Row<LOW>>::new(coefficient);
    ops::mul_add_with::<Row<LOW>>(&mut destination, &prepared, &source);
    assert_eq!(destination, expected);
}

pub fn exercise() {
    exercise_row::<0x1b>();
    exercise_row::<0xa1c6_7ae0_5c18_5c83>();
}

#[cfg(not(test))]
fn main() {
    exercise();
    println!("degree64_consumer: sparse and dense relations pass bulk arithmetic");
}
