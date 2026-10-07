//! Encode parity shards and recover erased data in the Reed–Solomon byte field.
//!
//! Run with `just example shard_recovery`. The example supplies a fixed linear
//! code; fgf applies coefficients but does not construct or invert coding
//! matrices. The recovery equations cover the erasure patterns shown here,
//! not unknown errors or a general-purpose decoder.

#![forbid(unsafe_code)]

use fgf::poly::RS;
use fgf::{Binary, Elem, Polynomial, backend_for, ops};

const ROW_LEN: usize = 32;
const TWO: Elem<Binary<8, Polynomial<RS>>> = Elem::<Binary<8, Polynomial<RS>>>::from_raw(2);
const THREE: Elem<Binary<8, Polynomial<RS>>> = Elem::<Binary<8, Polynomial<RS>>>::ONE.add(TWO);

fn main() {
    let a = *b"The first data shard has bytes!!";
    let b = *b"The other data shard has bytes!!";
    assert_eq!(a.len(), ROW_LEN);
    assert_eq!(b.len(), ROW_LEN);

    // p = a + b, q = a + 2*b, independently at every byte position.
    // Source-major order: [a->p, a->q, b->p, b->q].
    let encoding = ops::CoeffMatrix::<Binary<8, Polynomial<RS>>>::from_source_major(
        2,
        2,
        &[
            Elem::<Binary<8, Polynomial<RS>>>::ONE,
            Elem::<Binary<8, Polynomial<RS>>>::ONE,
            Elem::<Binary<8, Polynomial<RS>>>::ONE,
            TWO,
        ],
    );
    let mut parity = [0xa5; 2 * ROW_LEN];
    ops::mul_into_matrix_with(&mut parity, ROW_LEN, &encoding, &[&a, &b]);
    let (p, q) = parity.split_at(ROW_LEN);

    for (i, (&ai, &bi)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            p[i],
            (Elem::<Binary<8, Polynomial<RS>>>::from_raw(ai)
                + Elem::<Binary<8, Polynomial<RS>>>::from_raw(bi))
            .to_raw()
        );
        assert_eq!(
            q[i],
            (Elem::<Binary<8, Polynomial<RS>>>::from_raw(ai)
                + TWO * Elem::<Binary<8, Polynomial<RS>>>::from_raw(bi))
            .to_raw()
        );
    }

    // If only b is lost, b = p + a in characteristic two.
    let single = ops::CoeffVec::<Binary<8, Polynomial<RS>>>::new(&[
        Elem::<Binary<8, Polynomial<RS>>>::ONE,
        Elem::<Binary<8, Polynomial<RS>>>::ONE,
    ]);
    let mut recovered_b = [0xa5; ROW_LEN];
    ops::mul_into_gather_with(&mut recovered_b, single.as_ref(), &[p, &a]);
    assert_eq!(recovered_b, b);

    // If both data shards are lost, subtracting the equations gives
    // b = (p + q)/3, then a = (2*p + q)/3. Here '+' is field XOR, and
    // 3 is the field element 1 + 2, not integer addition modulo 256.
    assert_ne!(THREE, Elem::<Binary<8, Polynomial<RS>>>::ZERO);
    let inverse = THREE.inv();
    let recovery = ops::CoeffMatrix::<Binary<8, Polynomial<RS>>>::from_source_major(
        2,
        2,
        &[TWO * inverse, inverse, inverse, inverse],
    );
    let mut recovered = [0xa5; 2 * ROW_LEN];
    ops::mul_into_matrix_with(&mut recovered, ROW_LEN, &recovery, &[p, q]);
    assert_eq!(&recovered[..ROW_LEN], &a);
    assert_eq!(&recovered[ROW_LEN..], &b);

    println!(
        "Binary<8, Polynomial<RS>> backend: {:?}",
        backend_for::<Binary<8, Polynomial<RS>>>()
    );
    println!("Parity p: {p:02x?}");
    println!("Parity q: {q:02x?}");
    println!(
        "Recovered a: {}",
        std::str::from_utf8(&recovered[..ROW_LEN]).unwrap()
    );
    println!(
        "Recovered b: {}",
        std::str::from_utf8(&recovered[ROW_LEN..]).unwrap()
    );
}
