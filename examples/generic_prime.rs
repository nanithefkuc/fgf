//! A generic prime through the packed path.
//!
//! Run with `just example generic_prime`. The modulus `1_000_000_007` is a
//! non-special prime: no fold identities, plain widened modular arithmetic,
//! the default residue presentation at degree one. The packed operations
//! accept any lane bit pattern and emit canonical four-byte lanes for every
//! nonzero, non-unit coefficient (zero/unit coefficients take the
//! byte-preserving shortcut paths), and `backend_for` reports the row's
//! kernel backend; rows shorter than the routing floor run the portable
//! path.
//!
//! The second half is a small systematic-encode shape: sixteen data rows
//! fold into four parity rows through one prepared source-major coefficient
//! matrix, and a one-byte prime applies a prepared coefficient to one-byte
//! lanes.
#![forbid(unsafe_code)]

use fgf::{Elem, Field, FiniteField, backend_for, ops};

type Prime = FiniteField<1_000_000_007, 1>;
type BytePrime = FiniteField<101, 1>;

const SOURCES: usize = 16;
const PARITY: usize = 4;
const ROW: usize = 64;

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

    parity_matrix();
    byte_prime();
}

/// Encode `PARITY` parity rows from `SOURCES` data rows through one prepared
/// source-major coefficient matrix, and check every lane against element
/// arithmetic. The coefficients are a fixed table of nonzero, non-unit
/// residues: the example supplies them rather than building a higher-layer
/// matrix.
fn parity_matrix() {
    let element = |t: usize, i: usize| Elem::<Prime>::from_raw((t * 7_919 + i * 104_729) as u64);
    let data: Vec<Vec<u8>> = (0..SOURCES)
        .map(|t| ops::pack_to_vec::<Prime>(&(0..ROW).map(|i| element(t, i)).collect::<Vec<_>>()))
        .collect();
    let sources: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();

    let fixed: Vec<Elem<Prime>> = (0..SOURCES * PARITY)
        .map(|n| {
            let mut c = (n as u64).wrapping_mul(1_103_515_245).wrapping_add(12_345) % 1_000_000_007;
            if c < 2 {
                c += 2;
            }
            Elem::<Prime>::from_raw(c)
        })
        .collect();
    let matrix = ops::CoeffMatrix::<Prime>::from_source_major(SOURCES, PARITY, &fixed);

    let row_len = data[0].len();
    let mut parity = vec![0u8; PARITY * row_len];
    ops::mul_into_matrix_with::<Prime>(&mut parity, row_len, &matrix, &sources);

    let mut lanes = vec![Elem::<Prime>::ZERO; ROW];
    for (j, row) in parity.chunks_exact(row_len).enumerate() {
        ops::unpack::<Prime>(&mut lanes, row);
        for (i, &lane) in lanes.iter().enumerate() {
            let expected = (0..SOURCES)
                .map(|t| fixed[t * PARITY + j].mul(element(t, i)))
                .fold(Elem::<Prime>::ZERO, |sum, term| sum.add(term));
            assert_eq!(lane, expected, "parity row {j} lane {i}");
        }
    }
    println!(
        "{} ({:?}): {SOURCES} x {PARITY} prepared parity matrix agrees",
        Prime::NAME,
        backend_for::<Prime>()
    );
}

/// One-byte lanes: a prepared coefficient applied to every residue of `101`.
fn byte_prime() {
    let all: Vec<Elem<BytePrime>> = (0..101).map(Elem::<BytePrime>::from_raw).collect();
    let packed = ops::pack_to_vec::<BytePrime>(&all);
    let coeff = Elem::<BytePrime>::from_raw(57);
    let prepared = ops::Coeff::<BytePrime>::new(coeff);
    let mut scaled = vec![0u8; packed.len()];
    ops::mul_into_with::<BytePrime>(&mut scaled, &prepared, &packed);

    let mut lanes = vec![Elem::<BytePrime>::ZERO; all.len()];
    ops::unpack::<BytePrime>(&mut lanes, &scaled);
    let expected: Vec<_> = all.iter().map(|&x| coeff.mul(x)).collect();
    assert_eq!(lanes, expected);
    println!(
        "{} ({:?}): one-byte lanes, prepared coefficient agrees on every residue",
        BytePrime::NAME,
        backend_for::<BytePrime>()
    );
}
