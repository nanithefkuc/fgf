//! Prepared-map steady-state allocation: construction outside the counted
//! window, application inside it.
//!
//! `PreparedEmbedding::new` and `PreparedIsomorphism::new` own their
//! column entries by contract; every method below runs the compiled arms
//! over stack workspace only, so repeated application allocates nothing.

#![cfg(feature = "std")]
// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use fgf::field::FieldBuffer;
use fgf::poly::AES;
use fgf::{
    Components, Elem, Embedding, Extension, ExtensionSpec, Field, FieldElem, FiniteField,
    Isomorphism, Normal, PreparedEmbedding, PreparedIsomorphism, QuadMersenne31, Relation, Residue,
    bits, ops,
};

#[path = "common/zero_alloc.rs"]
mod common;
use common::{TEST_LOCK, count_allocations, noise};

type Gf16 = FiniteField<2, 16, Extension<fgf::Rijndael16, 2>>;
type Gf64 = FiniteField<2, 64, Extension<fgf::Rijndael64, 2>>;
type Fp64 = FiniteField<2, 64, Extension<fgf::FanPaar64, 2>>;
type Aes8 = FiniteField<2, 8, fgf::Polynomial<AES>>;
type Normal8 = FiniteField<2, 8, Normal<0x11B, 0x20>>;
struct Normal16Spec;
impl ExtensionSpec<2> for Normal16Spec {
    type Base = Normal8;
    type Storage = Components;
    const NAME: &'static str = "GF(2^16) over the normal-basis row";
    const RELATION: Relation = Relation::from_binary_words::<Normal8>(&[0x1, 0xFF]);
}
type Normal16 = FiniteField<2, 16, Extension<Normal16Spec, 2>>;

/// Steady-state application of one embedding over every method:
/// preparation outside the counted window, only application inside it.
fn assert_embedding_steady_state_zero_alloc<
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
>(
    map: &PreparedEmbedding<S, T>,
    sources: &[Elem<S>],
    targets: &[Elem<T>],
) {
    // Warm every method before counting.
    for &s in sources {
        core::hint::black_box(map.embed(s));
    }
    for &y in targets {
        core::hint::black_box(map.contains(y));
        core::hint::black_box(map.restrict(y));
        core::hint::black_box(map.frobenius(y));
        core::hint::black_box(map.trace(y));
        core::hint::black_box(map.norm(y));
    }
    let allocations = count_allocations(|| {
        for &s in sources {
            core::hint::black_box(map.embed(s));
        }
        for &y in targets {
            core::hint::black_box(map.contains(y));
            core::hint::black_box(map.restrict(y));
            core::hint::black_box(map.frobenius(y));
            core::hint::black_box(map.trace(y));
            core::hint::black_box(map.norm(y));
        }
    });
    assert_eq!(allocations, 0, "prepared embedding application allocated");
}

/// Deterministic sampled elements below the field order.
fn sampled<F: Field>(n: usize, seed: u64) -> Vec<Elem<F>> {
    let bytes = noise(n * 8, seed);
    bytes
        .chunks_exact(8)
        .enumerate()
        .map(|(i, lane)| {
            let wide = u128::from(u64::from_le_bytes(lane.try_into().unwrap()));
            Elem::<F>::from_index((wide ^ (i as u128).wrapping_mul(0x9E37)) % F::ORDER)
                .expect("a sampled index sits below the order")
        })
        .collect()
}

#[test]
fn binary_embedding_application_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let compact = Embedding::<Gf16, Gf64>::new().unwrap();
    let map = PreparedEmbedding::<Gf16, Gf64>::new(&compact);
    let sources: Vec<Elem<Gf16>> = sampled(32, 0xE0);
    let targets: Vec<Elem<Gf64>> = sampled(32, 0xE1);
    assert_embedding_steady_state_zero_alloc(&map, &sources, &targets);
}

#[test]
fn odd_embedding_application_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let compact = Embedding::<fgf::Mersenne31, QuadMersenne31>::new().unwrap();
    let map = PreparedEmbedding::<fgf::Mersenne31, QuadMersenne31>::new(&compact);
    let sources: Vec<Elem<fgf::Mersenne31>> = sampled(32, 0xE2);
    let targets: Vec<Elem<QuadMersenne31>> = sampled(32, 0xE3);
    assert_embedding_steady_state_zero_alloc(&map, &sources, &targets);
}

#[test]
fn prime_into_basis_embedding_application_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let compact = Embedding::<Aes8, Gf64>::new().unwrap();
    let map = PreparedEmbedding::<Aes8, Gf64>::new(&compact);
    let sources: Vec<Elem<Aes8>> = sampled(32, 0xE4);
    let targets: Vec<Elem<Gf64>> = sampled(32, 0xE5);
    assert_embedding_steady_state_zero_alloc(&map, &sources, &targets);
}

#[test]
fn isomorphism_application_allocates_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    let compact = Isomorphism::<Fp64, Gf64>::new().unwrap();
    let map = PreparedIsomorphism::<Fp64, Gf64>::new(&compact);
    let sources: Vec<Elem<Fp64>> = sampled(32, 0xE6);
    let targets: Vec<Elem<Gf64>> = sampled(32, 0xE7);
    for &s in &sources {
        core::hint::black_box(map.apply(s));
    }
    for &y in &targets {
        core::hint::black_box(map.unapply(y));
    }
    let allocations = count_allocations(|| {
        for &s in &sources {
            core::hint::black_box(map.apply(s));
        }
        for &y in &targets {
            core::hint::black_box(map.unapply(y));
        }
    });
    assert_eq!(allocations, 0, "prepared isomorphism application allocated");
}

/// Steady-state application of every row function: buffers sized and
/// preparation built outside the counted window, only the row body inside
/// it. Covers a binary and an odd non-injection map.
fn assert_rows_steady_state_zero_alloc<
    S: Field + FieldBuffer,
    T: Field<Characteristic = S::Characteristic> + FieldBuffer,
    U: Field + FieldBuffer,
    V: Field<Characteristic = U::Characteristic> + FieldBuffer,
>(
    emb: &PreparedEmbedding<S, T>,
    iso_map: &PreparedIsomorphism<U, V>,
    src_s: &[u8],
    src_t: &[u8],
    member_t: &[u8],
    src_u: &[u8],
    src_v: &[u8],
) {
    let count = src_t.len() / T::BYTES;
    let mut dst_t = vec![0u8; src_s.len() / S::BYTES * T::BYTES];
    let mut dst_s = vec![0u8; count * S::BYTES];
    let mut valid = vec![0u8; bits::bytes_for(count)];
    let count_uv = src_v.len() / V::BYTES;
    let mut dst_v = vec![0u8; src_u.len() / U::BYTES * V::BYTES];
    let mut dst_u = vec![0u8; count_uv * U::BYTES];
    // Warm every row before counting.
    ops::embed_into_with(&mut dst_t, emb, src_s);
    ops::trace_into_with(&mut dst_s, emb, src_t);
    ops::norm_into_with(&mut dst_s, emb, src_t);
    ops::restrict_into_with(&mut dst_s, &mut valid, emb, src_t);
    ops::restrict_into_with(&mut dst_s, &mut valid, emb, member_t);
    ops::contains_into_with(&mut valid, emb, src_t);
    ops::apply_into_with(&mut dst_v, iso_map, src_u);
    ops::unapply_into_with(&mut dst_u, iso_map, src_v);
    let allocations = count_allocations(|| {
        ops::embed_into_with(&mut dst_t, emb, src_s);
        ops::trace_into_with(&mut dst_s, emb, src_t);
        ops::norm_into_with(&mut dst_s, emb, src_t);
        ops::restrict_into_with(&mut dst_s, &mut valid, emb, src_t);
        ops::restrict_into_with(&mut dst_s, &mut valid, emb, member_t);
        ops::contains_into_with(&mut valid, emb, src_t);
        ops::apply_into_with(&mut dst_v, iso_map, src_u);
        ops::unapply_into_with(&mut dst_u, iso_map, src_v);
    });
    assert_eq!(allocations, 0, "prepared row application allocated");
}

type R3 = FiniteField<3, 1, Residue>;

struct G9AltSpec;
impl ExtensionSpec<2> for G9AltSpec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(9) as a^2 + 1";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0]);
}
type G9Alt = FiniteField<3, 2, Extension<G9AltSpec, 2>>;

struct G81ExtSpec;
impl ExtensionSpec<2> for G81ExtSpec {
    type Base = G9Alt;
    type Storage = Components;
    const NAME: &'static str = "GF(81) as t^2 - (1 + a)";
    const RELATION: Relation = Relation::from_coordinates(&[2, 2, 0, 0]);
}
type G81Ext = FiniteField<3, 4, Extension<G81ExtSpec, 2>>;

#[test]
fn prepared_rows_allocate_nothing() {
    let _guard = TEST_LOCK
        .lock()
        .expect("zero-allocation test lock poisoned");
    // Binary pair: the normal-basis row into the extension directly
    // over it (residue-column arm: the normal-basis columns are no
    // identity prefix). The iso arm runs on the shipped FanPaar64
    // presentation pair.
    let iso_compact = Isomorphism::<Fp64, Gf64>::new().unwrap();
    let iso = PreparedIsomorphism::<Fp64, Gf64>::new(&iso_compact);
    let src_u = noise(64 * Fp64::BYTES, 0xBE);
    let src_v = noise(64 * Gf64::BYTES, 0xBF);
    {
        // Compose the nonidentity normal automorphism with the
        // canonical lift: the tuple is no identity prefix, so the
        // binary arm runs the residue-column path.
        let image = (1u16..256)
            .map(|word| Elem::<Normal8>::from_raw(word as u8))
            .find(|image| Isomorphism::<Normal8, Normal8>::from_images(&[*image]).is_ok())
            .expect("a nonidentity normal automorphism exists");
        let automorphism = Isomorphism::<Normal8, Normal8>::from_images(&[image]).unwrap();
        let lift = Embedding::<Normal8, Normal16>::new().unwrap();
        let compact = automorphism.then_embedding(&lift);
        let map = PreparedEmbedding::<Normal8, Normal16>::new(&compact);
        let src_s = noise(64 * Normal8::BYTES, 0xB0);
        let src_t = noise(64 * Normal16::BYTES, 0xB1);
        let members: Vec<Elem<Normal16>> = (0..64)
            .map(|i| map.embed(Elem::<Normal8>::from_index(i % 256).unwrap()))
            .collect();
        let mut member_t = vec![0u8; 64 * Normal16::BYTES];
        for (i, m) in members.iter().enumerate() {
            Normal16::encode(
                &mut member_t[i * Normal16::BYTES..(i + 1) * Normal16::BYTES],
                *m,
            );
        }
        assert_rows_steady_state_zero_alloc(&map, &iso, &src_s, &src_t, &member_t, &src_u, &src_v);
    }
    // Odd pair: a conjugate-image inclusion of GF(4) into AES (a
    // non-identity image tuple, never the canonical construction), so
    // the compiled columns are no identity prefix. The iso arm is an
    // odd nonidentity isomorphism so residue-column application counts
    // in both directions.
    let odd_iso_compact = (0..9u128)
        .filter_map(|word| {
            Isomorphism::<G9Alt, G9Alt>::from_images(&[Elem::<G9Alt>::from_index(word).unwrap()])
                .ok()
        })
        .find(|iso| {
            iso.apply(Elem::<G9Alt>::from_index(3).unwrap())
                != Elem::<G9Alt>::from_index(3).unwrap()
        })
        .expect("the odd row admits a nonidentity automorphism");
    let odd_iso = PreparedIsomorphism::<G9Alt, G9Alt>::new(&odd_iso_compact);
    let odd_u = noise(64 * G9Alt::BYTES, 0xBC);
    let odd_v = noise(64 * G9Alt::BYTES, 0xBD);
    {
        // Conjugate-image search over the quartic target: the first two
        // valid image tuples of the quadratic source fix distinct maps.
        let mut tuples = Vec::new();
        for word in 0..81u128 {
            let candidate = Elem::<G81Ext>::from_index(word).unwrap();
            if Embedding::<G9Alt, G81Ext>::from_images(&[candidate]).is_ok() {
                tuples.push(candidate);
                if tuples.len() == 2 {
                    break;
                }
            }
        }
        assert_eq!(tuples.len(), 2, "the pair admits conjugate images");
        let compact = Embedding::<G9Alt, G81Ext>::from_images(&[tuples[1]]).unwrap();
        let map = PreparedEmbedding::<G9Alt, G81Ext>::new(&compact);
        let src_s = noise(64 * G9Alt::BYTES, 0xB2);
        let src_t = noise(64 * G81Ext::BYTES, 0xB3);
        let members: Vec<Elem<G81Ext>> = (0..64)
            .map(|i| map.embed(Elem::<G9Alt>::from_index(i % 9).unwrap()))
            .collect();
        let mut member_t = vec![0u8; 64 * G81Ext::BYTES];
        for (i, m) in members.iter().enumerate() {
            G81Ext::encode(
                &mut member_t[i * G81Ext::BYTES..(i + 1) * G81Ext::BYTES],
                *m,
            );
        }
        assert_rows_steady_state_zero_alloc(
            &map, &odd_iso, &src_s, &src_t, &member_t, &odd_u, &odd_v,
        );
    }
}
