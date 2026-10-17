//! Prepared map row acceptance: decode-method-encode equality, bitmap
//! discipline, mixed restriction, geometry panics, and the QM31 anchors.
//!
//! Every row entry point decodes each lane, applies the prepared scalar
//! method, and encodes canonically, so each row equals the per-element
//! scalar loop of decode into the prepared method into encode. Restriction
//! writes canonical preimages or canonical source zero beside an LSB-first
//! validity bitmap; membership writes the bitmap only.

#![cfg(feature = "alloc")]
// Toolchain-drift lint (not in the MSRV); see `src/lib.rs`.
#![allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]

use fgf::field::{BinaryField, FieldBuffer, PrimeCharacteristic, PrimeField};
use fgf::poly::AES;
use fgf::{
    Components, Elem, Embedding, Extension, ExtensionSpec, Field, FieldElem, FiniteField,
    Isomorphism, Polynomial, PreparedEmbedding, PreparedIsomorphism, QuadMersenne31, Relation,
    Residue, bits, ops,
};

type R3 = FiniteField<3, 1, Residue>;
type R7 = FiniteField<7, 1, Residue>;
type Aes8 = FiniteField<2, 8, Polynomial<AES>>;
type Gf16 = FiniteField<2, 16, fgf::Extension<fgf::Rijndael16, 2>>;
type Gf64 = FiniteField<2, 64, fgf::Extension<fgf::Rijndael64, 2>>;
type Fp64 = FiniteField<2, 64, fgf::Extension<fgf::FanPaar64, 2>>;
type Normal8 = FiniteField<2, 8, fgf::Normal<0x11B, 0x20>>;

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
    // `-(1 + a)`: constant `2 + 2a`, linear zero.
    const RELATION: Relation = Relation::from_coordinates(&[2, 2, 0, 0]);
}
type G81Ext = FiniteField<3, 4, Extension<G81ExtSpec, 2>>;

struct G343Spec;
impl ExtensionSpec<3> for G343Spec {
    type Base = R7;
    type Storage = Components;
    const NAME: &'static str = "GF(343) as x^3 - 2";
    const RELATION: Relation = Relation::from_coordinates(&[5, 0, 0]);
}
type G343 = FiniteField<7, 3, Extension<G343Spec, 3>>;

/// Deterministic pseudo-random bytes. No dependency, reproducible failures.
fn noise(len: usize, seed: u64) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

/// Row lengths straddle lane and unroll boundaries, plus one large count.
const COUNTS: [usize; 12] = [0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 1024];

fn encode_all<F: FieldBuffer>(elems: &[Elem<F>]) -> Vec<u8> {
    let mut bytes = vec![0u8; elems.len() * F::BYTES];
    for (i, elem) in elems.iter().enumerate() {
        F::encode(&mut bytes[i * F::BYTES..(i + 1) * F::BYTES], *elem);
    }
    bytes
}

/// Deterministic staged elements: boundaries first, then the LCG stream.
fn staged<F: Field>(n: usize, salt: u64) -> Vec<Elem<F>> {
    let mut out = Vec::with_capacity(n);
    for boundary in [0u128, 1, F::ORDER - 1] {
        if out.len() < n {
            out.push(Elem::<F>::from_index(boundary).expect("an index below the order"));
        }
    }
    let mut state = salt | 1;
    while out.len() < n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let wide =
            ((u128::from(state >> 33) << 31) | u128::from((state >> 2) & 0x7fff_ffff)) % F::ORDER;
        out.push(Elem::<F>::from_index(wide).expect("a sampled index sits below the order"));
    }
    out
}

/// Target elements outside the image: offset member images verified
/// against `contains`.
fn outsiders<S: FieldBuffer, T: FieldBuffer<Characteristic = S::Characteristic>>(
    map: &PreparedEmbedding<S, T>,
    members: &[Elem<T>],
    n: usize,
) -> Vec<Elem<T>> {
    // Every candidate is verified against `contains`. Candidates are
    // pseudo-randomized by an LCG over the target index space, so each
    // probe is an independent draw: sparse images resolve in
    // `n / density` expected probes instead of a full linear sweep, and
    // tiny targets cycle the outsiders found when the target holds fewer
    // outsiders than lanes.
    let mut out = Vec::new();
    let mut state = 0x243F_6A88_85A3_08D3u128
        | ((members.first().map_or(1, |member| member.to_index()) | 1) & 0xFFFF_FFFF);
    let mut probes = 0u64;
    while out.len() < n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let index = (state >> 33) % T::ORDER;
        let candidate = Elem::<T>::from_index(index).expect("an index below the order");
        if !map.contains(candidate) {
            out.push(candidate);
        }
        probes += 1;
        assert!(
            probes < 1_000_000,
            "degrees {} into {} admit too few outsiders",
            S::DEGREE,
            T::DEGREE
        );
    }
    out
}

/// `outsider_lanes` caps the outsider/mixed pool: pairs whose image is
/// sparse in a vast target (the Goldilocks square) cannot stage a full
/// 1024-lane outsider row in reasonable time, so they exercise the
/// membership rows over a smaller pool while every `COUNTS` length still
/// runs (repeating pool lanes where the row is longer than the pool).
fn check_embedding_rows<S, T>(seed: u64, outsider_lanes: usize)
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("a validated pair constructs");
    let map = PreparedEmbedding::<S, T>::new(&compact);
    let top_probe = *COUNTS.iter().max().unwrap_or(&1);
    for x in staged::<S>(top_probe, seed) {
        assert_eq!(
            map.embed(x),
            compact.embed(x),
            "prepared agrees with its map"
        );
    }
    assert_eq!(
        map.relative_degree(),
        T::DEGREE / S::DEGREE,
        "relative degree divides"
    );
    let top = *COUNTS.iter().max().unwrap_or(&1);
    let sources: Vec<Elem<S>> = staged(top, seed);
    let members: Vec<Elem<T>> = sources.iter().map(|x| map.embed(*x)).collect();
    let strangers = outsiders(&map, &members, outsider_lanes.min(top));
    let mixed: Vec<Elem<T>> = (0..top)
        .map(|i| {
            if i % 2 == 0 {
                members[i % members.len()]
            } else {
                strangers[(i / 2) % strangers.len()]
            }
        })
        .collect();
    let member_pool: Vec<Elem<T>> = (0..top).map(|i| members[i % members.len()]).collect();
    let stranger_pool: Vec<Elem<T>> = (0..top).map(|i| strangers[i % strangers.len()]).collect();
    let source_bytes = encode_all::<S>(&sources);
    let member_bytes = encode_all::<T>(&member_pool);
    let stranger_bytes = encode_all::<T>(&stranger_pool);
    let mixed_bytes = encode_all::<T>(&mixed);

    for &count in &COUNTS {
        // embed
        {
            let src = &source_bytes[..count * S::BYTES];
            let mut expected = vec![0u8; count * T::BYTES];
            for (d, s) in expected
                .chunks_exact_mut(T::BYTES)
                .zip(src.chunks_exact(S::BYTES))
            {
                T::encode(d, map.embed(S::decode(s)));
            }
            let mut dst = vec![0xA5u8; count * T::BYTES];
            ops::embed_into_with(&mut dst, &map, src);
            assert_eq!(dst, expected, "embed count {count}");
        }
        // trace
        {
            let src = &mixed_bytes[..count * T::BYTES];
            let mut expected = vec![0u8; count * S::BYTES];
            for (d, s) in expected
                .chunks_exact_mut(S::BYTES)
                .zip(src.chunks_exact(T::BYTES))
            {
                S::encode(d, map.trace(T::decode(s)));
            }
            let mut dst = vec![0xA5u8; count * S::BYTES];
            ops::trace_into_with(&mut dst, &map, src);
            assert_eq!(dst, expected, "trace count {count}");
        }
        // norm
        {
            let src = &mixed_bytes[..count * T::BYTES];
            let mut expected = vec![0u8; count * S::BYTES];
            for (d, s) in expected
                .chunks_exact_mut(S::BYTES)
                .zip(src.chunks_exact(T::BYTES))
            {
                S::encode(d, map.norm(T::decode(s)));
            }
            let mut dst = vec![0xA5u8; count * S::BYTES];
            ops::norm_into_with(&mut dst, &map, src);
            assert_eq!(dst, expected, "norm count {count}");
        }
        // restrict over members: all bits set, preimages match.
        {
            let src = &member_bytes[..count * T::BYTES];
            let mut dst = vec![0xA5u8; count * S::BYTES];
            let mut valid = vec![0xA5u8; bits::bytes_for(count)];
            ops::restrict_into_with(&mut dst, &mut valid, &map, src);
            for (i, (d, s)) in dst
                .chunks_exact(S::BYTES)
                .zip(src.chunks_exact(T::BYTES))
                .enumerate()
            {
                let expected = map.restrict(T::decode(s)).expect("a member restricts");
                let mut lane = vec![0u8; S::BYTES];
                S::encode(&mut lane, expected);
                assert_eq!(d, lane.as_slice(), "restrict member lane {i} count {count}");
                assert_eq!(
                    valid[i / 8] >> (i % 8) & 1,
                    1,
                    "restrict member bit {i} count {count}"
                );
            }
            for b in count..bits::bytes_for(count) * 8 {
                assert_eq!(
                    valid[b / 8] >> (b % 8) & 1,
                    0,
                    "restrict member tail bit {b} count {count}"
                );
            }
        }
        // restrict over nonmembers: canonical source zero, no bits set.
        {
            let src = &stranger_bytes[..count * T::BYTES];
            let mut dst = vec![0xA5u8; count * S::BYTES];
            let mut valid = vec![0xA5u8; bits::bytes_for(count)];
            ops::restrict_into_with(&mut dst, &mut valid, &map, src);
            let mut zero = vec![0u8; S::BYTES];
            S::encode(&mut zero, Elem::<S>::ZERO);
            for d in dst.chunks_exact(S::BYTES) {
                assert_eq!(d, zero.as_slice(), "restrict outsider zero count {count}");
            }
            assert!(
                valid.iter().all(|&b| b == 0),
                "restrict outsider bitmap count {count}"
            );
        }
        // contains over the mixed row: bitmap only.
        {
            let src = &mixed_bytes[..count * T::BYTES];
            let mut valid = vec![0xA5u8; bits::bytes_for(count)];
            ops::contains_into_with(&mut valid, &map, src);
            for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
                assert_eq!(
                    valid[i / 8] >> (i % 8) & 1,
                    u8::from(map.contains(T::decode(s))),
                    "contains bit {i} count {count}"
                );
            }
            for b in count..bits::bytes_for(count) * 8 {
                assert_eq!(
                    valid[b / 8] >> (b % 8) & 1,
                    0,
                    "contains tail bit {b} count {count}"
                );
            }
        }
    }
}

fn check_isomorphism_rows<S, T>()
where
    S: FieldBuffer,
    T: FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = Isomorphism::<S, T>::new().expect("a validated pair constructs");
    let map = PreparedIsomorphism::<S, T>::new(&compact);
    let probe: Vec<Elem<S>> = staged::<S>(16, 0x150);
    for &x in &probe {
        assert_eq!(
            map.apply(x),
            compact.apply(x),
            "prepared agrees with its map"
        );
    }
    let top = *COUNTS.iter().max().unwrap_or(&1);
    let sources: Vec<Elem<S>> = staged(top, 0x150);
    let images: Vec<Elem<T>> = sources.iter().map(|x| map.apply(*x)).collect();
    let source_bytes = encode_all::<S>(&sources);
    let image_bytes = encode_all::<T>(&images);
    for &count in &COUNTS {
        let src = &source_bytes[..count * S::BYTES];
        let mut expected = vec![0u8; count * T::BYTES];
        for (d, s) in expected
            .chunks_exact_mut(T::BYTES)
            .zip(src.chunks_exact(S::BYTES))
        {
            T::encode(d, map.apply(S::decode(s)));
        }
        let mut dst = vec![0xA5u8; count * T::BYTES];
        ops::apply_into_with(&mut dst, &map, src);
        assert_eq!(dst, expected, "apply count {count}");

        let src = &image_bytes[..count * T::BYTES];
        let mut expected = vec![0u8; count * S::BYTES];
        for (d, s) in expected
            .chunks_exact_mut(S::BYTES)
            .zip(src.chunks_exact(T::BYTES))
        {
            S::encode(d, map.unapply(T::decode(s)));
        }
        let mut dst = vec![0xA5u8; count * S::BYTES];
        ops::unapply_into_with(&mut dst, &map, src);
        assert_eq!(dst, expected, "unapply count {count}");
    }
}

#[test]
fn prepared_rows_match_the_scalar_loop() {
    check_embedding_rows::<Gf16, Gf64>(0x860, 1024);
    check_embedding_rows::<Aes8, Gf64>(0x861, 1024);
    check_embedding_rows::<R3, G81Ext>(0x862, 1024);
    check_embedding_rows::<R7, G343>(0x863, 1024);
    check_embedding_rows::<fgf::Mersenne31, QuadMersenne31>(0x864, 1024);
    check_embedding_rows::<fgf::Goldilocks, GldSquare>(0x865, 64);
    check_isomorphism_rows::<Fp64, Gf64>();
    check_isomorphism_rows::<Normal8, Aes8>();
}

type GldSquare = FiniteField<0xFFFF_FFFF_0000_0001, 2, fgf::Extension<GldSquareSpec, 2>>;
struct GldSquareSpec;
impl ExtensionSpec<2> for GldSquareSpec {
    type Base = fgf::Goldilocks;
    type Storage = Components;
    const NAME: &'static str = "GF(g^2) as t^2 - 7";
    const RELATION: Relation = Relation::from_coordinates(&[0xFFFF_FFFF_0000_0001 - 7, 0]);
}

/// Noncanonical alias lanes decode before the map runs: row output still
/// equals the scalar loop over the same bytes.
#[test]
fn noncanonical_alias_lanes_match_the_scalar_loop() {
    type M31 = fgf::Mersenne31;
    type Qm31 = QuadMersenne31;
    let compact = Embedding::<M31, Qm31>::new().expect("m31 embeds in qm31");
    let map = PreparedEmbedding::<M31, Qm31>::new(&compact);
    for &count in &COUNTS {
        let src = vec![0xFFu8; count * M31::BYTES];
        let mut expected = vec![0u8; count * Qm31::BYTES];
        for (d, s) in expected
            .chunks_exact_mut(Qm31::BYTES)
            .zip(src.chunks_exact(M31::BYTES))
        {
            Qm31::encode(d, map.embed(M31::decode(s)));
        }
        let mut dst = vec![0u8; count * Qm31::BYTES];
        ops::embed_into_with(&mut dst, &map, &src);
        assert_eq!(dst, expected, "alias embed count {count}");

        let tsrc = vec![0xFFu8; count * Qm31::BYTES];
        let mut texpected = vec![0u8; count * M31::BYTES];
        for (d, s) in texpected
            .chunks_exact_mut(M31::BYTES)
            .zip(tsrc.chunks_exact(Qm31::BYTES))
        {
            M31::encode(d, map.trace(Qm31::decode(s)));
        }
        let mut tdst = vec![0u8; count * M31::BYTES];
        ops::trace_into_with(&mut tdst, &map, &tsrc);
        assert_eq!(tdst, texpected, "alias trace count {count}");
    }
}

/// Mixed restriction: interleaved members and outsiders write preimages
/// beside canonical source zero, with the matching bitmap.
#[test]
fn mixed_restriction_marks_members_only() {
    let compact = Embedding::<R3, G81Ext>::new().expect("r3 embeds in g81");
    let map = PreparedEmbedding::<R3, G81Ext>::new(&compact);
    let members: Vec<Elem<G81Ext>> = staged::<R3>(64, 0x61_3D)
        .iter()
        .map(|x| map.embed(*x))
        .collect();
    let strangers = outsiders(&map, &members, 64);
    let mixed: Vec<Elem<G81Ext>> = (0..64)
        .map(|i| {
            if i % 2 == 0 {
                members[i / 2]
            } else {
                strangers[i / 2]
            }
        })
        .collect();
    let src = encode_all::<G81Ext>(&mixed);
    let mut dst = vec![0xA5u8; 64 * R3::BYTES];
    let mut valid = vec![0xA5u8; bits::bytes_for(64)];
    ops::restrict_into_with(&mut dst, &mut valid, &map, &src);
    let mut zero = vec![0u8; R3::BYTES];
    R3::encode(&mut zero, Elem::<R3>::ZERO);
    for i in 0..64 {
        let lane = &dst[i * R3::BYTES..(i + 1) * R3::BYTES];
        let bit = valid[i / 8] >> (i % 8) & 1;
        if i % 2 == 0 {
            let mut expected = vec![0u8; R3::BYTES];
            R3::encode(
                &mut expected,
                map.restrict(mixed[i]).expect("an even lane is a member"),
            );
            assert_eq!(lane, expected.as_slice(), "mixed lane {i}");
            assert_eq!(bit, 1, "mixed bit {i}");
        } else {
            assert_eq!(lane, zero.as_slice(), "mixed outsider lane {i}");
            assert_eq!(bit, 0, "mixed outsider bit {i}");
        }
    }
}

/// Valid zero is not a failure sentinel: embedding the source zero
/// restricts back to zero with its bit set.
#[test]
fn valid_zero_restricts_with_its_bit_set() {
    let compact = Embedding::<R3, G81Ext>::new().expect("r3 embeds in g81");
    let map = PreparedEmbedding::<R3, G81Ext>::new(&compact);
    let image = map.embed(Elem::<R3>::ZERO);
    assert!(map.contains(image), "the zero image is contained");
    assert_eq!(map.restrict(image), Some(Elem::<R3>::ZERO));
    let src = encode_all::<G81Ext>(&[image]);
    let mut dst = vec![0xA5u8; R3::BYTES];
    let mut valid = vec![0xA5u8; bits::bytes_for(1)];
    ops::restrict_into_with(&mut dst, &mut valid, &map, &src);
    let mut zero = vec![0u8; R3::BYTES];
    R3::encode(&mut zero, Elem::<R3>::ZERO);
    assert_eq!(dst, zero, "valid zero encodes as source zero");
    assert_eq!(valid, [1], "valid zero sets its bit");
    let mut only = vec![0xA5u8; bits::bytes_for(1)];
    ops::contains_into_with(&mut only, &map, &src);
    assert_eq!(only, [1], "contains marks the zero image");
}

/// Over the quadratic Mersenne extension the prepared trace is the
/// conjugate sum and the prepared norm is the scalar norm.
#[test]
fn qm31_prepared_rows_match_conjugation_and_norm() {
    type M31 = fgf::Mersenne31;
    type Qm31 = QuadMersenne31;
    let compact = Embedding::<M31, Qm31>::new().expect("m31 embeds in qm31");
    let map = PreparedEmbedding::<M31, Qm31>::new(&compact);
    let elems = [
        Elem::<Qm31>::from_raw([7, 11]),
        Elem::<Qm31>::from_raw([0x7FFF_FFFE, 0x7FFF_FFFF]),
        Elem::<Qm31>::ZERO,
        Elem::<Qm31>::ONE,
    ];
    for (i, &elem) in elems.iter().enumerate() {
        let sum = elem + elem.conjugate();
        let expected_trace = map
            .restrict(sum)
            .expect("the conjugate sum lies in the image");
        assert_eq!(map.trace(elem), expected_trace, "trace anchor {i}");
        assert_eq!(map.norm(elem), elem.norm(), "norm anchor {i}");
    }
    let src = encode_all::<Qm31>(&elems);
    let mut dst = vec![0u8; elems.len() * M31::BYTES];
    ops::norm_into_with(&mut dst, &map, &src);
    for (lane, &elem) in dst.chunks_exact(M31::BYTES).zip(elems.iter()) {
        assert_eq!(M31::decode(lane), elem.norm());
    }
}

/// Rows over a field above degree 64: the GF(3^66) tower exercises lane
/// counts past the 64-coordinate mark. Targets are built by member
/// arithmetic (embedding lifts and generator powers), never raw indices:
/// the presentation index space is far wider than any row under test.
#[test]
fn above_64_degree_rows_match_the_scalar_loop() {
    let compact = Embedding::<R3, G3e66>::new().expect("r3 embeds in g3e66");
    let map = PreparedEmbedding::<R3, G3e66>::new(&compact);
    assert_eq!(G3e66::DEGREE, 66, "the tower clears degree 64");
    // No `HasGenerator` on custom rows: powers of the lifted prime
    // element supply non-image targets beside the embedded members.
    let step = map.embed(Elem::<R3>::from_raw(2));
    let mut powers = vec![Elem::<G3e66>::ONE];
    for _ in 1..128 {
        powers.push(*powers.last().expect("a nonempty row") * step);
    }
    let lifts: Vec<Elem<G3e66>> = staged::<R3>(64, 0x66)
        .iter()
        .map(|x| map.embed(*x))
        .collect();
    // Target basis vectors past coordinate 63: prime-coordinate unit
    // vectors assembled from `from_prime_coordinates`, verified by
    // round-tripping through `prime_coordinates_into`.
    let mut basis: Vec<Elem<G3e66>> = Vec::new();
    // Target basis vectors past coordinate 63, assembled structurally
    // (`from_index` caps at 64 lanes): unit G3e11 coordinates lifted
    // through `from_components` at each level. Each unit round-trips
    // through encode/decode to prove the lane.
    for j in [0usize, 1, 10, 11, 32, 33, 63, 64, 65] {
        // Coordinate j lives in quadratic half j/33 with G3e33 coordinate
        // k = j%33: cubic slot k/11, G3e11 coordinate k%11.
        let k = j % 33;
        let unit11 = Elem::<G3e11>::from_index(3u128.pow((k % 11) as u32))
            .expect("a unit index sits below the G3e11 order");
        let (c0, c1, c2) = match k / 11 {
            0 => (unit11, Elem::<G3e11>::ZERO, Elem::<G3e11>::ZERO),
            1 => (Elem::<G3e11>::ZERO, unit11, Elem::<G3e11>::ZERO),
            _ => (Elem::<G3e11>::ZERO, Elem::<G3e11>::ZERO, unit11),
        };
        let unit33 = <G3e33 as fgf::ExtensionField<3>>::from_components([c0, c1, c2]);
        let unit = if j < 33 {
            <G3e66 as fgf::ExtensionField<2>>::from_components([unit33, Elem::<G3e33>::ZERO])
        } else {
            <G3e66 as fgf::ExtensionField<2>>::from_components([Elem::<G3e33>::ZERO, unit33])
        };
        let mut lane = vec![0u8; G3e66::BYTES];
        G3e66::encode(&mut lane, unit);
        assert_eq!(G3e66::decode(&lane), unit, "basis vector {j} round-trips");
        basis.push(unit);
    }
    // High-coordinate probes come first: trace/norm rows consume at
    // most 65 lanes, so a prime-field-first pool would never exercise
    // coordinates past 63.
    let pool: Vec<Elem<G3e66>> = basis.iter().copied().chain(lifts).chain(powers).collect();
    // Mixed outsiders: step-multiples that fall outside the image, with
    // compact-map expectations per lane.
    let rejected: Vec<Elem<G3e66>> = {
        let mut out = Vec::new();
        // The image is a subfield, closed under multiplication: probe
        // basis-vector sums, which leave the image at once.
        for (a, b) in basis.iter().zip(pool.iter()).take(64) {
            let probe = *a + *b;
            if !compact.contains(probe) {
                out.push(probe);
                if out.len() == 32 {
                    break;
                }
            }
        }
        // Fall back to scaled basis vectors if the sums all landed inside.
        let mut k = 0;
        while out.len() < 32 && k < 64 {
            k += 1;
            for v in basis.iter() {
                let probe = *v + pool[k % pool.len()];
                if !compact.contains(probe) && !out.contains(&probe) {
                    out.push(probe);
                    if out.len() == 32 {
                        break;
                    }
                }
            }
        }
        out
    };
    assert!(!rejected.is_empty(), "the tower admits outsiders");
    for &count in &[0usize, 1, 7, 8, 9, 65] {
        let elems: Vec<Elem<G3e66>> = (0..count).map(|i| pool[i % pool.len()]).collect();
        let src = encode_all::<G3e66>(&elems);
        let mut expected = vec![0u8; count * R3::BYTES];
        for (d, s) in expected
            .chunks_exact_mut(R3::BYTES)
            .zip(src.chunks_exact(G3e66::BYTES))
        {
            R3::encode(d, compact.trace(G3e66::decode(s)));
        }
        let mut dst = vec![0xA5u8; count * R3::BYTES];
        ops::trace_into_with(&mut dst, &map, &src);
        assert_eq!(dst, expected, "above-64 trace count {count}");

        // Mixed outsiders restrict to canonical zero with cleared bits,
        // per the compact map's own answers.
        {
            let mixed: Vec<Elem<G3e66>> = (0..count)
                .map(|i| {
                    if i % 2 == 0 {
                        pool[i % pool.len()]
                    } else {
                        rejected[(i / 2) % rejected.len()]
                    }
                })
                .collect();
            let msrc = encode_all::<G3e66>(&mixed);
            let mut mdst = vec![0xA5u8; count * R3::BYTES];
            let mut mvalid = vec![0xA5u8; bits::bytes_for(count)];
            ops::restrict_into_with(&mut mdst, &mut mvalid, &map, &msrc);
            for (i, (d, y)) in mdst
                .chunks_exact(R3::BYTES)
                .zip(msrc.chunks_exact(G3e66::BYTES))
                .enumerate()
            {
                let expect_compact = compact.restrict(G3e66::decode(y));
                let mut lane = vec![0u8; R3::BYTES];
                match expect_compact {
                    Some(preimage) => {
                        R3::encode(&mut lane, preimage);
                        assert_eq!(mvalid[i / 8] >> (i % 8) & 1, 1, "mixed bit {i}");
                    }
                    None => {
                        R3::encode(&mut lane, Elem::<R3>::ZERO);
                        assert_eq!(mvalid[i / 8] >> (i % 8) & 1, 0, "mixed bit {i}");
                    }
                }
                assert_eq!(map.restrict(G3e66::decode(y)), expect_compact);
                assert_eq!(d, lane.as_slice(), "mixed lane {i} count {count}");
            }
        }

        let mut ndst = vec![0xA5u8; count * R3::BYTES];
        ops::norm_into_with(&mut ndst, &map, &src);
        for (lane, s) in ndst
            .chunks_exact(R3::BYTES)
            .zip(src.chunks_exact(G3e66::BYTES))
        {
            assert_eq!(R3::decode(lane), compact.norm(G3e66::decode(s)));
        }

        let mut edst = vec![0xA5u8; count * R3::BYTES];
        let mut valid = vec![0xA5u8; bits::bytes_for(count)];
        ops::restrict_into_with(&mut edst, &mut valid, &map, &src);
        for (i, (d, s)) in edst
            .chunks_exact(R3::BYTES)
            .zip(src.chunks_exact(G3e66::BYTES))
            .enumerate()
        {
            let expected = map.restrict(G3e66::decode(s));
            let mut lane = vec![0u8; R3::BYTES];
            match expected {
                Some(preimage) => {
                    R3::encode(&mut lane, preimage);
                    assert_eq!(valid[i / 8] >> (i % 8) & 1, 1, "above-64 bit {i}");
                }
                None => {
                    R3::encode(&mut lane, Elem::<R3>::ZERO);
                    assert_eq!(valid[i / 8] >> (i % 8) & 1, 0, "above-64 bit {i}");
                }
            }
            assert_eq!(d, lane.as_slice(), "above-64 lane {i} count {count}");
        }
    }
}

struct G3e11Spec;
impl ExtensionSpec<11> for G3e11Spec {
    type Base = R3;
    type Storage = Components;
    const NAME: &'static str = "GF(3^11) as x^11 + 2x^9 + 1";
    const RELATION: Relation = Relation::from_coordinates(&[1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0]);
}
type G3e11 = FiniteField<3, 11, Extension<G3e11Spec, 11>>;
struct G3e33Spec;
impl ExtensionSpec<3> for G3e33Spec {
    type Base = G3e11;
    type Storage = Components;
    const NAME: &'static str = "GF(3^33) as x^3 + 2x + 1";
    const RELATION: Relation = Relation::from_coordinates(&[
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
    ]);
}
type G3e33 = FiniteField<3, 33, Extension<G3e33Spec, 3>>;
struct G3e66Spec;
impl ExtensionSpec<2> for G3e66Spec {
    type Base = G3e33;
    type Storage = Components;
    const NAME: &'static str = "GF(3^66) as t^2 + 1";
    const RELATION: Relation = Relation::from_coordinates(&[
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    ]);
}
type G3e66 = FiniteField<3, 66, Extension<G3e66Spec, 2>>;

/// Random-byte rows decode before the map runs: prepared rows equal the
/// per-element scalar loop on unstructured input.
#[test]
fn noise_rows_match_the_scalar_loop() {
    let compact = Embedding::<Aes8, Gf64>::new().expect("aes8 embeds in gf64");
    let map = PreparedEmbedding::<Aes8, Gf64>::new(&compact);
    for &count in &[7usize, 33, 65, 129] {
        let src = noise(count * Gf64::BYTES, 0x9015 + count as u64);
        let mut expected = vec![0u8; count * Aes8::BYTES];
        for (d, s) in expected
            .chunks_exact_mut(Aes8::BYTES)
            .zip(src.chunks_exact(Gf64::BYTES))
        {
            Aes8::encode(d, map.trace(Gf64::decode(s)));
        }
        let mut dst = vec![0xA5u8; count * Aes8::BYTES];
        ops::trace_into_with(&mut dst, &map, &src);
        assert_eq!(dst, expected, "noise trace count {count}");
    }
}

// ---------------------------------------------------------------------------
// Bit-packed GF(2) endpoint rows (`ops::map_bits`)
// ---------------------------------------------------------------------------

type Gf1 = FiniteField<2, 1, Polynomial<3>>;
type Res2 = FiniteField<2, 1, Residue>;
type Gf32 = FiniteField<2, 32, fgf::Extension<fgf::Rijndael32, 2>>;
type Cantor8 = FiniteField<2, 8, fgf::Cantor<0x11B, 0x20>>;

struct OverAesSpec;
impl ExtensionSpec<2> for OverAesSpec {
    type Base = Aes8;
    type Storage = Components;
    const NAME: &'static str = "GF(2^16) as t^2 + 0x53*t + 1";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[1, 0x53]);
}
type OverAes = FiniteField<2, 16, Extension<OverAesSpec, 2>>;

/// `t^3 + 2` over `Aes8`: irreducible (no root in GF(2^8)), so a legal
/// three-byte `Components` row — the odd lane width stays on the portable
/// body while remaining a valid `BinaryField + FieldBuffer` target.
struct D3AesSpec;
impl ExtensionSpec<3> for D3AesSpec {
    type Base = Aes8;
    type Storage = Components;
    const NAME: &'static str = "GF(2^24) as t^3 + 2";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[2, 0, 0]);
}
type D3Aes = FiniteField<2, 24, Extension<D3AesSpec, 3>>;

/// Bit counts straddle byte, 64-bit word, and vector boundaries, plus one
/// large tail-heavy count.
const BIT_COUNTS: [usize; 21] = [
    0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 511, 512, 513, 4099,
];

/// Bit `i` of a packed buffer.
fn bit_of(buf: &[u8], i: usize) -> u8 {
    (buf[i / 8] >> (i % 8)) & 1
}

/// Assemble `bits[i] in {0,1}` into the packed LSB-first form with cleared
/// tail bits — the expected shape of every packed output.
fn pack_bits(bits: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; bits::bytes_for(bits.len())];
    for (i, &bit) in bits.iter().enumerate() {
        out[i / 8] |= bit << (i % 8);
    }
    out
}

/// A packed source row plus one surplus byte, every byte noise: padding bits
/// and surplus are garbage, so a row that reads outside `[0, bits)` leaks.
fn noisy_packed(bits: usize, seed: u64) -> Vec<u8> {
    noise(bits::bytes_for(bits) + 1, seed | 1)
}

/// Target lanes outside the two-element image, staged by `contains` probes.
fn bits_outsiders<S, T>(map: &PreparedEmbedding<S, T>, seed: u64, n: usize) -> Vec<Elem<T>>
where
    S: Field,
    T: Field<Characteristic = S::Characteristic>,
{
    let mut out = Vec::with_capacity(n);
    let mut state = seed | 1;
    let mut probes = 0u64;
    while out.len() < n {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let index = u128::from(state >> 33) % T::ORDER;
        let candidate = Elem::<T>::from_index(index).expect("a sampled index sits below the order");
        if !map.contains(candidate) {
            out.push(candidate);
        }
        probes += 1;
        assert!(
            probes < 1_000_000,
            "the image of degree {} into {} is too dense for outsiders",
            S::DEGREE,
            T::DEGREE
        );
    }
    out
}

/// Every bit-packed row against the per-bit scalar loop: bits are read by
/// `i/8, i%8`, run through the prepared scalar methods, and compared lane
/// by lane and bit by bit.
fn check_bit_rows<S, T>(seed: u64)
where
    S: PrimeField<Characteristic = PrimeCharacteristic<2>>,
    T: BinaryField + FieldBuffer<Characteristic = S::Characteristic>,
{
    let compact = Embedding::<S, T>::new().expect("the prime field embeds in a binary row");
    let map = PreparedEmbedding::<S, T>::new(&compact);
    let zero = Elem::<S>::ZERO;
    let one = Elem::<S>::ONE;
    let top = *BIT_COUNTS.iter().max().unwrap_or(&1);

    // Packed source bits (garbage padding and surplus included) and the
    // member lanes their embed images define: valid zeros and valid ones
    // both occur because the noise bits mix.
    let source = noisy_packed(top, seed);
    let member_bytes: Vec<u8> = (0..top)
        .flat_map(|i| {
            let mut lane = [0u8; 8];
            T::encode(
                &mut lane[..T::BYTES],
                map.embed(if bit_of(&source, i) == 1 { one } else { zero }),
            );
            lane[..T::BYTES].to_vec()
        })
        .collect();
    let strangers = bits_outsiders::<S, T>(&map, seed ^ 0x5DEE, 65.min(top));
    let noise_lanes = encode_all::<T>(&staged::<T>(top, seed ^ 0xCECE));
    let mixed_bytes: Vec<u8> = (0..top)
        .flat_map(|i| {
            let elem = if i % 2 == 0 {
                map.embed(if bit_of(&source, i) == 1 { one } else { zero })
            } else {
                strangers[(i / 2) % strangers.len()]
            };
            let mut lane = [0u8; 8];
            T::encode(&mut lane[..T::BYTES], elem);
            lane[..T::BYTES].to_vec()
        })
        .collect();

    for &count in &BIT_COUNTS {
        let packed = bits::bytes_for(count);
        // embed: garbage-packed input with surplus, exact lane output.
        {
            let src = &source[..packed + 1];
            let mut expected = vec![0u8; count * T::BYTES];
            for (i, d) in expected.chunks_exact_mut(T::BYTES).enumerate() {
                T::encode(d, map.embed(if bit_of(src, i) == 1 { one } else { zero }));
            }
            let mut dst = vec![0xA5u8; count * T::BYTES];
            ops::map_bits::embed_into_with(&mut dst, &map, src, count);
            assert_eq!(dst, expected, "bit embed count {count}");
        }
        // trace and norm: noise lanes in, packed bits out.
        for (op, method) in [("trace", 0u8), ("norm", 1u8)] {
            let src = &noise_lanes[..count * T::BYTES];
            let mut expected = vec![0u8; count];
            for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
                let elem = T::decode(s);
                let image = if method == 0 {
                    map.trace(elem)
                } else {
                    map.norm(elem)
                };
                expected[i] = u8::from(image == one);
            }
            let mut dst = vec![0xA5u8; packed];
            if method == 0 {
                ops::map_bits::trace_into_with(&mut dst, &map, src, count);
            } else {
                ops::map_bits::norm_into_with(&mut dst, &map, src, count);
            }
            assert_eq!(dst, pack_bits(&expected[..count]), "bit {op} count {count}");
        }
        // contains over members and mixed rows.
        for (label, bytes) in [("members", &member_bytes), ("mixed", &mixed_bytes)] {
            let src = &bytes[..count * T::BYTES];
            let mut expected = vec![0u8; count];
            for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
                expected[i] = u8::from(map.contains(T::decode(s)));
            }
            let mut valid = vec![0xA5u8; packed];
            ops::map_bits::contains_into_with(&mut valid, &map, src, count);
            assert_eq!(
                valid,
                pack_bits(&expected[..count]),
                "bit contains {label} {count}"
            );
        }
        // restrict over members (value bit = source bit, all valid bits set)
        // and over mixed rows (outsiders write zero bits with no valid bit).
        for (label, bytes) in [("members", &member_bytes), ("mixed", &mixed_bytes)] {
            let src = &bytes[..count * T::BYTES];
            let mut expected_value = vec![0u8; count];
            let mut expected_valid = vec![0u8; count];
            for (i, s) in src.chunks_exact(T::BYTES).enumerate() {
                let elem = T::decode(s);
                expected_valid[i] = u8::from(map.contains(elem));
                expected_value[i] = match map.restrict(elem) {
                    Some(preimage) => u8::from(preimage == one),
                    None => 0,
                };
            }
            let mut dst = vec![0xA5u8; packed];
            let mut valid = vec![0xA5u8; packed];
            ops::map_bits::restrict_into_with(&mut dst, &mut valid, &map, src, count);
            assert_eq!(
                dst,
                pack_bits(&expected_value[..count]),
                "bit restrict value {label} {count}"
            );
            assert_eq!(
                valid,
                pack_bits(&expected_valid[..count]),
                "bit restrict valid {label} {count}"
            );
        }
    }
}

#[test]
fn bit_rows_match_the_scalar_loop() {
    check_bit_rows::<Gf1, Aes8>(0x101);
    check_bit_rows::<Gf1, Normal8>(0x102);
    check_bit_rows::<Gf1, Cantor8>(0x103);
    check_bit_rows::<Gf1, Gf16>(0x104);
    check_bit_rows::<Gf1, Gf32>(0x105);
    check_bit_rows::<Gf1, Gf64>(0x106);
    check_bit_rows::<Gf1, Fp64>(0x107);
    check_bit_rows::<Gf1, OverAes>(0x108);
    check_bit_rows::<Gf1, D3Aes>(0x109);
    check_bit_rows::<Res2, Gf64>(0x10A);
}

#[test]
fn bit_embed_known_answer() {
    // 0b0111_1001: bits 0, 3, 4, 5, 6 are set.
    let byte = 0b0111_1001u8;
    let map = PreparedEmbedding::<Gf1, Aes8>::new(&Embedding::<Gf1, Aes8>::new().unwrap());
    let mut dst = vec![0u8; 7];
    ops::map_bits::embed_into_with(&mut dst, &map, &[byte], 7);
    assert_eq!(dst, [0x01, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01]);

    // The normal-basis identity is 0xFF, not 0x01.
    let map = PreparedEmbedding::<Gf1, Normal8>::new(&Embedding::<Gf1, Normal8>::new().unwrap());
    let mut dst = vec![0u8; 7];
    ops::map_bits::embed_into_with(&mut dst, &map, &[byte], 7);
    assert_eq!(dst, [0xFF, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF]);
}

#[test]
fn binary_fieldbuffer_lanes_are_coordinate_words() {
    fn check<T: BinaryField + FieldBuffer>() {
        assert_eq!(
            8 * T::BYTES,
            T::DEGREE as usize,
            "{} lanes are full coordinate words",
            T::NAME
        );
        let mut lane = [0u8; 8];
        T::encode(&mut lane[..T::BYTES], Elem::<T>::ZERO);
        assert!(
            lane[..T::BYTES].iter().all(|&b| b == 0),
            "{} canonical zero lane is all zero bytes",
            T::NAME
        );
        // Exhaustive at one byte; staged samples above.
        let samples: Vec<Elem<T>> = if T::ORDER <= 256 {
            (0..T::ORDER)
                .map(|i| Elem::<T>::from_index(i).unwrap())
                .collect()
        } else {
            staged::<T>(257, 0xB17)
        };
        for elem in samples {
            T::encode(&mut lane[..T::BYTES], elem);
            let mut word = 0u64;
            for (shift, &byte) in lane[..T::BYTES].iter().enumerate() {
                word |= u64::from(byte) << (8 * shift);
            }
            assert_eq!(
                word,
                elem.to_index() as u64,
                "{} lane little-endian word equals the coordinate word",
                T::NAME
            );
        }
    }
    check::<Aes8>();
    check::<Normal8>();
    check::<Cantor8>();
    check::<Gf16>();
    check::<Gf32>();
    check::<Gf64>();
    check::<Fp64>();
    check::<OverAes>();
    check::<D3Aes>();
}
#[cfg(feature = "std")]
mod panics {
    use super::*;

    fn catch(body: impl FnOnce()) -> (String, bool) {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body));
        match result {
            Ok(()) => (String::new(), false),
            Err(payload) => {
                let message = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                    .unwrap_or_default();
                (message, true)
            }
        }
    }

    #[test]
    fn geometry_failures_name_the_operand_and_leave_outputs_unchanged() {
        let compact = Embedding::<Gf16, Gf64>::new().expect("gf16 embeds in gf64");
        let map = PreparedEmbedding::<Gf16, Gf64>::new(&compact);
        let iso_compact = Isomorphism::<Fp64, Gf64>::new().expect("fp64 is isomorphic to gf64");
        let iso = PreparedIsomorphism::<Fp64, Gf64>::new(&iso_compact);

        // embed: short dst names `dst`, destination untouched.
        let mut dst = vec![0xA5u8; 8];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::embed_into_with(&mut dst, &map, &[0u8; 16]);
        });
        assert!(panicked, "short dst must panic");
        assert!(
            message.contains("embed_into_with") && message.contains("dst"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed embed must not write dst");

        // embed: partial trailing source element names `src`.
        let mut dst = vec![0xA5u8; 32];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::embed_into_with(&mut dst, &map, &[0u8; 17]);
        });
        assert!(panicked, "partial src must panic");
        assert!(
            message.contains("embed_into_with") && message.contains("src"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed embed must not write dst");

        // trace: short dst names `dst`.
        let mut dst = vec![0xA5u8; 2];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::trace_into_with(&mut dst, &map, &[0u8; 16]);
        });
        assert!(panicked, "short trace dst must panic");
        assert!(
            message.contains("trace_into_with") && message.contains("dst"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed trace must not write dst");

        // norm: partial trailing target element.
        let mut dst = vec![0xA5u8; 4];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::norm_into_with(&mut dst, &map, &[0u8; 15]);
        });
        assert!(panicked, "partial norm src must panic");
        assert!(
            message.contains("norm_into_with") && message.contains("src"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed norm must not write dst");

        // restrict: wrong bitmap length names `valid`, both outputs untouched.
        let mut dst = vec![0xA5u8; 8];
        let mut valid = vec![0xA5u8; 3];
        let (d_before, v_before) = (dst.clone(), valid.clone());
        let (message, panicked) = catch(|| {
            ops::restrict_into_with(&mut dst, &mut valid, &map, &[0u8; 32]);
        });
        assert!(panicked, "wrong bitmap must panic");
        assert!(
            message.contains("restrict_into_with") && message.contains("valid"),
            "unexpected message: {message}"
        );
        assert_eq!(
            (dst, valid),
            (d_before, v_before),
            "failed restrict writes nothing"
        );

        // restrict: short dst with a right-sized bitmap.
        let mut dst = vec![0xA5u8; 4];
        let mut valid = vec![0xA5u8; bits::bytes_for(4)];
        let (d_before, v_before) = (dst.clone(), valid.clone());
        let (message, panicked) = catch(|| {
            ops::restrict_into_with(&mut dst, &mut valid, &map, &[0u8; 32]);
        });
        assert!(panicked, "short restrict dst must panic");
        assert!(
            message.contains("restrict_into_with") && message.contains("dst"),
            "unexpected message: {message}"
        );
        assert_eq!(
            (dst, valid),
            (d_before, v_before),
            "failed restrict writes nothing"
        );

        // contains: wrong bitmap length names `valid`.
        let mut valid = vec![0xA5u8; 5];
        let v_before = valid.clone();
        let (message, panicked) = catch(|| {
            ops::contains_into_with(&mut valid, &map, &[0u8; 32]);
        });
        assert!(panicked, "wrong contains bitmap must panic");
        assert!(
            message.contains("contains_into_with") && message.contains("valid"),
            "unexpected message: {message}"
        );
        assert_eq!(valid, v_before, "failed contains must not write valid");

        // apply / unapply geometry.
        let mut dst = vec![0xA5u8; 8];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::apply_into_with(&mut dst, &iso, &[0u8; 16]);
        });
        assert!(panicked, "short apply dst must panic");
        assert!(
            message.contains("apply_into_with") && message.contains("dst"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed apply must not write dst");

        let mut dst = vec![0xA5u8; 16];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::unapply_into_with(&mut dst, &iso, &[0u8; 8]);
        });
        assert!(panicked, "short unapply src must panic");
        assert!(
            message.contains("unapply_into_with"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed unapply must not write dst");
    }

    #[test]
    fn bit_geometry_failures_name_the_operand_and_leave_outputs_unchanged() {
        let map = PreparedEmbedding::<Gf1, Gf16>::new(&Embedding::<Gf1, Gf16>::new().unwrap());

        // embed: short packed coverage names `src`.
        let mut dst = vec![0xA5u8; 32];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::map_bits::embed_into_with(&mut dst, &map, &[0u8; 1], 16);
        });
        assert!(panicked, "short packed src must panic");
        assert!(
            message.contains("embed_into_with") && message.contains("src"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed bit embed must not write dst");

        // embed: wrong lane count names `dst`.
        let mut dst = vec![0xA5u8; 31];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::map_bits::embed_into_with(&mut dst, &map, &[0u8; 2], 16);
        });
        assert!(panicked, "wrong lane count must panic");
        assert!(
            message.contains("embed_into_with") && message.contains("dst"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed bit embed must not write dst");

        // trace: oversized bitmap names `dst` (exact outputs, not coverage).
        let mut dst = vec![0xA5u8; 3];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::map_bits::trace_into_with(&mut dst, &map, &[0u8; 32], 16);
        });
        assert!(panicked, "oversized packed dst must panic");
        assert!(
            message.contains("trace_into_with") && message.contains("dst"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "failed bit trace must not write dst");

        // restrict: oversized `valid` names `valid` with both outputs
        // untouched (the second output is validated before any write).
        let mut dst = vec![0xA5u8; 2];
        let mut valid = vec![0xA5u8; 3];
        let (d_before, v_before) = (dst.clone(), valid.clone());
        let (message, panicked) = catch(|| {
            ops::map_bits::restrict_into_with(&mut dst, &mut valid, &map, &[0u8; 32], 16);
        });
        assert!(panicked, "oversized valid must panic");
        assert!(
            message.contains("restrict_into_with") && message.contains("valid"),
            "unexpected message: {message}"
        );
        assert_eq!(
            (dst, valid),
            (d_before, v_before),
            "failed bit restrict writes nothing"
        );

        // contains: `bits == 0` with a nonempty bitmap names `valid`.
        let mut valid = vec![0xA5u8; 1];
        let v_before = valid.clone();
        let (message, panicked) = catch(|| {
            ops::map_bits::contains_into_with(&mut valid, &map, &[], 0);
        });
        assert!(panicked, "empty row with nonempty bitmap must panic");
        assert!(
            message.contains("contains_into_with") && message.contains("valid"),
            "unexpected message: {message}"
        );
        assert_eq!(valid, v_before, "failed bit contains must not write valid");

        // checked span product overflow, reached through the public row.
        let mut dst = vec![0xA5u8; 0];
        let before = dst.clone();
        let (message, panicked) = catch(|| {
            ops::map_bits::trace_into_with(&mut dst, &map, &[], usize::MAX);
        });
        assert!(panicked, "span overflow must panic");
        assert!(
            message.contains("trace_into_with")
                && message.contains("src")
                && message.contains("overflow"),
            "unexpected message: {message}"
        );
        assert_eq!(dst, before, "overflowing bit trace must not write dst");
    }
}
