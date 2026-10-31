//! The shared GF(2)-linear engine over raw binary-field coordinate words.
//!
//! A [`LinearMap`] carries the images of a coordinate word's bits as a
//! column table: input bit `k` contributes `columns[k]` to the output word,
//! so one table serves fixed multiplication by a coefficient and binary maps
//! alike, at up to sixty-four input and output bits. The engine
//! reads descriptors through the crate's characteristic-two word layer and
//! never couples to a concrete field row.
//!
//! [`multiplication`] builds the multiplication map of one coefficient over
//! a descriptor. Quadratic characteristic-two nodes derive their columns
//! recursively in the base — the four base maps of one coefficient pair
//! assemble both halves of every wide column without a full product per
//! column — and every other node reads its column images through
//! `word_mul`. Zero
//! coefficients return a zero table without column work, and construction
//! is stack-only at every recursion depth.
//!
//! The ISA children hold the vector bodies: SSE4.1 in `sse41`, AVX2 in
//! `avx2`. Each entry validates its geometry, applies the map through
//! lane-width equality against per-bit masks, and runs a scalar tail over
//! the column table. Broader rectangular maps can stage through padded lane
//! widths at the call site; this engine maps raw coordinate words and owns
//! no byte-layout conversion.

use crate::field::extension::arithmetic::{degree_of, word_mul};
use crate::field::extension::descriptor::{Descriptor, Node};

mod avx2;
mod sse41;

pub use avx2::{mul_add_avx2, mul_assign_avx2, mul_into_avx2};
pub use sse41::{mul_add_sse41, mul_assign_sse41, mul_into_sse41};

/// One GF(2)-linear map over raw coordinate words, as bit-column images.
///
/// Column `k` holds the image of the coordinate word with bit `k` set; the
/// output word is the XOR of the columns of the set input bits. Input bits
/// at or above `input_bits` contribute nothing, and result bits at or above
/// `output_bits` are zero even when a column carries higher bits. The table
/// never grows past the coordinate capacity of sixty-four bits.
#[derive(Clone, Copy, Debug)]
pub struct LinearMap {
    /// Image of each input bit; slot `k` is the image of bit `k`.
    pub columns: [u64; 64],
    /// Input width in bits; bits at or above contribute nothing.
    pub input_bits: usize,
    /// Output width in bits; result bits at or above are zero.
    pub output_bits: usize,
}

impl LinearMap {
    /// The output word the map assigns to the raw coordinate word `x`.
    ///
    /// Input bits at or above `input_bits` are ignored. The caller checks the
    /// map widths against the coordinate capacity before application.
    #[must_use]
    pub(crate) const fn apply(&self, x: u64) -> u64 {
        let mut acc = 0;
        let mut bit = 0;
        while bit < self.input_bits {
            if (x >> bit) & 1 == 1 {
                acc ^= self.columns[bit];
            }
            bit += 1;
        }
        acc & output_mask(self.output_bits)
    }
}

/// Set-bit mask of a word width in bits, saturating at the whole word.
#[must_use]
pub(crate) const fn output_mask(bits: usize) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    }
}

/// The multiplication map of one coefficient over the descriptor's root:
/// the map `x ↦ coefficient · x` on the root's coordinate words.
///
/// The descriptor must be characteristic two with a validated root of at
/// most the coordinate capacity, and `coefficient` must be a canonical
/// coordinate word of the root. Each application builds bounded stack
/// workspace without enlarging a reusable coefficient value or selecting a
/// backend.
///
/// # Panics
/// Panics when the root's absolute degree exceeds the coordinate capacity.
#[must_use]
pub fn multiplication(descriptor: &Descriptor, coefficient: u64) -> LinearMap {
    let bits = degree_of(descriptor, descriptor.root_index()) as usize;
    assert!(
        bits <= 64,
        "binary_linear: descriptor degree {bits} exceeds the 64-bit map capacity",
    );
    mul_map_at(descriptor, descriptor.root_index(), coefficient)
}

/// The multiplication map of `coefficient` over the node at `index`.
///
/// Quadratic characteristic-two nodes recurse in the base. Under
/// `t² = A·t + B`, the coefficient `c0 + c1·t` multiplies `x0 + x1·t` into
/// `(c0·x0 + B·c1·x1) + (c1·x0 + (c0 + A·c1)·x1)·t`, so the base maps of
/// `c0`, `c1`, `B·c1`, and `c0 + A·c1` supply both halves of every column:
/// column `k` is `M(c0)[k] | M(c1)[k] << m` and column `k + m` is
/// `M(B·c1)[k] | M(c0 + A·c1)[k] << m` for the base degree `m`. Every other
/// node reads each basis image through [`word_mul`].
fn mul_map_at(descriptor: &Descriptor, index: u8, coefficient: u64) -> LinearMap {
    let bits = degree_of(descriptor, index) as usize;
    if coefficient == 0 {
        return LinearMap {
            columns: [0; 64],
            input_bits: bits,
            output_bits: bits,
        };
    }
    let mut columns = [0u64; 64];
    if let Node::Extension {
        base,
        relative_degree: 2,
        base_degree,
        coefficients,
        characteristic: 2,
        ..
    } = descriptor.node(index)
    {
        let half = *base_degree as usize;
        let mask = output_mask(half);
        let low = mul_map_at(descriptor, *base, coefficient & mask);
        let high = mul_map_at(descriptor, *base, coefficient >> half);
        let scaled_constant = mul_map_at(
            descriptor,
            *base,
            word_mul(descriptor, *base, coefficients[0], coefficient >> half),
        );
        let scaled_cross = mul_map_at(
            descriptor,
            *base,
            (coefficient & mask)
                ^ word_mul(descriptor, *base, coefficients[1], coefficient >> half),
        );
        let mut slot = 0;
        while slot < half {
            columns[slot] = low.columns[slot] | (high.columns[slot] << half);
            columns[slot + half] =
                scaled_constant.columns[slot] | (scaled_cross.columns[slot] << half);
            slot += 1;
        }
    } else {
        let mut slot = 0;
        while slot < bits {
            columns[slot] = word_mul(descriptor, index, coefficient, 1u64 << slot);
            slot += 1;
        }
    }
    LinearMap {
        columns,
        input_bits: bits,
        output_bits: bits,
    }
}

/// Active input bits of one call, precomputed for the lane builders.
///
/// The vector bodies broadcast each active bit's mask and column once per
/// call, so zero columns drop out before the payload loop.
pub(crate) struct PreparedBits {
    /// Indices of the input bits whose column is nonzero, ascending.
    pub(crate) bits: [u8; 64],
    /// Occupied prefix length of `bits`.
    pub(crate) len: usize,
    /// Set bits of `output_bits`: the mask applied to every column image.
    pub(crate) out_mask: u64,
}

/// Collects the active input bits of `map`.
#[must_use]
pub(crate) const fn prepare_bits(map: &LinearMap) -> PreparedBits {
    let mut prepared = PreparedBits {
        bits: [0; 64],
        len: 0,
        out_mask: output_mask(map.output_bits),
    };
    let mut bit = 0;
    while bit < map.input_bits {
        if map.columns[bit] & prepared.out_mask != 0 {
            // The cursor stays below the coordinate capacity of sixty-four.
            #[allow(clippy::cast_possible_truncation)]
            let coordinate = bit as u8;
            prepared.bits[prepared.len] = coordinate;
            prepared.len += 1;
        }
        bit += 1;
    }
    prepared
}

/// Validates the shared geometry of one entry: an admissible element width,
/// whole elements, matching lengths, and a map that fits the element.
///
/// # Panics
/// Panics when any check fails, naming the failing operand.
pub(crate) fn check_geometry<const BYTES: usize>(
    name: &str,
    dst: &[u8],
    src: Option<&[u8]>,
    map: &LinearMap,
) {
    assert!(
        matches!(BYTES, 2 | 4 | 8),
        "{name}: element width {BYTES} is unsupported",
    );
    assert!(
        dst.len().is_multiple_of(BYTES),
        "{name}: buffer of {} bytes ends with a partial {BYTES}-byte element",
        dst.len(),
    );
    if let Some(src) = src {
        assert_eq!(
            dst.len(),
            src.len(),
            "{name}: dst is {} bytes but src is {} bytes",
            dst.len(),
            src.len(),
        );
    }
    assert!(
        map.input_bits <= BYTES * 8,
        "{name}: map input width {} exceeds the {BYTES}-byte element",
        map.input_bits,
    );
    assert!(
        map.output_bits <= BYTES * 8,
        "{name}: map output width {} exceeds the {BYTES}-byte element",
        map.output_bits,
    );
    assert!(
        map.input_bits <= 64 && map.output_bits <= 64,
        "{name}: map width exceeds the 64-bit coordinate capacity",
    );
}

/// Reads the little-endian coordinate word of one `BYTES`-byte element.
///
/// The element must hold exactly `BYTES` bytes, which the chunking in the
/// callers establishes.
#[must_use]
pub(crate) const fn read_word<const BYTES: usize>(element: &[u8]) -> u64 {
    let mut word = 0;
    let mut byte = 0;
    while byte < BYTES {
        word |= (element[byte] as u64) << (8 * byte);
        byte += 1;
    }
    word
}

/// Writes the little-endian coordinate word of one `BYTES`-byte element.
///
/// The element must hold exactly `BYTES` bytes, which the chunking in the
/// callers establishes.
pub(crate) const fn write_word<const BYTES: usize>(element: &mut [u8], word: u64) {
    let mut byte = 0;
    while byte < BYTES {
        // The element holds the low `BYTES` bytes of the word.
        #[allow(clippy::cast_possible_truncation)]
        let value = (word >> (8 * byte)) as u8;
        element[byte] = value;
        byte += 1;
    }
}

/// Scalar tail of the add disposition: `dst ^= map(src)` per element.
pub(crate) fn tail_add<const BYTES: usize>(dst_tail: &mut [u8], src_tail: &[u8], map: &LinearMap) {
    for (dst, src) in dst_tail
        .chunks_exact_mut(BYTES)
        .zip(src_tail.chunks_exact(BYTES))
    {
        let sum = read_word::<BYTES>(dst) ^ map.apply(read_word::<BYTES>(src));
        write_word::<BYTES>(dst, sum);
    }
}

/// Scalar tail of the overwrite disposition: `dst = map(src)` per element.
pub(crate) fn tail_into<const BYTES: usize>(dst_tail: &mut [u8], src_tail: &[u8], map: &LinearMap) {
    for (dst, src) in dst_tail
        .chunks_exact_mut(BYTES)
        .zip(src_tail.chunks_exact(BYTES))
    {
        write_word::<BYTES>(dst, map.apply(read_word::<BYTES>(src)));
    }
}

/// Scalar tail of the assign disposition: `dst = map(dst)` per element.
pub(crate) fn tail_assign<const BYTES: usize>(dst_tail: &mut [u8], map: &LinearMap) {
    for dst in dst_tail.chunks_exact_mut(BYTES) {
        let word = read_word::<BYTES>(dst);
        write_word::<BYTES>(dst, map.apply(word));
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation)]

    use super::{LinearMap, avx2, multiplication, output_mask, read_word, sse41, write_word};
    use crate::field::BinaryField;
    use crate::field::binary::tower::{
        FanPaar16Field, FanPaar32Field, FanPaar64Field, Rijndael16Field, Rijndael32Field,
    };
    use crate::field::binary::{AES, RS};
    use crate::field::extension::arithmetic::{degree_of, word_mul};
    use crate::field::extension::descriptor::{Descriptor, PrimePresentation, Storage};
    use crate::field::{Cantor, FiniteField, Polynomial};
    use crate::kernel::Backend;
    use crate::kernel::tests::{expect_geometry_panic, noise};
    use archmage::{SimdToken as _, X64V2Token, X64V3Token};
    use std::vec;
    use std::vec::Vec;

    /// Next raw word of the fixed-seed generator.
    fn lcg(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        *state
    }

    /// Whether the host resolves to one of the requested tiers, honoring
    /// `SIMD_BACKEND` through the single detection source.
    fn host_supports(supported: &'static [Backend]) -> bool {
        let resolved = simdispatch::Selection::new("SIMD_BACKEND")
            .supports(supported)
            .resolve();
        resolved != Backend::Scalar
    }

    /// Byte-width samples: the two short-circuits, an extreme, and random
    /// words.
    fn byte_samples(state: &mut u64) -> Vec<u8> {
        vec![
            0,
            1,
            0x53,
            lcg(state) as u8,
            lcg(state) as u8,
            lcg(state) as u8,
        ]
    }

    /// Half-word samples over the same shape.
    fn half_samples(state: &mut u64) -> Vec<u16> {
        vec![
            0,
            1,
            0x8000,
            lcg(state) as u16,
            lcg(state) as u16,
            lcg(state) as u16,
        ]
    }

    /// Word samples over the same shape.
    fn word_samples(state: &mut u64) -> Vec<u32> {
        vec![
            0,
            1,
            0x8000_0000,
            lcg(state) as u32,
            lcg(state) as u32,
            lcg(state) as u32,
        ]
    }

    /// Double-word samples over the same shape.
    fn wide_samples(state: &mut u64) -> Vec<u64> {
        vec![
            0,
            1,
            0x8000_0000_0000_0000,
            lcg(state),
            lcg(state),
            lcg(state),
        ]
    }

    /// A zero map over `bits` input and output bits.
    fn zero_map(bits: usize) -> LinearMap {
        LinearMap {
            columns: [0; 64],
            input_bits: bits,
            output_bits: bits,
        }
    }

    /// Checks each coefficient map against the field row's arithmetic.
    fn check_field_maps<F>(name: &str, samples: &[F::Raw])
    where
        F: BinaryField,
        F::Raw: Into<u64>,
    {
        let descriptor = <F as BinaryField>::DESCRIPTION.as_descriptor();
        for &coefficient in samples {
            let map = multiplication(descriptor, coefficient.into());
            for &word in samples {
                let expected = F::mul_raw(coefficient, word);
                assert_eq!(
                    map.apply(word.into()),
                    expected.into(),
                    "{name}: coefficient {coefficient:?}, word {word:?}",
                );
            }
        }
    }

    /// Fixed multiplication maps against the shipped rows' own arithmetic:
    /// polynomial leaves, an ordered-basis root, and quadratic towers at
    /// several depths.
    #[test]
    fn multiplication_matches_field_arithmetic() {
        let mut state = 0x5EED_BEEF;
        let bytes = byte_samples(&mut state);
        check_field_maps::<FiniteField<2, 8, Polynomial<RS>>>("rs", &bytes);
        check_field_maps::<FiniteField<2, 8, Polynomial<AES>>>("aes", &bytes);
        check_field_maps::<FiniteField<2, 8, Cantor<0x11B, 0x20>>>("cantor", &bytes);
        let halves = half_samples(&mut state);
        check_field_maps::<FanPaar16Field>("fanpaar16", &halves);
        check_field_maps::<Rijndael16Field>("rijndael16", &halves);
        let words = word_samples(&mut state);
        check_field_maps::<FanPaar32Field>("fanpaar32", &words);
        check_field_maps::<Rijndael32Field>("rijndael32", &words);
        let wides = wide_samples(&mut state);
        check_field_maps::<FanPaar64Field>("fanpaar64", &wides);
    }

    /// Column-by-column equality of a built map against the descriptor's
    /// own word multiplication at the root.
    fn check_columns(name: &str, descriptor: &Descriptor, coefficients: &[u64]) {
        let root = descriptor.root_index();
        let bits = degree_of(descriptor, root) as usize;
        for &coefficient in coefficients {
            let masked = coefficient & output_mask(bits);
            let map = multiplication(descriptor, masked);
            for slot in 0..bits {
                assert_eq!(
                    map.columns[slot],
                    word_mul(descriptor, root, masked, 1u64 << slot),
                    "{name}: coefficient {masked:#x} column {slot}",
                );
            }
        }
    }

    /// Quadratic nodes recurse in the base; the assembled columns must
    /// equal the direct word products.
    #[test]
    fn quadratic_columns_match_word_multiplication() {
        let mut state = 0x0D15_EA5E;
        let coefficients = vec![0, 1, 0x8000, lcg(&mut state), lcg(&mut state)];
        let towers = [
            (
                "fanpaar16",
                <FanPaar16Field as BinaryField>::DESCRIPTION.as_descriptor(),
            ),
            (
                "rijndael16",
                <Rijndael16Field as BinaryField>::DESCRIPTION.as_descriptor(),
            ),
            (
                "fanpaar32",
                <FanPaar32Field as BinaryField>::DESCRIPTION.as_descriptor(),
            ),
            (
                "fanpaar64",
                <FanPaar64Field as BinaryField>::DESCRIPTION.as_descriptor(),
            ),
        ];
        for (name, descriptor) in towers {
            check_columns(name, descriptor, &coefficients);
        }
        let custom = Descriptor::polynomial(8, 0x11D).append_quadratic_words(0x53, 0xCA);
        check_columns("custom quadratic", &custom, &coefficients);
    }

    /// Nodes without the quadratic shortcut: a cubic relation over GF(2),
    /// an ordered-basis root, and the GF(2) prime leaf.
    #[test]
    fn generic_node_columns_match_word_multiplication() {
        let mut state = 0x1CE_B1AD;
        let coefficients = vec![0, 1, 3, lcg(&mut state), lcg(&mut state)];

        let prime = Descriptor::prime(2, PrimePresentation::Residue);
        let cubic = Descriptor::extension(
            &prime,
            2,
            3,
            3,
            &[1, 1, 0],
            Storage::PackedPadded { width: 8 },
        );
        cubic.validate_relation();
        check_columns("cubic relation", &cubic, &coefficients);
        let basis =
            <FiniteField<2, 8, Cantor<0x11B, 0x20>> as BinaryField>::DESCRIPTION.as_descriptor();
        check_columns("cantor basis root", basis, &coefficients);
        check_columns("prime two leaf", &prime, &coefficients);
    }

    /// Arbitrary maps over full-lane words: the zero map, a map with one
    /// active column, and dense maps with some zero columns.
    fn arbitrary_maps(bits: usize) -> Vec<LinearMap> {
        let mut state = 0x5EED_51ED;
        let mut maps = vec![zero_map(bits)];
        let mut single = zero_map(bits);
        single.columns[bits - 1] = lcg(&mut state) & output_mask(bits);
        maps.push(single);
        for _ in 0..4 {
            let mut columns = [0u64; 64];
            for column in columns.iter_mut().take(bits) {
                if lcg(&mut state) & 7 != 0 {
                    *column = lcg(&mut state) & output_mask(bits);
                }
            }
            maps.push(LinearMap {
                columns,
                input_bits: bits,
                output_bits: bits,
            });
        }
        maps
    }

    /// Multiplication maps of one field row for a spread of coefficients.
    fn field_maps<F>(coefficients: &[F::Raw]) -> Vec<LinearMap>
    where
        F: BinaryField,
        F::Raw: Into<u64>,
    {
        let descriptor = <F as BinaryField>::DESCRIPTION.as_descriptor();
        coefficients
            .iter()
            .map(|&coefficient| multiplication(descriptor, coefficient.into()))
            .collect()
    }

    /// Byte lengths around the 16- and 32-byte lane boundaries for
    /// `BYTES`-byte elements: sub-lane, single-element, boundary, and
    /// multi-lane sizes including empty.
    fn lane_lengths<const BYTES: usize>() -> Vec<usize> {
        [0, 1, 2, 3, 5, 8, 13, 16, 17, 31, 32, 33, 64, 65, 96]
            .iter()
            .map(|&elements| elements * BYTES)
            .collect()
    }

    /// Scalar oracle of the add disposition.
    fn oracle_add<const BYTES: usize>(dst: &mut [u8], map: &LinearMap, src: &[u8]) {
        for (dst_elem, src_elem) in dst.chunks_exact_mut(BYTES).zip(src.chunks_exact(BYTES)) {
            let sum = read_word::<BYTES>(dst_elem) ^ map.apply(read_word::<BYTES>(src_elem));
            write_word::<BYTES>(dst_elem, sum);
        }
    }

    /// Scalar oracle of the overwrite disposition.
    fn oracle_into<const BYTES: usize>(dst: &mut [u8], map: &LinearMap, src: &[u8]) {
        for (dst_elem, src_elem) in dst.chunks_exact_mut(BYTES).zip(src.chunks_exact(BYTES)) {
            write_word::<BYTES>(dst_elem, map.apply(read_word::<BYTES>(src_elem)));
        }
    }

    /// Scalar oracle of the assign disposition.
    fn oracle_assign<const BYTES: usize>(dst: &mut [u8], map: &LinearMap) {
        for dst_elem in dst.chunks_exact_mut(BYTES) {
            let word = read_word::<BYTES>(dst_elem);
            write_word::<BYTES>(dst_elem, map.apply(word));
        }
    }

    /// Differential over one tier and element width: every disposition,
    /// arbitrary and field-derived maps, lane-boundary lengths, misaligned
    /// destinations and sources, and intact guard bytes around the
    /// operated region.
    fn check_tier<const BYTES: usize>(
        name: &str,
        maps: &[LinearMap],
        mul_add: impl Fn(&mut [u8], &LinearMap, &[u8]),
        mul_into: impl Fn(&mut [u8], &LinearMap, &[u8]),
        mul_assign: impl Fn(&mut [u8], &LinearMap),
    ) {
        for len in lane_lengths::<BYTES>() {
            let wide_src = noise(len + 3, 0x51ED_2701 ^ len as u64);
            for (map_index, map) in maps.iter().enumerate() {
                for head in [0usize, 1, 3] {
                    let padded = len + head + 5;
                    let mut storage = noise(padded, 0x0D15);
                    let mut expected = storage.clone();
                    {
                        let (_, area) = storage.split_at_mut(head);
                        let (region, _) = area.split_at_mut(len);
                        let (_, expected_area) = expected.split_at_mut(head);
                        let (expected_region, _) = expected_area.split_at_mut(len);
                        let shifted = &wide_src[head..head + len];
                        mul_add(region, map, shifted);
                        oracle_add::<BYTES>(expected_region, map, shifted);
                    }
                    assert_eq!(
                        storage, expected,
                        "{name}: mul_add len {len} map {map_index} head {head}",
                    );
                    let mut storage = noise(padded, 0x0D16);
                    let mut expected = storage.clone();
                    {
                        let (_, area) = storage.split_at_mut(head);
                        let (region, _) = area.split_at_mut(len);
                        let (_, expected_area) = expected.split_at_mut(head);
                        let (expected_region, _) = expected_area.split_at_mut(len);
                        let shifted = &wide_src[head..head + len];
                        mul_into(region, map, shifted);
                        oracle_into::<BYTES>(expected_region, map, shifted);
                    }
                    assert_eq!(
                        storage, expected,
                        "{name}: mul_into len {len} map {map_index} head {head}",
                    );
                    let mut storage = noise(padded, 0x0D17);
                    let mut expected = storage.clone();
                    {
                        let (_, area) = storage.split_at_mut(head);
                        let (region, _) = area.split_at_mut(len);
                        let (_, expected_area) = expected.split_at_mut(head);
                        let (expected_region, _) = expected_area.split_at_mut(len);
                        mul_assign(region, map);
                        oracle_assign::<BYTES>(expected_region, map);
                    }
                    assert_eq!(
                        storage, expected,
                        "{name}: mul_assign len {len} map {map_index} head {head}",
                    );
                }
            }
        }
    }

    /// SSE4.1 differentials at two-, four-, and eight-byte element widths.
    #[test]
    fn sse41_linear_differentials() {
        if !host_supports(&[Backend::V2]) {
            std::eprintln!("skipping: no SSE4.1 tier on this host");
            return;
        }
        let token = X64V2Token::summon().expect("guard passed: the V2 tier supplies its token");
        let mut state = 0xAB1E_5EED;
        let mut narrow = arbitrary_maps(16);
        narrow.extend(field_maps::<FiniteField<2, 8, Polynomial<RS>>>(
            &byte_samples(&mut state),
        ));
        narrow.extend(field_maps::<FanPaar16Field>(&half_samples(&mut state)));
        narrow.extend(field_maps::<Rijndael16Field>(&half_samples(&mut state)));
        check_tier::<2>(
            "binary_linear.sse41[2]",
            &narrow,
            |dst, map, src| sse41::mul_add_sse41::<2>(token, dst, map, src),
            |dst, map, src| sse41::mul_into_sse41::<2>(token, dst, map, src),
            |dst, map| sse41::mul_assign_sse41::<2>(token, dst, map),
        );
        let mut middle = arbitrary_maps(32);
        middle.extend(field_maps::<FanPaar32Field>(&word_samples(&mut state)));
        middle.extend(field_maps::<Rijndael16Field>(&half_samples(&mut state)));
        check_tier::<4>(
            "binary_linear.sse41[4]",
            &middle,
            |dst, map, src| sse41::mul_add_sse41::<4>(token, dst, map, src),
            |dst, map, src| sse41::mul_into_sse41::<4>(token, dst, map, src),
            |dst, map| sse41::mul_assign_sse41::<4>(token, dst, map),
        );
        let mut wide = arbitrary_maps(64);
        wide.extend(field_maps::<FanPaar64Field>(&wide_samples(&mut state)));
        wide.extend(field_maps::<FanPaar32Field>(&word_samples(&mut state)));
        check_tier::<8>(
            "binary_linear.sse41[8]",
            &wide,
            |dst, map, src| sse41::mul_add_sse41::<8>(token, dst, map, src),
            |dst, map, src| sse41::mul_into_sse41::<8>(token, dst, map, src),
            |dst, map| sse41::mul_assign_sse41::<8>(token, dst, map),
        );
    }

    /// AVX2 differentials at two-, four-, and eight-byte element widths.
    #[test]
    fn avx2_linear_differentials() {
        if !host_supports(&[Backend::V3]) {
            std::eprintln!("skipping: no AVX2 tier on this host");
            return;
        }
        let token = X64V3Token::summon().expect("guard passed: the V3 tier supplies its token");
        let mut state = 0xAB1E_CAFE;
        let mut narrow = arbitrary_maps(16);
        narrow.extend(field_maps::<FiniteField<2, 8, Polynomial<RS>>>(
            &byte_samples(&mut state),
        ));
        narrow.extend(field_maps::<FanPaar16Field>(&half_samples(&mut state)));
        narrow.extend(field_maps::<Rijndael16Field>(&half_samples(&mut state)));
        check_tier::<2>(
            "binary_linear.avx2[2]",
            &narrow,
            |dst, map, src| avx2::mul_add_avx2::<2>(token, dst, map, src),
            |dst, map, src| avx2::mul_into_avx2::<2>(token, dst, map, src),
            |dst, map| avx2::mul_assign_avx2::<2>(token, dst, map),
        );
        let mut middle = arbitrary_maps(32);
        middle.extend(field_maps::<FanPaar32Field>(&word_samples(&mut state)));
        middle.extend(field_maps::<Rijndael16Field>(&half_samples(&mut state)));
        check_tier::<4>(
            "binary_linear.avx2[4]",
            &middle,
            |dst, map, src| avx2::mul_add_avx2::<4>(token, dst, map, src),
            |dst, map, src| avx2::mul_into_avx2::<4>(token, dst, map, src),
            |dst, map| avx2::mul_assign_avx2::<4>(token, dst, map),
        );
        let mut wide = arbitrary_maps(64);
        wide.extend(field_maps::<FanPaar64Field>(&wide_samples(&mut state)));
        wide.extend(field_maps::<FanPaar32Field>(&word_samples(&mut state)));
        check_tier::<8>(
            "binary_linear.avx2[8]",
            &wide,
            |dst, map, src| avx2::mul_add_avx2::<8>(token, dst, map, src),
            |dst, map, src| avx2::mul_into_avx2::<8>(token, dst, map, src),
            |dst, map| avx2::mul_assign_avx2::<8>(token, dst, map),
        );
    }

    /// SSE4.1 geometry rejections: partial elements, length mismatches, a
    /// map too wide for its element, and an unsupported element width.
    #[test]
    fn sse41_geometry_rejections() {
        if !host_supports(&[Backend::V2]) {
            std::eprintln!("skipping: no SSE4.1 tier on this host");
            return;
        }
        let token = X64V2Token::summon().expect("guard passed: the V2 tier supplies its token");
        let map = zero_map(16);
        let mut dst = [0u8; 16];
        let src = [0u8; 16];
        expect_geometry_panic(
            || sse41::mul_add_sse41::<2>(token, &mut dst[..3], &map, &src[..3]),
            "ends with a partial 2-byte element",
        );
        expect_geometry_panic(
            || sse41::mul_add_sse41::<2>(token, &mut dst[..4], &map, &src[..2]),
            "dst is 4 bytes but src is 2 bytes",
        );
        let mut wide = zero_map(16);
        wide.input_bits = 24;
        expect_geometry_panic(
            || sse41::mul_add_sse41::<2>(token, &mut dst, &wide, &src),
            "map input width 24 exceeds the 2-byte element",
        );
        wide.input_bits = 16;
        wide.output_bits = 24;
        expect_geometry_panic(
            || sse41::mul_assign_sse41::<2>(token, &mut dst, &wide),
            "map output width 24 exceeds the 2-byte element",
        );
        expect_geometry_panic(
            || sse41::mul_into_sse41::<16>(token, &mut dst, &map, &src),
            "element width 16 is unsupported",
        );
    }

    /// AVX2 geometry rejections over the same shapes.
    #[test]
    fn avx2_geometry_rejections() {
        if !host_supports(&[Backend::V3]) {
            std::eprintln!("skipping: no AVX2 tier on this host");
            return;
        }
        let token = X64V3Token::summon().expect("guard passed: the V3 tier supplies its token");
        let map = zero_map(16);
        let mut dst = [0u8; 16];
        let src = [0u8; 16];
        expect_geometry_panic(
            || avx2::mul_add_avx2::<2>(token, &mut dst[..3], &map, &src[..3]),
            "ends with a partial 2-byte element",
        );
        expect_geometry_panic(
            || avx2::mul_add_avx2::<2>(token, &mut dst[..4], &map, &src[..2]),
            "dst is 4 bytes but src is 2 bytes",
        );
        let mut wide = zero_map(16);
        wide.input_bits = 24;
        expect_geometry_panic(
            || avx2::mul_add_avx2::<2>(token, &mut dst, &wide, &src),
            "map input width 24 exceeds the 2-byte element",
        );
        wide.input_bits = 16;
        wide.output_bits = 24;
        expect_geometry_panic(
            || avx2::mul_assign_avx2::<2>(token, &mut dst, &wide),
            "map output width 24 exceeds the 2-byte element",
        );
        expect_geometry_panic(
            || avx2::mul_into_avx2::<16>(token, &mut dst, &map, &src),
            "element width 16 is unsupported",
        );
    }
}
