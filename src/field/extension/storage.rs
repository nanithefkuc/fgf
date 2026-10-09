//! The sealed storage family of extension rows.
//!
//! A storage decides the row's raw word and its byte, display, and component
//! forms: [`Components`] keeps one base element per component, while
//! [`Packed<W>`](Packed) and [`PackedPadded<W>`](PackedPadded) keep the
//! row's coordinate word in one integer, component `i` occupying the bits
//! `[i*B, (i+1)*B)` of the base degree `B`. The public [`ExtensionStorage`]
//! trait names the family and its raw word; the conversions the generic row
//! body runs on live on the crate-private [`StorageRepr`] subtrait and its
//! [`StorageBuffer`] and [`StorageDisplay`] extensions, in the row-hook
//! pattern: never as public methods.

use core::fmt::Formatter;
use core::hash::Hash;
use core::marker::PhantomData;

use super::arithmetic;
use super::descriptor::Storage;
use super::private::Sealed;
use crate::field::binary::BinaryField;
use crate::field::private::RowAccess;
use crate::field::{Elem, Field, FieldBuffer};

/// The storage word types a packed extension row may select.
pub trait PackedWord: Sealed {}

impl Sealed for u8 {}
impl Sealed for u16 {}
impl Sealed for u32 {}
impl Sealed for u64 {}

impl PackedWord for u8 {}
impl PackedWord for u16 {}
impl PackedWord for u32 {}
impl PackedWord for u64 {}

/// One base element per component, in the base's canonical raw form.
pub struct Components;

/// The row's coordinate word fills the storage word exactly.
pub struct Packed<W: PackedWord>(PhantomData<W>);

/// The row's coordinate word sits below the storage word width; every
/// storage bit at or above the field degree stays zero.
pub struct PackedPadded<W: PackedWord>(PhantomData<W>);
// Manual marker impls: derives would bound the traits on `W`, which the
// storage contract does not supply. The markers implement `Clone`, `Copy`,
// `Default`, and `Debug` only.

#[allow(clippy::expl_impl_clone_on_copy)]
impl Clone for Components {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for Components {}

impl Default for Components {
    fn default() -> Self {
        Self
    }
}

impl core::fmt::Debug for Components {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
        formatter.write_str("Components")
    }
}

macro_rules! storage_marker {
    ($storage:ident) => {
        impl<W: PackedWord> Clone for $storage<W> {
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<W: PackedWord> Copy for $storage<W> {}

        impl<W: PackedWord> Default for $storage<W> {
            fn default() -> Self {
                Self(PhantomData)
            }
        }

        impl<W: PackedWord> core::fmt::Debug for $storage<W> {
            fn fmt(&self, formatter: &mut Formatter<'_>) -> core::fmt::Result {
                formatter.write_str(core::stringify!($storage))
            }
        }
    };
}

storage_marker!(Packed);
storage_marker!(PackedPadded);

/// The element storage a relation node selects, sealed to the family above.
///
/// `Raw` is the row's raw word: the base-component array for
/// [`Components`], the coordinate word for [`Packed<W>`](Packed) and
/// [`PackedPadded<W>`](PackedPadded), which require a binary base.
pub trait ExtensionStorage<B: Field, const D: usize>: Sealed + 'static {
    /// The raw word of one element.
    type Raw: Copy + Eq + Ord + Hash + core::fmt::Debug + 'static;
}

impl Sealed for Components {}

impl<B: Field, const D: usize> ExtensionStorage<B, D> for Components {
    type Raw = [<B as Field>::Raw; D];
}

/// The component and raw-word conversions of one storage, behind the
/// `Field` seal.
///
/// `join` canonicalizes every component through the base and clears
/// padding, so a joined raw word is always canonical. `split` reads the
/// components of any raw word the base accepts.
pub(crate) trait StorageRepr<B: Field, const D: usize>: ExtensionStorage<B, D> {
    /// The descriptor's storage tag.
    const TAG: Storage;
    /// The additive identity's raw word: zero components.
    const ZERO: Self::Raw;
    /// The multiplicative identity's raw word: the base identity in
    /// component zero.
    const ONE: Self::Raw;
    /// The canonical form of a raw word.
    fn canonical(raw: Self::Raw) -> Self::Raw;
    /// The base components of a raw word, in component order.
    fn split(raw: Self::Raw) -> [<B as Field>::Raw; D];
    /// The raw word of base components, canonicalized on the way in.
    fn join(components: [<B as Field>::Raw; D]) -> Self::Raw;
}

/// The fixed-width byte form of one storage, for the rows that carry one.
///
/// Implemented for [`Packed<W>`](Packed) — the little-endian word — and for
/// [`Components`] over a base with a byte form — the concatenated base
/// encodings. [`PackedPadded<W>`](PackedPadded) implements none: its
/// coordinate word is narrower than its storage word.
pub(crate) trait StorageBuffer<B: Field, const D: usize>: StorageRepr<B, D> {
    /// Width of one element's stable byte representation.
    const BYTES: usize;
    /// Storage bit width of one element: `8 * BYTES`.
    const STORAGE_BITS: u32;
    /// The raw word named by stable little-endian bytes.
    ///
    /// # Panics
    ///
    /// Panics when `bytes` holds the wrong width.
    fn decode(bytes: &[u8]) -> Self::Raw;
    /// Writes the stable little-endian bytes of `raw`.
    ///
    /// # Panics
    ///
    /// Panics when `bytes` holds the wrong width.
    fn encode(bytes: &mut [u8], raw: &Self::Raw);
}

/// The display form of one storage.
pub(crate) trait StorageDisplay<B: Field, const D: usize>: StorageRepr<B, D> {
    /// Writes one element's display form; `digits` is the hexadecimal digit
    /// count of a packed coordinate word.
    fn fmt(formatter: &mut Formatter<'_>, raw: &Self::Raw, digits: usize) -> core::fmt::Result;
}

impl<B: Field, const D: usize> StorageRepr<B, D> for Components {
    const TAG: Storage = Storage::Components;
    const ZERO: Self::Raw = [<B as Field>::ZERO_RAW; D];
    const ONE: Self::Raw = {
        let mut components = [<B as Field>::ZERO_RAW; D];
        components[0] = <B as Field>::ONE_RAW;
        components
    };

    fn canonical(raw: Self::Raw) -> Self::Raw {
        raw.map(<B as Field>::canonical_raw)
    }

    fn split(raw: Self::Raw) -> [<B as Field>::Raw; D] {
        raw
    }

    fn join(components: [<B as Field>::Raw; D]) -> Self::Raw {
        components.map(<B as Field>::canonical_raw)
    }
}

impl<B: FieldBuffer, const D: usize> StorageBuffer<B, D> for Components {
    const BYTES: usize = D * <B as FieldBuffer>::BYTES;
    // The base byte width times the component count stays far below the
    // `u32` range.
    #[allow(clippy::cast_possible_truncation)]
    const STORAGE_BITS: u32 = (8 * D * <B as FieldBuffer>::BYTES) as u32;

    fn decode(bytes: &[u8]) -> Self::Raw {
        let width = <B as FieldBuffer>::BYTES;
        core::array::from_fn(|i| {
            let component = &bytes[i * width..(i + 1) * width];
            <B as FieldBuffer>::decode(component).to_raw()
        })
    }

    fn encode(bytes: &mut [u8], raw: &Self::Raw) {
        let width = <B as FieldBuffer>::BYTES;
        for (i, component) in raw.iter().enumerate() {
            let element = Elem::<B> { raw: *component };
            <B as FieldBuffer>::encode(&mut bytes[i * width..(i + 1) * width], element);
        }
    }
}

impl<B: Field, const D: usize> StorageDisplay<B, D> for Components
where
    Elem<B>: core::fmt::Display,
{
    fn fmt(formatter: &mut Formatter<'_>, raw: &Self::Raw, _digits: usize) -> core::fmt::Result {
        formatter.write_str("[")?;
        for (i, component) in raw.iter().enumerate() {
            if i > 0 {
                formatter.write_str(", ")?;
            }
            let element = Elem::<B> { raw: *component };
            core::fmt::Display::fmt(&element, formatter)?;
        }
        formatter.write_str("]")
    }
}

/// Emit one packed storage over one word type.
///
/// `Packed` fills its word exactly; `PackedPadded` keeps the coordinate word
/// below the word width, with every bit at or above the field degree
/// cleared. `join` rebuilds the word from canonical components only, so the
/// padding bits of a padded word are zero after every join, and `canonical`
/// runs the same round trip to clear the padding of an arbitrary word.
macro_rules! packed_storage {
    ($storage:ident, $word:ty, $bits:literal) => {
        impl Sealed for $storage<$word> {}

        impl<B: BinaryField, const D: usize> ExtensionStorage<B, D> for $storage<$word> {
            type Raw = $word;
        }

        impl<B: BinaryField, const D: usize> StorageRepr<B, D> for $storage<$word> {
            const TAG: Storage = Storage::$storage { width: $bits };
            const ZERO: Self::Raw = 0;
            // The base identity word holds at most the base's degree bits,
            // below the coordinate word of the row.
            #[allow(clippy::cast_possible_truncation)]
            const ONE: Self::Raw = {
                let base = <B as RowAccess>::DESCRIPTOR;
                arithmetic::word_one(base, base.root_index()) as $word
            };

            fn canonical(raw: Self::Raw) -> Self::Raw {
                <Self as StorageRepr<B, D>>::join(<Self as StorageRepr<B, D>>::split(raw))
            }

            #[allow(clippy::cast_lossless)]
            fn split(raw: Self::Raw) -> [<B as Field>::Raw; D] {
                let word = raw as u64;
                let degree = <B as Field>::DEGREE as usize;
                core::array::from_fn(|i| {
                    let component = extract_bits(word, i * degree, degree);
                    // The extracted word holds at most the base's degree
                    // bits, so the conversion cannot see excess bits.
                    <B as BinaryField>::from_coordinates(component)
                        .expect("packed component word holds excess bits")
                        .to_raw()
                })
            }

            #[allow(clippy::cast_possible_truncation)]
            fn join(components: [<B as Field>::Raw; D]) -> Self::Raw {
                let degree = <B as Field>::DEGREE as usize;
                let mut word = 0u64;
                for (i, component) in components.iter().enumerate() {
                    let element = Elem::<B> {
                        raw: <B as Field>::canonical_raw(*component),
                    };
                    // A packed row places component `i` at the bits
                    // `[i*degree, (i+1)*degree)`; the shift stays at or
                    // below the word width.
                    word |= <B as BinaryField>::to_coordinates(element) << (i * degree);
                }
                word as $word
            }
        }

        impl<B: BinaryField, const D: usize> StorageDisplay<B, D> for $storage<$word> {
            fn fmt(
                formatter: &mut Formatter<'_>,
                raw: &Self::Raw,
                digits: usize,
            ) -> core::fmt::Result {
                write!(formatter, "{:0width$x}", *raw, width = digits)
            }
        }
    };
}

packed_storage!(Packed, u8, 8);
packed_storage!(Packed, u16, 16);
packed_storage!(Packed, u32, 32);
packed_storage!(Packed, u64, 64);
packed_storage!(PackedPadded, u8, 8);
packed_storage!(PackedPadded, u16, 16);
packed_storage!(PackedPadded, u32, 32);
packed_storage!(PackedPadded, u64, 64);

/// Emit the little-endian byte form of one packed word.
macro_rules! packed_buffer {
    ($word:ty, $bytes:literal, $bits:literal) => {
        impl<B: BinaryField, const D: usize> StorageBuffer<B, D> for Packed<$word> {
            const BYTES: usize = $bytes;
            const STORAGE_BITS: u32 = $bits;

            fn decode(bytes: &[u8]) -> Self::Raw {
                let bytes: [u8; $bytes] = bytes
                    .try_into()
                    .expect("packed element has the wrong byte width");
                <$word>::from_le_bytes(bytes)
            }

            fn encode(bytes: &mut [u8], raw: &Self::Raw) {
                bytes.copy_from_slice(&raw.to_le_bytes());
            }
        }
    };
}

packed_buffer!(u8, 1, 8);
packed_buffer!(u16, 2, 16);
packed_buffer!(u32, 4, 32);
packed_buffer!(u64, 8, 64);

/// The `bits` bits of `word` at offset `offset`.
///
/// `offset + bits` stays at or below 64, so the shifts never wrap.
pub(crate) const fn extract_bits(word: u64, offset: usize, bits: usize) -> u64 {
    if bits >= 64 {
        word
    } else {
        (word >> offset) & ((1 << bits) - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::finite_field::FiniteField;
    use crate::field::{AES, Polynomial};

    type Aes8 = FiniteField<2, 8, Polynomial<AES>>;
    type Gf4 = FiniteField<2, 2, Polynomial<7>>;
    type FanPaar4Row = FiniteField<2, 4, crate::field::Tower<crate::field::FanPaar4>>;
    type FanPaar8Row = FiniteField<2, 8, crate::field::Tower<crate::field::FanPaar8>>;
    type Rijndael16Row = FiniteField<2, 16, crate::field::Tower<crate::field::Rijndael16>>;
    type Rijndael32Row = FiniteField<2, 32, crate::field::Tower<crate::field::Rijndael32>>;

    /// One packed storage joins its components, splits back, canonicalizes
    /// padding, and carries the base identity in component zero.
    macro_rules! packed_case {
        ($storage:ident, $word:ty, $base:ty, $low:expr, $high:expr, $word_value:expr) => {{
            let components = [
                Elem::<$base>::from_raw($low).raw,
                Elem::<$base>::from_raw($high).raw,
            ];
            let joined = <$storage<$word> as StorageRepr<$base, 2>>::join(components);
            assert_eq!(joined, $word_value);
            assert_eq!(
                <$storage<$word> as StorageRepr<$base, 2>>::split(joined),
                components
            );
            assert_eq!(<$storage<$word> as StorageRepr<$base, 2>>::ONE, 1);
            assert_eq!(<$storage<$word> as StorageRepr<$base, 2>>::ZERO, 0);
            // The canonical form of the exact word is itself; a padded
            // word clears the bits above the field degree.
            assert_eq!(
                <$storage<$word> as StorageRepr<$base, 2>>::canonical(joined),
                joined
            );
        }};
    }

    #[test]
    fn packed_storages_round_trip() {
        // Exact-width words: `Packed<u8>` over GF(2^4), `Packed<u16>` over
        // GF(2^8), `Packed<u32>` over GF(2^16), `Packed<u64>` over GF(2^32).
        packed_case!(Packed, u8, FanPaar4Row, 0b1001, 0b0101, 0b0101_1001);
        packed_case!(Packed, u16, Aes8, 0x53, 0xC1, 0xC153);
        packed_case!(Packed, u32, Rijndael16Row, 0x00FF, 0x1234, 0x1234_00FF);
        packed_case!(
            Packed,
            u64,
            Rijndael32Row,
            0x00FF_FFFF,
            0x1234_5678,
            0x1234_5678_00FF_FFFF
        );
        // Padded words below their width.
        packed_case!(PackedPadded, u8, Gf4, 0b10, 0b01, 0b0110);
        packed_case!(PackedPadded, u16, FanPaar4Row, 0b1001, 0b0101, 0b0101_1001);
        packed_case!(PackedPadded, u32, FanPaar8Row, 0x53, 0xC1, 0xC153);
        packed_case!(PackedPadded, u64, FanPaar8Row, 0x53, 0xC1, 0xC153);
    }

    #[test]
    fn padded_canonical_clears_high_bits() {
        // The degree-four row over GF(4) holds four bits; the byte above
        // them is padding.
        let components = [
            Elem::<Gf4>::from_raw(0b10).raw,
            Elem::<Gf4>::from_raw(0b01).raw,
        ];
        let joined = <PackedPadded<u8> as StorageRepr<Gf4, 2>>::join(components);
        assert_eq!(joined, 0b0110);
        assert_eq!(
            <PackedPadded<u8> as StorageRepr<Gf4, 2>>::canonical(0xF6),
            joined
        );
    }

    /// The byte form of a packed word is its little-endian encoding, at
    /// every word width.
    #[test]
    fn packed_bytes_round_trip_every_word() {
        macro_rules! buffer_case {
            ($word:ty, $value:expr) => {{
                let mut bytes = [0u8; core::mem::size_of::<$word>()];
                <Packed<$word> as StorageBuffer<Aes8, 2>>::encode(&mut bytes, &$value);
                assert_eq!(bytes, $value.to_le_bytes());
                assert_eq!(
                    <Packed<$word> as StorageBuffer<Aes8, 2>>::decode(&bytes),
                    $value
                );
            }};
        }
        // Every word width carries the GF(2^16) pair `(0x53, 0xC1)` over
        // the AES base in its low bits.
        buffer_case!(u8, 0x53u8);
        buffer_case!(u16, 0xC153u16);
        buffer_case!(u32, 0x0000_C153u32);
        buffer_case!(u64, 0x0000_0000_0000_C153u64);
    }

    #[test]
    fn components_storage_round_trip() {
        let components = [
            Elem::<Aes8>::from_raw(0x53).raw,
            Elem::<Aes8>::from_raw(0xC1).raw,
        ];
        assert_eq!(
            <Components as StorageRepr<Aes8, 2>>::split(
                <Components as StorageRepr<Aes8, 2>>::join(components)
            ),
            components
        );
        assert_eq!(<Components as StorageRepr<Aes8, 2>>::ZERO, [0, 0]);
        assert_eq!(<Components as StorageRepr<Aes8, 2>>::ONE, [1, 0]);
        // The identity in component zero comes from the base identity,
        // which a tower row may spell differently from one.
        assert_eq!(
            <Components as StorageRepr<FanPaar4Row, 2>>::ONE,
            [Elem::<FanPaar4Row>::ONE.raw, 0]
        );
    }

    #[test]
    fn components_bytes_concatenate_base_encodings() {
        assert_eq!(<Components as StorageBuffer<Aes8, 2>>::BYTES, 2);
        assert_eq!(<Components as StorageBuffer<Aes8, 2>>::STORAGE_BITS, 16);
        let components = [
            Elem::<Aes8>::from_raw(0x53).raw,
            Elem::<Aes8>::from_raw(0xC1).raw,
        ];
        let mut bytes = [0u8; 2];
        <Components as StorageBuffer<Aes8, 2>>::encode(&mut bytes, &components);
        assert_eq!(bytes, [0x53, 0xC1]);
        assert_eq!(
            <Components as StorageBuffer<Aes8, 2>>::decode(&bytes),
            components
        );
    }
}
