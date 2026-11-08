//! Gather offset validation rejects every invalid offset before any
//! destination byte changes.
//!
//! The public operations validate the whole offset table before dispatch;
//! the direct AVX2 gather must do the same before choosing its lane or
//! per-source fallback shape. Every rejection case asserts the destination
//! bytes survived the panic untouched, so a fold that mutates before
//! validating fails here. No case pins panic prose.

use std::panic::{AssertUnwindSafe, catch_unwind};

use fgf::binary::AES;
use fgf::binary::Polynomial;
use fgf::prime::Goldilocks;
use fgf::{FiniteField, ops};

type Gf8 = FiniteField<2, 8, Polynomial<AES>>;

#[path = "../src/kernel/tests/fixtures.rs"]
mod fixtures;

use fixtures::noise_odd_seed as noise;

/// Asserts `body` panics on `dst` and that `dst` still holds `sentinel`
/// bytes afterwards, proving the rejection preceded every write.
fn rejects_without_mutating(sentinel: &[u8], body: impl FnOnce(&mut [u8])) {
    let mut got = sentinel.to_vec();
    let outcome = catch_unwind(AssertUnwindSafe(|| body(&mut got)));
    assert!(outcome.is_err(), "invalid gather offsets must panic");
    assert_eq!(
        got, sentinel,
        "a rejected gather must not write any destination byte",
    );
}

#[test]
fn invalid_later_offset_never_mutates() {
    let region = noise(256, 0x51);
    let dst = noise(32, 0x52);
    rejects_without_mutating(&dst, |got| {
        ops::add_gather_offsets::<Gf8>(got, &[0, 64, 4_096], &region);
    });
}

/// A field whose addition is not XOR folds one `add_assign` per source, but
/// the whole offset table is still validated before the fold starts.
#[test]
fn non_xor_residue_field_rejects_before_mutation() {
    let region = noise(256, 0x53);
    let dst = noise(64, 0x54);
    rejects_without_mutating(&dst, |got| {
        ops::add_gather_offsets::<Goldilocks>(got, &[0, 8, 65_536], &region);
    });
}

/// Only on a 32-bit `usize` does `u32::MAX + dst.len()` overflow. With
/// overflow checks off (`--release`), an unchecked add wraps the range check
/// to a pass and the fold reaches the bad slice with earlier sources already
/// XORed in; the checked sum rejects the table before dispatch. Geometry:
/// a 0x3c-byte region and a 0x3c-byte destination filled with 0xa5, sources
/// at offsets 0 and `u32::MAX`.
#[test]
#[cfg(target_pointer_width = "32")]
fn overflowing_offset_rejects_before_mutation_on_32_bit() {
    if std::env::var_os("FGF_REQUIRE_WRAPPING").is_some() {
        let sum = catch_unwind(|| std::hint::black_box(usize::MAX) + std::hint::black_box(1usize));
        assert_eq!(
            sum.ok(),
            Some(0),
            "the 32-bit acceptance job must disable overflow checks",
        );
    }
    let region = noise(0x3c, 0x55);
    let dst = vec![0xa5u8; 0x3c];
    rejects_without_mutating(&dst, |got| {
        ops::add_gather_offsets::<Gf8>(got, &[0, u32::MAX], &region);
    });
    let region = noise(64, 0x56);
    let dst = vec![0xa5u8; 64];
    rejects_without_mutating(&dst, |got| {
        ops::add_gather_offsets::<Goldilocks>(got, &[0, u32::MAX], &region);
    });
}
