//! SSE4.2 `QuadMersenne31` additive differentials: independent limb model,
//! portable reference, noncanonical alias canonicalization, unaligned
//! whole-element slices, and validation-before-write geometry.

use super::*;
use crate::field::mersenne31;
use crate::field::quad_mersenne31::QuadMersenne31;
use crate::kernel::prime;
use crate::kernel::tests::tier_gate;
use crate::kernel::x86;

/// Lengths straddling the 16-byte SSE4.2 lane and 64-byte tile boundaries:
/// empty, single-element tail, exact lanes, tile-minus-one, exact tile,
/// tile-plus-tail, and larger buffers.
const LENS: &[usize] = &[0, 8, 16, 24, 32, 48, 56, 64, 72, 96, 128, 264, 1024];

const P: u32 = mersenne31::MODULUS;

const VALUES: [Elem<QuadMersenne31>; 8] = [
    Elem::<QuadMersenne31>::ZERO,
    Elem::<QuadMersenne31>::ONE,
    Elem::<QuadMersenne31>::I,
    Elem::<QuadMersenne31>::from_raw([P - 1, P - 1]),
    Elem::<QuadMersenne31>::from_raw([1, P - 1]),
    Elem::<QuadMersenne31>::from_raw([0x5555_5555, 0x1234_5678]),
    Elem::<QuadMersenne31>::from_raw([P, 0]),
    Elem::<QuadMersenne31>::from_raw([u32::MAX, 2]),
];

fn o_canon(x: u32) -> u32 {
    u32::try_from(u128::from(x) % u128::from(P)).unwrap()
}

fn o_add(a: u32, b: u32) -> u32 {
    u32::try_from((u128::from(o_canon(a)) + u128::from(o_canon(b))) % u128::from(P)).unwrap()
}

fn o_sub(a: u32, b: u32) -> u32 {
    let d = i128::from(o_canon(a)) - i128::from(o_canon(b));
    u32::try_from(d.rem_euclid(i128::from(P))).unwrap()
}

fn fill_noncanonical(len: usize, shift: usize) -> Vec<u8> {
    const LIMBS: &[u32] = &[0, 1, P - 1, P, P + 1, 0x8000_0000, 0xFFFF_FFFE, u32::MAX];
    let mut out = Vec::with_capacity(len);
    let mut i = shift;
    while out.len() < len {
        let re = LIMBS[i % LIMBS.len()];
        let im = LIMBS[(i + 3) % LIMBS.len()];
        out.extend_from_slice(&re.to_le_bytes());
        out.extend_from_slice(&im.to_le_bytes());
        i += 1;
    }
    out.truncate(len);
    out
}

fn assert_canonical(label: &str, buf: &[u8]) {
    for (i, limb) in buf.chunks_exact(4).enumerate() {
        let lane = u32::from_le_bytes(limb.try_into().unwrap());
        assert!(lane < P, "{label}: limb {i} holds {lane:#x}, want < {P:#x}");
    }
}

fn model_pair_bytes(lhs: &[u8], rhs: &[u8], op: impl Fn(u32, u32) -> u32) -> Vec<u8> {
    assert_eq!(lhs.len(), rhs.len());
    let mut out = Vec::with_capacity(lhs.len());
    for (ca, cb) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
        let left = u32::from_le_bytes(ca.try_into().unwrap());
        let right = u32::from_le_bytes(cb.try_into().unwrap());
        out.extend_from_slice(&op(left, right).to_le_bytes());
    }
    out
}

fn model_scalar_bytes(
    buf: &[u8],
    value: Elem<QuadMersenne31>,
    op: impl Fn(u32, u32) -> u32,
) -> Vec<u8> {
    let [vr, vi] = value.to_raw();
    let mut out = Vec::with_capacity(buf.len());
    for chunk in buf.chunks_exact(8) {
        let ar = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let ai = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
        out.extend_from_slice(&op(ar, vr).to_le_bytes());
        out.extend_from_slice(&op(ai, vi).to_le_bytes());
    }
    out
}

#[test]
fn sse42_row_ops_match_model_and_reference() {
    tier_gate::run_entry(x86::quad_mersenne31::add_assign_sse42, |token, entry| {
        drive_prime_len::<QuadMersenne31>(
            LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::add_assign::<QuadMersenne31>(want, src);
            },
            "qm31 sse42",
            "add_assign",
        );
        for &len in LENS {
            let base = fill_noncanonical(len, 1);
            let src = fill_noncanonical(len, 0);
            let mut got = base.clone();
            entry(token, &mut got, &src);
            assert_eq!(
                got,
                model_pair_bytes(&base, &src, o_add),
                "sse42 add_assign noncanonical len {len}"
            );
            assert_canonical("sse42 add_assign", &got);
        }
    });
    tier_gate::run_entry(x86::quad_mersenne31::sub_assign_sse42, |token, entry| {
        drive_prime_len::<QuadMersenne31>(
            LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::sub_assign::<QuadMersenne31>(want, src);
            },
            "qm31 sse42",
            "sub_assign",
        );
        for &len in LENS {
            let base = fill_noncanonical(len, 1);
            let src = fill_noncanonical(len, 0);
            let mut got = base.clone();
            entry(token, &mut got, &src);
            assert_eq!(
                got,
                model_pair_bytes(&base, &src, o_sub),
                "sse42 sub_assign noncanonical len {len}"
            );
            assert_canonical("sse42 sub_assign", &got);
        }
    });
}

#[test]
fn sse42_scalar_ops_match_model_and_reference() {
    tier_gate::run_entry(
        x86::quad_mersenne31::add_assign_scalar_sse42,
        |token, entry| {
            drive_prime_scalar::<QuadMersenne31, Elem<QuadMersenne31>>(
                LENS,
                &VALUES,
                |e| e,
                |got, want, v, _, _| {
                    entry(token, got, v);
                    prime::add_assign_scalar::<QuadMersenne31>(want, v);
                },
                "qm31 sse42",
                "add_assign_scalar",
            );
            for &len in LENS {
                let base = fill_noncanonical(len, 2);
                for &v in &VALUES {
                    let mut got = base.clone();
                    entry(token, &mut got, v);
                    assert_eq!(
                        got,
                        model_scalar_bytes(&base, v, o_add),
                        "sse42 add_assign_scalar noncanonical len {len} val {v:?}"
                    );
                    assert_canonical("sse42 add_assign_scalar", &got);
                }
            }
        },
    );
    tier_gate::run_entry(
        x86::quad_mersenne31::sub_assign_scalar_sse42,
        |token, entry| {
            drive_prime_scalar::<QuadMersenne31, Elem<QuadMersenne31>>(
                LENS,
                &VALUES,
                |e| e,
                |got, want, v, _, _| {
                    entry(token, got, v);
                    prime::sub_assign_scalar::<QuadMersenne31>(want, v);
                },
                "qm31 sse42",
                "sub_assign_scalar",
            );
            for &len in LENS {
                let base = fill_noncanonical(len, 2);
                for &v in &VALUES {
                    let mut got = base.clone();
                    entry(token, &mut got, v);
                    assert_eq!(
                        got,
                        model_scalar_bytes(&base, v, o_sub),
                        "sse42 sub_assign_scalar noncanonical len {len} val {v:?}"
                    );
                    assert_canonical("sse42 sub_assign_scalar", &got);
                }
            }
        },
    );
}

#[test]
fn sse42_unaligned_slices() {
    let Some(token) = tier_gate::summon::<X64V2Token>() else {
        eprintln!("skipping: no SSE4.2 on this host");
        return;
    };
    let len = 88;
    let src_elems = fill_noncanonical(len, 0);
    let dst_elems = fill_noncanonical(len, 3);
    let val = Elem::<QuadMersenne31>::from_raw([0x1234_5678, P - 7]);
    for offset in [0usize, 1, 3, 5] {
        let mut src_storage = vec![0u8; len + 8];
        src_storage[offset..offset + len].copy_from_slice(&src_elems);
        let src = &src_storage[offset..offset + len];

        let mut dst_storage = vec![0u8; len + 8];
        dst_storage[offset..offset + len].copy_from_slice(&dst_elems);
        x86::quad_mersenne31::add_assign_sse42(token, &mut dst_storage[offset..offset + len], src);
        assert_eq!(
            &dst_storage[offset..offset + len],
            &model_pair_bytes(&dst_elems, &src_elems, o_add)[..],
            "unaligned sse42 add_assign offset {offset}"
        );

        let mut dst_storage = vec![0u8; len + 8];
        dst_storage[offset..offset + len].copy_from_slice(&dst_elems);
        x86::quad_mersenne31::sub_assign_sse42(token, &mut dst_storage[offset..offset + len], src);
        assert_eq!(
            &dst_storage[offset..offset + len],
            &model_pair_bytes(&dst_elems, &src_elems, o_sub)[..],
            "unaligned sse42 sub_assign offset {offset}"
        );

        let mut dst_storage = vec![0u8; len + 8];
        dst_storage[offset..offset + len].copy_from_slice(&dst_elems);
        x86::quad_mersenne31::add_assign_scalar_sse42(
            token,
            &mut dst_storage[offset..offset + len],
            val,
        );
        assert_eq!(
            &dst_storage[offset..offset + len],
            &model_scalar_bytes(&dst_elems, val, o_add)[..],
            "unaligned sse42 add_assign_scalar offset {offset}"
        );

        let mut dst_storage = vec![0u8; len + 8];
        dst_storage[offset..offset + len].copy_from_slice(&dst_elems);
        x86::quad_mersenne31::sub_assign_scalar_sse42(
            token,
            &mut dst_storage[offset..offset + len],
            val,
        );
        assert_eq!(
            &dst_storage[offset..offset + len],
            &model_scalar_bytes(&dst_elems, val, o_sub)[..],
            "unaligned sse42 sub_assign_scalar offset {offset}"
        );
    }
}

fn expect_geometry_panic(label: &str, dst: &mut [u8], invoke: impl FnOnce(&mut [u8])) {
    let before = dst.to_vec();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| invoke(dst))).is_err();
    assert!(panicked, "{label} must panic");
    assert_eq!(
        &*dst,
        &before[..],
        "{label} must not write before validating"
    );
}

#[test]
fn sse42_geometry_validation_precedes_writes() {
    let Some(token) = tier_gate::summon::<X64V2Token>() else {
        eprintln!("skipping: no SSE4.2 on this host");
        return;
    };
    let val = Elem::<QuadMersenne31>::from_raw([3, 5]);

    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("sse42 add partial", &mut partial, |d| {
        x86::quad_mersenne31::add_assign_sse42(token, d, &partial_src);
    });
    let mut partial = [1u8; 12];
    let partial_src = [1u8; 12];
    expect_geometry_panic("sse42 sub partial", &mut partial, |d| {
        x86::quad_mersenne31::sub_assign_sse42(token, d, &partial_src);
    });
    let mut partial = [1u8; 12];
    expect_geometry_panic("sse42 add_scalar partial", &mut partial, |d| {
        x86::quad_mersenne31::add_assign_scalar_sse42(token, d, val);
    });
    let mut partial = [1u8; 12];
    expect_geometry_panic("sse42 sub_scalar partial", &mut partial, |d| {
        x86::quad_mersenne31::sub_assign_scalar_sse42(token, d, val);
    });

    let mut mismatched = [1u8; 16];
    let short_src = [1u8; 8];
    expect_geometry_panic("sse42 add length mismatch", &mut mismatched, |d| {
        x86::quad_mersenne31::add_assign_sse42(token, d, &short_src);
    });
    let mut mismatched = [1u8; 16];
    let short_src = [1u8; 8];
    expect_geometry_panic("sse42 sub length mismatch", &mut mismatched, |d| {
        x86::quad_mersenne31::sub_assign_sse42(token, d, &short_src);
    });
}
