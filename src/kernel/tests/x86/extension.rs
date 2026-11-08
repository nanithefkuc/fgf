//! Custom-extension dispatch differentials: presentations without a
//! pinned strategy route through the typed scalar fallback on every tier.

use super::*;
use crate::{field::Normal, kernel::tests::tier_gate};

/// Transport a polynomial-coordinate word of the test normal base
/// `Normal<0x11B, 0x20>` into its normal-basis coordinates: the `B`
/// constant of the custom normal-base tower spec.
#[allow(clippy::cast_possible_truncation)]
const fn normal_transport(word: u64) -> u64 {
    let to_base = crate::field::binary::normal::to_base::<0x11B, 0x20, 8>();
    let from = crate::field::binary::normal::inverse_columns(to_base, 8);
    let mut image = 0u64;
    let mut i = 0;
    while i < 8 {
        if (word >> i) & 1 == 1 {
            image ^= from[i] as u64;
        }
        i += 1;
    }
    image
}

/// Differential coverage for the Reed-Solomon-rooted custom extension: a
/// description without a pinned strategy routes every dispatched
/// single-row operation through the typed scalar fallback, at lengths
/// straddling the 32-byte lane boundaries the vector kernels would use.
#[test]
fn rs_extension_kernels_match_scalar() {
    let coeffs: Vec<RsExtElem> = [0u16, 1, 2, 0x00ff, 0x0100, 0xff00, 0xffff, 0xbeef]
        .iter()
        .copied()
        .map(RsExtElem::from_raw)
        .collect();

    check_tower_mul_add(
        "rs16 fallback mul_add",
        LENGTHS,
        &coeffs,
        scalar::mul_add::<RsExtField>,
        |dst, c, src| {
            let prepared = <RsExtField as KernelDispatch>::prepare(RawDispatch, c);
            <RsExtField as KernelDispatch>::mul_add(RawDispatch, dst, &prepared, src);
        },
    );
    check_tower_mul_assign(
        "rs16 fallback mul_assign",
        LENGTHS,
        &coeffs,
        scalar::mul_assign::<RsExtField>,
        |dst, c| {
            let prepared = <RsExtField as KernelDispatch>::prepare(RawDispatch, c);
            <RsExtField as KernelDispatch>::mul_assign(RawDispatch, dst, &prepared);
        },
    );
    check_tower_mul_into(
        "rs16 fallback mul_into",
        LENGTHS,
        &coeffs,
        scalar::mul_add::<RsExtField>,
        |dst, c, src| {
            let prepared = <RsExtField as KernelDispatch>::prepare(RawDispatch, c);
            <RsExtField as KernelDispatch>::mul_into(RawDispatch, dst, &prepared, src);
        },
    );

    // The vector elementwise bodies fold the relation's linear term into
    // the Karatsuba cross sum, which holds only for a unit linear
    // coefficient — so the RS extension's elementwise route is the scalar
    // path, exercised in the dispatch test below.
}

/// The dispatched operations for the Reed–Solomon-rooted extension: the
/// one-shot raw-element forms resolve against the RS bank on access.
#[test]
#[allow(clippy::too_many_lines)]
fn rs_extension_dispatch_matches_scalar() {
    let Some(_) = tier_gate::summon::<X64V2Token>() else {
        eprintln!("skipping: no SSSE3 on this host");
        return;
    };
    for &len in LENGTHS {
        let src = noise(len, 0x238);
        for raw in [0u16, 1, 0x0100, 0xbeef] {
            let coeff = RsExtElem::from_raw(raw);
            let prepared = <RsExtField as KernelDispatch>::prepare(RawDispatch, coeff);

            let mut got = noise(len, 0x249);
            let mut want = got.clone();
            <RsExtField as KernelDispatch>::mul_add(RawDispatch, &mut got, &prepared, &src);
            scalar::mul_add::<RsExtField>(&mut want, coeff, &src);
            assert_eq!(
                got, want,
                "rs16 dispatch mul_add: len {len} coeff {raw:04x}"
            );

            let mut got = noise(len, 0x25a);
            let mut want = got.clone();
            <RsExtField as KernelDispatch>::mul_assign(RawDispatch, &mut got, &prepared);
            scalar::mul_assign::<RsExtField>(&mut want, coeff);
            assert_eq!(
                got, want,
                "rs16 dispatch mul_assign: len {len} coeff {raw:04x}"
            );

            let mut got = vec![0xaa; len];
            let mut want = vec![0u8; len];
            <RsExtField as KernelDispatch>::mul_into(RawDispatch, &mut got, &prepared, &src);
            scalar::mul_add::<RsExtField>(&mut want, coeff, &src);
            assert_eq!(
                got, want,
                "rs16 dispatch mul_into: len {len} coeff {raw:04x}"
            );
        }

        // Elementwise dispatch keeps the scalar path for the non-unit
        // linear coefficient.
        let a = noise(len, 0x26b);
        let b = noise(len, 0x27c);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        <RsExtField as KernelDispatch>::mul_elementwise(RawDispatch, &mut got, &a, &b);
        scalar::mul_elementwise::<RsExtField>(&mut want, &a, &b);
        assert_eq!(got, want, "rs16 dispatch elementwise: len {len}");
    }

    // Multi-row shapes over raw elements through dispatch.
    let row_len = 34;
    let nrows = 5;
    let src = noise(row_len, 0x28d);
    let coeffs: Vec<RsExtElem> = (0..nrows as u16)
        .map(|j| RsExtElem::from_raw(j * 8191))
        .collect();
    let mut got = noise(row_len * nrows, 0x29e);
    let mut want = got.clone();
    <RsExtField as KernelDispatch>::mul_add_scatter(RawDispatch, &mut got, row_len, &coeffs, &src);
    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
        scalar::mul_add::<RsExtField>(row, coeff, &src);
    }
    assert_eq!(got, want, "rs16 dispatch scatter");

    let srcs: Vec<Vec<u8>> = (0u16..3)
        .map(|t| noise(row_len, 0x2af + u64::from(t)))
        .collect();
    let src_refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
    let gather_coeffs: Vec<RsExtElem> = (0u16..3)
        .map(|t| RsExtElem::from_raw(t.wrapping_mul(40_961).wrapping_add(7)))
        .collect();
    let mut got = noise(row_len, 0x2c0);
    let mut want = got.clone();
    <RsExtField as KernelDispatch>::mul_add_gather(
        RawDispatch,
        &mut got,
        &gather_coeffs,
        &src_refs,
    );
    for (&coeff, &s) in gather_coeffs.iter().zip(&src_refs) {
        scalar::mul_add::<RsExtField>(&mut want, coeff, s);
    }
    assert_eq!(got, want, "rs16 dispatch gather");

    let matrix_coeffs: Vec<Vec<RsExtElem>> = (0u16..3)
        .map(|t| {
            (0..nrows)
                .map(|j| {
                    RsExtElem::from_raw(
                        t.wrapping_mul(7919)
                            .wrapping_add((j as u16).wrapping_mul(613)),
                    )
                })
                .collect()
        })
        .collect();
    let terms: Vec<(&[RsExtElem], &[u8])> = (0..3)
        .map(|t| (matrix_coeffs[t].as_slice(), src_refs[t]))
        .collect();
    let mut got = noise(row_len * nrows, 0x2d1);
    let mut want = got.clone();
    <RsExtField as KernelDispatch>::mul_add_matrix(RawDispatch, &mut got, row_len, nrows, &terms);
    for &(coeffs, s) in &terms {
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
            scalar::mul_add::<RsExtField>(row, coeff, s);
        }
    }
    assert_eq!(got, want, "rs16 dispatch matrix");

    // The overwrite-scatter default routes per-row `mul_into` through
    // dispatch: prior contents are ignored, zero fills, one copies.
    let mut got = noise(row_len * nrows, 0x2e2);
    let mut want = vec![0u8; row_len * nrows];
    <RsExtField as KernelDispatch>::mul_into_scatter(RawDispatch, &mut got, row_len, &coeffs, &src);
    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
        let prepared = <RsExtField as KernelDispatch>::prepare(RawDispatch, coeff);
        <RsExtField as KernelDispatch>::mul_into(RawDispatch, row, &prepared, &src);
    }
    assert_eq!(got, want, "rs16 dispatch overwrite scatter");
}

/// The AES-base tower with a non-unit linear coefficient: the only
/// spec that drives the GFNI `Compact` route with a general relation.
/// Every dispatched entry runs, against the portable scalar reference.
#[test]
fn aes_tower_dispatch_matches_scalar() {
    use crate::kernel::tower::gf16::Prepared;

    // A custom extension takes the scalar fallback on every tier: the
    // backend reports `Scalar` and the prepared form is the raw word.
    assert_eq!(
        <AesExtField as KernelDispatch>::backend(crate::field::dispatch::RawDispatch),
        crate::kernel::Backend::Scalar
    );
    assert!(matches!(
        <AesExtField as KernelDispatch>::prepare(RawDispatch, AesExtElem::from_raw(0x0104)),
        Prepared::Plain(_)
    ));
    assert!(!<AesExtField as KernelDispatch>::has_vector_elementwise(
        crate::field::dispatch::RawDispatch,
    ));

    let Some(_) = tier_gate::summon::<X64V2Token>() else {
        eprintln!("skipping: no SSSE3 on this host");
        return;
    };
    for &len in LENGTHS {
        let src = noise(len, 0x238);
        for raw in [0u16, 1, 0x0104, 0xbeef] {
            let coeff = AesExtElem::from_raw(raw);
            let prepared = <AesExtField as KernelDispatch>::prepare(RawDispatch, coeff);

            let mut got = noise(len, 0x249);
            let mut want = got.clone();
            <AesExtField as KernelDispatch>::mul_add(RawDispatch, &mut got, &prepared, &src);
            scalar::mul_add::<AesExtField>(&mut want, coeff, &src);
            assert_eq!(
                got, want,
                "aes16 dispatch mul_add: len {len} coeff {raw:04x}"
            );

            let mut got = noise(len, 0x25a);
            let mut want = got.clone();
            <AesExtField as KernelDispatch>::mul_assign(RawDispatch, &mut got, &prepared);
            scalar::mul_assign::<AesExtField>(&mut want, coeff);
            assert_eq!(
                got, want,
                "aes16 dispatch mul_assign: len {len} coeff {raw:04x}"
            );

            let mut got = vec![0xaa; len];
            let mut want = vec![0u8; len];
            <AesExtField as KernelDispatch>::mul_into(RawDispatch, &mut got, &prepared, &src);
            scalar::mul_add::<AesExtField>(&mut want, coeff, &src);
            assert_eq!(
                got, want,
                "aes16 dispatch mul_into: len {len} coeff {raw:04x}"
            );
        }

        // A non-unit linear coefficient keeps elementwise scalar.
        let a = noise(len, 0x26b);
        let b = noise(len, 0x27c);
        let mut got = vec![0; len];
        let mut want = vec![0; len];
        <AesExtField as KernelDispatch>::mul_elementwise(RawDispatch, &mut got, &a, &b);
        scalar::mul_elementwise::<AesExtField>(&mut want, &a, &b);
        assert_eq!(got, want, "aes16 dispatch elementwise: len {len}");
        let mut got = a.clone();
        let mut want = a.clone();
        <AesExtField as KernelDispatch>::mul_elementwise_assign(RawDispatch, &mut got, &b);
        scalar::mul_elementwise_assign::<AesExtField>(&mut want, &b);
        assert_eq!(got, want, "aes16 dispatch elementwise assign: len {len}");
    }

    aes_tower_multi_row_matches_scalar();
}

/// A custom extension over a normal-basis degree-8 base: bulk
/// operations run the typed scalar fallback on every tier and
/// `backend_for` reports `Scalar`.
#[test]
fn custom_normal_base_tower_bulk_matches_scalar() {
    type NormalBase = FiniteField<2, 8, Normal<0x11B, 0x20>>;

    #[derive(Clone, Copy)]
    struct OverNormal;

    impl crate::field::ExtensionSpec<2> for OverNormal {
        type Base = NormalBase;
        type Storage = Packed<u16>;
        const NAME: &'static str = "custom normal-base tower";
        // The constant term is the AES `0x20` transported into normal
        // coordinates; the linear coefficient is the normal base's one
        // (all bits set).
        const RELATION: Relation =
            Relation::from_binary_words::<NormalBase>(&[normal_transport(0x20), 0xFF]);
    }

    type Field = FiniteField<2, 16, crate::field::extension::Extension<OverNormal, 2>>;
    assert_eq!(Elem::<NormalBase>::ONE.to_raw(), 0xFF);

    assert_eq!(
        <Field as KernelDispatch>::backend(crate::field::dispatch::RawDispatch),
        crate::kernel::Backend::Scalar
    );

    let row_len = 34;
    let src = noise(row_len, 0x3a1);
    let coeffs: Vec<Elem<Field>> = (0..5u16)
        .map(|j| Elem::<Field>::from_raw(j.wrapping_mul(8191)))
        .collect();
    let mut got = noise(row_len * 5, 0x3b2);
    let mut want = got.clone();
    <Field as KernelDispatch>::mul_add_scatter(RawDispatch, &mut got, row_len, &coeffs, &src);
    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
        scalar::mul_add::<Field>(row, coeff, &src);
    }
    assert_eq!(got, want, "custom normal-base tower scatter");

    let prepared = <Field as KernelDispatch>::prepare(RawDispatch, coeffs[0]);
    let mut got = noise(row_len, 0x3c3);
    let mut want = got.clone();
    <Field as KernelDispatch>::mul_add(RawDispatch, &mut got, &prepared, &src);
    scalar::mul_add::<Field>(&mut want, coeffs[0], &src);
    assert_eq!(got, want, "custom normal-base tower mul_add");
}

/// The multi-row dispatched shapes for the AES-base general-relation
/// tower, raw-element and prepared forms, against the portable
/// scalar reference.
#[allow(clippy::too_many_lines)]
#[test]
fn aes_tower_multi_row_matches_scalar() {
    let Some(_) = tier_gate::summon::<X64V2Token>() else {
        eprintln!("skipping: no SSSE3 on this host");
        return;
    };
    // Multi-row shapes over raw elements through dispatch.
    let row_len = 34;
    let nrows = 5;
    let src = noise(row_len, 0x28d);
    let coeffs: Vec<AesExtElem> = (0..nrows as u16)
        .map(|j| AesExtElem::from_raw(j * 8191))
        .collect();
    let mut got = noise(row_len * nrows, 0x29e);
    let mut want = got.clone();
    <AesExtField as KernelDispatch>::mul_add_scatter(RawDispatch, &mut got, row_len, &coeffs, &src);
    for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&coeffs) {
        scalar::mul_add::<AesExtField>(row, coeff, &src);
    }
    assert_eq!(got, want, "aes16 dispatch scatter");

    let srcs: Vec<Vec<u8>> = (0u16..3)
        .map(|t| noise(row_len, 0x2af + u64::from(t)))
        .collect();
    let src_refs: Vec<&[u8]> = srcs.iter().map(Vec::as_slice).collect();
    let gather_coeffs: Vec<AesExtElem> = (0u16..3)
        .map(|t| AesExtElem::from_raw(t.wrapping_mul(40_961).wrapping_add(7)))
        .collect();
    let mut got = noise(row_len, 0x2c0);
    let mut want = got.clone();
    <AesExtField as KernelDispatch>::mul_add_gather(
        RawDispatch,
        &mut got,
        &gather_coeffs,
        &src_refs,
    );
    for (&coeff, &s) in gather_coeffs.iter().zip(&src_refs) {
        scalar::mul_add::<AesExtField>(&mut want, coeff, s);
    }
    assert_eq!(got, want, "aes16 dispatch gather");

    let matrix_coeffs: Vec<Vec<AesExtElem>> = (0u16..3)
        .map(|t| {
            (0..nrows)
                .map(|j| {
                    AesExtElem::from_raw(
                        t.wrapping_mul(7919)
                            .wrapping_add((j as u16).wrapping_mul(613)),
                    )
                })
                .collect()
        })
        .collect();
    let terms: Vec<(&[AesExtElem], &[u8])> = (0..3)
        .map(|t| (matrix_coeffs[t].as_slice(), src_refs[t]))
        .collect();
    let mut got = noise(row_len * nrows, 0x2d1);
    let mut want = got.clone();
    <AesExtField as KernelDispatch>::mul_add_matrix(RawDispatch, &mut got, row_len, nrows, &terms);
    for &(coeffs, s) in &terms {
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(coeffs) {
            scalar::mul_add::<AesExtField>(row, coeff, s);
        }
    }
    assert_eq!(got, want, "aes16 dispatch matrix");
    // The `_with` and `_plan` forms cover the same routes through
    // prepared state; the multi-row defaults apply prepared AXPY per
    // row and term.
    #[cfg(feature = "alloc")]
    {
        let values: Vec<AesExtElem> = (0..nrows as u16)
            .map(|j| AesExtElem::from_raw(j.wrapping_mul(7411).wrapping_add(3)))
            .collect();
        let prepared: Vec<<AesExtField as KernelDispatch>::Prepared> = values
            .iter()
            .map(|&c| <AesExtField as KernelDispatch>::prepare(RawDispatch, c))
            .collect();
        let mut got = noise(row_len * nrows, 0x2e2);
        let mut want = got.clone();
        <AesExtField as KernelDispatch>::mul_add_scatter_with(
            RawDispatch,
            &mut got,
            row_len,
            &prepared,
            &src,
        );
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&values) {
            scalar::mul_add::<AesExtField>(row, coeff, &src);
        }
        assert_eq!(got, want, "aes16 dispatch scatter with");
        let mut got = noise(row_len * nrows, 0x2e3);
        let mut want = got.clone();
        <AesExtField as KernelDispatch>::mul_add_scatter_plan(
            RawDispatch,
            &mut got,
            row_len,
            &values,
            &prepared,
            &src,
        );
        for (row, &coeff) in want.chunks_exact_mut(row_len).zip(&values) {
            scalar::mul_add::<AesExtField>(row, coeff, &src);
        }
        assert_eq!(got, want, "aes16 dispatch scatter plan");

        let mut got = noise(row_len, 0x2e4);
        let mut want = got.clone();
        <AesExtField as KernelDispatch>::mul_add_gather_with(
            RawDispatch,
            &mut got,
            &prepared[..src_refs.len()],
            &src_refs,
        );
        for (&coeff, &s) in values.iter().zip(&src_refs) {
            scalar::mul_add::<AesExtField>(&mut want, coeff, s);
        }
        assert_eq!(got, want, "aes16 dispatch gather with");
        let mut got = noise(row_len, 0x2e5);
        let mut want = got.clone();
        <AesExtField as KernelDispatch>::mul_add_gather_plan(
            RawDispatch,
            &mut got,
            &values[..src_refs.len()],
            &prepared[..src_refs.len()],
            &src_refs,
        );
        for (&coeff, &s) in values.iter().zip(&src_refs) {
            scalar::mul_add::<AesExtField>(&mut want, coeff, s);
        }
        assert_eq!(got, want, "aes16 dispatch gather plan");

        let nterms = src_refs.len();
        let flat: Vec<AesExtElem> = (0..nterms * nrows)
            .map(|k| AesExtElem::from_raw((k as u16).wrapping_mul(911).wrapping_add(17)))
            .collect();
        let flat_prepared: Vec<<AesExtField as KernelDispatch>::Prepared> = flat
            .iter()
            .map(|&c| <AesExtField as KernelDispatch>::prepare(RawDispatch, c))
            .collect();
        let mut got = noise(row_len * nrows, 0x2e6);
        let mut want = got.clone();
        <AesExtField as KernelDispatch>::mul_add_matrix_plan(
            RawDispatch,
            &mut got,
            row_len,
            nrows,
            &flat,
            &flat_prepared,
            &src_refs,
        );
        for (term, &s) in src_refs.iter().enumerate() {
            for (row, &coeff) in want
                .chunks_exact_mut(row_len)
                .zip(&flat[term * nrows..(term + 1) * nrows])
            {
                scalar::mul_add::<AesExtField>(row, coeff, s);
            }
        }
        assert_eq!(got, want, "aes16 dispatch matrix plan");
    }
}
