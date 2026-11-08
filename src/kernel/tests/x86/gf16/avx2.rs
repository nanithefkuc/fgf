//! AVX2 GF(2^16) differentials.

use super::*;
use crate::kernel::tests::tier_gate;

/// `TableCoefficient::with_tables` builds tables for the `Plain` prepared
/// form, which dispatch never produces on a shuffle host but direct callers
/// may hold.
#[test]
fn gf16_scatter_accepts_plain_prepared_coefficients() {
    tier_gate::run_entry(x86::gf16::mul_add_scatter_avx2, |v3, entry| {
        for &len in LENGTHS {
            let src = noise(len, 0x61);
            let mut got = noise(2 * len, 0x62);
            let mut want = got.clone();
            let coeffs = [gf16_coeff_at(1), gf16_coeff_at(2)];
            entry(v3, &mut got, len, &coeffs, &src);
            scalar::mul_add::<Rijndael16Field>(&mut want[..len], gf16_coeff_at(1), &src);
            scalar::mul_add::<Rijndael16Field>(&mut want[len..], gf16_coeff_at(2), &src);
            assert_eq!(got, want, "plain-prepared scatter at len {len}");
        }
    });
    for &len in LENGTHS {
        // The plain prepared form reaches the dispatched single-row path.
        let src = noise(len, 0x61);
        let prepared = crate::kernel::gf16::Prepared::Plain(gf16_coeff_at(1).to_raw());
        let mut dispatched = noise(len, 0x63);
        let mut reference = dispatched.clone();
        <Rijndael16Field as KernelDispatch>::mul_add(RawDispatch, &mut dispatched, &prepared, &src);
        scalar::mul_add::<Rijndael16Field>(&mut reference, gf16_coeff_at(1), &src);
        assert_eq!(
            dispatched, reference,
            "plain-prepared dispatch at len {len}"
        );
    }
}
