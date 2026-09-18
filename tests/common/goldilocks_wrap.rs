use fgf::goldilocks;

/// `a * b (mod p)` by direct 128-bit integer reduction.
pub(super) fn mulmod(a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(goldilocks::MODULUS)) as u64
}

/// `d + a * b (mod p)` by direct 128-bit integer reduction.
pub(super) fn addmulmod(d: u64, a: u64, b: u64) -> u64 {
    ((u128::from(d) + u128::from(a) * u128::from(b)) % u128::from(goldilocks::MODULUS)) as u64
}

/// Canonical operands whose pairwise products cross the wrap branch:
/// every power of two (pairs past `2^96` wrap), the `2^32` epsilon
/// neighborhood, the `2^48` neighborhood, the modulus extremes, and
/// 128-divisible lanes at and above `2^39`.
pub(super) fn boundary_family() -> Vec<u64> {
    let mut values = vec![0, 1, 2];
    values.extend((0..=63).map(|k| 1_u64 << k));
    values.extend([0xFFFF_FFFE, 0xFFFF_FFFF, 0x1_0000_0000]);
    values.extend([(1 << 48) - 1, 1 << 48, (1 << 48) + 1]);
    let p = goldilocks::MODULUS;
    values.extend([p - 1, p - 2, (p - 1) / 2]);
    values.extend([128 << 32, 128 * ((1 << 45) + 7)]);
    values
}

/// Pack canonical lanes into the stable little-endian representation.
pub(super) fn lanes_bytes(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Assert every lane of `got` against `want`, naming the operation,
/// coefficient, lane index, and source lane on failure.
pub(super) fn check_lanes(name: &str, coeff: u64, got: &[u8], want: &[u64]) {
    for (index, (chunk, &expected)) in got.as_chunks::<8>().0.iter().zip(want.iter()).enumerate() {
        let lane = u64::from_le_bytes(*chunk);
        assert_eq!(
            lane, expected,
            "{name} coeff {coeff:#x} lane {index}: got {lane:#x}, want {expected:#x}"
        );
    }
}
