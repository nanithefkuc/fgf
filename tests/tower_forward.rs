//! Forward layout fixtures for every shipped tower degree.
//!
//! Records the raw words of `ONE`, the generator, base inclusions, the
//! indeterminate, and fixed products from the pre-cutover types. These tests
//! stay unchanged through the tower cutover: they use only the surviving
//! alias spellings and the unified low-component-first packing.

use fgf::{AES, Binary, Elem, Polynomial};

type Gf16 = fgf::Binary<16, fgf::Tower<fgf::Rijndael16>>;
type Gf32 = fgf::Binary<32, fgf::Tower<fgf::Rijndael32>>;
type Gf64 = fgf::Binary<64, fgf::Tower<fgf::Rijndael64>>;
type FanPaar8Field = fgf::Binary<8, fgf::Tower<fgf::FanPaar8>>;
type FanPaar16Field = fgf::Binary<16, fgf::Tower<fgf::FanPaar16>>;
type FanPaar32Field = fgf::Binary<32, fgf::Tower<fgf::FanPaar32>>;
type FanPaar64Field = fgf::Binary<64, fgf::Tower<fgf::FanPaar64>>;

type Aes8 = Binary<8, Polynomial<AES>>;

#[test]
fn gf16_forward_layout() {
    assert_eq!(Elem::<Gf16>::ONE.to_raw(), 0x0001);
    assert_eq!(Elem::<Gf16>::GENERATOR.to_raw(), 0x0108);
    // Base inclusion preserves the low-component pattern.
    for (b, expect) in [
        (0x00u8, 0x0000u16),
        (0x01, 0x0001),
        (0x53, 0x0053),
        (0xFF, 0x00FF),
    ] {
        let lifted = Elem::<Gf16>::from_components(Elem::<Aes8>::from_raw(b), Elem::<Aes8>::ZERO);
        assert_eq!(lifted.to_raw(), expect);
    }
    // Indeterminate is the high component alone.
    let ind = Elem::<Gf16>::from_components(Elem::<Aes8>::ZERO, Elem::<Aes8>::ONE);
    assert_eq!(ind.to_raw(), 0x0100);
    // Fixed products: zero, one, high-only, low-only, full-width.
    for (x, y, expect) in [
        (0x0000u16, 0x0000u16, 0x0000u16),
        (0x0001, 0x0001, 0x0001),
        (0x3400, 0x1200, 0x05a0),
        (0x0012, 0x0034, 0x0005),
        (0x1234, 0x5678, 0xf029),
        (0xFFFF, 0x0108, 0xb96b),
    ] {
        assert_eq!(
            Elem::<Gf16>::from_raw(x)
                .mul(Elem::<Gf16>::from_raw(y))
                .to_raw(),
            expect,
            "{x:04x} * {y:04x}"
        );
    }
}

#[test]
fn gf32_forward_layout() {
    assert_eq!(Elem::<Gf32>::ONE.to_raw(), 0x00000001);
    assert_eq!(Elem::<Gf32>::GENERATOR.to_raw(), 0x00010002);
    for (b, expect) in [
        (0x0000u16, 0x00000000u32),
        (0x0001, 0x00000001),
        (0x1234, 0x00001234),
        (0xFFFF, 0x0000FFFF),
    ] {
        let lifted = Elem::<Gf32>::from_components(Elem::<Gf16>::from_raw(b), Elem::<Gf16>::ZERO);
        assert_eq!(lifted.to_raw(), expect);
    }
    let ind = Elem::<Gf32>::from_components(Elem::<Gf16>::ZERO, Elem::<Gf16>::ONE);
    assert_eq!(ind.to_raw(), 0x00010000);
    for (x, y, expect) in [
        (0x00000000u32, 0x00000000u32, 0x00000000u32),
        (0x00000001, 0x00000001, 0x00000001),
        (0x34000000, 0x12000000, 0x05a067c7),
        (0x00000012, 0x00000034, 0x00000005),
        (0x12345678, 0x9abcdef0, 0xa0943a44),
        (0xFFFFFFFF, 0x00010002, 0xe5e5e5e0),
    ] {
        assert_eq!(
            Elem::<Gf32>::from_raw(x)
                .mul(Elem::<Gf32>::from_raw(y))
                .to_raw(),
            expect,
            "{x:08x} * {y:08x}"
        );
    }
}

#[test]
fn gf64_forward_layout() {
    assert_eq!(Elem::<Gf64>::ONE.to_raw(), 0x0000000000000001);
    assert_eq!(Elem::<Gf64>::GENERATOR.to_raw(), 0x0000000100000004);
    for (b, expect) in [
        (0x00000000u32, 0x0000000000000000u64),
        (0x00000001, 0x0000000000000001),
        (0xdeadbeef, 0x00000000deadbeef),
        (0xFFFFFFFF, 0x00000000ffffffff),
    ] {
        let lifted = Elem::<Gf64>::from_components(Elem::<Gf32>::from_raw(b), Elem::<Gf32>::ZERO);
        assert_eq!(lifted.to_raw(), expect);
    }
    let ind = Elem::<Gf64>::from_components(Elem::<Gf32>::ZERO, Elem::<Gf32>::ONE);
    assert_eq!(ind.to_raw(), 0x0000000100000000);
    for (x, y, expect) in [
        (
            0x0000000000000000u64,
            0x0000000000000000u64,
            0x0000000000000000u64,
        ),
        (0x0000000000000001, 0x0000000000000001, 0x0000000000000001),
        (0x3400000000000000, 0x1200000000000000, 0x05a067c7a0a9c76e),
        (0x0000000000000012, 0x0000000000000034, 0x0000000000000005),
        (0x123456789abcdef0, 0x0fedcba987654321, 0x5cf6fa64e25bfd80),
        (0xFFFFFFFFFFFFFFFF, 0x0000000100000004, 0xd1d1d1d1d1d171d1),
    ] {
        assert_eq!(
            Elem::<Gf64>::from_raw(x)
                .mul(Elem::<Gf64>::from_raw(y))
                .to_raw(),
            expect,
            "{x:016x} * {y:016x}"
        );
    }
}

#[test]
fn fan_paar8_forward_layout() {
    assert_eq!(Elem::<FanPaar8Field>::ONE.to_raw(), 0x01);
    assert_eq!(Elem::<FanPaar8Field>::GENERATOR.to_raw(), 0x2d);
    // Indeterminate of the byte tower.
    assert_eq!(Elem::<FanPaar8Field>::from_raw(0x10).to_raw(), 0x10);
    for (x, y, expect) in [
        (0x00u8, 0x00u8, 0x00u8),
        (0x01, 0x01, 0x01),
        (0x1b, 0xa8, 0x09),
        (0x09, 0x2d, 0xec),
        (0xFF, 0x2d, 0x1c),
    ] {
        assert_eq!(
            Elem::<FanPaar8Field>::from_raw(x)
                .mul(Elem::<FanPaar8Field>::from_raw(y))
                .to_raw(),
            expect,
            "{x:02x} * {y:02x}"
        );
    }
}

#[test]
fn fan_paar16_forward_layout() {
    assert_eq!(Elem::<FanPaar16Field>::ONE.to_raw(), 0x0001);
    assert_eq!(Elem::<FanPaar16Field>::GENERATOR.to_raw(), 0xe2de);
    for (b, expect) in [
        (0x00u8, 0x0000u16),
        (0x01, 0x0001),
        (0x1b, 0x001b),
        (0xFF, 0x00FF),
    ] {
        let lifted = Elem::<FanPaar16Field>::from_components(
            Elem::<FanPaar8Field>::from_raw(b),
            Elem::<FanPaar8Field>::ZERO,
        );
        assert_eq!(lifted.to_raw(), expect);
        assert_eq!(
            Elem::<FanPaar16Field>::from_raw(u16::from(b)).to_raw(),
            expect
        );
    }
    let ind = Elem::<FanPaar16Field>::from_components(
        Elem::<FanPaar8Field>::ZERO,
        Elem::<FanPaar8Field>::ONE,
    );
    assert_eq!(ind.to_raw(), 0x0100);
    for (x, y, expect) in [
        (0x0000u16, 0x0000u16, 0x0000u16),
        (0x0001, 0x0001, 0x0001),
        (0x3400, 0x1200, 0x199b),
        (0x0012, 0x0034, 0x009b),
        (0x48a8, 0xf8a4, 0x3656),
        (0xFFFF, 0xe2de, 0xa6ac),
    ] {
        assert_eq!(
            Elem::<FanPaar16Field>::from_raw(x)
                .mul(Elem::<FanPaar16Field>::from_raw(y))
                .to_raw(),
            expect,
            "{x:04x} * {y:04x}"
        );
    }
}

#[test]
fn fan_paar32_forward_layout() {
    assert_eq!(Elem::<FanPaar32Field>::ONE.to_raw(), 0x00000001);
    assert_eq!(Elem::<FanPaar32Field>::GENERATOR.to_raw(), 0x03e21cea);
    for (b, expect) in [
        (0x0000u16, 0x00000000u32),
        (0x0001, 0x00000001),
        (0x48a8, 0x000048a8),
        (0xFFFF, 0x0000FFFF),
    ] {
        let lifted = Elem::<FanPaar32Field>::from_components(
            Elem::<FanPaar16Field>::from_raw(b),
            Elem::<FanPaar16Field>::ZERO,
        );
        assert_eq!(lifted.to_raw(), expect);
        assert_eq!(
            Elem::<FanPaar32Field>::from_raw(u32::from(b)).to_raw(),
            expect
        );
    }
    let ind = Elem::<FanPaar32Field>::from_components(
        Elem::<FanPaar16Field>::ZERO,
        Elem::<FanPaar16Field>::ONE,
    );
    assert_eq!(ind.to_raw(), 0x00010000);
    for (x, y, expect) in [
        (0x00000000u32, 0x00000000u32, 0x00000000u32),
        (0x00000001, 0x00000001, 0x00000001),
        (0x34000000, 0x12000000, 0x4a19199b),
        (0x00000012, 0x00000034, 0x0000009b),
        (0x12345678, 0x9abcdef0, 0x9f77a270),
        (0xFFFFFFFF, 0x03e21cea, 0x9698865e),
    ] {
        assert_eq!(
            Elem::<FanPaar32Field>::from_raw(x)
                .mul(Elem::<FanPaar32Field>::from_raw(y))
                .to_raw(),
            expect,
            "{x:08x} * {y:08x}"
        );
    }
}

#[test]
fn fan_paar64_forward_layout() {
    assert_eq!(Elem::<FanPaar64Field>::ONE.to_raw(), 0x0000000000000001);
    assert_eq!(
        Elem::<FanPaar64Field>::GENERATOR.to_raw(),
        0x070f870dcd9c1d88
    );
    for (b, expect) in [
        (0x00000000u32, 0x0000000000000000u64),
        (0x00000001, 0x0000000000000001),
        (0x12345678, 0x0000000012345678),
        (0xFFFFFFFF, 0x00000000ffffffff),
    ] {
        let lifted = Elem::<FanPaar64Field>::from_components(
            Elem::<FanPaar32Field>::from_raw(b),
            Elem::<FanPaar32Field>::ZERO,
        );
        assert_eq!(lifted.to_raw(), expect);
        assert_eq!(
            Elem::<FanPaar64Field>::from_raw(u64::from(b)).to_raw(),
            expect
        );
    }
    let ind = Elem::<FanPaar64Field>::from_components(
        Elem::<FanPaar32Field>::ZERO,
        Elem::<FanPaar32Field>::ONE,
    );
    assert_eq!(ind.to_raw(), 0x0000000100000000);
    for (x, y, expect) in [
        (
            0x0000000000000000u64,
            0x0000000000000000u64,
            0x0000000000000000u64,
        ),
        (0x0000000000000001, 0x0000000000000001, 0x0000000000000001),
        (0xc84d619110831cef, 0x000000000000a14f, 0x3565086d6b9ef595),
        (0xFFFFFFFFFFFFFFFF, 0x070f870dcd9c1d88, 0x80076b4104dc0875),
    ] {
        assert_eq!(
            Elem::<FanPaar64Field>::from_raw(x)
                .mul(Elem::<FanPaar64Field>::from_raw(y))
                .to_raw(),
            expect,
            "{x:016x} * {y:016x}"
        );
    }
}
