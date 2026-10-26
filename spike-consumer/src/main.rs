
#![allow(dead_code)]
use fgf::internals::kernel::tables::{spike_bank_digest, spike_runtime_columns, SpikeWide};
use fgf::{Cantor, Extension, ExtensionSpec, FanPaar8, FiniteField, Normal, Packed, Polynomial, Relation, Rijndael16, Rijndael32};

type Gf2 = FiniteField<2, 1, Polynomial<3>>;
type Gf16b = FiniteField<2, 4, Polynomial<0x13>>;
type Aes8 = FiniteField<2, 8, Polynomial<0x11B>>;

pub struct Custom8;
impl ExtensionSpec<2> for Custom8 {
    type Base = Gf16b; type Storage = Packed<u8>;
    const NAME: &'static str = "spike GF(2^4)[t]";
    const RELATION: Relation = Relation::from_binary_words::<Gf16b>(&[0x8, 0x1]);
}
pub struct Bin16;
impl ExtensionSpec<16> for Bin16 {
    type Base = Gf2; type Storage = Packed<u16>;
    const NAME: &'static str = "spike GF(2)[x]/deg16";
    const RELATION: Relation = Relation::from_binary_words::<Gf2>(&[1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}
pub struct Bin32;
impl ExtensionSpec<32> for Bin32 {
    type Base = Gf2; type Storage = Packed<u32>;
    const NAME: &'static str = "spike GF(2)[x]/deg32";
    const RELATION: Relation = Relation::from_binary_words::<Gf2>(&[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}
pub struct Bin64;
impl ExtensionSpec<64> for Bin64 {
    type Base = Gf2; type Storage = Packed<u64>;
    const NAME: &'static str = "spike GF(2)[x]/deg64";
    const RELATION: Relation = Relation::from_binary_words::<Gf2>(&[1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
}
pub struct Aes64;
impl ExtensionSpec<8> for Aes64 {
    type Base = Aes8; type Storage = Packed<u64>;
    const NAME: &'static str = "spike AES[t]/deg8";
    const RELATION: Relation = Relation::from_binary_words::<Aes8>(&[107, 48, 249, 14, 199, 221, 1, 228]);
}
pub struct Custom32;
impl ExtensionSpec<2> for Custom32 {
    type Base = FiniteField<2, 16, Extension<Rijndael16, 2>>; type Storage = Packed<u32>;
    const NAME: &'static str = "spike custom 32";
    const RELATION: Relation = Relation::from_binary_words::<FiniteField<2, 16, Extension<Rijndael16, 2>>>(&[0x2001, 0x0001]);
}
pub struct Custom64;
impl ExtensionSpec<2> for Custom64 {
    type Base = FiniteField<2, 32, Extension<Rijndael32, 2>>; type Storage = Packed<u64>;
    const NAME: &'static str = "spike custom 64";
    const RELATION: Relation = Relation::from_binary_words::<FiniteField<2, 32, Extension<Rijndael32, 2>>>(&[0x2ada_7f36, 0x0000_0001]);
}
type W16 = FiniteField<2, 16, Extension<Bin16, 16>>;
type W32 = FiniteField<2, 32, Extension<Bin32, 32>>;
type W64 = FiniteField<2, 64, Extension<Bin64, 64>>;
type A64 = FiniteField<2, 64, Extension<Aes64, 8>>;
type C32 = FiniteField<2, 32, Extension<Custom32, 2>>;
type C64 = FiniteField<2, 64, Extension<Custom64, 2>>;

fn main() {
    let mut acc = 0u64;
    #[cfg(feature = "banks")]
    {
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x11b, 0x20>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x11b, 0x21>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x11d, 0x20>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x11d, 0x21>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x12b, 0x8>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x12b, 0x9>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x12d, 0x8>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x12d, 0x9>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x139, 0x8>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x139, 0x9>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x13f, 0x8>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x13f, 0x9>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x14d, 0x20>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x14d, 0x21>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x15f, 0x20>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x15f, 0x21>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x163, 0x8>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x163, 0x9>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x165, 0x8>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x165, 0x9>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x169, 0x8>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x169, 0x9>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x171, 0x8>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Normal<0x171, 0x9>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Cantor<0x11B, 0x20>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Cantor<0x11D, 0x20>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Extension<FanPaar8, 2>>>();
        acc ^= spike_bank_digest::<FiniteField<2, 8, Extension<Custom8, 2>>>();
    }
    #[cfg(feature = "wide")]
    {
        const C: u64 = 0x1234_5678_9abc_def1;
        acc ^= SpikeWide::<W16, {C & 0xffff}>::COLUMNS[3];
        acc ^= SpikeWide::<W32, {C & 0xffff_ffff}>::COLUMNS[7];
        acc ^= SpikeWide::<W64, C>::COLUMNS[11];
        acc ^= SpikeWide::<A64, C>::COLUMNS[13];
        acc ^= SpikeWide::<C32, {C & 0xffff_ffff}>::COLUMNS[5];
        acc ^= SpikeWide::<C64, C>::COLUMNS[17];
        time::<W16>("u16 over GF(2), degree 16", 0xffff);
        time::<W32>("u32 over GF(2), degree 32", 0xffff_ffff);
        time::<W64>("u64 over GF(2), degree 64", u64::MAX);
        time::<A64>("u64 over AES, degree 8", u64::MAX);
        time::<C32>("u32 over Rijndael16, degree 2", 0xffff_ffff);
        time::<C64>("u64 over Rijndael32, degree 2", u64::MAX);
    }
    #[cfg(feature = "rows")]
    {
        rows::<W16>("u16 over GF(2), degree 16");
        rows::<W32>("u32 over GF(2), degree 32");
        rows::<W64>("u64 over GF(2), degree 64");
        rows::<A64>("u64 over AES, degree 8");
        rows::<C32>("u32 over Rijndael16, degree 2");
        rows::<C64>("u64 over Rijndael32, degree 2");
        rows::<FiniteField<2, 16, Extension<Rijndael16, 2>>>("shipped Rijndael16");
        rows::<FiniteField<2, 32, Extension<Rijndael32, 2>>>("shipped Rijndael32");
        rows::<FiniteField<2, 64, Extension<fgf::Rijndael64, 2>>>("shipped Rijndael64");
        rows::<FiniteField<2, 8, Normal<0x11B, 0x20>>>("Normal GF(2^8)");
        rows::<Aes8>("AES GF(2^8)");
    }
    println!("digest {acc:#x}");
}


#[cfg(feature = "rows")]
fn rows<F: fgf::FieldKernels>(label: &str) where fgf::Elem<F>: fgf::field::FieldElem {
    use fgf::field::FieldBuffer;
    let len = 4096usize;
    let src: Vec<u8> = (0..len).map(|i| (i * 131 + 7) as u8).collect();
    let mut dst = vec![0u8; len];
    let mut bytes = [0u8; 8];
    bytes[..F::BYTES].copy_from_slice(&src[8..8 + F::BYTES]);
    let coeff = F::decode(&bytes[..F::BYTES]);
    let n = 200u32;
    let start = std::time::Instant::now();
    for _ in 0..n { fgf::ops::mul_add::<F>(std::hint::black_box(&mut dst), std::hint::black_box(coeff), &src); }
    let row = start.elapsed().as_nanos() as f64 / f64::from(n);
    let start = std::time::Instant::now();
    let mut keep = 0usize;
    for _ in 0..n { let p = fgf::ops::Coeff::<F>::new(std::hint::black_box(coeff)); keep ^= std::hint::black_box(&p) as *const _ as usize; }
    let prep = start.elapsed().as_nanos() as f64 / f64::from(n);
    println!("row {label}: mul_add 4096 B {row:.0} ns, Coeff::new {prep:.0} ns, size_of Coeff {} B ({keep:x})", core::mem::size_of::<fgf::ops::Coeff<F>>());
}

fn time<F: fgf::field::Field>(label: &str, mask: u64) {
    let n = 2000u64;
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut sink = 0u64;
    let start = std::time::Instant::now();
    for _ in 0..n {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
        let cols = spike_runtime_columns::<F>(std::hint::black_box(x & mask));
        sink ^= cols[(x >> 58) as usize % 64];
    }
    let per = start.elapsed().as_nanos() as f64 / n as f64;
    println!("columns {label}: {per:.0} ns per coefficient (sink {sink:x})");
}
