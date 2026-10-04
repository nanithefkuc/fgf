//! Canonical inclusions between binary-field presentations.
//!
//! Run with `just example embeddings`, optionally with
//! `--no-default-features`. `Embedding<S, T>` prepares one canonical
//! inclusion through the frozen reference ladder; construction and
//! application allocate nothing, and repeated application reuses the
//! prepared columns without rebuilding them.

#![forbid(unsafe_code)]

use fgf::poly::{AES, REED_SOLOMON};
use fgf::{Elem, Embedding, EmbeddingError, FanPaar16, Gf, Gf1, Gf8, Gf16, Gf64, Poly};

type Aes = Gf8<Poly<AES>>;
type Nibble = Gf<4, Poly<0x13>>;

fn main() {
    // Below the AES field the ladder inclusions are smallest-raw-root maps,
    // not literal byte relabelling: the indeterminate of x^4 + x + 1 lands
    // on 0x5C inside the AES field.
    let small = Embedding::<Nibble, Aes>::new().unwrap();
    let x = Elem::<Nibble>::from_raw(2);
    let image = small.embed(x);
    assert_eq!(image, Elem::<Aes>::from_raw(0x5C));
    assert!(small.contains(image));
    assert_eq!(small.restrict(image), Some(x));
    assert_eq!(small.restrict(Elem::<Aes>::from_raw(2)), None);
    println!("GF(16) x embeds into AES as {image}; the literal byte 2 is outside the image.");

    // Equal degree is a change of presentation. The AES-to-Reed-Solomon map
    // is a ring isomorphism: it preserves one and products, and restriction
    // inverts it.
    let involution = Embedding::<Gf8<Poly<AES>>, Gf8<Poly<REED_SOLOMON>>>::new().unwrap();
    let (a, b) = (Elem::<Aes>::from_raw(0x53), Elem::<Aes>::from_raw(0xca));
    assert_eq!(
        involution.embed(Elem::<Aes>::ONE),
        Elem::<Gf8<Poly<REED_SOLOMON>>>::ONE
    );
    assert_eq!(
        involution.embed(a * b),
        involution.embed(a) * involution.embed(b)
    );
    assert_eq!(involution.restrict(involution.embed(a)), Some(a));
    println!(
        "AES {a} * {b} = {} maps to {} in the Reed-Solomon basis.",
        a * b,
        involution.embed(a * b)
    );

    // Above the AES field, inclusions keep the base in the low component.
    // The relative Frobenius, trace, and norm come with the embedding.
    let tower = Embedding::<Aes, Gf16>::new().unwrap();
    let (lo, hi) = (Elem::<Aes>::from_raw(0x53), Elem::<Aes>::from_raw(0xca));
    let z = Elem::<Gf16>::from_components(lo, hi);
    assert_eq!(
        tower.embed(lo),
        Elem::<Gf16>::from_components(lo, Elem::<Aes>::ZERO)
    );
    assert_eq!(
        tower.frobenius(z),
        Elem::<Gf16>::from_components(lo + hi, hi)
    );
    assert_eq!(tower.trace(z), hi);
    assert_eq!(
        tower.norm(z),
        lo.square() + lo * hi + Elem::<Aes>::from_raw(0x20) * hi.square()
    );
    let indeterminate = Elem::<Gf16>::from_raw(0x0100);
    assert!(!tower.contains(indeterminate));
    assert_eq!(tower.restrict(indeterminate), None);
    println!(
        "Gf16 z={z} has relative trace {hi} and norm {}.",
        tower.norm(z)
    );

    // Cross-basis at equal degree: the Fan-Paar tower through the same
    // ladder. Equal descriptions embed identically; different descriptions
    // still compare as fields, not as raw bytes.
    let cross = Embedding::<FanPaar16, Gf16>::new().unwrap();
    let (p, q) = (
        Elem::<FanPaar16>::from_raw(0x35C2),
        Elem::<FanPaar16>::from_raw(0x1234),
    );
    assert_eq!(cross.restrict(cross.embed(p * q)), Some(p * q));
    assert_ne!(cross.embed(p).to_raw(), p.to_raw());
    println!(
        "Fan-Paar {p} maps to {} in the Rijndael basis.",
        cross.embed(p)
    );

    // Through to degree 64: commuting triangles make the composite the
    // embedding of the composite pair, and the relative Frobenius has the
    // extension order.
    let wide = Embedding::<Gf16, Gf64>::new().unwrap();
    let lifted = wide.embed(z);
    assert!(wide.contains(lifted));
    assert_eq!(wide.restrict(lifted), Some(z));
    assert_eq!(
        wide.embed(tower.embed(lo)),
        Embedding::<Aes, Gf64>::new().unwrap().embed(lo)
    );
    let mut frobenius = lifted;
    for _ in 0..4 {
        frobenius = wide.frobenius(frobenius);
    }
    assert_eq!(frobenius, lifted);
    println!("Gf16 z={z} lifts into GF(2^64) as {lifted}; four Frobenius steps return it.");

    // Embedding<Gf1, T> carries the absolute trace and norm.
    let absolute = Embedding::<Gf1, Gf16>::new().unwrap();
    let mut conjugate_sum = Elem::<Gf16>::ZERO;
    let mut power = z;
    for _ in 0..16 {
        conjugate_sum += power;
        power = power.square();
    }
    assert_eq!(absolute.restrict(conjugate_sum), Some(absolute.trace(z)));
    println!(
        "Absolute trace of z is {}.",
        u8::from(absolute.trace(z).is_one())
    );

    // Degrees that do not divide are rejected at construction.
    assert_eq!(
        Embedding::<Gf16, Aes>::new().unwrap_err(),
        EmbeddingError::IncompatibleDegree {
            source: 16,
            target: 8
        }
    );
    println!("GF(2^16) into GF(2^8) is rejected: source degree does not divide target degree.");
}
