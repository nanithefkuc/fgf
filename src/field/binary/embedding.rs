//! Coherent embeddings between binary-field presentations.
//!
//! [`Embedding<S, T>`](Embedding) prepares the canonical inclusion of the
//! binary field `S` into the binary field `T`, for every pair whose source
//! degree divides its target degree. Construction and application allocate
//! nothing; the prepared state is fixed column arrays.
//!
//! # Reference ladder
//!
//! The canonical map factors through one frozen reference per degree:
//! GF(2) as [`Poly<3>`](super::poly::Poly), `Poly<0x7>` at degree two,
//! `Poly<0x13>` at degree four, the AES field at degree eight, and the
//! Rijndael towers [`Gf16`](super::tower::Gf16),
//! [`Gf32`](super::tower::Gf32), [`Gf64`](super::tower::Gf64) above it.
//! Consecutive inclusions are frozen: the identity into degree two, the
//! smallest raw reference root of the source polynomial below the AES
//! field, and the low component above it.
//!
//! For a presentation `R`, the canonical map `phi_R` into the reference of
//! the same degree sends a flat polynomial's indeterminate to the smallest
//! raw reference root of its polynomial, converts an ordered basis through
//! its polynomial coordinates, and transports a tower through its base
//! before mapping the indeterminate to a root of the transported relation.
//! Presentations of degree at most eight tabulate that root search over the
//! bounded field; larger towers solve the transported quadratic through
//! recursive Artin-Schreier reduction in the reference tower, with no
//! elimination and no enumeration above the byte field. The default
//! embedding of `S` into `T` is the composite of the source map, the frozen
//! inclusion, and the inverse target map, so equal descriptions embed
//! identically and triangles of inclusions commute.
//!
//! # Application
//!
//! [`embed`](Embedding::embed) applies the prepared columns directly.
//! [`restrict`](Embedding::restrict) inverts them through the ladder:
//! members are exactly the reference words whose components above the
//! source degree are zero, except that sources below degree eight use the
//! frozen inverse inclusions of the byte field. [`frobenius`](Embedding::frobenius),
//! [`trace`](Embedding::trace), and [`norm`](Embedding::norm) implement the
//! relative Galois action of the extension; `Embedding<Gf1, T>` supplies
//! the absolute trace and norm.
//!
//! ```
//! use fgf::{Elem, Embedding, Gf8, Gf16, Poly, AES};
//!
//! let embedding = Embedding::<Gf8<Poly<AES>>, Gf16>::new().unwrap();
//! let x = Elem::<Gf8<Poly<AES>>>::from_raw(0x53);
//! let lifted = embedding.embed(x);
//! assert!(embedding.contains(lifted));
//! assert_eq!(embedding.restrict(lifted), Some(x));
//! assert_eq!(embedding.frobenius(lifted), lifted);
//! ```

use core::fmt;
use core::marker::PhantomData;

use super::BinaryField;
use super::description::{BinaryDescription, NodeData};
use super::tower;
use crate::field::{Elem, FieldElem};

/// The error returned when a requested embedding cannot exist.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingError {
    /// The source degree does not divide the target degree.
    IncompatibleDegree {
        /// Total degree of the source field.
        source: u32,
        /// Total degree of the target field.
        target: u32,
    },
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let EmbeddingError::IncompatibleDegree { source, target } = self;
        write!(
            formatter,
            "source degree {source} does not divide target degree {target}"
        )
    }
}

impl core::error::Error for EmbeddingError {}

/// The reference description of the frozen ladder at degree one.
static REFERENCE_1: BinaryDescription = BinaryDescription::polynomial(1, 3);
/// The reference description of the frozen ladder at degree two.
static REFERENCE_2: BinaryDescription = BinaryDescription::polynomial(2, 0x7);
/// The reference description of the frozen ladder at degree four.
static REFERENCE_4: BinaryDescription = BinaryDescription::polynomial(4, 0x13);
/// The reference description of the frozen ladder at degree eight.
static REFERENCE_8: BinaryDescription = BinaryDescription::polynomial(8, 0x11B);

/// The frozen reference of `degree`.
fn reference(degree: u32) -> &'static BinaryDescription {
    match degree {
        1 => &REFERENCE_1,
        2 => &REFERENCE_2,
        4 => &REFERENCE_4,
        8 => &REFERENCE_8,
        16 => tower::RIJNDAEL16_DESC,
        32 => tower::RIJNDAEL32_DESC,
        64 => tower::RIJNDAEL64_DESC,
        _ => panic!("no reference field of degree {degree}"),
    }
}

/// Apply the GF(2)-linear map given by `columns` to `word`.
fn fold(columns: &[u64; 64], word: u64) -> u64 {
    let mut acc = 0;
    let mut rest = word;
    while rest != 0 {
        acc ^= columns[rest.trailing_zeros() as usize];
        rest &= rest - 1;
    }
    acc
}

/// Evaluate the full polynomial `poly` at `x` inside `field`.
///
/// Horner descent from bit `top`, so leading zero coefficients are free.
fn evaluate(field: &BinaryDescription, top: u32, poly: u128, x: u64) -> u64 {
    let mut acc = 0;
    let mut bit = top;
    loop {
        acc = field.mul(acc, x);
        if (poly >> bit) & 1 == 1 {
            acc ^= 1;
        }
        if bit == 0 {
            break;
        }
        bit -= 1;
    }
    acc
}

/// The smallest nonzero raw root of `poly` in the reference of degree `n`.
fn smallest_root(field: &BinaryDescription, n: u32, poly: u128) -> u64 {
    for candidate in 1..(1u64 << n) {
        if evaluate(field, n, poly, candidate) == 0 {
            return candidate;
        }
    }
    panic!("an irreducible polynomial has no root in the reference field")
}

/// Solve `w^2 + w = c` in the reference of degree `h`.
///
/// Degrees up to eight enumerate the bounded field directly. Larger
/// degrees recurse through the reference tower `Ref_h = Ref_k[s]`: the
/// high component fixes `q` up to one branch selected by an absolute
/// trace, and the low component recurses.
// The single-character locals mirror the tower identities they name.
#[allow(clippy::many_single_char_names)]
fn solve_artin_schreier(h: u32, c: u64) -> u64 {
    let ref_h = reference(h);
    if h <= 8 {
        for candidate in 0..(1u64 << h) {
            if ref_h.mul(candidate, candidate) ^ candidate == c {
                return candidate;
            }
        }
        panic!("reference artin-schreier equation has no solution");
    }
    let k = h / 2;
    let ref_k = reference(k);
    let NodeData::Quadratic { a, b, .. } = ref_h.root_data() else {
        panic!("the reference above degree eight is a quadratic tower");
    };
    let mask = (1u64 << k) - 1;
    let c0 = c & mask;
    let c1 = c >> k;
    let y = solve_artin_schreier(k, ref_k.mul(a, c1));
    let a_inv = ref_k.inv(a);
    let mut q = ref_k.mul(y, a_inv);
    let mut rhs = c0 ^ ref_k.mul(b, ref_k.mul(q, q));
    if ref_k.trace(rhs) != 0 {
        q ^= a_inv;
        rhs = c0 ^ ref_k.mul(b, ref_k.mul(q, q));
        assert!(
            ref_k.trace(rhs) == 0,
            "both artin-schreier branches have trace one"
        );
    }
    let p = solve_artin_schreier(k, rhs);
    p | (q << k)
}

/// Columns of one frozen consecutive reference inclusion `Ref_d -> Ref_2d`.
///
/// Below degree eight the indeterminate maps to the smallest raw root of
/// the source reference polynomial; above, the inclusion is the low
/// component.
fn ladder_step(d: u32) -> [u64; 64] {
    let mut columns = [0u64; 64];
    if d <= 4 {
        let NodeData::Polynomial { full, .. } = reference(d).root_data() else {
            panic!("the reference below degree eight is a flat polynomial");
        };
        let target = reference(2 * d);
        let root = smallest_root(target, 2 * d, full);
        let mut power = 1;
        for column in columns.iter_mut().take(d as usize) {
            *column = power;
            power = target.mul(power, root);
        }
    } else {
        for (index, column) in columns.iter_mut().enumerate().take(d as usize) {
            *column = 1 << index;
        }
    }
    columns
}

/// Columns of the frozen reference inclusion `Ref_m -> Ref_n`, for `m`
/// dividing `n`. Equal degrees yield the identity.
fn ladder_inclusion(m: u32, n: u32) -> [u64; 64] {
    let mut columns = [0u64; 64];
    for (index, column) in columns.iter_mut().enumerate().take(m as usize) {
        *column = 1 << index;
    }
    let mut degree = m;
    while degree < n {
        let step = ladder_step(degree);
        let mut lifted = [0u64; 64];
        for (index, column) in lifted.iter_mut().enumerate().take(m as usize) {
            *column = fold(&step, columns[index]);
        }
        columns = lifted;
        degree *= 2;
    }
    columns
}

/// The canonical map of one presentation into the reference of its degree,
/// as GF(2)-linear columns both ways.
struct PhiMap {
    degree: u32,
    forward: [u64; 64],
    inverse: [u64; 64],
}

/// The identity map at `degree`.
fn identity_phi(degree: u32) -> PhiMap {
    let mut forward = [0u64; 64];
    let mut inverse = [0u64; 64];
    for index in 0..degree as usize {
        forward[index] = 1 << index;
        inverse[index] = 1 << index;
    }
    PhiMap {
        degree,
        forward,
        inverse,
    }
}

/// Build the canonical map of the flat polynomial presentation `full`.
fn build_polynomial_phi(degree: u32, full: u128) -> PhiMap {
    let field = reference(degree);
    if let NodeData::Polynomial {
        degree: ref_degree,
        full: ref_full,
    } = field.root_data()
        && u32::from(ref_degree) == degree
        && ref_full == full
    {
        return identity_phi(degree);
    }
    let root = smallest_root(field, degree, full);
    let mut forward = [0u64; 64];
    let mut power = 1;
    for column in forward.iter_mut().take(degree as usize) {
        *column = power;
        power = field.mul(power, root);
    }
    let mut inverse = [0u64; 64];
    // The whole map and its inverse lookup are bounded by the byte field.
    let mut forward_table = [0u8; 256];
    let mut inverse_table = [0u8; 256];
    #[allow(clippy::cast_possible_truncation)]
    for (word, slot) in forward_table.iter_mut().enumerate().take(1usize << degree) {
        let image = fold(&forward, word as u64) as u8;
        *slot = image;
        inverse_table[usize::from(image)] = word as u8;
    }
    for (index, column) in inverse.iter_mut().enumerate().take(degree as usize) {
        *column = u64::from(inverse_table[1usize << index]);
    }
    PhiMap {
        degree,
        forward,
        inverse,
    }
}

/// Build the canonical map of the root presentation of `description`.
fn build_phi(description: &BinaryDescription) -> PhiMap {
    match description.root_data() {
        NodeData::Polynomial { degree, full } => build_polynomial_phi(u32::from(degree), full),
        NodeData::Basis {
            to_base, from_base, ..
        } => {
            let inner = build_phi(&description.base_description());
            let mut forward = [0u64; 64];
            for (index, column) in forward.iter_mut().enumerate().take(inner.degree as usize) {
                *column = fold(
                    &inner.forward,
                    super::description::apply(to_base, 1 << index),
                );
            }
            let mut inverse = [0u64; 64];
            for (index, column) in inverse.iter_mut().enumerate().take(inner.degree as usize) {
                *column = super::description::apply(from_base, fold(&inner.inverse, 1 << index));
            }
            PhiMap {
                degree: inner.degree,
                forward,
                inverse,
            }
        }
        NodeData::Quadratic { a, b, .. } => {
            let inner = build_phi(&description.base_description());
            build_quadratic_phi(&inner, a, b)
        }
    }
}

/// Build the canonical map of a quadratic tower over a base whose map is
/// `base`, under the relation `t^2 + a*t + b`.
///
/// Output degrees up to eight enumerate the bounded reference for the root
/// of the transported relation. Larger degrees solve it through the
/// Artin-Schreier reduction: `v = A'/a_r`, then `w` with
/// `w^2 + w = (B' + b_r*v^2)/A'^2` and `u = A'*w`, giving the root
/// `z = u + v*r`; the smaller of `z` and `z + A'` in raw order is stored.
// The single-character locals mirror the node fields they transport.
#[allow(clippy::many_single_char_names)]
fn build_quadratic_phi(base: &PhiMap, a: u64, b: u64) -> PhiMap {
    let n = 2 * base.degree;
    let transported_a = fold(&base.forward, a);
    let transported_b = fold(&base.forward, b);
    if n <= 8 {
        build_small_quadratic_phi(base, transported_a, transported_b)
    } else {
        build_wide_quadratic_phi(base, transported_a, transported_b)
    }
}

/// The bounded-root branch of [`build_quadratic_phi`] for output degrees
/// up to eight.
// The single-character locals mirror the node fields they transport.
#[allow(clippy::many_single_char_names)]
fn build_small_quadratic_phi(base: &PhiMap, transported_a: u64, transported_b: u64) -> PhiMap {
    let h = base.degree;
    let n = 2 * h;
    let ref_n = reference(n);
    // The base embeds through the frozen ladder inclusion, whose
    // low components are not literal words below the AES field.
    let iota = ladder_inclusion(h, n);
    let embedded_a = fold(&iota, transported_a);
    let embedded_b = fold(&iota, transported_b);
    let mut root = 0;
    for candidate in 1..(1u64 << n) {
        let square = ref_n.mul(candidate, candidate);
        if square ^ ref_n.mul(candidate, embedded_a) ^ embedded_b == 0 {
            root = candidate;
            break;
        }
    }
    assert!(root != 0, "transported relation has no reference root");
    let mut forward = [0u64; 64];
    for (index, column) in forward.iter_mut().enumerate().take(h as usize) {
        *column = fold(&iota, base.forward[index]);
    }
    for (index, column) in forward
        .iter_mut()
        .skip(h as usize)
        .enumerate()
        .take(h as usize)
    {
        *column = ref_n.mul(fold(&iota, base.forward[index]), root);
    }
    let mut inverse = [0u64; 64];
    // The whole map and its inverse lookup are bounded by the byte field.
    let mut forward_table = [0u8; 256];
    let mut inverse_table = [0u8; 256];
    #[allow(clippy::cast_possible_truncation)]
    for (word, slot) in forward_table.iter_mut().enumerate().take(1usize << n) {
        let image = fold(&forward, word as u64) as u8;
        *slot = image;
        inverse_table[usize::from(image)] = word as u8;
    }
    for (index, column) in inverse.iter_mut().enumerate().take(n as usize) {
        *column = u64::from(inverse_table[1usize << index]);
    }
    PhiMap {
        degree: n,
        forward,
        inverse,
    }
}

/// The Artin-Schreier branch of [`build_quadratic_phi`] for output degrees
/// above eight.
// The single-character locals mirror the node fields they transport.
#[allow(clippy::many_single_char_names)]
fn build_wide_quadratic_phi(base: &PhiMap, transported_a: u64, transported_b: u64) -> PhiMap {
    let h = base.degree;
    let n = 2 * h;
    let ref_h = reference(h);
    let ref_n = reference(n);
    let NodeData::Quadratic { a: a_r, b: b_r, .. } = ref_n.root_data() else {
        panic!("the reference above degree eight is a quadratic tower");
    };
    let v = ref_h.mul(transported_a, ref_h.inv(a_r));
    assert!(
        v != 0,
        "transported relation is reducible over the reference"
    );
    let norm_a = ref_h.mul(transported_a, transported_a);
    let rhs = ref_h.mul(
        transported_b ^ ref_h.mul(b_r, ref_h.mul(v, v)),
        ref_h.inv(norm_a),
    );
    let w = solve_artin_schreier(h, rhs);
    let mut u = ref_h.mul(transported_a, w);
    let mut z = u | (v << h);
    assert!(
        ref_n.mul(z, z) ^ ref_n.mul(z, transported_a) ^ transported_b == 0,
        "the solved root fails the transported relation"
    );
    if (z ^ transported_a) < z {
        u ^= transported_a;
        z = u | (v << h);
    }
    let mut forward = [0u64; 64];
    forward[..h as usize].copy_from_slice(&base.forward[..h as usize]);
    for (index, column) in forward
        .iter_mut()
        .skip(h as usize)
        .enumerate()
        .take(h as usize)
    {
        *column = ref_n.mul(base.forward[index], z);
    }
    let v_inv = ref_h.inv(v);
    let mut inverse = [0u64; 64];
    inverse[..h as usize].copy_from_slice(&base.inverse[..h as usize]);
    for (index, column) in inverse
        .iter_mut()
        .skip(h as usize)
        .enumerate()
        .take(h as usize)
    {
        // Recursive inverse transport: beta = q / v, alpha = p + beta*u.
        let beta = ref_h.mul(1 << index, v_inv);
        let alpha = ref_h.mul(beta, u);
        *column = fold(&base.inverse, alpha) | (fold(&base.inverse, beta) << h);
    }
    PhiMap {
        degree: n,
        forward,
        inverse,
    }
}

/// Table entry marking a byte that is not in a frozen inclusion's image.
const NONMEMBER: u8 = 0xFF;

/// The frozen inverse of the ladder inclusion `Ref_m -> Ref_top`, as a byte
/// lookup, for a source below degree eight.
// Inclusion images and source words below degree eight hold one byte.
#[allow(clippy::cast_possible_truncation)]
fn inverse_inclusion(m: u32, top: u32) -> [u8; 256] {
    let columns = ladder_inclusion(m, top);
    let mut table = [NONMEMBER; 256];
    for word in 0..(1u64 << m) {
        let image = fold(&columns, word) as u8;
        table[usize::from(image)] = word as u8;
    }
    table
}

/// Ladder membership data for restriction.
// Prepared state is fixed arrays by contract; boxing the table would
// allocate where the embedding promises none.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
enum Restriction {
    /// Equal degrees: the reference map itself is a bijection.
    Equal,
    /// Source degree at least eight: members are exactly the reference
    /// words with zero bits at and above `shift`.
    HighZero {
        /// The first bit above the embedded source.
        shift: u32,
    },
    /// Source degree below eight: zero bits at and above `shift`, then the
    /// frozen inverse-inclusion lookup of the byte field.
    Table {
        /// The first bit above the checked span.
        shift: u32,
        /// Inverse inclusion with `NONMEMBER` marking nonmembers.
        inverse: [u8; 256],
    },
}

/// A prepared, explicitly selected field inclusion of `S` into `T`.
///
/// The embedding is the canonical one through the frozen reference ladder:
/// the source's canonical map into the reference of its degree, the frozen
/// consecutive inclusions up to the target's degree, and the inverse of the
/// target's canonical map. Equal presentation descriptions therefore embed
/// identically, and inclusions commute as triangles.
///
/// Construction verifies the composed map on valid descriptions (unit,
/// inverse round trips, basis-pair multiplicativity, and Frobenius-fixed
/// images); a failed verification is a crate defect and panics. Descriptor
/// invalidity is the descriptions' own use-time validation, not an error
/// here.
///
/// The prepared state is fixed arrays; construction and every method are
/// allocation-free. Repeated application never reruns root construction.
pub struct Embedding<S: BinaryField, T: BinaryField> {
    basis_images: [u64; 64],
    target_to_reference: [u64; 64],
    source_from_reference: [u64; 64],
    source_degree: u32,
    target_degree: u32,
    restriction: Restriction,
    phantom: PhantomData<fn() -> (S, T)>,
}

// A derive would bound the impl on `S: Clone`; the manual impl needs only
// the field contract the type already carries.
#[allow(clippy::expl_impl_clone_on_copy)]
impl<S: BinaryField, T: BinaryField> Clone for Embedding<S, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: BinaryField, T: BinaryField> Copy for Embedding<S, T> {}

impl<S: BinaryField, T: BinaryField> fmt::Debug for Embedding<S, T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Embedding<{}, {}>", S::NAME, T::NAME)
    }
}

impl<S: BinaryField, T: BinaryField> Embedding<S, T> {
    /// Prepare the canonical embedding of `S` into `T`.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::IncompatibleDegree`] when `S::DEGREE`
    /// does not divide `T::DEGREE`; no other pair is rejected.
    ///
    /// # Panics
    ///
    /// Panics when verification of the composed map fails on otherwise
    /// valid descriptions. That is a crate defect, not an input condition.
    pub fn new() -> Result<Self, EmbeddingError> {
        let m = S::DEGREE;
        let n = T::DEGREE;
        if n % m != 0 {
            return Err(EmbeddingError::IncompatibleDegree {
                source: m,
                target: n,
            });
        }
        let source = build_phi(S::DESCRIPTION);
        let target = build_phi(T::DESCRIPTION);
        assert!(
            source.degree == m && target.degree == n,
            "a description degree disagrees with its field"
        );
        let inclusion = ladder_inclusion(m, n);
        let mut basis_images = [0u64; 64];
        for (index, image) in basis_images.iter_mut().enumerate().take(m as usize) {
            let reference = fold(&inclusion, source.forward[index]);
            *image = fold(&target.inverse, reference);
        }
        let restriction = if m == n {
            Restriction::Equal
        } else if m >= 8 {
            Restriction::HighZero { shift: m }
        } else {
            Restriction::Table {
                shift: if n >= 16 { 8 } else { n },
                inverse: inverse_inclusion(m, if n >= 8 { 8 } else { n }),
            }
        };
        let embedding = Self {
            basis_images,
            target_to_reference: target.forward,
            source_from_reference: source.inverse,
            source_degree: m,
            target_degree: n,
            restriction,
            phantom: PhantomData,
        };
        embedding.verify(&source, &target);
        Ok(embedding)
    }

    /// Map `value` into the target field.
    ///
    /// # Panics
    ///
    /// Panics only on a crate defect: a prepared image that fails target
    /// coordinate conversion, which construction verified impossible.
    #[must_use]
    pub fn embed(&self, value: Elem<S>) -> Elem<T> {
        let word = fold(&self.basis_images, S::to_coordinates(value));
        T::from_coordinates(word).expect("an embedding image fits the target degree")
    }

    /// Whether `value` lies in the image of this embedding.
    ///
    /// Membership is the fixed points of the relative Frobenius.
    #[must_use]
    pub fn contains(&self, value: Elem<T>) -> bool {
        self.frobenius(value) == value
    }

    /// The inverse of [`embed`](Self::embed) on the image, or `None`.
    ///
    /// # Panics
    ///
    /// Panics only on a crate defect: a restricted word that fails source
    /// coordinate conversion, which construction verified impossible.
    #[must_use]
    pub fn restrict(&self, value: Elem<T>) -> Option<Elem<S>> {
        self.restrict_coordinates(T::to_coordinates(value))
            .map(|word| {
                S::from_coordinates(word).expect("a restriction image fits the source degree")
            })
    }

    /// The relative Frobenius `x^(2^m)` of the extension.
    ///
    /// Computed by `m` repeated squarings, so no exponent is ever shifted
    /// into or out of a 64-bit word.
    #[must_use]
    pub fn frobenius(&self, value: Elem<T>) -> Elem<T> {
        let mut power = value;
        for _ in 0..self.source_degree {
            power = power.square();
        }
        power
    }

    /// The relative trace: the sum of the conjugates under the relative
    /// Frobenius, restricted to the source.
    ///
    /// Equal degrees reduce to restriction. `Embedding<Gf1, T>` supplies
    /// the absolute trace.
    ///
    /// # Panics
    ///
    /// Panics only on a crate defect: a conjugate sum outside the verified
    /// embedding image.
    #[must_use]
    pub fn trace(&self, value: Elem<T>) -> Elem<S> {
        let mut sum = value;
        let mut power = value;
        for _ in 1..self.target_degree / self.source_degree {
            power = self.frobenius(power);
            sum = sum.add(power);
        }
        self.restrict(sum)
            .expect("a conjugate sum lies in the embedding image")
    }

    /// The relative norm: the product of the conjugates under the relative
    /// Frobenius, restricted to the source.
    ///
    /// Equal degrees reduce to restriction. `Embedding<Gf1, T>` supplies
    /// the absolute norm.
    ///
    /// # Panics
    ///
    /// Panics only on a crate defect: a conjugate product outside the
    /// verified embedding image.
    #[must_use]
    pub fn norm(&self, value: Elem<T>) -> Elem<S> {
        let mut product = value;
        let mut power = value;
        for _ in 1..self.target_degree / self.source_degree {
            power = self.frobenius(power);
            product = product.mul(power);
        }
        self.restrict(product)
            .expect("a conjugate product lies in the embedding image")
    }

    /// Restriction over target coordinate words.
    fn restrict_coordinates(&self, word: u64) -> Option<u64> {
        let reference = fold(&self.target_to_reference, word);
        let low = match &self.restriction {
            Restriction::Equal => reference,
            Restriction::HighZero { shift } => {
                if reference >> shift == 0 {
                    reference
                } else {
                    return None;
                }
            }
            Restriction::Table { shift, inverse } => {
                if reference >> shift != 0 {
                    return None;
                }
                // The checked span holds one byte.
                #[allow(clippy::cast_possible_truncation)]
                let byte = (reference & 0xFF) as u8;
                let entry = inverse[usize::from(byte)];
                if entry == NONMEMBER {
                    return None;
                }
                u64::from(entry)
            }
        };
        Some(fold(&self.source_from_reference, low))
    }

    /// Verify the prepared map on valid descriptions.
    fn verify(&self, source: &PhiMap, target: &PhiMap) {
        let m = self.source_degree as usize;
        // Unit.
        let source_one = S::DESCRIPTION.one();
        let target_one = T::DESCRIPTION.one();
        assert_eq!(
            fold(&self.basis_images, source_one),
            target_one,
            "an embedding does not preserve one"
        );
        // Inverse coordinate round trips on every basis vector, both ways.
        for index in 0..m {
            assert_eq!(
                fold(&source.inverse, source.forward[index]),
                1 << index,
                "a source map is not invertible"
            );
        }
        for index in 0..self.target_degree as usize {
            assert_eq!(
                fold(&target.inverse, target.forward[index]),
                1 << index,
                "a target map is not invertible"
            );
        }
        // Basis images: Frobenius-fixed, restrict back, and multiplicative
        // in pairs. A unital map multiplicative on basis products is a
        // ring homomorphism.
        for left in 0..m {
            let image = fold(&self.basis_images, 1 << left);
            assert_eq!(
                T::DESCRIPTION.pow(image, 1u128 << self.source_degree),
                image,
                "a basis image is not Frobenius-fixed"
            );
            assert_eq!(
                self.restrict_coordinates(image),
                Some(1 << left),
                "restriction does not invert embedding"
            );
            for right in 0..m {
                let product = S::DESCRIPTION.mul(1 << left, 1 << right);
                assert_eq!(
                    fold(&self.basis_images, product),
                    T::DESCRIPTION.mul(
                        fold(&self.basis_images, 1 << left),
                        fold(&self.basis_images, 1 << right)
                    ),
                    "an embedding is not multiplicative on a basis pair"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shipped tower at degree sixteen and above maps its
    /// indeterminate to the smaller root of its transported relation, and
    /// the resulting map round-trips.
    fn check_tower_phi(description: &'static BinaryDescription) {
        let phi = build_phi(description);
        let n = phi.degree;
        let h = n / 2;
        for index in 0..n as usize {
            assert_eq!(fold(&phi.inverse, phi.forward[index]), 1 << index);
        }
        let NodeData::Quadratic { a, b, .. } = description.root_data() else {
            panic!("the shipped tower root is a quadratic node");
        };
        let transported_a = fold(&phi.forward, a);
        let transported_b = fold(&phi.forward, b);
        let ref_n = reference(n);
        // One is the first forward column's base image, so the root is the
        // image of the indeterminate itself.
        let z = phi.forward[h as usize];
        assert_eq!(
            ref_n.mul(z, z) ^ ref_n.mul(z, transported_a) ^ transported_b,
            0,
            "the shipped tower root fails its transported relation"
        );
        assert!(z <= z ^ transported_a, "the chosen root is not minimal");
    }

    #[test]
    fn shipped_tower_reference_maps_satisfy_their_relations() {
        check_tower_phi(tower::RIJNDAEL16_DESC);
        check_tower_phi(tower::RIJNDAEL32_DESC);
        check_tower_phi(tower::RIJNDAEL64_DESC);
        check_tower_phi(tower::FANPAAR16_DESC);
        check_tower_phi(tower::FANPAAR32_DESC);
        check_tower_phi(tower::FANPAAR64_DESC);
    }

    #[test]
    fn reference_ladder_steps_are_frozen() {
        // One maps to one; each step's nontrivial action is its polynomial
        // root search below degree eight and its literal low component
        // above.
        let step = ladder_step(1);
        assert_eq!(step[0], 1);
        for d in [8u32, 16, 32] {
            let step = ladder_step(d);
            for (index, column) in step.iter().enumerate().take(d as usize) {
                assert_eq!(*column, 1 << index);
            }
        }
        // The ladder inclusion of the whole chain is the low word.
        let columns = ladder_inclusion(8, 64);
        for (index, column) in columns.iter().enumerate().take(8) {
            assert_eq!(*column, 1 << index);
        }
    }

    #[test]
    fn artin_schreier_solutions_satisfy_their_equations() {
        for h in [16u32, 32, 64] {
            let field = reference(h);
            let mut state = 0x0123_4567_89ab_cdefu64;
            for _ in 0..8 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                // Trace-zero words are exactly the solvable right sides.
                if field.trace(state) == 0 {
                    let w = solve_artin_schreier(h, state);
                    assert_eq!(field.mul(w, w) ^ w, state);
                }
            }
        }
    }
}
