//! Characteristic-independent field embeddings and isomorphisms.
//!
//! [`Embedding<S, T>`](Embedding) prepares one injective field homomorphism
//! from `S` into `T`, for every pair of rows of equal characteristic whose
//! source degree divides the target degree. [`Isomorphism<S, T>`](Isomorphism)
//! is its equal-degree form with a total inverse. One implementation serves
//! every characteristic: the prepared state stores the image in `T` of each
//! generator level of `S`, embedding evaluates by Horner's rule in the
//! target's arithmetic, membership is the relative Frobenius, and
//! restriction recovers components through trace-dual bases without solving
//! a linear system.
//!
//! A map comes from two sources only: an inclusion forced by `Extension`
//! structure (the source itself, its prime subfield, or a row of the
//! target's own tower), or caller-supplied generator images validated
//! against the transported relations. Characteristic two keeps the frozen
//! reference ladder as its canonical-image rule, so every historical binary
//! result reproduces exactly; odd characteristic chooses no canonical map
//! between unrelated presentations.
//!
//! Construction and every method are allocation-free; the prepared state is
//! fixed arrays. Repeated application never reruns image construction.
//!
//! # Levels
//!
//! A prepared relation stores one image per generator level of the source,
//! prime field upward: one level per extension node, whose generator is the
//! indeterminate; one level for a flat polynomial leaf above degree one,
//! whose generator is the residue of the indeterminate; one level for a
//! basis leaf, whose generator is the underlying polynomial root expressed
//! in the basis. Rows of absolute degree one hold no levels.
//!
//! ```
//! use fgf::{AES, Elem, Embedding, Extension, FiniteField, Polynomial, Rijndael16};
//!
//! type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
//! let embedding = Embedding::<FiniteField<2, 8, Polynomial<AES>>, Gf16>::new().unwrap();
//! let x = Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53);
//! let lifted = embedding.embed(x);
//! assert!(embedding.contains(lifted));
//! assert_eq!(embedding.restrict(lifted), Some(x));
//! assert_eq!(embedding.frobenius(lifted), lifted);
//! ```

mod ladder;

use core::marker::PhantomData;

use super::binary::description::apply;
use super::extension::descriptor::{Descriptor, Node};
use super::{Elem, Field, FieldElem, PrimeIdentity};

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
    /// Neither a structural inclusion nor, in characteristic two, the
    /// frozen reference ladder names a canonical map for the pair.
    NoCanonicalMap,
    /// A supplied generator image fails its transported relation, or the
    /// image count differs from the source's generator count.
    InvalidImages,
}

impl core::fmt::Display for EmbeddingError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EmbeddingError::IncompatibleDegree { source, target } => write!(
                formatter,
                "source degree {source} does not divide target degree {target}"
            ),
            EmbeddingError::NoCanonicalMap => {
                formatter.write_str("no canonical map between the presentations")
            }
            EmbeddingError::InvalidImages => {
                formatter.write_str("generator images fail the transported relations")
            }
        }
    }
}

impl core::error::Error for EmbeddingError {}

// ---------------------------------------------------------------------------
// Generator levels
// ---------------------------------------------------------------------------

/// The shape of one generator level of a source presentation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum LevelKind {
    /// An extension node: the generator is the indeterminate, and the
    /// transported relation is checked against the node's coefficients.
    Extension,
    /// A flat polynomial leaf above degree one: the generator is the
    /// indeterminate residue, checked against the full polynomial.
    Polynomial {
        /// Total degree of the leaf.
        degree: u8,
        /// The full reduction polynomial including the leading term.
        full: u128,
    },
    /// A basis leaf: the generator is the underlying polynomial root
    /// expressed in the basis, checked against that polynomial.
    Basis {
        /// Index of the underlying polynomial node.
        poly: u8,
    },
}

/// One generator level of a source presentation, prime field upward.
#[derive(Clone, Copy)]
struct Level {
    /// The level's shape.
    kind: LevelKind,
    /// Descriptor node index holding the level's relation data.
    node: u8,
    /// Base node index the level extends, if any.
    base: Option<u8>,
    /// Absolute degree of the base over the prime field: one for a prime
    /// base, the base's degree otherwise.
    base_degree: u32,
    /// Offset of the level's coordinate span in the flattened presentation.
    span_offset: usize,
    /// Length of the level's coordinate span.
    span_len: usize,
}

/// The generator levels of the presentation rooted at `root`, prime field
/// upward: deeper levels first, so level zero sits nearest the prime field.
///
/// A basis leaf contributes its level and consumes its underlying
/// polynomial node, which supplies the relation but no separate level.
fn collect_levels(
    descriptor: &Descriptor,
    root: u8,
    offset: usize,
    out: &mut [Level; 8],
    count: &mut usize,
) {
    let push = |level: Level, out: &mut [Level; 8], count: &mut usize| {
        assert!(*count < 8, "a presentation holds at most eight levels");
        out[*count] = level;
        *count += 1;
    };
    match descriptor.node(root) {
        Node::Unused | Node::Prime { .. } => {}
        Node::Polynomial { degree, full } => {
            if *degree > 1 {
                push(
                    Level {
                        kind: LevelKind::Polynomial {
                            degree: *degree,
                            full: *full,
                        },
                        node: root,
                        base: None,
                        base_degree: 1,
                        span_offset: offset,
                        span_len: usize::from(*degree),
                    },
                    out,
                    count,
                );
            }
        }
        Node::Basis { base, .. } => {
            let span_len = super::extension::arithmetic::degree_of(descriptor, *base) as usize;
            push(
                Level {
                    kind: LevelKind::Basis { poly: *base },
                    node: root,
                    base: None,
                    base_degree: 1,
                    span_offset: offset,
                    span_len,
                },
                out,
                count,
            );
        }
        Node::Extension {
            base, base_degree, ..
        } => {
            collect_levels(descriptor, *base, offset, out, count);
            // Component zero starts the span; the indeterminate sits one
            // base degree in.
            let span_len = (*base_degree as usize) * level_relative(descriptor, root);
            push(
                Level {
                    kind: LevelKind::Extension,
                    node: root,
                    base: Some(*base),
                    base_degree: *base_degree,
                    span_offset: offset,
                    span_len,
                },
                out,
                count,
            );
        }
    }
}

/// The generator levels of `descriptor`, with its root degree.
fn levels(descriptor: &Descriptor) -> ([Level; 8], usize) {
    let mut out = [Level {
        kind: LevelKind::Extension,
        // Unused slots name no node: `u8::MAX` matches nothing.
        node: u8::MAX,
        base: None,
        base_degree: 0,
        span_offset: usize::MAX,
        span_len: 0,
    }; 8];
    let mut count = 0;
    collect_levels(descriptor, descriptor.root_index(), 0, &mut out, &mut count);
    (out, count)
}

/// The index of the level owned by `node`, or `u8::MAX` when the node owns
/// none (a prime leaf, a degree-one polynomial leaf, or a polynomial node
/// consumed by its basis leaf).
fn level_of(levels: &[Level; 8], count: usize, node: u8) -> Option<usize> {
    let mut i = 0;
    while i < count {
        if levels[i].node == node {
            return Some(i);
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// Coordinate surgery over descriptors
// ---------------------------------------------------------------------------

/// The flattened prime-subfield coordinates of one in the presentation
/// rooted at `root`: the multiplicative identity in descriptor order.
fn one_coordinates(descriptor: &Descriptor, root: u8, out: &mut [u64]) {
    match descriptor.node(root) {
        Node::Unused | Node::Prime { .. } => {
            out[0] = 1;
        }
        Node::Polynomial { degree, .. } => {
            out[0] = 1;
            let mut i = 1;
            while i < usize::from(*degree) {
                out[i] = 0;
                i += 1;
            }
        }
        Node::Basis { .. } => {
            // The basis one is the polynomial one transported through the
            // inverse map: pack the polynomial identity, convert, unpack.
            let mut poly = [0u64; 8];
            poly[0] = 1;
            let word = pack_bits(&poly);
            let Node::Basis { from_base, .. } = descriptor.node(root) else {
                panic!("a basis level reads its own node");
            };
            unpack_bits(apply(*from_base, word), out);
        }
        Node::Extension {
            base, base_degree, ..
        } => {
            one_coordinates(descriptor, *base, out);
            let mut i = *base_degree as usize;
            while i < out.len() {
                out[i] = 0;
                i += 1;
            }
        }
    }
}

/// The flattened prime-subfield coordinates of the level generator named
/// by `level`: the indeterminate of an extension or polynomial level, or
/// the underlying polynomial root expressed in the basis.
fn generator_coordinates(descriptor: &Descriptor, level: &Level, degree: usize, out: &mut [u64]) {
    let span = level.span_offset..level.span_offset + level.span_len;
    let mut i = span.start;
    while i < span.end {
        out[i] = 0;
        i += 1;
    }
    match level.kind {
        LevelKind::Extension => {
            // The indeterminate is the one of the base in component one.
            let mut one = [0u64; 128];
            one_coordinates(
                descriptor,
                level.base.expect("an extension level extends a base"),
                &mut one[..level.base_degree as usize],
            );
            let mut i = 0;
            while i < level.base_degree as usize {
                out[span.start + level.base_degree as usize + i] = one[i];
                i += 1;
            }
        }
        LevelKind::Polynomial { .. } => {
            out[span.start + 1] = 1;
        }
        LevelKind::Basis { poly } => {
            // The polynomial indeterminate transported into the basis.
            let mut poly_coords = [0u64; 8];
            poly_coords[1] = 1;
            let Node::Basis { from_base, .. } = descriptor.node(level.node) else {
                panic!("a basis level reads its own node");
            };
            let _ = poly;
            unpack_bits(
                apply(*from_base, pack_bits(&poly_coords)),
                &mut out[span.clone()],
            );
        }
    }
    let _ = degree;
}

/// Pack binary coordinate bits into a word, least significant first.
fn pack_bits(coordinates: &[u64]) -> u64 {
    let mut word = 0u64;
    let mut i = 0;
    while i < coordinates.len() {
        word |= (coordinates[i] & 1) << i;
        i += 1;
    }
    word
}

/// Unpack a word into binary coordinate bits, least significant first.
fn unpack_bits(word: u64, out: &mut [u64]) {
    let mut i = 0;
    while i < out.len() {
        out[i] = (word >> i) & 1;
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// The evaluation engine
// ---------------------------------------------------------------------------

/// Embed flattened `coords` of the presentation rooted at `at` into the
/// target row, through the level `images` in level order.
fn embed_coordinates<T: Field>(
    descriptor: &Descriptor,
    at: u8,
    coords: &[u64],
    levels: &[Level; 8],
    count: usize,
    images: &[Elem<T>],
) -> Elem<T> {
    match descriptor.node(at) {
        Node::Unused | Node::Prime { .. } => {
            Elem::<T>::from_residue(coords[0]).expect("a prime coordinate sits below the target")
        }
        Node::Polynomial { degree, .. } => {
            if *degree == 1 {
                Elem::<T>::from_residue(coords[0])
                    .expect("a prime coordinate sits below the target")
            } else {
                let image =
                    images[level_of(levels, count, at).expect("a polynomial level owns an image")];
                horner(image, coords)
            }
        }
        Node::Basis { .. } => {
            // Through the basis map into polynomial coordinates, then the
            // level evaluation.
            let Node::Basis { to_base, .. } = descriptor.node(at) else {
                panic!("a basis level reads its own node");
            };
            let mut poly = [0u64; 8];
            unpack_bits(apply(*to_base, pack_bits(coords)), &mut poly);
            let image = images[level_of(levels, count, at).expect("a basis level owns an image")];
            horner(image, &poly)
        }
        Node::Extension {
            base, base_degree, ..
        } => {
            let image =
                images[level_of(levels, count, at).expect("an extension level owns an image")];
            let span = *base_degree as usize;
            let relative = coords.len() / span;
            let mut acc = Elem::<T>::ZERO;
            let mut j = relative;
            while j > 0 {
                j -= 1;
                let component = embed_coordinates(
                    descriptor,
                    *base,
                    &coords[j * span..(j + 1) * span],
                    levels,
                    count,
                    images,
                );
                acc = acc * image + component;
            }
            acc
        }
    }
}
/// Horner evaluation of prime-residue coefficients, constant term first,
/// at `image` in the target's arithmetic.
fn horner<T: Field>(image: Elem<T>, coefficients: &[u64]) -> Elem<T> {
    let mut acc = Elem::<T>::ZERO;
    let mut i = coefficients.len();
    while i > 0 {
        i -= 1;
        let coefficient = Elem::<T>::from_residue(coefficients[i])
            .expect("a prime coordinate sits below the target");
        acc = acc * image + coefficient;
    }
    acc
}

/// Whether the transported relations of every level vanish at the supplied
/// images: the soundness check behind `from_images` and construction.
fn relations_hold<T: Field>(
    descriptor: &Descriptor,
    levels: &[Level; 8],
    count: usize,
    images: &[Elem<T>],
) -> bool {
    let mut l = 0;
    while l < count {
        let relative = level_relative(descriptor, levels[l].node);
        // The monic relation evaluated at the image by Horner descent from
        // the leading term.
        let mut acc = Elem::<T>::ONE;
        let mut j = relative;
        while j > 0 {
            j -= 1;
            acc = acc * images[l]
                + transport_coefficient(descriptor, levels[l].node, levels, count, images, j);
        }
        if acc != Elem::<T>::ZERO {
            return false;
        }
        l += 1;
    }
    true
}
/// Restrict `value` to the presentation rooted at `at`, writing its
/// flattened prime-subfield coordinates into `out`. Returns `false`
/// exactly when `value` lies outside the image.
/// Restrict `value` to the presentation rooted at `at`, writing its
/// flattened prime-subfield coordinates into `out`. Returns `false`
/// exactly when `value` lies outside the image.
fn restrict_coordinates<T: Field>(
    descriptor: &Descriptor,
    at: u8,
    value: Elem<T>,
    levels: &[Level; 8],
    count: usize,
    images: &[Elem<T>],
    out: &mut [u64],
) -> bool {
    // Membership is the fixed field of the relative Frobenius.
    if value.pow(order_of(descriptor, at)) != value {
        return false;
    }
    match descriptor.node(at) {
        Node::Unused | Node::Prime { .. } => {
            let Some(residue) = value.to_residue() else {
                return false;
            };
            out[0] = residue;
            true
        }
        Node::Polynomial { degree, .. } => {
            if *degree == 1 {
                let Some(residue) = value.to_residue() else {
                    return false;
                };
                out[0] = residue;
                return true;
            }
            // Flat levels hold prime residues directly.
            let mut poly = [0u64; 8];
            if out.len() > 8
                || !restrict_flat(
                    descriptor,
                    at,
                    value,
                    levels,
                    count,
                    images,
                    &mut poly[..out.len()],
                )
            {
                return false;
            }
            out.copy_from_slice(&poly[..out.len()]);
            true
        }
        Node::Basis { .. } => {
            // Recover polynomial coordinates through the dual basis, then
            // transport into the basis through the inverse map.
            let mut poly = [0u64; 8];
            let relative = out.len();
            if relative > 8
                || !restrict_flat(
                    descriptor,
                    at,
                    value,
                    levels,
                    count,
                    images,
                    &mut poly[..relative],
                )
            {
                return false;
            }
            let Node::Basis { from_base, .. } = descriptor.node(at) else {
                panic!("a basis level reads its own node");
            };
            unpack_bits(apply(*from_base, pack_bits(&poly[..relative])), out);
            true
        }
        Node::Extension {
            base, base_degree, ..
        } => {
            let span = *base_degree as usize;
            let relative = out.len() / span;
            let mut i = 0;
            while i < relative {
                let Some(component) =
                    restrict_component(descriptor, at, *base, value, i, levels, count, images)
                else {
                    return false;
                };
                let mut chunk = [0u64; 128];
                if !restrict_coordinates(
                    descriptor,
                    *base,
                    component,
                    levels,
                    count,
                    images,
                    &mut chunk[..span],
                ) {
                    return false;
                }
                out[i * span..(i + 1) * span].copy_from_slice(&chunk[..span]);
                i += 1;
            }
            true
        }
    }
}

/// Recover the `index`-th component of one extension level through its
/// trace-dual basis: the relative trace of `value` times the dual element
/// over the base order. Returns `None` when the level data is inconsistent.
#[allow(clippy::too_many_arguments)]
fn restrict_component<T: Field>(
    descriptor: &Descriptor,
    at: u8,
    base: u8,
    value: Elem<T>,
    index: usize,
    levels: &[Level; 8],
    count: usize,
    images: &[Elem<T>],
) -> Option<Elem<T>> {
    let sub = order_of(descriptor, base);
    let image = images[level_of(levels, count, at).expect("a restricted level owns an image")];
    let relative = level_relative(descriptor, at);
    let mut trace = Elem::<T>::ZERO;
    let mut power = value * dual_element(descriptor, at, image, levels, count, images, index)?;
    let mut k = 0;
    while k < relative {
        trace += power;
        power = power.pow(sub);
        k += 1;
    }
    Some(trace)
}

/// Recover the prime-residue components of one flat level through its
/// trace-dual basis over the prime field.
#[allow(clippy::too_many_arguments)]
fn restrict_flat<T: Field>(
    descriptor: &Descriptor,
    at: u8,
    value: Elem<T>,
    levels: &[Level; 8],
    count: usize,
    images: &[Elem<T>],
    out: &mut [u64],
) -> bool {
    let prime = u128::from(super::extension::arithmetic::characteristic_of(
        descriptor, at,
    ));
    let image = images[level_of(levels, count, at).expect("a restricted level owns an image")];
    let mut i = 0;
    while i < out.len() {
        let Some(dual) = dual_element(descriptor, at, image, levels, count, images, i) else {
            return false;
        };
        let mut trace = Elem::<T>::ZERO;
        let mut power = value * dual;
        let mut k = 0;
        let relative = out.len();
        while k < relative {
            trace += power;
            power = power.pow(prime);
            k += 1;
        }
        let Some(residue) = trace.to_residue() else {
            return false;
        };
        out[i] = residue;
        i += 1;
    }
    true
}

/// The `index`-th trace-dual basis element of the level at `at`, in the
/// target: `b_index / f'(image)` with `b` the synthetic quotient
/// coefficients of the transported monic relation. Returns `None` when the
/// transported data is inconsistent.
fn dual_element<T: Field>(
    descriptor: &Descriptor,
    at: u8,
    image: Elem<T>,
    levels: &[Level; 8],
    count: usize,
    images: &[Elem<T>],
    index: usize,
) -> Option<Elem<T>> {
    let relative = level_relative(descriptor, at);
    if index >= relative {
        return None;
    }
    // `b_index = Σ_{j>index} c_j image^{j-1-index}` with `c_relative = 1`,
    // by Horner descent from the leading term.
    let mut acc = Elem::<T>::ONE;
    let mut j = relative;
    while j > index + 1 {
        j -= 1;
        acc = acc * image + transport_coefficient(descriptor, at, levels, count, images, j);
    }
    let derivative = relation_derivative(descriptor, at, levels, count, images, image, relative);
    assert!(
        derivative != Elem::<T>::ZERO,
        "a transported relation is inseparable at its image"
    );
    Some(acc * derivative.inv())
}

/// The derivative of the transported monic relation at the image:
/// `Σ_{j≥1} j·c_j·image^{j-1}` with `c_relative = 1`.
fn relation_derivative<T: Field>(
    descriptor: &Descriptor,
    at: u8,
    levels: &[Level; 8],
    count: usize,
    images: &[Elem<T>],
    image: Elem<T>,
    relative: usize,
) -> Elem<T> {
    let mut acc = Elem::<T>::ZERO;
    let mut j = relative;
    while j > 1 {
        j -= 1;
        // Scalar multiplication by repeated addition, whatever the
        // characteristic.
        let coefficient = transport_coefficient(descriptor, at, levels, count, images, j);
        let mut scaled = Elem::<T>::ZERO;
        let mut k = 0;
        while k < j {
            scaled += coefficient;
            k += 1;
        }
        acc = acc * image + scaled;
    }
    // The leading term contributes `relative · image^{relative-1}`.
    let mut power = Elem::<T>::ONE;
    let mut k = 1;
    while k < relative {
        power *= image;
        k += 1;
    }
    let mut leading = Elem::<T>::ZERO;
    let mut k = 0;
    while k < relative {
        leading += power;
        k += 1;
    }
    acc + leading
}

/// The `index`-th transported non-leading coefficient of the level at
/// `at`, embedded in the target: one base element per coefficient for an
/// extension node, one polynomial bit for flat and basis levels.
fn transport_coefficient<T: Field>(
    descriptor: &Descriptor,
    at: u8,
    levels: &[Level; 8],
    count: usize,
    images: &[Elem<T>],
    index: usize,
) -> Elem<T> {
    match descriptor.node(at) {
        Node::Extension {
            base,
            base_degree,
            characteristic,
            base_words,
            coefficients,
            ..
        } => {
            if *characteristic == 2 {
                let mut chunk = [0u64; 128];
                let word = coefficients[index];
                let mut k = 0;
                while k < *base_degree as usize {
                    chunk[k] = (word >> k) & 1;
                    k += 1;
                }
                embed_coordinates(
                    descriptor,
                    *base,
                    &chunk[..*base_degree as usize],
                    levels,
                    count,
                    images,
                )
            } else {
                embed_coordinates(
                    descriptor,
                    *base,
                    &coefficients[index * *base_words..(index + 1) * *base_words],
                    levels,
                    count,
                    images,
                )
            }
        }
        Node::Polynomial { full, .. } => Elem::<T>::from_residue(((full >> index) & 1) as u64)
            .expect("a polynomial bit sits below the target"),
        Node::Basis { .. } => {
            let poly = basis_poly(descriptor, at);
            let Node::Polynomial { full, .. } = descriptor.node(poly) else {
                panic!("a basis level reads its polynomial node");
            };
            Elem::<T>::from_residue(((full >> index) & 1) as u64)
                .expect("a polynomial bit sits below the target")
        }
        Node::Unused | Node::Prime { .. } => {
            panic!("a prime node carries no transported relation");
        }
    }
}

/// The relative degree of the level at `at`: the extension degree, the
/// leaf degree, or the underlying polynomial degree.
fn level_relative(descriptor: &Descriptor, at: u8) -> usize {
    match descriptor.node(at) {
        Node::Extension {
            relative_degree, ..
        } => *relative_degree,
        Node::Polynomial { degree, .. } => usize::from(*degree),
        Node::Basis { .. } => {
            let poly = basis_poly(descriptor, at);
            let Node::Polynomial { degree, .. } = descriptor.node(poly) else {
                panic!("a basis level reads its polynomial node");
            };
            usize::from(*degree)
        }
        Node::Unused | Node::Prime { .. } => {
            panic!("a prime node carries no relative degree");
        }
    }
}

/// The underlying polynomial node of the basis leaf at `at`.
fn basis_poly(descriptor: &Descriptor, at: u8) -> u8 {
    let Node::Basis { base, .. } = descriptor.node(at) else {
        panic!("a basis level reads its own node");
    };
    *base
}

/// The order of the presentation rooted at `at`.
fn order_of(descriptor: &Descriptor, at: u8) -> u128 {
    super::extension::arithmetic::order_of(descriptor, at)
}

// ---------------------------------------------------------------------------
// Image sources
// ---------------------------------------------------------------------------

/// The structural images of the source's levels in the target, when the
/// source's descriptor matches the target's own or one of its tower bases:
/// each source level generator lifts through the upper tower levels as
/// constant components (converted through basis maps where the upper
/// level is a basis leaf). Returns `None` when no tower base matches.
fn structural_images<T: Field>(
    source_descriptor: &Descriptor,
    source_levels: &[Level; 8],
    source_count: usize,
    source_degree: usize,
    target_descriptor: &Descriptor,
    target_degree: usize,
) -> Option<([Elem<T>; 8], usize)> {
    // The upper tower nodes from the target root down to the match,
    // innermost last.
    let mut uppers = [0u8; 8];
    let mut upper_count = 0;
    let mut current = *target_descriptor;
    loop {
        if current.same_structure(source_descriptor) {
            break;
        }
        let root = current.root_index();
        match current.node(root) {
            Node::Extension { .. } | Node::Basis { .. } => {
                assert!(
                    upper_count < 8,
                    "a tower descent visits at most eight nodes"
                );
                uppers[upper_count] = root;
                upper_count += 1;
                current = current.base_descriptor();
            }
            _ => return None,
        }
    }
    let mut images = [Elem::<T>::ZERO; 8];
    let mut l = 0;
    while l < source_count {
        let mut coords = [0u64; 128];
        generator_coordinates(
            source_descriptor,
            &source_levels[l],
            source_degree,
            &mut coords,
        );
        let mut len = source_degree;
        // Lift innermost first, back up to the target root.
        let mut u = upper_count;
        while u > 0 {
            u -= 1;
            lift_coordinates(target_descriptor, uppers[u], &mut coords, &mut len);
        }
        assert_eq!(len, target_degree, "a structural lift spans the target");
        images[l] = Elem::<T>::from_raw(T::from_prime_coordinates(&coords[..target_degree]));
        l += 1;
    }
    Some((images, source_count))
}

/// Lift flattened coordinates one tower level outward: constant-component
/// padding through an extension level, the inverse basis map through a
/// basis leaf.
fn lift_coordinates(descriptor: &Descriptor, at: u8, coords: &mut [u64; 128], len: &mut usize) {
    match descriptor.node(at) {
        Node::Extension {
            base_degree,
            relative_degree,
            ..
        } => {
            assert_eq!(
                *len, *base_degree as usize,
                "a structural lift meets its base degree"
            );
            let mut i = *len;
            while i < *relative_degree * *base_degree as usize {
                coords[i] = 0;
                i += 1;
            }
            *len = *relative_degree * *base_degree as usize;
        }
        Node::Basis { from_base, .. } => {
            let packed = pack_bits(&coords[..*len]);
            let mut converted = [0u64; 128];
            unpack_bits(apply(*from_base, packed), &mut converted[..*len]);
            coords[..*len].copy_from_slice(&converted[..*len]);
        }
        Node::Unused | Node::Prime { .. } | Node::Polynomial { .. } => {
            panic!("a structural lift climbs a tower level");
        }
    }
}

// ---------------------------------------------------------------------------
// A prepared, explicitly selected field inclusion of `S` into `T`
// ---------------------------------------------------------------------------

/// A prepared, explicitly selected field inclusion of `S` into `T`.
///
/// The map is fixed by the image in `T` of each generator level of `S`,
/// prime field upward. Construction selects those images structurally,
/// through the frozen characteristic-two ladder, or from caller-supplied
/// values validated against the transported relations; every other method
/// evaluates them in the target's arithmetic. Equal presentation
/// descriptions embed identically, and inclusions commute as triangles.
///
/// The prepared state is fixed arrays; construction and every method are
/// allocation-free. Repeated application never reruns image construction.
pub struct Embedding<S: Field, T: Field<Characteristic = S::Characteristic>> {
    images: [Elem<T>; 8],
    count: u8,
    phantom: PhantomData<fn() -> (S, T)>,
}

// A derive would bound the impl on `S: Clone`; the manual impl needs only
// the field contract the type already carries.
#[allow(clippy::expl_impl_clone_on_copy)]
impl<S: Field, T: Field<Characteristic = S::Characteristic>> Clone for Embedding<S, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Field, T: Field<Characteristic = S::Characteristic>> Copy for Embedding<S, T> {}

impl<S: Field, T: Field<Characteristic = S::Characteristic>> core::fmt::Debug for Embedding<S, T> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "Embedding<{}, {}>", S::NAME, T::NAME)
    }
}

impl<S: Field, T: Field<Characteristic = S::Characteristic>> Embedding<S, T> {
    /// Prepare the canonical embedding of `S` into `T`.
    ///
    /// Returns, in order: [`EmbeddingError::IncompatibleDegree`] when
    /// `S::DEGREE` does not divide `T::DEGREE`; the structural inclusion
    /// when `S` is `T`, the prime subfield of `T`, or a row of `T`'s own
    /// tower; the canonical ladder embedding when both rows are
    /// characteristic-two presentations the ladder covers; and
    /// [`EmbeddingError::NoCanonicalMap`] otherwise. No root search runs
    /// outside the binary ladder's frozen rules.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::IncompatibleDegree`] on a degree mismatch
    /// and [`EmbeddingError::NoCanonicalMap`] when no map source covers
    /// the pair.
    ///
    /// # Panics
    ///
    /// Panics when a selected image fails its transported relation on an
    /// otherwise valid presentation. That is a crate defect, not an input
    /// condition.
    pub fn new() -> Result<Self, EmbeddingError> {
        let m = S::DEGREE;
        let n = T::DEGREE;
        if n % m != 0 {
            return Err(EmbeddingError::IncompatibleDegree {
                source: m,
                target: n,
            });
        }
        let source_descriptor = S::DESCRIPTOR;
        let target_descriptor = T::DESCRIPTOR;
        let (source_levels, source_count) = levels(source_descriptor);
        if m == 1 {
            // The prime subfield inclusion: degree-one sources hold no
            // levels, and the prime map is unique.
            return Ok(Self::checked(
                [Elem::<T>::ZERO; 8],
                0,
                source_descriptor,
                &source_levels,
                source_count,
            ));
        }
        if let Some((images, count)) = structural_images::<T>(
            source_descriptor,
            &source_levels,
            source_count,
            m as usize,
            target_descriptor,
            n as usize,
        ) {
            return Ok(Self::checked(
                images,
                count,
                source_descriptor,
                &source_levels,
                source_count,
            ));
        }
        let characteristic = <S::Characteristic as PrimeIdentity>::CHARACTERISTIC;
        if characteristic == 2
            && let (Some(source_binary), Some(target_binary)) =
                (S::BINARY_DESCRIPTION, T::BINARY_DESCRIPTION)
        {
            let mut words = [0u64; 8];
            let mut l = 0;
            while l < source_count {
                let mut coords = [0u64; 128];
                generator_coordinates(
                    source_descriptor,
                    &source_levels[l],
                    m as usize,
                    &mut coords,
                );
                // Generator words of a binary row fit the word.
                words[l] = u64::try_from(super::coordinates_index(&coords[..m as usize], 2))
                    .expect("a binary generator fits the word");
                l += 1;
            }
            if let Some(image_words) =
                ladder::canonical_images(source_binary, target_binary, &words[..source_count])
            {
                let mut images = [Elem::<T>::ZERO; 8];
                let mut l = 0;
                while l < source_count {
                    images[l] = Elem::<T>::from_index(u128::from(image_words[l]))
                        .expect("a ladder image fits the target");
                    l += 1;
                }
                return Ok(Self::checked(
                    images,
                    source_count,
                    source_descriptor,
                    &source_levels,
                    source_count,
                ));
            }
        }
        Err(EmbeddingError::NoCanonicalMap)
    }

    /// Prepare the embedding fixed by `images`: the image in `T` of each
    /// generator level of `S` in level order, validated against the
    /// transported relations.
    ///
    /// [`to_images`](Self::to_images) returns the stored images in the
    /// same order, so `from_images(e.to_images())` reproduces `e`.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::IncompatibleDegree`] on a degree mismatch
    /// first, then [`EmbeddingError::InvalidImages`] on a count mismatch
    /// or a failed relation.
    ///
    /// # Panics
    ///
    /// Panics when the source holds more than eight generator levels.
    /// Presentations are capped at eight descriptor nodes, so valid rows
    /// never reach this.
    pub fn from_images(images: &[Elem<T>]) -> Result<Self, EmbeddingError> {
        let m = S::DEGREE;
        let n = T::DEGREE;
        if n % m != 0 {
            return Err(EmbeddingError::IncompatibleDegree {
                source: m,
                target: n,
            });
        }
        let source_descriptor = S::DESCRIPTOR;
        let (source_levels, source_count) = levels(source_descriptor);
        if images.len() != source_count {
            return Err(EmbeddingError::InvalidImages);
        }
        let mut stored = [Elem::<T>::ZERO; 8];
        stored[..source_count].copy_from_slice(images);
        if !relations_hold(source_descriptor, &source_levels, source_count, &stored) {
            return Err(EmbeddingError::InvalidImages);
        }
        Ok(Self {
            images: stored,
            // Levels are capped at eight per presentation.
            count: u8::try_from(source_count).expect("a level count fits its byte"),
            phantom: PhantomData,
        })
    }

    /// Check selected images against their transported relations,
    /// panicking on failure: construction-time validation for maps the
    /// sources guarantee.
    fn checked(
        images: [Elem<T>; 8],
        count: usize,
        descriptor: &Descriptor,
        levels: &[Level; 8],
        level_count: usize,
    ) -> Self {
        assert_eq!(count, level_count, "a constructed map covers every level");
        assert!(
            relations_hold(descriptor, levels, level_count, &images),
            "a constructed image fails its transported relation"
        );
        Self {
            images,
            // Levels are capped at eight per presentation.
            count: u8::try_from(count).expect("a level count fits its byte"),
            phantom: PhantomData,
        }
    }

    /// The stored images in level order, so `from_images(e.to_images())`
    /// reproduces `e`.
    #[must_use]
    pub fn to_images(&self) -> &[Elem<T>] {
        &self.images[..self.count as usize]
    }

    /// The number of generator levels of `S`: zero at absolute degree one.
    #[must_use]
    pub fn generator_count() -> usize {
        levels(S::DESCRIPTOR).1
    }

    /// The generator of the `level`-th level of `S`, prime field upward.
    ///
    /// # Panics
    ///
    /// Panics when `level` reaches [`generator_count`](Self::generator_count).
    #[must_use]
    pub fn generator(level: usize) -> Elem<S> {
        let descriptor = S::DESCRIPTOR;
        let (source_levels, source_count) = levels(descriptor);
        assert!(
            level < source_count,
            "a generator level sits below the generator count"
        );
        let mut coords = [0u64; 128];
        generator_coordinates(
            descriptor,
            &source_levels[level],
            S::DEGREE as usize,
            &mut coords,
        );
        Elem::<S>::from_raw(S::from_prime_coordinates(&coords[..S::DEGREE as usize]))
    }

    /// Compose with the next embedding: `self.then(next)` applies `self`,
    /// then `next`, by image transport.
    #[must_use]
    pub fn then<U: Field<Characteristic = S::Characteristic>>(
        &self,
        next: &Embedding<T, U>,
    ) -> Embedding<S, U> {
        let mut images = [Elem::<U>::ZERO; 8];
        let mut l = 0;
        while l < self.count as usize {
            images[l] = next.embed(self.images[l]);
            l += 1;
        }
        Embedding {
            images,
            count: self.count,
            phantom: PhantomData,
        }
    }

    /// The relative degree `T::DEGREE / S::DEGREE`.
    #[must_use]
    pub fn relative_degree() -> u32 {
        T::DEGREE / S::DEGREE
    }

    /// Map `value` into the target field by Horner evaluation in the
    /// target's arithmetic, recursing into the base.
    #[must_use]
    pub fn embed(&self, value: Elem<S>) -> Elem<T> {
        let descriptor = S::DESCRIPTOR;
        let (source_levels, source_count) = levels(descriptor);
        let mut coords = [0u64; 128];
        S::prime_coordinates_into(&mut coords[..S::DEGREE as usize], value.to_raw());
        embed_coordinates(
            descriptor,
            descriptor.root_index(),
            &coords[..S::DEGREE as usize],
            &source_levels,
            source_count,
            &self.images,
        )
    }

    /// Whether `value` lies in the image of this embedding: the fixed
    /// field of the relative Frobenius.
    #[must_use]
    pub fn contains(&self, value: Elem<T>) -> bool {
        value.pow(S::ORDER) == value
    }

    /// The inverse of [`embed`](Self::embed) on the image, or `None`.
    ///
    /// Components recover level by level through the trace-dual basis,
    /// ending in the prime-subfield readout; a value outside the image
    /// restricts to `None` without solving a linear system.
    #[must_use]
    pub fn restrict(&self, value: Elem<T>) -> Option<Elem<S>> {
        let descriptor = S::DESCRIPTOR;
        let (source_levels, source_count) = levels(descriptor);
        let mut coords = [0u64; 128];
        if !restrict_coordinates(
            descriptor,
            descriptor.root_index(),
            value,
            &source_levels,
            source_count,
            &self.images,
            &mut coords[..S::DEGREE as usize],
        ) {
            return None;
        }
        Some(Elem::<S>::from_raw(S::from_prime_coordinates(
            &coords[..S::DEGREE as usize],
        )))
    }

    /// The relative Frobenius `x^|S|` of the extension.
    #[must_use]
    pub fn frobenius(&self, value: Elem<T>) -> Elem<T> {
        value.pow(S::ORDER)
    }

    /// The relative trace: the sum of the conjugates under the relative
    /// Frobenius, restricted to the source.
    ///
    /// Equal degrees reduce to restriction. `Embedding<FiniteField<2, 1, Polynomial<3>>, T>` supplies
    /// the absolute trace.
    ///
    /// # Panics
    ///
    /// Panics only on a crate defect: a conjugate sum outside the
    /// constructed embedding image.
    #[must_use]
    pub fn trace(&self, value: Elem<T>) -> Elem<S> {
        let mut sum = value;
        let mut power = value;
        let mut k = 1;
        while k < T::DEGREE / S::DEGREE {
            power = self.frobenius(power);
            sum += power;
            k += 1;
        }
        self.restrict(sum)
            .expect("a conjugate sum lies in the embedding image")
    }

    /// The relative norm: the product of the conjugates under the relative
    /// Frobenius, restricted to the source.
    ///
    /// Equal degrees reduce to restriction. `Embedding<FiniteField<2, 1, Polynomial<3>>, T>` supplies
    /// the absolute norm.
    ///
    /// # Panics
    ///
    /// Panics only on a crate defect: a conjugate product outside the
    /// constructed embedding image.
    #[must_use]
    pub fn norm(&self, value: Elem<T>) -> Elem<S> {
        let mut product = value;
        let mut power = value;
        let mut k = 1;
        while k < T::DEGREE / S::DEGREE {
            power = self.frobenius(power);
            product *= power;
            k += 1;
        }
        self.restrict(product)
            .expect("a conjugate product lies in the embedding image")
    }
}

// ---------------------------------------------------------------------------
// A prepared field isomorphism of `S` onto `T`
// ---------------------------------------------------------------------------

/// A prepared field isomorphism between rows of equal characteristic and
/// equal degree, with a total inverse.
///
/// An [`Embedding`] of equal degrees and the `Isomorphism` built from it
/// denote the same map and agree on every element. The prepared state adds
/// the reverse images to the forward ones; composition and inversion are
/// infallible and agree with sequential application.
pub struct Isomorphism<S: Field, T: Field<Characteristic = S::Characteristic>> {
    forward: [Elem<T>; 8],
    reverse: [Elem<S>; 8],
    count: u8,
    phantom: PhantomData<fn() -> (S, T)>,
}

// A derive would bound the impl on `S: Clone`; the manual impl needs only
// the field contract the type already carries.
#[allow(clippy::expl_impl_clone_on_copy)]
impl<S: Field, T: Field<Characteristic = S::Characteristic>> Clone for Isomorphism<S, T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<S: Field, T: Field<Characteristic = S::Characteristic>> Copy for Isomorphism<S, T> {}

impl<S: Field, T: Field<Characteristic = S::Characteristic>> core::fmt::Debug
    for Isomorphism<S, T>
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "Isomorphism<{}, {}>", S::NAME, T::NAME)
    }
}

impl<S: Field, T: Field<Characteristic = S::Characteristic>> Isomorphism<S, T> {
    /// Prepare the canonical isomorphism of `S` onto `T`.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::IncompatibleDegree`] unless the degrees
    /// are equal, before any other check, and
    /// [`EmbeddingError::NoCanonicalMap`] when no map source covers the
    /// pair.
    pub fn new() -> Result<Self, EmbeddingError> {
        if S::DEGREE != T::DEGREE {
            return Err(EmbeddingError::IncompatibleDegree {
                source: S::DEGREE,
                target: T::DEGREE,
            });
        }
        Self::from_embedding(Embedding::new()?)
    }

    /// Prepare the isomorphism fixed by `images`, in the order
    /// [`from_images`](Embedding::from_images) accepts.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::IncompatibleDegree`] on unequal degrees
    /// first, then [`EmbeddingError::InvalidImages`] on a count mismatch
    /// or a failed relation.
    pub fn from_images(images: &[Elem<T>]) -> Result<Self, EmbeddingError> {
        if S::DEGREE != T::DEGREE {
            return Err(EmbeddingError::IncompatibleDegree {
                source: S::DEGREE,
                target: T::DEGREE,
            });
        }
        Self::from_embedding(Embedding::from_images(images)?)
    }

    /// Prepare the isomorphism denoted by `embedding`: the reverse images
    /// restrict the target's generators once.
    ///
    /// # Errors
    ///
    /// Returns [`EmbeddingError::IncompatibleDegree`] on unequal degrees.
    ///
    /// # Panics
    ///
    /// Panics only on a crate defect: a target generator outside the
    /// constructed embedding image.
    pub fn from_embedding(embedding: Embedding<S, T>) -> Result<Self, EmbeddingError> {
        if S::DEGREE != T::DEGREE {
            return Err(EmbeddingError::IncompatibleDegree {
                source: S::DEGREE,
                target: T::DEGREE,
            });
        }
        let descriptor = T::DESCRIPTOR;
        let (target_levels, target_count) = levels(descriptor);
        let mut reverse = [Elem::<S>::ZERO; 8];
        let mut l = 0;
        while l < target_count {
            let mut coords = [0u64; 128];
            generator_coordinates(
                descriptor,
                &target_levels[l],
                T::DEGREE as usize,
                &mut coords,
            );
            let generator =
                Elem::<T>::from_raw(T::from_prime_coordinates(&coords[..T::DEGREE as usize]));
            reverse[l] = embedding
                .restrict(generator)
                .expect("a target generator lies in the embedding image");
            l += 1;
        }
        Ok(Self {
            forward: embedding.images,
            reverse,
            count: embedding.count,
            phantom: PhantomData,
        })
    }

    /// The forward embedding this isomorphism denotes.
    #[must_use]
    pub fn to_embedding(&self) -> Embedding<S, T> {
        Embedding {
            images: self.forward,
            count: self.count,
            phantom: PhantomData,
        }
    }

    /// The forward images in the order `from_images` accepts.
    #[must_use]
    pub fn to_images(&self) -> &[Elem<T>] {
        &self.forward[..self.count as usize]
    }

    /// Compose with the next isomorphism by image transport.
    #[must_use]
    pub fn then<U: Field<Characteristic = S::Characteristic>>(
        &self,
        next: &Isomorphism<T, U>,
    ) -> Isomorphism<S, U> {
        // The reverse composite runs the reverses in reverse order.
        let forward = self.to_embedding().then(&next.to_embedding());
        let reverse = next
            .inverse()
            .to_embedding()
            .then(&self.inverse().to_embedding());
        Isomorphism {
            forward: forward.images,
            reverse: reverse.images,
            count: forward.count,
            phantom: PhantomData,
        }
    }

    /// The inverse isomorphism: the two image sets swapped.
    #[must_use]
    pub fn inverse(&self) -> Isomorphism<T, S> {
        Isomorphism {
            forward: self.reverse,
            reverse: self.forward,
            count: self.count,
            phantom: PhantomData,
        }
    }

    /// Map `value` into the target field; equals the denoted embedding.
    #[must_use]
    pub fn apply(&self, value: Elem<S>) -> Elem<T> {
        self.to_embedding().embed(value)
    }

    /// Map `value` back into the source field: the total inverse.
    #[must_use]
    pub fn unapply(&self, value: Elem<T>) -> Elem<S> {
        let descriptor = T::DESCRIPTOR;
        let (target_levels, target_count) = levels(descriptor);
        let mut coords = [0u64; 128];
        T::prime_coordinates_into(&mut coords[..T::DEGREE as usize], value.to_raw());
        embed_coordinates(
            descriptor,
            descriptor.root_index(),
            &coords[..T::DEGREE as usize],
            &target_levels,
            target_count,
            &self.reverse,
        )
    }
}
