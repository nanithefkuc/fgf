//! Opaque binary-field descriptors over the shared descriptor body.
//!
//! A [`BinaryDescription`] wraps the general field descriptor with its
//! nodes restricted to the binary leaves and quadratic relation nodes. The
//! public sealed traits expose it only as an opaque contract; construction
//! stays inside the representation modules, and the element interpreter
//! lives with the general descriptor in `crate::field::extension`.

use crate::field::extension::arithmetic;
use crate::field::extension::irreducibility;
use crate::field::extension::{Descriptor, Node};

/// Opaque scalar description of a binary-field presentation.
///
/// No public construction: representations publish their own validated
/// constants, and every value construction forces that validation.
#[derive(Clone, Copy)]
pub struct BinaryDescription {
    inner: Descriptor,
}

impl BinaryDescription {
    /// The shared descriptor this binary description wraps.
    #[must_use]
    pub(crate) const fn as_descriptor(&self) -> &Descriptor {
        &self.inner
    }

    /// Defining data of the root node.
    #[must_use]
    pub(crate) const fn root_data(&self) -> NodeData {
        match self.inner.node(self.inner.root_index()) {
            Node::Unused | Node::Prime { .. } => panic!("unused node"),
            Node::Polynomial { degree, full } => NodeData::Polynomial {
                degree: *degree,
                full: *full,
            },
            Node::Basis {
                to_base, from_base, ..
            } => NodeData::Basis {
                to_base: *to_base,
                from_base: *from_base,
            },
            Node::Extension {
                relative_degree: 2,
                coefficients,
                ..
            } => NodeData::Quadratic {
                a: coefficients[1],
                b: coefficients[0],
            },
            Node::Extension { .. } => panic!("the root node holds no relation"),
        }
    }

    /// The sub-description rooted at the root node's base.
    #[must_use]
    pub(crate) const fn base_description(&self) -> BinaryDescription {
        Self {
            inner: self.inner.base_descriptor(),
        }
    }

    /// A lone polynomial root of the stated degree.
    #[must_use]
    pub(crate) const fn polynomial(degree: u8, full: u128) -> Self {
        Self {
            inner: Descriptor::polynomial(degree, full),
        }
    }

    /// Append an ordered-basis wrapper over `base`'s root.
    #[must_use]
    pub(crate) const fn append_basis(
        base: &BinaryDescription,
        to_base: [u8; 8],
        from_base: [u8; 8],
    ) -> Self {
        Self {
            inner: base.inner.append_basis(to_base, from_base),
        }
    }

    /// Append a quadratic extension over `base`'s root.
    #[must_use]
    pub(crate) const fn append_quadratic(base: &BinaryDescription, a: u64, b: u64) -> Self {
        Self {
            inner: base.inner.append_quadratic_words(a, b),
        }
    }

    /// Total degree of the root presentation.
    #[must_use]
    pub const fn degree(&self) -> u32 {
        self.inner.degree()
    }

    /// Whether the root node is a flat polynomial presentation.
    #[must_use]
    pub const fn is_polynomial_root(&self) -> bool {
        matches!(
            self.inner.node(self.inner.root_index()),
            Node::Polynomial { .. }
        )
    }

    /// Whether the root node is an ordered-basis presentation.
    #[must_use]
    pub const fn is_basis_root(&self) -> bool {
        matches!(self.inner.node(self.inner.root_index()), Node::Basis { .. })
    }

    /// The multiplicative identity in root coordinates.
    #[must_use]
    pub const fn one(&self) -> u64 {
        arithmetic::word_one(&self.inner, self.inner.root_index())
    }

    /// Field multiplication in root coordinates.
    #[must_use]
    pub const fn mul(&self, x: u64, y: u64) -> u64 {
        arithmetic::word_mul(&self.inner, self.inner.root_index(), x, y)
    }

    /// Field exponentiation in root coordinates.
    #[must_use]
    pub const fn pow(&self, x: u64, e: u128) -> u64 {
        arithmetic::word_pow(&self.inner, self.inner.root_index(), x, e)
    }

    /// Multiplicative inverse in root coordinates, mapping zero to zero.
    ///
    /// The arithmetic methods take canonical coordinate words, with no bits
    /// at or above `degree()`. On any other word `inv` stays total: a
    /// polynomial root inverts the low `degree()` bits of the word, and a
    /// word whose low bits are zero inverts to zero.
    #[must_use]
    pub const fn inv(&self, x: u64) -> u64 {
        arithmetic::word_inv(&self.inner, self.inner.root_index(), x)
    }

    /// Absolute trace to GF(2) in root coordinates.
    #[must_use]
    pub const fn trace(&self, x: u64) -> u64 {
        arithmetic::word_trace(&self.inner, self.inner.root_index(), x)
    }

    /// Use-time validation of the whole description.
    pub(crate) const fn validate(&self) {
        let descriptor = &self.inner;
        assert!(
            descriptor.node_count() > 0
                && descriptor.node_count() <= 8
                && descriptor.root_index() < descriptor.node_count(),
            "bad description shape"
        );
        let mut i = 0u8;
        while i < 8 {
            let used = i < descriptor.node_count();
            match descriptor.node(i) {
                Node::Prime { .. } => {
                    assert!(!used, "non-binary node inside a binary description");
                }
                Node::Unused => assert!(!used, "unused node inside description"),
                Node::Polynomial { degree, full } => {
                    assert!(used, "node outside description");
                    assert!(
                        *degree == 1 || *degree == 2 || *degree == 4 || *degree == 8,
                        "unsupported flat degree"
                    );
                    assert!(
                        poly_irreducible(*full, *degree as u32),
                        "polynomial is reducible"
                    );
                }
                Node::Basis { .. } => self.validate_basis_node(i, used),
                Node::Extension {
                    base,
                    relative_degree,
                    base_degree,
                    coefficients,
                    ..
                } => {
                    assert!(used && *base < i, "quadratic node order");
                    let h = arithmetic::degree_of(descriptor, *base);
                    assert!(h <= 32, "binary degree exceeds 64");
                    let (a, b) = (coefficients[1], coefficients[0]);
                    if h < 64 {
                        assert!(a >> h == 0, "tower coefficient has excess bits");
                        assert!(b >> h == 0, "tower coefficient has excess bits");
                    }
                    assert!(a != 0, "tower coefficient A is zero");
                    assert!(*relative_degree == 2, "non-quadratic binary node");
                    assert!(
                        h == *base_degree,
                        "quadratic node caches a wrong base degree"
                    );
                    assert!(
                        irreducibility::quadratic_relation_holds(descriptor, i),
                        "tower relation is reducible"
                    );
                }
            }
            i += 1;
        }
        assert!(self.degree() <= 64, "binary degree exceeds 64");
    }

    /// Validate the ordered-basis node at `index` of this descriptor:
    /// degree, column widths, padding, and the bounded-enumeration
    /// bijection.
    // The single-character locals mirror the node fields they check.
    #[allow(clippy::many_single_char_names)]
    const fn validate_basis_node(&self, index: u8, used: bool) {
        let descriptor = &self.inner;
        let (base, to_base, from_base) = match descriptor.node(index) {
            Node::Basis {
                base,
                to_base,
                from_base,
            } => (*base, *to_base, *from_base),
            _ => panic!("not a basis node"),
        };
        assert!(used && base < index, "basis node order");
        let n = arithmetic::degree_of(descriptor, base);
        assert!(n == 2 || n == 4 || n == 8, "unsupported basis degree");
        let mask = (1u64 << n) - 1;
        let mut k = 0usize;
        while k < 8 {
            // The counter never exceeds the column arrays.
            #[allow(clippy::cast_possible_truncation)]
            let column = k as u32;
            if column < n {
                assert!((to_base[k] as u64) <= mask, "basis vector has excess bits");
                assert!(
                    (from_base[k] as u64) <= mask,
                    "inverse column has excess bits"
                );
            } else {
                assert!(to_base[k] == 0, "basis padding must be zero");
                assert!(from_base[k] == 0, "inverse padding must be zero");
            }
            k += 1;
        }
        // Bounded-enumeration bijection: the stored inverse inverts the
        // forward map on every coordinate word.
        let count = 1u64 << n;
        let mut x = 0;
        while x < count {
            let forth = apply(to_base, x);
            let back = apply(from_base, forth);
            assert!(back == x, "basis map is not invertible");
            x += 1;
        }
        let mut y = 0;
        while y < count {
            let forth = apply(from_base, y);
            let back = apply(to_base, forth);
            assert!(back == y, "inverse map is not invertible");
            y += 1;
        }
    }

    /// Structural equality ignoring unused slots.
    ///
    /// Strategy selection compares complete descriptions through this
    /// comparator; hashes alone do not prove arithmetic equality.
    #[must_use]
    pub const fn same_structure(&self, other: &BinaryDescription) -> bool {
        self.inner.same_structure(&other.inner)
    }

    /// Whether `candidate` generates the multiplicative group.
    ///
    /// Checks the full-order power plus every cofactor quotient against the
    /// complete prime-factor list of `2^N - 1` at each supported degree.
    ///
    /// # Panics
    ///
    /// Panics in `const` evaluation when the descriptor's degree is not one
    /// of the supported binary degrees.
    #[must_use]
    pub const fn is_generator(&self, candidate: u64) -> bool {
        let n = self.degree();
        let factors: &[u64] = match n {
            1 => &[],
            2 => &[3],
            4 => &[3, 5],
            8 => &[3, 5, 17],
            16 => &[3, 5, 17, 257],
            32 => &[3, 5, 17, 257, 65_537],
            64 => &[3, 5, 17, 257, 641, 65_537, 6_700_417],
            _ => panic!("unsupported degree"),
        };
        let order: u128 = (1u128 << n) - 1;
        if candidate == 0 || self.pow(candidate, order) != self.one() {
            return false;
        }
        let mut i = 0;
        while i < factors.len() {
            if self.pow(candidate, order / factors[i] as u128) == self.one() {
                return false;
            }
            i += 1;
        }
        true
    }
}

/// Opaque discrete-logarithm tables for one degree-eight presentation.
///
/// Built from the presentation's description under its selected generator:
/// `EXP[i]` is the generator to the `i`, and `LOG` inverts it with
/// `LOG[0]` undefined. Only valid degree-eight coordinates carry one.
#[derive(Clone, Copy)]
pub struct ByteLogExp {
    exp: [u8; 255],
    log: [u8; 256],
}

impl ByteLogExp {
    /// Build the log/exp pair for `description` under `generator`.
    #[must_use]
    // The description's coordinates are one byte by construction; the loop
    // counters never exceed their table sizes.
    #[allow(clippy::cast_possible_truncation)]
    pub const fn build(description: &BinaryDescription, generator: u64) -> Self {
        let mut exp = [0u8; 255];
        let mut value = description.one();
        let mut i = 0;
        while i < 255 {
            exp[i] = value as u8;
            value = description.mul(value, generator);
            i += 1;
        }
        let mut log = [0u8; 256];
        let mut j = 0;
        while j < 255 {
            log[exp[j] as usize] = j as u8;
            j += 1;
        }
        Self { exp, log }
    }

    /// Multiply through the tables, with zero short-circuiting.
    #[inline]
    #[must_use]
    pub(crate) const fn mul(&self, left: u8, right: u8) -> u8 {
        if left == 0 || right == 0 {
            return 0;
        }
        let la = self.log[left as usize] as usize;
        let lb = self.log[right as usize] as usize;
        self.exp[(la + lb) % 255]
    }

    /// Multiplicative inverse through the tables, mapping zero to zero.
    #[inline]
    #[must_use]
    pub(crate) const fn inv(&self, value: u8) -> u8 {
        if value == 0 {
            return 0;
        }
        let l = self.log[value as usize] as usize;
        self.exp[(255 - l) % 255]
    }
}

/// Degree of the polynomial `p`, i.e. the index of its highest set bit.
pub(crate) const fn poly_degree(p: u128) -> u32 {
    p.ilog2()
}

/// Whether `p` of degree `n` is irreducible over GF(2).
///
/// Trial division by every polynomial of degree `1..=n/2`.
pub(crate) const fn poly_irreducible(p: u128, n: u32) -> bool {
    if poly_degree(p) != n {
        return false;
    }
    if n <= 1 {
        return true;
    }
    let mut q: u128 = 2;
    while poly_degree(q) <= n / 2 {
        let mut r = p;
        let dq = poly_degree(q);
        while r != 0 && poly_degree(r) >= dq {
            r ^= q << (poly_degree(r) - dq);
        }
        if r == 0 {
            return false;
        }
        q += 1;
    }
    true
}

/// Apply a byte-linear map: bit `i` of `x` selects `columns[i]`.
pub(crate) const fn apply(columns: [u8; 8], x: u64) -> u64 {
    let mut result = 0u64;
    let mut i = 0;
    while i < 8 {
        if (x >> i) & 1 == 1 {
            result ^= columns[i] as u64;
        }
        i += 1;
    }
    result
}

/// Bit-serial polynomial product, independent of every table.
///
/// The table builders below run through this path so that constructing the
/// pinned log/exp leaves never reads the leaves themselves.
pub(crate) const fn poly_mul_serial(full: u128, n: u32, x: u64, y: u64) -> u64 {
    let mut acc: u128 = 0;
    let mut k = 0;
    while k < n {
        if (y >> k) & 1 == 1 {
            acc ^= (x as u128) << k;
        }
        k += 1;
    }
    let mut k = 2 * n;
    while k > n {
        k -= 1;
        if (acc >> k) & 1 == 1 {
            acc ^= full << (k - n);
        }
    }
    // The reduction leaves fewer than `n <= 8` bits set.
    #[allow(clippy::cast_possible_truncation)]
    let reduced = acc as u64;
    reduced
}

/// The pinned AES log/exp leaf, built serially under the derived generator.
pub(crate) const PINNED_AES_LEAF: ByteLogExp = serial_log_exp(0x11B, 3);

/// The pinned Reed-Solomon log/exp leaf, built serially.
pub(crate) const PINNED_RS_LEAF: ByteLogExp = serial_log_exp(0x11D, 2);

/// Build one pinned leaf's log/exp pair without consulting any leaf table.
const fn serial_log_exp(full: u128, generator: u64) -> ByteLogExp {
    // Loop counters stay below the table sizes; the byte tower closes over
    // one byte.
    #[allow(clippy::cast_possible_truncation)]
    const fn build(full: u128, generator: u64) -> ByteLogExp {
        let mut exp = [0u8; 255];
        let mut value = 1u64;
        let mut i = 0;
        while i < 255 {
            // The degree-eight leaf holds one byte by construction.
            let byte: u8 = value as u8;
            exp[i] = byte;
            value = poly_mul_serial(full, 8, value, generator);
            i += 1;
        }
        let mut log = [0u8; 256];
        let mut j = 0;
        while j < 255 {
            log[exp[j] as usize] = j as u8;
            j += 1;
        }
        ByteLogExp { exp, log }
    }
    build(full, generator)
}

/// Crate-visible defining data of one description node.
#[derive(Clone, Copy)]
pub(crate) enum NodeData {
    /// A flat polynomial root.
    Polynomial {
        /// Total degree of the node.
        degree: u8,
        /// The full reduction polynomial including the leading term.
        full: u128,
    },
    /// An ordered basis over an earlier node.
    Basis {
        /// Basis vectors in base coordinates.
        to_base: [u8; 8],
        /// Inverse columns in base coordinates.
        from_base: [u8; 8],
    },
    /// A quadratic extension of an earlier node.
    Quadratic {
        /// Linear coefficient as a base-coordinate word.
        a: u64,
        /// Constant coefficient as a base-coordinate word.
        b: u64,
    },
}

#[cfg(test)]
mod tests {
    use core::hint::black_box;

    use super::*;
    use crate::field::FiniteField;
    use crate::field::binary::normal::inverse_columns;
    use crate::field::binary::{BinaryField, Cantor, Normal};
    use crate::field::extension::descriptor::{COORDINATES, NODES, Storage};
    use crate::field::{Elem, HasGenerator};
    use crate::{
        FanPaar2, FanPaar4, FanPaar8, FanPaar16, FanPaar32, FanPaar64, Polynomial, Rijndael16,
        Rijndael32, Rijndael64, Tower,
    };

    /// A quadratic relation node with packed coefficient words, for tests
    /// that place malformed nodes by hand.
    fn quadratic_words_node(base: u8, linear: u64, constant: u64) -> Node {
        let mut coefficients = [0u64; COORDINATES];
        coefficients[0] = constant;
        coefficients[1] = linear;
        Node::Extension {
            base,
            relative_degree: 2,
            characteristic: 2,
            base_degree: 1,
            base_words: 1,
            coefficients,
            storage: Storage::Packed { width: 8 },
        }
    }

    /// Deterministic 64-bit words from the crate's shared LCG constants.
    fn words(seed: u64) -> impl Iterator<Item = u64> {
        let mut state = seed | 1;
        core::iter::repeat_with(move || {
            let mut word = 0;
            for _ in 0..2 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                word = (word << 32) | (state >> 32);
            }
            word
        })
    }

    fn mask(n: u32) -> u64 {
        if n >= 64 { u64::MAX } else { (1 << n) - 1 }
    }

    /// Horner-order shift-and-add product modulo `full`, reducing after
    /// every doubling rather than once after a full carry-less product.
    fn xtime_mul(full: u128, n: u32, x: u64, y: u64) -> u64 {
        let mut acc: u128 = 0;
        for bit in (0..n).rev() {
            acc <<= 1;
            if (acc >> n) & 1 == 1 {
                acc ^= full;
            }
            if (y >> bit) & 1 == 1 {
                acc ^= u128::from(x);
            }
        }
        u64::try_from(acc).unwrap()
    }

    /// Carry-less polynomial product without reduction.
    fn clmul(x: u128, y: u128) -> u128 {
        (0..64)
            .filter(|bit| (y >> bit) & 1 == 1)
            .fold(0, |acc, bit| acc ^ (x << bit))
    }

    /// Coordinates of `x` under the basis vectors `columns`.
    fn combine(columns: [u8; 8], x: u64) -> u64 {
        (0..8)
            .filter(|bit| (x >> bit) & 1 == 1)
            .fold(0, |acc, bit| acc ^ u64::from(columns[bit]))
    }

    /// The coordinates `w` with `combine(columns, w) == target`, by Gaussian
    /// elimination rather than the stored inverse columns.
    fn solve(columns: [u8; 8], target: u64) -> u64 {
        // Echelon rows keyed by leading bit: (vector, combination).
        let mut echelon = [(0u64, 0u64); 8];
        let lead = |v: u64| usize::try_from(v.ilog2()).unwrap();
        for (i, &column) in columns.iter().enumerate() {
            let (mut v, mut c) = (u64::from(column), 1u64 << i);
            while v != 0 {
                let (ev, ec) = echelon[lead(v)];
                if ev == 0 {
                    echelon[lead(v)] = (v, c);
                    break;
                }
                (v, c) = (v ^ ev, c ^ ec);
            }
        }
        let (mut v, mut c) = (target, 0);
        while v != 0 {
            let (ev, ec) = echelon[lead(v)];
            assert_ne!(ev, 0, "target outside the basis span");
            (v, c) = (v ^ ev, c ^ ec);
        }
        c
    }

    /// Independent product at the root: Horner leaves, basis transport by
    /// elimination (never the stored inverse columns), and schoolbook
    /// quadratic extensions under `Y^2 = aY + b`.
    fn root_mul(desc: &BinaryDescription, x: u64, y: u64) -> u64 {
        match desc.root_data() {
            NodeData::Polynomial { degree, full } => xtime_mul(full, u32::from(degree), x, y),
            NodeData::Basis { to_base, .. } => {
                let base = desc.base_description();
                solve(
                    to_base,
                    root_mul(&base, combine(to_base, x), combine(to_base, y)),
                )
            }
            NodeData::Quadratic { a, b } => {
                let base = desc.base_description();
                let h = base.degree();
                let (x0, x1, y0, y1) = (x & mask(h), x >> h, y & mask(h), y >> h);
                let m = |u, v| root_mul(&base, u, v);
                let top = m(x1, y1);
                let lo = m(x0, y0) ^ m(b, top);
                let hi = m(x0, y1) ^ m(x1, y0) ^ m(a, top);
                lo | (hi << h)
            }
        }
    }

    /// Sum of the `n` Frobenius conjugates under the reference product.
    fn reference_trace(desc: &BinaryDescription, x: u64) -> u64 {
        let mut power = x;
        let mut acc = 0;
        for _ in 0..desc.degree() {
            acc ^= power;
            power = root_mul(desc, power, power);
        }
        acc
    }

    /// A unit-triangular, hence invertible, basis of degree `n`.
    fn triangular(n: u8) -> [u8; 8] {
        let mut columns = [0u8; 8];
        for (i, column) in columns.iter_mut().enumerate().take(usize::from(n)) {
            *column = (1u8 << i) | (0x55 & ((1u8 << i) - 1));
        }
        columns
    }

    fn triangular_over(base: &BinaryDescription) -> BinaryDescription {
        let n = u8::try_from(base.degree()).unwrap();
        BinaryDescription::append_basis(base, triangular(n), inverse_columns(triangular(n), n))
    }

    fn desc_of<F: BinaryField>() -> BinaryDescription {
        *F::DESCRIPTION
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    enum Kind {
        Polynomial,
        Basis,
        Quadratic,
    }

    /// Every validated presentation of degree at most eight under test,
    /// with its expected degree and root kind.
    // One descriptor occupies the shared node array's width.
    #[allow(clippy::large_stack_arrays)]
    fn small_fixtures() -> [(BinaryDescription, u32, Kind); 18] {
        let poly = BinaryDescription::polynomial;
        [
            (poly(1, 0x3), 1, Kind::Polynomial),
            (poly(2, 0x7), 2, Kind::Polynomial),
            (poly(4, 0x13), 4, Kind::Polynomial),
            (poly(4, 0x19), 4, Kind::Polynomial),
            (poly(4, 0x1F), 4, Kind::Polynomial),
            (poly(8, 0x11B), 8, Kind::Polynomial),
            (poly(8, 0x11D), 8, Kind::Polynomial),
            (poly(8, 0x12D), 8, Kind::Polynomial),
            (triangular_over(&poly(2, 0x7)), 2, Kind::Basis),
            (triangular_over(&poly(4, 0x19)), 4, Kind::Basis),
            (triangular_over(&poly(8, 0x11D)), 8, Kind::Basis),
            (triangular_over(&poly(8, 0x12D)), 8, Kind::Basis),
            (
                desc_of::<FiniteField<2, 8, Normal<0x11B, 0x20>>>(),
                8,
                Kind::Basis,
            ),
            (
                desc_of::<FiniteField<2, 8, Cantor<0x11B, 0x20>>>(),
                8,
                Kind::Basis,
            ),
            (
                desc_of::<FiniteField<2, 2, Tower<FanPaar2>>>(),
                2,
                Kind::Quadratic,
            ),
            (
                desc_of::<FiniteField<2, 4, Tower<FanPaar4>>>(),
                4,
                Kind::Quadratic,
            ),
            (
                desc_of::<FiniteField<2, 8, Tower<FanPaar8>>>(),
                8,
                Kind::Quadratic,
            ),
            (
                triangular_over(&desc_of::<FiniteField<2, 8, Tower<FanPaar8>>>()),
                8,
                Kind::Basis,
            ),
        ]
    }

    /// The pinned GF(2^16), GF(2^32), and GF(2^64) towers.
    // One descriptor occupies the shared node array's width.
    #[allow(clippy::large_stack_arrays)]
    fn tower_fixtures() -> [BinaryDescription; 6] {
        [
            desc_of::<FiniteField<2, 16, Tower<Rijndael16>>>(),
            desc_of::<FiniteField<2, 32, Tower<Rijndael32>>>(),
            desc_of::<FiniteField<2, 64, Tower<Rijndael64>>>(),
            desc_of::<FiniteField<2, 16, Tower<FanPaar16>>>(),
            desc_of::<FiniteField<2, 32, Tower<FanPaar32>>>(),
            desc_of::<FiniteField<2, 64, Tower<FanPaar64>>>(),
        ]
    }

    /// Multiplicative order of a nonzero `x` by repeated reference products.
    fn reference_order(desc: &BinaryDescription, x: u64) -> u64 {
        let one = desc.one();
        let mut power = x;
        let mut order = 1;
        while power != one {
            power = root_mul(desc, power, x);
            order += 1;
        }
        order
    }

    fn gcd(mut a: u128, mut b: u128) -> u128 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }

    #[test]
    fn small_presentations_match_independent_arithmetic() {
        for (fixture, degree, kind) in small_fixtures() {
            let desc = black_box(fixture);
            desc.validate();
            assert_eq!(desc.degree(), degree);
            assert_eq!(desc.is_polynomial_root(), kind == Kind::Polynomial);
            assert_eq!(desc.is_basis_root(), kind == Kind::Basis);
            let size = 1u64 << degree;
            let group = u128::from(size - 1);
            let one = desc.one();
            let mut trace_ones = 0;
            for x in 0..size {
                assert_eq!(root_mul(&desc, one, x), x, "identity at {x:#x}");
                // Every right operand below degree eight; a stride through
                // the bytes keeps degree eight to a sampled sweep.
                for y in (0..size).step_by(if degree == 8 { 5 } else { 1 }) {
                    assert_eq!(desc.mul(x, y), root_mul(&desc, x, y), "{x:#x} * {y:#x}");
                }
                let inverse = desc.inv(x);
                if x == 0 {
                    assert_eq!(inverse, 0);
                } else {
                    assert_eq!(root_mul(&desc, x, inverse), one, "inverse of {x:#x}");
                }
                let mut power = one;
                for e in 0..=40 {
                    assert_eq!(desc.pow(x, e), power, "{x:#x}^{e}");
                    power = root_mul(&desc, power, x);
                }
                if x != 0 {
                    assert_eq!(desc.pow(x, group), one, "Fermat at {x:#x}");
                    assert_eq!(desc.pow(x, 5 * group + 3), desc.pow(x, 3));
                }
                let trace = desc.trace(x);
                assert_eq!(trace, reference_trace(&desc, x));
                assert!(trace == 0 || trace == one, "trace lies in GF(2)");
                trace_ones += u64::from(trace == one);
            }
            assert_eq!(trace_ones, size / 2, "trace is balanced");
            // `g^i` has full order exactly when `gcd(i, 2^n - 1) == 1`, for a
            // generator `g` found by brute-force order.
            let generator = (1..size)
                .find(|&g| reference_order(&desc, g) == size - 1)
                .unwrap();
            assert!(!desc.is_generator(0));
            let mut power = one;
            for i in 1..size {
                power = root_mul(&desc, power, generator);
                let coprime = gcd(u128::from(i), group) == 1;
                assert_eq!(desc.is_generator(power), coprime, "g^{i}");
            }
        }
    }

    /// `inv` on words with bits at or above a polynomial leaf's degree: the
    /// low `degree()` bits alone name the element, a masked zero inverts to
    /// zero, and every other masked word inverts to its own inverse under
    /// the independent product. The degree-four case includes the reduction
    /// polynomial itself as input.
    #[test]
    fn polynomial_leaf_inverse_ignores_bits_above_the_degree() {
        let cases = [
            (0x13u128, 4u32, 0x10u64),
            (0x13, 4, 0x13),
            (0x13, 4, 0x1F),
            (0x13, 4, 0xF0),
            (0x13, 4, 0xFFFF),
            (0x13, 4, u64::MAX),
            (0x11B, 8, 0x100),
            (0x11B, 8, 0x1FF),
            (0x11B, 8, 0x10000),
            (0x11B, 8, u64::MAX),
        ];
        for (full, degree, word) in cases {
            let desc = BinaryDescription::polynomial(u8::try_from(degree).unwrap(), full);
            let masked = word & mask(degree);
            let inverse = desc.inv(black_box(word));
            assert_eq!(inverse, desc.inv(masked), "inverse of {word:#x}");
            if masked == 0 {
                assert_eq!(inverse, 0, "masked zero of {word:#x}");
            } else {
                assert_eq!(
                    xtime_mul(full, degree, masked, inverse),
                    1,
                    "inverse of {word:#x}"
                );
            }
        }
    }

    /// Sampled differential of one pinned tower. The reference trace walks
    /// `n` reference squarings, so it runs on the first `traced` samples;
    /// `exponents` are the powers `k` of the selected generator whose
    /// full-order verdict must equal `gcd(k, 2^n - 1) == 1`.
    fn check_tower<F: BinaryField + HasGenerator>(
        seed: u64,
        samples: usize,
        traced: usize,
        exponents: &[u128],
    ) {
        let desc = black_box(*F::DESCRIPTION);
        desc.validate();
        let n = F::DEGREE;
        assert_eq!(desc.degree(), n);
        assert!(!desc.is_polynomial_root() && !desc.is_basis_root());
        let elem = |x| F::from_coordinates(x).unwrap();
        let one = desc.one();
        let mut words = words(seed).map(|w| w & mask(n));
        for sample in 0..samples {
            let (x, y, z) = (
                words.next().unwrap(),
                words.next().unwrap(),
                words.next().unwrap(),
            );
            let product = desc.mul(x, y);
            assert_eq!(product, root_mul(&desc, x, y), "{x:#x} * {y:#x}");
            assert_eq!(product, F::to_coordinates(elem(x) * elem(y)));
            assert_eq!(root_mul(&desc, one, x), x);
            assert_eq!(root_mul(&desc, x, desc.inv(x)), one);
            let (e1, e2) = (u128::from(y), u128::from(z));
            assert_eq!(
                desc.pow(x, e1 + e2),
                root_mul(&desc, desc.pow(x, e1), desc.pow(x, e2))
            );
            assert_eq!(desc.pow(x, 1u128 << n), x, "Frobenius fixes {x:#x}");
            let trace = desc.trace(x);
            if sample < traced {
                assert_eq!(trace, reference_trace(&desc, x));
            }
            assert!(trace == 0 || trace == one);
            assert_eq!(desc.trace(x ^ y), trace ^ desc.trace(y));
        }
        assert_eq!(desc.inv(0), 0);
        let generator = F::to_coordinates(Elem::<F>::GENERATOR);
        let group = (1u128 << n) - 1;
        assert!(!desc.is_generator(0));
        assert!(!desc.is_generator(one));
        for &k in exponents {
            let candidate = desc.pow(generator, k);
            assert_eq!(desc.is_generator(candidate), gcd(k, group) == 1, "g^{k}");
        }
    }

    /// Divisors of the group orders mixed with coprime exponents.
    const EXPONENTS: [u128; 12] = [1, 2, 3, 5, 7, 11, 15, 17, 19, 255, 257, 641];

    #[test]
    fn rijndael_towers_match_independent_arithmetic() {
        check_tower::<FiniteField<2, 16, Tower<Rijndael16>>>(0x16, 200, 200, &EXPONENTS);
        check_tower::<FiniteField<2, 32, Tower<Rijndael32>>>(0x32, 200, 64, &EXPONENTS);
        check_tower::<FiniteField<2, 64, Tower<Rijndael64>>>(0x64, 100, 16, &EXPONENTS);
    }

    #[test]
    fn fan_paar_towers_match_independent_arithmetic() {
        check_tower::<FiniteField<2, 16, Tower<FanPaar16>>>(0x116, 32, 8, &EXPONENTS);
        check_tower::<FiniteField<2, 32, Tower<FanPaar32>>>(0x132, 8, 2, &EXPONENTS[..6]);
        check_tower::<FiniteField<2, 64, Tower<FanPaar64>>>(0x164, 2, 0, &EXPONENTS[..2]);
    }

    /// `EXP` walks the powers of the generator and `LOG` inverts it.
    fn check_log_exp(table: &ByteLogExp, generator: u64, one: u64, mul: impl Fn(u64, u64) -> u64) {
        let mut power = one;
        for i in 0..255u8 {
            assert_eq!(u64::from(table.exp[usize::from(i)]), power, "generator^{i}");
            assert_eq!(table.log[usize::from(table.exp[usize::from(i)])], i);
            power = mul(power, generator);
        }
        assert_eq!(power, one, "the generator has order 255");
        for x in 0..=255u8 {
            for y in (0..=255u8).step_by(3) {
                let expected = mul(u64::from(x), u64::from(y));
                assert_eq!(u64::from(table.mul(x, y)), expected, "{x:#x} * {y:#x}");
            }
            let inverse = u64::from(table.inv(x));
            if x == 0 {
                assert_eq!(inverse, 0);
            } else {
                assert_eq!(mul(u64::from(x), inverse), one);
            }
        }
    }

    #[test]
    fn byte_log_exp_tables_follow_the_generator() {
        for (fixture, degree, _) in small_fixtures() {
            if degree != 8 {
                continue;
            }
            let generator = (2..256)
                .find(|&g| reference_order(&fixture, g) == 255)
                .unwrap();
            let table = ByteLogExp::build(black_box(&fixture), black_box(generator));
            check_log_exp(&table, generator, fixture.one(), |x, y| {
                root_mul(&fixture, x, y)
            });
        }
    }

    #[test]
    fn pinned_leaves_follow_their_generator() {
        for (full, generator) in [(0x11B, 3), (0x11D, 2)] {
            let table = serial_log_exp(black_box(full), black_box(generator));
            check_log_exp(&table, generator, 1, |x, y| xtime_mul(full, 8, x, y));
        }
    }

    #[test]
    fn irreducibility_matches_product_enumeration() {
        for n in 1..=8u32 {
            let mut reducible = [false; 512];
            for left in 2u128..1 << n {
                let dl = left.ilog2();
                for right in 1u128 << (n - dl)..1 << (n - dl + 1) {
                    reducible[usize::try_from(clmul(left, right)).unwrap()] = true;
                }
            }
            for p in 1u128 << n..1 << (n + 1) {
                let index = usize::try_from(p).unwrap();
                assert_eq!(
                    poly_irreducible(black_box(p), n),
                    !reducible[index],
                    "{p:#x}"
                );
            }
        }
        assert!(!poly_irreducible(black_box(0x13), 8), "degree mismatch");
    }

    #[test]
    fn base_description_restricts_to_the_base_presentation() {
        let gf64 = black_box(desc_of::<FiniteField<2, 64, Tower<Rijndael64>>>());
        let gf32 = gf64.base_description();
        assert!(gf32.same_structure(&desc_of::<FiniteField<2, 32, Tower<Rijndael32>>>()));
        let gf16 = gf32.base_description();
        assert!(gf16.same_structure(&desc_of::<FiniteField<2, 16, Tower<Rijndael16>>>()));
        let leaf = gf16.base_description();
        assert!(leaf.same_structure(&BinaryDescription::polynomial(8, 0x11B)));
        let normal = black_box(desc_of::<FiniteField<2, 8, Normal<0x11B, 0x20>>>());
        assert!(
            normal
                .base_description()
                .same_structure(&BinaryDescription::polynomial(8, 0x11B))
        );
        for (x, y) in words(7).zip(words(8)).take(64) {
            let (x, y) = (x & 0xFFFF_FFFF, y & 0xFFFF_FFFF);
            assert_eq!(gf32.mul(x, y), root_mul(&gf32, x, y));
        }
    }

    #[test]
    // The compared pool holds whole descriptors by value.
    #[allow(clippy::large_stack_arrays)]
    fn same_structure_distinguishes_every_defining_datum() {
        let poly = BinaryDescription::polynomial;
        let mut all = [poly(1, 3); 32];
        let mut count = 0;
        for (desc, _, _) in small_fixtures() {
            all[count] = desc;
            count += 1;
        }
        for desc in tower_fixtures() {
            all[count] = desc;
            count += 1;
        }
        let leaf = poly(8, 0x11B);
        let extra = [
            BinaryDescription::append_quadratic(&leaf, 1, 0x21),
            BinaryDescription::append_quadratic(&leaf, 2, 0x20),
            BinaryDescription::append_basis(&poly(8, 0x11D), triangular(8), [0x80; 8]),
            BinaryDescription::append_quadratic(&poly(2, 0x7), 1, 1),
            BinaryDescription::append_basis(&poly(2, 0x7), triangular(2), [2, 1, 0, 0, 0, 0, 0, 0]),
            BinaryDescription::append_basis(&triangular_over(&poly(2, 0x7)), triangular(2), [0; 8]),
            BinaryDescription::append_quadratic(&triangular_over(&poly(2, 0x7)), 1, 1),
        ];
        for desc in extra {
            all[count] = desc;
            count += 1;
        }
        let all = &all[..count];
        for (i, left) in all.iter().enumerate() {
            for (j, right) in all.iter().enumerate() {
                assert_eq!(black_box(left).same_structure(right), i == j, "{i} vs {j}");
            }
        }
        assert!(desc_of::<FiniteField<2, 1, Polynomial<3>>>().same_structure(&poly(1, 3)));
        let rebuilt = triangular_over(&poly(8, 0x12D));
        assert!(black_box(rebuilt).same_structure(&triangular_over(&poly(8, 0x12D))));
    }

    /// A description whose root slot was never filled.
    const HOLLOW: BinaryDescription = BinaryDescription {
        inner: Descriptor::from_parts([Node::Unused; NODES], 1, 0),
    };

    #[test]
    #[should_panic(expected = "unused node")]
    fn root_data_rejects_an_unused_root() {
        let _ = black_box(HOLLOW).root_data();
    }

    #[test]
    #[should_panic(expected = "unused node")]
    fn degree_rejects_an_unused_root() {
        let _ = black_box(HOLLOW).degree();
    }

    #[test]
    #[should_panic(expected = "unused node")]
    fn one_rejects_an_unused_root() {
        let _ = black_box(HOLLOW).one();
    }

    #[test]
    #[should_panic(expected = "unused node")]
    fn mul_rejects_an_unused_root() {
        let _ = black_box(HOLLOW).mul(1, 1);
    }

    #[test]
    #[should_panic(expected = "the root node has no base")]
    fn base_description_rejects_a_polynomial_root() {
        let _ = black_box(BinaryDescription::polynomial(8, 0x11B)).base_description();
    }

    #[test]
    #[should_panic(expected = "unsupported degree")]
    fn is_generator_rejects_an_unsupported_degree() {
        let _ = black_box(BinaryDescription::polynomial(3, 0xB)).is_generator(2);
    }

    #[test]
    #[should_panic(expected = "description capacity exhausted")]
    fn append_quadratic_rejects_a_full_description() {
        let full = BinaryDescription::append_quadratic(
            &desc_of::<FiniteField<2, 64, Tower<FanPaar64>>>(),
            1,
            1,
        );
        let _ = BinaryDescription::append_quadratic(black_box(&full), 1, 1);
    }

    #[test]
    #[should_panic(expected = "description capacity exhausted")]
    fn append_basis_rejects_a_full_description() {
        let full = BinaryDescription::append_quadratic(
            &desc_of::<FiniteField<2, 64, Tower<FanPaar64>>>(),
            1,
            1,
        );
        let _ = BinaryDescription::append_basis(black_box(&full), [0; 8], [0; 8]);
    }

    /// Validation of `desc` must panic with the caller's expected fragment.
    fn validate(desc: &BinaryDescription) {
        black_box(desc).validate();
    }

    #[test]
    #[should_panic(expected = "bad description shape")]
    fn validate_rejects_an_empty_description() {
        validate(&BinaryDescription {
            inner: Descriptor::from_parts([Node::Unused; NODES], 0, 0),
        });
    }

    #[test]
    #[should_panic(expected = "unused node inside description")]
    fn validate_rejects_an_unused_root() {
        validate(&HOLLOW);
    }

    #[test]
    #[should_panic(expected = "node outside description")]
    fn validate_rejects_a_node_past_the_length() {
        let mut desc = BinaryDescription::polynomial(2, 0x7);
        desc.inner = desc.inner.with_node(1, *desc.inner.node(0));
        validate(&desc);
    }

    #[test]
    #[should_panic(expected = "unsupported flat degree")]
    fn validate_rejects_an_unsupported_flat_degree() {
        validate(&BinaryDescription::polynomial(3, 0xB));
    }

    #[test]
    #[should_panic(expected = "polynomial is reducible")]
    fn validate_rejects_a_reducible_polynomial() {
        validate(&BinaryDescription::polynomial(8, 0x101));
    }

    #[test]
    #[should_panic(expected = "basis node order")]
    fn validate_rejects_a_basis_over_itself() {
        let mut desc = BinaryDescription::polynomial(2, 0x7);
        desc.inner = desc.inner.with_node(
            0,
            Node::Basis {
                base: 0,
                to_base: triangular(2),
                from_base: triangular(2),
            },
        );
        validate(&desc);
    }

    #[test]
    #[should_panic(expected = "basis node order")]
    fn validate_rejects_a_basis_node_past_the_length() {
        // Identity maps over a degree-four leaf: a valid basis node whose
        // slot sits outside the occupied prefix.
        let mut desc = BinaryDescription::polynomial(4, 0x13);
        desc.inner = desc.inner.with_node(1, basis_identity4());
        validate(&desc);
    }

    /// The identity basis over the degree-four leaf, as a raw node.
    fn basis_identity4() -> Node {
        Node::Basis {
            base: 0,
            to_base: IDENTITY4,
            from_base: IDENTITY4,
        }
    }

    #[test]
    #[should_panic(expected = "unsupported basis degree")]
    fn validate_rejects_a_basis_over_gf2() {
        validate(&BinaryDescription::append_basis(
            &BinaryDescription::polynomial(1, 3),
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ));
    }

    fn basis4(to_base: [u8; 8], from_base: [u8; 8]) -> BinaryDescription {
        BinaryDescription::append_basis(&BinaryDescription::polynomial(4, 0x13), to_base, from_base)
    }

    const IDENTITY4: [u8; 8] = [1, 2, 4, 8, 0, 0, 0, 0];

    #[test]
    #[should_panic(expected = "basis vector has excess bits")]
    fn validate_rejects_a_wide_basis_vector() {
        validate(&basis4([0x11, 2, 4, 8, 0, 0, 0, 0], IDENTITY4));
    }

    #[test]
    #[should_panic(expected = "inverse column has excess bits")]
    fn validate_rejects_a_wide_inverse_column() {
        validate(&basis4(IDENTITY4, [0x11, 2, 4, 8, 0, 0, 0, 0]));
    }

    #[test]
    #[should_panic(expected = "basis padding must be zero")]
    fn validate_rejects_basis_padding() {
        validate(&basis4([1, 2, 4, 8, 1, 0, 0, 0], IDENTITY4));
    }

    #[test]
    #[should_panic(expected = "inverse padding must be zero")]
    fn validate_rejects_inverse_padding() {
        validate(&basis4(IDENTITY4, [1, 2, 4, 8, 0, 0, 0, 1]));
    }

    #[test]
    #[should_panic(expected = "basis map is not invertible")]
    fn validate_rejects_a_mismatched_inverse() {
        validate(&basis4(IDENTITY4, [2, 1, 4, 8, 0, 0, 0, 0]));
    }

    #[test]
    #[should_panic(expected = "quadratic node order")]
    fn validate_rejects_a_quadratic_over_itself() {
        let mut desc = BinaryDescription::polynomial(2, 0x7);
        desc.inner = desc.inner.with_node(0, quadratic_words_node(0, 1, 1));
        validate(&desc);
    }

    #[test]
    #[should_panic(expected = "binary degree exceeds 64")]
    fn validate_rejects_a_degree_above_sixty_four() {
        validate(&BinaryDescription::append_quadratic(
            &desc_of::<FiniteField<2, 64, Tower<FanPaar64>>>(),
            1,
            1,
        ));
    }

    #[test]
    #[should_panic(expected = "tower coefficient has excess bits")]
    fn validate_rejects_a_wide_linear_coefficient() {
        validate(&BinaryDescription::append_quadratic(
            &BinaryDescription::polynomial(8, 0x11B),
            0x100,
            0x20,
        ));
    }

    #[test]
    #[should_panic(expected = "tower coefficient has excess bits")]
    fn validate_rejects_a_wide_constant_coefficient() {
        validate(&BinaryDescription::append_quadratic(
            &BinaryDescription::polynomial(8, 0x11B),
            1,
            0x120,
        ));
    }

    #[test]
    #[should_panic(expected = "tower coefficient A is zero")]
    fn validate_rejects_a_zero_linear_coefficient() {
        validate(&BinaryDescription::append_quadratic(
            &BinaryDescription::polynomial(8, 0x11B),
            0,
            0x20,
        ));
    }

    #[test]
    #[should_panic(expected = "tower relation is reducible")]
    fn validate_rejects_a_reducible_tower_relation() {
        // `Y^2 + Y + 1` splits over GF(2^8): GF(4) is a subfield.
        validate(&BinaryDescription::append_quadratic(
            &BinaryDescription::polynomial(8, 0x11D),
            1,
            1,
        ));
    }
}
