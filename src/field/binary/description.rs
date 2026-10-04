//! Opaque binary-field descriptors and their const interpreter.
//!
//! A [`BinaryDescription`] is plain const data interpreted by bounded field
//! arithmetic helpers. One topologically ordered description contains
//! polynomial, ordered-basis, and quadratic-extension nodes. The public
//! sealed traits expose it only as an opaque contract; construction stays
//! inside the representation modules.

/// One node of a binary-field description.
#[derive(Clone, Copy)]
enum BinaryNode {
    /// An unused capacity slot past the description length.
    Unused,
    /// A flat polynomial root of the stated degree.
    Polynomial {
        /// Total degree of this node.
        degree: u8,
        /// The full reduction polynomial including the leading term.
        full: u128,
    },
    /// An ordered basis over an earlier node.
    Basis {
        /// Index of the base node.
        base: u8,
        /// Basis vectors in base coordinates; bit `i` selects `to_base[i]`.
        to_base: [u8; 8],
        /// Inverse columns; bit `j` of a base word selects `from_base[j]`.
        from_base: [u8; 8],
    },
    /// A quadratic extension of an earlier node.
    Quadratic {
        /// Index of the base node.
        base: u8,
        /// Linear coefficient as a base-coordinate word.
        a: u64,
        /// Constant coefficient as a base-coordinate word.
        b: u64,
    },
}

/// Opaque scalar description of a binary-field presentation.
///
/// No public construction: representations publish their own validated
/// constants, and every value construction forces that validation.
#[derive(Clone, Copy)]
pub struct BinaryDescription {
    nodes: [BinaryNode; 8],
    len: u8,
    root: u8,
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
const fn apply(columns: [u8; 8], x: u64) -> u64 {
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
const fn poly_mul_serial(full: u128, n: u32, x: u64, y: u64) -> u64 {
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
const PINNED_AES_LEAF: ByteLogExp = serial_log_exp(0x11B, 3);

/// The pinned Reed-Solomon log/exp leaf, built serially.
const PINNED_RS_LEAF: ByteLogExp = serial_log_exp(0x11D, 2);

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

impl BinaryDescription {
    /// A lone polynomial root of the stated degree.
    #[must_use]
    pub(crate) const fn polynomial(degree: u8, full: u128) -> Self {
        let mut nodes = [BinaryNode::Unused; 8];
        nodes[0] = BinaryNode::Polynomial { degree, full };
        Self {
            nodes,
            len: 1,
            root: 0,
        }
    }

    /// Append an ordered-basis wrapper over `base`'s root.
    #[must_use]
    pub(crate) const fn append_basis(
        base: &BinaryDescription,
        to_base: [u8; 8],
        from_base: [u8; 8],
    ) -> Self {
        assert!(base.len < 8, "description capacity exhausted");
        let mut next = *base;
        next.nodes[next.len as usize] = BinaryNode::Basis {
            base: base.root,
            to_base,
            from_base,
        };
        next.root = next.len;
        next.len += 1;
        next
    }

    /// Append a quadratic extension over `base`'s root.
    #[must_use]
    pub(crate) const fn append_quadratic(base: &BinaryDescription, a: u64, b: u64) -> Self {
        assert!(base.len < 8, "description capacity exhausted");
        let mut next = *base;
        next.nodes[next.len as usize] = BinaryNode::Quadratic {
            base: base.root,
            a,
            b,
        };
        next.root = next.len;
        next.len += 1;
        next
    }

    const fn degree_of(&self, index: u8) -> u32 {
        match self.nodes[index as usize] {
            BinaryNode::Unused => panic!("unused node"),
            BinaryNode::Polynomial { degree, .. } => degree as u32,
            BinaryNode::Basis { base, .. } => self.degree_of(base),
            BinaryNode::Quadratic { base, .. } => 2 * self.degree_of(base),
        }
    }

    /// Total degree of the root presentation.
    #[must_use]
    pub const fn degree(&self) -> u32 {
        self.degree_of(self.root)
    }

    /// Whether the root node is a flat polynomial presentation.
    #[must_use]
    pub const fn is_polynomial_root(&self) -> bool {
        matches!(
            self.nodes[self.root as usize],
            BinaryNode::Polynomial { .. }
        )
    }

    /// Whether the root node is an ordered-basis presentation.
    #[must_use]
    pub const fn is_basis_root(&self) -> bool {
        matches!(self.nodes[self.root as usize], BinaryNode::Basis { .. })
    }

    const fn one_of(&self, index: u8) -> u64 {
        match self.nodes[index as usize] {
            BinaryNode::Unused => panic!("unused node"),
            BinaryNode::Polynomial { .. } => 1,
            BinaryNode::Basis { from_base, .. } => apply(from_base, 1),
            BinaryNode::Quadratic { base, .. } => self.one_of(base),
        }
    }

    /// The multiplicative identity in root coordinates.
    #[must_use]
    pub const fn one(&self) -> u64 {
        self.one_of(self.root)
    }

    const fn mul_of(&self, index: u8, x: u64, y: u64) -> u64 {
        match self.nodes[index as usize] {
            BinaryNode::Unused => panic!("unused node"),
            BinaryNode::Polynomial { degree, full } => {
                let n = degree as u32;
                // Pinned degree-eight leaves take their log/exp tables; the
                // binding tables are built serially, so reading them here
                // closes no cycle. Leaf coordinates hold one byte.
                #[allow(clippy::cast_possible_truncation)]
                let pinned = if n == 8 && full == 0x11B {
                    Some(PINNED_AES_LEAF.mul(x as u8, y as u8) as u64)
                } else if n == 8 && full == 0x11D {
                    Some(PINNED_RS_LEAF.mul(x as u8, y as u8) as u64)
                } else {
                    None
                };
                match pinned {
                    Some(product) => product,
                    None => poly_mul_serial(full, n, x, y),
                }
            }
            BinaryNode::Basis {
                base,
                to_base,
                from_base,
            } => {
                let product = self.mul_of(base, apply(to_base, x), apply(to_base, y));
                apply(from_base, product)
            }
            BinaryNode::Quadratic { base, a, b } => {
                let h = self.degree_of(base);
                let mask = if h >= 64 { u64::MAX } else { (1u64 << h) - 1 };
                let (x0, x1, y0, y1) = (x & mask, x >> h, y & mask, y >> h);
                let ac = self.mul_of(base, x0, y0);
                let bd = self.mul_of(base, x1, y1);
                let mid = self.mul_of(base, x0 ^ x1, y0 ^ y1) ^ ac ^ bd;
                let lo = ac ^ self.mul_of(base, b, bd);
                let hi = mid ^ self.mul_of(base, a, bd);
                (lo & mask) | ((hi & mask) << h)
            }
        }
    }

    /// Field multiplication in root coordinates.
    #[must_use]
    pub const fn mul(&self, x: u64, y: u64) -> u64 {
        self.mul_of(self.root, x, y)
    }

    const fn pow_of(&self, index: u8, mut x: u64, mut e: u128) -> u64 {
        let mut result = self.one_of(index);
        while e > 0 {
            if e & 1 == 1 {
                result = self.mul_of(index, result, x);
            }
            x = self.mul_of(index, x, x);
            e >>= 1;
        }
        result
    }

    /// Field exponentiation in root coordinates.
    #[must_use]
    pub const fn pow(&self, x: u64, e: u128) -> u64 {
        self.pow_of(self.root, x, e)
    }

    const fn inv_of(&self, index: u8, x: u64) -> u64 {
        if x == 0 {
            return 0;
        }
        let n = self.degree_of(index);
        self.pow_of(index, x, (1u128 << n) - 2)
    }

    /// Multiplicative inverse in root coordinates, mapping zero to zero.
    #[must_use]
    pub const fn inv(&self, x: u64) -> u64 {
        self.inv_of(self.root, x)
    }

    const fn trace_of(&self, index: u8, x: u64) -> u64 {
        let n = self.degree_of(index);
        let mut acc = 0;
        let mut power = x;
        let mut k = 0;
        while k < n {
            acc ^= power;
            power = self.mul_of(index, power, power);
            k += 1;
        }
        acc
    }

    /// Absolute trace to GF(2) in root coordinates.
    #[must_use]
    pub const fn trace(&self, x: u64) -> u64 {
        self.trace_of(self.root, x)
    }

    /// Use-time validation of the whole description.
    // The single-character locals mirror the node fields they check.
    #[allow(clippy::many_single_char_names)]
    pub(crate) const fn validate(&self) {
        assert!(
            self.len > 0 && self.len <= 8 && self.root < self.len,
            "bad description shape"
        );
        let mut i = 0u8;
        while i < 8 {
            let used = i < self.len;
            match self.nodes[i as usize] {
                BinaryNode::Unused => assert!(!used, "unused node inside description"),
                BinaryNode::Polynomial { degree, full } => {
                    assert!(used, "node outside description");
                    assert!(
                        degree == 1 || degree == 2 || degree == 4 || degree == 8,
                        "unsupported flat degree"
                    );
                    assert!(
                        poly_irreducible(full, degree as u32),
                        "polynomial is reducible"
                    );
                }
                BinaryNode::Basis {
                    base,
                    to_base,
                    from_base,
                } => {
                    assert!(used && base < i, "basis node order");
                    let n = self.degree_of(base);
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
                    // Bounded-enumeration bijection: the stored inverse inverts
                    // the forward map on every coordinate word.
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
                BinaryNode::Quadratic { base, a, b } => {
                    assert!(used && base < i, "quadratic node order");
                    let h = self.degree_of(base);
                    assert!(h <= 32, "binary degree exceeds 64");
                    if h < 64 {
                        assert!(a >> h == 0, "tower coefficient has excess bits");
                        assert!(b >> h == 0, "tower coefficient has excess bits");
                    }
                    assert!(a != 0, "tower coefficient A is zero");
                    let a_inv = self.inv_of(base, a);
                    let scaled = self.mul_of(base, b, self.mul_of(base, a_inv, a_inv));
                    assert!(
                        self.trace_of(base, scaled) == self.one_of(base),
                        "tower relation is reducible"
                    );
                }
            }
            i += 1;
        }
        assert!(self.degree() <= 64, "binary degree exceeds 64");
    }

    /// Structural equality ignoring unused slots.
    ///
    /// Strategy selection compares complete descriptions through this
    /// comparator; hashes alone do not prove arithmetic equality.
    #[must_use]
    pub const fn same_structure(&self, other: &BinaryDescription) -> bool {
        if self.len != other.len || self.root != other.root {
            return false;
        }
        let mut i = 0;
        while i < self.len as usize {
            let same = match (self.nodes[i], other.nodes[i]) {
                (
                    BinaryNode::Polynomial {
                        degree: d0,
                        full: f0,
                    },
                    BinaryNode::Polynomial {
                        degree: d1,
                        full: f1,
                    },
                ) => d0 == d1 && f0 == f1,
                (
                    BinaryNode::Basis {
                        base: b0,
                        to_base: t0,
                        from_base: f0,
                    },
                    BinaryNode::Basis {
                        base: b1,
                        to_base: t1,
                        from_base: f1,
                    },
                ) => {
                    let mut eq = b0 == b1;
                    let mut k = 0;
                    while k < 8 {
                        eq = eq && t0[k] == t1[k] && f0[k] == f1[k];
                        k += 1;
                    }
                    eq
                }
                (
                    BinaryNode::Quadratic {
                        base: b0,
                        a: a0,
                        b: c0,
                    },
                    BinaryNode::Quadratic {
                        base: b1,
                        a: a1,
                        b: c1,
                    },
                ) => b0 == b1 && a0 == a1 && c0 == c1,
                _ => false,
            };
            if !same {
                return false;
            }
            i += 1;
        }
        true
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
        if candidate == 0 || self.pow_of(self.root, candidate, order) != self.one() {
            return false;
        }
        let mut i = 0;
        while i < factors.len() {
            if self.pow_of(self.root, candidate, order / factors[i] as u128) == self.one() {
                return false;
            }
            i += 1;
        }
        true
    }
}
