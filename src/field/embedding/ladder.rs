//! The frozen characteristic-two canonical-image rule.
//!
//! The generic embedding chooses no map between unrelated odd-characteristic
//! presentations, but in characteristic two every covered pair keeps the
//! historical canonical map: the source's canonical map into the reference
//! of its degree, the frozen consecutive inclusions up to the target's
//! degree, and the inverse of the target's canonical map. The canonical map
//! factors through one frozen reference per degree: GF(2) as
//! `Polynomial<3>`, `Polynomial<0x7>` at degree two, `Polynomial<0x13>` at
//! degree four, the AES field at degree eight, and the Rijndael towers
//! above it. Consecutive inclusions are frozen: the identity into degree
//! two, the smallest raw reference root of the source polynomial below the
//! AES field, and the low component above it.
//!
//! For a presentation `R`, the canonical map `phi_R` into the reference of
//! the same degree sends a flat polynomial's indeterminate to the smallest
//! raw reference root of its polynomial, converts an ordered basis through
//! its polynomial coordinates, and transports a tower through its base
//! before mapping the indeterminate to a root of the transported relation.
//! Presentations of degree at most eight tabulate that root search over the
//! bounded field; larger towers solve the transported quadratic through
//! recursive Artin-Schreier reduction in the reference tower, with no
//! elimination and no enumeration above the byte field.
//!
//! The column maps below are transient construction aids: they compute the
//! canonical images of the source's level generators once, and the prepared
//! embedding stores only those images. No column table survives
//! construction.

use super::super::binary::description::{BinaryDescription, NodeData, apply};
use super::super::binary::tower;

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

/// Whether the frozen reference ladder covers `description`: a ladder
/// degree at the root, and every relation node of absolute degree above
/// eight quadratic, so the reference towers above the byte field apply
/// unchanged. Flat presentations always pass.
pub(super) fn covers(description: &BinaryDescription) -> bool {
    use super::super::extension::descriptor::Node;
    let descriptor = description.as_descriptor();
    if !matches!(descriptor.degree(), 1 | 2 | 4 | 8 | 16 | 32 | 64) {
        return false;
    }
    let mut index = 0;
    while index < descriptor.node_count() {
        let covered = match descriptor.node(index) {
            Node::Extension {
                base_degree,
                relative_degree,
                ..
            } => {
                let absolute = (*base_degree as usize).saturating_mul(*relative_degree);
                *relative_degree == 2 || absolute <= 8
            }
            _ => true,
        };
        if !covered {
            return false;
        }
        index += 1;
    }
    true
}

/// The canonical images of the source's level-generator words in the
/// target's coordinates, in level order.
///
/// Returns `None` when either presentation lies outside the ladder's
/// coverage. The images reproduce the historical binary embedding exactly:
/// the composite of the source map, the frozen inclusion, and the inverse
/// target map applied to each generator word.
///
/// # Panics
///
/// Panics when a covered presentation's transported relation has no
/// reference root. That is a crate defect, not an input condition.
pub(super) fn canonical_images(
    source: &BinaryDescription,
    target: &BinaryDescription,
    generator_words: &[u64],
) -> Option<[u64; 8]> {
    if !covers(source) || !covers(target) {
        return None;
    }
    let from = build_phi(source);
    let to = build_phi(target);
    let inclusion = ladder_inclusion(from.degree, to.degree);
    let mut images = [0u64; 8];
    for (slot, word) in images.iter_mut().zip(generator_words.iter()) {
        let reference = fold(&inclusion, fold(&from.forward, *word));
        *slot = fold(&to.inverse, reference);
    }
    Some(images)
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
                *column = fold(&inner.forward, apply(to_base, 1 << index));
            }
            let mut inverse = [0u64; 64];
            for (index, column) in inverse.iter_mut().enumerate().take(inner.degree as usize) {
                *column = apply(from_base, fold(&inner.inverse, 1 << index));
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
        NodeData::Relation {
            relative_degree,
            coefficients,
            ..
        } => {
            let inner = build_phi(&description.base_description());
            let h = inner.degree;
            // The coverage check admits a non-quadratic relation node only
            // at an absolute degree the bounded reference reaches: the base
            // degree times the relative degree stays at or below eight.
            assert!(
                matches!((h, relative_degree), (1, 1..=8) | (2, 1..=4) | (4, 1..=2)),
                "a non-quadratic relation node sits above the bounded reference"
            );
            let transported: [u64; 8] = core::array::from_fn(|index| {
                coefficients
                    .get(index)
                    .map_or(0, |word| fold(&inner.forward, *word))
            });
            build_small_relation_phi(&inner, &transported[..relative_degree])
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
        build_small_relation_phi(base, &[transported_b, transported_a])
    } else {
        build_wide_quadratic_phi(base, transported_a, transported_b)
    }
}

/// Build the canonical map of one relation over a base whose map is
/// `base`, at an output degree up to eight, by enumerating the bounded
/// reference for the smallest nonzero root of the transported relation.
///
/// `transported` holds the relation's non-leading coefficients in
/// reference coordinates of the base's degree, constant term first; the
/// `j`-th component's basis vectors map to the `j`-th power of the root
/// times the base image.
// The bounded word counts stay below one byte, so the table casts are
// exact. The single-character locals mirror the node fields they transport.
#[allow(clippy::cast_possible_truncation, clippy::many_single_char_names)]
fn build_small_relation_phi(base: &PhiMap, transported: &[u64]) -> PhiMap {
    let h = base.degree;
    let d = transported.len();
    let n = h * d as u32;
    assert!(
        n <= 8,
        "the bounded reference enumeration covers degrees up to eight"
    );
    let ref_n = reference(n);
    // The base embeds through the frozen ladder inclusion, whose
    // low components are not literal words below the AES field.
    let iota = ladder_inclusion(h, n);
    let mut embedded = [0u64; 8];
    let mut j = 0;
    while j < d {
        embedded[j] = fold(&iota, transported[j]);
        j += 1;
    }
    // Horner descent from the monic leading term: the value is
    // `r^d + e_{d-1} r^{d-1} + ... + e_0`.
    let mut root = 0;
    for candidate in 1..(1u64 << n) {
        let mut acc = 1u64;
        let mut k = d;
        while k > 0 {
            k -= 1;
            acc = ref_n.mul(acc, candidate) ^ embedded[k];
        }
        if acc == 0 {
            root = candidate;
            break;
        }
    }
    assert!(root != 0, "transported relation has no reference root");
    let mut forward = [0u64; 64];
    for (index, column) in forward.iter_mut().enumerate().take(h as usize) {
        *column = fold(&iota, base.forward[index]);
    }
    let mut power = 1u64;
    for j in 1..d {
        power = ref_n.mul(power, root);
        for index in 0..h as usize {
            forward[j * h as usize + index] = ref_n.mul(fold(&iota, base.forward[index]), power);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shipped tower at degree sixteen and above maps its
    /// indeterminate to the smaller root of its transported relation, and
    /// the resulting map round-trips.
    fn check_tower_phi(description: &BinaryDescription) {
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
        use super::super::super::binary::BinaryField;

        check_tower_phi(tower::RIJNDAEL16_DESC);
        check_tower_phi(tower::RIJNDAEL32_DESC);
        check_tower_phi(tower::RIJNDAEL64_DESC);
        check_tower_phi(<tower::FanPaar16Field as BinaryField>::DESCRIPTION);
        check_tower_phi(<tower::FanPaar32Field as BinaryField>::DESCRIPTION);
        check_tower_phi(<tower::FanPaar64Field as BinaryField>::DESCRIPTION);
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

    #[test]
    #[should_panic(expected = "no reference field of degree")]
    fn reference_rejects_an_unsupported_degree() {
        let _ = reference(core::hint::black_box(3));
    }

    /// `x^2` has no nonzero root in GF(4): every candidate squares to a
    /// nonzero element.
    #[test]
    #[should_panic(expected = "has no root")]
    fn smallest_root_rejects_a_polynomial_without_nonzero_roots() {
        let _ = smallest_root(&REFERENCE_2, 2, core::hint::black_box(0x4));
    }

    /// `w^2 + w` vanishes on all of GF(2), so `w^2 + w = 1` has no solution.
    #[test]
    #[should_panic(expected = "has no solution")]
    fn artin_schreier_rejects_a_trace_one_right_side() {
        let _ = solve_artin_schreier(1, core::hint::black_box(1));
    }
}
