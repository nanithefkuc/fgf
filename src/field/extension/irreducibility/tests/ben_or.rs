//! Generalized Ben-Or reference for differential irreducibility checks.

use crate::field::extension::arithmetic::{
    WIDE, Wide, component, element_one, element_pow, element_sub, order_of, poly_degree,
    poly_divmod, relation_coefficient, set_component, wide_get, wide_put, zero_element,
};
use crate::field::extension::descriptor::{Descriptor, Node};

/// The generalized Ben-Or test for a relation node of relative degree at
/// least three: `gcd(f, t^(q^i) - t)` must be a unit for every
/// `1 <= i <= floor(D/2)`, `q` the base order.
pub(super) const fn ben_or_irreducible(descriptor: &Descriptor, index: u8) -> bool {
    let (base, relative_degree, base_words) = match descriptor.node(index) {
        Node::Extension {
            base,
            relative_degree,
            base_words,
            ..
        } => (*base, *relative_degree, *base_words),
        _ => panic!("not a relation node"),
    };
    let base_order = order_of(descriptor, base);
    let base_one = element_one(descriptor, base);
    let mut relation: Wide = [0; WIDE];
    let mut i = 0;
    while i < relative_degree {
        wide_put(
            &mut relation,
            i,
            base_words,
            &relation_coefficient(descriptor, index, i),
        );
        i += 1;
    }
    wide_put(&mut relation, relative_degree, base_words, &base_one);
    let mut power = zero_element();
    set_component(descriptor, index, &mut power, 1, &base_one);
    let mut round = 1;
    while round <= relative_degree / 2 {
        power = element_pow(descriptor, index, &power, base_order);
        let mut difference: Wide = [0; WIDE];
        i = 0;
        while i < relative_degree {
            wide_put(
                &mut difference,
                i,
                base_words,
                &component(descriptor, index, &power, i),
            );
            i += 1;
        }
        let shifted = element_sub(
            descriptor,
            base,
            &wide_get(&difference, 1, base_words),
            &base_one,
        );
        wide_put(&mut difference, 1, base_words, &shifted);
        let mut a = relation;
        let mut b = difference;
        while poly_degree(descriptor, base, &b, base_words, relative_degree) >= 0 {
            let (_, remainder) = poly_divmod(descriptor, base, &a, &b, base_words, relative_degree);
            a = b;
            b = remainder;
        }
        if poly_degree(descriptor, base, &a, base_words, relative_degree) > 0 {
            return false;
        }
        round += 1;
    }
    true
}
