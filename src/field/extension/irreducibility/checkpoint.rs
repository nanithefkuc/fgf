//! The exact single Ben-Or checkpoint at relative degree three.
//!
//! The relation is irreducible exactly when `gcd(f, t^q - t)` is a unit,
//! with `q` the base order. One checkpoint is sufficient at degree three;
//! no blanket two-checkpoint exit is extended to degrees four and five.

use super::super::descriptor::{COORDINATES, Descriptor};
use super::coefficient::{SCRATCH, Shape, block_mut, copy_blocks, identity, prefix, prefix_mut};
use super::polynomial::{clear_identity, gcd_unit, powmod};

/// Whether a relative-degree-three relation has a unit
/// `gcd(f, t^q - t)`.
pub(super) const fn single(
    descriptor: &Descriptor,
    shape: Shape,
    relation: &[u64],
    degree: usize,
) -> bool {
    let words = shape.words();
    let used = degree * words;
    let mut monic = [0u64; SCRATCH];
    copy_blocks(&mut monic, relation, shape, degree);
    identity(block_mut(&mut monic, degree, words), descriptor, shape);
    let mut t = [0u64; COORDINATES];
    identity(block_mut(&mut t, 1, words), descriptor, shape);
    let mut power = [0u64; COORDINATES];
    powmod(
        prefix_mut(&mut power, used),
        descriptor,
        shape,
        &monic,
        degree,
        prefix(&t, used),
        shape.order(descriptor),
    );
    clear_identity(&mut power, descriptor, shape, 1);
    gcd_unit(descriptor, shape, &monic, degree, prefix(&power, used))
}
