//! General field descriptors: const data deciding extension validity over
//! every canonical base.
//!
//! A [`Descriptor`] is the characteristic-independent generalization of the
//! binary description: prime leaves, the flat binary leaves, and relation
//! nodes in both characteristics, interpreted by bounded const arithmetic.
//! The subtree split is by responsibility: [`descriptor`] holds the data and
//! its checks, [`arithmetic`] the element interpreter, [`irreducibility`]
//! the relation tests and the validation entry point, and [`pinned`] the
//! shipped descriptors whose relations skip those tests.

pub mod arithmetic;
pub mod descriptor;
pub mod irreducibility;
pub mod pinned;

pub use descriptor::{Descriptor, Node};
