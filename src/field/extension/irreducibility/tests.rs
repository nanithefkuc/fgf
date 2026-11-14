//! Relation criteria and differential references for irreducibility validation.
//!
//! [`criteria`] covers quadratic, packed binary and published relations;
//! [`ben_or`] supplies the independent checkpoint algorithm, and
//! [`differential`] compares verdicts, factor products, basis identities and
//! resultant values.

mod ben_or;
mod criteria;
mod differential;
