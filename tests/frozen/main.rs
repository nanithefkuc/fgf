//! Frozen pre-cutover field results: presentation samples, shipped-instance
//! known answers, the ordered-pair embedding matrix, and backend context
//! records, each fixture file under `fixtures/` asserted against the values
//! the current crate computes.

#[macro_use]
mod common;

mod backends;
mod embeddings;
mod fields;
