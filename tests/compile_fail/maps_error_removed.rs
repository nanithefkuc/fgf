// expect: error[E0432]
//! The 3.x error enum is removed: `ImageError` replaces it with one
//! condition per constructor outcome, and the import itself is the
//! rejection.
use fgf::maps::EmbeddingError;
