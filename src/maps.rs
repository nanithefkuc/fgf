//! Maps between fields: embeddings and isomorphisms.
//!
//! [`Embedding<S, T>`](Embedding) prepares one proper inclusion of `S`
//! into `T`: the source degree strictly divides the target degree.
//! [`Isomorphism<S, T>`](Isomorphism) names the equal-degree map with a
//! total inverse. One implementation serves every characteristic: the
//! prepared state stores the image in the target of each generator level
//! of the source, embedding evaluates by Horner rule in the target
//! arithmetic, membership is the relative Frobenius, and restriction
//! recovers components through trace-dual bases without solving a linear
//! system.
//!
//! A map comes from two sources only: an inclusion forced by `Extension`
//! structure (the prime subfield of the target, or a row of its own
//! tower), or caller-supplied generator images validated against the
//! transported relations. Characteristic two uses the frozen reference
//! ladder as its canonical-image rule; odd characteristic chooses no
//! canonical map between unrelated presentations.
//!
//! Construction and every method are allocation-free; the prepared state is
//! fixed arrays. Repeated application never reruns image construction.
//! Map construction and evaluation are variable-time and unsuitable for
//! secret data.
//!
//! # Representation
//!
//! A prepared relation stores one image per generator level of the source,
//! prime field upward: one level per extension node, whose generator is the
//! indeterminate; one level for a flat polynomial leaf above degree one,
//! whose generator is the residue of the indeterminate; one level for a
//! basis leaf, whose generator is the underlying polynomial root expressed
//! in the basis. Rows of absolute degree one hold no levels.
//!
//! ```
//! use fgf::{
//!     Elem, FiniteField, binary::{AES, Polynomial, Rijndael16}, extension::Extension, maps::Embedding,
//! };
//!
//! type Gf16 = FiniteField<2, 16, Extension<Rijndael16, 2>>;
//! let embedding = Embedding::<FiniteField<2, 8, Polynomial<AES>>, Gf16>::new().unwrap();
//! let x = Elem::<FiniteField<2, 8, Polynomial<AES>>>::from_raw(0x53);
//! let lifted = embedding.embed(x);
//! assert!(embedding.contains(lifted));
//! assert_eq!(embedding.restrict(lifted), Some(x));
//! assert_eq!(embedding.frobenius(lifted), lifted);
//! ```

pub use crate::field::embedding::{Embedding, EmbeddingError, Isomorphism, Postcompose};
