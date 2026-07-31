//! # fgf — Faster Galois Fields
//!
//! **Name reservation. There is no API here yet.**
//!
//! `fgf` will be the published form of
//! [`fff`](https://github.com/nanithefkuc/fff), a dependency-free library for
//! binary finite field arithmetic: const-capable scalar elements for GF(2^8)
//! through GF(2^64) and canonical Fan-Paar towers, plus safe,
//! runtime-dispatched SIMD kernels over packed byte buffers for erasure
//! coders, proof systems, and similar consumers.
//!
//! The crate is developed in the open and distributed through git today. The
//! rename to `fgf` and the first real release here happen together at 1.0.0,
//! gated on AVX-512 validation on real silicon, a stable extracted backend
//! detection layer, and closing the remaining vectorization work.
//!
//! Until then:
//!
//! ```toml
//! [dependencies]
//! fff = { git = "https://github.com/nanithefkuc/fff" }
//! ```

#![no_std]
#![deny(missing_docs)]
#![deny(unsafe_code)]
