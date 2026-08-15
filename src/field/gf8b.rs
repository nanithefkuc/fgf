//! GF(2^8) under the AES/Rijndael polynomial `0x11B`.
//!
//! The field is `GF(2)[x] / (x^8 + x^4 + x^3 + x + 1)`. Addition and
//! subtraction are bitwise XOR. Multiplication uses compile-time discrete-log
//! tables built from generator `0x03`.
//!
//! This polynomial is not an arbitrary choice: it is the one implemented in
//! hardware by the x86 `GF2P8MULB` instruction (GFNI). Picking it lets the
//! SIMD backend issue a single instruction per 64 field multiplications
//! instead of a nibble-shuffle emulation. The sibling `gf8d`
//! field is the same construction under `0x11D` for Reed–Solomon interop.
//!
//! # Backends
//!
//! - [`Elem::mul_xtime`] is the reference: shift-and-XOR, `const`, no static
//!   storage. Every other backend is differentially tested against it.
//! - [`Elem::mul`] uses the `LOG`/`EXP` tables (511 bytes, in L1).
//!
//! # Compile-time coding matrices
//!
//! Every scalar operation here is `const`, so a Reed-Solomon generator matrix
//! is a `const` item rather than a lazily-built table. This builds the
//! 3-by-4 Vandermonde matrix `V[i][j] = x_j^i` at compile time:
//!
//! ```
//! use fgf::gf8b::Elem;
//!
//! const POINTS: [Elem; 4] = [Elem(1), Elem(2), Elem(3), Elem(4)];
//! const V: [[Elem; 4]; 3] = {
//!     let mut rows = [[Elem::ZERO; 4]; 3];
//!     let mut i = 0;
//!     while i < 3 {
//!         let mut j = 0;
//!         while j < 4 {
//!             rows[i][j] = POINTS[j].pow(i as u64);
//!             j += 1;
//!         }
//!         i += 1;
//!     }
//!     rows
//! };
//!
//! assert_eq!(V[0], [Elem::ONE; 4]);
//! assert_eq!(V[1], POINTS);
//! assert_eq!(V[2][3], Elem(4).square());
//!
//! // A parity symbol is one Vandermonde row dotted with the data symbols.
//! let data = [Elem(0x11), Elem(0x22), Elem(0x33), Elem(0x44)];
//! let parity: Elem = V[1].iter().zip(data).map(|(&v, d)| v.mul(d)).sum();
//! assert_eq!(parity, Elem(0x11).mul(Elem(1)).add(Elem(0x22).mul(Elem(2)))
//!     .add(Elem(0x33).mul(Elem(3))).add(Elem(0x44).mul(Elem(4))));
//!
//! // Division is total: `x / 0` is zero, in `const` context too.
//! const _: () = assert!(Elem(0x57).div(Elem::ZERO).to_raw() == 0);
//! ```

crate::field::flat8::flat_gf8!(Gf8B, 0x11B, 0x1B, 0x03, "GF(2^8)");
