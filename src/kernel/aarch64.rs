//! `AArch64` NEON kernels.
//!
//! NEON has no byte-wide field multiply, so both fields use the split-nibble
//! strategy: `VTBL`/`vqtbl1q_u8` performs a 16-entry lookup across all 16
//! lanes, and `c * x` is two lookups and an XOR against the precomputed
//! [`ScaleTable`](crate::kernel::tables::ScaleTable).
//!
//! Every entry is a safe [`archmage`] capability-token function taking the
//! exact token its instructions require (`NeonToken` for baseline NEON,
//! `NeonAesToken` where the crypto extension is used) and validating its own
//! geometry; inner loops are `#[rite]` helpers inside the token-proven
//! region. Unsafe remains only where no safe primitive expresses the
//! structure — the offset-addressed rows of the scatter and matrix bodies,
//! each carrying a per-item `#[allow(unsafe_code)]` and local SINCE–THUS
//! proofs.
//!
//! [`archmage`]: https://docs.rs/archmage

pub mod bytes;
pub mod gf16;
pub mod gf8;

pub(crate) use bytes::xor_neon;
