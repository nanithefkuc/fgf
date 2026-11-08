//! Prime-field integer-SIMD differentials for the Mersenne31 and
//! Goldilocks entries: the portable-reference drives per tier, the
//! non-canonical lane canonicalization table, and the AVX-512 alias
//! screen.

use super::*;
use crate::field::goldilocks::{self, Goldilocks};
use crate::field::mersenne31::Mersenne31;
use crate::kernel::prime;
use crate::kernel::tests::tier_gate;
use crate::kernel::x86;

// Multiples of 4 (Mersenne31) / 8 (Goldilocks) straddling the 16-byte SSE
// and 32-byte AVX2 lanes, with odd element tails.
const M31_LENS: &[usize] = &[0, 4, 8, 16, 20, 32, 36, 64, 68, 128, 260, 1020];
const GLD_LENS: &[usize] = &[0, 8, 16, 24, 32, 40, 64, 72, 128, 264, 1024];
const M31_COEFFS: [u32; 8] = [
    0,
    1,
    2,
    7,
    0x5555_5555,
    0x7FFF_FFFE,
    0x7FFF_FFFF, // non-canonical zero
    0xFFFF_FFFF, // non-canonical one
];
const GLD_COEFFS: [u64; 8] = [
    0,
    1,
    2,
    7,
    0x5555_5555_5555_5555,
    goldilocks::MODULUS - 1,
    goldilocks::MODULUS, // non-canonical zero
    u64::MAX,
];

#[test]
#[allow(clippy::too_many_lines)]
fn prime_avx2_kernels_match_reference() {
    tier_gate::run_entry(x86::mersenne31::add_assign_avx2, |token, entry| {
        drive_prime_len::<Mersenne31>(
            M31_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::add_assign::<Mersenne31>(want, src);
            },
            "m31 avx2",
            "add_assign",
        );
    });
    tier_gate::run_entry(x86::mersenne31::sub_assign_avx2, |token, entry| {
        drive_prime_len::<Mersenne31>(
            M31_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::sub_assign::<Mersenne31>(want, src);
            },
            "m31 avx2",
            "sub_assign",
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_elementwise_avx2, |token, entry| {
        drive_prime_len::<Mersenne31>(
            M31_LENS,
            |got, want, _, src, b2| {
                *got = vec![0u8; src.len()];
                *want = vec![0u8; src.len()];
                entry(token, got, src, b2);
                prime::mul_elementwise::<Mersenne31>(want, src, b2);
            },
            "m31 avx2",
            "mul_elementwise",
        );
    });
    tier_gate::run_entry(
        x86::mersenne31::mul_elementwise_assign_avx2,
        |token, entry| {
            drive_prime_len::<Mersenne31>(
                M31_LENS,
                |got, want, base, src, _| {
                    entry(token, got, src);
                    prime::mul_elementwise::<Mersenne31>(want, base, src);
                },
                "m31 avx2",
                "mul_elementwise_assign",
            );
        },
    );
    tier_gate::run_entry(x86::mersenne31::mul_add_avx2, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, src| {
                entry(token, got, c, src);
                prime::mul_add::<Mersenne31>(want, *e, src);
            },
            "m31 avx2",
            "mul_add",
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_into_avx2, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, src| {
                *want = src.to_vec();
                entry(token, got, c, src);
                prime::mul_assign::<Mersenne31>(want, *e);
            },
            "m31 avx2",
            "mul_into",
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_assign_avx2, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::mul_assign::<Mersenne31>(want, *e);
            },
            "m31 avx2",
            "mul_assign",
        );
    });
    tier_gate::run_entry(x86::mersenne31::add_assign_scalar_avx2, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::add_assign_scalar::<Mersenne31>(want, *e);
            },
            "m31 avx2",
            "add_assign_scalar",
        );
    });
    tier_gate::run_entry(x86::mersenne31::sub_assign_scalar_avx2, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::sub_assign_scalar::<Mersenne31>(want, *e);
            },
            "m31 avx2",
            "sub_assign_scalar",
        );
    });
    tier_gate::run_entry(x86::goldilocks::add_assign_avx2, |token, entry| {
        drive_prime_len::<Goldilocks>(
            GLD_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::add_assign::<Goldilocks>(want, src);
            },
            "gld avx2",
            "add_assign",
        );
    });
    tier_gate::run_entry(x86::goldilocks::sub_assign_avx2, |token, entry| {
        drive_prime_len::<Goldilocks>(
            GLD_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::sub_assign::<Goldilocks>(want, src);
            },
            "gld avx2",
            "sub_assign",
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_elementwise_avx2, |token, entry| {
        drive_prime_len::<Goldilocks>(
            GLD_LENS,
            |got, want, _, src, b2| {
                *got = vec![0u8; src.len()];
                *want = vec![0u8; src.len()];
                entry(token, got, src, b2);
                prime::mul_elementwise::<Goldilocks>(want, src, b2);
            },
            "gld avx2",
            "mul_elementwise",
        );
    });
    tier_gate::run_entry(
        x86::goldilocks::mul_elementwise_assign_avx2,
        |token, entry| {
            drive_prime_len::<Goldilocks>(
                GLD_LENS,
                |got, want, base, src, _| {
                    entry(token, got, src);
                    prime::mul_elementwise::<Goldilocks>(want, base, src);
                },
                "gld avx2",
                "mul_elementwise_assign",
            );
        },
    );
    tier_gate::run_entry(x86::goldilocks::mul_add_avx2, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, src| {
                entry(token, got, c, src);
                prime::mul_add::<Goldilocks>(want, *e, src);
            },
            "gld avx2",
            "mul_add",
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_into_avx2, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, src| {
                *want = src.to_vec();
                entry(token, got, c, src);
                prime::mul_assign::<Goldilocks>(want, *e);
            },
            "gld avx2",
            "mul_into",
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_assign_avx2, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::mul_assign::<Goldilocks>(want, *e);
            },
            "gld avx2",
            "mul_assign",
        );
    });
    tier_gate::run_entry(x86::goldilocks::add_assign_scalar_avx2, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::add_assign_scalar::<Goldilocks>(want, *e);
            },
            "gld avx2",
            "add_assign_scalar",
        );
    });
    tier_gate::run_entry(x86::goldilocks::sub_assign_scalar_avx2, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::sub_assign_scalar::<Goldilocks>(want, *e);
            },
            "gld avx2",
            "sub_assign_scalar",
        );
    });
}

#[test]
#[allow(clippy::too_many_lines)]
fn prime_sse42_kernels_match_reference() {
    tier_gate::run_entry(x86::mersenne31::add_assign_sse42, |token, entry| {
        drive_prime_len::<Mersenne31>(
            M31_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::add_assign::<Mersenne31>(want, src);
            },
            "m31 sse4.2",
            "add_assign",
        );
    });
    tier_gate::run_entry(x86::mersenne31::sub_assign_sse42, |token, entry| {
        drive_prime_len::<Mersenne31>(
            M31_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::sub_assign::<Mersenne31>(want, src);
            },
            "m31 sse4.2",
            "sub_assign",
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_elementwise_sse42, |token, entry| {
        drive_prime_len::<Mersenne31>(
            M31_LENS,
            |got, want, _, src, b2| {
                *got = vec![0u8; src.len()];
                *want = vec![0u8; src.len()];
                entry(token, got, src, b2);
                prime::mul_elementwise::<Mersenne31>(want, src, b2);
            },
            "m31 sse4.2",
            "mul_elementwise",
        );
    });
    tier_gate::run_entry(
        x86::mersenne31::mul_elementwise_assign_sse42,
        |token, entry| {
            drive_prime_len::<Mersenne31>(
                M31_LENS,
                |got, want, base, src, _| {
                    entry(token, got, src);
                    prime::mul_elementwise::<Mersenne31>(want, base, src);
                },
                "m31 sse4.2",
                "mul_elementwise_assign",
            );
        },
    );
    tier_gate::run_entry(x86::mersenne31::mul_add_sse42, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, src| {
                entry(token, got, c, src);
                prime::mul_add::<Mersenne31>(want, *e, src);
            },
            "m31 sse4.2",
            "mul_add",
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_into_sse42, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, src| {
                *want = src.to_vec();
                entry(token, got, c, src);
                prime::mul_assign::<Mersenne31>(want, *e);
            },
            "m31 sse4.2",
            "mul_into",
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_assign_sse42, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::mul_assign::<Mersenne31>(want, *e);
            },
            "m31 sse4.2",
            "mul_assign",
        );
    });
    tier_gate::run_entry(x86::mersenne31::add_assign_scalar_sse42, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::add_assign_scalar::<Mersenne31>(want, *e);
            },
            "m31 sse4.2",
            "add_assign_scalar",
        );
    });
    tier_gate::run_entry(x86::mersenne31::sub_assign_scalar_sse42, |token, entry| {
        drive_prime_scalar::<Mersenne31, u32>(
            M31_LENS,
            &M31_COEFFS,
            Elem::<Mersenne31>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::sub_assign_scalar::<Mersenne31>(want, *e);
            },
            "m31 sse4.2",
            "sub_assign_scalar",
        );
    });
    tier_gate::run_entry(x86::goldilocks::add_assign_sse42, |token, entry| {
        drive_prime_len::<Goldilocks>(
            GLD_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::add_assign::<Goldilocks>(want, src);
            },
            "gld sse4.2",
            "add_assign",
        );
    });
    tier_gate::run_entry(x86::goldilocks::sub_assign_sse42, |token, entry| {
        drive_prime_len::<Goldilocks>(
            GLD_LENS,
            |got, want, _, src, _| {
                entry(token, got, src);
                prime::sub_assign::<Goldilocks>(want, src);
            },
            "gld sse4.2",
            "sub_assign",
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_elementwise_sse42, |token, entry| {
        drive_prime_len::<Goldilocks>(
            GLD_LENS,
            |got, want, _, src, b2| {
                *got = vec![0u8; src.len()];
                *want = vec![0u8; src.len()];
                entry(token, got, src, b2);
                prime::mul_elementwise::<Goldilocks>(want, src, b2);
            },
            "gld sse4.2",
            "mul_elementwise",
        );
    });
    tier_gate::run_entry(
        x86::goldilocks::mul_elementwise_assign_sse42,
        |token, entry| {
            drive_prime_len::<Goldilocks>(
                GLD_LENS,
                |got, want, base, src, _| {
                    entry(token, got, src);
                    prime::mul_elementwise::<Goldilocks>(want, base, src);
                },
                "gld sse4.2",
                "mul_elementwise_assign",
            );
        },
    );
    tier_gate::run_entry(x86::goldilocks::mul_add_sse42, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, src| {
                entry(token, got, c, src);
                prime::mul_add::<Goldilocks>(want, *e, src);
            },
            "gld sse4.2",
            "mul_add",
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_into_sse42, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, src| {
                *want = src.to_vec();
                entry(token, got, c, src);
                prime::mul_assign::<Goldilocks>(want, *e);
            },
            "gld sse4.2",
            "mul_into",
        );
    });
    tier_gate::run_entry(x86::goldilocks::mul_assign_sse42, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::mul_assign::<Goldilocks>(want, *e);
            },
            "gld sse4.2",
            "mul_assign",
        );
    });
    tier_gate::run_entry(x86::goldilocks::add_assign_scalar_sse42, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::add_assign_scalar::<Goldilocks>(want, *e);
            },
            "gld sse4.2",
            "add_assign_scalar",
        );
    });
    tier_gate::run_entry(x86::goldilocks::sub_assign_scalar_sse42, |token, entry| {
        drive_prime_scalar::<Goldilocks, u64>(
            GLD_LENS,
            &GLD_COEFFS,
            Elem::<Goldilocks>::from_raw,
            |got, want, c, e, _| {
                entry(token, got, c);
                prime::sub_assign_scalar::<Goldilocks>(want, *e);
            },
            "gld sse4.2",
            "sub_assign_scalar",
        );
    });
}

/// Lane-straddling lengths past the 64-byte vector, each ending in an
/// AVX2 and a scalar remainder.
#[cfg(feature = "simd512")]
const M31_WIDE: &[usize] = &[0, 4, 60, 64, 68, 96, 128, 196, 260, 1020, 1028];
#[cfg(feature = "simd512")]
const GLD_WIDE: &[usize] = &[0, 8, 56, 64, 72, 96, 128, 200, 264, 1024, 1032];

#[cfg(feature = "simd512")]
#[test]
#[allow(clippy::too_many_lines)]
fn prime_avx512_kernels_match_reference() {
    tier_gate::run_entry(
        x86::mersenne31::add_assign_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_len::<Mersenne31>(
                M31_WIDE,
                |got, want, _, src, _| {
                    entry(token.v4(), got, src);
                    prime::add_assign::<Mersenne31>(want, src);
                },
                "m31 avx512",
                "add_assign",
            );
        },
    );
    tier_gate::run_entry(
        x86::mersenne31::sub_assign_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_len::<Mersenne31>(
                M31_WIDE,
                |got, want, _, src, _| {
                    entry(token.v4(), got, src);
                    prime::sub_assign::<Mersenne31>(want, src);
                },
                "m31 avx512",
                "sub_assign",
            );
        },
    );
    tier_gate::run_entry(
        x86::mersenne31::mul_elementwise_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_len::<Mersenne31>(
                M31_WIDE,
                |got, want, _, src, b2| {
                    *got = vec![0u8; src.len()];
                    *want = vec![0u8; src.len()];
                    entry(token.v4(), got, src, b2);
                    prime::mul_elementwise::<Mersenne31>(want, src, b2);
                },
                "m31 avx512",
                "mul_elementwise",
            );
        },
    );
    tier_gate::run_entry(
        x86::mersenne31::mul_elementwise_assign_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_len::<Mersenne31>(
                M31_WIDE,
                |got, want, base, src, _| {
                    entry(token.v4(), got, src);
                    prime::mul_elementwise::<Mersenne31>(want, base, src);
                },
                "m31 avx512",
                "mul_elementwise_assign",
            );
        },
    );
    tier_gate::run_entry(
        x86::mersenne31::mul_add_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Mersenne31, u32>(
                M31_WIDE,
                &M31_COEFFS,
                Elem::<Mersenne31>::from_raw,
                |got, want, c, e, src| {
                    entry(token.v4(), got, c, src);
                    prime::mul_add::<Mersenne31>(want, *e, src);
                },
                "m31 avx512",
                "mul_add",
            );
        },
    );
    tier_gate::run_entry(
        x86::mersenne31::mul_into_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Mersenne31, u32>(
                M31_WIDE,
                &M31_COEFFS,
                Elem::<Mersenne31>::from_raw,
                |got, want, c, e, src| {
                    *want = src.to_vec();
                    entry(token.v4(), got, c, src);
                    prime::mul_assign::<Mersenne31>(want, *e);
                },
                "m31 avx512",
                "mul_into",
            );
        },
    );
    tier_gate::run_entry(
        x86::mersenne31::mul_assign_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Mersenne31, u32>(
                M31_WIDE,
                &M31_COEFFS,
                Elem::<Mersenne31>::from_raw,
                |got, want, c, e, _| {
                    entry(token.v4(), got, c);
                    prime::mul_assign::<Mersenne31>(want, *e);
                },
                "m31 avx512",
                "mul_assign",
            );
        },
    );
    tier_gate::run_entry(
        x86::mersenne31::add_assign_scalar_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Mersenne31, u32>(
                M31_WIDE,
                &M31_COEFFS,
                Elem::<Mersenne31>::from_raw,
                |got, want, c, e, _| {
                    entry(token.v4(), got, c);
                    prime::add_assign_scalar::<Mersenne31>(want, *e);
                },
                "m31 avx512",
                "add_assign_scalar",
            );
        },
    );
    tier_gate::run_entry(
        x86::mersenne31::sub_assign_scalar_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Mersenne31, u32>(
                M31_WIDE,
                &M31_COEFFS,
                Elem::<Mersenne31>::from_raw,
                |got, want, c, e, _| {
                    entry(token.v4(), got, c);
                    prime::sub_assign_scalar::<Mersenne31>(want, *e);
                },
                "m31 avx512",
                "sub_assign_scalar",
            );
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::add_assign_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_len::<Goldilocks>(
                GLD_WIDE,
                |got, want, _, src, _| {
                    entry(token.v4(), got, src);
                    prime::add_assign::<Goldilocks>(want, src);
                },
                "gld avx512",
                "add_assign",
            );
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::sub_assign_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_len::<Goldilocks>(
                GLD_WIDE,
                |got, want, _, src, _| {
                    entry(token.v4(), got, src);
                    prime::sub_assign::<Goldilocks>(want, src);
                },
                "gld avx512",
                "sub_assign",
            );
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::mul_elementwise_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_len::<Goldilocks>(
                GLD_WIDE,
                |got, want, _, src, b2| {
                    *got = vec![0u8; src.len()];
                    *want = vec![0u8; src.len()];
                    entry(token.v4(), got, src, b2);
                    prime::mul_elementwise::<Goldilocks>(want, src, b2);
                },
                "gld avx512",
                "mul_elementwise",
            );
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::mul_elementwise_assign_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_len::<Goldilocks>(
                GLD_WIDE,
                |got, want, base, src, _| {
                    entry(token.v4(), got, src);
                    prime::mul_elementwise::<Goldilocks>(want, base, src);
                },
                "gld avx512",
                "mul_elementwise_assign",
            );
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::mul_add_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Goldilocks, u64>(
                GLD_WIDE,
                &GLD_COEFFS,
                Elem::<Goldilocks>::from_raw,
                |got, want, c, e, src| {
                    entry(token.v4(), got, c, src);
                    prime::mul_add::<Goldilocks>(want, *e, src);
                },
                "gld avx512",
                "mul_add",
            );
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::mul_into_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Goldilocks, u64>(
                GLD_WIDE,
                &GLD_COEFFS,
                Elem::<Goldilocks>::from_raw,
                |got, want, c, e, src| {
                    *want = src.to_vec();
                    entry(token.v4(), got, c, src);
                    prime::mul_assign::<Goldilocks>(want, *e);
                },
                "gld avx512",
                "mul_into",
            );
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::mul_assign_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Goldilocks, u64>(
                GLD_WIDE,
                &GLD_COEFFS,
                Elem::<Goldilocks>::from_raw,
                |got, want, c, e, _| {
                    entry(token.v4(), got, c);
                    prime::mul_assign::<Goldilocks>(want, *e);
                },
                "gld avx512",
                "mul_assign",
            );
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::add_assign_scalar_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Goldilocks, u64>(
                GLD_WIDE,
                &GLD_COEFFS,
                Elem::<Goldilocks>::from_raw,
                |got, want, c, e, _| {
                    entry(token.v4(), got, c);
                    prime::add_assign_scalar::<Goldilocks>(want, *e);
                },
                "gld avx512",
                "add_assign_scalar",
            );
        },
    );
    tier_gate::run_entry(
        x86::goldilocks::sub_assign_scalar_avx512,
        |token: X64V4xToken, entry| {
            drive_prime_scalar::<Goldilocks, u64>(
                GLD_WIDE,
                &GLD_COEFFS,
                Elem::<Goldilocks>::from_raw,
                |got, want, c, e, _| {
                    entry(token.v4(), got, c);
                    prime::sub_assign_scalar::<Goldilocks>(want, *e);
                },
                "gld avx512",
                "sub_assign_scalar",
            );
        },
    );
}
/// Non-canonical lanes through every prime entry: buffers whose lanes
/// hold `p`, wider aliases, and all-ones words must match the portable
/// prime reference lane for lane, and every stored lane must be
/// canonical — accumulate, in-place, overwrite, elementwise, broadcast,
/// and tails included, at lane-straddling lengths.
#[test]
// Nine entries times three tiers at a dozen lengths each: a table by nature.
#[allow(clippy::too_many_lines)]
fn prime_entries_canonicalize_noncanonical_lanes() {
    // Cycle lane words across a buffer, little-endian lanes.
    fn fill(len: usize, words: &[u64], lane_bytes: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut i = 0;
        while out.len() < len {
            out.extend_from_slice(&words[i % words.len()].to_le_bytes()[..lane_bytes]);
            i += 1;
        }
        out.truncate(len);
        out
    }
    // Every lane-sized chunk holds a value below the modulus.
    fn assert_canonical(label: &str, buf: &[u8], lane_bytes: usize, modulus: u64) {
        for (i, lane) in buf.chunks_exact(lane_bytes).enumerate() {
            let mut word = 0u64;
            for (j, &b) in lane.iter().enumerate() {
                word |= u64::from(b) << (8 * j);
            }
            assert!(
                word < modulus,
                "{label}: lane {i} holds {word:#x}, want < {modulus:#x}"
            );
        }
    }

    const P31: u64 = crate::field::mersenne31::MODULUS as u64;
    const PGLD: u64 = crate::field::goldilocks::MODULUS;
    const L32: &[u64] = &[
        0,
        1,
        P31 - 1,
        P31,
        P31 + 1,
        0x8000_0000,
        0xFFFF_FFFE,
        0xFFFF_FFFF,
    ];
    const C32: &[u64] = &[0, 1, 2, 0x53, P31, 0xFFFF_FFFF];
    const L64: &[u64] = &[
        0,
        1,
        PGLD - 1,
        PGLD,
        PGLD + 1,
        0x1_0000_0000,
        u64::MAX - 1,
        u64::MAX,
    ];
    const C64: &[u64] = &[0, 1, 2, 0x53, PGLD, u64::MAX];

    #[allow(clippy::too_many_arguments)]
    fn drive_m31(
        label: &str,
        add: impl Fn(&mut [u8], &[u8]),
        sub: impl Fn(&mut [u8], &[u8]),
        muladd: impl Fn(&mut [u8], u32, &[u8]),
        mulinto: impl Fn(&mut [u8], u32, &[u8]),
        mulassign: impl Fn(&mut [u8], u32),
        elemwise: impl Fn(&mut [u8], &[u8], &[u8]),
        elemwise_assign: impl Fn(&mut [u8], &[u8]),
        addscalar: impl Fn(&mut [u8], u32),
        subscalar: impl Fn(&mut [u8], u32),
    ) {
        use crate::field::FieldElem as _;
        use crate::field::mersenne31::Mersenne31;
        for &len in M31_LENS {
            let src = fill(len, L32, 4);
            let base = fill(len, &L32[3..], 4);
            let other = fill(len, &L32[5..], 4);

            let mut got = base.clone();
            let mut want = base.clone();
            add(&mut got, &src);
            prime::add_assign::<Mersenne31>(&mut want, &src);
            assert_eq!(got, want, "{label} add_assign len {len}");
            assert_canonical(label, &got, 4, P31);

            let mut got = base.clone();
            let mut want = base.clone();
            sub(&mut got, &src);
            prime::sub_assign::<Mersenne31>(&mut want, &src);
            assert_eq!(got, want, "{label} sub_assign len {len}");
            assert_canonical(label, &got, 4, P31);

            // A small-minuend, aliased-subtrahend pair the rotation
            // above never aligns: `0 - 0xFFFF_FFFF` must be `p - 1`.
            let sub_base = fill(len, &L32[1..], 4);
            let mut got = sub_base.clone();
            let mut want = sub_base.clone();
            sub(&mut got, &src);
            prime::sub_assign::<Mersenne31>(&mut want, &src);
            assert_eq!(got, want, "{label} sub_assign shifted len {len}");
            assert_canonical(label, &got, 4, P31);

            let mut got = vec![0u8; len];
            let mut want = vec![0u8; len];
            elemwise(&mut got, &src, &other);
            prime::mul_elementwise::<Mersenne31>(&mut want, &src, &other);
            assert_eq!(got, want, "{label} mul_elementwise len {len}");
            assert_canonical(label, &got, 4, P31);

            let mut got = base.clone();
            let mut want = base.clone();
            elemwise_assign(&mut got, &src);
            prime::mul_elementwise::<Mersenne31>(&mut want, &base, &src);
            assert_eq!(got, want, "{label} mul_elementwise_assign len {len}");
            assert_canonical(label, &got, 4, P31);

            for &c in C32 {
                let c32 = c as u32;
                let e = Elem::<Mersenne31>::from_raw(c32);
                let mut got = base.clone();
                let mut want = base.clone();
                muladd(&mut got, c32, &src);
                prime::mul_add::<Mersenne31>(&mut want, e, &src);
                assert_eq!(got, want, "{label} mul_add len {len} coeff {c:#x}");
                // A zero coefficient is a defined no-op at every level
                // (ops fast path, scalar reference, kernel entry), so
                // untouched input lanes are preserved, not canonicalized.
                if !e.is_zero() {
                    assert_canonical(label, &got, 4, P31);
                }

                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                mulinto(&mut got, c32, &src);
                prime::mul_add::<Mersenne31>(&mut want, e, &src);
                // Overwrite of a zero buffer is the product itself.
                assert_eq!(got, want, "{label} mul_into len {len} coeff {c:#x}");
                assert_canonical(label, &got, 4, P31);

                let mut got = base.clone();
                let mut want = base.clone();
                mulassign(&mut got, c32);
                prime::mul_assign::<Mersenne31>(&mut want, e);
                assert_eq!(got, want, "{label} mul_assign len {len} coeff {c:#x}");
                // A unit coefficient is a defined no-op, preserving
                // untouched input lanes.
                if !e.is_one() {
                    assert_canonical(label, &got, 4, P31);
                }

                let mut got = base.clone();
                let mut want = base.clone();
                addscalar(&mut got, c32);
                prime::add_assign_scalar::<Mersenne31>(&mut want, e);
                assert_eq!(
                    got, want,
                    "{label} add_assign_scalar len {len} coeff {c:#x}"
                );
                assert_canonical(label, &got, 4, P31);

                let mut got = base.clone();
                let mut want = base.clone();
                subscalar(&mut got, c32);
                prime::sub_assign_scalar::<Mersenne31>(&mut want, e);
                assert_eq!(
                    got, want,
                    "{label} sub_assign_scalar len {len} coeff {c:#x}"
                );
                assert_canonical(label, &got, 4, P31);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn drive_gld(
        label: &str,
        add: impl Fn(&mut [u8], &[u8]),
        sub: impl Fn(&mut [u8], &[u8]),
        muladd: impl Fn(&mut [u8], u64, &[u8]),
        mulinto: impl Fn(&mut [u8], u64, &[u8]),
        mulassign: impl Fn(&mut [u8], u64),
        elemwise: impl Fn(&mut [u8], &[u8], &[u8]),
        elemwise_assign: impl Fn(&mut [u8], &[u8]),
        addscalar: impl Fn(&mut [u8], u64),
        subscalar: impl Fn(&mut [u8], u64),
    ) {
        use crate::field::FieldElem as _;
        use crate::field::goldilocks::Goldilocks;
        for &len in GLD_LENS {
            let src = fill(len, L64, 8);
            let base = fill(len, &L64[3..], 8);
            let other = fill(len, &L64[5..], 8);

            let mut got = base.clone();
            let mut want = base.clone();
            add(&mut got, &src);
            prime::add_assign::<Goldilocks>(&mut want, &src);
            assert_eq!(got, want, "{label} add_assign len {len}");
            assert_canonical(label, &got, 8, PGLD);

            let mut got = base.clone();
            let mut want = base.clone();
            sub(&mut got, &src);
            prime::sub_assign::<Goldilocks>(&mut want, &src);
            assert_eq!(got, want, "{label} sub_assign len {len}");
            assert_canonical(label, &got, 8, PGLD);

            let mut got = vec![0u8; len];
            let mut want = vec![0u8; len];
            elemwise(&mut got, &src, &other);
            prime::mul_elementwise::<Goldilocks>(&mut want, &src, &other);
            assert_eq!(got, want, "{label} mul_elementwise len {len}");
            assert_canonical(label, &got, 8, PGLD);

            let mut got = base.clone();
            let mut want = base.clone();
            elemwise_assign(&mut got, &src);
            prime::mul_elementwise::<Goldilocks>(&mut want, &base, &src);
            assert_eq!(got, want, "{label} mul_elementwise_assign len {len}");
            assert_canonical(label, &got, 8, PGLD);

            for &c in C64 {
                let e = Elem::<Goldilocks>::from_raw(c);
                let mut got = base.clone();
                let mut want = base.clone();
                muladd(&mut got, c, &src);
                prime::mul_add::<Goldilocks>(&mut want, e, &src);
                assert_eq!(got, want, "{label} mul_add len {len} coeff {c:#x}");
                // A zero coefficient is a defined no-op at every level
                // (ops fast path, scalar reference, kernel entry), so
                // untouched input lanes are preserved, not canonicalized.
                if !e.is_zero() {
                    assert_canonical(label, &got, 8, PGLD);
                }

                let mut got = vec![0u8; len];
                let mut want = vec![0u8; len];
                mulinto(&mut got, c, &src);
                prime::mul_add::<Goldilocks>(&mut want, e, &src);
                assert_eq!(got, want, "{label} mul_into len {len} coeff {c:#x}");
                assert_canonical(label, &got, 8, PGLD);

                let mut got = base.clone();
                let mut want = base.clone();
                mulassign(&mut got, c);
                prime::mul_assign::<Goldilocks>(&mut want, e);
                assert_eq!(got, want, "{label} mul_assign len {len} coeff {c:#x}");
                // A unit coefficient is a defined no-op, preserving
                // untouched input lanes.
                if !e.is_one() {
                    assert_canonical(label, &got, 8, PGLD);
                }

                let mut got = base.clone();
                let mut want = base.clone();
                addscalar(&mut got, c);
                prime::add_assign_scalar::<Goldilocks>(&mut want, e);
                assert_eq!(
                    got, want,
                    "{label} add_assign_scalar len {len} coeff {c:#x}"
                );
                assert_canonical(label, &got, 8, PGLD);

                let mut got = base.clone();
                let mut want = base.clone();
                subscalar(&mut got, c);
                prime::sub_assign_scalar::<Goldilocks>(&mut want, e);
                assert_eq!(
                    got, want,
                    "{label} sub_assign_scalar len {len} coeff {c:#x}"
                );
                assert_canonical(label, &got, 8, PGLD);
            }
        }
    }

    /// Exhaustive boundary-word pairs for the Mersenne31 entries: for
    /// every ordered pair of the seven boundary words `0, 1, p-1, p,
    /// p+1, 2p, 2p+1 (= u32::MAX)`, one operand holds `x` in every
    /// lane and the other holds `y`, so every lane position sees every
    /// boundary value against every other in both operand roles. The
    /// expectations come from an independent `u128 % p` model, not the
    /// crate's reference.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn drive_m31_pairs(
        label: &str,
        add: impl Fn(&mut [u8], &[u8]),
        sub: impl Fn(&mut [u8], &[u8]),
        muladd: impl Fn(&mut [u8], u32, &[u8]),
        mulinto: impl Fn(&mut [u8], u32, &[u8]),
        mulassign: impl Fn(&mut [u8], u32),
        elemwise: impl Fn(&mut [u8], &[u8], &[u8]),
        elemwise_assign: impl Fn(&mut [u8], &[u8]),
        addscalar: impl Fn(&mut [u8], u32),
        subscalar: impl Fn(&mut [u8], u32),
    ) {
        const P: u32 = crate::field::mersenne31::MODULUS;
        const WORDS: [u32; 7] = [0, 1, P - 1, P, P + 1, 2 * P, u32::MAX];
        // Lengths straddling the SSE4.2 lane/tile (16/64 bytes), the
        // AVX2 lane/tile (32/128), and the AVX-512 lane (64), each
        // ending in a vector or scalar tail somewhere.
        const LENS: &[usize] = &[16, 20, 32, 64, 68, 128, 132];
        const COEFFS: [u32; 6] = [0, 1, 2, 0x53, P, u32::MAX];
        let wide = u128::from(P);
        let canon = |v: u32| u128::from(v) % wide;
        let word = |v: u128| u32::try_from(v).unwrap();
        let fill = |len: usize, w: u32| -> Vec<u8> {
            let mut out = Vec::with_capacity(len);
            while out.len() < len {
                out.extend_from_slice(&w.to_le_bytes());
            }
            out
        };

        for &len in LENS {
            let pairs = WORDS
                .iter()
                .flat_map(|&x| WORDS.iter().map(move |&y| (x, y)));
            for (x, y) in pairs {
                let (cx, cy) = (canon(x), canon(y));
                let dst_x = fill(len, x);
                let src_y = fill(len, y);

                let mut got = dst_x.clone();
                add(&mut got, &src_y);
                let want = fill(len, word((cx + cy) % wide));
                assert_eq!(got, want, "{label} pair add {x:#x} {y:#x} len {len}");

                let mut got = dst_x.clone();
                sub(&mut got, &src_y);
                let want = fill(len, word((cx + wide - cy) % wide));
                assert_eq!(got, want, "{label} pair sub {x:#x} {y:#x} len {len}");

                let mut got = vec![0u8; len];
                elemwise(&mut got, &dst_x, &src_y);
                let want = fill(len, word(cx * cy % wide));
                assert_eq!(got, want, "{label} pair elemwise {x:#x} {y:#x} len {len}");

                let mut got = dst_x.clone();
                elemwise_assign(&mut got, &src_y);
                assert_eq!(
                    got,
                    fill(len, word(cx * cy % wide)),
                    "{label} pair elemwise_assign {x:#x} {y:#x} len {len}"
                );

                for &cv in &COEFFS {
                    let c = canon(cv);
                    let mut got = dst_x.clone();
                    muladd(&mut got, cv, &src_y);
                    // A zero-valued coefficient is a defined no-op that
                    // preserves the untouched lanes byte for byte.
                    let expected = if c == 0 {
                        x
                    } else {
                        word((cx + c * cy) % wide)
                    };
                    assert_eq!(
                        got,
                        fill(len, expected),
                        "{label} pair mul_add {x:#x} {y:#x} coeff {cv:#x} len {len}"
                    );

                    let mut got = vec![0u8; len];
                    mulinto(&mut got, cv, &src_y);
                    let expected = if c == 0 { 0 } else { word(c * cy % wide) };
                    assert_eq!(
                        got,
                        fill(len, expected),
                        "{label} pair mul_into {x:#x} {y:#x} coeff {cv:#x} len {len}"
                    );

                    let mut got = dst_x.clone();
                    mulassign(&mut got, cv);
                    // A unit-valued coefficient is a defined no-op; a
                    // zero-valued one zeroes the buffer.
                    let expected = if c == 0 {
                        0
                    } else if c == 1 {
                        x
                    } else {
                        word(c * cx % wide)
                    };
                    assert_eq!(
                        got,
                        fill(len, expected),
                        "{label} pair mul_assign {x:#x} coeff {cv:#x} len {len}"
                    );
                }

                for &v in &WORDS {
                    let cv = canon(v);
                    let mut got = dst_x.clone();
                    addscalar(&mut got, v);
                    assert_eq!(
                        got,
                        fill(len, word((cx + cv) % wide)),
                        "{label} pair add_scalar {x:#x} value {v:#x} len {len}"
                    );

                    let mut got = dst_x.clone();
                    subscalar(&mut got, v);
                    assert_eq!(
                        got,
                        fill(len, word((cx + wide - cv) % wide)),
                        "{label} pair sub_scalar {x:#x} value {v:#x} len {len}"
                    );
                }
            }
        }
    }

    macro_rules! canonicalize_tier {
        ($m31_add:path, $m31_sub:path, $m31_muladd:path, $m31_mulinto:path,
         $m31_mulassign:path, $m31_elemwise:path, $m31_elemwise_assign:path,
         $m31_addscalar:path, $m31_subscalar:path, $gld_add:path, $gld_sub:path,
         $gld_muladd:path, $gld_mulinto:path, $gld_mulassign:path, $gld_elemwise:path,
         $gld_elemwise_assign:path, $gld_addscalar:path, $gld_subscalar:path,
         $m31_label:literal, $gld_label:literal, $wide:expr) => {
            tier_gate::run_entry($m31_add, |token, entry| {
                drive_m31(
                    $m31_label,
                    |dst, src| entry($wide(token), dst, src),
                    |dst, src| $m31_sub($wide(token), dst, src),
                    |dst, c, src| $m31_muladd($wide(token), dst, c, src),
                    |dst, c, src| $m31_mulinto($wide(token), dst, c, src),
                    |dst, c| $m31_mulassign($wide(token), dst, c),
                    |dst, a, b| $m31_elemwise($wide(token), dst, a, b),
                    |dst, s| $m31_elemwise_assign($wide(token), dst, s),
                    |dst, c| $m31_addscalar($wide(token), dst, c),
                    |dst, c| $m31_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($m31_sub, |token, entry| {
                drive_m31(
                    $m31_label,
                    |dst, src| $m31_add($wide(token), dst, src),
                    |dst, src| entry($wide(token), dst, src),
                    |dst, c, src| $m31_muladd($wide(token), dst, c, src),
                    |dst, c, src| $m31_mulinto($wide(token), dst, c, src),
                    |dst, c| $m31_mulassign($wide(token), dst, c),
                    |dst, a, b| $m31_elemwise($wide(token), dst, a, b),
                    |dst, s| $m31_elemwise_assign($wide(token), dst, s),
                    |dst, c| $m31_addscalar($wide(token), dst, c),
                    |dst, c| $m31_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($m31_muladd, |token, entry| {
                drive_m31(
                    $m31_label,
                    |dst, src| $m31_add($wide(token), dst, src),
                    |dst, src| $m31_sub($wide(token), dst, src),
                    |dst, c, src| entry($wide(token), dst, c, src),
                    |dst, c, src| $m31_mulinto($wide(token), dst, c, src),
                    |dst, c| $m31_mulassign($wide(token), dst, c),
                    |dst, a, b| $m31_elemwise($wide(token), dst, a, b),
                    |dst, s| $m31_elemwise_assign($wide(token), dst, s),
                    |dst, c| $m31_addscalar($wide(token), dst, c),
                    |dst, c| $m31_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($m31_mulinto, |token, entry| {
                drive_m31(
                    $m31_label,
                    |dst, src| $m31_add($wide(token), dst, src),
                    |dst, src| $m31_sub($wide(token), dst, src),
                    |dst, c, src| $m31_muladd($wide(token), dst, c, src),
                    |dst, c, src| entry($wide(token), dst, c, src),
                    |dst, c| $m31_mulassign($wide(token), dst, c),
                    |dst, a, b| $m31_elemwise($wide(token), dst, a, b),
                    |dst, s| $m31_elemwise_assign($wide(token), dst, s),
                    |dst, c| $m31_addscalar($wide(token), dst, c),
                    |dst, c| $m31_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($m31_mulassign, |token, entry| {
                drive_m31(
                    $m31_label,
                    |dst, src| $m31_add($wide(token), dst, src),
                    |dst, src| $m31_sub($wide(token), dst, src),
                    |dst, c, src| $m31_muladd($wide(token), dst, c, src),
                    |dst, c, src| $m31_mulinto($wide(token), dst, c, src),
                    |dst, c| entry($wide(token), dst, c),
                    |dst, a, b| $m31_elemwise($wide(token), dst, a, b),
                    |dst, s| $m31_elemwise_assign($wide(token), dst, s),
                    |dst, c| $m31_addscalar($wide(token), dst, c),
                    |dst, c| $m31_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($m31_elemwise, |token, entry| {
                drive_m31(
                    $m31_label,
                    |dst, src| $m31_add($wide(token), dst, src),
                    |dst, src| $m31_sub($wide(token), dst, src),
                    |dst, c, src| $m31_muladd($wide(token), dst, c, src),
                    |dst, c, src| $m31_mulinto($wide(token), dst, c, src),
                    |dst, c| $m31_mulassign($wide(token), dst, c),
                    |dst, a, b| entry($wide(token), dst, a, b),
                    |dst, s| $m31_elemwise_assign($wide(token), dst, s),
                    |dst, c| $m31_addscalar($wide(token), dst, c),
                    |dst, c| $m31_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($m31_elemwise_assign, |token, entry| {
                drive_m31(
                    $m31_label,
                    |dst, src| $m31_add($wide(token), dst, src),
                    |dst, src| $m31_sub($wide(token), dst, src),
                    |dst, c, src| $m31_muladd($wide(token), dst, c, src),
                    |dst, c, src| $m31_mulinto($wide(token), dst, c, src),
                    |dst, c| $m31_mulassign($wide(token), dst, c),
                    |dst, a, b| $m31_elemwise($wide(token), dst, a, b),
                    |dst, s| entry($wide(token), dst, s),
                    |dst, c| $m31_addscalar($wide(token), dst, c),
                    |dst, c| $m31_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($m31_addscalar, |token, entry| {
                drive_m31(
                    $m31_label,
                    |dst, src| $m31_add($wide(token), dst, src),
                    |dst, src| $m31_sub($wide(token), dst, src),
                    |dst, c, src| $m31_muladd($wide(token), dst, c, src),
                    |dst, c, src| $m31_mulinto($wide(token), dst, c, src),
                    |dst, c| $m31_mulassign($wide(token), dst, c),
                    |dst, a, b| $m31_elemwise($wide(token), dst, a, b),
                    |dst, s| $m31_elemwise_assign($wide(token), dst, s),
                    |dst, c| entry($wide(token), dst, c),
                    |dst, c| $m31_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($m31_subscalar, |token, entry| {
                drive_m31(
                    $m31_label,
                    |dst, src| $m31_add($wide(token), dst, src),
                    |dst, src| $m31_sub($wide(token), dst, src),
                    |dst, c, src| $m31_muladd($wide(token), dst, c, src),
                    |dst, c, src| $m31_mulinto($wide(token), dst, c, src),
                    |dst, c| $m31_mulassign($wide(token), dst, c),
                    |dst, a, b| $m31_elemwise($wide(token), dst, a, b),
                    |dst, s| $m31_elemwise_assign($wide(token), dst, s),
                    |dst, c| $m31_addscalar($wide(token), dst, c),
                    |dst, c| entry($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($gld_add, |token, entry| {
                drive_gld(
                    $gld_label,
                    |dst, src| entry($wide(token), dst, src),
                    |dst, src| $gld_sub($wide(token), dst, src),
                    |dst, c, src| $gld_muladd($wide(token), dst, c, src),
                    |dst, c, src| $gld_mulinto($wide(token), dst, c, src),
                    |dst, c| $gld_mulassign($wide(token), dst, c),
                    |dst, a, b| $gld_elemwise($wide(token), dst, a, b),
                    |dst, s| $gld_elemwise_assign($wide(token), dst, s),
                    |dst, c| $gld_addscalar($wide(token), dst, c),
                    |dst, c| $gld_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($gld_sub, |token, entry| {
                drive_gld(
                    $gld_label,
                    |dst, src| $gld_add($wide(token), dst, src),
                    |dst, src| entry($wide(token), dst, src),
                    |dst, c, src| $gld_muladd($wide(token), dst, c, src),
                    |dst, c, src| $gld_mulinto($wide(token), dst, c, src),
                    |dst, c| $gld_mulassign($wide(token), dst, c),
                    |dst, a, b| $gld_elemwise($wide(token), dst, a, b),
                    |dst, s| $gld_elemwise_assign($wide(token), dst, s),
                    |dst, c| $gld_addscalar($wide(token), dst, c),
                    |dst, c| $gld_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($gld_muladd, |token, entry| {
                drive_gld(
                    $gld_label,
                    |dst, src| $gld_add($wide(token), dst, src),
                    |dst, src| $gld_sub($wide(token), dst, src),
                    |dst, c, src| entry($wide(token), dst, c, src),
                    |dst, c, src| $gld_mulinto($wide(token), dst, c, src),
                    |dst, c| $gld_mulassign($wide(token), dst, c),
                    |dst, a, b| $gld_elemwise($wide(token), dst, a, b),
                    |dst, s| $gld_elemwise_assign($wide(token), dst, s),
                    |dst, c| $gld_addscalar($wide(token), dst, c),
                    |dst, c| $gld_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($gld_mulinto, |token, entry| {
                drive_gld(
                    $gld_label,
                    |dst, src| $gld_add($wide(token), dst, src),
                    |dst, src| $gld_sub($wide(token), dst, src),
                    |dst, c, src| $gld_muladd($wide(token), dst, c, src),
                    |dst, c, src| entry($wide(token), dst, c, src),
                    |dst, c| $gld_mulassign($wide(token), dst, c),
                    |dst, a, b| $gld_elemwise($wide(token), dst, a, b),
                    |dst, s| $gld_elemwise_assign($wide(token), dst, s),
                    |dst, c| $gld_addscalar($wide(token), dst, c),
                    |dst, c| $gld_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($gld_mulassign, |token, entry| {
                drive_gld(
                    $gld_label,
                    |dst, src| $gld_add($wide(token), dst, src),
                    |dst, src| $gld_sub($wide(token), dst, src),
                    |dst, c, src| $gld_muladd($wide(token), dst, c, src),
                    |dst, c, src| $gld_mulinto($wide(token), dst, c, src),
                    |dst, c| entry($wide(token), dst, c),
                    |dst, a, b| $gld_elemwise($wide(token), dst, a, b),
                    |dst, s| $gld_elemwise_assign($wide(token), dst, s),
                    |dst, c| $gld_addscalar($wide(token), dst, c),
                    |dst, c| $gld_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($gld_elemwise, |token, entry| {
                drive_gld(
                    $gld_label,
                    |dst, src| $gld_add($wide(token), dst, src),
                    |dst, src| $gld_sub($wide(token), dst, src),
                    |dst, c, src| $gld_muladd($wide(token), dst, c, src),
                    |dst, c, src| $gld_mulinto($wide(token), dst, c, src),
                    |dst, c| $gld_mulassign($wide(token), dst, c),
                    |dst, a, b| entry($wide(token), dst, a, b),
                    |dst, s| $gld_elemwise_assign($wide(token), dst, s),
                    |dst, c| $gld_addscalar($wide(token), dst, c),
                    |dst, c| $gld_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($gld_elemwise_assign, |token, entry| {
                drive_gld(
                    $gld_label,
                    |dst, src| $gld_add($wide(token), dst, src),
                    |dst, src| $gld_sub($wide(token), dst, src),
                    |dst, c, src| $gld_muladd($wide(token), dst, c, src),
                    |dst, c, src| $gld_mulinto($wide(token), dst, c, src),
                    |dst, c| $gld_mulassign($wide(token), dst, c),
                    |dst, a, b| $gld_elemwise($wide(token), dst, a, b),
                    |dst, s| entry($wide(token), dst, s),
                    |dst, c| $gld_addscalar($wide(token), dst, c),
                    |dst, c| $gld_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($gld_addscalar, |token, entry| {
                drive_gld(
                    $gld_label,
                    |dst, src| $gld_add($wide(token), dst, src),
                    |dst, src| $gld_sub($wide(token), dst, src),
                    |dst, c, src| $gld_muladd($wide(token), dst, c, src),
                    |dst, c, src| $gld_mulinto($wide(token), dst, c, src),
                    |dst, c| $gld_mulassign($wide(token), dst, c),
                    |dst, a, b| $gld_elemwise($wide(token), dst, a, b),
                    |dst, s| $gld_elemwise_assign($wide(token), dst, s),
                    |dst, c| entry($wide(token), dst, c),
                    |dst, c| $gld_subscalar($wide(token), dst, c),
                );
            });
            tier_gate::run_entry($gld_subscalar, |token, entry| {
                drive_gld(
                    $gld_label,
                    |dst, src| $gld_add($wide(token), dst, src),
                    |dst, src| $gld_sub($wide(token), dst, src),
                    |dst, c, src| $gld_muladd($wide(token), dst, c, src),
                    |dst, c, src| $gld_mulinto($wide(token), dst, c, src),
                    |dst, c| $gld_mulassign($wide(token), dst, c),
                    |dst, a, b| $gld_elemwise($wide(token), dst, a, b),
                    |dst, s| $gld_elemwise_assign($wide(token), dst, s),
                    |dst, c| $gld_addscalar($wide(token), dst, c),
                    |dst, c| entry($wide(token), dst, c),
                );
            });
        };
    }

    canonicalize_tier!(
        x86::mersenne31::add_assign_avx2,
        x86::mersenne31::sub_assign_avx2,
        x86::mersenne31::mul_add_avx2,
        x86::mersenne31::mul_into_avx2,
        x86::mersenne31::mul_assign_avx2,
        x86::mersenne31::mul_elementwise_avx2,
        x86::mersenne31::mul_elementwise_assign_avx2,
        x86::mersenne31::add_assign_scalar_avx2,
        x86::mersenne31::sub_assign_scalar_avx2,
        x86::goldilocks::add_assign_avx2,
        x86::goldilocks::sub_assign_avx2,
        x86::goldilocks::mul_add_avx2,
        x86::goldilocks::mul_into_avx2,
        x86::goldilocks::mul_assign_avx2,
        x86::goldilocks::mul_elementwise_avx2,
        x86::goldilocks::mul_elementwise_assign_avx2,
        x86::goldilocks::add_assign_scalar_avx2,
        x86::goldilocks::sub_assign_scalar_avx2,
        "m31 avx2 noncanonical",
        "gld avx2 noncanonical",
        |token: X64V3Token| token
    );
    tier_gate::run_entry(x86::mersenne31::add_assign_avx2, |token, entry| {
        drive_m31_pairs(
            "m31 avx2 pairs",
            |dst, src| entry(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_avx2(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_avx2(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_avx2(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::sub_assign_avx2, |token, entry| {
        drive_m31_pairs(
            "m31 avx2 pairs",
            |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
            |dst, src| entry(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_avx2(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_avx2(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_avx2(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_add_avx2, |token, entry| {
        drive_m31_pairs(
            "m31 avx2 pairs",
            |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
            |dst, c, src| entry(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_avx2(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_avx2(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_into_avx2, |token, entry| {
        drive_m31_pairs(
            "m31 avx2 pairs",
            |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_avx2(token, dst, c, src),
            |dst, c, src| entry(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_avx2(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_assign_avx2, |token, entry| {
        drive_m31_pairs(
            "m31 avx2 pairs",
            |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_avx2(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_avx2(token, dst, c, src),
            |dst, c| entry(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_elementwise_avx2, |token, entry| {
        drive_m31_pairs(
            "m31 avx2 pairs",
            |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_avx2(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_avx2(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_avx2(token, dst, c),
            |dst, a, b| entry(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
        );
    });
    tier_gate::run_entry(
        x86::mersenne31::mul_elementwise_assign_avx2,
        |token, entry| {
            drive_m31_pairs(
                "m31 avx2 pairs",
                |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
                |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
                |dst, c, src| x86::mersenne31::mul_add_avx2(token, dst, c, src),
                |dst, c, src| x86::mersenne31::mul_into_avx2(token, dst, c, src),
                |dst, c| x86::mersenne31::mul_assign_avx2(token, dst, c),
                |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
                |dst, s| entry(token, dst, s),
                |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
                |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
            );
        },
    );
    tier_gate::run_entry(x86::mersenne31::add_assign_scalar_avx2, |token, entry| {
        drive_m31_pairs(
            "m31 avx2 pairs",
            |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_avx2(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_avx2(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_avx2(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| entry(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_avx2(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::sub_assign_scalar_avx2, |token, entry| {
        drive_m31_pairs(
            "m31 avx2 pairs",
            |dst, src| x86::mersenne31::add_assign_avx2(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_avx2(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_avx2(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_avx2(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_avx2(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_avx2(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_avx2(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_avx2(token, dst, c),
            |dst, c| entry(token, dst, c),
        );
    });

    canonicalize_tier!(
        x86::mersenne31::add_assign_sse42,
        x86::mersenne31::sub_assign_sse42,
        x86::mersenne31::mul_add_sse42,
        x86::mersenne31::mul_into_sse42,
        x86::mersenne31::mul_assign_sse42,
        x86::mersenne31::mul_elementwise_sse42,
        x86::mersenne31::mul_elementwise_assign_sse42,
        x86::mersenne31::add_assign_scalar_sse42,
        x86::mersenne31::sub_assign_scalar_sse42,
        x86::goldilocks::add_assign_sse42,
        x86::goldilocks::sub_assign_sse42,
        x86::goldilocks::mul_add_sse42,
        x86::goldilocks::mul_into_sse42,
        x86::goldilocks::mul_assign_sse42,
        x86::goldilocks::mul_elementwise_sse42,
        x86::goldilocks::mul_elementwise_assign_sse42,
        x86::goldilocks::add_assign_scalar_sse42,
        x86::goldilocks::sub_assign_scalar_sse42,
        "m31 sse4.2 noncanonical",
        "gld sse4.2 noncanonical",
        |token: X64V2Token| token
    );
    tier_gate::run_entry(x86::mersenne31::add_assign_sse42, |token, entry| {
        drive_m31_pairs(
            "m31 sse4.2 pairs",
            |dst, src| entry(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_sse42(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_sse42(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_sse42(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::sub_assign_sse42, |token, entry| {
        drive_m31_pairs(
            "m31 sse4.2 pairs",
            |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
            |dst, src| entry(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_sse42(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_sse42(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_sse42(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_add_sse42, |token, entry| {
        drive_m31_pairs(
            "m31 sse4.2 pairs",
            |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
            |dst, c, src| entry(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_sse42(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_sse42(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_into_sse42, |token, entry| {
        drive_m31_pairs(
            "m31 sse4.2 pairs",
            |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_sse42(token, dst, c, src),
            |dst, c, src| entry(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_sse42(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_assign_sse42, |token, entry| {
        drive_m31_pairs(
            "m31 sse4.2 pairs",
            |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_sse42(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_sse42(token, dst, c, src),
            |dst, c| entry(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::mul_elementwise_sse42, |token, entry| {
        drive_m31_pairs(
            "m31 sse4.2 pairs",
            |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_sse42(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_sse42(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_sse42(token, dst, c),
            |dst, a, b| entry(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
        );
    });
    tier_gate::run_entry(
        x86::mersenne31::mul_elementwise_assign_sse42,
        |token, entry| {
            drive_m31_pairs(
                "m31 sse4.2 pairs",
                |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
                |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
                |dst, c, src| x86::mersenne31::mul_add_sse42(token, dst, c, src),
                |dst, c, src| x86::mersenne31::mul_into_sse42(token, dst, c, src),
                |dst, c| x86::mersenne31::mul_assign_sse42(token, dst, c),
                |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
                |dst, s| entry(token, dst, s),
                |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
                |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
            );
        },
    );
    tier_gate::run_entry(x86::mersenne31::add_assign_scalar_sse42, |token, entry| {
        drive_m31_pairs(
            "m31 sse4.2 pairs",
            |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_sse42(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_sse42(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_sse42(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| entry(token, dst, c),
            |dst, c| x86::mersenne31::sub_assign_scalar_sse42(token, dst, c),
        );
    });
    tier_gate::run_entry(x86::mersenne31::sub_assign_scalar_sse42, |token, entry| {
        drive_m31_pairs(
            "m31 sse4.2 pairs",
            |dst, src| x86::mersenne31::add_assign_sse42(token, dst, src),
            |dst, src| x86::mersenne31::sub_assign_sse42(token, dst, src),
            |dst, c, src| x86::mersenne31::mul_add_sse42(token, dst, c, src),
            |dst, c, src| x86::mersenne31::mul_into_sse42(token, dst, c, src),
            |dst, c| x86::mersenne31::mul_assign_sse42(token, dst, c),
            |dst, a, b| x86::mersenne31::mul_elementwise_sse42(token, dst, a, b),
            |dst, s| x86::mersenne31::mul_elementwise_assign_sse42(token, dst, s),
            |dst, c| x86::mersenne31::add_assign_scalar_sse42(token, dst, c),
            |dst, c| entry(token, dst, c),
        );
    });
    #[cfg(feature = "simd512")]
    {
        canonicalize_tier!(
            x86::mersenne31::add_assign_avx512,
            x86::mersenne31::sub_assign_avx512,
            x86::mersenne31::mul_add_avx512,
            x86::mersenne31::mul_into_avx512,
            x86::mersenne31::mul_assign_avx512,
            x86::mersenne31::mul_elementwise_avx512,
            x86::mersenne31::mul_elementwise_assign_avx512,
            x86::mersenne31::add_assign_scalar_avx512,
            x86::mersenne31::sub_assign_scalar_avx512,
            x86::goldilocks::add_assign_avx512,
            x86::goldilocks::sub_assign_avx512,
            x86::goldilocks::mul_add_avx512,
            x86::goldilocks::mul_into_avx512,
            x86::goldilocks::mul_assign_avx512,
            x86::goldilocks::mul_elementwise_avx512,
            x86::goldilocks::mul_elementwise_assign_avx512,
            x86::goldilocks::add_assign_scalar_avx512,
            x86::goldilocks::sub_assign_scalar_avx512,
            "m31 avx512 noncanonical",
            "gld avx512 noncanonical",
            |token: X64V4xToken| token.v4()
        );
        tier_gate::run_entry(
            x86::mersenne31::add_assign_avx512,
            |token: X64V4xToken, entry| {
                drive_m31_pairs(
                    "m31 avx512 pairs",
                    |dst, src| entry(token.v4(), dst, src),
                    |dst, src| x86::mersenne31::sub_assign_avx512(token.v4(), dst, src),
                    |dst, c, src| x86::mersenne31::mul_add_avx512(token.v4(), dst, c, src),
                    |dst, c, src| x86::mersenne31::mul_into_avx512(token.v4(), dst, c, src),
                    |dst, c| x86::mersenne31::mul_assign_avx512(token.v4(), dst, c),
                    |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token.v4(), dst, a, b),
                    |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token.v4(), dst, s),
                    |dst, c| x86::mersenne31::add_assign_scalar_avx512(token.v4(), dst, c),
                    |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token.v4(), dst, c),
                );
            },
        );
        tier_gate::run_entry(
            x86::mersenne31::sub_assign_avx512,
            |token: X64V4xToken, entry| {
                drive_m31_pairs(
                    "m31 avx512 pairs",
                    |dst, src| x86::mersenne31::add_assign_avx512(token.v4(), dst, src),
                    |dst, src| entry(token.v4(), dst, src),
                    |dst, c, src| x86::mersenne31::mul_add_avx512(token.v4(), dst, c, src),
                    |dst, c, src| x86::mersenne31::mul_into_avx512(token.v4(), dst, c, src),
                    |dst, c| x86::mersenne31::mul_assign_avx512(token.v4(), dst, c),
                    |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token.v4(), dst, a, b),
                    |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token.v4(), dst, s),
                    |dst, c| x86::mersenne31::add_assign_scalar_avx512(token.v4(), dst, c),
                    |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token.v4(), dst, c),
                );
            },
        );
        tier_gate::run_entry(
            x86::mersenne31::mul_add_avx512,
            |token: X64V4xToken, entry| {
                drive_m31_pairs(
                    "m31 avx512 pairs",
                    |dst, src| x86::mersenne31::add_assign_avx512(token.v4(), dst, src),
                    |dst, src| x86::mersenne31::sub_assign_avx512(token.v4(), dst, src),
                    |dst, c, src| entry(token.v4(), dst, c, src),
                    |dst, c, src| x86::mersenne31::mul_into_avx512(token.v4(), dst, c, src),
                    |dst, c| x86::mersenne31::mul_assign_avx512(token.v4(), dst, c),
                    |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token.v4(), dst, a, b),
                    |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token.v4(), dst, s),
                    |dst, c| x86::mersenne31::add_assign_scalar_avx512(token.v4(), dst, c),
                    |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token.v4(), dst, c),
                );
            },
        );
        tier_gate::run_entry(
            x86::mersenne31::mul_into_avx512,
            |token: X64V4xToken, entry| {
                drive_m31_pairs(
                    "m31 avx512 pairs",
                    |dst, src| x86::mersenne31::add_assign_avx512(token.v4(), dst, src),
                    |dst, src| x86::mersenne31::sub_assign_avx512(token.v4(), dst, src),
                    |dst, c, src| x86::mersenne31::mul_add_avx512(token.v4(), dst, c, src),
                    |dst, c, src| entry(token.v4(), dst, c, src),
                    |dst, c| x86::mersenne31::mul_assign_avx512(token.v4(), dst, c),
                    |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token.v4(), dst, a, b),
                    |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token.v4(), dst, s),
                    |dst, c| x86::mersenne31::add_assign_scalar_avx512(token.v4(), dst, c),
                    |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token.v4(), dst, c),
                );
            },
        );
        tier_gate::run_entry(
            x86::mersenne31::mul_assign_avx512,
            |token: X64V4xToken, entry| {
                drive_m31_pairs(
                    "m31 avx512 pairs",
                    |dst, src| x86::mersenne31::add_assign_avx512(token.v4(), dst, src),
                    |dst, src| x86::mersenne31::sub_assign_avx512(token.v4(), dst, src),
                    |dst, c, src| x86::mersenne31::mul_add_avx512(token.v4(), dst, c, src),
                    |dst, c, src| x86::mersenne31::mul_into_avx512(token.v4(), dst, c, src),
                    |dst, c| entry(token.v4(), dst, c),
                    |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token.v4(), dst, a, b),
                    |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token.v4(), dst, s),
                    |dst, c| x86::mersenne31::add_assign_scalar_avx512(token.v4(), dst, c),
                    |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token.v4(), dst, c),
                );
            },
        );
        tier_gate::run_entry(
            x86::mersenne31::mul_elementwise_avx512,
            |token: X64V4xToken, entry| {
                drive_m31_pairs(
                    "m31 avx512 pairs",
                    |dst, src| x86::mersenne31::add_assign_avx512(token.v4(), dst, src),
                    |dst, src| x86::mersenne31::sub_assign_avx512(token.v4(), dst, src),
                    |dst, c, src| x86::mersenne31::mul_add_avx512(token.v4(), dst, c, src),
                    |dst, c, src| x86::mersenne31::mul_into_avx512(token.v4(), dst, c, src),
                    |dst, c| x86::mersenne31::mul_assign_avx512(token.v4(), dst, c),
                    |dst, a, b| entry(token.v4(), dst, a, b),
                    |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token.v4(), dst, s),
                    |dst, c| x86::mersenne31::add_assign_scalar_avx512(token.v4(), dst, c),
                    |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token.v4(), dst, c),
                );
            },
        );
        tier_gate::run_entry(
            x86::mersenne31::mul_elementwise_assign_avx512,
            |token: X64V4xToken, entry| {
                drive_m31_pairs(
                    "m31 avx512 pairs",
                    |dst, src| x86::mersenne31::add_assign_avx512(token.v4(), dst, src),
                    |dst, src| x86::mersenne31::sub_assign_avx512(token.v4(), dst, src),
                    |dst, c, src| x86::mersenne31::mul_add_avx512(token.v4(), dst, c, src),
                    |dst, c, src| x86::mersenne31::mul_into_avx512(token.v4(), dst, c, src),
                    |dst, c| x86::mersenne31::mul_assign_avx512(token.v4(), dst, c),
                    |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token.v4(), dst, a, b),
                    |dst, s| entry(token.v4(), dst, s),
                    |dst, c| x86::mersenne31::add_assign_scalar_avx512(token.v4(), dst, c),
                    |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token.v4(), dst, c),
                );
            },
        );
        tier_gate::run_entry(
            x86::mersenne31::add_assign_scalar_avx512,
            |token: X64V4xToken, entry| {
                drive_m31_pairs(
                    "m31 avx512 pairs",
                    |dst, src| x86::mersenne31::add_assign_avx512(token.v4(), dst, src),
                    |dst, src| x86::mersenne31::sub_assign_avx512(token.v4(), dst, src),
                    |dst, c, src| x86::mersenne31::mul_add_avx512(token.v4(), dst, c, src),
                    |dst, c, src| x86::mersenne31::mul_into_avx512(token.v4(), dst, c, src),
                    |dst, c| x86::mersenne31::mul_assign_avx512(token.v4(), dst, c),
                    |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token.v4(), dst, a, b),
                    |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token.v4(), dst, s),
                    |dst, c| entry(token.v4(), dst, c),
                    |dst, c| x86::mersenne31::sub_assign_scalar_avx512(token.v4(), dst, c),
                );
            },
        );
        tier_gate::run_entry(
            x86::mersenne31::sub_assign_scalar_avx512,
            |token: X64V4xToken, entry| {
                drive_m31_pairs(
                    "m31 avx512 pairs",
                    |dst, src| x86::mersenne31::add_assign_avx512(token.v4(), dst, src),
                    |dst, src| x86::mersenne31::sub_assign_avx512(token.v4(), dst, src),
                    |dst, c, src| x86::mersenne31::mul_add_avx512(token.v4(), dst, c, src),
                    |dst, c, src| x86::mersenne31::mul_into_avx512(token.v4(), dst, c, src),
                    |dst, c| x86::mersenne31::mul_assign_avx512(token.v4(), dst, c),
                    |dst, a, b| x86::mersenne31::mul_elementwise_avx512(token.v4(), dst, a, b),
                    |dst, s| x86::mersenne31::mul_elementwise_assign_avx512(token.v4(), dst, s),
                    |dst, c| x86::mersenne31::add_assign_scalar_avx512(token.v4(), dst, c),
                    |dst, c| entry(token.v4(), dst, c),
                );
            },
        );
    }
}
/// The AVX-512 Mersenne multiplies screen operand lanes for bit 31:
/// per register group for the in-place entries, per call for the
/// out-of-place entries. A single high-bit alias at every dword lane
/// of one complete screened group, swept across the group's registers
/// in each operand role, plus a lane past the last whole group, the
/// sub-lane tail, and raw-alias coefficients for the M31 scalar
/// multiplies, must leave every lane equal to the portable reference
/// and canonical.
#[cfg(feature = "simd512")]
#[test]
// Five m31 and three qm31 entries per alias placement: a table by nature.
#[allow(clippy::too_many_lines)]
fn mersenne_avx512_screen_reaches_every_lane() {
    use crate::field::quad_mersenne31::QuadMersenne31;
    const P: u32 = crate::field::mersenne31::MODULUS;
    // Forty-nine 64-byte lanes and a sub-lane tail of one QM31 element.
    const LEN: usize = 3 * 1024 + 72;
    // Word positions: one complete screened group (64 dword lanes,
    // registers 0-3) swept lane by lane, the lane past the last whole
    // group, and the tail word; `None` is the all-canonical control.
    // Positions 300 and 767 from the earlier spot sweep lie inside that
    // group and stay covered by the sweep.
    const GROUP_WORDS: usize = 64;
    const AT_EXTRA: [Option<usize>; 3] = [None, Some(770), Some(784)];
    // `p`, then patterns whose bit 31 breaks the doubled operand or,
    // against a large partner, overflows the product split.
    const ALIASES: [u32; 5] = [P, 0x8000_0000, 0xE000_0000, 0xFFFF_FFFE, u32::MAX];
    const RAW_COEFFS: [u32; 5] = [P, 0x8000_0000, 0xE000_0000, 0xFFFF_FFFE, u32::MAX];
    let canonical = |seed: u64| -> Vec<u8> {
        let mut out = noise(LEN, seed);
        for lane in out.chunks_exact_mut(4) {
            let word = u32::from_le_bytes(lane.try_into().unwrap());
            lane.copy_from_slice(&Elem::<Mersenne31>::from_raw(word).to_raw().to_le_bytes());
        }
        out
    };
    // The alias lane meets the largest canonical word in the other
    // operand, the partner that drives the product split highest.
    let place = |mut buf: Vec<u8>, at: Option<usize>, word: u32| -> Vec<u8> {
        if let Some(index) = at {
            buf[4 * index..4 * index + 4].copy_from_slice(&word.to_le_bytes());
        }
        buf
    };
    let check = |label: &str, got: &[u8], want: &[u8]| {
        assert_eq!(got, want, "{label}");
        for (i, lane) in got.chunks_exact(4).enumerate() {
            let word = u32::from_le_bytes(lane.try_into().unwrap());
            assert!(word < P, "{label}: lane {i} holds {word:#x}");
        }
    };
    let coeff = P - 1;
    let m31 = Elem::<Mersenne31>::from_raw(coeff);
    let qm31 = Elem::<QuadMersenne31>::from_raw([coeff, P - 2]);
    for at in AT_EXTRA.into_iter().chain((0..GROUP_WORDS).map(Some)) {
        for alias in ALIASES {
            for (role, (x, y)) in [
                (
                    "first",
                    (
                        place(canonical(0x51), at, alias),
                        place(canonical(0x52), at, P - 1),
                    ),
                ),
                (
                    "second",
                    (
                        place(canonical(0x51), at, P - 1),
                        place(canonical(0x52), at, alias),
                    ),
                ),
            ] {
                let label = |op: &str| format!("{op} alias {alias:#x} at {at:?} in {role}");

                let mut want = x.clone();
                tier_gate::run_entry(
                    x86::mersenne31::mul_add_avx512,
                    |token: X64V4xToken, entry| {
                        let mut got = x.clone();
                        entry(token.v4(), &mut got, coeff, &y);
                        prime::mul_add::<Mersenne31>(&mut want, m31, &y);
                        check(&label("m31 mul_add"), &got, &want);
                    },
                );
                tier_gate::run_entry(
                    x86::mersenne31::mul_into_avx512,
                    |token: X64V4xToken, entry| {
                        let mut got = vec![0u8; LEN];
                        let mut want = vec![0u8; LEN];
                        entry(token.v4(), &mut got, coeff, &y);
                        prime::mul_add::<Mersenne31>(&mut want, m31, &y);
                        check(&label("m31 mul_into"), &got, &want);
                    },
                );
                tier_gate::run_entry(
                    x86::mersenne31::mul_assign_avx512,
                    |token: X64V4xToken, entry| {
                        let mut got = x.clone();
                        let mut want = x.clone();
                        entry(token.v4(), &mut got, coeff);
                        prime::mul_assign::<Mersenne31>(&mut want, m31);
                        check(&label("m31 mul_assign"), &got, &want);
                    },
                );
                tier_gate::run_entry(
                    x86::mersenne31::mul_elementwise_avx512,
                    |token: X64V4xToken, entry| {
                        let mut got = vec![0u8; LEN];
                        let mut want = vec![0u8; LEN];
                        entry(token.v4(), &mut got, &x, &y);
                        prime::mul_elementwise::<Mersenne31>(&mut want, &x, &y);
                        check(&label("m31 mul_elementwise"), &got, &want);
                    },
                );
                let mut want = x.clone();
                prime::mul_elementwise::<Mersenne31>(&mut want, &x, &y);
                tier_gate::run_entry(
                    x86::mersenne31::mul_elementwise_assign_avx512,
                    |token: X64V4xToken, entry| {
                        let mut got = x.clone();
                        entry(token.v4(), &mut got, &y);
                        check(&label("m31 mul_elementwise_assign"), &got, &want);
                    },
                );
                let mut want = x.clone();
                tier_gate::run_entry(
                    x86::quad_mersenne31::mul_add_avx512,
                    |token: X64V4xToken, entry| {
                        let mut got = x.clone();
                        entry(token.v4(), &mut got, qm31, &y);
                        prime::mul_add::<QuadMersenne31>(&mut want, qm31, &y);
                        check(&label("qm31 mul_add"), &got, &want);
                    },
                );
                tier_gate::run_entry(
                    x86::quad_mersenne31::mul_into_avx512,
                    |token: X64V4xToken, entry| {
                        let mut got = vec![0u8; LEN];
                        let mut want = vec![0u8; LEN];
                        entry(token.v4(), &mut got, qm31, &y);
                        prime::mul_add::<QuadMersenne31>(&mut want, qm31, &y);
                        check(&label("qm31 mul_into"), &got, &want);
                    },
                );
                let mut want = vec![0u8; LEN];
                prime::mul_elementwise::<QuadMersenne31>(&mut want, &x, &y);
                tier_gate::run_entry(
                    x86::quad_mersenne31::mul_elementwise_avx512,
                    |token: X64V4xToken, entry| {
                        let mut got = vec![0u8; LEN];
                        entry(token.v4(), &mut got, &x, &y);
                        check(&label("qm31 mul_elementwise"), &got, &want);
                    },
                );

                for raw in RAW_COEFFS {
                    let rc = Elem::<Mersenne31>::from_raw(raw);
                    // A zero- or unit-valued raw coefficient is a defined
                    // no-op mirroring the portable reference (`mul_assign`
                    // preserves the buffer byte for byte, `mul_into`
                    // zeroes or copies it), so only equality applies.
                    // Otherwise every stored lane must be canonical.
                    let check_canonical = !rc.is_zero() && !rc.is_one();
                    let run_coeff = |op: &str, got: &[u8], want: &[u8]| {
                        let coined = label(&format!("{op} coeff {raw:#x}"));
                        assert_eq!(got, want, "{coined}");
                        if check_canonical {
                            for (i, lane) in got.chunks_exact(4).enumerate() {
                                let word = u32::from_le_bytes(lane.try_into().unwrap());
                                assert!(word < P, "{coined}: lane {i} holds {word:#x}");
                            }
                        }
                    };
                    tier_gate::run_entry(
                        x86::mersenne31::mul_assign_avx512,
                        |token: X64V4xToken, entry| {
                            let mut got = x.clone();
                            let mut want = x.clone();
                            entry(token.v4(), &mut got, raw);
                            prime::mul_assign::<Mersenne31>(&mut want, rc);
                            run_coeff("m31 mul_assign", &got, &want);
                        },
                    );
                    tier_gate::run_entry(
                        x86::mersenne31::mul_into_avx512,
                        |token: X64V4xToken, entry| {
                            let mut got = vec![0u8; LEN];
                            let mut want = vec![0u8; LEN];
                            entry(token.v4(), &mut got, raw, &y);
                            prime::mul_add::<Mersenne31>(&mut want, rc, &y);
                            run_coeff("m31 mul_into", &got, &want);
                        },
                    );
                }
            }
        }
    }
}
