//! Request-bounded direct differential execution and entry receipts.

use super::FGF_TIERS;
use archmage::SimdToken;
use core::any::type_name;
use simdispatch::{Backend, Selection};
use std::collections::HashSet;
use std::io::Write;
use std::sync::{LazyLock, Mutex};

/// The dispatch tier required by a concrete capability token.
pub trait TokenTier: SimdToken {
    const TIER: Backend;
}

macro_rules! token_tier {
    ($token:ty, $tier:ident) => {
        impl TokenTier for $token {
            const TIER: Backend = Backend::$tier;
        }
    };
}
token_tier!(archmage::X64V1Token, V2);
token_tier!(archmage::X64V2Token, V2);
token_tier!(archmage::X64V3Token, V3);
token_tier!(archmage::X64V3GfniCryptoToken, V3GfniCrypto);
#[cfg(feature = "simd512")]
token_tier!(archmage::X64V4Token, V4x);
#[cfg(feature = "simd512")]
token_tier!(archmage::X64V4xToken, V4x);

/// Whether the request ceiling includes the entry's capability tier.
pub fn tier_allows(tier: Backend, ceiling: Backend) -> bool {
    // Backend orders stronger capabilities first.
    tier >= ceiling
}

static REQUEST_CEILING: LazyLock<Backend> = LazyLock::new(|| {
    const HARDWARE_ENV: &str = "FGF_TEST_HARDWARE_CEILING_DO_NOT_SET";
    assert!(
        std::env::var_os(HARDWARE_ENV).is_none(),
        "hardware ceiling variable must not be set"
    );
    let required = std::env::var("FGF_REQUIRE_TIER")
        .ok()
        .map(|name| match name.as_str() {
            "v4x" => Backend::V4x,
            "v3_gfni_crypto" => Backend::V3GfniCrypto,
            "v3" => Backend::V3,
            "v2" => Backend::V2,
            "scalar" => Backend::Scalar,
            _ => panic!("unsupported required tier: {name}"),
        });
    let hardware = Selection::new(HARDWARE_ENV).supports(FGF_TIERS).resolve();
    validate_required_tier(required, hardware);
    Selection::new("SIMD_BACKEND").supports(FGF_TIERS).resolve()
});

fn validate_required_tier(required: Option<Backend>, hardware: Backend) {
    if let Some(floor) = required {
        assert!(
            tier_allows(floor, hardware),
            "kernel capability missing: required={} hardware={}",
            floor.name(),
            hardware.name()
        );
    }
}

/// Materializes a token only within the host and request ceilings.
///
/// Geometry and allocation checks use this gate without earning receipts.
pub fn summon<T: TokenTier>() -> Option<T> {
    if !tier_allows(T::TIER, *REQUEST_CEILING) {
        return None;
    }
    Some(T::summon().expect("resolved differential tier must supply its capability token"))
}

#[allow(dead_code)]
static EXECUTED: LazyLock<Mutex<HashSet<&'static str>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

/// Runs an arithmetic differential and emits its receipt only after success.
///
/// The entry remains a function item, preserving its compiler definition path.
/// Repeated fixture cases emit one receipt per function-item instantiation.
#[allow(dead_code)]
pub fn run_entry<T: TokenTier, E>(entry: E, body: impl FnOnce(T, E)) {
    let Some(token) = summon::<T>() else {
        return;
    };
    let identity = type_name::<E>();
    body(token, entry);
    let mut executed = EXECUTED.lock().expect("entry receipt lock poisoned");
    if executed.insert(identity) {
        let path = definition_path(identity);
        // One framed write keeps merged libtest output from splitting a receipt.
        let receipt = format!(
            "\nkernel_differential executed tier={} entry={path}\n",
            T::TIER.name()
        );
        std::io::stderr()
            .lock()
            .write_all(receipt.as_bytes())
            .expect("writing kernel execution receipt");
    }
}

fn definition_path(name: &str) -> String {
    let mut path = String::new();
    let mut depth = 0usize;
    for ch in name.chars() {
        match ch {
            '<' => depth += 1,
            '>' => depth = depth.checked_sub(1).expect("unbalanced entry identity"),
            _ if depth == 0 => path.push(ch),
            _ => {}
        }
    }
    assert_eq!(depth, 0, "unbalanced entry identity");
    assert!(
        path.starts_with("fgf::kernel::x86::")
            && path.split("::").all(|part| {
                !part.is_empty() && part.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
            }),
        "entry identity is not a kernel definition path: {name}"
    );
    path
}

#[test]
fn ceiling_includes_only_weaker_or_equal_tiers() {
    let tiers = [
        Backend::V4x,
        Backend::V3GfniCrypto,
        Backend::V3,
        Backend::V2,
        Backend::Scalar,
    ];
    for (ceiling_index, &ceiling) in tiers.iter().enumerate() {
        for (entry_index, &entry) in tiers.iter().enumerate() {
            assert_eq!(tier_allows(entry, ceiling), entry_index >= ceiling_index);
        }
    }
}

#[test]
fn generic_entry_identity_keeps_the_definition() {
    assert_eq!(
        definition_path("fgf::kernel::x86::gf8::gfni::single::mul_add_gfni<true>"),
        "fgf::kernel::x86::gf8::gfni::single::mul_add_gfni"
    );
    assert_eq!(
        definition_path("fgf::kernel::x86::residue::avx2::mul_into_with<fgf::Row<2, 3>>"),
        "fgf::kernel::x86::residue::avx2::mul_into_with"
    );
}

#[test]
#[should_panic(expected = "not a kernel definition path")]
fn function_pointer_cannot_earn_a_receipt() {
    definition_path("fn(archmage::X64V3Token, &mut [u8])");
}

#[test]
#[should_panic(expected = "kernel capability missing")]
fn required_tier_rejects_weaker_hardware_without_an_entry() {
    validate_required_tier(Some(Backend::V4x), Backend::V3GfniCrypto);
}
