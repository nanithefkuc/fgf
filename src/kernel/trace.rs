//! Test-only append-only trace of vector work.
//!
//! Emission sites are `#[cfg(test)]` statements placed after the vector loop
//! they describe. Zero vector work emits nothing. Every entry records the
//! instruction tier of the loop that ran, semantic substage, unit kind, and
//! innermost enclosing scope. The trace does not influence dispatch or control
//! flow and carries no production counters.

extern crate std;

use std::cell::RefCell;
use std::vec::Vec;

use crate::kernel::Backend;

/// What one counted [`Entry::work`] unit means.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub(crate) enum Unit {
    /// Field elements processed through vector lanes.
    Elements,
    /// Bytes moved or transformed through vector lanes.
    Bytes,
    /// Bit lanes processed through vector lanes (bit-packed GF(2)).
    Bits,
    /// Logical words transformed through vector lanes.
    Words,
}

/// Semantic substage of the arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub(crate) enum Substage {
    Add,
    Sub,
    Mul,
    MulAdd,
    MulElementwise,
    Inverse,
    Dot,
    Linear,
    Pack,
    Unpack,
    BitAnd,
    BitAndNot,
    BitXor,
    BitWeight,
    BitParity,
}

/// Semantic enclosing operation for composed bodies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
pub(crate) enum Scope {
    Inverse,
    Division,
    Frobenius,
    Norm,
    Trace,
    Restriction,
    Membership,
}

/// One recorded stretch of vector work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    /// Emitting kernel identity; diagnostics only.
    pub kernel: &'static str,
    /// Semantic substage of the arithmetic.
    pub substage: Substage,
    /// Innermost enclosing scope at emission, if any.
    pub scope: Option<Scope>,
    /// Meaning of `work`.
    pub unit: Unit,
    /// Work the vector loop processed; always positive.
    pub work: usize,
    /// Instruction tier of the loop that ran.
    pub tier: Backend,
}

std::thread_local! {
    static ENTRIES: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
    static SCOPE: RefCell<Vec<Scope>> = const { RefCell::new(Vec::new()) };
}

/// Discard this thread's recorded entries and scope path.
pub(crate) fn clear() {
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    SCOPE.with(|scope| scope.borrow_mut().clear());
}

/// Record one stretch of vector work. Zero work records nothing.
pub(crate) fn emit(
    kernel: &'static str,
    substage: Substage,
    unit: Unit,
    work: usize,
    tier: Backend,
) {
    if work == 0 {
        return;
    }
    let scope = SCOPE.with(|scope| scope.borrow().last().copied());
    ENTRIES.with(|entries| {
        entries.borrow_mut().push(Entry {
            kernel,
            substage,
            scope,
            unit,
            work,
            tier,
        });
    });
}

/// Snapshot of this thread's entries in emission order.
pub(crate) fn snapshot() -> Vec<Entry> {
    ENTRIES.with(|entries| entries.borrow().clone())
}

/// Push a semantic enclosing-operation scope until the returned guard drops.
///
/// The guard pops the scope on drop, including during unwind.
#[must_use]
pub(crate) fn scope(scope: Scope) -> ScopeGuard {
    SCOPE.with(|scopes| scopes.borrow_mut().push(scope));
    ScopeGuard {}
}

/// Guard returned by [`scope`]; pops the scope path on drop.
pub(crate) struct ScopeGuard {}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPE.with(|scopes| {
            scopes.borrow_mut().pop();
        });
    }
}

/// Numeric strength of a tier, higher is stronger.
///
/// Scalar reports 0; other x86 tiers order by capability. Non-x86 variants
/// report scalar strength.
pub(crate) fn strength(tier: Backend) -> u8 {
    match tier {
        Backend::V4x => 4,
        Backend::V3GfniCrypto => 3,
        Backend::V3 => 2,
        Backend::V2 => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::Scope;
    use super::{Backend, Unit, clear, emit, scope, snapshot, strength};

    #[test]
    fn emission_order_is_preserved() {
        clear();
        emit("k", super::Substage::Mul, Unit::Elements, 4, Backend::V3);
        emit("k", super::Substage::Add, Unit::Elements, 2, Backend::V2);
        emit(
            "k",
            super::Substage::MulAdd,
            Unit::Bytes,
            16,
            Backend::V3GfniCrypto,
        );
        let entries = snapshot();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].substage, super::Substage::Mul);
        assert_eq!(entries[1].substage, super::Substage::Add);
        assert_eq!(entries[2].substage, super::Substage::MulAdd);
    }

    #[test]
    fn zero_work_records_nothing() {
        clear();
        emit("k", super::Substage::Mul, Unit::Elements, 0, Backend::V3);
        assert!(snapshot().is_empty());
    }

    #[test]
    fn clear_empties_the_trace() {
        emit("k", super::Substage::Mul, Unit::Elements, 8, Backend::V3);
        assert_eq!(snapshot().len(), 1);
        clear();
        assert!(snapshot().is_empty());
        emit("k", super::Substage::Add, Unit::Bytes, 4, Backend::V2);
        assert_eq!(snapshot().len(), 1);
    }

    #[test]
    fn scope_attaches_innermost_and_pops_on_drop() {
        clear();
        let _outer = scope(Scope::Inverse);
        emit("k", super::Substage::Mul, Unit::Elements, 4, Backend::V3);
        {
            let _inner = scope(Scope::Frobenius);
            emit(
                "k",
                super::Substage::MulAdd,
                Unit::Elements,
                8,
                Backend::V3GfniCrypto,
            );
        }
        emit("k", super::Substage::Add, Unit::Elements, 2, Backend::V2);
        let entries = snapshot();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].scope, Some(Scope::Inverse));
        assert_eq!(entries[1].scope, Some(Scope::Frobenius));
        assert_eq!(entries[2].scope, Some(Scope::Inverse));
    }

    #[test]
    fn scope_pops_on_unwind() {
        clear();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = scope(Scope::Inverse);
            emit("k", super::Substage::Mul, Unit::Elements, 4, Backend::V3);
            panic!("unwind through active scope guard");
        }));
        assert!(caught.is_err());
        // Emission inside the guarded body was recorded with the scope.
        let entries = snapshot();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].scope, Some(Scope::Inverse));
        // Post-unwind emission has no scope: the guard dropped during unwind.
        emit("k", super::Substage::Add, Unit::Elements, 2, Backend::V2);
        let entries = snapshot();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].scope, None);
    }

    #[test]
    fn entries_from_another_thread_are_invisible() {
        clear();
        emit("k", super::Substage::Mul, Unit::Elements, 4, Backend::V3);

        let handle = std::thread::spawn(|| {
            emit("k", super::Substage::Add, Unit::Bytes, 8, Backend::V2);
            snapshot()
        });
        let thread_entries = handle.join().unwrap();
        assert_eq!(thread_entries.len(), 1);
        assert_eq!(thread_entries[0].substage, super::Substage::Add);

        // The main thread does not see the spawned thread's entries.
        let main_entries = snapshot();
        assert_eq!(main_entries.len(), 1);
        assert_eq!(main_entries[0].substage, super::Substage::Mul);
    }

    #[test]
    fn strength_orders_x86_tiers() {
        assert!(strength(Backend::V4x) > strength(Backend::V3GfniCrypto));
        assert!(strength(Backend::V3GfniCrypto) > strength(Backend::V3));
        assert!(strength(Backend::V3) > strength(Backend::V2));
        assert!(strength(Backend::V2) > strength(Backend::Scalar));
    }

    #[test]
    fn strength_reports_zero_for_non_x86_variants() {
        assert_eq!(strength(Backend::Scalar), 0);
        assert_eq!(strength(Backend::V1), 0);
    }
}
