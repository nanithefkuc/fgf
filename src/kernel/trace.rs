//! Test-only append-only trace of vector work.
//!
//! Emission sites are `#[cfg(test)]` statements placed after the vector loop
//! they describe. Zero vector work emits nothing. Every entry records the
//! instruction tier of the loop that ran, the semantic substage, the unit
//! kind, and the innermost active `(scope, pass)` pair. Each independently
//! required pass of a composed operation opens its own `scope(…, pass)` guard
//! so that emissions from different passes carry different keys. The trace
//! does not influence dispatch or control flow and carries no production
//! counters.

extern crate std;

use std::cell::{Cell, RefCell};
use std::vec::Vec;

use crate::kernel::Backend;

/// What one counted [`Entry::work`] unit means.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
/// The vocabulary is closed; variants without a current emission site still
/// appear in ledger work rules, so the enum is not dead.
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
/// Closed vocabulary consumed by the route ledger; not every substage has a
/// current emission site.
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
/// Closed vocabulary consumed by the route ledger.
pub(crate) enum Scope {
    Inverse,
    Division,
    Frobenius,
    Norm,
    Trace,
    Restriction,
    Membership,
}

/// Independently required pass inside a semantic scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(dead_code)]
/// Closed vocabulary consumed by the route ledger.
pub(crate) enum Pass {
    Main,
    Prefix,
    Backward,
    Output,
    Conjugate,
    Accumulate,
    Project,
}

/// One recorded stretch of vector work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    /// Emitting kernel identity; diagnostics only.
    pub kernel: &'static str,
    /// Semantic substage of the arithmetic.
    pub substage: Substage,
    /// Innermost active scope and pass at emission, if any.
    pub scope: Option<(Scope, Pass)>,
    /// Meaning of `work`.
    pub unit: Unit,
    /// Work the vector loop processed; always positive.
    pub work: usize,
    /// Instruction tier of the loop that ran.
    pub tier: Backend,
    /// The emitting call has checked that its coefficient is neither zero nor one.
    pub general_only: bool,
}

std::thread_local! {
    static ENTRIES: RefCell<Vec<Entry>> = const { RefCell::new(Vec::new()) };
    static SCOPE: RefCell<Vec<(Scope, Pass)>> = const { RefCell::new(Vec::new()) };
    static ELEMENT_BYTES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static TERM: Cell<Option<Term>> = const { Cell::new(None) };
}

/// Discard this thread's recorded entries and scope path.
pub fn clear() {
    ENTRIES.with(|entries| entries.borrow_mut().clear());
    SCOPE.with(|scope| scope.borrow_mut().clear());
    ELEMENT_BYTES.with(|widths| widths.borrow_mut().clear());
    TERM.with(|term| term.set(None));
}

/// Record one stretch of vector work. Zero work records nothing.
///
/// Inside a [`term`] guard, `MulAdd` work takes the guard's class: unit-term
/// work records as `Add`, general-term work as qualified `MulAdd`.
#[allow(dead_code)]
pub(crate) fn emit(
    kernel: &'static str,
    substage: Substage,
    unit: Unit,
    work: usize,
    tier: Backend,
) {
    let (substage, general) = match (substage, TERM.with(Cell::get)) {
        (Substage::MulAdd, Some(Term::One)) => (Substage::Add, false),
        (Substage::MulAdd, Some(Term::General)) => (Substage::MulAdd, true),
        _ => (substage, false),
    };
    push(kernel, substage, unit, work, tier, general);
}

/// Coefficient class of one term a fused multi-row kernel multiplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum Term {
    /// The canonical zero: the term adds nothing.
    Zero,
    /// The canonical one: the term adds its source.
    One,
    /// Any other coefficient: the term multiplies.
    General,
}

impl Term {
    /// Class of a byte-field factor word: the coefficient byte itself when
    /// `native`, the coefficient's affine matrix qword otherwise.
    #[allow(dead_code)]
    pub(crate) fn of_byte_word(word: u64, native: bool) -> Self {
        match (word, native) {
            (0, _) => Self::Zero,
            (1, true) | (crate::kernel::tables::IDENTITY_AFFINE, false) => Self::One,
            _ => Self::General,
        }
    }
}

/// Record `work` per term for a fused multi-term stretch, split by class.
///
/// A fused kernel multiplies every staged term alike, so the class comes
/// from the coefficient word it multiplied by: zero-term work is unqualified
/// `MulAdd`, unit-term work is `Add` (the vector unit added the source), and
/// general-term work is `MulAdd` qualified as general.
#[allow(dead_code)]
pub(crate) fn emit_terms(
    kernel: &'static str,
    unit: Unit,
    work: usize,
    terms: impl IntoIterator<Item = Term>,
    tier: Backend,
) {
    record_terms(kernel, unit, work, terms, tier, Substage::Add);
}

/// [`emit_terms`] for the fold that seeds an overwrite: a unit seed is the
/// copy base, so its work is unqualified `MulAdd` rather than `Add`.
#[allow(dead_code)]
pub(crate) fn emit_seed_terms(
    kernel: &'static str,
    unit: Unit,
    work: usize,
    terms: impl IntoIterator<Item = Term>,
    tier: Backend,
) {
    record_terms(kernel, unit, work, terms, tier, Substage::MulAdd);
}

fn record_terms(
    kernel: &'static str,
    unit: Unit,
    work: usize,
    terms: impl IntoIterator<Item = Term>,
    tier: Backend,
    unit_substage: Substage,
) {
    let mut counts = [0usize; 3];
    for term in terms {
        counts[term as usize] += 1;
    }
    let [zero, one, general] =
        counts.map(|count| count.checked_mul(work).expect("trace term work overflow"));
    push(kernel, Substage::MulAdd, unit, zero, tier, false);
    push(kernel, unit_substage, unit, one, tier, false);
    push(kernel, Substage::MulAdd, unit, general, tier, true);
}

fn push(
    kernel: &'static str,
    substage: Substage,
    unit: Unit,
    work: usize,
    tier: Backend,
    general_only: bool,
) {
    if work == 0 {
        return;
    }
    let scope = SCOPE.with(|scope| scope.borrow().last().copied());
    let (unit, work) = if unit == Unit::Elements {
        ELEMENT_BYTES.with(|widths| match widths.borrow().last().copied() {
            Some(width) => (
                Unit::Bytes,
                work.checked_mul(width)
                    .expect("trace element byte count overflow"),
            ),
            None => (unit, work),
        })
    } else {
        (unit, work)
    };
    ENTRIES.with(|entries| {
        entries.borrow_mut().push(Entry {
            kernel,
            substage,
            scope,
            unit,
            work,
            tier,
            general_only,
        });
    });
}

/// Snapshot of this thread's entries in emission order.
pub(crate) fn snapshot() -> Vec<Entry> {
    ENTRIES.with(|entries| entries.borrow().clone())
}

/// Push a semantic scope and pass until the returned guard drops.
///
/// The guard pops the `(scope, pass)` pair on drop, including during unwind.
#[must_use]
#[allow(dead_code)]
pub(crate) fn scope(scope: Scope, pass: Pass) -> ScopeGuard {
    SCOPE.with(|scopes| scopes.borrow_mut().push((scope, pass)));
    ScopeGuard {}
}

/// Guard returned by [`scope`]; pops the scope path on drop.
#[allow(dead_code)]
pub(crate) struct ScopeGuard {}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPE.with(|scopes| {
            scopes.borrow_mut().pop();
        });
    }
}

/// Express nested element-kernel emissions as bytes of their actual layout.
///
/// Component compositions may use a base field with a different element
/// width. The innermost guard supplies that base width; byte emissions
/// already carry a layout-independent count and remain unchanged.
#[must_use]
pub(crate) fn element_bytes(width: usize) -> ElementBytesGuard {
    assert!(width > 0, "trace element width must be positive");
    ELEMENT_BYTES.with(|widths| widths.borrow_mut().push(width));
    ElementBytesGuard {}
}

/// Restores the enclosing element width when a component call returns.
pub(crate) struct ElementBytesGuard {}

impl Drop for ElementBytesGuard {
    fn drop(&mut self) {
        ELEMENT_BYTES.with(|widths| {
            widths.borrow_mut().pop();
        });
    }
}

/// Classify nested `MulAdd` emissions by the coefficient a caller checked.
///
/// The caller compares the coefficient against its field's canonical zero
/// and one, or its prepared form against their maps, in the same function
/// before opening this guard. Nested guards restore the prior class.
#[must_use]
pub(crate) fn term(class: Term) -> TermGuard {
    TermGuard {
        previous: TERM.with(|term| term.replace(Some(class))),
    }
}

/// Restores the enclosing coefficient class on return or unwind.
pub(crate) struct TermGuard {
    previous: Option<Term>,
}

impl Drop for TermGuard {
    fn drop(&mut self) {
        TERM.with(|term| term.set(self.previous));
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

/// Maximum non-vector elements a single row can lose to alignment peel and
/// scalar tail: ⌈2 × (`vector_bytes` − 1) / `elem_bytes`⌉, the per-row allowance
/// for `assert_vector_work_in` lower bounds.
#[allow(dead_code)]
pub(crate) fn row_loss(vector_bytes: usize, elem_bytes: usize) -> usize {
    (2usize * (vector_bytes - 1)).div_ceil(elem_bytes)
}

/// Assert the trace holds at least one entry at exactly `tier` with work > 0.
///
/// # Panics
///
/// Panics if no trace entry at `tier` with positive work is found.
#[allow(dead_code)]
pub fn assert_vector_work(tier: Backend) {
    let entries = snapshot();
    let mut found = false;
    for e in &entries {
        if e.tier == tier && e.work > 0 {
            found = true;
            break;
        }
    }
    assert!(
        found,
        "expected at least one trace entry at tier {} with positive work, \
         but snapshot has {} entries with tiers {:?}",
        tier.name(),
        entries.len(),
        entries.iter().map(|e| e.tier.name()).collect::<Vec<_>>(),
    );
}

/// Assert the summed work of trace entries at exactly `tier` lies in
/// `[min, max]`, and the summed work of all entries (any tier) is at most
/// `max` — catching double counting and overcounting.
///
/// # Panics
///
/// Panics if `min` is zero, or if either bound is violated.
#[track_caller]
#[allow(dead_code)]
pub fn assert_vector_work_in(tier: Backend, min: usize, max: usize) {
    assert!(min > 0, "min must be > 0 to prove positive work");
    let entries = snapshot();
    let tier_work: usize = entries
        .iter()
        .filter(|e| e.tier == tier)
        .map(|e| e.work)
        .sum();
    let total_work: usize = entries.iter().map(|e| e.work).sum();
    assert!(
        tier_work >= min,
        "expected at least {min} units of work at tier {}, got {tier_work} \
         ({} entries, total work {total_work})",
        tier.name(),
        entries.len(),
    );
    assert!(
        tier_work <= max,
        "work at tier {} exceeded upper bound: {tier_work} > {max} \
         ({} entries, total work {total_work})",
        tier.name(),
        entries.len(),
    );
    assert!(
        total_work <= max,
        "total work over all tiers exceeded upper bound: {total_work} > {max} \
         ({} entries)",
        entries.len(),
    );
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::Pass;
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
        assert_eq!(snapshot().len(), 0);
    }

    #[test]
    fn clear_empties_the_trace() {
        emit("k", super::Substage::Mul, Unit::Elements, 8, Backend::V3);
        assert_eq!(snapshot().len(), 1);
        clear();
        assert_eq!(snapshot().len(), 0);
        emit("k", super::Substage::Add, Unit::Bytes, 4, Backend::V2);
        assert_eq!(snapshot().len(), 1);
    }

    #[test]
    fn scope_attaches_innermost_and_pops_on_drop() {
        clear();
        let _outer = scope(Scope::Inverse, Pass::Main);
        emit("k", super::Substage::Mul, Unit::Elements, 4, Backend::V3);
        {
            let _inner = scope(Scope::Frobenius, Pass::Prefix);
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
        assert_eq!(entries[0].scope, Some((Scope::Inverse, Pass::Main)));
        assert_eq!(entries[1].scope, Some((Scope::Frobenius, Pass::Prefix)));
        assert_eq!(entries[2].scope, Some((Scope::Inverse, Pass::Main)));
    }

    #[test]
    fn scope_pops_on_unwind() {
        clear();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = scope(Scope::Inverse, Pass::Main);
            emit("k", super::Substage::Mul, Unit::Elements, 4, Backend::V3);
            panic!("unwind through active scope guard");
        }));
        assert!(caught.is_err());
        // Emission inside the guarded body was recorded with the scope/pass.
        let entries = snapshot();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].scope, Some((Scope::Inverse, Pass::Main)));
        // Post-unwind emission has no scope: the guard dropped during unwind.
        emit("k", super::Substage::Add, Unit::Elements, 2, Backend::V2);
        let entries = snapshot();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].scope, None);
    }

    #[test]
    fn general_qualification_restores_nested_and_unwound_contexts() {
        clear();
        emit(
            "plain",
            super::Substage::MulAdd,
            Unit::Elements,
            1,
            Backend::V3,
        );
        {
            let _outer = super::term(super::Term::General);
            {
                let _inner = super::term(super::Term::General);
                emit(
                    "inner",
                    super::Substage::MulAdd,
                    Unit::Elements,
                    2,
                    Backend::V3,
                );
            }
            emit(
                "outer",
                super::Substage::MulAdd,
                Unit::Elements,
                3,
                Backend::V3,
            );
            emit("add", super::Substage::Add, Unit::Elements, 4, Backend::V3);
        }
        let caught = std::panic::catch_unwind(|| {
            let _guard = super::term(super::Term::General);
            panic!("unwind through coefficient qualification");
        });
        assert!(caught.is_err());
        emit(
            "after",
            super::Substage::MulAdd,
            Unit::Elements,
            5,
            Backend::V3,
        );
        let entries = snapshot();
        assert!(!entries[0].general_only);
        assert!(entries[1].general_only);
        assert!(entries[2].general_only);
        assert!(!entries[3].general_only);
        assert!(!entries[4].general_only);
        let guard = super::term(super::Term::General);
        clear();
        emit(
            "cleared",
            super::Substage::MulAdd,
            Unit::Elements,
            6,
            Backend::V3,
        );
        assert!(!snapshot()[0].general_only);
        drop(guard);
    }

    /// The substage, qualification, and work of every recorded entry.
    fn kinds() -> std::vec::Vec<(super::Substage, bool, usize)> {
        snapshot()
            .iter()
            .map(|entry| (entry.substage, entry.general_only, entry.work))
            .collect()
    }

    #[test]
    fn unit_class_records_nested_multiply_add_as_addition() {
        clear();
        {
            let _unit = super::term(super::Term::One);
            emit("k", super::Substage::MulAdd, Unit::Elements, 7, Backend::V3);
            emit("k", super::Substage::Mul, Unit::Elements, 3, Backend::V3);
            {
                let _zero = super::term(super::Term::Zero);
                emit("k", super::Substage::MulAdd, Unit::Elements, 2, Backend::V3);
            }
        }
        assert_eq!(
            kinds(),
            [
                (super::Substage::Add, false, 7),
                (super::Substage::Mul, false, 3),
                (super::Substage::MulAdd, false, 2),
            ]
        );
    }

    #[test]
    fn fused_terms_split_work_by_coefficient_class() {
        use super::Term::{General, One, Zero};
        clear();
        super::emit_terms(
            "k",
            Unit::Elements,
            10,
            [General, One, Zero, General, One, General],
            Backend::V3,
        );
        super::emit_seed_terms("k", Unit::Elements, 4, [One, General, Zero], Backend::V3);
        super::emit_terms("k", Unit::Elements, 5, [Zero, Zero], Backend::V3);
        assert_eq!(
            kinds(),
            [
                (super::Substage::MulAdd, false, 10),
                (super::Substage::Add, false, 20),
                (super::Substage::MulAdd, true, 30),
                (super::Substage::MulAdd, false, 4),
                (super::Substage::MulAdd, false, 4),
                (super::Substage::MulAdd, true, 4),
                (super::Substage::MulAdd, false, 10),
            ]
        );
    }

    #[test]
    fn component_widths_normalize_nested_work_to_bytes() {
        clear();
        {
            let _outer = super::element_bytes(8);
            emit(
                "outer",
                super::Substage::Add,
                Unit::Elements,
                2,
                Backend::V3,
            );
            {
                let _inner = super::element_bytes(4);
                emit("base", super::Substage::Add, Unit::Elements, 3, Backend::V3);
                emit("bytes", super::Substage::Add, Unit::Bytes, 5, Backend::V3);
            }
            emit(
                "outer",
                super::Substage::Sub,
                Unit::Elements,
                1,
                Backend::V3,
            );
        }
        emit(
            "direct",
            super::Substage::Add,
            Unit::Elements,
            1,
            Backend::V3,
        );
        let observed: std::vec::Vec<_> = snapshot()
            .iter()
            .map(|entry| (entry.unit, entry.work))
            .collect();
        assert_eq!(
            observed,
            [
                (Unit::Bytes, 16),
                (Unit::Bytes, 12),
                (Unit::Bytes, 5),
                (Unit::Bytes, 8),
                (Unit::Elements, 1)
            ]
        );
    }

    #[test]
    fn component_width_restores_after_unwind() {
        clear();
        let _outer = super::element_bytes(8);
        let caught = std::panic::catch_unwind(|| {
            let _inner = super::element_bytes(4);
            panic!("unwind through component width guard");
        });
        assert!(caught.is_err());
        emit(
            "outer",
            super::Substage::Add,
            Unit::Elements,
            2,
            Backend::V3,
        );
        let entries = snapshot();
        assert_eq!((entries[0].unit, entries[0].work), (Unit::Bytes, 16));
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
