//! Borrowed coefficient/source providers for register-blocked kernels.

/// Matrix-like coefficient/source provider for the register-blocked kernels.
///
/// Coefficients are returned by value: an implementation may hold them
/// materialized (a slice or flat matrix of prepared forms) or resolve each
/// one on access from raw elements, which is how the dispatch layer serves
/// the raw-coefficient operations without allocating.
#[allow(clippy::len_without_is_empty)]
pub trait Matrix<C> {
    /// Number of terms.
    fn len(&self) -> usize;
    /// The coefficient of `term` for destination row `row`.
    fn coefficient(&self, term: usize, row: usize) -> C;
    /// The source buffer of `term`.
    fn source(&self, term: usize) -> &[u8];
}

impl<C: Copy> Matrix<C> for [(&[C], &[u8])] {
    #[inline]
    fn len(&self) -> usize {
        <[(&[C], &[u8])]>::len(self)
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> C {
        self[term].0[row]
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self[term].1
    }
}

/// Flat row-major coefficient matrix over borrowed sources.
///
/// The x86 dispatch's prepared-plan form; raw-element plans resolve through
/// an adapter at the dispatch layer instead.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub struct FlatMatrix<'a, C> {
    /// Flat row-major coefficients, `terms * nrows` entries.
    pub coefficients: &'a [C],
    /// Destination row count.
    pub nrows: usize,
    /// Source buffers, one per term.
    pub sources: &'a [&'a [u8]],
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
impl<C: Copy> Matrix<C> for FlatMatrix<'_, C> {
    #[inline]
    fn len(&self) -> usize {
        self.sources.len()
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> C {
        self.coefficients[term * self.nrows + row]
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.sources[term]
    }
}
