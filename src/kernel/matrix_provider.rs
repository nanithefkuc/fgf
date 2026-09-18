//! Borrowed coefficient/source providers for register-blocked x86 kernels.

/// Matrix-like coefficient/source provider for the register-blocked x86 kernels.
#[allow(clippy::len_without_is_empty)]
pub trait Matrix<C> {
    /// Number of terms.
    fn len(&self) -> usize;
    /// The coefficient of `term` for destination row `row`.
    fn coefficient(&self, term: usize, row: usize) -> &C;
    /// The source buffer of `term`.
    fn source(&self, term: usize) -> &[u8];
}

impl<C> Matrix<C> for [(&[C], &[u8])] {
    #[inline]
    fn len(&self) -> usize {
        <[(&[C], &[u8])]>::len(self)
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> &C {
        &self[term].0[row]
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self[term].1
    }
}

/// Flat row-major coefficient matrix over borrowed sources.
pub struct FlatMatrix<'a, C> {
    /// Flat row-major coefficients, `terms * nrows` entries.
    pub coefficients: &'a [C],
    /// Destination row count.
    pub nrows: usize,
    /// Source buffers, one per term.
    pub sources: &'a [&'a [u8]],
}

impl<C> Matrix<C> for FlatMatrix<'_, C> {
    #[inline]
    fn len(&self) -> usize {
        self.sources.len()
    }

    #[inline]
    fn coefficient(&self, term: usize, row: usize) -> &C {
        &self.coefficients[term * self.nrows + row]
    }

    #[inline]
    fn source(&self, term: usize) -> &[u8] {
        self.sources[term]
    }
}
