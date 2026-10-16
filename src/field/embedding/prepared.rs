//! Compiled embeddings and isomorphisms for repeated application.
//!
//! A [`PreparedEmbedding`] compiles one validated proper inclusion once so
//! repeated evaluation never re-runs image construction: the forward columns
//! evaluate the compact map on the prime-coordinate basis vectors, and the
//! left inverse is the weighted trace against one normalizer. A
//! [`PreparedIsomorphism`] compiles both directions of one equal-degree
//! presentation change the same way. See [`Embedding`](super::Embedding) for
//! the map mathematics; this module only changes when evaluation runs.
//!
//! Construction allocates (the column entries); application runs over stack
//! workspace only and allocates nothing. Map evaluation is variable-time
//! and unsuitable for secret data.

#[cfg(feature = "alloc")]
use alloc::boxed::Box;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use super::super::extension::descriptor::COORDINATES;
#[cfg(feature = "alloc")]
use super::super::private::RowAccess;
#[cfg(feature = "alloc")]
use super::super::{Elem, Field, FieldElem, PrimeIdentity, ResidueRow, Validate};
#[cfg(feature = "alloc")]
use super::frobenius::Frobenius;
use super::{Embedding, Isomorphism};

/// A compiled embedding arm over prime coordinates.
/// Columns are the scalar map evaluated on prime-coordinate basis vectors,
/// in source-major order: entries `[j * outputs + o]` hold coordinate `o`
/// of the image of basis vector `j`. No elimination, no matrix inversion,
/// no second decoder runs at construction or application. The source
/// endpoint rides in the owning struct; the arm only stores target-side
/// entries beside the prefix widths.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
enum PreparedArm<T: Field> {
    /// The forward column `j` is exactly `e_j` in the first `m` output
    /// positions (an identity prefix), so embedding copies the input
    /// coordinates and restriction reads them back.
    ///
    /// A permuted set of unit columns is not an injection: a normal-basis
    /// Frobenius composes that way, and only the exact prefix counts.
    /// The variant carries its own left inverse, which agrees with the
    /// weighted trace on the image and makes no claim off it.
    Injection {
        /// The source degree: the prefix width copied in both directions.
        /// The target width rides in the endpoint degrees; the arm copies
        /// the prefix and zero-pads the rest.
        inputs: u8,
    },
    /// Characteristic-two coordinate columns in source-major order.
    BinaryColumns {
        inputs: u8,
        outputs: u8,
        /// Forward entries: the coordinate word of each basis image.
        forward: Box<[u64]>,
        /// Left-inverse entries: the coordinate word of each `R(e_i)`.
        inverse: Box<[u64]>,
    },
    /// Odd-characteristic columns as prime-row elements, source-major.
    ResidueColumns {
        inputs: u8,
        outputs: u8,
        /// Forward entries: `inputs * outputs` coordinates.
        forward: Box<[Elem<ResidueRow<T>>]>,
        /// Left-inverse entries: `outputs * inputs` coordinates.
        inverse: Box<[Elem<ResidueRow<T>>]>,
    },
}

/// A compiled proper inclusion of `S` into `T`, ready for repeated use.
///
/// The map is fixed by the already-validated compact embedding this was
/// compiled from: construction evaluates that map on the prime-coordinate
/// basis vectors once, so application never re-runs image construction.
/// Application allocates nothing; only construction owns its column
/// entries.
///
/// Membership is the relative-Frobenius fixed field (`σ_m(x) == x` with
/// `m = S::DEGREE`), matching [`Embedding::contains`]: either predicate is
/// exact on a valid embedding, and this one avoids running the left
/// inverse on outsiders.
///
/// The type is `Send + Sync` exactly when the endpoint raw storage is:
/// preparation introduces no thread-affine state, only owned column
/// entries beside the compact map.
#[cfg(feature = "alloc")]
pub struct PreparedEmbedding<S: Field, T: Field<Characteristic = S::Characteristic>> {
    embedding: Embedding<S, T>,
    relative_degree: u32,
    arm: PreparedArm<T>,
    frobenius_map: Frobenius<T>,
    trace: TraceColumns<T>,
}

// A derive would bound the impl on `S: Clone`; the manual impls need only
// the field contract the type already carries.
#[cfg(feature = "alloc")]
impl<S: Field, T: Field<Characteristic = S::Characteristic>> Clone for PreparedEmbedding<S, T> {
    fn clone(&self) -> Self {
        Self {
            embedding: self.embedding,
            relative_degree: self.relative_degree,
            arm: self.arm.clone(),
            frobenius_map: self.frobenius_map.clone(),
            trace: self.trace.clone(),
        }
    }
}

#[cfg(feature = "alloc")]
impl<S: Field, T: Field<Characteristic = S::Characteristic>> core::fmt::Debug
    for PreparedEmbedding<S, T>
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "PreparedEmbedding<{}, {}>", S::NAME, T::NAME)
    }
}

/// A compiled isomorphism of `S` onto `T`, ready for repeated use.
///
/// Both directions compile from the already-validated compact
/// isomorphism: each side evaluates its scalar direction on the
/// prime-coordinate basis vectors once. Application allocates nothing;
///
/// The type is `Send + Sync` exactly when the endpoint raw storage is:
/// preparation introduces no thread-affine state, only owned column
/// entries beside the compact map.
#[cfg(feature = "alloc")]
pub struct PreparedIsomorphism<S: Field, T: Field<Characteristic = S::Characteristic>> {
    isomorphism: Isomorphism<S, T>,
    forward: PreparedIsoArm<T>,
    reverse: PreparedIsoArm<S>,
}

/// One compiled direction of an isomorphism: its prime-coordinate columns.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
enum PreparedIsoArm<T: Field> {
    Injection {
        degree: u8,
    },
    BinaryColumns {
        degree: u8,
        entries: Box<[u64]>,
    },
    ResidueColumns {
        degree: u8,
        entries: Box<[Elem<ResidueRow<T>>]>,
    },
}

// A derive would bound the impl on `S: Clone`; the manual impl needs only
// the field contract the type already carries.
#[cfg(feature = "alloc")]
impl<S: Field, T: Field<Characteristic = S::Characteristic>> Clone for PreparedIsomorphism<S, T> {
    fn clone(&self) -> Self {
        Self {
            isomorphism: self.isomorphism,
            forward: self.forward.clone(),
            reverse: self.reverse.clone(),
        }
    }
}

#[cfg(feature = "alloc")]
impl<S: Field, T: Field<Characteristic = S::Characteristic>> core::fmt::Debug
    for PreparedIsomorphism<S, T>
{
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "PreparedIsomorphism<{}, {}>", S::NAME, T::NAME)
    }
}

#[cfg(feature = "alloc")]
impl<S: Field, T: Field<Characteristic = S::Characteristic>> PreparedEmbedding<S, T> {
    /// Compile the already-validated map. Cannot fail.
    ///
    /// Construction allocates the forward columns (the scalar map on the
    /// prime-coordinate basis vectors) and the weighted-trace left
    /// inverse; every method below then runs allocation-free over stack
    /// workspace.
    #[must_use]
    pub fn new(embedding: &Embedding<S, T>) -> Self {
        let () = Validate::<S>::OK;
        let () = Validate::<T>::OK;
        let relative_degree = T::DEGREE / S::DEGREE;
        debug_assert!(
            relative_degree > 1,
            "a proper inclusion has degree above one"
        );
        let arm = compile::<S, T>(embedding);
        let trace = compile_trace::<S, T>(embedding);
        Self {
            embedding: *embedding,
            relative_degree,
            arm,
            frobenius_map: Frobenius::<T>::new(S::DEGREE),
            trace,
        }
    }

    /// The compact map this was compiled from.
    ///
    /// [`Embedding`] is `Copy`, so the accessor returns a copy of the
    /// stored map.
    #[must_use]
    pub fn embedding(&self) -> Embedding<S, T> {
        self.embedding
    }

    /// The relative degree `T::DEGREE / S::DEGREE`.
    #[must_use]
    pub fn relative_degree(&self) -> u32 {
        self.relative_degree
    }

    /// Map `value` into the target field.
    ///
    /// Equals the compact [`Embedding::embed`](Embedding::embed) at every
    /// element; runs the compiled columns instead of Horner evaluation.
    #[must_use]
    pub fn embed(&self, value: Elem<S>) -> Elem<T> {
        apply_forward::<S, T>(&self.arm, value)
    }

    /// Whether `value` lies in the image: the fixed field of the relative
    /// Frobenius `x^|S|`.
    ///
    /// Equals the compact [`Embedding::contains`](Embedding::contains) at
    /// every element.
    #[must_use]
    pub fn contains(&self, value: Elem<T>) -> bool {
        self.frobenius(value) == value
    }

    /// The inverse of [`embed`](Self::embed) on the image, or `None`.
    ///
    /// Equals the compact [`Embedding::restrict`](Embedding::restrict) at
    /// every element: members round-trip, outsiders read `None`.
    #[must_use]
    pub fn restrict(&self, value: Elem<T>) -> Option<Elem<S>> {
        if !self.contains(value) {
            return None;
        }
        Some(apply_inverse::<S, T>(&self.arm, value))
    }

    /// The relative Frobenius `x^|S|` of the extension.
    ///
    /// Equals the compact [`Embedding::frobenius`](Embedding::frobenius)
    /// at every element.
    #[must_use]
    pub fn frobenius(&self, value: Elem<T>) -> Elem<T> {
        value.frobenius_with(&self.frobenius_map)
    }

    /// The relative trace: the sum of the conjugates under the relative
    /// Frobenius, restricted to the source.
    ///
    /// Equals the compact [`Embedding::trace`](Embedding::trace) at every
    /// element.
    ///
    /// # Panics
    ///
    /// Panics only on a crate defect: a conjugate sum outside the
    /// constructed embedding image.
    #[must_use]
    pub fn trace(&self, value: Elem<T>) -> Elem<S> {
        apply_trace::<S, T>(&self.trace, value)
    }

    /// The relative norm: the product of the conjugates under the relative
    /// Frobenius, restricted to the source.
    ///
    /// Equals the compact [`Embedding::norm`](Embedding::norm) at every
    /// element. The restriction of the conjugate product runs through the
    /// compiled left inverse, which is exact on the image; no unchecked
    /// public conversion is involved. At relative degree two the product
    /// takes the `restrict(x·σ(x))` shape; wider degrees multiply out the
    /// full conjugate product.
    ///
    /// # Panics
    ///
    /// Panics only on a crate defect: a conjugate product outside the
    /// constructed embedding image.
    #[must_use]
    pub fn norm(&self, value: Elem<T>) -> Elem<S> {
        let mut product = value;
        let mut power = value;
        let mut k = 1;
        while k < self.relative_degree {
            power = self.frobenius(power);
            product *= power;
            k += 1;
        }
        assert!(
            self.contains(product),
            "a conjugate product lies in the embedding image"
        );
        apply_inverse::<S, T>(&self.arm, product)
    }
}

#[cfg(feature = "alloc")]
impl<S: Field, T: Field<Characteristic = S::Characteristic>> PreparedIsomorphism<S, T> {
    /// Compile the already-validated map. Cannot fail.
    ///
    /// Construction allocates the forward and reverse columns (each scalar
    /// direction on the prime-coordinate basis vectors); `apply` and
    /// `unapply` then run allocation-free over stack workspace.
    #[must_use]
    pub fn new(isomorphism: &Isomorphism<S, T>) -> Self {
        let () = Validate::<S>::OK;
        let () = Validate::<T>::OK;
        debug_assert_eq!(S::DEGREE, T::DEGREE, "an isomorphism has equal degrees");
        let compact = *isomorphism;
        let forward = compile_iso::<S, T>(|value| compact.apply(value));
        let reverse = compile_iso::<T, S>(|value| compact.unapply(value));
        Self {
            isomorphism: compact,
            forward,
            reverse,
        }
    }

    /// The compact map this was compiled from.
    ///
    /// [`Isomorphism`] is `Copy`, so the accessor returns a copy of the
    /// stored map.
    #[must_use]
    pub fn isomorphism(&self) -> Isomorphism<S, T> {
        self.isomorphism
    }

    /// Map `value` into the target field.
    ///
    /// Equals the compact [`Isomorphism::apply`](Isomorphism::apply) at
    /// every element.
    #[must_use]
    pub fn apply(&self, value: Elem<S>) -> Elem<T> {
        apply_iso::<S, T>(&self.forward, value)
    }

    /// Map `value` back into the source field: the total inverse.
    ///
    /// Equals the compact [`Isomorphism::unapply`](Isomorphism::unapply)
    /// at every element.
    #[must_use]
    pub fn unapply(&self, value: Elem<T>) -> Elem<S> {
        apply_iso::<T, S>(&self.reverse, value)
    }
}

/// Compile the forward columns and the weighted-trace left inverse of one
/// validated embedding.
#[cfg(feature = "alloc")]
fn compile<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
) -> PreparedArm<T> {
    let characteristic = <S::Characteristic as PrimeIdentity>::CHARACTERISTIC;
    let inputs = S::DEGREE as usize;
    let outputs = T::DEGREE as usize;
    // Every degree the crate builds fits the column tag width.
    #[allow(clippy::cast_possible_truncation)]
    let (tag_in, tag_out) = (inputs as u8, outputs as u8);
    if characteristic == 2 {
        let mut forward = Vec::with_capacity(inputs);
        for j in 0..inputs {
            forward.push(T::coordinate_word(
                embedding.embed(basis_vector::<S>(j)).raw,
            ));
        }
        if is_identity_prefix(&forward, inputs) {
            return PreparedArm::Injection { inputs: tag_in };
        }
        let inverse = compile_inverse_binary::<S, T>(embedding, &forward, inputs, outputs);
        PreparedArm::BinaryColumns {
            inputs: tag_in,
            outputs: tag_out,
            forward: forward.into_boxed_slice(),
            inverse: inverse.into_boxed_slice(),
        }
    } else {
        let mut forward = Vec::with_capacity(inputs * outputs);
        for j in 0..inputs {
            let image = embedding.embed(basis_vector::<S>(j));
            let mut coordinates = [0u64; COORDINATES];
            T::prime_coordinates_into(&mut coordinates[..outputs], image.raw);
            for coordinate in &coordinates[..outputs] {
                forward.push(Elem::<ResidueRow<T>>::from_raw(
                    ResidueRow::<T>::from_prime(*coordinate),
                ));
            }
        }
        if is_identity_prefix_residue::<T>(&forward, inputs, outputs) {
            return PreparedArm::Injection { inputs: tag_in };
        }
        let inverse = compile_inverse_residue::<S, T>(embedding, inputs, outputs);
        PreparedArm::ResidueColumns {
            inputs: tag_in,
            outputs: tag_out,
            forward: forward.into_boxed_slice(),
            inverse: inverse.into_boxed_slice(),
        }
    }
}

/// Whether every compiled forward column `j` is exactly `e_j` in the first
/// `m` output positions: the identity prefix that marks an injection.
///
/// A permuted set of unit columns fails this test by construction: only
/// the exact prefix counts, so a normal-basis Frobenius never routes here.
#[cfg(feature = "alloc")]
fn is_identity_prefix(columns: &[u64], inputs: usize) -> bool {
    if columns.len() != inputs {
        return false;
    }
    for (j, column) in columns.iter().enumerate() {
        if *column != 1u64 << j {
            return false;
        }
    }
    true
}

/// The residue analog of [`is_identity_prefix`]: column `j` holds one at
/// output position `j` and zero in the first `m` positions otherwise.
#[cfg(feature = "alloc")]
fn is_identity_prefix_residue<T: Field>(
    columns: &[Elem<ResidueRow<T>>],
    inputs: usize,
    outputs: usize,
) -> bool {
    if columns.len() != inputs * outputs {
        return false;
    }
    let one = Elem::<ResidueRow<T>>::ONE;
    let zero = Elem::<ResidueRow<T>>::ZERO;
    for j in 0..inputs {
        for o in 0..outputs {
            if columns[j * outputs + o] != if o == j { one } else { zero } {
                return false;
            }
        }
    }
    true
}

/// Compile the weighted-trace left inverse over characteristic two:
/// `R(e_i) = map.trace(c·e_i)` with `c = b · map.embed(t⁻¹)` for one `b`
/// with `t = map.trace(b) ≠ 0`.
///
/// Then `R∘E = id_S` by the trace projection, and `E∘R` projects onto the
/// image; the search below finds its `b` by scanning target basis
/// vectors, which terminates because the trace is a nonzero linear
/// form. The projection needs no condition on the characteristic or the
/// relative degree: the identity `Tr(c*s) = s*Tr(c) = s` holds for every
/// `s` in the image regardless.
#[cfg(feature = "alloc")]
fn compile_inverse_binary<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
    forward: &[u64],
    inputs: usize,
    outputs: usize,
) -> Vec<u64> {
    let normalizer = normalizer::<S, T>(embedding);
    let mut inverse = Vec::with_capacity(outputs);
    for i in 0..outputs {
        let probe = basis_vector::<T>(i);
        let traced = embedding.trace(normalizer * probe);
        inverse.push(S::coordinate_word(traced.raw));
    }
    debug_assert_inverse::<S, T>(embedding, forward, &inverse, inputs, outputs);
    inverse
}

/// Compile the weighted-trace left inverse over odd characteristic, in the
/// same `R(e_i) = map.trace(c·e_i)` form with prime-row entries.
#[cfg(feature = "alloc")]
fn compile_inverse_residue<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
    inputs: usize,
    outputs: usize,
) -> Vec<Elem<ResidueRow<T>>> {
    let normalizer = normalizer::<S, T>(embedding);
    let mut inverse = Vec::with_capacity(outputs * inputs);
    for i in 0..outputs {
        let probe = basis_vector::<T>(i);
        let traced = embedding.trace(normalizer * probe);
        let mut coordinates = [0u64; COORDINATES];
        S::prime_coordinates_into(&mut coordinates[..inputs], traced.raw);
        for coordinate in &coordinates[..inputs] {
            inverse.push(Elem::<ResidueRow<T>>::from_raw(
                ResidueRow::<T>::from_prime(*coordinate),
            ));
        }
    }
    inverse
}

/// One normalizer `c = b · map.embed(t⁻¹)` with `t = map.trace(b) ≠ 0`.
///
/// Scans the target basis vectors, then target index order; each probe
/// scales through [`scaled_probe`].
#[cfg(feature = "alloc")]
fn normalizer<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
) -> Elem<T> {
    let outputs = T::DEGREE as usize;
    (0..outputs)
        .map(|i| scaled_probe(embedding, basis_vector::<T>(i)))
        .find_map(|scaled| scaled)
        .expect("some target basis vector carries nonzero trace")
}

/// Scale one trace probe into normalizer form, or `None` when its trace
/// vanishes: `Some(b · map.embed(t⁻¹))` with `t = map.trace(b) ≠ 0`.
/// `t⁻¹` uses the total inverse convention (`inv(0) == 0`), which never
/// triggers here because `t ≠ 0`.
#[cfg(feature = "alloc")]
fn scaled_probe<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
    probe: Elem<T>,
) -> Option<Elem<T>> {
    let traced = embedding.trace(probe);
    if traced.is_zero() {
        return None;
    }
    Some(probe * embedding.embed(traced.inv()))
}

/// Debug-build check that the compiled inverse inverts the compiled
/// forward map on every source basis vector.
#[cfg(feature = "alloc")]
fn debug_assert_inverse<S: Field, T: Field<Characteristic = S::Characteristic>>(
    embedding: &Embedding<S, T>,
    forward: &[u64],
    inverse: &[u64],
    inputs: usize,
    outputs: usize,
) {
    debug_assert_eq!(forward.len(), inputs, "forward column count");
    debug_assert_eq!(inverse.len(), outputs, "inverse row count");
    for (j, column) in forward.iter().enumerate().take(inputs) {
        let basis = basis_vector::<S>(j);
        let embedded = T::coordinate_word(embedding.embed(basis).raw);
        debug_assert_eq!(embedded, *column, "forward column reproduces the map");
        let mut word = embedded;
        let mut image = 0u64;
        let mut i = 0;
        while word != 0 {
            if word & 1 == 1 {
                image ^= inverse[i];
            }
            word >>= 1;
            i += 1;
        }
        // The inverse is exact on the image; masking keeps the prefix the
        // source degree names. Degrees below 64 mask to the prefix width;
        // degree 64 keeps the full word, which no shift can name.
        let mask = mask_prefix(inputs);
        debug_assert_eq!(
            S::from_coordinate_word(image & mask),
            S::from_coordinate_word(1u64 << j & mask),
            "the left inverse inverts the forward map"
        );
    }
}

/// The prefix mask of the low `inputs` bits, without shifting by the word
/// width: degree 64 keeps the full word.
#[cfg(feature = "alloc")]
fn mask_prefix(inputs: usize) -> u64 {
    if inputs >= 64 {
        u64::MAX
    } else {
        (1u64 << inputs) - 1
    }
}

/// The `j`th prime-coordinate basis vector of `F`.
#[cfg(feature = "alloc")]
fn basis_vector<F: Field>(j: usize) -> Elem<F> {
    let degree = F::DEGREE as usize;
    let mut coordinates = [0u64; COORDINATES];
    coordinates[j] = 1;
    Elem::<F>::from_raw(F::from_prime_coordinates(&coordinates[..degree]))
}

/// Apply one compiled forward arm to an element.
#[cfg(feature = "alloc")]
fn apply_forward<S: Field, T: Field<Characteristic = S::Characteristic>>(
    arm: &PreparedArm<T>,
    value: Elem<S>,
) -> Elem<T> {
    match arm {
        PreparedArm::Injection { inputs, .. } => {
            debug_assert_eq!(usize::from(*inputs), S::DEGREE as usize, "prefix width");
            let mut coordinates = [0u64; COORDINATES];
            S::prime_coordinates_into(&mut coordinates[..S::DEGREE as usize], value.raw);
            let mut image = [0u64; COORDINATES];
            image[..S::DEGREE as usize].copy_from_slice(&coordinates[..S::DEGREE as usize]);
            Elem::<T>::from_raw(T::from_prime_coordinates(&image[..T::DEGREE as usize]))
        }
        PreparedArm::BinaryColumns {
            inputs,
            outputs,
            forward,
            ..
        } => {
            debug_assert_eq!(usize::from(*inputs), S::DEGREE as usize, "column count");
            debug_assert_eq!(usize::from(*outputs), T::DEGREE as usize, "row count");
            let word = S::coordinate_word(value.raw);
            let mut image = 0u64;
            let mut rest = word;
            let mut j = 0;
            while rest != 0 {
                if rest & 1 == 1 {
                    image ^= forward[j];
                }
                rest >>= 1;
                j += 1;
            }
            Elem {
                raw: T::from_coordinate_word(image),
            }
        }
        PreparedArm::ResidueColumns {
            inputs,
            outputs,
            forward,
            ..
        } => {
            let inputs = usize::from(*inputs);
            let outputs = usize::from(*outputs);
            debug_assert_eq!(inputs, S::DEGREE as usize, "column count");
            debug_assert_eq!(outputs, T::DEGREE as usize, "row count");
            let mut coordinates = [0u64; COORDINATES];
            S::prime_coordinates_into(&mut coordinates[..inputs], value.raw);
            let mut image = [Elem::<ResidueRow<T>>::ZERO; COORDINATES];
            for (j, coordinate) in coordinates[..inputs].iter().enumerate() {
                if *coordinate == 0 {
                    continue;
                }
                let factor = Elem::<ResidueRow<T>>::from_residue(*coordinate)
                    .expect("input coordinate below the characteristic");
                for (o, slot) in image[..outputs].iter_mut().enumerate() {
                    *slot += forward[j * outputs + o] * factor;
                }
            }
            let mut residues = [0u64; COORDINATES];
            for (o, slot) in image[..outputs].iter().enumerate() {
                residues[o] = slot
                    .to_residue()
                    .expect("degree-one readout covers the row");
            }
            Elem {
                raw: T::from_prime_coordinates(&residues[..outputs]),
            }
        }
    }
}

/// Apply one compiled left inverse to an image element.
///
/// The caller guarantees membership; on the image the weighted trace
/// inverts the forward map exactly.
#[cfg(feature = "alloc")]
/// The relative trace as compiled prime-coordinate columns: entry `i`
/// is `map.trace(e_i)` for the `i`-th target basis vector.
#[cfg(feature = "alloc")]
#[derive(Clone, Debug)]
enum TraceColumns<T: Field> {
    /// Characteristic two: one source coordinate word per target basis
    /// bit, applied by XOR on the target coordinate word.
    Binary {
        /// The target degree: the input bit count.
        inputs: u8,
        entries: Box<[u64]>,
    },
    /// Odd characteristic: source-major prime-row entries, one source
    /// row of `S::DEGREE` elements per target basis vector.
    Residue {
        inputs: u8,
        outputs: u8,
        entries: Box<[Elem<ResidueRow<T>>]>,
    },
}

/// Compile the trace columns of `map` over the target basis vectors.
#[cfg(feature = "alloc")]
fn compile_trace<S: Field, T: Field<Characteristic = S::Characteristic>>(
    map: &Embedding<S, T>,
) -> TraceColumns<T> {
    let outputs = T::DEGREE as usize;
    let inputs = S::DEGREE as usize;
    if <S::Characteristic as PrimeIdentity>::CHARACTERISTIC == 2 {
        let mut entries = alloc::vec![0u64; outputs];
        for (i, slot) in entries.iter_mut().enumerate() {
            let mut source = [0u64; COORDINATES];
            S::prime_coordinates_into(&mut source[..inputs], map.trace(basis_vector::<T>(i)).raw);
            *slot = S::coordinate_word(
                Elem::<S>::from_raw(S::from_prime_coordinates(&source[..inputs])).raw,
            );
        }
        return TraceColumns::Binary {
            inputs: u8::try_from(outputs).expect("a degree fits its byte"),
            entries: entries.into_boxed_slice(),
        };
    }
    let mut entries = alloc::vec![Elem::<ResidueRow<T>>::ZERO; outputs * inputs];
    for i in 0..outputs {
        let traced = map.trace(basis_vector::<T>(i));
        let mut source = [0u64; COORDINATES];
        S::prime_coordinates_into(&mut source[..inputs], traced.raw);
        for k in 0..inputs {
            entries[i * inputs + k] =
                Elem::<ResidueRow<T>>::from_raw(ResidueRow::<T>::from_prime(source[k]));
        }
    }
    TraceColumns::Residue {
        inputs: u8::try_from(inputs).expect("a degree fits its byte"),
        outputs: u8::try_from(outputs).expect("a degree fits its byte"),
        entries: entries.into_boxed_slice(),
    }
}

/// Apply the compiled trace columns: one linear evaluation, no
/// exponentiation and no restriction.
#[cfg(feature = "alloc")]
fn apply_trace<S: Field, T: Field<Characteristic = S::Characteristic>>(
    columns: &TraceColumns<T>,
    value: Elem<T>,
) -> Elem<S> {
    match columns {
        TraceColumns::Binary { inputs, entries } => {
            let inputs = usize::from(*inputs);
            debug_assert_eq!(inputs, T::DEGREE as usize, "input bit count");
            let word = T::coordinate_word(value.raw);
            let mut image = 0u64;
            let mut rest = word;
            let mut i = 0;
            while rest != 0 {
                if rest & 1 == 1 {
                    image ^= entries[i];
                }
                rest >>= 1;
                i += 1;
            }
            Elem {
                raw: S::from_coordinate_word(image),
            }
        }
        TraceColumns::Residue {
            inputs,
            outputs,
            entries,
        } => {
            let inputs = usize::from(*inputs);
            let outputs = usize::from(*outputs);
            debug_assert_eq!(outputs, T::DEGREE as usize, "column count");
            let mut coordinates = [0u64; COORDINATES];
            T::prime_coordinates_into(&mut coordinates[..outputs], value.raw);
            let mut image = [Elem::<ResidueRow<T>>::ZERO; COORDINATES];
            for (i, coordinate) in coordinates[..outputs].iter().enumerate() {
                if *coordinate == 0 {
                    continue;
                }
                let factor = Elem::<ResidueRow<T>>::from_residue(*coordinate)
                    .expect("input coordinate below the characteristic");
                for (k, slot) in image[..inputs].iter_mut().enumerate() {
                    *slot += entries[i * inputs + k] * factor;
                }
            }
            let mut residues = [0u64; COORDINATES];
            for (k, slot) in image[..inputs].iter().enumerate() {
                residues[k] = slot
                    .to_residue()
                    .expect("degree-one readout covers the row");
            }
            Elem {
                raw: S::from_prime_coordinates(&residues[..inputs]),
            }
        }
    }
}

fn apply_inverse<S: Field, T: Field<Characteristic = S::Characteristic>>(
    arm: &PreparedArm<T>,
    value: Elem<T>,
) -> Elem<S> {
    match arm {
        PreparedArm::Injection { inputs, .. } => {
            debug_assert_eq!(usize::from(*inputs), S::DEGREE as usize, "prefix width");
            let mut coordinates = [0u64; COORDINATES];
            T::prime_coordinates_into(&mut coordinates[..T::DEGREE as usize], value.raw);
            Elem::<S>::from_raw(S::from_prime_coordinates(
                &coordinates[..S::DEGREE as usize],
            ))
        }
        PreparedArm::BinaryColumns {
            outputs, inverse, ..
        } => {
            debug_assert_eq!(usize::from(*outputs), T::DEGREE as usize, "row count");
            let word = T::coordinate_word(value.raw);
            let mut image = 0u64;
            let mut rest = word;
            let mut i = 0;
            while rest != 0 {
                if rest & 1 == 1 {
                    image ^= inverse[i];
                }
                rest >>= 1;
                i += 1;
            }
            Elem {
                raw: S::from_coordinate_word(image),
            }
        }
        PreparedArm::ResidueColumns {
            inputs,
            outputs,
            inverse,
            ..
        } => {
            let inputs = usize::from(*inputs);
            let outputs = usize::from(*outputs);
            debug_assert_eq!(outputs, T::DEGREE as usize, "row count");
            debug_assert_eq!(inputs, S::DEGREE as usize, "column count");
            let mut coordinates = [0u64; COORDINATES];
            T::prime_coordinates_into(&mut coordinates[..outputs], value.raw);
            let mut image = [Elem::<ResidueRow<T>>::ZERO; COORDINATES];
            for (i, coordinate) in coordinates[..outputs].iter().enumerate() {
                if *coordinate == 0 {
                    continue;
                }
                let factor = Elem::<ResidueRow<T>>::from_residue(*coordinate)
                    .expect("input coordinate below the characteristic");
                for (k, slot) in image[..inputs].iter_mut().enumerate() {
                    *slot += inverse[i * inputs + k] * factor;
                }
            }
            let mut residues = [0u64; COORDINATES];
            for (k, slot) in image[..inputs].iter().enumerate() {
                residues[k] = slot
                    .to_residue()
                    .expect("degree-one readout covers the row");
            }
            Elem {
                raw: S::from_prime_coordinates(&residues[..inputs]),
            }
        }
    }
}

/// Compile one direction of an isomorphism: the scalar map on the
/// prime-coordinate basis vectors, with the same identity-prefix shortcut.
#[cfg(feature = "alloc")]
fn compile_iso<S: Field, T: Field<Characteristic = S::Characteristic>>(
    apply: impl Fn(Elem<S>) -> Elem<T>,
) -> PreparedIsoArm<T> {
    let characteristic = <S::Characteristic as PrimeIdentity>::CHARACTERISTIC;
    let degree = S::DEGREE as usize;
    debug_assert_eq!(
        degree,
        T::DEGREE as usize,
        "an isomorphism has equal degrees"
    );
    // Every degree the crate builds fits the column tag width.
    #[allow(clippy::cast_possible_truncation)]
    let tag = degree as u8;
    if characteristic == 2 {
        let mut entries = Vec::with_capacity(degree);
        for j in 0..degree {
            entries.push(T::coordinate_word(apply(basis_vector::<S>(j)).raw));
        }
        if is_identity_prefix(&entries, degree) {
            return PreparedIsoArm::Injection { degree: tag };
        }
        PreparedIsoArm::BinaryColumns {
            degree: tag,
            entries: entries.into_boxed_slice(),
        }
    } else {
        let mut entries = Vec::with_capacity(degree * degree);
        for j in 0..degree {
            let image = apply(basis_vector::<S>(j));
            let mut coordinates = [0u64; COORDINATES];
            T::prime_coordinates_into(&mut coordinates[..degree], image.raw);
            for coordinate in &coordinates[..degree] {
                entries.push(Elem::<ResidueRow<T>>::from_raw(
                    ResidueRow::<T>::from_prime(*coordinate),
                ));
            }
        }
        if is_identity_prefix_residue::<T>(&entries, degree, degree) {
            return PreparedIsoArm::Injection { degree: tag };
        }
        PreparedIsoArm::ResidueColumns {
            degree: tag,
            entries: entries.into_boxed_slice(),
        }
    }
}

/// Apply one compiled isomorphism direction to an element.
#[cfg(feature = "alloc")]
fn apply_iso<S: Field, T: Field<Characteristic = S::Characteristic>>(
    arm: &PreparedIsoArm<T>,
    value: Elem<S>,
) -> Elem<T> {
    match arm {
        PreparedIsoArm::Injection { degree } => {
            debug_assert_eq!(usize::from(*degree), S::DEGREE as usize, "prefix width");
            let mut coordinates = [0u64; COORDINATES];
            S::prime_coordinates_into(&mut coordinates[..S::DEGREE as usize], value.raw);
            Elem::<T>::from_raw(T::from_prime_coordinates(
                &coordinates[..T::DEGREE as usize],
            ))
        }
        PreparedIsoArm::BinaryColumns { degree, entries } => {
            debug_assert_eq!(usize::from(*degree), S::DEGREE as usize, "column count");
            let word = S::coordinate_word(value.raw);
            let mut image = 0u64;
            let mut rest = word;
            let mut j = 0;
            while rest != 0 {
                if rest & 1 == 1 {
                    image ^= entries[j];
                }
                rest >>= 1;
                j += 1;
            }
            Elem {
                raw: T::from_coordinate_word(image),
            }
        }
        PreparedIsoArm::ResidueColumns { degree, entries } => {
            let degree = usize::from(*degree);
            debug_assert_eq!(degree, S::DEGREE as usize, "column count");
            let mut coordinates = [0u64; COORDINATES];
            S::prime_coordinates_into(&mut coordinates[..degree], value.raw);
            let mut image = [Elem::<ResidueRow<T>>::ZERO; COORDINATES];
            for (j, coordinate) in coordinates[..degree].iter().enumerate() {
                if *coordinate == 0 {
                    continue;
                }
                let factor = Elem::<ResidueRow<T>>::from_residue(*coordinate)
                    .expect("input coordinate below the characteristic");
                for (o, slot) in image[..degree].iter_mut().enumerate() {
                    *slot += entries[j * degree + o] * factor;
                }
            }
            let mut residues = [0u64; COORDINATES];
            for (o, slot) in image[..degree].iter().enumerate() {
                residues[o] = slot
                    .to_residue()
                    .expect("degree-one readout covers the row");
            }
            Elem {
                raw: T::from_prime_coordinates(&residues[..degree]),
            }
        }
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    //! Unit coverage for the compilation helpers: the prefix helpers on
    //! both outcomes, the identity-prefix detectors on length and entry
    //! mismatches, and the trace-probe scaler on both outcomes.
    //!
    //! Behavior agreement lives in `tests/prepared_maps.rs`; these pin
    //! the small helpers the gate counts line by line.

    use super::super::super::Elem;
    use super::{is_identity_prefix, mask_prefix, scaled_probe};

    #[test]
    fn prefix_mask_covers_both_arms() {
        assert_eq!(mask_prefix(1), 1);
        assert_eq!(mask_prefix(8), 0xFF);
        assert_eq!(mask_prefix(63), u64::MAX >> 1);
        assert_eq!(mask_prefix(64), u64::MAX);
        assert_eq!(mask_prefix(80), u64::MAX);
    }

    #[test]
    fn identity_prefix_rejects_length_and_entry_mismatches() {
        // Length mismatch: fewer columns than inputs.
        assert!(!is_identity_prefix(&[1, 2], 3));
        // Entry mismatch at position one.
        assert!(!is_identity_prefix(&[1, 4], 2));
        // The exact prefix accepts.
        assert!(is_identity_prefix(&[1, 2, 4], 3));
        assert!(is_identity_prefix(&[], 0));
    }

    #[test]
    fn trace_probe_scales_or_skips() {
        use crate::field::{Mersenne31, QuadMersenne31};

        // The scaler returns `None` on a zero element's vanishing probe
        // and scales a unit probe through the prime inclusion.
        let compact = super::super::Embedding::<Mersenne31, QuadMersenne31>::new().unwrap();
        let probe = Elem::<QuadMersenne31>::ONE;
        let scaled = scaled_probe(&compact, probe);
        let expected = compact.embed(compact.trace(probe).inv());
        assert_eq!(scaled, Some(expected));
        assert_eq!(scaled_probe(&compact, Elem::<QuadMersenne31>::ZERO), None);
    }
}
