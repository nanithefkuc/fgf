//! In-process comparison of `fgf` against Intel ISA-L over GF(2^8)/`0x11D`.
//!
//! Both libraries implement the same field — the Reed-Solomon interop
//! polynomial `0x11D` that `fgf` exposes as [`Gf8D`] — so every arm is a
//! byte-for-byte comparison of the same result, not two approximations of
//! one. Every shape is validated against the other library's output before a
//! single timing is taken.
//!
//! The shapes are the ones an erasure codec actually issues: a
//! single-source region operation, a many-source dot product into one row,
//! and a full systematic encode of `p` parity rows from `k` sources.
//!
//! # Protocol
//!
//! The two implementations are interleaved inside one process against one
//! fixture set: each round takes a sample from both arms, and the order
//! alternates every round so that neither arm owns the warm side of a
//! thermal or frequency drift. A sample is the mean of 32 repetitions; the
//! reported figure is the median sample. Table construction, buffer
//! allocation, pointer-array staging, and backend resolution all happen
//! outside the timed region — the timed closures allocate nothing.
//!
//! The first line of the table is a control: the same `fgf` body registered
//! as both arms, over two distinct destinations. Its ratio is the harness's
//! own noise floor and is expected at 1.00x; a control that drifts
//! invalidates every ratio below it.
//!
//! # Running
//!
//! ```sh
//! FEC_GOLDEN_CORE=<p-core> just bench-isal   # from the fgf crate directory
//! ```
//!
//! Record the output in `BENCHMARKS.md` with the host, the resolved `fgf`
//! backend, the ISA-L version printed in the header, and the rustc version.

use std::hint::black_box;
use std::time::{Duration, Instant};

use fgf::{Gf8D, backend, backend_for, gf8d, ops};

/// Every fixture is page-aligned. ISA-L only requires 32 bytes, but equal
/// page offsets hold the 4 KiB store-to-load aliasing penalty identical for
/// both arms; with cache-line alignment alone the control drifts by up to
/// seven percent, depending on which destination the allocator placed
/// opposite the shared source.
const ALIGN: usize = 4096;
const REGION_BYTES: usize = 64 * 1024;
const GATHER_SOURCES: usize = 16;
const GATHER_LENGTHS: &[usize] = &[4 * 1024, 16 * 1024];
const ENCODE_SOURCES: usize = 10;
const ENCODE_ROWS: &[usize] = &[2, 4, 6];
const ENCODE_LENGTHS: &[usize] = &[4 * 1024, 16 * 1024, 64 * 1024];

const WARMUP: usize = 16;
const REPS: u32 = 32;
const MAX_SAMPLES: usize = 128;
const BUDGET: Duration = Duration::from_millis(1500);

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A 64-byte-aligned byte buffer built from an over-allocated `Vec`.
struct AlignedBuf {
    backing: Vec<u8>,
    offset: usize,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize, seed: u64) -> Self {
        let mut backing = vec![0u8; len + ALIGN];
        let offset = backing.as_ptr().align_offset(ALIGN);
        assert!(offset < ALIGN, "alignment padding is one ALIGN block");
        let mut state = seed | 1;
        for byte in &mut backing[offset..offset + len] {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            *byte = (state >> 33) as u8;
        }
        Self {
            backing,
            offset,
            len,
        }
    }

    fn as_slice(&self) -> &[u8] {
        &self.backing[self.offset..self.offset + self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.backing[self.offset..self.offset + self.len]
    }
}

/// The coding matrix both libraries encode with: `rows` by `k`, row-major,
/// every entry nonzero. ISA-L consumes it directly; `fgf` reads the same
/// entries transposed into one coefficient column per source.
fn coding_matrix(rows: usize, k: usize) -> Vec<u8> {
    (0..rows * k)
        .map(|index| (1 + ((index * 97 + 13) % 255)) as u8)
        .collect()
}

/// The dot-product coefficient vector, one nonzero entry per source.
fn gather_coefficients(sources: usize) -> Vec<u8> {
    (0..sources)
        .map(|source| (2 + ((source * 73 + 19) % 254)) as u8)
        .collect()
}

// ---------------------------------------------------------------------------
// ISA-L
// ---------------------------------------------------------------------------

mod isal {
    use std::marker::PhantomData;

    unsafe extern "C" {
        fn gf_vect_mul_init(c: u8, gftbl: *mut u8);
        fn gf_vect_mul(
            len: i32,
            gftbl: *mut u8,
            src: *mut core::ffi::c_void,
            dest: *mut core::ffi::c_void,
        ) -> i32;
        fn gf_vect_mad(
            len: i32,
            vec: i32,
            vec_i: i32,
            gftbls: *mut u8,
            src: *mut u8,
            dest: *mut u8,
        );
        fn ec_init_tables(k: i32, rows: i32, a: *mut u8, gftbls: *mut u8);
        fn ec_encode_data(
            len: i32,
            k: i32,
            rows: i32,
            gftbls: *mut u8,
            data: *mut *mut u8,
            coding: *mut *mut u8,
        );
        fn gf_vect_dot_prod(len: i32, vlen: i32, gftbls: *mut u8, src: *mut *mut u8, dest: *mut u8);
    }

    /// The version this binary linked against, from `pkg-config` at build time.
    pub const VERSION: &str = env!("ISAL_VERSION");

    /// Expand one coefficient into the 32-byte table `vect_mul` consumes.
    pub fn vect_mul_init(coefficient: u8) -> [u8; 32] {
        let mut table = [0u8; 32];
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `table` is a live local array of exactly the 32 bytes the
        //        documented `gf_vect_mul_init` output size requires.
        // THUS: the only store the call performs stays inside `table`.
        unsafe { gf_vect_mul_init(coefficient, table.as_mut_ptr()) };
        table
    }

    /// Expand a `rows` by `k` coding matrix into ISA-L's `32 * k * rows` bank.
    pub fn init_tables(k: usize, rows: usize, matrix: &[u8]) -> Vec<u8> {
        assert_eq!(matrix.len(), k * rows, "coding matrix is rows by k");
        let mut matrix = matrix.to_vec();
        let mut tables = vec![0u8; 32 * k * rows];
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `matrix` holds `k * rows` bytes and `tables` holds
        //        `32 * k * rows`, which are the documented input and output
        //        sizes for these `k` and `rows`.
        // THUS: every read and store the call performs is in bounds.
        //
        // ALIASING
        // SINCE: `matrix` is a private copy owned by this function and
        //        `tables` is a freshly allocated local.
        // THUS: the read and the write cannot overlap.
        unsafe {
            ec_init_tables(
                k as i32,
                rows as i32,
                matrix.as_mut_ptr(),
                tables.as_mut_ptr(),
            );
        }
        tables
    }

    /// `dest = coefficient * src`, ISA-L's single-source overwrite kernel.
    pub fn vect_mul(table: &mut [u8; 32], src: &[u8], dest: &mut [u8]) {
        assert_eq!(src.len(), dest.len(), "vect_mul: length mismatch");
        assert!(
            src.len().is_multiple_of(32),
            "vect_mul: length must be 32B-aligned"
        );
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `src` and `dest` are live slices of the same length, which
        //        is passed as `len`, and `table` is the 32-byte expansion
        //        `gf_vect_mul_init` produced.
        // THUS: every load and store the kernel performs is in bounds.
        //
        // ALIASING
        // SINCE: `dest` is exclusively borrowed, so it is disjoint from the
        //        shared `src` and from `table`.
        // THUS: the destination stores conflict with no live reference.
        let status = unsafe {
            gf_vect_mul(
                src.len() as i32,
                table.as_mut_ptr(),
                src.as_ptr().cast::<core::ffi::c_void>().cast_mut(),
                dest.as_mut_ptr().cast::<core::ffi::c_void>(),
            )
        };
        assert_eq!(status, 0, "gf_vect_mul rejected the geometry");
    }

    /// `dest ^= coefficient * src`, ISA-L's single-source accumulate kernel.
    ///
    /// `tables` is a one-coefficient bank from [`init_tables`] with `k` and
    /// `rows` both one.
    pub fn vect_mad(tables: &mut [u8], src: &[u8], dest: &mut [u8]) {
        assert_eq!(src.len(), dest.len(), "vect_mad: length mismatch");
        assert_eq!(
            tables.len(),
            32,
            "vect_mad: one coefficient is 32 table bytes"
        );
        // SAFETY:
        // MEMORY VALIDITY
        // SINCE: `src` and `dest` are live slices of the length passed as
        //        `len`, and `tables` holds the `32 * vec` bytes the
        //        documented contract requires for `vec == 1`.
        // THUS: every load and store the kernel performs is in bounds.
        //
        // ALIASING
        // SINCE: `dest` is exclusively borrowed and therefore disjoint from
        //        `src` and `tables`.
        // THUS: the accumulate stores conflict with no live reference.
        unsafe {
            gf_vect_mad(
                src.len() as i32,
                1,
                0,
                tables.as_mut_ptr(),
                src.as_ptr().cast_mut(),
                dest.as_mut_ptr(),
            );
        }
    }

    /// A staged many-source dot product: `dest = sum(coeffs[i] * srcs[i])`.
    ///
    /// The pointer array is built once at construction so the timed call
    /// allocates nothing.
    pub struct DotProd<'a> {
        tables: Vec<u8>,
        sources: Vec<*mut u8>,
        len: usize,
        dest: *mut u8,
        borrow: PhantomData<(&'a [u8], &'a mut [u8])>,
    }

    impl<'a> DotProd<'a> {
        pub fn new(coefficients: &[u8], srcs: &[&'a [u8]], dest: &'a mut [u8]) -> Self {
            assert_eq!(coefficients.len(), srcs.len(), "one coefficient per source");
            assert!(
                srcs.iter().all(|src| src.len() == dest.len()),
                "every source matches the destination length",
            );
            Self {
                tables: init_tables(srcs.len(), 1, coefficients),
                sources: srcs.iter().map(|src| src.as_ptr().cast_mut()).collect(),
                len: dest.len(),
                dest: dest.as_mut_ptr(),
                borrow: PhantomData,
            }
        }

        pub fn run(&mut self) {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the constructor checked that every source is `self.len`
            //        bytes and derived each pointer from a slice that the
            //        `'a` borrow keeps live, and `self.tables` is the
            //        `32 * vlen` bank `init_tables` produced for this source
            //        count.
            // THUS: every load and store the kernel performs is in bounds.
            //
            // ALIASING
            // SINCE: `dest` came from an exclusive borrow held for `'a`, and
            //        the sources came from shared borrows of distinct
            //        buffers; ISA-L documents `src` as read-only.
            // THUS: the destination stores conflict with no live reference.
            unsafe {
                gf_vect_dot_prod(
                    self.len as i32,
                    self.sources.len() as i32,
                    self.tables.as_mut_ptr(),
                    self.sources.as_mut_ptr(),
                    self.dest,
                );
            }
        }
    }

    /// A staged systematic encode: `rows` parity rows from `k` sources.
    pub struct Encode<'a> {
        tables: Vec<u8>,
        sources: Vec<*mut u8>,
        coding: Vec<*mut u8>,
        row_len: usize,
        borrow: PhantomData<(&'a [u8], &'a mut [u8])>,
    }

    impl<'a> Encode<'a> {
        pub fn new(matrix: &[u8], nrows: usize, srcs: &[&'a [u8]], rows: &'a mut [u8]) -> Self {
            let row_len = rows.len() / nrows;
            assert_eq!(rows.len(), row_len * nrows, "destination holds whole rows");
            assert!(
                srcs.iter().all(|src| src.len() == row_len),
                "every source is one row long",
            );
            Self {
                tables: init_tables(srcs.len(), nrows, matrix),
                sources: srcs.iter().map(|src| src.as_ptr().cast_mut()).collect(),
                coding: rows
                    .chunks_exact_mut(row_len)
                    .map(<[u8]>::as_mut_ptr)
                    .collect(),
                row_len,
                borrow: PhantomData,
            }
        }

        pub fn run(&mut self) {
            // SAFETY:
            // MEMORY VALIDITY
            // SINCE: the constructor derived every source pointer from a
            //        slice of exactly `row_len` bytes and every coding
            //        pointer from one `row_len` chunk of the destination,
            //        with `self.tables` the `32 * k * rows` bank
            //        `init_tables` produced for these counts; the `'a`
            //        borrow keeps all of them live.
            // THUS: every load and store the kernel performs is in bounds.
            //
            // ALIASING
            // SINCE: the coding rows are disjoint chunks of one exclusive
            //        borrow, the sources are shared borrows of distinct
            //        buffers, and ISA-L documents `data` as read-only.
            // THUS: no store conflicts with another live reference.
            unsafe {
                ec_encode_data(
                    self.row_len as i32,
                    self.sources.len() as i32,
                    self.coding.len() as i32,
                    self.tables.as_mut_ptr(),
                    self.sources.as_mut_ptr(),
                    self.coding.as_mut_ptr(),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

fn sample(body: &mut impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..REPS {
        body();
    }
    start.elapsed() / REPS
}

fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn throughput(bytes: usize, elapsed: Duration) -> f64 {
    bytes as f64 / elapsed.as_secs_f64() / (1024.0 * 1024.0 * 1024.0)
}

/// Time two implementations of one shape, interleaved, and print the pair.
///
/// `bytes` is the logical volume one call processes, so the two columns are
/// comparable across shapes. The ratio is the ISA-L median divided by the
/// `fgf` median: above 1.00 means `fgf` is faster.
fn compare(label: &str, bytes: usize, mut ours: impl FnMut(), mut theirs: impl FnMut()) {
    for _ in 0..WARMUP {
        ours();
        theirs();
    }

    let mut our_samples = Vec::with_capacity(MAX_SAMPLES);
    let mut their_samples = Vec::with_capacity(MAX_SAMPLES);
    let deadline = Instant::now() + BUDGET;
    let mut round = 0usize;
    while Instant::now() < deadline && our_samples.len() < MAX_SAMPLES {
        if round.is_multiple_of(2) {
            our_samples.push(sample(&mut ours));
            their_samples.push(sample(&mut theirs));
        } else {
            their_samples.push(sample(&mut theirs));
            our_samples.push(sample(&mut ours));
        }
        round += 1;
    }

    let our_median = median(our_samples);
    let their_median = median(their_samples);
    let ratio = their_median.as_secs_f64() / our_median.as_secs_f64();
    println!(
        "  {label:<34} {:>9.2?} {:>7.2} GiB/s | {:>9.2?} {:>7.2} GiB/s | {ratio:>5.2}x",
        our_median,
        throughput(bytes, our_median),
        their_median,
        throughput(bytes, their_median),
    );
}

fn header() {
    println!(
        "fgf — process backend {}, Gf8D backend {}",
        backend().name(),
        backend_for::<Gf8D>().name(),
    );
    println!(
        "ISA-L {} (runtime dispatch), GF(2^8) polynomial 0x11D",
        isal::VERSION
    );
    println!("ratio = ISA-L median / fgf median; above 1.00x means fgf is faster");
    println!(
        "  {:<34} {:>23} | {:>23} | {:>6}",
        "shape", "fgf", "ISA-L", "ratio"
    );
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

/// The harness's own noise floor: one `fgf` body timed against itself over
/// two distinct destinations. Anything other than 1.00x is interleaving bias.
fn control() {
    let src = AlignedBuf::new(REGION_BYTES, 0x100);
    let mut first = AlignedBuf::new(REGION_BYTES, 0x101);
    let mut second = AlignedBuf::new(REGION_BYTES, 0x102);
    let coefficient = gf8d::Elem::from_raw(0x53);

    compare(
        "control: mul_into vs itself",
        REGION_BYTES,
        || {
            ops::mul_into::<Gf8D>(
                black_box(first.as_mut_slice()),
                coefficient,
                black_box(src.as_slice()),
            );
        },
        || {
            ops::mul_into::<Gf8D>(
                black_box(second.as_mut_slice()),
                coefficient,
                black_box(src.as_slice()),
            );
        },
    );
}

/// Single-source region operations: `dst = c * src` and `dst ^= c * src`.
fn single_source() {
    let raw = 0x53u8;
    let coefficient = gf8d::Elem::from_raw(raw);
    let src = AlignedBuf::new(REGION_BYTES, 0x200);
    let mut ours = AlignedBuf::new(REGION_BYTES, 0x201);
    let mut theirs = AlignedBuf::new(REGION_BYTES, 0x202);
    let mut mul_table = isal::vect_mul_init(raw);
    let mut mad_tables = isal::init_tables(1, 1, &[raw]);

    ops::mul_into::<Gf8D>(ours.as_mut_slice(), coefficient, src.as_slice());
    isal::vect_mul(&mut mul_table, src.as_slice(), theirs.as_mut_slice());
    assert_eq!(
        ours.as_slice(),
        theirs.as_slice(),
        "mul_into and gf_vect_mul disagree",
    );

    ops::mul_add::<Gf8D>(ours.as_mut_slice(), coefficient, src.as_slice());
    isal::vect_mad(&mut mad_tables, src.as_slice(), theirs.as_mut_slice());
    assert_eq!(
        ours.as_slice(),
        theirs.as_slice(),
        "mul_add and gf_vect_mad disagree",
    );

    println!("{REGION_BYTES} B single source");
    compare(
        "dst = c * src",
        REGION_BYTES,
        || {
            ops::mul_into::<Gf8D>(
                black_box(ours.as_mut_slice()),
                coefficient,
                black_box(src.as_slice()),
            );
        },
        || {
            isal::vect_mul(
                &mut mul_table,
                black_box(src.as_slice()),
                black_box(theirs.as_mut_slice()),
            )
        },
    );
    compare(
        "dst ^= c * src",
        REGION_BYTES,
        || {
            ops::mul_add::<Gf8D>(
                black_box(ours.as_mut_slice()),
                coefficient,
                black_box(src.as_slice()),
            );
        },
        || {
            isal::vect_mad(
                &mut mad_tables,
                black_box(src.as_slice()),
                black_box(theirs.as_mut_slice()),
            )
        },
    );
}

/// Many sources into one row: `fgf`'s gather against `gf_vect_dot_prod`.
fn gather(len: usize) {
    let raw = gather_coefficients(GATHER_SOURCES);
    let coefficients: Vec<gf8d::Elem> = raw.iter().copied().map(gf8d::Elem::from_raw).collect();
    let buffers: Vec<AlignedBuf> = (0..GATHER_SOURCES)
        .map(|source| AlignedBuf::new(len, 0x300 + source as u64))
        .collect();
    let sources: Vec<&[u8]> = buffers.iter().map(AlignedBuf::as_slice).collect();
    let mut ours = AlignedBuf::new(len, 0x380);
    let mut theirs = AlignedBuf::new(len, 0x381);

    ops::mul_into_gather::<Gf8D>(ours.as_mut_slice(), &coefficients, &sources);
    {
        let mut check = isal::DotProd::new(&raw, &sources, theirs.as_mut_slice());
        check.run();
    }
    assert_eq!(
        ours.as_slice(),
        theirs.as_slice(),
        "mul_into_gather and gf_vect_dot_prod disagree",
    );
    let mut dot = isal::DotProd::new(&raw, &sources, theirs.as_mut_slice());

    println!("{len} B x {GATHER_SOURCES} sources -> 1 row");
    compare(
        "dst = sum(c[i] * src[i])",
        len * GATHER_SOURCES,
        || {
            ops::mul_into_gather::<Gf8D>(
                black_box(ours.as_mut_slice()),
                black_box(&coefficients),
                black_box(&sources),
            );
        },
        || dot.run(),
    );
}

/// Systematic encode: `nrows` parity rows from `ENCODE_SOURCES` sources.
fn encode(len: usize, nrows: usize) {
    let matrix = coding_matrix(nrows, ENCODE_SOURCES);
    let buffers: Vec<AlignedBuf> = (0..ENCODE_SOURCES)
        .map(|source| AlignedBuf::new(len, 0x400 + source as u64))
        .collect();
    let sources: Vec<&[u8]> = buffers.iter().map(AlignedBuf::as_slice).collect();
    let columns: Vec<Vec<gf8d::Elem>> = (0..ENCODE_SOURCES)
        .map(|source| {
            (0..nrows)
                .map(|row| gf8d::Elem::from_raw(matrix[row * ENCODE_SOURCES + source]))
                .collect()
        })
        .collect();
    let terms: Vec<(&[gf8d::Elem], &[u8])> = columns
        .iter()
        .zip(&sources)
        .map(|(coeffs, &src)| (coeffs.as_slice(), src))
        .collect();

    let mut ours = AlignedBuf::new(len * nrows, 0x480);
    let mut theirs = AlignedBuf::new(len * nrows, 0x481);
    let expected = {
        ops::mul_into_matrix::<Gf8D>(ours.as_mut_slice(), len, nrows, &terms);
        let mut encoder = isal::Encode::new(&matrix, nrows, &sources, theirs.as_mut_slice());
        encoder.run();
        ours.as_slice().to_vec()
    };
    assert_eq!(
        expected.as_slice(),
        theirs.as_slice(),
        "mul_into_matrix and ec_encode_data disagree",
    );

    let mut encoder = isal::Encode::new(&matrix, nrows, &sources, theirs.as_mut_slice());
    println!("{len} B x {ENCODE_SOURCES} sources -> {nrows} rows");
    compare(
        "systematic encode",
        len * ENCODE_SOURCES,
        || {
            ops::mul_into_matrix::<Gf8D>(
                black_box(ours.as_mut_slice()),
                len,
                nrows,
                black_box(&terms),
            );
        },
        || encoder.run(),
    );
}

fn main() {
    header();
    control();
    single_source();

    for &len in GATHER_LENGTHS {
        gather(len);
    }

    for &nrows in ENCODE_ROWS {
        for &len in ENCODE_LENGTHS {
            encode(len, nrows);
        }
    }
}
