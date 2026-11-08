//! Cell registry, deterministic fixtures, and the timed entry points.
//!
//! A cell is one timed public-API call shape. Its id is
//! `{field}/{operation}/{prep}/{row_len}x{rows}x{sources}/{alignment}`.
//! Registration order is fixed, so the inventory is byte-identical across
//! runs, hosts, and arms.

use fgf::ops::{self, bits};
use fgf::{Elem, FieldBuffer, FieldElem};

use crate::fields;
use crate::fixture;
use crate::oracle;

const SMALL: usize = 64;
const MID: usize = 4096;
const LARGE: usize = 65_536;
const OFFSET: usize = 32;
const ROW_GAP: usize = 128;
const PREFETCH_BYTES: usize = 32_768;
const NT_BYTES: usize = 2 * 1024 * 1024;

/// One registered benchmark shape.
pub struct Cell {
    pub id: String,
    pub field: &'static str,
    pub operation: &'static str,
    pub prep: &'static str,
    pub row_len: usize,
    pub rows: usize,
    pub sources: usize,
    pub alignment: String,
    pub align_offset: usize,
    pub family: &'static str,
    pub bytes: usize,
    pub run: fn(&Cell, &crate::Ctx) -> Option<f64>,
}

/// The page size the fixture places every live buffer within.
pub(crate) const PAGE_BYTES: usize = 4096;
/// Alignment used for buffers at or above the streaming-store size. This is
/// the huge-page size on the x86-64 hosts the campaign targets; an ordinary
/// page-aligned placement is the fallback on other hosts.
pub(crate) const LARGE_ALIGN: usize = 2 * 1024 * 1024;

/// The fixed intra-page offset of one fixture role.
///
/// Every offset is a multiple of 64 and distinct modulo [`PAGE_BYTES`], so a
/// `dst` and any source never share a 4 KiB page offset: the destination
/// store and the source load cannot hit the same page offset at the same
/// time and trigger 4 KiB store-to-load aliasing. Role offsets are constants
/// rather than allocation-order artifacts, so the placement is reproducible
/// across processes. `offset{k}` adds `k` to the destination only.
#[derive(Clone, Copy)]
enum Role {
    /// Destination row(s).
    Dst,
    /// First source row or one backing region of source rows.
    Src,
    /// Second source row, used by elementwise and division cells.
    Src2,
    /// The offset-addressed gather region.
    Region,
}

impl Role {
    const fn offset(self) -> usize {
        match self {
            Role::Dst => 64,
            Role::Src => 512,
            Role::Src2 => 960,
            Role::Region => 1408,
        }
    }
}

/// A byte buffer whose first live byte has the requested residue modulo 64,
/// inside a page-aligned allocation at the role's fixed page offset.
pub struct Aligned {
    storage: Vec<u8>,
    start: usize,
    len: usize,
    skew: usize,
    role: Role,
}

impl Aligned {
    fn from_bytes(bytes: &[u8], role: Role, skew: usize) -> Self {
        let align = if bytes.len() >= LARGE_ALIGN {
            LARGE_ALIGN
        } else {
            PAGE_BYTES
        };
        // The backing vector need not start page-aligned; the live slice does.
        let extra = align + Role::Region.offset() + 64 + skew;
        let mut storage = vec![0u8; bytes.len() + extra];
        let base = storage.as_ptr() as usize;
        let aligned = base.wrapping_add(align - 1) & !(align - 1);
        let start = aligned - base + role.offset() + skew;
        storage[start..start + bytes.len()].copy_from_slice(bytes);
        let value = Self {
            storage,
            start,
            len: bytes.len(),
            skew,
            role,
        };
        let address = value.as_slice().as_ptr() as usize;
        assert_eq!(address % 64, skew, "fixture 64-byte alignment");
        assert_eq!(
            address % PAGE_BYTES,
            (role.offset() + skew) % PAGE_BYTES,
            "fixture page offset"
        );
        value
    }

    fn from_packed<F: FieldBuffer>(len: usize, seed: u64, role: Role, skew: usize) -> Self {
        assert_eq!(
            len % F::BYTES,
            0,
            "cell row length {len} is not a whole number of {}-byte elements",
            F::BYTES
        );
        Self::from_bytes(&fixture::packed::<F>(len / F::BYTES, seed), role, skew)
    }

    fn as_slice(&self) -> &[u8] {
        &self.storage[self.start..self.start + self.len]
    }

    fn as_mut(&mut self) -> &mut [u8] {
        &mut self.storage[self.start..self.start + self.len]
    }

    fn duplicate(&self) -> Self {
        Self::from_bytes(self.as_slice(), self.role, self.skew)
    }
}

fn seed_for(id: &str) -> u64 {
    let mut state = 0xcbf2_9ce4_8422_2325u64;
    for byte in id.bytes() {
        state ^= u64::from(byte);
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    state
}

fn element<F: FieldBuffer>(seed: u64) -> Elem<F> {
    let mut state = seed | 1;
    let value = fixture::element::<F>(&mut state);
    // Keep timed coefficients off the zero and one shortcuts so a cell
    // measures the multiply path rather than the dispatch fast paths.
    if value.is_zero() || value.is_one() {
        <Elem<F> as FieldElem>::from_index(2).unwrap_or(value)
    } else {
        value
    }
}

fn check(cell: &Cell, actual: &[u8], expected: &[u8]) {
    fixture::assert_bytes_equal(&cell.id, actual, expected);
    println!("OUTPUT\t{}\t{}", cell.id, crate::sha256::hex(actual));
}

fn run_case(
    cell: &Cell,
    ctx: &crate::Ctx,
    dst: Aligned,
    expected: impl FnOnce(&mut [u8]),
    mut call: impl FnMut(&mut [u8]),
) -> Option<f64> {
    if ctx.check {
        let mut expected_buf = dst.duplicate();
        expected(expected_buf.as_mut());
        let mut actual = dst;
        call(actual.as_mut());
        check(cell, actual.as_slice(), expected_buf.as_slice());
        None
    } else {
        let mut actual = dst;
        let ns = ctx.measure(|| call(actual.as_mut()));
        // Keep the destination observable so no iteration's stores can be
        // dropped as dead.
        fixture::observe(actual.as_slice());
        Some(ns)
    }
}

fn check_result(cell: &Cell, actual: &[u8], expected: &[u8]) {
    fixture::assert_bytes_equal(&cell.id, actual, expected);
    println!("OUTPUT\t{}\t{}", cell.id, crate::sha256::hex(actual));
}

// --- registration -----------------------------------------------------------

struct Meta {
    field: &'static str,
    operation: &'static str,
    prep: &'static str,
    row_len: usize,
    rows: usize,
    sources: usize,
    align_offset: usize,
    family: &'static str,
}

fn push(cells: &mut Vec<Cell>, meta: Meta, run: fn(&Cell, &crate::Ctx) -> Option<f64>) {
    let alignment = if meta.align_offset == 0 {
        "aligned".to_string()
    } else {
        format!("offset{}", meta.align_offset)
    };
    let id = format!(
        "{}/{}/{}/{}x{}x{}/{alignment}",
        meta.field, meta.operation, meta.prep, meta.row_len, meta.rows, meta.sources
    );
    assert!(
        cells.iter().all(|cell| cell.id != id),
        "duplicate cell id {id}"
    );
    let bytes = meta
        .row_len
        .checked_mul(meta.rows)
        .and_then(|value| value.checked_mul(meta.sources))
        .expect("cell traffic overflows");
    cells.push(Cell {
        id,
        field: meta.field,
        operation: meta.operation,
        prep: meta.prep,
        row_len: meta.row_len,
        rows: meta.rows,
        sources: meta.sources,
        alignment,
        align_offset: meta.align_offset,
        family: meta.family,
        bytes,
        run,
    });
}

fn push_single<F: FieldBuffer>(
    cells: &mut Vec<Cell>,
    field: &'static str,
    family: &'static str,
    operation: &'static str,
    prep: &'static str,
    row_len: usize,
    align_offset: usize,
) {
    push(
        cells,
        Meta {
            field,
            operation,
            prep,
            row_len,
            rows: 1,
            sources: 1,
            align_offset,
            family,
        },
        run_single::<F>,
    );
}

fn push_rows<F: FieldBuffer>(
    cells: &mut Vec<Cell>,
    field: &'static str,
    family: &'static str,
    operation: &'static str,
    row_len: usize,
    rows: usize,
) {
    push(
        cells,
        Meta {
            field,
            operation,
            prep: "unprepared",
            row_len,
            rows,
            sources: 1,
            align_offset: 0,
            family,
        },
        run_rows::<F>,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_scatter<F: FieldBuffer>(
    cells: &mut Vec<Cell>,
    field: &'static str,
    family: &'static str,
    operation: &'static str,
    prep: &'static str,
    row_len: usize,
    rows: usize,
    align_offset: usize,
) {
    push(
        cells,
        Meta {
            field,
            operation,
            prep,
            row_len,
            rows,
            sources: 1,
            align_offset,
            family,
        },
        run_scatter::<F>,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_gather<F: FieldBuffer>(
    cells: &mut Vec<Cell>,
    field: &'static str,
    family: &'static str,
    operation: &'static str,
    prep: &'static str,
    row_len: usize,
    sources: usize,
    align_offset: usize,
) {
    push(
        cells,
        Meta {
            field,
            operation,
            prep,
            row_len,
            rows: 1,
            sources,
            align_offset,
            family,
        },
        run_gather::<F>,
    );
}

#[allow(clippy::too_many_arguments)]
fn push_matrix<F: FieldBuffer>(
    cells: &mut Vec<Cell>,
    field: &'static str,
    family: &'static str,
    operation: &'static str,
    prep: &'static str,
    row_len: usize,
    rows: usize,
    sources: usize,
    align_offset: usize,
) {
    let run = if operation.ends_with("_at") {
        run_matrix_at::<F> as fn(&Cell, &crate::Ctx) -> Option<f64>
    } else {
        run_matrix::<F> as fn(&Cell, &crate::Ctx) -> Option<f64>
    };
    push(
        cells,
        Meta {
            field,
            operation,
            prep,
            row_len,
            rows,
            sources,
            align_offset,
            family,
        },
        run,
    );
}

fn push_bits(
    cells: &mut Vec<Cell>,
    operation: &'static str,
    prep: &'static str,
    row_len: usize,
    rows: usize,
    sources: usize,
    align_offset: usize,
) {
    push(
        cells,
        Meta {
            field: fields::GF2_BITS,
            operation,
            prep,
            row_len,
            rows,
            sources,
            align_offset,
            family: fields::FAMILY_GF2,
        },
        run_bits,
    );
}

fn add_field<F: FieldBuffer>(cells: &mut Vec<Cell>, field: &'static str, family: &'static str) {
    for (operation, prep, row_len, offset) in [
        ("mul_add", "unprepared", SMALL, 0),
        ("mul_add", "unprepared", MID, 0),
        ("mul_add", "unprepared", LARGE, 0),
        ("mul_add", "prepared", SMALL, 0),
        ("mul_add", "prepared", MID, 0),
        ("mul_add", "prepared", LARGE, 0),
        ("mul_into", "unprepared", SMALL, 0),
        ("mul_into", "unprepared", MID, 0),
        ("mul_into", "unprepared", LARGE, 0),
        ("mul_into", "prepared", MID, 0),
        ("mul_into", "prepared", LARGE, 0),
        ("mul_assign", "unprepared", MID, 0),
        ("mul_assign", "prepared", MID, 0),
        ("mul_elementwise_into", "unprepared", MID, 0),
        ("mul_elementwise_assign", "unprepared", MID, 0),
        ("add_assign", "unprepared", MID, 0),
        ("sub_assign", "unprepared", MID, 0),
        ("add_assign_scalar", "unprepared", MID, 0),
        ("sub_assign_scalar", "unprepared", MID, 0),
        ("mul_add", "unprepared", MID, OFFSET),
        ("mul_into", "prepared", MID, OFFSET),
    ] {
        push_single::<F>(cells, field, family, operation, prep, row_len, offset);
    }
    push_rows::<F>(cells, field, family, "add_assign_rows", MID, 8);
    push_rows::<F>(cells, field, family, "sub_assign_rows", MID, 8);

    for (operation, prep, row_len, rows, offset) in [
        ("mul_add_scatter", "unprepared", MID, 2, 0),
        ("mul_add_scatter", "unprepared", MID, 16, 0),
        ("mul_add_scatter", "prepared", MID, 8, 0),
        ("mul_add_scatter", "prepared", LARGE, 8, 0),
        ("mul_add_scatter", "unprepared", SMALL, 16, 0),
        ("mul_into_scatter", "unprepared", MID, 8, 0),
        ("mul_add_scatter", "unprepared", MID, 8, OFFSET),
    ] {
        push_scatter::<F>(cells, field, family, operation, prep, row_len, rows, offset);
    }

    for (operation, prep, row_len, sources, offset) in [
        ("add_gather_offsets", "unprepared", MID, 8, 0),
        ("add_gather_offsets", "unprepared", MID, 8, OFFSET),
        ("mul_add_gather", "unprepared", MID, 8, 0),
        ("mul_add_gather", "prepared", MID, 8, 0),
        ("mul_into_gather", "unprepared", MID, 8, 0),
        ("mul_into_gather", "prepared", MID, 8, 0),
        ("mul_add_gather", "unprepared", LARGE, 8, 0),
        ("mul_add_gather", "unprepared", SMALL, 16, 0),
    ] {
        push_gather::<F>(
            cells, field, family, operation, prep, row_len, sources, offset,
        );
    }

    for (operation, prep, row_len, rows, sources, offset) in [
        ("mul_add_matrix", "unprepared", MID, 8, 8, 0),
        ("mul_add_matrix", "prepared", MID, 8, 8, 0),
        ("mul_into_matrix", "unprepared", MID, 8, 8, 0),
        ("mul_into_matrix", "prepared", MID, 8, 8, 0),
        ("mul_add_matrix", "unprepared", SMALL, 16, 8, 0),
        ("mul_add_matrix", "unprepared", LARGE, 8, 8, 0),
        ("mul_add_matrix", "unprepared", MID, 2, 4, 0),
        ("mul_add_matrix_at", "unprepared", MID, 8, 8, OFFSET),
        ("mul_add_matrix_at", "prepared", MID, 8, 8, OFFSET),
        ("mul_into_matrix_at", "unprepared", MID, 8, 8, 0),
        ("mul_into_matrix_at", "prepared", MID, 8, 8, 0),
        ("mul_into_matrix_at", "prepared", MID, 8, 8, OFFSET),
    ] {
        push_matrix::<F>(
            cells, field, family, operation, prep, row_len, rows, sources, offset,
        );
    }
}

fn add_bits(cells: &mut Vec<Cell>) {
    for (operation, prep, row_len, rows, sources, offset) in [
        ("xor_assign", "unprepared", SMALL, 1, 1, 0),
        ("xor_assign", "unprepared", 16, 1, 1, 0),
        ("xor_assign", "unprepared", 128, 1, 1, 0),
        ("xor_assign", "unprepared", 256, 1, 1, 0),
        ("xor_assign", "unprepared", MID, 1, 1, 0),
        ("xor_assign", "unprepared", LARGE, 1, 1, 0),
        ("xor_assign", "unprepared", MID, 1, 1, OFFSET),
        ("and_into", "unprepared", MID, 1, 2, 0),
        ("and_into", "unprepared", LARGE, 1, 2, 0),
        ("and_assign", "unprepared", MID, 1, 1, 0),
        ("andnot_assign", "unprepared", MID, 1, 1, 0),
        ("xor_range", "unprepared", MID, 1, 1, 0),
        ("xor_range", "unprepared", LARGE, 1, 1, 0),
        ("xor_range", "prepared", MID, 1, 1, 0),
        ("clear_range", "unprepared", MID, 1, 1, 0),
        ("set_range", "unprepared", MID, 1, 1, 0),
        ("weight", "unprepared", MID, 1, 1, 0),
        ("weight", "unprepared", LARGE, 1, 1, 0),
        ("dot_product", "unprepared", MID, 1, 2, 0),
        ("dot_product", "unprepared", LARGE, 1, 2, 0),
        ("xor_gather", "unprepared", MID, 1, 8, 0),
        ("xor_gather", "unprepared", LARGE, 1, 8, 0),
        ("xor_gather", "unprepared", MID, 1, 8, OFFSET),
    ] {
        push_bits(cells, operation, prep, row_len, rows, sources, offset);
    }
}

fn add_gf8_thresholds<F: FieldBuffer>(cells: &mut Vec<Cell>, field: &'static str) {
    // The affine in-place scale switches to the nibble shuffle at this
    // destination length; the small rows above stay on the affine body.
    push_single::<F>(
        cells,
        field,
        fields::FAMILY_GF,
        "mul_assign",
        "unprepared",
        LARGE,
        0,
    );
    push_single::<F>(
        cells,
        field,
        fields::FAMILY_GF,
        "mul_assign",
        "prepared",
        LARGE,
        0,
    );
    // Below `XOR_INLINE_MAX` the offset gather and the byte XOR run inline.
    push_gather::<F>(
        cells,
        field,
        fields::FAMILY_GF,
        "add_gather_offsets",
        "unprepared",
        16,
        8,
        0,
    );
    // The fused overwrite prefetches from this length; non-temporal stores
    // start at the streaming floor.
    for row_len in [24_576, PREFETCH_BYTES] {
        for prep in ["unprepared", "prepared"] {
            push_single::<F>(
                cells,
                field,
                fields::FAMILY_GF,
                "mul_into",
                prep,
                row_len,
                0,
            );
        }
    }
    // Scatter cohort fills: three rows leave a partial pair and single, five
    // rows leave one after a full four-row cohort.
    for rows in [1, 3, 5, 6, 7] {
        for prep in ["unprepared", "prepared"] {
            push_scatter::<F>(
                cells,
                field,
                fields::FAMILY_GF,
                "mul_add_scatter",
                prep,
                MID,
                rows,
                0,
            );
        }
    }
    for (operation, prep, row_len, rows, sources, offset) in [
        // FEW_TERMS: two terms keep the widest row group.
        ("mul_add_matrix", "unprepared", MID, 4, 2, 0),
        ("mul_add_matrix", "prepared", MID, 4, 2, 0),
        // MATRIX_PREFETCH_MIN: an 8 KiB row reaches the prefetching body.
        ("mul_add_matrix", "unprepared", 8_192, 8, 8, 0),
        ("mul_add_matrix", "prepared", 8_192, 8, 8, 0),
        // RESOLVE_CHUNK and the column-blocked walk: 33 terms cross the
        // resolve chunk; the row counts select each side of the column gate.
        ("mul_add_matrix", "unprepared", 16_384, 8, 33, 0),
        ("mul_add_matrix", "prepared", 16_384, 8, 33, 0),
        ("mul_add_matrix", "unprepared", 16_384, 4, 33, 0),
        ("mul_add_matrix", "unprepared", 16_384, 3, 33, 0),
        ("mul_add_matrix", "unprepared", MID, 8, 33, 0),
        // THREE_LANE_{MIN_TERMS,MIN_ROW,MAX_ROW}.
        ("mul_add_matrix", "unprepared", 512, 3, 6, 0),
        ("mul_add_matrix", "unprepared", 256, 3, 6, 0),
        ("mul_add_matrix", "unprepared", 512, 3, 5, 0),
        ("mul_add_matrix", "unprepared", 131_072, 3, 6, 0),
        ("mul_add_matrix", "prepared", 131_072, 3, 6, 0),
        ("mul_add_matrix", "unprepared", 131_136, 3, 6, 0),
        ("mul_add_matrix", "prepared", 131_136, 3, 6, 0),
        ("mul_add_matrix", "unprepared", MID, 8, 32, 0),
        ("mul_add_matrix", "unprepared", 8_192, 8, 32, 0),
        ("mul_add_matrix", "prepared", 8_192, 8, 32, 0),
        ("mul_add_matrix", "unprepared", 8_192, 8, 33, 0),
    ] {
        push_matrix::<F>(
            cells,
            field,
            fields::FAMILY_GF,
            operation,
            prep,
            row_len,
            rows,
            sources,
            offset,
        );
    }
}

fn add_qm31_thresholds(cells: &mut Vec<Cell>) {
    for operation in [
        "add_assign",
        "sub_assign",
        "add_assign_scalar",
        "sub_assign_scalar",
    ] {
        push_single::<fgf::extension::QuadMersenne31>(
            cells,
            fields::QM31,
            fields::FAMILY_M31,
            operation,
            "unprepared",
            32,
            0,
        );
    }
    for operation in [
        "mul_add",
        "mul_assign",
        "mul_into",
        "mul_elementwise_into",
        "mul_elementwise_assign",
    ] {
        push_single::<fgf::extension::QuadMersenne31>(
            cells,
            fields::QM31,
            fields::FAMILY_M31,
            operation,
            "unprepared",
            24,
            0,
        );
    }
}

fn add_residue_delayed<F: FieldBuffer>(cells: &mut Vec<Cell>, field: &'static str) {
    for (operation, row_len, rows, sources) in [
        ("mul_add_matrix", 256, 4, 8),
        ("mul_add_matrix", 256, 4, 7),
        ("mul_add_matrix", 256, 4, 9),
        ("mul_add_matrix", 128, 4, 8),
        ("mul_add_matrix", 256, 4, 4),
        ("mul_into_matrix", 256, 4, 8),
        ("mul_into_matrix", 128, 4, 8),
    ] {
        push_matrix::<F>(
            cells,
            field,
            fields::FAMILY_PRIME,
            operation,
            "prepared",
            row_len,
            rows,
            sources,
            0,
        );
    }
}

/// Add one `add_assign` cell over a binary field.
fn push_xor_axis<F: FieldBuffer>(cells: &mut Vec<Cell>, field: &'static str, row_len: usize) {
    push_single::<F>(
        cells,
        field,
        fields::FAMILY_GF,
        "add_assign",
        "unprepared",
        row_len,
        0,
    );
}

/// Cells for the uncited routing-threshold evidence sessions.
///
/// Each shape is an endpoint, integral grid point, or interval midpoint of
/// one threshold session's axis. Shapes the registry already registers are
/// reused by the session declarations rather than duplicated here.
fn add_threshold_evidence_cells(cells: &mut Vec<Cell>) {
    // `bytes.rs::XOR_INLINE_MAX` (31 bytes, `<=`). Byte-wide rows can sit
    // exactly on the cutoff; wider elements only represent whole elements,
    // so they contribute the representable grid values.
    for row_len in [16, 24, 32, 48] {
        push_xor_axis::<fields::Gf8Aes>(cells, fields::GF8_AES, row_len);
        push_xor_axis::<fields::Gf8Rs>(cells, fields::GF8_RS, row_len);
        push_xor_axis::<fields::Gf16Rijndael>(cells, fields::GF16_RIJNDAEL, row_len);
        push_xor_axis::<fields::Gf16FanPaar>(cells, fields::FANPAAR16, row_len);
        push_xor_axis::<fields::Gf32Rijndael>(cells, fields::GF32_RIJNDAEL, row_len);
        push_xor_axis::<fields::Gf64Rijndael>(cells, fields::GF64_RIJNDAEL, row_len);
        push_xor_axis::<fields::PackedRs16>(cells, fields::PACKED_RS16, row_len);
        push_xor_axis::<fields::CustomRijndael16>(cells, fields::CUSTOM_RIJNDAEL16, row_len);
    }
    for row_len in [31, 62] {
        push_xor_axis::<fields::Gf8Aes>(cells, fields::GF8_AES, row_len);
        push_xor_axis::<fields::Gf8Rs>(cells, fields::GF8_RS, row_len);
    }
    // Only byte and 16-bit elements represent the 62-byte endpoint; 32- and
    // 64-bit elements cannot.
    push_xor_axis::<fields::Gf16Rijndael>(cells, fields::GF16_RIJNDAEL, 62);
    push_xor_axis::<fields::Gf16FanPaar>(cells, fields::FANPAAR16, 62);
    push_xor_axis::<fields::PackedRs16>(cells, fields::PACKED_RS16, 62);
    push_xor_axis::<fields::CustomRijndael16>(cells, fields::CUSTOM_RIJNDAEL16, 62);
    // `bytes.rs:37` compares the same cutoff in the offset-addressed gather;
    // one gather target on each side of the grid covers that call site.
    for row_len in [24, 32] {
        push_gather::<fields::Gf8Aes>(
            cells,
            fields::GF8_AES,
            fields::FAMILY_GF,
            "add_gather_offsets",
            "unprepared",
            row_len,
            8,
            0,
        );
        push_gather::<fields::Gf8Rs>(
            cells,
            fields::GF8_RS,
            fields::FAMILY_GF,
            "add_gather_offsets",
            "unprepared",
            row_len,
            8,
            0,
        );
    }

    // `gf8.rs` in-place scale below 65 536 bytes, GFNI without native AES.
    // The 65 536-byte shapes are already registered by `add_gf8_thresholds`.
    for prep in ["unprepared", "prepared"] {
        for row_len in [32_768, 49_152, 98_304, 131_072] {
            push_single::<fields::Gf8Rs>(
                cells,
                fields::GF8_RS,
                fields::FAMILY_GF,
                "mul_assign",
                prep,
                row_len,
                0,
            );
        }
    }

    // `quad_mersenne31.rs::VECTOR_ADD_MIN_BYTES`: 32-byte shapes are
    // registered by `add_qm31_thresholds`.
    for operation in [
        "add_assign",
        "sub_assign",
        "add_assign_scalar",
        "sub_assign_scalar",
    ] {
        for row_len in [48, 64, 96, 128] {
            push_single::<fgf::extension::QuadMersenne31>(
                cells,
                fields::QM31,
                fields::FAMILY_M31,
                operation,
                "unprepared",
                row_len,
                0,
            );
        }
    }

    // `quad_mersenne31.rs::VECTOR_MUL_MIN_BYTES`: 24-byte shapes are
    // registered by `add_qm31_thresholds`; at 64 bytes the unprepared
    // `mul_add`/`mul_into` and prepared `mul_add` shapes already exist.
    for operation in [
        "mul_add",
        "mul_assign",
        "mul_into",
        "mul_elementwise_into",
        "mul_elementwise_assign",
    ] {
        for row_len in [16, 32, 48] {
            push_single::<fgf::extension::QuadMersenne31>(
                cells,
                fields::QM31,
                fields::FAMILY_M31,
                operation,
                "unprepared",
                row_len,
                0,
            );
        }
    }
    for (operation, prep) in [
        ("mul_into", "prepared"),
        ("mul_assign", "unprepared"),
        ("mul_assign", "prepared"),
        ("mul_elementwise_into", "unprepared"),
        ("mul_elementwise_assign", "unprepared"),
    ] {
        push_single::<fgf::extension::QuadMersenne31>(
            cells,
            fields::QM31,
            fields::FAMILY_M31,
            operation,
            prep,
            64,
            0,
        );
    }

    // `x86.rs::NT_STORE_MIN`: the 2 MiB unprepared GF(2^8)-AES and GF(2^16)
    // shapes already exist; every other streaming-floor shape is new.
    for row_len in [1_048_576, 1_572_864] {
        for prep in ["unprepared", "prepared"] {
            push_single::<fields::Gf8Aes>(
                cells,
                fields::GF8_AES,
                fields::FAMILY_GF,
                "mul_into",
                prep,
                row_len,
                0,
            );
            push_single::<fields::Gf8Rs>(
                cells,
                fields::GF8_RS,
                fields::FAMILY_GF,
                "mul_into",
                prep,
                row_len,
                0,
            );
            push_single::<fields::Gf16Rijndael>(
                cells,
                fields::GF16_RIJNDAEL,
                fields::FAMILY_GF,
                "mul_into",
                prep,
                row_len,
                0,
            );
        }
    }
    push_single::<fields::Gf8Rs>(
        cells,
        fields::GF8_RS,
        fields::FAMILY_GF,
        "mul_into",
        "unprepared",
        2_097_152,
        0,
    );
    push_single::<fields::Gf8Aes>(
        cells,
        fields::GF8_AES,
        fields::FAMILY_GF,
        "mul_into",
        "prepared",
        2_097_152,
        0,
    );
    push_single::<fields::Gf8Rs>(
        cells,
        fields::GF8_RS,
        fields::FAMILY_GF,
        "mul_into",
        "prepared",
        2_097_152,
        0,
    );
    push_single::<fields::Gf16Rijndael>(
        cells,
        fields::GF16_RIJNDAEL,
        fields::FAMILY_GF,
        "mul_into",
        "prepared",
        2_097_152,
        0,
    );
    for row_len in [3_145_728, 4_194_304] {
        for prep in ["unprepared", "prepared"] {
            push_single::<fields::Gf8Aes>(
                cells,
                fields::GF8_AES,
                fields::FAMILY_GF,
                "mul_into",
                prep,
                row_len,
                0,
            );
            push_single::<fields::Gf8Rs>(
                cells,
                fields::GF8_RS,
                fields::FAMILY_GF,
                "mul_into",
                prep,
                row_len,
                0,
            );
            push_single::<fields::Gf16Rijndael>(
                cells,
                fields::GF16_RIJNDAEL,
                fields::FAMILY_GF,
                "mul_into",
                prep,
                row_len,
                0,
            );
        }
    }

    // `x86/bytes/avx512.rs` XOR gates: one byte-XOR axis shared by every
    // binary element width.
    for row_len in [128, 192, 256, 384, 512, 768, 1024] {
        push_xor_axis::<fields::Gf8Aes>(cells, fields::GF8_AES, row_len);
        push_xor_axis::<fields::Gf8Rs>(cells, fields::GF8_RS, row_len);
        push_xor_axis::<fields::Gf16Rijndael>(cells, fields::GF16_RIJNDAEL, row_len);
        push_xor_axis::<fields::Gf16FanPaar>(cells, fields::FANPAAR16, row_len);
        push_xor_axis::<fields::Gf32Rijndael>(cells, fields::GF32_RIJNDAEL, row_len);
        push_xor_axis::<fields::Gf64Rijndael>(cells, fields::GF64_RIJNDAEL, row_len);
        push_xor_axis::<fields::PackedRs16>(cells, fields::PACKED_RS16, row_len);
        push_xor_axis::<fields::CustomRijndael16>(cells, fields::CUSTOM_RIJNDAEL16, row_len);
    }

    // `x86/gf8/avx512/gather.rs::GATHER_PEEL_MIN`.
    for prep in ["unprepared", "prepared"] {
        for row_len in [1536, 2304, 3072, 4608, 6144] {
            push_gather::<fields::Gf8Aes>(
                cells,
                fields::GF8_AES,
                fields::FAMILY_GF,
                "mul_add_gather",
                prep,
                row_len,
                8,
                0,
            );
            push_gather::<fields::Gf8Rs>(
                cells,
                fields::GF8_RS,
                fields::FAMILY_GF,
                "mul_add_gather",
                prep,
                row_len,
                8,
                0,
            );
        }
    }

    // `x86/gf8/avx512/scatter.rs::SCATTER_PEEL_MIN`.
    for prep in ["unprepared", "prepared"] {
        for row_len in [256, 384, 512, 768, 1024] {
            push_scatter::<fields::Gf8Aes>(
                cells,
                fields::GF8_AES,
                fields::FAMILY_GF,
                "mul_add_scatter",
                prep,
                row_len,
                8,
                0,
            );
            push_scatter::<fields::Gf8Rs>(
                cells,
                fields::GF8_RS,
                fields::FAMILY_GF,
                "mul_add_scatter",
                prep,
                row_len,
                8,
                0,
            );
        }
    }

    // `x86/gf8/avx512/matrix.rs::MATRIX_PEEL_MIN`: 4096 and 8192 are
    // registered by `add_field` and `add_gf8_thresholds`.
    for prep in ["unprepared", "prepared"] {
        for row_len in [6144, 12_288, 16_384] {
            push_matrix::<fields::Gf8Aes>(
                cells,
                fields::GF8_AES,
                fields::FAMILY_GF,
                "mul_add_matrix",
                prep,
                row_len,
                8,
                8,
                0,
            );
            push_matrix::<fields::Gf8Rs>(
                cells,
                fields::GF8_RS,
                fields::FAMILY_GF,
                "mul_add_matrix",
                prep,
                row_len,
                8,
                8,
                0,
            );
        }
    }
}

fn add_recip<F: FieldBuffer>(cells: &mut Vec<Cell>, field: &'static str, family: &'static str) {
    for (operation, row_len) in [
        ("inv_into", SMALL),
        ("inv_into", MID),
        ("inv_assign", SMALL),
        ("inv_assign", MID),
        ("div_elementwise_into", SMALL),
        ("div_elementwise_into", MID),
        ("div_elementwise_assign", SMALL),
        ("div_elementwise_assign", MID),
    ] {
        push_single::<F>(cells, field, family, operation, "unprepared", row_len, 0);
    }
}

/// Build the fixed cell list in registration order.
pub fn registry() -> Vec<Cell> {
    let mut cells = Vec::new();
    add_field::<fields::Gf8Aes>(&mut cells, fields::GF8_AES, fields::FAMILY_GF);
    add_field::<fields::Gf8Rs>(&mut cells, fields::GF8_RS, fields::FAMILY_GF);
    add_field::<fields::Gf16Rijndael>(&mut cells, fields::GF16_RIJNDAEL, fields::FAMILY_GF);
    add_field::<fields::Gf16FanPaar>(&mut cells, fields::FANPAAR16, fields::FAMILY_GF);
    add_field::<fields::Gf32Rijndael>(&mut cells, fields::GF32_RIJNDAEL, fields::FAMILY_GF);
    add_field::<fields::Gf64Rijndael>(&mut cells, fields::GF64_RIJNDAEL, fields::FAMILY_GF);
    add_field::<fields::ResidueU8>(&mut cells, fields::RESIDUE_101, fields::FAMILY_PRIME);
    add_field::<fields::ResidueU8High>(&mut cells, fields::RESIDUE_251, fields::FAMILY_PRIME);
    add_field::<fields::ResidueU32>(&mut cells, fields::RESIDUE_1E9P7, fields::FAMILY_PRIME);
    add_field::<fields::ResidueU64>(&mut cells, fields::RESIDUE_MAX, fields::FAMILY_PRIME);
    add_field::<fgf::prime::Mersenne31>(&mut cells, fields::M31, fields::FAMILY_M31);
    add_field::<fgf::prime::Goldilocks>(&mut cells, fields::GOLDILOCKS, fields::FAMILY_GDL);
    add_field::<fgf::extension::QuadMersenne31>(&mut cells, fields::QM31, fields::FAMILY_M31);
    add_field::<fields::PackedRs16>(&mut cells, fields::PACKED_RS16, fields::FAMILY_GF);
    add_field::<fields::ComponentsGf49>(&mut cells, fields::COMPONENTS_GF49, fields::FAMILY_EXT);
    add_field::<fields::CustomRijndael16>(&mut cells, fields::CUSTOM_RIJNDAEL16, fields::FAMILY_GF);
    add_gf8_thresholds::<fields::Gf8Aes>(&mut cells, fields::GF8_AES);
    add_gf8_thresholds::<fields::Gf8Rs>(&mut cells, fields::GF8_RS);
    add_qm31_thresholds(&mut cells);
    add_residue_delayed::<fields::ResidueU32>(&mut cells, fields::RESIDUE_1E9P7);
    add_residue_delayed::<fields::ResidueU64>(&mut cells, fields::RESIDUE_MAX);
    add_recip::<fgf::prime::Mersenne31>(&mut cells, fields::M31, fields::FAMILY_M31);
    add_recip::<fields::Gf8Aes>(&mut cells, fields::GF8_AES, fields::FAMILY_GF);
    // One streaming-store cell per routing family: GF(2^8) and GF(2^16).
    push_single::<fields::Gf8Aes>(
        &mut cells,
        fields::GF8_AES,
        fields::FAMILY_GF,
        "mul_into",
        "unprepared",
        NT_BYTES,
        0,
    );
    push_single::<fields::Gf16Rijndael>(
        &mut cells,
        fields::GF16_RIJNDAEL,
        fields::FAMILY_GF,
        "mul_into",
        "unprepared",
        NT_BYTES,
        0,
    );
    for prep in ["unprepared", "prepared"] {
        for row_len in [24_576, PREFETCH_BYTES] {
            push_single::<fields::Gf16Rijndael>(
                &mut cells,
                fields::GF16_RIJNDAEL,
                fields::FAMILY_GF,
                "mul_into",
                prep,
                row_len,
                0,
            );
        }
    }
    add_bits(&mut cells);
    add_threshold_evidence_cells(&mut cells);
    cells
}

// --- single-row runners -----------------------------------------------------

struct Single<F: FieldBuffer> {
    src: Aligned,
    dst: Aligned,
    rhs: Aligned,
    coeff: Elem<F>,
}

impl<F: FieldBuffer> Single<F> {
    fn new(cell: &Cell) -> Self {
        let seed = seed_for(&cell.id);
        Self {
            src: Aligned::from_packed::<F>(cell.row_len, seed ^ 0x51, Role::Src, 0),
            dst: Aligned::from_packed::<F>(cell.row_len, seed ^ 0x52, Role::Dst, cell.align_offset),
            rhs: Aligned::from_packed::<F>(cell.row_len, seed ^ 0x53, Role::Src2, 0),
            coeff: element::<F>(seed ^ 0x54),
        }
    }
}

fn run_single<F: FieldBuffer>(cell: &Cell, ctx: &crate::Ctx) -> Option<f64> {
    let fixture = Single::<F>::new(cell);
    let coeff = fixture.coeff;
    let src = fixture.src.as_slice();
    let rhs = fixture.rhs.as_slice();
    match (cell.operation, cell.prep) {
        ("mul_add", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::mul_add::<F>(expected, coeff, src),
            |dst| ops::mul_add::<F>(dst, coeff, src),
        ),
        ("mul_add", "prepared") => {
            let prepared = ops::Coeff::<F>::new(coeff);
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| oracle::mul_add::<F>(expected, coeff, src),
                |dst| ops::mul_add_with::<F>(dst, &prepared, src),
            )
        }
        ("mul_into", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::mul_into::<F>(expected, coeff, src),
            |dst| ops::mul_into::<F>(dst, coeff, src),
        ),
        ("mul_into", "prepared") => {
            let prepared = ops::Coeff::<F>::new(coeff);
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| oracle::mul_into::<F>(expected, coeff, src),
                |dst| ops::mul_into_with::<F>(dst, &prepared, src),
            )
        }
        ("mul_assign", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::mul_assign::<F>(expected, coeff),
            |dst| ops::mul_assign::<F>(dst, coeff),
        ),
        ("mul_assign", "prepared") => {
            let prepared = ops::Coeff::<F>::new(coeff);
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| oracle::mul_assign::<F>(expected, coeff),
                |dst| ops::mul_assign_with::<F>(dst, &prepared),
            )
        }
        ("mul_elementwise_into", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::elementwise_into::<F>(expected, src, rhs),
            |dst| ops::mul_elementwise_into::<F>(dst, src, rhs),
        ),
        ("mul_elementwise_assign", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::elementwise_assign::<F>(expected, src),
            |dst| ops::mul_elementwise_assign::<F>(dst, src),
        ),
        ("add_assign", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::add_assign::<F>(expected, src),
            |dst| ops::add_assign::<F>(dst, src),
        ),
        ("sub_assign", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::sub_assign::<F>(expected, src),
            |dst| ops::sub_assign::<F>(dst, src),
        ),
        ("add_assign_scalar", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::add_scalar::<F>(expected, coeff),
            |dst| ops::add_assign_scalar::<F>(dst, coeff),
        ),
        ("sub_assign_scalar", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::sub_scalar::<F>(expected, coeff),
            |dst| ops::sub_assign_scalar::<F>(dst, coeff),
        ),
        ("inv_into", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::inv_into::<F>(expected, src),
            |dst| ops::inv_into::<F>(dst, src),
        ),
        ("inv_assign", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::inv_assign::<F>(expected),
            |dst| ops::inv_assign::<F>(dst),
        ),
        ("div_elementwise_into", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::div_elementwise_into::<F>(expected, src, rhs),
            |dst| ops::div_elementwise_into::<F>(dst, src, rhs),
        ),
        ("div_elementwise_assign", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::div_elementwise_assign::<F>(expected, src),
            |dst| ops::div_elementwise_assign::<F>(dst, src),
        ),
        other => panic!("unregistered single-row operation {other:?}"),
    }
}

// --- row runners ------------------------------------------------------------

struct Rows {
    src: Aligned,
    dst: Aligned,
}

fn run_rows<F: FieldBuffer>(cell: &Cell, ctx: &crate::Ctx) -> Option<f64> {
    let seed = seed_for(&cell.id);
    let len = cell.row_len * cell.rows;
    let fixture = Rows {
        src: Aligned::from_packed::<F>(len, seed ^ 0x61, Role::Src, 0),
        dst: Aligned::from_packed::<F>(len, seed ^ 0x62, Role::Dst, cell.align_offset),
    };
    let src = fixture.src.as_slice();
    match cell.operation {
        "add_assign_rows" => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::add_assign::<F>(expected, src),
            |dst| ops::add_assign_rows::<F>(dst, cell.row_len, src),
        ),
        "sub_assign_rows" => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::sub_assign::<F>(expected, src),
            |dst| ops::sub_assign_rows::<F>(dst, cell.row_len, src),
        ),
        other => panic!("unregistered row operation {other:?}"),
    }
}

// --- scatter runners --------------------------------------------------------

struct Scatter<F: FieldBuffer> {
    src: Aligned,
    rows: Aligned,
    coeffs: Vec<Elem<F>>,
}

impl<F: FieldBuffer> Scatter<F> {
    fn new(cell: &Cell) -> Self {
        let seed = seed_for(&cell.id);
        let mut state = seed ^ 0x70;
        let coeffs = (0..cell.rows)
            .map(|index| element::<F>(fixture::lcg(&mut state) ^ index as u64))
            .collect();
        Self {
            src: Aligned::from_packed::<F>(cell.row_len, seed ^ 0x71, Role::Src, 0),
            rows: Aligned::from_packed::<F>(
                cell.row_len * cell.rows,
                seed ^ 0x72,
                Role::Dst,
                cell.align_offset,
            ),
            coeffs,
        }
    }
}

fn run_scatter<F: FieldBuffer>(cell: &Cell, ctx: &crate::Ctx) -> Option<f64> {
    let fixture = Scatter::<F>::new(cell);
    let src = fixture.src.as_slice();
    let coeffs = &fixture.coeffs;
    match (cell.operation, cell.prep) {
        ("mul_add_scatter", "unprepared") => run_case(
            cell,
            ctx,
            fixture.rows.duplicate(),
            |expected| oracle::scatter_add::<F>(expected, cell.row_len, coeffs, src),
            |rows| ops::mul_add_scatter::<F>(rows, cell.row_len, coeffs, src),
        ),
        ("mul_add_scatter", "prepared") => {
            let coeffvec = ops::CoeffVec::<F>::new(coeffs);
            let view = coeffvec.as_ref();
            run_case(
                cell,
                ctx,
                fixture.rows.duplicate(),
                |expected| oracle::scatter_add::<F>(expected, cell.row_len, coeffs, src),
                |rows| ops::mul_add_scatter_with::<F>(rows, cell.row_len, view, src),
            )
        }
        ("mul_into_scatter", "unprepared") => run_case(
            cell,
            ctx,
            fixture.rows.duplicate(),
            |expected| oracle::scatter_into::<F>(expected, cell.row_len, coeffs, src),
            |rows| ops::mul_into_scatter::<F>(rows, cell.row_len, coeffs, src),
        ),
        other => panic!("unregistered scatter operation {other:?}"),
    }
}

// --- gather runners ---------------------------------------------------------

struct Gather<F: FieldBuffer> {
    dst: Aligned,
    region: Aligned,
    coeffs: Vec<Elem<F>>,
}

impl<F: FieldBuffer> Gather<F> {
    fn new(cell: &Cell) -> Self {
        let seed = seed_for(&cell.id);
        let mut state = seed ^ 0x80;
        let coeffs = (0..cell.sources)
            .map(|index| element::<F>(fixture::lcg(&mut state) ^ index as u64))
            .collect();
        Self {
            dst: Aligned::from_packed::<F>(cell.row_len, seed ^ 0x81, Role::Dst, cell.align_offset),
            region: Aligned::from_packed::<F>(
                cell.row_len * cell.sources,
                seed ^ 0x82,
                Role::Region,
                0,
            ),
            coeffs,
        }
    }
}

fn run_gather<F: FieldBuffer>(cell: &Cell, ctx: &crate::Ctx) -> Option<f64> {
    let fixture = Gather::<F>::new(cell);
    let region = fixture.region.as_slice();
    let srcs: Vec<&[u8]> = (0..cell.sources)
        .map(|index| &region[index * cell.row_len..(index + 1) * cell.row_len])
        .collect();
    let coeffs = &fixture.coeffs;
    match (cell.operation, cell.prep) {
        ("add_gather_offsets", "unprepared") => {
            let offsets: Vec<u32> = (0..cell.sources)
                .map(|index| (index * cell.row_len) as u32)
                .collect();
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| {
                    for &offset in &offsets {
                        let start = offset as usize;
                        oracle::add_assign::<F>(expected, &region[start..start + cell.row_len]);
                    }
                },
                |dst| ops::add_gather_offsets::<F>(dst, &offsets, region),
            )
        }
        ("mul_add_gather", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::gather_add::<F>(expected, coeffs, &srcs),
            |dst| ops::mul_add_gather::<F>(dst, coeffs, &srcs),
        ),
        ("mul_add_gather", "prepared") => {
            let coeffvec = ops::CoeffVec::<F>::new(coeffs);
            let view = coeffvec.as_ref();
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| oracle::gather_add::<F>(expected, coeffs, &srcs),
                |dst| ops::mul_add_gather_with::<F>(dst, view, &srcs),
            )
        }
        ("mul_into_gather", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::gather_into::<F>(expected, coeffs, &srcs),
            |dst| ops::mul_into_gather::<F>(dst, coeffs, &srcs),
        ),
        ("mul_into_gather", "prepared") => {
            let coeffvec = ops::CoeffVec::<F>::new(coeffs);
            let view = coeffvec.as_ref();
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| oracle::gather_into::<F>(expected, coeffs, &srcs),
                |dst| ops::mul_into_gather_with::<F>(dst, view, &srcs),
            )
        }
        other => panic!("unregistered gather operation {other:?}"),
    }
}

// --- matrix runners ---------------------------------------------------------

struct Matrix<F: FieldBuffer> {
    dst: Aligned,
    sources: Aligned,
    coeff_sets: Vec<Vec<Elem<F>>>,
}

impl<F: FieldBuffer> Matrix<F> {
    fn new(cell: &Cell) -> Self {
        let seed = seed_for(&cell.id);
        let mut state = seed ^ 0x90;
        let coeff_sets = (0..cell.sources)
            .map(|_| {
                (0..cell.rows)
                    .map(|_| element::<F>(fixture::lcg(&mut state)))
                    .collect()
            })
            .collect();
        Self {
            dst: Aligned::from_packed::<F>(
                cell.row_len * cell.rows,
                seed ^ 0x91,
                Role::Dst,
                cell.align_offset,
            ),
            sources: Aligned::from_packed::<F>(
                cell.row_len * cell.sources,
                seed ^ 0x92,
                Role::Src,
                0,
            ),
            coeff_sets,
        }
    }

    fn srcs(&self, cell: &Cell) -> Vec<&[u8]> {
        let sources = self.sources.as_slice();
        (0..cell.sources)
            .map(|index| &sources[index * cell.row_len..(index + 1) * cell.row_len])
            .collect()
    }

    fn terms<'a>(&'a self, srcs: &'a [&'a [u8]]) -> Vec<(&'a [Elem<F>], &'a [u8])> {
        self.coeff_sets
            .iter()
            .zip(srcs)
            .map(|(coeffs, src)| (coeffs.as_slice(), *src))
            .collect()
    }

    fn flat(&self) -> Vec<Elem<F>> {
        self.coeff_sets.iter().flatten().copied().collect()
    }
}

fn run_matrix<F: FieldBuffer>(cell: &Cell, ctx: &crate::Ctx) -> Option<f64> {
    let fixture = Matrix::<F>::new(cell);
    let srcs = fixture.srcs(cell);
    let terms = fixture.terms(&srcs);
    match (cell.operation, cell.prep) {
        ("mul_add_matrix", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::matrix_add::<F>(expected, cell.row_len, cell.rows, &terms),
            |rows| ops::mul_add_matrix::<F>(rows, cell.row_len, cell.rows, &terms),
        ),
        ("mul_add_matrix", "prepared") => {
            let matrix =
                ops::CoeffMatrix::<F>::from_source_major(cell.sources, cell.rows, &fixture.flat());
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| oracle::matrix_add::<F>(expected, cell.row_len, cell.rows, &terms),
                |rows| ops::mul_add_matrix_with::<F>(rows, cell.row_len, &matrix, &srcs),
            )
        }
        ("mul_into_matrix", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| oracle::matrix_into::<F>(expected, cell.row_len, cell.rows, &terms),
            |rows| ops::mul_into_matrix::<F>(rows, cell.row_len, cell.rows, &terms),
        ),
        ("mul_into_matrix", "prepared") => {
            let matrix =
                ops::CoeffMatrix::<F>::from_source_major(cell.sources, cell.rows, &fixture.flat());
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| oracle::matrix_into::<F>(expected, cell.row_len, cell.rows, &terms),
                |rows| ops::mul_into_matrix_with::<F>(rows, cell.row_len, &matrix, &srcs),
            )
        }
        other => panic!("unregistered matrix operation {other:?}"),
    }
}

struct MatrixAt<F: FieldBuffer> {
    dst: Aligned,
    sources: Aligned,
    row_starts: Vec<usize>,
    coeff_sets: Vec<Vec<Elem<F>>>,
}

impl<F: FieldBuffer> MatrixAt<F> {
    fn new(cell: &Cell) -> Self {
        let seed = seed_for(&cell.id);
        let total = (cell.rows - 1) * (cell.row_len + ROW_GAP) + cell.row_len;
        let row_starts = (0..cell.rows)
            .map(|index| index * (cell.row_len + ROW_GAP))
            .collect();
        let mut state = seed ^ 0x93;
        let coeff_sets = (0..cell.sources)
            .map(|_| {
                (0..cell.rows)
                    .map(|_| element::<F>(fixture::lcg(&mut state)))
                    .collect()
            })
            .collect();
        Self {
            dst: Aligned::from_packed::<F>(total, seed ^ 0x94, Role::Dst, cell.align_offset),
            sources: Aligned::from_packed::<F>(
                cell.row_len * cell.sources,
                seed ^ 0x95,
                Role::Src,
                0,
            ),
            row_starts,
            coeff_sets,
        }
    }

    fn srcs(&self, cell: &Cell) -> Vec<&[u8]> {
        let sources = self.sources.as_slice();
        (0..cell.sources)
            .map(|index| &sources[index * cell.row_len..(index + 1) * cell.row_len])
            .collect()
    }

    fn terms<'a>(&'a self, srcs: &'a [&'a [u8]]) -> Vec<(&'a [Elem<F>], &'a [u8])> {
        self.coeff_sets
            .iter()
            .zip(srcs)
            .map(|(coeffs, src)| (coeffs.as_slice(), *src))
            .collect()
    }

    fn flat(&self) -> Vec<Elem<F>> {
        self.coeff_sets.iter().flatten().copied().collect()
    }
}

fn run_matrix_at<F: FieldBuffer>(cell: &Cell, ctx: &crate::Ctx) -> Option<f64> {
    let fixture = MatrixAt::<F>::new(cell);
    let srcs = fixture.srcs(cell);
    let terms = fixture.terms(&srcs);
    match (cell.operation, cell.prep) {
        ("mul_add_matrix_at", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| {
                oracle::matrix_at_add::<F>(expected, cell.row_len, &fixture.row_starts, &terms);
            },
            |dst| {
                ops::mul_add_matrix_at::<F>(dst, cell.row_len, &fixture.row_starts, &terms);
            },
        ),
        ("mul_add_matrix_at", "prepared") => {
            let matrix =
                ops::CoeffMatrix::<F>::from_source_major(cell.sources, cell.rows, &fixture.flat());
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| {
                    oracle::matrix_at_add::<F>(expected, cell.row_len, &fixture.row_starts, &terms);
                },
                |dst| {
                    ops::mul_add_matrix_at_with::<F>(
                        dst,
                        cell.row_len,
                        &fixture.row_starts,
                        &matrix,
                        &srcs,
                    );
                },
            )
        }
        ("mul_into_matrix_at", "unprepared") => run_case(
            cell,
            ctx,
            fixture.dst.duplicate(),
            |expected| {
                oracle::matrix_at_into::<F>(expected, cell.row_len, &fixture.row_starts, &terms);
            },
            |dst| {
                ops::mul_into_matrix_at::<F>(dst, cell.row_len, &fixture.row_starts, &terms);
            },
        ),
        ("mul_into_matrix_at", "prepared") => {
            let matrix =
                ops::CoeffMatrix::<F>::from_source_major(cell.sources, cell.rows, &fixture.flat());
            run_case(
                cell,
                ctx,
                fixture.dst.duplicate(),
                |expected| {
                    oracle::matrix_at_into::<F>(
                        expected,
                        cell.row_len,
                        &fixture.row_starts,
                        &terms,
                    );
                },
                |dst| {
                    ops::mul_into_matrix_at_with::<F>(
                        dst,
                        cell.row_len,
                        &fixture.row_starts,
                        &matrix,
                        &srcs,
                    );
                },
            )
        }
        other => panic!("unregistered matrix_at operation {other:?}"),
    }
}

// --- bit-packed runners -----------------------------------------------------

fn run_bits(cell: &Cell, ctx: &crate::Ctx) -> Option<f64> {
    let seed = seed_for(&cell.id);
    let len = cell.row_len;
    let bits = len * 8;
    match cell.operation {
        "xor_assign" => {
            let src = Aligned::from_bytes(&fixture::noise(len, seed ^ 0xa1), Role::Src, 0);
            let dst = Aligned::from_bytes(
                &fixture::noise(len, seed ^ 0xa2),
                Role::Dst,
                cell.align_offset,
            );
            let src = src.as_slice();
            run_case(
                cell,
                ctx,
                dst,
                |expected| oracle::bit_xor_assign(expected, src),
                |dst| bits::xor_assign(dst, src),
            )
        }
        "and_into" => {
            let a = Aligned::from_bytes(&fixture::noise(len, seed ^ 0xa3), Role::Src, 0);
            let b = Aligned::from_bytes(&fixture::noise(len, seed ^ 0xa4), Role::Src2, 0);
            let dst = Aligned::from_bytes(
                &fixture::noise(len, seed ^ 0xa5),
                Role::Dst,
                cell.align_offset,
            );
            let a = a.as_slice();
            let b = b.as_slice();
            run_case(
                cell,
                ctx,
                dst,
                |expected| oracle::bit_and_into(expected, a, b),
                |dst| bits::and_into(dst, a, b),
            )
        }
        "and_assign" => {
            let src = Aligned::from_bytes(&fixture::noise(len, seed ^ 0xa6), Role::Src, 0);
            let dst = Aligned::from_bytes(
                &fixture::noise(len, seed ^ 0xa7),
                Role::Dst,
                cell.align_offset,
            );
            let src = src.as_slice();
            run_case(
                cell,
                ctx,
                dst,
                |expected| oracle::bit_and_assign(expected, src),
                |dst| bits::and_assign(dst, src),
            )
        }
        "andnot_assign" => {
            let mask = Aligned::from_bytes(&fixture::noise(len, seed ^ 0xa8), Role::Src, 0);
            let dst = Aligned::from_bytes(
                &fixture::noise(len, seed ^ 0xa9),
                Role::Dst,
                cell.align_offset,
            );
            let mask = mask.as_slice();
            run_case(
                cell,
                ctx,
                dst,
                |expected| oracle::bit_andnot_assign(expected, mask),
                |dst| bits::andnot_assign(dst, mask),
            )
        }
        "xor_range" | "clear_range" | "set_range" => {
            let from = bits / 4;
            let to = bits - bits / 8;
            let src = Aligned::from_bytes(&fixture::noise(len, seed ^ 0xaa), Role::Src, 0);
            let dst = Aligned::from_bytes(
                &fixture::noise(len, seed ^ 0xab),
                Role::Dst,
                cell.align_offset,
            );
            let src = src.as_slice();
            match (cell.operation, cell.prep) {
                ("xor_range", "unprepared") => run_case(
                    cell,
                    ctx,
                    dst,
                    |expected| oracle::bit_xor_range(expected, bits, from, to, src),
                    |dst| bits::xor_range(dst, bits, from, to, src),
                ),
                ("xor_range", "prepared") => {
                    let plan = bits::XorRange::new(bits, from, to);
                    run_case(
                        cell,
                        ctx,
                        dst,
                        |expected| oracle::bit_xor_range(expected, bits, from, to, src),
                        |dst| bits::xor_range_with(dst, &plan, src),
                    )
                }
                ("clear_range", _) => run_case(
                    cell,
                    ctx,
                    dst,
                    |expected| oracle::bit_clear_range(expected, bits, from, to),
                    |dst| bits::clear_range(dst, bits, from, to),
                ),
                ("set_range", _) => run_case(
                    cell,
                    ctx,
                    dst,
                    |expected| {
                        oracle::bit_set_range(expected, bits, from, to);
                    },
                    |dst| bits::set_range(dst, bits, from, to),
                ),
                other => panic!("unregistered range operation {other:?}"),
            }
        }
        "weight" => {
            let src = Aligned::from_bytes(&fixture::noise(len, seed ^ 0xac), Role::Src, 0);
            let src = src.as_slice();
            let expected = u64::try_from(oracle::bit_weight(src, bits)).expect("weight fits u64");
            if ctx.check {
                let actual = u64::try_from(bits::weight(src, bits)).expect("weight fits u64");
                check_result(cell, &actual.to_le_bytes(), &expected.to_le_bytes());
                None
            } else {
                Some(ctx.measure(|| {
                    fixture::observe(bits::weight(src, bits));
                }))
            }
        }
        "dot_product" => {
            let a = Aligned::from_bytes(&fixture::noise(len, seed ^ 0xad), Role::Src, 0);
            let b = Aligned::from_bytes(&fixture::noise(len, seed ^ 0xae), Role::Src2, 0);
            let a = a.as_slice();
            let b = b.as_slice();
            let expected = oracle::bit_dot(a, b, bits);
            if ctx.check {
                #[allow(clippy::cast_possible_truncation)]
                let actual = bits::dot_product(a, b, bits).to_index() as u8;
                check_result(cell, &[actual], &[expected]);
                None
            } else {
                Some(ctx.measure(|| {
                    fixture::observe(bits::dot_product(a, b, bits));
                }))
            }
        }
        "xor_gather" => {
            let dst = Aligned::from_bytes(
                &fixture::noise(len, seed ^ 0xaf),
                Role::Dst,
                cell.align_offset,
            );
            let region = Aligned::from_bytes(
                &fixture::noise(len * cell.sources, seed ^ 0xb0),
                Role::Region,
                0,
            );
            let region = region.as_slice();
            let srcs: Vec<&[u8]> = (0..cell.sources)
                .map(|index| &region[index * len..(index + 1) * len])
                .collect();
            let selector = (0x9e37_79b9_7f4a_7c15u64 & ((1u64 << cell.sources.min(63)) - 1)) | 1;
            run_case(
                cell,
                ctx,
                dst,
                |expected| oracle::bit_gather(expected, selector, &srcs),
                |dst| bits::xor_gather(dst, bits, selector, &srcs),
            )
        }
        other => panic!("unregistered bit operation {other:?}"),
    }
}
