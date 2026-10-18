//! Accumulate arriving GF(2^16) symbols into reusable parity buffers.
//!
//! Run with `just example streaming_encode`. Prepared matrix columns drive
//! scatter updates; a changed source updates parity through its delta. The
//! same coefficients also write selected output rows in a larger frame.

#![forbid(unsafe_code)]

use fgf::{Elem, backend_for, ops};

// Callers name their own fields; the canonical `FiniteField<2, N, R>` type is the API.
type Gf16 = fgf::FiniteField<2, 16, fgf::Extension<fgf::Rijndael16, 2>>;

const SYMBOLS: usize = 8;
const ROW_LEN: usize = SYMBOLS * 2;

fn main() {
    let first = std::array::from_fn::<_, SYMBOLS, _>(|i| Elem::<Gf16>::from_raw(0x1200 + i as u16));
    let second =
        std::array::from_fn::<_, SYMBOLS, _>(|i| Elem::<Gf16>::from_raw(0x3400 + i as u16));
    let third = std::array::from_fn::<_, SYMBOLS, _>(|i| Elem::<Gf16>::from_raw(0x5600 + i as u16));
    let mut payloads = [[0u8; ROW_LEN]; 3];
    for (dst, elements) in payloads.iter_mut().zip([first, second, third]) {
        ops::pack::<Gf16>(dst, &elements);
    }

    // Each source contributes to two parity rows. The byte buffers contain
    // little-endian tower elements, not independent GF(256) bytes.
    let coefficients = [
        Elem::<Gf16>::ONE,
        Elem::<Gf16>::from_raw(0x0108),
        Elem::<Gf16>::ONE,
        Elem::<Gf16>::from_raw(0x0203),
        Elem::<Gf16>::ONE,
        Elem::<Gf16>::from_raw(0x0405),
    ];
    let matrix = ops::CoeffMatrix::<Gf16>::from_source_major(3, 2, &coefficients);
    let mut parity = [0u8; 2 * ROW_LEN];
    for (source, payload) in payloads.iter().enumerate() {
        ops::mul_add_scatter_with(
            &mut parity,
            ROW_LEN,
            matrix.source(source).unwrap(),
            payload,
        );
    }

    // An incremental edit needs only delta = new - old, not a full re-encode.
    let mut replacement = second;
    replacement[3] += Elem::<Gf16>::from_raw(0xabcd);
    let mut replacement_bytes = [0u8; ROW_LEN];
    ops::pack::<Gf16>(&mut replacement_bytes, &replacement);
    let mut delta = replacement_bytes;
    ops::sub_assign::<Gf16>(&mut delta, &payloads[1]);
    ops::mul_add_scatter_with(&mut parity, ROW_LEN, matrix.source(1).unwrap(), &delta);
    payloads[1] = replacement_bytes;

    let sources: [&[u8]; 3] = payloads.each_ref().map(|row| row.as_slice());
    let mut batch = [0xa5; 2 * ROW_LEN];
    ops::mul_into_matrix_with(&mut batch, ROW_LEN, &matrix, &sources);
    assert_eq!(parity, batch);

    // '_at' addresses destination rows. Headers and the unused row remain
    // untouched. The prepared form reuses the coding matrix directly; like
    // the raw form it accumulates, so the output slots still start at zero.
    const HEADER: usize = 4;
    let offsets = [HEADER, HEADER + 2 * ROW_LEN];
    let mut frame = [0x5a; HEADER + 3 * ROW_LEN];
    for start in offsets {
        frame[start..start + ROW_LEN].fill(0);
    }
    ops::mul_add_matrix_at_with(&mut frame, ROW_LEN, &offsets, &matrix, &sources);
    assert_eq!(&frame[..HEADER], &[0x5a; HEADER]);
    assert_eq!(
        &frame[HEADER + ROW_LEN..HEADER + 2 * ROW_LEN],
        &[0x5a; ROW_LEN]
    );
    for (output, start) in offsets.into_iter().enumerate() {
        assert_eq!(
            &frame[start..start + ROW_LEN],
            &parity[output * ROW_LEN..(output + 1) * ROW_LEN]
        );
    }

    // Overwrite writes the same rows without reading them first.
    let mut direct = [0x5a; HEADER + 3 * ROW_LEN];
    ops::mul_into_matrix_at_with(&mut direct, ROW_LEN, &offsets, &matrix, &sources);
    assert_eq!(direct, frame);

    let mut elements = [Elem::ZERO; SYMBOLS];
    ops::unpack::<Gf16>(&mut elements, &parity[ROW_LEN..]);
    println!("Gf16 backend: {:?}", backend_for::<Gf16>());
    println!("Updated weighted parity: {elements:?}");
    println!("Streaming, batch, and offset-addressed outputs agree; frame metadata is intact.");
}
