# Bit-packed GF(2)

Use `bits` for one bit per GF(2) vector element. Supply the logical length
explicitly; byte count alone cannot identify the final live bit.

## Scalar values and vector storage

Polynomial GF(2) has scalar arithmetic but no byte-lane `FieldBuffer` or
`FieldKernels`. Its vectors do not use `ops::pack`:

```rust
use fgf::{FiniteField, Elem, Polynomial};

type F = FiniteField<2, 1, Polynomial<3>>;
assert_eq!(Elem::<F>::ONE + Elem::<F>::ONE, Elem::<F>::ZERO);
assert_eq!(Elem::<F>::ONE * Elem::<F>::ONE, Elem::<F>::ONE);
```

Vector addition is XOR; elementwise multiplication is AND. Scalar coefficients
are zero or one, so no coefficient preparation is needed.

## Indexing and logical length

Element `i` occupies bit `i % 8` of byte `i / 8`, least-significant bit first. A logical length of ten requires two bytes, but six bits of the final byte are padding. Extra supplied bytes are padding as well.

```rust
use fgf::bits;

let mut vector = [0u8; bits::bytes_for(10)];
bits::set_range(&mut vector, 10, 0, 1);
bits::set_range(&mut vector, 10, 9, 10);
assert_eq!(vector, [0x01, 0x02]);
assert_eq!(bits::weight(&vector, 10), 2);
```

The buffer length cannot recover the exact live bit count. Range endpoints are half-open: `[from, to)`, with `from <= to <= bits`. Operations also require enough storage for the logical length.

## Padding belongs to the caller

Padding is caller-owned. Range operations preserve bits outside the window;
whole-buffer XOR includes all supplied bytes. `weight` and `dot_product` ignore
padding without repairing it:

```rust
use fgf::bits;

let mut received = [0x01u8, 0xfe];
assert_eq!(bits::weight(&received, 10), 2);
let source = [0x01u8, 0x02];
let live = bits::XorRange::new(10, 0, 10);
bits::xor_range_with(&mut received, &source, &live);
assert_eq!(received, [0, 0xfc]);
assert_eq!(bits::weight(&received, 10), 0);
```

The high padding bits remain set. `XorRange` prepares geometry and masks,
not coefficients. Mask the final byte and clear surplus bytes if the wire
format requires zero padding.

## Dot products compute parity equations

A GF(2) dot product is the parity of the positions where both vectors contain one. It returns a typed GF(2) element; `is_one()` tests the parity.

For a seven-position Hamming code, let position one be bit zero. The three parity-check rows select positions according to the three bits of their one-based binary labels:

| Check | Selected positions | Packed row |
| --- | --- | --- |
| Low label bit | 1, 3, 5, 7 | `0b0101_0101` |
| Middle label bit | 2, 3, 6, 7 | `0b0110_0110` |
| High label bit | 4, 5, 6, 7 | `0b0111_1000` |

A codeword has zero parity against each row. Flipping one position toggles exactly the check bits of that position's label, so the resulting syndrome identifies that single error.

```rust
use fgf::bits;

fn syndrome(word: &[u8]) -> u8 {
    let checks = [[0b0101_0101u8], [0b0110_0110], [0b0111_1000]];
    checks.iter().enumerate().fold(0, |value, (bit, check)| {
        value | (u8::from(bits::dot_product(word, check, 7).is_one()) << bit)
    })
}

let codeword = [0b0110_0110u8];
assert_eq!(syndrome(&codeword), 0);
let mut received = codeword;
received[0] ^= 1 << 4;
assert_eq!(syndrome(&received), 5);
received[0] ^= 1 << (5 - 1);
assert_eq!(received, codeword);
let with_padding = [codeword[0] | 0x80];
assert_eq!(syndrome(&with_padding), 0);
```

This correction assumes at most one error. Multiple errors can produce the
same syndrome; `fgf` supplies parity arithmetic, not a decoding policy.

## Subset XOR combines supplied rows

`xor_gather` combines source rows selected by bits of a `u64` selector. Selector bit `s` means source `s` contributes. It does not denote a bit range within each source. The selected sources still have equal byte lengths, and the selector must not name a source outside the supplied collection.

```rust
use fgf::bits;

let a = [0b0000_0111u8];
let b = [0b0001_1001u8];
let c = [0b0010_1010u8];
let sources: [&[u8]; 3] = [&a, &b, &c];
let mut output = [0u8; 1];
bits::xor_gather(&mut output, 7, 0b101, &sources);
assert_eq!(output, [0b0010_1101]);
```

Run the [binary syndrome example](https://github.com/nanithefkuc/fgf/blob/v3.0.1/examples/binary_syndrome.rs)
with `just example binary_syndrome --no-default-features`. Exact range and
padding contracts belong to the
[bits reference](https://docs.rs/fgf/3.0.1/fgf/bits/index.html).

