# fgf — Faster Galois Fields

**This is a name reservation. There is no API here yet.**

`fgf` will be the published form of [`fff`](https://github.com/nanithefkuc/fff),
a dependency-free Rust library for binary finite field arithmetic:
const-capable scalar elements for GF(2^8) through GF(2^64) and the canonical
Fan–Paar towers, plus safe, runtime-dispatched SIMD kernels over packed byte
buffers. It is the arithmetic layer underneath erasure coders, proof systems,
and other consumers that operate on packed field elements; it is deliberately
not a codec.

The crate is developed in the open and distributed through git today. The
rename to `fgf` and the first real release here happen together at 1.0.0,
gated on:

1. AVX-512 kernels validated on hardware that can execute them.
2. A stable, extracted backend detection layer.
3. Closing the remaining vectorization work rather than shipping around it.

Until then:

```toml
[dependencies]
fff = { git = "https://github.com/nanithefkuc/fff" }
```

## License

MIT. See [LICENSE](LICENSE).
