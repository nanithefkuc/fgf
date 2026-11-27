# Binary tower regressions

[#149](https://github.com/nanithefkuc/fgf/issues/149) and [#150](https://github.com/nanithefkuc/fgf/issues/150).

| Field | Operation | Process selection | 2.1.0 (ns/element) | 4.0.1 (ns/element) | `1ce2125` (ns/element) | Fix (ns/element) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Fan–Paar8 | Chained `mul` | Scalar | 1.795 | 906.250 | 905.952 | 1.791 |
| Fan–Paar8 | Chained `inv` | Scalar | 174.270 | 1783.668 | 1785.228 | 3.976 |
| Fan–Paar8 | `ops::mul_add` | Scalar | 1.168 | 921.362 | 922.798 | 0.587 |
| Fan–Paar8 | `ops::mul_add` | Default | 1.166 | 921.399 | 921.452 | 0.030 |
| Fan–Paar16 | Chained `mul` | Scalar | 18.929 | 265.108 | 265.068 | 18.793 |
| Fan–Paar16 | Chained `inv` | Scalar | 238.048 | 203.734 | 203.697 | 60.820 |
| Fan–Paar16 | `ops::mul_add` | Scalar | 17.971 | 270.858 | 270.685 | 18.033 |
| Fan–Paar16 | `ops::mul_add` | Default | 0.167 | 0.172 | 0.172 | 0.168 |
| Fan–Paar32 | Chained `mul` | Scalar | 65.572 | 812.943 | 811.692 | 67.076 |
| Fan–Paar32 | Chained `inv` | Scalar | 407.222 | 1114.206 | 1114.306 | 225.644 |
| Fan–Paar32 | `ops::mul_add` | Scalar | 65.905 | 888.421 | 894.042 | 67.347 |
| Fan–Paar32 | `ops::mul_add` | Default | 0.835 | 0.841 | 0.838 | 0.837 |
| Fan–Paar64 | Chained `mul` | Scalar | 215.526 | 2452.807 | 2453.322 | 217.565 |
| Fan–Paar64 | Chained `inv` | Scalar | 848.331 | 3782.730 | 3778.381 | 645.088 |
| Fan–Paar64 | `ops::mul_add` | Scalar | 213.848 | 2923.244 | 2939.007 | 218.527 |
| Fan–Paar64 | `ops::mul_add` | Default | 4.724 | 4.715 | 4.715 | 4.726 |
| Rijndael32 | `ops::mul_add` | Scalar | 20.328 | 67.433 | 67.482 | 20.416 |
| Rijndael64 | Chained `mul` | Scalar | 76.025 | 83.321 | 83.428 | 83.507 |
| Rijndael64 | `ops::mul_add` | Scalar | 77.275 | 262.706 | 263.551 | 78.662 |
| AES byte field (control) | `ops::mul_add` | Scalar | 0.590 | 0.598 | 0.590 | 0.592 |
| AES byte field (control) | `ops::mul_add` | Default | 0.012 | 0.012 | 0.012 | 0.011 |

- Intel Core i5-1135G7 (Willow Cove), Linux 7.2.7, Rust 1.98.1. Release builds
  with all features; each process pinned to core 1 with `taskset`.
- Median of five interleaved rounds from one session. Each round includes both
  published releases, the pre-fix revision, and this change. `Scalar` sets
  `SIMD_BACKEND=scalar`; `Default` leaves the override unset.
- The default process selected `v4x`. The Fan–Paar8 row ran on scalar in the
  baselines and AVX2 in the fix; the wider Fan–Paar rows used AVX2 in all builds.
- Deterministic full-width inputs, 4096 elements per row. Scalar cases chain
  eight passes over decoded elements and include an addition in each dependency
  step. Row cases warm the selected kernel and then time 256 one-shot calls with
  a fixed nontrivial coefficient, including coefficient preparation. Allocation
  and decoding of the input fixtures are outside the timed regions.
