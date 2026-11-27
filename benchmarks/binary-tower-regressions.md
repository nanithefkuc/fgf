# Binary tower regressions

[#149](https://github.com/nanithefkuc/fgf/issues/149) and [#150](https://github.com/nanithefkuc/fgf/issues/150).

| Run | Field | Operation | Process selection | 2.1.0 (ns/element) | 4.0.1 (ns/element) | `1ce2125` (ns/element) | `c1bd014` (ns/element) | Fixed-width towers (ns/element) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | Fan–Paar8 | Chained `mul` | Scalar | 1.795 | 906.250 | 905.952 | 1.791 | - |
| A | Fan–Paar8 | Chained `inv` | Scalar | 174.270 | 1783.668 | 1785.228 | 3.976 | - |
| A | Fan–Paar8 | `ops::mul_add` | Scalar | 1.168 | 921.362 | 922.798 | 0.587 | - |
| A | Fan–Paar8 | `ops::mul_add` | Default | 1.166 | 921.399 | 921.452 | 0.030 | - |
| A | Fan–Paar16 | Chained `mul` | Scalar | 18.929 | 265.108 | 265.068 | 18.793 | - |
| A | Fan–Paar16 | Chained `inv` | Scalar | 238.048 | 203.734 | 203.697 | 60.820 | - |
| A | Fan–Paar16 | `ops::mul_add` | Scalar | 17.971 | 270.858 | 270.685 | 18.033 | - |
| A | Fan–Paar16 | `ops::mul_add` | Default | 0.167 | 0.172 | 0.172 | 0.168 | - |
| A | Fan–Paar32 | Chained `mul` | Scalar | 65.572 | 812.943 | 811.692 | 67.076 | - |
| A | Fan–Paar32 | Chained `inv` | Scalar | 407.222 | 1114.206 | 1114.306 | 225.644 | - |
| A | Fan–Paar32 | `ops::mul_add` | Scalar | 65.905 | 888.421 | 894.042 | 67.347 | - |
| A | Fan–Paar32 | `ops::mul_add` | Default | 0.835 | 0.841 | 0.838 | 0.837 | - |
| A | Fan–Paar64 | Chained `mul` | Scalar | 215.526 | 2452.807 | 2453.322 | 217.565 | - |
| A | Fan–Paar64 | Chained `inv` | Scalar | 848.331 | 3782.730 | 3778.381 | 645.088 | - |
| A | Fan–Paar64 | `ops::mul_add` | Scalar | 213.848 | 2923.244 | 2939.007 | 218.527 | - |
| A | Fan–Paar64 | `ops::mul_add` | Default | 4.724 | 4.715 | 4.715 | 4.726 | - |
| A | Rijndael32 | `ops::mul_add` | Scalar | 20.328 | 67.433 | 67.482 | 20.416 | - |
| A | Rijndael64 | Chained `mul` | Scalar | 76.025 | 83.321 | 83.428 | 83.507 | - |
| A | Rijndael64 | `ops::mul_add` | Scalar | 77.275 | 262.706 | 263.551 | 78.662 | - |
| A | AES byte field (control) | `ops::mul_add` | Scalar | 0.590 | 0.598 | 0.590 | 0.592 | - |
| A | AES byte field (control) | `ops::mul_add` | Default | 0.012 | 0.012 | 0.012 | 0.011 | - |
| B | Fan–Paar8 | Elimination (64 × 64) | Coverage, scalar | 3533.5014 | - | - | 399.2413 | 354.1029 |
| B | Fan–Paar16 | Elimination (64 × 64) | Coverage, scalar | 6774.3501 | - | - | 9386.1898 | 1176.8939 |
| B | Fan–Paar32 | Elimination (64 × 64) | Coverage, scalar | 14130.0962 | - | - | 17027.6443 | 6490.3833 |
| B | Fan–Paar64 | Elimination (64 × 64) | Coverage, scalar | 38319.8474 | - | - | 40693.6187 | 11527.0974 |
| B | AES byte field (control) | Elimination (64 × 64) | Coverage, scalar | 333.5601 | - | - | 361.4575 | 365.7767 |
| B | Fan–Paar8 | Chained `mul` | Coverage, scalar | 55.2234 | - | - | 143.0175 | 59.9781 |
| B | Fan–Paar16 | Chained `mul` | Coverage, scalar | 176.6852 | - | - | 261.0508 | 88.0434 |
| B | Fan–Paar32 | Chained `mul` | Coverage, scalar | 541.3078 | - | - | 627.8298 | 173.8440 |
| B | Fan–Paar64 | Chained `mul` | Coverage, scalar | 1675.2318 | - | - | 1740.5993 | 434.6664 |
| B | Fan–Paar8 | Chained `inv` | Coverage, scalar | 1288.3158 | - | - | 129.4538 | 77.0532 |
| B | Fan–Paar16 | Chained `inv` | Coverage, scalar | 1770.9417 | - | - | 594.1123 | 110.6861 |
| B | Fan–Paar32 | Chained `inv` | Coverage, scalar | 3007.9710 | - | - | 1824.7405 | 252.3823 |
| B | Fan–Paar64 | Chained `inv` | Coverage, scalar | 6215.0854 | - | - | 4975.0378 | 700.9149 |
| B | Fan–Paar8 | `ops::mul_add` (64 lanes) | Coverage, scalar | 165.7560 | - | - | 14.2039 | 13.7421 |
| B | Fan–Paar16 | `ops::mul_add` (64 lanes) | Coverage, scalar | 307.8372 | - | - | 427.3811 | 50.2687 |
| B | Fan–Paar32 | `ops::mul_add` (64 lanes) | Coverage, scalar | 648.0230 | - | - | 781.0578 | 295.8833 |
| B | Fan–Paar64 | `ops::mul_add` (64 lanes) | Coverage, scalar | 1734.2247 | - | - | 1855.3777 | 527.4349 |
| B | Fan–Paar8 | Elimination (64 × 64) | Release, scalar | 31.5122 | - | - | 17.2283 | 16.8778 |
| B | Fan–Paar16 | Elimination (64 × 64) | Release, scalar | 417.3160 | - | - | 426.6417 | 48.1373 |
| B | Fan–Paar32 | Elimination (64 × 64) | Release, scalar | 1481.0182 | - | - | 1525.9464 | 217.4419 |
| B | Fan–Paar64 | Elimination (64 × 64) | Release, scalar | 4768.5801 | - | - | 4843.1730 | 656.0214 |
| B | AES byte field (control) | Elimination (64 × 64) | Release, scalar | 17.2921 | - | - | 17.5494 | 17.5310 |
| B | Fan–Paar8 | Elimination (64 × 64) | Release, default | 32.0729 | - | - | 7.1470 | 5.9189 |
| B | Fan–Paar16 | Elimination (64 × 64) | Release, default | 60.1864 | - | - | 58.8816 | 17.4719 |
| B | Fan–Paar32 | Elimination (64 × 64) | Release, default | 197.6597 | - | - | 198.9927 | 43.8953 |
| B | Fan–Paar64 | Elimination (64 × 64) | Release, default | 441.0396 | - | - | 451.1067 | 143.7490 |
| B | AES byte field (control) | Elimination (64 × 64) | Release, default | 9.0484 | - | - | 7.8560 | 7.8157 |
| B | Fan–Paar8 | `ops::mul_add` (8 lanes) | Release, default | 1.6966 | - | - | 1.8759 | 1.2984 |
| B | Fan–Paar16 | `ops::mul_add` (8 lanes) | Release, default | 2.7802 | - | - | 2.7102 | 2.1186 |
| B | Fan–Paar32 | `ops::mul_add` (8 lanes) | Release, default | 6.0821 | - | - | 6.6274 | 2.6197 |
| B | Fan–Paar64 | `ops::mul_add` (8 lanes) | Release, default | 20.9253 | - | - | 22.5286 | 8.3357 |
| B | Fan–Paar8 | `ops::mul_add` (4096 lanes) | Release, default | 1.1969 | - | - | 0.0326 | 0.0324 |
| B | Fan–Paar16 | `ops::mul_add` (4096 lanes) | Release, default | 0.1562 | - | - | 0.1646 | 0.1588 |
| B | Fan–Paar32 | `ops::mul_add` (4096 lanes) | Release, default | 0.8790 | - | - | 0.8700 | 0.8453 |
| B | Fan–Paar64 | `ops::mul_add` (4096 lanes) | Release, default | 4.4784 | - | - | 4.4440 | 4.4407 |
| B | AES byte field (control) | `ops::mul_add` (4096 lanes) | Release, default | 0.0122 | - | - | 0.0127 | 0.0128 |

- Both runs use an Intel Core i5-1135G7 (Willow Cove), Linux 7.2.7, and Rust
  1.98.1, with all features and each process pinned to core 1 with `taskset`.
  Values are medians of five interleaved rounds within each run. Compare only
  builds measured in the same run. `Scalar` sets `SIMD_BACKEND=scalar`;
  `Default` leaves the override unset.
- Run A uses release builds, deterministic full-width inputs, and rows of 4096
  elements with a fixed nontrivial coefficient. Scalar cases chain eight
  passes over decoded elements and include an addition in each dependency step.
  Row cases warm the kernel then time 256 one-shot calls with coefficient
  preparation included. Allocation and input decoding are untimed.
- Run B uses optimized release and a separate scalar coverage build with
  optimization disabled, debug assertions and overflow checks enabled, and
  coverage instrumentation in both the library and driver. It times chained
  element operations with an addition in each step, one-shot row calls including
  coefficient preparation, and a fixed 64 × 64 forward-elimination workload.
  Elimination includes resetting the matrix from a fixture; its timings are
  normalized by matrix elements. It exercises the public field and row
  operations used in matrix elimination, not the `gfm` suite or its CI setup.
- Default processes select `v4x`. For byte Fan–Paar rows, 2.1.0 and the other
  pre-fix builds in run A use scalar, while `c1bd014` uses AVX2 plus its tail.
  The fixed-width build uses the scalar nibble loop directly below an SSSE3
  lane and AVX2 plus its tail for longer rows. Wider Fan–Paar rows use AVX2
  plus its scalar tail in all default builds.
