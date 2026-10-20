# GF16 presentations

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../BENCHMARKS.md) · [Field definitions](../labels.md)

Each library uses its native encoding. Field maps and fixtures are validated outside timing. **Fixed** reuses coefficients; **Cycling** rotates sixteen prepared sets. Conversion caller loops live on the [maps page](maps.md).

## Find a presentation

- [gf16-cantor-base](#gf16-cantor-base)
- [gf16-fanpaar16](#gf16-fanpaar16)
- [gf16-flat-0x1002d](#gf16-flat-0x1002d)
- [gf16-flat-0x1100b](#gf16-flat-0x1100b)
- [gf16-normal-base](#gf16-normal-base)
- [gf16-rijndael16](#gf16-rijndael16)
- [gf16-rs-quad](#gf16-rs-quad)

## gf16-cantor-base

### Native presentations

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 0.278 / 0.538 | 7.883 / 12.99 | 20.97 / 35.70 | 1.058 / 2.307 |
| mul_add | Fixed | 0.278 / 0.538 | 7.884 / 13.01 | 20.96 / 36.89 | 1.051 / 2.308 |
| mul_into | Cycling | 0.279 / 0.567 | 8.606 / 14.13 | 24.08 / 41.11 | 1.166 / 2.643 |
| mul_into | Fixed | 0.279 / 0.567 | 8.609 / 14.11 | 24.08 / 41.11 | 1.165 / 2.645 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-cantor-base | `fgf::ops::mul_add` |
| gf16-cantor-base | `fgf::ops::mul_into` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather_with | 0.277 / 0.540 | 5.838 / 10.14 | 19.90 / 33.30 | 1.036 / 2.278 |
| mul_into_gather_with | 0.278 / 0.543 | 5.856 / 10.02 | 20.01 / 33.56 | 1.041 / 2.378 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-cantor-base | `fgf::ops::mul_add_gather_with` |
| gf16-cantor-base | `fgf::ops::mul_into_gather_with` |

</details>

**Counted bytes:** 160 KiB · **Elements:** 81,920 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_matrix_with | 0.279 / 0.545 | 5.844 / 9.989 | 21.28 / 34.23 | 1.055 / 2.392 |
| mul_into_matrix_with | 0.279 / 0.546 | 5.888 / 10.02 | 21.48 / 34.60 | 1.065 / 2.414 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-cantor-base | `fgf::ops::mul_add_matrix_with` |
| gf16-cantor-base | `fgf::ops::mul_into_matrix_with` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | `reed-solomon-simd` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_assign | Cycling | 0.281 / 0.569 | 8.616 / 14.13 | 24.08 / 41.30 |
| mul_assign | Fixed | 0.281 / 0.572 | 8.622 / 14.13 | 24.09 / 41.31 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign`

</details>

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| scalar_mul | 16.5 / 9.57 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::mul`

</details>

### Preparation and setup

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_coeff | 3.01 / 1.28 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::Coeff::new`

</details>

**Elements:** 16 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_gather | 50.1 / 27.6 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::new`

</details>

**Elements:** 40 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_matrix | 59.7 / 33.8 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::from_source_major`

</details>

<details>
<summary>Duplicate-fgf control ratios</summary>

Ratios divide duplicate-fgf latency by primary-fgf latency; medians and ranges span five runs.



| Operation | Coefficients | Counted bytes | Elements | Sources | Destinations | Median ratio | Run range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0003 / 1.0001 | 0.9999–1.0010 / 0.9997–1.0014 |
| mul_add | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0002 / 1.0002 | 1.0000–1.0012 / 0.9997–1.0007 |
| mul_add_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0001 / 1.0004 | 0.9998–1.0007 / 1.0000–1.0006 |
| mul_add_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0001 / 0.9999 | 1.0000–1.0002 / 0.9997–1.0002 |
| mul_assign | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 0.9999 | 0.9998–1.0000 / 0.9997–0.9999 |
| mul_assign | Fixed | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 0.9999 | 0.9998–0.9999 / 0.9998–0.9999 |
| mul_into | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9999–1.0001 / 0.9996–1.0002 |
| mul_into | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9999–1.0000 / 0.9996–1.0003 |
| mul_into_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0002 / 1.0001 | 0.9995–1.0006 / 0.9997–1.0003 |
| mul_into_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 0.9999 / 0.9999 | 0.9999–1.0002 / 0.9998–1.0001 |

</details>


## gf16-fanpaar16

### Native presentations

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** v4x / v3_gfni_crypto

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 10.8 / 16.6 | 7.88 / 13.1 | 21.0 / 36.5 | 1.06 / 2.41 |
| mul_add | Fixed | 10.8 / 16.6 | 7.89 / 13.1 | 21.0 / 36.4 | 1.06 / 2.41 |
| mul_into | Cycling | 11.4 / 17.3 | 8.61 / 14.1 | 24.1 / 41.1 | 1.17 / 2.53 |
| mul_into | Fixed | 11.4 / 17.3 | 8.61 / 14.1 | 24.1 / 41.1 | 1.17 / 2.63 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-fanpaar16 | `fgf::ops::mul_add` |
| gf16-fanpaar16 | `fgf::ops::mul_into` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather_with | 10.3 / 16.3 | 5.83 / 9.97 | 19.9 / 33.9 | 1.04 / 2.35 |
| mul_into_gather_with | 10.3 / 16.3 | 5.85 / 9.89 | 20.0 / 33.7 | 1.05 / 2.30 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-fanpaar16 | `fgf::ops::mul_add_gather_with` |
| gf16-fanpaar16 | `fgf::ops::mul_into_gather_with` |

</details>

**Counted bytes:** 160 KiB · **Elements:** 81,920 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_matrix_with | 10.4 / 16.3 | 5.84 / 9.98 | 21.3 / 34.3 | 1.06 / 2.31 |
| mul_into_matrix_with | 10.3 / 16.2 | 5.88 / 9.91 | 21.5 / 35.1 | 1.07 / 2.34 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-fanpaar16 | `fgf::ops::mul_add_matrix_with` |
| gf16-fanpaar16 | `fgf::ops::mul_into_matrix_with` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** v4x / v3_gfni_crypto

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | `reed-solomon-simd` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_assign | Cycling | 11.4 / 17.3 | 8.62 / 14.1 | 24.1 / 41.3 |
| mul_assign | Fixed | 11.4 / 17.3 | 8.62 / 14.1 | 24.1 / 41.4 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign`

</details>

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| scalar_mul | 319 / 205 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::mul`

</details>

### Preparation and setup

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_coeff | 18.7 / 10.6 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::Coeff::new`

</details>

**Elements:** 16 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_gather | 314 / 167 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::new`

</details>

**Elements:** 40 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_matrix | 733 / 384 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::from_source_major`

</details>

<details>
<summary>Duplicate-fgf control ratios</summary>

Ratios divide duplicate-fgf latency by primary-fgf latency; medians and ranges span five runs.



| Operation | Coefficients | Counted bytes | Elements | Sources | Destinations | Median ratio | Run range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9996–1.0022 / 0.9996–1.0000 |
| mul_add | Fixed | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 1.0000 | 0.9995–1.0000 / 0.9995–1.0001 |
| mul_add_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0001 / 1.0001 | 0.9997–1.0004 / 1.0000–1.0182 |
| mul_add_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 0.9999 / 1.0001 | 0.9997–1.0002 / 0.9999–1.0006 |
| mul_assign | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9994–1.0000 / 0.9996–1.0000 |
| mul_assign | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9995–1.0000 / 0.9995–1.0000 |
| mul_into | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9998–1.0000 / 0.9996–1.0001 |
| mul_into | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 1.0000–1.0001 / 0.9996–1.0001 |
| mul_into_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0004 / 0.9997 | 0.9962–1.0006 / 0.9997–1.0002 |
| mul_into_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0001 / 1.0001 | 0.9996–1.0002 / 1.0000–1.0146 |

</details>


## gf16-flat-0x1002d

### Native presentations

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 0.000639 / 0.001234 | 7.884660 / 12.95992 | 20.96708 / 34.28309 | 1.046033 / 2.400047 |
| mul_add | Fixed | 0.000653 / 0.001262 | 7.885361 / 12.93837 | 20.96663 / 35.26483 | 1.059758 / 2.379028 |
| mul_into | Cycling | 0.000639 / 0.001235 | 8.607336 / 14.10891 | 24.07698 / 39.63845 | 1.166280 / 2.647996 |
| mul_into | Fixed | 0.000653 / 0.001264 | 8.611890 / 14.12401 | 24.08990 / 41.10566 | 1.165972 / 2.646453 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-flat-0x1002d | `fgf::ops::mul_add` |
| gf16-flat-0x1002d | `fgf::ops::mul_into` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather_with | 0.000642 / 0.001239 | 5.835903 / 9.980556 | 19.78800 / 33.40588 | 1.048192 / 2.354184 |
| mul_into_gather_with | 0.000642 / 0.001240 | 5.854340 / 10.00510 | 19.96611 / 33.66122 | 1.054457 / 2.366369 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-flat-0x1002d | `fgf::ops::mul_add_gather_with` |
| gf16-flat-0x1002d | `fgf::ops::mul_into_gather_with` |

</details>

**Counted bytes:** 160 KiB · **Elements:** 81,920 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_matrix_with | 0.000643 / 0.001242 | 5.851380 / 9.976243 | 21.21228 / 34.49264 | 1.051395 / 2.365869 |
| mul_into_matrix_with | 0.000644 / 0.001242 | 5.882340 / 10.01825 | 21.41847 / 35.24936 | 1.062190 / 2.391903 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-flat-0x1002d | `fgf::ops::mul_add_matrix_with` |
| gf16-flat-0x1002d | `fgf::ops::mul_into_matrix_with` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | `reed-solomon-simd` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_assign | Cycling | 0.000639 / 0.001234 | 8.616829 / 14.11472 | 24.08812 / 41.30320 |
| mul_assign | Fixed | 0.000653 / 0.001264 | 8.622383 / 14.11758 | 24.09020 / 41.32330 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign`

</details>

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| scalar_mul | 3250 / 1739 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::mul`

</details>

### Preparation and setup

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_coeff | 3.01 / 1.28 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::Coeff::new`

</details>

**Elements:** 16 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_gather | 58.4 / 28.0 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::new`

</details>

**Elements:** 40 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_matrix | 66.0 / 33.6 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::from_source_major`

</details>

<details>
<summary>Duplicate-fgf control ratios</summary>

Ratios divide duplicate-fgf latency by primary-fgf latency; medians and ranges span five runs.



| Operation | Coefficients | Counted bytes | Elements | Sources | Destinations | Median ratio | Run range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 1.0000 | 0.9996–1.0001 / 0.9996–1.0000 |
| mul_add | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0001 / 1.0000 | 0.9996–1.0007 / 1.0000–1.0000 |
| mul_add_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0004 / 1.0000 | 1.0000–1.0004 / 1.0000–1.0002 |
| mul_add_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0000 / 1.0000 | 1.0000–1.0006 / 0.9999–1.0000 |
| mul_assign | Cycling | 64 KiB | 32,768 | 1 | 1 | 0.9996 / 1.0000 | 0.9995–0.9998 / 1.0000–1.0003 |
| mul_assign | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9998–1.0001 / 0.9998–1.0001 |
| mul_into | Cycling | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 1.0000 | 0.9998–1.0002 / 0.9996–1.0002 |
| mul_into | Fixed | 64 KiB | 32,768 | 1 | 1 | 0.9997 / 1.0000 | 0.9993–1.0007 / 0.9999–1.0000 |
| mul_into_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0000 / 1.0000 | 0.9994–1.0007 / 0.9996–1.0000 |
| mul_into_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0002 / 1.0000 | 1.0000–1.0007 / 0.9999–1.0003 |

</details>


## gf16-flat-0x1100b

### Native presentations

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 0.000651 / 0.001258 | 7.883197 / 12.94420 | 20.96719 / 35.61465 | 1.059651 / 2.395902 |
| mul_add | Fixed | 0.000626 / 0.001208 | 7.885743 / 12.98974 | 20.97406 / 35.07169 | 1.063265 / 2.312003 |
| mul_into | Cycling | 0.000650 / 0.001258 | 8.606881 / 14.10789 | 24.08812 / 39.52094 | 1.166504 / 2.533486 |
| mul_into | Fixed | 0.000626 / 0.001209 | 8.611131 / 14.12238 | 24.09154 / 41.05166 | 1.166362 / 2.533801 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-flat-0x1100b | `fgf::ops::mul_add` |
| gf16-flat-0x1100b | `fgf::ops::mul_into` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather_with | 0.000639 / 0.001233 | 5.830607 / 9.968077 | 19.81882 / 33.12094 | 1.041422 / 2.352959 |
| mul_into_gather_with | 0.000639 / 0.001233 | 5.852200 / 9.900218 | 19.97551 / 33.54905 | 1.048732 / 2.367378 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-flat-0x1100b | `fgf::ops::mul_add_gather_with` |
| gf16-flat-0x1100b | `fgf::ops::mul_into_gather_with` |

</details>

**Counted bytes:** 160 KiB · **Elements:** 81,920 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_matrix_with | 0.000637 / 0.001230 | 5.842698 / 9.969562 | 21.21007 / 34.24156 | 1.058190 / 2.388534 |
| mul_into_matrix_with | 0.000637 / 0.001230 | 5.880696 / 9.903080 | 21.44274 / 35.08020 | 1.067481 / 2.403016 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-flat-0x1100b | `fgf::ops::mul_add_matrix_with` |
| gf16-flat-0x1100b | `fgf::ops::mul_into_matrix_with` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | `reed-solomon-simd` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_assign | Cycling | 0.000650 / 0.001257 | 8.615917 / 14.11299 | 24.08381 / 41.30145 |
| mul_assign | Fixed | 0.000626 / 0.001207 | 8.620709 / 14.13055 | 24.08812 / 41.32395 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign`

</details>

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| scalar_mul | 3248 / 1742 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::mul`

</details>

### Preparation and setup

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_coeff | 3.01 / 1.28 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::Coeff::new`

</details>

**Elements:** 16 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_gather | 37.4 / 28.1 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::new`

</details>

**Elements:** 40 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_matrix | 65.6 / 34.2 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::from_source_major`

</details>

<details>
<summary>Duplicate-fgf control ratios</summary>

Ratios divide duplicate-fgf latency by primary-fgf latency; medians and ranges span five runs.



| Operation | Coefficients | Counted bytes | Elements | Sources | Destinations | Median ratio | Run range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0001 / 1.0000 | 0.9999–1.0006 / 0.9996–1.0002 |
| mul_add | Fixed | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 1.0000 | 0.9998–1.0000 / 0.9999–1.0000 |
| mul_add_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0000 / 1.0000 | 0.9998–1.0005 / 0.9991–1.0000 |
| mul_add_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0002 / 1.0000 | 1.0000–1.0003 / 0.9999–1.0004 |
| mul_assign | Cycling | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 1.0000 | 0.9998–1.0003 / 0.9998–1.0005 |
| mul_assign | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0001 / 1.0001 | 1.0000–1.0002 / 1.0001–1.0002 |
| mul_into | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9999–1.0006 / 1.0000–1.0005 |
| mul_into | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9998–1.0003 / 0.9999–1.0009 |
| mul_into_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0001 / 1.0000 | 0.9995–1.0004 / 1.0000–1.0006 |
| mul_into_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 0.9999 / 1.0000 | 0.9996–1.0001 / 0.9999–1.0001 |

</details>


## gf16-normal-base

### Native presentations

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 0.278 / 0.544 | 7.881 / 12.96 | 20.96 / 35.52 | 1.065 / 2.308 |
| mul_add | Fixed | 0.278 / 0.543 | 7.883 / 13.06 | 20.97 / 36.49 | 1.062 / 2.308 |
| mul_into | Cycling | 0.280 / 0.561 | 8.605 / 14.12 | 24.08 / 41.09 | 1.166 / 2.530 |
| mul_into | Fixed | 0.280 / 0.561 | 8.609 / 14.13 | 24.07 / 41.12 | 1.166 / 2.530 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-normal-base | `fgf::ops::mul_add` |
| gf16-normal-base | `fgf::ops::mul_into` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather_with | 0.278 / 0.540 | 5.838 / 10.02 | 19.90 / 33.39 | 1.045 / 2.272 |
| mul_into_gather_with | 0.278 / 0.547 | 5.855 / 10.03 | 20.02 / 33.65 | 1.051 / 2.287 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-normal-base | `fgf::ops::mul_add_gather_with` |
| gf16-normal-base | `fgf::ops::mul_into_gather_with` |

</details>

**Counted bytes:** 160 KiB · **Elements:** 81,920 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_matrix_with | 0.279 / 0.552 | 5.855 / 9.847 | 21.29 / 34.36 | 1.059 / 2.300 |
| mul_into_matrix_with | 0.279 / 0.554 | 5.886 / 10.03 | 21.50 / 35.39 | 1.067 / 2.322 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-normal-base | `fgf::ops::mul_add_matrix_with` |
| gf16-normal-base | `fgf::ops::mul_into_matrix_with` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | `reed-solomon-simd` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_assign | Cycling | 0.283 / 0.563 | 8.616 / 14.13 | 24.09 / 41.31 |
| mul_assign | Fixed | 0.282 / 0.563 | 8.621 / 14.14 | 24.09 / 41.33 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign`

</details>

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| scalar_mul | 16.5 / 9.72 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::mul`

</details>

### Preparation and setup

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_coeff | 3.01 / 1.28 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::Coeff::new`

</details>

**Elements:** 16 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_gather | 49.8 / 27.7 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::new`

</details>

**Elements:** 40 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_matrix | 66.9 / 33.7 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::from_source_major`

</details>

<details>
<summary>Duplicate-fgf control ratios</summary>

Ratios divide duplicate-fgf latency by primary-fgf latency; medians and ranges span five runs.



| Operation | Coefficients | Counted bytes | Elements | Sources | Destinations | Median ratio | Run range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0003 / 1.0003 | 1.0000–1.0011 / 0.9998–1.0007 |
| mul_add | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0003 / 1.0003 | 1.0000–1.0010 / 0.9994–1.0021 |
| mul_add_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0002 / 1.0001 | 0.9996–1.0007 / 0.9994–1.0005 |
| mul_add_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0001 / 0.9999 | 0.9997–1.0004 / 0.9985–1.0003 |
| mul_assign | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 0.9999 | 0.9998–1.0001 / 0.9997–0.9999 |
| mul_assign | Fixed | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 0.9999 | 0.9997–1.0000 / 0.9996–1.0000 |
| mul_into | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0001 / 1.0000 | 0.9998–1.0001 / 0.9996–1.0001 |
| mul_into | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 0.9999 | 0.9999–1.0000 / 0.9996–1.0001 |
| mul_into_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0001 / 1.0000 | 0.9998–1.0008 / 0.9997–1.0004 |
| mul_into_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0001 / 1.0001 | 0.9997–1.0005 / 0.9989–1.0003 |

</details>


## gf16-rijndael16

### Native presentations

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** v4x / v3_gfni_crypto

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 40.6 / 53.4 | 7.88 / 13.0 | 20.9 / 35.8 | 1.05 / 2.31 |
| mul_add | Fixed | 40.7 / 53.5 | 7.88 / 13.0 | 20.9 / 36.1 | 1.05 / 2.31 |
| mul_into | Cycling | 35.9 / 60.7 | 8.61 / 14.1 | 24.0 / 41.1 | 1.17 / 2.65 |
| mul_into | Fixed | 36.0 / 60.8 | 8.61 / 14.1 | 24.1 / 41.1 | 1.17 / 2.65 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-rijndael16 | `fgf::ops::mul_add` |
| gf16-rijndael16 | `fgf::ops::mul_into` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather_with | 44.0 / 71.8 | 5.84 / 9.97 | 19.9 / 33.3 | 1.04 / 2.27 |
| mul_into_gather_with | 60.5 / 55.3 | 5.86 / 9.88 | 20.0 / 34.3 | 1.06 / 2.29 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-rijndael16 | `fgf::ops::mul_add_gather_with` |
| gf16-rijndael16 | `fgf::ops::mul_into_gather_with` |

</details>

**Counted bytes:** 160 KiB · **Elements:** 81,920 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_matrix_with | 58.6 / 57.2 | 5.85 / 9.85 | 21.3 / 34.4 | 1.06 / 2.30 |
| mul_into_matrix_with | 56.0 / 55.5 | 5.89 / 9.89 | 21.5 / 35.4 | 1.07 / 2.28 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-rijndael16 | `fgf::ops::mul_add_matrix_with` |
| gf16-rijndael16 | `fgf::ops::mul_into_matrix_with` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** v4x / v3_gfni_crypto

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | `reed-solomon-simd` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_assign | Cycling | 51.1 / 72.5 | 8.61 / 14.1 | 24.0 / 41.2 |
| mul_assign | Fixed | 51.1 / 72.6 | 8.62 / 14.1 | 24.1 / 41.3 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign`

</details>

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| scalar_mul | 16.0 / 9.23 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::mul`

</details>

### Preparation and setup

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_coeff | 7.17 / 3.22 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::Coeff::new`

</details>

**Elements:** 16 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_gather | 120 / 65.8 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::new`

</details>

**Elements:** 40 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** v4x / v3_gfni_crypto

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_matrix | 241 / 132 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::from_source_major`

</details>

<details>
<summary>Duplicate-fgf control ratios</summary>

Ratios divide duplicate-fgf latency by primary-fgf latency; medians and ranges span five runs.



| Operation | Coefficients | Counted bytes | Elements | Sources | Destinations | Median ratio | Run range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 0.9999 | 0.9986–1.0001 / 0.9995–1.0001 |
| mul_add | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0011 / 1.0020 | 1.0002–1.0014 / 1.0016–1.0023 |
| mul_add_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0025 / 0.9999 | 1.0003–1.0036 / 0.9977–1.0073 |
| mul_add_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0028 / 1.0003 | 0.9999–1.0052 / 0.9994–1.0009 |
| mul_assign | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 0.9997 | 0.9996–1.0002 / 0.9986–1.0021 |
| mul_assign | Fixed | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 1.0001 | 0.9997–1.0001 / 0.9972–1.0020 |
| mul_into | Cycling | 64 KiB | 32,768 | 1 | 1 | 0.9996 / 0.9998 | 0.9994–1.0002 / 0.9997–1.0000 |
| mul_into | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0005 / 1.0018 | 0.9995–1.0018 / 1.0010–1.0021 |
| mul_into_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0007 / 0.9996 | 0.9989–1.0019 / 0.9986–1.0005 |
| mul_into_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0036 / 0.9969 | 0.9993–1.0172 / 0.9933–1.0005 |

</details>


## gf16-rs-quad

### Native presentations

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 0.246 / 0.482 | 7.882 / 13.06 | 20.96 / 35.43 | 1.064 / 2.401 |
| mul_add | Fixed | 0.246 / 0.480 | 7.885 / 13.06 | 20.98 / 36.40 | 1.061 / 2.406 |
| mul_into | Cycling | 0.248 / 0.506 | 8.605 / 14.12 | 24.07 / 41.00 | 1.167 / 2.650 |
| mul_into | Fixed | 0.247 / 0.506 | 8.609 / 14.11 | 24.08 / 41.03 | 1.166 / 2.651 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-rs-quad | `fgf::ops::mul_add` |
| gf16-rs-quad | `fgf::ops::mul_into` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_gather_with | 0.246 / 0.481 | 5.836 / 10.00 | 19.84 / 33.39 | 1.045 / 2.359 |
| mul_into_gather_with | 0.246 / 0.484 | 5.856 / 10.01 | 19.95 / 34.32 | 1.055 / 2.299 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-rs-quad | `fgf::ops::mul_add_gather_with` |
| gf16-rs-quad | `fgf::ops::mul_into_gather_with` |

</details>

**Counted bytes:** 160 KiB · **Elements:** 81,920 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) | GF-Complete (GiB/s) | Leopard-RS 1.x (GiB/s) | `reed-solomon-erasure` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_add_matrix_with | 0.247 / 0.485 | 5.854 / 10.13 | 21.23 / 34.26 | 1.056 / 2.386 |
| mul_into_matrix_with | 0.247 / 0.487 | 5.886 / 10.02 | 21.43 / 35.21 | 1.065 / 2.331 |

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| gf16-rs-quad | `fgf::ops::mul_add_matrix_with` |
| gf16-rs-quad | `fgf::ops::mul_into_matrix_with` |

</details>

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **fgf routes:** scalar / scalar

| Operation | Coefficients | `fgf` (GiB/s) | GF-Complete (GiB/s) | `reed-solomon-simd` (GiB/s) |
| --- | --- | --- | --- | --- |
| mul_assign | Cycling | 0.249 / 0.509 | 8.614 / 14.13 | 24.08 / 41.30 |
| mul_assign | Fixed | 0.249 / 0.509 | 8.619 / 14.11 | 24.09 / 41.32 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign`

</details>

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| scalar_mul | 17.4 / 10.3 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::mul`

</details>

### Preparation and setup

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_coeff | 3.01 / 1.28 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::Coeff::new`

</details>

**Elements:** 16 · **Sources:** 16 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_gather | 51.1 / 28.7 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::new`

</details>

**Elements:** 40 · **Sources:** 10 · **Destinations:** 4 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (ns/op) |
| --- | --- |
| prepare_matrix | 61.1 / 31.2 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::from_source_major`

</details>

<details>
<summary>Duplicate-fgf control ratios</summary>

Ratios divide duplicate-fgf latency by primary-fgf latency; medians and ranges span five runs.



| Operation | Coefficients | Counted bytes | Elements | Sources | Destinations | Median ratio | Run range |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mul_add | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0001 / 1.0001 | 1.0000–1.0006 / 0.9996–1.0006 |
| mul_add | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0002 / 1.0000 | 1.0000–1.0007 / 0.9995–1.0014 |
| mul_add_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0000 / 1.0001 | 0.9996–1.0001 / 0.9966–1.0002 |
| mul_add_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0000 / 0.9999 | 1.0000–1.0001 / 0.9990–1.0009 |
| mul_assign | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 0.9999 | 0.9999–1.0000 / 0.9998–0.9999 |
| mul_assign | Fixed | 64 KiB | 32,768 | 1 | 1 | 0.9999 / 0.9999 | 0.9998–0.9999 / 0.9998–0.9999 |
| mul_into | Cycling | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9999–1.0001 / 0.9996–1.0001 |
| mul_into | Fixed | 64 KiB | 32,768 | 1 | 1 | 1.0000 / 1.0000 | 0.9999–1.0001 / 0.9996–1.0001 |
| mul_into_gather_with | Fixed | 64 KiB | 32,768 | 16 | 1 | 1.0000 / 1.0001 | 0.9997–1.0001 / 0.9966–1.0003 |
| mul_into_matrix_with | Fixed | 160 KiB | 81,920 | 10 | 4 | 1.0000 / 1.0000 | 0.9999–1.0002 / 0.9990–1.0009 |

</details>

## Measurement details

[Campaign, backend routes, and validation](public-api.md#measurement-details)

Differences comparable to duplicate-fgf control variability do not establish a performance difference. Counted bytes are the throughput numerator, not necessarily one row's size.
