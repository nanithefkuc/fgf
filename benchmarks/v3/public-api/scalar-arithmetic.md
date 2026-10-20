# Scalar arithmetic

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [add](#steady-scalar-add-add-latency)
  - [div](#steady-scalar-div-div-latency)
  - [inv](#steady-scalar-inv-inv-latency)
  - [mul](#steady-scalar-mul-mul-latency)
  - [neg](#steady-scalar-neg-neg-latency)
  - [operators](#steady-scalar-operators-operators-latency)
  - [pow](#steady-scalar-pow-pow-latency)
  - [queries](#steady-scalar-queries-queries-latency)
  - [square](#steady-scalar-square-square-latency)
  - [sub](#steady-scalar-sub-sub-latency)
- [Query and scalar latency](#query-and-scalar-latency)
  - [coefficient matrix emptiness](#query-scalar-collection-coeffmatrix-is-empty-coefficient-matrix-emptiness-latency)
  - [coefficient matrix output count](#query-scalar-collection-coeffmatrix-output-count-coefficient-matrix-output-count-latency)
  - [coefficient matrix value iterator](#query-scalar-collection-coeffmatrix-values-coefficient-matrix-value-iterator-latency)
  - [coefficient vector indexed read](#query-scalar-collection-coeffvec-get-coefficient-vector-indexed-read-latency)
  - [coefficient vector emptiness](#query-scalar-collection-coeffvec-is-empty-coefficient-vector-emptiness-latency)
  - [coefficient vector value iterator](#query-scalar-collection-coeffvec-values-coefficient-vector-value-iterator-latency)
  - [borrowed coefficient vector value iterator](#query-scalar-collection-coeffvecref-values-borrowed-coefficient-vector-value-iterator-latency)
  - [description one](#query-scalar-description-one-description-one-latency)
  - [description structure comparison](#query-scalar-description-same-structure-description-structure-comparison-latency)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-scalar-add-add-latency"></a>

### add

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 1.003 / 0.469 |
| gf125 | 4.350 / 2.134 |
| gf127 | 1.003 / 0.462 |
| gf131 | 1.003 / 0.462 |
| gf1e9p7 | 1.003 / 0.469 |
| gf2p1-poly | 0.745 / 0.426 |
| gf2p1-residue | 1.003 / 0.469 |
| gf2p16-custom-cantor | 0.710 / 0.427 |
| gf2p16-custom-normal | 0.708 / 0.427 |
| gf2p16-custom-rijndael | 0.710 / 0.427 |
| gf2p16-custom-rs | 0.710 / 0.427 |
| gf2p16-fanpaar | 0.706 / 0.426 |
| gf2p16-rijndael | 0.709 / 0.427 |
| gf2p2-cantor | 0.753 / 0.427 |
| gf2p2-fanpaar | 0.753 / 0.426 |
| gf2p2-normal | 0.753 / 0.427 |
| gf2p2-poly | 0.753 / 0.426 |
| gf2p32-fanpaar | 0.753 / 0.426 |
| gf2p32-rijndael | 0.745 / 0.426 |
| gf2p4-cantor | 0.753 / 0.426 |
| gf2p4-fanpaar | 0.753 / 0.426 |
| gf2p4-normal | 0.753 / 0.427 |
| gf2p4-poly | 0.753 / 0.426 |
| gf2p5-padded | 0.748 / 0.426 |
| gf2p64-fanpaar | 0.815 / 0.426 |
| gf2p64-rijndael | 0.815 / 0.426 |
| gf2p8-aes | 0.729 / 0.426 |
| gf2p8-cantor | 0.729 / 0.427 |
| gf2p8-fanpaar | 0.753 / 0.426 |
| gf2p8-normal | 0.735 / 0.426 |
| gf2p8-rs | 0.735 / 0.426 |
| gf49 | 4.112 / 1.897 |
| gf65521 | 1.003 / 0.469 |
| gf65537 | 1.003 / 0.469 |
| gf81 | 5.664 / 2.779 |
| gf9 | 2.941 / 1.340 |
| gfmaxp | 1.003 / 0.469 |
| gfu32max | 1.003 / 0.469 |
| goldilocks | 1.614 / 0.853 |
| goldilocks-residue | 1.003 / 0.469 |
| goldilocks-squared | 3.970 / 2.044 |
| m31 | 1.671 / 0.853 |
| m31-residue | 1.003 / 0.469 |
| maxprime-squared | 2.341 / 1.138 |
| qm31 | 3.721 / 1.711 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::add`, `fgf::field::Field::add_raw`

</details>

<a id="steady-scalar-div-div-latency"></a>

### div

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 17.62 / 8.584 |
| gf125 | 175.1 / 78.89 |
| gf127 | 25.03 / 10.52 |
| gf131 | 17.79 / 8.023 |
| gf1e9p7 | 310.3 / 137.5 |
| gf2p1-poly | 11.33 / 5.679 |
| gf2p1-residue | 0.827 / 0.426 |
| gf2p16-custom-cantor | 30.13 / 12.82 |
| gf2p16-custom-normal | 30.67 / 13.20 |
| gf2p16-custom-rijndael | 30.15 / 12.81 |
| gf2p16-custom-rs | 30.08 / 12.81 |
| gf2p16-fanpaar | 561.6 / 354.2 |
| gf2p16-rijndael | 27.41 / 12.95 |
| gf2p2-cantor | 62.21 / 28.72 |
| gf2p2-fanpaar | 0.668 / 0.426 |
| gf2p2-normal | 0.668 / 0.426 |
| gf2p2-poly | 32.09 / 14.53 |
| gf2p32-fanpaar | 2290 / 1404 |
| gf2p32-rijndael | 98.81 / 55.05 |
| gf2p4-cantor | 95.20 / 47.30 |
| gf2p4-fanpaar | 657.3 / 292.8 |
| gf2p4-normal | 63.02 / 28.61 |
| gf2p4-poly | 65.45 / 33.00 |
| gf2p5-padded | 908.9 / 445.1 |
| gf2p64-fanpaar | 7396 / 4506 |
| gf2p64-rijndael | 303.0 / 162.1 |
| gf2p8-aes | 2.354 / 1.228 |
| gf2p8-cantor | 2.354 / 1.231 |
| gf2p8-fanpaar | 3411 / 1518 |
| gf2p8-normal | 2.354 / 1.227 |
| gf2p8-rs | 2.354 / 1.231 |
| gf49 | 64.33 / 32.77 |
| gf65521 | 177.9 / 75.71 |
| gf65537 | 99.39 / 41.81 |
| gf81 | 98.20 / 43.02 |
| gf9 | 22.56 / 8.579 |
| gfmaxp | 733.6 / 397.3 |
| gfu32max | 369.9 / 186.4 |
| goldilocks | 759.9 / 406.6 |
| goldilocks-residue | 751.4 / 405.8 |
| goldilocks-squared | 756.2 / 456.2 |
| m31 | 348.8 / 181.8 |
| m31-residue | 358.5 / 180.9 |
| maxprime-squared | 779.5 / 430.7 |
| qm31 | 387.6 / 212.2 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::div`

</details>

<a id="steady-scalar-inv-inv-latency"></a>

### inv

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 13.24 / 6.946 |
| gf125 | 128.3 / 56.34 |
| gf127 | 18.03 / 8.571 |
| gf131 | 13.04 / 6.203 |
| gf1e9p7 | 301.5 / 133.7 |
| gf2p1-poly | 4.038 / 2.036 |
| gf2p1-residue | 0.468 / 0.228 |
| gf2p16-custom-cantor | 13.82 / 6.600 |
| gf2p16-custom-normal | 15.42 / 7.067 |
| gf2p16-custom-rijndael | 13.84 / 6.597 |
| gf2p16-custom-rs | 13.84 / 6.597 |
| gf2p16-fanpaar | 241.5 / 152.5 |
| gf2p16-rijndael | 13.89 / 6.626 |
| gf2p2-cantor | 17.43 / 7.728 |
| gf2p2-fanpaar | 68.21 / 32.29 |
| gf2p2-normal | 17.39 / 7.737 |
| gf2p2-poly | 19.48 / 9.000 |
| gf2p32-fanpaar | 1324 / 833.0 |
| gf2p32-rijndael | 69.00 / 35.73 |
| gf2p4-cantor | 41.48 / 20.51 |
| gf2p4-fanpaar | 429.8 / 191.5 |
| gf2p4-normal | 33.55 / 15.70 |
| gf2p4-poly | 19.37 / 8.955 |
| gf2p5-padded | 631.1 / 310.6 |
| gf2p64-fanpaar | 4488 / 2734 |
| gf2p64-rijndael | 199.7 / 107.6 |
| gf2p8-aes | 1.003 / 0.533 |
| gf2p8-cantor | 1.003 / 0.533 |
| gf2p8-fanpaar | 2262 / 1010 |
| gf2p8-normal | 1.003 / 0.533 |
| gf2p8-rs | 1.003 / 0.533 |
| gf49 | 29.40 / 15.68 |
| gf65521 | 168.8 / 72.55 |
| gf65537 | 93.58 / 40.30 |
| gf81 | 50.87 / 23.23 |
| gf9 | 8.506 / 3.344 |
| gfmaxp | 729.3 / 394.9 |
| gfu32max | 366.4 / 182.5 |
| goldilocks | 703.2 / 404.0 |
| goldilocks-residue | 752.0 / 400.5 |
| goldilocks-squared | 728.4 / 426.2 |
| m31 | 347.1 / 180.7 |
| m31-residue | 359.5 / 175.8 |
| maxprime-squared | 749.9 / 408.3 |
| qm31 | 367.5 / 192.0 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::inv`, `fgf::field::Field::inv_raw`

</details>

<a id="steady-scalar-mul-mul-latency"></a>

### mul

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 3.343 / 2.133 |
| gf125 | 34.60 / 14.05 |
| gf127 | 3.343 / 2.133 |
| gf131 | 3.343 / 2.133 |
| gf1e9p7 | 3.343 / 2.133 |
| gf2p1-poly | 6.945 / 3.555 |
| gf2p1-residue | 0.776 / 0.426 |
| gf2p16-custom-cantor | 7.438 / 3.660 |
| gf2p16-custom-normal | 7.393 / 3.702 |
| gf2p16-custom-rijndael | 7.431 / 3.686 |
| gf2p16-custom-rs | 7.491 / 3.689 |
| gf2p16-fanpaar | 316.7 / 192.8 |
| gf2p16-rijndael | 8.706 / 3.995 |
| gf2p2-cantor | 27.83 / 12.53 |
| gf2p2-fanpaar | 44.50 / 20.59 |
| gf2p2-normal | 27.92 / 12.51 |
| gf2p2-poly | 12.14 / 5.679 |
| gf2p32-fanpaar | 964.1 / 591.9 |
| gf2p32-rijndael | 26.80 / 12.96 |
| gf2p4-cantor | 37.19 / 16.92 |
| gf2p4-fanpaar | 227.9 / 101.8 |
| gf2p4-normal | 37.21 / 16.94 |
| gf2p4-poly | 21.12 / 9.414 |
| gf2p5-padded | 298.2 / 145.6 |
| gf2p64-fanpaar | 2909 / 1769 |
| gf2p64-rijndael | 99.04 / 50.94 |
| gf2p8-aes | 1.504 / 0.785 |
| gf2p8-cantor | 1.504 / 0.786 |
| gf2p8-fanpaar | 1145 / 509.1 |
| gf2p8-normal | 1.504 / 0.785 |
| gf2p8-rs | 1.504 / 0.785 |
| gf49 | 17.51 / 9.171 |
| gf65521 | 3.343 / 2.133 |
| gf65537 | 1.630 / 0.758 |
| gf81 | 38.42 / 14.99 |
| gf9 | 8.008 / 3.340 |
| gfmaxp | 3.343 / 2.133 |
| gfu32max | 3.343 / 2.133 |
| goldilocks | 2.597 / 1.221 |
| goldilocks-residue | 3.353 / 2.133 |
| goldilocks-squared | 20.48 / 10.39 |
| m31 | 2.582 / 1.183 |
| m31-residue | 3.343 / 2.133 |
| maxprime-squared | 14.20 / 8.531 |
| qm31 | 10.96 / 4.507 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::mul`, `fgf::field::Field::mul_raw`

</details>

<a id="steady-scalar-neg-neg-latency"></a>

### neg

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf101 | 0.557 / 0.293 | 0/5 / 0/5 |
| gf125 | 2.357 / 1.115 | 0/5 / 0/5 |
| gf127 | 0.557 / 0.284 | 0/5 / 0/5 |
| gf131 | 0.585 / 0.284 | 0/5 / 0/5 |
| gf1e9p7 | 0.585 / 0.299 | 0/5 / 0/5 |
| gf2p1-poly | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p1-residue | 0.585 / 0.284 | 0/5 / 0/5 |
| gf2p16-custom-cantor | 0.433 / 0.213 | 0/5 / 2/5 |
| gf2p16-custom-normal | 0.433 / 0.213 | 0/5 / 1/5 |
| gf2p16-custom-rijndael | 0.433 / 0.213 | 0/5 / 1/5 |
| gf2p16-custom-rs | 0.433 / 0.213 | 0/5 / 3/5 |
| gf2p16-fanpaar | 0.433 / 0.213 | 0/5 / 3/5 |
| gf2p16-rijndael | 0.432 / 0.213 | 0/5 / 1/5 |
| gf2p2-cantor | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p2-fanpaar | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p2-normal | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p2-poly | 0.427 / 0.213 | 0/5 / 0/5 |
| gf2p32-fanpaar | 0.427 / 0.213 | 0/5 / 3/5 |
| gf2p32-rijndael | 0.427 / 0.213 | 0/5 / 2/5 |
| gf2p4-cantor | 0.427 / 0.213 | 0/5 / 2/5 |
| gf2p4-fanpaar | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p4-normal | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p4-poly | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p5-padded | 0.413 / 0.213 | 0/5 / 1/5 |
| gf2p64-fanpaar | 0.468 / 0.213 | 0/5 / 0/5 |
| gf2p64-rijndael | 0.468 / 0.213 | 0/5 / 2/5 |
| gf2p8-aes | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p8-cantor | 0.427 / 0.213 | 0/5 / 0/5 |
| gf2p8-fanpaar | 0.427 / 0.213 | 0/5 / 0/5 |
| gf2p8-normal | 0.427 / 0.213 | 0/5 / 2/5 |
| gf2p8-rs | 0.427 / 0.213 | 0/5 / 1/5 |
| gf49 | 2.672 / 1.173 | 0/5 / 0/5 |
| gf65521 | 0.557 / 0.284 | 0/5 / 0/5 |
| gf65537 | 0.585 / 0.284 | 0/5 / 0/5 |
| gf81 | 3.113 / 1.493 | 0/5 / 0/5 |
| gf9 | 1.638 / 0.782 | 0/5 / 0/5 |
| gfmaxp | 0.585 / 0.299 | 0/5 / 0/5 |
| gfu32max | 0.557 / 0.293 | 0/5 / 0/5 |
| goldilocks | 1.003 / 0.469 | 0/5 / 0/5 |
| goldilocks-residue | 0.585 / 0.284 | 0/5 / 0/5 |
| goldilocks-squared | 2.617 / 1.179 | 0/5 / 0/5 |
| m31 | 1.062 / 0.555 | 0/5 / 0/5 |
| m31-residue | 0.557 / 0.284 | 0/5 / 0/5 |
| maxprime-squared | 1.504 / 0.782 | 0/5 / 0/5 |
| qm31 | 2.416 / 1.176 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::neg`, `fgf::field::Field::neg_raw`

</details>

<a id="steady-scalar-operators-operators-latency"></a>

### operators

**Elements:** 1 · **Sources:** 8 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 75.975 / 31.028 |
| gf125 | 460.39 / 222.63 |
| gf127 | 76.711 / 31.796 |
| gf131 | 72.035 / 31.007 |
| gf1e9p7 | 363.27 / 176.37 |
| gf2p1-poly | 79.450 / 39.018 |
| gf2p1-residue | 10.921 / 4.9800 |
| gf2p16-custom-cantor | 142.35 / 76.126 |
| gf2p16-custom-normal | 144.80 / 79.101 |
| gf2p16-custom-rijndael | 142.36 / 76.081 |
| gf2p16-custom-rs | 142.39 / 76.010 |
| gf2p16-fanpaar | 3424.4 / 2089.0 |
| gf2p16-rijndael | 141.19 / 74.633 |
| gf2p2-cantor | 313.49 / 141.61 |
| gf2p2-fanpaar | 527.78 / 247.12 |
| gf2p2-normal | 322.50 / 146.21 |
| gf2p2-poly | 138.75 / 66.181 |
| gf2p32-fanpaar | 10979 / 6728.3 |
| gf2p32-rijndael | 353.64 / 182.10 |
| gf2p4-cantor | 431.40 / 205.12 |
| gf2p4-fanpaar | 2715.2 / 1219.7 |
| gf2p4-normal | 440.64 / 210.37 |
| gf2p4-poly | 248.60 / 131.60 |
| gf2p5-padded | 2839.2 / 1376.5 |
| gf2p64-fanpaar | 33602 / 20432 |
| gf2p64-rijndael | 1155.2 / 605.76 |
| gf2p8-aes | 59.704 / 34.428 |
| gf2p8-cantor | 59.702 / 34.430 |
| gf2p8-fanpaar | 13741 / 6092.8 |
| gf2p8-normal | 59.687 / 34.432 |
| gf2p8-rs | 59.702 / 34.428 |
| gf49 | 266.67 / 131.58 |
| gf65521 | 230.84 / 110.57 |
| gf65537 | 135.96 / 57.179 |
| gf81 | 627.18 / 283.66 |
| gf9 | 175.07 / 79.934 |
| gfmaxp | 775.32 / 431.64 |
| gfu32max | 414.73 / 217.58 |
| goldilocks | 753.29 / 446.86 |
| goldilocks-residue | 810.66 / 438.04 |
| goldilocks-squared | 1004.7 / 585.21 |
| m31 | 396.14 / 219.14 |
| m31-residue | 408.65 / 212.85 |
| maxprime-squared | 968.04 / 534.10 |
| qm31 | 550.14 / 282.55 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`core::ops::Add for fgf::Elem`, `core::ops::Sub for fgf::Elem`, `core::ops::Mul for fgf::Elem`, `core::ops::Div for fgf::Elem`, `core::ops::Neg for fgf::Elem`, `core::ops::AddAssign for fgf::Elem`, `core::ops::SubAssign for fgf::Elem`, `core::ops::MulAssign for fgf::Elem`, `core::ops::DivAssign for fgf::Elem`, `core::iter::Sum for fgf::Elem`, `core::iter::Product for fgf::Elem`

</details>

<a id="steady-scalar-pow-pow-latency"></a>

### pow

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 20.648 / 12.796 |
| gf125 | 98.718 / 41.932 |
| gf127 | 20.646 / 12.796 |
| gf131 | 20.631 / 12.796 |
| gf1e9p7 | 20.627 / 12.795 |
| gf2p1-poly | 44.709 / 21.899 |
| gf2p1-residue | 3.1894 / 1.4930 |
| gf2p16-custom-cantor | 50.128 / 25.845 |
| gf2p16-custom-normal | 51.042 / 26.248 |
| gf2p16-custom-rijndael | 50.146 / 26.231 |
| gf2p16-custom-rs | 50.345 / 26.021 |
| gf2p16-fanpaar | 1914.6 / 1171.2 |
| gf2p16-rijndael | 53.839 / 26.361 |
| gf2p2-cantor | 171.15 / 75.943 |
| gf2p2-fanpaar | 270.61 / 124.29 |
| gf2p2-normal | 171.18 / 75.977 |
| gf2p2-poly | 75.487 / 34.457 |
| gf2p32-fanpaar | 5794.2 / 3561.9 |
| gf2p32-rijndael | 154.37 / 78.084 |
| gf2p4-cantor | 226.59 / 102.12 |
| gf2p4-fanpaar | 1382.1 / 619.57 |
| gf2p4-normal | 226.60 / 102.40 |
| gf2p4-poly | 130.47 / 57.595 |
| gf2p5-padded | 1527.1 / 749.82 |
| gf2p64-fanpaar | 17476 / 10625 |
| gf2p64-rijndael | 539.35 / 280.78 |
| gf2p8-aes | 22.717 / 12.377 |
| gf2p8-cantor | 22.716 / 12.376 |
| gf2p8-fanpaar | 6888.7 / 3055.9 |
| gf2p8-normal | 22.729 / 12.386 |
| gf2p8-rs | 22.734 / 12.374 |
| gf49 | 110.48 / 58.065 |
| gf65521 | 21.053 / 12.796 |
| gf65537 | 13.259 / 4.9832 |
| gf81 | 287.59 / 125.37 |
| gf9 | 47.606 / 19.805 |
| gfmaxp | 20.630 / 12.796 |
| gfu32max | 20.623 / 12.796 |
| goldilocks | 20.778 / 8.9735 |
| goldilocks-residue | 20.678 / 12.796 |
| goldilocks-squared | 122.56 / 62.121 |
| m31 | 18.689 / 6.9125 |
| m31-residue | 20.646 / 12.796 |
| maxprime-squared | 83.791 / 49.053 |
| qm31 | 72.331 / 27.753 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::pow`

</details>

<a id="steady-scalar-queries-queries-latency"></a>

### queries

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 1.504 / 0.711 |
| gf125 | 3.163 / 1.464 |
| gf127 | 1.671 / 0.782 |
| gf131 | 1.170 / 0.555 |
| gf1e9p7 | 1.170 / 0.555 |
| gf2p1-poly | 0.668 / 0.427 |
| gf2p1-residue | 0.743 / 0.427 |
| gf2p16-custom-cantor | 0.891 / 0.549 |
| gf2p16-custom-normal | 0.891 / 0.549 |
| gf2p16-custom-rijndael | 0.891 / 0.549 |
| gf2p16-custom-rs | 0.891 / 0.549 |
| gf2p16-fanpaar | 0.891 / 0.549 |
| gf2p16-rijndael | 0.891 / 0.549 |
| gf2p2-cantor | 0.836 / 0.427 |
| gf2p2-fanpaar | 0.836 / 0.427 |
| gf2p2-normal | 0.836 / 0.427 |
| gf2p2-poly | 0.891 / 0.427 |
| gf2p32-fanpaar | 0.836 / 0.427 |
| gf2p32-rijndael | 0.836 / 0.426 |
| gf2p4-cantor | 0.836 / 0.427 |
| gf2p4-fanpaar | 0.836 / 0.427 |
| gf2p4-normal | 0.891 / 0.427 |
| gf2p4-poly | 0.891 / 0.427 |
| gf2p5-padded | 0.891 / 0.427 |
| gf2p64-fanpaar | 1.003 / 0.427 |
| gf2p64-rijndael | 1.003 / 0.427 |
| gf2p8-aes | 0.836 / 0.426 |
| gf2p8-cantor | 0.891 / 0.427 |
| gf2p8-fanpaar | 0.836 / 0.427 |
| gf2p8-normal | 0.836 / 0.427 |
| gf2p8-rs | 0.891 / 0.427 |
| gf49 | 3.554 / 1.573 |
| gf65521 | 1.504 / 0.711 |
| gf65537 | 1.337 / 0.640 |
| gf81 | 5.490 / 2.747 |
| gf9 | 2.438 / 1.079 |
| gfmaxp | 1.003 / 0.469 |
| gfu32max | 1.170 / 0.555 |
| goldilocks | 1.170 / 0.533 |
| goldilocks-residue | 1.003 / 0.480 |
| goldilocks-squared | 1.844 / 0.854 |
| m31 | 1.337 / 0.640 |
| m31-residue | 1.671 / 0.782 |
| maxprime-squared | 1.674 / 0.782 |
| qm31 | 2.005 / 0.999 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::from_raw`, `fgf::field::FieldElem::is_zero`, `fgf::field::FieldElem::is_one`, `fgf::Elem::to_raw`

</details>

<a id="steady-scalar-square-square-latency"></a>

### square

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 3.342 / 2.133 |
| gf125 | 23.03 / 9.778 |
| gf127 | 3.342 / 2.133 |
| gf131 | 3.342 / 2.133 |
| gf1e9p7 | 3.342 / 2.133 |
| gf2p1-poly | 6.822 / 3.477 |
| gf2p1-residue | 0.439 / 0.213 |
| gf2p16-custom-cantor | 7.071 / 3.365 |
| gf2p16-custom-normal | 7.071 / 3.441 |
| gf2p16-custom-rijndael | 7.153 / 3.372 |
| gf2p16-custom-rs | 7.150 / 3.369 |
| gf2p16-fanpaar | 316.7 / 192.9 |
| gf2p16-rijndael | 8.662 / 3.783 |
| gf2p2-cantor | 27.84 / 12.51 |
| gf2p2-fanpaar | 45.08 / 20.57 |
| gf2p2-normal | 27.79 / 12.44 |
| gf2p2-poly | 12.16 / 5.568 |
| gf2p32-fanpaar | 963.6 / 603.1 |
| gf2p32-rijndael | 26.80 / 13.00 |
| gf2p4-cantor | 37.13 / 16.88 |
| gf2p4-fanpaar | 228.3 / 101.6 |
| gf2p4-normal | 37.15 / 16.87 |
| gf2p4-poly | 21.25 / 9.392 |
| gf2p5-padded | 185.4 / 90.91 |
| gf2p64-fanpaar | 2908 / 1779 |
| gf2p64-rijndael | 99.56 / 51.06 |
| gf2p8-aes | 1.003 / 0.534 |
| gf2p8-cantor | 1.003 / 0.534 |
| gf2p8-fanpaar | 1145 / 506.8 |
| gf2p8-normal | 1.003 / 0.513 |
| gf2p8-rs | 1.003 / 0.534 |
| gf49 | 15.07 / 6.729 |
| gf65521 | 3.342 / 2.133 |
| gf65537 | 1.401 / 0.640 |
| gf81 | 42.80 / 16.85 |
| gf9 | 5.571 / 2.139 |
| gfmaxp | 3.342 / 2.133 |
| gfu32max | 3.342 / 2.133 |
| goldilocks | 2.229 / 1.039 |
| goldilocks-residue | 3.342 / 2.133 |
| goldilocks-squared | 20.16 / 10.16 |
| m31 | 2.025 / 0.857 |
| m31-residue | 3.342 / 2.133 |
| maxprime-squared | 10.71 / 6.402 |
| qm31 | 10.98 / 4.453 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::square`

</details>

<a id="steady-scalar-sub-sub-latency"></a>

### sub

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 0.819 / 0.426 |
| gf125 | 3.132 / 1.503 |
| gf127 | 0.811 / 0.426 |
| gf131 | 0.811 / 0.427 |
| gf1e9p7 | 0.811 / 0.426 |
| gf2p1-poly | 0.745 / 0.427 |
| gf2p1-residue | 0.811 / 0.426 |
| gf2p16-custom-cantor | 0.710 / 0.427 |
| gf2p16-custom-normal | 0.706 / 0.426 |
| gf2p16-custom-rijndael | 0.708 / 0.427 |
| gf2p16-custom-rs | 0.710 / 0.427 |
| gf2p16-fanpaar | 0.710 / 0.427 |
| gf2p16-rijndael | 0.709 / 0.426 |
| gf2p2-cantor | 0.745 / 0.426 |
| gf2p2-fanpaar | 0.753 / 0.426 |
| gf2p2-normal | 0.753 / 0.426 |
| gf2p2-poly | 0.753 / 0.426 |
| gf2p32-fanpaar | 0.745 / 0.426 |
| gf2p32-rijndael | 0.735 / 0.426 |
| gf2p4-cantor | 0.753 / 0.426 |
| gf2p4-fanpaar | 0.746 / 0.426 |
| gf2p4-normal | 0.753 / 0.427 |
| gf2p4-poly | 0.753 / 0.426 |
| gf2p5-padded | 0.747 / 0.426 |
| gf2p64-fanpaar | 0.819 / 0.426 |
| gf2p64-rijndael | 0.819 / 0.426 |
| gf2p8-aes | 0.735 / 0.426 |
| gf2p8-cantor | 0.735 / 0.427 |
| gf2p8-fanpaar | 0.753 / 0.426 |
| gf2p8-normal | 0.735 / 0.427 |
| gf2p8-rs | 0.735 / 0.426 |
| gf49 | 3.337 / 1.410 |
| gf65521 | 0.811 / 0.426 |
| gf65537 | 0.814 / 0.426 |
| gf81 | 4.319 / 1.946 |
| gf9 | 2.146 / 0.960 |
| gfmaxp | 0.819 / 0.426 |
| gfu32max | 0.819 / 0.426 |
| goldilocks | 1.836 / 0.960 |
| goldilocks-residue | 0.819 / 0.426 |
| goldilocks-squared | 4.195 / 2.242 |
| m31 | 1.671 / 0.853 |
| m31-residue | 0.819 / 0.426 |
| maxprime-squared | 1.639 / 0.747 |
| qm31 | 3.721 / 1.711 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::sub`, `fgf::field::Field::sub_raw`

</details>

## Query and scalar latency

Black-boxed scalar and getter calls; these report per-call latency, never a throughput.

<a id="query-scalar-collection-coeffmatrix-is-empty-coefficient-matrix-emptiness-latency"></a>

### coefficient matrix emptiness

**Elements:** 12 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 0.668 / 0.426 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::is_empty`

</details>

<a id="query-scalar-collection-coeffmatrix-output-count-coefficient-matrix-output-count-latency"></a>

### coefficient matrix output count

**Elements:** 12 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 0.668 / 0.426 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::output_count`

</details>

<a id="query-scalar-collection-coeffmatrix-values-coefficient-matrix-value-iterator-latency"></a>

### coefficient matrix value iterator

**Elements:** 12 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 0.835 / 0.426 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::values`

</details>

<a id="query-scalar-collection-coeffvec-get-coefficient-vector-indexed-read-latency"></a>

### coefficient vector indexed read

**Elements:** 8 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 0.936 / 0.427 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::get`

</details>

<a id="query-scalar-collection-coeffvec-is-empty-coefficient-vector-emptiness-latency"></a>

### coefficient vector emptiness

**Elements:** 8 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 0.668 / 0.426 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::is_empty`

</details>

<a id="query-scalar-collection-coeffvec-values-coefficient-vector-value-iterator-latency"></a>

### coefficient vector value iterator

**Elements:** 8 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 0.835 / 0.426 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::values`

</details>

<a id="query-scalar-collection-coeffvecref-values-borrowed-coefficient-vector-value-iterator-latency"></a>

### borrowed coefficient vector value iterator

**Elements:** 8 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 1.246 / 0.640 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVecRef::values`

</details>

<a id="query-scalar-description-one-description-one-latency"></a>

### description one

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 1.170 / 0.587 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::BinaryField::DESCRIPTION`, `fgf::BinaryDescription::one`

</details>

<a id="query-scalar-description-same-structure-description-structure-comparison-latency"></a>

### description structure comparison

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 4.01 / 2.35 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::BinaryField::DESCRIPTION`, `fgf::BinaryDescription::same_structure`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
