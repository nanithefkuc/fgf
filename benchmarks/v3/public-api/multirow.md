# Scatter, gather, and matrix

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [mul_into_matrix](#steady-bulk-matrix-into-one-shot-mul-into-matrix-throughput)
  - [mul_into_gather_with](#steady-details-maps-gather-into-prepared-mul-into-gather-with-throughput)
  - [mul_into_matrix_with](#steady-details-maps-matrix-into-prepared-mul-into-matrix-with-throughput)
  - [mul_add_gather](#steady-spread-gather-mul-add-gather-throughput)
  - [mul_add_matrix (gather empty sources)](#steady-spread-gather-empty-sources-mul-add-matrix-gather-empty-sources-latency)
  - [add_gather_offsets](#steady-spread-gather-offsets-add-gather-offsets-throughput)
  - [mul_add_gather_with](#steady-spread-gather-prepared-mul-add-gather-with-throughput)
  - [mul_add_scatter (gather single source)](#steady-spread-gather-single-source-mul-add-scatter-gather-single-source-throughput)
  - [mul_add_matrix](#steady-spread-matrix-mul-add-matrix-throughput)
  - [mul_add_matrix_at](#steady-spread-matrix-at-mul-add-matrix-at-throughput)
  - [mul_add_matrix (matrix empty terms)](#steady-spread-matrix-empty-terms-mul-add-matrix-matrix-empty-terms-latency)
  - [mul_add_matrix_with](#steady-spread-matrix-prepared-mul-add-matrix-with-throughput)
  - [mul_add_scatter](#steady-spread-scatter-mul-add-scatter-throughput)
  - [mul_add_scatter_with](#steady-spread-scatter-prepared-mul-add-scatter-with-throughput)
  - [mul_add_scatter (scatter zero coeffs)](#steady-spread-scatter-zero-coeffs-mul-add-scatter-scatter-zero-coeffs-throughput)
- [Query and scalar latency](#query-and-scalar-latency)
  - [coeff_value](#query-details-maps-coeff-value-coeff-value-latency)
  - [coeffvecref_get](#query-details-maps-coeffvecref-get-coeffvecref-get-latency)
  - [coeffvecref_is_empty](#query-details-maps-coeffvecref-is-empty-coeffvecref-is-empty-latency)
  - [coeffvecref_len](#query-details-maps-coeffvecref-len-coeffvecref-len-latency)
  - [has_vector_elementwise](#query-details-maps-has-vector-elementwise-has-vector-elementwise-latency)
  - [relation_to_coordinates](#query-details-maps-relation-to-coordinates-relation-to-coordinates-latency)
  - [vector_elementwise_min_bytes](#query-details-maps-vector-elementwise-min-bytes-vector-elementwise-min-bytes-latency)
- [Preparation and setup](#preparation-and-setup)
  - [relation_from_binary_words](#setup-details-maps-relation-from-binary-words-relation-from-binary-words-latency)
  - [relation_from_coordinates](#setup-details-maps-relation-from-coordinates-relation-from-coordinates-latency)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-bulk-matrix-into-one-shot-mul-into-matrix-throughput"></a>

### mul_into_matrix

**Sources:** 4 · **Destinations:** 3

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 48 KiB | 49,152 | 0.173213 / 0.327949 |
| gf125 | 49140 B | 16,380 | 0.064261 / 0.146293 |
| gf127 | 48 KiB | 49,152 | 0.179641 / 0.331376 |
| gf131 | 48 KiB | 49,152 | 0.198886 / 0.368900 |
| gf1e9p7 | 48 KiB | 12,288 | 0.583779 / 1.334875 |
| gf2p1-residue | 48 KiB | 6,144 | 7.838792 / 19.77470 |
| gf2p16-custom-cantor | 48 KiB | 24,576 | 0.278605 / 0.559799 |
| gf2p16-custom-normal | 48 KiB | 24,576 | 0.276940 / 0.546054 |
| gf2p16-custom-rijndael | 48 KiB | 24,576 | 0.283497 / 0.574765 |
| gf2p16-custom-rs | 48 KiB | 24,576 | 0.285250 / 0.574512 |
| gf2p16-fanpaar | 48 KiB | 24,576 | 8.809840 / 15.37898 |
| gf2p16-rijndael | 48 KiB | 24,576 | 33.00830 / 44.21451 |
| gf2p32-fanpaar | 48 KiB | 12,288 | 3.504009 / 5.763592 |
| gf2p32-rijndael | 48 KiB | 12,288 | 14.82605 / 27.31440 |
| gf2p64-fanpaar | 48 KiB | 6,144 | 0.956589 / 1.390750 |
| gf2p64-rijndael | 48 KiB | 6,144 | 7.137502 / 13.34407 |
| gf2p8-aes | 48 KiB | 49,152 | 111.5103 / 149.8913 |
| gf2p8-cantor | 48 KiB | 49,152 | 0.636173 / 1.402011 |
| gf2p8-fanpaar | 48 KiB | 49,152 | 0.000812 / 0.001836 |
| gf2p8-normal | 48 KiB | 49,152 | 0.633740 / 1.395988 |
| gf2p8-rs | 48 KiB | 49,152 | 111.5524 / 166.3226 |
| gf49 | 48 KiB | 24,576 | 0.054869 / 0.128720 |
| gf65521 | 48 KiB | 24,576 | 0.468100 / 0.864451 |
| gf65537 | 48 KiB | 12,288 | 0.833747 / 1.872732 |
| gf81 | 48 KiB | 12,288 | 0.059089 / 0.132612 |
| gf9 | 48 KiB | 24,576 | 0.128126 / 0.295494 |
| gfmaxp | 48 KiB | 6,144 | 1.834954 / 3.470898 |
| gfu32max | 48 KiB | 12,288 | 0.690982 / 1.572254 |
| goldilocks | 48 KiB | 6,144 | 1.425894 / 3.658451 |
| goldilocks-residue | 48 KiB | 6,144 | 1.671318 / 3.467151 |
| goldilocks-squared | 48 KiB | 3,072 | 0.522224 / 1.033114 |
| m31 | 48 KiB | 12,288 | 1.307280 / 2.169933 |
| m31-residue | 48 KiB | 12,288 | 0.645985 / 1.461360 |
| maxprime-squared | 48 KiB | 3,072 | 0.764366 / 1.725091 |
| qm31 | 48 KiB | 6,144 | 9.550292 / 11.35960 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_into_matrix`

</details>

<a id="steady-details-maps-gather-into-prepared-mul-into-gather-with-throughput"></a>

### mul_into_gather_with

**Counted bytes:** 16 KiB · **Sources:** 4 · **Destinations:** 1

| Field | Elements | Throughput (GiB/s) |
| --- | --- | --- |
| gf101 | 16,384 | 14.38 / 14.19 |
| gf2p16-custom-rs | 8,192 | 0.281 / 0.557 |
| gf2p16-fanpaar | 8,192 | 9.396 / 16.15 |
| gf2p16-rijndael | 8,192 | 41.72 / 57.91 |
| gf2p8-aes | 16,384 | 86.73 / 85.26 |
| gf2p8-rs | 16,384 | 83.59 / 113.0 |
| gf65521 | 8,192 | 0.575 / 1.057 |
| gfmaxp | 2,048 | 5.449 / 4.863 |
| gfu32max | 4,096 | 9.013 / 10.33 |
| goldilocks | 2,048 | 9.169 / 8.573 |
| m31 | 4,096 | 14.69 / 18.72 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_into_gather_with`, `fgf::ops::CoeffVec::new`, `fgf::ops::CoeffVec::as_ref`

</details>

<a id="steady-details-maps-matrix-into-prepared-mul-into-matrix-with-throughput"></a>

### mul_into_matrix_with

**Counted bytes:** 48 KiB · **Sources:** 4 · **Destinations:** 3

| Field | Elements | Throughput (GiB/s) |
| --- | --- | --- |
| gf101 | 49,152 | 12.18 / 11.86 |
| gf2p16-custom-rs | 24,576 | 0.280 / 0.547 |
| gf2p16-fanpaar | 24,576 | 9.126 / 15.74 |
| gf2p16-rijndael | 24,576 | 34.14 / 44.77 |
| gf2p8-aes | 49,152 | 109.5 / 137.5 |
| gf2p8-rs | 49,152 | 109.4 / 160.4 |
| gf65521 | 24,576 | 0.755 / 1.841 |
| gfmaxp | 6,144 | 5.175 / 4.580 |
| gfu32max | 12,288 | 7.756 / 8.847 |
| goldilocks | 6,144 | 8.721 / 7.760 |
| m31 | 12,288 | 13.82 / 16.80 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_into_matrix_with`, `fgf::ops::CoeffMatrix::from_source_major`

</details>

<a id="steady-spread-gather-mul-add-gather-throughput"></a>

### mul_add_gather

**Sources:** 8 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 32 KiB | 32,768 | 0.172254 / 0.326934 |
| gf125 | 32760 B | 10,920 | 0.065563 / 0.149786 |
| gf127 | 32 KiB | 32,768 | 0.179006 / 0.328597 |
| gf131 | 32 KiB | 32,768 | 0.198850 / 0.370235 |
| gf1e9p7 | 32 KiB | 8,192 | 0.576441 / 1.329974 |
| gf2p1-residue | 32 KiB | 4,096 | 8.311642 / 16.96931 |
| gf2p16-custom-cantor | 32 KiB | 16,384 | 0.236424 / 0.439104 |
| gf2p16-custom-normal | 32 KiB | 16,384 | 0.224104 / 0.416122 |
| gf2p16-custom-rijndael | 32 KiB | 16,384 | 0.218589 / 0.412134 |
| gf2p16-custom-rs | 32 KiB | 16,384 | 0.220503 / 0.409506 |
| gf2p16-fanpaar | 32 KiB | 16,384 | 9.017131 / 15.56289 |
| gf2p16-rijndael | 32 KiB | 16,384 | 38.55095 / 73.02448 |
| gf2p32-fanpaar | 32 KiB | 8,192 | 3.547524 / 5.717213 |
| gf2p32-rijndael | 32 KiB | 8,192 | 15.42531 / 28.43157 |
| gf2p64-fanpaar | 32 KiB | 4,096 | 0.948292 / 1.392360 |
| gf2p64-rijndael | 32 KiB | 4,096 | 7.267171 / 13.55420 |
| gf2p8-aes | 32 KiB | 32,768 | 64.76153 / 105.2017 |
| gf2p8-cantor | 32 KiB | 32,768 | 0.647136 / 1.253887 |
| gf2p8-fanpaar | 32 KiB | 32,768 | 0.000814 / 0.001827 |
| gf2p8-normal | 32 KiB | 32,768 | 0.646386 / 1.255628 |
| gf2p8-rs | 32 KiB | 32,768 | 64.81433 / 106.2423 |
| gf49 | 32 KiB | 16,384 | 0.055024 / 0.129056 |
| gf65521 | 32 KiB | 16,384 | 0.478581 / 0.863313 |
| gf65537 | 32 KiB | 8,192 | 0.834024 / 1.869305 |
| gf81 | 32 KiB | 8,192 | 0.063163 / 0.143522 |
| gf9 | 32 KiB | 16,384 | 0.123546 / 0.283959 |
| gfmaxp | 32 KiB | 4,096 | 1.720572 / 3.478381 |
| gfu32max | 32 KiB | 8,192 | 0.705784 / 1.564342 |
| goldilocks | 32 KiB | 4,096 | 1.717063 / 3.848098 |
| goldilocks-residue | 32 KiB | 4,096 | 1.620947 / 3.462955 |
| goldilocks-squared | 32 KiB | 2,048 | 0.523076 / 1.034538 |
| m31 | 32 KiB | 8,192 | 1.310576 / 2.173578 |
| m31-residue | 32 KiB | 8,192 | 0.636118 / 1.450645 |
| maxprime-squared | 32 KiB | 2,048 | 0.762151 / 1.728001 |
| qm31 | 32 KiB | 4,096 | 9.634024 / 11.61003 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_gather`

</details>

<a id="steady-spread-gather-empty-sources-mul-add-matrix-gather-empty-sources-latency"></a>

### mul_add_matrix (gather empty sources)

**Elements:** 0 · **Sources:** 0 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 1.337 / 0.853 |
| m31 | 2.339 / 1.493 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_matrix`, `fgf::ops::mul_into_matrix`, `fgf::ops::mul_add_gather`

</details>

<a id="steady-spread-gather-offsets-add-gather-offsets-throughput"></a>

### add_gather_offsets

**Sources:** 8 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 32 KiB | 32,768 | 2.887 / 4.925 |
| gf125 | 32760 B | 10,920 | 0.399 / 0.979 |
| gf127 | 32 KiB | 32,768 | 2.778 / 4.570 |
| gf131 | 32 KiB | 32,768 | 0.584 / 1.136 |
| gf1e9p7 | 32 KiB | 8,192 | 2.518 / 4.259 |
| gf2p1-residue | 32 KiB | 4,096 | 9.386 / 20.17 |
| gf2p16-custom-cantor | 32 KiB | 16,384 | 52.04 / 90.06 |
| gf2p16-custom-normal | 32 KiB | 16,384 | 52.01 / 89.93 |
| gf2p16-custom-rijndael | 32 KiB | 16,384 | 51.46 / 89.27 |
| gf2p16-custom-rs | 32 KiB | 16,384 | 50.94 / 90.43 |
| gf2p16-fanpaar | 32 KiB | 16,384 | 51.78 / 90.56 |
| gf2p16-rijndael | 32 KiB | 16,384 | 52.24 / 89.53 |
| gf2p32-fanpaar | 32 KiB | 8,192 | 52.64 / 97.98 |
| gf2p32-rijndael | 32 KiB | 8,192 | 53.34 / 97.54 |
| gf2p64-fanpaar | 32 KiB | 4,096 | 54.92 / 100.3 |
| gf2p64-rijndael | 32 KiB | 4,096 | 55.07 / 100.7 |
| gf2p8-aes | 32 KiB | 32,768 | 53.44 / 89.46 |
| gf2p8-cantor | 32 KiB | 32,768 | 54.89 / 89.30 |
| gf2p8-fanpaar | 32 KiB | 32,768 | 54.10 / 90.60 |
| gf2p8-normal | 32 KiB | 32,768 | 52.72 / 90.80 |
| gf2p8-rs | 32 KiB | 32,768 | 54.07 / 89.61 |
| gf49 | 32 KiB | 16,384 | 0.203 / 0.438 |
| gf65521 | 32 KiB | 16,384 | 1.281 / 2.741 |
| gf65537 | 32 KiB | 8,192 | 4.149 / 6.946 |
| gf81 | 32 KiB | 8,192 | 0.174 / 0.275 |
| gf9 | 32 KiB | 16,384 | 0.391 / 0.958 |
| gfmaxp | 32 KiB | 4,096 | 5.195 / 10.79 |
| gfu32max | 32 KiB | 8,192 | 1.518 / 3.266 |
| goldilocks | 32 KiB | 4,096 | 36.14 / 21.63 |
| goldilocks-residue | 32 KiB | 4,096 | 5.046 / 10.75 |
| goldilocks-squared | 32 KiB | 2,048 | 2.036 / 3.897 |
| m31 | 32 KiB | 8,192 | 29.79 / 34.19 |
| m31-residue | 32 KiB | 8,192 | 3.321 / 5.552 |
| maxprime-squared | 32 KiB | 2,048 | 3.285 / 7.635 |
| qm31 | 32 KiB | 4,096 | 20.64 / 33.29 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::add_gather_offsets`

</details>

<a id="steady-spread-gather-prepared-mul-add-gather-with-throughput"></a>

### mul_add_gather_with

**Sources:** 8 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 32 KiB | 32,768 | 12.06883 / 11.96722 |
| gf125 | 32760 B | 10,920 | 0.065813 / 0.152532 |
| gf127 | 32 KiB | 32,768 | 12.03983 / 11.81050 |
| gf131 | 32 KiB | 32,768 | 0.470545 / 0.810552 |
| gf1e9p7 | 32 KiB | 8,192 | 7.825715 / 8.785774 |
| gf2p1-residue | 32 KiB | 4,096 | 8.240241 / 19.96611 |
| gf2p16-custom-cantor | 32 KiB | 16,384 | 0.279577 / 0.549272 |
| gf2p16-custom-normal | 32 KiB | 16,384 | 0.278045 / 0.538542 |
| gf2p16-custom-rijndael | 32 KiB | 16,384 | 0.279955 / 0.549595 |
| gf2p16-custom-rs | 32 KiB | 16,384 | 0.279753 / 0.546715 |
| gf2p16-fanpaar | 32 KiB | 16,384 | 9.417693 / 16.09943 |
| gf2p16-rijndael | 32 KiB | 16,384 | 38.36759 / 72.89037 |
| gf2p32-fanpaar | 32 KiB | 8,192 | 3.556413 / 5.728509 |
| gf2p32-rijndael | 32 KiB | 8,192 | 16.66902 / 30.35839 |
| gf2p64-fanpaar | 32 KiB | 4,096 | 0.946179 / 1.393330 |
| gf2p64-rijndael | 32 KiB | 4,096 | 8.767334 / 16.17781 |
| gf2p8-aes | 32 KiB | 32,768 | 64.44346 / 105.0644 |
| gf2p8-cantor | 32 KiB | 32,768 | 0.651547 / 1.266769 |
| gf2p8-fanpaar | 32 KiB | 32,768 | 0.000813 / 0.001833 |
| gf2p8-normal | 32 KiB | 32,768 | 0.650577 / 1.264578 |
| gf2p8-rs | 32 KiB | 32,768 | 67.99839 / 104.4624 |
| gf49 | 32 KiB | 16,384 | 0.056178 / 0.131909 |
| gf65521 | 32 KiB | 16,384 | 0.470276 / 0.872068 |
| gf65537 | 32 KiB | 8,192 | 7.828108 / 8.779291 |
| gf81 | 32 KiB | 8,192 | 0.060934 / 0.140885 |
| gf9 | 32 KiB | 16,384 | 0.127681 / 0.302075 |
| gfmaxp | 32 KiB | 4,096 | 5.142426 / 4.607905 |
| gfu32max | 32 KiB | 8,192 | 7.830657 / 8.770186 |
| goldilocks | 32 KiB | 4,096 | 8.808171 / 7.970018 |
| goldilocks-residue | 32 KiB | 4,096 | 5.264576 / 4.649028 |
| goldilocks-squared | 32 KiB | 2,048 | 0.522098 / 1.023921 |
| m31 | 32 KiB | 8,192 | 13.95011 / 17.27306 |
| m31-residue | 32 KiB | 8,192 | 7.990383 / 8.944305 |
| maxprime-squared | 32 KiB | 2,048 | 0.760430 / 1.726559 |
| qm31 | 32 KiB | 4,096 | 9.620339 / 11.99100 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_gather_with`, `fgf::ops::CoeffVec::new`, `fgf::ops::CoeffVec::as_ref`

</details>

<a id="steady-spread-gather-single-source-mul-add-scatter-gather-single-source-throughput"></a>

### mul_add_scatter (gather single source)

**Counted bytes:** 32 KiB · **Sources:** 1 · **Destinations:** 1

| Field | Elements | Throughput (GiB/s) |
| --- | --- | --- |
| gf2p8-aes | 32,768 | 721 / 935 |
| m31 | 8,192 | 153 / 193 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_scatter`, `fgf::ops::mul_add_gather`, `fgf::ops::mul_into_gather`

</details>

<a id="steady-spread-matrix-mul-add-matrix-throughput"></a>

### mul_add_matrix

**Sources:** 8 · **Destinations:** 8

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 0.174155 / 0.326396 |
| gf125 | 262080 B | 87,360 | 0.061588 / 0.133648 |
| gf127 | 256 KiB | 262,144 | 0.179783 / 0.322134 |
| gf131 | 256 KiB | 262,144 | 0.199091 / 0.371284 |
| gf1e9p7 | 256 KiB | 65,536 | 0.589003 / 1.335120 |
| gf2p1-residue | 256 KiB | 32,768 | 8.539969 / 19.98823 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.278766 / 0.550170 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.280750 / 0.547759 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.280384 / 0.562082 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.278364 / 0.549185 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 8.972954 / 15.44693 |
| gf2p16-rijndael | 256 KiB | 131,072 | 48.23913 / 56.43423 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 3.629676 / 5.825199 |
| gf2p32-rijndael | 256 KiB | 65,536 | 15.10596 / 31.09459 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 0.960669 / 1.401428 |
| gf2p64-rijndael | 256 KiB | 32,768 | 7.469158 / 13.55508 |
| gf2p8-aes | 256 KiB | 262,144 | 125.3516 / 145.5019 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.636946 / 1.400757 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000812 / 0.001835 |
| gf2p8-normal | 256 KiB | 262,144 | 0.632283 / 1.380080 |
| gf2p8-rs | 256 KiB | 262,144 | 110.1359 / 155.7455 |
| gf49 | 256 KiB | 131,072 | 0.056459 / 0.130542 |
| gf65521 | 256 KiB | 131,072 | 0.472766 / 0.851660 |
| gf65537 | 256 KiB | 65,536 | 0.842583 / 1.916110 |
| gf81 | 256 KiB | 65,536 | 0.056503 / 0.122940 |
| gf9 | 256 KiB | 131,072 | 0.125241 / 0.281332 |
| gfmaxp | 256 KiB | 32,768 | 1.863893 / 3.489300 |
| gfu32max | 256 KiB | 65,536 | 0.694178 / 1.570491 |
| goldilocks | 256 KiB | 32,768 | 0.810914 / 1.424735 |
| goldilocks-residue | 256 KiB | 32,768 | 1.687451 / 3.483381 |
| goldilocks-squared | 256 KiB | 16,384 | 0.520367 / 1.035073 |
| m31 | 256 KiB | 65,536 | 1.317636 / 2.175234 |
| m31-residue | 256 KiB | 65,536 | 0.653041 / 1.461978 |
| maxprime-squared | 256 KiB | 16,384 | 0.766008 / 1.733485 |
| qm31 | 256 KiB | 32,768 | 9.854115 / 11.65954 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_matrix`

</details>

<a id="steady-spread-matrix-at-mul-add-matrix-at-throughput"></a>

### mul_add_matrix_at

**Sources:** 8 · **Destinations:** 8

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 11.90712 / 11.87122 |
| gf125 | 262080 B | 87,360 | 0.061376 / 0.134286 |
| gf127 | 256 KiB | 262,144 | 11.90669 / 11.88184 |
| gf131 | 256 KiB | 262,144 | 0.463403 / 0.799381 |
| gf1e9p7 | 256 KiB | 65,536 | 7.775396 / 8.766152 |
| gf2p1-residue | 256 KiB | 32,768 | 8.981289 / 20.09186 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.279769 / 0.544734 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.279707 / 0.537674 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.281124 / 0.548870 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.281856 / 0.545912 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 8.655129 / 15.51096 |
| gf2p16-rijndael | 256 KiB | 131,072 | 26.09683 / 47.25684 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 3.627889 / 5.783010 |
| gf2p32-rijndael | 256 KiB | 65,536 | 14.38279 / 29.18946 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 0.964063 / 1.396441 |
| gf2p64-rijndael | 256 KiB | 32,768 | 7.424242 / 13.56874 |
| gf2p8-aes | 256 KiB | 262,144 | 104.5511 / 117.2181 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.636650 / 1.398396 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000813 / 0.001829 |
| gf2p8-normal | 256 KiB | 262,144 | 0.635901 / 1.399664 |
| gf2p8-rs | 256 KiB | 262,144 | 104.4282 / 126.6440 |
| gf49 | 256 KiB | 131,072 | 0.056360 / 0.131976 |
| gf65521 | 256 KiB | 131,072 | 0.758283 / 1.839511 |
| gf65537 | 256 KiB | 65,536 | 7.814532 / 8.794137 |
| gf81 | 256 KiB | 65,536 | 0.056572 / 0.122967 |
| gf9 | 256 KiB | 131,072 | 0.128621 / 0.294321 |
| gfmaxp | 256 KiB | 32,768 | 5.066485 / 4.642350 |
| gfu32max | 256 KiB | 65,536 | 7.788014 / 8.766782 |
| goldilocks | 256 KiB | 32,768 | 8.490703 / 7.854411 |
| goldilocks-residue | 256 KiB | 32,768 | 5.068642 / 4.676005 |
| goldilocks-squared | 256 KiB | 16,384 | 0.525635 / 1.042500 |
| m31 | 256 KiB | 65,536 | 13.83076 / 17.26672 |
| m31-residue | 256 KiB | 65,536 | 7.752711 / 8.763320 |
| maxprime-squared | 256 KiB | 16,384 | 0.765474 / 1.697431 |
| qm31 | 256 KiB | 32,768 | 9.577241 / 11.61565 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_matrix_at`

</details>

<a id="steady-spread-matrix-empty-terms-mul-add-matrix-matrix-empty-terms-latency"></a>

### mul_add_matrix (matrix empty terms)

**Elements:** 0 · **Sources:** 0 · **Destinations:** 8

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 1.77 / 1.07 |
| m31 | 2.00 / 1.28 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_matrix`, `fgf::ops::mul_into_matrix`, `fgf::ops::mul_add_gather`

</details>

<a id="steady-spread-matrix-prepared-mul-add-matrix-with-throughput"></a>

### mul_add_matrix_with

**Sources:** 8 · **Destinations:** 8

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 12.39481 / 12.00195 |
| gf125 | 262080 B | 87,360 | 0.061903 / 0.134873 |
| gf127 | 256 KiB | 262,144 | 12.36006 / 12.00785 |
| gf131 | 256 KiB | 262,144 | 0.469923 / 0.808605 |
| gf1e9p7 | 256 KiB | 65,536 | 7.983539 / 8.964553 |
| gf2p1-residue | 256 KiB | 32,768 | 8.557930 / 20.14971 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.279692 / 0.543207 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.279017 / 0.534722 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.281595 / 0.547445 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.279484 / 0.538389 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 9.344693 / 16.25775 |
| gf2p16-rijndael | 256 KiB | 131,072 | 48.11032 / 56.65802 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 3.628948 / 5.825807 |
| gf2p32-rijndael | 256 KiB | 65,536 | 16.18943 / 32.29605 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 0.960923 / 1.402802 |
| gf2p64-rijndael | 256 KiB | 32,768 | 9.011456 / 16.12421 |
| gf2p8-aes | 256 KiB | 262,144 | 125.7389 / 142.9439 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.635719 / 1.395124 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000812 / 0.001836 |
| gf2p8-normal | 256 KiB | 262,144 | 0.635164 / 1.393186 |
| gf2p8-rs | 256 KiB | 262,144 | 109.8560 / 155.0853 |
| gf49 | 256 KiB | 131,072 | 0.055511 / 0.129601 |
| gf65521 | 256 KiB | 131,072 | 0.758573 / 1.843388 |
| gf65537 | 256 KiB | 65,536 | 7.985661 / 8.968298 |
| gf81 | 256 KiB | 65,536 | 0.056685 / 0.122503 |
| gf9 | 256 KiB | 131,072 | 0.123678 / 0.284029 |
| gfmaxp | 256 KiB | 32,768 | 5.239477 / 4.672347 |
| gfu32max | 256 KiB | 65,536 | 7.991543 / 8.971058 |
| goldilocks | 256 KiB | 32,768 | 8.559280 / 7.878586 |
| goldilocks-residue | 256 KiB | 32,768 | 5.238577 / 4.669275 |
| goldilocks-squared | 256 KiB | 16,384 | 0.527202 / 1.041458 |
| m31 | 256 KiB | 65,536 | 14.15452 / 17.53002 |
| m31-residue | 256 KiB | 65,536 | 7.851191 / 8.797662 |
| maxprime-squared | 256 KiB | 16,384 | 0.765702 / 1.729658 |
| qm31 | 256 KiB | 32,768 | 9.836992 / 11.72260 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_matrix_with`, `fgf::ops::CoeffMatrix::from_source_major`

</details>

<a id="steady-spread-scatter-mul-add-scatter-throughput"></a>

### mul_add_scatter

**Sources:** 1 · **Destinations:** 8

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 32 KiB | 32,768 | 0.174030 / 0.318701 |
| gf125 | 32760 B | 10,920 | 0.067487 / 0.163376 |
| gf127 | 32 KiB | 32,768 | 0.180337 / 0.326478 |
| gf131 | 32 KiB | 32,768 | 0.197706 / 0.368405 |
| gf1e9p7 | 32 KiB | 8,192 | 0.591207 / 1.332580 |
| gf2p1-residue | 32 KiB | 4,096 | 7.393039 / 20.09515 |
| gf2p16-custom-cantor | 32 KiB | 16,384 | 0.280155 / 0.541871 |
| gf2p16-custom-normal | 32 KiB | 16,384 | 0.277805 / 0.542470 |
| gf2p16-custom-rijndael | 32 KiB | 16,384 | 0.276291 / 0.543202 |
| gf2p16-custom-rs | 32 KiB | 16,384 | 0.280800 / 0.546894 |
| gf2p16-fanpaar | 32 KiB | 16,384 | 9.152436 / 15.69581 |
| gf2p16-rijndael | 32 KiB | 16,384 | 30.66104 / 66.68796 |
| gf2p32-fanpaar | 32 KiB | 8,192 | 3.577362 / 5.616641 |
| gf2p32-rijndael | 32 KiB | 8,192 | 14.93755 / 30.98604 |
| gf2p64-fanpaar | 32 KiB | 4,096 | 0.961721 / 1.385211 |
| gf2p64-rijndael | 32 KiB | 4,096 | 7.286402 / 13.65331 |
| gf2p8-aes | 32 KiB | 32,768 | 73.80901 / 111.9505 |
| gf2p8-cantor | 32 KiB | 32,768 | 0.644308 / 1.417502 |
| gf2p8-fanpaar | 32 KiB | 32,768 | 0.000814 / 0.001830 |
| gf2p8-normal | 32 KiB | 32,768 | 0.641483 / 1.406121 |
| gf2p8-rs | 32 KiB | 32,768 | 73.21766 / 108.4271 |
| gf49 | 32 KiB | 16,384 | 0.059262 / 0.138631 |
| gf65521 | 32 KiB | 16,384 | 0.462298 / 0.868751 |
| gf65537 | 32 KiB | 8,192 | 0.832188 / 1.858264 |
| gf81 | 32 KiB | 8,192 | 0.063135 / 0.149025 |
| gf9 | 32 KiB | 16,384 | 0.126190 / 0.297288 |
| gfmaxp | 32 KiB | 4,096 | 1.746119 / 3.490749 |
| gfu32max | 32 KiB | 8,192 | 0.712341 / 1.559326 |
| goldilocks | 32 KiB | 4,096 | 1.678877 / 3.857217 |
| goldilocks-residue | 32 KiB | 4,096 | 1.673973 / 3.457462 |
| goldilocks-squared | 32 KiB | 2,048 | 0.534024 / 1.045772 |
| m31 | 32 KiB | 8,192 | 1.275818 / 2.168539 |
| m31-residue | 32 KiB | 8,192 | 0.641383 / 1.462822 |
| maxprime-squared | 32 KiB | 2,048 | 0.768037 / 1.729556 |
| qm31 | 32 KiB | 4,096 | 9.821954 / 11.55619 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_scatter`

</details>

<a id="steady-spread-scatter-prepared-mul-add-scatter-with-throughput"></a>

### mul_add_scatter_with

**Sources:** 1 · **Destinations:** 8

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 32 KiB | 32,768 | 12.46894 / 11.94727 |
| gf125 | 32760 B | 10,920 | 0.069630 / 0.169228 |
| gf127 | 32 KiB | 32,768 | 12.44203 / 11.96874 |
| gf131 | 32 KiB | 32,768 | 0.464445 / 0.794519 |
| gf1e9p7 | 32 KiB | 8,192 | 7.992154 / 8.740209 |
| gf2p1-residue | 32 KiB | 4,096 | 7.446452 / 20.22300 |
| gf2p16-custom-cantor | 32 KiB | 16,384 | 0.279638 / 0.546663 |
| gf2p16-custom-normal | 32 KiB | 16,384 | 0.279135 / 0.544711 |
| gf2p16-custom-rijndael | 32 KiB | 16,384 | 0.279720 / 0.553394 |
| gf2p16-custom-rs | 32 KiB | 16,384 | 0.279375 / 0.548650 |
| gf2p16-fanpaar | 32 KiB | 16,384 | 9.528291 / 16.27219 |
| gf2p16-rijndael | 32 KiB | 16,384 | 30.62088 / 66.22377 |
| gf2p32-fanpaar | 32 KiB | 8,192 | 3.580930 / 5.620725 |
| gf2p32-rijndael | 32 KiB | 8,192 | 16.15106 / 33.10719 |
| gf2p64-fanpaar | 32 KiB | 4,096 | 0.962123 / 1.383421 |
| gf2p64-rijndael | 32 KiB | 4,096 | 8.749754 / 16.18273 |
| gf2p8-aes | 32 KiB | 32,768 | 73.52916 / 111.8423 |
| gf2p8-cantor | 32 KiB | 32,768 | 0.631786 / 1.389594 |
| gf2p8-fanpaar | 32 KiB | 32,768 | 0.000814 / 0.001832 |
| gf2p8-normal | 32 KiB | 32,768 | 0.644126 / 1.399592 |
| gf2p8-rs | 32 KiB | 32,768 | 70.98237 / 107.4195 |
| gf49 | 32 KiB | 16,384 | 0.055243 / 0.129263 |
| gf65521 | 32 KiB | 16,384 | 0.470031 / 0.872492 |
| gf65537 | 32 KiB | 8,192 | 8.000333 / 8.742658 |
| gf81 | 32 KiB | 8,192 | 0.064084 / 0.150541 |
| gf9 | 32 KiB | 16,384 | 0.127470 / 0.296195 |
| gfmaxp | 32 KiB | 4,096 | 5.337366 / 4.573978 |
| gfu32max | 32 KiB | 8,192 | 7.989176 / 8.740045 |
| goldilocks | 32 KiB | 4,096 | 8.938951 / 8.038799 |
| goldilocks-residue | 32 KiB | 4,096 | 5.373654 / 4.669289 |
| goldilocks-squared | 32 KiB | 2,048 | 0.534145 / 1.048183 |
| m31 | 32 KiB | 8,192 | 14.29227 / 16.95164 |
| m31-residue | 32 KiB | 8,192 | 7.985598 / 8.746798 |
| maxprime-squared | 32 KiB | 2,048 | 0.768564 / 1.728637 |
| qm31 | 32 KiB | 4,096 | 9.802089 / 11.50647 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_scatter_with`, `fgf::ops::CoeffVec::new`, `fgf::ops::CoeffVec::as_ref`

</details>

<a id="steady-spread-scatter-zero-coeffs-mul-add-scatter-scatter-zero-coeffs-throughput"></a>

### mul_add_scatter (scatter zero coeffs)

**Counted bytes:** 32 KiB · **Sources:** 1 · **Destinations:** 8

| Field | Elements | Throughput (GiB/s) |
| --- | --- | --- |
| gf2p8-aes | 32,768 | 8057.7 / 14310 |
| m31 | 8,192 | 7941.0 / 13010 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_scatter`, `fgf::ops::mul_add_gather`, `fgf::ops::mul_into_gather`

</details>

## Query and scalar latency

Black-boxed scalar and getter calls; these report per-call latency, never a throughput.

<a id="query-details-maps-coeff-value-coeff-value-latency"></a>

### coeff_value

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 0.668 / 0.426 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::Coeff::value`, `fgf::ops::Coeff::new`

</details>

<a id="query-details-maps-coeffvecref-get-coeffvecref-get-latency"></a>

### coeffvecref_get

**Elements:** 8 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 1.424 / 0.853 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVecRef::get`, `fgf::ops::CoeffVec::new`, `fgf::ops::CoeffVec::as_ref`

</details>

<a id="query-details-maps-coeffvecref-is-empty-coeffvecref-is-empty-latency"></a>

### coeffvecref_is_empty

**Elements:** 8 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 0.880 / 0.533 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVecRef::is_empty`, `fgf::ops::CoeffVec::new`, `fgf::ops::CoeffVec::as_ref`

</details>

<a id="query-details-maps-coeffvecref-len-coeffvecref-len-latency"></a>

### coeffvecref_len

**Elements:** 8 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 0.985 / 0.589 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVecRef::len`, `fgf::ops::CoeffVec::new`, `fgf::ops::CoeffVec::as_ref`

</details>

<a id="query-details-maps-has-vector-elementwise-has-vector-elementwise-latency"></a>

### has_vector_elementwise

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| qm31 | 1.003 / 0.640 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::kernel::FieldKernels::has_vector_elementwise`

</details>

<a id="query-details-maps-relation-to-coordinates-relation-to-coordinates-latency"></a>

### relation_to_coordinates

**Elements:** 2 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf49 | 7.85 / 4.08 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Relation::to_coordinates`

</details>

<a id="query-details-maps-vector-elementwise-min-bytes-vector-elementwise-min-bytes-latency"></a>

### vector_elementwise_min_bytes

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| qm31 | 0.501 / 0.320 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::kernel::vector_elementwise_min_bytes`

</details>

## Preparation and setup

Construction and preparation costs excluded from steady-state panels; owned results are destroyed before the next iteration.

<a id="setup-details-maps-relation-from-binary-words-relation-from-binary-words-latency"></a>

### relation_from_binary_words

**Elements:** 16 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-custom-rs | 1.671 / 0.853 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Relation::from_binary_words`

</details>

<a id="setup-details-maps-relation-from-coordinates-relation-from-coordinates-latency"></a>

### relation_from_coordinates

**Elements:** 2 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf49 | 0.633 / 0.380 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Relation::from_coordinates`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
