# Single-row multiplication

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [add_assign (empty buffers)](#steady-bulk-empty-buffers-add-assign-empty-buffers-latency)
  - [mul_add](#steady-bulk-mul-add-mul-add-throughput)
  - [mul_add (unit add)](#steady-bulk-mul-add-mul-add-unit-add-throughput)
  - [mul_add_with](#steady-bulk-mul-add-prepared-mul-add-with-throughput)
  - [mul_add_with (unit add)](#steady-bulk-mul-add-prepared-mul-add-with-unit-add-throughput)
  - [mul_add (nonresident buffers)](#steady-bulk-mul-add-nonresident-mul-add-nonresident-buffers-throughput)
  - [mul_add (short buffers)](#steady-bulk-mul-add-short-mul-add-short-buffers-throughput)
  - [mul_assign](#steady-bulk-mul-assign-mul-assign-throughput)
  - [mul_assign (unit no-op)](#steady-bulk-mul-assign-mul-assign-unit-no-op-latency)
  - [mul_assign_with](#steady-bulk-mul-assign-prepared-mul-assign-with-throughput)
  - [mul_assign_with (unit no-op)](#steady-bulk-mul-assign-prepared-mul-assign-with-unit-no-op-latency)
  - [mul_into](#steady-bulk-mul-into-mul-into-throughput)
  - [mul_into (unit copy)](#steady-bulk-mul-into-mul-into-unit-copy-throughput)
  - [mul_into_with](#steady-bulk-mul-into-prepared-mul-into-with-throughput)
  - [mul_into_with (unit copy)](#steady-bulk-mul-into-prepared-mul-into-with-unit-copy-throughput)
  - [mul_into (nonresident buffers)](#steady-bulk-mul-into-nonresident-mul-into-nonresident-buffers-throughput)
  - [mul_into (short buffers)](#steady-bulk-mul-into-short-mul-into-short-buffers-throughput)
  - [mul_add (unit coeff)](#steady-bulk-unit-coeff-mul-add-unit-coeff-throughput)
  - [mul_add (zero no-op)](#steady-bulk-zero-coeff-mul-add-zero-no-op-latency)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-bulk-empty-buffers-add-assign-empty-buffers-latency"></a>

### add_assign (empty buffers)

**Elements:** 0 · **Sources:** 0 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael | 20.1 / 10.2 |
| gf2p8-aes | 15.5 / 7.68 |
| m31 | 22.5 / 12.6 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::add_assign`, `fgf::ops::mul_add`, `fgf::ops::mul_into`, `fgf::ops::mul_assign`

</details>

<a id="steady-bulk-mul-add-mul-add-throughput"></a>

### mul_add

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 11.77298 / 13.46073 |
| gf125 | 262128 B | 87,376 | 0.067609 / 0.149972 |
| gf127 | 256 KiB | 262,144 | 11.76900 / 13.45480 |
| gf131 | 256 KiB | 262,144 | 0.463633 / 0.799395 |
| gf1e9p7 | 256 KiB | 65,536 | 7.842112 / 8.960933 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.277652 / 0.538334 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.275910 / 0.527561 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.277677 / 0.542762 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.277674 / 0.534321 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 10.72285 / 16.67627 |
| gf2p16-rijndael | 256 KiB | 131,072 | 33.96739 / 47.91769 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 3.941034 / 6.210116 |
| gf2p32-rijndael | 256 KiB | 65,536 | 18.47521 / 34.18858 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 0.966832 / 1.457490 |
| gf2p64-rijndael | 256 KiB | 32,768 | 10.36223 / 16.32270 |
| gf2p8-aes | 256 KiB | 262,144 | 38.57987 / 56.17163 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.634730 / 1.400180 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000814 / 0.001830 |
| gf2p8-normal | 256 KiB | 262,144 | 0.633579 / 1.393907 |
| gf2p8-rs | 256 KiB | 262,144 | 38.47954 / 55.90858 |
| gf49 | 256 KiB | 131,072 | 0.054657 / 0.126698 |
| gf65521 | 256 KiB | 131,072 | 0.470063 / 0.872759 |
| gf65537 | 256 KiB | 65,536 | 7.841325 / 8.959453 |
| gf81 | 256 KiB | 65,536 | 0.052922 / 0.113176 |
| gf9 | 256 KiB | 131,072 | 0.107641 / 0.224652 |
| gfmaxp | 256 KiB | 32,768 | 5.037995 / 4.553960 |
| gfu32max | 256 KiB | 65,536 | 7.842584 / 8.930694 |
| goldilocks | 256 KiB | 32,768 | 9.507080 / 7.929443 |
| goldilocks-residue | 256 KiB | 32,768 | 5.701772 / 4.751909 |
| goldilocks-squared | 256 KiB | 16,384 | 0.517196 / 1.024458 |
| m31 | 256 KiB | 65,536 | 13.98005 / 17.89953 |
| m31-residue | 256 KiB | 65,536 | 7.842080 / 8.943699 |
| maxprime-squared | 256 KiB | 16,384 | 0.755144 / 1.735291 |
| qm31 | 256 KiB | 32,768 | 10.12517 / 12.20749 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add`

</details>

<a id="steady-bulk-mul-add-mul-add-unit-add-throughput"></a>

### mul_add (unit add)

**Counted bytes:** 256 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2p1-residue | 9.09 / 20.9 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add`

</details>

<a id="steady-bulk-mul-add-prepared-mul-add-with-throughput"></a>

### mul_add_with

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 11.77184 / 13.58696 |
| gf125 | 262128 B | 87,376 | 0.061827 / 0.136922 |
| gf127 | 256 KiB | 262,144 | 11.76418 / 13.46936 |
| gf131 | 256 KiB | 262,144 | 0.458295 / 0.811406 |
| gf1e9p7 | 256 KiB | 65,536 | 7.842710 / 8.969946 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.277677 / 0.535403 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.275765 / 0.525899 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.277932 / 0.534330 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.277669 / 0.533081 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 10.72975 / 16.70080 |
| gf2p16-rijndael | 256 KiB | 131,072 | 33.96570 / 47.95769 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 3.934183 / 6.200023 |
| gf2p32-rijndael | 256 KiB | 65,536 | 18.81641 / 33.16677 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 0.991995 / 1.457977 |
| gf2p64-rijndael | 256 KiB | 32,768 | 10.40629 / 16.42206 |
| gf2p8-aes | 256 KiB | 262,144 | 38.58825 / 56.16624 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.634570 / 1.396045 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000813 / 0.001835 |
| gf2p8-normal | 256 KiB | 262,144 | 0.634366 / 1.394873 |
| gf2p8-rs | 256 KiB | 262,144 | 38.48371 / 55.95371 |
| gf49 | 256 KiB | 131,072 | 0.054828 / 0.130092 |
| gf65521 | 256 KiB | 131,072 | 0.470396 / 0.862078 |
| gf65537 | 256 KiB | 65,536 | 7.842238 / 8.977697 |
| gf81 | 256 KiB | 65,536 | 0.054926 / 0.118465 |
| gf9 | 256 KiB | 131,072 | 0.103875 / 0.221733 |
| gfmaxp | 256 KiB | 32,768 | 5.590660 / 4.552187 |
| gfu32max | 256 KiB | 65,536 | 7.841828 / 8.964141 |
| goldilocks | 256 KiB | 32,768 | 9.498988 / 7.929218 |
| goldilocks-residue | 256 KiB | 32,768 | 5.700990 / 4.751851 |
| goldilocks-squared | 256 KiB | 16,384 | 0.517540 / 1.024294 |
| m31 | 256 KiB | 65,536 | 13.98025 / 17.89838 |
| m31-residue | 256 KiB | 65,536 | 7.841388 / 8.971182 |
| maxprime-squared | 256 KiB | 16,384 | 0.753082 / 1.735866 |
| qm31 | 256 KiB | 32,768 | 10.12559 / 12.18890 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_with`, `fgf::ops::Coeff::new`, `fgf::ops::PreparedCoefficient`

</details>

<a id="steady-bulk-mul-add-prepared-mul-add-with-unit-add-throughput"></a>

### mul_add_with (unit add)

**Counted bytes:** 256 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2p1-residue | 9.09 / 20.9 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add_with`, `fgf::ops::Coeff::new`, `fgf::ops::PreparedCoefficient`

</details>

<a id="steady-bulk-mul-add-nonresident-mul-add-nonresident-buffers-throughput"></a>

### mul_add (nonresident buffers)

**Counted bytes:** 128 MiB · **Sources:** 1 · **Destinations:** 1

| Field | Elements | Throughput (GiB/s) |
| --- | --- | --- |
| gf2p16-rijndael | 67,108,864 | 5.89 / 13.2 |
| gf2p8-aes | 134,217,728 | 5.88 / 13.3 |
| gf2p8-rs | 134,217,728 | 5.87 / 13.3 |
| gfmaxp | 16,777,216 | 4.68 / 4.30 |
| goldilocks | 16,777,216 | 5.76 / 7.23 |
| m31 | 33,554,432 | 5.86 / 12.6 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add`

</details>

<a id="steady-bulk-mul-add-short-mul-add-short-buffers-throughput"></a>

### mul_add (short buffers)

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf2p16-rijndael | 64 B | 32 | 3.85 / 5.71 |
| gf2p8-aes | 64 B | 64 | 4.83 / 13.3 |
| gf2p8-rs | 64 B | 64 | 4.61 / 12.7 |
| gfmaxp | 128 B | 16 | 3.92 / 3.49 |
| goldilocks | 128 B | 16 | 6.27 / 7.08 |
| m31 | 64 B | 16 | 5.68 / 9.29 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add`

</details>

<a id="steady-bulk-mul-assign-mul-assign-throughput"></a>

### mul_assign

**Sources:** 0 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 42.32388 / 51.89758 |
| gf125 | 262128 B | 87,376 | 0.064832 / 0.143128 |
| gf127 | 256 KiB | 262,144 | 42.32388 / 51.89513 |
| gf131 | 256 KiB | 262,144 | 1.591539 / 2.633852 |
| gf1e9p7 | 256 KiB | 65,536 | 17.30021 / 20.19657 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.281056 / 0.554377 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.281473 / 0.558269 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.282592 / 0.566083 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.282629 / 0.565723 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 11.45158 / 17.34693 |
| gf2p16-rijndael | 256 KiB | 131,072 | 48.41946 / 73.78079 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 4.036041 / 6.256567 |
| gf2p32-rijndael | 256 KiB | 65,536 | 23.36207 / 37.43914 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 1.037201 / 1.454805 |
| gf2p64-rijndael | 256 KiB | 32,768 | 10.85480 / 16.94215 |
| gf2p8-aes | 256 KiB | 262,144 | 51.47285 / 73.48432 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.700796 / 1.527480 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000741 / 0.001832 |
| gf2p8-normal | 256 KiB | 262,144 | 0.710587 / 1.380428 |
| gf2p8-rs | 256 KiB | 262,144 | 51.47750 / 52.59206 |
| gf49 | 256 KiB | 131,072 | 0.073359 / 0.166548 |
| gf65521 | 256 KiB | 131,072 | 1.663365 / 2.911531 |
| gf65537 | 256 KiB | 65,536 | 17.30573 / 20.29582 |
| gf81 | 256 KiB | 65,536 | 0.066541 / 0.144526 |
| gf9 | 256 KiB | 131,072 | 0.253719 / 0.625054 |
| gfmaxp | 256 KiB | 32,768 | 6.841979 / 5.662052 |
| gfu32max | 256 KiB | 65,536 | 17.30297 / 20.13180 |
| goldilocks | 256 KiB | 32,768 | 11.86618 / 11.42920 |
| goldilocks-residue | 256 KiB | 32,768 | 6.842171 / 5.572889 |
| goldilocks-squared | 256 KiB | 16,384 | 0.675342 / 1.253076 |
| m31 | 256 KiB | 65,536 | 19.64985 / 26.10597 |
| m31-residue | 256 KiB | 65,536 | 17.30067 / 20.29544 |
| maxprime-squared | 256 KiB | 16,384 | 0.997108 / 1.746995 |
| qm31 | 256 KiB | 32,768 | 9.677943 / 16.06717 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign`

</details>

<a id="steady-bulk-mul-assign-mul-assign-unit-no-op-latency"></a>

### mul_assign (unit no-op)

**Elements:** 32,768 · **Sources:** 0 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p1-residue | 1.337 / 0.853 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign`

</details>

<a id="steady-bulk-mul-assign-prepared-mul-assign-with-throughput"></a>

### mul_assign_with

**Sources:** 0 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 42.32878 / 51.89942 |
| gf125 | 262128 B | 87,376 | 0.066576 / 0.142274 |
| gf127 | 256 KiB | 262,144 | 42.31247 / 51.90003 |
| gf131 | 256 KiB | 262,144 | 1.592019 / 2.633692 |
| gf1e9p7 | 256 KiB | 65,536 | 17.30021 / 20.13328 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.281151 / 0.555780 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.281585 / 0.561240 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.282593 / 0.569949 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.282277 / 0.566153 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 11.45998 / 17.32907 |
| gf2p16-rijndael | 256 KiB | 131,072 | 48.57022 / 73.79231 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 4.026946 / 6.239478 |
| gf2p32-rijndael | 256 KiB | 65,536 | 23.40799 / 37.70678 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 1.037115 / 1.455929 |
| gf2p64-rijndael | 256 KiB | 32,768 | 10.89853 / 16.99330 |
| gf2p8-aes | 256 KiB | 262,144 | 51.48965 / 73.50755 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.697677 / 1.527744 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000809 / 0.001836 |
| gf2p8-normal | 256 KiB | 262,144 | 0.700192 / 1.507087 |
| gf2p8-rs | 256 KiB | 262,144 | 51.48887 / 52.59178 |
| gf49 | 256 KiB | 131,072 | 0.073351 / 0.166271 |
| gf65521 | 256 KiB | 131,072 | 1.670879 / 2.911505 |
| gf65537 | 256 KiB | 65,536 | 17.30175 / 20.13346 |
| gf81 | 256 KiB | 65,536 | 0.063764 / 0.133299 |
| gf9 | 256 KiB | 131,072 | 0.147441 / 0.295868 |
| gfmaxp | 256 KiB | 32,768 | 6.842386 / 5.662069 |
| gfu32max | 256 KiB | 65,536 | 17.30466 / 20.13291 |
| goldilocks | 256 KiB | 32,768 | 11.86610 / 11.42880 |
| goldilocks-residue | 256 KiB | 32,768 | 6.841428 / 5.572985 |
| goldilocks-squared | 256 KiB | 16,384 | 0.674624 / 1.255536 |
| m31 | 256 KiB | 65,536 | 19.66691 / 26.10165 |
| m31-residue | 256 KiB | 65,536 | 17.30021 / 20.13217 |
| maxprime-squared | 256 KiB | 16,384 | 0.989325 / 1.746989 |
| qm31 | 256 KiB | 32,768 | 9.678566 / 16.24693 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign_with`, `fgf::ops::Coeff::new`

</details>

<a id="steady-bulk-mul-assign-prepared-mul-assign-with-unit-no-op-latency"></a>

### mul_assign_with (unit no-op)

**Elements:** 32,768 · **Sources:** 0 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p1-residue | 1.337 / 0.853 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_assign_with`, `fgf::ops::Coeff::new`

</details>

<a id="steady-bulk-mul-into-mul-into-throughput"></a>

### mul_into

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 36.04296 / 50.52333 |
| gf125 | 262128 B | 87,376 | 0.073192 / 0.164578 |
| gf127 | 256 KiB | 262,144 | 36.06391 / 50.50581 |
| gf131 | 256 KiB | 262,144 | 1.588247 / 2.633039 |
| gf1e9p7 | 256 KiB | 65,536 | 17.28750 / 19.91179 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.278901 / 0.556685 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.279402 / 0.559206 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.280448 / 0.567619 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.279990 / 0.568160 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 11.29274 / 17.38259 |
| gf2p16-rijndael | 256 KiB | 131,072 | 38.51482 / 60.77322 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 3.998553 / 6.283943 |
| gf2p32-rijndael | 256 KiB | 65,536 | 21.30868 / 37.45104 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 0.983586 / 1.458470 |
| gf2p64-rijndael | 256 KiB | 32,768 | 10.80908 / 16.91896 |
| gf2p8-aes | 256 KiB | 262,144 | 38.18874 / 62.17463 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.681610 / 1.481769 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000810 / 0.001833 |
| gf2p8-normal | 256 KiB | 262,144 | 0.686638 / 1.481810 |
| gf2p8-rs | 256 KiB | 262,144 | 38.24071 / 62.15772 |
| gf49 | 256 KiB | 131,072 | 0.073555 / 0.164450 |
| gf65521 | 256 KiB | 131,072 | 1.739510 / 2.911431 |
| gf65537 | 256 KiB | 65,536 | 17.28720 / 19.89808 |
| gf81 | 256 KiB | 65,536 | 0.071178 / 0.150443 |
| gf9 | 256 KiB | 131,072 | 0.149209 / 0.289247 |
| gfmaxp | 256 KiB | 32,768 | 6.830828 / 5.572969 |
| gfu32max | 256 KiB | 65,536 | 17.28659 / 19.87774 |
| goldilocks | 256 KiB | 32,768 | 11.99171 / 11.06166 |
| goldilocks-residue | 256 KiB | 32,768 | 6.830494 / 5.656608 |
| goldilocks-squared | 256 KiB | 16,384 | 0.673289 / 1.261890 |
| m31 | 256 KiB | 65,536 | 19.54024 / 25.26480 |
| m31-residue | 256 KiB | 65,536 | 17.28827 / 19.91359 |
| maxprime-squared | 256 KiB | 16,384 | 0.925439 / 1.746997 |
| qm31 | 256 KiB | 32,768 | 13.68866 / 16.51691 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_into`

</details>

<a id="steady-bulk-mul-into-mul-into-unit-copy-throughput"></a>

### mul_into (unit copy)

**Counted bytes:** 256 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2p1-residue | 36.9 / 60.3 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_into`

</details>

<a id="steady-bulk-mul-into-prepared-mul-into-with-throughput"></a>

### mul_into_with

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 36.05716 / 50.51335 |
| gf125 | 262128 B | 87,376 | 0.065954 / 0.145830 |
| gf127 | 256 KiB | 262,144 | 36.05113 / 50.50692 |
| gf131 | 256 KiB | 262,144 | 1.585100 / 2.702532 |
| gf1e9p7 | 256 KiB | 65,536 | 17.28643 / 19.91666 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.278848 / 0.557341 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.279276 / 0.560596 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.280517 / 0.567832 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.280382 / 0.566612 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 11.30228 / 17.39854 |
| gf2p16-rijndael | 256 KiB | 131,072 | 38.57567 / 60.31725 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 3.997931 / 6.285683 |
| gf2p32-rijndael | 256 KiB | 65,536 | 21.34635 / 37.54782 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 0.984332 / 1.458588 |
| gf2p64-rijndael | 256 KiB | 32,768 | 10.82094 / 17.00780 |
| gf2p8-aes | 256 KiB | 262,144 | 38.15852 / 62.16671 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.686494 / 1.483029 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000812 / 0.001834 |
| gf2p8-normal | 256 KiB | 262,144 | 0.637208 / 1.365216 |
| gf2p8-rs | 256 KiB | 262,144 | 38.23734 / 62.16975 |
| gf49 | 256 KiB | 131,072 | 0.073618 / 0.164539 |
| gf65521 | 256 KiB | 131,072 | 1.739626 / 2.911414 |
| gf65537 | 256 KiB | 65,536 | 17.28674 / 19.85816 |
| gf81 | 256 KiB | 65,536 | 0.060053 / 0.126108 |
| gf9 | 256 KiB | 131,072 | 0.149068 / 0.297836 |
| gfmaxp | 256 KiB | 32,768 | 6.830494 / 5.569155 |
| gfu32max | 256 KiB | 65,536 | 17.28659 / 19.91070 |
| goldilocks | 256 KiB | 32,768 | 11.99200 / 11.02732 |
| goldilocks-residue | 256 KiB | 32,768 | 6.830661 / 5.656985 |
| goldilocks-squared | 256 KiB | 16,384 | 0.673006 / 1.261934 |
| m31 | 256 KiB | 65,536 | 19.54405 / 25.26313 |
| m31-residue | 256 KiB | 65,536 | 17.28689 / 19.91251 |
| maxprime-squared | 256 KiB | 16,384 | 0.934965 / 1.746942 |
| qm31 | 256 KiB | 32,768 | 13.68904 / 16.57283 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_into_with`, `fgf::ops::Coeff::new`

</details>

<a id="steady-bulk-mul-into-prepared-mul-into-with-unit-copy-throughput"></a>

### mul_into_with (unit copy)

**Counted bytes:** 256 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2p1-residue | 36.9 / 60.4 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_into_with`, `fgf::ops::Coeff::new`

</details>

<a id="steady-bulk-mul-into-nonresident-mul-into-nonresident-buffers-throughput"></a>

### mul_into (nonresident buffers)

**Counted bytes:** 128 MiB · **Sources:** 1 · **Destinations:** 1

| Field | Elements | Throughput (GiB/s) |
| --- | --- | --- |
| gf2p16-rijndael | 67,108,864 | 5.99 / 19.7 |
| gf2p8-aes | 134,217,728 | 8.36 / 19.9 |
| gf2p8-rs | 134,217,728 | 8.41 / 20.0 |
| gfmaxp | 16,777,216 | 5.62 / 5.24 |
| goldilocks | 16,777,216 | 5.84 / 9.71 |
| m31 | 33,554,432 | 5.93 / 13.0 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_into`

</details>

<a id="steady-bulk-mul-into-short-mul-into-short-buffers-throughput"></a>

### mul_into (short buffers)

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf2p16-rijndael | 64 B | 32 | 3.15 / 5.61 |
| gf2p8-aes | 64 B | 64 | 4.79 / 13.3 |
| gf2p8-rs | 64 B | 64 | 4.74 / 13.3 |
| gfmaxp | 128 B | 16 | 4.27 / 4.47 |
| goldilocks | 128 B | 16 | 7.80 / 9.47 |
| m31 | 64 B | 16 | 6.13 / 10.4 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_into`

</details>

<a id="steady-bulk-unit-coeff-mul-add-unit-coeff-throughput"></a>

### mul_add (unit coeff)

**Counted bytes:** 256 KiB · **Sources:** 1 · **Destinations:** 1

| Field | Elements | Throughput (GiB/s) |
| --- | --- | --- |
| gf2p16-rijndael | 131,072 | 33.9 / 61.3 |
| gf2p8-aes | 262,144 | 33.9 / 61.3 |
| m31 | 65,536 | 29.9 / 34.7 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add`, `fgf::ops::mul_into`, `fgf::ops::add_assign`

</details>

<a id="steady-bulk-zero-coeff-mul-add-zero-no-op-latency"></a>

### mul_add (zero no-op)

**Sources:** 1 · **Destinations:** 1

| Field | Elements | Latency (ns/op) |
| --- | --- | --- |
| gf2p16-rijndael | 131,072 | 3.34 / 2.13 |
| gf2p8-aes | 262,144 | 3.34 / 1.92 |
| m31 | 65,536 | 3.68 / 2.35 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_add`, `fgf::ops::mul_into`, `fgf::ops::add_assign`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
