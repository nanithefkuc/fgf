# Buffer addition and subtraction

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [add_assign](#steady-bulk-add-assign-add-assign-throughput)
  - [add_assign (nonresident buffers)](#steady-bulk-add-assign-nonresident-add-assign-nonresident-buffers-throughput)
  - [add_assign (short buffers)](#steady-bulk-add-assign-short-add-assign-short-buffers-throughput)
  - [add_assign_rows](#steady-bulk-add-rows-add-assign-rows-throughput)
  - [add_assign_scalar](#steady-bulk-add-scalar-add-assign-scalar-throughput)
  - [sub_assign](#steady-bulk-sub-assign-sub-assign-throughput)
  - [sub_assign_scalar](#steady-bulk-sub-scalar-sub-assign-scalar-throughput)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-bulk-add-assign-add-assign-throughput"></a>

### add_assign

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 2.911 / 4.993 |
| gf125 | 262128 B | 87,376 | 0.399 / 0.956 |
| gf127 | 256 KiB | 262,144 | 2.784 / 4.619 |
| gf131 | 256 KiB | 262,144 | 0.605 / 1.132 |
| gf1e9p7 | 256 KiB | 65,536 | 2.506 / 4.295 |
| gf2p1-residue | 256 KiB | 32,768 | 9.092 / 20.91 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 33.90 / 61.16 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 33.89 / 61.14 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 33.89 / 61.14 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 33.88 / 61.15 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 33.90 / 61.15 |
| gf2p16-rijndael | 256 KiB | 131,072 | 33.90 / 61.16 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 33.89 / 61.15 |
| gf2p32-rijndael | 256 KiB | 65,536 | 33.88 / 61.14 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 33.87 / 61.14 |
| gf2p64-rijndael | 256 KiB | 32,768 | 33.88 / 61.15 |
| gf2p8-aes | 256 KiB | 262,144 | 33.87 / 61.16 |
| gf2p8-cantor | 256 KiB | 262,144 | 33.79 / 61.16 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 33.90 / 61.17 |
| gf2p8-normal | 256 KiB | 262,144 | 33.88 / 61.16 |
| gf2p8-rs | 256 KiB | 262,144 | 33.88 / 61.16 |
| gf49 | 256 KiB | 131,072 | 0.202 / 0.437 |
| gf65521 | 256 KiB | 131,072 | 1.269 / 2.758 |
| gf65537 | 256 KiB | 65,536 | 4.171 / 7.019 |
| gf81 | 256 KiB | 65,536 | 0.221 / 0.362 |
| gf9 | 256 KiB | 131,072 | 0.390 / 0.930 |
| gfmaxp | 256 KiB | 32,768 | 5.066 / 11.02 |
| gfu32max | 256 KiB | 65,536 | 1.514 / 3.169 |
| goldilocks | 256 KiB | 32,768 | 31.76 / 21.79 |
| goldilocks-residue | 256 KiB | 32,768 | 4.916 / 11.00 |
| goldilocks-squared | 256 KiB | 16,384 | 2.025 / 3.951 |
| m31 | 256 KiB | 65,536 | 26.18 / 36.93 |
| m31-residue | 256 KiB | 65,536 | 3.322 / 5.688 |
| maxprime-squared | 256 KiB | 16,384 | 3.229 / 7.756 |
| qm31 | 256 KiB | 32,768 | 20.00 / 36.65 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::add_assign`

</details>

<a id="steady-bulk-add-assign-nonresident-add-assign-nonresident-buffers-throughput"></a>

### add_assign (nonresident buffers)

**Counted bytes:** 128 MiB · **Sources:** 1 · **Destinations:** 1

| Field | Elements | Throughput (GiB/s) |
| --- | --- | --- |
| gf2p16-rijndael | 67,108,864 | 5.90 / 13.5 |
| gf2p8-aes | 134,217,728 | 5.90 / 13.4 |
| gf2p8-rs | 134,217,728 | 5.90 / 13.4 |
| gfmaxp | 16,777,216 | 4.64 / 8.79 |
| goldilocks | 16,777,216 | 5.94 / 12.7 |
| m31 | 33,554,432 | 5.93 / 13.1 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::add_assign`

</details>

<a id="steady-bulk-add-assign-short-add-assign-short-buffers-throughput"></a>

### add_assign (short buffers)

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf2p16-rijndael | 64 B | 32 | 7.43 / 13.3 |
| gf2p8-aes | 64 B | 64 | 8.11 / 14.0 |
| gf2p8-rs | 64 B | 64 | 8.11 / 15.5 |
| gfmaxp | 128 B | 16 | 5.01 / 9.55 |
| goldilocks | 128 B | 16 | 12.6 / 14.2 |
| m31 | 64 B | 16 | 7.13 / 10.7 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::add_assign`

</details>

<a id="steady-bulk-add-rows-add-assign-rows-throughput"></a>

### add_assign_rows

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 2.910 / 4.990 |
| gf125 | 262128 B | 87,376 | 0.399 / 0.921 |
| gf127 | 256 KiB | 262,144 | 2.784 / 4.618 |
| gf131 | 256 KiB | 262,144 | 0.577 / 1.080 |
| gf1e9p7 | 256 KiB | 65,536 | 2.506 / 4.294 |
| gf2p1-residue | 256 KiB | 32,768 | 9.108 / 20.92 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 33.89 / 61.17 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 33.88 / 61.18 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 33.89 / 61.17 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 33.90 / 61.17 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 33.91 / 61.17 |
| gf2p16-rijndael | 256 KiB | 131,072 | 33.87 / 61.18 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 33.90 / 61.18 |
| gf2p32-rijndael | 256 KiB | 65,536 | 33.89 / 61.17 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 33.89 / 61.18 |
| gf2p64-rijndael | 256 KiB | 32,768 | 33.90 / 61.17 |
| gf2p8-aes | 256 KiB | 262,144 | 33.87 / 61.16 |
| gf2p8-cantor | 256 KiB | 262,144 | 33.79 / 61.17 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 33.91 / 61.14 |
| gf2p8-normal | 256 KiB | 262,144 | 33.89 / 61.16 |
| gf2p8-rs | 256 KiB | 262,144 | 33.88 / 61.16 |
| gf49 | 256 KiB | 131,072 | 0.203 / 0.451 |
| gf65521 | 256 KiB | 131,072 | 1.269 / 2.758 |
| gf65537 | 256 KiB | 65,536 | 4.170 / 7.249 |
| gf81 | 256 KiB | 65,536 | 0.216 / 0.353 |
| gf9 | 256 KiB | 131,072 | 0.391 / 0.935 |
| gfmaxp | 256 KiB | 32,768 | 5.062 / 10.48 |
| gfu32max | 256 KiB | 65,536 | 1.514 / 3.233 |
| goldilocks | 256 KiB | 32,768 | 31.78 / 21.75 |
| goldilocks-residue | 256 KiB | 32,768 | 4.931 / 11.01 |
| goldilocks-squared | 256 KiB | 16,384 | 2.017 / 3.951 |
| m31 | 256 KiB | 65,536 | 26.19 / 36.97 |
| m31-residue | 256 KiB | 65,536 | 3.322 / 5.684 |
| maxprime-squared | 256 KiB | 16,384 | 3.230 / 7.756 |
| qm31 | 256 KiB | 32,768 | 19.90 / 36.65 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::add_assign_rows`

</details>

<a id="steady-bulk-add-scalar-add-assign-scalar-throughput"></a>

### add_assign_scalar

**Sources:** 0 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 5.242 / 8.958 |
| gf125 | 262128 B | 87,376 | 0.443 / 1.064 |
| gf127 | 256 KiB | 262,144 | 5.324 / 8.676 |
| gf131 | 256 KiB | 262,144 | 0.670 / 1.325 |
| gf1e9p7 | 256 KiB | 65,536 | 4.291 / 7.194 |
| gf2p1-residue | 256 KiB | 32,768 | 7.991 / 17.45 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 51.84 / 64.84 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 51.83 / 64.81 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 51.84 / 64.78 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 51.83 / 64.83 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 51.73 / 64.75 |
| gf2p16-rijndael | 256 KiB | 131,072 | 51.81 / 64.76 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 51.85 / 64.89 |
| gf2p32-rijndael | 256 KiB | 65,536 | 51.64 / 64.62 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 51.83 / 64.74 |
| gf2p64-rijndael | 256 KiB | 32,768 | 50.86 / 63.87 |
| gf2p8-aes | 256 KiB | 262,144 | 51.85 / 64.95 |
| gf2p8-cantor | 256 KiB | 262,144 | 51.85 / 64.96 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 51.85 / 64.99 |
| gf2p8-normal | 256 KiB | 262,144 | 41.60 / 73.45 |
| gf2p8-rs | 256 KiB | 262,144 | 51.84 / 64.82 |
| gf49 | 256 KiB | 131,072 | 0.254 / 0.563 |
| gf65521 | 256 KiB | 131,072 | 1.589 / 3.441 |
| gf65537 | 256 KiB | 65,536 | 6.558 / 10.88 |
| gf81 | 256 KiB | 65,536 | 0.470 / 1.055 |
| gf9 | 256 KiB | 131,072 | 0.424 / 1.033 |
| gfmaxp | 256 KiB | 32,768 | 6.153 / 13.94 |
| gfu32max | 256 KiB | 65,536 | 2.228 / 4.701 |
| goldilocks | 256 KiB | 32,768 | 49.21 / 39.21 |
| goldilocks-residue | 256 KiB | 32,768 | 6.085 / 13.67 |
| goldilocks-squared | 256 KiB | 16,384 | 2.957 / 5.840 |
| m31 | 256 KiB | 65,536 | 44.77 / 59.38 |
| m31-residue | 256 KiB | 65,536 | 5.641 / 9.417 |
| maxprime-squared | 256 KiB | 16,384 | 4.331 / 10.47 |
| qm31 | 256 KiB | 32,768 | 34.03 / 50.80 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::add_assign_scalar`

</details>

<a id="steady-bulk-sub-assign-sub-assign-throughput"></a>

### sub_assign

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 2.827 / 4.835 |
| gf125 | 262128 B | 87,376 | 0.382 / 0.906 |
| gf127 | 256 KiB | 262,144 | 2.721 / 4.491 |
| gf131 | 256 KiB | 262,144 | 12.73 / 23.21 |
| gf1e9p7 | 256 KiB | 65,536 | 2.870 / 4.889 |
| gf2p1-residue | 256 KiB | 32,768 | 7.224 / 17.45 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 33.90 / 61.15 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 33.89 / 61.15 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 33.89 / 61.16 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 33.89 / 61.15 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 33.90 / 61.15 |
| gf2p16-rijndael | 256 KiB | 131,072 | 33.90 / 61.16 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 33.89 / 61.15 |
| gf2p32-rijndael | 256 KiB | 65,536 | 33.87 / 61.15 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 33.87 / 61.14 |
| gf2p64-rijndael | 256 KiB | 32,768 | 33.87 / 61.15 |
| gf2p8-aes | 256 KiB | 262,144 | 33.87 / 61.16 |
| gf2p8-cantor | 256 KiB | 262,144 | 33.81 / 61.16 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 33.91 / 61.16 |
| gf2p8-normal | 256 KiB | 262,144 | 33.88 / 61.17 |
| gf2p8-rs | 256 KiB | 262,144 | 33.89 / 61.17 |
| gf49 | 256 KiB | 131,072 | 0.195 / 0.416 |
| gf65521 | 256 KiB | 131,072 | 1.319 / 3.275 |
| gf65537 | 256 KiB | 65,536 | 4.927 / 8.409 |
| gf81 | 256 KiB | 65,536 | 0.391 / 0.956 |
| gf9 | 256 KiB | 131,072 | 0.372 / 0.951 |
| gfmaxp | 256 KiB | 32,768 | 6.382 / 14.96 |
| gfu32max | 256 KiB | 65,536 | 1.574 / 3.520 |
| goldilocks | 256 KiB | 32,768 | 32.76 / 24.66 |
| goldilocks-residue | 256 KiB | 32,768 | 5.990 / 14.95 |
| goldilocks-squared | 256 KiB | 16,384 | 0.860 / 1.430 |
| m31 | 256 KiB | 65,536 | 26.18 / 36.96 |
| m31-residue | 256 KiB | 65,536 | 3.978 / 6.650 |
| maxprime-squared | 256 KiB | 16,384 | 3.881 / 9.306 |
| qm31 | 256 KiB | 32,768 | 19.88 / 34.54 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::sub_assign`

</details>

<a id="steady-bulk-sub-scalar-sub-assign-scalar-throughput"></a>

### sub_assign_scalar

**Sources:** 0 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 4.666 / 8.016 |
| gf125 | 262128 B | 87,376 | 0.564 / 1.353 |
| gf127 | 256 KiB | 262,144 | 4.693 / 7.710 |
| gf131 | 256 KiB | 262,144 | 1.084 / 1.678 |
| gf1e9p7 | 256 KiB | 65,536 | 4.989 / 8.487 |
| gf2p1-residue | 256 KiB | 32,768 | 10.52 / 24.89 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 51.84 / 64.76 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 51.84 / 64.82 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 51.84 / 64.86 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 51.83 / 64.76 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 51.73 / 64.72 |
| gf2p16-rijndael | 256 KiB | 131,072 | 51.80 / 64.77 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 51.85 / 64.84 |
| gf2p32-rijndael | 256 KiB | 65,536 | 51.62 / 64.55 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 51.83 / 64.81 |
| gf2p64-rijndael | 256 KiB | 32,768 | 50.86 / 63.90 |
| gf2p8-aes | 256 KiB | 262,144 | 51.84 / 64.89 |
| gf2p8-cantor | 256 KiB | 262,144 | 51.85 / 64.92 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 51.85 / 64.99 |
| gf2p8-normal | 256 KiB | 262,144 | 41.60 / 73.45 |
| gf2p8-rs | 256 KiB | 262,144 | 51.84 / 64.87 |
| gf49 | 256 KiB | 131,072 | 0.285 / 0.637 |
| gf65521 | 256 KiB | 131,072 | 1.713 / 4.360 |
| gf65537 | 256 KiB | 65,536 | 8.148 / 13.23 |
| gf81 | 256 KiB | 65,536 | 0.577 / 1.377 |
| gf9 | 256 KiB | 131,072 | 0.526 / 1.333 |
| gfmaxp | 256 KiB | 32,768 | 7.854 / 19.37 |
| gfu32max | 256 KiB | 65,536 | 2.361 / 5.675 |
| goldilocks | 256 KiB | 32,768 | 51.76 / 38.70 |
| goldilocks-residue | 256 KiB | 32,768 | 7.547 / 19.36 |
| goldilocks-squared | 256 KiB | 16,384 | 3.335 / 6.340 |
| m31 | 256 KiB | 65,536 | 44.97 / 59.30 |
| m31-residue | 256 KiB | 65,536 | 6.590 / 10.75 |
| maxprime-squared | 256 KiB | 16,384 | 5.672 / 13.96 |
| qm31 | 256 KiB | 32,768 | 33.77 / 50.79 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::sub_assign_scalar`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
