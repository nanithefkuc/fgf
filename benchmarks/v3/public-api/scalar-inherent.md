# Inherent scalar arithmetic

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [add (inherent)](#steady-scalar-inherent-add-add-inherent-latency)
  - [canonical (inherent)](#steady-scalar-inherent-canonical-canonical-inherent-latency)
  - [div (inherent)](#steady-scalar-inherent-div-div-inherent-latency)
  - [from_bytes (inherent)](#steady-scalar-inherent-from-bytes-from-bytes-inherent-latency)
  - [from_raw (inherent)](#steady-scalar-inherent-from-raw-from-raw-inherent-latency)
  - [inv (inherent)](#steady-scalar-inherent-inv-inv-inherent-latency)
  - [mul (inherent)](#steady-scalar-inherent-mul-mul-inherent-latency)
  - [neg (inherent)](#steady-scalar-inherent-neg-neg-inherent-latency)
  - [pow (inherent)](#steady-scalar-inherent-pow-pow-inherent-latency)
  - [square (inherent)](#steady-scalar-inherent-square-square-inherent-latency)
  - [sub (inherent)](#steady-scalar-inherent-sub-sub-inherent-latency)
  - [to_bytes (inherent)](#steady-scalar-inherent-to-bytes-to-bytes-inherent-latency)
- [Query and scalar latency](#query-and-scalar-latency)
  - [is_one (inherent)](#query-scalar-inherent-is-one-is-one-inherent-latency)
  - [is_zero (inherent)](#query-scalar-inherent-is-zero-is-zero-inherent-latency)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-scalar-inherent-add-add-inherent-latency"></a>

### add (inherent)

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 1.003 / 0.469 |
| gf2p1-poly | 0.753 / 0.426 |
| gf2p16-rijndael | 0.668 / 0.427 |
| gf2p2-poly | 0.745 / 0.426 |
| gf2p32-rijndael | 0.745 / 0.426 |
| gf2p4-poly | 0.735 / 0.426 |
| gf2p64-rijndael | 0.830 / 0.426 |
| gf2p8-aes | 0.735 / 0.426 |
| goldilocks | 1.607 / 0.853 |
| m31 | 1.671 / 0.853 |
| qm31 | 3.721 / 1.710 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::add (inherent const)`

</details>

<a id="steady-scalar-inherent-canonical-canonical-inherent-latency"></a>

### canonical (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 0.668 / 0.427 |
| gf2p1-poly | 0.425 / 0.213 |
| gf2p16-rijndael | 0.425 / 0.213 |
| gf2p2-poly | 0.427 / 0.213 |
| gf2p32-rijndael | 0.427 / 0.426 |
| gf2p4-poly | 0.427 / 0.213 |
| gf2p64-rijndael | 0.468 / 0.426 |
| gf2p8-aes | 0.425 / 0.213 |
| qm31 | 0.468 / 0.213 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::canonical (inherent const)`

</details>

<a id="steady-scalar-inherent-div-div-inherent-latency"></a>

### div (inherent)

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 310.8 / 137.8 |
| gf2p1-poly | 0.668 / 0.426 |
| gf2p16-rijndael | 25.79 / 12.13 |
| gf2p2-poly | 16.28 / 7.735 |
| gf2p32-rijndael | 96.17 / 54.36 |
| gf2p4-poly | 25.11 / 11.51 |
| gf2p64-rijndael | 305.0 / 163.0 |
| gf2p8-aes | 2.368 / 1.246 |
| goldilocks | 455.3 / 231.8 |
| m31 | 235.9 / 107.1 |
| qm31 | 389.7 / 208.1 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::div (inherent const)`

</details>

<a id="steady-scalar-inherent-from-bytes-from-bytes-inherent-latency"></a>

### from_bytes (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf2p1-poly | 0.420 / 0.213 | 0/5 / 3/5 |
| gf2p16-rijndael | 0.433 / 0.213 | 0/5 / 1/5 |
| gf2p2-poly | 0.413 / 0.213 | 0/5 / 2/5 |
| gf2p32-rijndael | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p4-poly | 0.413 / 0.213 | 0/5 / 2/5 |
| gf2p64-rijndael | 0.468 / 0.213 | 0/5 / 1/5 |
| gf2p8-aes | 0.427 / 0.213 | 0/5 / 1/5 |
| goldilocks | 0.625 / 0.320 | 0/5 / 0/5 |
| m31 | 0.780 / 0.427 | 0/5 / 0/5 |
| qm31 | 1.639 / 0.784 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Elem::from_bytes (inherent const)`

</details>

<a id="steady-scalar-inherent-from-raw-from-raw-inherent-latency"></a>

### from_raw (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf1e9p7 | 0.668 / 0.426 | 0/5 / 0/5 |
| gf2p1-poly | 0.413 / 0.213 | 0/5 / 0/5 |
| gf2p16-rijndael | 0.433 / 0.213 | 0/5 / 4/5 |
| gf2p2-poly | 0.413 / 0.213 | 0/5 / 1/5 |
| gf2p32-rijndael | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p4-poly | 0.413 / 0.213 | 0/5 / 3/5 |
| gf2p64-rijndael | 0.468 / 0.213 | 0/5 / 4/5 |
| gf2p8-aes | 0.427 / 0.213 | 0/5 / 1/5 |
| goldilocks | 0.625 / 0.320 | 0/5 / 0/5 |
| m31 | 0.780 / 0.427 | 0/5 / 0/5 |
| qm31 | 1.575 / 0.749 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Elem::from_raw (inherent const)`

</details>

<a id="steady-scalar-inherent-inv-inv-inherent-latency"></a>

### inv (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 301.5 / 133.7 |
| gf2p1-poly | 4.013 / 2.002 |
| gf2p16-rijndael | 12.91 / 6.284 |
| gf2p2-poly | 19.43 / 8.964 |
| gf2p32-rijndael | 66.92 / 35.25 |
| gf2p4-poly | 27.37 / 13.23 |
| gf2p64-rijndael | 197.9 / 106.4 |
| gf2p8-aes | 1.003 / 0.533 |
| goldilocks | 453.0 / 231.4 |
| m31 | 236.3 / 102.3 |
| qm31 | 365.8 / 191.5 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::inv (inherent const)`

</details>

<a id="steady-scalar-inherent-mul-mul-inherent-latency"></a>

### mul (inherent)

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 3.343 / 2.133 |
| gf2p1-poly | 6.800 / 3.485 |
| gf2p16-rijndael | 7.064 / 3.278 |
| gf2p2-poly | 11.92 / 5.595 |
| gf2p32-rijndael | 24.92 / 12.38 |
| gf2p4-poly | 20.87 / 9.379 |
| gf2p64-rijndael | 91.63 / 47.35 |
| gf2p8-aes | 1.504 / 0.785 |
| goldilocks | 2.020 / 0.900 |
| m31 | 2.582 / 1.181 |
| qm31 | 9.758 / 3.847 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::mul (inherent const)`

</details>

<a id="steady-scalar-inherent-neg-neg-inherent-latency"></a>

### neg (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf1e9p7 | 0.557 / 0.284 | 0/5 / 0/5 |
| gf2p1-poly | 0.425 / 0.213 | 0/5 / 0/5 |
| gf2p16-rijndael | 0.425 / 0.213 | 0/5 / 0/5 |
| gf2p2-poly | 0.427 / 0.213 | 0/5 / 0/5 |
| gf2p32-rijndael | 0.427 / 0.426 | 0/5 / 1/5 |
| gf2p4-poly | 0.427 / 0.213 | 0/5 / 0/5 |
| gf2p64-rijndael | 0.468 / 0.426 | 0/5 / 1/5 |
| gf2p8-aes | 0.426 / 0.213 | 0/5 / 0/5 |
| goldilocks | 0.893 / 0.427 | 0/5 / 0/5 |
| m31 | 1.062 / 0.555 | 0/5 / 0/5 |
| qm31 | 2.416 / 1.112 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Elem::neg (inherent const)`

</details>

<a id="steady-scalar-inherent-pow-pow-inherent-latency"></a>

### pow (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 19.4 / 10.7 |
| gf2p1-poly | 44.0 / 21.6 |
| gf2p16-rijndael | 42.6 / 22.6 |
| gf2p2-poly | 74.8 / 34.3 |
| gf2p32-rijndael | 116 / 59.7 |
| gf2p4-poly | 130 / 57.5 |
| gf2p64-rijndael | 353 / 186 |
| gf2p8-aes | 22.7 / 12.4 |
| goldilocks | 16.2 / 6.73 |
| m31 | 18.7 / 6.90 |
| qm31 | 50.8 / 21.7 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::pow (inherent const)`

</details>

<a id="steady-scalar-inherent-square-square-inherent-latency"></a>

### square (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 3.342 / 2.133 |
| gf2p1-poly | 6.776 / 3.413 |
| gf2p16-rijndael | 3.962 / 1.918 |
| gf2p2-poly | 11.96 / 5.527 |
| gf2p32-rijndael | 11.77 / 5.983 |
| gf2p4-poly | 21.23 / 9.387 |
| gf2p64-rijndael | 42.67 / 22.10 |
| gf2p8-aes | 1.003 / 0.499 |
| goldilocks | 1.728 / 0.787 |
| m31 | 2.025 / 0.857 |
| qm31 | 7.244 / 2.795 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::square (inherent const)`

</details>

<a id="steady-scalar-inherent-sub-sub-inherent-latency"></a>

### sub (inherent)

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 0.811 / 0.426 |
| gf2p1-poly | 0.753 / 0.426 |
| gf2p16-rijndael | 0.692 / 0.426 |
| gf2p2-poly | 0.745 / 0.426 |
| gf2p32-rijndael | 0.745 / 0.426 |
| gf2p4-poly | 0.735 / 0.426 |
| gf2p64-rijndael | 0.820 / 0.426 |
| gf2p8-aes | 0.735 / 0.427 |
| goldilocks | 1.816 / 0.960 |
| m31 | 1.671 / 0.853 |
| qm31 | 3.721 / 1.712 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::sub (inherent const)`

</details>

<a id="steady-scalar-inherent-to-bytes-to-bytes-inherent-latency"></a>

### to_bytes (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf2p1-poly | 0.427 / 0.213 | 0/5 / 1/5 |
| gf2p16-rijndael | 0.433 / 0.213 | 0/5 / 1/5 |
| gf2p2-poly | 0.427 / 0.213 | 0/5 / 3/5 |
| gf2p32-rijndael | 0.427 / 0.213 | 0/5 / 2/5 |
| gf2p4-poly | 0.427 / 0.213 | 0/5 / 2/5 |
| gf2p64-rijndael | 0.468 / 0.213 | 0/5 / 1/5 |
| gf2p8-aes | 0.427 / 0.213 | 0/5 / 3/5 |
| goldilocks | 0.468 / 0.213 | 0/5 / 2/5 |
| m31 | 0.427 / 0.213 | 0/5 / 1/5 |
| qm31 | 0.468 / 0.213 | 0/5 / 3/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Elem::to_bytes (inherent const)`

</details>

## Query and scalar latency

Black-boxed scalar and getter calls; these report per-call latency, never a throughput.

<a id="query-scalar-inherent-is-one-is-one-inherent-latency"></a>

### is_one (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf1e9p7 | 0.446 / 0.213 | 0/5 / 1/5 |
| gf2p1-poly | 0.446 / 0.213 | 0/5 / 1/5 |
| gf2p16-rijndael | 0.446 / 0.240 | 0/5 / 0/5 |
| gf2p2-poly | 0.446 / 0.213 | 0/5 / 1/5 |
| gf2p32-rijndael | 0.446 / 0.213 | 0/5 / 1/5 |
| gf2p4-poly | 0.446 / 0.217 | 0/5 / 0/5 |
| gf2p64-rijndael | 0.502 / 0.219 | 0/5 / 0/5 |
| gf2p8-aes | 0.446 / 0.213 | 0/5 / 1/5 |
| qm31 | 0.557 / 0.261 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Elem::is_one (inherent const)`

</details>

<a id="query-scalar-inherent-is-zero-is-zero-inherent-latency"></a>

### is_zero (inherent)

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf1e9p7 | 0.457 / 0.213 | 0/5 / 1/5 |
| gf2p1-poly | 0.446 / 0.213 | 0/5 / 2/5 |
| gf2p16-rijndael | 0.446 / 0.240 | 0/5 / 0/5 |
| gf2p2-poly | 0.446 / 0.213 | 0/5 / 2/5 |
| gf2p32-rijndael | 0.432 / 0.213 | 0/5 / 0/5 |
| gf2p4-poly | 0.446 / 0.217 | 0/5 / 1/5 |
| gf2p64-rijndael | 0.457 / 0.213 | 0/5 / 2/5 |
| gf2p8-aes | 0.446 / 0.213 | 0/5 / 1/5 |
| qm31 | 0.501 / 0.241 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Elem::is_zero (inherent const)`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
