# Bit-packed operations

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [and_assign](#steady-bits-and-assign-and-assign-throughput)
  - [and_into](#steady-bits-and-into-and-into-throughput)
  - [andnot_assign](#steady-bits-andnot-assign-andnot-assign-throughput)
  - [clear_range](#steady-bits-clear-range-clear-range-throughput)
  - [dot_product](#steady-bits-dot-product-dot-product-throughput)
  - [xor_range (empty buffers)](#steady-bits-empty-buffers-xor-range-empty-buffers-latency)
  - [xor_range (empty range)](#steady-bits-empty-range-xor-range-empty-range-latency)
  - [set_range](#steady-bits-set-range-set-range-throughput)
  - [weight](#steady-bits-weight-weight-throughput)
  - [xor_assign](#steady-bits-xor-assign-xor-assign-throughput)
  - [xor_gather](#steady-bits-xor-gather-xor-gather-throughput)
  - [xor_range](#steady-bits-xor-range-xor-range-throughput)
  - [xor_range_with](#steady-bits-xor-range-prepared-xor-range-with-throughput)
- [Query and scalar latency](#query-and-scalar-latency)
  - [bytes_for](#query-bits-bytes-for-bytes-for-latency)
- [Preparation and setup](#preparation-and-setup)
  - [xor_range_new](#setup-bits-xor-range-new-xor-range-new-latency)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-bits-and-assign-and-assign-throughput"></a>

### and_assign

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 1 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 21.6 / 49.8 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::and_assign`

</details>

<a id="steady-bits-and-into-and-into-throughput"></a>

### and_into

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 2 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 18.6 / 47.1 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::and_into`

</details>

<a id="steady-bits-andnot-assign-andnot-assign-throughput"></a>

### andnot_assign

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 1 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 21.5 / 49.8 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::andnot_assign`

</details>

<a id="steady-bits-clear-range-clear-range-throughput"></a>

### clear_range

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 0 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 54.0 / 71.7 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::clear_range`

</details>

<a id="steady-bits-dot-product-dot-product-throughput"></a>

### dot_product

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 2 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 19.1 / 41.3 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::dot_product`, `fgf::field::FieldElem::is_one`

</details>

<a id="steady-bits-empty-buffers-xor-range-empty-buffers-latency"></a>

### xor_range (empty buffers)

**Elements:** 0 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2-packed-bits | 4.36 / 2.77 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::bits::xor_range`, `fgf::bits::xor_range_with`, `fgf::bits::xor_gather`, `fgf::bits::xor_assign`

</details>

<a id="steady-bits-empty-range-xor-range-empty-range-latency"></a>

### xor_range (empty range)

**Elements:** 64 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2-packed-bits | 8.02 / 5.12 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::bits::xor_range`, `fgf::bits::xor_range_with`, `fgf::bits::xor_gather`, `fgf::bits::xor_assign`

</details>

<a id="steady-bits-set-range-set-range-throughput"></a>

### set_range

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 0 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 54.0 / 71.7 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::set_range`

</details>

<a id="steady-bits-weight-weight-throughput"></a>

### weight

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 1 · **Destinations:** 0

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 68.0 / 14.4 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::weight`

</details>

<a id="steady-bits-xor-assign-xor-assign-throughput"></a>

### xor_assign

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf2-packed-bits | 128 MiB | 1,073,741,824 | 5.94 / 13.7 |
| gf2-packed-bits | 256 KiB | 2,097,152 | 33.9 / 61.3 |
| gf2-packed-bits | 64 B | 512 | 8.11 / 14.0 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::xor_assign`

</details>

<a id="steady-bits-xor-gather-xor-gather-throughput"></a>

### xor_gather

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 4 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 5.34 / 9.43 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::xor_gather`

</details>

<a id="steady-bits-xor-range-xor-range-throughput"></a>

### xor_range

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 1 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 33.9 / 45.5 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::xor_range`

</details>

<a id="steady-bits-xor-range-prepared-xor-range-with-throughput"></a>

### xor_range_with

**Counted bytes:** 256 KiB · **Elements:** 2,097,152 · **Sources:** 1 · **Destinations:** 1

| Field | Throughput (GiB/s) |
| --- | --- |
| gf2-packed-bits | 33.9 / 45.4 |

<details>
<summary>Exact public calls</summary>

`fgf::bits::xor_range_with`, `fgf::bits::XorRange::new`

</details>

## Query and scalar latency

Black-boxed scalar and getter calls; these report per-call latency, never a throughput.

<a id="query-bits-bytes-for-bytes-for-latency"></a>

### bytes_for

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2-packed-bits | 0.668 / 0.341 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::bits::bytes_for`

</details>

## Preparation and setup

Construction and preparation costs excluded from steady-state panels; owned results are destroyed before the next iteration.

<a id="setup-bits-xor-range-new-xor-range-new-latency"></a>

### xor_range_new

**Elements:** 2,097,152 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2-packed-bits | 3.69 / 1.81 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::bits::XorRange::new`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
