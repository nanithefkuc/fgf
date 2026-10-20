# Field-specific arithmetic

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [canonical raw](#steady-details-math-canonical-raw-canonical-raw-latency)
  - [conjugate](#steady-details-math-conjugate-conjugate-latency)
  - [description pow](#steady-details-math-description-pow-description-pow-latency)
  - [fp invert](#steady-details-math-fp-invert-fp-invert-latency)
  - [fp mul alpha](#steady-details-math-fp-mul-alpha-fp-mul-alpha-latency)
  - [fp multiply](#steady-details-math-fp-multiply-fp-multiply-latency)
  - [fp square](#steady-details-math-fp-square-fp-square-latency)
  - [from components](#steady-details-math-from-components-from-components-latency)
  - [from residue](#steady-details-math-from-residue-from-residue-latency)
  - [mul alpha](#steady-details-math-mul-alpha-mul-alpha-latency)
  - [norm](#steady-details-math-norm-norm-latency)
  - [to components](#steady-details-math-to-components-to-components-latency)
  - [to residue](#steady-details-math-to-residue-to-residue-latency)
- [Query and scalar latency](#query-and-scalar-latency)
  - [description basis_root query](#query-details-math-description-is-basis-root-description-basis-root-query-latency)
  - [description polynomial_root query](#query-details-math-description-is-polynomial-root-description-polynomial-root-query-latency)
- [Preparation and setup](#preparation-and-setup)
  - [log_exp build](#setup-details-math-log-exp-build-log-exp-build-latency)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-details-math-canonical-raw-canonical-raw-latency"></a>

### canonical raw

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 0.668 / 0.426 |
| goldilocks | 0.625 / 0.320 |
| m31 | 0.835 / 0.427 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::Field::canonical_raw`

</details>

<a id="steady-details-math-conjugate-conjugate-latency"></a>

### conjugate

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| qm31 | 1.219 / 0.642 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::conjugate (inherent const)`

</details>

<a id="steady-details-math-description-pow-description-pow-latency"></a>

### description pow

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 27.5 / 14.9 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::binary::BinaryDescription::pow`

</details>

<a id="steady-details-math-fp-invert-fp-invert-latency"></a>

### fp invert

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p64-fanpaar | 4486 / 2835 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::binary::tower::fp_invert`

</details>

<a id="steady-details-math-fp-mul-alpha-fp-mul-alpha-latency"></a>

### fp mul alpha

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p64-fanpaar | 16.4 / 9.70 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::binary::tower::fp_mul_alpha`

</details>

<a id="steady-details-math-fp-multiply-fp-multiply-latency"></a>

### fp multiply

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p64-fanpaar | 2908 / 1781 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::binary::tower::fp_multiply`

</details>

<a id="steady-details-math-fp-square-fp-square-latency"></a>

### fp square

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p64-fanpaar | 537 / 329 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::binary::tower::fp_square`

</details>

<a id="steady-details-math-from-components-from-components-latency"></a>

### from components

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf2p16-rijndael | 0.428 / 0.213 | 0/5 / 1/5 |
| qm31 | 1.003 / 0.509 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Elem::from_components (inherent const)`

</details>

<a id="steady-details-math-from-residue-from-residue-latency"></a>

### from residue

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 0.468 / 0.213 |
| m31 | 0.457 / 0.213 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::from_residue (inherent const)`

</details>

<a id="steady-details-math-mul-alpha-mul-alpha-latency"></a>

### mul alpha

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-fanpaar | 11.4 / 7.68 |
| gf2p32-fanpaar | 14.4 / 9.41 |
| gf2p8-fanpaar | 9.03 / 5.97 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::mul_alpha (inherent const)`

</details>

<a id="steady-details-math-norm-norm-latency"></a>

### norm

**Elements:** 1 · **Sources:** 2 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| qm31 | 4.65 / 1.72 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::norm (inherent const)`

</details>

<a id="steady-details-math-to-components-to-components-latency"></a>

### to components

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf2p16-rijndael | 0.432 / 0.213 | 0/5 / 3/5 |
| qm31 | 0.468 / 0.213 | 0/5 / 1/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Elem::to_components (inherent const)`

</details>

<a id="steady-details-math-to-residue-to-residue-latency"></a>

### to residue

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf1e9p7 | 0.627 / 0.361 |
| m31 | 0.631 / 0.361 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Elem::to_residue (inherent const)`

</details>

## Query and scalar latency

Black-boxed scalar and getter calls; these report per-call latency, never a throughput.

<a id="query-details-math-description-is-basis-root-description-basis-root-query-latency"></a>

### description basis_root query

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-normal | 0.780 / 0.427 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::binary::BinaryDescription::is_basis_root`

</details>

<a id="query-details-math-description-is-polynomial-root-description-polynomial-root-query-latency"></a>

### description polynomial_root query

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 0.780 / 0.427 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::binary::BinaryDescription::is_polynomial_root`

</details>

## Preparation and setup

Construction and preparation costs excluded from steady-state panels; owned results are destroyed before the next iteration.

<a id="setup-details-math-log-exp-build-log-exp-build-latency"></a>

### log_exp build

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 2051 / 1198 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::binary::ByteLogExp::build`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
