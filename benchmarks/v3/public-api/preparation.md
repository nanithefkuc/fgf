# Preparation and construction

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Preparation and setup](#preparation-and-setup)
  - [coeff preparation](#setup-setup-coeff-new-coeff-preparation-latency)
  - [coefficient matrix preparation](#setup-setup-coeffmatrix-new-coefficient-matrix-preparation-latency)
  - [coeff vector preparation](#setup-setup-coeffvec-new-coeff-vector-preparation-latency)

## Preparation and setup

Construction and preparation costs excluded from steady-state panels; owned results are destroyed before the next iteration.

<a id="setup-setup-coeff-new-coeff-preparation-latency"></a>

### coeff preparation

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-fanpaar | 10.03 / 6.078 |
| gf2p16-rijndael | 2.683 / 1.280 |
| gf2p8-aes | 0.585 / 0.299 |
| gfmaxp | 3.624 / 2.133 |
| m31 | 0.427 / 0.213 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::Coeff::new`

</details>

<a id="setup-setup-coeffmatrix-new-coefficient-matrix-preparation-latency"></a>

### coefficient matrix preparation

**Elements:** 64 · **Sources:** 8 · **Destinations:** 8

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-fanpaar | 651 / 395 |
| gf2p16-rijndael | 157 / 81.0 |
| gf2p8-aes | 1.67 / 1.07 |
| gfmaxp | 1.67 / 1.07 |
| m31 | 1.67 / 1.07 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::from_source_major`

</details>

<a id="setup-setup-coeffvec-new-coeff-vector-preparation-latency"></a>

### coeff vector preparation

**Elements:** 8 · **Sources:** 8 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-fanpaar | 84.4 / 54.6 |
| gf2p16-rijndael | 22.7 / 11.1 |
| gf2p8-aes | 12.9 / 6.85 |
| gfmaxp | 3.34 / 1.92 |
| m31 | 6.02 / 1.92 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::new`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
