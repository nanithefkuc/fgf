# Elementwise buffer products

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [mul_elementwise](#steady-bulk-elementwise-mul-elementwise-throughput)
  - [mul_elementwise_assign](#steady-bulk-elementwise-assign-mul-elementwise-assign-throughput)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-bulk-elementwise-mul-elementwise-throughput"></a>

### mul_elementwise

**Sources:** 2 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 1.904705 / 3.403356 |
| gf125 | 262128 B | 87,376 | 0.054525 / 0.110463 |
| gf127 | 256 KiB | 262,144 | 1.648671 / 2.778434 |
| gf131 | 256 KiB | 262,144 | 6.112895 / 10.26885 |
| gf1e9p7 | 256 KiB | 65,536 | 5.438505 / 6.503047 |
| gf2p1-residue | 256 KiB | 32,768 | 13.58270 / 26.03308 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.260171 / 0.527019 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.261484 / 0.528321 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.260096 / 0.530350 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.260446 / 0.528967 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 0.005886 / 0.009017 |
| gf2p16-rijndael | 256 KiB | 131,072 | 27.83234 / 30.47595 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 0.003865 / 0.006313 |
| gf2p32-rijndael | 256 KiB | 65,536 | 0.132437 / 0.276942 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 0.002470 / 0.004089 |
| gf2p64-rijndael | 256 KiB | 32,768 | 0.036320 / 0.071505 |
| gf2p8-aes | 256 KiB | 262,144 | 28.24231 / 41.61794 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.538442 / 1.139779 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000812 / 0.001836 |
| gf2p8-normal | 256 KiB | 262,144 | 0.599322 / 1.139823 |
| gf2p8-rs | 256 KiB | 262,144 | 28.30671 / 33.27656 |
| gf49 | 256 KiB | 131,072 | 0.081678 / 0.158649 |
| gf65521 | 256 KiB | 131,072 | 1.207702 / 2.907401 |
| gf65537 | 256 KiB | 65,536 | 5.735612 / 6.501986 |
| gf81 | 256 KiB | 65,536 | 0.053176 / 0.103243 |
| gf9 | 256 KiB | 131,072 | 0.124518 / 0.250647 |
| gfmaxp | 256 KiB | 32,768 | 2.200263 / 3.493969 |
| gfu32max | 256 KiB | 65,536 | 5.445556 / 6.501575 |
| goldilocks | 256 KiB | 32,768 | 10.18977 / 10.10605 |
| goldilocks-residue | 256 KiB | 32,768 | 2.172836 / 3.493925 |
| goldilocks-squared | 256 KiB | 16,384 | 0.628918 / 1.185431 |
| m31 | 256 KiB | 65,536 | 14.71602 / 19.62405 |
| m31-residue | 256 KiB | 65,536 | 6.307789 / 6.498719 |
| maxprime-squared | 256 KiB | 16,384 | 0.882231 / 1.746956 |
| qm31 | 256 KiB | 32,768 | 11.12480 / 13.33765 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_elementwise`

</details>

<a id="steady-bulk-elementwise-assign-mul-elementwise-assign-throughput"></a>

### mul_elementwise_assign

**Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Elements | Throughput (GiB/s) |
| --- | --- | --- | --- |
| gf101 | 256 KiB | 262,144 | 1.904130 / 3.419061 |
| gf125 | 262128 B | 87,376 | 0.053800 / 0.107965 |
| gf127 | 256 KiB | 262,144 | 1.644989 / 2.784157 |
| gf131 | 256 KiB | 262,144 | 6.114330 / 10.32176 |
| gf1e9p7 | 256 KiB | 65,536 | 5.418033 / 6.078158 |
| gf2p1-residue | 256 KiB | 32,768 | 13.85598 / 27.45364 |
| gf2p16-custom-cantor | 256 KiB | 131,072 | 0.254721 / 0.519486 |
| gf2p16-custom-normal | 256 KiB | 131,072 | 0.254748 / 0.523098 |
| gf2p16-custom-rijndael | 256 KiB | 131,072 | 0.255452 / 0.521375 |
| gf2p16-custom-rs | 256 KiB | 131,072 | 0.254913 / 0.521526 |
| gf2p16-fanpaar | 256 KiB | 131,072 | 0.005884 / 0.009619 |
| gf2p16-rijndael | 256 KiB | 131,072 | 26.07682 / 33.80412 |
| gf2p32-fanpaar | 256 KiB | 65,536 | 0.003866 / 0.006300 |
| gf2p32-rijndael | 256 KiB | 65,536 | 0.131804 / 0.271972 |
| gf2p64-fanpaar | 256 KiB | 32,768 | 0.002476 / 0.004017 |
| gf2p64-rijndael | 256 KiB | 32,768 | 0.037633 / 0.071276 |
| gf2p8-aes | 256 KiB | 262,144 | 28.13761 / 50.76685 |
| gf2p8-cantor | 256 KiB | 262,144 | 0.605813 / 1.164365 |
| gf2p8-fanpaar | 256 KiB | 262,144 | 0.000814 / 0.001833 |
| gf2p8-normal | 256 KiB | 262,144 | 0.547435 / 1.161282 |
| gf2p8-rs | 256 KiB | 262,144 | 27.84398 / 43.94114 |
| gf49 | 256 KiB | 131,072 | 0.082394 / 0.153165 |
| gf65521 | 256 KiB | 131,072 | 1.163660 / 2.910572 |
| gf65537 | 256 KiB | 65,536 | 5.414458 / 6.089320 |
| gf81 | 256 KiB | 65,536 | 0.052790 / 0.101720 |
| gf9 | 256 KiB | 131,072 | 0.125571 / 0.243435 |
| gfmaxp | 256 KiB | 32,768 | 2.206333 / 3.493925 |
| gfu32max | 256 KiB | 65,536 | 5.417687 / 6.089149 |
| goldilocks | 256 KiB | 32,768 | 11.48107 / 11.00489 |
| goldilocks-residue | 256 KiB | 32,768 | 2.191715 / 3.493906 |
| goldilocks-squared | 256 KiB | 16,384 | 0.630950 / 1.187816 |
| m31 | 256 KiB | 65,536 | 14.86510 / 19.66427 |
| m31-residue | 256 KiB | 65,536 | 6.359651 / 6.069036 |
| maxprime-squared | 256 KiB | 16,384 | 0.896065 / 1.746942 |
| qm31 | 256 KiB | 32,768 | 7.559265 / 13.59916 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::mul_elementwise_assign`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
