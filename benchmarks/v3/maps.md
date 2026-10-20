# Field maps

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../BENCHMARKS.md) · [Field definitions](../labels.md)

Map panels time public scalar calls and caller loops over validated exact isomorphisms; no bulk-map API exists. Map fixtures are validated outside the timed region, and map records report the scalar route.

[Public operations](public-api.md) · [GF16 presentations](gf16-presentations.md)

Pairs identify **S → T** in the prepared map. For embeddings, `restrict`, `trace`, and `norm` run **T → S**; `frobenius` runs **T → T**.

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [apply](#steady-maps-apply-apply-latency)
  - [apply_unapply](#steady-maps-apply-unapply-apply-unapply-latency)
  - [contains](#steady-maps-contains-contains-latency)
  - [frobenius](#steady-maps-frobenius-frobenius-latency)
  - [norm](#steady-maps-norm-norm-latency)
  - [restrict](#steady-maps-restrict-restrict-latency)
  - [trace](#steady-maps-trace-trace-latency)
- [Query and scalar latency](#query-and-scalar-latency)
  - [iso_to_images](#query-details-maps-iso-to-images-iso-to-images-latency)
  - [generator_count](#query-maps-direct-generator-count-generator-count-latency)
  - [fail_canonical](#query-maps-fail-canonical-fail-canonical-latency)
  - [fail_degree](#query-maps-fail-degree-fail-degree-latency)
  - [fail_images](#query-maps-fail-images-fail-images-latency)
  - [generator queries](#query-maps-generators-generator-queries-latency)
  - [iso_conversions](#query-maps-iso-conversions-iso-conversions-latency)
  - [relative_degree](#query-maps-relative-degree-relative-degree-latency)
- [Preparation and setup](#preparation-and-setup)
  - [iso_from_images](#setup-details-maps-iso-from-images-iso-from-images-latency)
  - [iso_then](#setup-details-maps-iso-then-iso-then-latency)
  - [from_embedding](#setup-maps-direct-from-embedding-from-embedding-latency)
  - [new](#setup-maps-direct-new-new-latency)
- [End-to-end compositions](#end-to-end-compositions)
  - [build](#end-to-end-maps-build-build-latency)
  - [from_images](#end-to-end-maps-from-images-from-images-latency)
  - [then](#end-to-end-maps-then-then-latency)

- [Conversion caller loops](#conversion-caller-loops)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-maps-apply-apply-latency"></a>

### apply

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p1-poly → gf2p2-poly | 34.5144 / 20.4742 |
| gf2p1-poly → gf2p5-padded | 34.5073 / 20.6861 |
| gf2p16-fanpaar → gf2p32-fanpaar | 29535.0 / 17850.1 |
| gf2p16-rijndael → gf2p16-fanpaar | 5800.79 / 3540.75 |
| gf2p16-rijndael → gf2p32-rijndael | 308.430 / 148.306 |
| gf2p2-fanpaar → gf2p4-fanpaar | 508.183 / 231.550 |
| gf2p2-poly → gf2p2-cantor | 93.8948 / 46.0050 |
| gf2p2-poly → gf2p2-normal | 95.6014 / 46.2403 |
| gf2p2-poly → gf2p4-poly | 82.0577 / 40.7945 |
| gf2p32-fanpaar → gf2p64-fanpaar | 181611 / 110245 |
| gf2p32-rijndael → gf2p64-rijndael | 1523.98 / 831.982 |
| gf2p4-fanpaar → gf2p8-fanpaar | 7009.60 / 3132.31 |
| gf2p4-poly → gf2p4-cantor | 189.696 / 92.4967 |
| gf2p4-poly → gf2p4-normal | 190.051 / 92.6393 |
| gf2p4-poly → gf2p8-aes | 43.4036 / 22.6352 |
| gf2p8-aes → gf2p16-custom-rijndael | 97.4248 / 50.6259 |
| gf2p8-aes → gf2p16-rijndael | 80.5723 / 39.0666 |
| gf2p8-aes → gf2p8-aes | 82.8525 / 40.7007 |
| gf2p8-aes → gf2p8-cantor | 89.5807 / 45.1704 |
| gf2p8-aes → gf2p8-normal | 88.6612 / 45.6179 |
| gf2p8-aes → gf2p8-rs | 87.0169 / 45.1776 |
| gf2p8-cantor → gf2p16-custom-cantor | 109.685 / 63.9811 |
| gf2p8-fanpaar → gf2p16-fanpaar | 4709.67 / 2874.66 |
| gf2p8-normal → gf2p16-custom-normal | 106.634 / 64.7578 |
| gf2p8-rs → gf2p16-custom-rs | 97.1943 / 50.4868 |
| gf3 → gf9 | 20.3857 / 12.5822 |
| gf5 → gf125 | 20.3864 / 12.5828 |
| gf7 → gf49 | 32.8693 / 20.4710 |
| gf9 → gf81 | 129.858 / 57.9654 |
| gfmaxp → maxprime-squared | 20.3862 / 12.7859 |
| goldilocks → goldilocks-squared | 20.3864 / 12.8151 |
| m31 → qm31 | 34.1340 / 20.4733 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::embed`

</details>

<a id="steady-maps-apply-unapply-apply-unapply-latency"></a>

### apply_unapply

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael → gf2p16-fanpaar | 6172 / 3884 |
| gf2p2-poly → gf2p2-cantor | 233.1 / 118.6 |
| gf2p2-poly → gf2p2-normal | 234.9 / 118.8 |
| gf2p4-poly → gf2p4-cantor | 423.4 / 222.1 |
| gf2p4-poly → gf2p4-normal | 425.6 / 222.2 |
| gf2p8-aes → gf2p8-aes | 150.7 / 73.72 |
| gf2p8-aes → gf2p8-cantor | 190.2 / 105.7 |
| gf2p8-aes → gf2p8-normal | 186.6 / 105.5 |
| gf2p8-aes → gf2p8-rs | 162.1 / 81.66 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::new`, `fgf::Isomorphism::apply`, `fgf::Isomorphism::unapply`

</details>

<a id="steady-maps-contains-contains-latency"></a>

### contains

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p1-poly → gf2p2-poly | 36.903 / 16.857 |
| gf2p1-poly → gf2p5-padded | 561.36 / 274.57 |
| gf2p16-fanpaar → gf2p32-fanpaar | 17342 / 10634 |
| gf2p16-rijndael → gf2p16-fanpaar | 5734.9 / 3501.1 |
| gf2p16-rijndael → gf2p32-rijndael | 393.00 / 195.31 |
| gf2p2-fanpaar → gf2p4-fanpaar | 910.41 / 407.24 |
| gf2p2-poly → gf2p2-cantor | 113.46 / 50.889 |
| gf2p2-poly → gf2p2-normal | 113.71 / 50.918 |
| gf2p2-poly → gf2p4-poly | 86.465 / 40.365 |
| gf2p32-fanpaar → gf2p64-fanpaar | 98977 / 60155 |
| gf2p32-rijndael → gf2p64-rijndael | 2309.9 / 1142.7 |
| gf2p4-fanpaar → gf2p8-fanpaar | 6863.1 / 3060.6 |
| gf2p4-poly → gf2p4-cantor | 227.98 / 107.60 |
| gf2p4-poly → gf2p4-normal | 227.86 / 107.75 |
| gf2p4-poly → gf2p8-aes | 7.0516 / 3.5910 |
| gf2p8-aes → gf2p16-custom-rijndael | 23.641 / 11.367 |
| gf2p8-aes → gf2p16-rijndael | 99.001 / 50.492 |
| gf2p8-aes → gf2p8-aes | 18.506 / 9.8810 |
| gf2p8-aes → gf2p8-cantor | 18.486 / 9.8675 |
| gf2p8-aes → gf2p8-normal | 18.462 / 9.8657 |
| gf2p8-aes → gf2p8-rs | 18.428 / 9.7401 |
| gf2p8-cantor → gf2p16-custom-cantor | 23.651 / 11.328 |
| gf2p8-fanpaar → gf2p16-fanpaar | 3166.0 / 1929.6 |
| gf2p8-normal → gf2p16-custom-normal | 23.683 / 11.341 |
| gf2p8-rs → gf2p16-custom-rs | 23.818 / 11.385 |
| gf3 → gf9 | 16.152 / 6.4370 |
| gf5 → gf125 | 63.642 / 26.778 |
| gf7 → gf49 | 37.951 / 16.824 |
| gf9 → gf81 | 231.21 / 101.70 |
| gfmaxp → maxprime-squared | 1483.4 / 945.64 |
| goldilocks → goldilocks-squared | 1258.1 / 592.06 |
| m31 → qm31 | 758.12 / 295.35 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::contains`

</details>

<a id="steady-maps-frobenius-frobenius-latency"></a>

### frobenius

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p1-poly → gf2p2-poly | 36.261 / 16.565 |
| gf2p1-poly → gf2p5-padded | 560.96 / 274.23 |
| gf2p16-fanpaar → gf2p32-fanpaar | 17348 / 10648 |
| gf2p16-rijndael → gf2p16-fanpaar | 5732.9 / 3588.1 |
| gf2p16-rijndael → gf2p32-rijndael | 392.34 / 195.13 |
| gf2p2-fanpaar → gf2p4-fanpaar | 910.80 / 406.42 |
| gf2p2-poly → gf2p2-cantor | 112.61 / 50.859 |
| gf2p2-poly → gf2p2-normal | 112.38 / 50.800 |
| gf2p2-poly → gf2p4-poly | 85.928 / 40.228 |
| gf2p32-fanpaar → gf2p64-fanpaar | 98994 / 60104 |
| gf2p32-rijndael → gf2p64-rijndael | 2313.2 / 1144.9 |
| gf2p4-fanpaar → gf2p8-fanpaar | 6872.5 / 3059.9 |
| gf2p4-poly → gf2p4-cantor | 227.31 / 107.45 |
| gf2p4-poly → gf2p4-normal | 227.03 / 107.52 |
| gf2p4-poly → gf2p8-aes | 6.1676 / 3.4434 |
| gf2p8-aes → gf2p16-custom-rijndael | 23.039 / 11.117 |
| gf2p8-aes → gf2p16-rijndael | 97.937 / 50.207 |
| gf2p8-aes → gf2p8-aes | 17.503 / 9.0708 |
| gf2p8-aes → gf2p8-cantor | 17.505 / 8.9832 |
| gf2p8-aes → gf2p8-normal | 17.504 / 8.9801 |
| gf2p8-aes → gf2p8-rs | 17.504 / 9.0737 |
| gf2p8-cantor → gf2p16-custom-cantor | 23.030 / 11.094 |
| gf2p8-fanpaar → gf2p16-fanpaar | 3166.2 / 1940.8 |
| gf2p8-normal → gf2p16-custom-normal | 23.034 / 11.093 |
| gf2p8-rs → gf2p16-custom-rs | 23.112 / 11.099 |
| gf3 → gf9 | 14.900 / 5.9195 |
| gf5 → gf125 | 63.135 / 25.334 |
| gf7 → gf49 | 37.264 / 16.675 |
| gf9 → gf81 | 230.87 / 101.46 |
| gfmaxp → maxprime-squared | 1483.1 / 945.36 |
| goldilocks → goldilocks-squared | 1257.3 / 591.64 |
| m31 → qm31 | 751.52 / 296.07 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::frobenius`

</details>

<a id="steady-maps-norm-norm-latency"></a>

### norm

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p1-poly → gf2p2-poly | 104.2011 / 47.99316 |
| gf2p1-poly → gf2p5-padded | 3579.630 / 1760.777 |
| gf2p16-fanpaar → gf2p32-fanpaar | 670534.8 / 407206.1 |
| gf2p16-rijndael → gf2p16-fanpaar | 266135.3 / 166776.5 |
| gf2p16-rijndael → gf2p32-rijndael | 14093.38 / 7119.800 |
| gf2p2-fanpaar → gf2p4-fanpaar | 9271.182 / 4150.167 |
| gf2p2-poly → gf2p2-cantor | 769.2540 / 344.0036 |
| gf2p2-poly → gf2p2-normal | 775.0760 / 343.9764 |
| gf2p2-poly → gf2p4-poly | 683.6418 / 310.0296 |
| gf2p32-fanpaar → gf2p64-fanpaar | 4646101 / 2823234 |
| gf2p32-rijndael → gf2p64-rijndael | 82261.63 / 44772.63 |
| gf2p4-fanpaar → gf2p8-fanpaar | 136341.8 / 60666.75 |
| gf2p4-poly → gf2p4-cantor | 3594.214 / 1652.633 |
| gf2p4-poly → gf2p4-normal | 3590.893 / 1649.542 |
| gf2p4-poly → gf2p8-aes | 562.3031 / 272.1616 |
| gf2p8-aes → gf2p16-custom-rijndael | 1369.162 / 692.8934 |
| gf2p8-aes → gf2p16-rijndael | 3337.933 / 1780.955 |
| gf2p8-aes → gf2p8-aes | 2086.596 / 1145.738 |
| gf2p8-aes → gf2p8-cantor | 589.1389 / 306.2046 |
| gf2p8-aes → gf2p8-normal | 584.6698 / 305.8356 |
| gf2p8-aes → gf2p8-rs | 584.8909 / 306.7527 |
| gf2p8-cantor → gf2p16-custom-cantor | 1368.894 / 691.8333 |
| gf2p8-fanpaar → gf2p16-fanpaar | 92155.88 / 56036.75 |
| gf2p8-normal → gf2p16-custom-normal | 1386.586 / 723.6115 |
| gf2p8-rs → gf2p16-custom-rs | 1369.627 / 693.2045 |
| gf3 → gf9 | 67.11750 / 26.39357 |
| gf5 → gf125 | 271.1501 / 123.7821 |
| gf7 → gf49 | 122.9308 / 54.19006 |
| gf9 → gf81 | 2048.204 / 898.7982 |
| gfmaxp → maxprime-squared | 3058.364 / 1891.000 |
| goldilocks → goldilocks-squared | 2545.538 / 1194.476 |
| m31 → qm31 | 1542.143 / 604.9597 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::norm`

</details>

<a id="steady-maps-restrict-restrict-latency"></a>

### restrict

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p1-poly → gf2p2-poly | 56.88693 / 26.52615 |
| gf2p1-poly → gf2p5-padded | 580.5318 / 284.1041 |
| gf2p16-fanpaar → gf2p32-fanpaar | 652157.1 / 396760.5 |
| gf2p16-rijndael → gf2p16-fanpaar | 266158.0 / 166000.8 |
| gf2p16-rijndael → gf2p32-rijndael | 13752.63 / 6888.067 |
| gf2p2-fanpaar → gf2p4-fanpaar | 8161.385 / 3648.241 |
| gf2p2-poly → gf2p2-cantor | 769.6875 / 343.8310 |
| gf2p2-poly → gf2p2-normal | 770.3560 / 344.1000 |
| gf2p2-poly → gf2p4-poly | 575.0414 / 261.1412 |
| gf2p32-fanpaar → gf2p64-fanpaar | 4543290 / 2759789 |
| gf2p32-rijndael → gf2p64-rijndael | 79457.63 / 43587.13 |
| gf2p4-fanpaar → gf2p8-fanpaar | 128259.5 / 57076.38 |
| gf2p4-poly → gf2p4-cantor | 3595.750 / 1653.902 |
| gf2p4-poly → gf2p4-normal | 3595.679 / 1651.625 |
| gf2p4-poly → gf2p8-aes | 554.5213 / 274.8399 |
| gf2p8-aes → gf2p16-custom-rijndael | 1334.845 / 676.5957 |
| gf2p8-aes → gf2p16-rijndael | 3281.150 / 1742.813 |
| gf2p8-aes → gf2p8-aes | 2092.315 / 1148.667 |
| gf2p8-aes → gf2p8-cantor | 587.1114 / 305.5067 |
| gf2p8-aes → gf2p8-normal | 587.4636 / 305.6916 |
| gf2p8-aes → gf2p8-rs | 587.9485 / 305.4067 |
| gf2p8-cantor → gf2p16-custom-cantor | 1335.541 / 675.7607 |
| gf2p8-fanpaar → gf2p16-fanpaar | 88692.75 / 54109.63 |
| gf2p8-normal → gf2p16-custom-normal | 1352.340 / 707.6154 |
| gf2p8-rs → gf2p16-custom-rs | 1335.347 / 675.0563 |
| gf3 → gf9 | 49.92906 / 19.12023 |
| gf5 → gf125 | 89.37198 / 37.08973 |
| gf7 → gf49 | 94.84113 / 54.70957 |
| gf9 → gf81 | 1771.047 / 783.7542 |
| gfmaxp → maxprime-squared | 1501.848 / 930.6143 |
| goldilocks → goldilocks-squared | 1279.184 / 601.2099 |
| m31 → qm31 | 774.7339 / 304.9110 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::restrict`, `fgf::Embedding::contains`

</details>

<a id="steady-maps-trace-trace-latency"></a>

### trace

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p1-poly → gf2p2-poly | 92.91820 / 42.61932 |
| gf2p1-poly → gf2p5-padded | 2880.778 / 1380.190 |
| gf2p16-fanpaar → gf2p32-fanpaar | 669537.5 / 406603.4 |
| gf2p16-rijndael → gf2p16-fanpaar | 266514.6 / 168607.6 |
| gf2p16-rijndael → gf2p32-rijndael | 11250.33 / 6112.824 |
| gf2p2-fanpaar → gf2p4-fanpaar | 9041.636 / 4062.720 |
| gf2p2-poly → gf2p2-cantor | 769.3984 / 343.7138 |
| gf2p2-poly → gf2p2-normal | 773.7705 / 344.3974 |
| gf2p2-poly → gf2p4-poly | 663.0942 / 300.3158 |
| gf2p32-fanpaar → gf2p64-fanpaar | 4645067 / 2820860 |
| gf2p32-rijndael → gf2p64-rijndael | 72100.25 / 41330.00 |
| gf2p4-fanpaar → gf2p8-fanpaar | 135147.3 / 60008.25 |
| gf2p4-poly → gf2p4-cantor | 3619.321 / 1662.683 |
| gf2p4-poly → gf2p4-normal | 3597.679 / 1650.297 |
| gf2p4-poly → gf2p8-aes | 316.0000 / 169.0635 |
| gf2p8-aes → gf2p16-custom-rijndael | 710.7073 / 337.0727 |
| gf2p8-aes → gf2p16-rijndael | 2600.776 / 1446.371 |
| gf2p8-aes → gf2p8-aes | 2091.191 / 1154.560 |
| gf2p8-aes → gf2p8-cantor | 587.2690 / 305.3114 |
| gf2p8-aes → gf2p8-normal | 587.5000 / 308.5676 |
| gf2p8-aes → gf2p8-rs | 586.8622 / 307.0530 |
| gf2p8-cantor → gf2p16-custom-cantor | 711.8952 / 335.4598 |
| gf2p8-fanpaar → gf2p16-fanpaar | 91833.38 / 55937.75 |
| gf2p8-normal → gf2p16-custom-normal | 715.5203 / 345.2806 |
| gf2p8-rs → gf2p16-custom-rs | 710.2742 / 336.2212 |
| gf3 → gf9 | 61.05477 / 24.19174 |
| gf5 → gf125 | 246.0387 / 112.0153 |
| gf7 → gf49 | 107.4004 / 48.31713 |
| gf9 → gf81 | 2017.744 / 891.1818 |
| gfmaxp → maxprime-squared | 3007.265 / 1866.444 |
| goldilocks → goldilocks-squared | 2534.325 / 1201.494 |
| m31 → qm31 | 1532.895 / 602.4467 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::trace`

</details>

## Query and scalar latency

Black-boxed scalar and getter calls; these report per-call latency, never a throughput.

<a id="query-details-maps-iso-to-images-iso-to-images-latency"></a>

### iso_to_images

**Elements:** 8 · **Sources:** 0 · **Destinations:** 0

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p8-aes → gf2p8-rs | 0.936 / 0.427 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::to_images`, `fgf::Isomorphism::new`

</details>

<a id="query-maps-direct-generator-count-generator-count-latency"></a>

### generator_count

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p8-aes → gf2p8-rs | 23.4 / 14.7 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::generator_count`

</details>

<a id="query-maps-fail-canonical-fail-canonical-latency"></a>

### fail_canonical

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf49 → gf49-alternate | 168 / 98.0 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::new`, `fgf::EmbeddingError::NoCanonicalMap`

</details>

<a id="query-maps-fail-degree-fail-degree-latency"></a>

### fail_degree

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Map pair (S → T) | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf2p16-rijndael → gf2p8-aes | 0.334 / 0.213 | 2/5 / 2/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::new`, `fgf::EmbeddingError::IncompatibleDegree`

</details>

<a id="query-maps-fail-images-fail-images-latency"></a>

### fail_images

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p8-aes → gf2p16-rijndael | 23.4 / 14.7 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::from_images`, `fgf::EmbeddingError::InvalidImages`

</details>

<a id="query-maps-generators-generator-queries-latency"></a>

### generator queries

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf2p16-rijndael | 126.7 / 78.29 | 0/5 / 0/5 |
| gf2p5-padded | 59.39 / 34.64 | 0/5 / 0/5 |
| gf2p64-fanpaar | 784.5 / 491.2 | 0/5 / 0/5 |
| gf2p64-rijndael | 403.0 / 218.0 | 0/5 / 0/5 |
| gf2p8-aes | 42.91 / 26.99 | 0/5 / 0/5 |
| gf49 | 58.49 / 32.77 | 0/5 / 0/5 |
| gf81 | 131.9 / 67.43 | 0/5 / 0/5 |
| goldilocks | 0.334 / 0.213 | 1/5 / 3/5 |
| m31 | 0.334 / 0.213 | 3/5 / 1/5 |
| qm31 | 58.68 / 32.76 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::generator_count`, `fgf::Embedding::generator`

</details>

<a id="query-maps-iso-conversions-iso-conversions-latency"></a>

### iso_conversions

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p16-rijndael → gf2p16-fanpaar | 6175 / 3841 |
| gf2p2-poly → gf2p2-cantor | 234.7 / 118.7 |
| gf2p2-poly → gf2p2-normal | 236.7 / 118.9 |
| gf2p4-poly → gf2p4-cantor | 418.3 / 220.2 |
| gf2p4-poly → gf2p4-normal | 418.7 / 220.4 |
| gf2p8-aes → gf2p8-aes | 175.5 / 92.82 |
| gf2p8-aes → gf2p8-cantor | 181.2 / 100.0 |
| gf2p8-aes → gf2p8-normal | 186.6 / 106.8 |
| gf2p8-aes → gf2p8-rs | 162.5 / 81.01 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::from_embedding`, `fgf::Isomorphism::to_embedding`, `fgf::Isomorphism::inverse`

</details>

<a id="query-maps-relative-degree-relative-degree-latency"></a>

### relative_degree

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Map pair (S → T) | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf2p1-poly → gf2p2-poly | 0.334 / 0.213 | 1/5 / 3/5 |
| gf2p1-poly → gf2p5-padded | 0.334 / 0.213 | 2/5 / 4/5 |
| gf2p16-fanpaar → gf2p32-fanpaar | 0.334 / 0.213 | 2/5 / 3/5 |
| gf2p16-rijndael → gf2p16-fanpaar | 0.334 / 0.213 | 3/5 / 1/5 |
| gf2p16-rijndael → gf2p32-rijndael | 0.334 / 0.213 | 2/5 / 3/5 |
| gf2p2-fanpaar → gf2p4-fanpaar | 0.334 / 0.213 | 2/5 / 2/5 |
| gf2p2-poly → gf2p2-cantor | 0.334 / 0.213 | 2/5 / 4/5 |
| gf2p2-poly → gf2p2-normal | 0.334 / 0.213 | 2/5 / 3/5 |
| gf2p2-poly → gf2p4-poly | 0.334 / 0.213 | 4/5 / 3/5 |
| gf2p32-fanpaar → gf2p64-fanpaar | 0.334 / 0.213 | 2/5 / 1/5 |
| gf2p32-rijndael → gf2p64-rijndael | 0.334 / 0.213 | 0/5 / 2/5 |
| gf2p4-fanpaar → gf2p8-fanpaar | 0.334 / 0.213 | 1/5 / 2/5 |
| gf2p4-poly → gf2p4-cantor | 0.334 / 0.213 | 1/5 / 3/5 |
| gf2p4-poly → gf2p4-normal | 0.334 / 0.213 | 1/5 / 3/5 |
| gf2p4-poly → gf2p8-aes | 0.334 / 0.213 | 1/5 / 3/5 |
| gf2p8-aes → gf2p16-custom-rijndael | 0.334 / 0.213 | 0/5 / 2/5 |
| gf2p8-aes → gf2p16-rijndael | 0.334 / 0.213 | 3/5 / 2/5 |
| gf2p8-aes → gf2p8-aes | 0.334 / 0.213 | 2/5 / 3/5 |
| gf2p8-aes → gf2p8-cantor | 0.334 / 0.213 | 3/5 / 3/5 |
| gf2p8-aes → gf2p8-normal | 0.334 / 0.213 | 2/5 / 2/5 |
| gf2p8-aes → gf2p8-rs | 0.334 / 0.213 | 1/5 / 3/5 |
| gf2p8-cantor → gf2p16-custom-cantor | 0.334 / 0.213 | 4/5 / 3/5 |
| gf2p8-fanpaar → gf2p16-fanpaar | 0.334 / 0.213 | 5/5 / 4/5 |
| gf2p8-normal → gf2p16-custom-normal | 0.334 / 0.213 | 1/5 / 3/5 |
| gf2p8-rs → gf2p16-custom-rs | 0.334 / 0.213 | 2/5 / 4/5 |
| gf3 → gf9 | 0.334 / 0.213 | 0/5 / 2/5 |
| gf5 → gf125 | 0.334 / 0.213 | 0/5 / 4/5 |
| gf7 → gf49 | 0.334 / 0.213 | 1/5 / 3/5 |
| gf9 → gf81 | 0.334 / 0.213 | 1/5 / 4/5 |
| gfmaxp → maxprime-squared | 0.334 / 0.213 | 4/5 / 3/5 |
| goldilocks → goldilocks-squared | 0.334 / 0.213 | 1/5 / 2/5 |
| m31 → qm31 | 0.334 / 0.213 | 2/5 / 2/5 |

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::relative_degree`

</details>

## Preparation and setup

Construction and preparation costs excluded from steady-state panels; owned results are destroyed before the next iteration.

<a id="setup-details-maps-iso-from-images-iso-from-images-latency"></a>

### iso_from_images

**Elements:** 8 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p8-aes → gf2p8-rs | 723 / 395 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::from_images`, `fgf::Isomorphism::to_images`, `fgf::Isomorphism::new`

</details>

<a id="setup-details-maps-iso-then-iso-then-latency"></a>

### iso_then

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p8-aes → gf2p8-rs → gf2p8-aes | 96.4 / 58.4 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::then`, `fgf::Isomorphism::inverse`, `fgf::Isomorphism::new`

</details>

<a id="setup-maps-direct-from-embedding-from-embedding-latency"></a>

### from_embedding

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p8-aes → gf2p8-rs | 638 / 342 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::from_embedding`

</details>

<a id="setup-maps-direct-new-new-latency"></a>

### new

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p8-aes → gf2p8-rs | 1861 / 980.2 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::new`

</details>

## End-to-end compositions

Public compositions from construction through consumption, timed as one call.

<a id="end-to-end-maps-build-build-latency"></a>

### build

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p1-poly → gf2p2-poly | 19.0553 / 11.9430 |
| gf2p1-poly → gf2p5-padded | 19.0497 / 11.9430 |
| gf2p16-fanpaar → gf2p32-fanpaar | 51573.8 / 31256.1 |
| gf2p16-rijndael → gf2p16-fanpaar | 18126.4 / 10784.4 |
| gf2p16-rijndael → gf2p32-rijndael | 675.646 / 339.563 |
| gf2p2-fanpaar → gf2p4-fanpaar | 729.417 / 348.972 |
| gf2p2-poly → gf2p2-cantor | 238.872 / 138.651 |
| gf2p2-poly → gf2p2-normal | 243.986 / 134.397 |
| gf2p2-poly → gf2p4-poly | 973.708 / 514.593 |
| gf2p32-fanpaar → gf2p64-fanpaar | 334229 / 203297 |
| gf2p32-rijndael → gf2p64-rijndael | 2869.16 / 1618.64 |
| gf2p4-fanpaar → gf2p8-fanpaar | 9632.36 / 4317.92 |
| gf2p4-poly → gf2p4-cantor | 355.677 / 192.115 |
| gf2p4-poly → gf2p4-normal | 342.286 / 188.545 |
| gf2p4-poly → gf2p8-aes | 4160.31 / 2315.86 |
| gf2p8-aes → gf2p16-custom-rijndael | 232.043 / 144.783 |
| gf2p8-aes → gf2p16-rijndael | 274.137 / 157.299 |
| gf2p8-aes → gf2p8-aes | 114.183 / 79.1516 |
| gf2p8-aes → gf2p8-cantor | 246.143 / 160.256 |
| gf2p8-aes → gf2p8-normal | 248.177 / 156.856 |
| gf2p8-aes → gf2p8-rs | 1213.88 / 634.255 |
| gf2p8-cantor → gf2p16-custom-cantor | 246.324 / 147.275 |
| gf2p8-fanpaar → gf2p16-fanpaar | 7764.62 / 4696.14 |
| gf2p8-normal → gf2p16-custom-normal | 243.328 / 146.872 |
| gf2p8-rs → gf2p16-custom-rs | 231.954 / 143.367 |
| gf3 → gf9 | 20.3869 / 12.7950 |
| gf5 → gf125 | 19.0505 / 11.9433 |
| gf7 → gf49 | 19.7183 / 11.9437 |
| gf9 → gf81 | 316.232 / 167.559 |
| gfmaxp → maxprime-squared | 20.3863 / 12.7956 |
| goldilocks → goldilocks-squared | 20.3869 / 12.7957 |
| m31 → qm31 | 19.7181 / 11.9438 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::new`

</details>

<a id="end-to-end-maps-from-images-from-images-latency"></a>

### from_images

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p1-poly → gf2p2-poly | 38.8567 / 23.6743 |
| gf2p1-poly → gf2p5-padded | 38.7733 / 23.6731 |
| gf2p16-fanpaar → gf2p32-fanpaar | 102733 / 62252.1 |
| gf2p16-rijndael → gf2p16-fanpaar | 26573.5 / 16213.8 |
| gf2p16-rijndael → gf2p32-rijndael | 1109.91 / 554.518 |
| gf2p2-fanpaar → gf2p4-fanpaar | 1256.30 / 596.969 |
| gf2p2-poly → gf2p2-cantor | 327.817 / 181.715 |
| gf2p2-poly → gf2p2-normal | 331.978 / 179.232 |
| gf2p2-poly → gf2p4-poly | 1055.01 / 550.750 |
| gf2p32-fanpaar → gf2p64-fanpaar | 668191 / 405690 |
| gf2p32-rijndael → gf2p64-rijndael | 5300.26 / 3030.33 |
| gf2p4-fanpaar → gf2p8-fanpaar | 18976.1 / 8520.25 |
| gf2p4-poly → gf2p4-cantor | 528.798 / 282.332 |
| gf2p4-poly → gf2p4-normal | 535.326 / 279.555 |
| gf2p4-poly → gf2p8-aes | 4220.25 / 2343.40 |
| gf2p8-aes → gf2p16-custom-rijndael | 342.375 / 207.276 |
| gf2p8-aes → gf2p16-rijndael | 405.159 / 236.742 |
| gf2p8-aes → gf2p8-aes | 204.459 / 131.755 |
| gf2p8-aes → gf2p8-cantor | 336.126 / 211.114 |
| gf2p8-aes → gf2p8-normal | 330.484 / 208.197 |
| gf2p8-aes → gf2p8-rs | 1293.67 / 684.095 |
| gf2p8-cantor → gf2p16-custom-cantor | 387.056 / 210.612 |
| gf2p8-fanpaar → gf2p16-fanpaar | 15241.3 / 9218.45 |
| gf2p8-normal → gf2p16-custom-normal | 362.765 / 209.620 |
| gf2p8-rs → gf2p16-custom-rs | 346.504 / 207.153 |
| gf3 → gf9 | 40.4440 / 25.3793 |
| gf5 → gf125 | 40.4372 / 25.3786 |
| gf7 → gf49 | 37.7686 / 23.6748 |
| gf9 → gf81 | 460.211 / 232.787 |
| gfmaxp → maxprime-squared | 40.4400 / 24.9537 |
| goldilocks → goldilocks-squared | 40.4414 / 25.3797 |
| m31 → qm31 | 37.7690 / 23.6736 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::from_images`, `fgf::Embedding::to_images`

</details>

<a id="end-to-end-maps-then-then-latency"></a>

### then

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Map pair (S → T) | Latency (ns/op) |
| --- | --- |
| gf2p2-poly → gf2p4-poly → gf2p8-aes | 91.814 / 55.879 |
| gf2p4-fanpaar → gf2p8-fanpaar → gf2p16-fanpaar | 11501 / 7145.7 |
| gf2p8-aes → gf2p16-rijndael → gf2p32-rijndael | 378.30 / 197.63 |
| gf3 → gf9 → gf81 | 46.454 / 29.644 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::Embedding::then`

</details>

## Conversion caller loops

Public per-element `Isomorphism` calls over 64 KiB. Packing inside or outside the timed body is labelled separately.

### gf16-cantor-base

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) |
| --- | --- |
| Convert (packing timed) | 0.0114 / 0.0228 |
| Convert (packing outside timing) | 0.0115 / 0.0231 |

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::apply`

</details>

### gf16-fanpaar16

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) |
| --- | --- |
| Convert (packing timed) | 0.000319 / 0.000494 |
| Convert (packing outside timing) | 0.000319 / 0.000496 |

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::apply`

</details>

### gf16-flat-0x1002d

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) |
| --- | --- |
| Convert (packing timed) | 0.0000469 / 0.0000917 |
| Convert (packing outside timing) | 0.0000469 / 0.0000915 |

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::apply`

</details>

### gf16-flat-0x1100b

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) |
| --- | --- |
| Convert (packing timed) | 0.0000438 / 0.0000854 |
| Convert (packing outside timing) | 0.0000438 / 0.0000854 |

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::apply`

</details>

### gf16-normal-base

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) |
| --- | --- |
| Convert (packing timed) | 0.0114 / 0.0223 |
| Convert (packing outside timing) | 0.0117 / 0.0225 |

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::apply`

</details>

### gf16-rs-quad

**Counted bytes:** 64 KiB · **Elements:** 32,768 · **Sources:** 1 · **Destinations:** 1 · **Coefficients:** Fixed · **fgf routes:** scalar / scalar

| Operation | `fgf` (GiB/s) |
| --- | --- |
| Convert (packing timed) | 0.0114 / 0.0229 |
| Convert (packing outside timing) | 0.0115 / 0.0230 |

<details>
<summary>Exact public calls</summary>

`fgf::Isomorphism::apply`

</details>

## Measurement details

[Campaign, backend routes, and timer-floor interpretation](public-api.md#measurement-details)

Map calls report the scalar route. A process backend is not evidence of SIMD execution for these calls.
