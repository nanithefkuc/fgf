# Encoding and coordinate conversion

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Steady-state operations](#steady-state-operations)
  - [component conversion](#steady-convert-components-component-conversion-latency)
  - [decode](#steady-convert-decode-decode-throughput)
  - [encode](#steady-convert-encode-encode-throughput)
  - [specialized reduction](#steady-convert-fold-reduce-specialized-reduction-latency)
  - [index round trip](#steady-convert-index-index-round-trip-latency)
  - [even_composite](#steady-convert-is-prime-even-composite-latency)
  - [large_prime](#steady-convert-is-prime-large-prime-latency)
  - [semiprime](#steady-convert-is-prime-semiprime-latency)
  - [pack](#steady-convert-pack-pack-throughput)
  - [integer reduction](#steady-convert-reduce-integer-reduction-latency)
  - [residue conversion](#steady-convert-residue-residue-conversion-latency)
  - [unpack](#steady-convert-unpack-unpack-throughput)
- [End-to-end compositions](#end-to-end-compositions)
  - [pack_to_vec](#end-to-end-convert-pack-to-vec-pack-to-vec-throughput)

## Steady-state operations

Repeated public calls on prebuilt fixtures. Explicit prepared coefficients stay outside timing; work internal to the named public call remains inside it.

<a id="steady-convert-components-component-conversion-latency"></a>

### component conversion

**Destinations:** 1

| Field | Elements | Sources | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- | --- | --- |
| gf125 | 3 | 3 | 2.005 / 1.066 | 0/5 / 0/5 |
| gf2p16-custom-cantor | 2 | 2 | 0.432 / 0.213 | 0/5 / 3/5 |
| gf2p16-custom-normal | 2 | 2 | 0.432 / 0.213 | 0/5 / 3/5 |
| gf2p16-custom-rijndael | 2 | 2 | 0.432 / 0.213 | 0/5 / 3/5 |
| gf2p16-custom-rs | 2 | 2 | 0.432 / 0.213 | 0/5 / 4/5 |
| gf2p16-fanpaar | 2 | 2 | 0.432 / 0.213 | 0/5 / 3/5 |
| gf2p16-rijndael | 2 | 2 | 0.432 / 0.213 | 0/5 / 4/5 |
| gf2p2-fanpaar | 2 | 2 | 0.439 / 0.218 | 0/5 / 0/5 |
| gf2p32-fanpaar | 2 | 2 | 0.413 / 0.213 | 0/5 / 1/5 |
| gf2p32-rijndael | 2 | 2 | 0.427 / 0.213 | 0/5 / 2/5 |
| gf2p4-fanpaar | 2 | 2 | 0.439 / 0.219 | 0/5 / 0/5 |
| gf2p5-padded | 5 | 5 | 3.898 / 1.770 | 0/5 / 0/5 |
| gf2p64-fanpaar | 2 | 2 | 48.17 / 26.28 | 0/5 / 0/5 |
| gf2p64-rijndael | 2 | 2 | 46.56 / 24.96 | 0/5 / 0/5 |
| gf2p8-fanpaar | 2 | 2 | 0.439 / 0.226 | 0/5 / 0/5 |
| gf49 | 2 | 2 | 2.143 / 1.137 | 0/5 / 0/5 |
| gf81 | 2 | 2 | 2.865 / 1.486 | 0/5 / 0/5 |
| gf9 | 2 | 2 | 1.337 / 0.683 | 0/5 / 0/5 |
| goldilocks-squared | 2 | 2 | 1.131 / 0.534 | 0/5 / 0/5 |
| maxprime-squared | 2 | 2 | 1.109 / 0.498 | 0/5 / 0/5 |
| qm31 | 2 | 2 | 1.799 / 0.786 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::ExtensionField::from_components`, `fgf::ExtensionField::to_components`

</details>

<a id="steady-convert-decode-decode-throughput"></a>

### decode

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Throughput (GiB/s) |
| --- | --- | --- |
| gf101 | 1 B | 0.892 / 1.456 |
| gf125 | 3 B | 1.392 / 2.620 |
| gf127 | 1 B | 0.616 / 1.186 |
| gf131 | 1 B | 0.890 / 1.456 |
| gf1e9p7 | 4 B | 2.787 / 5.813 |
| gf2p1-residue | 8 B | 6.962 / 11.65 |
| gf2p16-custom-cantor | 2 B | 1.858 / 2.912 |
| gf2p16-custom-normal | 2 B | 1.858 / 2.912 |
| gf2p16-custom-rijndael | 2 B | 1.858 / 2.912 |
| gf2p16-custom-rs | 2 B | 1.858 / 2.912 |
| gf2p16-fanpaar | 2 B | 1.858 / 2.912 |
| gf2p16-rijndael | 2 B | 1.858 / 2.912 |
| gf2p32-fanpaar | 4 B | 3.716 / 5.823 |
| gf2p32-rijndael | 4 B | 3.716 / 5.823 |
| gf2p64-fanpaar | 8 B | 0.148 / 0.284 |
| gf2p64-rijndael | 8 B | 0.151 / 0.295 |
| gf2p8-aes | 1 B | 1.239 / 2.911 |
| gf2p8-cantor | 1 B | 1.239 / 2.911 |
| gf2p8-fanpaar | 1 B | 0.929 / 1.456 |
| gf2p8-normal | 1 B | 1.239 / 2.911 |
| gf2p8-rs | 1 B | 1.239 / 2.911 |
| gf49 | 2 B | 0.679 / 1.285 |
| gf65521 | 2 B | 1.857 / 2.912 |
| gf65537 | 4 B | 3.185 / 5.823 |
| gf81 | 4 B | 1.953 / 3.410 |
| gf9 | 2 B | 1.238 / 2.381 |
| gfmaxp | 8 B | 6.731 / 11.65 |
| gfu32max | 4 B | 2.784 / 5.240 |
| goldilocks | 8 B | 7.431 / 15.88 |
| goldilocks-residue | 8 B | 6.727 / 11.65 |
| goldilocks-squared | 16 B | 7.558 / 13.97 |
| m31 | 4 B | 3.715 / 6.709 |
| m31-residue | 4 B | 2.786 / 5.226 |
| maxprime-squared | 16 B | 8.917 / 17.47 |
| qm31 | 8 B | 2.409 / 4.954 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldBuffer::decode`

</details>

<a id="steady-convert-encode-encode-throughput"></a>

### encode

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Counted bytes | Throughput (GiB/s) |
| --- | --- | --- |
| gf101 | 1 B | 0.697 / 1.092 |
| gf125 | 3 B | 1.509 / 2.620 |
| gf127 | 1 B | 0.697 / 1.092 |
| gf131 | 1 B | 0.697 / 1.092 |
| gf1e9p7 | 4 B | 2.787 / 4.368 |
| gf2p1-residue | 8 B | 5.574 / 8.735 |
| gf2p16-custom-cantor | 2 B | 1.394 / 2.184 |
| gf2p16-custom-normal | 2 B | 1.394 / 2.184 |
| gf2p16-custom-rijndael | 2 B | 1.394 / 2.184 |
| gf2p16-custom-rs | 2 B | 1.394 / 2.184 |
| gf2p16-fanpaar | 2 B | 1.393 / 2.184 |
| gf2p16-rijndael | 2 B | 1.393 / 2.184 |
| gf2p32-fanpaar | 4 B | 2.787 / 4.368 |
| gf2p32-rijndael | 4 B | 2.787 / 4.368 |
| gf2p64-fanpaar | 8 B | 5.574 / 8.735 |
| gf2p64-rijndael | 8 B | 5.574 / 8.735 |
| gf2p8-aes | 1 B | 0.697 / 1.092 |
| gf2p8-cantor | 1 B | 0.697 / 1.092 |
| gf2p8-fanpaar | 1 B | 0.697 / 1.092 |
| gf2p8-normal | 1 B | 0.697 / 1.092 |
| gf2p8-rs | 1 B | 0.697 / 1.092 |
| gf49 | 2 B | 1.320 / 2.184 |
| gf65521 | 2 B | 1.394 / 2.184 |
| gf65537 | 4 B | 2.787 / 4.368 |
| gf81 | 4 B | 1.566 / 2.911 |
| gf9 | 2 B | 1.319 / 2.184 |
| gfmaxp | 8 B | 5.574 / 8.735 |
| gfu32max | 4 B | 2.787 / 4.368 |
| goldilocks | 8 B | 5.574 / 8.735 |
| goldilocks-residue | 8 B | 5.574 / 8.735 |
| goldilocks-squared | 16 B | 2.123 / 3.494 |
| m31 | 4 B | 2.787 / 4.368 |
| m31-residue | 4 B | 2.787 / 4.368 |
| maxprime-squared | 16 B | 2.123 / 3.494 |
| qm31 | 8 B | 5.574 / 8.735 |

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldBuffer::encode`

</details>

<a id="steady-convert-fold-reduce-specialized-reduction-latency"></a>

### specialized reduction

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| goldilocks | 2.334 / 0.928 |
| m31 | 0.835 / 0.427 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

| Fields | Public callable paths |
| --- | --- |
| goldilocks | `fgf::goldilocks::reduce`, `fgf::goldilocks::reduce_wide` |
| m31 | `fgf::mersenne31::reduce` |

</details>

<a id="steady-convert-index-index-round-trip-latency"></a>

### index round trip

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 0.936 / 0.427 |
| gf125 | 0.937 / 0.489 |
| gf127 | 0.936 / 0.427 |
| gf131 | 0.910 / 0.427 |
| gf1e9p7 | 0.910 / 0.427 |
| gf2p1-poly | 0.937 / 0.489 |
| gf2p1-residue | 0.942 / 0.489 |
| gf2p16-custom-cantor | 0.937 / 0.489 |
| gf2p16-custom-normal | 0.937 / 0.489 |
| gf2p16-custom-rijndael | 0.937 / 0.489 |
| gf2p16-custom-rs | 0.937 / 0.489 |
| gf2p16-fanpaar | 0.937 / 0.489 |
| gf2p16-rijndael | 0.937 / 0.489 |
| gf2p2-cantor | 0.937 / 0.489 |
| gf2p2-fanpaar | 0.937 / 0.489 |
| gf2p2-normal | 0.937 / 0.489 |
| gf2p2-poly | 0.937 / 0.489 |
| gf2p32-fanpaar | 25.13 / 13.37 |
| gf2p32-rijndael | 24.51 / 12.77 |
| gf2p4-cantor | 0.937 / 0.489 |
| gf2p4-fanpaar | 0.937 / 0.489 |
| gf2p4-normal | 0.937 / 0.485 |
| gf2p4-poly | 0.937 / 0.485 |
| gf2p5-padded | 0.937 / 0.489 |
| gf2p64-fanpaar | 217.1 / 109.3 |
| gf2p64-rijndael | 207.6 / 106.7 |
| gf2p8-aes | 0.937 / 0.489 |
| gf2p8-cantor | 0.937 / 0.493 |
| gf2p8-fanpaar | 0.937 / 0.493 |
| gf2p8-normal | 0.937 / 0.493 |
| gf2p8-rs | 0.937 / 0.493 |
| gf49 | 0.936 / 0.427 |
| gf65521 | 0.910 / 0.427 |
| gf65537 | 0.937 / 0.489 |
| gf81 | 0.942 / 0.489 |
| gf9 | 0.910 / 0.427 |
| gfmaxp | 0.910 / 0.427 |
| gfu32max | 0.936 / 0.427 |
| goldilocks | 0.914 / 0.427 |
| goldilocks-residue | 0.910 / 0.427 |
| goldilocks-squared | 0.916 / 0.427 |
| m31 | 0.936 / 0.427 |
| m31-residue | 0.936 / 0.427 |
| maxprime-squared | 0.910 / 0.427 |
| qm31 | 0.935 / 0.427 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::from_index`, `fgf::field::FieldElem::to_index`, `fgf::RangeError`

</details>

<a id="steady-convert-is-prime-even-composite-latency"></a>

### even_composite

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| u64 | 2.34 / 1.92 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::is_prime`

</details>

<a id="steady-convert-is-prime-large-prime-latency"></a>

### large_prime

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| u64 | 4715 / 2847 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::is_prime`

</details>

<a id="steady-convert-is-prime-semiprime-latency"></a>

### semiprime

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| u64 | 665 / 342 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::is_prime`

</details>

<a id="steady-convert-pack-pack-throughput"></a>

### pack

**Elements:** 8,192 · **Sources:** 8,192 · **Destinations:** 1

| Field | Counted bytes | Throughput (GiB/s) |
| --- | --- | --- |
| gf101 | 8 KiB | 5707.58 / 8944.38 |
| gf125 | 24 KiB | 17122.6 / 26833.3 |
| gf127 | 8 KiB | 5707.66 / 8944.39 |
| gf131 | 8 KiB | 5707.58 / 8944.20 |
| gf1e9p7 | 32 KiB | 22830.4 / 35778.0 |
| gf2p1-residue | 64 KiB | 45660.3 / 71555.8 |
| gf2p16-custom-cantor | 16 KiB | 11415.5 / 17889.1 |
| gf2p16-custom-normal | 16 KiB | 11415.1 / 17889.1 |
| gf2p16-custom-rijndael | 16 KiB | 11415.4 / 17889.1 |
| gf2p16-custom-rs | 16 KiB | 11415.4 / 17888.7 |
| gf2p16-fanpaar | 16 KiB | 11415.1 / 17888.6 |
| gf2p16-rijndael | 16 KiB | 11415.8 / 17888.6 |
| gf2p32-fanpaar | 32 KiB | 22828.8 / 35777.5 |
| gf2p32-rijndael | 32 KiB | 22829.5 / 35776.2 |
| gf2p64-fanpaar | 64 KiB | 45663.3 / 71553.7 |
| gf2p64-rijndael | 64 KiB | 45662.3 / 71555.9 |
| gf2p8-aes | 8 KiB | 5707.42 / 8944.40 |
| gf2p8-cantor | 8 KiB | 5707.63 / 8944.31 |
| gf2p8-fanpaar | 8 KiB | 5707.55 / 8944.44 |
| gf2p8-normal | 8 KiB | 5707.78 / 8944.41 |
| gf2p8-rs | 8 KiB | 5707.63 / 8944.38 |
| gf49 | 16 KiB | 11415.1 / 17888.6 |
| gf65521 | 16 KiB | 11415.1 / 17889.3 |
| gf65537 | 32 KiB | 22830.8 / 35777.9 |
| gf81 | 32 KiB | 22831.1 / 35778.6 |
| gf9 | 16 KiB | 11414.9 / 17889.1 |
| gfmaxp | 64 KiB | 45661.8 / 71552.0 |
| gfu32max | 32 KiB | 22830.0 / 35776.7 |
| goldilocks | 64 KiB | 45660.6 / 71555.7 |
| goldilocks-residue | 64 KiB | 45660.9 / 71555.8 |
| goldilocks-squared | 128 KiB | 91319.9 / 143113 |
| m31 | 32 KiB | 22831.5 / 35779.3 |
| m31-residue | 32 KiB | 22831.1 / 35778.4 |
| maxprime-squared | 128 KiB | 91323.5 / 143110 |
| qm31 | 64 KiB | 45661.0 / 71556.1 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::pack`

</details>

<a id="steady-convert-reduce-integer-reduction-latency"></a>

### integer reduction

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 668.4 / 284.5 |
| gf127 | 838.4 / 395.8 |
| gf131 | 663.1 / 296.8 |
| gf1e9p7 | 4257 / 1920 |
| gf2p1-residue | 4.345 / 2.559 |
| gf65521 | 2123 / 987.4 |
| gf65537 | 1478 / 569.7 |
| gfmaxp | 9399 / 5705 |
| gfu32max | 4745 / 2548 |
| goldilocks-residue | 8685 / 4652 |
| m31-residue | 4624 / 2478 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::prime::reduce`, `fgf::field::prime::reduce_wide`

</details>

<a id="steady-convert-residue-residue-conversion-latency"></a>

### residue conversion

**Elements:** 1 · **Sources:** 1 · **Destinations:** 1

| Field | Latency (ns/op) |
| --- | --- |
| gf101 | 0.637 / 0.320 |
| gf125 | 0.640 / 0.320 |
| gf127 | 0.637 / 0.320 |
| gf131 | 0.640 / 0.320 |
| gf1e9p7 | 0.640 / 0.320 |
| gf2p1-poly | 0.637 / 0.320 |
| gf2p1-residue | 0.637 / 0.320 |
| gf2p16-custom-cantor | 0.640 / 0.320 |
| gf2p16-custom-normal | 0.640 / 0.320 |
| gf2p16-custom-rijndael | 0.640 / 0.320 |
| gf2p16-custom-rs | 0.640 / 0.320 |
| gf2p16-fanpaar | 0.640 / 0.320 |
| gf2p16-rijndael | 0.640 / 0.320 |
| gf2p2-cantor | 0.634 / 0.320 |
| gf2p2-fanpaar | 0.637 / 0.320 |
| gf2p2-normal | 0.640 / 0.320 |
| gf2p2-poly | 0.637 / 0.320 |
| gf2p32-fanpaar | 0.640 / 0.320 |
| gf2p32-rijndael | 0.640 / 0.320 |
| gf2p4-cantor | 0.634 / 0.320 |
| gf2p4-fanpaar | 0.640 / 0.320 |
| gf2p4-normal | 0.637 / 0.320 |
| gf2p4-poly | 0.637 / 0.320 |
| gf2p5-padded | 0.640 / 0.320 |
| gf2p64-fanpaar | 39.38 / 19.71 |
| gf2p64-rijndael | 36.84 / 19.46 |
| gf2p8-aes | 0.637 / 0.320 |
| gf2p8-cantor | 0.637 / 0.320 |
| gf2p8-fanpaar | 0.640 / 0.320 |
| gf2p8-normal | 0.637 / 0.320 |
| gf2p8-rs | 0.637 / 0.320 |
| gf49 | 0.640 / 0.320 |
| gf65521 | 0.640 / 0.320 |
| gf65537 | 0.640 / 0.320 |
| gf81 | 0.640 / 0.320 |
| gf9 | 0.640 / 0.320 |
| gfmaxp | 0.637 / 0.320 |
| gfu32max | 0.640 / 0.320 |
| goldilocks | 0.634 / 0.320 |
| goldilocks-residue | 0.634 / 0.320 |
| goldilocks-squared | 0.637 / 0.320 |
| m31 | 0.640 / 0.320 |
| m31-residue | 0.640 / 0.320 |
| maxprime-squared | 0.637 / 0.320 |
| qm31 | 0.640 / 0.320 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::field::FieldElem::from_residue`, `fgf::field::FieldElem::to_residue`, `fgf::RangeError`

</details>

<a id="steady-convert-unpack-unpack-throughput"></a>

### unpack

**Elements:** 8,192 · **Sources:** 1 · **Destinations:** 8,192

| Field | Counted bytes | Throughput (GiB/s) |
| --- | --- | --- |
| gf101 | 8 KiB | 5707.93 / 8944.49 |
| gf125 | 24 KiB | 17122.4 / 26833.7 |
| gf127 | 8 KiB | 5707.72 / 8944.71 |
| gf131 | 8 KiB | 5707.51 / 8944.43 |
| gf1e9p7 | 32 KiB | 22830.5 / 35777.9 |
| gf2p1-residue | 64 KiB | 45661.7 / 71554.8 |
| gf2p16-custom-cantor | 16 KiB | 11415.3 / 17889.4 |
| gf2p16-custom-normal | 16 KiB | 11415.5 / 17889.4 |
| gf2p16-custom-rijndael | 16 KiB | 11415.4 / 17889.5 |
| gf2p16-custom-rs | 16 KiB | 11415.5 / 17889.2 |
| gf2p16-fanpaar | 16 KiB | 11415.0 / 17889.4 |
| gf2p16-rijndael | 16 KiB | 11415.6 / 17889.3 |
| gf2p32-fanpaar | 32 KiB | 35.9276 / 57.9720 |
| gf2p32-rijndael | 32 KiB | 35.9251 / 57.9813 |
| gf2p64-fanpaar | 64 KiB | 0.14581 / 0.28001 |
| gf2p64-rijndael | 64 KiB | 0.15199 / 0.29892 |
| gf2p8-aes | 8 KiB | 5707.41 / 8943.90 |
| gf2p8-cantor | 8 KiB | 5707.46 / 8944.37 |
| gf2p8-fanpaar | 8 KiB | 5707.51 / 8944.26 |
| gf2p8-normal | 8 KiB | 5707.57 / 8944.38 |
| gf2p8-rs | 8 KiB | 5707.49 / 8944.11 |
| gf49 | 16 KiB | 11415.2 / 17888.8 |
| gf65521 | 16 KiB | 11415.0 / 17889.1 |
| gf65537 | 32 KiB | 22829.6 / 35778.1 |
| gf81 | 32 KiB | 2.04187 / 3.54924 |
| gf9 | 16 KiB | 11415.3 / 17888.5 |
| gfmaxp | 64 KiB | 45660.4 / 71555.3 |
| gfu32max | 32 KiB | 22830.1 / 35778.4 |
| goldilocks | 64 KiB | 45659.7 / 71554.5 |
| goldilocks-residue | 64 KiB | 45657.7 / 71552.9 |
| goldilocks-squared | 128 KiB | 121753 / 190800 |
| m31 | 32 KiB | 22829.8 / 35778.4 |
| m31-residue | 32 KiB | 22830.2 / 35777.8 |
| maxprime-squared | 128 KiB | 8.31719 / 23.2252 |
| qm31 | 64 KiB | 3.54972 / 6.38437 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::unpack`

</details>

## End-to-end compositions

Public compositions from construction through consumption, timed as one call.

<a id="end-to-end-convert-pack-to-vec-pack-to-vec-throughput"></a>

### pack_to_vec

**Elements:** 8,192 · **Sources:** 8,192 · **Destinations:** 1

| Field | Counted bytes | Throughput (GiB/s) |
| --- | --- | --- |
| gf101 | 8 KiB | 12053.3 / 21811.5 |
| gf125 | 24 KiB | 36160.3 / 59283.7 |
| gf127 | 8 KiB | 12163.1 / 21811.5 |
| gf131 | 8 KiB | 12053.3 / 21812.4 |
| gf1e9p7 | 32 KiB | 47908.7 / 78868.5 |
| gf2p1-residue | 64 KiB | 96425.0 / 157738 |
| gf2p16-custom-cantor | 16 KiB | 23968.7 / 39302.8 |
| gf2p16-custom-normal | 16 KiB | 23968.4 / 39302.2 |
| gf2p16-custom-rijndael | 16 KiB | 23968.4 / 39301.8 |
| gf2p16-custom-rs | 16 KiB | 23968.4 / 39304.3 |
| gf2p16-fanpaar | 16 KiB | 23968.6 / 39302.9 |
| gf2p16-rijndael | 16 KiB | 23968.3 / 39300.9 |
| gf2p32-fanpaar | 32 KiB | 48350.1 / 95406.0 |
| gf2p32-rijndael | 32 KiB | 48248.7 / 95406.4 |
| gf2p64-fanpaar | 64 KiB | 96684.2 / 190811 |
| gf2p64-rijndael | 64 KiB | 96653.5 / 190813 |
| gf2p8-aes | 8 KiB | 12053.4 / 21812.1 |
| gf2p8-cantor | 8 KiB | 12053.2 / 21809.9 |
| gf2p8-fanpaar | 8 KiB | 12053.0 / 21810.2 |
| gf2p8-normal | 8 KiB | 12052.7 / 21810.9 |
| gf2p8-rs | 8 KiB | 12053.2 / 21811.8 |
| gf49 | 16 KiB | 23953.6 / 39303.5 |
| gf65521 | 16 KiB | 24085.4 / 39303.7 |
| gf65537 | 32 KiB | 48339.3 / 78866.3 |
| gf81 | 32 KiB | 48319.0 / 95405.1 |
| gf9 | 16 KiB | 24096.6 / 39303.8 |
| gfmaxp | 64 KiB | 95889.3 / 157732 |
| gfu32max | 32 KiB | 48298.9 / 78868.2 |
| goldilocks | 64 KiB | 95888.4 / 157729 |
| goldilocks-residue | 64 KiB | 95656.6 / 157735 |
| goldilocks-squared | 128 KiB | 192857 / 315461 |
| m31 | 32 KiB | 47943.3 / 78864.4 |
| m31-residue | 32 KiB | 48210.1 / 78862.6 |
| maxprime-squared | 128 KiB | 193373 / 315461 |
| qm31 | 64 KiB | 96428.8 / 157741 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::pack_to_vec`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
