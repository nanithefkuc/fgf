# Queries and prepared-state access

Every result cell reports **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)**. Throughput: higher is better; latency: lower is better. A dash means no measurement.

[All benchmarks](../../../BENCHMARKS.md) · [Field definitions](../../labels.md)

[Public operations index](../public-api.md)

## Find a result

- [Query and scalar latency](#query-and-scalar-latency)
  - [backend parsing](#query-query-backend-backend-parsing-latency)
  - [process backend](#query-query-backend-process-backend-latency)
  - [tier ladder](#query-query-backend-tier-ladder-latency)
  - [field backend](#query-query-backend-for-field-field-backend-latency)
  - [vector elementwise capability](#query-query-backend-vector-elementwise-vector-elementwise-capability-latency)
  - [coeff value read](#query-query-coeff-accessors-coeff-value-read-latency)
  - [coefficient matrix reads](#query-query-coeffmatrix-accessors-coefficient-matrix-reads-latency)
  - [coeff vector reads](#query-query-coeffvec-accessors-coeff-vector-reads-latency)
  - [binary description accessors](#query-query-description-binary-description-accessors-latency)

## Query and scalar latency

Black-boxed scalar and getter calls; these report per-call latency, never a throughput.

<a id="query-query-backend-backend-parsing-latency"></a>

### backend parsing

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| process | 0.334 / 0.213 | 0/5 / 4/5 |

<details>
<summary>Exact public calls</summary>

`fgf::ParseBackendError`, `core::str::FromStr for fgf::Backend`

</details>

<a id="query-query-backend-process-backend-latency"></a>

### process backend

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| process | 1.671 / 0.853 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::backend`, `fgf::KernelBackend::has_native_mul`, `fgf::KernelBackend::has_blocked_rows`, `fgf::Backend::name`

</details>

<a id="query-query-backend-tier-ladder-latency"></a>

### tier ladder

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| process | 0.334 / 0.213 | 1/5 / 3/5 |

<details>
<summary>Exact public calls</summary>

`fgf::FGF_TIERS`

</details>

<a id="query-query-backend-for-field-field-backend-latency"></a>

### field backend

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf101 | 1.006 / 0.533 | 0/5 / 0/5 |
| gf125 | 0.334 / 0.213 | 1/5 / 3/5 |
| gf127 | 1.006 / 0.533 | 0/5 / 0/5 |
| gf131 | 0.334 / 0.213 | 0/5 / 2/5 |
| gf1e9p7 | 1.005 / 0.533 | 0/5 / 0/5 |
| gf2p1-residue | 0.334 / 0.213 | 1/5 / 3/5 |
| gf2p16-custom-cantor | 0.334 / 0.213 | 3/5 / 2/5 |
| gf2p16-custom-normal | 0.334 / 0.213 | 1/5 / 1/5 |
| gf2p16-custom-rijndael | 0.334 / 0.213 | 2/5 / 2/5 |
| gf2p16-custom-rs | 0.334 / 0.213 | 2/5 / 1/5 |
| gf2p16-fanpaar | 1.337 / 0.640 | 0/5 / 0/5 |
| gf2p16-rijndael | 0.780 / 0.427 | 0/5 / 0/5 |
| gf2p32-fanpaar | 1.337 / 0.640 | 0/5 / 0/5 |
| gf2p32-rijndael | 1.003 / 0.569 | 0/5 / 0/5 |
| gf2p64-fanpaar | 1.337 / 0.640 | 0/5 / 0/5 |
| gf2p64-rijndael | 1.003 / 0.569 | 0/5 / 0/5 |
| gf2p8-aes | 0.780 / 0.427 | 0/5 / 0/5 |
| gf2p8-cantor | 0.334 / 0.213 | 0/5 / 1/5 |
| gf2p8-fanpaar | 0.334 / 0.213 | 1/5 / 3/5 |
| gf2p8-normal | 0.334 / 0.213 | 3/5 / 3/5 |
| gf2p8-rs | 0.780 / 0.427 | 0/5 / 0/5 |
| gf49 | 0.334 / 0.213 | 1/5 / 2/5 |
| gf65521 | 0.334 / 0.213 | 1/5 / 0/5 |
| gf65537 | 1.006 / 0.533 | 0/5 / 0/5 |
| gf81 | 0.334 / 0.213 | 2/5 / 2/5 |
| gf9 | 0.334 / 0.213 | 2/5 / 0/5 |
| gfmaxp | 1.006 / 0.533 | 0/5 / 0/5 |
| gfu32max | 1.005 / 0.533 | 0/5 / 0/5 |
| goldilocks | 1.006 / 0.533 | 0/5 / 0/5 |
| goldilocks-residue | 1.006 / 0.533 | 0/5 / 0/5 |
| goldilocks-squared | 0.334 / 0.213 | 1/5 / 2/5 |
| m31 | 1.004 / 0.533 | 0/5 / 0/5 |
| m31-residue | 1.006 / 0.533 | 0/5 / 0/5 |
| maxprime-squared | 0.334 / 0.213 | 1/5 / 3/5 |
| qm31 | 1.006 / 0.533 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::backend_for`, `fgf::FieldKernels::backend`

</details>

<a id="query-query-backend-vector-elementwise-vector-elementwise-capability-latency"></a>

### vector elementwise capability

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf101 | 0.334 / 0.213 | 4/5 / 4/5 |
| gf125 | 0.347 / 0.213 | 0/5 / 3/5 |
| gf127 | 0.334 / 0.213 | 0/5 / 3/5 |
| gf131 | 0.347 / 0.213 | 0/5 / 2/5 |
| gf1e9p7 | 1.003 / 0.640 | 0/5 / 0/5 |
| gf2p1-residue | 0.347 / 0.213 | 0/5 / 3/5 |
| gf2p16-custom-cantor | 0.347 / 0.213 | 0/5 / 3/5 |
| gf2p16-custom-normal | 0.347 / 0.213 | 0/5 / 2/5 |
| gf2p16-custom-rijndael | 0.347 / 0.213 | 0/5 / 5/5 |
| gf2p16-custom-rs | 0.347 / 0.213 | 0/5 / 3/5 |
| gf2p16-fanpaar | 0.334 / 0.213 | 2/5 / 3/5 |
| gf2p16-rijndael | 0.668 / 0.341 | 0/5 / 0/5 |
| gf2p32-fanpaar | 0.334 / 0.213 | 2/5 / 2/5 |
| gf2p32-rijndael | 0.334 / 0.213 | 2/5 / 3/5 |
| gf2p64-fanpaar | 0.334 / 0.213 | 1/5 / 2/5 |
| gf2p64-rijndael | 0.334 / 0.213 | 0/5 / 1/5 |
| gf2p8-aes | 0.668 / 0.341 | 0/5 / 0/5 |
| gf2p8-cantor | 0.347 / 0.213 | 0/5 / 2/5 |
| gf2p8-fanpaar | 0.348 / 0.213 | 0/5 / 2/5 |
| gf2p8-normal | 0.347 / 0.213 | 0/5 / 5/5 |
| gf2p8-rs | 0.891 / 0.533 | 0/5 / 0/5 |
| gf49 | 0.347 / 0.213 | 0/5 / 2/5 |
| gf65521 | 0.347 / 0.213 | 0/5 / 4/5 |
| gf65537 | 1.003 / 0.640 | 0/5 / 0/5 |
| gf81 | 0.347 / 0.213 | 0/5 / 2/5 |
| gf9 | 0.347 / 0.213 | 0/5 / 2/5 |
| gfmaxp | 0.334 / 0.213 | 0/5 / 1/5 |
| gfu32max | 1.003 / 0.640 | 0/5 / 0/5 |
| goldilocks | 1.003 / 0.533 | 0/5 / 0/5 |
| goldilocks-residue | 0.334 / 0.213 | 1/5 / 2/5 |
| goldilocks-squared | 0.348 / 0.213 | 0/5 / 3/5 |
| m31 | 1.003 / 0.533 | 0/5 / 0/5 |
| m31-residue | 1.003 / 0.640 | 0/5 / 0/5 |
| maxprime-squared | 0.347 / 0.213 | 0/5 / 3/5 |
| qm31 | 1.337 / 0.853 | 0/5 / 0/5 |

<details>
<summary>Exact public calls</summary>

`fgf::has_vector_elementwise`, `fgf::kernel::FieldKernels::vector_elementwise_min_bytes`

</details>

<a id="query-query-coeff-accessors-coeff-value-read-latency"></a>

### coeff value read

**Elements:** 1 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) | Floor-limited rounds |
| --- | --- | --- |
| gf2p16-fanpaar | 0.334 / 0.213 | 3/5 / 3/5 |
| gf2p16-rijndael | 0.334 / 0.213 | 1/5 / 4/5 |
| gf2p8-aes | 0.334 / 0.213 | 0/5 / 3/5 |
| gfmaxp | 0.334 / 0.213 | 2/5 / 4/5 |
| m31 | 0.334 / 0.213 | 2/5 / 2/5 |

<details>
<summary>Exact public calls</summary>

`fgf::ops::PreparedCoefficient::value`

</details>

<a id="query-query-coeffmatrix-accessors-coefficient-matrix-reads-latency"></a>

### coefficient matrix reads

**Elements:** 64 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-fanpaar | 50.3 / 27.3 |
| gf2p16-rijndael | 50.6 / 27.3 |
| gf2p8-aes | 38.4 / 19.4 |
| gfmaxp | 147 / 73.2 |
| m31 | 39.3 / 19.6 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffMatrix::source_count`, `fgf::ops::CoeffMatrix::output_count`, `fgf::ops::CoeffMatrix::len`, `fgf::ops::CoeffMatrix::is_empty`, `fgf::ops::CoeffMatrix::get_at`, `fgf::ops::CoeffMatrix::source`, `fgf::ops::CoeffMatrix::values`

</details>

<a id="query-query-coeffvec-accessors-coeff-vector-reads-latency"></a>

### coeff vector reads

**Elements:** 8 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p16-fanpaar | 7.11 / 3.98 |
| gf2p16-rijndael | 7.11 / 3.99 |
| gf2p8-aes | 5.18 / 3.34 |
| gfmaxp | 21.3 / 10.3 |
| m31 | 9.96 / 5.80 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::ops::CoeffVec::len`, `fgf::ops::CoeffVec::is_empty`, `fgf::ops::CoeffVec::get`, `fgf::ops::CoeffVec::coeffs`, `fgf::ops::CoeffVec::values`, `fgf::ops::CoeffVecRef::into_coeffs`, `fgf::ops::CoeffVecRef::values`, `fgf::ops::CoeffRef::value`

</details>

<a id="query-query-description-binary-description-accessors-latency"></a>

### binary description accessors

**Elements:** 0 · **Sources:** 0 · **Destinations:** 0

| Field | Latency (ns/op) |
| --- | --- |
| gf2p8-aes | 261.3 / 132.4 |
| gf2p8-cantor | 1347 / 618.4 |
| gf2p8-normal | 1314 / 598.0 |
| gf2p8-rs | 370.6 / 184.9 |

No measured rounds are at or below the timer-floor control.

<details>
<summary>Exact public calls</summary>

`fgf::BinaryField::DESCRIPTION`, `fgf::BinaryDescription::degree`, `fgf::BinaryDescription::one`, `fgf::BinaryDescription::mul`, `fgf::BinaryDescription::inv`, `fgf::BinaryDescription::trace`, `fgf::BinaryDescription::same_structure`, `fgf::BinaryDescription::is_generator`

</details>

## Measurement details

[Campaign, backend routes, timer floor, and validation](../public-api.md#measurement-details)
