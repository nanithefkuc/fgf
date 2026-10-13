# Generic prime residues (v3)

Every numeric cell uses **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)** order; unavailable measurements are `-`. Shared context and reproduction: [BENCHMARKS.md](../../BENCHMARKS.md). [Field labels](../labels.md) fix modulus and lane representation.

Current public calls under the routed request. Backend factors live in the [comparison](../comparison/residue.md); generic residues have no released-v2 counterpart.

## Setup

| Setting | Value |
| --- | --- |
| Revision | `f3cd58c39e6e3672216912dd3e6131da1c4b670a` |
| Toolchain | 1.98.1 |
| Aggregation | Median of five per-run placement medians. |
| Placements | Offsets 0, 16, 32 within each run. |
| Order | Routed, AVX2, scalar requests rotate within each round. |
| Resolved process backend (v3) | v4x / - |
| Resolved process backend (avx2) | v3 / - |
| Resolved process backend (scalar) | scalar / - |

## Self-timings

### Prepared single-row multiplication

Numerator: row bytes per call.

<details>
<summary>mul_into_with</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res101 | 0.842 / - | 1.867 / - | 2.443 / - | 8.847 / - | 24.57 / - | 43.13 / - |
| res131 | 1.389 / - | 3.170 / - | 4.434 / - | 5.650 / - | 6.436 / - | 6.350 / - |
| res65521 | 1.435 / - | 1.607 / - | 1.325 / - | 1.511 / - | 1.681 / - | 1.729 / - |
| res1e9p7 | 2.698 / - | 5.570 / - | 9.655 / - | 14.24 / - | 16.48 / - | 16.97 / - |
| res1280000003 | 2.783 / - | 5.939 / - | 9.630 / - | 14.33 / - | 16.60 / - | 16.99 / - |
| reswordmax | 2.784 / - | 5.941 / - | 9.798 / - | 14.28 / - | 16.67 / - | 16.97 / - |
| reswide | 3.817 / - | 5.024 / - | 5.327 / - | 5.485 / - | 5.541 / - | 5.610 / - |
| res2 | 12.73 / - | 22.26 / - | 39.60 / - | 64.72 / - | 93.50 / - | 136.3 / - |

</details>

<details>
<summary>mul_add_with</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res101 | 0.484 / - | 0.684 / - | 1.682 / - | 7.424 / - | 12.65 / - | 12.09 / - |
| res131 | 0.646 / - | 0.895 / - | 1.026 / - | 1.050 / - | 1.066 / - | 1.071 / - |
| res65521 | 0.824 / - | 0.874 / - | 0.804 / - | 0.841 / - | 0.889 / - | 0.898 / - |
| res1e9p7 | 1.849 / - | 3.864 / - | 6.017 / - | 6.894 / - | 7.491 / - | 8.058 / - |
| res1280000003 | 1.856 / - | 3.864 / - | 6.070 / - | 6.993 / - | 7.599 / - | 8.064 / - |
| reswordmax | 1.842 / - | 3.903 / - | 6.096 / - | 6.929 / - | 7.484 / - | 8.057 / - |
| reswide | 3.474 / - | 4.341 / - | 4.512 / - | 4.668 / - | 4.805 / - | 4.835 / - |
| res2 | 7.118 / - | 8.657 / - | 9.342 / - | 8.148 / - | 9.195 / - | 9.546 / - |

</details>

<details>
<summary>mul_assign_with</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res101 | 0.815 / - | 1.849 / - | 2.371 / - | 5.117 / - | 14.84 / - | 28.90 / - |
| res131 | 1.112 / - | 2.966 / - | 4.453 / - | 5.792 / - | 6.264 / - | 6.004 / - |
| res65521 | 1.292 / - | 1.405 / - | 1.251 / - | 1.450 / - | 1.596 / - | 1.645 / - |
| res1e9p7 | 3.563 / - | 5.822 / - | 7.555 / - | 9.016 / - | 13.45 / - | 14.71 / - |
| res1280000003 | 3.563 / - | 5.977 / - | 7.598 / - | 9.052 / - | 13.45 / - | 14.66 / - |
| reswordmax | 3.564 / - | 5.975 / - | 7.604 / - | 9.061 / - | 13.45 / - | 14.69 / - |
| reswide | 2.170 / - | 3.194 / - | 4.064 / - | 4.652 / - | 5.316 / - | 5.326 / - |
| res2 | 22.12 / - | 32.33 / - | 59.28 / - | 74.75 / - | 101.1 / - | 93.53 / - |

</details>

### One-shot single-row multiplication

Numerator: row bytes per call.

<details>
<summary>mul_into</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res101 | 0.849 / - | 1.872 / - | 2.443 / - | 8.900 / - | 23.75 / - | 43.20 / - |
| res131 | 1.389 / - | 3.179 / - | 4.685 / - | 6.042 / - | 6.432 / - | 6.372 / - |
| res65521 | 1.483 / - | 1.599 / - | 1.347 / - | 1.518 / - | 1.682 / - | 1.730 / - |
| res1e9p7 | 2.700 / - | 5.824 / - | 9.652 / - | 14.26 / - | 16.49 / - | 16.95 / - |
| res1280000003 | 2.783 / - | 5.932 / - | 9.687 / - | 14.35 / - | 16.58 / - | 16.98 / - |
| reswordmax | 2.783 / - | 5.938 / - | 9.764 / - | 14.28 / - | 16.67 / - | 16.95 / - |
| reswide | 3.806 / - | 5.058 / - | 5.316 / - | 5.486 / - | 5.540 / - | 5.610 / - |
| res2 | 12.73 / - | 22.28 / - | 39.61 / - | 64.74 / - | 94.92 / - | 136.3 / - |

</details>

### Elementwise multiplication

Numerator: one operand's bytes per call.

<details>
<summary>mul_elementwise</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res1e9p7 | 1.08 / - | 2.90 / - | 4.83 / - | 5.74 / - | 5.93 / - | 6.17 / - |
| res1280000003 | 1.40 / - | 2.89 / - | 4.83 / - | 5.74 / - | 6.09 / - | 6.17 / - |
| reswordmax | 1.23 / - | 2.89 / - | 4.83 / - | 5.75 / - | 5.92 / - | 6.17 / - |

</details>

<details>
<summary>mul_elementwise_assign</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res1e9p7 | 2.05 / - | 3.81 / - | 5.40 / - | 5.83 / - | 6.01 / - | 6.22 / - |
| res1280000003 | 2.05 / - | 3.82 / - | 5.43 / - | 5.81 / - | 5.98 / - | 6.23 / - |
| reswordmax | 2.04 / - | 3.82 / - | 5.43 / - | 5.82 / - | 5.93 / - | 6.22 / - |

</details>

### Matrix multiplication

Four outputs; numerator: row bytes × sources × outputs. Each lane occupies four bytes.

<details>
<summary>mul_into_matrix_with</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 6.51 / - | 7.89 / - |
| res65537 | 8 | 10.6 / - | 12.7 / - |
| res65537 | 16 | 12.4 / - | 14.4 / - |
| res65537 | 32 | 13.2 / - | 15.2 / - |
| res67m | 4 | 6.37 / - | 7.88 / - |
| res67m | 8 | 10.6 / - | 12.4 / - |
| res67m | 16 | 12.4 / - | 14.4 / - |
| res67m | 32 | 13.1 / - | 15.2 / - |
| res268m | 4 | 6.36 / - | 7.88 / - |
| res268m | 8 | 10.7 / - | 12.9 / - |
| res268m | 16 | 12.4 / - | 14.4 / - |
| res268m | 32 | 12.6 / - | 14.2 / - |
| res536m | 4 | 6.35 / - | 7.87 / - |
| res536m | 8 | 10.4 / - | 12.4 / - |
| res536m | 16 | 11.2 / - | 12.8 / - |
| res536m | 32 | 11.3 / - | 12.5 / - |
| res858m | 4 | 6.55 / - | 7.89 / - |
| res858m | 8 | 9.10 / - | 10.6 / - |
| res858m | 16 | 9.35 / - | 10.4 / - |
| res858m | 32 | 9.89 / - | 10.7 / - |
| res1e9p7 | 4 | 6.44 / - | 7.88 / - |
| res1e9p7 | 8 | 8.97 / - | 10.4 / - |
| res1e9p7 | 16 | 9.36 / - | 10.5 / - |
| res1e9p7 | 32 | 9.40 / - | 10.2 / - |
| res1280000003 | 4 | 6.50 / - | 7.89 / - |
| res1280000003 | 8 | 6.51 / - | 8.02 / - |
| res1280000003 | 16 | 6.73 / - | 8.09 / - |
| res1280000003 | 32 | 6.75 / - | 8.13 / - |
| reswordmax | 4 | 6.52 / - | 7.89 / - |
| reswordmax | 8 | 6.60 / - | 8.02 / - |
| reswordmax | 16 | 6.74 / - | 8.09 / - |
| reswordmax | 32 | 6.74 / - | 8.12 / - |

</details>

<details>
<summary>mul_add_matrix_with</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 6.68 / - | 8.16 / - |
| res65537 | 8 | 10.2 / - | 12.1 / - |
| res65537 | 16 | 12.1 / - | 13.7 / - |
| res65537 | 32 | 12.9 / - | 15.0 / - |
| res67m | 4 | 6.50 / - | 8.15 / - |
| res67m | 8 | 10.2 / - | 12.3 / - |
| res67m | 16 | 12.1 / - | 13.8 / - |
| res67m | 32 | 12.9 / - | 15.0 / - |
| res268m | 4 | 6.49 / - | 8.16 / - |
| res268m | 8 | 10.2 / - | 12.2 / - |
| res268m | 16 | 12.0 / - | 13.8 / - |
| res268m | 32 | 12.5 / - | 14.0 / - |
| res536m | 4 | 6.65 / - | 8.15 / - |
| res536m | 8 | 10.3 / - | 12.3 / - |
| res536m | 16 | 10.9 / - | 12.3 / - |
| res536m | 32 | 11.1 / - | 12.3 / - |
| res858m | 4 | 6.69 / - | 8.16 / - |
| res858m | 8 | 8.64 / - | 10.1 / - |
| res858m | 16 | 9.16 / - | 10.0 / - |
| res858m | 32 | 9.77 / - | 10.6 / - |
| res1e9p7 | 4 | 6.53 / - | 8.16 / - |
| res1e9p7 | 8 | 8.70 / - | 10.2 / - |
| res1e9p7 | 16 | 9.20 / - | 9.99 / - |
| res1e9p7 | 32 | 9.32 / - | 10.0 / - |
| res1280000003 | 4 | 6.81 / - | 8.16 / - |
| res1280000003 | 8 | 6.65 / - | 8.16 / - |
| res1280000003 | 16 | 6.74 / - | 8.16 / - |
| res1280000003 | 32 | 6.76 / - | 8.16 / - |
| reswordmax | 4 | 6.65 / - | 8.15 / - |
| reswordmax | 8 | 6.66 / - | 8.16 / - |
| reswordmax | 16 | 6.73 / - | 8.16 / - |
| reswordmax | 32 | 6.67 / - | 8.16 / - |

</details>

<details>
<summary>Four prepared mul_into_gather_with calls</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 7.48 / - | 9.33 / - |
| res65537 | 8 | 7.14 / - | 8.68 / - |
| res65537 | 16 | 6.94 / - | 8.39 / - |
| res65537 | 32 | 6.80 / - | 8.25 / - |
| res67m | 4 | 7.56 / - | 9.33 / - |
| res67m | 8 | 7.11 / - | 8.68 / - |
| res67m | 16 | 6.78 / - | 8.39 / - |
| res67m | 32 | 6.68 / - | 8.25 / - |
| res268m | 4 | 7.55 / - | 9.33 / - |
| res268m | 8 | 7.14 / - | 8.68 / - |
| res268m | 16 | 6.78 / - | 8.38 / - |
| res268m | 32 | 6.70 / - | 8.25 / - |
| res536m | 4 | 7.50 / - | 9.33 / - |
| res536m | 8 | 7.18 / - | 8.68 / - |
| res536m | 16 | 6.84 / - | 8.38 / - |
| res536m | 32 | 6.70 / - | 8.25 / - |
| res858m | 4 | 7.70 / - | 9.34 / - |
| res858m | 8 | 7.33 / - | 8.68 / - |
| res858m | 16 | 6.92 / - | 8.39 / - |
| res858m | 32 | 6.80 / - | 8.25 / - |
| res1e9p7 | 4 | 7.67 / - | 9.33 / - |
| res1e9p7 | 8 | 7.05 / - | 8.68 / - |
| res1e9p7 | 16 | 6.82 / - | 8.38 / - |
| res1e9p7 | 32 | 6.70 / - | 8.25 / - |
| res1280000003 | 4 | 7.51 / - | 9.33 / - |
| res1280000003 | 8 | 7.19 / - | 8.68 / - |
| res1280000003 | 16 | 6.98 / - | 8.39 / - |
| res1280000003 | 32 | 6.86 / - | 8.25 / - |
| reswordmax | 4 | 7.50 / - | 9.33 / - |
| reswordmax | 8 | 7.18 / - | 8.68 / - |
| reswordmax | 16 | 7.02 / - | 8.39 / - |
| reswordmax | 32 | 6.85 / - | 8.25 / - |

</details>

<details>
<summary>Four prepared mul_add_gather_with calls</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 6.83 / - | 8.10 / - |
| res65537 | 8 | 6.69 / - | 8.11 / - |
| res65537 | 16 | 6.68 / - | 8.11 / - |
| res65537 | 32 | 6.69 / - | 8.11 / - |
| res67m | 4 | 6.56 / - | 8.10 / - |
| res67m | 8 | 6.61 / - | 8.11 / - |
| res67m | 16 | 6.60 / - | 8.11 / - |
| res67m | 32 | 6.55 / - | 8.11 / - |
| res268m | 4 | 6.67 / - | 8.10 / - |
| res268m | 8 | 6.72 / - | 8.11 / - |
| res268m | 16 | 6.63 / - | 8.11 / - |
| res268m | 32 | 6.60 / - | 8.11 / - |
| res536m | 4 | 6.62 / - | 8.10 / - |
| res536m | 8 | 6.61 / - | 8.11 / - |
| res536m | 16 | 6.59 / - | 8.11 / - |
| res536m | 32 | 6.59 / - | 8.11 / - |
| res858m | 4 | 6.66 / - | 8.10 / - |
| res858m | 8 | 6.74 / - | 8.11 / - |
| res858m | 16 | 6.75 / - | 8.11 / - |
| res858m | 32 | 6.75 / - | 8.11 / - |
| res1e9p7 | 4 | 6.53 / - | 8.10 / - |
| res1e9p7 | 8 | 6.63 / - | 8.11 / - |
| res1e9p7 | 16 | 6.65 / - | 8.11 / - |
| res1e9p7 | 32 | 6.60 / - | 8.11 / - |
| res1280000003 | 4 | 6.90 / - | 8.10 / - |
| res1280000003 | 8 | 6.67 / - | 8.11 / - |
| res1280000003 | 16 | 6.76 / - | 8.11 / - |
| res1280000003 | 32 | 6.72 / - | 8.11 / - |
| reswordmax | 4 | 6.71 / - | 8.10 / - |
| reswordmax | 8 | 6.90 / - | 8.11 / - |
| reswordmax | 16 | 6.82 / - | 8.11 / - |
| reswordmax | 32 | 6.73 / - | 8.11 / - |

</details>

<details>
<summary>Prepare CoeffMatrix + mul_into_matrix_with</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 6.07 / - | 7.87 / - |
| res65537 | 8 | 9.48 / - | 12.9 / - |
| res65537 | 16 | 11.0 / - | 14.3 / - |
| res65537 | 32 | 11.7 / - | 15.1 / - |
| res67m | 4 | 5.73 / - | 7.87 / - |
| res67m | 8 | 9.02 / - | 12.3 / - |
| res67m | 16 | 10.4 / - | 14.5 / - |
| res67m | 32 | 10.9 / - | 15.2 / - |
| res268m | 4 | 5.72 / - | 7.87 / - |
| res268m | 8 | 9.08 / - | 12.8 / - |
| res268m | 16 | 10.4 / - | 13.9 / - |
| res268m | 32 | 10.6 / - | 14.2 / - |
| res536m | 4 | 5.72 / - | 7.86 / - |
| res536m | 8 | 8.90 / - | 12.4 / - |
| res536m | 16 | 9.41 / - | 12.8 / - |
| res536m | 32 | 9.57 / - | 12.4 / - |
| res858m | 4 | 5.88 / - | 7.87 / - |
| res858m | 8 | 7.81 / - | 10.5 / - |
| res858m | 16 | 8.11 / - | 10.3 / - |
| res858m | 32 | 8.59 / - | 10.8 / - |
| res1e9p7 | 4 | 5.82 / - | 7.86 / - |
| res1e9p7 | 8 | 7.72 / - | 10.4 / - |
| res1e9p7 | 16 | 7.98 / - | 10.5 / - |
| res1e9p7 | 32 | 8.15 / - | 10.1 / - |
| res1280000003 | 4 | 5.70 / - | 7.87 / - |
| res1280000003 | 8 | 5.88 / - | 8.01 / - |
| res1280000003 | 16 | 6.08 / - | 8.08 / - |
| res1280000003 | 32 | 6.13 / - | 8.11 / - |
| reswordmax | 4 | 5.83 / - | 7.87 / - |
| reswordmax | 8 | 5.91 / - | 8.01 / - |
| reswordmax | 16 | 6.07 / - | 8.08 / - |
| reswordmax | 32 | 6.10 / - | 8.11 / - |

</details>

<details>
<summary>Prepare CoeffMatrix + mul_add_matrix_with</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 6.11 / - | 8.14 / - |
| res65537 | 8 | 9.12 / - | 12.2 / - |
| res65537 | 16 | 10.7 / - | 13.7 / - |
| res65537 | 32 | 11.5 / - | 14.9 / - |
| res67m | 4 | 5.79 / - | 8.13 / - |
| res67m | 8 | 8.75 / - | 12.2 / - |
| res67m | 16 | 10.0 / - | 13.6 / - |
| res67m | 32 | 10.8 / - | 14.9 / - |
| res268m | 4 | 5.78 / - | 8.13 / - |
| res268m | 8 | 8.65 / - | 12.1 / - |
| res268m | 16 | 10.1 / - | 13.6 / - |
| res268m | 32 | 10.5 / - | 14.0 / - |
| res536m | 4 | 5.90 / - | 8.13 / - |
| res536m | 8 | 8.70 / - | 12.2 / - |
| res536m | 16 | 9.30 / - | 12.2 / - |
| res536m | 32 | 9.45 / - | 12.2 / - |
| res858m | 4 | 5.96 / - | 8.13 / - |
| res858m | 8 | 7.50 / - | 10.0 / - |
| res858m | 16 | 7.95 / - | 10.0 / - |
| res858m | 32 | 8.48 / - | 10.6 / - |
| res1e9p7 | 4 | 5.83 / - | 8.13 / - |
| res1e9p7 | 8 | 7.49 / - | 10.1 / - |
| res1e9p7 | 16 | 7.93 / - | 10.0 / - |
| res1e9p7 | 32 | 8.06 / - | 9.99 / - |
| res1280000003 | 4 | 6.07 / - | 8.13 / - |
| res1280000003 | 8 | 5.99 / - | 8.14 / - |
| res1280000003 | 16 | 6.11 / - | 8.15 / - |
| res1280000003 | 32 | 6.14 / - | 8.14 / - |
| reswordmax | 4 | 5.94 / - | 8.13 / - |
| reswordmax | 8 | 5.99 / - | 8.14 / - |
| reswordmax | 16 | 6.09 / - | 8.14 / - |
| reswordmax | 32 | 6.04 / - | 8.14 / - |

</details>

## Competitors

No matched competitor measurements are available.

## Caveats

- Backend requests use one current artifact, not separate released residue implementations.
- Short rows can take a narrower or scalar path. Exact operation routes and field caps stay in the evidence.
- mul_assign_with includes restoring the staged destination inside the timed call.
- Prepared matrix application excludes construction; prepare-and-apply panels include CoeffMatrix allocation and construction.
- Gather panels compose four prepared output gathers at the same matrix geometry.
- Direct internals timings are excluded from public tables.
- Controls and per-round variability stay in the raw evidence; differences within that variability do not establish a win.
