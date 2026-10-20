# Generic prime residues

Every numeric cell uses **Willow Cove (i5-1135G7) / Golden Cove (i7-12700K)** order; unavailable measurements are `-`. Shared context and reproduction: [BENCHMARKS.md](../../BENCHMARKS.md). [Field labels](../labels.md) fix modulus and lane representation.

Throughput: **higher is better**. A lane is one field element; its byte width is listed in the field definitions.

## Find a result

- [Prepared single-row multiplication](#prepared-single-row-multiplication)
- [One-shot single-row multiplication](#one-shot-single-row-multiplication)
- [Elementwise multiplication](#elementwise-multiplication)
- [Matrix multiplication](#matrix-multiplication)
- [Measurement details](#measurement-details)

## Self-timings

### Prepared single-row multiplication

Numerator: row bytes per call.

#### mul_into_with

<details>
<summary>Show all fields and row lengths</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res101 | 0.810 / 1.742 | 1.719 / 3.869 | 2.424 / 4.760 | 8.769 / 5.822 | 25.41 / 18.64 | 39.47 / 38.73 |
| res131 | 1.389 / 2.684 | 2.968 / 5.815 | 4.450 / 8.727 | 5.400 / 10.53 | 6.437 / 10.89 | 6.292 / 10.57 |
| res65521 | 1.435 / 2.414 | 1.603 / 2.619 | 1.328 / 2.713 | 1.508 / 2.799 | 1.681 / 2.886 | 1.729 / 2.905 |
| res1e9p7 | 2.699 / 5.579 | 5.571 / 9.612 | 9.624 / 14.21 | 14.30 / 15.11 | 16.30 / 18.93 | 16.92 / 19.90 |
| res1280000003 | 2.783 / 5.768 | 5.933 / 9.972 | 9.716 / 14.59 | 14.40 / 15.41 | 16.39 / 18.99 | 16.91 / 19.91 |
| reswordmax | 2.783 / 5.815 | 5.988 / 9.973 | 9.847 / 14.71 | 14.43 / 14.67 | 16.49 / 19.00 | 16.90 / 19.68 |
| reswide | 3.932 / 5.602 | 5.076 / 6.067 | 5.331 / 6.118 | 5.492 / 6.274 | 5.531 / 6.317 | 5.609 / 6.249 |
| res2 | 13.67 / 23.28 | 23.76 / 39.91 | 41.93 / 69.83 | 64.68 / 111.6 | 90.23 / 198.7 | 136.3 / 225.9 |

</details>

#### mul_add_with

<details>
<summary>Show all fields and row lengths</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res101 | 0.481 / 0.759 | 0.683 / 1.437 | 1.783 / 2.844 | 7.365 / 5.137 | 13.17 / 12.09 | 12.20 / 13.32 |
| res131 | 0.646 / 1.144 | 0.896 / 1.561 | 1.027 / 1.787 | 1.049 / 1.902 | 1.067 / 1.934 | 1.071 / 1.945 |
| res65521 | 0.823 / 2.000 | 0.873 / 2.152 | 0.792 / 2.184 | 0.848 / 2.216 | 0.888 / 2.178 | 0.898 / 2.218 |
| res1e9p7 | 1.845 / 3.225 | 3.854 / 6.068 | 6.034 / 7.133 | 6.911 / 7.977 | 7.413 / 8.828 | 8.061 / 9.014 |
| res1280000003 | 1.875 / 3.221 | 3.895 / 6.121 | 6.106 / 7.141 | 7.014 / 7.988 | 7.420 / 8.835 | 8.059 / 9.054 |
| reswordmax | 1.862 / 3.221 | 3.906 / 6.106 | 6.099 / 7.172 | 6.957 / 7.995 | 7.381 / 8.584 | 8.060 / 9.010 |
| reswide | 3.575 / 3.407 | 4.386 / 3.493 | 4.522 / 5.058 | 4.678 / 5.141 | 4.796 / 5.063 | 4.834 / 5.136 |
| res2 | 7.117 / 13.91 | 8.659 / 16.66 | 9.340 / 18.60 | 8.146 / 19.68 | 9.198 / 20.62 | 9.553 / 20.45 |

</details>

#### mul_assign_with

<details>
<summary>Show all fields and row lengths</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res101 | 0.805 / 1.486 | 1.845 / 3.981 | 2.369 / 4.611 | 5.113 / 6.705 | 14.85 / 18.81 | 28.79 / 34.75 |
| res131 | 1.235 / 1.993 | 2.964 / 4.983 | 4.455 / 8.183 | 5.801 / 10.26 | 6.268 / 10.71 | 6.001 / 9.970 |
| res65521 | 1.294 / 2.391 | 1.363 / 2.567 | 1.249 / 2.754 | 1.450 / 2.823 | 1.599 / 2.850 | 1.645 / 2.867 |
| res1e9p7 | 3.563 / 7.348 | 5.975 / 11.88 | 7.552 / 15.31 | 9.025 / 14.55 | 12.59 / 18.37 | 14.62 / 17.93 |
| res1280000003 | 3.562 / 7.349 | 5.979 / 12.69 | 7.605 / 15.32 | 9.057 / 14.45 | 12.58 / 18.37 | 14.59 / 17.93 |
| reswordmax | 3.564 / 7.349 | 5.981 / 12.68 | 7.605 / 15.32 | 9.065 / 14.55 | 12.59 / 18.36 | 14.65 / 17.93 |
| reswide | 2.205 / 3.493 | 3.224 / 5.811 | 4.062 / 6.011 | 4.652 / 6.048 | 5.262 / 5.988 | 5.319 / 6.024 |
| res2 | 22.13 / 34.86 | 32.37 / 55.79 | 59.36 / 101.4 | 74.88 / 131.4 | 73.44 / 153.3 | 102.8 / 160.1 |

</details>

### One-shot single-row multiplication

Numerator: row bytes per call.

#### mul_into

<details>
<summary>Show all fields and row lengths</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res101 | 0.810 / 1.742 | 1.715 / 3.871 | 2.442 / 4.759 | 8.784 / 5.821 | 24.57 / 18.66 | 39.59 / 38.85 |
| res131 | 1.387 / 2.683 | 3.178 / 5.816 | 4.688 / 8.726 | 6.037 / 10.53 | 6.431 / 10.91 | 6.373 / 10.98 |
| res65521 | 1.482 / 2.410 | 1.602 / 2.610 | 1.350 / 2.705 | 1.518 / 2.795 | 1.683 / 2.887 | 1.728 / 2.905 |
| res1e9p7 | 2.692 / 5.579 | 5.877 / 9.617 | 9.648 / 14.22 | 14.31 / 15.30 | 16.41 / 18.91 | 16.91 / 19.90 |
| res1280000003 | 2.782 / 5.587 | 5.876 / 9.968 | 9.688 / 14.58 | 14.40 / 15.24 | 16.46 / 19.00 | 16.96 / 19.69 |
| reswordmax | 2.781 / 5.812 | 5.999 / 9.974 | 9.841 / 14.69 | 14.42 / 14.82 | 16.57 / 19.01 | 16.90 / 19.64 |
| reswide | 3.935 / 5.558 | 5.052 / 5.926 | 5.320 / 6.116 | 5.488 / 6.196 | 5.535 / 6.275 | 5.609 / 6.254 |
| res2 | 13.68 / 23.27 | 23.75 / 39.89 | 41.93 / 69.84 | 64.76 / 111.7 | 90.01 / 194.3 | 136.3 / 226.0 |

</details>

### Elementwise multiplication

Numerator: one operand's bytes per call.

#### mul_elementwise

<details>
<summary>Show all fields and row lengths</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res1e9p7 | 1.08 / 2.06 | 2.90 / 5.01 | 4.80 / 5.55 | 5.70 / 5.95 | 5.88 / 6.36 | 6.22 / 6.48 |
| res1280000003 | 1.40 / 2.26 | 2.88 / 5.02 | 4.66 / 5.54 | 5.70 / 5.95 | 5.90 / 6.37 | 6.22 / 6.45 |
| reswordmax | 1.23 / 2.61 | 2.89 / 5.02 | 4.47 / 5.54 | 5.71 / 5.95 | 5.87 / 6.36 | 6.20 / 6.48 |

</details>

#### mul_elementwise_assign

<details>
<summary>Show all fields and row lengths</summary>

| Field | 8 lanes (GiB/s) | 16 lanes (GiB/s) | 32 lanes (GiB/s) | 64 lanes (GiB/s) | 256 lanes (GiB/s) | 1024 lanes (GiB/s) |
| --- | --- | --- | --- | --- | --- | --- |
| res1e9p7 | 2.04 / 3.15 | 3.82 / 5.36 | 5.03 / 5.73 | 5.80 / 6.10 | 5.95 / 6.26 | 6.22 / 6.50 |
| res1280000003 | 2.04 / 3.15 | 3.82 / 5.36 | 5.03 / 5.74 | 5.73 / 6.10 | 5.93 / 6.41 | 6.23 / 6.50 |
| reswordmax | 2.04 / 3.15 | 3.82 / 5.36 | 5.04 / 5.73 | 5.75 / 6.10 | 5.92 / 6.27 | 6.22 / 6.45 |

</details>

### Matrix multiplication

Four outputs; numerator: row bytes × sources × outputs. Each lane occupies four bytes.

#### mul_into_matrix_with

<details>
<summary>Show all fields and row lengths</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 6.36 / 7.82 | 7.89 / 8.91 |
| res65537 | 8 | 10.6 / 12.9 | 12.4 / 14.8 |
| res65537 | 16 | 12.3 / 14.4 | 14.5 / 16.4 |
| res65537 | 32 | 13.0 / 16.2 | 14.8 / 18.4 |
| res67m | 4 | 6.45 / 7.81 | 7.88 / 8.89 |
| res67m | 8 | 10.4 / 12.9 | 12.3 / 14.7 |
| res67m | 16 | 12.3 / 14.4 | 14.4 / 16.4 |
| res67m | 32 | 12.9 / 16.3 | 14.8 / 18.4 |
| res268m | 4 | 6.47 / 7.81 | 7.88 / 8.91 |
| res268m | 8 | 10.6 / 13.0 | 12.6 / 14.9 |
| res268m | 16 | 12.3 / 14.4 | 14.4 / 16.3 |
| res268m | 32 | 12.4 / 15.0 | 14.0 / 16.7 |
| res536m | 4 | 6.46 / 7.81 | 7.88 / 8.88 |
| res536m | 8 | 10.5 / 13.0 | 12.7 / 14.8 |
| res536m | 16 | 11.1 / 13.3 | 12.8 / 15.0 |
| res536m | 32 | 11.2 / 13.3 | 12.3 / 14.5 |
| res858m | 4 | 6.55 / 7.83 | 7.89 / 8.89 |
| res858m | 8 | 8.91 / 10.6 | 10.3 / 11.8 |
| res858m | 16 | 9.30 / 10.8 | 10.4 / 11.9 |
| res858m | 32 | 9.77 / 11.3 | 10.5 / 12.2 |
| res1e9p7 | 4 | 6.37 / 7.82 | 7.88 / 8.90 |
| res1e9p7 | 8 | 8.98 / 10.6 | 10.6 / 11.8 |
| res1e9p7 | 16 | 9.30 / 10.8 | 10.5 / 11.8 |
| res1e9p7 | 32 | 9.27 / 10.6 | 10.1 / 11.6 |
| res1280000003 | 4 | 6.41 / 7.90 | 7.89 / 8.91 |
| res1280000003 | 8 | 6.59 / 7.89 | 8.02 / 9.01 |
| res1280000003 | 16 | 6.68 / 7.94 | 8.09 / 9.07 |
| res1280000003 | 32 | 6.78 / 7.99 | 8.13 / 9.08 |
| reswordmax | 4 | 6.68 / 7.89 | 7.89 / 8.89 |
| reswordmax | 8 | 6.57 / 7.88 | 8.02 / 9.00 |
| reswordmax | 16 | 6.73 / 7.94 | 8.09 / 9.05 |
| reswordmax | 32 | 6.73 / 7.97 | 8.13 / 9.08 |

</details>

#### mul_add_matrix_with

<details>
<summary>Show all fields and row lengths</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 6.39 / 7.94 | 8.16 / 9.11 |
| res65537 | 8 | 10.2 / 12.1 | 12.3 / 13.8 |
| res65537 | 16 | 12.1 / 14.4 | 13.8 / 16.4 |
| res65537 | 32 | 12.9 / 15.6 | 15.0 / 17.7 |
| res67m | 4 | 6.68 / 7.94 | 8.16 / 9.10 |
| res67m | 8 | 10.2 / 12.1 | 12.1 / 13.9 |
| res67m | 16 | 12.1 / 14.3 | 13.8 / 16.4 |
| res67m | 32 | 12.8 / 15.7 | 15.0 / 17.7 |
| res268m | 4 | 6.47 / 7.93 | 8.16 / 9.10 |
| res268m | 8 | 10.1 / 12.1 | 12.3 / 13.8 |
| res268m | 16 | 12.1 / 14.4 | 14.0 / 16.3 |
| res268m | 32 | 12.5 / 14.8 | 14.0 / 16.5 |
| res536m | 4 | 6.68 / 7.94 | 8.16 / 9.11 |
| res536m | 8 | 10.2 / 12.1 | 12.1 / 13.9 |
| res536m | 16 | 10.8 / 12.7 | 12.3 / 14.3 |
| res536m | 32 | 11.1 / 13.1 | 12.2 / 14.4 |
| res858m | 4 | 6.76 / 7.97 | 8.16 / 9.10 |
| res858m | 8 | 8.52 / 10.0 | 9.99 / 11.2 |
| res858m | 16 | 9.09 / 10.5 | 10.0 / 11.5 |
| res858m | 32 | 9.69 / 11.1 | 10.6 / 11.8 |
| res1e9p7 | 4 | 6.67 / 7.94 | 8.16 / 9.10 |
| res1e9p7 | 8 | 8.61 / 10.0 | 10.0 / 11.1 |
| res1e9p7 | 16 | 9.15 / 10.5 | 9.99 / 11.4 |
| res1e9p7 | 32 | 9.32 / 10.5 | 10.0 / 11.3 |
| res1280000003 | 4 | 6.82 / 7.97 | 8.16 / 9.10 |
| res1280000003 | 8 | 6.79 / 7.92 | 8.16 / 9.11 |
| res1280000003 | 16 | 6.78 / 7.95 | 8.16 / 9.11 |
| res1280000003 | 32 | 6.74 / 7.98 | 8.16 / 9.10 |
| reswordmax | 4 | 6.61 / 7.97 | 8.16 / 9.10 |
| reswordmax | 8 | 6.65 / 7.93 | 8.16 / 9.10 |
| reswordmax | 16 | 6.65 / 7.95 | 8.16 / 9.10 |
| reswordmax | 32 | 6.76 / 7.96 | 8.16 / 9.10 |

</details>

#### Four prepared mul_into_gather_with calls

<details>
<summary>Show all fields and row lengths</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 7.58 / 9.13 | 9.33 / 10.5 |
| res65537 | 8 | 7.05 / 8.53 | 8.68 / 9.77 |
| res65537 | 16 | 6.91 / 8.13 | 8.39 / 9.43 |
| res65537 | 32 | 6.62 / 8.06 | 8.25 / 9.27 |
| res67m | 4 | 7.67 / 9.13 | 9.33 / 10.5 |
| res67m | 8 | 7.11 / 8.52 | 8.68 / 9.77 |
| res67m | 16 | 6.80 / 8.06 | 8.39 / 9.43 |
| res67m | 32 | 6.70 / 7.99 | 8.25 / 9.26 |
| res268m | 4 | 7.67 / 9.13 | 9.33 / 10.5 |
| res268m | 8 | 7.10 / 8.52 | 8.68 / 9.77 |
| res268m | 16 | 6.81 / 8.10 | 8.38 / 9.43 |
| res268m | 32 | 6.68 / 8.02 | 8.25 / 9.26 |
| res536m | 4 | 7.69 / 9.16 | 9.33 / 10.5 |
| res536m | 8 | 7.19 / 8.54 | 8.68 / 9.77 |
| res536m | 16 | 6.73 / 8.14 | 8.38 / 9.42 |
| res536m | 32 | 6.68 / 8.03 | 8.25 / 9.26 |
| res858m | 4 | 7.84 / 9.17 | 9.33 / 10.5 |
| res858m | 8 | 7.21 / 8.57 | 8.68 / 9.77 |
| res858m | 16 | 6.97 / 8.14 | 8.39 / 9.43 |
| res858m | 32 | 6.86 / 8.04 | 8.25 / 9.27 |
| res1e9p7 | 4 | 7.57 / 9.14 | 9.33 / 10.5 |
| res1e9p7 | 8 | 7.20 / 8.53 | 8.68 / 9.78 |
| res1e9p7 | 16 | 6.81 / 8.13 | 8.38 / 9.43 |
| res1e9p7 | 32 | 6.70 / 8.04 | 8.25 / 9.27 |
| res1280000003 | 4 | 7.68 / 9.18 | 9.33 / 10.5 |
| res1280000003 | 8 | 7.19 / 8.58 | 8.68 / 9.77 |
| res1280000003 | 16 | 6.98 / 8.19 | 8.39 / 9.43 |
| res1280000003 | 32 | 6.85 / 8.10 | 8.25 / 9.27 |
| reswordmax | 4 | 7.92 / 9.12 | 9.33 / 10.5 |
| reswordmax | 8 | 7.23 / 8.53 | 8.68 / 9.77 |
| reswordmax | 16 | 7.00 / 8.11 | 8.39 / 9.43 |
| reswordmax | 32 | 6.85 / 8.01 | 8.25 / 9.26 |

</details>

#### Four prepared mul_add_gather_with calls

<details>
<summary>Show all fields and row lengths</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 6.42 / 7.99 | 8.10 / 9.10 |
| res65537 | 8 | 6.67 / 7.99 | 8.11 / 9.10 |
| res65537 | 16 | 6.60 / 7.85 | 8.11 / 9.11 |
| res65537 | 32 | 6.64 / 7.89 | 8.11 / 9.11 |
| res67m | 4 | 6.77 / 8.01 | 8.10 / 9.11 |
| res67m | 8 | 6.66 / 8.01 | 8.11 / 9.10 |
| res67m | 16 | 6.62 / 7.88 | 8.11 / 9.12 |
| res67m | 32 | 6.60 / 7.92 | 8.11 / 9.11 |
| res268m | 4 | 6.62 / 8.00 | 8.10 / 9.10 |
| res268m | 8 | 6.60 / 7.99 | 8.11 / 9.10 |
| res268m | 16 | 6.64 / 7.86 | 8.11 / 9.11 |
| res268m | 32 | 6.62 / 7.90 | 8.11 / 9.11 |
| res536m | 4 | 6.55 / 8.01 | 8.10 / 9.10 |
| res536m | 8 | 6.68 / 8.02 | 8.11 / 9.10 |
| res536m | 16 | 6.61 / 7.87 | 8.11 / 9.11 |
| res536m | 32 | 6.57 / 7.92 | 8.11 / 9.11 |
| res858m | 4 | 6.91 / 8.03 | 8.10 / 9.10 |
| res858m | 8 | 6.87 / 8.03 | 8.11 / 9.10 |
| res858m | 16 | 6.77 / 7.89 | 8.11 / 9.11 |
| res858m | 32 | 6.74 / 7.93 | 8.11 / 9.11 |
| res1e9p7 | 4 | 6.79 / 7.99 | 8.10 / 9.10 |
| res1e9p7 | 8 | 6.64 / 8.01 | 8.11 / 9.10 |
| res1e9p7 | 16 | 6.64 / 7.88 | 8.11 / 9.11 |
| res1e9p7 | 32 | 6.61 / 7.90 | 8.11 / 9.11 |
| res1280000003 | 4 | 6.86 / 8.03 | 8.10 / 9.11 |
| res1280000003 | 8 | 6.87 / 8.04 | 8.11 / 9.11 |
| res1280000003 | 16 | 6.70 / 7.90 | 8.11 / 9.11 |
| res1280000003 | 32 | 6.73 / 7.94 | 8.11 / 9.11 |
| reswordmax | 4 | 6.64 / 8.00 | 8.10 / 9.09 |
| reswordmax | 8 | 6.84 / 8.00 | 8.11 / 9.10 |
| reswordmax | 16 | 6.64 / 7.87 | 8.11 / 9.12 |
| reswordmax | 32 | 6.76 / 7.89 | 8.11 / 9.12 |

</details>

#### Prepare CoeffMatrix + mul_into_matrix_with

<details>
<summary>Show all fields and row lengths</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 5.96 / 7.50 | 7.87 / 8.90 |
| res65537 | 8 | 9.56 / 12.2 | 12.4 / 14.8 |
| res65537 | 16 | 11.1 / 13.7 | 14.5 / 16.3 |
| res65537 | 32 | 11.6 / 15.2 | 14.8 / 18.4 |
| res67m | 4 | 5.80 / 7.26 | 7.87 / 8.88 |
| res67m | 8 | 8.84 / 11.4 | 12.3 / 14.8 |
| res67m | 16 | 10.4 / 12.7 | 14.3 / 16.4 |
| res67m | 32 | 10.8 / 14.0 | 14.8 / 18.4 |
| res268m | 4 | 5.83 / 7.25 | 7.87 / 8.87 |
| res268m | 8 | 9.07 / 11.4 | 12.6 / 14.8 |
| res268m | 16 | 10.4 / 12.6 | 14.3 / 16.3 |
| res268m | 32 | 10.5 / 13.1 | 13.9 / 16.7 |
| res536m | 4 | 5.82 / 7.27 | 7.87 / 8.89 |
| res536m | 8 | 9.04 / 11.4 | 12.5 / 14.8 |
| res536m | 16 | 9.42 / 11.8 | 12.7 / 15.0 |
| res536m | 32 | 9.50 / 11.7 | 12.2 / 14.6 |
| res858m | 4 | 5.89 / 7.27 | 7.87 / 8.87 |
| res858m | 8 | 7.68 / 9.53 | 10.3 / 11.7 |
| res858m | 16 | 8.15 / 9.81 | 10.4 / 11.9 |
| res858m | 32 | 8.49 / 10.2 | 10.5 / 12.2 |
| res1e9p7 | 4 | 5.72 / 7.25 | 7.87 / 8.88 |
| res1e9p7 | 8 | 7.80 / 9.53 | 10.5 / 11.7 |
| res1e9p7 | 16 | 8.17 / 9.77 | 10.5 / 11.8 |
| res1e9p7 | 32 | 8.14 / 9.64 | 10.1 / 11.5 |
| res1280000003 | 4 | 5.74 / 7.34 | 7.87 / 8.90 |
| res1280000003 | 8 | 5.89 / 7.35 | 8.01 / 8.99 |
| res1280000003 | 16 | 6.02 / 7.41 | 8.07 / 9.04 |
| res1280000003 | 32 | 6.17 / 7.43 | 8.11 / 9.07 |
| reswordmax | 4 | 5.95 / 7.32 | 7.87 / 8.88 |
| reswordmax | 8 | 5.93 / 7.34 | 8.00 / 8.99 |
| reswordmax | 16 | 6.07 / 7.39 | 8.07 / 9.04 |
| reswordmax | 32 | 6.10 / 7.42 | 8.11 / 9.07 |

</details>

#### Prepare CoeffMatrix + mul_add_matrix_with

<details>
<summary>Show all fields and row lengths</summary>

| Field | Sources | 64 lanes (GiB/s) | 4096 lanes (GiB/s) |
| --- | --- | --- | --- |
| res65537 | 4 | 5.90 / 7.60 | 8.14 / 9.09 |
| res65537 | 8 | 9.22 / 11.5 | 12.2 / 13.8 |
| res65537 | 16 | 10.8 / 13.6 | 13.8 / 16.3 |
| res65537 | 32 | 11.5 / 14.7 | 14.9 / 17.7 |
| res67m | 4 | 5.92 / 7.32 | 8.13 / 9.09 |
| res67m | 8 | 8.69 / 10.8 | 12.1 / 13.9 |
| res67m | 16 | 10.2 / 12.6 | 13.7 / 16.3 |
| res67m | 32 | 10.8 / 13.5 | 14.9 / 17.7 |
| res268m | 4 | 5.75 / 7.32 | 8.13 / 9.08 |
| res268m | 8 | 8.76 / 10.8 | 12.1 / 13.8 |
| res268m | 16 | 10.1 / 12.5 | 14.0 / 16.3 |
| res268m | 32 | 10.4 / 13.0 | 14.0 / 16.4 |
| res536m | 4 | 5.92 / 7.35 | 8.13 / 9.08 |
| res536m | 8 | 8.74 / 10.8 | 12.1 / 13.8 |
| res536m | 16 | 9.36 / 11.4 | 12.2 / 14.2 |
| res536m | 32 | 9.48 / 11.7 | 12.2 / 14.4 |
| res858m | 4 | 5.99 / 7.35 | 8.13 / 9.09 |
| res858m | 8 | 7.49 / 9.10 | 9.95 / 11.1 |
| res858m | 16 | 8.03 / 9.50 | 10.0 / 11.5 |
| res858m | 32 | 8.44 / 10.0 | 10.5 / 11.8 |
| res1e9p7 | 4 | 5.92 / 7.32 | 8.13 / 9.09 |
| res1e9p7 | 8 | 7.43 / 9.14 | 10.1 / 11.1 |
| res1e9p7 | 16 | 8.04 / 9.48 | 9.91 / 11.4 |
| res1e9p7 | 32 | 8.12 / 9.52 | 10.0 / 11.2 |
| res1280000003 | 4 | 6.03 / 7.44 | 8.13 / 9.09 |
| res1280000003 | 8 | 6.09 / 7.41 | 8.14 / 9.10 |
| res1280000003 | 16 | 6.10 / 7.40 | 8.14 / 9.09 |
| res1280000003 | 32 | 6.12 / 7.42 | 8.15 / 9.10 |
| reswordmax | 4 | 5.88 / 7.43 | 8.13 / 9.08 |
| reswordmax | 8 | 5.98 / 7.42 | 8.14 / 9.09 |
| reswordmax | 16 | 6.01 / 7.40 | 8.14 / 9.09 |
| reswordmax | 32 | 6.11 / 7.41 | 8.15 / 9.09 |

</details>

## Competitors

No matched competitor measurements are available.

## Measurement details

<details>
<summary>Setup, backend requests, and caveats</summary>

| Setting | Value |
| --- | --- |
| Revision | `63b8d168a7e4b81043fafe4db1ffb9918dc1c231` |
| Toolchain | 1.98.1 |
| Aggregation | Median of five per-run placement medians. |
| Placements | Offsets 0, 16, 32 within each run. |
| Order | Routed, AVX2, scalar requests rotate within each round. |
| Resolved process backend (v3) | v4x / v3_gfni_crypto |
| Resolved process backend (avx2) | v3 / v3 |
| Resolved process backend (scalar) | scalar / scalar |

- Backend requests use one current artifact, not separate released residue implementations.
- Short rows can take a narrower or scalar path. Exact operation routes and field caps stay in the evidence.
- mul_assign_with includes restoring the staged destination inside the timed call.
- Prepared matrix application excludes construction; prepare-and-apply panels include CoeffMatrix allocation and construction.
- Gather panels compose four prepared output gathers at the same matrix geometry.
- Direct internals timings are excluded from public tables.
- Controls and per-round variability stay in the raw evidence; differences within that variability do not establish a win.

</details>
