# C++ comparison benchmarks

This crate compares `ckmeans` with the C++ implementation in the
[Ckmeans.1d.dp](https://cran.r-project.org/package=Ckmeans.1d.dp) R package. The `ckmeans-1d-dp`
Python package wraps the same C++ code. The crate is not published.

## Run the benchmarks

From this directory:

```bash
cargo bench --bench compare
```

To run one group, give a filter:

```bash
cargo bench --bench compare -- ckmeans_py
```

To check that the two implementations give the same clusters:

```bash
cargo nextest r
```

To compare the number of clusters that the two implementations select with the BIC:

```bash
cargo run --release --example bic_selection
```

## What is measured

Each benchmark runs these functions on the same seeded input:

| ID | Call |
|----|------|
| `rust_indices` | `ckmeans::ckmeans_indices`, which `ckmeans_py` calls |
| `rust_clusters` | `ckmeans::ckmeans` |
| `rust_optimal` | `ckmeans::ckmeans_optimal` with `k_min = k_max = k` |
| `cpp_linear` | `kmeans_1d_dp()` with method `linear`, the default in the R and Python packages |
| `cpp_loglinear` | `kmeans_1d_dp()` with method `loglinear` |

The groups are:

| Group | Input | n | k |
|-------|-------|---|---|
| `ckmeans_py_uniform_n110000` | Uniform(1, 3) | 110,000 | 5, 20 |
| `ckmeans_py_uniform_n1000000` | Uniform(1, 3) | 1,000,000 | 5, 20 |
| `varying_k_n110k` | Uniform(0, 1000) | 110,000 | 3, 7, 15, 30, 50 |
| `bimodal_n110k` | Normal(0, 1) and Normal(100, 1) | 110,000 | 7 |
| `bic_k1_9_n110000` | Four normals, means 0, 10, 20, 30 | 110,000 | BIC selection in 1..=9 |
| `bic_k1_9_n1000000` | Four normals, means 0, 10, 20, 30 | 1,000,000 | BIC selection in 1..=9 |

The `ckmeans_py` groups use the inputs of the benchmark in the `ckmeans_py` repository.

The `bic_k1_9` groups run only `rust_optimal` with `CkmeansConfig::default()`, and the C++ code
with `Kmin = 1` and `Kmax = 9`.

## Build settings

`build.rs` compiles the C++ sources that the `ckmeans-1d-dp` Python package compiles, with
`-O3 -DNDEBUG` and `-std=c++11`. The Rust code uses the `ckmeans_py` release settings,
`lto = true` and `codegen-units = 1`.

## Differences between the two calls

- The C++ call uses `estimate_k = "BIC"` with `Kmin = Kmax = k`, as the Python package does
  by default. In this configuration, `kmeans_1d_dp()` computes one BIC value after the dynamic
  program. This step takes 6 to 10 percent of the C++ time for the `ckmeans_py` cases.
  `rust_indices` and `rust_clusters` do not do this step.
- `rust_optimal` computes the same BIC value as the C++ code, with the same O(nk) Gaussian
  mixture likelihood.
- The C++ call returns a cluster label for each value, and the centre, within-cluster sum of
  squares and size of each cluster. `ckmeans_indices` returns the sorted data and the index
  range of each cluster. `ckmeans` also copies each cluster into a new `Vec`.
- The Rust wrapper of the C++ code checks that `k` is in range and that all values are
  finite. It also takes a mutex, because `kmeans_1d_dp()` sorts through a static pointer.

## Results

Apple M2 Pro, rustc 1.98.0, Apple clang 17.0.0. The times are Criterion point estimates from
one run of the full suite.

Uniform(1, 3), the `ckmeans_py` cases:

| n | k | `rust_indices` | `rust_clusters` | `rust_optimal` | `cpp_linear` | `cpp_loglinear` |
|---|---|--------------|---------------|--------------|------------|---------------|
| 110,000 | 5 | 16.0 ms | 16.0 ms | 18.2 ms | 25.4 ms | 24.6 ms |
| 110,000 | 20 | 60.4 ms | 60.5 ms | 66.4 ms | 103.8 ms | 90.3 ms |
| 1,000,000 | 5 | 167.2 ms | 168.8 ms | 190.5 ms | 261.1 ms | 255.5 ms |
| 1,000,000 | 20 | 665.6 ms | 665.8 ms | 710.2 ms | 971.0 ms | 936.0 ms |

Uniform(0, 1000), n = 110,000:

| k | `rust_indices` | `rust_clusters` | `rust_optimal` | `cpp_linear` | `cpp_loglinear` |
|---|--------------|---------------|--------------|------------|---------------|
| 3 | 9.7 ms | 9.7 ms | 11.1 ms | 16.2 ms | 16.3 ms |
| 7 | 23.4 ms | 23.2 ms | 25.8 ms | 37.9 ms | 35.6 ms |
| 15 | 47.1 ms | 46.6 ms | 50.2 ms | 79.5 ms | 71.2 ms |
| 30 | 87.9 ms | 86.8 ms | 93.2 ms | 157.4 ms | 129.5 ms |
| 50 | 135.1 ms | 135.5 ms | 145.3 ms | 258.0 ms | 193.3 ms |

Bimodal, n = 110,000:

| k | `rust_indices` | `rust_clusters` | `rust_optimal` | `cpp_linear` | `cpp_loglinear` |
|---|--------------|---------------|--------------|------------|---------------|
| 7 | 21.4 ms | 21.1 ms | 23.7 ms | 32.1 ms | 29.0 ms |

BIC selection of k in 1..=9:

| n | `rust_optimal` | `cpp_linear` | `cpp_loglinear` |
|---|----------------|--------------|-----------------|
| 110,000 | 36.8 ms | 54.9 ms | 50.2 ms |
| 1,000,000 | 382.5 ms | 502.3 ms | 497.3 ms |

## Licence

The C++ sources in `vendor/Ckmeans.1d.dp` are licensed under the GNU Lesser General Public
License, version 3 or later. See `vendor/Ckmeans.1d.dp/README.md`.
