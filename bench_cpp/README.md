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

Apple M2 Pro, rustc 1.99.0, Apple clang 17.0.0. The times are Criterion point estimates from
one run of the full suite.

Uniform(1, 3), the `ckmeans_py` cases:

| n | k | `rust_indices` | `rust_clusters` | `rust_optimal` | `cpp_linear` | `cpp_loglinear` |
|---|---|--------------|---------------|--------------|------------|---------------|
| 110,000 | 5 | 13.2 ms | 12.9 ms | 14.5 ms | 26.0 ms | 25.3 ms |
| 110,000 | 20 | 49.7 ms | 49.8 ms | 52.7 ms | 105.3 ms | 90.7 ms |
| 1,000,000 | 5 | 126.7 ms | 128.0 ms | 142.9 ms | 254.9 ms | 249.8 ms |
| 1,000,000 | 20 | 513.5 ms | 521.5 ms | 556.1 ms | 995.4 ms | 940.5 ms |

Uniform(0, 1000), n = 110,000:

| k | `rust_indices` | `rust_clusters` | `rust_optimal` | `cpp_linear` | `cpp_loglinear` |
|---|--------------|---------------|--------------|------------|---------------|
| 3 | 7.6 ms | 7.8 ms | 9.1 ms | 16.5 ms | 16.1 ms |
| 7 | 18.2 ms | 17.9 ms | 20.0 ms | 37.0 ms | 36.2 ms |
| 15 | 38.2 ms | 37.3 ms | 40.4 ms | 81.2 ms | 72.2 ms |
| 30 | 71.4 ms | 71.3 ms | 75.5 ms | 160.5 ms | 141.6 ms |
| 50 | 114.4 ms | 113.5 ms | 120.0 ms | 256.0 ms | 193.0 ms |

Bimodal, n = 110,000:

| k | `rust_indices` | `rust_clusters` | `rust_optimal` | `cpp_linear` | `cpp_loglinear` |
|---|--------------|---------------|--------------|------------|---------------|
| 7 | 16.0 ms | 15.9 ms | 17.6 ms | 31.3 ms | 28.7 ms |

BIC selection of k in 1..=9:

| n | `rust_optimal` | `cpp_linear` | `cpp_loglinear` |
|---|----------------|--------------|-----------------|
| 110,000 | 26.7 ms | 54.5 ms | 50.9 ms |
| 1,000,000 | 263.1 ms | 510.2 ms | 515.9 ms |

## Licence

The C++ sources in `vendor/Ckmeans.1d.dp` are licensed under the GNU Lesser General Public
License, version 3 or later. See `vendor/Ckmeans.1d.dp/README.md`.
