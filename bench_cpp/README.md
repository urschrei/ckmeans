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

The `ckmeans_py` groups use the inputs of the benchmark in the `ckmeans_py` repository.

## Build settings

`build.rs` compiles the C++ sources that the `ckmeans-1d-dp` Python package compiles, with
`-O3 -DNDEBUG` and `-std=c++11`. The Rust code uses the `ckmeans_py` release settings,
`lto = true` and `codegen-units = 1`.

## Differences between the two calls

- The C++ call uses `estimate_k = "BIC"` with `Kmin = Kmax = k`, as the Python package does
  by default. In this configuration, `kmeans_1d_dp()` computes one BIC value after the dynamic
  program. This step takes 6 to 10 percent of the C++ time for the `ckmeans_py` cases.
  `rust_indices` and `rust_clusters` do not do this step.
- `rust_optimal` also computes one BIC value, but with a different formula. The C++ code
  evaluates the Gaussian mixture density at each value, which is O(nk). `ckmeans_optimal`
  computes the BIC from the size and within-cluster sum of squares of each cluster, which is
  O(k) after one pass over the clusters. `ckmeans_optimal` also sorts the data twice.
- The C++ call returns a cluster label for each value, and the centre, within-cluster sum of
  squares and size of each cluster. `ckmeans_indices` returns the sorted data and the index
  range of each cluster. `ckmeans` also copies each cluster into a new `Vec`.
- The Rust wrapper of the C++ code checks that `k` is in range and that all values are
  finite. It also takes a mutex, because `kmeans_1d_dp()` sorts through a static pointer.

## Results

Apple M2 Pro, rustc 1.98.0, Apple clang 17.0.0. The times are Criterion point estimates.

| n | k | `rust_indices` | `rust_optimal` | `cpp_linear` | `cpp_loglinear` |
|---|---|----------------|----------------|--------------|-----------------|
| 110,000 | 5 | 15.8 ms | 17.7 ms | 24.7 ms | 23.9 ms |
| 110,000 | 20 | 59.4 ms | 61.1 ms | 101.1 ms | 91.9 ms |
| 1,000,000 | 5 | 165.1 ms | 186.5 ms | 253.1 ms | 249.3 ms |
| 1,000,000 | 20 | 656.3 ms | 670.2 ms | 990.7 ms | 928.2 ms |

## Licence

The C++ sources in `vendor/Ckmeans.1d.dp` are licensed under the GNU Lesser General Public
License, version 3 or later. See `vendor/Ckmeans.1d.dp/README.md`.
