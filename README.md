# Ckmeans

[![Documentation](https://img.shields.io/docsrs/ckmeans/latest.svg)](https://docs.rs/ckmeans/latest)


```rust
use ckmeans::ckmeans;

let input = vec![
    1.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0,
    2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 5.0, 7.0,
    1.0, 5.0, 82.0, 1.0, 1.3, 1.1, 78.0,
];
let expected = vec![
    vec![
        1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 2.0, 2.0,
        2.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0,
    ],
    vec![12.0, 13.0, 14.0, 15.0, 16.0],
    vec![78.0, 82.0],
];
let result = ckmeans(&input, 3).unwrap();
assert_eq!(result, expected);
```

## Optimal k Selection

If you don't know the optimal number of clusters in advance, `ckmeans_optimal` can determine it
automatically using the Bayesian Information Criterion (BIC), following Song & Zhong (2020).
The BIC is that of a Gaussian mixture with one component for each cluster. `ckmeans_optimal`
selects the same k as the `Ckmeans.1d.dp` R package, which reports the negative of the BIC
values that `ckmeans_optimal` returns:

```rust
use ckmeans::{ckmeans_optimal, CkmeansConfig};

let data = vec![1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0];
// CkmeansConfig::default() evaluates k = 1..=9, capped at the number of
// distinct values in the data (3 here)
let result = ckmeans_optimal(&data, CkmeansConfig::default()).unwrap();
// result.k == 3 (optimal number of clusters)
// result.clusters contains the three clusters
// result.stats contains per-cluster centre, size, and within-cluster sum of squares
// result.bic contains BIC values for each candidate k
```

Ckmeans clustering is an improvement on 1-dimensional (univariate) heuristic-based clustering approaches such as [Jenks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization). The algorithm was developed by [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf) (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach to the problem of clustering numeric data into groups with the least within-group sum-of-squared-deviations.

Minimising the difference within groups (what Wang & Song refer to as `withinss`, or within sum-of-squares) means that groups are optimally homogeneous within and the data is split into representative groups. This is very useful for visualisation, where one may wish to represent a continuous variable in discrete colour or style groups. This function can provide groups that emphasise differences between data.

## Data Types

While this library supports both integer and floating-point types, **`f64` is the recommended type** for most clustering use cases. Continuous data is the primary target for optimal clustering. Integer inputs are promoted to `f64` internally for the clustering computation, so they cluster at the same speed; values beyond f64's exact integer range (2^53) may lose precision, which is the main reason to prefer `f64`.

## Numerical Limits

NaN values cannot be clustered: all functions return `CkmeansErr::NanError` for input that contains NaN. The following limits apply to other input (see the `ckmeans` and `ckmeans_optimal` documentation for examples):

- **Infinite values** are accepted, but a cluster that contains one has no finite sum of squares, so `ckmeans` returns a partition of the input with no optimality guarantee. `ckmeans_optimal` returns NaN BIC values for such input and uses `k_min`; the statistics of a cluster with an infinite value are NaN.
- **Very large ranges**: the costs are computed in `f64` from cumulative sums. If the data spans a very large range (for example, values that differ by 1 alongside values of order 1e8), cost differences below `f64` resolution are lost. The result can then be sub-optimal by that amount, and equal values can be put in adjacent clusters. `roundbreaks` then returns the first value of the upper class as the break.
- **Large integers**: integer values with a magnitude above 2^53 lose precision in the calculation. The clusters still contain the original values.

## How It Works

The algorithm fills two matrices using dynamic programming:
- **S matrix**: stores the minimum within-cluster sum-of-squares for clustering the first `i` elements into `k` clusters
- **J matrix**: stores backtracking indices to reconstruct the optimal cluster boundaries

For each column `k` (number of clusters), the algorithm finds the optimal split point `j` for each position `i` by minimising `SSQ(j, i) + S[k-1][j-1]`, where `SSQ(j, i)` is the sum-of-squares for elements `j` to `i` (computed in O(1) using prefix sums).

Performance comes from the monotonicity of the optimal split point: the optimal split for position `i` is always >= the optimal split for position `i-1`. This is the precondition for a divide-and-conquer dynamic-programming optimisation, which bounds the search at each step and processes each column in O(n log n) time, giving O(kn log n) overall. (The same monotonicity is what the SMAWK algorithm exploits to reach O(n) per column; in practice, though, these divide-and-conquer bounds keep each column close to linear with a smaller constant factor than SMAWK.)

Like the [original R implementation](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html), this implementation can automatically determine the optimal number of clusters using `ckmeans_optimal`, which evaluates candidates using the Bayesian Information Criterion (BIC). It also provides the `roundbreaks` method to aid labelling.

# FFI
A C-compatible FFI implementation is available, along with libraries for major platforms. See the [header file](include/header.h) and a basic C example in the [`examples`](examples) folder. The FFI functions have been verified not to leak memory (see comment in example). `ckmeans_ffi` reports errors (such as NaN input or an invalid number of classes) through a `CkmeansStatus` out-parameter and returns a result with a `NULL` data pointer; it does not abort the calling process.

# WASM
A WASM module is also available, giving access to both `ckmeans` and `roundbreaks`. Generate the module using [`wasm-bindgen`](https://rustwasm.github.io/docs/wasm-bindgen/) and the appropriate target, or use the [NPM package](https://www.npmjs.com/package/@urschrei/ckmeans).

# Implementation

This implementation builds on David Schnurr's JavaScript package (<https://github.com/schnerd/ckmeans>) and Bill Mill's Python + Numpy implementation (<https://github.com/llimllib/ckmeans>), with several key differences:

| Feature | Schnurr / Mill | This Implementation |
|---------|---------------|---------------------|
| Matrix layout | Nested arrays | Flat contiguous array for cache locality |
| Column filling | Recursive or iterative two-pointer | Stack-based divide-and-conquer with single-pass inner loop |
| SSQ computation | Computed twice per candidate in two-pointer approach | Computed exactly once per candidate |
| Memory allocation | Per-column stack allocation | Pre-allocated stack reused across columns |

The single-pass inner loop is the most significant change: the original two-pointer approach computed `SSQ(j, i)` for both the high and low pointers in each iteration, effectively computing SSQ twice for each index in the search range. This implementation computes SSQ exactly once per index, which significantly benefits f64 performance where floating-point arithmetic dominates.

# Performance

On an M2 Pro, to produce 7 clusters from normally-distributed f64 data:

| Data Size | Time |
|-----------|------|
| 10,000 | 1.7 ms |
| 50,000 | 9.9 ms |
| 110,000 | 23 ms |
| 500,000 | 115 ms |
| 1,000,000 | 243 ms |

Scaling with cluster count (110k f64 values):

| Clusters (k) | Time |
|--------------|------|
| 3 | 9.6 ms |
| 7 | 23 ms |
| 15 | 47 ms |
| 30 | 89 ms |
| 50 | 138 ms |

## Comparison with the C++ implementation

The `bench_cpp` crate compares this library with the C++ implementation of the
[Ckmeans.1d.dp](https://cran.r-project.org/package=Ckmeans.1d.dp) R package on the same inputs.
See [`bench_cpp/README.md`](bench_cpp/README.md).

## Profile-Guided Optimisation (PGO)
This library supports PGO builds for enhanced performance. PGO typically provides 10-30% performance improvements by optimising hot paths based on real-world usage patterns.

### Building with PGO
To build an optimised version using PGO:

```bash
# Run the automated PGO build script
chmod +x scripts/pgo-build.sh
./scripts/pgo-build.sh
```

The script will:
1. Build with instrumentation to collect profile data
2. Run comprehensive training workloads (k=3 to 25)
3. Build the final optimised binary using collected profiles

Optimised binaries will be available in `target/pgo-optimized/`.

### Using PGO in Production
- For Rust projects: Use the `.rlib` file from `target/pgo-optimized/`
- For C/FFI: Use the platform-specific shared library (`.so`, `.dylib`, or `.dll`)
- For maximum performance, ensure your use case matches the training profile (k values between 3-25)

## Complexity
$O(kn \log n)$. Other approaches such as Hilferink's [`CalcNaturalBreaks`](https://www.geodms.nl/CalcNaturalBreaks) or k-means have comparable complexity, but do _not_ guarantee optimality. In practice, they require many rounds to approach an optimal result, so in practice they're slower.
### Note
Wang and Song (2011) state that the algorithm runs in $O(k^2n)$ in their introduction. They have since updated their dynamic programming algorithm (see the August 2016 note [here](https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920fc9aabab184a2acff29e7965ce4f90173/src/Ckmeans.1d.dp.cpp#L91-L95)), reported there as $O(kn)$. The divide-and-conquer search reproduced here is $O(kn \log n)$ in the worst case, with the monotonicity bounds keeping it close to linear in practice.

## Testing

In addition to the unit tests, [`src/properties.rs`](src/properties.rs) contains property-based tests written with [`hegeltest`](https://crates.io/crates/hegeltest). They compare the clusters with a brute-force search and a reference dynamic program, and test the error cases, `ckmeans_optimal`, `roundbreaks` and the FFI. Run all tests with `cargo nextest r` or `cargo test`.

## Possible Improvements

- **SIMD**: The split-point search is vectorised for ranges of 16 or more candidates, with 128-bit vectors on the default targets. On x86-64, 256-bit AVX2 vectors could double the lane width.
- **Parallelisation**: Columns could be processed in parallel using rayon (though dependencies between columns limit this)

# References
1. [Wang, H., & Song, M. (2011). Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming. The R Journal, 3(2), 29.](https://doi.org/10.32614/RJ-2011-015)
2. <https://observablehq.com/@visionscarto/natural-breaks>
