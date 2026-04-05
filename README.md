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
automatically using the Bayesian Information Criterion (BIC), following Song & Zhong (2020):

```rust
use ckmeans::ckmeans_optimal;

let data = vec![1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0];
let result = ckmeans_optimal(&data, None, None).unwrap();
// result.k == 3 (optimal number of clusters)
// result.clusters contains the three clusters
// result.stats contains per-cluster center, size, and within-cluster sum of squares
// result.bic contains BIC values for each candidate k (default range: 1..=9)
```

Ckmeans clustering is an improvement on 1-dimensional (univariate) heuristic-based clustering approaches such as [Jenks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization). The algorithm was developed by [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf) (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach to the problem of clustering numeric data into groups with the least within-group sum-of-squared-deviations.

Minimising the difference within groups (what Wang & Song refer to as `withinss`, or within sum-of-squares) means that groups are optimally homogeneous within and the data is split into representative groups. This is very useful for visualisation, where one may wish to represent a continuous variable in discrete colour or style groups. This function can provide groups that emphasise differences between data.

## Data Types

While this library supports both integer and floating-point types, **`f64` is the recommended type** for most clustering use cases. Continuous data is the primary target for optimal clustering, and the implementation is optimised for floating-point performance. Integer types work but may see reduced performance due to algorithmic trade-offs that favour f64.

## How It Works

The algorithm fills two matrices using dynamic programming:
- **S matrix**: stores the minimum within-cluster sum-of-squares for clustering the first `i` elements into `k` clusters
- **J matrix**: stores backtracking indices to reconstruct the optimal cluster boundaries

For each column `k` (number of clusters), the algorithm finds the optimal split point `j` for each position `i` by minimising `SSQ(j, i) + S[k-1][j-1]`, where `SSQ(j, i)` is the sum-of-squares for elements `j` to `i` (computed in O(1) using prefix sums).

The key to linear-time performance is the SMAWK algorithm's monotonicity property: the optimal split point for position `i` is always >= the optimal split point for position `i-1`. This allows a divide-and-conquer approach that processes each column in O(n) time, giving O(kn) total complexity.

Like the [original R implementation](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html), this implementation can automatically determine the optimal number of clusters using `ckmeans_optimal`, which evaluates candidates using the Bayesian Information Criterion (BIC). It also provides the `roundbreaks` method to aid labelling.

# FFI
A C-compatible FFI implementation is available, along with libraries for major platforms. See the [header file](include/header.h) and a basic C example in the [`examples`](examples) folder. The FFI functions have been verified not to leak memory (see comment in example).

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
$O(kn)$. Other approaches such as Hilferink's [`CalcNaturalBreaks`](https://www.geodms.nl/CalcNaturalBreaks) or k-means have comparable complexity, but do _not_ guarantee optimality. In practice, they require many rounds to approach an optimal result, so in practice they're slower.
### Note
Wang and Song (2011) state that the algorithm runs in $O(k^2n)$ in their introduction. However, they have since updated their dynamic programming algorithm (see August 2016 note [here](https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920fc9aabab184a2acff29e7965ce4f90173/src/Ckmeans.1d.dp.cpp#L91-L95)) which reduces the complexity to linear time. This approach has been used in the extant implementations listed above, and reproduced here.

## Possible Improvements

- **SIMD**: The SSQ computation could potentially benefit from SIMD vectorisation
- **Parallelisation**: Columns could be processed in parallel using rayon (though dependencies between columns limit this)
- **Integer optimisation**: The current implementation favours f64; a separate code path with early-exit optimisation could improve integer performance
- **Property-based tests**: Additional testing coverage

# References
1. [Wang, H., & Song, M. (2011). Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming. The R Journal, 3(2), 29.](https://doi.org/10.32614/RJ-2011-015)
2. <https://observablehq.com/@visionscarto/natural-breaks>
