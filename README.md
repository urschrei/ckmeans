# Ckmeans

[![Documentation](https://img.shields.io/docsrs/ckmeans/latest.svg)](https://docs.rs/ckmeans/latest)

Ckmeans clustering is an improvement on 1-dimensional (univariate) heuristic-based clustering approaches such as [Jenks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization). The algorithm was developed by [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf) (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach to the problem of clustering numeric data into groups with the least within-group sum-of-squared-deviations.

Minimizing the difference within groups – what Wang & Song refer to as `withinss`, or within sum-of-squares – means that groups are optimally homogenous within and the data is split into representative groups. This is very useful for visualization, where one may wish to represent a continuous variable in discrete colour or style groups. This function can provide groups that emphasize differences between data.

Being a dynamic approach, this algorithm is based on two matrices that store incrementally-computed values for squared deviations and backtracking indexes.

Unlike the [original implementation](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html), this implementation does not include any code to automatically determine the optimal number of clusters: this information needs to be explicitly provided. It **does** provide the `roundbreaks` method to aid labelling, however.

# FFI
A C-compatible FFI implementation is available, along with libraries for major platforms. See the [header file](include/header.h) and a basic C example in the [`examples`](examples) folder. The FFI functions have been verified not to leak memory (see comment in example).

# WASM
A WASM module is also available, giving access to both `ckmeans` and `roundbreaks`. Generate the module using [`wasm-bindgen`](https://rustwasm.github.io/docs/wasm-bindgen/) and the appropriate target, or use the [NPM package](https://www.npmjs.com/package/@urschrei/ckmeans).

# Implementation
This is a port (including documentation) of David Schnurr's package <https://github.com/schnerd/ckmeans>, incorporating some improvements from Bill Mill's Python + Numpy implementation at <https://github.com/llimllib/ckmeans>.

# Performance
On an M2 Pro, to produce 7 classes:

1. 110k uniformly-distributed i32 values between 0 and 250: ~12 ms
2. 110k normally-distributed f64 values with a mean of 3.0 and a standard deviation of 1.0: 38 ms

## Complexity
$O(kn)$. Other approaches such as Hilferink's [`CalcNaturalBreaks`](https://www.geodms.nl/CalcNaturalBreaks) or k-means have comparable complexity, but do _not_ guarantee optimality. In practice, they require many rounds to approach an optimal result, so in practice they're slower.
### Note
Wang and Song (2011) state that the algorithm runs in $O(k^2n)$ in their introduction. However, they have since updated their dynamic programming algorithm (see August 2016 note [here](https://github.com/cran/Ckmeans.1d.dp/blob/f7f2920fc9aabab184a2acff29e7965ce4f90173/src/Ckmeans.1d.dp.cpp#L91-L95)) which reduces the complexity to linear time. This approach has been used in the extant implementations listed above, and reproduced here.

## Possible Improvements
### Perf
The "matrices" are nested vectors and thus don't have optimal memory layout. In addition, we're not trying to leverage any of the fast linear algebra libraries that might be available if we used e.g. [`ndarray`](https://crates.io/crates/ndarray).

### Tests
Perhaps some property-based tests

# References
1. [Wang, H., & Song, M. (2011). Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming. The R Journal, 3(2), 29.](https://doi.org/10.32614/RJ-2011-015)
2. <https://observablehq.com/@visionscarto/natural-breaks>
