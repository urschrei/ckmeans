# Ckmeans

[![Documentation](https://img.shields.io/docsrs/ckmeans/latest.svg)](https://docs.rs/ckmeans/latest)

Ckmeans clustering is an improvement on 1-dimensional (univariate) heuristic-based clustering approaches such as [Jenks](https://en.wikipedia.org/wiki/Jenks_natural_breaks_optimization). The algorithm was developed by [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf) (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach to the problem of clustering numeric data into groups with the least within-group sum-of-squared-deviations.

Minimizing the difference within groups – what Wang & Song refer to as `withinss`, or within sum-of-squares – means that groups are optimally homogenous within and the data is split into representative groups. This is very useful for visualization, where one may wish to represent a continuous variable in discrete colour or style groups. This function can provide groups that emphasize differences between data.

Being a dynamic approach, this algorithm is based on two matrices that store incrementally-computed values for squared deviations and backtracking indexes.

Unlike the [original implementation](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html), this implementation does not include any code to automatically determine the optimal number of clusters: this information needs to be explicitly provided. It **does** provide the `roundbreaks` method to aid labelling, however.

# FFI
A C-compatible FFI implementation is available, along with libraries for major platforms. See the [header file](include/header.h) and a basic C example in the [`examples`](examples) folder.

# Implementation
This is a port (including documentation) of David Schnurr's package <https://github.com/schnerd/ckmeans>, incorporating some improvements from Bill Mill's Python + Numpy implementation at <https://github.com/llimllib/ckmeans>.

# Performance
On an M2 Pro, the algorithm will classify 110k uniformly-distributed i32 values between 0 and 250 into 7 classes in ~12 ms.

## Complexity
$O(n^2k)$. Other approaches such as Hilferink's [`CalcNaturalBreaks`](https://www.geodms.nl/CalcNaturalBreaks) or k-means are faster, but do _not_ guarantee optimality. In practice, they require many rounds to approach an optimal result, so in practice they're the same speed or far slower (in the case of k-means, see Wang and Song (2011), Figure 4) for more than 7 classes while failing to guarantee optimality or reproducibility.

## Possible Improvements
### Perf
The "matrices" are nested vectors and thus don't have optimal memory layout. In addition, we're not trying to leverage any of the fast linear algebra libraries that might be available if we used e.g. [`ndarray`](https://crates.io/crates/ndarray).

### Tests
Perhaps some property-based tests

# References
1. [Wang, H., & Song, M. (2011). Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming. The R Journal, 3(2), 29.](https://doi.org/10.32614/RJ-2011-015)
2. <https://observablehq.com/@visionscarto/natural-breaks>
