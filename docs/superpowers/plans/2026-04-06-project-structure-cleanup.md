# Project Structure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the crate to follow Rust project structure conventions -- extract algorithm internals from `lib.rs`, fix import style, and correct item ordering.

**Architecture:** Extract private algorithm helpers into `src/algo.rs`, leaving `lib.rs` as a table of contents with public types, public API functions, and re-exports. Fix compound imports to individual imports throughout. Reorder items so public precedes private and callers precede callees.

**Tech Stack:** Rust, cargo nextest, cargo fmt, cargo clippy, jj

---

### Task 1: Extract algorithm internals into `src/algo.rs`

**Files:**
- Create: `src/algo.rs`
- Modify: `src/lib.rs`

The following private items move from `lib.rs` to `algo.rs`:
- `numeric_sort` (line 116)
- `unique_count_sorted` (line 123)
- `FlatMatrix` struct + impl (lines 132-156)
- `ssq` (line 159)
- `fill_matrix_column` (line 177)
- `fill_matrices` (line 239)
- `compute_cluster_stats` (line 293)
- `compute_bic` (line 321)

These are all `pub(crate)` from `algo.rs` so `lib.rs` can call them.

- [ ] **Step 1: Create `src/algo.rs` with extracted internals**

```rust
use num_traits::Float;
use num_traits::Num;
use num_traits::NumCast;
use num_traits::cast::FromPrimitive;
use std::fmt::Debug;

use crate::CkNum;
use crate::ClusterStats;

/// return a sorted **copy** of the input. Will blow up in the presence of NaN
pub(crate) fn numeric_sort<T: CkNum>(arr: &[T]) -> Vec<T> {
    let mut xs = arr.to_vec();
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    xs
}

/// Assumes sorted input (so be sure only to use on `numeric_sort` output!)
pub(crate) fn unique_count_sorted<T: CkNum>(input: &mut [T]) -> usize {
    if input.is_empty() {
        0
    } else {
        1 + input.windows(2).filter(|win| win[0] != win[1]).count()
    }
}

/// Flat matrix structure for better cache locality
pub(crate) struct FlatMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: CkNum> FlatMatrix<T> {
    pub(crate) fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::zero(); rows * cols],
            rows,
            cols,
        }
    }

    #[inline]
    pub(crate) fn get(&self, row: usize, col: usize) -> T {
        self.data[row * self.cols + col]
    }

    #[inline]
    pub(crate) fn set(&mut self, row: usize, col: usize, value: T) {
        self.data[row * self.cols + col] = value;
    }
}

#[inline(always)]
fn ssq<T: CkNum>(j: usize, i: usize, sumx: &[T], sumxsq: &[T]) -> Option<T> {
    let n = T::from_usize(i - j + 1)?;
    let sji = if j > 0 {
        let sum_diff = sumx[i] - sumx[j - 1];
        let muji = sum_diff / n;
        sumxsq[i] - sumxsq[j - 1] - n * muji * muji
    } else {
        let n_plus_one = T::from_usize(i + 1)?;
        sumxsq[i] - (sumx[i] * sumx[i]) / n_plus_one
    };
    if sji < T::zero() {
        Some(T::zero())
    } else {
        Some(sji)
    }
}

#[allow(clippy::too_many_arguments)]
fn fill_matrix_column<T: CkNum>(
    imin: usize,
    imax: usize,
    column: usize,
    matrix: &mut FlatMatrix<T>,
    backtrack_matrix: &mut FlatMatrix<usize>,
    sumx: &[T],
    sumxsq: &[T],
    stack: &mut Vec<(usize, usize)>,
) -> Option<()> {
    // Reuse the pre-allocated stack for divide-and-conquer traversal
    stack.clear();
    stack.push((imin, imax));

    while let Some((imin, imax)) = stack.pop() {
        if imin > imax {
            continue;
        }

        // Start at midpoint between imin and imax
        let i = imin + (imax - imin) / 2;

        // Compute SMAWK bounds for j
        let mut jlow = column;
        if imin > column {
            jlow = jlow.max(backtrack_matrix.get(column, imin - 1));
        }
        jlow = jlow.max(backtrack_matrix.get(column - 1, i));

        let mut jhigh = i;
        if imax < matrix.cols - 1 {
            jhigh = jhigh.min(backtrack_matrix.get(column, imax + 1));
        }

        // Find minimum cost split point with a single pass through the range.
        // This computes ssq exactly once per j (the old two-pointer approach
        // computed ssq twice for each index).
        let mut best_j = jlow;
        let mut best_cost = ssq(jlow, i, sumx, sumxsq)? + matrix.get(column - 1, jlow - 1);

        for j in (jlow + 1)..=jhigh {
            let cost = ssq(j, i, sumx, sumxsq)? + matrix.get(column - 1, j - 1);
            if cost < best_cost {
                best_cost = cost;
                best_j = j;
            }
        }

        matrix.set(column, i, best_cost);
        backtrack_matrix.set(column, i, best_j);

        // Push right range first (so left is processed first when popped)
        if i < imax {
            stack.push((i + 1, imax));
        }
        if imin < i {
            stack.push((imin, i - 1));
        }
    }
    Some(())
}

pub(crate) fn fill_matrices<T: CkNum>(
    data: &[T],
    matrix: &mut FlatMatrix<T>,
    backtrack_matrix: &mut FlatMatrix<usize>,
    nclusters: usize,
) -> Option<()> {
    let nvalues = data.len();
    let mut sumx = Vec::with_capacity(nvalues);
    let mut sumxsq = Vec::with_capacity(nvalues);
    let shift = data[nvalues / 2];
    // Initialize first row in matrix & backtrack_matrix

    // Pre-compute sumx and sumxsq
    sumx.push(data[0] - shift);
    sumxsq.push((data[0] - shift) * (data[0] - shift));

    for i in 1..nvalues {
        let shifted = data[i] - shift;
        sumx.push(sumx[i - 1] + shifted);
        sumxsq.push(sumxsq[i - 1] + shifted * shifted);
    }

    // Initialize matrix for k = 0
    for i in 0..nvalues {
        matrix.set(0, i, ssq(0, i, &sumx, &sumxsq)?);
        backtrack_matrix.set(0, i, 0);
    }

    // Pre-allocate stack for divide-and-conquer (reused across columns)
    // Maximum depth is log2(n) + 1 for binary tree traversal
    let stack_capacity = ((nvalues as f64).log2().ceil() as usize).max(1) + 1;
    let mut stack = Vec::with_capacity(stack_capacity);

    for k in 1..nclusters {
        let imin = if k < nclusters {
            k.max(1)
        } else {
            // No need to compute matrix[k - 1][0] ... matrix[k - 1][n - 2]
            nvalues - 1
        };
        fill_matrix_column(
            imin,
            nvalues - 1,
            k,
            matrix,
            backtrack_matrix,
            &sumx,
            &sumxsq,
            &mut stack,
        )?;
    }
    Some(())
}

/// Compute per-cluster statistics (center, size, withinss) for a set of clusters.
pub(crate) fn compute_cluster_stats<T: CkNum>(
    clusters: &[Vec<T>],
) -> Option<Vec<ClusterStats<T>>> {
    clusters
        .iter()
        .map(|cluster| {
            let size = cluster.len();
            let n = T::from_usize(size)?;
            let sum: T = cluster.iter().copied().fold(T::zero(), |acc, x| acc + x);
            let center = sum / n;
            let withinss = cluster
                .iter()
                .copied()
                .fold(T::zero(), |acc, x| acc + (x - center) * (x - center));
            Some(ClusterStats {
                center,
                size,
                withinss,
            })
        })
        .collect()
}

/// Compute the BIC for a clustering result under a Gaussian mixture model.
///
/// Following Song & Zhong (2020):
/// - Log-likelihood per cluster j: -n_j/2 * ln(2*pi) - n_j/2 * ln(sigma_j^2) - (n_j - 1)/2
/// - For singleton clusters (n_j = 1), sigma_j^2 = total_variance / n
/// - Number of parameters: p = 3k - 1
/// - BIC = -2 * log(L) + p * ln(n)
pub(crate) fn compute_bic<T: CkNum + Float>(
    stats: &[ClusterStats<T>],
    n: usize,
    total_variance: T,
) -> Option<T> {
    let k = stats.len();
    let n_t = T::from_usize(n)?;
    let two = T::from_f64(2.0)?;
    let two_pi = T::from_f64(std::f64::consts::TAU)?;
    let ln_two_pi = two_pi.ln();

    // Fallback variance for singleton clusters
    let singleton_var = total_variance / n_t;

    let mut log_likelihood = T::zero();

    for stat in stats {
        let n_j = T::from_usize(stat.size)?;
        let sigma_sq = if stat.size <= 1 {
            singleton_var
        } else {
            stat.withinss / n_j
        };

        // Guard against zero variance (all identical values in cluster)
        if sigma_sq <= T::zero() {
            // Perfectly homogeneous cluster -- skip the variance penalty.
            // Only the constant terms contribute.
            log_likelihood = log_likelihood - n_j / two * ln_two_pi;
        } else {
            log_likelihood = log_likelihood
                - n_j / two * ln_two_pi
                - n_j / two * sigma_sq.ln()
                - (n_j - T::one()) / two;
        }
    }

    // p = 3k - 1: k means + k variances + (k-1) mixing proportions
    let p = T::from_usize(3 * k - 1)?;
    let bic = -two * log_likelihood + p * n_t.ln();
    Some(bic)
}
```

- [ ] **Step 2: Rewrite `src/lib.rs`**

The new `lib.rs` follows these conventions:
- Private imports first, then public re-exports grouped together
- All imports are individual (no compound `use`)
- Public types first (central/titular items)
- Public API functions next (caller before callee: `ckmeans_optimal` calls `ckmeans`, `ckmeans` calls `ckmeans_indices`)
- Private helper `roundbreaks` is public but depends on `ckmeans`, so it comes after
- `mod` declarations and re-exports grouped at top
- Tests at bottom

```rust
//! Ckmeans clustering is an improvement on heuristic-based 1-dimensional (univariate) clustering
//! approaches such as Jenks. The algorithm was developed by
//! [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf)
//! (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach
//! to the problem of clustering numeric data into groups with the least
//! within-group sum-of-squared-deviations.
//!
//! # Example
//!
//! ```
//! use ckmeans::ckmeans;
//!
//! let input = vec![
//!     1.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0,
//!     2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 5.0, 7.0,
//!     1.0, 5.0, 82.0, 1.0, 1.3, 1.1, 78.0,
//! ];
//! let expected = vec![
//!     vec![
//!         1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 2.0, 2.0,
//!         2.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0,
//!     ],
//!     vec![12.0, 13.0, 14.0, 15.0, 16.0],
//!     vec![78.0, 82.0],
//! ];
//!
//! let result = ckmeans(&input, 3).unwrap();
//! assert_eq!(result, expected);
//! ```

use num_traits::Float;
use num_traits::Num;
use num_traits::NumCast;
use num_traits::cast::FromPrimitive;
use std::fmt::Debug;

mod algo;
mod errors;
#[cfg(not(target_arch = "wasm32"))]
mod ffi;
mod wasm;

pub use crate::errors::CkmeansErr;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::ffi::ExternalArray;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::ffi::InternalArray;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::ffi::WrapperArray;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::ffi::ckmeans_ffi;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::ffi::drop_ckmeans_result;
pub use crate::wasm::ckmeans_optimal_wasm;
pub use crate::wasm::ckmeans_wasm;
pub use crate::wasm::roundbreaks_wasm;

/// A trait that encompasses most common numeric types (integer **and** floating point)
pub trait CkNum: Num + Copy + NumCast + PartialOrd + FromPrimitive + Debug {}
impl<T: Num + Copy + NumCast + PartialOrd + FromPrimitive + Debug> CkNum for T {}

/// Result type for ckmeans_indices: (sorted_data, cluster_ranges)
pub type ClusterIndices<T> = (Vec<T>, Vec<(usize, usize)>);

/// Per-cluster statistics returned by [`ckmeans_optimal`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterStats<T> {
    /// Mean value of the cluster.
    pub center: T,
    /// Number of elements in the cluster.
    pub size: usize,
    /// Within-cluster sum of squares.
    pub withinss: T,
}

/// Configuration for [`ckmeans_optimal`].
///
/// The default evaluates k = 1 through 9:
///
/// | Field   | Default |
/// |---------|---------|
/// | `k_min` | 1       |
/// | `k_max` | 9       |
///
/// # Example
/// ```
/// use ckmeans::CkmeansConfig;
///
/// // Use defaults: evaluates k = 1..=9
/// let config = CkmeansConfig::default();
/// assert_eq!(config.k_min, 1);
/// assert_eq!(config.k_max, 9);
///
/// // Custom range: evaluates k = 2..=15
/// let config = CkmeansConfig { k_min: 2, k_max: 15, ..Default::default() };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CkmeansConfig {
    /// Minimum number of clusters to evaluate. Default: **1**.
    pub k_min: u8,
    /// Maximum number of clusters to evaluate. Default: **9**.
    pub k_max: u8,
}

impl Default for CkmeansConfig {
    fn default() -> Self {
        Self { k_min: 1, k_max: 9 }
    }
}

/// Result of [`ckmeans_optimal`], containing the clustering, the chosen k,
/// BIC values for each candidate k, and per-cluster statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct CkmeansResult<T> {
    /// Clustered data, same format as [`ckmeans`] output.
    pub clusters: Vec<Vec<T>>,
    /// The chosen number of clusters.
    pub k: u8,
    /// BIC value for each candidate k evaluated, as `(k, bic)` pairs.
    pub bic: Vec<(u8, T)>,
    /// Per-cluster statistics for the chosen clustering.
    pub stats: Vec<ClusterStats<T>>,
}

/// Determine the optimal number of clusters using the Bayesian Information
/// Criterion (BIC) and return the clustering result with per-cluster statistics.
///
/// This follows the approach of Song & Zhong (2020), evaluating each candidate k
/// in the range `k_min..=k_max` and selecting the k that minimises BIC.
///
/// # Arguments
/// * `data` - The input data to cluster
/// * `config` - Clustering configuration; use [`CkmeansConfig::default()`] for defaults
///   (k = 1..=9)
///
/// # References
/// 1. Song, M., & Zhong, H. (2020). Efficient weighted univariate clustering maps
///    outstanding dysregulated genomic zones in human cancers. Bioinformatics, 36(20), 5027-5036.
///
/// # Example
///
/// ```
/// use ckmeans::ckmeans_optimal;
/// use ckmeans::CkmeansConfig;
///
/// let data = vec![1.0, 1.0, 1.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0];
/// let result = ckmeans_optimal(&data, CkmeansConfig::default()).unwrap();
/// assert_eq!(result.k, 3);
/// ```
pub fn ckmeans_optimal<T: CkNum + Float>(
    data: &[T],
    config: CkmeansConfig,
) -> Result<CkmeansResult<T>, CkmeansErr> {
    let k_min = config.k_min;
    let k_max = config.k_max;

    if k_min == 0 {
        return Err(CkmeansErr::TooFewClassesError);
    }
    if k_min > k_max {
        return Err(CkmeansErr::InvalidRangeError);
    }
    if (k_min as usize) > data.len() {
        return Err(CkmeansErr::TooManyClassesError);
    }

    // Cap k_max to data length
    let k_max = k_max.min(data.len() as u8);

    // Check for all-identical values
    let sorted = algo::numeric_sort(data);
    if sorted.first() == sorted.last() {
        let stats = algo::compute_cluster_stats(std::slice::from_ref(&sorted))
            .ok_or(CkmeansErr::ConversionError)?;
        return Ok(CkmeansResult {
            clusters: vec![sorted],
            k: 1,
            bic: vec![(1, T::zero())],
            stats,
        });
    }

    // Compute total variance for singleton cluster fallback in BIC
    let n = data.len();
    let n_t = T::from_usize(n).ok_or(CkmeansErr::ConversionError)?;
    let sum: T = data.iter().copied().fold(T::zero(), |acc, x| acc + x);
    let mean = sum / n_t;
    let total_variance = data
        .iter()
        .copied()
        .fold(T::zero(), |acc, x| acc + (x - mean) * (x - mean))
        / n_t;

    let mut best_k: u8 = k_min;
    let mut best_bic = T::infinity();
    let mut best_clusters: Vec<Vec<T>> = Vec::new();
    let mut best_stats: Vec<ClusterStats<T>> = Vec::new();
    let mut all_bics: Vec<(u8, T)> = Vec::with_capacity((k_max - k_min + 1) as usize);

    for k in k_min..=k_max {
        let clusters = ckmeans(data, k)?;
        let stats =
            algo::compute_cluster_stats(&clusters).ok_or(CkmeansErr::ConversionError)?;
        let bic =
            algo::compute_bic(&stats, n, total_variance).ok_or(CkmeansErr::ConversionError)?;

        all_bics.push((k, bic));

        if bic < best_bic {
            best_bic = bic;
            best_k = k;
            best_clusters = clusters;
            best_stats = stats;
        }
    }

    Ok(CkmeansResult {
        clusters: best_clusters,
        k: best_k,
        bic: all_bics,
        stats: best_stats,
    })
}

/// Minimizing the difference within groups -- what Wang & Song refer to as
/// `withinss`, or within sum-of-squares, means that groups are **optimally
/// homogenous** within and the data is split into representative groups.
/// This is very useful for visualization, where one may wish to represent
/// a continuous variable in discrete colour or style groups. This function
/// can provide groups -- or "classes" -- that emphasize differences between data.
///
/// Being a dynamic approach, this algorithm is based on two matrices that
/// store incrementally-computed values for squared deviations and backtracking
/// indexes.
///
/// If you do not know the optimal number of clusters, use [`ckmeans_optimal`]
/// which evaluates candidates using the Bayesian Information Criterion (BIC).
///
/// # Notes
/// Most common numeric (integer or floating point) types can be clustered
///
/// # References
/// 1. [Wang, H., & Song, M. (2011). Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming. The R Journal, 3(2), 29.](https://doi.org/10.32614/RJ-2011-015)
/// 2. <https://observablehq.com/@visionscarto/natural-breaks>
///
/// # Example
///
/// ```
/// use ckmeans::ckmeans;
///
/// let input = vec![
///     1.0f64, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0, 2.0, 3.0, 5.0, 7.0, 1.0, 2.0, 5.0, 7.0,
///     1.0, 5.0, 82.0, 1.0, 1.3, 1.1, 78.0,
/// ];
/// let expected = vec![
///     vec![
///         1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 2.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0,
///     ],
///     vec![12.0, 13.0, 14.0, 15.0, 16.0],
///     vec![78.0, 82.0],
/// ];
///
/// let result = ckmeans(&input, 3).unwrap();
/// assert_eq!(result, expected);
/// ```
pub fn ckmeans<T: CkNum>(data: &[T], nclusters: u8) -> Result<Vec<Vec<T>>, CkmeansErr> {
    let (sorted, indices) = ckmeans_indices(data, nclusters)?;

    // Convert indices to actual clusters
    let mut clusters = Vec::with_capacity(indices.len());
    for range in indices {
        let mut cluster = Vec::with_capacity(range.1 - range.0 + 1);
        cluster.extend_from_slice(&sorted[range.0..=range.1]);
        clusters.push(cluster);
    }
    Ok(clusters)
}

/// Cluster data and return the sorted data with cluster index ranges.
/// This avoids copying data into separate cluster vectors.
///
/// Returns: (sorted_data, cluster_ranges) where cluster_ranges contains (start, end) inclusive indices
///
/// # Example
/// ```
/// use ckmeans::ckmeans_indices;
///
/// let input = vec![1.0, 12.0, 13.0, 2.0, 3.0, 5.0, 82.0, 78.0];
/// let (sorted, indices) = ckmeans_indices(&input, 3).unwrap();
/// // sorted contains all values in sorted order
/// // indices contains [(0, 4), (5, 6), (7, 7)] representing the three clusters
/// ```
pub fn ckmeans_indices<T: CkNum>(
    data: &[T],
    nclusters: u8,
) -> Result<ClusterIndices<T>, CkmeansErr> {
    if nclusters == 0 {
        return Err(CkmeansErr::TooFewClassesError);
    }
    if nclusters as usize > data.len() {
        return Err(CkmeansErr::TooManyClassesError);
    }
    let nvalues = data.len();
    let mut sorted = algo::numeric_sort(data);
    // we'll use this as the maximum number of clusters
    let unique_count = algo::unique_count_sorted(&mut sorted);
    // if all of the input values are identical, there's one cluster
    // with all of the input in it.
    if unique_count == 1 {
        return Ok((sorted, vec![(0, nvalues - 1)]));
    }
    let nclusters = unique_count.min(nclusters as usize);

    // named 'S' originally
    let mut matrix = algo::FlatMatrix::new(nclusters, nvalues);
    // named 'J' originally - store as usize to avoid conversions
    let mut backtrack_matrix = algo::FlatMatrix::<usize>::new(nclusters, nvalues);

    // This is a dynamic programming approach to solving the problem of minimizing
    // within-cluster sum of squares. It's similar to linear regression
    // in this way, and this calculation incrementally computes the
    // sum of squares that are later read.
    algo::fill_matrices(&sorted, &mut matrix, &mut backtrack_matrix, nclusters)
        .ok_or(CkmeansErr::ConversionError)?;

    // The real work of Ckmeans clustering happens in the matrix generation:
    // the generated matrices encode all possible clustering combinations, and
    // once they're generated we can solve for the best clustering groups
    // very quickly.
    let mut indices: Vec<(usize, usize)> = Vec::with_capacity(nclusters);
    let mut cluster_right = backtrack_matrix.cols - 1;

    // Backtrack the clusters from the dynamic programming matrix. This
    // starts at the bottom-right corner of the matrix (if the top-left is 0, 0),
    // and moves the cluster target with the loop.
    for cluster in (0..backtrack_matrix.rows).rev() {
        let cluster_left = backtrack_matrix.get(cluster, cluster_right);

        // Store the indices instead of copying data
        indices.push((cluster_left, cluster_right));
        if cluster > 0 {
            cluster_right = cluster_left - 1;
        }
    }
    indices.reverse();
    Ok((sorted, indices))
}

/// The boundaries of the classes returned by [ckmeans] are "ugly" in the sense that the values
/// returned are the lower bound of each cluster, which can't be used for labelling, since they
/// might have many decimal places. To create a legend, the values should be rounded -- but the
/// rounding might be either too loose (and would result in spurious decimal places), or too strict,
/// resulting in classes ranging "from x to x". A better approach is to choose the roundest number that
/// separates the lowest point from a class from the highest point
/// in the _preceding_ class -- thus giving just enough precision to distinguish the classes.
///
/// This function is closer to what Jenks returns: `nclusters - 1` "breaks" in the data, useful for
/// labelling.
///
/// # Original Implementation
/// <https://observablehq.com/@visionscarto/natural-breaks#round>
pub fn roundbreaks<T: Float + Debug + FromPrimitive>(
    data: &[T],
    nclusters: u8,
) -> Result<Vec<T>, CkmeansErr> {
    let ckm = ckmeans(data, nclusters)?;
    ckm.windows(2)
        .map(|pair| {
            let p = T::from(10.0).ok_or(CkmeansErr::ConversionError)?.powf(
                (T::one()
                    - (*pair[1].first().ok_or(CkmeansErr::HighWindowError)?
                        - *pair[0].last().ok_or(CkmeansErr::LowWindowError)?)
                    .log10())
                .floor(),
            );
            Ok((((*pair[1].first().ok_or(CkmeansErr::HighWindowError)?
                + *pair[0].last().ok_or(CkmeansErr::LowWindowError)?)
                / T::from(2.0).ok_or(CkmeansErr::ConversionError)?)
                * p)
                .floor()
                / p)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clustering_integers() {
        // ... (unchanged)
    }

    // ... all other tests unchanged
}
```

- [ ] **Step 3: Run tests**

Run: `cargo nextest r`
Expected: All 15 tests pass

- [ ] **Step 4: Run fmt and clippy**

Run: `cargo fmt && cargo clippy`
Expected: No warnings or errors

- [ ] **Step 5: Commit**

Run: `jj fix && jj commit -m "[WIP: claude] Extract algorithm internals into src/algo.rs"`

---

### Task 2: Fix compound imports in `src/lib.rs`

**Files:**
- Modify: `src/lib.rs` (imports at top)

Already handled in Task 1 -- the rewritten `lib.rs` uses individual imports throughout. This task is a verification step.

- [ ] **Step 1: Verify no compound imports remain in lib.rs**

Run: `grep -n '{.*,.*}' src/lib.rs` (should find nothing outside test code / struct literals)

- [ ] **Step 2: Commit if any fixes needed**

Run: `jj fix && jj commit -m "[WIP: claude] Fix remaining compound imports"`

---

### Task 3: Fix item ordering in `src/wasm.rs`

**Files:**
- Modify: `src/wasm.rs`

Move private helpers (`inner_vec_to_js_array`, `wrapper_vec_to_js_array`) after the public functions.

- [ ] **Step 1: Reorder `src/wasm.rs`**

```rust
use crate::CkmeansConfig;
use crate::ckmeans;
use crate::ckmeans_optimal;
use crate::roundbreaks;
use js_sys::Array;
use js_sys::Float64Array;
use js_sys::Number;
use js_sys::Object;
use js_sys::Reflect;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
/// A WASM wrapper for ckmeans
pub fn ckmeans_wasm(data: &[f64], nclusters: u8) -> Result<Array, JsError> {
    Ok(wrapper_vec_to_js_array(&ckmeans(data, nclusters)?))
}

#[wasm_bindgen]
/// A WASM wrapper for roundbreaks
pub fn roundbreaks_wasm(data: &[f64], nclusters: u8) -> Result<Float64Array, JsError> {
    Ok(inner_vec_to_js_array(&roundbreaks(data, nclusters)?))
}

#[wasm_bindgen]
/// A WASM wrapper for ckmeans_optimal
pub fn ckmeans_optimal_wasm(
    data: &[f64],
    k_min: Option<u8>,
    k_max: Option<u8>,
) -> Result<JsValue, JsError> {
    let config = CkmeansConfig {
        k_min: k_min.unwrap_or(1),
        k_max: k_max.unwrap_or(9),
    };
    let result = ckmeans_optimal(data, config)?;
    let obj = Object::new();

    // clusters: Array of Float64Arrays
    let clusters_js = wrapper_vec_to_js_array(&result.clusters);
    Reflect::set(&obj, &"clusters".into(), &clusters_js).unwrap();

    // k: number
    Reflect::set(&obj, &"k".into(), &Number::from(result.k as f64)).unwrap();

    // bic: Array of {k, value} objects
    let bic_arr = Array::new();
    for (k, value) in &result.bic {
        let entry = Object::new();
        Reflect::set(&entry, &"k".into(), &Number::from(*k as f64)).unwrap();
        Reflect::set(&entry, &"value".into(), &Number::from(*value)).unwrap();
        bic_arr.push(&entry);
    }
    Reflect::set(&obj, &"bic".into(), &bic_arr).unwrap();

    // stats: Array of {center, size, withinss} objects
    let stats_arr = Array::new();
    for stat in &result.stats {
        let entry = Object::new();
        Reflect::set(&entry, &"center".into(), &Number::from(stat.center)).unwrap();
        Reflect::set(&entry, &"size".into(), &Number::from(stat.size as f64)).unwrap();
        Reflect::set(&entry, &"withinss".into(), &Number::from(stat.withinss)).unwrap();
        stats_arr.push(&entry);
    }
    Reflect::set(&obj, &"stats".into(), &stats_arr).unwrap();

    Ok(obj.into())
}

/// convert individual ckmeans result classes to WASM-compatible Arrays
fn inner_vec_to_js_array(data: &[f64]) -> Float64Array {
    Float64Array::from(data)
}

/// Convert a ckmeans result to an Array suitable for use by a JS function
// NB: it's crucial to only work with slices here, as taking ownership of data will cause
// dangling references
fn wrapper_vec_to_js_array(data: &[Vec<f64>]) -> Array {
    data.iter().map(|v| inner_vec_to_js_array(v)).collect()
}
```

- [ ] **Step 2: Run tests**

Run: `cargo nextest r`
Expected: All 15 tests pass

- [ ] **Step 3: Run fmt and clippy**

Run: `cargo fmt && cargo clippy`

- [ ] **Step 4: Commit**

Run: `jj fix && jj commit -m "[WIP: claude] Reorder wasm.rs: public functions before private helpers"`

---

### Task 4: Final verification

- [ ] **Step 1: Full test suite**

Run: `cargo nextest r`
Expected: All 15 tests pass

- [ ] **Step 2: Clippy clean**

Run: `cargo clippy -- -D warnings`
Expected: No warnings

- [ ] **Step 3: Doc tests**

Run: `cargo test --doc`
Expected: All doc tests pass
