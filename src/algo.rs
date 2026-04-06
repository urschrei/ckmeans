use num_traits::Float;

use crate::CkNum;
use crate::ClusterStats;

/// Return a sorted copy of the input. Will blow up in the presence of NaN.
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
    pub(crate) rows: usize,
    pub(crate) cols: usize,
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
        let imin = k.max(1);
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
pub(crate) fn compute_cluster_stats<T: CkNum>(clusters: &[Vec<T>]) -> Option<Vec<ClusterStats<T>>> {
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
