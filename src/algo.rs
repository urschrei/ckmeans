use fearless_simd::{Level, dispatch};
use num_traits::Float;

use crate::CkNum;
use crate::ClusterStats;

/// Return a sorted copy of the input, or `None` if the input contains a value
/// that is not comparable with itself (NaN).
pub(crate) fn numeric_sort<T: CkNum>(arr: &[T]) -> Option<Vec<T>> {
    if arr.iter().any(|x| x.partial_cmp(x).is_none()) {
        return None;
    }
    let mut xs = arr.to_vec();
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Some(xs)
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

    #[inline]
    pub(crate) fn row(&self, row: usize) -> &[T] {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }
}

/// Within-cluster sum of squares for the sorted segment `j..=i`.
///
/// The dynamic program is evaluated in `f64` regardless of the input element
/// type. Only the cluster *boundaries* (stored as `usize` indices) feed the
/// returned clustering, so the accumulator type does not affect the element
/// type of the result. Accumulating in `f64` also avoids the overflow the
/// previous element-typed accumulation was prone to for large integer inputs
/// (e.g. the squared deviations of `i32` data readily exceed `i32::MAX`).
#[inline(always)]
fn ssq(j: usize, i: usize, sumx: &[f64], sumxsq: &[f64]) -> f64 {
    let n = (i - j + 1) as f64;
    let sji = if j > 0 {
        let sum_diff = sumx[i] - sumx[j - 1];
        let muji = sum_diff / n;
        sumxsq[i] - sumxsq[j - 1] - n * muji * muji
    } else {
        let n_plus_one = (i + 1) as f64;
        sumxsq[i] - (sumx[i] * sumx[i]) / n_plus_one
    };
    if sji < 0.0 { 0.0 } else { sji }
}

/// Number of independent minimum searches in the lane loop of [`min_split`].
const LANES: usize = 4;

/// Below this range length, [`min_split`] uses a scalar loop. The set-up and
/// reduction of the lane loop cost more than they save on short ranges, and
/// most ranges in the divide-and-conquer search are short.
const LANE_THRESHOLD: usize = 16;

/// Return the minimum of `ssq(j, i) + prev[j - 1]` for `j` in `jlow..=jhigh`,
/// and the first `j` that gives it. `jlow` must be at least 1.
///
/// For long ranges, each of [`LANES`] lanes keeps its own minimum, so that the
/// compiler can vectorise the loop. Each cost uses the same operations as
/// [`ssq`], so the result is identical to that of a scalar search.
#[inline(always)]
fn min_split(
    i: usize,
    jlow: usize,
    jhigh: usize,
    prev: &[f64],
    sumx: &[f64],
    sumxsq: &[f64],
) -> (f64, usize) {
    debug_assert!(jlow >= 1 && jlow <= jhigh && jhigh <= i);
    // Offset m in these slices is j = jlow + m. Because j > 0, the costs use
    // the j > 0 branch of ssq.
    let start = jlow - 1;
    let len = jhigh - jlow + 1;
    let sx = &sumx[start..start + len];
    let sq = &sumxsq[start..start + len];
    let prev = &prev[start..start + len];
    let (sx_i, sq_i) = (sumx[i], sumxsq[i]);
    // Segment length at offset 0. Integers below 2^53 are exact in f64, so
    // n0 - m is equal to (i - j + 1) as f64.
    let n0 = (i - jlow + 1) as f64;
    let cost = |sxm: f64, sqm: f64, pm: f64, n: f64| {
        let muji = (sx_i - sxm) / n;
        let sji = sq_i - sqm - n * muji * muji;
        (if sji < 0.0 { 0.0 } else { sji }) + pm
    };

    if len < LANE_THRESHOLD {
        let mut best_cost = cost(sx[0], sq[0], prev[0], n0);
        let mut best_m = 0;
        for m in 1..len {
            let c = cost(sx[m], sq[m], prev[m], n0 - m as f64);
            if c < best_cost {
                best_cost = c;
                best_m = m;
            }
        }
        return (best_cost, jlow + best_m);
    }

    let mut lane_cost = [f64::INFINITY; LANES];
    let mut lane_m = [usize::MAX; LANES];
    let chunks = sx
        .chunks_exact(LANES)
        .zip(sq.chunks_exact(LANES))
        .zip(prev.chunks_exact(LANES));
    for (chunk, ((sxc, sqc), pc)) in chunks.enumerate() {
        let base = chunk * LANES;
        for lane in 0..LANES {
            let c = cost(sxc[lane], sqc[lane], pc[lane], n0 - (base + lane) as f64);
            let better = c < lane_cost[lane];
            lane_cost[lane] = if better { c } else { lane_cost[lane] };
            lane_m[lane] = if better { base + lane } else { lane_m[lane] };
        }
    }

    // Each lane holds its first minimum. Of equal minima, keep the smallest
    // offset.
    let mut best_cost = f64::INFINITY;
    let mut best_m = usize::MAX;
    for lane in 0..LANES {
        if lane_cost[lane] < best_cost || (lane_cost[lane] == best_cost && lane_m[lane] < best_m) {
            best_cost = lane_cost[lane];
            best_m = lane_m[lane];
        }
    }
    for m in (len / LANES) * LANES..len {
        let c = cost(sx[m], sq[m], prev[m], n0 - m as f64);
        if c < best_cost {
            best_cost = c;
            best_m = m;
        }
    }
    // If no cost is less than infinity, use the first j, as the scalar search
    // does.
    if best_m == usize::MAX {
        best_m = 0;
        best_cost = cost(sx[0], sq[0], prev[0], n0);
    }
    (best_cost, jlow + best_m)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn fill_matrix_column(
    imin: usize,
    imax: usize,
    column: usize,
    matrix: &mut FlatMatrix<f64>,
    backtrack_matrix: &mut FlatMatrix<usize>,
    sumx: &[f64],
    sumxsq: &[f64],
    stack: &mut Vec<(usize, usize)>,
) {
    // Reuse the pre-allocated stack for divide-and-conquer traversal
    stack.clear();
    stack.push((imin, imax));

    while let Some((imin, imax)) = stack.pop() {
        if imin > imax {
            continue;
        }

        // Start at midpoint between imin and imax
        let i = imin + (imax - imin) / 2;

        // Bound the optimal split point j using the monotonicity of optimal
        // splits (it is non-decreasing in both i and the cluster count). These
        // bounds drive the divide-and-conquer search; they are not the SMAWK
        // algorithm.
        let mut jlow = column;
        if imin > column {
            jlow = jlow.max(backtrack_matrix.get(column, imin - 1));
        }
        jlow = jlow.max(backtrack_matrix.get(column - 1, i));

        let mut jhigh = i;
        if imax < matrix.cols - 1 {
            jhigh = jhigh.min(backtrack_matrix.get(column, imax + 1));
        }

        let (best_cost, best_j) = min_split(i, jlow, jhigh, matrix.row(column - 1), sumx, sumxsq);
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
}

pub(crate) fn fill_matrices<T: CkNum>(
    data: &[T],
    matrix: &mut FlatMatrix<f64>,
    backtrack_matrix: &mut FlatMatrix<usize>,
    nclusters: usize,
) -> Option<()> {
    let nvalues = data.len();
    let mut sumx = Vec::with_capacity(nvalues);
    let mut sumxsq = Vec::with_capacity(nvalues);
    // Scale by a power of two so that the largest magnitude is near 1. Then the
    // squared deviations cannot overflow for finite input. Scaling by a power of
    // two is exact, and it multiplies every cost by the same factor, so it does
    // not change the split points. The input is sorted, so the largest magnitude
    // is at one of the ends. `to_f64` is the only fallible step; it cannot fail
    // for the standard numeric types but is propagated as `None`
    // (ConversionError) to be safe.
    let max_abs = data[0]
        .to_f64()?
        .abs()
        .max(data[nvalues - 1].to_f64()?.abs());
    let scale = if max_abs > 0.0 && max_abs.is_finite() {
        // Limit the scale to 2^1022 so that it stays finite for subnormal input.
        let exponent = (max_abs.log2().floor() as i32).max(f64::MIN_EXP - 1);
        2f64.powi(-exponent)
    } else {
        1.0
    };
    // Shift by a central value to improve the conditioning of the cumulative
    // sums.
    let shift = data[nvalues / 2].to_f64()? * scale;

    // Pre-compute sumx and sumxsq in f64
    let first = data[0].to_f64()? * scale - shift;
    sumx.push(first);
    sumxsq.push(first * first);
    for i in 1..nvalues {
        let shifted = data[i].to_f64()? * scale - shift;
        sumx.push(sumx[i - 1] + shifted);
        sumxsq.push(sumxsq[i - 1] + shifted * shifted);
    }

    // Initialize matrix for k = 0
    for i in 0..nvalues {
        matrix.set(0, i, ssq(0, i, &sumx, &sumxsq));
        backtrack_matrix.set(0, i, 0);
    }

    // Pre-allocate stack for divide-and-conquer (reused across columns)
    // Maximum depth is log2(n) + 1 for binary tree traversal
    let stack_capacity = ((nvalues as f64).log2().ceil() as usize).max(1) + 1;
    let mut stack = Vec::with_capacity(stack_capacity);

    // Compile the column loop for each SIMD level, and select the best level
    // that the CPU supports. The functions that the loop calls are
    // #[inline(always)], so they also compile for that level.
    dispatch!(Level::new(), _simd => {
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
            );
        }
    });
    Some(())
}

/// Fill the dynamic programming matrices for up to `nclusters` clusters of `sorted`, and return
/// the backtrack matrix. Each row depends only on the rows above it, so the first k rows give
/// the clustering for k clusters, for each k in `1..=nclusters`.
pub(crate) fn backtrack_matrix<T: CkNum>(
    sorted: &[T],
    nclusters: usize,
) -> Option<FlatMatrix<usize>> {
    // named 'S' originally. The dynamic program runs in f64 regardless of the
    // input element type; only the backtrack indices feed the result.
    let mut matrix = FlatMatrix::<f64>::new(nclusters, sorted.len());
    // named 'J' originally - store as usize to avoid conversions
    let mut backtrack_matrix = FlatMatrix::<usize>::new(nclusters, sorted.len());
    // This is a dynamic programming approach to solving the problem of minimizing
    // within-cluster sum of squares. It's similar to linear regression
    // in this way, and this calculation incrementally computes the
    // sum of squares that are later read.
    fill_matrices(sorted, &mut matrix, &mut backtrack_matrix, nclusters)?;
    Some(backtrack_matrix)
}

/// Return the first and last index of each cluster for `nclusters` clusters. The backtrack
/// matrix must have at least `nclusters` rows.
pub(crate) fn backtrack(
    backtrack_matrix: &FlatMatrix<usize>,
    nclusters: usize,
) -> Vec<(usize, usize)> {
    debug_assert!(nclusters <= backtrack_matrix.rows);
    let mut indices: Vec<(usize, usize)> = Vec::with_capacity(nclusters);
    let mut cluster_right = backtrack_matrix.cols - 1;

    // Backtrack the clusters from the dynamic programming matrix. This
    // starts at the last column of row nclusters - 1 (if the top-left is 0, 0),
    // and moves the cluster target with the loop.
    for cluster in (0..nclusters).rev() {
        let cluster_left = backtrack_matrix.get(cluster, cluster_right);

        // Store the indices instead of copying data
        indices.push((cluster_left, cluster_right));
        if cluster > 0 {
            cluster_right = cluster_left - 1;
        }
    }
    indices.reverse();
    indices
}

/// Return the roundest number `b` with `low < b <= high`.
///
/// The function uses the largest power of ten that has a multiple in the
/// interval, and returns the multiple nearest the midpoint. If no candidate is
/// in the interval (for example because the gap is subnormal, a bound is
/// infinite, or `low >= high`), it returns `high`. `None` is a failed numeric
/// conversion.
pub(crate) fn round_break<T: Float>(low: T, high: T) -> Option<T> {
    if low >= high || !low.is_finite() || !high.is_finite() {
        return Some(high);
    }
    let two = T::from(2.0)?;
    let middle = low / two + high / two;
    let coarsest = low.abs().max(high.abs()).log10().ceil().to_i32()?;
    // The gap overflows to infinity only if it spans most of the float range.
    let finest = (high - low)
        .log10()
        .floor()
        .to_i32()
        .map_or(coarsest, |e| e - 1);
    for exponent in (finest..=coarsest).rev() {
        if let Some(candidate) = nearest_multiple_in(low, high, middle, exponent) {
            return Some(candidate);
        }
    }
    Some(high)
}

/// Return the multiple of `10^exponent` nearest `middle` with
/// `low < multiple <= high`, if one exists.
fn nearest_multiple_in<T: Float>(low: T, high: T, middle: T, exponent: i32) -> Option<T> {
    let ten = T::from(10.0)?;
    // Divide by a positive power of ten for negative exponents, so that the
    // result is the float nearest the decimal value.
    let factor = ten.powi(exponent.abs());
    if !factor.is_finite() {
        return None;
    }
    let to_units = |x: T| if exponent < 0 { x * factor } else { x / factor };
    let from_units = |k: T| if exponent < 0 { k / factor } else { k * factor };
    let nearest = to_units(middle).round();
    [nearest, nearest - T::one(), nearest + T::one()]
        .into_iter()
        .map(from_units)
        .find(|&candidate| low < candidate && candidate <= high)
}

/// Compute per-cluster statistics (center, size, withinss) for a set of sorted clusters.
pub(crate) fn compute_cluster_stats<T: CkNum>(clusters: &[Vec<T>]) -> Option<Vec<ClusterStats<T>>> {
    clusters
        .iter()
        .map(|cluster| {
            let size = cluster.len();
            let n = T::from_usize(size)?;
            // Sum the offsets from the first value, not the values. Then a
            // cluster of equal finite values gets that value as its centre. The cluster
            // is sorted: rounding can push the centre above the last value, so
            // clamp it.
            let (&low, &high) = (cluster.first()?, cluster.last()?);
            let offset: T = cluster
                .iter()
                .copied()
                .fold(T::zero(), |acc, x| acc + (x - low));
            let mut center = low + offset / n;
            if center > high {
                center = high;
            }
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

/// Return the mean and the sample variance of `x`. Use offsets from the middle value, as
/// `shifted_data_variance()` in the Ckmeans.1d.dp C++ code does.
fn shifted_mean_variance(x: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let middle = x[(x.len() - 1) / 2];
    let (sum, sum_sq) = x.iter().fold((0.0, 0.0), |(sum, sum_sq), &v| {
        let d = v - middle;
        (sum + d, sum_sq + d * d)
    });
    let mean = sum / n + middle;
    // Rounding can make the difference negative
    let variance = if x.len() > 1 {
        ((sum_sq - sum * sum / n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };
    (mean, variance)
}

/// Compute the BIC of a clustering of `sorted` under a Gaussian mixture model. Each element of
/// `ranges` gives the first and last index of a cluster. A lower value is better.
///
/// This is the calculation in `select_levels()` of the Ckmeans.1d.dp C++ code (Song & Zhong,
/// 2020), which reports the negative of this value:
/// - Cluster j gives a component with weight n_j / n, the mean of the cluster, and the sample
///   variance of the cluster.
/// - Let d be the smallest distance from the cluster to an adjacent value outside it. A
///   component with zero variance gets the variance d^2 / 36. A component of one value gets d^2.
///   No variance is less than `f64::MIN_POSITIVE`.
/// - The log-likelihood ln L is the sum over all values of the log of the mixture density.
/// - BIC = -2 ln L + (3k - 1) ln n.
///
/// The calculation is in `f64`, and it uses the log of each density. It is O(nk), but it does not
/// call `exp()` for a density that is less than 2^-60 of the largest density at that value, and it
/// calls `ln()` once for each 64 values.
/// Where the density of a value underflows, the C++ code gives an infinite BIC, and this
/// function gives a finite value.
pub(crate) fn compute_bic<T: CkNum + Float>(sorted: &[T], ranges: &[(usize, usize)]) -> Option<T> {
    let x: Vec<f64> = sorted.iter().map(|v| v.to_f64()).collect::<Option<_>>()?;
    let n = x.len() as f64;

    // (ln(weight / sqrt(2 pi variance)), mean, 1 / (2 * variance)) for each component
    let components: Vec<(f64, f64, f64)> = ranges
        .iter()
        .map(|&(left, right)| {
            let size = right - left + 1;
            let (mean, mut variance) = shifted_mean_variance(&x[left..=right]);
            if variance == 0.0 || size == 1 {
                let gap_left = left.checked_sub(1).map(|i| x[left] - x[i]);
                let gap_right = x.get(right + 1).map(|&next| next - x[right]);
                let gap = match (gap_left, gap_right) {
                    (Some(l), Some(r)) => Some(l.min(r)),
                    (l, r) => l.or(r),
                };
                // There is no adjacent value only if all values are in one cluster
                if let Some(d) = gap {
                    if variance == 0.0 {
                        variance = d * d / 36.0;
                    }
                    if size == 1 {
                        variance = d * d;
                    }
                }
            }
            // The variance underflows to zero if the values are very close. The C++ code
            // does not set a minimum, and its BIC is NaN for these clusters.
            variance = variance.max(f64::MIN_POSITIVE);
            let log_coeff = (size as f64 / n).ln() - 0.5 * (std::f64::consts::TAU * variance).ln();
            (log_coeff, mean, 0.5 / variance)
        })
        .collect();

    // Sum the densities relative to the largest, and skip the exp() of each density that is less
    // than 2^-60 of the largest. The skipped densities change the sum by at most k * 2^-60 of
    // its value.
    let skip = 60.0 * std::f64::consts::LN_2;
    let mut log_densities = vec![0.0; components.len()];
    let mut log_likelihood = 0.0;
    // Each relative sum is in [1, k], and k is at most 255. The product of 64 relative sums is
    // at most 255^64 < 1e155, so one ln() for each 64 values is sufficient.
    for chunk in x.chunks(64) {
        let mut product = 1.0;
        for &v in chunk {
            let mut largest = f64::NEG_INFINITY;
            for (log_density, &(log_coeff, mean, inv_two_var)) in
                log_densities.iter_mut().zip(&components)
            {
                *log_density = log_coeff - (v - mean) * (v - mean) * inv_two_var;
                if log_density.is_nan() {
                    return T::from_f64(f64::NAN);
                }
                largest = largest.max(*log_density);
            }
            log_likelihood += largest;
            if largest == f64::NEG_INFINITY {
                continue;
            }
            product *= log_densities
                .iter()
                .filter(|&&log_density| log_density > largest - skip)
                .map(|&log_density| (log_density - largest).exp())
                .sum::<f64>();
        }
        log_likelihood += product.ln();
    }

    // p = 3k - 1: k means, k variances and k - 1 weights
    let p = (3 * ranges.len() - 1) as f64;
    T::from_f64(-2.0 * log_likelihood + p * n.ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hegel::TestCase;
    use hegel::generators::{self as gs, Generator};

    /// Return the minimum cost and the first `j` that gives it, with a scalar
    /// search over `jlow..=jhigh`.
    fn scalar_min_split(
        i: usize,
        jlow: usize,
        jhigh: usize,
        prev: &[f64],
        sumx: &[f64],
        sumxsq: &[f64],
    ) -> (f64, usize) {
        let mut best_j = jlow;
        let mut best_cost = ssq(jlow, i, sumx, sumxsq) + prev[jlow - 1];
        for j in (jlow + 1)..=jhigh {
            let cost = ssq(j, i, sumx, sumxsq) + prev[j - 1];
            if cost < best_cost {
                best_cost = cost;
                best_j = j;
            }
        }
        (best_cost, best_j)
    }

    /// Values from a small pool make equal costs frequent, so the test
    /// examines the tie-break.
    #[hegel::test(test_cases = 2000)]
    fn min_split_agrees_with_scalar_search(tc: TestCase) {
        let len = tc.draw(gs::integers::<usize>().min_value(2).max_value(200));
        let mut data = tc.draw(
            gs::vecs(hegel::one_of!(
                gs::floats::<f64>().min_value(-1.0).max_value(1.0),
                gs::integers::<u8>().max_value(4).map(f64::from),
            ))
            .min_size(len)
            .max_size(len),
        );
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let prev = tc.draw(
            gs::vecs(hegel::one_of!(
                gs::floats::<f64>().min_value(0.0).max_value(10.0),
                gs::integers::<u8>().max_value(2).map(f64::from),
            ))
            .min_size(len)
            .max_size(len),
        );
        let mut sumx = Vec::with_capacity(len);
        let mut sumxsq = Vec::with_capacity(len);
        let (mut sx, mut sq) = (0.0, 0.0);
        for &x in &data {
            sx += x;
            sq += x * x;
            sumx.push(sx);
            sumxsq.push(sq);
        }
        let i = tc.draw(gs::integers::<usize>().min_value(1).max_value(len - 1));
        let jlow = tc.draw(gs::integers::<usize>().min_value(1).max_value(i));
        let jhigh = tc.draw(gs::integers::<usize>().min_value(jlow).max_value(i));
        let expected = scalar_min_split(i, jlow, jhigh, &prev, &sumx, &sumxsq);
        let actual = min_split(i, jlow, jhigh, &prev, &sumx, &sumxsq);
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1, expected.1);
    }
}
