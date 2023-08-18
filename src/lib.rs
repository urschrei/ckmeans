//! Ckmeans clustering is an improvement on heuristic-based 1-dimensional clustering
//! approaches such as Jenks. The algorithm was developed by
//! [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf)
//! (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach
//! to the problem of clustering numeric data into groups with the least
//! within-group sum-of-squared-deviations.

use num_traits::cast::FromPrimitive;
use num_traits::Float;
use num_traits::{Num, NumCast};
use std::fmt::Debug;

#[cfg(not(target_arch = "wasm32"))]
mod ffi;
#[cfg(not(target_arch = "wasm32"))]
pub use crate::ffi::{
    ckmeans_ffi, drop_ckmeans_result, ExternalArray, InternalArray, WrapperArray,
};

mod errors;
pub use crate::errors::CkmeansErr;

/// A trait that encompasses most common numeric types (integer **and** floating point)
pub trait CkNum: Num + Copy + NumCast + PartialOrd + FromPrimitive + Debug {}
impl<T: Num + Copy + NumCast + PartialOrd + FromPrimitive + Debug> CkNum for T {}

/// return a sorted **copy** of the input. Will blow up in the presence of NaN
fn numeric_sort<T: CkNum>(arr: &[T]) -> Vec<T> {
    let mut xs = arr.to_vec();
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    xs
}

/// Assumes sorted input (so be sure only to use on `numeric_sort` output!)
fn unique_count_sorted<T: CkNum>(input: &mut [T]) -> usize {
    if input.is_empty() {
        0
    } else {
        1 + input.windows(2).filter(|win| win[0] != win[1]).count()
    }
}

fn make_matrix<T: CkNum>(columns: usize, rows: usize) -> Vec<Vec<T>> {
    let matrix: Vec<Vec<T>> = (0..columns).map(|_| vec![T::zero(); rows]).collect();
    matrix
}

#[inline(always)]
fn ssq<T: CkNum>(j: usize, i: usize, sumx: &[T], sumxsq: &[T]) -> Option<T> {
    let sji = if j > 0 {
        let muji = (sumx[i] - sumx[j - 1]) / T::from_usize(i - j + 1)?;
        sumxsq[i] - sumxsq[j - 1] - T::from_usize(i - j + 1)? * muji * muji
    } else {
        sumxsq[i] - (sumx[i] * sumx[i]) / T::from_usize(i + 1)?
    };
    if sji < T::zero() {
        Some(T::zero())
    } else {
        Some(sji)
    }
}

fn fill_matrix_column<T: CkNum>(
    imin: usize,
    imax: usize,
    column: usize,
    matrix: &mut Vec<Vec<T>>,
    backtrack_matrix: &mut Vec<Vec<T>>,
    sumx: &[T],
    sumxsq: &[T],
) -> Option<()>
where
{
    if imin > imax {
        return Some(());
    }
    // Start at midpoint between imin and imax
    let i = imin + (imax - imin) / 2;
    matrix[column][i] = matrix[column - 1][i - 1];
    backtrack_matrix[column][i] = T::from_usize(i)?;
    let mut jlow = column;
    if imin > column {
        jlow = (jlow).max(T::to_usize(&backtrack_matrix[column][imin - 1])?);
    }
    jlow = (jlow).max(T::to_usize(&backtrack_matrix[column - 1][i])?);
    let mut jhigh = i - 1; // the upper end for j
    if imax < matrix[0].len() - 1 {
        jhigh = jhigh.min(T::to_usize(&backtrack_matrix[column][imax + 1])?);
    }
    for j in (jlow..jhigh + 1).rev() {
        let sji = ssq(j, i, sumx, sumxsq)?;
        if sji + matrix[column - 1][jlow - 1] >= matrix[column][i] {
            break;
        }
        let sjlowi = ssq(jlow, i, sumx, sumxsq)?;

        let ssqjlow = sjlowi + matrix[column - 1][jlow - 1];
        if ssqjlow < matrix[column][i] {
            // shrink the lower bound
            matrix[column][i] = ssqjlow;
            backtrack_matrix[column][i] = T::from_usize(jlow)?;
        }
        jlow += 1;

        let ssqj = sji + matrix[column - 1][j - 1];
        if ssqj < matrix[column][i] {
            matrix[column][i] = ssqj;
            backtrack_matrix[column][i] = T::from_usize(j)?;
        }
    }
    fill_matrix_column(imin, i - 1, column, matrix, backtrack_matrix, sumx, sumxsq)?;
    fill_matrix_column(i + 1, imax, column, matrix, backtrack_matrix, sumx, sumxsq)?;
    Some(())
}

fn fill_matrices<T: CkNum>(
    data: &[T],
    matrix: &mut Vec<Vec<T>>,
    backtrack_matrix: &mut Vec<Vec<T>>,
    nclusters: usize,
) -> Option<()>
where
{
    let nvalues = data.len();
    let mut sumx: Vec<T> = vec![T::zero(); nvalues];
    let mut sumxsq: Vec<T> = vec![T::zero(); nvalues];
    let shift = data[nvalues / 2];
    // Initialize first row in matrix & backtrack_matrix
    for i in 0..nvalues {
        if i == 0 {
            sumx[0] = data[0] - shift;
            sumxsq[0] = (data[0] - shift) * (data[0] - shift);
        } else {
            sumx[i] = sumx[i - 1] + data[i] - shift;
            sumxsq[i] = sumxsq[i - 1] + (data[i] - shift) * (data[i] - shift);
        }
        // Initialize for k = 0
        matrix[0][i] = ssq(0, i, &sumx, &sumxsq)?;
        backtrack_matrix[0][i] = T::zero();
    }
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
        )?;
    }
    Some(())
}

/// Minimizing the difference within groups – what Wang & Song refer to as
/// `withinss`, or within sum-of-squares, means that groups are **optimally
/// homogenous** within and the data is split into representative groups.
/// This is very useful for visualization, where one may wish to represent
/// a continuous variable in discrete colour or style groups. This function
/// can provide groups – or "classes" – that emphasize differences between data.
///
/// Being a dynamic approach, this algorithm is based on two matrices that
/// store incrementally-computed values for squared deviations and backtracking
/// indexes.
///
/// Unlike the [original implementation](https://cran.r-project.org/web/packages/Ckmeans.1d.dp/index.html),
/// this implementation does not include any code to automatically determine
/// the optimal number of clusters: this information needs to be explicitly
/// provided.
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
    if nclusters == 0 {
        return Err(CkmeansErr::TooFewClassesError);
    }
    if usize::try_from(nclusters)? > data.len() {
        return Err(CkmeansErr::TooManyClassesError);
    }
    let nvalues = data.len();
    let mut sorted = numeric_sort(data);
    // we'll use this as the maximum number of clusters
    let unique_count = unique_count_sorted(&mut sorted);
    // if all of the input values are identical, there's one cluster
    // with all of the input in it.
    if unique_count == 1 {
        return Ok(vec![sorted]);
    }
    let nclusters = unique_count.min(usize::try_from(nclusters)?);

    // named 'S' originally
    let mut matrix = make_matrix(nclusters, nvalues);
    // named 'J' originally
    let mut backtrack_matrix = make_matrix(nclusters, nvalues);

    // This is a dynamic programming approach to solving the problem of minimizing
    // within-cluster sum of squares. It's similar to linear regression
    // in this way, and this calculation incrementally computes the
    // sum of squares that are later read.
    fill_matrices(&sorted, &mut matrix, &mut backtrack_matrix, nclusters)
        .ok_or(CkmeansErr::ConversionError)?;

    // The real work of Ckmeans clustering happens in the matrix generation:
    // the generated matrices encode all possible clustering combinations, and
    // once they're generated we can solve for the best clustering groups
    // very quickly.
    let mut clusters: Vec<Vec<T>> = Vec::with_capacity(nclusters);
    let mut cluster_right = backtrack_matrix[0].len() - 1;

    // Backtrack the clusters from the dynamic programming matrix. This
    // starts at the bottom-right corner of the matrix (if the top-left is 0, 0),
    // and moves the cluster target with the loop.
    for cluster in (0..backtrack_matrix.len()).rev() {
        let cluster_left = T::to_usize(&backtrack_matrix[cluster][cluster_right])
            .ok_or(CkmeansErr::ConversionError)?;

        // fill the cluster from the sorted input by taking a slice of the
        // array. the backtrack matrix makes this easy: it stores the
        // indexes where the cluster should start and end.
        clusters.push(sorted[cluster_left..cluster_right + 1].to_vec());
        if cluster > 0 {
            cluster_right = cluster_left - 1;
        }
    }
    clusters.reverse();
    Ok(clusters)
}

/// The boundaries of the classes returned by [ckmeans] are "ugly" in the sense that the values
/// returned are the lower bound of each cluster, which can’t be used for labelling, since they
/// might have many decimal places. To create a legend, the values should be rounded — but the
/// rounding might be either too loose (and would result in spurious decimal places), or too strict,
/// resulting in classes ranging “from x to x”. A better approach is to choose the roundest number that
/// separates the lowest point from a class from the highest point
/// in the _preceding_ class — thus giving just enough precision to distinguish the classes.
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
        .collect::<Result<Vec<T>, _>>()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_clustering_integers() {
        let i = vec![
            1, 12, 13, 14, 15, 16, 2, 2, 3, 5, 7, 1, 2, 5, 7, 1, 5, 82, 1, 1, 1, 78,
        ];
        let expected = vec![
            vec![1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 5, 5, 5, 7, 7],
            vec![12, 13, 14, 15, 16],
            vec![78, 82],
        ];
        let res = ckmeans(&i, 3).unwrap();
        assert_eq!(res, expected)
    }

    #[test]
    fn test_clustering_floats() {
        let i = vec![
            1f64, 12., 13., 14., 15., 16., 2., 2., 3., 5., 7., 1., 2., 5., 7., 1., 5., 82., 1.,
            1.3, 1.1, 78.,
        ];
        let expected = vec![
            vec![
                1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 2.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0,
            ],
            vec![12., 13., 14., 15., 16.],
            vec![78., 82.],
        ];
        let res = ckmeans(&i, 3).unwrap();
        assert_eq!(res, expected)
    }
    #[test]
    fn test_roundbreaks() {
        let i = vec![
            1f64, 12., 13., 14., 15., 16., 2., 2., 3., 5., 7., 1., 2., 5., 7., 1., 5., 82., 1.,
            1.3, 1.1, 78.,
        ];
        let expected = vec![9.0, 40.0];
        let res = roundbreaks(&i, 3).unwrap();
        assert_eq!(res, expected)
    }
}
