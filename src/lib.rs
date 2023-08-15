//! Ckmeans clustering is an improvement on heuristic-based clustering
//! approaches like Jenks. The algorithm was developed by
//! [Haizhou Wang and Mingzhou Song](http://journal.r-project.org/archive/2011-2/RJournal_2011-2_Wang+Song.pdf)
//! (2011) as a [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) approach
//! to the problem of clustering numeric data into groups with the least
//! within-group sum-of-squared-deviations.
use std::{error::Error, num::TryFromIntError};

/// return a sorted **copy** of the input
fn numeric_sort(arr: &[i32]) -> Vec<i32> {
    let mut xs = arr.to_vec();
    xs.sort_unstable();
    xs
}

/// Assumes sorted input (so be sure only to use on `numeric_sort` output!)
fn unique_count_sorted(input: &mut [i32]) -> usize {
    if input.is_empty() {
        0
    } else {
        1 + input.windows(2).filter(|win| win[0] != win[1]).count()
    }
}

fn make_matrix(columns: usize, rows: usize) -> Vec<Vec<i32>> {
    let matrix: Vec<Vec<i32>> = (0..columns).map(|_| vec![0; rows]).collect();
    matrix
}

fn ssq(j: usize, i: usize, sumx: &[i32], sumxsq: &[i32]) -> Result<i32, TryFromIntError> {
    let sji = if j > 0 {
        let muji = (sumx[i] - sumx[j - 1]) / i32::try_from(i - j + 1)?;
        sumxsq[i] - sumxsq[j - 1] - i32::try_from(i - j + 1)? * muji.pow(2)
    } else {
        sumxsq[i] - (sumx[i] * sumx[i]) / i32::try_from(i + 1)?
    };
    if sji < 0 {
        Ok(0)
    } else {
        Ok(sji)
    }
}

fn fill_matrix_column(
    imin: usize,
    imax: usize,
    column: usize,
    matrix: &mut Vec<Vec<i32>>,
    backtrack_matrix: &mut Vec<Vec<i32>>,
    sumx: &[i32],
    sumxsq: &[i32],
) -> Result<(), TryFromIntError> {
    if imin > imax {
        return Ok(());
    }
    // Start at midpoint between imin and imax
    let i = imin + (imax - imin) / 2;
    matrix[column][i] = matrix[column - 1][i - 1];
    backtrack_matrix[column][i] = i32::try_from(i)?;
    let mut jlow = column;
    if imin > column {
        jlow = (jlow).max(usize::try_from(backtrack_matrix[column][imin - 1])?);
    }
    jlow = (jlow).max(usize::try_from(backtrack_matrix[column - 1][i])?);
    let mut jhigh = i - 1; // the upper end for j
    if imax < matrix[0].len() - 1 {
        jhigh = jhigh.min(usize::try_from(backtrack_matrix[column][imax + 1])?);
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
            backtrack_matrix[column][i] = i32::try_from(jlow)?;
        }
        jlow += 1;

        let ssqj = sji + matrix[column - 1][j - 1];
        if ssqj < matrix[column][i] {
            matrix[column][i] = ssqj;
            backtrack_matrix[column][i] = i32::try_from(j)?;
        }
    }
    fill_matrix_column(imin, i - 1, column, matrix, backtrack_matrix, sumx, sumxsq)?;
    fill_matrix_column(i + 1, imax, column, matrix, backtrack_matrix, sumx, sumxsq)?;
    Ok(())
}

fn fill_matrices(
    data: &[i32],
    matrix: &mut Vec<Vec<i32>>,
    backtrack_matrix: &mut Vec<Vec<i32>>,
    nclusters: usize,
) -> Result<(), TryFromIntError> {
    let nvalues = data.len();
    let mut sumx: Vec<i32> = vec![0; nvalues];
    let mut sumxsq: Vec<i32> = vec![0; nvalues];
    let shift = data[nvalues / 2];
    // Initialize first row in matrix & backtrack_matrix
    for i in 0..nvalues {
        // (0..nvalues).enumerate().for_each(|(_, i)| {
        if i == 0 {
            sumx[0] = data[0] - shift;
            sumxsq[0] = (data[0] - shift).pow(2);
        } else {
            sumx[i] = sumx[i - 1] + data[i] - shift;
            sumxsq[i] = sumxsq[i - 1] + (data[i] - shift) * (data[i] - shift);
        }
        // Initialize for k = 0
        matrix[0][i] = ssq(0, i, &sumx, &sumxsq)?;
        backtrack_matrix[0][i] = 0;
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
    Ok(())
}

/// Minimizing the difference within groups - what Wang & Song refer to as
/// `withinss`, or within sum-of-squares, means that groups are optimally
/// homogenous within and the data is split into representative groups.
/// This is very useful for visualization, where you may want to represent
/// a continuous variable in discrete colour or style groups. This function
/// can provide groups that emphasize differences between data.
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
/// # References
/// 1. [Wang, H., & Song, M. (2011). Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming. The R Journal, 3(2), 29.](https://doi.org/10.32614/RJ-2011-015)
/// 2. <https://observablehq.com/@visionscarto/natural-breaks>
pub fn ckmeans(data: &[i32], nclusters: i8) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    if nclusters == 0 {
        return Err("Can't generate 0 classes. Try a positive number.".into());
    }
    if usize::try_from(nclusters).expect("Couldn't convert i8 to usize") > data.len() {
        return Err("Can't generate more classes than data values".into());
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

    // This is a dynamic programming way to solve the problem of minimizing
    // within-cluster sum of squares. It's similar to linear regression
    // in this way, and this calculation incrementally computes the
    // sum of squares that are later read.
    fill_matrices(&sorted, &mut matrix, &mut backtrack_matrix, nclusters)?;

    // The real work of Ckmeans clustering happens in the matrix generation:
    // the generated matrices encode all possible clustering combinations, and
    // once they're generated we can solve for the best clustering groups
    // very quickly.
    let mut clusters: Vec<Vec<i32>> = Vec::with_capacity(nclusters);
    let mut cluster_right = backtrack_matrix[0].len() - 1;

    // Backtrack the clusters from the dynamic programming matrix. This
    // starts at the bottom-right corner of the matrix (if the top-left is 0, 0),
    // and moves the cluster target with the loop.
    for cluster in (0..backtrack_matrix.len()).rev() {
        let cluster_left = usize::try_from(backtrack_matrix[cluster][cluster_right])?;

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_clustering() {
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
}
