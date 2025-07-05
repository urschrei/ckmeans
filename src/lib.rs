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
mod wasm;
pub use crate::wasm::{ckmeans_wasm, roundbreaks_wasm};

mod errors;
pub use crate::errors::CkmeansErr;

/// Result type for ckmeans_indices: (sorted_data, cluster_ranges)
pub type ClusterIndices<T> = (Vec<T>, Vec<(usize, usize)>);

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

/// Flat matrix structure for better cache locality
struct FlatMatrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: CkNum> FlatMatrix<T> {
    fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::zero(); rows * cols],
            rows,
            cols,
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> T {
        self.data[row * self.cols + col]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, value: T) {
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

fn fill_matrix_column<T: CkNum>(
    imin: usize,
    imax: usize,
    column: usize,
    matrix: &mut FlatMatrix<T>,
    backtrack_matrix: &mut FlatMatrix<usize>,
    sumx: &[T],
    sumxsq: &[T],
) -> Option<()> {
    // Stack to simulate recursion: (imin, imax)
    // Maximum depth is log2(n) for binary tree traversal
    let capacity = if imax > imin {
        ((imax - imin + 1) as f64).log2().ceil() as usize + 1
    } else {
        1
    };
    let mut stack = Vec::with_capacity(capacity);
    stack.push((imin, imax));

    while let Some((imin, imax)) = stack.pop() {
        if imin > imax {
            continue;
        }

        // Start at midpoint between imin and imax
        let i = imin + (imax - imin) / 2;
        matrix.set(column, i, matrix.get(column - 1, i - 1));
        backtrack_matrix.set(column, i, i);
        let mut jlow = column;
        if imin > column {
            jlow = (jlow).max(backtrack_matrix.get(column, imin - 1));
        }
        jlow = (jlow).max(backtrack_matrix.get(column - 1, i));
        let mut jhigh = i - 1; // the upper end for j
        if imax < matrix.cols - 1 {
            jhigh = jhigh.min(backtrack_matrix.get(column, imax + 1));
        }
        for j in (jlow..=jhigh).rev() {
            let sji = ssq(j, i, sumx, sumxsq)?;
            if sji + matrix.get(column - 1, jlow - 1) >= matrix.get(column, i) {
                break;
            }
            let sjlowi = ssq(jlow, i, sumx, sumxsq)?;

            let ssqjlow = sjlowi + matrix.get(column - 1, jlow - 1);
            if ssqjlow < matrix.get(column, i) {
                // shrink the lower bound
                matrix.set(column, i, ssqjlow);
                backtrack_matrix.set(column, i, jlow);
            }
            jlow += 1;

            let ssqj = sji + matrix.get(column - 1, j - 1);
            if ssqj < matrix.get(column, i) {
                matrix.set(column, i, ssqj);
                backtrack_matrix.set(column, i, j);
            }
        }

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

fn fill_matrices<T: CkNum>(
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
    let mut sorted = numeric_sort(data);
    // we'll use this as the maximum number of clusters
    let unique_count = unique_count_sorted(&mut sorted);
    // if all of the input values are identical, there's one cluster
    // with all of the input in it.
    if unique_count == 1 {
        return Ok((sorted, vec![(0, nvalues - 1)]));
    }
    let nclusters = unique_count.min(nclusters as usize);

    // named 'S' originally
    let mut matrix = FlatMatrix::new(nclusters, nvalues);
    // named 'J' originally - store as usize to avoid conversions
    let mut backtrack_matrix = FlatMatrix::<usize>::new(nclusters, nvalues);

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
/// returned are the lower bound of each cluster, which can’t be used for labelling, since they
/// might have many decimal places. To create a legend, the values should be rounded — but the
/// rounding might be either too loose (and would result in spurious decimal places), or too strict,
/// resulting in classes ranging “from x to x”. A better approach is to choose the roundest number that
/// separates the lowest point from a class from the highest point
/// in the _preceding_ class — thus giving just enough precision to distinguish the classes.
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
        let i = vec![
            1, 12, 13, 14, 15, 16, 2, 2, 3, 5, 7, 1, 2, 5, 7, 1, 5, 82, 1, 1, 1, 78,
        ];
        let expected = vec![
            vec![1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 5, 5, 5, 7, 7],
            vec![12, 13, 14, 15, 16],
            vec![78, 82],
        ];
        let res = ckmeans(&i, 3).unwrap();
        assert_eq!(res, expected);
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
        assert_eq!(res, expected);
    }
    #[test]
    fn test_roundbreaks() {
        // this is the example data from Observable
        let numbers = vec![
            2.186_880_969_969_168,
            2.693_032_112_550_337,
            3.931_802_019_562_82,
            1.885_849_946_021_524_8,
            3.289_199_280_095_574_3,
            3.958_993_479_120_063_6,
            1.769_479_604_490_788_3,
            3.170_823_246_601_968_6,
            3.713_810_381_488_059_4,
            1.808_400_336_999_689_7,
            2.947_289_457_238_938,
            4.079_716_765_942_393,
            2.019_481_213_579_366_5,
            3.367_740_732_705_309,
            3.946_018_838_624_863,
            1.705_437_159_404_827,
            2.882_595_951_213_505_5,
            3.985_849_681_199_300_5,
            1.181_831_719_229_522_5,
            3.195_988_261_190_872_3,
            3.797_595_260_081_815_5,
            2.326_000_355_045_864_3,
            3.079_009_759_179_15,
            3.972_033_030_026_513_4,
            2.092_492_802_579_194_7,
            3.414_985_824_526_252_8,
            3.999_784_164_947_667,
            2.038_037_585_951_181_5,
            2.561_038_262_661_651,
            4.235_818_645_004_002,
            1.940_708_488_392_699_5,
            3.098_653_285_349_226_6,
            4.187_154_244_559_768,
            1.937_453_693_800_098_8,
            3.301_289_735_332_64,
            4.073_525_037_840_789,
            1.621_995_058_521_678_7,
            2.951_665_262_252_112_2,
            3.837_950_983_403_334_5,
            1.830_122_797_495_506_3,
            3.058_706_691_625_945_3,
            4.038_198_749_869_919,
            1.846_312_203_680_539_3,
            2.874_469_711_585_369,
            3.739_250_728_698_104,
            1.883_918_293_609_806_7,
            3.339_856_903_842_820_7,
            4.041_222_857_529_328,
            2.158_078_147_639_536_3,
            2.789_573_267_050_152_5,
            3.670_343_274_157_198,
            2.075_207_620_258_03,
            3.026_907_608_629_054_7,
            4.068_500_990_714_395,
            2.150_766_291_210_518_8,
            2.786_945_124_615_935_6,
            4.060_256_411_136_024,
            2.160_538_596_948_854,
            3.591_166_842_041_902_3,
            3.936_942_667_005_152,
            1.782_374_974_381_155_4,
            2.398_556_760_000_693,
            3.858_072_677_727_023_3,
            2.276_857_198_356_403,
            2.667_470_377_021_615_5,
            3.989_313_264_412_629_2,
            2.126_263_439_397_846,
            2.887_148_841_450_299,
            4.006_450_415_376_944_5,
            2.384_214_172_538_217,
            3.158_494_526_369_261_6,
            3.917_391_700_186_437,
            2.045_842_731_040_772,
            2.896_606_957_300_526_6,
            4.061_877_233_021_508,
            2.502_073_084_064_212_7,
            2.467_027_330_439_492,
            4.059_122_876_412_696_5,
            1.926_764_841_918_448_9,
            3.184_736_160_134_685,
            3.798_709_413_777_763_3,
            2.407_891_897_908_353,
            3.113_035_170_650_594_5,
            4.305_405_220_398_142,
            3.020_832_358_740_857,
            3.089_645_992_537_316_3,
            4.337_513_332_122_615,
            2.151_389_798_283_614_8,
            3.389_065_214_915_219_4,
            4.147_203_390_905_856,
            1.744_226_739_948_633_8,
            3.120_497_422_595_793_3,
            4.202_793_281_453_125_5,
            1.820_521_814_474_723_5,
            2.514_571_652_914_36,
            4.196_065_588_365_519,
            2.324_601_357_283_982_8,
            3.650_545_226_495_212,
            3.995_107_700_529_213_3,
            1.478_649_263_021_916_3,
            3.184_728_841_579_185,
            4.047_736_697_459_269_5,
            2.098_834_005_399_845,
            3.303_175_986_466_117,
            3.789_228_807_165_397_2,
            2.624_470_149_286_327_4,
            2.889_489_714_416_799_2,
            3.970_265_133_933_609,
            2.113_416_007_212_771,
            2.837_726_973_525_549_5,
            4.098_857_511_413,
            2.220_921_415_338_863_4,
            3.002_366_818_476_632,
            4.283_496_512_420_427,
            1.558_156_616_826_829_5,
            2.980_756_646_018_853,
            3.910_127_425_359_612,
            1.490_739_212_060_197,
            2.895_513_344_823_693,
            4.247_576_525_913_251,
            2.189_206_900_634_349_8,
            2.998_806_078_461_756,
            3.859_209_377_540_056_4,
            1.586_726_854_167_168_6,
            3.152_649_521_951_604,
            3.824_440_845_159_143_6,
            3.110_387_134_652_678_6,
            2.484_970_361_633_695_6,
            4.156_025_918_520_517,
            1.511_172_143_351_513_5,
            3.750_816_172_762_316,
            3.932_338_638_345_120_4,
            2.076_949_927_679_19,
            3.208_154_398_640_064_5,
            4.045_289_073_742_084,
            1.976_044_516_014_291_9,
            3.031_327_170_083_975,
            4.204_267_226_311_512,
            1.939_214_053_185_361,
            3.323_788_480_108_269,
            3.747_981_740_872_601_3,
            2.782_673_358_706_148_7,
            3.240_250_893_521_295,
            3.778_322_920_441_067,
            2.402_897_199_559_954_6,
            3.049_035_907_252_789_3,
            3.941_361_122_588_998_6,
            1.914_818_926_554_815_2,
            2.944_080_059_198_642,
            4.267_406_847_388_78,
            2.058_582_756_874_975_5,
            2.782_341_912_819_006,
            3.797_216_217_333_120_5,
            1.619_726_539_827_832_2,
            3.255_938_819_736_093_2,
            4.220_325_996_536_666,
            1.981_927_944_233_005_8,
            3.053_309_950_903_032,
            4.005_034_457_047_913_5,
            3.454_880_797_184_877_4,
            3.171_306_397_902_508_7,
            3.792_042_274_564_493_3,
            2.687_431_903_142_606,
            3.043_534_128_823_437_4,
            3.936_218_880_097_710_5,
            1.384_430_767_708_770_2,
            2.995_055_995_775_735_4,
            3.904_845_590_592_921,
            3.061_504_913_041_073,
            3.130_758_322_831_024_3,
            4.317_604_148_100_834,
            1.291_369_155_045_615,
            3.443_142_692_197_454,
            4.167_123_158_127_977,
            1.285_783_110_744_627_8,
            2.743_134_331_614_822_7,
            3.705_373_366_302_743_5,
            2.380_416_856_603_215_3,
            2.887_674_496_702_909,
            3.990_541_840_841_131_3,
            1.671_923_655_611_446_7,
            2.985_002_677_898_493_5,
            4.146_972_397_533_571,
            2.079_701_208_980_980_5,
            2.951_096_908_063_335,
            3.793_706_799_242_936_5,
            3.001_063_205_135_309,
            3.134_924_808_221_958_5,
            4.022_206_004_589_426,
            1.545_495_981_764_916,
            2.896_630_804_004_962_6,
            4.026_750_229_754_802,
            2.455_019_376_713_662_5,
            3.104_846_667_702_584,
            4.170_108_463_306_901,
            1.367_053_071_130_132_3,
            2.832_456_174_517_439_3,
            4.098_799_538_338_867,
            1.814_066_923_533_963,
            2.581_112_819_622_158,
            3.779_921_234_228_462_3,
            1.115_008_897_674_243,
            3.103_260_260_015_172,
            3.937_589_020_729_37,
            2.411_956_649_166_637,
            3.351_700_352_820_514,
            4.022_640_879_373_81,
            2.714_506_909_365_993,
            2.844_309_361_015_004,
            3.787_479_119_468_31,
            0.683_725_055_383_285_7,
            2.971_586_707_439_505,
            4.311_768_255_823_228,
            1.435_791_482_915_398_4,
            2.931_274_470_207_522,
            3.906_562_719_609_756,
            0.758_811_983_900_105_7,
            3.136_088_252_208_502,
            3.885_505_169_010_581,
            2.831_050_009_570_089,
            3.236_269_875_814_2,
            3.982_431_100_799_526,
            1.979_936_487_407_217_4,
            2.612_954_769_202_015_6,
            3.922_712_312_227_067,
            1.817_078_767_026_889_8,
            2.778_751_474_760_038,
            3.774_179_421_631_728_4,
            1.574_152_595_188_286_2,
            3.313_796_027_497_986,
            3.982_980_119_091_688,
            1.963_145_283_190_319_6,
            3.019_040_075_427_340_8,
            3.635_010_529_230_896_7,
            2.139_289_839_228_415_4,
            2.878_753_250_007_488_6,
            3.877_829_442_211_61,
            2.195_492_943_242_401_7,
            3.147_517_117_687_641,
            3.799_171_079_529_07,
            2.250_022_309_368_775_5,
            2.748_635_946_016_38,
            3.657_938_699_979_082_4,
            2.036_446_584_820_199,
            2.480_283_293_481_005_7,
            4.043_549_388_062_252,
            3.135_722_451_985_087,
            3.545_251_180_888_751_5,
            3.969_342_547_129_601_3,
            1.822_993_803_027_427_3,
        ];
        let expected = vec![2.43, 3.5];
        let res = roundbreaks(&numbers, 3).unwrap();
        assert_eq!(res, expected);
    }
}
