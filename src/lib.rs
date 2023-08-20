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
mod wasm;
pub use crate::wasm::{ckmeans_wasm, roundbreaks_wasm};

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
    let mut backtrack_matrix = matrix.clone();

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
        // this is the example data from Observable
        let numbers = vec![
            2.186880969969168,
            2.693032112550337,
            3.93180201956282,
            1.8858499460215248,
            3.2891992800955743,
            3.9589934791200636,
            1.7694796044907883,
            3.1708232466019686,
            3.7138103814880594,
            1.8084003369996897,
            2.947289457238938,
            4.079716765942393,
            2.0194812135793665,
            3.367740732705309,
            3.946018838624863,
            1.705437159404827,
            2.8825959512135055,
            3.9858496811993005,
            1.1818317192295225,
            3.1959882611908723,
            3.7975952600818155,
            2.3260003550458643,
            3.07900975917915,
            3.9720330300265134,
            2.0924928025791947,
            3.4149858245262528,
            3.999784164947667,
            2.0380375859511815,
            2.561038262661651,
            4.235818645004002,
            1.9407084883926995,
            3.0986532853492266,
            4.187154244559768,
            1.9374536938000988,
            3.30128973533264,
            4.073525037840789,
            1.6219950585216787,
            2.9516652622521122,
            3.8379509834033345,
            1.8301227974955063,
            3.0587066916259453,
            4.038198749869919,
            1.8463122036805393,
            2.874469711585369,
            3.739250728698104,
            1.8839182936098067,
            3.3398569038428207,
            4.041222857529328,
            2.1580781476395363,
            2.7895732670501525,
            3.670343274157198,
            2.07520762025803,
            3.0269076086290547,
            4.068500990714395,
            2.1507662912105188,
            2.7869451246159356,
            4.060256411136024,
            2.160538596948854,
            3.5911668420419023,
            3.936942667005152,
            1.7823749743811554,
            2.398556760000693,
            3.8580726777270233,
            2.276857198356403,
            2.6674703770216155,
            3.9893132644126292,
            2.126263439397846,
            2.887148841450299,
            4.0064504153769445,
            2.384214172538217,
            3.1584945263692616,
            3.917391700186437,
            2.045842731040772,
            2.8966069573005266,
            4.061877233021508,
            2.5020730840642127,
            2.467027330439492,
            4.0591228764126965,
            1.9267648419184489,
            3.184736160134685,
            3.7987094137777633,
            2.407891897908353,
            3.1130351706505945,
            4.305405220398142,
            3.020832358740857,
            3.0896459925373163,
            4.337513332122615,
            2.1513897982836148,
            3.3890652149152194,
            4.147203390905856,
            1.7442267399486338,
            3.1204974225957933,
            4.2027932814531255,
            1.8205218144747235,
            2.51457165291436,
            4.196065588365519,
            2.3246013572839828,
            3.650545226495212,
            3.9951077005292133,
            1.4786492630219163,
            3.184728841579185,
            4.0477366974592695,
            2.098834005399845,
            3.303175986466117,
            3.7892288071653972,
            2.6244701492863274,
            2.8894897144167992,
            3.970265133933609,
            2.113416007212771,
            2.8377269735255495,
            4.098857511413,
            2.2209214153388634,
            3.002366818476632,
            4.283496512420427,
            1.5581566168268295,
            2.980756646018853,
            3.910127425359612,
            1.490739212060197,
            2.895513344823693,
            4.247576525913251,
            2.1892069006343498,
            2.998806078461756,
            3.8592093775400564,
            1.5867268541671686,
            3.152649521951604,
            3.8244408451591436,
            3.1103871346526786,
            2.4849703616336956,
            4.156025918520517,
            1.5111721433515135,
            3.750816172762316,
            3.9323386383451204,
            2.07694992767919,
            3.2081543986400645,
            4.045289073742084,
            1.9760445160142919,
            3.031327170083975,
            4.204267226311512,
            1.939214053185361,
            3.323788480108269,
            3.7479817408726013,
            2.7826733587061487,
            3.240250893521295,
            3.778322920441067,
            2.4028971995599546,
            3.0490359072527893,
            3.9413611225889986,
            1.9148189265548152,
            2.944080059198642,
            4.26740684738878,
            2.0585827568749755,
            2.782341912819006,
            3.7972162173331205,
            1.6197265398278322,
            3.2559388197360932,
            4.220325996536666,
            1.9819279442330058,
            3.053309950903032,
            4.0050344570479135,
            3.4548807971848774,
            3.1713063979025087,
            3.7920422745644933,
            2.687431903142606,
            3.0435341288234374,
            3.9362188800977105,
            1.3844307677087702,
            2.9950559957757354,
            3.904845590592921,
            3.061504913041073,
            3.1307583228310243,
            4.317604148100834,
            1.291369155045615,
            3.443142692197454,
            4.167123158127977,
            1.2857831107446278,
            2.7431343316148227,
            3.7053733663027435,
            2.3804168566032153,
            2.887674496702909,
            3.9905418408411313,
            1.6719236556114467,
            2.9850026778984935,
            4.146972397533571,
            2.0797012089809805,
            2.951096908063335,
            3.7937067992429365,
            3.001063205135309,
            3.1349248082219585,
            4.022206004589426,
            1.545495981764916,
            2.8966308040049626,
            4.026750229754802,
            2.4550193767136625,
            3.104846667702584,
            4.170108463306901,
            1.3670530711301323,
            2.8324561745174393,
            4.098799538338867,
            1.814066923533963,
            2.581112819622158,
            3.7799212342284623,
            1.115008897674243,
            3.103260260015172,
            3.93758902072937,
            2.411956649166637,
            3.351700352820514,
            4.02264087937381,
            2.714506909365993,
            2.844309361015004,
            3.78747911946831,
            0.6837250553832857,
            2.971586707439505,
            4.311768255823228,
            1.4357914829153984,
            2.931274470207522,
            3.906562719609756,
            0.7588119839001057,
            3.136088252208502,
            3.885505169010581,
            2.831050009570089,
            3.2362698758142,
            3.982431100799526,
            1.9799364874072174,
            2.6129547692020156,
            3.922712312227067,
            1.8170787670268898,
            2.778751474760038,
            3.7741794216317284,
            1.5741525951882862,
            3.313796027497986,
            3.982980119091688,
            1.9631452831903196,
            3.0190400754273408,
            3.6350105292308967,
            2.1392898392284154,
            2.8787532500074886,
            3.87782944221161,
            2.1954929432424017,
            3.147517117687641,
            3.79917107952907,
            2.2500223093687755,
            2.74863594601638,
            3.6579386999790824,
            2.036446584820199,
            2.4802832934810057,
            4.043549388062252,
            3.135722451985087,
            3.5452511808887515,
            3.9693425471296013,
            1.8229938030274273,
        ];
        let expected = vec![2.43, 3.5];
        let res = roundbreaks(&numbers, 3).unwrap();
        assert_eq!(res, expected)
    }
}
