//! Property-based tests (Hegel) for the public clustering API.

use hegel::TestCase;
use hegel::generators::{self as gs, Generator, PrintableGenerator};

use crate::{
    CkNum, CkmeansConfig, CkmeansErr, ckmeans, ckmeans_indices, ckmeans_optimal, roundbreaks,
};

/// Largest input the structural properties draw.
const MAX_LEN: usize = 60;

/// Sort a copy of `data`. The input must not contain NaN.
fn sorted<T: CkNum>(data: &[T]) -> Vec<T> {
    let mut xs = data.to_vec();
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs
}

/// Number of distinct values in `data`. The input must not contain NaN.
fn distinct_count<T: CkNum>(data: &[T]) -> usize {
    let mut xs = sorted(data);
    xs.dedup();
    xs.len()
}

/// Draw a cluster count in `1..=len` (capped at `u8::MAX`).
fn draw_k(tc: &TestCase, len: usize) -> u8 {
    let max = u8::try_from(len).unwrap_or(u8::MAX);
    tc.draw(gs::integers::<u8>().min_value(1).max_value(max))
}

/// Properties of `ckmeans` and `ckmeans_indices` that hold for every element
/// type. `$elements` draws one element: it mixes the full domain of the type
/// (NaN excluded) with a small pool of values, so that inputs with duplicates
/// and with fewer distinct values than `k` occur often.
macro_rules! structural_properties {
    ($name:ident, $t:ty, $elements:expr) => {
        mod $name {
            use super::*;

            fn data() -> impl PrintableGenerator<Vec<$t>> {
                gs::vecs($elements).min_size(1).max_size(MAX_LEN)
            }

            #[hegel::test(test_cases = 2000)]
            fn clusters_concatenate_to_sorted_input(tc: TestCase) {
                let data = tc.draw(data());
                let k = draw_k(&tc, data.len());
                let clusters = ckmeans(&data, k).unwrap();
                assert_eq!(clusters.concat(), sorted(&data));
            }

            #[hegel::test(test_cases = 2000)]
            fn ranges_are_contiguous_and_non_empty(tc: TestCase) {
                let data = tc.draw(data());
                let k = draw_k(&tc, data.len());
                let (_, ranges) = ckmeans_indices(&data, k).unwrap();
                assert_eq!(ranges.first().unwrap().0, 0);
                assert_eq!(ranges.last().unwrap().1, data.len() - 1);
                for &(start, end) in &ranges {
                    assert!(start <= end, "empty range in {ranges:?}");
                }
                for pair in ranges.windows(2) {
                    assert_eq!(pair[1].0, pair[0].1 + 1, "gap or overlap in {ranges:?}");
                }
            }

            #[hegel::test(test_cases = 2000)]
            fn cluster_count_is_k_capped_at_distinct_values(tc: TestCase) {
                let data = tc.draw(data());
                let k = draw_k(&tc, data.len());
                let clusters = ckmeans(&data, k).unwrap();
                assert_eq!(clusters.len(), distinct_count(&data).min(k as usize));
            }

            #[hegel::test(test_cases = 2000)]
            fn ckmeans_agrees_with_ckmeans_indices(tc: TestCase) {
                let data = tc.draw(data());
                let k = draw_k(&tc, data.len());
                let clusters = ckmeans(&data, k).unwrap();
                let (sorted_data, ranges) = ckmeans_indices(&data, k).unwrap();
                let from_ranges: Vec<Vec<$t>> = ranges
                    .iter()
                    .map(|&(start, end)| sorted_data[start..=end].to_vec())
                    .collect();
                assert_eq!(clusters, from_ranges);
            }

            #[hegel::test(test_cases = 500)]
            fn zero_clusters_is_rejected(tc: TestCase) {
                let data = tc.draw(gs::vecs($elements).max_size(MAX_LEN));
                let Err(CkmeansErr::TooFewClassesError) = ckmeans_indices(&data, 0) else {
                    panic!("k = 0 was not rejected with TooFewClassesError");
                };
            }

            #[hegel::test(test_cases = 500)]
            fn more_clusters_than_values_is_rejected(tc: TestCase) {
                let data = tc.draw(gs::vecs($elements).max_size(MAX_LEN));
                let k = tc.draw(gs::integers::<u8>().min_value(data.len() as u8 + 1));
                let Err(CkmeansErr::TooManyClassesError) = ckmeans_indices(&data, k) else {
                    panic!("k = {k} > len = {} was not rejected", data.len());
                };
            }
        }
    };
}

structural_properties!(
    f64_elements,
    f64,
    hegel::one_of!(
        gs::floats::<f64>().allow_nan(false),
        gs::integers::<u8>().max_value(8).map(|x| x as f64),
    )
);
structural_properties!(
    f32_elements,
    f32,
    hegel::one_of!(
        gs::floats::<f32>().allow_nan(false),
        gs::integers::<u8>().max_value(8).map(|x| x as f32),
    )
);
structural_properties!(
    i32_elements,
    i32,
    hegel::one_of!(
        gs::integers::<i32>(),
        gs::integers::<i32>().min_value(-4).max_value(4),
    )
);
structural_properties!(
    i64_elements,
    i64,
    hegel::one_of!(
        gs::integers::<i64>(),
        gs::integers::<i64>().min_value(-4).max_value(4),
    )
);
structural_properties!(
    u8_elements,
    u8,
    hegel::one_of!(gs::integers::<u8>(), gs::integers::<u8>().max_value(8))
);

/// Every entry point rejects input that contains NaN. `$t` is a float type.
macro_rules! nan_properties {
    ($name:ident, $t:ty) => {
        mod $name {
            use super::*;

            /// Draw data (infinities included) and insert one to three NaNs of either
            /// sign at drawn positions.
            fn draw_data_with_nan(tc: &TestCase) -> Vec<$t> {
                let mut data =
                    tc.draw(gs::vecs(gs::floats::<$t>().allow_nan(false)).max_size(MAX_LEN));
                let count = tc.draw(gs::integers::<usize>().min_value(1).max_value(3));
                for _ in 0..count {
                    let position = tc.draw(gs::integers::<usize>().max_value(data.len()));
                    let nan = tc.draw(gs::sampled_from(vec![<$t>::NAN, -<$t>::NAN]));
                    data.insert(position, nan);
                }
                data
            }

            #[hegel::test(test_cases = 1000)]
            fn ckmeans_indices_rejects_nan(tc: TestCase) {
                let data = draw_data_with_nan(&tc);
                let k = draw_k(&tc, data.len());
                let Err(CkmeansErr::NanError) = ckmeans_indices(&data, k) else {
                    panic!("NaN input was not rejected with NanError");
                };
            }

            #[hegel::test(test_cases = 1000)]
            fn ckmeans_rejects_nan(tc: TestCase) {
                let data = draw_data_with_nan(&tc);
                let k = draw_k(&tc, data.len());
                let Err(CkmeansErr::NanError) = ckmeans(&data, k) else {
                    panic!("NaN input was not rejected with NanError");
                };
            }

            #[hegel::test(test_cases = 1000)]
            fn ckmeans_optimal_rejects_nan(tc: TestCase) {
                let data = draw_data_with_nan(&tc);
                let k_min = draw_k(&tc, data.len());
                let k_max = tc.draw(gs::integers::<u8>().min_value(k_min));
                let Err(CkmeansErr::NanError) =
                    ckmeans_optimal(&data, CkmeansConfig { k_min, k_max })
                else {
                    panic!("NaN input was not rejected with NanError");
                };
            }

            #[hegel::test(test_cases = 1000)]
            fn roundbreaks_rejects_nan(tc: TestCase) {
                let data = draw_data_with_nan(&tc);
                let k = draw_k(&tc, data.len());
                let Err(CkmeansErr::NanError) = roundbreaks(&data, k) else {
                    panic!("NaN input was not rejected with NanError");
                };
            }
        }
    };
}

nan_properties!(f64_nan, f64);
nan_properties!(f32_nan, f32);

/// Optimality of `ckmeans` against independent reference solutions.
mod optimality {
    use super::*;

    /// Values in a range where squared deviations cannot overflow, mixed with a
    /// small pool of values so that ties and duplicates occur often.
    fn element() -> impl PrintableGenerator<f64> {
        hegel::one_of!(
            gs::floats::<f64>().min_value(-1e3).max_value(1e3),
            gs::integers::<u8>().max_value(8).map(f64::from),
        )
    }

    fn draw_data(tc: &TestCase, max_len: usize) -> Vec<f64> {
        let len = tc.draw(gs::integers::<usize>().min_value(1).max_value(max_len));
        tc.draw(gs::vecs(element()).min_size(len).max_size(len))
    }

    /// Two-pass within-cluster sum of squares.
    fn ssq(cluster: &[f64]) -> f64 {
        let mean = cluster.iter().sum::<f64>() / cluster.len() as f64;
        cluster.iter().map(|x| (x - mean) * (x - mean)).sum()
    }

    fn total_ssq(clusters: &[Vec<f64>]) -> f64 {
        clusters.iter().map(|c| ssq(c)).sum()
    }

    /// Absolute tolerance for comparing sums of squares of `data`.
    fn tolerance(data: &[f64]) -> f64 {
        1e-9 * ssq(data) + 1e-12
    }

    /// Smallest total sum of squares over every split of `sorted` into
    /// `groups` contiguous, non-empty groups.
    fn brute_force(sorted: &[f64], groups: usize) -> f64 {
        if groups == 1 {
            return ssq(sorted);
        }
        (1..=sorted.len() - (groups - 1))
            .map(|split| ssq(&sorted[..split]) + brute_force(&sorted[split..], groups - 1))
            .fold(f64::INFINITY, f64::min)
    }

    /// Smallest total sum of squares over every split of `sorted` into
    /// `groups` contiguous, non-empty groups, by an O(groups * n^2) dynamic
    /// program with no bounds on the split search.
    fn reference_dp(sorted: &[f64], groups: usize) -> f64 {
        let n = sorted.len();
        // segment[j][i] is the sum of squares of sorted[j..=i] (Welford).
        let mut segment = vec![vec![0.0; n]; n];
        for (j, row) in segment.iter_mut().enumerate() {
            let (mut mean, mut m2) = (0.0, 0.0);
            for (i, &x) in sorted.iter().enumerate().skip(j) {
                let count = (i - j + 1) as f64;
                let delta = x - mean;
                mean += delta / count;
                m2 += delta * (x - mean);
                row[i] = m2;
            }
        }
        let mut cost: Vec<f64> = (0..n).map(|i| segment[0][i]).collect();
        for c in 1..groups {
            let mut next = vec![f64::INFINITY; n];
            for i in c..n {
                for j in c..=i {
                    next[i] = next[i].min(cost[j - 1] + segment[j][i]);
                }
            }
            cost = next;
        }
        cost[n - 1]
    }

    #[hegel::test(test_cases = 2000)]
    fn ckmeans_is_no_worse_than_brute_force(tc: TestCase) {
        let data = draw_data(&tc, 10);
        let k = draw_k(&tc, data.len());
        let clusters = ckmeans(&data, k).unwrap();
        let best = brute_force(&sorted(&data), clusters.len());
        let found = total_ssq(&clusters);
        assert!(
            found <= best + tolerance(&data),
            "ckmeans total {found} > brute-force optimum {best} for {clusters:?}"
        );
    }

    #[hegel::test(test_cases = 500)]
    fn ckmeans_is_no_worse_than_reference_dp(tc: TestCase) {
        let data = draw_data(&tc, 200);
        let k = draw_k(&tc, data.len().min(20));
        let clusters = ckmeans(&data, k).unwrap();
        let best = reference_dp(&sorted(&data), clusters.len());
        let found = total_ssq(&clusters);
        assert!(
            found <= best + tolerance(&data),
            "ckmeans total {found} > reference optimum {best}"
        );
    }

    #[hegel::test(test_cases = 500)]
    fn total_ssq_does_not_increase_with_k(tc: TestCase) {
        let data = draw_data(&tc, 60);
        let tol = tolerance(&data);
        let mut previous = f64::INFINITY;
        for k in 1..=data.len() as u8 {
            let current = total_ssq(&ckmeans(&data, k).unwrap());
            assert!(
                current <= previous + tol,
                "total sum of squares rose from {previous} to {current} at k = {k}"
            );
            previous = current;
        }
    }
}

/// Relations that hold for float input of any finite magnitude.
mod float_relations {
    use super::*;

    /// Multiples of 0.001 up to 1000 in magnitude: every non-zero value is at
    /// least 2^-10 and at most 2^10 in magnitude, so scaling by 2^m for
    /// |m| <= 1000 stays in the normal range.
    fn element() -> impl PrintableGenerator<f64> {
        hegel::one_of!(
            gs::integers::<i32>()
                .min_value(-1_000_000)
                .max_value(1_000_000)
                .map(|x| f64::from(x) / 1000.0),
            gs::integers::<u8>().max_value(8).map(f64::from),
        )
    }

    #[hegel::test(test_cases = 2000)]
    fn scaling_by_a_power_of_two_does_not_change_ranges(tc: TestCase) {
        let data = tc.draw(gs::vecs(element()).min_size(1).max_size(MAX_LEN));
        let k = draw_k(&tc, data.len());
        let exponent = tc.draw(gs::integers::<i32>().min_value(-1000).max_value(1000));
        let factor = 2f64.powi(exponent);
        let scaled: Vec<f64> = data.iter().map(|x| x * factor).collect();
        let (_, ranges) = ckmeans_indices(&data, k).unwrap();
        let (_, scaled_ranges) = ckmeans_indices(&scaled, k).unwrap();
        assert_eq!(ranges, scaled_ranges);
    }
}

/// Relations between clusterings of related inputs.
mod relations {
    use super::*;

    /// Largest magnitude that every tested element type (f32 included)
    /// represents exactly.
    const EXACT: i32 = 1 << 24;

    fn element() -> impl PrintableGenerator<i32> {
        hegel::one_of!(
            gs::integers::<i32>().min_value(-EXACT).max_value(EXACT),
            gs::integers::<i32>().min_value(-4).max_value(4),
        )
    }

    fn draw_data(tc: &TestCase) -> Vec<i32> {
        tc.draw(gs::vecs(element()).min_size(1).max_size(MAX_LEN))
    }

    #[hegel::test(test_cases = 2000)]
    fn translation_does_not_change_ranges(tc: TestCase) {
        let data = draw_data(&tc);
        let k = draw_k(&tc, data.len());
        let offset = tc.draw(gs::integers::<i32>().min_value(-EXACT).max_value(EXACT));
        let shifted: Vec<i32> = data.iter().map(|x| x + offset).collect();
        let (_, ranges) = ckmeans_indices(&data, k).unwrap();
        let (_, shifted_ranges) = ckmeans_indices(&shifted, k).unwrap();
        assert_eq!(ranges, shifted_ranges);
    }

    #[hegel::test(test_cases = 2000)]
    fn element_types_agree_on_ranges(tc: TestCase) {
        let data = draw_data(&tc);
        let k = draw_k(&tc, data.len());
        let as_i64: Vec<i64> = data.iter().map(|&x| i64::from(x)).collect();
        let as_f64: Vec<f64> = data.iter().map(|&x| f64::from(x)).collect();
        let as_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let (_, expected) = ckmeans_indices(&data, k).unwrap();
        assert_eq!(ckmeans_indices(&as_i64, k).unwrap().1, expected, "i64");
        assert_eq!(ckmeans_indices(&as_f64, k).unwrap().1, expected, "f64");
        assert_eq!(ckmeans_indices(&as_f32, k).unwrap().1, expected, "f32");
    }

    #[hegel::test(test_cases = 2000)]
    fn u8_agrees_with_f64_on_ranges(tc: TestCase) {
        let data = tc.draw(gs::vecs(gs::integers::<u8>()).min_size(1).max_size(MAX_LEN));
        let k = draw_k(&tc, data.len());
        let as_f64: Vec<f64> = data.iter().map(|&x| f64::from(x)).collect();
        assert_eq!(
            ckmeans_indices(&data, k).unwrap().1,
            ckmeans_indices(&as_f64, k).unwrap().1
        );
    }

    #[hegel::test(test_cases = 2000)]
    fn equal_values_share_a_cluster(tc: TestCase) {
        let data = draw_data(&tc);
        let k = draw_k(&tc, data.len());
        let clusters = ckmeans(&data, k).unwrap();
        for pair in clusters.windows(2) {
            assert!(
                pair[0].last() < pair[1].first(),
                "equal values split across clusters: {clusters:?}"
            );
        }
    }
}

/// Properties of `ckmeans_optimal`. `$t` is a float type.
macro_rules! optimal_properties {
    ($name:ident, $t:ty) => {
        mod $name {
            use super::*;

            /// Values in a range where BIC terms stay finite, mixed with a small
            /// pool of values so that ties and duplicates occur often.
            fn element() -> impl PrintableGenerator<$t> {
                hegel::one_of!(
                    gs::floats::<$t>().min_value(-1e3).max_value(1e3),
                    gs::integers::<u8>().max_value(8).map(<$t>::from),
                )
            }

            /// Draw a length that is either short or near 256, where a `u8`
            /// conversion of the length wraps round.
            fn draw_len(tc: &TestCase) -> usize {
                tc.draw(hegel::one_of!(
                    gs::integers::<usize>().min_value(1).max_value(40),
                    gs::integers::<usize>().min_value(250).max_value(270),
                ))
            }

            fn draw_data(tc: &TestCase) -> Vec<$t> {
                let len = draw_len(tc);
                tc.draw(gs::vecs(element()).min_size(len).max_size(len))
            }

            /// Draw a valid configuration: `1 <= k_min <= distinct values`, and
            /// `k_max` up to 12 above `k_min`.
            fn draw_config(tc: &TestCase, data: &[$t]) -> CkmeansConfig {
                let k_min = draw_k(tc, distinct_count(data));
                let span = tc.draw(gs::integers::<u8>().max_value(12));
                CkmeansConfig {
                    k_min,
                    k_max: k_min.saturating_add(span),
                }
            }

            #[hegel::test(test_cases = 500)]
            fn clusters_match_ckmeans_at_chosen_k(tc: TestCase) {
                let data = draw_data(&tc);
                let config = draw_config(&tc, &data);
                let result = ckmeans_optimal(&data, config).unwrap();
                assert_eq!(result.clusters, ckmeans(&data, result.k).unwrap());
            }

            #[hegel::test(test_cases = 500)]
            fn bic_covers_k_min_to_capped_k_max(tc: TestCase) {
                let data = draw_data(&tc);
                let config = draw_config(&tc, &data);
                let result = ckmeans_optimal(&data, config).unwrap();
                let cap = u8::try_from(distinct_count(&data)).unwrap_or(u8::MAX);
                let expected: Vec<u8> = (config.k_min..=config.k_max.min(cap)).collect();
                let evaluated: Vec<u8> = result.bic.iter().map(|&(k, _)| k).collect();
                assert_eq!(evaluated, expected);
            }

            #[hegel::test(test_cases = 500)]
            fn default_config_evaluates_up_to_nine(tc: TestCase) {
                let data = draw_data(&tc);
                let result = ckmeans_optimal(&data, CkmeansConfig::default()).unwrap();
                let cap = u8::try_from(distinct_count(&data)).unwrap_or(u8::MAX);
                assert_eq!(result.bic.len(), usize::from(cap.min(9)));
            }

            #[hegel::test(test_cases = 500)]
            fn cluster_count_equals_chosen_k(tc: TestCase) {
                let data = draw_data(&tc);
                let config = draw_config(&tc, &data);
                let result = ckmeans_optimal(&data, config).unwrap();
                assert_eq!(result.clusters.len(), usize::from(result.k));
            }

            #[hegel::test(test_cases = 1000)]
            fn cluster_count_equals_chosen_k_for_any_values(tc: TestCase) {
                let data = tc.draw(
                    gs::vecs(gs::floats::<$t>().allow_nan(false))
                        .min_size(1)
                        .max_size(MAX_LEN),
                );
                let config = draw_config(&tc, &data);
                let result = ckmeans_optimal(&data, config).unwrap();
                assert_eq!(result.clusters.len(), usize::from(result.k));
                assert_eq!(result.clusters.concat(), sorted(&data));
            }

            #[hegel::test(test_cases = 500)]
            fn chosen_k_has_first_minimum_bic(tc: TestCase) {
                let data = draw_data(&tc);
                let config = draw_config(&tc, &data);
                let result = ckmeans_optimal(&data, config).unwrap();
                let (best_k, _) = result
                    .bic
                    .iter()
                    .copied()
                    .reduce(|best, next| if next.1 < best.1 { next } else { best })
                    .unwrap();
                assert_eq!(result.k, best_k, "BIC values: {:?}", result.bic);
            }

            #[hegel::test(test_cases = 500)]
            fn bic_values_are_finite(tc: TestCase) {
                let data = draw_data(&tc);
                let config = draw_config(&tc, &data);
                let result = ckmeans_optimal(&data, config).unwrap();
                for &(k, bic) in &result.bic {
                    assert!(bic.is_finite(), "BIC for k = {k} is {bic}");
                }
            }

            #[hegel::test(test_cases = 500)]
            fn stats_describe_clusters(tc: TestCase) {
                let data = draw_data(&tc);
                let config = draw_config(&tc, &data);
                let result = ckmeans_optimal(&data, config).unwrap();
                assert_eq!(result.stats.len(), result.clusters.len());
                for (stat, cluster) in result.stats.iter().zip(&result.clusters) {
                    assert_eq!(stat.size, cluster.len());
                    let (lo, hi) = (cluster[0], cluster[cluster.len() - 1]);
                    assert!(
                        lo <= stat.center && stat.center <= hi,
                        "{stat:?} {cluster:?}"
                    );
                    assert!(stat.withinss >= 0.0, "{stat:?}");
                }
            }

            #[hegel::test(test_cases = 500)]
            fn k_min_zero_is_rejected(tc: TestCase) {
                let data = tc.draw(gs::vecs(element()).max_size(MAX_LEN));
                let k_max = tc.draw(gs::integers::<u8>());
                let config = CkmeansConfig { k_min: 0, k_max };
                let Err(CkmeansErr::TooFewClassesError) = ckmeans_optimal(&data, config) else {
                    panic!("k_min = 0 was not rejected with TooFewClassesError");
                };
            }

            #[hegel::test(test_cases = 500)]
            fn inverted_range_is_rejected(tc: TestCase) {
                let data = tc.draw(gs::vecs(element()).max_size(MAX_LEN));
                let k_min = tc.draw(gs::integers::<u8>().min_value(2));
                let k_max = tc.draw(gs::integers::<u8>().min_value(1).max_value(k_min - 1));
                let config = CkmeansConfig { k_min, k_max };
                let Err(CkmeansErr::InvalidRangeError) = ckmeans_optimal(&data, config) else {
                    panic!("k_min > k_max was not rejected with InvalidRangeError");
                };
            }

            #[hegel::test(test_cases = 500)]
            fn k_min_above_distinct_values_is_rejected(tc: TestCase) {
                let data = tc.draw(gs::vecs(element()).max_size(MAX_LEN));
                let distinct = distinct_count(&data) as u8;
                let k_min = tc.draw(gs::integers::<u8>().min_value(distinct + 1));
                let k_max = tc.draw(gs::integers::<u8>().min_value(k_min));
                let config = CkmeansConfig { k_min, k_max };
                let Err(CkmeansErr::TooManyClassesError) = ckmeans_optimal(&data, config) else {
                    panic!("k_min = {k_min} > {distinct} distinct values was not rejected");
                };
            }
        }
    };
}

optimal_properties!(f64_optimal, f64);
optimal_properties!(f32_optimal, f32);
