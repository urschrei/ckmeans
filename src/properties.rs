//! Property-based tests (Hegel) for the public clustering API.

use hegel::TestCase;
use hegel::generators::{self as gs, Generator, PrintableGenerator};

use crate::{CkNum, CkmeansErr, ckmeans, ckmeans_indices};

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
