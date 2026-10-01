//! Check that the Rust and C++ implementations give the same clusters for the benchmark inputs.

use bench_cpp::{CppError, Method, ckmeans_cpp_cluster};
use ckmeans::ckmeans_indices;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Normal, Uniform};

/// Compare the cluster sizes and the cluster of each value in sorted order.
fn assert_same_clusters(data: &[f64], k: u8, method: Method) {
    let (_, ranges) = ckmeans_indices(data, k).unwrap();
    let cpp = ckmeans_cpp_cluster(data, k as usize, method).unwrap();

    let rust_sizes: Vec<f64> = ranges.iter().map(|&(a, b)| (b - a + 1) as f64).collect();
    assert_eq!(
        rust_sizes,
        cpp.size[..ranges.len()],
        "method {}",
        method.name()
    );

    // The C++ labels are in input order. Sort them by value to compare with the Rust ranges.
    let mut order: Vec<usize> = (0..data.len()).collect();
    order.sort_by(|&i, &j| data[i].total_cmp(&data[j]));
    for (label, &(start, end)) in ranges.iter().enumerate() {
        for &i in &order[start..=end] {
            assert_eq!(cpp.cluster[i] as usize, label, "method {}", method.name());
        }
    }
}

#[test]
fn same_clusters_uniform() {
    let mut rng = StdRng::seed_from_u64(1);
    let range = Uniform::new(1.0, 3.0).unwrap();
    for (n, k) in [(1_000, 5), (10_000, 20), (110_000, 5), (110_000, 50)] {
        let data: Vec<f64> = (0..n).map(|_| rng.sample(range)).collect();
        for method in [Method::Linear, Method::LogLinear] {
            assert_same_clusters(&data, k, method);
        }
    }
}

#[test]
fn same_clusters_bimodal() {
    let mut rng = StdRng::seed_from_u64(2);
    let low = Normal::new(0.0, 1.0).unwrap();
    let high = Normal::new(100.0, 1.0).unwrap();
    let mut data: Vec<f64> = (0..5_000).map(|_| rng.sample(low)).collect();
    data.extend((0..5_000).map(|_| rng.sample(high)));
    for method in [Method::Linear, Method::LogLinear] {
        assert_same_clusters(&data, 7, method);
    }
}

#[test]
fn rejects_invalid_input() {
    assert_eq!(
        ckmeans_cpp_cluster(&[1.0, 2.0], 0, Method::Linear),
        Err(CppError::InvalidK)
    );
    assert_eq!(
        ckmeans_cpp_cluster(&[1.0, 2.0], 3, Method::Linear),
        Err(CppError::InvalidK)
    );
    assert_eq!(
        ckmeans_cpp_cluster(&[1.0, f64::NAN], 1, Method::Linear),
        Err(CppError::NonFinite)
    );
}
