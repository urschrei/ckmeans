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

/// Compare the BIC values and the selected k of `ckmeans_optimal` with those of the C++ code.
fn assert_same_bic(data: &[f64], k_max: u8) {
    let config = ckmeans::CkmeansConfig { k_min: 1, k_max };
    let rust = ckmeans::ckmeans_optimal(data, config).unwrap();
    let cpp = bench_cpp::ckmeans_cpp_optimal(data, 1, usize::from(k_max), Method::Linear).unwrap();
    assert_eq!(usize::from(rust.k), cpp.k, "selected k");
    assert_eq!(rust.bic.len(), cpp.bic.len());
    for (&(rust_k, rust_bic), &(cpp_k, cpp_bic)) in rust.bic.iter().zip(&cpp.bic) {
        assert_eq!(usize::from(rust_k), cpp_k);
        // The C++ code reports the negative value
        let tolerance = 1e-9 * cpp_bic.abs().max(1.0);
        assert!(
            (rust_bic + cpp_bic).abs() <= tolerance,
            "k = {rust_k}: Rust BIC {rust_bic}, C++ BIC {cpp_bic}"
        );
    }
}

#[test]
fn same_bic_mixtures() {
    let mut rng = StdRng::seed_from_u64(3);
    for components in 1..=6 {
        for gap in [3.0, 10.0] {
            let data: Vec<f64> = (0..600)
                .map(|i| {
                    let mean = (i % components) as f64 * gap;
                    rng.sample(Normal::new(mean, 1.0).unwrap())
                })
                .collect();
            assert_same_bic(&data, 9);
        }
    }
}

#[test]
fn same_bic_ties_and_small_samples() {
    let mut rng = StdRng::seed_from_u64(4);
    // Many equal values give clusters with zero variance
    let integers: Vec<f64> = (0..1000)
        .map(|_| f64::from(rng.random_range(0..10u8)))
        .collect();
    assert_same_bic(&integers, 9);
    // Small samples give clusters of one value
    for _ in 0..20 {
        let small: Vec<f64> = (0..20)
            .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
            .collect();
        assert_same_bic(&small, 9);
    }
}
