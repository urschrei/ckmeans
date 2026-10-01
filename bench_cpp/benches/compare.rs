//! Compare ckmeans with the Ckmeans.1d.dp C++ implementation on identical seeded inputs.

use std::hint::black_box;

use bench_cpp::{Method, ckmeans_cpp_cluster, ckmeans_cpp_optimal};
use ckmeans::{CkmeansConfig, ckmeans, ckmeans_indices, ckmeans_optimal};
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput, measurement::WallTime};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};

const SEED: u64 = 42;

fn sample<D: Distribution<f64>>(rng: &mut StdRng, dist: D, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.sample(&dist)).collect()
}

/// Run each implementation on the same data, with `parameter` as the benchmark ID parameter.
fn bench_implementations(
    group: &mut BenchmarkGroup<'_, WallTime>,
    parameter: &str,
    data: &[f64],
    k: u8,
) {
    group.throughput(Throughput::Elements(data.len() as u64));
    group.bench_with_input(BenchmarkId::new("rust_indices", parameter), data, |b, d| {
        b.iter(|| ckmeans_indices(black_box(d), black_box(k)).unwrap());
    });
    group.bench_with_input(
        BenchmarkId::new("rust_clusters", parameter),
        data,
        |b, d| {
            b.iter(|| ckmeans(black_box(d), black_box(k)).unwrap());
        },
    );
    // BIC selection over k..=k, as in the C++ calls
    let config = CkmeansConfig { k_min: k, k_max: k };
    group.bench_with_input(BenchmarkId::new("rust_optimal", parameter), data, |b, d| {
        b.iter(|| ckmeans_optimal(black_box(d), black_box(config)).unwrap());
    });
    for method in [Method::Linear, Method::LogLinear] {
        let id = BenchmarkId::new(format!("cpp_{}", method.name()), parameter);
        group.bench_with_input(id, data, |b, d| {
            b.iter(|| ckmeans_cpp_cluster(black_box(d), black_box(k as usize), method).unwrap());
        });
    }
}

/// The cases in the ckmeans_py benchmark: Uniform(1, 3), n = 110k or 1M, k = 5 or 20
fn bench_ckmeans_py_cases(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let range = Uniform::new(1.0, 3.0).unwrap();
    for n in [110_000, 1_000_000] {
        let mut group = c.benchmark_group(format!("ckmeans_py_uniform_n{n}"));
        if n >= 1_000_000 {
            group.sample_size(20);
        }
        let data = sample(&mut rng, range, n);
        for k in [5, 20] {
            bench_implementations(&mut group, &format!("k{k}"), &data, k);
        }
        group.finish();
    }
}

/// Uniform(0, 1000), n = 110k, increasing k
fn bench_varying_k(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let range = Uniform::new(0.0, 1000.0).unwrap();
    let data = sample(&mut rng, range, 110_000);
    let mut group = c.benchmark_group("varying_k_n110k");
    for k in [3, 7, 15, 30, 50] {
        bench_implementations(&mut group, &format!("k{k}"), &data, k);
    }
    group.finish();
}

/// Two separated normal distributions, n = 110k, k = 7
fn bench_bimodal(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut data = sample(&mut rng, Normal::new(0.0, 1.0).unwrap(), 55_000);
    data.extend(sample(&mut rng, Normal::new(100.0, 1.0).unwrap(), 55_000));
    let mut group = c.benchmark_group("bimodal_n110k");
    bench_implementations(&mut group, "k7", &data, 7);
    group.finish();
}

/// BIC selection of k in 1..=9, the default range in the R, Python and Rust packages.
/// The input is a mixture of four separated normal distributions.
fn bench_bic_selection(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let components: Vec<Normal<f64>> = (0..4)
        .map(|i| Normal::new(f64::from(i) * 10.0, 1.0).unwrap())
        .collect();
    let config = CkmeansConfig::default();
    for n in [110_000, 1_000_000] {
        let mut group = c.benchmark_group(format!("bic_k1_9_n{n}"));
        if n >= 1_000_000 {
            group.sample_size(10);
        }
        group.throughput(Throughput::Elements(n as u64));
        let data: Vec<f64> = (0..n)
            .map(|i| rng.sample(components[i % components.len()]))
            .collect();
        group.bench_with_input("rust_optimal", &data, |b, d| {
            b.iter(|| ckmeans_optimal(black_box(d), black_box(config)).unwrap());
        });
        for method in [Method::Linear, Method::LogLinear] {
            let id = format!("cpp_{}", method.name());
            group.bench_with_input(id, &data, |b, d| {
                b.iter(|| ckmeans_cpp_optimal(black_box(d), 1, 9, method).unwrap());
            });
        }
        group.finish();
    }
}

criterion::criterion_group!(
    benches,
    bench_ckmeans_py_cases,
    bench_varying_k,
    bench_bimodal,
    bench_bic_selection
);
criterion::criterion_main!(benches);
