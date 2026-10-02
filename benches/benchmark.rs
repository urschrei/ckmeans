#[macro_use]
extern crate criterion;
extern crate ckmeans;
extern crate rand;

use rand::RngExt;
use rand_distr::Normal;
use rand_distr::Uniform;
use std::hint::black_box;

use ckmeans::{CkmeansConfig, ckmeans, ckmeans_optimal};
use criterion::{BenchmarkId, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("110k i32, Uniform: 0 - 250", |bencher| {
        let mut rng = rand::rng();
        let range = Uniform::new(0, 250).unwrap();

        let data: Vec<i32> = (0..110_000).map(|_| rng.sample(range)).collect();
        bencher.iter(|| {
            ckmeans(black_box(&data), black_box(7)).unwrap();
        });
    });

    c.bench_function("110k f64, Gaussian: mu = 3, sigma = 1", |bencher| {
        let mut rng = rand::rng();
        let range = Normal::new(3.0, 1.0).unwrap();

        let data: Vec<f64> = (0..110_000).map(|_| rng.sample(range)).collect();
        bencher.iter(|| {
            ckmeans(black_box(&data), black_box(7)).unwrap();
        });
    });
}

/// Benchmarks varying n (data size) with fixed k=7
fn bench_varying_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("varying_n_k7");

    for n in [10_000, 50_000, 110_000, 500_000, 1_000_000] {
        let mut rng = rand::rng();
        let range = Uniform::new(0.0, 1000.0).unwrap();
        let data: Vec<f64> = (0..n).map(|_| rng.sample(range)).collect();

        group.bench_with_input(BenchmarkId::new("n", n), &data, |b, data| {
            b.iter(|| ckmeans(black_box(data), black_box(7)).unwrap());
        });
    }
    group.finish();
}

/// Benchmarks varying k (cluster count) with fixed n=110k
fn bench_varying_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("varying_k_n110k");

    let mut rng = rand::rng();
    let range = Uniform::new(0.0, 1000.0).unwrap();
    let data: Vec<f64> = (0..110_000).map(|_| rng.sample(range)).collect();

    for k in [3u8, 7, 15, 30, 50] {
        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| ckmeans(black_box(&data), black_box(k)).unwrap());
        });
    }
    group.finish();
}

/// Benchmarks with both high n and high k
fn bench_high_n_and_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("high_n_and_k");

    let mut rng = rand::rng();
    let range = Uniform::new(0.0, 1000.0).unwrap();

    // n=500k, k=20
    let data_500k: Vec<f64> = (0..500_000).map(|_| rng.sample(range)).collect();
    group.bench_function("n500k_k20", |b| {
        b.iter(|| ckmeans(black_box(&data_500k), black_box(20)).unwrap());
    });

    // n=1M, k=10
    let data_1m: Vec<f64> = (0..1_000_000).map(|_| rng.sample(range)).collect();
    group.bench_function("n1M_k10", |b| {
        b.iter(|| ckmeans(black_box(&data_1m), black_box(10)).unwrap());
    });

    group.finish();
}

/// Benchmarks with pathological data distributions
fn bench_pathological(c: &mut Criterion) {
    let mut group = c.benchmark_group("pathological");

    let mut rng = rand::rng();

    // Bimodal: two tight clusters with a gap (should have tight monotonicity bounds)
    let cluster1 = Normal::new(0.0, 1.0).unwrap();
    let cluster2 = Normal::new(100.0, 1.0).unwrap();
    let mut bimodal: Vec<f64> = (0..55_000).map(|_| rng.sample(cluster1)).collect();
    bimodal.extend((0..55_000).map(|_| rng.sample(cluster2)));
    group.bench_function("bimodal_110k_k7", |b| {
        b.iter(|| ckmeans(black_box(&bimodal), black_box(7)).unwrap());
    });

    // Nearly uniform: worst case for the monotonicity bounds (all values roughly equidistant)
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    let nearly_uniform: Vec<f64> = (0..110_000).map(|_| rng.sample(uniform)).collect();
    group.bench_function("uniform_110k_k50", |b| {
        b.iter(|| ckmeans(black_box(&nearly_uniform), black_box(50)).unwrap());
    });

    group.finish();
}

/// Benchmarks `ckmeans_optimal` with the default range of k (1 to 9)
fn bench_optimal(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimal_k1_9");
    let mut rng = rand::rng();
    let config = CkmeansConfig::default();

    // A mixture of four separated normal distributions, as in bench_cpp
    let components: Vec<Normal<f64>> = (0..4)
        .map(|i| Normal::new(f64::from(i) * 10.0, 1.0).unwrap())
        .collect();
    for n in [110_000, 1_000_000] {
        let data: Vec<f64> = (0..n)
            .map(|i| rng.sample(components[i % components.len()]))
            .collect();
        group.bench_with_input(BenchmarkId::new("four_normals", n), &data, |b, data| {
            b.iter(|| ckmeans_optimal(black_box(data), black_box(config)).unwrap());
        });
    }

    let uniform = Uniform::new(0.0, 1000.0).unwrap();
    let data: Vec<f64> = (0..110_000).map(|_| rng.sample(uniform)).collect();
    group.bench_with_input(BenchmarkId::new("uniform", 110_000), &data, |b, data| {
        b.iter(|| ckmeans_optimal(black_box(data), black_box(config)).unwrap());
    });
    group.finish();
}

criterion_group!(
    benches,
    criterion_benchmark,
    bench_varying_n,
    bench_varying_k,
    bench_high_n_and_k,
    bench_pathological,
    bench_optimal,
);
criterion_main!(benches);
