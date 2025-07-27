#[macro_use]
extern crate criterion;

use ckmeans::ckmeans;
use criterion::{BenchmarkId, Criterion};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::hint::black_box;

/// Benchmark suite for validating PGO performance improvements
/// Tests specifically the k=3-25 range that PGO was trained on
fn pgo_validation_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("pgo_validation");

    // Test data sizes
    let test_sizes = vec![1_000, 10_000, 50_000, 100_000];

    // Test specific k values that are common in practice
    let k_values = vec![3, 5, 7, 10, 15, 20, 25];

    let mut rng = rand::rng();

    // Benchmark 1: Uniform distribution (common for binning)
    for &size in &test_sizes {
        let uniform = Uniform::new(0, 1000).unwrap();
        let data: Vec<i32> = (0..size).map(|_| uniform.sample(&mut rng)).collect();

        for &k in &k_values {
            if k < size {
                group.bench_with_input(
                    BenchmarkId::new(format!("uniform_i32_k{}", k), size),
                    &(&data, k),
                    |b, &(data, k)| {
                        b.iter(|| {
                            black_box(ckmeans(data, k as u8).unwrap());
                        });
                    },
                );
            }
        }
    }

    // Benchmark 2: Normal distribution (common in real-world data)
    for &size in &test_sizes {
        let normal = Normal::new(50.0, 15.0).unwrap();
        let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();

        for &k in &k_values {
            if k < size {
                group.bench_with_input(
                    BenchmarkId::new(format!("normal_f64_k{}", k), size),
                    &(&data, k),
                    |b, &(data, k)| {
                        b.iter(|| {
                            black_box(ckmeans(data, k as u8).unwrap());
                        });
                    },
                );
            }
        }
    }

    // Benchmark 3: Multimodal distribution (tests cluster separation)
    for &size in &[10_000, 50_000] {
        let mut data: Vec<f64> = Vec::with_capacity(size);
        let modes = vec![
            Normal::new(10.0, 2.0).unwrap(),
            Normal::new(50.0, 3.0).unwrap(),
            Normal::new(90.0, 2.5).unwrap(),
            Normal::new(150.0, 4.0).unwrap(),
            Normal::new(200.0, 3.0).unwrap(),
        ];

        for i in 0..size {
            let mode = &modes[i % modes.len()];
            data.push(mode.sample(&mut rng));
        }

        for &k in &[5, 10, 15, 20] {
            group.bench_with_input(
                BenchmarkId::new(format!("multimodal_k{}", k), size),
                &(&data, k),
                |b, &(data, k)| {
                    b.iter(|| {
                        black_box(ckmeans(data, k as u8).unwrap());
                    });
                },
            );
        }
    }

    // Benchmark 4: Edge case - small k, large data
    let large_data: Vec<i32> = (0..100_000).map(|i| i % 1000).collect();
    for &k in &[3, 4, 5] {
        group.bench_with_input(
            BenchmarkId::new("edge_small_k", k),
            &(&large_data, k),
            |b, &(data, k)| {
                b.iter(|| {
                    black_box(ckmeans(data, k as u8).unwrap());
                });
            },
        );
    }

    // Benchmark 5: Edge case - large k, moderate data
    let moderate_data: Vec<f64> = (0..10_000).map(|_| rng.random::<f64>() * 100.0).collect();
    for &k in &[20, 22, 25] {
        group.bench_with_input(
            BenchmarkId::new("edge_large_k", k),
            &(&moderate_data, k),
            |b, &(data, k)| {
                b.iter(|| {
                    black_box(ckmeans(data, k as u8).unwrap());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, pgo_validation_benches);
criterion_main!(benches);
