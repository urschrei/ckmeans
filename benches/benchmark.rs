#[macro_use]
extern crate criterion;
extern crate ckmeans;
extern crate rand;

use rand::Rng; // 0.6.5
use rand_distr::Normal;
use rand_distr::Uniform;
use std::hint::black_box;

use ckmeans::ckmeans;
use criterion::Criterion;

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

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
