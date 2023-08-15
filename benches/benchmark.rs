#[macro_use]
extern crate criterion;
extern crate ckmeans;
extern crate rand;

use rand::{distributions::Uniform, Rng}; // 0.6.5

use ckmeans::ckmeans;
use criterion::{black_box, Criterion};

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("110k", |bencher| {
        let mut rng = rand::thread_rng();
        let range = Uniform::new(0, 250);

        let data: Vec<i32> = (0..110000).map(|_| rng.sample(&range)).collect();
        bencher.iter(|| {
            ckmeans(black_box(&data), black_box(7)).unwrap();
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
