//! Compare the number of clusters that `ckmeans_optimal` and the C++ implementation select with
//! the BIC, for k in 1..=9.
//!
//! Run with `cargo run --release --example bic_selection`.

use bench_cpp::{Method, ckmeans_cpp_optimal};
use ckmeans::{CkmeansConfig, ckmeans_optimal};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Exp, LogNormal, Normal, Uniform};

const K_MIN: u8 = 1;
const K_MAX: u8 = 9;
const SEEDS: u64 = 5;

/// Values from `components` normal distributions with means 0, `gap`, 2 * `gap`, ...
fn mixture(rng: &mut StdRng, components: usize, gap: f64, sd: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let mean = (i % components) as f64 * gap;
            rng.sample(Normal::new(mean, sd).unwrap())
        })
        .collect()
}

fn datasets(seed: u64) -> Vec<(String, Vec<f64>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sets = vec![
        (
            "uniform(0, 1), n=1000".to_string(),
            (0..1000)
                .map(|_| rng.sample(Uniform::new(0.0, 1.0).unwrap()))
                .collect(),
        ),
        (
            "normal(0, 1), n=1000".to_string(),
            (0..1000)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect(),
        ),
        (
            "exp(1), n=1000".to_string(),
            (0..1000)
                .map(|_| rng.sample(Exp::new(1.0).unwrap()))
                .collect(),
        ),
        (
            "lognormal(0, 1), n=1000".to_string(),
            (0..1000)
                .map(|_| rng.sample(LogNormal::new(0.0, 1.0).unwrap()))
                .collect(),
        ),
        (
            "integers 0..10, n=1000".to_string(),
            (0..1000)
                .map(|_| f64::from(rng.random_range(0..10u8)))
                .collect(),
        ),
        (
            "normal(0, 1), n=20".to_string(),
            (0..20)
                .map(|_| rng.sample(Normal::new(0.0, 1.0).unwrap()))
                .collect(),
        ),
    ];
    for components in 2..=6 {
        sets.push((
            format!("{components} separated normals, n=600"),
            mixture(&mut rng, components, 10.0, 1.0, 600),
        ));
        sets.push((
            format!("{components} overlapping normals, n=600"),
            mixture(&mut rng, components, 3.0, 1.0, 600),
        ));
    }
    sets
}

fn main() {
    let config = CkmeansConfig {
        k_min: K_MIN,
        k_max: K_MAX,
    };
    let mut total = 0;
    let mut differ = 0;
    println!("| Input | Seed | Rust k | C++ k |");
    println!("|-------|------|--------|-------|");
    for seed in 0..SEEDS {
        for (name, data) in datasets(seed) {
            let rust = ckmeans_optimal(&data, config).unwrap();
            let kmax = usize::from(K_MAX).min(data.len());
            let cpp = ckmeans_cpp_optimal(&data, usize::from(K_MIN), kmax, Method::Linear).unwrap();
            total += 1;
            let marker = if usize::from(rust.k) == cpp.k {
                ""
            } else {
                differ += 1;
                " *"
            };
            println!("| {name} | {seed} | {} | {}{marker} |", rust.k, cpp.k);
        }
    }
    println!("\n{differ} of {total} inputs select a different k (marked *).");
}
