#[cfg(test)]
use ckmeans::ckmeans;
#[cfg(test)]
use rand::Rng;
#[cfg(test)]
use rand_distr::{Distribution, Normal, Uniform};
#[cfg(test)]
use std::hint::black_box;

/// Generate representative workloads for PGO training
/// Covers k values from 3 to 25 with various data distributions
#[cfg(test)]
pub fn run_pgo_workloads() {
    println!("Running PGO training workloads...");

    // Test different data sizes
    let data_sizes = vec![100, 1_000, 10_000, 50_000, 100_000];

    // Test k values from 3 to 25
    let k_values: Vec<usize> = (3..=25).collect();

    let mut rng = rand::rng();

    // Uniform distribution tests
    println!("Testing uniform distributions...");
    for &size in &data_sizes {
        let uniform = Uniform::new(0, 1000).unwrap();
        let data: Vec<i32> = (0..size).map(|_| uniform.sample(&mut rng)).collect();

        for &k in &k_values {
            if k < size {
                let _ = black_box(ckmeans(&data, k as u8));
            }
        }
    }

    // Normal distribution tests with different parameters
    println!("Testing normal distributions...");
    let normal_params = vec![
        (0.0, 1.0),    // Standard normal
        (100.0, 20.0), // Centered at 100, wider spread
        (50.0, 5.0),   // Centered at 50, narrow spread
        (-25.0, 10.0), // Negative center
    ];

    for &size in &data_sizes {
        for &(mean, std_dev) in &normal_params {
            let normal = Normal::new(mean, std_dev).unwrap();
            let data: Vec<f64> = (0..size).map(|_| normal.sample(&mut rng)).collect();

            for &k in &k_values {
                if k < size {
                    let _ = black_box(ckmeans(&data, k as u8));
                }
            }
        }
    }

    // Multimodal distribution tests (mixture of normals)
    println!("Testing multimodal distributions...");
    for &size in &data_sizes {
        let mut data: Vec<f64> = Vec::with_capacity(size);
        let modes = vec![
            Normal::new(10.0, 2.0).unwrap(),
            Normal::new(50.0, 3.0).unwrap(),
            Normal::new(90.0, 2.5).unwrap(),
        ];

        for i in 0..size {
            let mode = &modes[i % modes.len()];
            data.push(mode.sample(&mut rng));
        }

        for &k in &k_values {
            if k < size {
                let _ = black_box(ckmeans(&data, k as u8));
            }
        }
    }

    // Skewed distribution tests
    println!("Testing skewed distributions...");
    for &size in &data_sizes {
        // Create exponentially distributed data
        let mut data: Vec<f64> = Vec::with_capacity(size);
        for _ in 0..size {
            // Simple exponential distribution approximation
            data.push(-rng.random::<f64>().ln() * 10.0);
        }

        for &k in &k_values {
            if k < size {
                let _ = black_box(ckmeans(&data, k as u8));
            }
        }
    }

    // Edge cases: very small datasets
    println!("Testing edge cases...");
    for k in 3..=10 {
        let small_data: Vec<i32> = (0..k + 2).map(|i| i as i32).collect();
        let _ = black_box(ckmeans(&small_data, k as u8));
    }

    // Real-world-like data: timestamps (sorted data)
    println!("Testing sorted data (timestamps)...");
    for &size in &data_sizes {
        let mut data: Vec<i64> = Vec::with_capacity(size);
        let mut timestamp = 1_000_000_000i64;
        for _ in 0..size {
            timestamp += rng.random_range(1..1000);
            data.push(timestamp);
        }

        for &k in &[3, 5, 10, 15, 20, 25] {
            if k < size {
                let _ = black_box(ckmeans(&data, k as u8));
            }
        }
    }

    println!("PGO training workloads complete!");
}

#[cfg(test)]
fn main() {
    run_pgo_workloads();
}

#[cfg(not(test))]
fn main() {
    println!(
        "This binary requires dev-dependencies and should be built with 'cargo test --no-run'"
    );
    std::process::exit(1);
}
