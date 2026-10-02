# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). For releases
before 2.1.0, see the [GitHub releases](https://github.com/urschrei/ckmeans/releases).

## [Unreleased]

### Changed

- The search for the optimal split point divides long ranges into four independent lanes, so
  that the compiler can vectorise it. On x86-64, the library selects AVX2 at run time if the CPU
  supports it, with [`fearless_simd`](https://crates.io/crates/fearless_simd). For 110,000 to
  1,000,000 values, `ckmeans` is 14 to 19 percent faster on an Apple M2 Pro, and 22 to 32 percent
  faster on an AMD EPYC with AVX2. The clusters do not change.
- `ckmeans_optimal` calculates the BIC values faster. It converts the input to `f64` one time,
  vectorises the calculation of the log densities, and does not call `exp()` for the largest
  density at each value. For k = 1 to 9 on an Apple M2 Pro, `ckmeans_optimal` is 27 percent
  faster than in version 2.1.0 for 110,000 values, and 31 percent faster for 1,000,000 values.
  The BIC values do not change.
- The minimum supported Rust version is 1.89.

## [2.1.0] - 2026-10-01

### Fixed

- `ckmeans_optimal` selected `k_max` for most inputs. It now calculates the BIC of a Gaussian
  mixture with one component for each cluster, as the
  [Ckmeans.1d.dp](https://cran.r-project.org/package=Ckmeans.1d.dp) R package does, and selects
  the same k as that package. For most inputs, the selected k and the values in
  `CkmeansResult::bic` change. The R package reports the negative of these BIC values.
  This change also applies to `ckmeans_optimal_wasm`.

### Changed

- `ckmeans_optimal` fills the dynamic programming matrices one time, for `k_max`, and sorts the
  input one time. For the default range of k, 1 to 9, it is approximately 3.5 times faster than
  in version 2.0.0.

### Added

- The `bench_cpp` crate, which compares this library with the C++ implementation of the
  Ckmeans.1d.dp R package. It is not part of the published crate. See
  [`bench_cpp/README.md`](bench_cpp/README.md).

[Unreleased]: https://github.com/urschrei/ckmeans/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/urschrei/ckmeans/compare/v2.0.0...v2.1.0
