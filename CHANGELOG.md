# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). For releases
before 2.1.0, see the [GitHub releases](https://github.com/urschrei/ckmeans/releases).

## [Unreleased]

### Changed

- The search for the optimal split point divides long ranges into four independent lanes, so
  that the compiler can vectorise it. On an Apple M2 Pro, `ckmeans` is 14 to 19 percent faster
  for 110,000 to 1,000,000 values. The clusters do not change.

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
