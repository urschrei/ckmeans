//! A safe wrapper around the Ckmeans.1d.dp C++ implementation, for use in comparison benchmarks.
//!
//! The wrapper calls `kmeans_1d_dp()` in the same way as the `ckmeans-1d-dp` Python package does
//! with its default arguments: unweighted input, the L2 criterion, and BIC level selection over
//! a range that contains only `k`.

use std::ffi::c_int;
use std::sync::Mutex;

unsafe extern "C" {
    fn ckmeans_cpp(
        x: *const f64,
        n: usize,
        k: usize,
        method: c_int,
        cluster: *mut c_int,
        centers: *mut f64,
        withinss: *mut f64,
        size: *mut f64,
    ) -> c_int;
}

/// `kmeans_1d_dp()` sorts through a static pointer, so concurrent calls are not safe.
static CPP_LOCK: Mutex<()> = Mutex::new(());

/// The algorithm that fills the dynamic programming matrix
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// SMAWK, O(kn). The default in the R and Python packages.
    Linear,
    /// Divide and conquer, O(kn log n)
    LogLinear,
    /// O(kn^2)
    Quadratic,
}

impl Method {
    fn code(self) -> c_int {
        match self {
            Method::Linear => 0,
            Method::LogLinear => 1,
            Method::Quadratic => 2,
        }
    }

    /// The name that the R and Python packages use for this method
    pub fn name(self) -> &'static str {
        match self {
            Method::Linear => "linear",
            Method::LogLinear => "loglinear",
            Method::Quadratic => "quadratic",
        }
    }
}

/// The output of `kmeans_1d_dp()`. Each `Vec` except `cluster` has `k` elements.
/// If the input has fewer than `k` unique values, the elements after that count are not set.
#[derive(Clone, Debug, PartialEq)]
pub struct CppClustering {
    /// The zero-based cluster number of each input value, in input order
    pub cluster: Vec<i32>,
    /// The mean of each cluster
    pub centers: Vec<f64>,
    /// The within-cluster sum of squares of each cluster
    pub withinss: Vec<f64>,
    /// The number of values in each cluster
    pub size: Vec<f64>,
}

/// The errors that [`ckmeans_cpp_cluster`] can return
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CppError {
    /// `k` is zero or larger than the input length
    InvalidK,
    /// The input contains a NaN or infinite value
    NonFinite,
    /// The C++ code raised an exception
    Exception,
}

/// Cluster `x` into `k` clusters with the C++ implementation.
///
/// # Errors
/// Returns [`CppError`] if `k` is not in `1..=x.len()`, if `x` contains a value that is not
/// finite, or if the C++ code raises an exception.
pub fn ckmeans_cpp_cluster(x: &[f64], k: usize, method: Method) -> Result<CppClustering, CppError> {
    if k == 0 || k > x.len() {
        return Err(CppError::InvalidK);
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(CppError::NonFinite);
    }
    let mut result = CppClustering {
        cluster: vec![0; x.len()],
        centers: vec![0.0; k],
        withinss: vec![0.0; k],
        size: vec![0.0; k],
    };
    let status = {
        let _guard = CPP_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: x has x.len() elements, cluster has x.len() elements, and the other output
        // buffers have k elements, which is the maximum number of clusters that the C++ code
        // writes. The lock prevents concurrent calls.
        unsafe {
            ckmeans_cpp(
                x.as_ptr(),
                x.len(),
                k,
                method.code(),
                result.cluster.as_mut_ptr(),
                result.centers.as_mut_ptr(),
                result.withinss.as_mut_ptr(),
                result.size.as_mut_ptr(),
            )
        }
    };
    match status {
        0 => Ok(result),
        _ => Err(CppError::Exception),
    }
}
