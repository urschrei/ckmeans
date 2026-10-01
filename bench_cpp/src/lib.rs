//! A safe wrapper around the Ckmeans.1d.dp C++ implementation, for use in comparison benchmarks.
//!
//! The wrapper calls `kmeans_1d_dp()` in the same way as the `ckmeans-1d-dp` Python package does
//! with its default arguments: unweighted input, the L2 criterion, and BIC level selection.
//! [`ckmeans_cpp_cluster`] uses a range of levels that contains only `k`.

use std::ffi::c_int;
use std::sync::Mutex;

unsafe extern "C" {
    fn ckmeans_cpp(
        x: *const f64,
        n: usize,
        kmin: usize,
        kmax: usize,
        method: c_int,
        cluster: *mut c_int,
        centers: *mut f64,
        withinss: *mut f64,
        size: *mut f64,
        bic: *mut f64,
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

/// The output of `kmeans_1d_dp()`. Each `Vec` except `cluster` has one element for each
/// cluster that was requested. If the input has fewer unique values than that, the elements
/// after that count are not set.
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

/// The output of [`ckmeans_cpp_optimal`]
#[derive(Clone, Debug, PartialEq)]
pub struct CppOptimal {
    /// The selected number of clusters
    pub k: usize,
    /// The BIC value for each evaluated k, as `(k, bic)` pairs. The C++ code calculates
    /// `2 ln L - p ln n`, so a larger value is better. This is the negative of the value that
    /// `ckmeans::ckmeans_optimal` reports.
    pub bic: Vec<(usize, f64)>,
    /// The clustering for the selected k. The `Vec`s other than `cluster` have `k` elements.
    pub clustering: CppClustering,
}

/// The errors that the functions in this crate can return
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CppError {
    /// `kmin` is zero, `kmin` is larger than `kmax`, or `kmax` is larger than the input length
    InvalidK,
    /// The input contains a NaN or infinite value
    NonFinite,
    /// The C++ code raised an exception
    Exception,
}

/// Call `kmeans_1d_dp()` with BIC selection of k in `kmin..=kmax`.
/// Return the clustering and the BIC buffer, which has `kmax - kmin + 1` elements.
/// The C++ code does not write the BIC elements for k larger than the number of unique values,
/// and these elements are NaN.
fn run(
    x: &[f64],
    kmin: usize,
    kmax: usize,
    method: Method,
) -> Result<(CppClustering, Vec<f64>), CppError> {
    if kmin == 0 || kmin > kmax || kmax > x.len() {
        return Err(CppError::InvalidK);
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(CppError::NonFinite);
    }
    let mut clustering = CppClustering {
        cluster: vec![0; x.len()],
        centers: vec![0.0; kmax],
        withinss: vec![0.0; kmax],
        size: vec![0.0; kmax],
    };
    let mut bic = vec![f64::NAN; kmax - kmin + 1];
    let status = {
        let _guard = CPP_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: x and cluster have x.len() elements. centers, withinss and size have kmax
        // elements, which is the maximum number of clusters that the C++ code writes. bic has
        // kmax - kmin + 1 elements. The lock prevents concurrent calls.
        unsafe {
            ckmeans_cpp(
                x.as_ptr(),
                x.len(),
                kmin,
                kmax,
                method.code(),
                clustering.cluster.as_mut_ptr(),
                clustering.centers.as_mut_ptr(),
                clustering.withinss.as_mut_ptr(),
                clustering.size.as_mut_ptr(),
                bic.as_mut_ptr(),
            )
        }
    };
    match status {
        0 => Ok((clustering, bic)),
        _ => Err(CppError::Exception),
    }
}

/// Cluster `x` into `k` clusters with the C++ implementation.
///
/// # Errors
/// Returns [`CppError`] if `k` is not in `1..=x.len()`, if `x` contains a value that is not
/// finite, or if the C++ code raises an exception.
pub fn ckmeans_cpp_cluster(x: &[f64], k: usize, method: Method) -> Result<CppClustering, CppError> {
    run(x, k, k, method).map(|(clustering, _)| clustering)
}

/// Cluster `x` with the C++ implementation, and select the number of clusters in `kmin..=kmax`
/// with the BIC.
///
/// # Errors
/// Returns [`CppError`] if `kmin` is zero, if `kmin` is larger than `kmax`, if `kmax` is larger
/// than `x.len()`, if `x` contains a value that is not finite, or if the C++ code raises an
/// exception.
pub fn ckmeans_cpp_optimal(
    x: &[f64],
    kmin: usize,
    kmax: usize,
    method: Method,
) -> Result<CppOptimal, CppError> {
    let (mut clustering, bic) = run(x, kmin, kmax, method)?;
    let k = clustering
        .cluster
        .iter()
        .max()
        .map_or(0, |&c| c as usize + 1);
    clustering.centers.truncate(k);
    clustering.withinss.truncate(k);
    clustering.size.truncate(k);
    let bic = (kmin..=kmax)
        .zip(bic)
        .filter(|(_, value)| !value.is_nan())
        .collect();
    Ok(CppOptimal { k, bic, clustering })
}
