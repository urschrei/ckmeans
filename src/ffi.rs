//! The FFI module for Ckmeans

use libc::c_uchar;
use libc::c_void;
use libc::size_t;
use std::f64;
use std::panic;
use std::ptr;
use std::slice;

use crate::CkmeansErr;
use crate::ckmeans;

/// Status of a [`ckmeans_ffi`] call.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CkmeansStatus {
    /// The call succeeded.
    Ok = 0,
    /// `classes` is 0.
    TooFewClasses = 1,
    /// `classes` is greater than the number of data values.
    TooManyClasses = 2,
    /// The data contains NaN.
    NanInput = 3,
    /// The data pointer is null and the length is not 0.
    NullData = 4,
    /// An internal error occurred.
    InternalError = 5,
}

impl From<&CkmeansErr> for CkmeansStatus {
    fn from(err: &CkmeansErr) -> Self {
        match err {
            CkmeansErr::TooFewClassesError => CkmeansStatus::TooFewClasses,
            CkmeansErr::TooManyClassesError => CkmeansStatus::TooManyClasses,
            CkmeansErr::NanError => CkmeansStatus::NanInput,
            CkmeansErr::ConversionError
            | CkmeansErr::LowWindowError
            | CkmeansErr::HighWindowError
            | CkmeansErr::InfallibleError
            | CkmeansErr::InvalidRangeError => CkmeansStatus::InternalError,
        }
    }
}

/// Wrapper for a void pointer to a sequence of [`InternalArray`]s, and the sequence length. Used for FFI.
///
/// Each sequence entry represents a single [ckmeans] result class.
#[repr(C)]
pub struct WrapperArray {
    pub data: *const c_void,
    pub len: size_t,
}

/// Wrapper for a void pointer to a sequence of floats representing a single [ckmeans] result class, and the
/// sequence length. Used for FFI.
#[repr(C)]
pub struct InternalArray {
    pub data: *const c_void,
    pub len: size_t,
}

/// Wrapper for a void pointer to a sequence of floats representing data to be clustered using
/// [ckmeans], and the sequence length. Used for FFI.
#[repr(C)]
pub struct ExternalArray {
    pub data: *const c_void,
    pub len: size_t,
}

/// Borrow the values of an [`ExternalArray`] without taking ownership. Returns
/// `None` if the data pointer is null and the length is not 0.
///
/// # Safety
///
/// If `arr.data` is not null, it must point to `arr.len` initialised, aligned
/// `f64` values that stay valid and unchanged for `'a`.
unsafe fn external_slice<'a>(arr: &ExternalArray) -> Option<&'a [f64]> {
    if arr.data.is_null() {
        return (arr.len == 0).then_some(&[]);
    }
    Some(unsafe { slice::from_raw_parts(arr.data.cast(), arr.len) })
}

/// Leak the clusters so that they can be returned across the FFI boundary.
/// [`reclaim_clusters`] takes ownership again.
fn leak_clusters(clusters: Vec<Vec<f64>>) -> WrapperArray {
    let classes: Box<[InternalArray]> = clusters
        .into_iter()
        .map(|cluster| {
            let boxed = cluster.into_boxed_slice();
            InternalArray {
                len: boxed.len(),
                data: Box::into_raw(boxed).cast(),
            }
        })
        .collect();
    WrapperArray {
        len: classes.len(),
        data: Box::into_raw(classes).cast(),
    }
}

/// Take ownership of clusters that [`leak_clusters`] leaked.
///
/// # Safety
///
/// `result` must be a value returned by [`leak_clusters`], and ownership must
/// not be taken more than once.
unsafe fn reclaim_clusters(result: WrapperArray) -> Vec<Vec<f64>> {
    let classes: *mut [InternalArray] = ptr::slice_from_raw_parts_mut(result.data as _, result.len);
    unsafe { Box::from_raw(classes) }
        .into_iter()
        .map(|class| {
            let values: *mut [f64] = ptr::slice_from_raw_parts_mut(class.data as _, class.len);
            unsafe { Box::from_raw(values) }.into_vec()
        })
        .collect()
}

/// An FFI wrapper for [ckmeans].
///
/// On success, the function writes [`CkmeansStatus::Ok`] to `status` and returns the clusters. On
/// failure, it writes the error to `status` and returns a [`WrapperArray`] with a null `data`
/// pointer and a `len` of 0. `status` can be null. The function does not panic across the FFI
/// boundary.
///
/// Data returned by this function **must** be freed by calling [`drop_ckmeans_result`].
///
/// # Safety
///
/// - If `data.data` is not null, it must point to `data.len` initialised, aligned `f64` values.
/// - If `status` is not null, it must point to memory that is valid for a write of a
///   [`CkmeansStatus`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ckmeans_ffi(
    data: ExternalArray,
    classes: c_uchar,
    status: *mut CkmeansStatus,
) -> WrapperArray {
    let outcome = match unsafe { external_slice(&data) } {
        None => Err(CkmeansStatus::NullData),
        Some(values) => panic::catch_unwind(|| ckmeans(values, classes))
            .map_err(|_| CkmeansStatus::InternalError)
            .and_then(|result| result.map_err(|err| CkmeansStatus::from(&err))),
    };
    let (code, result) = match outcome {
        Ok(clusters) => (CkmeansStatus::Ok, leak_clusters(clusters)),
        Err(code) => (
            code,
            WrapperArray {
                data: ptr::null(),
                len: 0,
            },
        ),
    };
    if !status.is_null() {
        unsafe { status.write(code) };
    }
    result
}

/// Drop data returned by [`ckmeans_ffi`]. A result with a null `data` pointer is ignored.
///
/// # Safety
///
/// `result` must be a value returned by [`ckmeans_ffi`], and it must not be dropped more than
/// once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn drop_ckmeans_result(result: WrapperArray) {
    if result.data.is_null() {
        return;
    }
    drop(unsafe { reclaim_clusters(result) });
}

#[cfg(test)]
mod tests {
    use super::*;
    use hegel::TestCase;
    use hegel::generators as gs;

    /// Borrow `data` as an [`ExternalArray`].
    fn borrow(data: &[f64]) -> ExternalArray {
        ExternalArray {
            data: data.as_ptr().cast(),
            len: data.len(),
        }
    }

    /// Call [`ckmeans_ffi`] and return the status and the result.
    fn call(data: ExternalArray, classes: u8) -> (CkmeansStatus, WrapperArray) {
        let mut status = CkmeansStatus::InternalError;
        let result = unsafe { ckmeans_ffi(data, classes, &mut status) };
        (status, result)
    }

    /// Assert that `result` is the empty error result, then drop it.
    fn assert_empty(result: WrapperArray) {
        assert!(result.data.is_null());
        assert_eq!(result.len, 0);
        unsafe { drop_ckmeans_result(result) };
    }

    #[test]
    fn ffi() {
        let i = vec![
            1f64, 12., 13., 14., 15., 16., 2., 2., 3., 5., 7., 1., 2., 5., 7., 1., 5., 82., 1.,
            1.3, 1.1, 78.,
        ];
        let res = unsafe { reclaim_clusters(ckmeans_ffi(borrow(&i), 3, ptr::null_mut())) };
        let expected = vec![
            vec![
                1.0, 1.0, 1.0, 1.0, 1.1, 1.3, 2.0, 2.0, 2.0, 3.0, 5.0, 5.0, 5.0, 7.0, 7.0,
            ],
            vec![12., 13., 14., 15., 16.],
            vec![78., 82.],
        ];
        assert_eq!(res, expected);
    }

    #[hegel::test(test_cases = 1000)]
    fn ffi_round_trip_matches_ckmeans(tc: TestCase) {
        let data = tc.draw(
            gs::vecs(gs::floats::<f64>().allow_nan(false))
                .min_size(1)
                .max_size(60),
        );
        let max = u8::try_from(data.len()).unwrap_or(u8::MAX);
        let k = tc.draw(gs::integers::<u8>().min_value(1).max_value(max));
        let (status, result) = call(borrow(&data), k);
        assert_eq!(status, CkmeansStatus::Ok);
        let result = unsafe { reclaim_clusters(result) };
        assert_eq!(result, ckmeans(&data, k).unwrap());
    }

    #[hegel::test(test_cases = 1000)]
    fn ffi_reports_nan_input(tc: TestCase) {
        let mut data = tc.draw(gs::vecs(gs::floats::<f64>().allow_nan(false)).max_size(60));
        let position = tc.draw(gs::integers::<usize>().max_value(data.len()));
        data.insert(position, f64::NAN);
        let max = u8::try_from(data.len()).unwrap_or(u8::MAX);
        let k = tc.draw(gs::integers::<u8>().min_value(1).max_value(max));
        let (status, result) = call(borrow(&data), k);
        assert_eq!(status, CkmeansStatus::NanInput);
        assert_empty(result);
    }

    #[hegel::test(test_cases = 500)]
    fn ffi_reports_zero_classes(tc: TestCase) {
        let data = tc.draw(gs::vecs(gs::floats::<f64>()).max_size(60));
        let (status, result) = call(borrow(&data), 0);
        assert_eq!(status, CkmeansStatus::TooFewClasses);
        assert_empty(result);
    }

    #[hegel::test(test_cases = 500)]
    fn ffi_reports_too_many_classes(tc: TestCase) {
        let data = tc.draw(gs::vecs(gs::floats::<f64>().allow_nan(false)).max_size(60));
        let k = tc.draw(gs::integers::<u8>().min_value(data.len() as u8 + 1));
        let (status, result) = call(borrow(&data), k);
        assert_eq!(status, CkmeansStatus::TooManyClasses);
        assert_empty(result);
    }

    #[hegel::test(test_cases = 500)]
    fn ffi_reports_null_data(tc: TestCase) {
        let len = tc.draw(gs::integers::<usize>().min_value(1));
        let k = tc.draw(gs::integers::<u8>());
        let external = ExternalArray {
            data: ptr::null(),
            len,
        };
        let (status, result) = call(external, k);
        assert_eq!(status, CkmeansStatus::NullData);
        assert_empty(result);
    }

    #[hegel::test(test_cases = 500)]
    fn ffi_treats_null_empty_data_as_empty(tc: TestCase) {
        let k = tc.draw(gs::integers::<u8>());
        let external = ExternalArray {
            data: ptr::null(),
            len: 0,
        };
        let (status, result) = call(external, k);
        let expected = CkmeansStatus::from(&ckmeans::<f64>(&[], k).unwrap_err());
        assert_eq!(status, expected);
        assert_empty(result);
    }
}
