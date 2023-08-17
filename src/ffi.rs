use libc::{c_double, c_schar, c_void, size_t};
use std::f64;
use std::ptr;
use std::slice;

use crate::ckmeans;

/// Wrapper for a void pointer to a sequence of [`Array`](struct.Array.html)s, and the sequence length. Used for FFI.
///
/// Each sequence entry represents a Ckmeans class.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct WrapperArray {
    pub data: *const InternalArray,
    pub len: size_t,
}

/// Wrapper for a void pointer to a sequence of floats representing a Ckmeans class, and the
/// sequence length. Used for FFI.
///
/// `data` is a `Vec<c_double>`.
#[repr(C)]
#[derive(Clone)]
pub struct InternalArray {
    pub data: *const c_void,
    pub len: size_t,
}

/// Wrapper for a void pointer to a sequence of floats representing data to be clustered using
/// Ckmeans, and the sequence length. Used for FFI.
///
/// `data` is a `Vec<c_double>`.
#[repr(C)]
pub struct ExternalArray {
    pub data: *const c_void,
    pub len: size_t,
}

/// We don't need to take ownership of incoming data to be clustered: that happens in CkMeans
impl From<&ExternalArray> for &[f64] {
    fn from(arr: &ExternalArray) -> Self {
        unsafe { slice::from_raw_parts(arr.data as *mut f64, arr.len) }
    }
}

// Convert individual Ckmeans result classes into things that can be leaked across the FFI boundary
impl From<Vec<f64>> for InternalArray {
    fn from(v: Vec<f64>) -> Self {
        let boxed = v.into_boxed_slice();
        let blen = boxed.len();
        let rawp = Box::into_raw(boxed);
        InternalArray {
            data: rawp as *const libc::c_void,
            len: blen as libc::size_t,
        }
    }
}

impl From<Vec<f64>> for ExternalArray {
    fn from(v: Vec<f64>) -> Self {
        let boxed = v.into_boxed_slice();
        let blen = boxed.len();
        let rawp = Box::into_raw(boxed);
        ExternalArray {
            data: rawp as *const libc::c_void,
            len: blen as libc::size_t,
        }
    }
}

impl From<Vec<Vec<f64>>> for WrapperArray {
    fn from(arr: Vec<Vec<f64>>) -> Self {
        let iarrs: Vec<InternalArray> = arr.into_iter().map(|member| member.into()).collect();
        let boxed = iarrs.into_boxed_slice();
        let blen = boxed.len();
        let rawp = Box::into_raw(boxed);
        WrapperArray {
            data: rawp as *const InternalArray,
            len: blen as libc::size_t,
        }
    }
}

// Reconstitute individual CkMeans result classes so they can be eventually dropped
impl From<InternalArray> for Vec<f64> {
    fn from(arr: InternalArray) -> Self {
        // we originated this data, so pointer-to-slice -> box -> vec
        unsafe {
            let p = ptr::slice_from_raw_parts_mut(arr.data as *mut f64, arr.len);
            Box::from_raw(p).to_vec()
        }
    }
}

// Reconstitute a CkMeans result that has been returned across the FFI boundary so it can be dropped
impl From<WrapperArray> for Vec<Vec<f64>> {
    fn from(arr: WrapperArray) -> Self {
        let arrays = unsafe {
            let p = ptr::slice_from_raw_parts_mut(arr.data as *mut InternalArray, arr.len);
            Box::from_raw(p).to_vec()
        };
        arrays.into_iter().map(|arr| arr.into()).collect()
    }
}

#[no_mangle]
pub extern "C" fn ckmeans_ffi(data: &ExternalArray, classes: c_schar) -> WrapperArray {
    ckmeans(data.into(), classes).unwrap().into()
}

#[no_mangle]
pub extern "C" fn drop_ckmeans_result(result: WrapperArray) {
    let _: Vec<Vec<f64>> = result.into();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi() {
        let i = vec![
            1f64, 12., 13., 14., 15., 16., 2., 2., 3., 5., 7., 1., 2., 5., 7., 1., 5., 82., 1.,
            1.3, 1.1, 78.,
        ];
        let res = ckmeans_ffi(&i.into(), 3);
        drop_ckmeans_result(res);
    }
}
