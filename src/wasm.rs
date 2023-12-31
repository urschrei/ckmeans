use crate::ckmeans;
use crate::roundbreaks;
use js_sys::Array;
use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

/// convert individual ckmeans result classes to WASM-compatible Arrays
fn inner_vec_to_js_array(data: &[f64]) -> Float64Array {
    Float64Array::from(data)
}

/// Convert a ckmeans result to an Array suitable for use by a JS function
// NB: it's crucial to only work with slices here, as taking ownership of data will cause
// dangling references
fn wrapper_vec_to_js_array(data: &[Vec<f64>]) -> Array {
    data.iter().map(|v| inner_vec_to_js_array(v)).collect()
}

#[wasm_bindgen]
/// A WASM wrapper for ckmeans
pub fn ckmeans_wasm(data: &[f64], nclusters: u8) -> Result<Array, JsError> {
    Ok(wrapper_vec_to_js_array(&ckmeans(data, nclusters)?))
}

#[wasm_bindgen]
/// A WASM wrapper for roundbreaks
pub fn roundbreaks_wasm(data: &[f64], nclusters: u8) -> Result<Float64Array, JsError> {
    Ok(inner_vec_to_js_array(&roundbreaks(data, nclusters)?))
}
