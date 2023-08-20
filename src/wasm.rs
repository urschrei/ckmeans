use crate::ckmeans;
use crate::roundbreaks;
use js_sys::Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

/// convert individual ckmeans result classes to WASM-compatible Arrays
fn inner_vec_to_js_array(data: Vec<f64>) -> Array {
    data.into_iter().map(JsValue::from).collect()
}

/// Convert a ckmeans result to an Array suitable for use by a JS function
fn wrapper_vec_to_js_array(data: Vec<Vec<f64>>) -> Array {
    data.into_iter().map(inner_vec_to_js_array).collect()
}

#[wasm_bindgen]
/// A WASM wrapper for ckmeans
pub fn ckmeans_wasm(data: &[f64], nclusters: u8) -> Result<Array, JsError> {
    Ok(wrapper_vec_to_js_array(ckmeans(data, nclusters)?))
}

#[wasm_bindgen]
/// A WASM wrapper for roundbreaks
pub fn roundbreaks_wasm(data: &[f64], nclusters: u8) -> Result<Array, JsError> {
    Ok(inner_vec_to_js_array(roundbreaks(data, nclusters)?))
}
