use crate::ckmeans;
use js_sys::Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsValue;

fn inner_vec_to_js_array(data: Vec<f64>) -> Array {
    let array = Array::new();
    for num in data {
        array.push(&JsValue::from_f64(num));
    }
    array
}

// Convert a ckmeans result to an Array suitable for use by a JS function
pub fn wrapper_vec_to_js_array(data: Vec<Vec<f64>>) -> Array {
    let outer_array = Array::new();
    for inner_vec in data {
        outer_array.push(&inner_vec_to_js_array(inner_vec));
    }
    outer_array
}

#[wasm_bindgen]
pub fn ckmeans_wasm(data: &[f64], nclusters: u8) -> Result<Array, JsError> {
    // Your implementation here...

    let res = ckmeans(data, nclusters).unwrap();
    Ok(wrapper_vec_to_js_array(res))
}
