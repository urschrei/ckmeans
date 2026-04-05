use crate::ckmeans;
use crate::ckmeans_optimal;
use crate::roundbreaks;
use js_sys::Array;
use js_sys::Float64Array;
use js_sys::Number;
use js_sys::Object;
use js_sys::Reflect;
use wasm_bindgen::JsValue;
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

#[wasm_bindgen]
/// A WASM wrapper for ckmeans_optimal
pub fn ckmeans_optimal_wasm(
    data: &[f64],
    k_min: Option<u8>,
    k_max: Option<u8>,
) -> Result<JsValue, JsError> {
    let result = ckmeans_optimal(data, k_min, k_max)?;
    let obj = Object::new();

    // clusters: Array of Float64Arrays
    let clusters_js = wrapper_vec_to_js_array(&result.clusters);
    Reflect::set(&obj, &"clusters".into(), &clusters_js).unwrap();

    // k: number
    Reflect::set(&obj, &"k".into(), &Number::from(result.k as f64)).unwrap();

    // bic: Array of {k, value} objects
    let bic_arr = Array::new();
    for (k, value) in &result.bic {
        let entry = Object::new();
        Reflect::set(&entry, &"k".into(), &Number::from(*k as f64)).unwrap();
        Reflect::set(&entry, &"value".into(), &Number::from(*value)).unwrap();
        bic_arr.push(&entry);
    }
    Reflect::set(&obj, &"bic".into(), &bic_arr).unwrap();

    // stats: Array of {center, size, withinss} objects
    let stats_arr = Array::new();
    for stat in &result.stats {
        let entry = Object::new();
        Reflect::set(&entry, &"center".into(), &Number::from(stat.center)).unwrap();
        Reflect::set(&entry, &"size".into(), &Number::from(stat.size as f64)).unwrap();
        Reflect::set(&entry, &"withinss".into(), &Number::from(stat.withinss)).unwrap();
        stats_arr.push(&entry);
    }
    Reflect::set(&obj, &"stats".into(), &stats_arr).unwrap();

    Ok(obj.into())
}
