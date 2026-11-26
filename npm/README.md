# Ckmeans

A WebAssembly build of the [ckmeans](https://crates.io/crates/ckmeans) Rust crate, an implementation of Wang and Song's Ckmeans algorithm for optimal clustering of one-dimensional data.

Ckmeans is a dynamic programming approach to the problem of clustering numeric data into groups with the least within-group sum of squared deviations. It improves on heuristic approaches such as Jenks natural breaks. It is useful for visualisation, where a continuous variable is shown in discrete colour or style groups.

The module requires WebAssembly SIMD (`simd128`). All current browsers support it, as does Node.js 16.4 or later.

## Installation

```sh
npm install @urschrei/ckmeans
```

## Usage

### Browser or bundler

```javascript
import init, { ckmeans_wasm, roundbreaks_wasm } from "@urschrei/ckmeans";

// Load the module one time, before you call any of its functions
await init();

const data = [3.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0, 2.0, 3.0, 5.0, 7.0,
  1.0, 2.0, 5.0, 7.0, 1.0, 5.0, 82.0, 1.0, 1.3, 1.1, 78.0];

const clusters = ckmeans_wasm(data, 3);
// [
//   Float64Array [1, 1, 1, 1.1, 1.3, 2, 2, 2, 3, 3, 5, 5, 5, 7, 7],
//   Float64Array [12, 13, 14, 15, 16],
//   Float64Array [78, 82]
// ]

const breaks = roundbreaks_wasm(data, 3);
// Float64Array [10, 50]
```

To load the module from a CDN, import it from the CDN URL:

```javascript
import init, { ckmeans_wasm } from "https://cdn.jsdelivr.net/npm/@urschrei/ckmeans@@VERSION@/ckmeans.js";

await init();
```

### Node.js

```javascript
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { initSync, ckmeans_wasm } from "@urschrei/ckmeans";

const require = createRequire(import.meta.url);
initSync({ module: readFileSync(require.resolve("@urschrei/ckmeans/ckmeans_bg.wasm")) });

const clusters = ckmeans_wasm([1.0, 1.0, 50.0, 50.0, 100.0], 3);
```

### Observable

```javascript
ckmeans = {
  const module = await import("https://cdn.jsdelivr.net/npm/@urschrei/ckmeans@@VERSION@/ckmeans.js");
  await module.default();
  return module;
}
```

## Functions

Each function accepts the data as a `Float64Array` or as an array of numbers. The data does not have to be sorted. A function throws an `Error` if the data contains `NaN`, or if the number of clusters is 0 or greater than the number of values.

### `ckmeans_wasm(data, nclusters)`

Divides `data` into `nclusters` clusters (1 to 255). Returns an array of `Float64Array`s, one for each cluster, in ascending order.

### `roundbreaks_wasm(data, nclusters)`

Returns a `Float64Array` of `nclusters - 1` rounded break values between the clusters, for use as labels.

### `ckmeans_optimal_wasm(data, k_min, k_max)`

Selects the number of clusters with the Bayesian Information Criterion (BIC), and divides `data` into that number of clusters. `k_min` and `k_max` are optional, with the default values 1 and 9. If `k_max` is greater than the number of distinct values, the function uses the number of distinct values. The function throws an `Error` if `k_min` is 0, if `k_min` is greater than `k_max`, or if `k_min` is greater than the number of distinct values. The function selects the same number of clusters as the [Ckmeans.1d.dp](https://cran.r-project.org/package=Ckmeans.1d.dp) R package, which reports the negative of these BIC values.

```javascript
const result = ckmeans_optimal_wasm([1, 1, 1, 50, 50, 50, 100, 100, 100]);
// {
//   clusters: [Float64Array [1, 1, 1], Float64Array [50, 50, 50], Float64Array [100, 100, 100]],
//   k: 3,
//   bic: [{ k: 1, value: ... }, { k: 2, value: ... }, { k: 3, value: ... }],
//   stats: [{ center: 1, size: 3, withinss: 0 }, ...]
// }
```

`bic` contains the BIC value for each value of k that the function evaluated. `stats` contains the centre, size and within-cluster sum of squares of each cluster.

## Limits

The [crate README](https://github.com/urschrei/ckmeans#numerical-limits) describes the results for infinite values, very large ranges and large integers.

## Licence

[Blue Oak Model License 1.0.0](https://blueoakcouncil.org/license/1.0.0)

## References

1. [Wang, H., & Song, M. (2011). Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming. The R Journal, 3(2), 29.](https://doi.org/10.32614/RJ-2011-015)
