// Load the built npm package in Node.js and check the examples in npm/README.md.
// Usage: node npm/smoke-test.mjs <package directory>
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const dir = resolve(process.argv[2] ?? "pkg");
const mod = await import(pathToFileURL(join(dir, "ckmeans.js")).href);
mod.initSync({ module: readFileSync(join(dir, "ckmeans_bg.wasm")) });

const toArrays = (clusters) => clusters.map((c) => Array.from(c));

const data = [3.0, 12.0, 13.0, 14.0, 15.0, 16.0, 2.0, 2.0, 3.0, 5.0, 7.0,
  1.0, 2.0, 5.0, 7.0, 1.0, 5.0, 82.0, 1.0, 1.3, 1.1, 78.0];
const expected = [
  [1, 1, 1, 1.1, 1.3, 2, 2, 2, 3, 3, 5, 5, 5, 7, 7],
  [12, 13, 14, 15, 16],
  [78, 82],
];

assert.deepEqual(toArrays(mod.ckmeans_wasm(data, 3)), expected);
assert.deepEqual(toArrays(mod.ckmeans_wasm(Float64Array.from(data), 3)), expected);
assert.deepEqual(Array.from(mod.roundbreaks_wasm(data, 3)), [10, 50]);

const optimal = mod.ckmeans_optimal_wasm([1, 1, 1, 50, 50, 50, 100, 100, 100]);
assert.equal(optimal.k, 3);
assert.deepEqual(toArrays(optimal.clusters), [[1, 1, 1], [50, 50, 50], [100, 100, 100]]);
assert.deepEqual(optimal.bic.map((b) => b.k), [1, 2, 3]);
assert.deepEqual(optimal.stats[0], { center: 1, size: 3, withinss: 0 });

const bounded = mod.ckmeans_optimal_wasm(data, 2, 4);
assert.ok(bounded.k >= 2 && bounded.k <= 4);
assert.deepEqual(bounded.bic.map((b) => b.k), [2, 3, 4]);

assert.throws(() => mod.ckmeans_wasm([1, NaN, 3], 2), Error);
assert.throws(() => mod.ckmeans_wasm(data, 0), Error);
assert.throws(() => mod.ckmeans_wasm([1, 2], 3), Error);
assert.throws(() => mod.roundbreaks_wasm([], 1), Error);
assert.throws(() => mod.ckmeans_optimal_wasm(data, 0, 3), Error);
assert.throws(() => mod.ckmeans_optimal_wasm(data, 4, 3), Error);

console.log(`smoke test passed: ${dir}`);
