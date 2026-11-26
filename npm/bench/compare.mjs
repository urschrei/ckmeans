// Compare the speed of the WASM build with two JavaScript implementations:
// simple-statistics and David Schnurr's ckmeans.
// Usage: node compare.mjs <package directory>
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { ckmeans as ssCkmeans } from "simple-statistics";

const require = createRequire(import.meta.url);
const schnerd = require("ckmeans");

const dir = resolve(process.argv[2] ?? "../../pkg");
const wasm = await import(pathToFileURL(join(dir, "ckmeans.js")).href);
wasm.initSync({ module: readFileSync(join(dir, "ckmeans_bg.wasm")) });

// Seeded generator, so that each run uses the same data
let seed = 7;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
const gauss = () => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
// A mixture of five normal distributions
const make = (n) => Array.from({ length: n }, (_, i) => (i % 5) * 100 + gauss() * 15);

// Run f for at least budgetMs after one warm-up call, and return the mean time of a call
const time = (f, budgetMs) => {
  f();
  let runs = 0;
  const t0 = performance.now();
  do {
    f();
    runs++;
  } while (performance.now() - t0 < budgetMs);
  return (performance.now() - t0) / runs;
};

const k = 7;
const widths = [9, 11, 12, 13, 10];
const columns = ["n", "wasm(F64)", "wasm(Array)", "simple-stats", "schnerd"];
console.log(`k=${k}; mean time of a call in ms; Node.js ${process.version}`);
console.log(columns.map((c, i) => c.padStart(widths[i])).join(" "));

let failed = false;
for (const n of [1_000, 10_000, 110_000, 1_000_000]) {
  const arr = make(n);
  const f64 = Float64Array.from(arr);

  // Schnurr's ckmeans returns the first value of each cluster
  const clusters = wasm.ckmeans_wasm(f64, k).map((c) => Array.from(c));
  const same = JSON.stringify(clusters) === JSON.stringify(ssCkmeans(arr, k))
    && JSON.stringify(clusters.map((c) => c[0])) === JSON.stringify(Array.from(schnerd(arr, k)));
  failed ||= !same;

  const budget = n >= 1_000_000 ? 3000 : 1500;
  const times = [
    time(() => wasm.ckmeans_wasm(f64, k), budget),
    time(() => wasm.ckmeans_wasm(arr, k), budget),
    time(() => ssCkmeans(arr, k), budget),
    time(() => schnerd(arr, k), budget),
  ];
  const row = [String(n), ...times.map((t) => t.toFixed(2))];
  console.log(row.map((c, i) => c.padStart(widths[i])).join(" "), same ? "" : "RESULTS DIFFER");
}

if (failed) {
  console.error("The implementations returned different clusters for at least one input.");
  process.exit(1);
}
