#!/usr/bin/env bash
set -euo pipefail

# Build the @urschrei/ckmeans npm package in pkg/ and test it with Node.js.
# The package version is the crate version in Cargo.toml.
# Publish with: npm publish ./pkg --access public

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Remove the files of earlier builds, so that the package contains only the files of this build.
rm -rf pkg

# simd128 lets fearless_simd use WebAssembly SIMD. WebAssembly has no run-time
# feature detection, so the module needs a host that supports SIMD.
RUSTFLAGS="-C target-feature=+simd128" \
    wasm-pack build --target web --release --scope urschrei --out-dir pkg

version="$(node -p 'require("./pkg/package.json").version')"
sed "s/@VERSION@/${version}/g" npm/README.md > pkg/README.md

node npm/smoke-test.mjs ./pkg
npm pack --dry-run ./pkg

echo
echo "Built @urschrei/ckmeans ${version} in pkg/."
echo "Publish with: npm publish ./pkg --access public"
