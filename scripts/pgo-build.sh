#!/usr/bin/env bash
set -euo pipefail

# PGO Build Script for ckmeans
# This script automates the Profile-Guided Optimization build process

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colours for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Colour

# Platform detection
PLATFORM=$(uname -s)
ARCH=$(uname -m)

echo -e "${GREEN}=== PGO Build for ckmeans ===${NC}"
echo "Platform: $PLATFORM $ARCH"
echo

# Clean previous PGO data
echo -e "${YELLOW}Cleaning previous PGO data...${NC}"
rm -rf target/pgo-profiles
mkdir -p target/pgo-profiles

# Step 1: Build with PGO instrumentation
echo -e "${GREEN}Step 1: Building with PGO instrumentation...${NC}"
# Build PGO training binary in the tools directory
cd "$PROJECT_ROOT/tools"
RUSTFLAGS="-Cprofile-generate=$PROJECT_ROOT/target/pgo-profiles" \
    cargo build --bin pgo_training --profile pgo-generate
cd "$PROJECT_ROOT"

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build with PGO instrumentation${NC}"
    exit 1
fi

# Step 2: Run training workloads
echo -e "${GREEN}Step 2: Running training workloads...${NC}"
./tools/target/pgo-generate/pgo_training

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to run training workloads${NC}"
    exit 1
fi

# Step 3: Merge profiling data
echo -e "${GREEN}Step 3: Merging profiling data...${NC}"

# First check for Rust's bundled LLVM tools
RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
RUST_LLVM_PROFDATA=$(find "$RUSTUP_HOME" -name "llvm-profdata" -type f 2>/dev/null | head -n1)

if [ -n "$RUST_LLVM_PROFDATA" ] && [ -x "$RUST_LLVM_PROFDATA" ]; then
    PROFDATA_CMD="$RUST_LLVM_PROFDATA"
    echo "Using Rust's bundled llvm-profdata: $PROFDATA_CMD"
elif command -v llvm-profdata &> /dev/null; then
    PROFDATA_CMD="llvm-profdata"
elif command -v llvm-profdata-18 &> /dev/null; then
    PROFDATA_CMD="llvm-profdata-18"
elif command -v llvm-profdata-17 &> /dev/null; then
    PROFDATA_CMD="llvm-profdata-17"
else
    echo -e "${RED}llvm-profdata not found. Please install LLVM tools or ensure rustup's llvm-tools-preview is installed.${NC}"
    echo "You can install it with: rustup component add llvm-tools-preview"
    exit 1
fi

$PROFDATA_CMD merge -o "$PROJECT_ROOT/target/pgo-profiles/merged.profdata" \
    "$PROJECT_ROOT/target/pgo-profiles"/*.profraw

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to merge profiling data${NC}"
    exit 1
fi

# Step 4: Build with PGO optimization
echo -e "${GREEN}Step 4: Building with PGO optimization...${NC}"
RUSTFLAGS="-Cprofile-use=$PROJECT_ROOT/target/pgo-profiles/merged.profdata -Cllvm-args=-pgo-warn-missing-function" \
    cargo build --profile pgo-use

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to build with PGO optimization${NC}"
    exit 1
fi

# Step 5: Copy optimized binary to a clear location
echo -e "${GREEN}Step 5: Packaging PGO-optimized build...${NC}"
mkdir -p target/pgo-optimized
cp target/pgo-use/libckmeans.* target/pgo-optimized/ 2>/dev/null || true
cp target/pgo-use/ckmeans* target/pgo-optimized/ 2>/dev/null || true

echo
echo -e "${GREEN}=== PGO Build Complete! ===${NC}"
echo "Optimized binaries are in: target/pgo-optimized/"
echo
echo "To use the PGO-optimized library in your project:"
echo "  - Static library: target/pgo-optimized/libckmeans.rlib"
echo "  - Dynamic library: target/pgo-optimized/libckmeans.so (or .dylib on macOS)"

# Optional: Run benchmarks to compare performance
if [ "${RUN_BENCHMARKS:-0}" = "1" ]; then
    echo
    echo -e "${YELLOW}Running performance comparison...${NC}"
    echo "Standard release build:"
    cargo bench --bench benchmark
    
    echo
    echo "PGO-optimized build:"
    CARGO_TARGET_DIR=target/pgo-bench cargo bench --bench benchmark
fi