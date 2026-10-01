use std::path::Path;

/// The C++ sources that the ckmeans-1d-dp Python package compiles, without its pybind11 wrapper
const SOURCES: [&str; 11] = [
    "Ckmeans.1d.dp.cpp",
    "dynamic_prog.cpp",
    "EWL2_dynamic_prog.cpp",
    "EWL2_fill_log_linear.cpp",
    "EWL2_fill_quadratic.cpp",
    "EWL2_fill_SMAWK.cpp",
    "fill_log_linear.cpp",
    "fill_quadratic.cpp",
    "fill_SMAWK.cpp",
    "select_levels.cpp",
    "weighted_select_levels.cpp",
];

fn main() {
    let vendor = Path::new("vendor/Ckmeans.1d.dp");
    println!("cargo::rerun-if-changed=cpp");
    println!("cargo::rerun-if-changed={}", vendor.display());

    // Use -O3 and NDEBUG in all profiles, as the ckmeans-1d-dp Python build does
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++11")
        .opt_level(3)
        .define("NDEBUG", None)
        .warnings(false)
        .include(vendor)
        .file("cpp/shim.cpp");
    for source in SOURCES {
        build.file(vendor.join(source));
    }
    build.compile("ckmeans_cpp");
}
