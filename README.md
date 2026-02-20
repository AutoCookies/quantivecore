# QuantCore v1.0.0

Production-grade bit-level GEMM engine (binary/ternary) with stable C ABI.

## Stable ABI

Only `include/quantcore/c_api.h` is ABI-stable for 1.x.

Exported symbols:
- `qc_version`
- `qc_binary_gemm`
- `qc_ternary_gemm`

## Version

Current release: **1.0.0**.

## Build

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Install

```bash
cmake -S . -B build-release
cmake --build build-release -j
cmake --install build-release
```

## Reproducible builds

```bash
export SOURCE_DATE_EPOCH=1700000000
cmake -S . -B build-r1 -DQC_REPRODUCIBLE=ON
cmake --build build-r1 -j
```

## Performance freeze baseline (binary 2048)

See `bench/regression_baseline.json` and CI regression gate.

## Deployment examples

- C++: `examples/simple_inference.cpp`
- Go: `examples/go_inference_example/main.go`

## Release checklist (v1.0.0)

- [x] C ABI frozen
- [x] Shared/static packaging + pkg-config
- [x] Sanitizer jobs configured
- [x] Cross-platform CI matrix configured
- [x] Go wrapper includes `BinaryGEMM`, `TernaryGEMM`, `Version`
- [x] Reproducible build check configured
