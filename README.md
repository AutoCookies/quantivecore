# QuantCore

QuantCore is a bit-level linear algebra engine focused on binary (1-bit) and ternary (2-bit plane-packed) GEMM/GEMV.

## ISA Support Matrix

| ISA | Status | Notes |
| --- | --- | --- |
| Scalar C++20 | ✅ | Correctness reference path |
| AVX2 | ✅ | Production fallback SIMD |
| AVX-512F | ✅ | Runtime-dispatched fast path |
| AVX-512VPOPCNTDQ | ✅ (if CPU supports) | Popcount acceleration for AVX-512 path |
| AVX-512VNNI | ⚙️ optional | Reserved for future fused scoring path |
| AMX tile | 🧪 experimental | Optional, isolated source path |

## Dispatch Behavior

Runtime dispatch order for binary GEMM:
1. AMX path when available.
2. AVX-512F + AVX-512VPOPCNTDQ blocked kernel.
3. AVX2 kernel.
4. Scalar reference.

Ternary GEMM follows AVX-512 -> AVX2 -> Scalar.

Dispatch checks are guarded by `__builtin_cpu_supports` so unsupported systems never execute unsupported code.

## Cache Blocking

Blocked kernels use compile-time configurable tile sizes:
- `QUANTCORE_BLOCK_MB` (default 64)
- `QUANTCORE_BLOCK_NB` (default 64)
- `QUANTCORE_BLOCK_KB_BLOCKS` (default 8 blocks of 64 elements)

## NUMA / Threading

Threaded execution partitions the M dimension deterministically and pins worker threads via Linux affinity when available.
If NUMA/affinity APIs are unavailable, behavior gracefully falls back.

## Build

```bash
cmake -S . -B build -DQUANTCORE_ENABLE_ASAN=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Go Wrapper

```bash
cmake -S . -B build-release
cmake --build build-release -j
cd go && go test ./...
```

## Performance Regression Guard

CI runs:
- `bench_gemm` with warmup + median timing
- comparison against `bench/regression_baseline.json`
- fails on >5% slowdown

To disable AMX path at runtime, use hardware without AMX support (dispatch auto-falls back).
