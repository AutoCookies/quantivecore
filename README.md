# QuantCore

QuantCore is a bit-level GEMM/GEMV engine for binary (1-bit) and ternary (2-plane) math.

## Phase 4 Highlights

- Runtime auto-tuner for cache blocking (`MB/NB/KB_blocks`) with deterministic median timing search.
- Runtime micro-architecture profile capture in benchmark (`/proc/cpuinfo` model reporting).
- Roofline-oriented reporting with arithmetic intensity + measured memory bandwidth probe.
- Hardened dispatch path with explicit dimension validation and safe ISA fallback chain.
- CI performance regression gate with latency ceilings and dominance checks.

## ISA Matrix

| ISA | Status |
| --- | --- |
| Scalar | ✅ reference |
| AVX2 | ✅ |
| AVX-512F | ✅ |
| AVX-512VPOPCNTDQ | ✅ when available |
| AMX tile | 🧪 optional path |

## Dispatch order (binary)

AMX -> AVX-512 (blocked, tuned) -> AVX2 -> Scalar.

## Tuning API

`include/quantcore/blocking.hpp` exposes:
- `current_blocking_strategy()`
- `set_blocking_strategy(...)`
- `reset_blocking_strategy()`
- `autotune_blocking_binary(...)`

## Benchmark + Roofline

```bash
cmake -S . -B build-release
cmake --build build-release -j
./build-release/bench_gemm bench/current_bench.json
python3 bench/check_regression.py bench/regression_baseline.json bench/current_bench.json
```

Benchmark outputs latency metrics, tuned tile sizes, memory bandwidth estimate, and memory-bound roofline throughput estimate.
