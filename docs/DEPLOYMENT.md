# Deployment Guide

## ABI guarantee

For v1.x, only `include/quantcore/c_api.h` is stable.

## Thread-safety contract

- `qc_binary_gemm` / `qc_ternary_gemm`: thread-safe and reentrant.
- Autotune/blocking control (`set_blocking_strategy`, `autotune_blocking_binary`) uses internal synchronization.
- For deterministic serving, configure tuning once during init, then run inference.

## Deterministic mode

- Use fixed packed inputs and fixed threading policy.
- Avoid changing global blocking strategy concurrently.

## Memory alignment contract

- Packed matrices are expected as contiguous `uint64_t` arrays.
- C API accepts unaligned pointers; aligned buffers are recommended for best performance.
