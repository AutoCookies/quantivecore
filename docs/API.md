# API Contract

## Stable C ABI (`c_api.h`)

```c
const char* qc_version(void);
void qc_binary_gemm(...);
void qc_ternary_gemm(...);
```

This ABI is frozen for 1.x.

## Go wrapper stability

`go/quantcore` provides stable 1.x functions:
- `Version() string`
- `BinaryGEMM(...)`
- `TernaryGEMM(...)`

No per-element cgo calls; one cgo call per GEMM op.
