#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(QUANTCORE_BUILD_SHARED)
    #define QC_API __declspec(dllexport)
  #elif defined(QUANTCORE_USE_SHARED)
    #define QC_API __declspec(dllimport)
  #else
    #define QC_API
  #endif
#else
  #define QC_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define QC_VERSION_MAJOR 1
#define QC_VERSION_MINOR 0
#define QC_VERSION_PATCH 0

QC_API const char* qc_version(void);

QC_API void qc_binary_gemm(size_t m, size_t n, size_t k, const uint64_t* a_data,
                           const uint64_t* b_data, int32_t* c_data);

QC_API void qc_ternary_gemm(size_t m, size_t n, size_t k, const uint64_t* a_pos, const uint64_t* a_neg,
                            const uint64_t* b_pos, const uint64_t* b_neg, int32_t* c_data);

#ifdef __cplusplus
}
#endif
