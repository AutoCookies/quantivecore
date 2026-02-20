#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quantcore/bitpack.hpp"

namespace quantcore {

void ternary_gemm_scalar(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c);
void ternary_gemm_avx2(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c);
void ternary_gemm_avx512(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c,
                         bool blocked);
void ternary_gemm(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c,
                  bool use_threads = false);

}  // namespace quantcore
