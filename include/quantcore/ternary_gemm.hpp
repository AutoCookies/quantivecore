#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quantcore/bitpack.hpp"

namespace quantcore {

void ternary_gemm_scalar(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c);
void ternary_gemm_avx2(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c);
void ternary_gemm(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c,
                  bool use_threads = false);

}  // namespace quantcore

extern "C" {

void qc_ternary_gemm(std::size_t m, std::size_t n, std::size_t k, const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                     const std::uint64_t* b_pos, const std::uint64_t* b_neg, std::int32_t* c_data);

}
