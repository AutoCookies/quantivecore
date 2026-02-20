#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quantcore/bitpack.hpp"

namespace quantcore {

void binary_gemm_scalar(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c);
void binary_gemm_avx2(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c);
void binary_gemm(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c, bool use_threads = false);

void binary_gemv_scalar(const PackedBinaryMatrix& a, const std::vector<std::uint64_t>& x, std::size_t x_cols,
                        std::vector<std::int32_t>& y);

bool avx2_supported();

}  // namespace quantcore

extern "C" {

void qc_binary_gemm(std::size_t m, std::size_t n, std::size_t k, const std::uint64_t* a_data,
                    const std::uint64_t* b_data, std::int32_t* c_data);

}
