#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "quantcore/bitpack.hpp"

namespace quantcore {

void binary_gemm_scalar(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c);
void binary_gemm_avx2(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c);
void binary_gemm_avx512(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c,
                        bool blocked);
void binary_gemm_amx(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c);
void binary_gemm(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c,
                 bool use_threads = false);

void binary_gemv_scalar(const PackedBinaryMatrix& a, const std::vector<std::uint64_t>& x, std::size_t x_cols,
                        std::vector<std::int32_t>& y);

bool avx2_supported();
bool avx512f_supported();
bool avx512vpopcntdq_supported();
bool amx_tile_supported();

}  // namespace quantcore
