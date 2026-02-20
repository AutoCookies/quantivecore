#include "quantcore/binary_gemm.hpp"
#include "quantcore/blocking.hpp"
#include "quantcore/ternary_gemm.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <vector>

namespace quantcore {

void parallel_for_rows_numa(std::size_t rows, bool use_threads,
                            const std::function<void(std::size_t, std::size_t)>& fn);

bool avx2_supported() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}

bool avx512f_supported() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx512f") != 0;
#else
    return false;
#endif
}

bool avx512vpopcntdq_supported() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx512vpopcntdq") != 0;
#else
    return false;
#endif
}

bool amx_tile_supported() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("amx-tile") != 0;
#else
    return false;
#endif
}

void binary_gemm(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c, bool use_threads) {
    c.assign(a.rows * b.rows, 0);
    const bool use_avx512 = avx512f_supported() && avx512vpopcntdq_supported();

    parallel_for_rows_numa(a.rows, use_threads, [&](std::size_t start, std::size_t finish) {
        for (std::size_t i = start; i < finish; ++i) {
            PackedBinaryMatrix one_row{1,
                                       a.cols,
                                       a.blocks_per_row,
                                       std::vector<std::uint64_t>(a.data.begin() + static_cast<std::ptrdiff_t>(i * a.blocks_per_row),
                                                                  a.data.begin() + static_cast<std::ptrdiff_t>((i + 1) * a.blocks_per_row))};
            std::vector<std::int32_t> row_out;
            if (amx_tile_supported()) {
                binary_gemm_amx(one_row, b, row_out);
            } else if (use_avx512) {
                binary_gemm_avx512(one_row, b, row_out, true);
            } else if (avx2_supported()) {
                binary_gemm_avx2(one_row, b, row_out);
            } else {
                binary_gemm_scalar(one_row, b, row_out);
            }
            std::copy(row_out.begin(), row_out.end(), c.begin() + static_cast<std::ptrdiff_t>(i * b.rows));
        }
    });
}

void ternary_gemm(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c,
                  bool use_threads) {
    c.assign(a.rows * b.rows, 0);
    const bool use_avx512 = avx512f_supported() && avx512vpopcntdq_supported();

    parallel_for_rows_numa(a.rows, use_threads, [&](std::size_t start, std::size_t finish) {
        for (std::size_t i = start; i < finish; ++i) {
            PackedTernaryMatrix one_row;
            one_row.rows = 1;
            one_row.cols = a.cols;
            one_row.blocks_per_row = a.blocks_per_row;
            one_row.positive = std::vector<std::uint64_t>(a.positive.begin() + static_cast<std::ptrdiff_t>(i * a.blocks_per_row),
                                                          a.positive.begin() + static_cast<std::ptrdiff_t>((i + 1) * a.blocks_per_row));
            one_row.negative = std::vector<std::uint64_t>(a.negative.begin() + static_cast<std::ptrdiff_t>(i * a.blocks_per_row),
                                                          a.negative.begin() + static_cast<std::ptrdiff_t>((i + 1) * a.blocks_per_row));

            std::vector<std::int32_t> row_out;
            if (use_avx512) {
                ternary_gemm_avx512(one_row, b, row_out, true);
            } else if (avx2_supported()) {
                ternary_gemm_avx2(one_row, b, row_out);
            } else {
                ternary_gemm_scalar(one_row, b, row_out);
            }
            std::copy(row_out.begin(), row_out.end(), c.begin() + static_cast<std::ptrdiff_t>(i * b.rows));
        }
    });
}

}  // namespace quantcore

extern "C" {

void qc_binary_gemm(std::size_t m, std::size_t n, std::size_t k, const std::uint64_t* a_data,
                    const std::uint64_t* b_data, std::int32_t* c_data) {
    const std::size_t blocks = quantcore::blocks_for_cols(k);
    quantcore::PackedBinaryMatrix a{m, k, blocks, std::vector<std::uint64_t>(a_data, a_data + m * blocks)};
    quantcore::PackedBinaryMatrix b{n, k, blocks, std::vector<std::uint64_t>(b_data, b_data + n * blocks)};
    std::vector<std::int32_t> c;
    quantcore::binary_gemm(a, b, c, false);
    std::copy(c.begin(), c.end(), c_data);
}

void qc_ternary_gemm(std::size_t m, std::size_t n, std::size_t k, const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                     const std::uint64_t* b_pos, const std::uint64_t* b_neg, std::int32_t* c_data) {
    const std::size_t blocks = quantcore::blocks_for_cols(k);
    quantcore::PackedTernaryMatrix a{m, k, blocks, std::vector<std::uint64_t>(a_pos, a_pos + m * blocks),
                                     std::vector<std::uint64_t>(a_neg, a_neg + m * blocks)};
    quantcore::PackedTernaryMatrix b{n, k, blocks, std::vector<std::uint64_t>(b_pos, b_pos + n * blocks),
                                     std::vector<std::uint64_t>(b_neg, b_neg + n * blocks)};
    std::vector<std::int32_t> c;
    quantcore::ternary_gemm(a, b, c, false);
    std::copy(c.begin(), c.end(), c_data);
}

}
