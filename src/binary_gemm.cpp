#include "quantcore/binary_gemm.hpp"

#include <bit>
#include <stdexcept>

namespace quantcore {

namespace {

[[nodiscard]] std::uint64_t valid_mask(std::size_t block, std::size_t cols) {
    const std::size_t start = block * 64U;
    if (start + 64U <= cols) {
        return ~std::uint64_t{0};
    }
    const std::size_t remainder = cols - start;
    if (remainder == 0) {
        return 0;
    }
    return (std::uint64_t{1} << remainder) - 1U;
}

[[nodiscard]] std::int32_t dot_binary_blocks(const std::uint64_t* lhs, const std::uint64_t* rhs, std::size_t blocks,
                                             std::size_t k) {
    std::size_t equal = 0;
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto mask = valid_mask(b, k);
        const auto same = ~(lhs[b] ^ rhs[b]) & mask;
        equal += std::popcount(same);
    }
    return static_cast<std::int32_t>(2 * equal) - static_cast<std::int32_t>(k);
}

}  // namespace

void binary_gemm_scalar(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c) {
    if (a.cols != b.cols) {
        throw std::invalid_argument("binary_gemm: k dimension mismatch");
    }
    if (a.blocks_per_row != b.blocks_per_row) {
        throw std::invalid_argument("binary_gemm: packed block mismatch");
    }

    c.assign(a.rows * b.rows, 0);
    for (std::size_t i = 0; i < a.rows; ++i) {
        const std::uint64_t* a_row = &a.data[i * a.blocks_per_row];
        for (std::size_t j = 0; j < b.rows; ++j) {
            const std::uint64_t* b_row = &b.data[j * b.blocks_per_row];
            c[i * b.rows + j] = dot_binary_blocks(a_row, b_row, a.blocks_per_row, a.cols);
        }
    }
}

void binary_gemv_scalar(const PackedBinaryMatrix& a, const std::vector<std::uint64_t>& x, std::size_t x_cols,
                        std::vector<std::int32_t>& y) {
    if (a.cols != x_cols) {
        throw std::invalid_argument("binary_gemv: k dimension mismatch");
    }
    const auto x_blocks = blocks_for_cols(x_cols);
    if (x.size() != x_blocks) {
        throw std::invalid_argument("binary_gemv: packed vector size mismatch");
    }

    y.assign(a.rows, 0);
    for (std::size_t i = 0; i < a.rows; ++i) {
        const std::uint64_t* a_row = &a.data[i * a.blocks_per_row];
        y[i] = dot_binary_blocks(a_row, x.data(), a.blocks_per_row, a.cols);
    }
}

}  // namespace quantcore
