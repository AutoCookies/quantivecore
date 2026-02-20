#include "quantcore/ternary_gemm.hpp"

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

[[nodiscard]] std::int32_t dot_ternary_blocks(const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                                              const std::uint64_t* b_pos, const std::uint64_t* b_neg,
                                              std::size_t blocks, std::size_t k) {
    std::int32_t acc = 0;
    for (std::size_t blk = 0; blk < blocks; ++blk) {
        const auto mask = valid_mask(blk, k);
        const auto app = a_pos[blk] & b_pos[blk] & mask;
        const auto ann = a_neg[blk] & b_neg[blk] & mask;
        const auto apn = a_pos[blk] & b_neg[blk] & mask;
        const auto anp = a_neg[blk] & b_pos[blk] & mask;

        acc += static_cast<std::int32_t>(std::popcount(app));
        acc += static_cast<std::int32_t>(std::popcount(ann));
        acc -= static_cast<std::int32_t>(std::popcount(apn));
        acc -= static_cast<std::int32_t>(std::popcount(anp));
    }
    return acc;
}

}  // namespace

void ternary_gemm_scalar(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c) {
    if (a.cols != b.cols) {
        throw std::invalid_argument("ternary_gemm: k dimension mismatch");
    }

    c.assign(a.rows * b.rows, 0);
    for (std::size_t i = 0; i < a.rows; ++i) {
        const std::uint64_t* a_pos = &a.positive[i * a.blocks_per_row];
        const std::uint64_t* a_neg = &a.negative[i * a.blocks_per_row];
        for (std::size_t j = 0; j < b.rows; ++j) {
            const std::uint64_t* b_pos = &b.positive[j * b.blocks_per_row];
            const std::uint64_t* b_neg = &b.negative[j * b.blocks_per_row];
            c[i * b.rows + j] = dot_ternary_blocks(a_pos, a_neg, b_pos, b_neg, a.blocks_per_row, a.cols);
        }
    }
}

}  // namespace quantcore
