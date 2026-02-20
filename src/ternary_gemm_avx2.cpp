#include "quantcore/ternary_gemm.hpp"

#include <array>
#include <bit>
#include <immintrin.h>

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

__attribute__((target("avx2,popcnt")))
[[nodiscard]] std::int32_t dot_ternary_avx2(const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                                            const std::uint64_t* b_pos, const std::uint64_t* b_neg,
                                            std::size_t blocks, std::size_t k) {
    std::int32_t acc = 0;
    std::size_t blk = 0;
    alignas(32) std::array<std::uint64_t, 4> pp{};
    alignas(32) std::array<std::uint64_t, 4> nn{};
    alignas(32) std::array<std::uint64_t, 4> pn{};
    alignas(32) std::array<std::uint64_t, 4> np{};

    for (; blk + 4 <= blocks; blk += 4) {
        const auto ap = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_pos + blk));
        const auto an = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a_neg + blk));
        const auto bp = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_pos + blk));
        const auto bn = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b_neg + blk));

        _mm256_store_si256(reinterpret_cast<__m256i*>(pp.data()), _mm256_and_si256(ap, bp));
        _mm256_store_si256(reinterpret_cast<__m256i*>(nn.data()), _mm256_and_si256(an, bn));
        _mm256_store_si256(reinterpret_cast<__m256i*>(pn.data()), _mm256_and_si256(ap, bn));
        _mm256_store_si256(reinterpret_cast<__m256i*>(np.data()), _mm256_and_si256(an, bp));

        for (std::size_t lane = 0; lane < 4; ++lane) {
            const auto mask = valid_mask(blk + lane, k);
            acc += static_cast<std::int32_t>(std::popcount(pp[lane] & mask));
            acc += static_cast<std::int32_t>(std::popcount(nn[lane] & mask));
            acc -= static_cast<std::int32_t>(std::popcount(pn[lane] & mask));
            acc -= static_cast<std::int32_t>(std::popcount(np[lane] & mask));
        }
    }

    for (; blk < blocks; ++blk) {
        const auto mask = valid_mask(blk, k);
        acc += static_cast<std::int32_t>(std::popcount((a_pos[blk] & b_pos[blk]) & mask));
        acc += static_cast<std::int32_t>(std::popcount((a_neg[blk] & b_neg[blk]) & mask));
        acc -= static_cast<std::int32_t>(std::popcount((a_pos[blk] & b_neg[blk]) & mask));
        acc -= static_cast<std::int32_t>(std::popcount((a_neg[blk] & b_pos[blk]) & mask));
    }

    return acc;
}

}  // namespace

void ternary_gemm_avx2(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c) {
    c.assign(a.rows * b.rows, 0);
    for (std::size_t i = 0; i < a.rows; ++i) {
        const std::uint64_t* a_pos = &a.positive[i * a.blocks_per_row];
        const std::uint64_t* a_neg = &a.negative[i * a.blocks_per_row];
        for (std::size_t j = 0; j < b.rows; ++j) {
            const std::uint64_t* b_pos = &b.positive[j * b.blocks_per_row];
            const std::uint64_t* b_neg = &b.negative[j * b.blocks_per_row];
            c[i * b.rows + j] = dot_ternary_avx2(a_pos, a_neg, b_pos, b_neg, a.blocks_per_row, a.cols);
        }
    }
}

}  // namespace quantcore
