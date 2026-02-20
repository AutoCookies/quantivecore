#include "quantcore/binary_gemm.hpp"

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
[[nodiscard]] std::int32_t dot_binary_avx2(const std::uint64_t* lhs, const std::uint64_t* rhs, std::size_t blocks,
                                           std::size_t k) {
    std::size_t equal = 0;
    std::size_t b = 0;
    alignas(32) std::array<std::uint64_t, 4> lanes{};

    for (; b + 4 <= blocks; b += 4) {
        const auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lhs + b));
        const auto vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rhs + b));
        const auto vx = _mm256_xor_si256(va, vb);
        _mm256_store_si256(reinterpret_cast<__m256i*>(lanes.data()), vx);

        for (std::size_t lane = 0; lane < 4; ++lane) {
            const auto mask = valid_mask(b + lane, k);
            equal += std::popcount((~lanes[lane]) & mask);
        }
    }

    for (; b < blocks; ++b) {
        const auto mask = valid_mask(b, k);
        equal += std::popcount((~(lhs[b] ^ rhs[b])) & mask);
    }

    return static_cast<std::int32_t>(2 * equal) - static_cast<std::int32_t>(k);
}

}  // namespace

void binary_gemm_avx2(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c) {
    c.assign(a.rows * b.rows, 0);
    for (std::size_t i = 0; i < a.rows; ++i) {
        const std::uint64_t* a_row = &a.data[i * a.blocks_per_row];
        for (std::size_t j = 0; j < b.rows; ++j) {
            const std::uint64_t* b_row = &b.data[j * b.blocks_per_row];
            c[i * b.rows + j] = dot_binary_avx2(a_row, b_row, a.blocks_per_row, a.cols);
        }
    }
}

}  // namespace quantcore
