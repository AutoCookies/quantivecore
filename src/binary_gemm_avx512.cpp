#include "quantcore/binary_gemm.hpp"
#include "quantcore/blocking.hpp"

#include <algorithm>
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

__attribute__((target("avx512f,avx512vpopcntdq")))
std::int32_t dot_binary_vpopcnt(const std::uint64_t* lhs, const std::uint64_t* rhs, std::size_t blocks, std::size_t k) {
    std::size_t equal = 0;
    std::size_t b = 0;
    for (; b + 8 <= blocks; b += 8) {
        alignas(64) std::uint64_t masks[8];
        for (std::size_t lane = 0; lane < 8; ++lane) {
            masks[lane] = valid_mask(b + lane, k);
        }

        const auto va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(lhs + b));
        const auto vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(rhs + b));
        const auto vm = _mm512_load_si512(reinterpret_cast<const __m512i*>(masks));
        const auto veq = _mm512_and_si512(_mm512_xor_si512(_mm512_xor_si512(va, vb), _mm512_set1_epi64(-1)), vm);
        const auto pc = _mm512_popcnt_epi64(veq);
        alignas(64) std::uint64_t lane_pc[8];
        _mm512_store_si512(reinterpret_cast<__m512i*>(lane_pc), pc);
        for (std::size_t lane = 0; lane < 8; ++lane) {
            equal += lane_pc[lane];
        }
    }
    for (; b < blocks; ++b) {
        equal += std::popcount((~(lhs[b] ^ rhs[b])) & valid_mask(b, k));
    }
    return static_cast<std::int32_t>(2 * equal) - static_cast<std::int32_t>(k);
}

}  // namespace

void binary_gemm_avx512(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c,
                        bool blocked) {
    c.assign(a.rows * b.rows, 0);
    const BlockingStrategy strategy = default_blocking_strategy();
    const std::size_t mb = blocked ? strategy.mb : a.rows;
    const std::size_t nb = blocked ? strategy.nb : b.rows;

    for (std::size_t i0 = 0; i0 < a.rows; i0 += mb) {
        const std::size_t i_max = std::min(a.rows, i0 + mb);
        for (std::size_t j0 = 0; j0 < b.rows; j0 += nb) {
            const std::size_t j_max = std::min(b.rows, j0 + nb);
            for (std::size_t i = i0; i < i_max; ++i) {
                const std::uint64_t* a_row = &a.data[i * a.blocks_per_row];
                for (std::size_t j = j0; j < j_max; ++j) {
                    const std::uint64_t* b_row = &b.data[j * b.blocks_per_row];
                    c[i * b.rows + j] = dot_binary_vpopcnt(a_row, b_row, a.blocks_per_row, a.cols);
                }
            }
        }
    }
}

}  // namespace quantcore
