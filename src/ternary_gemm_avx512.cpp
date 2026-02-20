#include "quantcore/ternary_gemm.hpp"
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
std::int32_t dot_ternary_vpopcnt(const std::uint64_t* a_pos, const std::uint64_t* a_neg, const std::uint64_t* b_pos,
                                 const std::uint64_t* b_neg, std::size_t blocks, std::size_t k) {
    std::int32_t acc = 0;
    std::size_t blk = 0;
    for (; blk + 8 <= blocks; blk += 8) {
        alignas(64) std::uint64_t masks[8];
        for (std::size_t lane = 0; lane < 8; ++lane) {
            masks[lane] = valid_mask(blk + lane, k);
        }
        const auto vm = _mm512_load_si512(reinterpret_cast<const __m512i*>(masks));

        const auto ap = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_pos + blk));
        const auto an = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(a_neg + blk));
        const auto bp = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_pos + blk));
        const auto bn = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(b_neg + blk));

        const auto pp = _mm512_popcnt_epi64(_mm512_and_si512(_mm512_and_si512(ap, bp), vm));
        const auto nn = _mm512_popcnt_epi64(_mm512_and_si512(_mm512_and_si512(an, bn), vm));
        const auto pn = _mm512_popcnt_epi64(_mm512_and_si512(_mm512_and_si512(ap, bn), vm));
        const auto np = _mm512_popcnt_epi64(_mm512_and_si512(_mm512_and_si512(an, bp), vm));

        alignas(64) std::uint64_t ppc[8], nnc[8], pnc[8], npc[8];
        _mm512_store_si512(reinterpret_cast<__m512i*>(ppc), pp);
        _mm512_store_si512(reinterpret_cast<__m512i*>(nnc), nn);
        _mm512_store_si512(reinterpret_cast<__m512i*>(pnc), pn);
        _mm512_store_si512(reinterpret_cast<__m512i*>(npc), np);

        for (std::size_t lane = 0; lane < 8; ++lane) {
            acc += static_cast<std::int32_t>(ppc[lane]);
            acc += static_cast<std::int32_t>(nnc[lane]);
            acc -= static_cast<std::int32_t>(pnc[lane]);
            acc -= static_cast<std::int32_t>(npc[lane]);
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

void ternary_gemm_avx512(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c,
                         bool blocked) {
    c.assign(a.rows * b.rows, 0);
    const BlockingStrategy strategy = current_blocking_strategy();
    const std::size_t mb = blocked ? strategy.mb : a.rows;
    const std::size_t nb = blocked ? strategy.nb : b.rows;

    for (std::size_t i0 = 0; i0 < a.rows; i0 += mb) {
        const std::size_t i_max = std::min(a.rows, i0 + mb);
        for (std::size_t j0 = 0; j0 < b.rows; j0 += nb) {
            const std::size_t j_max = std::min(b.rows, j0 + nb);
            for (std::size_t i = i0; i < i_max; ++i) {
                const std::uint64_t* a_pos = &a.positive[i * a.blocks_per_row];
                const std::uint64_t* a_neg = &a.negative[i * a.blocks_per_row];
                for (std::size_t j = j0; j < j_max; ++j) {
                    const std::uint64_t* b_pos = &b.positive[j * b.blocks_per_row];
                    const std::uint64_t* b_neg = &b.negative[j * b.blocks_per_row];
                    c[i * b.rows + j] = dot_ternary_vpopcnt(a_pos, a_neg, b_pos, b_neg, a.blocks_per_row, a.cols);
                }
            }
        }
    }
}

}  // namespace quantcore
