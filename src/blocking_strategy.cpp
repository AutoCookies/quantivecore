#include "quantcore/blocking.hpp"

#include "quantcore/binary_gemm.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <vector>

namespace quantcore {

#ifndef QUANTCORE_BLOCK_MB
#define QUANTCORE_BLOCK_MB 64
#endif
#ifndef QUANTCORE_BLOCK_NB
#define QUANTCORE_BLOCK_NB 64
#endif
#ifndef QUANTCORE_BLOCK_KB_BLOCKS
#define QUANTCORE_BLOCK_KB_BLOCKS 8
#endif

namespace {

std::mutex g_strategy_mu;
BlockingStrategy g_strategy{QUANTCORE_BLOCK_MB, QUANTCORE_BLOCK_NB, QUANTCORE_BLOCK_KB_BLOCKS};

double measure_avx512_ms(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, int warmup, int iters) {
    std::vector<std::int32_t> out;
    for (int i = 0; i < warmup; ++i) {
        binary_gemm_avx512(a, b, out, true);
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        binary_gemm_avx512(a, b, out, true);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = t1 - t0;
        samples.push_back(ms.count());
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

}  // namespace

BlockingStrategy default_blocking_strategy() {
    return BlockingStrategy{QUANTCORE_BLOCK_MB, QUANTCORE_BLOCK_NB, QUANTCORE_BLOCK_KB_BLOCKS};
}

BlockingStrategy current_blocking_strategy() {
    std::lock_guard<std::mutex> lock(g_strategy_mu);
    return g_strategy;
}

void set_blocking_strategy(BlockingStrategy strategy) {
    std::lock_guard<std::mutex> lock(g_strategy_mu);
    g_strategy = strategy;
}

void reset_blocking_strategy() {
    set_blocking_strategy(default_blocking_strategy());
}

BlockingStrategy autotune_blocking_binary(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, int warmup,
                                          int iterations) {
    if (!(avx512f_supported() && avx512vpopcntdq_supported())) {
        return current_blocking_strategy();
    }

    const std::vector<BlockingStrategy> candidates = {
        {32, 32, 4}, {64, 64, 8}, {128, 64, 8}, {64, 128, 8}, {128, 128, 16},
    };

    const BlockingStrategy original = current_blocking_strategy();
    BlockingStrategy best = original;
    double best_ms = std::numeric_limits<double>::max();

    for (const auto candidate : candidates) {
        set_blocking_strategy(candidate);
        const double score = measure_avx512_ms(a, b, warmup, iterations);
        if (score < best_ms) {
            best_ms = score;
            best = candidate;
        }
    }

    set_blocking_strategy(best);
    return best;
}

}  // namespace quantcore
