#pragma once

#include <cstddef>

#include "quantcore/bitpack.hpp"

namespace quantcore {

struct BlockingStrategy {
    std::size_t mb;
    std::size_t nb;
    std::size_t kb_blocks;
};

BlockingStrategy default_blocking_strategy();
BlockingStrategy current_blocking_strategy();
void set_blocking_strategy(BlockingStrategy strategy);
void reset_blocking_strategy();

BlockingStrategy autotune_blocking_binary(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, int warmup,
                                          int iterations);

}  // namespace quantcore
