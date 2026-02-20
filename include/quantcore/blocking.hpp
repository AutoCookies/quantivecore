#pragma once

#include <cstddef>

namespace quantcore {

struct BlockingStrategy {
    std::size_t mb;
    std::size_t nb;
    std::size_t kb_blocks;
};

BlockingStrategy default_blocking_strategy();

}  // namespace quantcore
