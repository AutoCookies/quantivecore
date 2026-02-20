#include "quantcore/blocking.hpp"

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

BlockingStrategy default_blocking_strategy() {
    return BlockingStrategy{QUANTCORE_BLOCK_MB, QUANTCORE_BLOCK_NB, QUANTCORE_BLOCK_KB_BLOCKS};
}

}  // namespace quantcore
