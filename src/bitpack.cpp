#include "quantcore/bitpack.hpp"

#include <stdexcept>

namespace quantcore {

namespace {

[[nodiscard]] std::size_t linear_index(std::size_t row, std::size_t col, std::size_t cols) {
    return row * cols + col;
}

}  // namespace

std::size_t blocks_for_cols(std::size_t cols) {
    return (cols + 63U) / 64U;
}

PackedBinaryMatrix pack_binary_matrix(const std::vector<std::int8_t>& values, std::size_t rows, std::size_t cols) {
    if (values.size() != rows * cols) {
        throw std::invalid_argument("binary input size mismatch");
    }

    PackedBinaryMatrix packed;
    packed.rows = rows;
    packed.cols = cols;
    packed.blocks_per_row = blocks_for_cols(cols);
    packed.data.assign(rows * packed.blocks_per_row, 0);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t block_idx = c / 64U;
            const std::size_t bit_idx = c % 64U;
            const std::size_t packed_idx = r * packed.blocks_per_row + block_idx;
            if (values[linear_index(r, c, cols)] > 0) {
                packed.data[packed_idx] |= (std::uint64_t{1} << bit_idx);
            }
        }
    }

    return packed;
}

std::vector<std::int8_t> unpack_binary_matrix(const PackedBinaryMatrix& packed) {
    std::vector<std::int8_t> out(packed.rows * packed.cols, -1);

    for (std::size_t r = 0; r < packed.rows; ++r) {
        for (std::size_t c = 0; c < packed.cols; ++c) {
            const std::size_t block_idx = c / 64U;
            const std::size_t bit_idx = c % 64U;
            const std::size_t packed_idx = r * packed.blocks_per_row + block_idx;
            const bool bit = (packed.data[packed_idx] >> bit_idx) & std::uint64_t{1};
            out[linear_index(r, c, packed.cols)] = bit ? 1 : -1;
        }
    }

    return out;
}

PackedTernaryMatrix pack_ternary_matrix(const std::vector<std::int8_t>& values, std::size_t rows, std::size_t cols) {
    if (values.size() != rows * cols) {
        throw std::invalid_argument("ternary input size mismatch");
    }

    PackedTernaryMatrix packed;
    packed.rows = rows;
    packed.cols = cols;
    packed.blocks_per_row = blocks_for_cols(cols);
    packed.positive.assign(rows * packed.blocks_per_row, 0);
    packed.negative.assign(rows * packed.blocks_per_row, 0);

    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            const std::size_t block_idx = c / 64U;
            const std::size_t bit_idx = c % 64U;
            const std::size_t packed_idx = r * packed.blocks_per_row + block_idx;
            const auto v = values[linear_index(r, c, cols)];
            if (v > 0) {
                packed.positive[packed_idx] |= (std::uint64_t{1} << bit_idx);
            } else if (v < 0) {
                packed.negative[packed_idx] |= (std::uint64_t{1} << bit_idx);
            }
        }
    }

    return packed;
}

std::vector<std::int8_t> unpack_ternary_matrix(const PackedTernaryMatrix& packed) {
    std::vector<std::int8_t> out(packed.rows * packed.cols, 0);

    for (std::size_t r = 0; r < packed.rows; ++r) {
        for (std::size_t c = 0; c < packed.cols; ++c) {
            const std::size_t block_idx = c / 64U;
            const std::size_t bit_idx = c % 64U;
            const std::size_t packed_idx = r * packed.blocks_per_row + block_idx;
            const bool pos = (packed.positive[packed_idx] >> bit_idx) & std::uint64_t{1};
            const bool neg = (packed.negative[packed_idx] >> bit_idx) & std::uint64_t{1};
            out[linear_index(r, c, packed.cols)] = pos ? 1 : (neg ? -1 : 0);
        }
    }

    return out;
}

}  // namespace quantcore
