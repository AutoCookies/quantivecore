#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace quantcore {

struct PackedBinaryMatrix {
    std::size_t rows{};
    std::size_t cols{};
    std::size_t blocks_per_row{};
    std::vector<std::uint64_t> data;
};

struct PackedTernaryMatrix {
    std::size_t rows{};
    std::size_t cols{};
    std::size_t blocks_per_row{};
    std::vector<std::uint64_t> positive;
    std::vector<std::uint64_t> negative;
};

std::size_t blocks_for_cols(std::size_t cols);

PackedBinaryMatrix pack_binary_matrix(const std::vector<std::int8_t>& values, std::size_t rows, std::size_t cols);
std::vector<std::int8_t> unpack_binary_matrix(const PackedBinaryMatrix& packed);

PackedTernaryMatrix pack_ternary_matrix(const std::vector<std::int8_t>& values, std::size_t rows, std::size_t cols);
std::vector<std::int8_t> unpack_ternary_matrix(const PackedTernaryMatrix& packed);

}  // namespace quantcore
