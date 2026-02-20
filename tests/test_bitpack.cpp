#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "quantcore/bitpack.hpp"

TEST_CASE("binary pack/unpack roundtrip", "[bitpack]") {
    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 70;
    const std::vector<std::int8_t> input = {
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        1, -1, 1, -1, 1, -1, 1, -1, 1, -1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
        -1, 1, -1, 1, -1, 1, -1, 1, -1, 1};

    const auto packed = quantcore::pack_binary_matrix(input, rows, cols);
    const auto unpacked = quantcore::unpack_binary_matrix(packed);

    REQUIRE(unpacked == input);
}

TEST_CASE("ternary pack/unpack roundtrip", "[bitpack]") {
    const std::vector<std::int8_t> input = {1, 0, -1, 1, -1, 0, 1, 0, -1, 1, -1, 0, 1, 0, -1, 1, -1, 0};
    const auto packed = quantcore::pack_ternary_matrix(input, 3, 6);
    const auto unpacked = quantcore::unpack_ternary_matrix(packed);

    REQUIRE(unpacked == input);
}
