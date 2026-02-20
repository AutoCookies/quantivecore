#include <catch2/catch_test_macros.hpp>

#include <random>
#include <vector>

#include "quantcore/binary_gemm.hpp"
#include "quantcore/ternary_gemm.hpp"

namespace {

std::vector<std::int32_t> reference(const std::vector<std::int8_t>& a, const std::vector<std::int8_t>& b, std::size_t m,
                                    std::size_t n, std::size_t k) {
    std::vector<std::int32_t> out(m * n, 0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            int acc = 0;
            for (std::size_t t = 0; t < k; ++t) {
                acc += static_cast<int>(a[i * k + t]) * static_cast<int>(b[j * k + t]);
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

}  // namespace

TEST_CASE("binary randomized property tests", "[property][binary]") {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dim_dist(8, 64);
    std::uniform_int_distribution<int> bit_dist(0, 1);

    for (int it = 0; it < 20; ++it) {
        const std::size_t m = static_cast<std::size_t>(dim_dist(rng));
        const std::size_t n = static_cast<std::size_t>(dim_dist(rng));
        const std::size_t k = static_cast<std::size_t>(dim_dist(rng) + (it % 3 == 0 ? 67 : 0));

        std::vector<std::int8_t> a(m * k, -1);
        std::vector<std::int8_t> b(n * k, -1);
        for (auto& v : a) {
            v = bit_dist(rng) ? 1 : -1;
        }
        for (auto& v : b) {
            v = bit_dist(rng) ? 1 : -1;
        }

        const auto pa = quantcore::pack_binary_matrix(a, m, k);
        const auto pb = quantcore::pack_binary_matrix(b, n, k);

        std::vector<std::int32_t> c;
        quantcore::binary_gemm_scalar(pa, pb, c);
        const auto ref = reference(a, b, m, n, k);
        REQUIRE(c == ref);
    }
}

TEST_CASE("ternary randomized property tests", "[property][ternary]") {
    std::mt19937 rng(4321);
    std::uniform_int_distribution<int> dim_dist(8, 64);
    std::uniform_int_distribution<int> ternary_dist(0, 2);

    for (int it = 0; it < 20; ++it) {
        const std::size_t m = static_cast<std::size_t>(dim_dist(rng));
        const std::size_t n = static_cast<std::size_t>(dim_dist(rng));
        const std::size_t k = static_cast<std::size_t>(dim_dist(rng) + (it % 4 == 0 ? 129 : 0));

        std::vector<std::int8_t> a(m * k, 0);
        std::vector<std::int8_t> b(n * k, 0);
        for (auto& v : a) {
            const int r = ternary_dist(rng);
            v = (r == 0) ? -1 : ((r == 1) ? 0 : 1);
        }
        for (auto& v : b) {
            const int r = ternary_dist(rng);
            v = (r == 0) ? -1 : ((r == 1) ? 0 : 1);
        }

        const auto pa = quantcore::pack_ternary_matrix(a, m, k);
        const auto pb = quantcore::pack_ternary_matrix(b, n, k);

        std::vector<std::int32_t> c;
        quantcore::ternary_gemm_scalar(pa, pb, c);
        const auto ref = reference(a, b, m, n, k);
        REQUIRE(c == ref);
    }
}
