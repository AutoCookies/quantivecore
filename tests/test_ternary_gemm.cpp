#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "quantcore/binary_gemm.hpp"
#include "quantcore/ternary_gemm.hpp"

namespace {

std::vector<std::int32_t> reference_ternary(const std::vector<std::int8_t>& a, const std::vector<std::int8_t>& b,
                                            std::size_t m, std::size_t n, std::size_t k) {
    std::vector<std::int32_t> out(m * n, 0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            std::int32_t acc = 0;
            for (std::size_t t = 0; t < k; ++t) {
                acc += a[i * k + t] * b[j * k + t];
            }
            out[i * n + j] = acc;
        }
    }
    return out;
}

}  // namespace

TEST_CASE("ternary gemm deterministic golden", "[ternary]") {
    const std::vector<std::int8_t> a = {1, 0, -1, 1, -1, 0, 0, 1, 1, -1, 0, -1};
    const std::vector<std::int8_t> b = {1, -1, 0, 1, 0, -1, 1, 0, -1, 1, -1, 0};

    const auto pa = quantcore::pack_ternary_matrix(a, 3, 4);
    const auto pb = quantcore::pack_ternary_matrix(b, 3, 4);

    std::vector<std::int32_t> c;
    quantcore::ternary_gemm_scalar(pa, pb, c);
    const auto ref = reference_ternary(a, b, 3, 3, 4);
    REQUIRE(c == ref);
}

TEST_CASE("ternary avx2 and threaded match scalar", "[ternary][avx2]") {
    constexpr std::size_t m = 11;
    constexpr std::size_t n = 7;
    constexpr std::size_t k = 193;

    std::vector<std::int8_t> a(m * k, 0);
    std::vector<std::int8_t> b(n * k, 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = (i % 3 == 0) ? -1 : ((i % 3 == 1) ? 0 : 1);
    }
    for (std::size_t i = 0; i < b.size(); ++i) {
        b[i] = (i % 4 == 0) ? 1 : ((i % 4 == 1) ? -1 : 0);
    }

    const auto pa = quantcore::pack_ternary_matrix(a, m, k);
    const auto pb = quantcore::pack_ternary_matrix(b, n, k);

    std::vector<std::int32_t> scalar;
    quantcore::ternary_gemm_scalar(pa, pb, scalar);

    std::vector<std::int32_t> dispatched;
    quantcore::ternary_gemm(pa, pb, dispatched, true);
    REQUIRE(dispatched == scalar);

    if (quantcore::avx2_supported()) {
        std::vector<std::int32_t> avx2;
        quantcore::ternary_gemm_avx2(pa, pb, avx2);
        REQUIRE(avx2 == scalar);
    }
}
