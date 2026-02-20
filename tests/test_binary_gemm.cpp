#include <catch2/catch_test_macros.hpp>

#include <vector>

#include "quantcore/binary_gemm.hpp"
#include "quantcore/blocking.hpp"

namespace {

std::vector<std::int32_t> reference_binary(const std::vector<std::int8_t>& a, const std::vector<std::int8_t>& b,
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

TEST_CASE("binary gemm deterministic golden", "[binary]") {
    const std::vector<std::int8_t> a = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1};
    const std::vector<std::int8_t> b = {1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1};
    const auto pa = quantcore::pack_binary_matrix(a, 3, 4);
    const auto pb = quantcore::pack_binary_matrix(b, 3, 4);

    std::vector<std::int32_t> c;
    quantcore::binary_gemm_scalar(pa, pb, c);
    const auto ref = reference_binary(a, b, 3, 3, 4);

    REQUIRE(c == ref);
}

TEST_CASE("binary gemv deterministic", "[binary]") {
    const std::vector<std::int8_t> a = {1, -1, 1, -1, -1, -1, 1, 1};
    const auto pa = quantcore::pack_binary_matrix(a, 2, 4);

    const std::vector<std::int8_t> x = {1, 1, -1, -1};
    const auto px = quantcore::pack_binary_matrix(x, 1, 4);

    std::vector<std::int32_t> y;
    quantcore::binary_gemv_scalar(pa, px.data, 4, y);

    REQUIRE(y.size() == 2);
    REQUIRE(y[0] == 0);
    REQUIRE(y[1] == -4);
}

TEST_CASE("binary avx2 and threaded match scalar", "[binary][avx2]") {
    constexpr std::size_t m = 17;
    constexpr std::size_t n = 13;
    constexpr std::size_t k = 131;

    std::vector<std::int8_t> a(m * k, 1);
    std::vector<std::int8_t> b(n * k, -1);
    for (std::size_t i = 0; i < a.size(); i += 3) {
        a[i] = -1;
    }
    for (std::size_t i = 0; i < b.size(); i += 5) {
        b[i] = 1;
    }

    const auto pa = quantcore::pack_binary_matrix(a, m, k);
    const auto pb = quantcore::pack_binary_matrix(b, n, k);

    std::vector<std::int32_t> scalar;
    quantcore::binary_gemm_scalar(pa, pb, scalar);

    std::vector<std::int32_t> dispatched;
    quantcore::binary_gemm(pa, pb, dispatched, true);
    REQUIRE(dispatched == scalar);

    if (quantcore::avx2_supported()) {
        std::vector<std::int32_t> avx2;
        quantcore::binary_gemm_avx2(pa, pb, avx2);
        REQUIRE(avx2 == scalar);
    }
}


TEST_CASE("binary avx512 and amx match scalar", "[binary][avx512][amx]") {
    constexpr std::size_t m = 9;
    constexpr std::size_t n = 10;
    constexpr std::size_t k = 257;

    std::vector<std::int8_t> a(m * k, 1);
    std::vector<std::int8_t> b(n * k, -1);
    for (std::size_t i = 0; i < a.size(); i += 2) { a[i] = -1; }
    for (std::size_t i = 0; i < b.size(); i += 3) { b[i] = 1; }

    const auto pa = quantcore::pack_binary_matrix(a, m, k);
    const auto pb = quantcore::pack_binary_matrix(b, n, k);

    std::vector<std::int32_t> scalar;
    quantcore::binary_gemm_scalar(pa, pb, scalar);

    if (quantcore::avx512f_supported() && quantcore::avx512vpopcntdq_supported()) {
        std::vector<std::int32_t> naive;
        std::vector<std::int32_t> blocked;
        quantcore::binary_gemm_avx512(pa, pb, naive, false);
        quantcore::binary_gemm_avx512(pa, pb, blocked, true);
        REQUIRE(naive == scalar);
        REQUIRE(blocked == scalar);
    }

    if (quantcore::amx_tile_supported()) {
        std::vector<std::int32_t> amx;
        quantcore::binary_gemm_amx(pa, pb, amx);
        REQUIRE(amx == scalar);
    }
}


TEST_CASE("autotune updates strategy sanely", "[binary][tuning]") {
    constexpr std::size_t m = 8;
    constexpr std::size_t n = 8;
    constexpr std::size_t k = 128;
    std::vector<std::int8_t> a(m * k, 1);
    std::vector<std::int8_t> b(n * k, -1);

    const auto pa = quantcore::pack_binary_matrix(a, m, k);
    const auto pb = quantcore::pack_binary_matrix(b, n, k);

    const auto before = quantcore::current_blocking_strategy();
    const auto tuned = quantcore::autotune_blocking_binary(pa, pb, 1, 1);
    const auto after = quantcore::current_blocking_strategy();

    REQUIRE(after.mb == tuned.mb);
    REQUIRE(after.nb == tuned.nb);
    REQUIRE(after.kb_blocks == tuned.kb_blocks);

    quantcore::set_blocking_strategy(before);
}
