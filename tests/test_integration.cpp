#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <vector>

#include "quantcore/binary_gemm.hpp"
#include "quantcore/blocking.hpp"

TEST_CASE("integration stress deterministic dispatch", "[integration]") {
    const bool heavy = std::getenv("QC_HEAVY_INTEGRATION") != nullptr;
    const std::size_t dim = heavy ? 4096 : 256;
    const int loops = heavy ? 10000 : 8;

    std::vector<std::int8_t> a(dim * dim, 1);
    std::vector<std::int8_t> b(dim * dim, -1);
    for (std::size_t i = 0; i < a.size(); i += 17) {
        a[i] = -1;
    }
    for (std::size_t i = 0; i < b.size(); i += 19) {
        b[i] = 1;
    }

    const auto pa = quantcore::pack_binary_matrix(a, dim, dim);
    const auto pb = quantcore::pack_binary_matrix(b, dim, dim);

    quantcore::autotune_blocking_binary(pa, pb, 1, 1);

    std::vector<std::int32_t> out;
    for (int i = 0; i < loops; ++i) {
        quantcore::binary_gemm(pa, pb, out, true);
    }

    REQUIRE(out.size() == dim * dim);
}
