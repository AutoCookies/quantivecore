#include <chrono>
#include <cstdint>
#include <iostream>
#include <vector>

#include "quantcore/binary_gemm.hpp"

int main() {
    constexpr std::size_t m = 256;
    constexpr std::size_t n = 256;
    constexpr std::size_t k = 256;

    std::vector<std::int8_t> a(m * k, 1);
    std::vector<std::int8_t> b(n * k, -1);

    const auto pa = quantcore::pack_binary_matrix(a, m, k);
    const auto pb = quantcore::pack_binary_matrix(b, n, k);

    auto run = [&](const char* name, bool threads) {
        std::vector<std::int32_t> c;
        const auto t0 = std::chrono::high_resolution_clock::now();
        quantcore::binary_gemm(pa, pb, c, threads);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = t1 - t0;

        const double ops = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
        const double gops = ops / (ms.count() * 1e6);
        const double throughput = static_cast<double>(m) * static_cast<double>(n) / ms.count();

        std::cout << name << ": " << ms.count() << " ms, " << gops << " GOPS-eq, throughput=" << throughput
                  << " outputs/ms, batch=1" << '\n';
    };

    run("dispatch(single-thread)", false);
    run("dispatch(multi-thread)", true);

    if (quantcore::avx2_supported()) {
        std::vector<std::int32_t> c;
        const auto t0 = std::chrono::high_resolution_clock::now();
        quantcore::binary_gemm_avx2(pa, pb, c);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = t1 - t0;
        const double ops = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
        const double gops = ops / (ms.count() * 1e6);
        std::cout << "avx2(single-thread): " << ms.count() << " ms, " << gops << " GOPS-eq" << '\n';
    }

    return 0;
}
