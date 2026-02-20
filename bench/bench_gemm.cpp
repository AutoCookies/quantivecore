#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "quantcore/binary_gemm.hpp"

namespace {

double median_ms(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double run_mode(const quantcore::PackedBinaryMatrix& a, const quantcore::PackedBinaryMatrix& b, int warmup, int iters,
                const std::string& mode) {
    std::vector<std::int32_t> out;
    auto run_once = [&]() {
        if (mode == "scalar") {
            quantcore::binary_gemm_scalar(a, b, out);
        } else if (mode == "avx2") {
            quantcore::binary_gemm_avx2(a, b, out);
        } else if (mode == "dispatch_st") {
            quantcore::binary_gemm(a, b, out, false);
        } else {
            quantcore::binary_gemm(a, b, out, true);
        }
    };

    for (int i = 0; i < warmup; ++i) {
        run_once();
    }

    std::vector<double> timings;
    timings.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        run_once();
        const auto t1 = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double, std::milli> ms = t1 - t0;
        timings.push_back(ms.count());
    }

    return median_ms(std::move(timings));
}

}  // namespace

int main(int argc, char** argv) {
    const std::size_t m = 2048;
    const std::size_t n = 2048;
    const std::size_t k = 2048;

    int warmup = 2;
    int iters = 5;
    std::string json_output;
    if (argc > 1) {
        json_output = argv[1];
    }

    std::vector<std::int8_t> a(m * k, 1);
    std::vector<std::int8_t> b(n * k, -1);
    for (std::size_t i = 0; i < a.size(); i += 11) {
        a[i] = -1;
    }
    for (std::size_t i = 0; i < b.size(); i += 7) {
        b[i] = 1;
    }

    const auto pa = quantcore::pack_binary_matrix(a, m, k);
    const auto pb = quantcore::pack_binary_matrix(b, n, k);

    const double scalar_ms = run_mode(pa, pb, warmup, iters, "scalar");
    const double avx2_ms = quantcore::avx2_supported() ? run_mode(pa, pb, warmup, iters, "avx2") : scalar_ms;
    const double dispatch_st_ms = run_mode(pa, pb, warmup, iters, "dispatch_st");
    const double dispatch_mt_ms = run_mode(pa, pb, warmup, iters, "dispatch_mt");

    const double ops = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double avx2_gops = ops / (avx2_ms * 1e6);

    std::cout << "scalar(ms): " << scalar_ms << '\n';
    std::cout << "avx2(ms): " << avx2_ms << '\n';
    std::cout << "dispatch_st(ms): " << dispatch_st_ms << '\n';
    std::cout << "dispatch_mt(ms): " << dispatch_mt_ms << '\n';
    std::cout << "avx2 GOPS-eq: " << avx2_gops << '\n';

    if (!json_output.empty()) {
        std::ofstream out(json_output);
        out << "{\n"
            << "  \"binary_2048\": {\n"
            << "    \"scalar_ms\": " << scalar_ms << ",\n"
            << "    \"avx2_ms\": " << avx2_ms << ",\n"
            << "    \"dispatch_st_ms\": " << dispatch_st_ms << ",\n"
            << "    \"dispatch_mt_ms\": " << dispatch_mt_ms << "\n"
            << "  }\n"
            << "}\n";
    }

    return 0;
}
