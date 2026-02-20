#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "quantcore/binary_gemm.hpp"
#include "quantcore/blocking.hpp"

namespace {

double median_ms(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double probe_memory_bandwidth_gbps() {
    constexpr std::size_t bytes = 256ULL * 1024ULL * 1024ULL;
    std::vector<std::uint8_t> src(bytes, 3);
    std::vector<std::uint8_t> dst(bytes, 0);

    const auto t0 = std::chrono::high_resolution_clock::now();
    std::copy(src.begin(), src.end(), dst.begin());
    const auto t1 = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> sec = t1 - t0;
    const double gb = static_cast<double>(bytes) / 1e9;
    return gb / sec.count();
}

std::string cpu_model_name() {
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto pos = line.find(':');
            if (pos != std::string::npos) {
                return line.substr(pos + 2);
            }
        }
    }
    return "unknown";
}

double run_mode(const quantcore::PackedBinaryMatrix& a, const quantcore::PackedBinaryMatrix& b, int warmup, int iters,
                const std::string& mode) {
    std::vector<std::int32_t> out;
    auto run_once = [&]() {
        if (mode == "scalar") {
            quantcore::binary_gemm_scalar(a, b, out);
        } else if (mode == "avx2") {
            quantcore::binary_gemm_avx2(a, b, out);
        } else if (mode == "avx512_naive") {
            quantcore::binary_gemm_avx512(a, b, out, false);
        } else if (mode == "avx512_blocked") {
            quantcore::binary_gemm_avx512(a, b, out, true);
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

    const int warmup = 2;
    const int iters = 5;
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

    const auto tuned = quantcore::autotune_blocking_binary(pa, pb, 1, 2);
    const double scalar_ms = run_mode(pa, pb, warmup, iters, "scalar");
    const double avx2_ms = quantcore::avx2_supported() ? run_mode(pa, pb, warmup, iters, "avx2") : scalar_ms;

    double avx512_naive_ms = avx2_ms;
    double avx512_blocked_ms = avx2_ms;
    if (quantcore::avx512f_supported() && quantcore::avx512vpopcntdq_supported()) {
        avx512_naive_ms = run_mode(pa, pb, warmup, iters, "avx512_naive");
        avx512_blocked_ms = run_mode(pa, pb, warmup, iters, "avx512_blocked");
    }

    const double dispatch_st_ms = run_mode(pa, pb, warmup, iters, "dispatch_st");
    const double dispatch_mt_ms = run_mode(pa, pb, warmup, iters, "dispatch_mt");

    const double ops = static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
    const double ai = ops / (static_cast<double>(pa.data.size() + pb.data.size()) * 8.0 + static_cast<double>(m * n) * 4.0);
    const double mem_bw = probe_memory_bandwidth_gbps();
    const double roofline_memory_gops = ai * mem_bw;

    std::cout << "cpu_model: " << cpu_model_name() << '\n';
    std::cout << "autotuned blocks: MB=" << tuned.mb << " NB=" << tuned.nb << " KB_blocks=" << tuned.kb_blocks << '\n';
    std::cout << "scalar(ms): " << scalar_ms << '\n';
    std::cout << "avx2(ms): " << avx2_ms << '\n';
    std::cout << "avx512_naive(ms): " << avx512_naive_ms << '\n';
    std::cout << "avx512_blocked(ms): " << avx512_blocked_ms << '\n';
    std::cout << "dispatch_st(ms): " << dispatch_st_ms << '\n';
    std::cout << "dispatch_mt(ms): " << dispatch_mt_ms << '\n';
    std::cout << "memory_bw(GB/s): " << mem_bw << '\n';
    std::cout << "roofline_memory_bound(GOPS-eq): " << roofline_memory_gops << '\n';

    if (!json_output.empty()) {
        std::ofstream out(json_output);
        out << "{\n"
            << "  \"binary_2048\": {\n"
            << "    \"scalar_ms\": " << scalar_ms << ",\n"
            << "    \"avx2_ms\": " << avx2_ms << ",\n"
            << "    \"avx512_naive_ms\": " << avx512_naive_ms << ",\n"
            << "    \"avx512_blocked_ms\": " << avx512_blocked_ms << ",\n"
            << "    \"dispatch_st_ms\": " << dispatch_st_ms << ",\n"
            << "    \"dispatch_mt_ms\": " << dispatch_mt_ms << ",\n"
            << "    \"memory_bw_gbps\": " << mem_bw << ",\n"
            << "    \"roofline_memory_gops\": " << roofline_memory_gops << ",\n"
            << "    \"tuned_mb\": " << tuned.mb << ",\n"
            << "    \"tuned_nb\": " << tuned.nb << ",\n"
            << "    \"tuned_kb_blocks\": " << tuned.kb_blocks << "\n"
            << "  }\n"
            << "}\n";
    }

    return 0;
}
