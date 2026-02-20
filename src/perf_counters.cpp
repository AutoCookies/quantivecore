#include <chrono>
#include <cstdint>
#include <functional>

namespace quantcore {

struct PerfSample {
    double elapsed_ms{};
    std::uint64_t cycles{};
    std::uint64_t instructions{};
    double ipc{};
};

PerfSample measure_wall_time_only(const std::function<void()>& fn) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    fn();
    const auto t1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> ms = t1 - t0;

    PerfSample s;
    s.elapsed_ms = ms.count();
    return s;
}

}  // namespace quantcore
