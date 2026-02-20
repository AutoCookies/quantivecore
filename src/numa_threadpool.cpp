#include <algorithm>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace quantcore {

void parallel_for_rows_numa(std::size_t rows, bool use_threads,
                            const std::function<void(std::size_t, std::size_t)>& fn) {
    const std::size_t hw = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t workers = use_threads ? std::min(rows, hw) : 1;
    if (workers <= 1 || rows < 2) {
        fn(0, rows);
        return;
    }

    const std::size_t chunk = (rows + workers - 1) / workers;
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (std::size_t tid = 0; tid < workers; ++tid) {
        const std::size_t begin = tid * chunk;
        const std::size_t end = std::min(rows, begin + chunk);
        if (begin >= end) {
            break;
        }
        pool.emplace_back([=, &fn]() {
#if defined(__linux__)
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(static_cast<int>(tid % hw), &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
            fn(begin, end);
        });
    }

    for (auto& th : pool) {
        th.join();
    }
}

}  // namespace quantcore
