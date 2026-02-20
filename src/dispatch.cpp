#include "quantcore/binary_gemm.hpp"
#include "quantcore/ternary_gemm.hpp"

#include <algorithm>
#include <thread>

namespace quantcore {

bool avx2_supported() {
#if defined(__x86_64__) || defined(_M_X64)
    return __builtin_cpu_supports("avx2") != 0;
#else
    return false;
#endif
}

namespace {

template <typename Fn>
void parallel_rows(std::size_t rows, bool use_threads, Fn&& fn) {
    const std::size_t hw = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t workers = use_threads ? std::min(rows, hw) : 1;

    if (workers <= 1 || rows < 2) {
        fn(0, rows);
        return;
    }

    std::vector<std::thread> pool;
    pool.reserve(workers);

    const std::size_t chunk = (rows + workers - 1) / workers;
    for (std::size_t t = 0; t < workers; ++t) {
        const std::size_t begin = t * chunk;
        const std::size_t end = std::min(rows, begin + chunk);
        if (begin >= end) {
            break;
        }
        pool.emplace_back([&, begin, end]() { fn(begin, end); });
    }

    for (auto& th : pool) {
        th.join();
    }
}

}  // namespace

void binary_gemm(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c, bool use_threads) {
    c.assign(a.rows * b.rows, 0);

    parallel_rows(a.rows, use_threads, [&](std::size_t start, std::size_t finish) {
        std::vector<std::int32_t> row_out;
        for (std::size_t i = start; i < finish; ++i) {
            PackedBinaryMatrix one_row{1, a.cols, a.blocks_per_row,
                                       std::vector<std::uint64_t>(a.data.begin() + static_cast<std::ptrdiff_t>(i * a.blocks_per_row),
                                                                  a.data.begin() + static_cast<std::ptrdiff_t>((i + 1) * a.blocks_per_row))};
            if (avx2_supported()) {
                binary_gemm_avx2(one_row, b, row_out);
            } else {
                binary_gemm_scalar(one_row, b, row_out);
            }
            std::copy(row_out.begin(), row_out.end(), c.begin() + static_cast<std::ptrdiff_t>(i * b.rows));
        }
    });
}

void ternary_gemm(const PackedTernaryMatrix& a, const PackedTernaryMatrix& b, std::vector<std::int32_t>& c,
                  bool use_threads) {
    c.assign(a.rows * b.rows, 0);

    parallel_rows(a.rows, use_threads, [&](std::size_t start, std::size_t finish) {
        std::vector<std::int32_t> row_out;
        for (std::size_t i = start; i < finish; ++i) {
            PackedTernaryMatrix one_row;
            one_row.rows = 1;
            one_row.cols = a.cols;
            one_row.blocks_per_row = a.blocks_per_row;
            one_row.positive = std::vector<std::uint64_t>(a.positive.begin() + static_cast<std::ptrdiff_t>(i * a.blocks_per_row),
                                                          a.positive.begin() + static_cast<std::ptrdiff_t>((i + 1) * a.blocks_per_row));
            one_row.negative = std::vector<std::uint64_t>(a.negative.begin() + static_cast<std::ptrdiff_t>(i * a.blocks_per_row),
                                                          a.negative.begin() + static_cast<std::ptrdiff_t>((i + 1) * a.blocks_per_row));
            if (avx2_supported()) {
                ternary_gemm_avx2(one_row, b, row_out);
            } else {
                ternary_gemm_scalar(one_row, b, row_out);
            }
            std::copy(row_out.begin(), row_out.end(), c.begin() + static_cast<std::ptrdiff_t>(i * b.rows));
        }
    });
}

}  // namespace quantcore

extern "C" {

void qc_binary_gemm(std::size_t m, std::size_t n, std::size_t k, const std::uint64_t* a_data,
                    const std::uint64_t* b_data, std::int32_t* c_data) {
    const std::size_t blocks = quantcore::blocks_for_cols(k);
    quantcore::PackedBinaryMatrix a{m, k, blocks, std::vector<std::uint64_t>(a_data, a_data + m * blocks)};
    quantcore::PackedBinaryMatrix b{n, k, blocks, std::vector<std::uint64_t>(b_data, b_data + n * blocks)};
    std::vector<std::int32_t> c;
    quantcore::binary_gemm(a, b, c, false);
    std::copy(c.begin(), c.end(), c_data);
}

void qc_ternary_gemm(std::size_t m, std::size_t n, std::size_t k, const std::uint64_t* a_pos, const std::uint64_t* a_neg,
                     const std::uint64_t* b_pos, const std::uint64_t* b_neg, std::int32_t* c_data) {
    const std::size_t blocks = quantcore::blocks_for_cols(k);
    quantcore::PackedTernaryMatrix a{m, k, blocks, std::vector<std::uint64_t>(a_pos, a_pos + m * blocks),
                                     std::vector<std::uint64_t>(a_neg, a_neg + m * blocks)};
    quantcore::PackedTernaryMatrix b{n, k, blocks, std::vector<std::uint64_t>(b_pos, b_pos + n * blocks),
                                     std::vector<std::uint64_t>(b_neg, b_neg + n * blocks)};
    std::vector<std::int32_t> c;
    quantcore::ternary_gemm(a, b, c, false);
    std::copy(c.begin(), c.end(), c_data);
}

}
