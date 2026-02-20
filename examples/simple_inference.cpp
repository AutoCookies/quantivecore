#include <cstdint>
#include <iostream>
#include <vector>

#include "quantcore/c_api.h"

int main() {
    std::vector<std::uint64_t> a = {0b1010};
    std::vector<std::uint64_t> b = {0b1100};
    std::vector<std::int32_t> c(1);
    qc_binary_gemm(1, 1, 4, a.data(), b.data(), c.data());
    std::cout << "QuantCore " << qc_version() << " result=" << c[0] << "\n";
    return 0;
}
