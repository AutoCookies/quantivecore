#include "quantcore/binary_gemm.hpp"

namespace quantcore {

void binary_gemm_amx(const PackedBinaryMatrix& a, const PackedBinaryMatrix& b, std::vector<std::int32_t>& c) {
    // Experimental AMX path placeholder: preserve correctness by delegating to best available non-AMX kernel.
    if (avx512f_supported() && avx512vpopcntdq_supported()) {
        binary_gemm_avx512(a, b, c, true);
        return;
    }
    if (avx2_supported()) {
        binary_gemm_avx2(a, b, c);
        return;
    }
    binary_gemm_scalar(a, b, c);
}

}  // namespace quantcore
