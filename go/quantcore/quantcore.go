package quantcore

/*
#cgo CXXFLAGS: -std=c++20
#cgo CFLAGS: -I../../include
#cgo LDFLAGS: -L../../build-release -lquantcore -lstdc++
#include <stdint.h>
#include <stddef.h>

void qc_binary_gemm(size_t m, size_t n, size_t k, const uint64_t* a_data,
                    const uint64_t* b_data, int32_t* c_data);
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func BinaryGemm(m, n, k int, aPacked []uint64, bPacked []uint64) ([]int32, error) {
	blocks := (k + 63) / 64
	if len(aPacked) != m*blocks {
		return nil, fmt.Errorf("aPacked length mismatch")
	}
	if len(bPacked) != n*blocks {
		return nil, fmt.Errorf("bPacked length mismatch")
	}

	out := make([]int32, m*n)
	C.qc_binary_gemm(
		C.size_t(m),
		C.size_t(n),
		C.size_t(k),
		(*C.uint64_t)(unsafe.Pointer(&aPacked[0])),
		(*C.uint64_t)(unsafe.Pointer(&bPacked[0])),
		(*C.int32_t)(unsafe.Pointer(&out[0])),
	)
	return out, nil
}
