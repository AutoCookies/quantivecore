package quantcore

/*
#cgo CFLAGS: -I../../include
#cgo LDFLAGS: -L../../build-release -lquantcore -lstdc++
#include <quantcore/c_api.h>
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func Version() string {
	return C.GoString(C.qc_version())
}

func BinaryGEMM(m, n, k int, aPacked []uint64, bPacked []uint64) ([]int32, error) {
	blocks := (k + 63) / 64
	if len(aPacked) != m*blocks {
		return nil, fmt.Errorf("aPacked length mismatch")
	}
	if len(bPacked) != n*blocks {
		return nil, fmt.Errorf("bPacked length mismatch")
	}
	out := make([]int32, m*n)
	C.qc_binary_gemm(
		C.size_t(m), C.size_t(n), C.size_t(k),
		(*C.uint64_t)(unsafe.Pointer(&aPacked[0])),
		(*C.uint64_t)(unsafe.Pointer(&bPacked[0])),
		(*C.int32_t)(unsafe.Pointer(&out[0])),
	)
	return out, nil
}

func TernaryGEMM(m, n, k int, aPos []uint64, aNeg []uint64, bPos []uint64, bNeg []uint64) ([]int32, error) {
	blocks := (k + 63) / 64
	if len(aPos) != m*blocks || len(aNeg) != m*blocks {
		return nil, fmt.Errorf("A planes length mismatch")
	}
	if len(bPos) != n*blocks || len(bNeg) != n*blocks {
		return nil, fmt.Errorf("B planes length mismatch")
	}
	out := make([]int32, m*n)
	C.qc_ternary_gemm(
		C.size_t(m), C.size_t(n), C.size_t(k),
		(*C.uint64_t)(unsafe.Pointer(&aPos[0])),
		(*C.uint64_t)(unsafe.Pointer(&aNeg[0])),
		(*C.uint64_t)(unsafe.Pointer(&bPos[0])),
		(*C.uint64_t)(unsafe.Pointer(&bNeg[0])),
		(*C.int32_t)(unsafe.Pointer(&out[0])),
	)
	return out, nil
}
