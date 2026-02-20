package quantcore

import "testing"

func packBinary(vals []int8, rows, cols int) []uint64 {
	blocks := (cols + 63) / 64
	out := make([]uint64, rows*blocks)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			if vals[r*cols+c] > 0 {
				block := c / 64
				bit := uint(c % 64)
				out[r*blocks+block] |= (uint64(1) << bit)
			}
		}
	}
	return out
}

func TestBinaryGemm(t *testing.T) {
	a := []int8{1, -1, 1, -1, -1, 1, -1, 1}
	b := []int8{1, 1, -1, -1, -1, -1, 1, 1}

	ap := packBinary(a, 2, 4)
	bp := packBinary(b, 2, 4)

	got, err := BinaryGemm(2, 2, 4, ap, bp)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	want := []int32{0, 0, 0, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("mismatch at %d: got=%d want=%d", i, got[i], want[i])
		}
	}
}
