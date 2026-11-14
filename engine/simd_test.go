package engine

import (
	"math"
	"testing"
)

// close reports whether a and b agree to float32 rounding. The assembly adds
// in a different order from the plain loop, so the last bits can differ.
func close(a, b float32) bool {
	return math.Abs(float64(a-b)) <= 1e-4*math.Max(1, math.Abs(float64(b)))
}

func TestL2SqMatchesGeneric(t *testing.T) {
	// Lengths on both sides of the 16-float block, with and without a tail.
	for _, n := range []int{1, 3, 8, 15, 16, 17, 31, 32, 33, 100, 128, 129, 300} {
		a, b := randVecs(1, n, uint64(n)), randVecs(1, n, uint64(n)+100)
		if got, want := L2Sq(a, b), l2sqGeneric(a, b); !close(got, want) {
			t.Errorf("n=%d: got %v, want %v", n, got, want)
		}
	}
}

func TestL2SqOfIdenticalVectorsIsExactlyZero(t *testing.T) {
	a := randVecs(1, 128, 1)
	if got := L2Sq(a, a); got != 0 {
		t.Fatalf("got %v, want 0", got)
	}
}

func TestL2SqPanicsOnShortSecondVector(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic")
		}
	}()
	L2Sq(make([]float32, 64), make([]float32, 32))
}

func TestDot4MatchesGeneric(t *testing.T) {
	// 12 and 20 are not multiples of 8, so they take the generic path.
	for _, n := range []int{8, 12, 16, 20, 64, 128, 256} {
		x, c := randVecs(1, n, 1), randVecs(4, n, 2)
		var got, want [4]float32
		dot4(x, c, &got)
		dot4Generic(x, c, &want)
		for j := range got {
			if !close(got[j], want[j]) {
				t.Errorf("n=%d row %d: got %v, want %v", n, j, got[j], want[j])
			}
		}
	}
}

func TestDot4RowsStayInOrder(t *testing.T) {
	// Each row is a different constant, so a swapped lane would show at once.
	n := 16
	x := make([]float32, n)
	c := make([]float32, 4*n)
	for i := range x {
		x[i] = 1
	}
	for row := 0; row < 4; row++ {
		for i := 0; i < n; i++ {
			c[row*n+i] = float32(row + 1)
		}
	}
	var got [4]float32
	dot4(x, c, &got)
	want := [4]float32{16, 32, 48, 64}
	if got != want {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func BenchmarkL2Sq128(b *testing.B) {
	x, y := randVecs(1, 128, 1), randVecs(1, 128, 2)
	var s float32
	for i := 0; i < b.N; i++ {
		s += L2Sq(x, y)
	}
	_ = s
}

func BenchmarkL2Sq128Generic(b *testing.B) {
	x, y := randVecs(1, 128, 1), randVecs(1, 128, 2)
	var s float32
	for i := 0; i < b.N; i++ {
		s += l2sqGeneric(x, y)
	}
	_ = s
}

func BenchmarkDists1024(b *testing.B) {
	c := NewCentroids(randVecs(1024, 128, 1), 128)
	q := randVecs(1, 128, 2)
	out := make([]float32, 1024)
	for i := 0; i < b.N; i++ {
		c.Dists(q, out)
	}
}

func TestL2Sq4IsBitIdenticalToL2Sq(t *testing.T) {
	// Exact equality, not closeness. Flat and IVF group rows differently, so
	// the grouped kernel must give the same bits as the single-row one.
	for _, n := range []int{16, 32, 128, 12, 20, 129} {
		q, x := randVecs(1, n, 1), randVecs(4, n, 2)
		var got [4]float32
		l2sq4(q, x, &got)
		for j := 0; j < 4; j++ {
			if want := L2Sq(q, x[j*n:(j+1)*n]); got[j] != want {
				t.Errorf("n=%d row %d: got %v, want %v", n, j, got[j], want)
			}
		}
	}
}

func BenchmarkScan12500(b *testing.B) {
	// 12,500 vectors per query, like nprobe 64 at 200k vectors
	x, q := randVecs(12500, 128, 1), randVecs(1, 128, 2)
	top := NewTopK(10)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		top.Reset(10)
		scanRows(top, q, x, nil, 0)
	}
}
