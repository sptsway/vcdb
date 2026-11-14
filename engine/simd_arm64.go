//go:build arm64 && !purego

package engine

// Implemented in simd_arm64.s with NEON, 128-bit vectors of four float32.

// l2sqNEON returns the squared L2 distance of n floats. n must be a multiple of 16.
//
//go:noescape
func l2sqNEON(a, b *float32, n int) float32

// dot4NEON writes the dot products of x with four consecutive rows of c.
// n is the row length and must be a multiple of 8.
//
//go:noescape
func dot4NEON(x, c *float32, n int, out *[4]float32)

// l2sq4NEON writes the squared distance from q to four consecutive rows of x.
// n is the row length and must be a multiple of 16. Each result is bit-for-bit
// what l2sqNEON gives for that row alone.
//
//go:noescape
func l2sq4NEON(q, x *float32, n int, out *[4]float32)

func l2sq(a, b []float32) float32 {
	b = b[:len(a)] // panics if b is shorter, so the assembly never reads past b
	n := len(a) &^ 15
	var s float32
	if n > 0 {
		s = l2sqNEON(&a[0], &b[0], n)
	}
	if n < len(a) { // the last few floats, when len is not a multiple of 16
		s += l2sqGeneric(a[n:], b[n:])
	}
	return s
}

func dot4(x, c []float32, out *[4]float32) {
	_ = c[4*len(x)-1] // panics if c is too short
	if len(x) == 0 || len(x)%8 != 0 {
		dot4Generic(x, c, out)
		return
	}
	dot4NEON(&x[0], &c[0], len(x), out)
}

func l2sq4(q, x []float32, out *[4]float32) {
	_ = x[4*len(q)-1] // panics if x is too short
	if len(q) == 0 || len(q)%16 != 0 {
		l2sq4Generic(q, x, out)
		return
	}
	l2sq4NEON(&q[0], &x[0], len(q), out)
}
