package engine

// L2Sq returns the squared L2 distance between a and b.
//
// The square root is skipped because it never changes the order of
// neighbours. On arm64 the work runs in NEON assembly
// (simd_arm64.s). Other CPUs use the plain Go loop below.
//
// Flat and IVF both call this one function, so their distances for the
// same pair of vectors are bit-for-bit equal. That makes exact tests fair.
func L2Sq(a, b []float32) float32 { return l2sq(a, b) }

// l2sqGeneric is the portable version. It is also the reference that the
// assembly is tested against. Four running sums let the CPU overlap the adds.
func l2sqGeneric(a, b []float32) float32 {
	b = b[:len(a)] // one bounds check here lets the loop skip the rest
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(a); i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < len(a); i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}

// dot4Generic writes into out the dot products of x with four consecutive
// rows of c. Each row has len(x) floats.
func dot4Generic(x, c []float32, out *[4]float32) {
	n := len(x)
	c0, c1, c2, c3 := c[:n], c[n:2*n], c[2*n:3*n], c[3*n:4*n]
	var s0, s1, s2, s3 float32
	for i, v := range x {
		s0 += v * c0[i]
		s1 += v * c1[i]
		s2 += v * c2[i]
		s3 += v * c3[i]
	}
	*out = [4]float32{s0, s1, s2, s3}
}

// l2sq4Generic writes into out the distance from q to four consecutive rows of
// x. It calls l2sq per row, so it matches L2Sq exactly on every CPU.
func l2sq4Generic(q, x []float32, out *[4]float32) {
	n := len(q)
	for j := range out {
		out[j] = l2sq(q, x[j*n:(j+1)*n])
	}
}

// scanRows offers every row of rows to top. Row i has id ids[i], or base+i
// when ids is nil. Rows go through four at a time, which shares the load of q.
//
// Flat and IVF both scan with this one function.
func scanRows(top *TopK, q, rows []float32, ids []int64, base int64) {
	d := len(q)
	n := len(rows) / d
	id := func(i int) int64 {
		if ids != nil {
			return ids[i]
		}
		return base + int64(i)
	}
	var r [4]float32
	i := 0
	for ; i+4 <= n; i += 4 {
		l2sq4(q, rows[i*d:(i+4)*d], &r)
		for j, dist := range r {
			if dist <= top.Worst() { // anything farther cannot get in
				top.Push(id(i+j), dist)
			}
		}
	}
	for ; i < n; i++ {
		if dist := l2sq(q, rows[i*d:(i+1)*d]); dist <= top.Worst() {
			top.Push(id(i), dist)
		}
	}
}
