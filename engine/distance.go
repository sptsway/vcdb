package engine

// L2Sq returns the squared L2 distance between a and b.
//
// The square root is skipped because it never changes the order of
// neighbours. Four running sums let the CPU overlap the additions.
// Flat and IVF both call this one function, so their distances for the
// same pair of vectors are bit-for-bit equal. That makes exact tests fair.
func L2Sq(a, b []float32) float32 {
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
