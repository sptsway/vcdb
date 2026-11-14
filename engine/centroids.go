package engine

// Centroids is a set of K points of dimension Dim, stored back to back.
// The squared norm of each point is saved so a distance costs one dot product.
type Centroids struct {
	Dim, K int
	Data   []float32
	norms  []float32
}

// NewCentroids wraps data (K*dim floats) and computes the norms.
func NewCentroids(data []float32, dim int) *Centroids {
	c := &Centroids{Dim: dim, K: len(data) / dim, Data: data}
	c.norms = make([]float32, c.K)
	for j := range c.norms {
		row := data[j*dim : (j+1)*dim]
		c.norms[j] = dot(row, row)
	}
	return c
}

func dot(a, b []float32) float32 {
	b = b[:len(a)]
	var s float32
	for i, v := range a {
		s += v * b[i]
	}
	return s
}

// Dists writes into out the squared distance from x to every centroid.
//
// It uses |x - c|^2 = |x|^2 - 2 x.c + |c|^2. Four centroids are handled per
// pass over x (dot4), so each loaded element of x is used four times.
// Rounding can push a tiny distance below zero. It is clipped to zero.
func (c *Centroids) Dists(x []float32, out []float32) {
	d := c.Dim
	xn := dot(x, x)
	j := 0
	var r [4]float32
	for ; j+4 <= c.K; j += 4 {
		dot4(x, c.Data[j*d:(j+4)*d], &r)
		out[j] = clip(xn + c.norms[j] - 2*r[0])
		out[j+1] = clip(xn + c.norms[j+1] - 2*r[1])
		out[j+2] = clip(xn + c.norms[j+2] - 2*r[2])
		out[j+3] = clip(xn + c.norms[j+3] - 2*r[3])
	}
	for ; j < c.K; j++ {
		out[j] = clip(xn + c.norms[j] - 2*dot(x, c.Data[j*d:(j+1)*d]))
	}
}

func clip(v float32) float32 {
	if v < 0 {
		return 0
	}
	return v
}

// Nearest returns the closest centroid to x and its squared distance.
// scratch must hold K floats. Ties go to the lower index.
func (c *Centroids) Nearest(x []float32, scratch []float32) (int, float32) {
	c.Dists(x, scratch)
	best := 0
	for j := 1; j < c.K; j++ {
		if scratch[j] < scratch[best] {
			best = j
		}
	}
	return best, scratch[best]
}

// Assign finds the nearest centroid of every vector in x. Each vector is
// independent, so the work is split across goroutines without changing the answer.
func (c *Centroids) Assign(x []float32, workers int) (labels []int32, dists []float32) {
	n := len(x) / c.Dim
	labels = make([]int32, n)
	dists = make([]float32, n)
	parallelFor(n, workers, func(lo, hi int) {
		scratch := make([]float32, c.K)
		for i := lo; i < hi; i++ {
			j, d := c.Nearest(x[i*c.Dim:(i+1)*c.Dim], scratch)
			labels[i], dists[i] = int32(j), d
		}
	})
	return labels, dists
}
