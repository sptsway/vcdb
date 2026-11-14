package engine

import "math/rand/v2"

// KMeans runs Lloyd's algorithm. Each round has two steps.
//  1. Assign every point to its nearest centroid.
//  2. Move every centroid to the mean of the points assigned to it.
//
// The total squared distance (the loss) can only stay equal or fall.
//
// The seed fixes the starting centroids, so a run repeats exactly. The
// assignment step splits across goroutines. The update step runs on one
// goroutine on purpose: float sums then come out the same for any worker count.
//
// It returns the final centroids and the loss measured at each round.
func KMeans(x []float32, dim, k, iters int, seed uint64, workers int) (*Centroids, []float64) {
	n := len(x) / dim
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))

	// Start from k distinct random points of the data.
	data := make([]float32, k*dim)
	for j, i := range rng.Perm(n)[:k] {
		copy(data[j*dim:(j+1)*dim], x[i*dim:(i+1)*dim])
	}
	cents := NewCentroids(data, dim)

	losses := make([]float64, 0, iters)
	sums := make([]float64, k*dim)
	counts := make([]int, k)
	for range iters {
		labels, dists := cents.Assign(x, workers)
		var loss float64
		for _, d := range dists {
			loss += float64(d)
		}
		losses = append(losses, loss)

		clear(sums)
		clear(counts)
		for i, l := range labels {
			counts[l]++
			row := sums[int(l)*dim : (int(l)+1)*dim]
			for j, v := range x[i*dim : (i+1)*dim] {
				row[j] += float64(v)
			}
		}
		next := make([]float32, k*dim)
		for j := 0; j < k; j++ {
			if counts[j] == 0 {
				// A centroid that lost all its points restarts at a random point.
				i := rng.IntN(n)
				copy(next[j*dim:(j+1)*dim], x[i*dim:(i+1)*dim])
				continue
			}
			inv := 1 / float64(counts[j])
			for t := 0; t < dim; t++ {
				next[j*dim+t] = float32(sums[j*dim+t] * inv)
			}
		}
		cents = NewCentroids(next, dim)
	}
	return cents, losses
}
