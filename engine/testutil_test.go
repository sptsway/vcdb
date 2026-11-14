package engine

import "math/rand/v2"

// randVecs makes n vectors of the given dimension from a fixed seed.
func randVecs(n, dim int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, seed+1))
	x := make([]float32, n*dim)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	return x
}
