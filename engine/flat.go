// Package engine is a small vector index written from scratch.
//
// Flat is exact brute-force search and the correctness oracle. KMeans trains
// centroids. IVF is the inverted-file index built from both.
package engine

import "fmt"

// Flat stores every vector in one slice and compares the query with all of them.
// A vector's id is its position: 0, 1, 2, ...
type Flat struct {
	dim  int
	data []float32 // row i lives at data[i*dim : (i+1)*dim]
}

func NewFlat(dim int) *Flat { return &Flat{dim: dim} }

func (f *Flat) Dim() int { return f.dim }
func (f *Flat) Len() int { return len(f.data) / f.dim }

// Add appends vectors. x holds them back to back, so len(x) is a multiple of dim.
func (f *Flat) Add(x []float32) {
	if len(x)%f.dim != 0 {
		panic(fmt.Sprintf("engine: %d floats is not a multiple of dim %d", len(x), f.dim))
	}
	f.data = append(f.data, x...)
}

// Search returns the k nearest vectors to q, best first.
// It returns fewer than k if the index holds fewer vectors.
func (f *Flat) Search(q []float32, k int) []Neighbor {
	top := NewTopK(k)
	scanRows(top, q, f.data, nil, 0)
	return top.Sorted()
}

// SearchBatch answers many queries at once. qs holds the queries back to back.
// workers <= 0 uses every CPU core.
func (f *Flat) SearchBatch(qs []float32, k, workers int) [][]Neighbor {
	nq := len(qs) / f.dim
	out := make([][]Neighbor, nq)
	parallelFor(nq, workers, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = f.Search(qs[i*f.dim:(i+1)*f.dim], k)
		}
	})
	return out
}
