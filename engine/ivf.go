package engine

import (
	"fmt"
	"sync"
)

// IVF is an inverted-file index with exact distances inside each list.
//
// Training splits space into nlist regions with k-means. Each vector is
// stored in the list of its nearest centroid. A search scans only the nprobe
// lists whose centroids are closest to the query. More lists means better
// recall and lower speed. With nprobe = nlist the search is exact.
//
// Search calls may run at the same time. Add and Train must not overlap with
// any other call.
type IVF struct {
	dim, nlist int
	seed       uint64
	cents      *Centroids
	vecs       [][]float32 // list l holds its vectors back to back
	ids        [][]int64   // and their ids, in the same order
	ntotal     int
	scratch    sync.Pool // per-query buffers, reused to avoid allocation
}

func NewIVF(dim, nlist int, seed uint64) *IVF {
	ix := &IVF{
		dim: dim, nlist: nlist, seed: seed,
		vecs: make([][]float32, nlist),
		ids:  make([][]int64, nlist),
	}
	ix.scratch.New = func() any { return make([]float32, nlist) }
	return ix
}

func (ix *IVF) Len() int { return ix.ntotal }

// Train learns the list centroids from a sample of vectors. It returns the
// k-means loss of each round. workers <= 0 uses every CPU core.
func (ix *IVF) Train(x []float32, iters, workers int) []float64 {
	var losses []float64
	ix.cents, losses = KMeans(x, ix.dim, ix.nlist, iters, ix.seed, workers)
	return losses
}

// Add stores vectors. It can be called any number of times after Train.
// ids may be nil, which numbers the vectors from the current size upward.
func (ix *IVF) Add(x []float32, ids []int64, workers int) {
	ix.mustBeTrained()
	n := len(x) / ix.dim
	if len(x)%ix.dim != 0 || (ids != nil && len(ids) != n) {
		panic(fmt.Sprintf("engine: %d floats and %d ids do not match dim %d", len(x), len(ids), ix.dim))
	}
	labels, _ := ix.cents.Assign(x, workers)
	for i, l := range labels {
		ix.vecs[l] = append(ix.vecs[l], x[i*ix.dim:(i+1)*ix.dim]...)
		id := int64(ix.ntotal + i)
		if ids != nil {
			id = ids[i]
		}
		ix.ids[l] = append(ix.ids[l], id)
	}
	ix.ntotal += n
}

// Search returns the k nearest vectors to q, best first.
func (ix *IVF) Search(q []float32, k, nprobe int) []Neighbor {
	ix.mustBeTrained()
	nprobe = min(nprobe, ix.nlist)

	// Step 1: rank the centroids by distance to the query.
	cd := ix.scratch.Get().([]float32)
	defer ix.scratch.Put(cd)
	ix.cents.Dists(q, cd)
	closest := NewTopK(nprobe)
	for l, d := range cd {
		closest.Push(int64(l), d)
	}

	// Step 2: scan the chosen lists with exact distances.
	top := NewTopK(k)
	for _, c := range closest.items {
		vecs, ids := ix.vecs[c.ID], ix.ids[c.ID]
		for i, id := range ids {
			top.Push(id, L2Sq(q, vecs[i*ix.dim:(i+1)*ix.dim]))
		}
	}
	return top.Sorted()
}

// SearchBatch answers many queries at once. qs holds the queries back to back.
func (ix *IVF) SearchBatch(qs []float32, k, nprobe, workers int) [][]Neighbor {
	nq := len(qs) / ix.dim
	out := make([][]Neighbor, nq)
	parallelFor(nq, workers, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = ix.Search(qs[i*ix.dim:(i+1)*ix.dim], k, nprobe)
		}
	})
	return out
}

func (ix *IVF) mustBeTrained() {
	if ix.cents == nil {
		panic("engine: IVF used before Train")
	}
}
