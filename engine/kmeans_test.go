package engine

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// blobs makes overlapping Gaussian groups, so k-means needs several rounds.
func blobs(n, dim, k int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, seed+7))
	centers := make([]float64, k*dim)
	for i := range centers {
		centers[i] = r.NormFloat64() * 2
	}
	x := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		c := r.IntN(k)
		for j := 0; j < dim; j++ {
			x[i*dim+j] = float32(centers[c*dim+j] + r.NormFloat64())
		}
	}
	return x
}

func TestKMeansLossFalls(t *testing.T) {
	_, losses := KMeans(blobs(2000, 8, 8, 1), 8, 8, 15, 0, 0)
	if losses[len(losses)-1] >= losses[0] {
		t.Fatalf("loss did not fall: %v", losses)
	}
	for i := 1; i < len(losses); i++ {
		// Lloyd's loss never rises. The tolerance covers float rounding only.
		if losses[i] > losses[i-1]*(1+1e-5) {
			t.Fatalf("loss rose at round %d: %v", i, losses)
		}
	}
}

func TestKMeansSameSeedSameResult(t *testing.T) {
	x := blobs(1000, 8, 8, 1)
	c1, l1 := KMeans(x, 8, 8, 5, 3, 0)
	c2, l2 := KMeans(x, 8, 8, 5, 3, 0)
	if !slices.Equal(c1.Data, c2.Data) || !slices.Equal(l1, l2) {
		t.Fatal("same seed gave different results")
	}
}

func TestKMeansIgnoresWorkerCount(t *testing.T) {
	x := blobs(1000, 8, 8, 1)
	c1, _ := KMeans(x, 8, 8, 5, 3, 1)
	c4, _ := KMeans(x, 8, 8, 5, 3, 4)
	if !slices.Equal(c1.Data, c4.Data) {
		t.Fatal("worker count changed the result")
	}
}

func TestAssignPicksNearestCentroid(t *testing.T) {
	x := blobs(200, 8, 5, 2)
	c := NewCentroids(slices.Clone(x[:5*8]), 8)
	labels, dists := c.Assign(x, 2)
	for i := 0; i < 200; i++ {
		best, bestD := 0, math.Inf(1)
		for j := 0; j < 5; j++ {
			d := float64(L2Sq(x[i*8:(i+1)*8], c.Data[j*8:(j+1)*8]))
			if d < bestD {
				best, bestD = j, d
			}
		}
		if int(labels[i]) != best || math.Abs(float64(dists[i])-bestD) > 1e-3 {
			t.Fatalf("point %d: got %d at %v, want %d at %v", i, labels[i], dists[i], best, bestD)
		}
	}
}

func TestDistsHandlesK(t *testing.T) {
	// K = 6 exercises both the blocks of four and the leftover loop.
	x := randVecs(1, 16, 5)
	c := NewCentroids(randVecs(6, 16, 6), 16)
	out := make([]float32, 6)
	c.Dists(x, out)
	for j := range out {
		want := L2Sq(x, c.Data[j*16:(j+1)*16])
		if math.Abs(float64(out[j]-want)) > 1e-3 {
			t.Fatalf("centroid %d: got %v, want %v", j, out[j], want)
		}
	}
}

func TestKMeansRestartsEmptyClusters(t *testing.T) {
	// Twenty identical points. All centroids start on the same spot, so
	// at least two of the three win no points. The run must stay finite.
	x := make([]float32, 20*4)
	for i := range x {
		x[i] = 1
	}
	c, losses := KMeans(x, 4, 3, 3, 0, 0)
	for _, v := range c.Data {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("centroid is not finite")
		}
	}
	for _, l := range losses {
		if l != 0 {
			t.Fatalf("loss = %v, want 0", losses)
		}
	}
}
