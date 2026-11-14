package engine

import (
	"math"
	"sort"
	"testing"
)

const dim = 16

// slowSearch is the obvious version: float64, full sort, no heap.
func slowSearch(data []float32, q []float32, k int) []int64 {
	n := len(data) / dim
	type pair struct {
		id int64
		d  float64
	}
	ps := make([]pair, n)
	for i := 0; i < n; i++ {
		var d float64
		for j := 0; j < dim; j++ {
			diff := float64(data[i*dim+j]) - float64(q[j])
			d += diff * diff
		}
		ps[i] = pair{int64(i), d}
	}
	sort.Slice(ps, func(a, b int) bool {
		if ps[a].d != ps[b].d {
			return ps[a].d < ps[b].d
		}
		return ps[a].id < ps[b].id
	})
	ids := make([]int64, k)
	for i := range ids {
		ids[i] = ps[i].id
	}
	return ids
}

func TestFlatMatchesSlowLoop(t *testing.T) {
	x, qs := randVecs(500, dim, 1), randVecs(20, dim, 2)
	f := NewFlat(dim)
	f.Add(x)
	for i := 0; i < 20; i++ {
		q := qs[i*dim : (i+1)*dim]
		want := slowSearch(x, q, 10)
		for j, n := range f.Search(q, 10) {
			if n.ID != want[j] {
				t.Fatalf("query %d rank %d: got id %d, want %d", i, j, n.ID, want[j])
			}
		}
	}
}

func TestFlatStoredVectorFindsItselfFirst(t *testing.T) {
	x := randVecs(500, dim, 1)
	f := NewFlat(dim)
	f.Add(x)
	got := f.Search(x[123*dim:124*dim], 5)
	if got[0].ID != 123 || got[0].Dist != 0 {
		t.Fatalf("got %+v, want id 123 at distance 0", got[0])
	}
}

func TestFlatResultsAreSorted(t *testing.T) {
	f := NewFlat(dim)
	f.Add(randVecs(500, dim, 1))
	got := f.Search(randVecs(1, dim, 9), 20)
	for i := 1; i < len(got); i++ {
		if got[i].Dist < got[i-1].Dist {
			t.Fatalf("not sorted at %d: %v", i, got)
		}
	}
}

func TestFlatAddTwiceKeepsIDsInOrder(t *testing.T) {
	x := randVecs(500, dim, 1)
	f := NewFlat(dim)
	f.Add(x[:200*dim])
	f.Add(x[200*dim:])
	if f.Len() != 500 {
		t.Fatalf("Len = %d, want 500", f.Len())
	}
	if got := f.Search(x[450*dim:451*dim], 1); got[0].ID != 450 {
		t.Fatalf("got id %d, want 450", got[0].ID)
	}
}

func TestFlatFewerVectorsThanK(t *testing.T) {
	f := NewFlat(dim)
	f.Add(randVecs(3, dim, 1))
	if got := f.Search(randVecs(1, dim, 2), 5); len(got) != 3 {
		t.Fatalf("got %d results, want 3", len(got))
	}
}

func TestFlatBatchEqualsSingle(t *testing.T) {
	f := NewFlat(dim)
	f.Add(randVecs(300, dim, 1))
	qs := randVecs(40, dim, 2)
	batch := f.SearchBatch(qs, 10, 4)
	for i := range batch {
		single := f.Search(qs[i*dim:(i+1)*dim], 10)
		for j := range single {
			if batch[i][j] != single[j] {
				t.Fatalf("query %d rank %d differs", i, j)
			}
		}
	}
}

func TestAddRejectsRaggedInput(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic")
		}
	}()
	NewFlat(dim).Add(make([]float32, dim+1))
}

func TestTopKBreaksTiesByID(t *testing.T) {
	top := NewTopK(3)
	for _, n := range []Neighbor{{7, 1}, {3, 1}, {5, 1}, {9, 0.5}} {
		top.Push(n.ID, n.Dist)
	}
	got := top.Sorted()
	want := []int64{9, 3, 5}
	for i, n := range got {
		if n.ID != want[i] {
			t.Fatalf("got %v, want ids %v", got, want)
		}
	}
}

func TestTopKWorstIsInfUntilFull(t *testing.T) {
	top := NewTopK(2)
	top.Push(1, 5)
	if !math.IsInf(float64(top.Worst()), 1) {
		t.Fatal("Worst should be +Inf before the heap is full")
	}
	top.Push(2, 3)
	if top.Worst() != 5 {
		t.Fatalf("Worst = %v, want 5", top.Worst())
	}
}
