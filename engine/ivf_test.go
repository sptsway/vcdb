package engine

import (
	"math/rand/v2"
	"slices"
	"testing"
)

const nlist = 16

// clustered makes Gaussian groups, so the lists actually mean something.
func clustered(n int, seed uint64) []float32 { return blobs(n, dim, nlist, seed) }

func buildIVF(x []float32) *IVF {
	ix := NewIVF(dim, nlist, 0)
	ix.Train(x, 20, 0)
	ix.Add(x, nil, 0)
	return ix
}

func oracle(x []float32) *Flat {
	f := NewFlat(dim)
	f.Add(x)
	return f
}

func recall(found, truth [][]Neighbor) float64 {
	hits, total := 0, 0
	for i := range truth {
		want := map[int64]bool{}
		for _, n := range truth[i] {
			want[n.ID] = true
		}
		for _, n := range found[i] {
			if want[n.ID] {
				hits++
			}
		}
		total += len(truth[i])
	}
	return float64(hits) / float64(total)
}

func TestIVFFullProbeEqualsFlatExactly(t *testing.T) {
	x, qs := clustered(3000, 1), clustered(100, 2)
	got := buildIVF(x).SearchBatch(qs, 10, nlist, 0)
	want := oracle(x).SearchBatch(qs, 10, 0)
	for i := range want {
		if !slices.Equal(got[i], want[i]) { // ids and distances, bit for bit
			t.Fatalf("query %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestIVFStoredVectorFindsItselfFirst(t *testing.T) {
	x := clustered(3000, 1)
	ix := buildIVF(x)
	for _, nprobe := range []int{1, nlist} {
		for _, row := range []int{0, 17, 1234, 2999} {
			got := ix.Search(x[row*dim:(row+1)*dim], 3, nprobe)
			if got[0].ID != int64(row) || got[0].Dist != 0 {
				t.Fatalf("nprobe %d row %d: got %+v", nprobe, row, got[0])
			}
		}
	}
}

func TestIVFVectorsAddedAfterTrainingAreFound(t *testing.T) {
	x := clustered(3000, 1)
	ix := NewIVF(dim, nlist, 0)
	ix.Train(x[:1500*dim], 20, 0) // learn the lists from half the data
	ix.Add(x[:1500*dim], nil, 0)
	ix.Add(x[1500*dim:], nil, 0) // a second add, later
	if ix.Len() != 3000 {
		t.Fatalf("Len = %d, want 3000", ix.Len())
	}
	qs := clustered(50, 2)
	got, want := ix.SearchBatch(qs, 10, nlist, 0), oracle(x).SearchBatch(qs, 10, 0)
	for i := range want {
		if !slices.Equal(got[i], want[i]) {
			t.Fatalf("query %d differs after a later add", i)
		}
	}
}

func TestIVFRecallGrowsWithNprobe(t *testing.T) {
	x, qs := clustered(3000, 1), clustered(200, 3)
	ix := buildIVF(x)
	truth := oracle(x).SearchBatch(qs, 10, 0)
	r1 := recall(ix.SearchBatch(qs, 10, 1, 0), truth)
	r4 := recall(ix.SearchBatch(qs, 10, 4, 0), truth)
	rAll := recall(ix.SearchBatch(qs, 10, nlist, 0), truth)
	if !(r1 < r4 && r4 <= rAll && rAll == 1) {
		t.Fatalf("recall at nprobe 1, 4, all = %v, %v, %v", r1, r4, rAll)
	}
}

func TestIVFCustomIDs(t *testing.T) {
	x := clustered(500, 1)
	ix := NewIVF(dim, 4, 0)
	ix.Train(x, 10, 0)
	ids := make([]int64, 500)
	for i := range ids {
		ids[i] = int64(10_000 + i)
	}
	ix.Add(x, ids, 0)
	if got := ix.Search(x[5*dim:6*dim], 1, 4); got[0].ID != 10_005 {
		t.Fatalf("got id %d, want 10005", got[0].ID)
	}
}

func TestIVFEmptyListsAreSkipped(t *testing.T) {
	x := clustered(40, 1)
	ix := NewIVF(dim, 32, 0)
	ix.Train(x, 10, 0)
	ix.Add(x[:10*dim], nil, 0) // 10 vectors over 32 lists leaves at least 22 empty
	empty := 0
	for _, ids := range ix.ids {
		if len(ids) == 0 {
			empty++
		}
	}
	if empty < 22 {
		t.Fatalf("only %d empty lists", empty)
	}
	got, want := ix.Search(x[:dim], 5, 32), oracle(x[:10*dim]).Search(x[:dim], 5)
	if !slices.Equal(got, want) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestIVFParallelAddEqualsSerial(t *testing.T) {
	x := clustered(2000, 1)
	build := func(workers int) *IVF {
		ix := NewIVF(dim, nlist, 0)
		ix.Train(x, 10, 1)
		ix.Add(x, nil, workers)
		return ix
	}
	a, b := build(1), build(4)
	for l := range a.ids {
		if !slices.Equal(a.ids[l], b.ids[l]) || !slices.Equal(a.vecs[l], b.vecs[l]) {
			t.Fatalf("list %d differs between 1 and 4 workers", l)
		}
	}
}

func TestIVFUseBeforeTrainPanics(t *testing.T) {
	for name, call := range map[string]func(*IVF){
		"add":    func(ix *IVF) { ix.Add(make([]float32, dim), nil, 0) },
		"search": func(ix *IVF) { ix.Search(make([]float32, dim), 1, 1) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s before Train should panic", name)
				}
			}()
			call(NewIVF(dim, nlist, 0))
		}()
	}
}

func TestIVFConcurrentSearchesAgree(t *testing.T) {
	x := clustered(2000, 1)
	ix := buildIVF(x)
	qs := clustered(64, 4)
	want := ix.SearchBatch(qs, 10, 4, 1)
	r := rand.New(rand.NewPCG(1, 2))
	for trial := 0; trial < 5; trial++ {
		got := ix.SearchBatch(qs, 10, 4, 2+r.IntN(6))
		for i := range want {
			if !slices.Equal(got[i], want[i]) {
				t.Fatalf("trial %d query %d differs", trial, i)
			}
		}
	}
}
