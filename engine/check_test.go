package engine

import (
	"fmt"
	"strings"
	"testing"
)

// Each misuse must panic with a message that names the problem, not a raw
// index-out-of-range error from deep inside the code.
func TestMisuseGivesClearMessages(t *testing.T) {
	x := randVecs(200, dim, 1)
	ix := NewIVF(dim, 4, 0)
	ix.Train(x, 2, 0)
	ix.Add(x, nil, 0)
	flat := NewFlat(dim)
	flat.Add(x)
	q := x[:dim]

	cases := []struct {
		name string
		call func()
		want string
	}{
		{"ivf k=0", func() { ix.Search(q, 0, 2) }, "k is 0"},
		{"ivf nprobe=0", func() { ix.Search(q, 5, 0) }, "nprobe is 0"},
		{"ivf short query", func() { ix.Search(q[:dim/2], 3, 2) }, "the query has 8 floats"},
		{"ivf long query", func() { ix.Search(x[:dim+1], 3, 2) }, "the query has 17 floats"},
		{"ivf ragged batch", func() { ix.SearchBatch(x[:dim+3], 3, 2, 0) }, "not a multiple of dim"},
		{"ivf ids count", func() { ix.Add(x[:2*dim], []int64{1}, 0) }, "2 vectors but 1 ids"},
		{"train too few", func() { NewIVF(dim, 8, 0).Train(x[:4*dim], 2, 0) }, "at least nlist (8)"},
		{"train ragged", func() { NewIVF(dim, 2, 0).Train(x[:dim+1], 2, 0) }, "not a multiple of dim"},
		{"kmeans k=0", func() { KMeans(x, dim, 0, 2, 0, 0) }, "k is 0"},
		{"flat k=0", func() { flat.Search(q, 0) }, "k is 0"},
		{"flat short query", func() { flat.Search(q[:4], 3) }, "the query has 4 floats"},
		{"flat ragged batch", func() { flat.SearchBatch(x[:dim+1], 3, 0) }, "not a multiple of dim"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				msg := fmt.Sprint(recover())
				if !strings.HasPrefix(msg, "engine: ") || !strings.Contains(msg, c.want) {
					t.Fatalf("panic %q, want an engine: message containing %q", msg, c.want)
				}
			}()
			c.call()
		})
	}
}
