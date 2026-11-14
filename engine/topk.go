package engine

import (
	"math"
	"slices"
)

// Neighbor is one search result.
type Neighbor struct {
	ID   int64
	Dist float32 // squared L2 distance
}

// better reports whether a ranks ahead of b. Smaller distance wins.
// On a tie the smaller id wins, which keeps results repeatable.
func better(a, b Neighbor) bool {
	return a.Dist < b.Dist || (a.Dist == b.Dist && a.ID < b.ID)
}

// TopK keeps the k best neighbours seen so far.
//
// It is a max-heap: the root is the worst neighbour kept. A new candidate
// replaces the root only if it is better. That costs O(log k) per accepted
// candidate and nothing for the many that are rejected.
type TopK struct {
	k     int
	items []Neighbor
}

func NewTopK(k int) *TopK {
	return &TopK{k: k, items: make([]Neighbor, 0, k)}
}

// Worst returns the distance a candidate must beat to get in.
// It is +Inf until k neighbours have been collected.
func (t *TopK) Worst() float32 {
	if len(t.items) < t.k {
		return float32(math.Inf(1))
	}
	return t.items[0].Dist
}

func (t *TopK) Push(id int64, dist float32) {
	n := Neighbor{ID: id, Dist: dist}
	if len(t.items) < t.k {
		t.items = append(t.items, n)
		t.up(len(t.items) - 1)
		return
	}
	if better(n, t.items[0]) {
		t.items[0] = n
		t.down(0)
	}
}

// Sorted returns the kept neighbours, best first. It empties the heap.
func (t *TopK) Sorted() []Neighbor {
	out := t.items
	slices.SortFunc(out, func(a, b Neighbor) int {
		switch {
		case better(a, b):
			return -1
		case better(b, a):
			return 1
		}
		return 0
	})
	t.items = nil
	return out
}

// up moves item i toward the root while it is worse than its parent.
func (t *TopK) up(i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !better(t.items[p], t.items[i]) {
			break
		}
		t.items[p], t.items[i] = t.items[i], t.items[p]
		i = p
	}
}

// down moves item i away from the root while a child is worse than it.
func (t *TopK) down(i int) {
	n := len(t.items)
	for {
		worst := i
		for _, c := range [2]int{2*i + 1, 2*i + 2} {
			if c < n && better(t.items[worst], t.items[c]) {
				worst = c
			}
		}
		if worst == i {
			return
		}
		t.items[i], t.items[worst] = t.items[worst], t.items[i]
		i = worst
	}
}
