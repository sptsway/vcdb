package engine

import "testing"

func benchIVF(b *testing.B, nprobe int) {
	x := randVecs(60000, 128, 1)
	ix := NewIVF(128, 1024, 0)
	ix.Train(x, 2, 0)
	ix.Add(x, nil, 0)
	qs := randVecs(256, 128, 2)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j := i % 256
		ix.Search(qs[j*128:(j+1)*128], 10, nprobe)
	}
}

func BenchmarkIVFSearchNprobe1(b *testing.B)  { benchIVF(b, 1) }
func BenchmarkIVFSearchNprobe16(b *testing.B) { benchIVF(b, 16) }
