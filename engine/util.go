package engine

import (
	"runtime"
	"sync"
	"sync/atomic"
)

// parallelFor runs fn over 0..n in small chunks shared by a pool of goroutines.
// workers <= 0 means one worker per CPU core.
//
// Workers take the next chunk when they finish the last one, instead of being
// handed equal slices up front. On a chip with fast and slow cores this keeps
// the fast cores busy while a slow core finishes its chunk.
func parallelFor(n, workers int, fn func(lo, hi int)) {
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	workers = min(workers, n)
	if workers <= 1 {
		if n > 0 {
			fn(0, n)
		}
		return
	}
	grain := min(max(n/(workers*32), 1), 1024)
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				lo := int(next.Add(int64(grain))) - grain
				if lo >= n {
					return
				}
				fn(lo, min(lo+grain, n))
			}
		}()
	}
	wg.Wait()
}
