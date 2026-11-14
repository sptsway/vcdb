package engine

import (
	"runtime"
	"sync"
)

// parallelFor splits 0..n into chunks and runs fn on each chunk in its own
// goroutine. workers <= 0 means one worker per CPU core.
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
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(lo, hi)
		}()
	}
	wg.Wait()
}
