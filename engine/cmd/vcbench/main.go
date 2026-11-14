// Command vcbench times the Go engine and writes CSV files into results/.
//
// It reads the raw binary files made by bench/export_data.py, so it sees the
// same vectors and the same ground truth as the Python benchmark against Faiss.
//
// Fairness rules match bench/compare.py:
//   - nlist and k-means iterations are the same as the Faiss run.
//   - Query speed is one query per call on one goroutine. A batched run on all
//     cores is reported as a separate, labelled row.
//   - Each query figure is the fastest of several passes. Other programs on the
//     machine can only slow a pass down, never speed it up.
//   - Inserts use one worker. The same inserts on all cores are separate rows.
package main

import (
	"encoding/binary"
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"slices"
	"strconv"
	"syscall"
	"time"

	"github.com/sptsway/vcdb/engine"
)

const (
	dim = 128
	k   = 10
)

var (
	passes       = 7 // timed passes per query figure
	nprobes      = []int{1, 4, 16, 64}
	writeBatches = []int{1, 100, 1000, 10000}
)

func main() {
	dir := flag.String("dir", "data/export/200k", "folder written by bench/export_data.py")
	tag := flag.String("tag", "200k", "file suffix for the results")
	nlist := flag.Int("nlist", 1024, "number of lists")
	iters := flag.Int("iters", 20, "k-means iterations")
	out := flag.String("out", "results", "folder for the CSV files")
	cpuprofile := flag.String("cpuprofile", "", "write a CPU profile to this file")
	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			panic(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	train := readF32(filepath.Join(*dir, "train.f32"))
	queries := readF32(filepath.Join(*dir, "queries.f32"))
	extra := readF32(filepath.Join(*dir, "extra.f32"))
	gt := readI32(filepath.Join(*dir, "gt.i32"))
	n, nq := len(train)/dim, len(queries)/dim
	fmt.Printf("run %s: n=%d nlist=%d queries=%d\n", *tag, n, *nlist, nq)

	// ---- check the oracle against the exported ground truth ----------------
	agree := oracleAgreement(train, queries, gt, 500)
	fmt.Printf("  Go flat agrees with ground truth on %.4f of 500 queries\n", agree)

	// ---- build --------------------------------------------------------------
	ix := engine.NewIVF(dim, *nlist, 0)
	t := time.Now()
	losses := ix.Train(train, *iters, 0)
	trainS := time.Since(t).Seconds()
	t = time.Now()
	ix.Add(train, nil, 0)
	addS := time.Since(t).Seconds()
	fmt.Printf("  train %.2fs  add %.2fs\n", trainS, addS)

	w := newWriter(*out)
	w.csv(fmt.Sprintf("go_build_%s.csv", *tag),
		[]string{"impl", "n", "nlist", "train_s", "add_s", "add_vectors_per_s"},
		[][]string{{"vecstore_go", itoa(n), itoa(*nlist), f2(trainS), f2(addS), f0(float64(n) / addS)}})
	rows := [][]string{}
	for i, l := range losses {
		rows = append(rows, []string{itoa(i + 1), fmt.Sprintf("%.1f", l)})
	}
	w.csv(fmt.Sprintf("go_kmeans_loss_%s.csv", *tag), []string{"iteration", "loss"}, rows)

	// ---- reads --------------------------------------------------------------
	rows = nil
	for _, nprobe := range nprobes {
		res := ix.SearchBatch(queries, k, nprobe, 0)
		rec := recallAtK(res, gt)
		single := best(passes, func() float64 { return qps(nq, func() { searchEach(ix, queries, nprobe) }) },
			func() { searchEach(ix, queries[:200*dim], nprobe) })
		batch := best(passes, func() float64 { return qps(nq, func() { ix.SearchBatch(queries, k, nprobe, 0) }) }, nil)
		rows = append(rows,
			[]string{"vecstore_go", itoa(n), itoa(*nlist), itoa(nprobe), fmt.Sprintf("%.4f", rec), f0(single)},
			[]string{"vecstore_go_batch", itoa(n), itoa(*nlist), itoa(nprobe), fmt.Sprintf("%.4f", rec), f0(batch)})
		fmt.Printf("  nprobe=%3d  recall %.4f  single %8.0f qps  batch %9.0f qps\n", nprobe, rec, single, batch)
	}
	w.csv(fmt.Sprintf("go_recall_qps_%s.csv", *tag),
		[]string{"impl", "n", "nlist", "nprobe", "recall_at_10", "qps"}, rows)

	// ---- writes -------------------------------------------------------------
	rows, used := nil, 0
	for _, batch := range writeBatches {
		total := max(batch, 5000)
		vps := addRate(ix, extra[used*dim:(used+total)*dim], batch, total, 1)
		used += total
		rows = append(rows, []string{"vecstore_go", itoa(batch), itoa(total), f0(vps)})
		fmt.Printf("  batch=%5d  %9.0f vectors/s (1 worker)\n", batch, vps)
	}
	// The same inserts again with every core. One vector at a time cannot be
	// split across cores, so batch size 1 is left out.
	for _, batch := range writeBatches[1:] {
		total := max(batch, 5000)
		vps := addRate(ix, extra[:total*dim], batch, total, 0)
		rows = append(rows, []string{"vecstore_go_batch", itoa(batch), itoa(total), f0(vps)})
		fmt.Printf("  batch=%5d  %9.0f vectors/s (all cores)\n", batch, vps)
	}
	w.csv(fmt.Sprintf("go_write_qps_%s.csv", *tag),
		[]string{"impl", "batch_size", "vectors", "vectors_per_s"}, rows)

	// ---- run information ----------------------------------------------------
	peak := peakRSSMB()
	w.upsert("go_run_info.csv",
		[]string{"run", "n", "nlist", "kmeans_iters", "queries", "go", "goos_goarch", "cores",
			"peak_rss_mb", "flat_agreement_with_ground_truth"},
		[]string{*tag, itoa(n), itoa(*nlist), itoa(*iters), itoa(nq), runtime.Version(),
			runtime.GOOS + "/" + runtime.GOARCH, itoa(runtime.NumCPU()), f0(peak),
			fmt.Sprintf("%.4f", agree)})
	fmt.Printf("peak memory %.0f MB (cap is 8192 MB)\n", peak)
}

// searchEach runs the queries one at a time on the calling goroutine.
func searchEach(ix *engine.IVF, queries []float32, nprobe int) {
	for i := 0; i < len(queries)/dim; i++ {
		ix.Search(queries[i*dim:(i+1)*dim], k, nprobe)
	}
}

// addRate inserts total vectors in batches and returns vectors per second.
func addRate(ix *engine.IVF, vecs []float32, batch, total, workers int) float64 {
	t := time.Now()
	for i := 0; i < total; i += batch {
		ix.Add(vecs[i*dim:(i+batch)*dim], nil, workers)
	}
	return float64(total) / time.Since(t).Seconds()
}

// qps times fn and returns count / seconds.
func qps(count int, fn func()) float64 {
	t := time.Now()
	fn()
	return float64(count) / time.Since(t).Seconds()
}

// best runs measure reps times, after an optional warm-up, and returns the highest rate.
func best(reps int, measure func() float64, warm func()) float64 {
	if warm != nil {
		warm()
	}
	top := 0.0
	for i := 0; i < reps; i++ {
		top = max(top, measure())
	}
	return top
}

// recallAtK is the share of the true k neighbours that the index returned.
func recallAtK(res [][]engine.Neighbor, gt []int32) float64 {
	hits := 0
	for i, r := range res {
		want := gt[i*k : (i+1)*k]
		for _, n := range r {
			if slices.Contains(want, int32(n.ID)) {
				hits++
			}
		}
	}
	return float64(hits) / float64(len(res)*k)
}

// oracleAgreement runs the Go flat index on a sample of queries and reports
// how much of the exported ground truth it reproduces.
func oracleAgreement(train, queries []float32, gt []int32, sample int) float64 {
	flat := engine.NewFlat(dim)
	flat.Add(train)
	sample = min(sample, len(queries)/dim)
	return recallAtK(flat.SearchBatch(queries[:sample*dim], k, 0), gt[:sample*k])
}

func peakRSSMB() float64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return math.NaN()
	}
	if runtime.GOOS == "darwin" {
		return float64(ru.Maxrss) / (1 << 20) // bytes on macOS
	}
	return float64(ru.Maxrss) / (1 << 10) // kilobytes on Linux
}

func readF32(path string) []float32 {
	b := mustRead(path)
	x := make([]float32, len(b)/4)
	for i := range x {
		x[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return x
}

func readI32(path string) []int32 {
	b := mustRead(path)
	x := make([]int32, len(b)/4)
	for i := range x {
		x[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return x
}

func mustRead(path string) []byte {
	b, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "run `make export` first:", err)
		os.Exit(1)
	}
	return b
}

type writer struct{ dir string }

func newWriter(dir string) writer {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		panic(err)
	}
	return writer{dir}
}

func (w writer) csv(name string, header []string, rows [][]string) {
	f, err := os.Create(filepath.Join(w.dir, name))
	if err != nil {
		panic(err)
	}
	defer f.Close()
	cw := csv.NewWriter(f)
	cw.Write(header)
	cw.WriteAll(rows)
	cw.Flush()
	fmt.Println("  wrote", filepath.Join("results", name))
}

// upsert replaces the row with the same first column, or appends it.
func (w writer) upsert(name string, header, row []string) {
	var rows [][]string
	if f, err := os.Open(filepath.Join(w.dir, name)); err == nil {
		old, _ := csv.NewReader(f).ReadAll()
		f.Close()
		for _, r := range old[1:] {
			if r[0] != row[0] {
				rows = append(rows, r)
			}
		}
	}
	w.csv(name, header, append(rows, row))
}

func itoa(v int) string   { return strconv.Itoa(v) }
func f0(v float64) string { return fmt.Sprintf("%.0f", v) }
func f2(v float64) string { return fmt.Sprintf("%.2f", v) }
