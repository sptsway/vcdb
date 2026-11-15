# Walkthrough

This page explains the code in plain sentences. Read it next to the source files.

The engine is written in Go and lives in `engine/`. Python is used only to drive the benchmark against Faiss.

## The problem

A vector is a list of numbers. Here each vector has 128 numbers.

Two vectors are close when the distance between them is small.

Nearest-neighbour search takes a query vector. It returns the k stored vectors closest to it.

The simple way is to compare the query with every stored vector. That is exact but slow.

An index makes search faster. It gives up a little accuracy to look at fewer vectors.

## distance.go: one distance function

`L2Sq` returns the squared L2 distance between two vectors. The square root is skipped because it never changes the order.

Every index calls this one function. So two indexes give the same distance for the same pair of vectors.

That is what lets the tests demand exact equality instead of "close enough".

## topk.go: keeping the best k

`TopK` is a bounded heap. It holds the k best results seen so far.

The worst kept result sits at the root. A new candidate must beat the root to get in.

Most candidates lose at once. They cost one comparison.

Ties are broken by the smaller id. This makes results repeatable.

## flat.go: the exact search

`Flat` stores every vector in one slice. A vector's id is its row number.

`Search` compares the query with every row and keeps the best k in a `TopK`.

This index is slow on big data. Its job is to be obviously correct.

Every other index is tested against it.

## centroids.go and kmeans.go: finding the regions

K-means groups vectors around k centres. Each centre is called a centroid.

It repeats two steps. First, assign each vector to its nearest centroid.

Second, move each centroid to the average of the vectors assigned to it.

The loss is the total squared distance from each vector to its centroid.

The loss can only stay equal or fall after each round. A test checks this.

The assignment step runs on all cores. The update step runs on one core on purpose.

That makes the answer the same for any number of cores. A test checks this too.

The starting centroids are random data points. A fixed seed picks them.

Sometimes a centroid wins no points. We restart it at a random data point.

`Centroids.Dists` finds the distance from one vector to every centroid. It uses a dot product.

It handles four centroids per pass, so each loaded number is used four times.

## ivf.go: the index

IVF means inverted file. The name comes from search engines.

Training runs k-means on a sample. This gives `nlist` centroids.

Each centroid owns one list. A list holds the vectors closest to its centroid.

`Add` finds the nearest centroid for each new vector. It appends the vector to that list.

`Add` can run again after training. New vectors go into the right list.

`Search` has two steps. First it ranks the centroids by distance to the query.

Second it scans only the closest `nprobe` lists. It computes exact distances there.

A small `nprobe` is fast. It can miss a neighbour that sits in a list nobody scanned.

A large `nprobe` is slow and accurate. If `nprobe` equals `nlist`, every list is scanned.

Then the answer equals brute force. A test checks that ids and distances match exactly.

Many goroutines can call `Search` at the same time. `Add` and `Train` must run alone.

## simd_arm64.s: why there is assembly

A plain Go loop handles about one number per cycle. The Go compiler does not use vector instructions on its own.

Apple chips have NEON. One NEON instruction works on four numbers at once.

So the hot loops are written directly in arm64 assembly. There are three of them.

`l2sqNEON` computes one squared distance. `l2sq4NEON` computes four at once against one query.

`dot4NEON` computes the dot product of one vector with four centroids.

Each kernel has a plain Go twin in `distance.go`. The tests compare the two on many lengths.

The four-row kernel gives the same bits as the one-row kernel. A test demands exact equality.

On other CPUs the plain Go loops run instead. The build tag `purego` forces that path.

## The tests

Go tests sit next to the code in `engine/`. They run with the race detector on.

They check that full-probe search equals brute force exactly. They check that a stored vector finds itself first.

They check that vectors added after training are found. They check that recall grows as `nprobe` grows.

They check the assembly against the plain loops. They check that the worker count never changes a result.

## The benchmark

`bench/fetch.py` downloads the SIFT dataset. It checks the file size first.

`bench/export_data.py` writes the vectors and the true neighbours as raw files. Faiss and Go then read the same data.

`bench/compare.py` builds `faiss.IndexIVFFlat` on those vectors.

`engine/cmd/vcbench` builds the Go index on the same vectors. It does all its own timing.

Both use the same `nlist` and the same number of k-means iterations.

Recall is the share of the true 10 nearest neighbours that an index returns.

The true neighbours come from `faiss.IndexFlatL2`. The Go exact index is checked against them.

Queries per second are measured one query per call. Faiss is limited to one thread.

Each speed is the fastest of seven passes. Other programs can only slow a pass down.

Write speed is measured by adding vectors in batches of 1, 100, 1000 and 10000.

Every result goes into a CSV in `results/`. `bench/plot.py` draws the charts from those files.

## What this project does not claim

It does not claim a new idea. It does not claim to beat Faiss.

Faiss is written in optimised C++ with years of tuning. Our engine is small and readable.

The goal is to show that the method is understood and the results are correct.
