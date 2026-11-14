# Walkthrough

This page explains the code in plain sentences. Read it next to the source files.

## The problem

A vector is a list of numbers. Here each vector has 128 numbers.

Two vectors are close when the distance between them is small.

Nearest-neighbour search takes a query vector. It returns the k stored vectors closest to it.

The simple way is to compare the query with every stored vector. That is exact but slow.

An index makes search faster. It gives up a little accuracy to look at fewer vectors.

## flat.py: the exact search

`FlatIndex` stores every vector in one array. Its id is its row number.

Distance is the squared L2 distance. The square root is skipped because it never changes the order.

The function `sq_l2` expands the distance as |q|² − 2 q·x + |x|². One matrix product then does the work.

Rounding can push a tiny distance just below zero. We clip it back to zero.

The function `topk` keeps the k smallest distances in each row. It uses `argpartition`, which avoids a full sort.

Ties are broken by the smaller id. This makes results repeatable.

If fewer than k vectors exist, the result is padded. Distance becomes infinity and the id becomes −1.

`search` handles queries in blocks. This keeps the distance table near 256 MB.

This index is slow on big data. Its job is to be obviously correct.

Every other index is tested against it.

## kmeans.py: finding the regions

K-means groups vectors around k centres. Each centre is called a centroid.

It repeats two steps. First, assign each vector to its nearest centroid.

Second, move each centroid to the average of the vectors assigned to it.

The loss is the total squared distance from each vector to its centroid.

The loss can only stay equal or fall after each round. A test checks this.

The function `assign` finds the nearest centroid for each row. It works in blocks of 8192 rows.

The starting centroids are random data points. A fixed seed picks them.

The same seed gives the same result every time. A test checks this too.

Sometimes a centroid wins no points. We restart it at a random data point.

## ivf.py: the index

IVF means inverted file. The name comes from search engines.

Training runs k-means on a sample. This gives `nlist` centroids.

Each centroid owns one list. A list holds the vectors closest to its centroid.

`add` finds the nearest centroid for each new vector. It appends the vector to that list.

`add` can run again after training. New vectors go into the right list.

Each list keeps three arrays: the vectors, their ids and their squared norms.

The norms are saved at add time. Search then skips recomputing them.

`search` has two steps. First it ranks the centroids by distance to the query.

Second it scans only the closest `nprobe` lists. It computes exact distances there.

A small `nprobe` is fast. It can miss a neighbour that sits in a list nobody scanned.

A large `nprobe` is slow and accurate. If `nprobe` equals `nlist`, every list is scanned.

Then the answer equals brute force. A test checks that the ids match exactly.

## The tests

`test_flat.py` compares the flat index with a plain Python loop.

`test_kmeans.py` checks that the loss falls and that seeds repeat.

`test_ivf.py` checks full-probe equality with the flat index.

It checks that a stored vector finds itself first. It checks that later adds are found.

It also checks that recall grows as `nprobe` grows.

## The benchmark

`bench/fetch.py` downloads the SIFT dataset. It checks the file size first.

`bench/compare.py` builds our index and `faiss.IndexIVFFlat` on the same vectors.

Both use the same `nlist` and the same number of k-means iterations.

Recall is the share of the true 10 nearest neighbours that an index returns.

The true neighbours come from our own flat index. We cross-check them against Faiss.

Queries per second are measured one query per call. Faiss is limited to one thread.

Write speed is measured by adding vectors in batches of 1, 100, 1000 and 10000.

Every result goes into a CSV in `results/`. `bench/plot.py` draws the charts from those files.

## What this project does not claim

It does not claim a new idea. It does not claim to beat Faiss.

Faiss is written in optimised C++. Ours is Python and NumPy, and it is slower.

The goal is to show that the method is understood and the results are correct.
