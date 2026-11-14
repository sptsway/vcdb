"""Lloyd's k-means, written from scratch.

Each round has two steps.
  1. Assign every point to its nearest centroid.
  2. Move every centroid to the mean of the points assigned to it.
The total squared distance (the loss) can only stay equal or fall.
"""
import numpy as np


def assign(x, centroids, block=8192):
    """Return (label, squared distance) of the nearest centroid for each row of x.

    Works in blocks of rows so the distance matrix stays small.
    """
    c_norms = np.einsum("ij,ij->i", centroids, centroids)
    labels = np.empty(len(x), dtype=np.int64)
    dists = np.empty(len(x), dtype=np.float32)
    for i in range(0, len(x), block):
        xb = x[i : i + block]
        # |x - c|^2 = |x|^2 - 2 x.c + |c|^2. The |x|^2 term is the same for
        # every centroid, so it is left out while picking the nearest one.
        partial = c_norms[None, :] - 2.0 * (xb @ centroids.T)
        j = partial.argmin(axis=1)
        labels[i : i + block] = j
        x_norms = np.einsum("ij,ij->i", xb, xb)
        best = partial[np.arange(len(xb)), j] + x_norms
        dists[i : i + block] = np.maximum(best, 0.0)
    return labels, dists


def kmeans(x, k, iters=20, seed=0):
    """Cluster x into k groups. Returns (centroids, loss per iteration).

    The seed fixes the starting centroids, so a run can be repeated exactly.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, dim = x.shape
    rng = np.random.default_rng(seed)
    # Start from k distinct random points of the data.
    centroids = x[rng.choice(n, size=k, replace=False)].copy()
    losses = []
    for _ in range(iters):
        labels, dists = assign(x, centroids)
        losses.append(float(dists.sum(dtype=np.float64)))

        counts = np.bincount(labels, minlength=k)
        sums = np.empty((k, dim), dtype=np.float64)
        for d in range(dim):  # one bincount per dimension adds up the members
            sums[:, d] = np.bincount(labels, weights=x[:, d], minlength=k)
        new = sums / np.maximum(counts, 1)[:, None]

        # A centroid that lost all its points restarts at a random data point.
        empty = counts == 0
        if empty.any():
            new[empty] = x[rng.choice(n, size=int(empty.sum()), replace=False)]
        centroids = new.astype(np.float32)
    return centroids, losses
