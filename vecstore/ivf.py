"""IVF-Flat: an inverted-file index with exact distances inside each list.

Training splits space into nlist regions with k-means. Each vector is stored
in the list of its nearest centroid. A search looks only at the nprobe lists
whose centroids are closest to the query. More lists means better recall and
lower speed. With nprobe = nlist the search is exact.
"""
import numpy as np

from vecstore.flat import topk
from vecstore.kmeans import assign, kmeans


class IVFFlat:
    def __init__(self, dim, nlist, seed=0):
        self.dim = dim
        self.nlist = nlist
        self.seed = seed
        self.centroids = None
        self.ntotal = 0
        # One block of vectors, ids and squared norms per list.
        self.vecs = [np.zeros((0, dim), np.float32) for _ in range(nlist)]
        self.ids = [np.zeros(0, np.int64) for _ in range(nlist)]
        self.norms = [np.zeros(0, np.float32) for _ in range(nlist)]

    def train(self, x, iters=20):
        """Learn the list centroids from a sample of vectors."""
        self.centroids, self.train_losses = kmeans(x, self.nlist, iters, self.seed)
        self.centroid_norms = np.einsum("ij,ij->i", self.centroids, self.centroids)

    def add(self, x, ids=None):
        """Store vectors. Can be called any number of times after training."""
        if self.centroids is None:
            raise RuntimeError("train the index before adding vectors")
        x = np.ascontiguousarray(x, dtype=np.float32)
        if ids is None:
            ids = np.arange(self.ntotal, self.ntotal + len(x), dtype=np.int64)
        ids = np.asarray(ids, dtype=np.int64)

        labels, _ = assign(x, self.centroids)
        # Sort by list so each list's new vectors sit together.
        order = np.argsort(labels, kind="stable")
        cuts = np.cumsum(np.bincount(labels, minlength=self.nlist))
        start = 0
        for lst in range(self.nlist):
            rows = order[start : cuts[lst]]
            start = cuts[lst]
            if len(rows) == 0:
                continue
            new = x[rows]
            self.vecs[lst] = np.vstack([self.vecs[lst], new])
            self.ids[lst] = np.concatenate([self.ids[lst], ids[rows]])
            self.norms[lst] = np.concatenate(
                [self.norms[lst], np.einsum("ij,ij->i", new, new)]
            )
        self.ntotal += len(x)

    def search(self, q, k, nprobe):
        """Return (distances, ids), each of shape (nq, k)."""
        if self.centroids is None:
            raise RuntimeError("train the index before searching")
        q = np.atleast_2d(np.asarray(q, dtype=np.float32))
        nprobe = min(nprobe, self.nlist)

        # Step 1: rank the centroids for every query at once.
        cd = self.centroid_norms[None, :] - 2.0 * (q @ self.centroids.T)
        if nprobe < self.nlist:
            probe = np.argpartition(cd, nprobe - 1, axis=1)[:, :nprobe]
        else:
            probe = np.tile(np.arange(self.nlist), (len(q), 1))

        # Step 2: scan the chosen lists one query at a time.
        out_d, out_i = [], []
        for qi, lists in zip(q, probe):
            dist, found = [], []
            for lst in lists:
                if len(self.ids[lst]) == 0:
                    continue
                dist.append(self.norms[lst] - 2.0 * (self.vecs[lst] @ qi))
                found.append(self.ids[lst])
            if dist:
                dist = np.concatenate(dist)[None, :] + float(qi @ qi)
                found = np.concatenate(found)
            else:
                dist, found = np.zeros((1, 0), np.float32), np.zeros(0, np.int64)
            d, i = topk(np.maximum(dist, 0.0), found, k)
            out_d.append(d)
            out_i.append(i)
        return np.vstack(out_d), np.vstack(out_i)
