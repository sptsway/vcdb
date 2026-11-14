"""Brute-force search. This is the correctness oracle for every other index.

Distance is squared L2. We never take the square root because it does not
change the order of neighbours.
"""
import numpy as np


def sq_l2(q, x, x_norms):
    """Squared L2 from each row of q to each row of x.

    Uses |q - x|^2 = |q|^2 - 2 q.x + |x|^2 so one matrix product does the work.
    """
    q_norms = np.einsum("ij,ij->i", q, q)
    d = q_norms[:, None] - 2.0 * (q @ x.T) + x_norms[None, :]
    return np.maximum(d, 0.0)  # rounding can push a tiny distance below zero


def topk(dist, ids, k):
    """Pick the k smallest distances in each row. Ties go to the smaller id.

    dist is (nq, m). ids is (m,) or (nq, m). Rows with fewer than k
    candidates are padded with distance inf and id -1.
    """
    ids = np.broadcast_to(ids, dist.shape)
    if dist.shape[1] > k:
        # argpartition finds the k smallest without sorting everything.
        part = np.argpartition(dist, k - 1, axis=1)[:, :k]
        dist = np.take_along_axis(dist, part, axis=1)
        ids = np.take_along_axis(ids, part, axis=1)
    order = np.lexsort((ids, dist), axis=1)  # sort by distance, then id
    dist = np.take_along_axis(dist, order, axis=1)
    ids = np.take_along_axis(ids, order, axis=1)
    short = k - dist.shape[1]
    if short > 0:
        dist = np.pad(dist, ((0, 0), (0, short)), constant_values=np.inf)
        ids = np.pad(ids, ((0, 0), (0, short)), constant_values=-1)
    return dist.astype(np.float32), ids.astype(np.int64)


class FlatIndex:
    def __init__(self, dim):
        self.dim = dim
        self.x = np.zeros((0, dim), dtype=np.float32)
        self.norms = np.zeros(0, dtype=np.float32)

    @property
    def ntotal(self):
        return len(self.x)

    def add(self, x):
        """Append vectors. Their ids are their positions: 0, 1, 2, ..."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        self.x = np.vstack([self.x, x])
        self.norms = np.concatenate([self.norms, np.einsum("ij,ij->i", x, x)])

    def search(self, q, k):
        """Return (distances, ids), each of shape (nq, k)."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float32))
        # Handle queries in blocks so the distance matrix stays near 256 MB.
        block = max(1, 2**26 // max(self.ntotal, 1))
        dists, ids = [], []
        for i in range(0, len(q), block):
            d = sq_l2(q[i : i + block], self.x, self.norms)
            all_ids = np.arange(self.ntotal)
            bd, bi = topk(d, all_ids, k)
            dists.append(bd)
            ids.append(bi)
        return np.vstack(dists), np.vstack(ids)
