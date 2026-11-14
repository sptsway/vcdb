import numpy as np

from vecstore.kmeans import assign, kmeans


def blobs(n=2000, dim=8, k=8, seed=0):
    """Overlapping Gaussian groups, so k-means needs several rounds."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, dim)) * 2.0
    pick = rng.integers(0, k, n)
    return (centers[pick] + rng.standard_normal((n, dim))).astype(np.float32)


def test_loss_falls_over_iterations():
    _, losses = kmeans(blobs(), k=8, iters=15, seed=0)
    assert losses[-1] < losses[0]
    # Lloyd's loss never rises. The tolerance covers float rounding only.
    for before, after in zip(losses, losses[1:]):
        assert after <= before * (1 + 1e-6)


def test_same_seed_gives_same_result():
    x = blobs()
    c1, l1 = kmeans(x, k=8, iters=5, seed=3)
    c2, l2 = kmeans(x, k=8, iters=5, seed=3)
    assert np.array_equal(c1, c2)
    assert l1 == l2


def test_assign_picks_the_nearest_centroid():
    x = blobs(200)
    centroids = x[:5].copy()
    labels, dists = assign(x, centroids)
    for i in range(len(x)):
        true = ((x[i] - centroids) ** 2).sum(axis=1)
        assert labels[i] == true.argmin()
        assert abs(dists[i] - true.min()) < 1e-3


def test_empty_clusters_are_restarted():
    # Twenty identical points. All centroids start on the same spot, so at
    # least two of the three win no points. The run must still stay finite.
    x = np.ones((20, 4), dtype=np.float32)
    centroids, losses = kmeans(x, k=3, iters=3, seed=0)
    assert np.isfinite(centroids).all()
    assert losses == [0.0, 0.0, 0.0]
