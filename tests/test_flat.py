import numpy as np

from vecstore.flat import FlatIndex, topk


def make_data(n=500, dim=16, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim)).astype(np.float32)


def slow_search(x, q, k):
    """Obvious loop version. Slow but easy to trust."""
    out = []
    for qi in q:
        d = [float(np.sum((xi.astype(np.float64) - qi) ** 2)) for xi in x]
        out.append(np.argsort(d, kind="stable")[:k])
    return np.array(out)


def test_matches_slow_loop():
    x, q = make_data(), make_data(20, seed=1)
    index = FlatIndex(16)
    index.add(x)
    _, ids = index.search(q, 10)
    assert np.array_equal(ids, slow_search(x, q, 10))


def test_stored_vector_finds_itself_first():
    x = make_data()
    index = FlatIndex(16)
    index.add(x)
    d, ids = index.search(x[123], 5)
    assert ids[0, 0] == 123
    assert d[0, 0] < 1e-4


def test_distances_are_sorted():
    x, q = make_data(), make_data(5, seed=2)
    index = FlatIndex(16)
    index.add(x)
    d, _ = index.search(q, 10)
    assert np.all(np.diff(d, axis=1) >= 0)


def test_add_twice_keeps_ids_in_order():
    x = make_data()
    index = FlatIndex(16)
    index.add(x[:200])
    index.add(x[200:])
    assert index.ntotal == 500
    _, ids = index.search(x[450], 1)
    assert ids[0, 0] == 450


def test_fewer_vectors_than_k_is_padded():
    index = FlatIndex(16)
    index.add(make_data(3))
    d, ids = index.search(make_data(1, seed=5), 5)
    assert list(ids[0, 3:]) == [-1, -1]
    assert np.isinf(d[0, 3:]).all()


def test_topk_breaks_ties_by_id():
    dist = np.array([[1.0, 1.0, 1.0, 0.5]], dtype=np.float32)
    _, ids = topk(dist, np.array([7, 3, 5, 9]), 3)
    assert list(ids[0]) == [9, 3, 5]
