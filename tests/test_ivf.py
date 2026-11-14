import numpy as np
import pytest

from vecstore.flat import FlatIndex
from vecstore.ivf import IVFFlat

DIM, NLIST = 16, 16


def make_data(n=3000, seed=0):
    """Gaussian groups, so the lists actually mean something."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((NLIST, DIM)) * 3.0
    pick = rng.integers(0, NLIST, n)
    return (centers[pick] + rng.standard_normal((n, DIM))).astype(np.float32)


def build(x, nlist=NLIST):
    index = IVFFlat(DIM, nlist, seed=0)
    index.train(x)
    index.add(x)
    return index


def oracle(x):
    flat = FlatIndex(DIM)
    flat.add(x)
    return flat


def recall(found, truth):
    hits = sum(len(set(f) & set(t)) for f, t in zip(found, truth))
    return hits / truth.size


def test_full_probe_equals_flat_exactly():
    x, q = make_data(), make_data(100, seed=1)
    d_ivf, i_ivf = build(x).search(q, 10, nprobe=NLIST)
    d_flat, i_flat = oracle(x).search(q, 10)
    assert np.array_equal(i_ivf, i_flat)
    assert np.allclose(d_ivf, d_flat, rtol=1e-4, atol=1e-4)


def test_stored_vector_finds_itself_first():
    x = make_data()
    index = build(x)
    for nprobe in (1, NLIST):
        for row in (0, 17, 1234, 2999):
            d, ids = index.search(x[row], 3, nprobe)
            assert ids[0, 0] == row
            assert d[0, 0] < 1e-3


def test_vectors_added_after_training_are_found():
    x = make_data()
    index = IVFFlat(DIM, NLIST, seed=0)
    index.train(x[:1500])  # learn the lists from half the data
    index.add(x[:1500])
    index.add(x[1500:])  # a second add, later
    assert index.ntotal == 3000
    q = make_data(50, seed=2)
    _, i_ivf = index.search(q, 10, nprobe=NLIST)
    _, i_flat = oracle(x).search(q, 10)
    assert np.array_equal(i_ivf, i_flat)


def test_recall_grows_with_nprobe():
    x, q = make_data(), make_data(200, seed=3)
    index = build(x)
    _, truth = oracle(x).search(q, 10)
    scores = [recall(index.search(q, 10, p)[1], truth) for p in (1, 4, NLIST)]
    assert scores[0] < scores[1] <= scores[2]
    assert scores[2] == 1.0


def test_custom_ids_are_returned():
    x = make_data(500)
    index = IVFFlat(DIM, 4, seed=0)
    index.train(x)
    index.add(x, ids=np.arange(500) + 10_000)
    _, ids = index.search(x[5], 1, nprobe=4)
    assert ids[0, 0] == 10_005


def test_empty_lists_are_skipped():
    x = make_data(40)
    index = IVFFlat(DIM, 32, seed=0)
    index.train(x)
    index.add(x[:10])  # 10 vectors over 32 lists leaves at least 22 empty
    assert sum(len(i) == 0 for i in index.ids) >= 22
    _, ids = index.search(x[:5], 5, nprobe=32)
    _, truth = oracle(x[:10]).search(x[:5], 5)
    assert np.array_equal(ids, truth)


def test_search_before_train_is_an_error():
    index = IVFFlat(DIM, NLIST)
    with pytest.raises(RuntimeError):
        index.add(make_data(10))
    with pytest.raises(RuntimeError):
        index.search(make_data(1), 1, 1)
