"""Exact nearest neighbours from faiss.IndexFlatL2, cached on disk.

Faiss is the reference library here, so the ground truth does not depend on
any code in this repository. The Go engine's own exact index is checked
against it: bench/export_data.py writes the ids to disk and vcbench reports
how much of them its brute-force search reproduces.
"""
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
K = 10


def ground_truth(train, queries, tag):
    """Ids of the K true nearest neighbours of each query, shape (nq, K)."""
    cache = ROOT / "data" / f"gt_{tag}.npy"
    if cache.exists():
        ids = np.load(cache)
        if len(ids) >= len(queries):
            return ids[: len(queries)]
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    _, ids = index.search(queries, K)
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, ids)
    return ids
