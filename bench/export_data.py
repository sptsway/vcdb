"""Write the benchmark data as raw little-endian binary files for the Go benchmark.

Files go to data/export/<tag>/:
  train.f32    n x 128 float32 vectors to index
  queries.f32  query vectors
  gt.i32       10 true nearest-neighbour ids per query (from vecstore's flat oracle)
  extra.f32    30,000 vectors used only for the insert benchmark

The Python and Go benchmarks therefore read the same vectors and the same
ground truth.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vecstore.flat import FlatIndex  # noqa: E402

K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--queries", type=int, default=10_000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or ("1m" if args.n >= 1_000_000 else f"{args.n // 1000}k")

    with h5py.File(ROOT / "data" / "sift-128-euclidean.hdf5") as f:
        full = np.array(f["train"])
        queries = np.ascontiguousarray(np.array(f["test"])[: args.queries])
    train = np.ascontiguousarray(full[: args.n])
    # Same rule as bench/compare.py, so both benchmarks insert the same vectors.
    extra = full[args.n : args.n + 30_000]
    if len(extra) < 30_000:
        extra = full[:30_000]

    cache = ROOT / "data" / f"gt_{tag}.npy"
    if cache.exists():
        gt = np.load(cache)
    else:
        flat = FlatIndex(train.shape[1])
        flat.add(train)
        _, gt = flat.search(queries, K)
        np.save(cache, gt)

    out = ROOT / "data" / "export" / tag
    out.mkdir(parents=True, exist_ok=True)
    train.astype("<f4").tofile(out / "train.f32")
    queries.astype("<f4").tofile(out / "queries.f32")
    np.ascontiguousarray(extra).astype("<f4").tofile(out / "extra.f32")
    gt[: len(queries)].astype("<i4").tofile(out / "gt.i32")
    print(f"wrote {out}: {len(train)} vectors, {len(queries)} queries, {K} neighbours each")


if __name__ == "__main__":
    main()
