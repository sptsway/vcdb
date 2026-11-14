"""Compare vecstore's IVF-Flat with faiss.IndexIVFFlat on sift-128-euclidean.

Writes CSVs into results/. Nothing here is typed in by hand.

Fairness rules:
  * Both indexes train on the same vectors with the same nlist and the
    same number of k-means iterations.
  * Query speed is measured one query per call. Faiss is held to one thread.
    Each figure is the fastest of several passes. Other programs on the
    machine can only slow a pass down, never speed it up.
  * Ground truth comes from vecstore's own flat index. It is cross-checked
    against faiss.IndexFlatL2 on a sample of queries.
"""
import argparse
import csv
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import faiss
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vecstore.flat import FlatIndex  # noqa: E402
from vecstore.ivf import IVFFlat  # noqa: E402

K = 10
NPROBES = [1, 4, 16, 64]
WRITE_BATCHES = [1, 100, 1000, 10000]
RESULTS = ROOT / "results"
MAX_THREADS = faiss.omp_get_max_threads()  # read before we pin faiss to one


def write_csv(name, header, rows):
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote results/{name}")


def recall_at_k(found, truth):
    """Share of the true top-k ids that the index returned."""
    hits = sum(len(set(f) & set(t)) for f, t in zip(found, truth))
    return hits / truth.size


def ground_truth(train, queries, tag):
    """Exact neighbours from vecstore's flat index, cached on disk."""
    cache = ROOT / "data" / f"gt_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    flat = FlatIndex(train.shape[1])
    flat.add(train)
    _, ids = flat.search(queries, K)
    np.save(cache, ids)
    return ids


def faiss_agreement(train, queries, truth, sample=1000):
    """How often vecstore's flat ids equal faiss.IndexFlatL2 on a sample."""
    ref = faiss.IndexFlatL2(train.shape[1])
    ref.add(train)
    _, ids = ref.search(queries[:sample], K)
    return recall_at_k(truth[:sample], ids)


def time_queries(search_one, queries, repeats=7, warmup=200):
    """Best queries per second over several passes of one-query calls."""
    for q in queries[:warmup]:
        search_one(q)
    rates = []
    for _ in range(repeats):
        t = time.perf_counter()
        for q in queries:
            search_one(q)
        rates.append(len(queries) / (time.perf_counter() - t))
    return max(rates)


def time_adds(add_fn, extra, batch, total):
    """Insert `total` vectors in batches of `batch`. Returns vectors per second."""
    t = time.perf_counter()
    for i in range(0, total, batch):
        add_fn(extra[i : i + batch])
    return total / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000, help="vectors to index")
    ap.add_argument("--nlist", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20, help="k-means iterations")
    ap.add_argument("--queries", type=int, default=10_000)
    ap.add_argument("--tag", default=None, help="file suffix, e.g. 200k")
    args = ap.parse_args()
    tag = args.tag or ("1m" if args.n >= 1_000_000 else f"{args.n // 1000}k")

    with h5py.File(ROOT / "data" / "sift-128-euclidean.hdf5") as f:
        full = np.array(f["train"])
        queries = np.array(f["test"])[: args.queries]
    train = np.ascontiguousarray(full[: args.n])
    dim = train.shape[1]
    print(f"run {tag}: n={args.n} nlist={args.nlist} queries={len(queries)}")

    # Vectors used only for the insert benchmark. They are not in the index.
    extra = full[args.n : args.n + 30_000]
    if len(extra) < 30_000:
        extra = full[:30_000]

    print("ground truth")
    truth = ground_truth(train, queries, tag)
    agree = faiss_agreement(train, queries, truth)
    print(f"  agreement with faiss.IndexFlatL2: {agree:.4f}")

    # ---- build both indexes -------------------------------------------------
    print("build vecstore")
    mine = IVFFlat(dim, args.nlist, seed=0)
    t = time.perf_counter()
    mine.train(train, iters=args.iters)
    mine_train = time.perf_counter() - t
    t = time.perf_counter()
    mine.add(train)
    mine_add = time.perf_counter() - t

    print("build faiss")
    quant = faiss.IndexFlatL2(dim)
    ref = faiss.IndexIVFFlat(quant, dim, args.nlist)
    ref.cp.niter = args.iters
    ref.cp.seed = 0
    ref.cp.max_points_per_centroid = 10**6  # use every vector, no subsampling
    t = time.perf_counter()
    ref.train(train)
    ref_train = time.perf_counter() - t
    t = time.perf_counter()
    ref.add(train)
    ref_add = time.perf_counter() - t

    write_csv(
        f"build_{tag}.csv",
        ["impl", "n", "nlist", "train_s", "add_s", "add_vectors_per_s"],
        [
            ["vecstore", args.n, args.nlist, f"{mine_train:.2f}", f"{mine_add:.2f}",
             f"{args.n / mine_add:.0f}"],
            ["faiss", args.n, args.nlist, f"{ref_train:.2f}", f"{ref_add:.2f}",
             f"{args.n / ref_add:.0f}"],
        ],
    )
    write_csv(
        f"kmeans_loss_{tag}.csv",
        ["iteration", "loss"],
        [[i + 1, f"{v:.1f}"] for i, v in enumerate(mine.train_losses)],
    )

    # ---- read benchmark -----------------------------------------------------
    print("queries")
    rows = []
    for nprobe in NPROBES:
        ref.nprobe = nprobe
        _, i_mine = mine.search(queries, K, nprobe)
        faiss.omp_set_num_threads(1)
        _, i_ref = ref.search(queries, K)
        qps_mine = time_queries(lambda q: mine.search(q, K, nprobe), queries)
        qps_ref = time_queries(lambda q: ref.search(q[None, :], K), queries)
        faiss.omp_set_num_threads(MAX_THREADS)
        qps_batch = max(
            _timed(lambda: ref.search(queries, K), len(queries)) for _ in range(7)
        )
        for name, ids, qps in [
            ("vecstore", i_mine, qps_mine),
            ("faiss", i_ref, qps_ref),
            ("faiss_batch", i_ref, qps_batch),
        ]:
            rows.append([name, args.n, args.nlist, nprobe,
                         f"{recall_at_k(ids, truth):.4f}", f"{qps:.0f}"])
        print(f"  nprobe={nprobe:>3}  vecstore {qps_mine:8.0f} qps"
              f"  faiss {qps_ref:8.0f} qps  recall {recall_at_k(i_mine, truth):.4f}")
    write_csv(f"recall_qps_{tag}.csv",
              ["impl", "n", "nlist", "nprobe", "recall_at_10", "qps"], rows)

    # ---- write benchmark ----------------------------------------------------
    print("inserts")
    faiss.omp_set_num_threads(1)
    rows, used = [], 0
    for batch in WRITE_BATCHES:
        total = max(batch, 5000)
        chunk = np.ascontiguousarray(extra[used : used + total])
        used += total
        rows.append(["vecstore", batch, total,
                     f"{time_adds(mine.add, chunk, batch, total):.0f}"])
        rows.append(["faiss", batch, total,
                     f"{time_adds(ref.add, chunk, batch, total):.0f}"])
        print(f"  batch={batch:>5}  vecstore {rows[-2][3]:>8} v/s  faiss {rows[-1][3]:>8} v/s")
    # The same inserts again with faiss free to use every core.
    faiss.omp_set_num_threads(MAX_THREADS)
    for batch in WRITE_BATCHES:
        total = max(batch, 5000)
        chunk = np.ascontiguousarray(extra[:total])
        rows.append(["faiss_batch", batch, total,
                     f"{time_adds(ref.add, chunk, batch, total):.0f}"])
    write_csv(f"write_qps_{tag}.csv",
              ["impl", "batch_size", "vectors", "vectors_per_s"], rows)

    # ---- run information ----------------------------------------------------
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True).stdout.strip()
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # bytes on macOS
    header = ["run", "n", "nlist", "kmeans_iters", "queries", "python", "numpy",
              "faiss", "cpu", "peak_rss_mb", "flat_agreement_with_faiss"]
    row = [tag, args.n, args.nlist, args.iters, len(queries), platform.python_version(),
           np.__version__, faiss.__version__, cpu, f"{peak_mb:.0f}", f"{agree:.4f}"]
    info = RESULTS / "run_info.csv"
    old = []
    if info.exists():
        with open(info, newline="") as f:
            old = [r for r in list(csv.reader(f))[1:] if r[0] != tag]
    write_csv("run_info.csv", header, old + [row])
    print(f"peak memory {peak_mb:.0f} MB (cap is 8192 MB)")


def _timed(fn, count):
    t = time.perf_counter()
    fn()
    return count / (time.perf_counter() - t)


if __name__ == "__main__":
    main()
