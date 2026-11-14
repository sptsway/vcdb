# Environment

Checked on 14 November 2026. Every version below was printed by the tool itself.

| Item | Value |
|---|---|
| Machine | Apple M4 Pro, 12 cores, 24 GB RAM |
| OS | macOS (Darwin 27.0.0) |
| GPU | none. CPU only |
| Python | 3.13.17 (virtualenv in `.venv/`) |
| numpy | 2.5.4 |
| faiss-cpu | 1.16.0 (baseline only. No Faiss code is copied) |
| pytest | 9.2.0 |
| h5py | 3.16.0 |
| matplotlib | 3.11.2 |

## Dataset

`sift-128-euclidean` from ann-benchmarks.

- URL: http://ann-benchmarks.com/sift-128-euclidean.hdf5
- Checked with a HEAD request: HTTP 200, 525,128,288 bytes (about 500 MiB).
- Contents: 1,000,000 train vectors, 10,000 query vectors, 128 dimensions.

## Memory cap

macOS does not enforce `ulimit -v`. The cap of 8 GB is checked instead.
Each benchmark run records its peak resident memory in `results/run_info.csv`.

## Setup

```
make setup
make test
```
