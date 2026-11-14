"""Download sift-128-euclidean from ann-benchmarks into data/.

The URL and size were checked with a HEAD request on 14 Nov 2026.
The script checks the size again before and after the download.
"""
import sys
import urllib.request
from pathlib import Path

URL = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
DEST = Path(__file__).resolve().parent.parent / "data" / "sift-128-euclidean.hdf5"
LIMIT = 5 * 1024**3  # the project rule: never fetch more than 5 GB unasked
# The server answers 403 to Python's default User-Agent, so we name ourselves.
HEADERS = {"User-Agent": "vcdb-fetch/1.0"}


def main():
    head = urllib.request.urlopen(
        urllib.request.Request(URL, method="HEAD", headers=HEADERS)
    )
    size = int(head.headers["Content-Length"])
    print(f"remote size: {size:,} bytes")
    if size > LIMIT:
        sys.exit("file is over 5 GB, refusing to download")
    if DEST.exists() and DEST.stat().st_size == size:
        print(f"already have {DEST}")
        return
    DEST.parent.mkdir(exist_ok=True)

    def progress(blocks, block_size, total):
        done = min(blocks * block_size, total)
        print(f"\r{done / total:6.1%}", end="", flush=True)

    opener = urllib.request.build_opener()
    opener.addheaders = list(HEADERS.items())
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(URL, DEST, progress)
    print()
    got = DEST.stat().st_size
    if got != size:
        sys.exit(f"size mismatch: expected {size}, got {got}")
    print(f"saved {DEST} ({got:,} bytes)")


if __name__ == "__main__":
    main()
