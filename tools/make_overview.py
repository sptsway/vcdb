"""Render OVERVIEW.md from status.csv and results/*.csv.

Also refreshes the results tables and headline numbers inside README.md,
between the <!-- name:start --> and <!-- name:end --> markers.

Nothing is typed by hand into the results sections. If a CSV is missing the
section says "no result yet".
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALID = {"planned", "in progress", "done"}

# (heading, csv path). Python and Faiss results first, then the Go engine's.
RESULTS = []
for _tag, _size in (("200k", "200k vectors"), ("1m", "1M vectors")):
    for _prefix, _who in (("", "Python and Faiss"), ("go_", "Go engine")):
        RESULTS += [
            (f"Recall and queries per second, {_size}, {_who}", f"results/{_prefix}recall_qps_{_tag}.csv"),
            (f"Build time, {_size}, {_who}", f"results/{_prefix}build_{_tag}.csv"),
            (f"Insert throughput by batch size, {_size}, {_who}", f"results/{_prefix}write_qps_{_tag}.csv"),
        ]
RESULTS += [
    ("Run environment, Python and Faiss", "results/run_info.csv"),
    ("Run environment, Go engine", "results/go_run_info.csv"),
]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def md_table(rows):
    head, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def status_section():
    rows = read_csv(ROOT / "status.csv")[1:]
    out = ["| Item | Status | File |", "|---|---|---|"]
    for item, status, file in rows:
        if status not in VALID:
            sys.exit(f"bad status {status!r} for {item!r}")
        # A row may only say "done" if its file exists.
        if status == "done" and not (ROOT / file).exists():
            sys.exit(f"{item!r} is done but {file} is missing")
        out.append(f"| {item} | {status} | {('`' + file + '`') if file else ''} |")
    return "\n".join(out)


def results_section():
    out = []
    for heading, rel in RESULTS:
        out.append(f"### {heading}\n")
        path = ROOT / rel
        out.append(md_table(read_csv(path)) if path.exists() else "no result yet")
        out.append("")
    return "\n".join(out)


def load(name):
    """Read a results CSV as a list of dicts. None if the file is missing."""
    path = ROOT / "results" / name
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_all(stem, tag):
    """Python/Faiss rows and Go rows for one run, joined."""
    return (load(f"{stem}_{tag}.csv") or []) + (load(f"go_{stem}_{tag}.csv") or [])


def tags():
    """Result sets present on disk, smallest first: ['200k', '1m']."""
    return [t for t in ("200k", "1m") if load(f"recall_qps_{t}.csv")]


def title_for(tag):
    row = load(f"build_{tag}.csv")[0]
    return f"{int(row['n']):,} vectors, nlist {row['nlist']}"


# CSV key -> short name in the tables. Order is the column order.
NAMES = {"vecstore_go": "Go", "vecstore": "Python", "faiss": "faiss"}
SINGLE = list(NAMES)


def present(rows):
    return [k for k in SINGLE if any(r["impl"] == k for r in rows)]


def value(rows, impl, col, **where):
    for r in rows:
        if r["impl"] == impl and all(r[c] == v for c, v in where.items()):
            return float(r[col])
    return None


def speed_table(tag):
    rows = load_all("recall_qps", tag)
    impls = present(rows)
    nprobes = sorted({r["nprobe"] for r in rows}, key=int)
    out = ["Recall@10", "",
           "| nprobe | " + " | ".join(NAMES[k] for k in impls) + " |",
           "|---|" + "---|" * len(impls)]
    for n in nprobes:
        out.append(f"| {n} | " + " | ".join(
            f"{value(rows, k, 'recall_at_10', nprobe=n):.4f}" for k in impls) + " |")

    cols = impls + [k for k in ("vecstore_go_batch", "faiss_batch") if any(r["impl"] == k for r in rows)]
    label = {**NAMES, "vecstore_go_batch": "Go batched, all cores", "faiss_batch": "faiss batched, all cores"}
    head = [label[k] for k in cols]
    extra = []
    if "vecstore_go" in impls and "faiss" in impls:
        extra.append("faiss speed over Go")
    if "vecstore_go" in impls and "vecstore" in impls:
        extra.append("Go speed over Python")
    out += ["", "Queries per second. One query per call, one thread, unless marked batched.", "",
            "| nprobe | " + " | ".join(head + extra) + " |", "|---|" + "---|" * (len(head) + len(extra))]
    for n in nprobes:
        cells = [f"{value(rows, k, 'qps', nprobe=n):,.0f}" for k in cols]
        go, fa, py = (value(rows, k, "qps", nprobe=n) for k in ("vecstore_go", "faiss", "vecstore"))
        if "faiss speed over Go" in extra:
            cells.append(f"{fa / go:.2f}x")
        if "Go speed over Python" in extra:
            cells.append(f"{go / py:.2f}x")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    if "faiss speed over Go" in extra:
        out += ["", "A faiss-over-Go value below 1.00x means the Go engine is faster."]
    return "\n".join(out)


def write_table(tag):
    rows = load_all("write_qps", tag)
    cols = present(rows) + [k for k in ("vecstore_go_batch", "faiss_batch") if any(r["impl"] == k for r in rows)]
    label = {**NAMES, "vecstore_go_batch": "Go, all cores", "faiss_batch": "faiss, all cores"}
    out = ["| vectors per add call | " + " | ".join(label[k] for k in cols) + " |",
           "|---|" + "---|" * len(cols)]
    for b in sorted({r["batch_size"] for r in rows}, key=int):
        cells = []
        for k in cols:
            v = value(rows, k, "vectors_per_s", batch_size=b)
            cells.append(f"{v:,.0f}" if v is not None else "n/a")
        out.append(f"| {int(b):,} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def build_table(tag):
    rows = load_all("build", tag)
    out = ["| index | train (s) | add all vectors (s) |", "|---|---|---|"]
    for k in present(rows):
        out.append(f"| {NAMES[k]} | {value(rows, k, 'train_s'):.1f} | {value(rows, k, 'add_s'):.1f} |")
    return "\n".join(out)


def results_for_readme():
    out = []
    for tag in reversed(tags()):  # largest first
        out += [f"#### {title_for(tag)}", "", "Reads", "", speed_table(tag), "",
                "Writes after the index is built. Vectors per second. One thread unless marked all cores.", "", write_table(tag), "",
                "Build. Every index trains for the same number of k-means iterations.", "",
                build_table(tag), ""]
    return "\n".join(out) if out else "no result yet"


def headline():
    """A few sentences computed from the CSVs."""
    lines = []
    for tag in reversed(tags()):
        rows = load_all("recall_qps", tag)
        main = "vecstore_go" if any(r["impl"] == "vecstore_go" for r in rows) else "vecstore"
        by = lambda impl: {r["nprobe"]: r for r in rows if r["impl"] == impl}
        m, f, py = by(main), by("faiss"), by("vecstore")
        top = max(m, key=int)
        gap = max(abs(float(m[k]["recall_at_10"]) - float(f[k]["recall_at_10"])) for k in m)
        share = [float(m[k]["qps"]) / float(f[k]["qps"]) for k in m]
        who = "The Go engine" if main == "vecstore_go" else "vecstore"
        text = (f"- **{title_for(tag)}.** {who} reaches recall@10 of {float(m[top]['recall_at_10']):.3f} "
                f"at nprobe {top}. It never differs from faiss by more than {gap:.4f}. "
                f"On one thread it runs at {min(share):.0%} to {max(share):.0%} of faiss's query speed")
        if main == "vecstore_go" and py:
            up = [float(m[k]["qps"]) / float(py[k]["qps"]) for k in m]
            text += f" and at {min(up):.1f}x to {max(up):.1f}x the Python reference"
        info = next((r for r in (load("go_run_info.csv") or []) if r["run"] == tag), None)
        info = info or next(r for r in load("run_info.csv") if r["run"] == tag)
        lines.append(text + f". Peak memory was {int(float(info['peak_rss_mb'])):,} MB.")
    return "\n".join(lines) if lines else "no result yet"


def inject(text, name, body):
    """Replace the text between <!-- name:start --> and <!-- name:end -->."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if start not in text:
        return text
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}\n{end}{tail}"


def update_readme():
    path = ROOT / "README.md"
    text = path.read_text()
    text = inject(text, "headline", headline())
    text = inject(text, "results", results_for_readme())
    path.write_text(text)
    print("updated README.md")


def main():
    text = f"""# Overview

Generated by `make status`. Do not edit by hand.

## The question

Can a small index written from scratch return correct nearest neighbours?
How close does it get to a reference library on recall and speed?

This project claims no new idea. It claims no speed advantage over Faiss.

## Status

{status_section()}

## Results

Every number below is read from a CSV in `results/`.

{results_section()}"""
    (ROOT / "OVERVIEW.md").write_text(text)
    print("wrote OVERVIEW.md")
    update_readme()


if __name__ == "__main__":
    main()
