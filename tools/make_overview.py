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

# (heading, csv path). Faiss results first, then the Go engine's.
RESULTS = []
for _tag, _size in (("200k", "200k vectors"), ("1m", "1M vectors")):
    for _prefix, _who in (("", "Faiss"), ("go_", "Go engine")):
        RESULTS += [
            (f"Recall and queries per second, {_size}, {_who}", f"results/{_prefix}recall_qps_{_tag}.csv"),
            (f"Build time, {_size}, {_who}", f"results/{_prefix}build_{_tag}.csv"),
            (f"Insert throughput by batch size, {_size}, {_who}", f"results/{_prefix}write_qps_{_tag}.csv"),
        ]
RESULTS += [
    ("Run environment, Faiss", "results/run_info.csv"),
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
    for item, status, file, _ in rows:
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
    """Faiss rows and Go rows for one run, joined."""
    return (load(f"{stem}_{tag}.csv") or []) + (load(f"go_{stem}_{tag}.csv") or [])


def tags():
    """Result sets present on disk, smallest first: ['200k', '1m']."""
    return [t for t in ("200k", "1m") if load(f"recall_qps_{t}.csv")]


def title_for(tag):
    row = load(f"build_{tag}.csv")[0]
    return f"{int(row['n']):,} vectors, nlist {row['nlist']}"


# CSV key -> short name in the tables. Order is the column order.
NAMES = {"vecstore_go": "Go", "faiss": "Faiss"}
SINGLE = list(NAMES)


def present(rows):
    return [k for k in SINGLE if any(r["impl"] == k for r in rows)]


def value(rows, impl, col, **where):
    for r in rows:
        if r["impl"] == impl and all(r[c] == v for c, v in where.items()):
            return float(r[col])
    return None


BATCH = {"vecstore_go_batch": "Go, all cores", "faiss_batch": "Faiss, all cores"}


def table(head, rows):
    """A Markdown table from a header list and a list of row lists."""
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows])


def speed_table(tag):
    rows = load_all("recall_qps", tag)
    impls = present(rows)
    nprobes = sorted({r["nprobe"] for r in rows}, key=int)
    recall = table(["nprobe"] + [NAMES[k] for k in impls],
                   [[n] + [f"{value(rows, k, 'recall_at_10', nprobe=n):.4f}" for k in impls]
                    for n in nprobes])

    cols = impls + [k for k in BATCH if any(r["impl"] == k for r in rows)]
    ratio = "vecstore_go" in impls and "faiss" in impls
    body = []
    for n in nprobes:
        cells = [n] + [f"{value(rows, k, 'qps', nprobe=n):,.0f}" for k in cols]
        if ratio:
            go, fa = (value(rows, k, "qps", nprobe=n) for k in ("vecstore_go", "faiss"))
            cells.append(f"{fa / go:.2f}x")
        body.append(cells)
    head = ["nprobe"] + [{**NAMES, **BATCH}[k] for k in cols] + (["Faiss ÷ Go"] if ratio else [])
    out = ["**Recall@10**", "", recall, "",
           "**Queries per second.** One query per call on one thread, unless marked all cores.", "",
           table(head, body)]
    if ratio:
        out += ["", "Faiss ÷ Go compares the one-thread columns. Below 1.00x, Go is faster."]
    return "\n".join(out)


def write_table(tag):
    rows = load_all("write_qps", tag)
    cols = present(rows) + [k for k in BATCH if any(r["impl"] == k for r in rows)]
    body = []
    for b in sorted({r["batch_size"] for r in rows}, key=int):
        cells = [f"{int(b):,}"]
        for k in cols:
            v = value(rows, k, "vectors_per_s", batch_size=b)
            cells.append(f"{v:,.0f}" if v is not None else "n/a")
        body.append(cells)
    return "\n".join([
        "**Inserts after the index is built.** Vectors per second. One thread unless marked all cores.", "",
        table(["Vectors per call"] + [{**NAMES, **BATCH}[k] for k in cols], body)])


def build_table(tag):
    rows = load_all("build", tag)
    body = [[NAMES[k], f"{value(rows, k, 'train_s'):.1f}", f"{value(rows, k, 'add_s'):.1f}"]
            for k in present(rows)]
    return "\n".join([
        "**Build time.** Both train for the same number of k-means rounds.", "",
        table(["Index", "Train (s)", "Add all vectors (s)"], body)])


def results_for_readme():
    out = []
    for tag in reversed(tags()):  # largest first
        out += ["<details>", f"<summary><b>{title_for(tag)}</b></summary>", "",
                speed_table(tag), "", write_table(tag), "", build_table(tag), "",
                "</details>", ""]
    return "\n".join(out).rstrip() if out else "no result yet"


def headline():
    """A summary table computed from the CSVs, one column per dataset size."""
    cols = []
    for tag in reversed(tags()):
        rows = load_all("recall_qps", tag)
        by = lambda impl: {r["nprobe"]: r for r in rows if r["impl"] == impl}
        go, fa = by("vecstore_go"), by("faiss")
        if not go or not fa:
            continue
        top = max(go, key=int)
        gap = max(abs(float(go[k]["recall_at_10"]) - float(fa[k]["recall_at_10"])) for k in go)
        share = [float(go[k]["qps"]) / float(fa[k]["qps"]) for k in go]
        build = load_all("build", tag)
        train = value(build, "vecstore_go", "train_s") / value(build, "faiss", "train_s")
        info = next(r for r in (load("go_run_info.csv") or []) if r["run"] == tag)
        n = int(load(f"build_{tag}.csv")[0]["n"])
        cols.append((f"{n:,} vectors", [
            go[top]["nlist"],
            f"{float(go[top]['recall_at_10']):.3f}",
            f"{gap:.4f}",
            f"{min(share):.0%} to {max(share):.0%}",
            f"{train:.1f}x as long",
            f"{int(float(info['peak_rss_mb'])):,} MB",
        ]))
    if not cols:
        return "no result yet"
    labels = [
        "Lists (nlist)",
        f"Go recall@10 at nprobe {top}",
        "Largest recall gap to Faiss",
        "Go query speed on one thread, as a share of Faiss",
        "Go training time, compared with Faiss",
        "Go peak memory",
    ]
    return table([""] + [c[0] for c in cols],
                 [[label] + [c[1][i] for c in cols] for i, label in enumerate(labels)])


def inject(text, name, body):
    """Replace the text between <!-- name:start --> and <!-- name:end -->."""
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    if start not in text:
        return text
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{start}\n{body}\n{end}{tail}"


def roadmap():
    """Checkbox list from status.csv, grouped by the roadmap column.

    Rows with an empty roadmap column are project chores. They show in
    OVERVIEW.md but not in the README roadmap.
    """
    phases = {}
    for item, status, _, phase in read_csv(ROOT / "status.csv")[1:]:
        if phase:
            phases.setdefault(phase, []).append((item, status))
    out = []
    for phase, items in phases.items():
        out += [f"**{phase}**", ""]
        for item, status in items:
            box = "x" if status == "done" else " "
            note = " (in progress)" if status == "in progress" else ""
            out.append(f"- [{box}] {item}{note}")
        out.append("")
    return "\n".join(out).rstrip()


def update_readme():
    path = ROOT / "README.md"
    text = path.read_text()
    text = inject(text, "headline", headline())
    text = inject(text, "roadmap", roadmap())
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
