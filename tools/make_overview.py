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

# (heading, csv path). Add a row here when a new benchmark appears.
RESULTS = [
    ("Recall and queries per second, 200k vectors", "results/recall_qps_200k.csv"),
    ("Build and write throughput, 200k vectors", "results/build_200k.csv"),
    ("Insert throughput by batch size, 200k vectors", "results/write_qps_200k.csv"),
    ("Recall and queries per second, 1M vectors", "results/recall_qps_1m.csv"),
    ("Build and write throughput, 1M vectors", "results/build_1m.csv"),
    ("Insert throughput by batch size, 1M vectors", "results/write_qps_1m.csv"),
    ("Run environment", "results/run_info.csv"),
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


def tags():
    """Result sets present on disk, smallest first: ['200k', '1m']."""
    found = [t for t in ("200k", "1m") if load(f"recall_qps_{t}.csv")]
    return found


def title_for(tag):
    row = load(f"build_{tag}.csv")[0]
    return f"{int(row['n']):,} vectors, nlist {row['nlist']}"


def speed_table(tag):
    rows = load(f"recall_qps_{tag}.csv")
    get = lambda impl, nprobe, col: next(
        float(r[col]) for r in rows if r["impl"] == impl and r["nprobe"] == nprobe
    )
    out = ["| nprobe | recall@10 vecstore | recall@10 faiss | queries/s vecstore "
           "| queries/s faiss | faiss speed over vecstore |", "|---|---|---|---|---|---|"]
    for nprobe in sorted({r["nprobe"] for r in rows}, key=int):
        qv, qf = get("vecstore", nprobe, "qps"), get("faiss", nprobe, "qps")
        out.append(f"| {nprobe} | {get('vecstore', nprobe, 'recall_at_10'):.4f} "
                   f"| {get('faiss', nprobe, 'recall_at_10'):.4f} | {qv:,.0f} | {qf:,.0f} "
                   f"| {qf / qv:.1f}x |")
    return "\n".join(out)


def write_table(tag):
    rows = load(f"write_qps_{tag}.csv")
    out = ["| vectors per add call | vecstore vectors/s | faiss vectors/s |", "|---|---|---|"]
    for b in sorted({r["batch_size"] for r in rows}, key=int):
        v = next(float(r["vectors_per_s"]) for r in rows if r["impl"] == "vecstore" and r["batch_size"] == b)
        f = next(float(r["vectors_per_s"]) for r in rows if r["impl"] == "faiss" and r["batch_size"] == b)
        out.append(f"| {int(b):,} | {v:,.0f} | {f:,.0f} |")
    return "\n".join(out)


def build_table(tag):
    rows = load(f"build_{tag}.csv")
    out = ["| index | train (s) | add all vectors (s) |", "|---|---|---|"]
    for r in rows:
        out.append(f"| {r['impl']} | {float(r['train_s']):.1f} | {float(r['add_s']):.1f} |")
    return "\n".join(out)


def results_for_readme():
    out = []
    for tag in reversed(tags()):  # largest first
        out += [f"#### {title_for(tag)}", "", "Reads", "", speed_table(tag), "",
                "Writes after the index is built", "", write_table(tag), "",
                "Build", "", build_table(tag), ""]
    return "\n".join(out) if out else "no result yet"


def headline():
    """A few sentences computed from the CSVs."""
    lines = []
    for tag in reversed(tags()):
        rows = load(f"recall_qps_{tag}.csv")
        by = lambda impl: {r["nprobe"]: r for r in rows if r["impl"] == impl}
        v, f = by("vecstore"), by("faiss")
        gap = max(abs(float(v[k]["recall_at_10"]) - float(f[k]["recall_at_10"])) for k in v)
        ratios = [float(f[k]["qps"]) / float(v[k]["qps"]) for k in v]
        top = max(v, key=int)
        info = next(r for r in load("run_info.csv") if r["run"] == tag)
        lines.append(
            f"- **{title_for(tag)}.** Recall@10 reaches {float(v[top]['recall_at_10']):.3f} "
            f"at nprobe {top}. It never differs from faiss by more than {gap:.4f}. "
            f"Faiss on one thread is {min(ratios):.1f}x to {max(ratios):.1f}x faster per query. "
            f"Peak memory was {int(info['peak_rss_mb']):,} MB."
        )
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
