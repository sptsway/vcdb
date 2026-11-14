"""Draw the README charts from results/*.csv. No number is typed in here.

Each chart is written twice (light and dark) into docs/img/. The README picks
the right one with a <picture> tag. Colours come from a palette that passed the
colour-blind and contrast checks in both themes.

Three implementations are drawn. Colour follows the implementation, never its
rank: Go is blue, the Python reference is aqua, Faiss is orange. A dashed line
means batched search on all cores.
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "img"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df",
                  blue="#2a78d6", orange="#eb6834", aqua="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#34332f",
                 blue="#3987e5", orange="#d95926", aqua="#199e70"),
}
# key in the CSV -> (legend label, colour, line style, marker)
SERIES = {
    "vecstore_go": ("vecstore, Go engine", "blue", "-", "o"),
    "vecstore": ("vecstore, Python reference", "aqua", "-", "D"),
    "faiss": ("faiss IVFFlat, 1 thread", "orange", "-", "s"),
    "vecstore_go_batch": ("vecstore Go, batched, all cores", "blue", "--", "^"),
    "faiss_batch": ("faiss, batched, all cores", "orange", "--", "v"),
}
SINGLE = ("vecstore_go", "vecstore", "faiss")
SIZES = {"200k": "200,000 vectors, nlist 1024", "1m": "1,000,000 vectors, nlist 4096"}


def read(name):
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def merged(stem, tag):
    """Python/Faiss rows plus Go rows for the same run, or None if absent."""
    rows = read(f"{stem}_{tag}.csv") + read(f"go_{stem}_{tag}.csv")
    return rows or None


def human(v, _=None):
    """Three significant figures: 5014 -> 5.01k, 1200000 -> 1.2M."""
    if v >= 1e6:
        return f"{v / 1e6:.3g}M"
    if v >= 1e3:
        return f"{v / 1e3:.3g}k"
    return f"{v:.3g}"


def setup(theme, title, subtitle, xlabel, ylabel):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["ink2"], length=0, labelsize=9)
    ax.set_xlabel(xlabel, color=t["ink2"], fontsize=9.5, labelpad=8)
    ax.set_ylabel(ylabel, color=t["ink2"], fontsize=9.5, labelpad=8)
    fig.text(0.07, 0.955, title, color=t["ink"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.07, 0.895, subtitle, color=t["ink2"], fontsize=9.5, va="top")
    return fig, ax, t


def finish(fig, ax, t, name, tag, theme, legend=True):
    if legend:
        leg = ax.legend(frameon=False, fontsize=8.5, loc="best", labelcolor=t["ink2"])
        for line in leg.get_lines():
            line.set_linewidth(2)
    fig.subplots_adjust(left=0.13, right=0.94, top=0.80, bottom=0.15)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}_{tag}_{theme}.png", facecolor=t["surface"])
    plt.close(fig)


def line(ax, t, key, xs, ys, size=8, z=3):
    label, colour, style, marker = SERIES[key]
    ax.plot(xs, ys, color=t[colour], linestyle=style, linewidth=2, marker=marker,
            markersize=size, markeredgecolor=t["surface"], markeredgewidth=2, label=label,
            zorder=z)


def end_label(ax, t, x, y, text, dy=0):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(9, dy), fontsize=8.5,
                color=t["ink"], va="center", annotation_clip=False)


def spread(ax, ys, gap_px=22):
    """Vertical offsets (points) that keep end labels at least gap_px apart.

    ys are data values at the right edge. Labels are pushed apart from their
    neighbours in screen space, so close lines never share a label spot.
    """
    ax.figure.canvas.draw()  # the transform is only exact after a draw
    screen = [ax.transData.transform((0, y))[1] for y in ys]
    order = sorted(range(len(ys)), key=lambda i: screen[i])
    placed = {}
    for rank, i in enumerate(order):
        want = screen[i]
        if rank and want < placed[order[rank - 1]] + gap_px:
            want = placed[order[rank - 1]] + gap_px
        placed[i] = want
    to_points = 72 / ax.figure.dpi
    return [(placed[i] - screen[i]) * to_points for i in range(len(ys))]


def by_impl(rows, xkey, ykey):
    out = {}
    for r in rows:
        out.setdefault(r["impl"], []).append((float(r[xkey]), float(r[ykey])))
    return out


def log_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(human))
    ax.yaxis.set_minor_formatter(NullFormatter())


def positions(data):
    """Map each x value to 0, 1, 2... so unevenly spaced values sit evenly."""
    return {n: i for i, n in enumerate(sorted({x for v in data.values() for x, _ in v}))}


def categorical_x(ax, pos, fmt):
    ax.set_xticks(list(pos.values()), [fmt(n) for n in pos])
    ax.set_xlim(-0.2, max(pos.values()) + 0.85)


def chart_recall(tag, rows, theme):
    fig, ax, t = setup(theme, "Recall at 10 as nprobe grows",
                       f"{SIZES[tag]}. Higher is better. The lines overlap.",
                       "nprobe (lists searched per query)", "recall@10")
    data = by_impl(rows, "nprobe", "recall_at_10")
    pos = positions(data)
    # Big marker at the back, small at the front, so all three stay visible.
    for key, size, z in (("faiss", 15, 3), ("vecstore", 10, 4), ("vecstore_go", 5, 5)):
        if key in data:
            line(ax, t, key, [pos[x] for x, _ in data[key]], [y for _, y in data[key]], size, z)
    last = max(pos.values())
    for key, name, dy in (("vecstore_go", "Go", 14), ("vecstore", "Python", 0), ("faiss", "faiss", -14)):
        if key in data:
            end_label(ax, t, last, data[key][-1][1], f"{name} {data[key][-1][1]:.4f}", dy)
    categorical_x(ax, pos, lambda n: f"{int(n)}")
    ax.set_ylim(0, 1.05)
    finish(fig, ax, t, "recall_vs_nprobe", tag, theme)


def chart_qps(tag, rows, theme):
    fig, ax, t = setup(theme, "Queries per second as nprobe grows",
                       f"{SIZES[tag]}. One query per call unless dashed. Log scale.",
                       "nprobe (lists searched per query)", "queries per second")
    data = by_impl(rows, "nprobe", "qps")
    pos = positions(data)
    ends = []
    for key in ("vecstore_go", "vecstore", "faiss", "vecstore_go_batch", "faiss_batch"):
        if key not in data:
            continue
        pts = data[key]
        line(ax, t, key, [pos[x] for x, _ in pts], [y for _, y in pts])
        if key in SINGLE:  # direct labels on the like-for-like lines only
            ends.append((pos[pts[-1][0]], pts[-1][1]))
    log_axis(ax)
    categorical_x(ax, pos, lambda n: f"{int(n)}")
    for (x, y), dy in zip(ends, spread(ax, [y for _, y in ends])):
        end_label(ax, t, x, y, human(y), dy)
    finish(fig, ax, t, "qps_vs_nprobe", tag, theme)


def chart_tradeoff(tag, rows, theme):
    fig, ax, t = setup(theme, "Recall against speed",
                       f"{SIZES[tag]}. Up and right is better. Points are nprobe 1, 4, 16, 64.",
                       "recall@10", "queries per second")
    recall = by_impl(rows, "nprobe", "recall_at_10")
    qps = by_impl(rows, "nprobe", "qps")
    for key in SINGLE:
        if key not in qps:
            continue
        xs = [y for _, y in recall[key]]
        ys = [y for _, y in qps[key]]
        line(ax, t, key, xs, ys)
    # Number each nprobe step once, above whichever line is highest there.
    steps = sorted({n for k in SINGLE if k in qps for n, _ in qps[k]})
    for n in steps:
        top_key = max((k for k in SINGLE if k in qps),
                      key=lambda k: dict(qps[k])[n])
        x = dict(zip([a for a, _ in qps[top_key]], [y for _, y in recall[top_key]]))[n]
        ax.annotate(f"{int(n)}", (x, dict(qps[top_key])[n]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color=t["ink2"])
    values = [y for k in SINGLE if k in qps for _, y in qps[k]]
    top, low = max(values), min(values)
    log_axis(ax)
    ax.set_ylim(low / 1.8, top * 1.5)
    ax.set_yticks([v for v in (1e3, 3e3, 1e4, 3e4, 1e5, 3e5) if low / 1.8 <= v <= top * 1.5])
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_xlim(0.2, 1.05)
    finish(fig, ax, t, "recall_vs_qps", tag, theme)


def chart_writes(tag, rows, theme):
    fig, ax, t = setup(theme, "Insert throughput by batch size",
                       f"{SIZES[tag]}. Vectors added after the index is built. Dashed: all cores. Log scale.",
                       "vectors per add call", "vectors inserted per second")
    data = by_impl(rows, "batch_size", "vectors_per_s")
    pos = positions(data)
    ends = []
    for key in ("vecstore_go", "vecstore", "faiss", "vecstore_go_batch", "faiss_batch"):
        if key not in data:
            continue
        pts = data[key]
        line(ax, t, key, [pos[x] for x, _ in pts], [y for _, y in pts])
        ends.append((pos[pts[-1][0]], pts[-1][1]))
    log_axis(ax)
    categorical_x(ax, pos, lambda n: f"{int(n):,}")
    for (x, y), dy in zip(ends, spread(ax, [y for _, y in ends])):
        end_label(ax, t, x, y, human(y), dy)
    finish(fig, ax, t, "write_throughput", tag, theme)


def chart_build(tag, rows, theme):
    fig, ax, t = setup(theme, "Time to build the index",
                       f"{SIZES[tag]}. Every index trains for 20 k-means iterations. Lower is better.",
                       "", "seconds")
    keys = [k for k in SINGLE if any(r["impl"] == k for r in rows)]
    train = {r["impl"]: float(r["train_s"]) for r in rows}
    add = {r["impl"]: float(r["add_s"]) for r in rows}
    names = {"vecstore_go": "Go engine", "vecstore": "Python reference", "faiss": "faiss"}
    peak = max(train[k] + add[k] for k in keys)
    for i, key in enumerate(keys):
        colour = SERIES[key][1]
        ax.bar(i, train[key], width=0.5, color=t[colour], zorder=3)
        ax.bar(i, add[key], width=0.5, bottom=train[key], color=t[colour], alpha=0.45, zorder=3)
        ax.annotate(f"train {train[key]:.1f}s", (i, train[key] / 2), ha="center", va="center",
                    fontsize=9, color=t["ink"], fontweight="bold")
        ax.annotate(f"add {add[key]:.1f}s", (i, train[key] + add[key]), ha="center",
                    textcoords="offset points", xytext=(0, 6), fontsize=9, color=t["ink"])
    ax.set_xticks(range(len(keys)), [names[k] for k in keys])
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_ylim(0, peak * 1.12)
    finish(fig, ax, t, "build_time", tag, theme, legend=False)


def chart_loss(tag, curves, theme):
    fig, ax, t = setup(theme, "k-means loss falls every iteration",
                       f"{SIZES[tag]}. Total squared distance to the nearest centroid. Lower is better.",
                       "iteration", "loss")
    for key, rows in curves.items():
        xs = [int(r["iteration"]) for r in rows]
        ys = [float(r["loss"]) for r in rows]
        line(ax, t, key, xs, ys, size=6)
    ax.set_xticks([1, 5, 10, 15, 20])
    power = len(str(int(max(float(r["loss"]) for rows in curves.values() for r in rows)))) - 1
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 10**power:.1f}e{power}"))
    finish(fig, ax, t, "kmeans_loss", tag, theme)


def main():
    made = 0
    for tag in SIZES:
        recall = merged("recall_qps", tag)
        if recall is None:
            print(f"{tag}: no result yet, skipping")
            continue
        build, writes = merged("build", tag), merged("write_qps", tag)
        curves = {k: read(n) for k, n in (("vecstore", f"kmeans_loss_{tag}.csv"),
                                          ("vecstore_go", f"go_kmeans_loss_{tag}.csv")) if read(n)}
        for theme in THEMES:
            chart_recall(tag, recall, theme)
            chart_qps(tag, recall, theme)
            chart_tradeoff(tag, recall, theme)
            if writes:
                chart_writes(tag, writes, theme)
            if build:
                chart_build(tag, build, theme)
            if curves:
                chart_loss(tag, curves, theme)
            made += 1
        print(f"{tag}: charts written to docs/img/")
    print(f"{made} theme runs done")


if __name__ == "__main__":
    main()
