from __future__ import annotations

from pathlib import Path

import numpy as np

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
BLUE = "#2a78d6"
RED = "#e34948"


def series_color(i: int) -> str:
    return SERIES[i % len(SERIES)]


def pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": cycler(color=list(SERIES)),
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "font.family": "sans-serif",
        }
    )
    return plt


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("azul_rojo", [BLUE, NEUTRAL, RED])


def _save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_curves(series: list[dict], path: str | Path, title: str, xlabel: str, ylabel: str, reference: float | None = None, ylim=None) -> None:
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    if reference is not None:
        ax.axhline(reference, color=MUTED, linewidth=0.8)
    for i, s in enumerate(series):
        color = s.get("color", series_color(i))
        x = np.asarray(s["x"], dtype=float)
        y = np.asarray(s["y"], dtype=float)
        if "lo" in s and "hi" in s:
            ax.fill_between(x, s["lo"], s["hi"], color=color, alpha=0.12, linewidth=0)
        ax.plot(x, y, color=color, label=s["label"])
        if len(x):
            ax.plot(x[-1:], y[-1:], "o", color=color, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
            if len(series) <= 4:
                ax.annotate(f"{y[-1]:.2f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_SECONDARY)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(loc="best")
    _save(fig, path)


def plot_intervals(
    labels: list[str],
    values: list[float],
    lows: list[float],
    highs: list[float],
    path: str | Path,
    title: str,
    xlabel: str,
    reference: float | None = None,
    highlight: list[bool] | None = None,
    fmt: str = "{:.2f}",
) -> None:
    plt = pyplot()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.38 * n))
    y = np.arange(n)[::-1]
    if reference is not None:
        ax.axvline(reference, color=MUTED, linewidth=0.8)
    for i in range(n):
        color = INK_SECONDARY if highlight is not None and highlight[i] else BLUE
        ax.plot([lows[i], highs[i]], [y[i], y[i]], color=color, linewidth=1.6)
        ax.plot([values[i]], [y[i]], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        text = f"{fmt.format(values[i])}  [{fmt.format(lows[i])}, {fmt.format(highs[i])}]"
        ax.annotate(text, (highs[i], y[i]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    lo = min(lows) if lows else 0.0
    hi = max(highs) if highs else 1.0
    span = max(hi - lo, 1e-9)
    ax.set_xlim(lo - 0.08 * span, hi + 0.45 * span)
    _save(fig, path)


def plot_matrix(
    row_labels: list[str],
    col_labels: list[str],
    matrix: np.ndarray,
    path: str | Path,
    title: str,
    center: float = 0.5,
    span: float = 0.5,
    colorbar: str = "",
    annotate: bool | None = None,
    fmt: str = "{:.2f}",
) -> None:
    plt = pyplot()
    matrix = np.asarray(matrix, dtype=float)
    rows, cols = matrix.shape
    fig, ax = plt.subplots(figsize=(1.6 + 0.62 * cols, 1.2 + 0.5 * rows))
    im = ax.imshow(matrix, cmap=diverging_cmap(), vmin=center - span, vmax=center + span, aspect="auto")
    ax.set_xticks(range(cols), col_labels, rotation=35, ha="right")
    ax.set_yticks(range(rows), row_labels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if annotate is None:
        annotate = rows * cols <= 144
    if annotate:
        for i in range(rows):
            for j in range(cols):
                if np.isfinite(matrix[i, j]):
                    strong = abs(matrix[i, j] - center) > 0.6 * span
                    ax.text(j, i, fmt.format(matrix[i, j]), ha="center", va="center", fontsize=7, color="#ffffff" if strong else INK)
    bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    if colorbar:
        bar.set_label(colorbar, color=INK_SECONDARY, fontsize=8)
    ax.set_title(title, loc="left")
    _save(fig, path)


def plot_grouped_bars(
    categories: list[str],
    groups: list[dict],
    path: str | Path,
    title: str,
    xlabel: str,
    bound: list[float] | None = None,
) -> None:
    plt = pyplot()
    n = len(categories)
    k = max(1, len(groups))
    fig, ax = plt.subplots(figsize=(7.2, 1.0 + 0.32 * n * k))
    height = 0.8 / k
    base = np.arange(n)[::-1].astype(float)
    for g, group in enumerate(groups):
        pos = base + 0.4 - height * (g + 0.5)
        values = np.asarray(group["values"], dtype=float)
        ax.barh(pos, values, height=height * 0.82, color=group.get("color", series_color(g)), label=group["label"])
        if "lo" in group and "hi" in group:
            lo = values - np.asarray(group["lo"], dtype=float)
            hi = np.asarray(group["hi"], dtype=float) - values
            ax.errorbar(values, pos, xerr=[np.maximum(lo, 0), np.maximum(hi, 0)], fmt="none", ecolor=INK_SECONDARY, elinewidth=0.9, capsize=2)
    if bound is not None:
        ax.plot(bound, base, "|", color=INK, markersize=12, label="cota: entropía del rasgo")
    ax.set_yticks(base, categories)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    if k > 1 or bound is not None:
        ax.legend(loc="lower right")
    _save(fig, path)


def plot_infoset_curve(curve: list[dict], path: str | Path, title: str = "MCCFR: infosets contra iteraciones") -> None:
    plt = pyplot()
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    shared = any("total_iterations" in r and int(r["total_iterations"]) != int(r["iteration"]) for r in curve)
    if shared:
        first = min(int(r["worker"]) for r in curve)
        rows = sorted((r for r in curve if int(r["worker"]) == first), key=lambda r: int(r["total_iterations"]))
        groups = [("tabla compartida", rows, "total_iterations")]
        xlabel = "iteraciones sumando todos los workers"
    else:
        workers = sorted({int(r["worker"]) for r in curve})
        groups = [(f"worker {w}", sorted((r for r in curve if int(r["worker"]) == w), key=lambda r: int(r["iteration"])), "iteration") for w in workers]
        xlabel = "iteraciones"
    for i, (label, rows, key) in enumerate(groups):
        it = np.array([float(r[key]) for r in rows])
        count = np.array([float(r["infosets"]) for r in rows])
        color = series_color(i)
        left.plot(it, count / 1e6, color=color, label=label)
        if len(it) > 1:
            step = np.diff(it)
            keep = step > 0
            rate = np.diff(count)[keep] / step[keep] * 1000.0
            right.plot(it[1:][keep], np.maximum(rate, 1e-3), color=color, label=label)
    left.set_title(title, loc="left")
    left.set_xlabel(xlabel)
    left.set_ylabel("infosets (millones)")
    left.set_ylim(bottom=0)
    right.set_title("infosets nuevos cada 1000 iteraciones", loc="left")
    right.set_xlabel(xlabel)
    right.set_yscale("log")
    if len(groups) > 1:
        left.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, path)
