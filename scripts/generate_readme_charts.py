#!/usr/bin/env python3
"""Generate performance comparison charts for the README.

Reads benchmark JSON and produces:
  - assets/speedup.png      — speedup factors by operation
  - assets/latency.png      — absolute latency comparison (log scale)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_PATH = os.path.join(ROOT, "data", "bench_torch_harmonics_latest.json")
OUT_DIR = os.path.join(ROOT, "assets")


# Muted two-colour palette (ref = warm grey, holysht = teal)
C_REF = "#9e9e9e"
C_HOLYSHT = "#00897b"
C_SPEEDUP = "#00897b"
C_TEXT = "#333333"
C_LIGHT = "#888888"


def load_data():
    with open(BENCH_PATH) as f:
        return json.load(f)


def tufte_ax(ax):
    """Strip chart junk — keep only left and bottom spines, no ticks on top/right."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=C_LIGHT, which="both", length=3)
    ax.tick_params(axis="both", which="minor", length=0)


def short_name(test_name, grid):
    """Produce a compact label like 'scalar fwd 256x512'."""
    name = test_name.lower()
    if "vector" in name:
        prefix = "vector"
    else:
        prefix = "scalar"

    if "roundtrip" in name:
        op = "roundtrip"
    elif "backward" in name:
        op = "fwd+bwd"
    elif "inverse" in name:
        op = "inverse"
    elif "bf16" in name and "vector" in name:
        op = "fwd (bf16)"
        prefix = "vector"
    elif "bf16" in name:
        op = "fwd (bf16)"
        prefix = "scalar"
    elif "synthesis" in name.lower() or "sparse" in name.lower():
        op = "synthesis"
        prefix = "Y_n^m"
    else:
        op = "forward"

    grid_short = grid.split("(")[0].strip()
    return f"{prefix} {op} {grid_short}"


def make_speedup_chart(data):
    """Horizontal lollipop chart of speedup factors."""
    results = data["results"]

    # Sort by category then grid size for visual grouping
    labels = []
    speedups = []
    for r in results:
        labels.append(short_name(r["test_name"], r["grid"]))
        speedups.append(r["speedup"])

    n = len(labels)
    y = np.arange(n)

    fig, ax = plt.subplots(figsize=(7.5, 0.38 * n + 1.0))
    tufte_ax(ax)

    # Thin lines from 1x to the dot (lollipop stems)
    for i in range(n):
        ax.plot([1, speedups[i]], [y[i], y[i]], color=C_SPEEDUP, linewidth=0.8, alpha=0.5)

    # Dots
    ax.scatter(speedups, y, color=C_SPEEDUP, s=36, zorder=5, edgecolors="none")

    # Direct labels
    for i in range(n):
        offset = 0.15
        ha = "left"
        ax.text(speedups[i] + offset, y[i], f"{speedups[i]:.1f}x",
                va="center", ha=ha, fontsize=8, color=C_TEXT)

    # Reference line at 1x
    ax.axvline(x=1, color="#dddddd", linewidth=0.8, zorder=0)
    ax.text(1, n + 0.2, "1x (torch-harmonics)", fontsize=7, color=C_LIGHT,
            ha="center", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=C_TEXT)
    ax.set_xlim(0, max(speedups) + 1.5)
    ax.set_ylim(-0.7, n - 0.3)
    ax.invert_yaxis()

    # Minimal x-axis
    ax.set_xlabel("speedup factor", fontsize=9, color=C_LIGHT)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))

    ax.set_title("HOLYSHT speedup over torch-harmonics", fontsize=11,
                 color=C_TEXT, loc="left", pad=10)

    # Subtitle with hardware info
    fig.text(0.125, 0.01,
             f"NVIDIA GB10  |  PyTorch {data['pytorch']}  |  CUDA {data['cuda']}  |  batch 4",
             fontsize=7, color=C_LIGHT, ha="left")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(OUT_DIR, "speedup.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


def make_latency_chart(data):
    """Paired dot chart of absolute latency (log scale), grouped by category."""
    results = data["results"]

    # Group into categories for small-multiples panels
    categories = {
        "scalar forward": [],
        "scalar inverse": [],
        "scalar roundtrip": [],
        "vector forward": [],
        "vector inverse": [],
        "training (fwd+bwd)": [],
        "bf16": [],
    }

    for r in results:
        name = r["test_name"].lower()
        grid = r["grid"].split("(")[0].strip()

        if "synthesis" in name or "sparse" in name:
            continue  # skip the single sparse case

        if "bf16" in name:
            cat = "bf16"
        elif "backward" in name and "vector" in name:
            cat = "training (fwd+bwd)"
        elif "backward" in name:
            cat = "training (fwd+bwd)"
        elif "roundtrip" in name:
            cat = "scalar roundtrip"
        elif "vector" in name and "inverse" in name:
            cat = "vector inverse"
        elif "vector" in name:
            cat = "vector forward"
        elif "inverse" in name:
            cat = "scalar inverse"
        else:
            cat = "scalar forward"

        categories[cat].append((grid, r["ref_ms"], r["holysht_ms"], r["speedup"]))

    # Remove empty categories
    categories = {k: v for k, v in categories.items() if v}

    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(2.4 * n_cats + 0.8, 4.0),
                              sharey=False)
    if n_cats == 1:
        axes = [axes]

    for ax, (cat, entries) in zip(axes, categories.items()):
        tufte_ax(ax)

        grids = [e[0] for e in entries]
        refs = [e[1] for e in entries]
        holys = [e[2] for e in entries]

        y = np.arange(len(grids))

        # Connecting lines (grey, thin)
        for i in range(len(grids)):
            ax.plot([refs[i], holys[i]], [y[i], y[i]],
                    color="#dddddd", linewidth=1.5, zorder=0)

        # Dots
        ax.scatter(refs, y, color=C_REF, s=28, zorder=5, edgecolors="none", label="torch-harmonics")
        ax.scatter(holys, y, color=C_HOLYSHT, s=28, zorder=5, edgecolors="none", label="HOLYSHT")

        # Direct labels (ms values)
        for i in range(len(grids)):
            # Label on the right of the rightmost dot
            right_val = max(refs[i], holys[i])
            ax.text(right_val * 1.25, y[i], f"{refs[i]:.1f}",
                    va="center", ha="left", fontsize=6.5, color=C_REF)
            ax.text(holys[i] * 0.7, y[i], f"{holys[i]:.2f}",
                    va="center", ha="right", fontsize=6.5, color=C_HOLYSHT)

        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels(grids, fontsize=7.5, color=C_TEXT)
        ax.set_ylim(-0.5, len(grids) - 0.5)
        ax.invert_yaxis()

        ax.set_title(cat, fontsize=8.5, color=C_TEXT, loc="left", pad=6)

        # Minimal x ticks
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.2f}"))
        ax.tick_params(axis="x", labelsize=6.5)

    # X-axis label on the middle axis
    mid = n_cats // 2
    axes[mid].set_xlabel("latency (ms, log scale)", fontsize=8, color=C_LIGHT)

    # Legend on first axis
    axes[0].legend(fontsize=6.5, loc="lower right", frameon=False,
                   labelcolor=[C_REF, C_HOLYSHT])

    fig.suptitle("Latency comparison by operation", fontsize=11,
                 color=C_TEXT, x=0.02, ha="left", y=0.98)

    fig.tight_layout(rect=[0, 0.02, 1, 0.93])
    out = os.path.join(OUT_DIR, "latency.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    data = load_data()
    os.makedirs(OUT_DIR, exist_ok=True)
    make_speedup_chart(data)
    make_latency_chart(data)
    print("Done.")
