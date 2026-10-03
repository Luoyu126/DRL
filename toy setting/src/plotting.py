from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "reward_exploration_matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def density_panel(grid, fields, path, trajectories=None, points=None):
    cols = len(fields)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4), squeeze=False)
    extent = [grid[..., 0].min(), grid[..., 0].max(), grid[..., 1].min(), grid[..., 1].max()]
    for ax, (title, values) in zip(axes[0], fields.items()):
        image = ax.imshow(values, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        fig.colorbar(image, ax=ax, shrink=0.8)
        if trajectories is not None:
            for trajectory in trajectories:
                ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.65, lw=0.8)
        if points is not None:
            ax.scatter(points[:, 0], points[:, 1], s=8, c="white", edgecolor="black", linewidth=0.2)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def field_plot(grid, background, vectors, path, title, stride=8):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    extent = [grid[..., 0].min(), grid[..., 0].max(), grid[..., 1].min(), grid[..., 1].max()]
    im = ax.imshow(background, origin="lower", extent=extent, aspect="auto", cmap="viridis")
    g, v = grid[::stride, ::stride], vectors[::stride, ::stride]
    ax.quiver(g[..., 0], g[..., 1], v[..., 0], v[..., 1], color="white", alpha=0.8)
    ax.set(title=title, xlabel="x1", ylabel="x2")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scatter_compare(groups, bounds, path, title):
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    for ax, (name, samples) in zip(axes[0], groups.items()):
        ax.scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.5)
        ax.set(xlim=bounds[0], ylim=bounds[1], title=name, xlabel="x1", ylabel="x2")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trajectory_compare(groups, background, grid, path):
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    extent = [grid[..., 0].min(), grid[..., 0].max(), grid[..., 1].min(), grid[..., 1].max()]
    for ax, (name, trajectories) in zip(axes[0], groups.items()):
        ax.imshow(background, origin="lower", extent=extent, aspect="auto", cmap="Greys", alpha=0.55)
        for trajectory in trajectories:
            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.7, lw=0.8)
        ax.set(title=name, xlabel="x1", ylabel="x2")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def visit_heatmaps(groups, rows, cols, path):
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    for ax, (name, counts) in zip(axes[0], groups.items()):
        im = ax.imshow(np.asarray(counts).reshape(rows, cols), origin="lower", cmap="magma")
        ax.set(title=name, xlabel="restart column", ylabel="restart row")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def categorized_scatter(samples, categories, bounds, path, title):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for category in np.unique(categories):
        mask = categories == category
        ax.scatter(samples[mask, 0], samples[mask, 1], s=24, alpha=0.75, label=category)
    ax.set(xlim=bounds[0], ylim=bounds[1], title=title, xlabel="x1", ylabel="x2")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def line_plot(series, path, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, (x, y) in series.items():
        ax.plot(x, y, marker="o", label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
