from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .distributions import classify_modes


def mode_fractions(samples, cfg):
    labels = classify_modes(samples, cfg["mu_a"], cfg["mu_b"], cfg["basin_radius"])
    return {label: float(np.mean(labels == label)) for label in ("A", "B", "other")}


def mmd_rbf(x, y, max_points=1000):
    x, y = np.asarray(x)[:max_points], np.asarray(y)[:max_points]
    z = np.concatenate([x, y])
    d2 = np.sum((z[:, None] - z[None, :]) ** 2, axis=-1)
    bandwidth = np.median(d2[d2 > 0]) + 1e-12
    kxx = np.exp(-np.sum((x[:, None] - x[None, :]) ** 2, axis=-1) / bandwidth)
    kyy = np.exp(-np.sum((y[:, None] - y[None, :]) ** 2, axis=-1) / bandwidth)
    kxy = np.exp(-np.sum((x[:, None] - y[None, :]) ** 2, axis=-1) / bandwidth)
    return float(kxx.mean() + kyy.mean() - 2 * kxy.mean())


def sliced_wasserstein(x, y, rng, projections=64):
    dirs = rng.normal(size=(projections, 2))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    n = min(len(x), len(y))
    idx_x, idx_y = rng.choice(len(x), n, replace=False), rng.choice(len(y), n, replace=False)
    px, py = np.asarray(x)[idx_x] @ dirs.T, np.asarray(y)[idx_y] @ dirs.T
    return float(np.mean(np.abs(np.sort(px, axis=0) - np.sort(py, axis=0))))


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)

