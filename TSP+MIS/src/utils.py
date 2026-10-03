from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsp_mis_matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_config(path=None, smoke=False):
    with open(path or ROOT / "config.json", encoding="utf-8") as handle:
        cfg = json.load(handle)
    if smoke:
        for key in ("mis", "tsp"):
            cfg[key]["train_graphs"] = 100
            cfg[key]["test_graphs"] = 20
            cfg[key]["train_steps"] = 120
            cfg[key]["finetune_steps"] = 60
            cfg[key]["restart_budget"] = 8
            cfg[key]["distill_budget"] = 4
            cfg[key]["samples_per_graph"] = 3
    return cfg


def output_dirs(base=None):
    root = Path(base) if base else ROOT / "outputs"
    result = {name: root / name for name in ("metrics", "figures", "models")}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def bar_plot(groups, path, ylabel, title):
    names, values, errors = [], [], []
    for name, stats in groups.items():
        names.append(name); values.append(stats[0]); errors.append(stats[1])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, values, yerr=errors, capsize=3)
    ax.set(ylabel=ylabel, title=title)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def line_plot(series, path, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (x, y) in series.items():
        ax.plot(x, y, marker="o", label=name)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

