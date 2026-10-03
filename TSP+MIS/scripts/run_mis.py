from __future__ import annotations

import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsp_mis_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from _common import make_parser, setup
from src.mis import (distill_labels, evaluate, generate_graphs, graph_contexts,
                     heuristic_labels, mask_to_bits, maximum_independent_set)
from src.nn import ConditionalBitDenoiser
from src.utils import bar_plot, line_plot, save_json


def plot_example(graph, solutions, path):
    n = len(graph); theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack([np.cos(theta), np.sin(theta)])
    fig, axes = plt.subplots(1, len(solutions), figsize=(4 * len(solutions), 4), squeeze=False)
    for ax, (name, solution) in zip(axes[0], solutions.items()):
        for i, j in zip(*np.triu_indices(n, 1)):
            if graph[i, j]:
                ax.plot(positions[[i, j], 0], positions[[i, j], 1], color="lightgray", lw=.5)
        colors = ["tab:red" if solution[i] else "tab:blue" for i in range(n)]
        ax.scatter(positions[:, 0], positions[:, 1], c=colors, s=55)
        ax.set(title=f"{name} (size={int(np.sum(solution))})", aspect="equal"); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(cfg, out):
    spec = cfg["mis"]; rng = np.random.default_rng(cfg["seed"])
    started = time.perf_counter()
    train_graphs = generate_graphs(spec["train_graphs"], spec["nodes"], spec["edge_probability"], rng)
    test_graphs = generate_graphs(spec["test_graphs"], spec["nodes"], spec["edge_probability"], rng)
    train_contexts, test_contexts = graph_contexts(train_graphs), graph_contexts(test_graphs)
    train_labels = heuristic_labels(train_graphs, rng)
    exact_started = time.perf_counter()
    exact_masks = [maximum_independent_set(graph) for graph in test_graphs]
    exact_seconds = time.perf_counter() - exact_started
    model = ConditionalBitDenoiser(train_contexts.shape[1], spec["nodes"], rng)
    history = model.fit(train_contexts, train_labels, spec["train_steps"], spec["batch_size"], spec["lr"], rng, reset_stats=True)
    model.save(out["models"] / "mis_pretrained.pkl")
    before, random_trace, ucb_trace, eval_seconds = evaluate(
        model, test_graphs, test_contexts, exact_masks, spec, np.random.default_rng(cfg["seed"] + 1))
    distilled_labels, label_improvements = distill_labels(model, train_graphs, train_contexts, spec,
                                                           np.random.default_rng(cfg["seed"] + 2))
    fine_history = model.fit(train_contexts, distilled_labels, spec["finetune_steps"], spec["batch_size"],
                             spec["lr"] * .5, rng)
    model.save(out["models"] / "mis_distilled.pkl")
    after, _, _, post_eval_seconds = evaluate(
        model, test_graphs, test_contexts, exact_masks, spec, np.random.default_rng(cfg["seed"] + 1))
    report = {
        "problem": "maximum_independent_set", "nodes": spec["nodes"], "edge_probability": spec["edge_probability"],
        "train_graphs": spec["train_graphs"], "test_graphs": spec["test_graphs"],
        "pretrain_final_bce": history[-1][1], "finetune_final_bce": fine_history[-1][1],
        "distill_label_improvement_mean": float(np.mean(label_improvements)),
        "before_distillation": before, "after_distillation": after,
        "cost": {"exact_label_seconds": exact_seconds, "pre_eval_seconds": eval_seconds,
                 "post_eval_seconds": post_eval_seconds, "total_seconds": time.perf_counter() - started,
                 "restart_budget": spec["restart_budget"], "distill_budget": spec["distill_budget"]},
    }
    save_json(report, out["metrics"] / "mis_cpu_validation.json")
    bar_plot({name: (stats["size_mean"], stats["size_std"]) for name, stats in before.items()},
             out["figures"] / "mis_methods.png", "independent-set size", "MIS CPU baselines")
    line_plot({"random restart": (np.arange(1, len(random_trace) + 1), random_trace),
               "UCB restart": (np.arange(1, len(ucb_trace) + 1), ucb_trace)},
              out["figures"] / "mis_budget_curve.png", "objective queries", "best size", "MIS anytime discovery")
    opt = mask_to_bits(exact_masks[0], spec["nodes"])
    # Visualize heuristic training target against exact solution on one graph.
    plot_example(test_graphs[0], {"heuristic": heuristic_labels(test_graphs[:1], rng)[0], "optimal": opt},
                 out["figures"] / "mis_example.png")
    return report


if __name__ == "__main__":
    args = make_parser("CPU MIS validation").parse_args(); cfg, out = setup(args)
    print(json.dumps(run(cfg, out), indent=2))
