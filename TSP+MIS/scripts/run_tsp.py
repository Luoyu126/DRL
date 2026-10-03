from __future__ import annotations

import json
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsp_mis_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from _common import make_parser, setup
from src.nn import ConditionalBitDenoiser
from src.tsp import (distill_labels, evaluate, generate_instances, held_karp,
                     heuristic_labels, instance_contexts, tour_length)
from src.utils import bar_plot, line_plot, save_json


def plot_example(points, tours, path):
    fig, axes = plt.subplots(1, len(tours), figsize=(4 * len(tours), 4), squeeze=False)
    for ax, (name, tour) in zip(axes[0], tours.items()):
        cycle = np.append(tour, tour[0])
        ax.plot(points[cycle, 0], points[cycle, 1], marker="o")
        for i, (x, y) in enumerate(points):
            ax.text(x, y, str(i), fontsize=7)
        ax.set(title=f"{name} (L={tour_length(points, tour):.3f})", aspect="equal")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run(cfg, out):
    spec = cfg["tsp"]; rng = np.random.default_rng(cfg["seed"] + 1000)
    started = time.perf_counter()
    train_instances = generate_instances(spec["train_graphs"], spec["nodes"], rng)
    test_instances = generate_instances(spec["test_graphs"], spec["nodes"], rng)
    train_contexts = instance_contexts(train_instances)
    test_contexts = instance_contexts(test_instances)
    train_labels, heuristic_tours = heuristic_labels(train_instances)
    exact_started = time.perf_counter(); optimal_tours = [held_karp(points) for points in test_instances]
    exact_seconds = time.perf_counter() - exact_started
    solution_dim = spec["nodes"] * (spec["nodes"] - 1) // 2
    model = ConditionalBitDenoiser(train_contexts.shape[1], solution_dim, rng)
    history = model.fit(train_contexts, train_labels, spec["train_steps"], spec["batch_size"], spec["lr"], rng, reset_stats=True)
    model.save(out["models"] / "tsp_pretrained.pkl")
    before, random_trace, ucb_trace, eval_seconds = evaluate(
        model, test_instances, test_contexts, optimal_tours, spec, np.random.default_rng(cfg["seed"] + 1001))
    distilled_labels, improvements = distill_labels(model, train_instances, train_contexts, spec,
                                                     np.random.default_rng(cfg["seed"] + 1002))
    fine_history = model.fit(train_contexts, distilled_labels, spec["finetune_steps"], spec["batch_size"],
                             spec["lr"] * .5, rng)
    model.save(out["models"] / "tsp_distilled.pkl")
    after, _, _, post_eval_seconds = evaluate(
        model, test_instances, test_contexts, optimal_tours, spec, np.random.default_rng(cfg["seed"] + 1001))
    report = {
        "problem": "euclidean_tsp", "context_representation": "upper_triangle_edge_distances",
        "nodes": spec["nodes"], "train_graphs": spec["train_graphs"],
        "test_graphs": spec["test_graphs"], "pretrain_final_bce": history[-1][1],
        "finetune_final_bce": fine_history[-1][1], "distill_length_improvement_mean": float(np.mean(improvements)),
        "before_distillation": before, "after_distillation": after,
        "cost": {"exact_label_seconds": exact_seconds, "pre_eval_seconds": eval_seconds,
                 "post_eval_seconds": post_eval_seconds, "total_seconds": time.perf_counter() - started,
                 "restart_budget": spec["restart_budget"], "distill_budget": spec["distill_budget"]},
    }
    save_json(report, out["metrics"] / "tsp_cpu_validation.json")
    bar_plot({name: (stats["optimality_gap_mean"] * 100, 0) for name, stats in before.items()},
             out["figures"] / "tsp_methods.png", "optimality gap (%)", "TSP CPU baselines")
    line_plot({"random restart": (np.arange(1, len(random_trace) + 1), random_trace),
               "UCB restart": (np.arange(1, len(ucb_trace) + 1), ucb_trace)},
              out["figures"] / "tsp_budget_curve.png", "objective queries", "best tour length", "TSP anytime discovery")
    plot_example(test_instances[0], {"nearest neighbor": heuristic_labels(test_instances[:1])[1][0],
                                     "optimal": optimal_tours[0]}, out["figures"] / "tsp_example.png")
    return report


if __name__ == "__main__":
    args = make_parser("CPU TSP validation").parse_args(); cfg, out = setup(args)
    print(json.dumps(run(cfg, out), indent=2))
