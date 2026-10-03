from __future__ import annotations

import json

import numpy as np

from _common import parser, setup
from src.experiment import build_problem, restart_experiment
from src.filters import filter_candidates
from src.metrics import mode_fractions, save_json
from src.mlp import StandardizedRewardMLP
from src.plotting import categorized_scatter, density_panel


def reward_dataset(cfg, ref, reward, rng, kind):
    n = cfg["reward_train"]["samples"]
    bounds = np.asarray(cfg["bounds"])
    if kind == "dense_grid":
        x = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, 2))
    elif kind == "biased_queries":
        global_n = max(1, n // 10)
        x = np.concatenate([ref.sample(n - global_n, rng), rng.uniform(bounds[:, 0], bounds[:, 1], (global_n, 2))])
    else:
        raise ValueError(kind)
    return x, reward.value(x)


def run(cfg, out):
    ref, reward, xs, ys, grid, target, _, r_b = build_problem(cfg)
    rng = np.random.default_rng(cfg["seed"])
    reports, models = {}, {}
    for kind in ("dense_grid", "biased_queries"):
        x, y = reward_dataset(cfg, ref, reward, rng, kind)
        model = StandardizedRewardMLP(rng)
        train = cfg["reward_train"]
        history = model.fit(x, y, steps=train["steps"], batch_size=train["batch_size"], lr=train["lr"], rng=rng)
        pred = model.value(grid.reshape(-1, 2)).reshape(grid.shape[:2])
        true = reward.value(grid)
        grad_pred = model.gradient(grid.reshape(-1, 2)).reshape(grid.shape)
        grad_true = reward.gradient(grid)
        reports[kind] = {"value_mse": float(np.mean((pred - true) ** 2)),
                         "gradient_mse": float(np.mean((grad_pred - grad_true) ** 2)),
                         "final_train_loss": history[-1][1],
                         "max_predicted_reward": float(pred.max()), "max_true_reward": float(true.max())}
        density_panel(grid, {"R_true": true, f"R_mlp_{kind}": pred, "absolute error": np.abs(pred - true)},
                      out["figures"] / f"setting_{cfg['setting'].lower()}_phase2_{kind}.png")
        model.save(out["models"] / f"setting_{cfg['setting'].lower()}_reward_{kind}.pkl")
        models[kind] = model
    exact = restart_experiment(cfg, ref, reward, "ucb", cfg["seed"])
    learned = restart_experiment(cfg, ref, reward, "ucb", cfg["seed"], models["biased_queries"])
    categories = filter_candidates(learned["endpoints"], reward, cfg, models["biased_queries"])
    accepted = np.char.startswith(categories.astype(str), "accepted")
    reports["comparison"] = {
        "true_reward_gradient": mode_fractions(exact["endpoints"], cfg),
        "reward_mlp_no_filter": mode_fractions(learned["endpoints"], cfg),
        "reward_mlp_true_reward_filter": mode_fractions(learned["endpoints"][accepted], cfg) if accepted.any() else {"A": 0, "B": 0, "other": 0},
        "filter_counts": {name: int(np.sum(categories == name)) for name in np.unique(categories)},
        "mean_exploitation_gap": float(np.mean(models["biased_queries"].value(learned["endpoints"]) - reward.value(learned["endpoints"])))
    }
    categorized_scatter(learned["endpoints"], categories, cfg["bounds"],
                        out["figures"] / f"setting_{cfg['setting'].lower()}_phase2_filter.png",
                        "External-filter decisions")
    report = {"setting": cfg["setting"], "calibrated_r_b": r_b, **reports}
    save_json(report, out["metrics"] / f"setting_{cfg['setting'].lower()}_phase2.json")
    return report


if __name__ == "__main__":
    args = parser("Phase 2: reward MLP and external filter").parse_args()
    cfg, out = setup(args)
    print(json.dumps(run(cfg, out), indent=2))
