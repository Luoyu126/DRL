from __future__ import annotations

import copy
import json

import numpy as np

from _common import parser, setup
from src.experiment import build_problem, sample_grid_target
from src.filters import filter_candidates
from src.langevin import run_langevin
from src.metrics import mmd_rbf, mode_fractions, save_json, sliced_wasserstein
from src.mlp import StandardizedRewardMLP
from src.plotting import field_plot, line_plot, scatter_compare
from src.restarts import RestartGrid
from src.score_mlp import DiffusionSchedule, ScoreMLP


def train_reward(cfg, ref, reward, rng):
    n = cfg["reward_train"]["samples"]
    global_n = max(1, n // 10)
    bounds = np.asarray(cfg["bounds"])
    x = np.concatenate([ref.sample(n - global_n, rng), rng.uniform(bounds[:, 0], bounds[:, 1], (global_n, 2))])
    model = StandardizedRewardMLP(rng)
    train = cfg["reward_train"]
    model.fit(x, reward.value(x), steps=train["steps"], batch_size=train["batch_size"], lr=train["lr"], rng=rng)
    return model


def exploration_buffer(cfg, score_model, schedule, reward_model, true_reward, rng):
    rcfg, lcfg = cfg["restart"], cfg["langevin"]
    cells = RestartGrid(cfg["bounds"], rcfg["rows"], rcfg["cols"])
    endpoints = []
    t_min = 1 / schedule.steps
    reward_prior = reward_model.value(cells.centers())
    def guided_score(x):
        base = score_model.score(x, t_min, schedule)
        # The t_min-as-clean approximation can be unreliable in low-density tails.
        # Norm clipping is recorded here as an explicit numerical safeguard.
        norm = np.linalg.norm(base, axis=1, keepdims=True)
        base = base * np.minimum(1.0, 8.0 / (norm + 1e-12))
        return base + reward_model.gradient(x) / cfg["beta"]
    for _ in range(lcfg["chains"]):
        cell = cells.choose_ucb(rng, rcfg["ucb_c"], rcfg["novelty_gamma"], endpoints, reward_prior)
        initial = cells.sample_cell(cell, rng)[None]
        end, _ = run_langevin(initial, guided_score, lcfg["steps"], lcfg["step_size"], rng, cfg["bounds"])
        end = end[0]
        endpoints.append(end)
        cells.update(cell, float(true_reward.value(end)), True)
    endpoints = np.asarray(endpoints)
    categories = filter_candidates(endpoints, true_reward, cfg, reward_model)
    accepted_b = endpoints[categories == "accepted_B"]
    return accepted_b, categories, endpoints


def run(cfg, out):
    ref, reward, xs, ys, grid, target, _, r_b = build_problem(cfg)
    rng = np.random.default_rng(cfg["seed"])
    train = cfg["score_train"]
    schedule = DiffusionSchedule(train["diffusion_steps"])
    reference_data = ref.sample(train["samples"], rng)
    score_model = ScoreMLP(rng)
    history = score_model.fit(reference_data, schedule, train["steps"], train["batch_size"], train["lr"], rng)
    sample_n = cfg["experiment"]["sample_count"]
    before = score_model.sample(sample_n, schedule, rng, cfg["bounds"])
    reward_model = train_reward(cfg, ref, reward, rng)
    buffer_b, categories, candidates = exploration_buffer(cfg, score_model, schedule, reward_model, reward, rng)
    target_samples = sample_grid_target(target, grid, sample_n, rng)
    lambdas, b_rates, mmds, swds, generated = [], [], [], [], {}
    last_distilled = score_model
    for lam in cfg["experiment"]["lambdas"]:
        distilled = copy.deepcopy(score_model)
        n_new = int(round(train["samples"] * lam))
        if n_new and len(buffer_b):
            new = buffer_b[rng.integers(0, len(buffer_b), n_new)] + rng.normal(scale=0.08, size=(n_new, 2))
            old = ref.sample(train["samples"] - n_new, rng)
            mixed = np.concatenate([old, new])
        else:
            mixed = ref.sample(train["samples"], rng)
        distilled.fit(mixed, schedule, max(1, train["steps"] // 2), train["batch_size"], train["lr"] * 0.5, rng)
        samples = distilled.sample(sample_n, schedule, rng, cfg["bounds"])
        key = f"lambda={lam:g}"
        generated[key] = samples
        lambdas.append(lam)
        b_rates.append(mode_fractions(samples, cfg)["B"])
        mmds.append(mmd_rbf(samples, target_samples))
        swds.append(sliced_wasserstein(samples, target_samples, rng))
        distilled.save(out["models"] / f"setting_{cfg['setting'].lower()}_score_lambda_{lam:g}.pkl")
        last_distilled = distilled
    t_min = 1 / schedule.steps
    score_eval = ref.sample(2000, rng)
    score_error = float(np.mean((score_model.score(score_eval, t_min, schedule) - ref.score(score_eval)) ** 2))
    report = {"setting": cfg["setting"], "calibrated_r_b": r_b, "t_min": t_min,
              "pretrain_final_loss": history[-1][1], "reference_weighted_clean_score_mse_at_t_min_approximation": score_error,
              "clean_mcmc_score_norm_clip": 8.0,
              "pretrain_mode_fractions": mode_fractions(before, cfg), "accepted_B_buffer_size": len(buffer_b),
              "filter_counts": {name: int(np.sum(categories == name)) for name in np.unique(categories)},
              "lambda_sweep": [{"lambda": float(l), "B_fraction": float(b), "mmd": float(m), "sliced_wasserstein": float(w)}
                               for l, b, m, w in zip(lambdas, b_rates, mmds, swds)]}
    prefix = f"setting_{cfg['setting'].lower()}"
    score_model.save(out["models"] / f"{prefix}_score_pretrained.pkl")
    scatter_compare({"before DSM": before, **generated}, cfg["bounds"], out["figures"] / f"{prefix}_phase3_generated.png", "DSM distillation")
    score_grid = grid.reshape(-1, 2)
    field_plot(grid, target, score_model.score(score_grid, t_min, schedule).reshape(grid.shape),
               out["figures"] / f"{prefix}_phase3_score_before.png", "Pretrained score at t_min")
    field_plot(grid, target, last_distilled.score(score_grid, t_min, schedule).reshape(grid.shape),
               out["figures"] / f"{prefix}_phase3_score_after.png", "Distilled score at t_min")
    line_plot({"B fraction": (lambdas, b_rates)}, out["figures"] / f"{prefix}_phase3_discovery.png", "lambda", "B fraction", "Discovery vs distillation mix")
    line_plot({"MMD": (lambdas, mmds), "sliced W1": (lambdas, swds)}, out["figures"] / f"{prefix}_phase3_fidelity.png", "lambda", "distance", "Target fidelity")
    save_json(report, out["metrics"] / f"{prefix}_phase3.json")
    return report


if __name__ == "__main__":
    args = parser("Phase 3: score MLP and DSM distillation").parse_args()
    cfg, out = setup(args)
    print(json.dumps(run(cfg, out), indent=2))
