from __future__ import annotations

import json
from copy import deepcopy

import numpy as np

from _common import parser, setup
from src.distributions import classify_modes
from src.experiment import build_problem, restart_experiment, sample_grid_target
from src.langevin import run_gradient_ascent, run_langevin, target_score
from src.metrics import mmd_rbf, mode_fractions, save_json, sliced_wasserstein
from src.plotting import scatter_compare, trajectory_compare, visit_heatmaps
from src.restarts import oracle_initial


def run(cfg, out):
    ref, reward, xs, ys, grid, target, _, r_b = build_problem(cfg)
    all_results, visual = {}, {}
    for seed in cfg["experiment"]["seeds"]:
        rng = np.random.default_rng(seed)
        n, lcfg = cfg["langevin"]["chains"], cfg["langevin"]
        base = ref.sample(n, rng)
        score = lambda x: target_score(x, ref, reward, cfg["beta"])
        local, local_traj = run_langevin(base, score, lcfg["steps"], lcfg["step_size"], rng, cfg["bounds"], keep_trajectory=True)
        high, _ = run_langevin(base, score, lcfg["steps"], lcfg["step_size"] * 2.5, rng, cfg["bounds"], noise_scale=1.4)
        ascent = run_gradient_ascent(base, score, lcfg["steps"], lcfg["step_size"], cfg["bounds"])
        oracle, _ = run_langevin(oracle_initial(cfg["mu_b"], n, rng), score, lcfg["steps"], lcfg["step_size"], rng, cfg["bounds"])
        random_result = restart_experiment(cfg, ref, reward, "random", seed)
        ucb_result = restart_experiment(cfg, ref, reward, "ucb", seed)
        importance_pool = ref.sample(max(1000, 20 * n), rng)
        weights = np.exp((reward.value(importance_pool) - reward.value(importance_pool).max()) / cfg["beta"])
        importance = importance_pool[rng.choice(len(importance_pool), n, p=weights / weights.sum())]
        target_samples = sample_grid_target(target, grid, n, rng)
        methods = {"base_sampling": base, "gradient_ascent": ascent, "local_guidance": local,
                   "high_noise": high, "importance_resampling": importance,
                   "random_restart": random_result["endpoints"], "ucb_restart": ucb_result["endpoints"],
                   "oracle_restart": oracle}
        seed_result = {}
        for name, samples in methods.items():
            seed_result[name] = {"mode_fractions": mode_fractions(samples, cfg),
                                 "mmd_to_target": mmd_rbf(samples, target_samples),
                                 "sliced_wasserstein": sliced_wasserstein(samples, target_samples, rng)}
        for name, result in (("random_restart", random_result), ("ucb_restart", ucb_result)):
            seed_result[name].update(first_b_chain=result["first_b_chain"], discovered_b=result["discovered_b"],
                                     score_evaluations=result["score_evaluations"], reward_queries=result["reward_queries"],
                                     wall_seconds=result["wall_seconds"])
        all_results[str(seed)] = seed_result
        if not visual:
            visual = methods
            local_paths = [local_traj[:, i] for i in range(min(6, local_traj.shape[1]))]
            trajectory_compare({"local guidance": local_paths, "random restart": random_result["trajectories"][:6],
                                "UCB restart": ucb_result["trajectories"][:6]}, target, grid,
                               out["figures"] / f"setting_{cfg['setting'].lower()}_phase1_trajectories.png")
            visit_heatmaps({"random restart": random_result["visits"], "UCB restart": ucb_result["visits"]},
                           cfg["restart"]["rows"], cfg["restart"]["cols"],
                           out["figures"] / f"setting_{cfg['setting'].lower()}_phase1_restart_visits.png")
    summary = {}
    for method in next(iter(all_results.values())):
        b = [r[method]["mode_fractions"]["B"] for r in all_results.values()]
        summary[method] = {"B_fraction_mean": float(np.mean(b)), "B_fraction_std": float(np.std(b))}
        if method in ("random_restart", "ucb_restart"):
            times = [r[method]["first_b_chain"] for r in all_results.values() if r[method]["first_b_chain"] is not None]
            summary[method].update(discovery_rate=float(np.mean([r[method]["discovered_b"] for r in all_results.values()])),
                                   median_first_b=float(np.median(times)) if times else None)
    report = {"setting": cfg["setting"], "calibrated_r_b": r_b, "per_seed": all_results, "summary": summary}
    scatter_compare(visual, cfg["bounds"], out["figures"] / f"setting_{cfg['setting'].lower()}_phase1_samples.png", "Exact-component baselines")
    save_json(report, out["metrics"] / f"setting_{cfg['setting'].lower()}_phase1.json")
    return report


if __name__ == "__main__":
    args = parser("Phase 1: exact score and reward baselines").parse_args()
    cfg, out = setup(args)
    print(json.dumps(run(cfg, out)["summary"], indent=2))
