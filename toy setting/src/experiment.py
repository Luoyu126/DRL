from __future__ import annotations

import time

import numpy as np

from .distributions import classify_modes, make_grid, normalized_target, reference_from_config
from .langevin import run_langevin, target_score
from .restarts import RestartGrid
from .rewards import calibrate_reward, reward_from_config


def build_problem(cfg):
    ref = reference_from_config(cfg)
    xs, ys, grid = make_grid(cfg["bounds"], cfg["grid_size"])
    r_b = calibrate_reward(cfg, ref, grid, xs, ys) if cfg["r_b"] == "auto" else float(cfg["r_b"])
    reward = reward_from_config(cfg, r_b)
    density, log_z = normalized_target(ref, reward, grid, xs, ys, cfg["beta"])
    return ref, reward, xs, ys, grid, density, log_z, r_b


def restart_experiment(cfg, ref, reward, strategy, seed, learned_reward=None):
    rng = np.random.default_rng(seed)
    rcfg, lcfg = cfg["restart"], cfg["langevin"]
    grid = RestartGrid(cfg["bounds"], rcfg["rows"], rcfg["cols"])
    score_reward = reward if learned_reward is None else learned_reward
    score_fn = lambda x: target_score(x, ref, score_reward, cfg["beta"])
    endpoints, trajectories, replay = [], [], []
    first_b = None
    start_time = time.perf_counter()
    for chain in range(lcfg["chains"]):
        if strategy == "random":
            cell = grid.choose_random(rng)
        elif strategy == "ucb":
            cell = grid.choose_ucb(rng, rcfg["ucb_c"], rcfg["novelty_gamma"], replay)
        else:
            raise ValueError(strategy)
        initial = grid.sample_cell(cell, rng)[None, :]
        endpoint, traj = run_langevin(initial, score_fn, lcfg["steps"], lcfg["step_size"], rng,
                                        cfg["bounds"], keep_trajectory=True)
        endpoint = endpoint[0]
        label = classify_modes(endpoint[None], cfg["mu_a"], cfg["mu_b"], cfg["basin_radius"])[0]
        true_value = float(reward.value(endpoint))
        grid.update(cell, true_value, label in ("A", "B"))
        endpoints.append(endpoint)
        trajectories.append(traj[:, 0])
        replay.append(endpoint)
        if label == "B" and first_b is None:
            first_b = chain + 1
    return {
        "endpoints": np.asarray(endpoints), "trajectories": trajectories, "visits": grid.counts,
        "first_b_chain": first_b, "discovered_b": first_b is not None,
        "score_evaluations": lcfg["chains"] * lcfg["steps"], "reward_queries": lcfg["chains"],
        "wall_seconds": time.perf_counter() - start_time,
    }


def sample_grid_target(density, grid, n, rng):
    probabilities = density.ravel() / density.sum()
    idx = rng.choice(len(probabilities), n, p=probabilities)
    flat = grid.reshape(-1, 2)
    return flat[idx]

