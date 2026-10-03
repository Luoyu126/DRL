from __future__ import annotations

import copy
import json
import time

import numpy as np

from _common import parser, setup
from src.complex_setting import (build_complex_problem, classify_named_modes,
                                 named_fractions)
from src.experiment import sample_grid_target
from src.langevin import run_langevin, target_score
from src.metrics import mmd_rbf, save_json, sliced_wasserstein
from src.mlp import StandardizedRewardMLP
from src.plotting import (categorized_scatter, density_panel, line_plot,
                          scatter_compare, trajectory_compare, visit_heatmaps)
from src.restarts import RestartGrid
from src.score_mlp import DiffusionSchedule, ScoreMLP


NOVEL_MODES = ("broad", "needle", "boundary")


def _mode_metrics(samples, peaks, target_samples, rng):
    fractions = named_fractions(samples, peaks)
    return {
        "mode_fractions": fractions,
        "novel_mode_coverage": int(sum(fractions.get(name, 0) >= 0.005 for name in NOVEL_MODES)),
        "novel_fraction": float(sum(fractions.get(name, 0) for name in NOVEL_MODES)),
        "mmd_to_target": mmd_rbf(samples, target_samples),
        "sliced_wasserstein": sliced_wasserstein(samples, target_samples, rng),
    }


def _first_hits(labels, names=NOVEL_MODES):
    result = {}
    for name in names:
        indices = np.flatnonzero(labels == name)
        result[name] = int(indices[0] + 1) if len(indices) else None
    return result


def run_restart_policy(cfg, ref, proposal_reward, true_reward, peaks, strategy, seed,
                       chains=None, use_prior=False, score_model=None, schedule=None, step_sizes=None,
                       novelty_gamma=None):
    rng = np.random.default_rng(seed)
    lcfg, rcfg = cfg["langevin"], cfg["restart"]
    chains = chains or lcfg["chains"]
    cells = RestartGrid(cfg["bounds"], rcfg["rows"], rcfg["cols"])
    endpoints, trajectories = [], []
    prior = proposal_reward.value(cells.centers()) if use_prior else None
    if score_model is None:
        base_score = ref.score
    else:
        t_min = 1 / schedule.steps
        def base_score(x):
            score = score_model.score(x, t_min, schedule)
            norm = np.linalg.norm(score, axis=1, keepdims=True)
            return score * np.minimum(1.0, 8.0 / (norm + 1e-12))
    score_fn = lambda x: base_score(x) + proposal_reward.gradient(x) / cfg["beta"]
    step_sizes = step_sizes or [lcfg["step_size"]]
    started = time.perf_counter()
    for _ in range(chains):
        if strategy == "random":
            cell = cells.choose_random(rng)
        elif strategy == "ucb":
            gamma = rcfg["novelty_gamma"] if novelty_gamma is None else novelty_gamma
            cell = cells.choose_ucb(rng, rcfg["ucb_c"], gamma, endpoints, prior)
        else:
            raise ValueError(strategy)
        initial = cells.sample_cell(cell, rng)[None, :]
        proposals = [run_langevin(initial, score_fn, lcfg["steps"], step_size, rng,
                                  cfg["bounds"], keep_trajectory=True) for step_size in step_sizes]
        energies = [float(ref.logpdf(item[0])[0] + proposal_reward.value(item[0])[0] / cfg["beta"])
                    for item in proposals]
        endpoint_array, trajectory = proposals[int(np.argmax(energies))]
        endpoint = endpoint_array[0]
        label = classify_named_modes(endpoint[None], peaks)[0]
        cells.update(cell, float(true_reward.value(endpoint)), label != "background")
        endpoints.append(endpoint)
        trajectories.append(trajectory[:, 0])
    endpoints = np.asarray(endpoints)
    labels = classify_named_modes(endpoints, peaks)
    return {"endpoints": endpoints, "labels": labels, "trajectories": trajectories,
            "visits": cells.counts, "first_hits": _first_hits(labels),
            "score_evaluations": chains * lcfg["steps"] * len(step_sizes), "reward_queries": chains,
            "wall_seconds": time.perf_counter() - started}


def exact_exploration(cfg, ref, reward, peaks, grid, target, out):
    per_seed, budget_curves, first_visual = {}, {"random": [], "ucb": []}, None
    max_budget = max(cfg["experiment"]["budgets"])
    for seed in cfg["experiment"]["seeds"]:
        rng = np.random.default_rng(seed)
        local_initial = ref.sample(max_budget, rng)
        exact_score = lambda x: target_score(x, ref, reward, cfg["beta"])
        local, local_traj = run_langevin(local_initial, exact_score, cfg["langevin"]["steps"],
                                        cfg["langevin"]["step_size"], rng, cfg["bounds"], keep_trajectory=True)
        noisy, _ = run_langevin(local_initial, exact_score, cfg["langevin"]["steps"],
                                cfg["langevin"]["step_size"] * 2.2, rng, cfg["bounds"], noise_scale=1.5)
        random_result = run_restart_policy(cfg, ref, reward, reward, peaks, "random", seed, max_budget)
        ucb_result = run_restart_policy(cfg, ref, reward, reward, peaks, "ucb", seed, max_budget)
        multiscale_steps = [cfg["langevin"]["step_size"], 0.004]
        random_multi = run_restart_policy(cfg, ref, reward, reward, peaks, "random", seed, max_budget,
                                          step_sizes=multiscale_steps)
        ucb_multi = run_restart_policy(cfg, ref, reward, reward, peaks, "ucb", seed, max_budget,
                                       step_sizes=multiscale_steps)
        oracle_initial = np.concatenate([
            np.asarray(next(p["mu"] for p in peaks if p["name"] == name))[None] + rng.normal(scale=.12, size=(max_budget // 3, 2))
            for name in NOVEL_MODES
        ])
        oracle, _ = run_langevin(oracle_initial, exact_score, cfg["langevin"]["steps"],
                                 cfg["langevin"]["step_size"], rng, cfg["bounds"])
        target_samples = sample_grid_target(target, grid, max_budget, rng)
        methods = {"reference": local_initial, "local": local, "high_noise": noisy,
                   "random": random_result["endpoints"], "ucb": ucb_result["endpoints"], "oracle": oracle}
        methods.update(random_multiscale=random_multi["endpoints"], ucb_multiscale=ucb_multi["endpoints"])
        seed_report = {name: _mode_metrics(samples, peaks, target_samples, rng) for name, samples in methods.items()}
        for name, result in (("random", random_result), ("ucb", ucb_result),
                             ("random_multiscale", random_multi), ("ucb_multiscale", ucb_multi)):
            seed_report[name].update(first_hits=result["first_hits"], score_evaluations=result["score_evaluations"],
                                     reward_queries=result["reward_queries"], wall_seconds=result["wall_seconds"])
        per_seed[str(seed)] = seed_report
        for strategy, result in (("random", random_result), ("ucb", ucb_result)):
            budget_curves[strategy].append([
                len(set(result["labels"][:budget]).intersection(NOVEL_MODES))
                for budget in cfg["experiment"]["budgets"]
            ])
        if first_visual is None:
            first_visual = (local_traj, random_result, ucb_result)
    summary = {}
    for method in next(iter(per_seed.values())):
        values = [report[method] for report in per_seed.values()]
        summary[method] = {
            "novel_fraction_mean": float(np.mean([v["novel_fraction"] for v in values])),
            "novel_fraction_std": float(np.std([v["novel_fraction"] for v in values])),
            "novel_mode_coverage_mean": float(np.mean([v["novel_mode_coverage"] for v in values])),
        }
        if method in ("random", "ucb", "random_multiscale", "ucb_multiscale"):
            summary[method]["median_first_hits"] = {
                mode: (float(np.median(hits)) if (hits := [v["first_hits"][mode] for v in values if v["first_hits"][mode] is not None]) else None)
                for mode in NOVEL_MODES
            }
    curves = {name: np.mean(values, axis=0).tolist() for name, values in budget_curves.items()}
    local_traj, random_result, ucb_result = first_visual
    local_paths = [local_traj[:, i] for i in range(min(6, local_traj.shape[1]))]
    trajectory_compare({"local": local_paths, "random": random_result["trajectories"][:6], "UCB": ucb_result["trajectories"][:6]},
                       target, grid, out["figures"] / "setting_c_exact_trajectories.png")
    visit_heatmaps({"random": random_result["visits"], "UCB": ucb_result["visits"]},
                   cfg["restart"]["rows"], cfg["restart"]["cols"], out["figures"] / "setting_c_restart_visits.png")
    line_plot({name: (cfg["experiment"]["budgets"], curve) for name, curve in curves.items()},
              out["figures"] / "setting_c_budget_coverage.png", "chain budget", "novel modes discovered", "Budget sensitivity")
    stability = {}
    rng = np.random.default_rng(cfg["seed"] + 909)
    exact_score = lambda x: target_score(x, ref, reward, cfg["beta"])
    for step_size in (cfg["langevin"]["step_size"], 0.008, 0.004, 0.002):
        stability[str(step_size)] = {}
        for peak in peaks:
            initial = np.asarray(peak["mu"])[None] + rng.normal(scale=0.08, size=(96, 2))
            endpoint, _ = run_langevin(initial, exact_score, cfg["langevin"]["steps"], step_size, rng, cfg["bounds"])
            labels = classify_named_modes(endpoint, peaks)
            stability[str(step_size)][peak["name"]] = float(np.mean(labels == peak["name"]))
    line_plot({name: ([float(x) for x in stability], [stability[x][name] for x in stability])
               for name in [p["name"] for p in peaks]}, out["figures"] / "setting_c_kernel_stability.png",
              "Langevin step size", "retention probability", "Mode stability versus kernel scale")
    return {"per_seed": per_seed, "summary": summary, "budget_mode_coverage": curves,
            "kernel_stability": stability}


def make_reward_dataset(kind, cfg, ref, reward, peaks, rng):
    n = cfg["reward_train"]["samples"]
    bounds = np.asarray(cfg["bounds"])
    if kind == "dense":
        x = rng.uniform(bounds[:, 0], bounds[:, 1], (n, 2))
    elif kind == "biased":
        global_n = n // 20
        x = np.concatenate([ref.sample(n - global_n, rng), rng.uniform(bounds[:, 0], bounds[:, 1], (global_n, 2))])
    elif kind == "blind":
        x = ref.sample(n, rng)
    elif kind == "peak_aware":
        uniform_n = int(n * .65)
        per_peak = (n - uniform_n) // len(peaks)
        focused = []
        for peak in peaks:
            tau = np.asarray(peak["tau"])
            focused.append(np.asarray(peak["mu"])[None] + rng.normal(size=(per_peak, 2)) * tau)
        x = np.concatenate([rng.uniform(bounds[:, 0], bounds[:, 1], (uniform_n, 2)), *focused])
        if len(x) < n:
            x = np.concatenate([x, rng.uniform(bounds[:, 0], bounds[:, 1], (n - len(x), 2))])
    else:
        raise ValueError(kind)
    y = reward.value(x)
    if kind in ("biased", "blind"):
        y = y + rng.normal(scale=0.04 * (np.std(y) + 1e-6), size=len(y))
    return x, y


def reward_model_stress(cfg, ref, reward, peaks, grid, target, out):
    rng = np.random.default_rng(cfg["seed"])
    true_grid = reward.value(grid)
    models, report = {}, {}
    for kind in ("dense", "biased", "blind", "peak_aware", "peak_aware_wide"):
        model_rng = np.random.default_rng(cfg["seed"] + 978) if kind == "peak_aware_wide" else rng
        dataset_kind = "peak_aware" if kind == "peak_aware_wide" else kind
        x, y = make_reward_dataset(dataset_kind, cfg, ref, reward, peaks, model_rng)
        hidden = 128 if kind == "peak_aware_wide" else 64
        model = StandardizedRewardMLP(model_rng, hidden=hidden)
        train = cfg["reward_train"]
        steps = max(train["steps"], 5000) if kind == "peak_aware_wide" else train["steps"]
        lr = .001 if kind == "peak_aware_wide" else train["lr"]
        history = model.fit(x, y, steps=steps, batch_size=train["batch_size"], lr=lr, rng=model_rng)
        flat = grid.reshape(-1, 2)
        pred = model.value(flat).reshape(grid.shape[:2])
        pred_grad = model.gradient(flat).reshape(grid.shape)
        true_grad = reward.gradient(grid)
        top = pred >= np.quantile(pred, .99)
        true_high = true_grid >= np.quantile(true_grid, .90)
        report[kind] = {"value_mse": float(np.mean((pred - true_grid) ** 2)),
                        "gradient_mse": float(np.mean((pred_grad - true_grad) ** 2)),
                        "top_1pct_spurious_rate": float(np.mean(~true_high[top])),
                        "max_prediction": float(pred.max()), "final_train_loss": history[-1][1],
                        "hidden_width": hidden, "train_steps": steps}
        density_panel(grid, {"R_true": true_grid, f"R_{kind}": pred, "absolute error": np.abs(pred - true_grid)},
                      out["figures"] / f"setting_c_reward_{kind}.png")
        model.save(out["models"] / f"setting_c_reward_{kind}.pkl")
        models[kind] = model
    comparisons = {}
    for kind, model in models.items():
        result = run_restart_policy(cfg, ref, model, reward, peaks, "ucb", cfg["seed"] + 101,
                                    use_prior=True, step_sizes=[cfg["langevin"]["step_size"], 0.004])
        pred = model.value(result["endpoints"])
        true = reward.value(result["endpoints"])
        labels = result["labels"]
        # A quantile is misleading because most of the box has reward near
        # zero. Use a meaningful fraction of the best true reward instead.
        threshold = 0.25 * float(np.max(true_grid))
        categories = np.full(len(labels), "rejected_low_true_reward", dtype=object)
        accepted = (labels != "background") & (true >= threshold)
        categories[accepted] = np.char.add("accepted_", labels[accepted].astype(str))
        spurious = (pred >= np.quantile(pred, .75)) & (true < threshold)
        categories[spurious] = "rejected_spurious_proxy_peak"
        comparisons[kind] = {"unfiltered": named_fractions(result["endpoints"], peaks),
                             "accepted_count": int(accepted.sum()),
                             "spurious_rejected": int(spurious.sum()),
                             "exploitation_gap_mean": float(np.mean(pred - true)),
                             "filter_counts": {name: int(np.sum(categories == name)) for name in np.unique(categories)}}
        if kind == "blind":
            categorized_scatter(result["endpoints"], categories, cfg["bounds"],
                                out["figures"] / "setting_c_blind_filter.png", "Blind reward model: oracle filter")
    report["exploration"] = comparisons
    return models, report


def _filtered_novel_buffer(endpoints, peaks, true_reward, grid):
    labels = classify_named_modes(endpoints, peaks)
    values = true_reward.value(endpoints)
    threshold = np.quantile(true_reward.value(grid), .65)
    keep = np.isin(labels, NOVEL_MODES) & (values >= threshold)
    return endpoints[keep], labels[keep], {name: int(np.sum(labels[keep] == name)) for name in NOVEL_MODES}


def _resample_buffer(buffer, labels, n, rng, strategy, target_masses):
    if n == 0 or len(buffer) == 0:
        return np.empty((0, 2))
    available = [name for name in NOVEL_MODES if np.any(labels == name)]
    if strategy == "raw":
        return buffer[rng.integers(0, len(buffer), n)]
    weights = np.asarray([target_masses[name] for name in available], dtype=float)
    weights = weights / weights.sum()
    chosen_modes = rng.choice(available, n, p=weights)
    result = []
    for name in chosen_modes:
        pool = buffer[labels == name]
        result.append(pool[rng.integers(0, len(pool))])
    return np.asarray(result)


def score_distillation(cfg, ref, reward, peaks, target_masses, exploration_reward_model, grid, target, out):
    rng = np.random.default_rng(cfg["seed"] + 303)
    train = cfg["score_train"]
    schedule = DiffusionSchedule(train["diffusion_steps"])
    reference_data = ref.sample(train["samples"], rng)
    score_model = ScoreMLP(rng, hidden=64, coordinate_scale=4.0)
    history = score_model.fit(reference_data, schedule, train["steps"], train["batch_size"], train["lr"], rng)
    n = cfg["experiment"]["sample_count"]
    before = score_model.sample(n, schedule, rng, cfg["bounds"])
    candidates = run_restart_policy(cfg, ref, exploration_reward_model, reward, peaks, "ucb", cfg["seed"] + 404,
                                    use_prior=True, score_model=score_model, schedule=schedule,
                                    step_sizes=[cfg["langevin"]["step_size"], 0.004])
    buffer, buffer_labels, counts = _filtered_novel_buffer(candidates["endpoints"], peaks, reward, grid)
    target_samples = sample_grid_target(target, grid, n, rng)
    results, generated = [], {"pretrained": before}
    last_model = score_model
    for lam in cfg["experiment"]["lambdas"]:
        model = copy.deepcopy(score_model)
        n_new = int(round(train["samples"] * lam))
        new = _resample_buffer(buffer, buffer_labels, n_new, rng, "raw", target_masses)
        old = ref.sample(train["samples"] - len(new), rng)
        mixed = np.concatenate([old, new])
        model.fit(mixed, schedule, max(1, train["steps"] // 2), train["batch_size"], train["lr"] * .5, rng)
        samples = model.sample(n, schedule, rng, cfg["bounds"])
        metrics = _mode_metrics(samples, peaks, target_samples, rng)
        results.append({"lambda": float(lam), "replay": "raw", **metrics})
        generated[f"raw lambda={lam:g}"] = samples
        model.save(out["models"] / f"setting_c_score_raw_lambda_{lam:g}.pkl")
        last_model = model
    lam = max(cfg["experiment"]["lambdas"])
    weighted = copy.deepcopy(score_model)
    n_new = int(round(train["samples"] * lam))
    new = _resample_buffer(buffer, buffer_labels, n_new, rng, "target_weighted", target_masses)
    mixed = np.concatenate([ref.sample(train["samples"] - len(new), rng), new])
    weighted.fit(mixed, schedule, max(1, train["steps"] // 2), train["batch_size"], train["lr"] * .5, rng)
    weighted_samples = weighted.sample(n, schedule, rng, cfg["bounds"])
    results.append({"lambda": float(lam), "replay": "target_weighted", **_mode_metrics(weighted_samples, peaks, target_samples, rng)})
    generated[f"target-weighted lambda={lam:g}"] = weighted_samples
    weighted.save(out["models"] / f"setting_c_score_target_weighted_lambda_{lam:g}.pkl")
    # Sanity upper bound: if every true novel mode is represented in replay,
    # can the same DSM architecture learn them? This isolates upstream
    # exploration/reward failures from downstream score-model capacity.
    oracle_labels = classify_named_modes(target_samples, peaks)
    oracle_keep = np.isin(oracle_labels, NOVEL_MODES)
    oracle_pool, oracle_pool_labels = target_samples[oracle_keep], oracle_labels[oracle_keep]
    oracle_model = copy.deepcopy(score_model)
    oracle_new = _resample_buffer(oracle_pool, oracle_pool_labels, n_new, rng, "target_weighted", target_masses)
    oracle_mixed = np.concatenate([ref.sample(train["samples"] - len(oracle_new), rng), oracle_new])
    oracle_model.fit(oracle_mixed, schedule, max(1, train["steps"] // 2), train["batch_size"], train["lr"] * .5, rng)
    oracle_samples = oracle_model.sample(n, schedule, rng, cfg["bounds"])
    results.append({"lambda": float(lam), "replay": "oracle_complete", **_mode_metrics(oracle_samples, peaks, target_samples, rng)})
    generated[f"oracle-complete lambda={lam:g}"] = oracle_samples
    oracle_model.save(out["models"] / f"setting_c_score_oracle_complete_lambda_{lam:g}.pkl")
    score_model.save(out["models"] / "setting_c_score_pretrained.pkl")
    scatter_compare(generated, cfg["bounds"], out["figures"] / "setting_c_distilled_samples.png", "Complex-setting DSM")
    raw = [item for item in results if item["replay"] == "raw"]
    line_plot({name: ([x["lambda"] for x in raw], [x["mode_fractions"][name] for x in raw]) for name in NOVEL_MODES},
              out["figures"] / "setting_c_distillation_modes.png", "lambda", "generated fraction", "Per-mode distillation")
    return {"pretrain_final_loss": history[-1][1], "pretrained": _mode_metrics(before, peaks, target_samples, rng),
            "buffer_size": len(buffer), "buffer_counts": counts, "candidate_first_hits": candidates["first_hits"],
            "sweep": results}


def run(cfg, out):
    ref, reward, peaks, xs, ys, grid, target, log_z, target_masses, calibration = build_complex_problem(cfg)
    ref_density = ref.pdf(grid)
    ref_density /= np.trapezoid(np.trapezoid(ref_density, xs, axis=1), ys)
    density_panel(grid, {"p_ref": ref_density, "multi-peak reward": reward.value(grid), "p_star": target},
                  out["figures"] / "setting_c_ground_truth.png")
    ground_truth = {"log_Z": log_z, "calibrated_peaks": peaks, "target_masses": target_masses,
                    "calibration_iterations": len(calibration), "calibration_max_error": calibration[-1],
                    "challenges": ["multiple separated optima", "narrow needle mode", "boundary mode",
                                   "negative-reward hazards", "biased reward data", "proxy exploitation",
                                   "finite restart budget", "replay imbalance and mode collapse"]}
    exact = exact_exploration(cfg, ref, reward, peaks, grid, target, out)
    reward_models, reward_stress = reward_model_stress(cfg, ref, reward, peaks, grid, target, out)
    distillation = score_distillation(cfg, ref, reward, peaks, target_masses, reward_models["peak_aware_wide"], grid, target, out)
    report = {"setting": cfg["setting"], "ground_truth": ground_truth, "exact_exploration": exact,
              "reward_model_stress": reward_stress, "score_distillation": distillation}
    save_json(report, out["metrics"] / "setting_c_complex_multimodal.json")
    return report


if __name__ == "__main__":
    args = parser("Phase 4: complex multimodal stress test").parse_args()
    cfg, out = setup(args)
    report = run(cfg, out)
    print(json.dumps({"target_masses": report["ground_truth"]["target_masses"],
                      "exploration": report["exact_exploration"]["summary"],
                      "reward_stress": report["reward_model_stress"],
                      "distillation": report["score_distillation"]}, indent=2))
