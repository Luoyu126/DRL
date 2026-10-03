from __future__ import annotations

import json

import numpy as np

from _common import parser, setup
from src.distributions import grid_masses
from src.experiment import build_problem
from src.langevin import target_score
from src.metrics import save_json
from src.plotting import density_panel, field_plot


def run(cfg, out):
    ref, reward, xs, ys, grid, target, log_z, r_b = build_problem(cfg)
    ref_density = ref.pdf(grid)
    ref_density /= np.trapezoid(np.trapezoid(ref_density, xs, axis=1), ys, axis=0)
    ref_mass = grid_masses(ref_density, grid, xs, ys, cfg["mu_a"], cfg["mu_b"])
    target_mass = grid_masses(target, grid, xs, ys, cfg["mu_a"], cfg["mu_b"])
    grad_at_a = float(np.linalg.norm(reward.gradient(np.asarray(cfg["mu_a"])[None])[0]))
    peak_a = float(reward.value(np.asarray(cfg["mu_a"])[None])[0])
    peak_b = float(reward.value(np.asarray(cfg["mu_b"])[None])[0])
    report = {"setting": cfg["setting"], "calibrated_r_b": r_b, "beta": cfg["beta"], "log_Z": log_z,
              "reference_mass": ref_mass, "target_mass": target_mass, "reward_at_A": peak_a,
              "reward_at_B": peak_b, "reward_gradient_norm_at_A": grad_at_a,
              "checks": {"B_reward_higher": peak_b > peak_a,
                         "target_has_B": target_mass["B"] > 0.05,
                         "small_cross_bump_gradient_at_A": grad_at_a < 0.05}}
    prefix = f"setting_{cfg['setting'].lower()}"
    density_panel(grid, {"p_ref": ref_density, "R_true": reward.value(grid), "p_star": target},
                  out["figures"] / f"{prefix}_phase0_densities.png")
    field_plot(grid, target, target_score(grid, ref, reward, cfg["beta"]),
               out["figures"] / f"{prefix}_phase0_target_score.png", "Target density and exact score")
    save_json(report, out["metrics"] / f"{prefix}_phase0.json")
    return report


if __name__ == "__main__":
    args = parser("Phase 0: grid ground truth").parse_args()
    cfg, out = setup(args)
    print(json.dumps(run(cfg, out), indent=2))

