from __future__ import annotations

import json

from _common import parser, setup
from phase0_ground_truth import run as phase0
from phase1_exact_components import run as phase1
from phase2_reward_mlp import run as phase2
from phase3_score_distillation import run as phase3
from src.metrics import save_json


if __name__ == "__main__":
    args = parser("Run all reward-guided exploration phases").parse_args()
    cfg, out = setup(args)
    reports = {"phase0": phase0(cfg, out), "phase1": phase1(cfg, out),
               "phase2": phase2(cfg, out), "phase3": phase3(cfg, out)}
    summary = {
        "setting": cfg["setting"],
        "checks": {
            "target_has_B": reports["phase0"]["checks"]["target_has_B"],
            "global_beats_local": reports["phase1"]["summary"]["random_restart"]["B_fraction_mean"] > reports["phase1"]["summary"]["local_guidance"]["B_fraction_mean"],
            "ucb_beats_random_discovery": reports["phase1"]["summary"]["ucb_restart"]["discovery_rate"] >= reports["phase1"]["summary"]["random_restart"]["discovery_rate"],
            "filter_observed_spurious": reports["phase2"]["comparison"]["filter_counts"].get("rejected_spurious_reward_model_peak", 0) > 0,
            "distillation_increases_B": max(x["B_fraction"] for x in reports["phase3"]["lambda_sweep"]) > reports["phase3"]["pretrain_mode_fractions"]["B"],
        },
    }
    save_json(summary, out["metrics"] / f"setting_{cfg['setting'].lower()}_summary.json")
    print(json.dumps(summary, indent=2))

