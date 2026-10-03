from __future__ import annotations

import json

from _common import make_parser, setup
from run_mis import run as run_mis
from run_tsp import run as run_tsp
from src.utils import save_json


if __name__ == "__main__":
    args = make_parser("Run MIS and TSP CPU validations").parse_args(); cfg, out = setup(args)
    mis, tsp = run_mis(cfg, out), run_tsp(cfg, out)
    summary = {
        "mis": {"base_gap": mis["before_distillation"]["reference_model"]["optimality_gap_mean"],
                "local_gap": mis["before_distillation"]["local"]["optimality_gap_mean"],
                "random_gap": mis["before_distillation"]["random_restart"]["optimality_gap_mean"],
                "ucb_gap": mis["before_distillation"]["ucb_restart"]["optimality_gap_mean"],
                "distilled_base_gap": mis["after_distillation"]["reference_model"]["optimality_gap_mean"]},
        "tsp": {"base_gap": tsp["before_distillation"]["reference_model"]["optimality_gap_mean"],
                "local_gap": tsp["before_distillation"]["local_2opt"]["optimality_gap_mean"],
                "random_gap": tsp["before_distillation"]["random_restart"]["optimality_gap_mean"],
                "ucb_gap": tsp["before_distillation"]["ucb_restart"]["optimality_gap_mean"],
                "distilled_base_gap": tsp["after_distillation"]["reference_model"]["optimality_gap_mean"]},
    }
    save_json(summary, out["metrics"] / "summary.json")
    print(json.dumps(summary, indent=2))

