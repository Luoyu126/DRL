from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import save_json


def compact(mis, tsp):
    return {
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


if __name__ == "__main__":
    metrics = ROOT / "outputs" / "metrics"
    mis = json.loads((metrics / "mis_cpu_validation.json").read_text())
    tsp = json.loads((metrics / "tsp_cpu_validation.json").read_text())
    result = compact(mis, tsp); save_json(result, metrics / "summary.json")
    print(json.dumps(result, indent=2))

