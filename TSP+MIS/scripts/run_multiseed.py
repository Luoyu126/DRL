from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from run_mis import run as run_mis
from run_tsp import run as run_tsp
from src.utils import load_config, output_dirs, save_json


SEEDS = (42, 123, 777)


def extract(mis, tsp):
    return {
        "mis_base": mis["before_distillation"]["reference_model"]["optimality_gap_mean"],
        "mis_local": mis["before_distillation"]["local"]["optimality_gap_mean"],
        "mis_random": mis["before_distillation"]["random_restart"]["optimality_gap_mean"],
        "mis_ucb": mis["before_distillation"]["ucb_restart"]["optimality_gap_mean"],
        "mis_distilled": mis["after_distillation"]["reference_model"]["optimality_gap_mean"],
        "tsp_base": tsp["before_distillation"]["reference_model"]["optimality_gap_mean"],
        "tsp_local": tsp["before_distillation"]["local_2opt"]["optimality_gap_mean"],
        "tsp_random": tsp["before_distillation"]["random_restart"]["optimality_gap_mean"],
        "tsp_ucb": tsp["before_distillation"]["ucb_restart"]["optimality_gap_mean"],
        "tsp_distilled": tsp["after_distillation"]["reference_model"]["optimality_gap_mean"],
    }


if __name__ == "__main__":
    cfg = load_config(); per_seed = {}
    for seed in SEEDS:
        if seed == 42:
            metrics = ROOT / "outputs" / "metrics"
            mis = json.loads((metrics / "mis_cpu_validation.json").read_text())
            tsp = json.loads((metrics / "tsp_cpu_validation.json").read_text())
        else:
            cfg_seed = json.loads(json.dumps(cfg)); cfg_seed["seed"] = seed
            out = output_dirs(ROOT / "outputs" / "multiseed" / f"seed_{seed}")
            mis, tsp = run_mis(cfg_seed, out), run_tsp(cfg_seed, out)
        per_seed[str(seed)] = extract(mis, tsp)
        print(json.dumps({"completed_seed": seed, **per_seed[str(seed)]}), flush=True)
    keys = next(iter(per_seed.values())).keys()
    aggregate = {key: {"mean": float(np.mean([x[key] for x in per_seed.values()])),
                       "std": float(np.std([x[key] for x in per_seed.values()]))} for key in keys}
    report = {"seeds": list(SEEDS), "per_seed": per_seed, "aggregate": aggregate}
    save_json(report, ROOT / "outputs" / "metrics" / "multiseed_summary.json")
    print(json.dumps(report, indent=2))

