from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path, smoke: bool = False) -> dict:
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["config_path"] = str(Path(path).resolve())
    if smoke:
        cfg = deepcopy(cfg)
        cfg["grid_size"] = 61
        if str(cfg["setting"]).startswith("C_"):
            cfg["langevin"].update(steps=80, chains=30)
            cfg["reward_train"].update(samples=1800, steps=350, batch_size=96)
            cfg["score_train"].update(samples=4500, steps=650, batch_size=192, diffusion_steps=40)
            cfg["experiment"].update(seeds=[2, 5], budgets=[10, 20, 30], lambdas=[0.0, 0.2, 0.4], sample_count=700)
        else:
            cfg["langevin"].update(steps=80, burn_in=20, chains=24)
            cfg["reward_train"].update(samples=2000, steps=400, batch_size=96)
            cfg["score_train"].update(samples=5000, steps=800, batch_size=192, diffusion_steps=40)
            cfg["experiment"].update(seeds=[3, 7], lambdas=[0.0, 0.2, 0.4], sample_count=750)
    return cfg


def output_dirs(root: str | Path | None = None) -> dict[str, Path]:
    base = Path(root) if root else ROOT / "outputs"
    result = {name: base / name for name in ("figures", "animations", "metrics", "models")}
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result
