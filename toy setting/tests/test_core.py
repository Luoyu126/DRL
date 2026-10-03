import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.distributions import make_grid, normalized_target, reference_from_config
from src.mlp import StandardizedRewardMLP
from src.rewards import BimodalReward, calibrate_reward, reward_from_config


class CoreTests(unittest.TestCase):
    def test_analytic_reward_gradient(self):
        reward = BimodalReward([-1.5, 0], [1.5, 0], .45, .45, 1, 5)
        x = np.array([[.2, -.3]])
        eps = 1e-5
        numeric = np.column_stack([(reward.value(x + np.eye(2)[i] * eps) - reward.value(x - np.eye(2)[i] * eps)) / (2 * eps) for i in range(2)])
        self.assertTrue(np.allclose(reward.gradient(x), numeric, atol=1e-6))

    def test_target_calibration(self):
        cfg = load_config(ROOT / "configs" / "setting_a.yaml", smoke=True)
        ref = reference_from_config(cfg)
        xs, ys, grid = make_grid(cfg["bounds"], cfg["grid_size"])
        r_b = calibrate_reward(cfg, ref, grid, xs, ys)
        reward = reward_from_config(cfg, r_b)
        target, _ = normalized_target(ref, reward, grid, xs, ys, cfg["beta"])
        self.assertTrue(np.isclose(np.trapezoid(np.trapezoid(target, xs, axis=1), ys), 1.0))

    def test_reward_mlp_gradient_shape(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(100, 2)); y = np.sum(x * x, axis=1)
        model = StandardizedRewardMLP(rng, hidden=8)
        model.fit(x, y, steps=3, batch_size=16, lr=.001, rng=rng)
        self.assertEqual(model.gradient(x[:4]).shape, (4, 2))
