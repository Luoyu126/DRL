import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.complex_setting import GaussianBumpReward, build_complex_problem, classify_named_modes
from src.config import load_config


class ComplexSettingTests(unittest.TestCase):
    def test_anisotropic_reward_gradient(self):
        reward = GaussianBumpReward([{"name": "x", "mu": [0, 0], "tau": [.3, .8], "height": 2.0}])
        x = np.array([[.1, -.2]])
        eps = 1e-5
        numeric = np.column_stack([(reward.value(x + np.eye(2)[i] * eps) - reward.value(x - np.eye(2)[i] * eps)) / (2 * eps) for i in range(2)])
        self.assertTrue(np.allclose(reward.gradient(x), numeric, atol=1e-6))

    def test_named_mode_classification(self):
        peaks = [{"name": "a", "mu": [0, 0], "radius": .5}, {"name": "b", "mu": [2, 0], "radius": .5}]
        labels = classify_named_modes(np.array([[.1, 0], [2.1, 0], [1, 1]]), peaks)
        self.assertEqual(labels.tolist(), ["a", "b", "background"])

    def test_multimodal_mass_calibration(self):
        cfg = load_config(ROOT / "configs" / "setting_c_multimodal.yaml", smoke=True)
        *_, masses, history = build_complex_problem(cfg)
        for peak in cfg["reward_peaks"]:
            self.assertLess(abs(masses[peak["name"]] - peak["target_mass"]), .012)
        self.assertLess(history[-1], .012)
