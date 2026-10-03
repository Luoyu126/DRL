from __future__ import annotations

import numpy as np

from .distributions import classify_modes


def filter_candidates(samples, true_reward, cfg, predicted_reward=None, reward_threshold=None):
    samples = np.asarray(samples)
    labels = classify_modes(samples, cfg["mu_a"], cfg["mu_b"], cfg["basin_radius"])
    values = true_reward.value(samples)
    if reward_threshold is None:
        reward_threshold = 0.45 * cfg["r_a"]
    accepted = values >= reward_threshold
    categories = np.full(len(samples), "rejected_low_reward", dtype=object)
    categories[accepted & (labels == "A")] = "accepted_A"
    categories[accepted & (labels == "B")] = "accepted_B"
    categories[accepted & (labels == "other")] = "rejected_unstable"
    if predicted_reward is not None:
        pred = predicted_reward.value(samples)
        exploit = (pred > np.quantile(pred, 0.75)) & (values < reward_threshold)
        categories[exploit] = "rejected_spurious_reward_model_peak"
    return categories

