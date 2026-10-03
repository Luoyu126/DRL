from __future__ import annotations

import numpy as np


class BimodalReward:
    def __init__(self, mu_a, mu_b, tau_a, tau_b, r_a, r_b):
        self.mus = np.asarray([mu_a, mu_b], dtype=float)
        self.taus = np.asarray([tau_a, tau_b], dtype=float)
        self.heights = np.asarray([r_a, r_b], dtype=float)

    def bumps(self, x):
        delta = np.asarray(x)[..., None, :] - self.mus
        return self.heights * np.exp(-np.sum(delta**2, axis=-1) / (2 * self.taus**2))

    def value(self, x):
        return self.bumps(x).sum(axis=-1)

    def gradient(self, x):
        x = np.asarray(x)
        delta = x[..., None, :] - self.mus
        return np.sum(self.bumps(x)[..., None] * (-delta / self.taus[None, :, None] ** 2), axis=-2)


def reward_from_config(cfg, r_b=None):
    if r_b is None:
        r_b = cfg["r_b"]
        if r_b == "auto":
            raise ValueError("r_b must be calibrated before constructing reward")
    return BimodalReward(cfg["mu_a"], cfg["mu_b"], cfg["tau_a"], cfg["tau_b"], cfg["r_a"], r_b)


def calibrate_reward(cfg, ref, grid, xs, ys, target_b_mass=None):
    from .distributions import grid_masses, normalized_target

    goal = cfg["target_b_mass"] if target_b_mass is None else target_b_mass
    lo, hi = cfg["r_a"], 30.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        reward = reward_from_config(cfg, mid)
        density, _ = normalized_target(ref, reward, grid, xs, ys, cfg["beta"])
        mass = grid_masses(density, grid, xs, ys, cfg["mu_a"], cfg["mu_b"])["B"]
        if mass < goal:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

