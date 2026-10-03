from __future__ import annotations

import numpy as np

from .distributions import GaussianMixture, integrate_grid, make_grid, normalized_target


class GaussianBumpReward:
    """Sum of anisotropic positive and negative Gaussian reward bumps."""

    def __init__(self, components):
        self.names = [item["name"] for item in components]
        self.mus = np.asarray([item["mu"] for item in components], dtype=float)
        self.taus = np.asarray([item["tau"] for item in components], dtype=float)
        self.heights = np.asarray([item["height"] for item in components], dtype=float)

    def bumps(self, x):
        delta = np.asarray(x)[..., None, :] - self.mus
        exponent = -0.5 * np.sum((delta / self.taus) ** 2, axis=-1)
        return self.heights * np.exp(exponent)

    def value(self, x):
        return self.bumps(x).sum(axis=-1)

    def gradient(self, x):
        delta = np.asarray(x)[..., None, :] - self.mus
        return np.sum(self.bumps(x)[..., None] * (-delta / self.taus**2), axis=-2)


def complex_reference(cfg):
    spec = cfg["reference"]
    return GaussianMixture(spec["means"], spec["sigmas"], spec["weights"])


def classify_named_modes(samples, peaks):
    samples = np.asarray(samples)
    centers = np.asarray([p["mu"] for p in peaks])
    radii = np.asarray([p["radius"] for p in peaks])
    distances = np.linalg.norm(samples[:, None, :] - centers[None, :, :], axis=-1)
    scaled = distances / radii[None, :]
    nearest = np.argmin(scaled, axis=1)
    labels = np.full(len(samples), "background", dtype=object)
    inside = scaled[np.arange(len(samples)), nearest] <= 1
    names = np.asarray([p["name"] for p in peaks], dtype=object)
    labels[inside] = names[nearest[inside]]
    return labels


def named_grid_masses(density, grid, xs, ys, peaks):
    labels = classify_named_modes(grid.reshape(-1, 2), peaks).reshape(grid.shape[:2])
    result = {p["name"]: integrate_grid(density * (labels == p["name"]), xs, ys) for p in peaks}
    result["background"] = integrate_grid(density * (labels == "background"), xs, ys)
    return result


def calibrate_multimodal_reward(cfg, ref, grid, xs, ys):
    peaks = [dict(p) for p in cfg["reward_peaks"]]
    hazards = [dict(h) for h in cfg["reward_hazards"]]
    # Coordinate updates use exact numerical masses. The Gaussian bump is not
    # constant within a basin, so a damped log-ratio update is more stable than
    # the separated-mode closed-form approximation.
    heights = np.full(len(peaks), 2.0)
    targets = np.asarray([p["target_mass"] for p in peaks])
    history = []
    for iteration in range(180):
        for p, height in zip(peaks, heights):
            p["height"] = float(height)
        reward = GaussianBumpReward(peaks + hazards)
        density, _ = normalized_target(ref, reward, grid, xs, ys, cfg["beta"])
        masses_dict = named_grid_masses(density, grid, xs, ys, peaks)
        masses = np.asarray([masses_dict[p["name"]] for p in peaks])
        error = np.log((targets + 1e-8) / (masses + 1e-8))
        heights = np.clip(heights + 0.32 * cfg["beta"] * error, 0.0, 45.0)
        history.append(float(np.max(np.abs(masses - targets))))
        if history[-1] < 0.004:
            break
    for p, height in zip(peaks, heights):
        p["height"] = float(height)
    reward = GaussianBumpReward(peaks + hazards)
    return reward, peaks, history


def build_complex_problem(cfg):
    ref = complex_reference(cfg)
    xs, ys, grid = make_grid(cfg["bounds"], cfg["grid_size"])
    reward, peaks, calibration_history = calibrate_multimodal_reward(cfg, ref, grid, xs, ys)
    target, log_z = normalized_target(ref, reward, grid, xs, ys, cfg["beta"])
    masses = named_grid_masses(target, grid, xs, ys, peaks)
    return ref, reward, peaks, xs, ys, grid, target, log_z, masses, calibration_history


def named_fractions(samples, peaks):
    labels = classify_named_modes(samples, peaks)
    names = [p["name"] for p in peaks] + ["background"]
    return {name: float(np.mean(labels == name)) for name in names}

