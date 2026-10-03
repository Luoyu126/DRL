from __future__ import annotations

import numpy as np


class GaussianMixture:
    def __init__(self, means, sigmas, weights):
        self.means = np.asarray(means, dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)
        weights = np.asarray(weights, dtype=float)
        keep = weights > 0
        self.means, self.sigmas, weights = self.means[keep], self.sigmas[keep], weights[keep]
        self.weights = weights / weights.sum()

    def component_pdf(self, x):
        x = np.asarray(x)
        delta = x[..., None, :] - self.means
        d = x.shape[-1]
        norm = (2 * np.pi * self.sigmas**2) ** (-d / 2)
        return norm * np.exp(-0.5 * np.sum(delta**2, axis=-1) / self.sigmas**2)

    def pdf(self, x):
        return np.sum(self.component_pdf(x) * self.weights, axis=-1)

    def logpdf(self, x):
        return np.log(self.pdf(x) + 1e-300)

    def score(self, x):
        x = np.asarray(x)
        comp = self.component_pdf(x) * self.weights
        responsibilities = comp / (comp.sum(axis=-1, keepdims=True) + 1e-300)
        component_scores = -(x[..., None, :] - self.means) / self.sigmas[None, :, None] ** 2
        return np.sum(responsibilities[..., None] * component_scores, axis=-2)

    def sample(self, n, rng):
        component = rng.choice(len(self.weights), size=n, p=self.weights)
        return self.means[component] + rng.normal(size=(n, 2)) * self.sigmas[component, None]


def reference_from_config(cfg):
    return GaussianMixture([cfg["mu_a"], cfg["mu_b"]], [cfg["sigma_a"], cfg["sigma_b"]], cfg["ref_weights"])


def make_grid(bounds, size):
    xs = np.linspace(*bounds[0], size)
    ys = np.linspace(*bounds[1], size)
    xx, yy = np.meshgrid(xs, ys)
    return xs, ys, np.stack([xx, yy], axis=-1)


def integrate_grid(values, xs, ys):
    return float(np.trapezoid(np.trapezoid(values, xs, axis=1), ys, axis=0))


def basin_masks(grid, mu_a, mu_b):
    da = np.sum((grid - np.asarray(mu_a)) ** 2, axis=-1)
    db = np.sum((grid - np.asarray(mu_b)) ** 2, axis=-1)
    return da <= db, db < da


def normalized_target(ref, reward, grid, xs, ys, beta):
    log_unnorm = ref.logpdf(grid) + reward.value(grid) / beta
    shifted = np.exp(log_unnorm - np.max(log_unnorm))
    z_shifted = integrate_grid(shifted, xs, ys)
    return shifted / z_shifted, float(np.max(log_unnorm) + np.log(z_shifted))


def grid_masses(density, grid, xs, ys, mu_a, mu_b):
    mask_a, mask_b = basin_masks(grid, mu_a, mu_b)
    return {"A": integrate_grid(density * mask_a, xs, ys), "B": integrate_grid(density * mask_b, xs, ys)}


def classify_modes(samples, mu_a, mu_b, radius=1.0):
    samples = np.asarray(samples)
    da = np.linalg.norm(samples - np.asarray(mu_a), axis=-1)
    db = np.linalg.norm(samples - np.asarray(mu_b), axis=-1)
    labels = np.full(len(samples), "other", dtype=object)
    labels[(da <= radius) & (da <= db)] = "A"
    labels[(db <= radius) & (db < da)] = "B"
    return labels

