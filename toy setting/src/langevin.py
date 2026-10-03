from __future__ import annotations

import numpy as np


def target_score(x, reference, reward, beta):
    return reference.score(x) + reward.gradient(x) / beta


def run_langevin(initial, score_fn, steps, step_size, rng, bounds=None, noise_scale=1.0, keep_trajectory=False):
    x = np.asarray(initial, dtype=float).copy()
    trajectory = [x.copy()] if keep_trajectory else None
    for _ in range(steps):
        x += 0.5 * step_size * score_fn(x) + np.sqrt(step_size) * noise_scale * rng.normal(size=x.shape)
        if bounds is not None:
            x = np.clip(x, np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1])
        if keep_trajectory:
            trajectory.append(x.copy())
    return x, None if trajectory is None else np.asarray(trajectory)


def run_gradient_ascent(initial, score_fn, steps, step_size, bounds=None):
    x = np.asarray(initial, dtype=float).copy()
    for _ in range(steps):
        x += 0.5 * step_size * score_fn(x)
        if bounds is not None:
            x = np.clip(x, np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1])
    return x

