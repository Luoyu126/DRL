from __future__ import annotations

import pickle

import numpy as np

from .mlp import MLP


def time_embedding(t, dim=8):
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    frequencies = 2.0 ** np.arange(dim // 2)[None, :] * np.pi
    return np.concatenate([np.sin(t * frequencies), np.cos(t * frequencies)], axis=1)


class DiffusionSchedule:
    def __init__(self, steps=100):
        self.steps = steps
        # Nichol-Dhariwal cosine schedule: alpha_bar(T) is close to zero even
        # for smoke-test step counts, making the N(0,I) reverse initialization valid.
        t = np.linspace(0, 1, steps + 1)
        alpha_bar = np.cos(((t + 0.008) / 1.008) * np.pi / 2) ** 2
        alpha_bar /= alpha_bar[0]
        self.betas = np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 1e-5, 0.999)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

    def forward(self, x0, indices, noise):
        ab = self.alpha_bars[indices, None]
        return np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise


class ScoreMLP:
    def __init__(self, rng, hidden=64, time_dim=8, coordinate_scale=3.0):
        self.time_dim = time_dim
        self.coordinate_scale = coordinate_scale
        self.model = MLP([2 + time_dim, hidden, hidden, 2], rng, output_scale=0.5)

    def features(self, x, t):
        x = np.asarray(x)
        if np.ndim(t) == 0:
            t = np.full(len(x), t)
        return np.concatenate([x / self.coordinate_scale, time_embedding(t, self.time_dim)], axis=1)

    def predict_noise(self, x, t):
        return self.model.forward(self.features(x, t))

    def score(self, x, t, schedule):
        index = np.clip((np.asarray(t) * (schedule.steps - 1)).astype(int), 0, schedule.steps - 1)
        sigma = np.sqrt(1 - schedule.alpha_bars[index])
        if np.ndim(sigma) == 0:
            sigma = np.full(len(x), sigma)
        return -self.predict_noise(x, t) / np.maximum(sigma[:, None], 1e-3)

    def fit(self, clean_samples, schedule, steps, batch_size, lr, rng):
        history = []
        for step in range(steps):
            idx = rng.integers(0, len(clean_samples), min(batch_size, len(clean_samples)))
            x0 = clean_samples[idx]
            ti = rng.integers(0, schedule.steps, len(idx))
            noise = rng.normal(size=x0.shape)
            xt = schedule.forward(x0, ti, noise)
            t = (ti + 1) / schedule.steps
            pred, cache = self.model.forward(self.features(xt, t), cache=True)
            error = pred - noise
            loss = float(np.mean(error**2))
            self.model.adam(self.model.backward(cache, 2 * error / error.size), lr)
            if step % max(1, steps // 20) == 0 or step == steps - 1:
                history.append((step, loss))
        return history

    def sample(self, n, schedule, rng, clip=None):
        x = rng.normal(size=(n, 2))
        for i in reversed(range(schedule.steps)):
            t = np.full(n, (i + 1) / schedule.steps)
            eps = self.predict_noise(x, t)
            alpha, beta, ab = schedule.alphas[i], schedule.betas[i], schedule.alpha_bars[i]
            mean = (x - beta / np.sqrt(1 - ab) * eps) / np.sqrt(alpha)
            x = mean if i == 0 else mean + np.sqrt(beta) * rng.normal(size=x.shape)
            if clip is not None:
                x = np.clip(x, np.asarray(clip)[:, 0], np.asarray(clip)[:, 1])
        return x

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
