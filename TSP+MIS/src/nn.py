from __future__ import annotations

import pickle

import numpy as np


def silu(x):
    return x / (1 + np.exp(-np.clip(x, -40, 40)))


def silu_grad(x):
    s = 1 / (1 + np.exp(-np.clip(x, -40, 40)))
    return s * (1 + x * (1 - s))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


class MLP:
    def __init__(self, sizes, rng):
        self.weights, self.biases = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            self.weights.append(rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)))
            self.biases.append(np.zeros(n_out))
        self.m = [np.zeros_like(p) for p in self.parameters()]
        self.v = [np.zeros_like(p) for p in self.parameters()]
        self.iteration = 0

    def parameters(self):
        return [p for pair in zip(self.weights, self.biases) for p in pair]

    def forward(self, x, cache=False):
        h, acts, zs = np.asarray(x, dtype=float), [np.asarray(x, dtype=float)], []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            zs.append(z)
            h = z if i == len(self.weights) - 1 else silu(z)
            acts.append(h)
        return (h, (acts, zs)) if cache else h

    def backward(self, cache, delta):
        acts, zs = cache
        pairs = []
        for i in reversed(range(len(self.weights))):
            pairs.append((acts[i].T @ delta, delta.sum(0)))
            if i:
                delta = (delta @ self.weights[i].T) * silu_grad(zs[i - 1])
        pairs.reverse()
        return [g for pair in pairs for g in pair]

    def adam(self, grads, lr, clip=10):
        self.iteration += 1
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if norm > clip:
            grads = [g * clip / (norm + 1e-12) for g in grads]
        for i, (parameter, grad) in enumerate(zip(self.parameters(), grads)):
            self.m[i] = .9 * self.m[i] + .1 * grad
            self.v[i] = .999 * self.v[i] + .001 * grad * grad
            mh = self.m[i] / (1 - .9**self.iteration)
            vh = self.v[i] / (1 - .999**self.iteration)
            parameter -= lr * mh / (np.sqrt(vh) + 1e-8)

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)


class ConditionalBitDenoiser:
    """Fixed-size graph-conditioned Bernoulli denoiser for CPU experiments."""

    def __init__(self, context_dim, solution_dim, rng, hidden=128):
        self.context_dim, self.solution_dim = context_dim, solution_dim
        self.model = MLP([context_dim + solution_dim + 4, hidden, hidden, solution_dim], rng)
        self.context_mean = np.zeros(context_dim)
        self.context_std = np.ones(context_dim)

    def features(self, context, noisy, noise_level):
        context = (context - self.context_mean) / self.context_std
        t = np.asarray(noise_level).reshape(-1, 1)
        emb = np.concatenate([t, t * t, np.sin(np.pi * t), np.cos(np.pi * t)], axis=1)
        return np.concatenate([context, noisy * 2 - 1, emb], axis=1)

    def fit(self, contexts, clean, steps, batch_size, lr, rng, reset_stats=False):
        if reset_stats:
            self.context_mean = contexts.mean(0)
            self.context_std = contexts.std(0) + 1e-6
        history = []
        for step in range(steps):
            idx = rng.integers(0, len(clean), min(batch_size, len(clean)))
            x0 = clean[idx]
            level = rng.uniform(.03, .5, len(idx))
            flips = rng.random(x0.shape) < level[:, None]
            xt = np.logical_xor(x0 > .5, flips).astype(float)
            logits, cache = self.model.forward(self.features(contexts[idx], xt, level), cache=True)
            probs = sigmoid(logits)
            loss = -np.mean(x0 * np.log(probs + 1e-8) + (1 - x0) * np.log(1 - probs + 1e-8))
            self.model.adam(self.model.backward(cache, (probs - x0) / x0.size), lr)
            if step % max(1, steps // 20) == 0 or step == steps - 1:
                history.append((step, float(loss)))
        return history

    def probabilities(self, contexts, noisy=None, noise_level=.05):
        contexts = np.asarray(contexts)
        if noisy is None:
            noisy = np.full((len(contexts), self.solution_dim), .5)
        level = np.full(len(contexts), noise_level) if np.ndim(noise_level) == 0 else noise_level
        return sigmoid(self.model.forward(self.features(contexts, noisy, level)))

    def sample_probabilities(self, context, count, rng, steps=12):
        contexts = np.repeat(np.asarray(context)[None], count, axis=0)
        x = (rng.random((count, self.solution_dim)) < .5).astype(float)
        for level in np.linspace(.5, .02, steps):
            probs = self.probabilities(contexts, x, level)
            temperature = level / .5
            sampling_probs = (1 - temperature) * probs + temperature * .5
            x = (rng.random(x.shape) < sampling_probs).astype(float)
        return self.probabilities(contexts, x, .01)

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

