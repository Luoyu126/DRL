from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -40, 40)))


def silu_grad(x):
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))
    return sig * (1.0 + x * (1.0 - sig))


class MLP:
    """Small NumPy MLP with Adam and exact input gradients."""

    def __init__(self, sizes, rng, output_scale=1.0):
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.weights.append(rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out)) * output_scale)
            self.biases.append(np.zeros(fan_out))
        self.m = [np.zeros_like(p) for p in self.parameters()]
        self.v = [np.zeros_like(p) for p in self.parameters()]
        self.step = 0

    def parameters(self):
        return [item for pair in zip(self.weights, self.biases) for item in pair]

    def forward(self, x, cache=False):
        h = np.asarray(x, dtype=float)
        activations, preacts = [h], []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w + b
            preacts.append(z)
            h = z if i == len(self.weights) - 1 else silu(z)
            activations.append(h)
        return (h, (activations, preacts)) if cache else h

    def backward(self, cache, grad_output):
        activations, preacts = cache
        delta = grad_output
        grads = []
        for i in reversed(range(len(self.weights))):
            dw = activations[i].T @ delta
            db = delta.sum(axis=0)
            grads.append((dw, db))
            if i:
                delta = (delta @ self.weights[i].T) * silu_grad(preacts[i - 1])
        grads.reverse()
        return [item for pair in grads for item in pair]

    def input_gradient(self, x, output_index=0):
        out, (acts, preacts) = self.forward(x, cache=True)
        delta = np.zeros_like(out)
        delta[:, output_index] = 1.0
        for i in reversed(range(len(self.weights))):
            delta = delta @ self.weights[i].T
            if i:
                delta *= silu_grad(preacts[i - 1])
        return delta

    def adam(self, grads, lr, clip=10.0):
        self.step += 1
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if norm > clip:
            grads = [g * clip / (norm + 1e-12) for g in grads]
        for i, (p, g) in enumerate(zip(self.parameters(), grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9**self.step)
            vh = self.v[i] / (1 - 0.999**self.step)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)

    def train_mse(self, x, y, steps, batch_size, lr, rng):
        history = []
        for step in range(steps):
            idx = rng.integers(0, len(x), min(batch_size, len(x)))
            pred, cache = self.forward(x[idx], cache=True)
            error = pred - y[idx]
            loss = float(np.mean(error**2))
            grad = 2 * error / error.size
            self.adam(self.backward(cache, grad), lr)
            if step % max(1, steps // 20) == 0 or step == steps - 1:
                history.append((step, loss))
        return history

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)


class StandardizedRewardMLP:
    def __init__(self, rng, hidden=64):
        self.model = MLP([2, hidden, hidden, 1], rng)
        self.x_mean = np.zeros(2)
        self.x_std = np.ones(2)
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, x, y, **kwargs):
        self.x_mean, self.x_std = x.mean(0), x.std(0) + 1e-6
        self.y_mean, self.y_std = float(y.mean()), float(y.std() + 1e-6)
        xn = (x - self.x_mean) / self.x_std
        yn = ((y - self.y_mean) / self.y_std)[:, None]
        return self.model.train_mse(xn, yn, **kwargs)

    def value(self, x):
        xn = (np.asarray(x) - self.x_mean) / self.x_std
        return self.model.forward(xn)[..., 0] * self.y_std + self.y_mean

    def gradient(self, x):
        x = np.asarray(x)
        shape = x.shape
        flat = x.reshape(-1, 2)
        xn = (flat - self.x_mean) / self.x_std
        grad = self.model.input_gradient(xn) * self.y_std / self.x_std
        return grad.reshape(shape)

    def save(self, path):
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

