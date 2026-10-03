from __future__ import annotations

import numpy as np


class RestartGrid:
    def __init__(self, bounds, rows, cols):
        self.bounds = np.asarray(bounds, dtype=float)
        self.rows, self.cols = rows, cols
        x_edges = np.linspace(*bounds[0], cols + 1)
        y_edges = np.linspace(*bounds[1], rows + 1)
        self.cells = []
        for iy in range(rows):
            for ix in range(cols):
                self.cells.append(((x_edges[ix], x_edges[ix + 1]), (y_edges[iy], y_edges[iy + 1])))
        self.counts = np.zeros(len(self.cells), dtype=int)
        self.reward_sum = np.zeros(len(self.cells))
        self.success = np.zeros(len(self.cells), dtype=int)

    def sample_cell(self, index, rng):
        cell = self.cells[index]
        return np.array([rng.uniform(*cell[0]), rng.uniform(*cell[1])])

    def centers(self):
        return np.asarray([[(c[0][0] + c[0][1]) / 2, (c[1][0] + c[1][1]) / 2] for c in self.cells])

    def update(self, index, reward, success):
        self.counts[index] += 1
        self.reward_sum[index] += reward
        self.success[index] += int(success)

    def choose_random(self, rng):
        return int(rng.integers(len(self.cells)))

    def choose_ucb(self, rng, c=1.4, novelty_gamma=0.25, replay=None, prior_values=None):
        if prior_values is not None:
            prior_values = np.asarray(prior_values, dtype=float)
            observed = np.divide(self.reward_sum, self.counts, out=prior_values.copy(), where=self.counts > 0)
            scale = np.std(prior_values) + 1e-12
            value = (observed - np.mean(prior_values)) / scale
            explore = c * np.sqrt(np.log(self.counts.sum() + 2) / (self.counts + 1))
            novelty = np.zeros(len(self.cells))
            if replay is not None and len(replay):
                d = self.centers()[:, None, :] - np.asarray(replay)[None, :, :]
                novelty = np.sqrt(np.sum(d * d, axis=-1)).min(axis=1)
                novelty /= novelty.max() + 1e-12
            # Tiny jitter avoids deterministic cell-index tie breaking.
            return int(np.argmax(value + explore + novelty_gamma * novelty + rng.normal(0, 1e-9, len(value))))
        unseen = np.flatnonzero(self.counts == 0)
        if len(unseen):
            # Random tie breaking makes the initial exploration unbiased.
            return int(rng.choice(unseen))
        mean = self.reward_sum / self.counts
        explore = c * np.sqrt(np.log(self.counts.sum() + 1) / (self.counts + 1))
        novelty = np.zeros(len(self.cells))
        if replay is not None and len(replay):
            d = self.centers()[:, None, :] - np.asarray(replay)[None, :, :]
            novelty = np.sqrt(np.sum(d * d, axis=-1)).min(axis=1)
            novelty /= novelty.max() + 1e-12
        return int(np.argmax(mean + explore + novelty_gamma * novelty))


def oracle_initial(mu_b, n, rng, scale=0.2):
    return np.asarray(mu_b) + rng.normal(scale=scale, size=(n, 2))
