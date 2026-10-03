from __future__ import annotations

import time

import numpy as np


def generate_instances(count, n, rng):
    # Mix uniform and mildly clustered instances to create varied local-search landscapes.
    points = rng.random((count, n, 2))
    for i in range(0, count, 3):
        centers = rng.random((3, 2))
        assignments = rng.integers(0, 3, n)
        points[i] = np.clip(centers[assignments] + rng.normal(scale=.08, size=(n, 2)), 0, 1)
    return points


def distance_matrix(points):
    delta = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def instance_contexts(instances):
    """Permutation-indexed weighted graph representation aligned with edge outputs."""
    n = instances.shape[1]
    rows, cols = np.triu_indices(n, 1)
    return np.asarray([distance_matrix(points)[rows, cols] for points in instances])


def tour_length(points, tour):
    tour = np.asarray(tour, dtype=int)
    nxt = np.roll(tour, -1)
    return float(np.linalg.norm(points[tour] - points[nxt], axis=1).sum())


def nearest_neighbor(points, start=0, noise=0.0, rng=None):
    distances = distance_matrix(points)
    if noise and rng is not None:
        distances = distances + rng.normal(scale=noise, size=distances.shape)
    unvisited = set(range(len(points))); unvisited.remove(start)
    tour, current = [start], start
    while unvisited:
        current = min(unvisited, key=lambda j: distances[current, j])
        unvisited.remove(current); tour.append(current)
    return np.asarray(tour, dtype=int)


def two_opt(points, tour, max_rounds=50):
    tour = np.asarray(tour, dtype=int).copy()
    n = len(tour)
    for _ in range(max_rounds):
        best_delta, best_move = -1e-12, None
        for i in range(n - 1):
            a, b = tour[i], tour[(i + 1) % n]
            for j in range(i + 2, n if i else n - 1):
                c, d = tour[j], tour[(j + 1) % n]
                delta = (np.linalg.norm(points[a] - points[c]) + np.linalg.norm(points[b] - points[d])
                         - np.linalg.norm(points[a] - points[b]) - np.linalg.norm(points[c] - points[d]))
                if delta < best_delta:
                    best_delta, best_move = delta, (i + 1, j)
        if best_move is None:
            break
        lo, hi = best_move; tour[lo:hi + 1] = tour[lo:hi + 1][::-1]
    return canonical_tour(tour)


def canonical_tour(tour):
    tour = list(map(int, tour)); zero = tour.index(0)
    tour = tour[zero:] + tour[:zero]
    reverse = [tour[0]] + list(reversed(tour[1:]))
    return np.asarray(min(tour, reverse), dtype=int)


def held_karp(points):
    """Exact TSP dynamic program with city 0 fixed as the start."""
    n = len(points); size = 1 << n; distances = distance_matrix(points)
    dp = np.full((size, n), np.inf); parent = np.full((size, n), -1, dtype=np.int16)
    dp[1, 0] = 0
    for mask in range(size):
        if not (mask & 1):
            continue
        for last in range(n):
            base = dp[mask, last]
            if not np.isfinite(base):
                continue
            remaining = ((1 << n) - 1) ^ mask
            while remaining:
                bit = remaining & -remaining; nxt = bit.bit_length() - 1
                new_mask = mask | bit; value = base + distances[last, nxt]
                if value < dp[new_mask, nxt]:
                    dp[new_mask, nxt] = value; parent[new_mask, nxt] = last
                remaining ^= bit
    full = size - 1
    last = min(range(1, n), key=lambda j: dp[full, j] + distances[j, 0])
    path, mask = [last], full
    while last != 0:
        previous = int(parent[mask, last]); mask ^= 1 << last; last = previous; path.append(last)
    return canonical_tour(list(reversed(path)))


def edge_indices(n):
    return list(zip(*np.triu_indices(n, 1)))


def tour_to_edges(tour, n):
    lookup = {tuple(edge): i for i, edge in enumerate(edge_indices(n))}
    result = np.zeros(len(lookup), dtype=float)
    for a, b in zip(tour, np.roll(tour, -1)):
        result[lookup[tuple(sorted((int(a), int(b))))]] = 1
    return result


def edges_to_matrix(values, n):
    matrix = np.zeros((n, n), dtype=float)
    rows, cols = np.triu_indices(n, 1)
    matrix[rows, cols] = values; matrix[cols, rows] = values
    return matrix


def decode_edge_scores(scores, n, stochastic_noise=0.0, rng=None):
    matrix = edges_to_matrix(scores, n)
    if stochastic_noise and rng is not None:
        matrix = matrix + rng.normal(scale=stochastic_noise, size=matrix.shape)
    tour, unvisited, current = [0], set(range(1, n)), 0
    while unvisited:
        current = max(unvisited, key=lambda j: matrix[current, j])
        unvisited.remove(current); tour.append(current)
    return canonical_tour(tour)


def heuristic_labels(instances):
    tours = [nearest_neighbor(points, 0) for points in instances]
    return np.asarray([tour_to_edges(tour, len(points)) for points, tour in zip(instances, tours)]), tours


def _perturb_tour(tour, moves, rng):
    result = np.asarray(tour).copy(); n = len(result)
    for _ in range(moves):
        i, j = sorted(rng.choice(np.arange(1, n), 2, replace=False))
        result[i:j + 1] = result[i:j + 1][::-1]
    return canonical_tour(result)


def _arm_candidate(points, model_tour, arm, rng):
    n = len(points)
    if arm == 0:
        tour = _perturb_tour(model_tour, 1, rng)
    elif arm == 1:
        tour = nearest_neighbor(points, int(rng.integers(n)))
    elif arm == 2:
        tour = canonical_tour(rng.permutation(n))
    elif arm == 3:
        tour = nearest_neighbor(points, int(rng.integers(n)), noise=.08, rng=rng)
    elif arm == 4:
        center = points.mean(0)
        tour = canonical_tour(np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])))
    else:
        raise ValueError(arm)
    return two_opt(points, tour)


def restart_search(points, model_tour, budget, rng, strategy):
    arms = 5; counts = np.zeros(arms, dtype=int); totals = np.zeros(arms)
    best, best_length, trace = None, np.inf, []
    for step in range(budget):
        if strategy == "random":
            arm = int(rng.integers(arms))
        elif strategy == "ucb":
            unseen = np.flatnonzero(counts == 0)
            if len(unseen):
                arm = int(unseen[0])
            else:
                mean = totals / counts
                arm = int(np.argmax(mean + .15 * np.sqrt(np.log(step + 1) / counts)))
        else:
            raise ValueError(strategy)
        candidate = _arm_candidate(points, model_tour, arm, rng)
        length = tour_length(points, candidate); reward = -length
        counts[arm] += 1; totals[arm] += reward
        if length < best_length:
            best, best_length = candidate, length
        trace.append(best_length)
    return best, np.asarray(trace), counts


def decode_model_samples(model, context, points, count, rng):
    probabilities = model.sample_probabilities(context, count, rng)
    tours = [decode_edge_scores(p, len(points), stochastic_noise=.04, rng=rng) for p in probabilities]
    return tours, probabilities


def evaluate(model, instances, contexts, optimal_tours, cfg, rng):
    methods = {name: [] for name in ("reference_model", "local_2opt", "random_restart", "ucb_restart", "optimal")}
    random_traces, ucb_traces, first_random, first_ucb = [], [], [], []
    started = time.perf_counter()
    for points, context, optimal in zip(instances, contexts, optimal_tours):
        samples, _ = decode_model_samples(model, context, points, cfg["samples_per_graph"], rng)
        base = min(samples, key=lambda t: tour_length(points, t))
        local = two_opt(points, base)
        random_best, random_trace, _ = restart_search(points, base, cfg["restart_budget"], rng, "random")
        ucb_best, ucb_trace, _ = restart_search(points, base, cfg["restart_budget"], rng, "ucb")
        optimum = tour_length(points, optimal)
        for name, tour in (("reference_model", base), ("local_2opt", local), ("random_restart", random_best),
                           ("ucb_restart", ucb_best), ("optimal", optimal)):
            methods[name].append(tour_length(points, tour))
        random_traces.append(random_trace); ucb_traces.append(ucb_trace)
        hit = np.flatnonzero(random_trace <= optimum * 1.01); first_random.append(int(hit[0] + 1) if len(hit) else None)
        hit = np.flatnonzero(ucb_trace <= optimum * 1.01); first_ucb.append(int(hit[0] + 1) if len(hit) else None)
    optimum = np.asarray(methods["optimal"])
    summary = {}
    for name, values in methods.items():
        values = np.asarray(values)
        summary[name] = {"length_mean": float(values.mean()), "length_std": float(values.std()),
                         "optimality_gap_mean": float(np.mean((values - optimum) / optimum)),
                         "within_1pct_rate": float(np.mean(values <= optimum * 1.01))}
    summary["random_restart"]["median_queries_to_within_1pct"] = _median_nonnull(first_random)
    summary["ucb_restart"]["median_queries_to_within_1pct"] = _median_nonnull(first_ucb)
    return summary, np.mean(random_traces, 0), np.mean(ucb_traces, 0), time.perf_counter() - started


def _median_nonnull(values):
    valid = [x for x in values if x is not None]
    return float(np.median(valid)) if valid else None


def distill_labels(model, instances, contexts, cfg, rng):
    labels, improvements = [], []
    for points, context in zip(instances, contexts):
        probability = model.probabilities(context[None])[0]
        base = decode_edge_scores(probability, len(points))
        best, _, _ = restart_search(points, base, cfg["distill_budget"], rng, "ucb")
        labels.append(tour_to_edges(best, len(points)))
        improvements.append(tour_length(points, base) - tour_length(points, best))
    return np.asarray(labels), improvements
