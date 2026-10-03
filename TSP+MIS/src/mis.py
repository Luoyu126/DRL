from __future__ import annotations

import time

import numpy as np


def generate_graphs(count, n, edge_probability, rng):
    upper = rng.random((count, n, n)) < edge_probability
    upper = np.triu(upper, 1)
    return np.logical_or(upper, np.transpose(upper, (0, 2, 1))).astype(np.uint8)


def graph_contexts(graphs):
    n = graphs.shape[1]
    rows, cols = np.triu_indices(n, 1)
    return graphs[:, rows, cols].astype(float)


def mask_to_bits(mask, n):
    return np.asarray([(mask >> i) & 1 for i in range(n)], dtype=float)


def bits_to_mask(bits):
    mask = 0
    for i, value in enumerate(bits):
        if value:
            mask |= 1 << i
    return mask


def maximum_independent_set(graph):
    """Exact bitset branch-and-bound, intended for n <= 24."""
    n = len(graph)
    adjacency = [sum((int(graph[i, j]) << j) for j in range(n)) for i in range(n)]
    best_mask, best_size = 0, 0

    def search(candidates, chosen):
        nonlocal best_mask, best_size
        chosen_size = chosen.bit_count()
        if chosen_size + candidates.bit_count() <= best_size:
            return
        if candidates == 0:
            if chosen_size > best_size:
                best_mask, best_size = chosen, chosen_size
            return
        vertices = [i for i in range(n) if candidates & (1 << i)]
        vertex = max(vertices, key=lambda i: (adjacency[i] & candidates).bit_count())
        bit = 1 << vertex
        search(candidates & ~bit & ~adjacency[vertex], chosen | bit)
        search(candidates & ~bit, chosen)

    search((1 << n) - 1, 0)
    return best_mask


def greedy_independent(graph, scores):
    selected = np.zeros(len(graph), dtype=np.uint8)
    blocked = np.zeros(len(graph), dtype=bool)
    for node in np.argsort(-np.asarray(scores)):
        if not blocked[node]:
            selected[node] = 1
            blocked[node] = True
            blocked |= graph[node].astype(bool)
    return selected


def local_improve(graph, solution, max_rounds=12):
    current = np.asarray(solution, dtype=np.uint8).copy()
    degrees = graph.sum(1)
    # Fill any immediately available nodes.
    current = greedy_independent(graph, current * 100 - degrees)
    for _ in range(max_rounds):
        improved = None
        selected = np.flatnonzero(current)
        for remove in selected:
            trial = current.copy(); trial[remove] = 0
            blocked = (graph @ trial) > 0
            available = np.flatnonzero((trial == 0) & ~blocked)
            for node in available[np.argsort(degrees[available])]:
                if trial[node] == 0 and not np.any(graph[node] & trial):
                    trial[node] = 1
            if trial.sum() > current.sum():
                improved = trial; break
        if improved is None:
            break
        current = improved
    return current


def heuristic_labels(graphs, rng):
    labels = []
    for graph in graphs:
        degree = graph.sum(1)
        scores = -degree + rng.normal(scale=.35, size=len(graph))
        labels.append(greedy_independent(graph, scores))
    return np.asarray(labels, dtype=float)


def decode_model_samples(model, context, graph, count, rng):
    probabilities = model.sample_probabilities(context, count, rng)
    return [greedy_independent(graph, p) for p in probabilities], probabilities


def _arm_candidate(graph, model_probability, arm, rng):
    degree = graph.sum(1)
    if arm == 0:
        scores = model_probability + rng.normal(scale=.08, size=len(graph))
    elif arm == 1:
        scores = -degree + rng.normal(scale=.5, size=len(graph))
    elif arm == 2:
        scores = rng.normal(size=len(graph))
    elif arm == 3:
        scores = model_probability + rng.normal(scale=.35, size=len(graph))
    elif arm == 4:
        anchor = int(rng.integers(len(graph)))
        scores = rng.normal(size=len(graph)); scores[anchor] += 4
    else:
        raise ValueError(arm)
    return local_improve(graph, greedy_independent(graph, scores))


def restart_search(graph, model_probability, budget, rng, strategy):
    arms = 5
    counts, totals = np.zeros(arms, dtype=int), np.zeros(arms)
    best, best_size, trace, arm_trace = None, -1, [], []
    for step in range(budget):
        if strategy == "random":
            arm = int(rng.integers(arms))
        elif strategy == "ucb":
            unseen = np.flatnonzero(counts == 0)
            if len(unseen):
                arm = int(unseen[0])
            else:
                mean = totals / counts
                arm = int(np.argmax(mean + 1.2 * np.sqrt(np.log(step + 1) / counts)))
        else:
            raise ValueError(strategy)
        candidate = _arm_candidate(graph, model_probability, arm, rng)
        value = int(candidate.sum())
        counts[arm] += 1; totals[arm] += value
        if value > best_size:
            best, best_size = candidate, value
        trace.append(best_size); arm_trace.append(arm)
    return best, np.asarray(trace), np.asarray(arm_trace), counts


def evaluate(model, graphs, contexts, exact_masks, cfg, rng):
    methods = {name: [] for name in ("reference_model", "local", "random_restart", "ucb_restart", "optimal")}
    random_traces, ucb_traces, first_random, first_ucb = [], [], [], []
    start = time.perf_counter()
    for graph, context, optimal_mask in zip(graphs, contexts, exact_masks):
        optimal = mask_to_bits(optimal_mask, len(graph))
        samples, probabilities = decode_model_samples(model, context, graph, cfg["samples_per_graph"], rng)
        base = max(samples, key=lambda x: x.sum())
        local = local_improve(graph, base)
        model_probability = probabilities.mean(0)
        random_best, random_trace, _, _ = restart_search(graph, model_probability, cfg["restart_budget"], rng, "random")
        ucb_best, ucb_trace, _, _ = restart_search(graph, model_probability, cfg["restart_budget"], rng, "ucb")
        for name, solution in (("reference_model", base), ("local", local), ("random_restart", random_best),
                               ("ucb_restart", ucb_best), ("optimal", optimal)):
            methods[name].append(int(solution.sum()))
        optimum = int(optimal.sum())
        random_traces.append(random_trace); ucb_traces.append(ucb_trace)
        hit = np.flatnonzero(random_trace >= optimum); first_random.append(int(hit[0] + 1) if len(hit) else None)
        hit = np.flatnonzero(ucb_trace >= optimum); first_ucb.append(int(hit[0] + 1) if len(hit) else None)
    optimum = np.asarray(methods["optimal"])
    summary = {}
    for name, values in methods.items():
        values = np.asarray(values)
        summary[name] = {"size_mean": float(values.mean()), "size_std": float(values.std()),
                         "optimality_gap_mean": float(np.mean((optimum - values) / optimum)),
                         "optimal_hit_rate": float(np.mean(values == optimum))}
    summary["random_restart"]["median_queries_to_optimum"] = _median_nonnull(first_random)
    summary["ucb_restart"]["median_queries_to_optimum"] = _median_nonnull(first_ucb)
    return summary, np.mean(random_traces, 0), np.mean(ucb_traces, 0), time.perf_counter() - start


def _median_nonnull(values):
    valid = [x for x in values if x is not None]
    return float(np.median(valid)) if valid else None


def distill_labels(model, graphs, contexts, cfg, rng):
    labels, improvements = [], []
    for graph, context in zip(graphs, contexts):
        probability = model.probabilities(context[None])[0]
        base = greedy_independent(graph, probability)
        best, _, _, _ = restart_search(graph, probability, cfg["distill_budget"], rng, "ucb")
        labels.append(best); improvements.append(int(best.sum()) - int(base.sum()))
    return np.asarray(labels, dtype=float), improvements
