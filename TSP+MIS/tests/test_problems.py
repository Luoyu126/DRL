import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

from src.mis import greedy_independent, maximum_independent_set, mask_to_bits
from src.tsp import held_karp, instance_contexts, tour_length, two_opt


class ProblemTests(unittest.TestCase):
    def test_mis_exact_cycle(self):
        graph = np.zeros((5, 5), dtype=np.uint8)
        for i in range(5): graph[i, (i + 1) % 5] = graph[(i + 1) % 5, i] = 1
        solution = mask_to_bits(maximum_independent_set(graph), 5)
        self.assertEqual(int(solution.sum()), 2)
        self.assertEqual(int(solution @ graph @ solution), 0)

    def test_mis_greedy_feasible(self):
        graph = np.ones((6, 6), dtype=np.uint8) - np.eye(6, dtype=np.uint8)
        solution = greedy_independent(graph, np.arange(6))
        self.assertEqual(int(solution.sum()), 1)

    def test_mis_exact_matches_bruteforce(self):
        rng = np.random.default_rng(9)
        for _ in range(8):
            upper = np.triu(rng.random((8, 8)) < .3, 1)
            graph = np.logical_or(upper, upper.T).astype(np.uint8)
            exact = mask_to_bits(maximum_independent_set(graph), 8)
            brute = 0
            for mask in range(1 << 8):
                bits = mask_to_bits(mask, 8)
                if not np.any((graph @ bits) * bits):
                    brute = max(brute, int(bits.sum()))
            self.assertEqual(int(exact.sum()), brute)

    def test_tsp_exact_square(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        tour = held_karp(points)
        self.assertAlmostEqual(tour_length(points, tour), 4.0, places=8)

    def test_two_opt_nonworsening(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        tour = np.array([0, 2, 1, 3])
        self.assertLessEqual(tour_length(points, two_opt(points, tour)), tour_length(points, tour))

    def test_tsp_context_matches_edges(self):
        points = np.array([[[0, 0], [3, 0], [0, 4]]], dtype=float)
        self.assertEqual(instance_contexts(points).tolist(), [[3.0, 4.0, 5.0]])
