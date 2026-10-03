# Formal CPU results

The formal runs completed for three independent seeds. Each seed uses 800 MIS training graphs, 1,000 TSP training graphs, and 160 exactly evaluated test instances per task. Values below are mean ± standard deviation across seeds.

| Method | MIS optimality gap | TSP optimality gap |
|---|---:|---:|
| Conditional solution model | 13.45 ± 0.11% | 26.77 ± 0.73% |
| Local refinement | 5.28 ± 0.39% | 0.58 ± 0.07% |
| Random global restart | 0.044 ± 0.062% | 0.00 ± 0.00% |
| UCB global restart | 0.079 ± 0.069% | 0.00 ± 0.00% |
| Distilled solution model | 6.97 ± 0.69% | 23.84 ± 0.73% |

## Interpretation

1. Global restart clearly improves over one local trajectory on both tasks under the tested budget. On average, MIS local search retains a 5.28% gap and TSP 2-opt retains a 0.58% gap; global restart nearly closes both.
2. The discovered solutions contain learnable signal. Direct MIS generation improves from a 13.45% gap to 6.97% after distillation. Direct TSP generation improves more modestly, from 26.77% to 23.84%.
3. UCB has no demonstrated advantage here. Random restart is marginally better on MIS and tied on TSP. Both small problems are easy enough that a strong restart arm often reaches the optimum in its first query, leaving little room for bandit allocation.
4. TSP exposes a model bottleneck. The small fixed-size MLP learns useful edge-distance correlations but does not reliably generate a globally consistent tour heatmap. A permutation-equivariant GNN or structured tour decoder is the appropriate next model.
5. Distillation is not uniformly beneficial to every downstream procedure: although direct MIS generation improves, local search from distilled samples becomes worse in this run. Replay changes the starting-solution distribution and therefore needs fidelity/diversity control.

## CPU cost

- Typical per-seed MIS total: about 20 seconds; exact test labels take well under one second.
- Typical per-seed TSP total: about two minutes; Held–Karp exact test labels take about eight seconds.
- Both experiments use exact objective values and always-feasible decoders.

This is positive evidence for the core “global discovery → replay distillation” idea on MIS and weaker positive evidence on TSP. It does not support an UCB novelty claim at these problem sizes.

Full per-seed and aggregate values are stored in `outputs/metrics/multiseed_summary.json`.
