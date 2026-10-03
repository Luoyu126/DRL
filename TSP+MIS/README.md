# CPU validation: TSP + MIS

This project tests reward/objective-guided global exploration and replay distillation on two graph combinatorial-optimization problems without a GPU.

- MIS: a solution is an `n`-bit node-selection vector. The exact objective is independent-set cardinality, and feasibility is enforced by greedy decoding.
- TSP: a solution is an upper-triangular binary edge vector. The graph context is the aligned edge-distance vector; decoded tours are always Hamiltonian cycles and the exact objective is negative tour length.
- The conditional solution model is a small NumPy Bernoulli denoiser. It is intentionally CPU-sized and is not a reproduction of the 12-layer DIFUSCO GNN.
- Random and UCB policies receive identical objective-query budgets and choose among the same restart operators.

## Run

```bash
./setup_env.sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/run_all.py --smoke
.venv/bin/python scripts/run_all.py
.venv/bin/python scripts/run_multiseed.py
```

Individual tasks:

```bash
.venv/bin/python scripts/run_mis.py
.venv/bin/python scripts/run_tsp.py
```

Formal configuration uses MIS-18 and Euclidean TSP-12, 800/1000 training graphs, 160 test graphs, exact test optima, 32 restart queries per graph, and 12 distillation-search queries per training graph. The checked results include three independent seeds (`42`, `123`, and `777`).

## Outputs

- `outputs/metrics`: complete JSON reports and a compact summary.
- `outputs/figures`: method comparison, budget curve and example-solution plots.
- `outputs/models`: pretrained and distilled model checkpoints.

See `RESULTS.md` for interpretation. All reported experiments ran on CPU.
