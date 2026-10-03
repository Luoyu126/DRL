# MLP Reward-Guided Exploration Toy

This directory implements all four phases in `MLP_REWARD_EXPLORATION_PLAN.md`. It uses a small NumPy neural-network implementation so the experiment is reproducible without a framework-specific runtime. All generated artifacts stay under `outputs/`.

## Setup

```bash
./setup_env.sh
```

The YAML files are deliberately JSON-compatible, so no YAML parser is required.

## Run

Fast end-to-end validation:

```bash
.venv/bin/python scripts/run_all.py --config configs/setting_a.yaml --smoke
.venv/bin/python scripts/run_all.py --config configs/setting_b.yaml --smoke
```

Full configured experiment:

```bash
.venv/bin/python scripts/run_all.py --config configs/setting_a.yaml
.venv/bin/python scripts/run_all.py --config configs/setting_b.yaml
```

Complex multimodal stress test:

```bash
.venv/bin/python scripts/phase4_complex_multimodal.py --config configs/setting_c_multimodal.yaml --smoke
.venv/bin/python scripts/phase4_complex_multimodal.py --config configs/setting_c_multimodal.yaml
```

Setting C adds broad, needle-like and boundary reward modes, two negative-reward hazards, reward-data coverage/capacity ablations, fixed versus multi-scale MCMC, budget curves, raw versus target-weighted replay, and an oracle-complete replay upper bound.

Each phase can also be run independently. Every script accepts `--config`, `--output`, and `--smoke`.

## Outputs

- `outputs/figures`: density, score, reward-error, trajectory, sample, discovery and fidelity plots.
- `outputs/metrics`: machine-readable per-phase and hypothesis summary JSON.
- `outputs/models`: pickled Reward MLP and Score MLP checkpoints.
- `outputs/animations`: reserved for optional trajectory animations.

The Phase 1 comparison fixes chain counts, chain lengths, reward-query counts, score evaluations, and seeds for random versus UCB restarts. Phase 3 records `t_min` explicitly because clean-space use of the diffusion score is an approximation. Summary checks are observations, not assertions: negative results remain in the reports as required by the plan.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```
