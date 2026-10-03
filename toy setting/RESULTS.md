# Experiment results

Formal configured runs were completed for both settings. Raw metrics are in `outputs/metrics/`; figures and model checkpoints are in their sibling output directories.

| Metric | Setting A | Setting B |
|---|---:|---:|
| Calibrated `rB` | 7.383 | 17.280 |
| Numerical target B mass | 0.500 | 0.350 |
| Local-guidance B fraction | 0.0188 | 0.0094 |
| Random-restart B fraction | 0.4094 | 0.2844 |
| UCB-restart B fraction | 0.4156 | 0.2781 |
| Random median chains to first B | 5 | 4 |
| UCB median chains to first B | 2 | 3 |
| Accepted Phase 3 B buffer | 11 | 43 |
| Pretrained diffusion B fraction | 0.0043 | 0.0003 |
| Distilled B fraction (`lambda=0.4`) | 0.2930 | 0.2460 |
| Distilled MMD (`lambda=0.4`) | 0.0705 | 0.0513 |

## Interpretation

- The grid integration verifies a real reward-tilted B mode in both settings.
- Local guidance usually misses B, while global restart discovers it in every formal seed.
- UCB reduces median time-to-first-B in both settings. It has slightly higher final B coverage than random in Setting A, but slightly lower coverage in Setting B. The evidence therefore supports faster discovery, not a universal UCB coverage advantage.
- Dense reward data fits both peaks well. Biased data is materially worse, especially in Setting B (value MSE 0.435 versus 0.0077 for dense data), and learned-reward exploration correspondingly finds fewer B endpoints.
- The formal runs did not produce endpoints classified as spurious high-predicted-reward peaks. The external filter removes low-true-reward endpoints, but H3's specific false-attractor reduction claim is not demonstrated by these seeds.
- DSM distillation raises direct, MCMC-free B generation monotonically over the configured lambda sweep and improves MMD through `lambda=0.4` in both settings.

The reverse sampler still leaves about 20% of pretrained samples outside the radius-based A/B basins. The clean-space use of the learned score is explicitly an approximation at `t_min=0.01` and uses a recorded score-norm clip of 8.0. These limitations should be addressed before treating the toy results as a polished method claim.

## Setting C: complex multimodal stress test

Setting C contains four positive basins (`home`, `broad`, `needle`, and `boundary`) plus two negative-reward hazards. Numerical target masses are 20.9%, 25.0%, 16.0%, and 14.0%, with 24.1% background mass.

### What worked

- Fixed-step local guidance covered only 1.2 of the three novel modes on average.
- Fixed-step random/UCB restart covered 2/3 modes; neither could retain the narrow `needle` mode.
- Multi-scale random and UCB restart covered all 3/3 novel modes for every formal seed. Multi-scale UCB produced a 60.1% total novel-mode fraction versus 22.7% for multi-scale random, at twice the score-evaluation cost of the fixed kernel.
- A 128-wide peak-aware Reward MLP reduced grid value MSE from 2.44 (64-wide peak-aware) to 1.08 and enabled learned-reward proposals to reach all three novel modes when paired with the exact reference score.
- Target-weighted replay reduced distilled-sampler MMD from 0.111 (raw replay, `lambda=0.4`) to 0.026 despite an imbalanced discovered buffer.

### Failure modes found

- A single Langevin step size can erase a genuine narrow target mode even when initialized at its center. Global restart alone does not solve kernel mismatch.
- Uniform dense reward data is not necessarily dense enough for a needle mode. Reference-biased and blind datasets have low training loss while missing important reward structure.
- The blind Reward MLP has a 43.7% false-positive rate among its grid top-1% predictions, showing that training loss is not a reliable safety signal under distribution shift.
- During blind-model exploration, the external true-reward filter rejected 39 high-proxy/low-true-reward endpoints and accepted only one endpoint, directly demonstrating the propose-then-filter safeguard under severe coverage bias.
- Combining learned reward and learned score changed which mode dominated exploration. The final learned-score replay contained 150 `boundary`, 3 `broad`, and 0 `needle` samples: global discovery can still collapse before distillation.
- Raw replay taught almost only the dominant boundary mode. Target-weighted replay restored broad/boundary balance but could not recover an absent needle mode.
- With an oracle-complete replay containing every mode, the same DSM architecture generated all 3/3 novel modes and reached MMD 0.021. Thus the remaining needle failure is upstream data acquisition/replay coverage, not a fundamental inability of the score MLP to represent it.

Overall, the complex experiment supports the core global-exploration idea, but shows that a robust version needs multi-scale transition kernels, reward-data coverage checks, diversity-aware allocation, and per-mode replay control. Plain UCB plus one MCMC scale is insufficient.
