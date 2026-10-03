# MLP Reward-Guided Exploration Toy：执行计划

## 1. 目标

在二维 toy setting 中验证下面这条完整链路：

```text
reference distribution / pretrained score
                  +
        differentiable reward model
                  ↓
reward-tilted target p*(x)
                  ↓
global restart + local MCMC
                  ↓
发现当前 sampler 很少访问的高-reward mode B
                  ↓
将 MCMC 得到的 clean x₀ 样本重新加噪
                  ↓
通过 DSM 蒸馏进 time-conditioned score MLP
```

核心目标不是修改 score 的定义，而是检验：当普通 local reward guidance 困在已知 mode A 时，主动分配 restart budget 是否能更高效地发现远处、reward 更高的 mode B，并最终让 diffusion sampler 直接生成 B。

## 2. 核心研究假设

### H1：Local guidance 会受 basin 限制

如果 reward 在 A、B 附近各有一个局部峰，但 B 更高且距离较远，那么从 A 初始化的局部 gradient/Langevin dynamics 可能一直留在 A。

### H2：Global restart 可以提高 B 的发现率

Random restart 有机会进入 B 的 attraction basin；根据 reward、novelty 和 visit count 分配预算的 UCB/bandit restart 应当比随机 restart 更快、更稳定。

### H3：Propose-then-filter 可以降低 learned reward 的假发现

Reward MLP 在训练分布之外可能产生虚假的高-reward attractor。使用解析 `R_true` 作为只用于验收的 oracle，可以检验 filter 是否能拒绝 reward-model exploitation。

### H4：DSM 可以 amortize MCMC

将通过验证的 MCMC clean samples 作为新的 `x₀`，重新加噪训练 time-conditioned score MLP 后，新 diffusion sampler 应当无需运行探索型 MCMC，就能直接产生 B。

## 3. Toy distribution 与 reward

### 3.1 坐标范围

初始使用：

```text
x₁ ∈ [-4, 4]
x₂ ∈ [-3, 3]
μA = (-1.5, 0)
μB = (+1.5, 0)
```

所有二维 density、reward、score、MCMC trajectory 和 restart cell 都在该矩形内可视化。

### 3.2 两个 reference 难度级别

#### Setting A：Rare but represented mode

\[
p_{\mathrm{ref}}(x)
=0.99\,\mathcal N(x;\mu_A,\sigma_A^2I)
+0.01\,\mathcal N(x;\mu_B,\sigma_B^2I).
\]

用途：验证 rare-mode discovery 和 sampler coverage。B 已经存在于 reference score 中，因此不会混入“模型在 OOD 区域胡乱外推”的问题。

#### Setting B：Reward-induced effective mode

\[
p_{\mathrm{ref}}(x)
=\mathcal N(x;\mu_A,\sigma_A^2I).
\]

Gaussian 在全空间有非零 density，但普通采样几乎只会看到 A。通过足够强的 B reward bump，让 reward-tilted target 在 B 附近形成第二个有效 mode。

用途：验证 reward 是否能把 reference 中极低概率的区域提升为 target mode。

推荐先完成 Setting A，再运行 Setting B。

### 3.3 双峰 reward landscape

Reward 在 A、B 附近都有局部峰，但 B 更高：

\[
R_{\mathrm{true}}(x)
=r_A\exp\left(-\frac{\|x-\mu_A\|^2}{2\tau_A^2}\right)
+r_B\exp\left(-\frac{\|x-\mu_B\|^2}{2\tau_B^2}\right),
\qquad r_B>r_A.
\]

初始参数建议：

```text
σA = σB = 0.55
τA = τB = 0.45
rA = 1.0
β  = 1.0
```

`rB` 不写死，应通过 grid calculation 调整，使 target 中 B 的质量达到指定水平。

若两个 reference modes 分离良好且 mode 内 reward 近似常数，则：

\[
\frac{p^*(B)}{p^*(A)}
\approx
\frac{p_{\mathrm{ref}}(B)}{p_{\mathrm{ref}}(A)}
\exp\left(\frac{r_B-r_A}{\beta}\right).
\]

对于 `0.99 / 0.01` reference，若希望 tilted target 接近 `50 / 50`：

\[
r_B-r_A\approx\beta\log 99\approx4.595\beta.
\]

最终参数以二维数值积分得到的真实 target mass 为准，不只依赖上述近似。

### 3.4 Reward-tilted target

\[
p^*(x)
=\frac{1}{Z}
p_{\mathrm{ref}}(x)
\exp\left(\frac{R(x)}{\beta}\right).
\]

在 clean `x₀` 空间，其 score 为：

\[
s^*(x)
=\nabla_x\log p^*(x)
=s_{\mathrm{ref}}(x,0)
+\frac{\nabla_xR(x)}{\beta}.
\]

## 4. 分阶段实施

每次只将一个解析组件替换成 MLP，避免 score error、reward error 和 exploration error 同时出现。

### Phase 0：数值 ground truth 与可视化

实现二维 grid 上的：

- `p_ref(x)`；
- `R_true(x)`；
- 未归一化和归一化的 `p*(x)`；
- `s_ref(x)`；
- `∇R_true(x)`；
- `s*(x)`；
- A/B basin mass 的数值积分。

必须先验证：

1. A、B 都是 reward local maxima；
2. B reward 高于 A；
3. target `p*` 的确具有期望的两个 modes；
4. A 附近的 B-bump gradient 足够小，避免 local baseline 被人为设置成“从任何地方都知道 B 在哪里”。

### Phase 0 验收标准

- 图上能同时看到 `p_ref`、reward、`p*` 和 `s*`；
- 数值报告 reference 与 target 的 A/B mass；
- `rB`、`β` 可交互或通过配置修改；
- Setting A 与 Setting B 均能稳定构造目标 density。

### Phase 1：解析 score + 解析 reward

先使用完全正确的：

\[
s^*_{\mathrm{oracle}}(x)
=s_{\mathrm{ref,true}}(x)
+\frac{\nabla R_{\mathrm{true}}(x)}{\beta}.
\]

Local Langevin update：

\[
x_{k+1}
=x_k
+\frac{\eta}{2}s^*_{\mathrm{oracle}}(x_k)
+\sqrt{\eta}\,\xi_k,
\qquad \xi_k\sim\mathcal N(0,I).
\]

先验证 sampling、burn-in、cluster detection、restart allocation 和统计指标。该阶段失败说明 exploration pipeline 本身有问题，与 MLP 无关。

### Phase 1 条件

1. `base_sampling`：直接从 `p_ref` 采样；
2. `local_guidance`：从 `p_ref` samples 启动 reward-tilted Langevin；
3. `high_noise`：增加 Langevin noise/step size；
4. `random_restart`：在全局网格或边界框随机启动；
5. `ucb_restart`：按 value + uncertainty/novelty + visit count 分配 restart；
6. `oracle_restart`：直接在 B basin 附近启动，只用于 sanity check。

### Phase 1 验收标准

- Oracle restart 几乎总能在 B 附近形成稳定样本云；
- Local guidance 明显比 global restart 更难首次发现 B；
- Random restart 能以非零概率发现 B；
- UCB restart 在相同 reward-query/MCMC-step budget 下，比 random restart 有更短的 median time-to-first-B 或更高 discovery rate；
- MCMC 样本对 `p*` 的 mode weights 误差、MMD 或 Wasserstein 指标可计算。

### Phase 2：解析 score + Reward MLP

### 4.2.1 Reward dataset

构造：

\[
\mathcal D_R=\{(x_i,R_{\mathrm{true}}(x_i))\}.
\]

至少准备两种数据覆盖：

- `dense_grid`：充分覆盖整个二维区域，用于确认 MLP 容量足够；
- `biased_queries`：大部分点来自 A 和当前 policy，少量点来自 global exploration，用于模拟真实 reward-data bias。

### 4.2.2 Reward MLP

```text
input:  (x₁, x₂)
hidden: 64, SiLU
hidden: 64, SiLU
output: scalar reward
loss:   mean squared error
```

通过自动微分获得：

\[
\nabla_xR_\phi(x).
\]

MCMC proposal drift 使用：

\[
s_{\mathrm{proposal}}(x)
=s_{\mathrm{ref,true}}(x)
+\frac{\nabla R_\phi(x)}{\beta}.
\]

`R_true` 不提供给 proposal，只用于：

- 评估 reward prediction error；
- 检测 reward hacking；
- propose-then-filter；
- 计算真实性能指标。

### Phase 2 需要对比

1. `true_reward_gradient`；
2. `reward_mlp_no_filter`；
3. `reward_mlp_true_reward_filter`；
4. 可选：`reward_ensemble_uncertainty_filter`。

### Phase 2 验收标准

- Reward MLP 能拟合 A、B 两个峰；
- 绘制整个 grid 上的 reward error 和 gradient error；
- 明确记录是否出现 B 之外的 spurious high-reward attractor；
- 若存在假 attractor，external filter 应显著降低其进入 DSM buffer 的比例；
- 报告 learned reward 相对 true reward 对 discovery rate 和 target fidelity 的影响。

### Phase 3：Score MLP + Reward MLP

### 4.3.1 Time-conditioned score MLP

输入：

```text
(x₁, x₂, time embedding)
```

初始网络：

```text
time embedding: sinusoidal 或小型 learned embedding
hidden: 64, SiLU
hidden: 64, SiLU
output: (score₁, score₂)
```

DDPM-style forward process：

\[
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
\]

DSM score loss：

\[
\mathcal L_{\mathrm{DSM}}
=\mathbb E_{x_0,t,\epsilon}
\left[
\lambda(t)
\left\|
s_\theta(x_t,t)+\frac{\epsilon}{\sigma_t}
\right\|^2
\right].
\]

也可以先使用 noise prediction：

\[
\mathcal L_\epsilon
=\mathbb E\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right],
\qquad
s_\theta(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sigma_t}.
\]

### 4.3.2 MCMC 暂时只在 clean space 运行

第一版不直接在 noisy `x_t` 空间加入 terminal reward，因为正确 tilted noisy score 需要：

\[
h_t(x_t)
=\mathbb E\left[
e^{R(x_0)/\beta}\mid x_t
\right].
\]

因此先在 `x₀` 空间使用：

\[
s_{\mathrm{MCMC}}(x)
=s_\theta(x,t\approx0)
+\frac{\nabla R_\phi(x)}{\beta}.
\]

若最小训练时间不是严格 `t=0`，需要显式记录使用的 `t_min`，并把这一近似作为 ablation。

### 4.3.3 DSM distillation

1. Global policy 选择 restart cells；
2. Reward-guided MCMC 产生候选 clean samples；
3. External filter 保留有效 B samples；
4. 建立 replay buffer；
5. 按 `1-λ` 混合旧样本、按 `λ` 混合新样本；
6. 对 mixed `x₀` 重新随机采样 `t, ε`；
7. 更新完整的 `sθ(x_t,t)`；
8. 从 Gaussian `x_T` 运行 reverse sampler，检查是否能直接产生 B。

### Phase 3 验收标准

- Score MLP 在解析 mixture score 上达到可接受误差；
- 预训练 sampler 的 A/B 生成比例与 reference 接近；
- exploration buffer 中通过 filter 的 B 样本能稳定增加；
- DSM 更新后，不依赖 MCMC 的 reverse sampler 产生 B 的概率明显提升；
- 同时报告 target fidelity，防止模型只追求 discovery 而错误学习 mode weights；
- 对 `λ ∈ {0, 0.05, 0.1, 0.2, 0.4}` 做 sweep。

## 5. UCB / Global restart 设计

将二维区域切成规则 grid cells。每个 cell 维护：

```text
N_i       visit count
reward_i  filtered mean true reward 或 learned reward
novelty_i 距离已有 replay samples 的程度
success_i 进入稳定 cluster 的次数
```

初始 UCB score：

\[
U_i
=\widehat R_i
+c\sqrt{\frac{\log(N+1)}{N_i+1}}
+\gamma\,\mathrm{Novelty}_i.
\]

由于 score/reward model 会变化，后期加入 sliding window 或 discounted counts；第一版先固定模型验证 allocation logic。

必须与 uniform random restart 使用完全相同的：

- reward-query budget；
- chain 数量；
- 每条 chain 步数；
- 总 score evaluations；
- random seeds。

## 6. Cluster 与 filter

第一版二维 toy 可以使用简单规则：

- chain endpoint 距离小于 `ε_cluster`；
- 至少 `min_cluster_size` 个不同 restart seeds 到达；
- 在多个 MCMC step sizes 下仍然存在；
- `R_true` 超过阈值；
- 可选：target energy/log-density 超过阈值。

Filter 输出必须区分：

```text
accepted_A
accepted_B
rejected_low_reward
rejected_unstable
rejected_spurious_reward_model_peak
```

## 7. 指标

### Discovery

- 是否发现 B；
- time/reward queries/score evaluations to first B；
- 固定预算下的 B discovery rate；
- 不同随机种子下的均值、标准差和 median。

### Fidelity

- learned A/B mass 与 `p*` 的误差；
- Wasserstein distance；
- MMD；
- grid 上 score MSE；
- reverse sampler distribution 与 `p*` 的差异。

### Reward

- 平均 `R_true`；
- 平均 `Rφ`；
- reward-model exploitation gap：`Rφ - R_true`；
- 被 filter 拒绝的高-predicted-reward samples 数量。

### Cost

- reward queries；
- score evaluations；
- MCMC steps；
- wall-clock time；
- 有效样本数和 ESS（若使用 importance weights）。

## 8. 必须包含的 baselines

1. Plain reference/diffusion sampling；
2. Local reward gradient ascent（无 Langevin noise）；
3. Local reward-guided Langevin；
4. Higher-noise Langevin；
5. Importance resampling：`x ~ p_ref`, `w = exp(R/β)`；
6. Random global restart + MCMC；
7. UCB global restart + MCMC；
8. Oracle B restart；
9. Reward MLP without filter；
10. Reward MLP with external filter。

如果 local guidance 已经稳定覆盖 B，或 UCB 与 random restart 没有显著差异，应当如实认为相应组件没有带来价值。

## 9. Ablations

- `β`：reward/reference trade-off；
- `rB-rA`：B 的相对吸引力；
- `τB`：B reward basin 宽度；
- B 与 A 的距离；
- Langevin step size 和 noise；
- restart region/grid resolution；
- chain length 与 burn-in；
- reward dataset coverage；
- reward MLP capacity；
- filter 开/关；
- `λ`：旧数据与探索数据混合比例；
- exact score vs score MLP；
- exact reward vs reward MLP。

## 10. 可视化产物

至少保存以下图或动画：

1. `p_ref(x)` heatmap + score arrows；
2. `R_true(x)` 与 `Rφ(x)`；
3. reward prediction/gradient error map；
4. `p*(x)` heatmap + target score arrows；
5. local/random/UCB MCMC trajectories；
6. restart cell visit heatmap；
7. accepted/rejected clusters；
8. DSM 更新前后的 score field；
9. DSM 更新前后的 generated A/B samples；
10. discovery-vs-fidelity 曲线和 `λ` sweep。

## 11. 建议工程结构

```text
concept learning/
├── README.md
├── index.html
├── MLP_REWARD_EXPLORATION_PLAN.md
├── requirements.txt
├── configs/
│   ├── setting_a.yaml
│   └── setting_b.yaml
├── src/
│   ├── distributions.py
│   ├── rewards.py
│   ├── reward_mlp.py
│   ├── score_mlp.py
│   ├── langevin.py
│   ├── restarts.py
│   ├── filters.py
│   ├── metrics.py
│   └── plotting.py
├── scripts/
│   ├── phase0_ground_truth.py
│   ├── phase1_exact_components.py
│   ├── phase2_reward_mlp.py
│   └── phase3_score_distillation.py
└── outputs/
    ├── figures/
    ├── animations/
    └── metrics/
```

当前系统 Python 环境尚未安装 `torch`、`numpy`、`matplotlib` 或 `scipy`。正式实现前应创建项目级虚拟环境并将依赖固定在 `requirements.txt` 中，不修改系统 Python。

## 12. 实施顺序 Checklist

- [ ] 创建隔离的 Python 环境与固定依赖；
- [ ] 实现二维 reference distributions；
- [ ] 实现双峰 `R_true`；
- [ ] 数值归一化并绘制 `p*`；
- [ ] 校准 `rB, β, τB`，保证 target 确有 B mode；
- [ ] 实现 exact target score；
- [ ] 实现 local Langevin 和轨迹可视化；
- [ ] 实现 random/global/oracle restart；
- [ ] 实现 UCB restart；
- [ ] 完成 Phase 1 公平预算比较；
- [ ] 构建 reward dataset；
- [ ] 训练 Reward MLP 并绘制 value/gradient error；
- [ ] 实现 true-reward filter；
- [ ] 完成 Phase 2 reward hacking/filter 实验；
- [ ] 实现 time-conditioned Score MLP；
- [ ] 完成 reference DSM pretraining；
- [ ] 建立 accepted-sample replay buffer；
- [ ] 实现 `λ` 混合与 DSM distillation；
- [ ] 完成 Phase 3 reverse generation；
- [ ] 运行多随机种子与全部 baselines；
- [ ] 输出 discovery、fidelity、reward、cost 报告；
- [ ] 回到 working notes，根据结果收窄或否定 novelty claim。

## 13. 最终成功标准

只有同时满足以下条件，才能认为核心 hypothesis 得到支持：

1. Grid ground truth 证明 tilted target 的 B mode 真实存在；
2. 相同预算下，local guidance 经常漏掉 B；
3. Global restart 显著提高 B discovery；
4. UCB 相对 random restart 有可重复的额外收益；
5. Filter 能减少 Reward MLP 导致的假 cluster；
6. DSM distillation 后，独立 diffusion generation 的 B 概率提高；
7. 提高 discovery 的同时，target mode weights 和整体分布误差仍然可接受；
8. 结果在多个 seeds 下成立，而非单次幸运发现。

如果第 3 条不成立，global exploration 没有必要；如果第 4 条不成立，bandit allocation 没有贡献；如果第 5 条不成立，filter 没有贡献。所有 negative result 都应保留并用于修改最终方法定位。
