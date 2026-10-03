# Reward-Guided Exploration for Generative Combinatorial Optimization

> **Repository layout:** The `main` branch intentionally contains only this project overview. All source code, configurations, tests, subproject documentation, and versioned experiment artifacts are maintained on the [`develop` branch](https://github.com/Luoyu126/DRL/tree/develop). Please switch to `develop` before running or modifying the project.

## 1. 项目目标与当前研究路线

本项目研究一个生成模型与搜索相互促进的闭环：

```text
条件生成模型产生候选解
        ↓
objective-guided search 发现更优或不同的解
        ↓
筛选高质量且多样化的解，形成 replay buffer
        ↓
使用原生成模型的训练目标继续训练
        ↓
模型在未见实例上产生更好的初始解
        ↓
更好的初始解进一步帮助下一轮搜索
```

这里的核心不只是“搜索可以把一个解变好”，而是验证：

> 搜索发现的解能否被稳定地蒸馏回生成模型，并转化为模型在未见问题实例上的可泛化能力。

前期已经用低维 toy setting 和小规模 CPU 图优化原型检查了基本机制。下一阶段不同时展开所有组合优化问题，而是聚焦 **diffusion + Euclidean TSP**，先在一条最匹配当前方法的路线上进行严格、可控的验证。

---

## 2. 方法共识

### 2.1 探索与 replay 的含义

当前方法可以理解为一种由搜索驱动的数据增强或自举训练：

1. 使用原始数据训练条件生成模型；
2. 从模型生成的候选解出发进行局部搜索、扰动或全局重启；
3. 收集搜索最终得到的高质量解；
4. 将这些解与原始训练数据按比例混合；
5. 从预训练 checkpoint 出发，继续使用原来的生成模型损失训练。

当前 replay 的基本单位是搜索得到的最终解，而不是整条搜索轨迹。工程阶段属于 post-training、fine-tuning 或 continued training，但训练范式仍然是原模型的去噪/生成目标，不是直接用强化学习更新模型。Objective 主要用于搜索、评价和样本筛选，不一定直接进入 diffusion loss。

普通数据增强通常对已有样本做旋转、加噪等变换；这里希望通过搜索主动发现原训练集中缺失或稀少的优质 solution modes。

### 2.2 UCB 在当前框架中的位置

UCB（Upper Confidence Bound）是一种在多个搜索操作之间分配尝试预算的 bandit 策略。它同时考虑：

- 某个操作过去取得的平均收益；
- 该操作是否因为尝试次数少而仍具有不确定性。

在 TSP 中，不同 arm 可以对应 2-opt、relocate、3-opt、double-bridge、模型重新采样或随机 insertion restart。收益可以是路线长度改善、单位时间改善、是否刷新最优解，或者是否发现新的高质量路线结构。

UCB 目前只是待验证的 operator-selection 方法，不应预设为核心贡献。它必须与 random selection 和固定规则在相同计算预算下比较。

### 2.3 从 toy setting 迁移到组合优化时发生了什么

TSP 和 MIS 通常具有可精确计算的 objective：

- TSP：给定合法 tour，可以准确计算总路程；
- MIS：给定合法节点集合，可以准确检查冲突并计算集合大小。

真正缺少的通常是：

- 完整的目标解分布；
- 该分布的 exact score；
- 大规模实例的 exact optimum；
- 所有高质量 solution modes 的完整信息。

因此，真实组合优化实验不再以“生成分布是否逼近一个已知二维密度”为主要证据，而要考察：

- 固定计算预算内能否找到更好的解；
- 能否发现高质量且结构不同的解；
- 搜索结果能否改善模型直接生成的能力；
- 改善能否泛化到未参与 replay 的新实例。

---

## 3. TSP 与 MIS 情境

### 3.1 TSP

一个 Euclidean TSP 实例是一组城市坐标。任意两个城市之间的边权是它们的距离。一个合法解是一条 Hamiltonian cycle：每个城市恰好访问一次，最后回到起点。目标是让整条 tour 尽可能短。

给定一条合法路线，其长度很容易精确计算；困难在于合法路线数量随城市数快速增长，无法穷举找到最短路线。

同一个实例可能有许多长度接近但边结构不同的优质路线。这些路线可以形成不同的 solution basins。某条路线可能已经无法通过一次简单局部修改继续缩短，却仍然可以通过更大范围的结构重组到达更好的 basin。

TSP 具有需要在表示和评估中处理的对称性：同一条环路可以从不同城市开始书写，也可以反向书写。因此，路线多样性不能只比较排列字符串，还应比较 edge overlap、结构差异和去重后的 tour。

### 3.2 MIS

Maximum Independent Set 的输入是一张无向图，输出是一组互不相邻的节点，目标是选择尽可能多的节点。给定候选集合后，合法性和 objective 都可以精确计算。

需要区分 maximal independent set 和 maximum independent set：前者只是无法直接再添加节点，后者才是全图最大的独立集。局部贪心很容易停在 maximal 但非 maximum 的解。

MIS 与 TSP 虽然都是图组合优化，但输出结构不同：MIS 更像带冲突约束的节点二元选择，TSP 则是带全局环约束的排列或边集合。因此二者通常需要不同的模型、decoder 和搜索邻域。

当前正式路线先聚焦 TSP；已有 MIS 原型仅作为前期验证记录，见第 9 节。

---

## 4. TSP 的经典求解组件

TSP 方法可以拆成两部分：先构造一条初始路线，再从该路线出发进行改进。不同构造器与搜索器可以组合。

### 4.1 初始化/构造方法

- **Nearest Neighbor**：每次访问最近的未访问城市，速度快但容易留下昂贵的收尾边。
- **Nearest/Farthest/Cheapest/Random Insertion**：从小环开始，逐渐把剩余城市插入路线。
- **Clarke–Wright Savings**：根据合并路径能节省多少距离来构造路线。
- **Christofides**：针对 metric TSP 的经典近似算法，具有理论近似保证。
- **神经生成模型**：根据整张城市图采样一条或多条路线。

### 4.2 局部改进操作

- **2-opt**：删除两条边，反转中间路线段，再以另一种方式连接；特别适合消除交叉边。
- **3-opt / k-opt**：同时替换三条或更多边，邻域更强但成本更高。
- **Relocate / Or-opt**：移动一个城市或一段连续城市到路线的其他位置。
- **Swap**：交换两个城市的位置。
- **Lin–Kernighan**：动态决定交换深度的可变 k-opt 方法。
- **LKH**：Lin–Kernighan 的高性能扩展，是需要面对的强经典基线。

2-opt 的直观例子：

```text
原路线：A → B → C → D → E → F → A
删除边：B—C、E—F
新增边：B—E、C—F
新路线：A → B → E → D → C → F → A
```

数字 2 表示一次替换两条边；中间的 `C → D → E` 被整体反转。反复执行能够缩短路线的 2-opt，最终得到的是 2-opt local optimum，并不保证是全局最优。

### 4.3 逃出局部最优

- **Random restart / Multi-start**：重新生成独立起点，每个起点都运行相同 local search，最后保留最好路线。
- **Double-bridge move**：切断四条边并重排若干大段；保留多数原有结构，同时跳出 2-opt basin。
- **Iterated Local Search（ILS）**：反复执行 `local search → perturbation → local search → acceptance`；TSP 中常用 double-bridge 作为 perturbation。
- **Simulated Annealing**：允许以随温度变化的概率暂时接受更差的路线。
- **Tabu Search**：暂时禁止最近使用过的操作，减少循环并强迫搜索新区域。
- **GRASP、Genetic Algorithm、Ant Colony Optimization**：分别通过随机化贪心、种群演化和信息素统计进行更大范围探索。

Multi-start、double-bridge 和 ILS 都是已有经典方法。“重启比单条 2-opt 更好”不能单独作为新贡献。它们应当成为我们方法的基线和组成部件。

可以把 TSP 操作理解为不同搜索尺度：

| 尺度 | 示例 |
|---|---|
| 很小 | swap、relocate |
| 小 | 2-opt |
| 中 | 3-opt、Or-opt |
| 可变 | Lin–Kernighan |
| 大扰动 | double-bridge |
| 全局 | 随机构造或模型重新生成 |

---

## 5. 神经网络求解 TSP 的主要路线

### 5.1 自回归构造模型

模型编码全部城市，然后每一步选择下一个尚未访问的城市。已访问城市被 mask，因此最终自然得到一个排列。

- Pointer Network 通过 attention 指向输入节点；
- Attention Model 使用 Transformer 风格的编码与解码；
- POMO 利用多起点和多个等价优质解进行并行 rollout。

训练可以采用两种方式：

- 监督学习：模仿 exact solver 或高质量 heuristic 产生的路线；
- 强化学习：直接把 tour length 作为信号，不要求每个实例都有最优标签。

推理时通常还会结合多次采样、不同起点、坐标增强、beam search 或 2-opt。

### 5.2 边概率/heatmap 模型

模型一次性为城市对打分，预测哪些边可能属于高质量 tour。由于独立选取高分边可能产生节点度数错误或多个 subtour，仍需 greedy decoder、beam search 或组合搜索将 heatmap 转换成合法路线。

### 5.3 Diffusion 生成模型

把 tour 表示为边选择矩阵或离散边向量，对干净路线逐步加噪，并训练条件模型根据城市图进行去噪。推理时从噪声结构出发，逐渐恢复路线，再通过 decoder/repair 得到合法 tour。

这一范式适合生成多样化解，也最贴合本项目现有的探索与 replay 思路。DIFUSCO 是 TSP/MIS 图 diffusion 的代表性工作。

### 5.4 Neural improvement

模型不从零构造 tour，而是观察当前路线并选择下一步修改，例如直接选择两个 2-opt 切点、选择搜索 operator、判断是否 restart，或预测哪些 move 值得优先检查。

### 5.5 Neural-guided classical solver

神经网络只指导成熟求解器的关键决策：

- NeuroLKH 学习候选边和 penalty，再由 LKH 完成主体搜索；
- DeepACO 学习启发式信息，再由蚁群系统采样和更新。

总体上更现实的路线通常是把神经模型的快速预测、跨实例泛化，与经典算法的可行性保证和精细搜索结合起来，而不是要求神经网络完全取代强求解器。

代表资料：

- [Pointer Networks](https://arxiv.org/abs/1506.03134)
- [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)
- [POMO](https://arxiv.org/abs/2010.16011)
- [DIFUSCO](https://arxiv.org/abs/2302.08224)
- [Learning 2-opt Heuristics via Deep Reinforcement Learning](https://arxiv.org/abs/2004.01608)
- [NeuroLKH](https://arxiv.org/abs/2110.07983)
- [DeepACO](https://arxiv.org/abs/2309.14032)

---

## 6. 下一阶段的核心问题

下一阶段不以“立即成为最强 TSP solver”为目标，而是先回答一个受控问题：

> 在相同 diffusion backbone、训练数据和计算预算下，搜索产生的数据经过合理筛选与 replay 后，能否让模型在未见 TSP 实例上稳定优于原模型、普通 continued training 和朴素搜索 replay？

建议首先复现一个成熟的 diffusion TSP backbone，优先考虑 DIFUSCO。随后只修改探索、样本选择和 replay 部分，尽量不同时更换模型结构，避免无法判断收益来源。

第一阶段不需要从头复现所有神经 TSP 方法，也不要求 beat 所有方法。其他范式主要用作性能坐标和强 baseline。若最终声称“新的最强 TSP solver”，才需要全面挑战各路线的强方法；若声称“生成模型的通用自我改进方法”，首要证据是对相同 backbone 的稳定增益、严格消融和跨实例泛化。

---

## 7. 重点实验计划：Diffusion + TSP + Search Replay

### Phase 0：明确任务、数据与预算

初始任务建议使用二维 Euclidean TSP：

- 从 TSP-50 开始，机制稳定后扩展到 TSP-100；
- train、validation、test 图严格分离；
- 搜索和 replay 只使用训练图；
- validation 用于选择 replay 比例、搜索预算和 checkpoint；
- test 图不得参与 replay；
- 明确每种方法的采样数、objective evaluation 次数、训练步数、CPU/GPU 时间和搜索 wall-clock time。

需要分别报告两种设置：

1. **Offline self-improvement**：只在训练图上搜索和更新，在未见测试图上评估；
2. **Test-time adaptation**：允许针对同一个测试实例边搜索边更新。

优先验证第一种，因为它能说明模型学习了可迁移的求解规律，而不是记住特定实例。

### Phase 1：复现 diffusion backbone

目标是确认基础系统可信，而不是立即加入新方法：

- 复现或运行官方 diffusion TSP 实现；
- 核对 TSP-50/TSP-100 的直接生成质量；
- 检查输出合法率和 decoder/repair 行为；
- 检查多次采样是否真的产生不同 tour；
- 加入标准 2-opt，确认输出存在合理的可改进空间；
- 保存统一格式的路线、edge set、tour length、采样时间和搜索时间。

如果基础复现与官方结果差距过大，应先解决 backbone、数据生成、decoder 或评估问题，再进入 replay 实验。

### Phase 2：建立经典搜索基线

固定同一个初始解来源，比较：

```text
模型直接生成
模型 + 2-opt
模型 + 3-opt 或 relocate
模型 + double-bridge ILS
random/heuristic multi-start + 相同 local search
model multi-start + 相同 local search
LKH（强性能参照）
```

建议把搜索 operator 组织成多尺度集合：

```text
arm 1: 2-opt
arm 2: relocate / Or-opt
arm 3: 3-opt
arm 4: double-bridge + local search
arm 5: diffusion model resample
arm 6: randomized insertion restart
```

先比较固定 schedule 和 random operator selection，再把 UCB 作为候选策略加入。所有策略必须共享相同的 objective-query 或 wall-clock 预算。

这一阶段只回答搜索问题，不更新模型：

- 不同初始化与搜索组合的质量—时间曲线如何；
- 模型 restart 是否优于随机或启发式 restart；
- 多尺度 operator 是否比单一 2-opt 更有效；
- UCB 是否稳定优于 random，而不是只在单个 seed 上领先。

### Phase 3：构造 replay buffer

对训练实例运行统一预算的搜索，保存最终高质量候选解及元数据：

- instance ID；
- 初始模型及 checkpoint；
- 初始 tour 与最终 tour；
- 使用的搜索 operator 和计算预算；
- tour length 与相对改进；
- edge set；
- 与同实例已有路线的结构差异；
- 是否刷新该实例当前最好结果。

需要比较多种 buffer 构造方式：

- **quality-only**：只保留最短路线；
- **quality + diversity**：兼顾长度和 edge-level 新颖性；
- **per-instance balanced**：避免少数容易搜索的实例占据 replay；
- **operator-balanced**：避免某个高产 operator 完全主导 buffer；
- **random-search replay**：作为朴素 replay 对照；
- **our exploration replay**：使用最终提出的探索和筛选策略。

### Phase 4：继续训练与关键消融

所有实验从同一个预训练 checkpoint 出发。核心对照矩阵：

| 组别 | 搜索数据 | 继续训练 | 作用 |
|---|---:|---:|---|
| Pretrained diffusion | 否 | 否 | 原始基线 |
| Continued training | 否 | 是，只用原数据 | 排除只是多训练的影响 |
| Search only | 是 | 否 | 测量测试时搜索收益 |
| Naive replay | 是 | 是 | 测量普通搜索数据增强 |
| Quality-only replay | 是 | 是 | 测量只追求最短路线的效果 |
| Quality + diversity replay | 是 | 是 | 测量多样性控制 |
| Proposed exploration replay | 是 | 是 | 完整方法 |

必须控制：

- 新增路线数量；
- 训练步数与学习率；
- 原始数据和 replay 的混合比例；
- 搜索使用的总 objective evaluations 或总时间；
- 每个实例贡献的样本数；
- 测试时采样数和后处理预算。

还应加入一个很重要的“额外标签”基线：用相同计算预算让普通 ILS 或 LKH 生成同样数量的新路线，再进行相同 continued training。这用于回答提升是否只是因为调用强求解器获得了更多标签。

### Phase 5：评估模型是否真正吸收搜索发现

在完全未见过的测试图上，分别评估：

#### A. 不带搜索的模型能力

- greedy/direct generation tour length；
- best-of-K sampling；
- optimality gap（存在 exact optimum 或可信 reference 时）；
- 合法率；
- 同一实例多次采样的 edge diversity；
- 推理时间。

这是验证蒸馏是否成功的核心结果。

#### B. 带相同搜索预算的最终能力

- pretrained model + search；
- replay-updated model + 同样 search；
- 最终 tour length；
- 达到指定质量阈值所需时间；
- quality-versus-budget curve；
- 不同规模上的泛化。

如果更新后的模型不带搜索时更好，并且在相同搜索预算下也得到更好的最终解，说明模型与搜索形成了正反馈。

### Phase 6：稳定性与泛化

至少进行：

- 多个训练 seed 和搜索 seed；
- TSP-50 同分布测试；
- TSP-50 训练、TSP-100 测试的规模外推；
- 不同城市坐标分布的 distribution shift；
- replay ratio、buffer size、diversity threshold 消融；
- 搜索轮数和自举轮数消融。

报告均值、标准差和失败案例，不能只展示最好 seed。

### Phase 7：决定是否扩大研究范围

只有在以下信号成立后再扩展：

1. 相同 backbone 上的 replay 增益稳定；
2. continued-training 和额外标签基线无法解释全部增益；
3. 测试图上直接生成质量提升；
4. quality + diversity replay 优于朴素 quality-only replay；
5. 收益在多个 seed 和至少两个规模上存在。

随后可以选择：

- 第二个 diffusion/generative TSP backbone；
- 一个自回归 backbone，检验方法是否跨模型范式；
- MIS，检验是否跨问题；
- 更大规模 TSP；
- 与 POMO、NeuroLKH、LKH 等进行更完整的横向比较。

---

## 8. 结果解释与贡献边界

不同结果支持的结论不同：

| 观察 | 能支持的结论 | 不能直接支持的结论 |
|---|---|---|
| 2-opt 改善模型输出 | 输出仍有局部改进空间 | replay 有效 |
| ILS 优于单次 2-opt | 大尺度扰动有价值 | 我们的方法新颖 |
| Search only 改善最终路线 | 搜索算法有效 | 模型能力提高 |
| Replay 后训练图变好 | 可能发生记忆或适配 | 能泛化到新图 |
| Replay 后未见测试图的直接生成变好 | 搜索知识被模型吸收并泛化 | 已经是最强 TSP solver |
| Proposed replay 优于等预算 naive replay | 探索/筛选策略具有额外价值 | 对所有模型和任务都通用 |

若最终只证明 diffusion backbone 上的受控提升，合适的贡献表述是“面向生成式组合优化模型的搜索—replay 自我改进机制”。若要声称通用性，至少需要第二个 backbone 或第二个问题；若要声称 TSP state of the art，则需要与各范式最强方法进行严格、统一预算的完整比较。

---

## 9. 已有仓库内容索引

以下工作已经存在于仓库中。本文件只提供入口，不重复同步其实现、实验结果和详细结论；以各子目录文档为准。

### Toy setting

- 实验计划：[toy setting/MLP_REWARD_EXPLORATION_PLAN.md](https://github.com/Luoyu126/DRL/blob/develop/toy%20setting/MLP_REWARD_EXPLORATION_PLAN.md)
- 运行说明：[toy setting/README.md](https://github.com/Luoyu126/DRL/blob/develop/toy%20setting/README.md)
- 结果与解释：[toy setting/RESULTS.md](https://github.com/Luoyu126/DRL/blob/develop/toy%20setting/RESULTS.md)
- 配置、代码和输出：[`toy setting/`](https://github.com/Luoyu126/DRL/tree/develop/toy%20setting)

该目录包含二维 reward-guided exploration、局部搜索与全局重启、random/UCB、多尺度搜索、reward/score 学习和 replay distillation 等前期实验。

### 小规模 CPU TSP + MIS 原型

- 运行说明：[TSP+MIS/README.md](https://github.com/Luoyu126/DRL/blob/develop/TSP%2BMIS/README.md)
- 实验设计：[TSP+MIS/DESIGN.md](https://github.com/Luoyu126/DRL/blob/develop/TSP%2BMIS/DESIGN.md)
- 结果与解释：[TSP+MIS/RESULTS.md](https://github.com/Luoyu126/DRL/blob/develop/TSP%2BMIS/RESULTS.md)
- 配置、代码和输出：[`TSP+MIS/`](https://github.com/Luoyu126/DRL/tree/develop/TSP%2BMIS)

该目录是小规模 NumPy/CPU 机制验证，不是正式 DIFUSCO 或大型 GNN 复现。后续 diffusion + TSP 实验应作为新的正式实验线，不应把该原型的模型能力当成神经 TSP 的最终结论。

### Concept learning

- 入口：[concept learning/README.md](https://github.com/Luoyu126/DRL/blob/develop/concept%20learning/README.md)

该目录与本阶段 TSP 计划分开维护。

---

## 10. 当前推荐的最小可行研究路径

```text
1. 复现一个可信的 diffusion TSP backbone
2. 建立 2-opt、double-bridge ILS、multi-start 和 LKH 参照
3. 固定预算，确认模型 restart 与不同搜索尺度的实际行为
4. 在训练图上收集 quality/diversity-controlled replay
5. 从相同 checkpoint 进行受控 continued training
6. 在未见测试图上分别测 direct generation 和 search-assisted performance
7. 用 continued training、naive replay、等预算 solver labels 排除混淆因素
8. 多 seed、跨规模验证后，再决定扩展到第二个 backbone 或 MIS
```

现阶段的首要目标不是复现并击败所有 TSP 方法，而是用一套严格的实验回答：**搜索发现是否能够以超过朴素数据增强的方式，持续提升 diffusion TSP 模型的可泛化生成能力。**
