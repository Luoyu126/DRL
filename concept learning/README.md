# 双高斯 Density：Score Learning 的 Step 0–1

这个零依赖小动画先建立后续讨论 DSM 和 DDPM score function 所需的共同对象：

后续的 reward-guided MCMC、Reward MLP、Score MLP 与 DSM distillation 实验见 [MLP Reward-Guided Exploration Toy 执行计划](./MLP_REWARD_EXPLORATION_PLAN.md)。

\[
p(x)=\pi\,\mathcal N(x;\mu_A,\sigma^2 I)+(1-\pi)\,\mathcal N(x;\mu_B,\sigma^2 I),\qquad x\in\mathbb R^2.
\]

- 左图：二维位置 `(x₁, x₂)` 上的 density landscape，高度是 `p(x)`。
- 右图：同一密度的俯视图，以及从 `p(x)` 独立抽取的样本。
- 青色箭头：浏览器内的两层 MLP 通过 DSM 在线学到的 score field。
- 紫色配对：clean sample `x₀`、noisy sample `x̃` 和条件 DSM target `(x₀-x̃)/σₙ²`。
- 橙色箭头：加噪分布的解析 score，仅在点击“揭晓解析答案”后显示，模型训练不读取它。
- 页面底部的手动单步区：一次构造一条训练数据，逐步显示 `x₀`、`ε`、`xₜ`、DSM label 和最终 squared loss，并代入当次采样的具体数值。
- 控件：改变混合权重 `π` 和两个峰的宽度 `σ`，观察山高和样本比例的变化。

## 运行

直接用浏览器打开 `index.html`，或在当前目录启动本地静态服务器：

```bash
cd "/home/chenyy/DRL/concept learning"
python3 -m http.server 8000
```

然后访问 <http://localhost:8000>。

## 当前形成的核心理解

### 1. Score 是一张 density 上的局部箭头场

对于一个 density `p(x)`，score function 是

\[
s(x)=\nabla_x\log p(x).
\]

它只描述当前位置上升最快的方向，是局部信息。对于多峰分布，峰附近的 score 主要反映局部峰形；不同 mode 的全局权重主要通过峰间 valley 中的箭头变化表现出来。

### 2. 一条 DSM 训练数据如何产生

先从数据分布采样 clean sample：

\[
x_0\sim p_{\mathrm{data}}.
\]

再采样独立高斯噪声并构造 noisy sample：

\[
\epsilon\sim\mathcal N(0,I),\qquad
x_t=x_0+\sigma_n\epsilon.
\]

已知条件分布

\[
q(x_t\mid x_0)=\mathcal N(x_0,\sigma_n^2I),
\]

所以能够直接计算 conditional score，作为监督 label：

\[
\begin{aligned}
y(x_t,x_0)
&=\nabla_{x_t}\log q(x_t\mid x_0)\\
&=\frac{x_0-x_t}{\sigma_n^2}\\
&=-\frac{\epsilon}{\sigma_n}.
\end{aligned}
\]

这个 label 的方向一定从 `x_t` 指向 `x₀`。从 `x_t` 到 `x₀` 的几何位移是 `x₀-x_t`，而 label 又除以了 `σₙ²`，因此二者方向相同，但长度通常不同。

网络用一条普通的回归损失学习它：

\[
\mathcal L
=\frac12\left\|s_\theta(x_t)-y(x_t,x_0)\right\|^2.
\]

### 3. 杂乱的 conditional labels 为什么能得到 density score

同一个 noisy position `x_t` 可能由许多不同的 `x₀` 加噪得到，所以单条 conditional label 会有很强的随机性，loss 也不必下降到零。

平方损失的最优预测是给定 `x_t` 后所有 conditional labels 的平均：

\[
\begin{aligned}
s_\theta^*(x_t)
&=\mathbb E\left[
\nabla_{x_t}\log q(x_t\mid x_0)
\mid x_t
\right]\\
&=\nabla_{x_t}\log p_{\sigma_n}(x_t).
\end{aligned}
\]

因此，DSM 虽然只使用容易计算的 conditional score 当 label，最终学到的却是整个 noisy marginal density `p_{σₙ}(x)` 的 score。

### 4. 多噪声 DSM：把 density 理解成许多平行切片

把单一 `σₙ` 换成多个噪声尺度后，可以把整体想成一叠“平行空间”：

```text
σ = large    高度平滑的 pσ(x)    score s(x, σlarge)
σ = medium   中度平滑的 pσ(x)    score s(x, σmedium)
σ = small    接近数据的 pσ(x)    score s(x, σsmall)
σ = 0        原始数据分布 p₀(x)
```

每张切片都在相同的 `x` 空间中，但具有不同的 noisy density 和不同的 score。通常不会为每一层训练独立网络，而是使用共享参数的条件网络：

\[
s_\theta(x,\sigma)\approx\nabla_x\log p_\sigma(x).
\]

### 5. DDPM 仍然在学习这些 density 切片的 score

DDPM 的前向过程可以直接写成：

\[
x_t
=\sqrt{\bar\alpha_t}\,x_0
+\sqrt{1-\bar\alpha_t}\,\epsilon
=c_t x_0+\sigma_t\epsilon,
\qquad
c_t=\sqrt{\bar\alpha_t},\quad
\sigma_t=\sqrt{1-\bar\alpha_t}.
\]

固定一个 `t`，就得到一张 noisy marginal density `p_t(x)`。因此整个 DDPM 可以理解成：

\[
\{p_t(x)\}_{t=0}^{T},
\]

而网络学习的是每张时间切片内部、沿 `x` 方向的 score：

\[
\boxed{
s_\theta(x,t)\approx\nabla_x\log p_t(x)
}
\]

这里不是在学习完整的 joint density `p(x,t)`，也不是计算 `t` 方向的梯度：

\[
s_\theta(x,t)\neq\nabla_{(x,t)}\log p(x,t).
\]

时间 `t` 只是告诉共享网络：当前输入位于哪一张 density 切片。

### 6. DDPM 的 noise prediction 是 score prediction 的另一种参数化

在 DDPM 的第 `t` 层，conditional DSM label 是

\[
\begin{aligned}
\nabla_{x_t}\log q(x_t\mid x_0)
&=-\frac{x_t-c_t x_0}{\sigma_t^2}\\
&=-\frac{\epsilon}{\sigma_t}.
\end{aligned}
\]

标准 DDPM 通常不直接输出 score，而是训练网络预测噪声：

\[
\mathcal L_{\mathrm{simple}}
=\mathbb E\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right].
\]

二者可以直接转换：

\[
\boxed{
s_\theta(x_t,t)
=-\frac{\epsilon_\theta(x_t,t)}{\sigma_t}
}
\]

所以 DDPM 的 noise-prediction training，本质上是重新参数化、重新加权的多噪声尺度 DSM。

### 7. DSM 与 DDPM 的分工

“平行 density 切片”的直觉对 DSM 和 DDPM 都成立，但 DDPM 还额外规定了切片之间的连接：

```text
前向加噪：p₀ → p₁ → p₂ → ... → pT
反向生成：pT → pT₋₁ → ... → p₁ → p₀
```

- DSM 回答：每一张 noisy density 切片上的 score 应该怎样学习？
- DDPM 回答：如何规定前向 Markov noising process，并利用学到的 score 构造逐步反向转移？
- 训练时可以随机抽一个 `t` 并直接构造 `x_t`，不必真的逐步从 `x₀` 运行到 `x_t`。
- 生成时从近似高斯的 `x_T` 出发，必须沿时间切片逐步反向移动到 `x₀`。

DDPM 的反向均值可以写成 score 形式：

\[
\mu_\theta(x_t,t)
=\frac{1}{\sqrt{1-\beta_t}}
\left(x_t+\beta_t s_\theta(x_t,t)\right),
\]

再进行反向采样：

\[
x_{t-1}=\mu_\theta(x_t,t)+\tilde\sigma_t z.
\]

因此当前最重要的总结是：

\[
\boxed{
\begin{aligned}
&\text{DDPM 学习整条 density 演化路径 }\{p_t(x)\}_{t=0}^{T}
\text{ 上的空间 score field，}\\
&\text{再使用 score 构造相邻时间切片之间的反向转移。}
\end{aligned}
}
\]

## 后续学习路线

1. 已完成：解析 score，`s(x) = ∇ₓ log p(x)`。
2. 当前：固定噪声尺度的 DSM，展示 `x₀`、`x̃`、conditional target 和 MLP 学到的 marginal score。
3. 加入 DDPM 时间轴：展示 `x₀ → x_t`、随 `t` 改变的 `p_t(x)` 与 `s(x,t)`。
4. 最后回到 working notes：观察 score 的局部爬坡性质、mode weights 在 valley 中留下的信息，以及探索缺失 mode 时的 blind spot。
