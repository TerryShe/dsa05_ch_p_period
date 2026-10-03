# MAPPO-NeurIPS2022：先建立可靠的循环 MAPPO 基线

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games |
| 作者 | Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, Yi Wu |
| 来源 | NeurIPS 2022，Datasets and Benchmarks Track；早期预印本 arXiv:2103.01955 |
| 官方页面 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html) |
| PDF 来源 | [会议正式版](https://proceedings.neurips.cc/paper_files/paper/2022/file/9c1535a02f0ce079433344e14d910597-Paper-Datasets_and_Benchmarks.pdf) |
| 本地 PDF | [MAPPO-NeurIPS2022.pdf](MAPPO-NeurIPS2022.pdf)，14 页 |
| 代码 | [作者实现](https://github.com/marlbenchmark/on-policy) |
| 阅读范围 | §3 方法与实现、§4 实验设置、§5 五项消融，PDF 第 3–10 页；附录未逐项复核 |
| 本次归档 | 2026-10-02 |

## 论文的核心内容

这篇工作证明，经过合适实现与调参的 MAPPO 在多种合作任务中具有很强的表现。它把主要影响因素分为价值归一化、Critic 输入、数据复用、PPO 截断和批量大小。价值归一化作用于回归目标，计算 GAE 时再还原价值尺度；同构智能体可以共享参数，集中式价值网络只在训练时使用。作者在 MPE、SMAC、GRF 和 Hanabi 上评估，并强调公平的调参预算。

## 在不增加 OBS 的条件下，能借鉴什么

**完全兼容现有信息边界。** ValueNorm、Critic 结构、优化批次和更新约束都属于训练过程；Actor 继续接收原有本地 OBS。论文的 agent-specific Critic 是训练网络，不意味着给 Actor 新增其他 UE 的状态。

对 DSA05 最有价值的是基线校准，而不是再把“使用 MAPPO”当作算法贡献：

1. 当前 `trainer_v2.py` 归一化了优势，但 Critic 仍回归原始 returns。这两种归一化处理的对象不同，不能由前者推断后者对奖励尺度不敏感。应先比较 A 与经过价值尺度稳定化的训练方案，再判断对偶权重是否无效。
2. ValueNorm 的均值和方差必须随检查点保存。GAE、自举和评估加载保持一致；不能只缩放标签而保留未还原的价值输出。
3. 当前按 active UE 平均 Actor 损失。混合 N 训练时，大 N 环境可能提供更多有效样本，因此占更大梯度权重。可比较“逐 UE 平均”和“先逐环境平均”，并记录各 N 的有效样本量；这是 DSA05 的实现诊断，不是本文已证明的结论。
4. 环境截断时要自举，真正终止时不自举；循环状态重放应保留相同的片段起始 hidden state。长回合与短反向传播片段应分开设置。

## 对 EXP006 的具体价值

EXP006 已显示中等负载的可靠性改善，但 1% 条件下容量没有超过最强本地规则。论文提醒我们：需要先排除价值拟合、更新幅度和数据使用方式的影响。它不能说明调参后一定提高容量，也不能替代针对尾部可靠性的目标设计。

建议先做小规模训练诊断：保留原始 A；添加仅价值归一化的 A；再添加业务周期归一化奖励的对照。记录 Critic explained variance、原始尺度预测误差、梯度裁剪比例、KL 和不同 N 的贡献。每次只变一个因素。

## 新颖性与引用建议

- **参考优先级：高，实施优先级：最高。** 它是后续算法比较的基线依据。
- 价值归一化和 PPO 调参本身不足以支撑目标期刊的主贡献。
- 不能把本文在游戏上的经验超参数直接视为无线可靠性场景的最优设置。
- 可在方法与实验设置中引用，用于解释共享 Actor、CTDE、价值尺度处理和公平预算。

综合路线见[固定 OBS 文献综述](../../docs/research/literature-fixed-obs-marl_20261002.md)。
