# CRL-Timely-Asilomar2025：按时吞吐量约束下的约束深度强化学习网络控制

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | A Constrained RL Approach for Cost-Efficient Delivery of Latency-Sensitive Applications |
| 作者 | Ozan Aygün, Vincenzo Norman Vitale, Antonia M. Tulino, Hao Feng, Elza Erkip, Jaime Llorca（NYU、那不勒斯大学、Intel Labs、特伦托大学、CTTC） |
| 来源 | 2025 Asilomar Conference on Signals, Systems, and Computers（arXiv 页面标注已接收）；arXiv:2603.04353（v1，2026-03-04） |
| 本地 PDF | [CRL-Timely-Asilomar2025.pdf](CRL-Timely-Asilomar2025.pdf)，7 页，arXiv v1 |
| 阅读范围 | 2026-10-14；全文浏览：§I–§II 模型、§III 约束 MDP、§IV 对偶次梯度与 DRL 原始更新、§V 仿真设置与结果 |

## 研究问题

最小代价的时延约束网络控制（MDNC）：多跳网络中多个业务流（commodity），每个包有寿命（TTL），过期即失效。**每个业务流的按时吞吐量（按时送达率）须不低于可靠性目标 δ_c**，同时最小化资源分配的代价（例如功耗）。作者指出，基于平均时延的随机优化方法（背压、UMW 等）处理不了逐包截止期。

## 方法（CDRL-NC）

- 建模为约束 MDP；拉格朗日松弛后用**对偶次梯度**：每个业务流一个乘子 λ_c，每轮先用 DRL 做 V 个回合的原始更新，再按"按时吞吐量与目标之差"更新乘子（乘子在训练轮之间更新，不进入观测）。
- 多智能体：一个集中式路由智能体加各节点的分布式调度智能体，用 MADDPG 协同训练。

## 实验

事件驱动的网络仿真，边缘网络拓扑；对照为 UMW 等。CDRL-NC 在满足可靠性目标的同时代价更低，在基线达不到目标的场景中也能满足；乘子随约束满足逐渐下降。

## 与 DSA05 的关系

**相同**
- 逐业务流的"按时送达率 ≥ 目标"约束，与我们"每个 UE 违约率 ≤ ε"同型；用逐流的拉格朗日乘子处理。
- 说明"按时吞吐量约束 + 约束 DRL"是 2025 年仍在做的问题。

**不同**
- 场景：多跳网络的路由与调度，有集中式路由智能体；DSA05 是单 AP 上行多址接入，完全分布式。
- 乘子的用法：在训练轮之间更新、不进入策略观测（相当于我们的变体 B）；不在执行时在线调整。
- 没有物理层衰落与干扰的细节。

## 引用建议

- 相关工作：按时吞吐量（timely throughput）约束下的约束 DRL；与 Hou–Kumar 的按时吞吐量理论一起，作为逐 UE 可靠性约束的背景。
