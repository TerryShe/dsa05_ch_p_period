# 文献库

每篇论文一个子文件夹，文件夹名是论文的缩写加年份。文件夹里有论文 PDF（从 arXiv 下载）和 `analysis.md`（针对本课题的阅读分析）。阶段性的综合阅读笔记放在 `docs/research/`，例如 [变体 C 的相关工作与新颖性判断](../docs/research/literature-variant-c_20260925.md)。

| 文件夹 | 论文 | 来源 | 与本课题的关系 |
|---|---|---|---|
| [SACRL-2021](SACRL-2021/analysis.md) | Calvo-Fullana 等，State Augmented Constrained RL | arXiv:2102.11941 | 变体 C 的理论来源：乘子进入状态，执行时运行对偶动态 |
| [SALA-TSP2022](SALA-TSP2022/analysis.md) | NaderiAlizadeh 等，State-Augmented Learnable Algorithms for RRM | IEEE TSP 2022 | 无线领域"逐用户对偶作为输入"的代表作；集中式 GNN，没有状态动态 |
| [FSAL-DVR-2025](FSAL-DVR-2025/analysis.md) | Uslu 等，Fast State-Augmented Learning with Dual Variable Regression | arXiv:2506.18748 | 对偶初始化与训练乘子分布；可启发本地化的 μ0 回归 |
| [SACC-MARL-2026](SACC-MARL-2026/analysis.md) | Amaya-Corredor 等，State Augmentation and Consensus for Separable Dynamics | TMLR 2026 | 多智能体状态增广；动态可分离、全局约束、需要邻居共识 |
| [HeLyMARL-2026](HeLyMARL-2026/analysis.md) | Jeong 等，Heterogeneous MARL for RRM under Coupled Finite-Horizon Constraints | arXiv:2608.01745 | 机制最接近：逐智能体虚拟队列进入奖励和观测；有限时域硬预算，需要信令 |
| [LyDROO-TWC2021](LyDROO-TWC2021/analysis.md) | Bi 等，Lyapunov-Guided DRL for Online Computation Offloading | IEEE TWC 2021 | 无线领域"虚拟队列 + 深度强化学习"的经典；集中式单智能体 |
| [LyMARL-DS-2024](LyMARL-DS-2024/analysis.md) | Zhang 等，Lyapunov-guided MARL for Delay-Sensitive Wireless Scheduling | arXiv:2411.01766 | 约束类型最接近：逐用户时延违约概率与虚拟队列；有集中调度器 |
| [DPD-CMARL-2024](DPD-CMARL-2024/analysis.md) | Kahe 与 Kebriaei，Distributed Primal-Dual Method for Constrained MARL | arXiv:2410.15335 | 共享约束、乘子共识、完全分布式学习；关系较远 |
| [DePAint-2023](DePAint-2023/analysis.md) | Hassan 等，DePAint | Applied Intelligence（arXiv 标注已接收） | 逐智能体的峰值与平均约束、邻居通信；关系较远 |

## 尚缺的文献

以上都来自围绕"约束处理与状态增广"的检索。问题本身这一侧的文献还没有系统检索，写论文前需要补上：

- 能耗最优的包调度，例如 lazy scheduling：把发送摊开、降低功率以节能，这正是本课题中央最优解的形态；
- 截止期约束的无线调度与去中心化策略，例如 dsa04 第二阶段文档中提到的 Singh 与 Kumar 的工作；
- 免授权（grant-free）上行接入与 IoT 周期业务；
- 多信道随机接入中的功率控制；
- 近三年 IEEE TWC、TCOM、JSAC、TCCN、IoT-J 上的 MARL 频谱接入与功率控制论文。
