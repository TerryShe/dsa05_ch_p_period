# MARLSPA-TCOM2025：循环 MAPPO 做多小区频谱与功率分配

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Multi-Agent Reinforcement Learning for Multi-Cell Spectrum and Power Allocation |
| 作者 | Yiming Zhang, Dongning Guo |
| 来源 | IEEE Transactions on Communications, 2025，DOI 10.1109/TCOMM.2025.3534565（据作者代码库 README 的引用信息核实）；本地为 arXiv:2312.05746v2 预印本（2024 年修订），未与出版社终版逐一比对 |
| 代码 | https://github.com/yimingzhang2026/MARL-multicell |
| 本地 PDF | [MARLSPA-TCOM2025.pdf](MARLSPA-TCOM2025.pdf)（13 页） |
| 阅读范围 | 摘要、引言、第 II 节 MARL 框架（Dec-POMDP-IR）与冲突图模型开头 |

## 问题与方法

- **场景**：多小区下行，每个 AP 是一个智能体，决定服务哪个用户、用哪些子带、用多大功率。业务随机到达，目标是**最小化平均包时延**。
- **框架**：Dec-POMDP-IR，即部分可观测、每个智能体有自己的奖励，允许邻域内交换观测。不假设智能体之间转移独立，也不使用共同的合作奖励。
- **算法**：循环神经网络加 MAPPO，考虑队列动态。作者把 CTDE 改成去中心化的训练与执行，使每个智能体的网络规模和训练开销不随网络规模增长。
- **结果**：在冲突图模型和蜂窝模型上，QoS 与需要全局信息的集中式方法（如 WMMSE、FP）相当，执行开销小得多，并能推广到不同规模与负载。

## 与 DSA05 的关系

- **算法上最接近**：循环 MAPPO、子带加功率联合决策、以集中式优化为参照、关注队列与业务而不是满载速率。
- **主要差别**：
  1. 智能体是下行 AP（每个小区一个）vs DSA05 的上行 UE；
  2. 目标是平均时延 vs 截止期、交付率约束与能耗；
  3. 使用个体奖励和邻域信息交换 vs DSA05 用团队奖励、集中式 Critic，执行时不交换信息；
  4. 随机到达 vs 异步确定性周期。
- **评价上的对照**：本文以 WMMSE/FP 为参照；DSA05 给出能耗的中央上下界，并在测试集上用两层 bootstrap 做统计判读，这可以作为方法论上的优势陈述。

## 引用建议

- 相关工作：近年 MAPPO 在频谱与功率分配上的应用。
- 发表信息：引用 TCOM 2025 正式版。
