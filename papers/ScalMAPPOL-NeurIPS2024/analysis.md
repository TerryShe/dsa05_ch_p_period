# ScalMAPPOL-NeurIPS2024：可扩展的多智能体约束策略优化

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Scalable Constrained Policy Optimization for Safe Multi-agent Reinforcement Learning |
| 作者 | Lijun Zhang, Lin Li, Wei Wei, Huizhong Song, Yaodong Yang, Jiye Liang（山西大学、北京大学） |
| 来源 | NeurIPS 2024（38th Conference on Neural Information Processing Systems） |
| 本地 PDF | [ScalMAPPOL-NeurIPS2024.pdf](ScalMAPPOL-NeurIPS2024.pdf)，33 页（含附录与清单），[会议论文集 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/fa76985f05e0a25c66528308dda33de0-Paper-Conference.pdf) |
| 阅读范围 | 2026-10-14；摘要、§1、方法与定理的陈述、实验设置与结果概要；证明与附录未读 |

## 研究问题

安全 MARL：多智能体协作最大化回报，同时满足各自的代价约束。指出已有方法（MAPPO-L、CMIX 等）依赖全局状态的集中式价值函数，在通信或计算受限、规模较大的系统中不可用。

## 方法

- 结合信任域方法的界与截断优势函数的界，为每个智能体给出新的本地策略优化目标。
- 证明：每个智能体按顺序更新、优化 κ 跳策略时，安全约束与联合策略的改进都能满足。
- 实用算法 Scal-MAPPO-L：每个智能体只看自己与邻居的局部状态，训练时不依赖全局状态与其他智能体的动作。
- 依赖网络结构的"影响随距离指数衰减"假设。

## 实验

Safe ManyAgent Ant、Safe Ant、Safe Coupled HalfCheetah 等安全多智能体基准（机器人控制）。每个智能体只看约一半智能体的状态时，Scal-MAPPO-L 的回报与约束满足与使用全局状态的 MAPPO-L 相当；κ ≥ 3 时接近。

## 与 DSA05 的关系

- 方法层面：逐智能体约束下去中心化训练的 MAPPO-Lagrangian 变体，是"约束 MARL 已有成熟算法"的证据。
- 不适用之处：它的可扩展性依赖局部性（κ 跳截断）。DSA05 是单 AP 上行，所有 UE 在同一组子信道上竞争，没有这种局部结构。
- 约束是机器人安全代价（每步都有信号），不是稀疏的截止期违约。

## 引用建议

- 相关工作：安全/约束 MARL 的代表算法之一；说明我们没有提出新的约束 MARL 算法。
