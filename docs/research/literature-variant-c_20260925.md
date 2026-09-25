---
doc_id: LIT-DSA05-001
type: literature-note
date: 2026-09-25
related: EXP-DSA05-001
---

# 变体 C（逐 UE 局部对偶）的相关工作与新颖性判断

EXP-DSA05-001 定稿时约定：变体 C 的新颖性在写作前另做文献检索确认。本文记录 2026-09-25 的初步检索（网络检索与摘要阅读，尚非系统综述）。

## 结论

**变体 C 的核心机制本身不新。** "把对偶变量或虚拟队列放进策略输入和逐时隙奖励、执行时在线更新"这一做法，在无线资源管理中已有成体系的工作，最近也已扩展到多智能体、逐智能体一个对偶变量的形式。所以不能把"逐 UE 对偶变量作为状态"当作论文的算法贡献。

本课题与已有工作真正不同的地方，是**问题与信息结构**，以及**可以量化最优差距的中央上下界**。论文的贡献需要围绕这些来定位；如果要有算法上的新意，还得另外找切入点（见文末）。

## 最相关的工作

| 工作 | 机制 | 与本课题的区别 |
|---|---|---|
| NaderiAlizadeh, Eisen, Ribeiro, [State-Augmented Learnable Algorithms for Resource Management in Wireless Networks](https://arxiv.org/abs/2207.02242)，IEEE TSP 2022 | 把逐用户约束的对偶变量作为策略输入，执行时按对偶梯度在线演化 | 集中式策略（掌握网络状态），不是只用本地观测的匿名多智能体 |
| Uslu, NaderiAlizadeh, Eisen, Ribeiro, [Fast State-Augmented Learning for Wireless Resource Allocation with Dual Variable Regression](https://arxiv.org/abs/2506.18748)，2025 | 状态增广 GNN，逐用户对偶变量作为节点输入 | 同上，集中式，基于图的全局信息 |
| Calvo-Fullana, Paternain, Chamon, Ribeiro，状态增广约束强化学习（arXiv:2102.11941） | 状态增广约束 RL 的理论框架 | 单智能体理论，是变体 C 的来源 |
| Amaya-Corredor, Calvo-Fullana, Jonsson, [Scalable Constrained MARL via State Augmentation and Consensus for Separable Dynamics](https://arxiv.org/html/2605.30461)，2026 | 每个智能体维护本地对偶变量、作为状态输入，执行时与邻居做共识 | 约束是**共享的全局资源约束**，需要邻居通信；场景为智能电网，没有干扰耦合 |
| Jeong, Ko, Hong, Hong, [HeLyMARL](https://arxiv.org/pdf/2608.01745)，2026 | 每个 BS/用户一个虚拟队列（能耗、切换预算），放进状态和逐时隙奖励，用 HAPPO 训练 | 有限时域预算约束，蜂窝下行关联与 BS 开关，追求 PF 吞吐；与本课题的机制最接近 |
| Bi, Huang, Wang, Zhang, [LyDROO](https://arxiv.org/abs/2010.01370)，IEEE TWC 2021 | Lyapunov 引导的 DRL，虚拟队列进入每帧的子问题 | 单智能体（集中式），MEC 卸载 |
| Zhang, Wei, Fan, Liu, Huang, [Lyapunov-guided MARL for Delay-Sensitive Wireless Scheduling](https://arxiv.org/abs/2411.01766)，2024 | 把时延违约约束转换为逐时隙的队列稳定性，MARL 决定优先级，底层调度器分配资源 | 有集中式调度器；目标是时延抖动与违约率，不涉及能耗和功率 |

## 本课题的差异点（论文可以站住的部分）

1. **信息结构**：UE 匿名，只有本地因果观测（自身反馈与公共 RSSI），没有他人 CSI，执行时也没有任何通信或共识。上面的工作要么集中式，要么需要邻居通信。
2. **约束结构**：每个 UE 的长期违约率约束，只在截止期结算时触发（事件驱动、稀疏），并且经由同信道干扰相互耦合。第 1 阶段的调参结果显示，用全网平均违约率驱动一个全局乘子（变体 B）会让策略长期牺牲少数 UE（最差 UE 的完成率降到约 0.25），而逐 UE 约束（变体 A、C）没有出现这种现象。
3. **场景**：异构周期、异步相位的截止期业务，HARQ-IR 累积互信息，离散功率与子信道联合选择，在上行能耗最小化这一目标下。
4. **可量化的最优差距**：N≤K 时有精确最优，N>K 时有经过环境重放核验的上下界（正交分时 MILP 与逐 UE 单独最优）。大多数 MARL 无线论文只和启发式比较。
5. **泛化**：训练池有 100 万个拓扑与相位实例，测试集完全不参与训练和选模。

## 仍可争取的算法贡献（待与用户讨论）

- **理论**：匿名智能体各自持有本地对偶变量时，逐 UE 约束的可行性保证（在状态增广理论的基础上扩展到多智能体、干扰耦合、事件触发的更新）。
- **信用分配**：违约只在截止期出现，稀疏且延迟。可以按"进度赤字"做基于势函数的奖励塑形（这种塑形不改变最优策略），检验它能否提高可行率、缩短训练时间。
- **周期结构的协调**：业务是周期性的，理想调度是周期重复的正交分时（上界 MILP 正是这样），可以研究让 UE 学出稳定的"时隙—子信道"节奏。

以上只是初步检索，定稿论文前需要系统地检索 IEEE Xplore 中 TWC、TCOM、JSAC、TCCN、IoT-J 近三年的相关文献。
