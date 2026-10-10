# SeqDQN-EuCNC2023：异步周期上行与严格截止期的分布式学习

## 基本信息与核验

| 项目 | 内容 |
|---|---|
| 标题 | SeqDQN: Multi-Agent Deep Reinforcement Learning for Uplink URLLC with Strict Deadlines |
| 作者 | Benoît-Marie Robaglia, Marceau Coupechoux, Dimitrios Tsilimantos, Apostolos Destounis |
| 发表 | EuCNC/6G Summit 2023，pp. 623–628；DOI [10.1109/EuCNC/6GSummit58263.2023.10188325](https://doi.org/10.1109/EuCNC/6GSummit58263.2023.10188325) |
| 原始来源 | [作者主页 PDF](https://marceaucoupechoux.wp.imt.fr/files/2023/04/eucnc23.pdf)；[作者机构书目](https://researchportal.ip-paris.fr/fr/publications/seqdqn-multi-agent-deep-reinforcement-learning-for-uplink-urllc-w/) |
| 本地版本 | [SeqDQN-EuCNC2023.pdf](SeqDQN-EuCNC2023.pdf)，6 页，作者公开稿，未逐字比对出版社终版 |
| 阅读 | 2026-10-10；摘要、§II 系统/业务/Dec-POMDP、§III 算法、§IV 设置与结论；未复现 |

## 原文做了什么

单 BS 上行、单信道碰撞模型；每设备单包缓冲，异构周期与相位偏移，支持概率周期到达和确定性周期到达。设备只决定发送或等待；执行观测为自身包龄及 BS 上一时隙的成功/碰撞/空闲三值反馈。包在截止期前可重传，不累积跨时隙服务比特。

SeqDQN 使用 GRU-DQN，按截止期将设备分组，从短截止期组开始顺序训练，各探索阶段只更新一个设备，其余策略固定。BS 在阶段切换时协调训练身份；“执行去中心化”不等于整个生命周期完全没有中心协调。评估指标是总体按期送达比例，比较 iDQN、QMIX、随机接入和轮询。

## 对 DSA05 当前 V2 的判断

**必读的直接邻近工作**：上行、异步异构周期、硬截止期、本地历史和分布式 MARL 已经存在，不能作为 DSA05 新问题的首创依据。这里的“无设备间通信”仍允许公共 BS 反馈，和 DSA05 L0 的表述应保持一致。

仍可研究的差异是衰落下多子信道与功率、跨时隙服务累积、对其他 UE 紧迫性的低开销摘要，以及最差 UE 违约率；这些差异需要通信机制和实验收益支撑，不能仅列场景参数。

## 建议如何使用

- 优先用于相关工作与信息边界表；将 SeqDQN 的三值反馈与 L0 RSSI、L1 紧迫摘要分别列明。
- 如复现，只能称“SeqDQN 式顺序学习的改编”。把二值接入改为 21 动作且保留自身 CSI，会改变问题；需固定观测、训练交互量和模型选择流程。
- 不建议立刻用它替换 EXP-011：EXP-009 已考察训练侧瓶颈，当前更关键的是摘要内容及接收代价。顺序更新是可能的反证型基线，不是新机制的替代证据。
- 原文的经验优势不构成 DSA05 部分可观测干扰问题的收敛保证。
