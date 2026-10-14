---
doc_id: LIT-DSA05-RL-ENGINEERING
type: literature-note
date: 2026-10-14
status: draft
scope: 为"用 RL 解决一个实际通信工程问题"寻找可站住的创新点；检验"可靠性欠账驱动的 MARL"是否已被覆盖
---

# 文献检索：可靠性欠账驱动的 MARL 与可能的工程问题（2026-10-14）

本文是检索记录，不是论文文本。

**检索范围与局限**：先读本库 `papers/` 中已有的笔记，再做约 16 组网页检索（其中 6 组专门补查欠账/虚拟队列/逐智能体约束的 MARL，新下载并分析 9 篇，见 `papers/README.md` 2026-10-14 一节）（通用搜索引擎，含 arXiv、IEEE、3GPP 文档服务器上的结果），并读了 3 篇的原文或摘要页。没有系统检索 IEEE Xplore 与 Scopus。"没有找到"只说明在这一范围内没有找到，不等于不存在。

## 1. 结论

1. **"可靠性欠账（逐设备约束状态）驱动的 MARL"本身不能作为创新点**。核心机制已有先例，其中三篇本库早已收录（EXP-008 时期）：
   - SALA（TSP 2022）：逐用户对偶变量作为策略输入、执行时在线更新，用于无线资源管理（集中式 GNN）；
   - HeLyMARL（arXiv 2026-08）：逐智能体虚拟队列同时进入奖励与观测，执行时在线更新（观测包含其他智能体的队列，需要信息交换）；
   - LyMARL-DS（arXiv 2024-11）：逐用户时延违约概率约束、虚拟队列加 MARL（底层由集中式调度器执行）。
   - 本库 HeLyMARL 笔记已写明："论文不能宣称'逐智能体约束状态进入观测'是我们的创新。"2026-10-14 我在提出这一方向前没有先查本库，这是疏漏。
2. **仍可能站住的只是组合与场景**：只用本地信息、匿名共享 Actor、事件触发的违约率约束、干扰耦合的上行多址与衰落。这属于"在新场景中应用已知机制"，单独撑不起期刊层次的方法贡献。
3. **检索中发现一个未被覆盖、且符合"RL 解决实际工程问题"的方向：生存时间（survival time）约束下的分布式周期上行接入**（§3）。在本次范围内，没有找到用 RL 处理生存时间或连续丢包的分布式上行接入工作。3GPP Rel-17 的标准做法是一个固定的反应式规则（失败后激活 PDCP 复制）。

## 2. 已覆盖的方向（不能作为创新点）

| 方向 | 代表工作 | 与 DSA05 的关系 |
|---|---|---|
| 逐设备约束状态进入策略（对偶、虚拟队列、欠账） | SALA（TSP 2022）、SACRL（arXiv 2021）、HeLyMARL（arXiv 2026）、LyMARL-DS（arXiv 2024）、FSAL-DVR（arXiv 2025） | 机制相同；区别只在信息结构（本地、匿名）与约束类型（事件触发的违约率） |
| 逐智能体约束与本地乘子的约束 MARL 算法 | [MARLTO](../../papers/MARLTO-2025/analysis.md)（arXiv 2025）、[DPD-CP](../../papers/DPD-CP-2025/analysis.md)（arXiv 2025）、[Scal-MAPPO-L](../../papers/ScalMAPPOL-NeurIPS2024/analysis.md)（NeurIPS 2024）；共享乘子的 [HC-MAPPO-L](../../papers/HCMAPPOL-2026/analysis.md)（arXiv 2026） | 约束 MARL 已有成熟算法；它们依赖转移独立、邻居通信或局部性，约束多为每步都有信号的代价 |
| 虚拟队列（漂移加惩罚）与 RL 的结合 | LyDROO（TWC 2021）、[DPP-RL](../../papers/DPP-RL-2025/analysis.md)（arXiv 2025） | 后者指出逐时隙的漂移加惩罚奖励与 RL 的长期目标不匹配，"欠账 × 违约"的奖励需要专门处理 |
| 按时吞吐量（送达率）约束的约束 DRL | [CRL-Timely](../../papers/CRL-Timely-Asilomar2025/analysis.md)（Asilomar 2025） | 逐业务流乘子在训练轮之间更新；多跳路由与调度 |
| 上行 URLLC、截止期、分布式 MARL | SeqDQN（EuCNC 2023）、[uMRA-HAPPO](../../papers/uMRA-HAPPO-GLOBECOM2025/analysis.md)（GLOBECOM 2025）、REEMAC（TMLCN 2026）、XR-SBPC（WCNC 2025） | 问题组合已有 |
| 最差用户目标、截止期、免授权 | DCGFRA（TCCN 2025） | 目标已有 |
| 完全分布式的公平接入 | [FSRL](../../papers/FSRL-WiOpt2025/analysis.md)（WiOpt 2025） | 只凭自身的成功/碰撞反馈、以 Jain 指数为公平目标；碰撞模型，没有衰落 |
| 多个配置授权上的学习 | GF-NOMA 多 CG 的协作 DDQN（arXiv 2111.09284） | 基站侧分配资源，突发业务；不是 UE 侧在共享 CG 上自主选择 |

**uMRA-HAPPO（GLOBECOM 2025，Telecom Paris，SeqDQN 同组）**，本次读了原文：
- 单 BS 上行；Poisson 到达，每个包有截止期，可重传至截止；每个帧中 UE 选择一组时隙发送副本，BS 做 SIC（现代随机接入）；碰撞模型，没有衰落；
- 目标是总体的按期成功比例（URLLC 分数），不是逐设备；
- **BS 每帧广播全体 UE 估计缓存按剩余寿命的汇总向量**，让 UE 了解即将到来的业务及其紧迫程度。这与我们的紧迫摘要思路相近，相关工作中须正面对比；
- 用 HAPPO，训练时由 BS 协调顺序更新。

## 3. 一个未被覆盖的工程问题：生存时间

### 3.1 工程背景

- **需求**：3GPP 的时间敏感通信（TSC）定义了生存时间：应用能容忍一定的连续消息丢失；超过生存时间（例如连续丢失两个周期的消息），服务即视为不可用（TS 22.104）。工业控制关心的是"不连续丢包"，不只是平均丢包率。
- **标准中的做法**（Rel-17 IIoT，RAN2 讨论稿，终版条款待核对 TS 38.321/38.331）：
  - 生存时间业务用配置授权承载；
  - 一次发送失败（收到重传授权）后，UE 进入"生存时间状态"，对该承载激活 PDCP 复制，后续包在多条链路上重复发送，以避免连续失败；
  - 这是一个固定的反应式规则，代价是复制占用的额外资源；RAN2 曾讨论 N > 1 次连续失败才触发等变体。

### 3.2 为什么适合用 RL

- **连续丢包在衰落下是相关的**：f_D = 10 Hz 时，快类周期两端的信道相关系数约 0.90，中类约 0.64（场景 v2.0 §2.2）。一次失败往往意味着下一次也危险，独立丢包的估计会低估生存时间的违约。
- **失败之后怎么反应是一个序贯决策**：换哪个子信道、用多大功率、何时发送，取决于自己的信道历史、干扰状况和其他 UE 的行为，规则难以兼顾；复制则以固定的资源代价换可靠性。
- **约束是逐设备、依赖历史的**：总失败包数的奖励不区分孤立丢包与连续丢包，所以这里需要新的训练目标，而不只是现有方法换一个评价指标。逐设备的"生存状态"（已连续失败的次数）是有物理含义的本地状态。
- **与现有基础设施吻合**：周期业务、截止期即下一次到达、多子信道与功率、衰落、按 UE 统计；紧迫摘要可以改为携带"处于生存状态"的标志。

### 3.3 相关但不同的工作（须在设计前细查）

- **信息年龄（AoI）**：对周期更新，"生存时间被违反"等价于"年龄超过某个门限"。
  - 已有：峰值年龄违约概率的集中式安全 DRL（[SafeDRL-PAoI](../../papers/SafeDRL-PAoI-2025/analysis.md)，arXiv 2507.08653，多传感器、有限码长；其中的随机 MATI 约束相邻两次成功上报的间隔，与生存时间同类）；衰落下的平均 AoI 多智能体 DRQN（GLOBECOM 2019）；随机接入的年龄违约概率分析（arXiv 2112.01182、2007.10030）；单链路衰落的峰值年龄违约（arXiv 1903.06771）。
  - 本次没有找到：分布式 MARL、多子信道加功率、衰落、以逐设备的年龄违约或生存时间为约束的上行接入。
- **专利**：存在多项降低连续丢包、按剩余生存时间动态调整可靠性目标的专利（非学习方法）。

### 3.4 需要先回答的问题

1. **在我们的场景中，连续丢包是否显著多于独立假设的预测？** 这决定问题是否非平凡。现有评估输出只存每个 UE 的违约总数，不存逐包序列；需要对若干方法重放少量回合并记录逐包结果。代价小（几小时内），只用已读的开发实例。
2. **标准基线怎样在场景中实现**：PDCP 复制相当于同一时隙在两个子信道上发送，现有动作空间是每个时隙一个（子信道, 功率）。忠实实现需要扩展动作，属于场景的新版本；或者在现有动作内实现近似版本（例如进入生存状态后用最好的子信道、满功率）。
3. **指标**：每个设备的生存时间违约次数或可用度，最差设备；生存时间取几个周期须依据 TS 22.104 的用例写定。
4. **新颖性的深入检索**：IEEE Xplore 上"survival time / consecutive packet loss / burst error / peak AoI violation"与"reinforcement learning / multi-agent / random access / configured grant"的组合，以及 Rel-17/18 的 3GPP 终版条款。

## 4. 来源

本库：`papers/SALA-TSP2022`、`papers/SACRL-2021`、`papers/HeLyMARL-2026`、`papers/LyMARL-DS-2024`、`papers/FSAL-DVR-2025`、`papers/SeqDQN-EuCNC2023`、`papers/REEMAC-TMLCN2026`、`papers/DCGFRA-2024`、`papers/HouKumar-INFOCOM2010`。

网页：
- [uMRA-HAPPO，GLOBECOM 2025（作者主页 PDF）](https://marceaucoupechoux.wp.imt.fr/files/2025/11/globecom25.pdf)
- [FSRL，WiOpt 2025（arXiv 2503.24296）](https://arxiv.org/abs/2503.24296)
- [GCN-DQN 工业 URLLC（arXiv 2506.15011）](https://arxiv.org/html/2506.15011v1)：未正式定义概率保证，参考价值低
- [GF-NOMA 多 CG 的 DDQN（arXiv 2111.09284）](https://arxiv.org/html/2111.09284)
- [配置授权综述（arXiv 2210.08940）](https://arxiv.org/pdf/2210.08940)
- [HeLyMARL（arXiv 2608.01745）](https://arxiv.org/html/2608.01745)
- [逐智能体约束的任务卸载 MARL（arXiv 2509.01257）](https://arxiv.org/html/2509.01257)
- [峰值年龄违约的安全 DRL（arXiv 2507.08653）](https://arxiv.org/abs/2507.08653)
- [衰落下的 AoI 多智能体 DRQN（GLOBECOM 2019）](http://wcan.ee.psu.edu/papers/SL_Globecom_19.pdf)
- [短包衰落的峰值年龄违约（arXiv 1903.06771）](https://arxiv.org/pdf/1903.06771)
- [优先级随机接入的 AoI（arXiv 2112.01182）](https://arxiv.org/pdf/2112.01182)
- [IRSA 的 AoI（arXiv 2007.10030）](https://arxiv.org/pdf/2007.10030)
- [3GPP 38.824 §8.2.1：每个 BWP 多个有效 CG 配置](https://www.itecspec.com/3gpp/38.824/s/8.2.1)
- [RAN2 #117-e 生存时间讨论稿](https://3gpp.org/ftp/tsg_ran/wg2_rl2/TSGR2_117-e/Inbox/Drafts/%5BOffline-504%5D%5BIIoT%5D%20QoS%20additional%20open%20issues%20(Nokia)/%5BDRAFT%5D%20R2-22xxxxx%20Summary%20of%20%5BAT117-e%5D%5B504%5D%5BIIoT%5D_V0-Rapp.docx)
- [RAN2 #115-e 生存时间状态的基线 CR 文本提案](https://www.3gpp.org/ftp/Email_Discussions/RAN2/%5BRAN2%23115-e%5D/%5BPost115-e%5D%5B513%5D%5BIIoT%5D%20QoS%20survival%20time%20(Huawei)/Phase%202/R2-210xxxx%20TP%20of%20baseline%20CR%20for%20Survival%20Time%20state%20operation_v05_Rapp_mark.docx)
- [3GPP 工业 5G 页面](https://www.3gpp.org/technologies/ind-5g)
