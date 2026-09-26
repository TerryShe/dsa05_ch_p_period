---
doc_id: LIT-DSA05-003
type: literature-note
date: 2026-09-26
related: [DES-DSA05-PERIODIC, EXP-DSA05-001, EXP-DSA05-003, EXP-DSA05-004]
---

# 问题侧的相关工作与 DSA05 的定位

前两份笔记（[变体 C](literature-variant-c_20260925.md)、[反事实信用](literature-counterfactual-credit_20260925.md)）围绕方法。本笔记补上问题本身这一侧，为论文的引言、相关工作和贡献陈述做准备。2026-09-26 检索，读了 17 篇；同日又核对了 dsa04 的 [AP-DECA 文献集](../../../dsa_marl_test04_mcs_p_subchannel/papers/AP-DECA/README.md)，补入其中 3 篇（XR-SBPC、PT-CD、FRL-EE），其余 3 篇与本库重复。PDF 与逐篇分析在 [`papers/`](../../papers/README.md)。每篇只读了摘要、引言、系统模型和主要结论，没有逐一核对证明与实验细节。

## 1. 四条相关研究线

### 1.1 能耗最优的截止期调度（lazy scheduling）

| 工作 | 核心结论 | 与 DSA05 的关系 |
|---|---|---|
| [LazySched](../../papers/LazySched-ToN2002/analysis.md)（ToN 2002） | 发送能量是传输时长的凸减函数，最优调度尽量拉长、摊平发送 | DSA05 的单 UE 最优（中央下界）是它的离散版本：一个包、一个周期、离散功率档 |
| [MoveRight](../../papers/MoveRight-TIT2004/analysis.md)（TIT 2004） | 多用户时分的最优离线调度（MoveRight）；增益不同且允许多用户解码时，时分严格次优 | DSA05 的中央上界是时分调度；TIN 接收机下合理，SIC 下不是下界。解释了我们的 MAC 和速率下界为何过松 |
| [CalcEE](../../papers/CalcEE-ToN2009/analysis.md)（ToN 2009） | 截止期、缓冲区等 QoS 约束统一为最小发送曲线；最优发送曲线是"绷紧的绳子" | 单 UE 问题中 D_min(t) = A(t − P)，最优是周期内恒定低速率；离散功率档下需要枚举 |
| [FRL-EE](../../papers/FRL-EE-2010/analysis.md)（arXiv 2010；TSP 2011、TMC 2013） | 单链路节能控制，能耗计入电路开关与切换功率；分离已知与未知动态，后决策状态加虚拟经验使学习快两个数量级 | 说明"只计发射能耗"这一假设的影响；样本效率的思路，但多智能体下"未知动态与动作无关"不成立 |

**这条线都是单发送端或集中式的**：前三篇是离线最优，FRL-EE 是单链路在线学习；都没有分布式决策，也没有同信道干扰。它给 DSA05 提供了"为什么要少发、低功率、长时间发"的理论依据，以及中央参照的背景。

### 1.2 截止期约束的无线调度（timely throughput）

| 工作 | 核心结论 | 与 DSA05 的关系 |
|---|---|---|
| [Hou 与 Kumar](../../papers/HouKumar-INFOCOM2010/analysis.md)（INFOCOM 2010） | 帧同步周期业务、每客户端交付率要求；最大债务优先类策略可行性最优 | DSA05 的每 UE 违约率约束就是交付率要求；"债务"与变体 C 的逐 UE 对偶同源 |
| [TimelyFlows](../../papers/TimelyFlows-ToN2017/analysis.md)（ToN 2017） | 一般业务模式 (offset, 周期, 截止期, 到达概率)；按时容量区域是凸多面体；LDF 不再可行性最优 | DSA05 的异步异构周期是它的确定性特例，是建模的直接依据 |
| [Singh 与 Kumar](../../papers/DecDeadline-CDC2015/analysis.md)（CDC 2015） | 把硬的同时发送约束放松为平均约束，拉格朗日价格把问题分解为逐包决策，完全去中心化 | 没有干扰时价格分解就够了；DSA05 的难点正是干扰使决策无法分解 |
| [TSRA](../../papers/TSRA-2022/analysis.md)（arXiv 2022） | 截止期约束下基于 R-learning 的随机接入，用 MDP 上界衡量差距 | 同样的问题骨架，但为碰撞信道、单信道、无功率控制、只有一个学习设备 |

**这条线以 AP 集中调度或单信道碰撞模型为主，目标是可行性或按时吞吐量，不计能耗。**

### 1.3 免授权与配置授权的周期上行

| 工作 | 核心结论 | 与 DSA05 的关系 |
|---|---|---|
| [CG-URLLC](../../papers/CG-URLLC-2021/analysis.md)（IEEE ComStd Mag 2021） | NR 配置授权：为周期小包预配置周期性资源，可配多次重复与冗余版本 | 周期上行业务的工程依据；中央上界可以解释为"理想的集中配置"；重复加冗余版本对应 HARQ-IR 累积 |
| [DCGFRA](../../papers/DCGFRA-2024/analysis.md)（TCCN，arXiv 2024） | 截止期约束的免授权接入：分布式导频选择加连续功率控制；利用模型构造可微目标，样本效率高于无模型 MARL | 目标期刊上最接近的问题之一；目标是最差用户丢包率而非能耗，每个包一个时隙，决策近似为逐时隙问题 |
| [PT-CD](../../papers/PT-CD-TWC2024/analysis.md)（TWC 2024） | LPWAN 周期业务相位异步，易产生持续碰撞；网关预测碰撞并集中分配偏移与信道 | 周期结构既是难点也可利用；中央上界相当于理想的集中偏移分配；提示利用周期性的规则基线 |

### 1.4 MARL 的频谱接入与功率控制

| 工作 | 核心结论 | 与 DSA05 的关系 |
|---|---|---|
| [XR-SBPC](../../papers/XR-SBPC-WCNC2025/analysis.md)（WCNC 2025） | 体内子网下行，XR 帧带偏移与抖动周期到达，时延预算内按 Shannon 速率累积服务；OFF 加子频带加离散功率；SAC-Discrete 集中训练分布执行；观测含邻居上一帧动作 | **问题结构最接近**；目标为可靠性、不计能耗，干扰网络而非单 AP 上行，观测比 DSA05 多 |
| [MARLV2X](../../papers/MARLV2X-JSAC2019/analysis.md)（JSAC 2019） | V2V 智能体选子带加 4 档功率（含关），周期负载须在时间预算内按累积容量送达；指纹 DQN | **动作与交付要求最接近**；但回合同步、目标为容量加交付率、假设即时干扰 CSI、无最优参照 |
| [RLPC](../../papers/RLPC-2022/analysis.md)（IoT-J，arXiv 2022） | 单链路、周期包、截止期为下一个周期、离散功率，可靠性约束下最小化期望功率；拉格朗日奖励 | **目标与约束完全同型的单链路版本**；DSA05 是它的多用户、有干扰、分布式推广 |
| [MARLSPA](../../papers/MARLSPA-TCOM2025/analysis.md)（TCOM 2025） | 循环 MAPPO 做多小区子带与功率分配，最小化时延，接近集中式优化 | 算法最接近；下行 AP 智能体、个体奖励、邻域信息交换 |
| [DMRL-DSA](../../papers/DMRL-DSA-TWC2019/analysis.md)（TWC 2019） | 只凭 ACK、LSTM 加 DQN 的多信道分布式接入 | 早期代表；碰撞信道、饱和业务、无功率控制 |
| [MADRL-PA](../../papers/MADRL-PA-JSAC2019/analysis.md)（JSAC 2019） | 分布式执行的 DQN 功率控制，接近 WMMSE/FP | 单频带、饱和业务、需要邻居信息交换 |
| [DLMA](../../papers/DLMA-JSAC2019/analysis.md)（JSAC 2019） | DRL 节点学会与 TDMA/ALOHA 共存 | 学习利用他人的时间结构；单信道、单个学习节点 |
| [CMAB-IoT](../../papers/CMAB-IoT-2020/analysis.md)（TWC，arXiv 2020） | 多玩家老虎机：不通信也能收敛到无冲突的信道分配 | 为 N ≤ K 时学到无冲突分配提供理论依据；无状态、无功率 |
| [DOL-TC](../../papers/DOL-TC-2026/analysis.md)（arXiv 2026） | 6G 工业子网中告警的分布式学习接入，中心广播竞争签名 | 关系较远；事件驱动、碰撞信道 |

## 2. DSA05 的定位

在我们检索到的范围内，**没有工作同时具备下面六点**。DSA05 的问题新颖性应当按这个组合来陈述，而不是声称其中某一点是首创。

**需要注意**：XR-SBPC（WCNC 2025）已经同时具备第 2、3 点的主体：带偏移的周期到达、跨时隙累积 Shannon 服务、OFF 加子频带加离散功率，并且用分布式 MARL 求解。所以"异步周期业务 + 子信道与功率联合选择 + MARL"这一组合本身**不是**新问题。DSA05 的区别主要落在第 1 点（单 AP 上行多址、严格本地因果信息；XR-SBPC 同样把干扰当噪声，TIN 本身不是区别）、第 4 点（能耗目标加逐 UE 约束）以及第 5、6 点（最优参照与测试协议）。

1. **上行、完全分布式**：UE 只用本地因果信息，执行时不交换信息，只有 AP 广播的逐子信道公共 RSSI 和自己的互信息反馈；
2. **子信道加离散功率的联合动作**，接收端 TIN，同信道干扰按 SINR 建模，一个包跨时隙 HARQ-IR 累积互信息；
3. **异步、异构周期业务**，截止期为本 UE 的下一次到达；
4. **在每 UE 交付率约束下最小化能耗**；
5. **给出中央参照**：N ≤ K 时精确最优，N > K 时给出单独最优下界与正交时分上界，并报告与它们的差距；
6. **按冻结协议在独立测试集上做统计判读**。

与最接近的四篇的区别需要在论文中写明：

| 最接近的工作 | 它有的 | 它没有的（DSA05 有） |
|---|---|---|
| XR-SBPC（WCNC 2025） | 带偏移的周期业务、截止期内累积 Shannon 服务、OFF 加子频带加功率档、分布式 DRL | 能耗目标与逐 UE 约束、单 AP 上行多址、不观测他人动作与信道、最优参照、测试协议 |
| MARLV2X（JSAC 2019） | 子带加功率档、时间预算内累积送达、MARL | 异步异构周期、能耗目标与逐 UE 约束、只用本地因果信息、最优参照 |
| RLPC（IoT-J） | 周期包、下一周期截止、离散功率、可靠性约束下最小功率 | 多用户、干扰、子信道选择、分布式协调 |
| DCGFRA（TCCN） | 上行免授权、截止期、资源选择加功率、分布式 | 能耗目标、跨时隙累积、周期业务、最优参照 |

## 3. 对论文和后续工作的影响

- **贡献陈述的重心**：问题表述、中央参照和严格评价，而不是约束式 MAPPO 本身。先前的方法侧检索已经确认，变体 C 的机制不新；测试集上 A 与 C 也没有显著差异。可以写的发现包括：
  - 长训后策略学会了在时间上错开发送，占空比与中央上界相近；
  - 熵退火与训练预算对可行性影响很大；
  - N=4 时恢复了精确最优的结构；
  - 负载越高，节能越多，离上界也越远；
  - 残余失败集中在远端、长周期的 UE。
- **中央参照的写法**：用 LazySched 与 CalcEE 解释单独最优下界，用 MoveRight 解释时分上界。并说明上界针对 TIN 接收机；允许 SIC 时同时发送可能更省能，这也是我们的 MAC 和速率下界过松的原因。
- **业务模型的依据**：用 TimelyFlows 的一般业务模式和 CG-URLLC 的配置授权说明异步周期业务的合理性。
- **EXP-DSA05-003（如果继续）**：DCGFRA 用已知模型构造可微目标来提高样本效率。这与"利用已知模拟器做反事实"方向相同，重写设计时必须引用并区分。
- **可以考虑的对照实验**（需要另行立项，暂不做）：
  - 用 MARLV2X 的奖励设计（送完后给常数奖励、不计能耗）训练同一网络，说明能耗目标与约束处理的必要性；
  - RLPC 式的单 UE 拉格朗日策略，在多用户下作为"不协调"的基线；
  - **逐 UE 固定功率档的消融**（dsa04 的 AP-DECA 分析提出）：每个 UE 只选子信道和开关，功率固定为一档，检验时变功率是否确有收益。DSA05 的速率随功率连续变化（Shannon），与 AP-DECA 的固定速率门限模型不同，时变功率的收益更可能存在，但目前没有直接证据；
  - **利用周期相位的规则基线**（PT-CD 的启示）：成功后保持同一相位与子信道，连续失败后随机重选。

  测试集已经使用，上述任何新方法或新基线只能在验证集上开发，最终比较需要新的测试计划或启用预留实例集。
- **能耗口径**：论文需要把"只计发射能耗"写成明确假设，并用 FRL-EE 说明计入电路与开关功耗后最优策略可能从"慢发"转向"快发完再休眠"。

## 4. 检索到但未读全文的工作

- **Hou、Borkar、Kumar，"A Theory of QoS for Wireless"**（INFOCOM 2009）：交付率要求框架的原始论文。机构库拒绝访问，没有找到公开全文；内容通过 Hou 与 Kumar（INFOCOM 2010）和 TimelyFlows 的回顾了解。
- **Huang、Wong、Schober，"Throughput Optimization for Grant-Free Multiple Access with Multiagent Deep Reinforcement Learning"**（IEEE TWC 2021）：免授权接入中的多智能体导频选择。没有找到 arXiv 版本。
- **Zhang 等，"Deep reinforcement learning for throughput improvement of the uplink grant-free NOMA system"**（IEEE IoT-J 2020）：未检索全文。
- **Caire 与 Tuninetti，"The throughput of hybrid-ARQ protocols for the Gaussian collision channel"**（IEEE TIT 2001）：HARQ-IR 累积互信息模型的经典出处，系统模型中应当引用。没有找到公开全文。
- **Jeannerot 等，"Joint Slot and Power Optimization for Grant Free Random Access with Unknown and Heterogeneous Device Activity"**（arXiv:2407.18809）：免授权接入中时隙与功率联合优化，SIC 接收机。只读了摘要。
- **Uyoata 等，"Learning Power Control Protocol for In-Factory 6G Subnetworks"**（EuCNC 2025，arXiv:2505.05967）：MAPPO 学习信令与功率控制协议。只读了摘要。
- **Learn2MAC**：TSRA 引用的在线学习多址方案，号称兼顾时延保证与低能耗，尚未查到出处与全文。

## 5. 仍然缺少的

- 近年的综述，用于引言中的概括性引用，例如 MARL 用于无线资源管理的综述。
- 衰落信道下截止期约束的最小能耗，为 DSA05 的衰落扩展做准备。Zafer 与 Modiano 的衰落版本（作者主页 J53、J58）已找到链接，还没有下载。
- 能耗模型中的电路功耗：FRL-EE 覆盖了单链路情形；多用户、周期业务下电路功耗的影响仍缺文献。
