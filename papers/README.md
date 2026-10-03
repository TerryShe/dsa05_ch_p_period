# 文献库

每篇论文一个子文件夹，文件夹名是论文的缩写加年份。文件夹里有论文 PDF（从会议/出版社公开页面、arXiv 或作者主页下载）和 `analysis.md`（针对本课题的阅读分析）。阶段性的综合阅读笔记放在 `docs/research/`：[变体 C 的相关工作与新颖性判断](../docs/research/literature-variant-c_20260925.md)、[反事实信用分配的相关工作](../docs/research/literature-counterfactual-credit_20260925.md)、[问题侧的相关工作与 DSA05 的定位](../docs/research/literature-problem-side_20260926.md)、[固定 OBS 下的 MARL 路线与优先级](../docs/research/literature-fixed-obs-marl_20261002.md)。

文件夹名中的年份：期刊或会议年份确定时用发表年份，否则用 arXiv 首次提交的年份。

少数 PDF 的来源限制再分发（例如机构库封面或 IEEE 页脚注明仅限个人使用），这些 PDF 只保留本地副本，列在 [`.gitignore`](.gitignore) 中，不纳入 Git；对应的 `analysis.md` 注明了下载地址。

## 固定 OBS 的 MARL 算法研究（2026-10-02 检索）

新增 11 篇 PDF 与分析，并补读已有 GPAE。重点是训练目标、尾部覆盖、本地历史表示与信用分配；执行仍只使用现有本地信息。详细适配、阅读范围与实验建议见各篇笔记及[综合分析](../docs/research/literature-fixed-obs-marl_20261002.md)。SOTO 原方案需要执行通信，仅借鉴其公平目标；WCMASAC 需重新核对执行信息与共享参数结构。

| 文件夹 | 论文/方法 | 来源 | 与本课题的关系 |
|---|---|---|---|
| [MAPPO-NeurIPS2022](MAPPO-NeurIPS2022/analysis.md) | Yu 等，The Surprising Effectiveness of PPO | NeurIPS 2022 | 价值归一化、Critic 和训练预算；先建立稳定基线 |
| [CoherentPG-NeurIPS2015](CoherentPG-NeurIPS2015/analysis.md) | Tamar 等，Policy Gradient for Coherent Risk Measures | NIPS 2015 | 部署尾部目标与正确的风险梯度；区分 1% 包门槛和 5% 部署尾部 |
| [LIAM-NeurIPS2021](LIAM-NeurIPS2021/analysis.md) | Papoudakis 等，Agent Modelling under Partial Observability | NeurIPS 2021 | 用训练监督改善本地历史表示；可只监督现有 GRU |
| [PLR-ICML2021](PLR-ICML2021/analysis.md) | Jiang 等，Prioritized Level Replay | ICML 2021 | 优先采集有学习潜力的困难训练部署，保持 OBS |
| [DAE-ICML2022](DAE-ICML2022/analysis.md) | Li 等，Difference Advantage Estimation | ICML 2022 | 多步逐 UE 信用与策略偏差权衡 |
| [DOP-ICLR2021](DOP-ICLR2021/analysis.md) | Wang 等，Off-Policy Multi-Agent Decomposed Policy Gradients | ICLR 2021；本地 arXiv v2 | 分解式训练 Critic 与数据复用；预印本未逐项比对终版 |
| [RiskQ-NeurIPS2023](RiskQ-NeurIPS2023/analysis.md) | Shen 等，Risk-sensitive MARL Value Factorization | NeurIPS 2023 | 风险敏感局部/联合价值一致性；风险价值型对照 |
| [RMIX-NeurIPS2021](RMIX-NeurIPS2021/analysis.md) | Qiu 等，Learning Risk-Sensitive Policies | NeurIPS 2021 | 从本地历史动态调整风险偏好 |
| [WCMASAC-AAAI2026](WCMASAC-AAAI2026/analysis.md) | Liu 等，Safe MARL via Distributional Safety Critic | AAAI 2026 | 分布成本 Critic 与 CVaR 约束；需改造适配 |
| [SOTO-ICML2021](SOTO-ICML2021/analysis.md) | Zimmer 等，Learning Fair Policies in Decentralized Cooperative MARL | ICML 2021 | 公平福利梯度；原方案执行共享邻居回报，不能原样采用 |
| [RUDDER-NeurIPS2019](RUDDER-NeurIPS2019/analysis.md) | Arjona-Medina 等，Return Decomposition for Delayed Rewards | NeurIPS 2019 | 截止期延迟信用；需区分时间/智能体归因与折扣守恒 |
| [GPAE-2026](GPAE-2026/analysis.md)（本次补读） | Kim 等，Generalized Per-Agent Advantage Estimation | AAMAS 2026 | 逐 UE 训练 Critic 与多步优势；优先做 on-policy 对照 |

## 方法侧（约束处理、状态增广、信用分配）

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
| [COMA-2018](COMA-2018/analysis.md) | Foerster 等，Counterfactual Multi-Agent Policy Gradients | AAAI 2018 | 反事实基线的起点；学联合 Q，按自己的策略边缘化 |
| [DrReinforce-2021](DrReinforce-2021/analysis.md) | Castellini 等，Difference Rewards Policy Gradients | AAMAS 2021 / NCA 2022 | 奖励已知时的即时奖励差分；最接近 EXP-DSA05-003 的对照 |
| [GPAE-2026](GPAE-2026/analysis.md) | Kim 等，Generalized Per-Agent Advantage Estimation | AAMAS 2026 | 不依赖模型的逐智能体多步优势 |

## 问题侧（2026-09-26 检索）

| 文件夹 | 论文 | 来源 | 与本课题的关系 |
|---|---|---|---|
| [LazySched-ToN2002](LazySched-ToN2002/analysis.md) | Uysal-Biyikoglu 等，Energy-Efficient Packet Transmission Over a Wireless Link | IEEE/ACM ToN 2002 | lazy scheduling：单 UE 最优（中央下界）的思想来源 |
| [MoveRight-TIT2004](MoveRight-TIT2004/analysis.md) | Uysal-Biyikoglu 与 El Gamal，On Adaptive Transmission for Energy Efficiency | IEEE TIT 2004 | 多用户时分的最优离线调度；SIC 下时分次优，解释中央上界与 MAC 下界 |
| [CalcEE-ToN2009](CalcEE-ToN2009/analysis.md) | Zafer 与 Modiano，A Calculus Approach to Energy-Efficient Data Transmission | IEEE/ACM ToN 2009 | 截止期约束下最小能耗的"绷紧的绳子"解；单 UE 问题的一般框架 |
| [HouKumar-INFOCOM2010](HouKumar-INFOCOM2010/analysis.md) | Hou 与 Kumar，Scheduling Heterogeneous Real-Time Traffic | IEEE INFOCOM 2010 | 交付率要求与"债务"调度；逐 UE 违约约束与变体 C 的来源 |
| [TimelyFlows-ToN2017](TimelyFlows-ToN2017/analysis.md) | Deng 等，Timely Wireless Flows with General Traffic Patterns | IEEE/ACM ToN 2017 | 一般（异步）业务模式的按时容量区域；异步周期建模的依据 |
| [DecDeadline-CDC2015](DecDeadline-CDC2015/analysis.md) | Singh 与 Kumar，Decentralized Throughput Maximizing Policies | IEEE CDC 2015 | 拉格朗日价格分解得到去中心化截止期调度；不含干扰 |
| [TSRA-2022](TSRA-2022/analysis.md) | Deng 等，RL for Improved Random Access in Delay-Constrained Networks | arXiv 2022 | 截止期约束下的 RL 随机接入；碰撞信道、单信道 |
| [DCGFRA-2024](DCGFRA-2024/analysis.md) | Bai、Chen、Larsson，Delay-Constrained Grant-Free Random Access in MIMO | IEEE TCCN（已接收） | 免授权、截止期、分布式导频选择加功率；目标为最差丢包率；模型辅助提高样本效率 |
| [CG-URLLC-2021](CG-URLLC-2021/analysis.md) | Gerami 与 Singh，Configured Grant for URLLC | IEEE ComStd Magazine 2021 | 周期上行业务的工程依据；中央上界即理想的集中配置 |
| [MARLV2X-JSAC2019](MARLV2X-JSAC2019/analysis.md) | Liang、Ye、Li，Spectrum Sharing in Vehicular Networks Based on MARL | IEEE JSAC 2019 | **动作与交付要求最接近**：子带加功率档、时间预算内累积送达 |
| [RLPC-2022](RLPC-2022/analysis.md) | Guo 等，RL Based Power Control for Reliable Mission-Critical Transmission | IEEE IoT-J（已接收） | **目标与约束同型的单链路版本**：周期包、离散功率、可靠性约束下最小功率 |
| [MARLSPA-TCOM2025](MARLSPA-TCOM2025/analysis.md) | Zhang 与 Guo，MARL for Multi-Cell Spectrum and Power Allocation | IEEE TCOM 2025 | 循环 MAPPO 做子带加功率，算法最接近 |
| [DMRL-DSA-TWC2019](DMRL-DSA-TWC2019/analysis.md) | Naparstek 与 Cohen，Deep Multi-User RL for Distributed DSA | IEEE TWC 2019 | 只凭 ACK 的多信道分布式接入，LSTM 加 DQN |
| [MADRL-PA-JSAC2019](MADRL-PA-JSAC2019/analysis.md) | Nasir 与 Guo，Multi-Agent DRL for Dynamic Power Allocation | IEEE JSAC 2019 | 分布式执行的 DRL 功率控制，接近 WMMSE/FP |
| [DLMA-JSAC2019](DLMA-JSAC2019/analysis.md) | Yu、Wang、Liew，Deep-RL Multiple Access | IEEE JSAC 2019 | DRL MAC 学会利用他人的时间结构 |
| [CMAB-IoT-2020](CMAB-IoT-2020/analysis.md) | Wang 等，Decentralized Learning for Channel Allocation in IoT | IEEE TWC（arXiv 注明） | 多玩家老虎机：不通信也能学到无冲突分配（对应 N=4 结果） |
| [DOL-TC-2026](DOL-TC-2026/analysis.md) | Abdelrahman 等，Distributed Online Learning for Time-Critical Communication | arXiv 2026 | 6G 工业子网的分布式学习接入；关系较远 |

## 来自 dsa04 AP-DECA 文献集（2026-09-26 核对）

dsa04 的 [`papers/AP-DECA/`](../../dsa_marl_test04_mcs_p_subchannel/papers/AP-DECA/README.md) 收录了 6 篇。其中 3 篇与本库重复（LPS = LazySched-ToN2002，V2V-MARL = MARLV2X-JSAC2019，MC-SPA = MARLSPA-TCOM2025，文件逐字节相同），另外 3 篇补入如下：

| 文件夹 | 论文 | 来源 | 与本课题的关系 |
|---|---|---|---|
| [XR-SBPC-WCNC2025](XR-SBPC-WCNC2025/analysis.md) | Bagherinejad 等，DRL-based Distributed Joint Sub-band Allocation and Power Control for XR | IEEE WCNC 2025 | **问题结构最接近**：带偏移的周期业务、跨时隙累积 Shannon 服务、OFF 加子频带加功率档、分布式 DRL；目标为可靠性，不计能耗（PDF 仅本地） |
| [PT-CD-TWC2024](PT-CD-TWC2024/analysis.md) | Kaburaki 等，Adaptive Resource Allocation Utilizing Periodic Traffic and Clock Drift | IEEE TWC 2024 | 周期业务导致持续碰撞；集中式预测并分配偏移与信道；启示利用周期性的规则基线（PDF 仅本地） |
| [FRL-EE-2010](FRL-EE-2010/analysis.md) | Mastronarde 与 van der Schaar，Fast RL for Energy-Efficient Wireless Communications | arXiv 2010（相关正式版 TSP 2011、TMC 2013） | 分离已知与未知动态提高学习效率；能耗计入电路与开关功率 |

## 尚缺的文献

- MARL 用于无线资源管理的近年综述，用于引言的概括性引用；
- 衰落信道下截止期约束的最小能耗（Zafer 与 Modiano 的衰落版本已找到链接，未下载），为衰落扩展做准备；
- 电路功耗对能耗最优调度的影响：FRL-EE-2010 部分覆盖（单链路，计入开关与切换功率），多用户场景仍缺；
- 检索到但未读全文的工作（Hou、Borkar、Kumar 2009，Huang 等 TWC 2021，Caire 与 Tuninetti 2001 等）列在[问题侧笔记](../docs/research/literature-problem-side_20260926.md) §4。
