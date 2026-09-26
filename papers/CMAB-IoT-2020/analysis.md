# CMAB-IoT-2020：多玩家老虎机下的去中心化无冲突信道分配

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Decentralized Learning for Channel Allocation in IoT Networks over Unlicensed Bandwidth as a Contextual Multi-player Multi-armed Bandit Game |
| 作者 | Wenbo Wang, Amir Leshem, Dusit Niyato, Zhu Han |
| 来源 | IEEE Transactions on Wireless Communications（arXiv 页面注明，卷期未查）；arXiv:2003.13314（2020 年提交，2021 年修订） |
| 本地 PDF | [CMAB-IoT-2020.pdf](CMAB-IoT-2020.pdf)（32 页，单栏） |
| 阅读范围 | 摘要、引言、相关工作开头 |

## 问题与方法

- **场景**：自组织 IoT 链路以 underlay 方式复用主网络的多个信道。设备感知能力弱，只能从自己用过的信道得到部分观测，信道质量还受主用户活动影响。
- **建模**：上下文多玩家多臂老虎机博弈。设备是玩家，信道是臂，主用户活动是上下文。
- **方法**：纯去中心化的三阶段策略学习，通过试错使所有链路共同收敛到社会最优、无冲突的信道分配，遗憾为多对数量级（次线性）。

## 与 DSA05 的关系

- **"不通信也能学会无冲突分配"的理论支撑**：DSA05 在 N ≤ K（N=4）时，学到的策略几乎没有冲突，每个 UE 占用不同子信道，与精确最优相差 1% 以内。多玩家老虎机文献为这种现象提供了收敛与遗憾的理论依据。
- **主要差别**：
  1. 老虎机没有状态：每次选臂的收益只取决于当时的选择；DSA05 有截止期、剩余比特与周期结构，是有状态的序贯决策；
  2. 无功率控制，目标为吞吐；
  3. DSA05 在 N > K 时不存在无冲突的静态分配，必须在时间上错开，这超出了多玩家老虎机的"一人一臂"设定。

## 引用建议

- 相关工作：去中心化、无信令的信道分配学习；可在 N=4 结果的讨论中引用。
