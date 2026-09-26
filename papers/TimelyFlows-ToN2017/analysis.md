# TimelyFlows-ToN2017：一般业务模式下的按时吞吐量容量区域

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Timely Wireless Flows with General Traffic Patterns: Capacity Region and Scheduling Algorithms |
| 作者 | Lei Deng, Chih-Chun Wang, Minghua Chen, Shizhen Zhao |
| 来源 | IEEE/ACM Transactions on Networking, vol. 25, no. 6, 2017（会议版：INFOCOM 2016） |
| 本地 PDF | [TimelyFlows-ToN2017.pdf](TimelyFlows-ToN2017.pdf)（作者主页版本，15 页） |
| 阅读范围 | 摘要、引言（三个基本问题与贡献表）、相关工作、系统模型与业务模式 |

## 问题与方法

- Hou 与 Kumar 的框架（见 [HouKumar-INFOCOM2010](../HouKumar-INFOCOM2010/analysis.md)）只适用于**帧同步**的业务：所有流同一周期、同时到达、同时过期。本文处理**一般业务模式**。
- **业务模型**：每个流由 (offset, prd, D, B) 描述：
  - offset 为到达过程的起始偏移；
  - prd 为到达间隔（周期）；
  - D 为截止期；
  - B 为每个周期到达的概率。
- **结果**：
  - 按时吞吐量问题本质上是一个无限时域 MDP；
  - 按时容量区域是有限的凸多面体；
  - 给出网络效用最优、可行性最优的调度器 RAC（维度灾难），以及低复杂度的 RAC-Approx 和 L-LDF；
  - 指出最大债务优先（LDF）对一般业务模式**不是**可行性最优的。
- 场景：单跳下行，AP 每个时隙只调度一条链路，删除信道（成功概率 p_k），有 ACK。

## 与 DSA05 的关系

- **DSA05 的业务正是本文的"一般业务模式"的确定性特例**：每个 UE 有自己的相位（offset）、周期 P ∈ {10, 20, 40} ms、截止期 D = P、到达概率 B = 1。这为"异步、异构周期"的建模提供了直接的文献依据。
- **"LDF 在一般业务模式下不是可行性最优"的结论**提示：DSA05 的异步周期结构让简单的优先级规则失效，需要考虑时间结构。这可以作为使用带记忆（GRU）的策略、而不是简单优先级规则的一个理由。
- **主要差别**：
  1. 本文是 AP 集中调度、单信道、一个时隙一条链路；DSA05 是上行分布式、K 个子信道、多个 UE 可能同时在同一子信道上发送并相互干扰；
  2. 本文的目标是按时吞吐量（可行性或效用），不计能耗；DSA05 在交付率约束下最小化能耗，功率是决策变量；
  3. 本文用 MDP 精确求解，维度灾难；DSA05 的中央参照用 MILP 求循环时分上界，另用单独最优作下界。

## 引用建议

- 系统模型：异步周期业务（offset、周期、截止期）的建模依据。
- 相关工作：截止期调度从帧同步到一般业务模式的发展。
