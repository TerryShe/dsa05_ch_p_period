# CalcEE-ToN2009：用累积曲线求截止期约束下的最小能耗

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | A Calculus Approach to Energy-Efficient Data Transmission with Quality-of-Service Constraints |
| 作者 | Murtaza Zafer, Eytan Modiano |
| 来源 | IEEE/ACM Transactions on Networking, 2009 |
| 本地 PDF | [CalcEE-ToN2009.pdf](CalcEE-ToN2009.pdf)（作者主页预印本，14 页） |
| 阅读范围 | 摘要、引言、系统模型、第 III 节的问题表述与"绷紧的绳子"解 |

## 问题与方法

- 单个发送端，功率—速率函数 g(r) 凸增（例如 Shannon 公式给出的每比特功率），可以连续调速率。
- 用**累积曲线**描述数据流：到达曲线 A(t)，发送曲线 D(t)，以及由 QoS 约束导出的**最小发送曲线** D_min(t)。因果性和 QoS 约束统一写成 D_min(t) ≤ D(t) ≤ A(t)。
  - 逐包截止期 d：D_min(t) = A(t − d)；
  - 缓冲区上限 B：D_min(t) = max(A(t) − B, 0)；
  - 网络演算中的服务曲线也可以这样表示。
- **最优解**：在时不变的功率—速率函数下，最优发送曲线是夹在 D_min 与 A 之间的最短路径，即"绷紧的绳子"。这统一并推广了 lazy scheduling 与 MoveRight 等早先的结果。
- 另有在线启发式，以及时变功率—速率函数（衰落）的推广。

## 与 DSA05 的关系

- **DSA05 的单 UE 问题正好是这个框架的一个特例**：周期 P、截止期 = 下一次到达，所以 D_min(t) = A(t − P)，A 与 D_min 是一对错开一个周期的阶梯。最优的"绷紧的绳子"在每个周期内是一条直线，即以恒定低速率把包摊满整个周期。
- **离散化后的差别**：DSA05 的功率档是离散的、时隙是 1 ms，所以最优解不是恒定速率，而是若干时隙上功率档的组合。这正是我们要逐包枚举的原因（`optimal_packet_plan`）。本文的连续解可以作为下界的直观解释。
- **多用户仍然缺失**：本文与 lazy scheduling 一样只有一个发送端。

## 引用建议

- 系统模型或中央参照：截止期约束下最小能耗的一般框架。
- 可以在"单 UE 最优"一段与 [LazySched-ToN2002](../LazySched-ToN2002/analysis.md) 一起引用。
