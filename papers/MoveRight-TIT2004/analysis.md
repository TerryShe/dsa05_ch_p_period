# MoveRight-TIT2004：多址信道上的能耗最优离线调度

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | On Adaptive Transmission for Energy Efficiency in Wireless Data Networks |
| 作者 | Elif Uysal-Biyikoglu, Abbas El Gamal |
| 来源 | IEEE Transactions on Information Theory, vol. 50, no. 12, 2004 |
| 本地 PDF | [MoveRight-TIT2004.pdf](MoveRight-TIT2004.pdf)（作者主页版本，14 页） |
| 阅读范围 | 摘要、引言、第 II 节（已有结果回顾）、第 III 节多址信道部分（Lemma 1–2、Theorem 1、FlowRight 算法） |

## 问题与方法

- 把单链路 lazy scheduling（见 [LazySched-ToN2002](../LazySched-ToN2002/analysis.md)）推广到多址（上行）、广播和衰落信道：多个用户的包在 [0, T) 内到达，必须在截止时间 T 之前发完，使总发射能量最小。
- **时分（time-division）**：用户在时间上不重叠。早先的 **MoveRight** 算法可以求出最优的时分调度。
- **多址（multiple-access）**：用户同时发送，接收端做多用户解码（连续干扰消除）。
  - Theorem 1：各用户信道增益相同时，时分就能达到最小能量；增益不同时，时分严格次优，最优点是多址能量区域的角点，对应按增益顺序连续解码。
  - 多址的离线问题是线性约束下的凸优化，作者提出 **FlowRight** 迭代算法求解。
- 在线启发式：look-ahead water-filling，用一个小缓冲同时适应积压和信道。

## 与 DSA05 的关系

- **DSA05 的中央上界就是一个时分调度**：正交循环分时的 MILP（`solve_orthogonal_cycle`）让同一子信道上的 UE 在时间上错开，相当于 K 个并行子信道上的 MoveRight 式时分问题，再加上离散功率档和周期性。
- **本文说明时分不一定最优**：如果接收端能做连续干扰消除，增益不同的用户同时发送可以更省能。DSA05 的接收机按 TIN（把干扰当噪声）建模，此时同时发送只会损失互信息，所以时分上界在 TIN 下是合理的参照。但论文里应当说明：
  1. 我们的上下界针对 TIN 接收机；
  2. 下界（逐 UE 单独最优）对任何接收机都成立，因为它忽略了所有干扰；
  3. 我们用多址和速率构造的线性规划下界（`scripts/explore_mac_sum_rate_bound.py`，见[上下界报告](../../docs/results/central-bounds_20260924/report.md) §4.2）在 10 个抽查实例中只收紧了 1 个。原因是这个松弛允许类似 SIC 的叠加：在当前 SNR 下，超过 K 个 UE 同时发送几乎不需要额外代价。这与本文"多用户解码下同时发送可以很省能"的结论一致，可以一起讨论。
- **学到的策略的行为与时分一致**：长训后冲突率随 N 上升但远低于基线，占空比与中央上界相近，说明策略在"学时分"。

## 引用建议

- 中央参照：时分上界的理论背景；多址 SIC 下界的讨论。
- 相关工作：多用户能耗最优包调度的经典工作。
