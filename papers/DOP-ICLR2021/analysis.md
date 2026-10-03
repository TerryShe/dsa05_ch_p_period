# DOP-ICLR2021：用分解式 Critic 降低信用估计难度

## 基本信息

| 项目 | 内容 |
|---|---|
| 发表标题 | DOP: Off-Policy Multi-Agent Decomposed Policy Gradients |
| 作者 | Yihan Wang, Beining Han, Tonghan Wang, Heng Dong, Chongjie Zhang |
| 来源 | ICLR 2021；本地为 arXiv:2007.12322v2（2020-10-04）预印本 |
| 官方会议 PDF | [OpenReview](https://openreview.net/pdf/3f81a57de106341a8eae39f9a550980090df7241.pdf)；本次网页可读取，但直接下载返回 403 |
| 实际 PDF 来源 | [作者 arXiv 版本](https://arxiv.org/pdf/2007.12322v2) |
| 本地 PDF | [DOP-ICLR2021.pdf](DOP-ICLR2021.pdf)，20 页；页首仍标注 Preprint / Under review，未与会议终版逐项比对 |
| 代码 | [作者实现](https://github.com/TonghanWang/DOP) |
| 阅读范围 | §3 失配分析、§4 分解 Critic 与 off-policy 梯度、§5 实验概览，PDF 第 3–8 页；附录证明未完整阅读 |
| 本次归档 | 2026-10-02 |

## 论文的核心内容

DOP 把集中式 Q 写成非负加权的逐智能体 Critic 之和：

\[
Q_{\mathrm{tot}}(\tau,a)
=\sum_i k_i(\tau)Q_i(\tau,a_i)+b(\tau).
\]

该结构降低了离策略期望计算的复杂度，并给各策略提供不同的信用。注意 Q_i 的 τ 是集中训练的联合历史，不是只能看 UE i 的本地历史。原文讨论表示偏差与梯度方差的权衡，在 SMAC 和 MPE 上评估随机与确定性策略版本。

## 不增加 OBS 的兼容性

Actor 可继续只看本地历史；所有 Q_i、权重与 replay 都用于集中训练。这里“逐智能体 Critic”不等于要把这个 Critic 部署到 UE，也不等于 UE 需要知道其他人的状态。

对于共享 Actor，逐 UE 的训练信用仍可不同，参数梯度再累加到同一个网络。因此无需为了不同信用增加 UE ID 或独立策略。

## 对 DSA05 的价值

1. DSA05 有 21 类动作，线性分解后可以对每个 UE 的所有动作做有限求和。比枚举全部联合动作有明显的计算结构优势。
2. 它提供了第三类信用路线：既不是即时奖励差分，也不是直接模拟所有反事实，而是改变训练 Critic 的结构。
3. 对混合 N，需要可变集合输入和共享输出 head；论文原网络不能作为现成的无 ID、变长 UE 实现。
4. 单标量团队 V 与逐 UE 动作价值之间存在结构差别。接入后应验证信用确实区分了动作，而不是只复制同一个团队信号。

## 迁移风险

**线性动作分解可能难以表达同信道碰撞与互相避让。** 联合价值的交互项并不一定可由上述结构精确表示。论文给出的是特定条件下的策略改进分析，不是任意无线 Dec-POMDP 的无偏 Q 表示保证。

离策略复用还涉及其他 UE 的策略变化和循环 hidden state。若直接用旧 hidden state 更新当前网络，可能引入额外失配。实验必须把 Critic 分解收益与 replay 收益分开。

仅降低平均团队违约仍不能保证所有 UE 达标；风险或公平目标需要另行定义。Q_i 也不能当作 UE i 的真实包违约率。

## 建议的最小比较（本课题推演）

先做同预算的 on-policy 分解式 Critic，对照 MAPPO 与 GPAE；若价值误差和样本效率有改善，再加入原方法的 replay 机制。报告动作价值预测、各 UE 优势差异、误差随 N 的变化和每训练时隙的计算成本。

如果方法因分解约束压低高负载表现，可保留为失败对照，不能通过隐含新增邻居 OBS 修补后继续声称信息边界不变。

## 新颖性与引用建议

**参考优先级：中高，实施优先级低于 GPAE on-policy。** 它是把价值分解用于策略梯度的核心引用。可用来支撑信用估计结构与计算复杂度比较；不宜声称“首次分解式策略梯度”。

综合路线见[固定 OBS 文献综述](../../docs/research/literature-fixed-obs-marl_20261002.md)。
