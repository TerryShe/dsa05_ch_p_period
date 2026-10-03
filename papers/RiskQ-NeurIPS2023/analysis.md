# RiskQ-NeurIPS2023：风险敏感的集中与分布价值一致性

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization |
| 作者 | Siqi Shen, Chennan Ma, Chao Li, Weiquan Liu, Yongquan Fu, Songzhu Mei, Xinwang Liu, Cheng Wang |
| 来源 | NeurIPS 2023；arXiv:2311.01753 |
| 官方页面 | [NeurIPS](https://papers.nips.cc/paper_files/paper/2023/hash/6d3040941a2d57ead4043556a70dd728-Abstract-Conference.html) |
| PDF 来源 | [会议正式版，含附录](https://papers.nips.cc/paper_files/paper/2023/file/6d3040941a2d57ead4043556a70dd728-Paper-Conference.pdf) |
| 本地 PDF | [RiskQ-NeurIPS2023.pdf](RiskQ-NeurIPS2023.pdf)，35 页 |
| 代码 | [作者实现](https://github.com/xmu-rl-3dv/RiskQ) |
| 阅读范围 | §2 风险定义、§3 相关工作、§4 RIGM 与分位混合、§5 实验框架，PDF 第 2–8 页；附录证明与全部结果未逐项复核 |
| 本次归档 | 2026-10-02 |

## 论文的核心内容

RiskQ 定义 RIGM：各智能体按本地风险值选动作，应与联合分布按相同风险准则选动作一致。一般的均值价值分解不自动满足这种一致性。方法对同一分位水平的逐智能体 utility 做非负加权混合，训练联合回报分布；执行时 GRU 只根据本地历史计算风险值并选动作。论文给出针对 VaR 与 distortion risk measures 的一致性结果，并在风险游戏和 SMAC 上评估。

结构可概括为

\[
\theta_{\mathrm{joint}}(\omega)
=\sum_i k_i\theta_i(\omega),\qquad k_i\ge0.
\]

其保证针对论文定义的混合结构与 utility，不能脱离假设理解为任意联合回报都能精确表示。

## 不增加 OBS 的兼容性

- 执行端的 utility 网络只接收本地观测历史，集中式 mixer 不部署到 UE，符合 CTDE。
- DSA05 的 21 个离散动作适合价值型方案；可以保留自己的 CSI、RSSI 和现有本地历史，替换训练算法。
- 固定风险参数不需要作为额外 OBS。训练时知道 N，与 Actor 执行时不知道 N 可以同时成立。
- 原实现面向固定规模任务。用于混合 N 时，还要处理共享参数、变长集合、无 UE 标识和信道置换结构，不能把带 agent ID 的现成输入直接搬进来。

## 与 DSA05 目标的关键差别

**RiskQ 的逐智能体 utility 不是该 UE 实际违约率的分布。** 若团队奖励仍是每时隙平均违约包数，RiskQ 优化的尾部也仍是这个团队回报的尾部。它不会自动变成“所有 UE 都不超过 1%”。

此外，原文常用的是奖励分布的低尾风险；DSA05 可采用成本分布的高尾风险。二者可通过符号转换关联，但风险参数含义、训练目标和报告名称必须一致。

干扰与避让存在强烈的非单调联合作用：一个 UE 的最好动作取决于别人是否同时发送。RIGM 分解有表示约束；不能因为网络满足 RIGM，就声称它消除了信息限制或精确学到 EDF 的联合决策。

## 可执行的参考方案（本课题推演）

把 RiskQ 作为独立的风险价值型基线，比较三项：相同结构下的风险中性版本、固定风险水平版本、原循环 MAPPO。所有方案使用相同 OBS 和训练实例池，并匹配环境交互预算、调参预算和评估随机数。

先在开发实例验证分布预测与实际尾部的关系，再决定是否值得长训。指标不能只看回报：必须报告 F_0.01、最差 UE 违约率、业务类别差异和 N*。若完整部署成本需要长轨迹，必须单独处理统计时域，而不是只替换 Q 网络头。

## 新颖性与引用建议

**参考优先级：高，作为近期理论对照的价值高于直接改造全部代码的优先级。** 它是“风险敏感价值分解已经有人做过”的必要引用。本文也解释了为什么直接给 QMIX 换分布输出头不足以建立风险一致性。

不建议把“RiskQ 用于无线接入”作为唯一贡献。更值得研究的是 DSA05 的部署层尾部目标，以及在严格相同 OBS 下，目标对齐能否改善边缘 UE 与承载能力。

参见[RMIX](../RMIX-NeurIPS2021/analysis.md)和[固定 OBS 文献综述](../../docs/research/literature-fixed-obs-marl_20261002.md)。
