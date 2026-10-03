# RMIX-NeurIPS2021：本地历史驱动的动态风险策略

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | RMIX: Learning Risk-Sensitive Policies for Cooperative Reinforcement Learning Agents |
| 作者 | Wei Qiu, Xinrun Wang, Runsheng Yu, Rundong Wang, Xu He, Bo An, Svetlana Obraztsova, Zinovi Rabinovich |
| 来源 | NeurIPS 2021；arXiv:2102.08159 |
| 官方页面 | [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/c2626d850c80ea07e7511bbae4c76f4b-Abstract.html) |
| PDF 来源 | [会议正式版](https://papers.nips.cc/paper/2021/file/c2626d850c80ea07e7511bbae4c76f4b-Paper.pdf) |
| 本地 PDF | [RMIX-NeurIPS2021.pdf](RMIX-NeurIPS2021.pdf)，14 页 |
| 代码 | [作者实现](https://github.com/yetanotherpolicy/rmix) |
| 阅读范围 | §2 风险定义、§3 CVaR 与风险预测器、§4 理论框架、§5 实验设置，PDF 第 2–8 页；附录证明未逐项复核 |
| 本次归档 | 2026-10-02 |

## 论文的核心内容

RMIX 学习逐智能体回报 utility 的分布，以低尾 CVaR 选择动作，再通过 QMIX 风格的单调 mixer 训练。动态风险水平预测器根据当前与历史分布表示调整风险偏好；本地分布用 CVaR 值作为辅助伪奖励训练。集中式混合值本身不是全局 CVaR，原文 §3.3 明确说明了这一点。实验包括多智能体 cliff navigation 与 SMAC。

## 不增加 OBS 的兼容性

原文的风险预测器使用本地历史和内部表示，因此**不需要额外广播风险水平**。如果 DSA05 只冻结原始 OBS，这种内部计算兼容；如果还冻结内部网络接口，则可先选择固定风险水平的版本，或只改变 GRU 后的训练结构。

不能把训练时知道的最紧迫 UE、真实 N 或联合信道状态作为执行端的风险参数。那会越过当前边界，与本文从本地历史生成风险参数不是一回事。

## 与 RiskQ 的关系

RMIX 是有参考价值的早期风险合作方法，但不能把各 UE 的 CVaR 经单调混合后的标量称为“团队回报分布的 CVaR”。RiskQ 进一步讨论了这种集中与分布风险一致性的不足。

这说明需要区分三个对象：真实逐 UE 的违约率；用于价值分解的逐智能体 utility；部署层“所有 UE 合格”的风险。三者没有自动的数值对应。

## 在 DSA05 中怎样使用（本课题推演）

更建议把它用作风险策略基线，而非首个主方法。比较风险中性分布式价值网络、固定 CVaR 网络和动态风险网络；保持相同 Actor 观测来源、共享结构、训练池和环境时隙预算。

如果动态风险有效，进一步检查风险偏好是否随自己的服务裕量、RSSI 历史和截止期变化。即使观察到关联，也不能据此断言它知道其他 UE 的真实紧迫性。

## 迁移风险

1. 低尾团队回报可能鼓励保守但低效的动作，未必改善最差 UE 的包违约率。奖励与可靠性目标须先对齐。
2. 极度避险可能减少 OFF，让更多 UE 同时满功率发送，反而恶化干扰。风险敏感并不天然等于合作避让。
3. 参数更多，伪奖励也引入新的学习动态；必须比较固定风险和等参数对照，避免把容量增加解释为风险机制。
4. 原论文的风险预测器旨在缓解时间一致性问题，不能把它当作对任意部分可观测无线问题的时间一致性证明。
5. 迁移到无 ID、可变 N 时，需要重新设计共享 utility 与 mixer；不应直接沿用 SMAC 的全部输入字段。

## 新颖性与引用建议

**参考优先级：中高，实施优先级低于目标对齐与 GPAE。** 它证明从本地历史调整风险偏好已有先例，也提供风险中性/固定风险/动态风险三层消融思路。

可用于相关工作及风险价值型基线。若主方法仍是 MAPPO，优先借鉴风险对象的分析，避免仅因“较新框架”就整体替换已经有协调收益的策略。

参见[RiskQ](../RiskQ-NeurIPS2023/analysis.md)和[固定 OBS 文献综述](../../docs/research/literature-fixed-obs-marl_20261002.md)。
