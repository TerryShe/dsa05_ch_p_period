# HCMAPPOL-2026：分层约束 MAPPO-Lagrangian（端边协同 DNN 推理）

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Safe Multi-Agent Deep Reinforcement Learning for Privacy-Aware Edge-Device Collaborative DNN Inference |
| 作者 | Hong Wang, Xuwei Fan, Zhipeng Cheng, Yachao Yuan, Minghui Min, Minghui Liwang, Xiaoyu Xia（苏州大学等） |
| 来源 | arXiv:2603.00129（v1，2026-02-23），IEEE 期刊模板，未见发表处 |
| 本地 PDF | [HCMAPPOL-2026.pdf](HCMAPPOL-2026.pdf)，14 页，arXiv v1 |
| 阅读范围 | 2026-10-14；摘要、§I、§IV 的拉格朗日部分与算法 1 的结构；系统模型与实验只看了概要 |

## 研究问题

端边协同 DNN 推理：联合决定模型部署、用户与服务器关联、模型切分与资源分配，最小化能耗与隐私代价，约束为长期推理时延。建模为约束 MDP。

## 方法

- 三层策略：自回归的模型部署策略；带拉格朗日的用户关联与切分策略；基于注意力的资源分配策略。
- **约束处理**：对时延约束引入一个**共享的**非负拉格朗日乘子 λ，按约束违反程度做投影梯度上升；用户智能体最大化 J_R − λ(J_C − J̄_C)。集中式 Critic。

## 与 DSA05 的关系

- 这是"MAPPO 加一个全局乘子"的典型做法，相当于我们 EXP-006/008 中的变体 B（全局 μ，在迭代之间更新）。它说明这类做法在无线/边缘计算的 MARL 中很常见。
- 不同：约束是平均时延（连续信号），不是逐设备的违约率；集中式 Critic、层次化策略；领域是边缘计算，不是多址接入。
- 关系较远，只作为"共享乘子 MAPPO-L"的例子。

## 引用建议

- 只在需要列举"MAPPO-Lagrangian 在无线/边缘场景中的应用"时引用，不作为主要对比对象。
