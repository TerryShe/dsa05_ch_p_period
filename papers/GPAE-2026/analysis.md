# GPAE-2026：广义逐智能体优势估计

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Generalized Per-Agent Advantage Estimation for Multi-Agent Policy Optimization |
| 作者 | Seongmin Kim, Giseung Park, Woojun Kim, Jiwon Jeon, Seungyul Han, Youngchul Sung |
| 来源 | AAMAS 2026；arXiv:2603.02654 |
| 本地 PDF | [GPAE-2026.pdf](GPAE-2026.pdf)（20 页） |
| 阅读范围 | 摘要、引言、相关工作中对 GAE / COMA / DAE 的比较；方法细节只做了浏览 |

## 方法

- 指出 MAPPO 用 GAE 给所有智能体同一个团队优势，无法区分各自的贡献；COMA 只用一步估计，方差和偏差都大。
- 提出 GPAE：用逐智能体的价值迭代算子，通过动作概率间接估计每个智能体的价值，从而得到多步的逐智能体优势，不需要直接学 Q 函数。再用双截断的重要性采样比，支持离策略数据。
- 文中还提到 Difference Advantage Estimation（DAE），也是差分式的优势估计，**尚未阅读**。

## 与 DSA05 的关系

- 它诊断的问题与我们在 EXP-DSA05-002 中的推测一致：团队共享优势无法把"让路"的功劳归给让路的智能体。
- 它是通用的 MARL 算法改进，不利用环境模型；我们的场景在训练时有廉价、精确的模拟器，可以直接算反事实，这是两者的根本差别。
- 如果 EXP-DSA05-003 有效，GPAE 可以作为一个不依赖模型的逐智能体优势基线来对比。

## 引用建议

- 相关工作：MAPPO 团队优势在信用分配上的局限，以及逐智能体优势估计的最新进展。
