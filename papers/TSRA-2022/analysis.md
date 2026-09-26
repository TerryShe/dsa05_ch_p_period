# TSRA-2022：截止期约束下的强化学习随机接入

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Reinforcement Learning for Improved Random Access in Delay-Constrained Heterogeneous Wireless Networks |
| 作者 | Lei Deng, Danzhou Wu, Zilong Liu, Yijin Zhang, Yunghsiang S. Han |
| 来源 | arXiv:2205.02057（2022）。脚注称初步版本发表于 IEEE GLOBECOM 2021 Workshop；arXiv 页面标注它与 arXiv:2103.14917（两用户情形，同一组作者）有大量文本重叠。期刊出处未查到 |
| 本地 PDF | [TSRA-2022.pdf](TSRA-2022.pdf)（36 页，单栏预印本） |
| 阅读范围 | 摘要、引言与相关工作 |

## 问题与方法

- 上行随机接入，多个设备通过**碰撞信道**向 AP 发送有截止期的业务：同一时隙两个设备同时发送就都失败。
- 从两设备问题入手：设备 1 使用 ALOHA；设备 2 不知道设备 1 的协议，要最大化系统的按时吞吐量。
  - 用 MDP 得到基于模型的上界，用来衡量学习方案与最优的差距；
  - 截止期为 1 时推导出设备 2 的最优策略（二元策略）。
- 提出 **TSRA**（tiny state-space R-learning random access）：用平均回报的 R-learning，状态空间很小；可推广到多设备。
- 与 DLMA（见 [DLMA-JSAC2019](../DLMA-JSAC2019/analysis.md)）和 Learn2MAC 比较，TSRA 的按时吞吐量更高、计算量更低、**发送功耗更低**。

## 与 DSA05 的关系

- **同样的问题骨架**：上行、截止期、随机接入、分布式学习。本文把"学习得到的接入策略比 ALOHA 类协议更好"推进到了有截止期的场景。
- **主要差别**：
  1. 碰撞信道 vs DSA05 的 SINR 与累积互信息（冲突不一定失败，只是互信息减少）；
  2. 单信道、无功率控制 vs DSA05 的 K 个子信道与 6 档功率；
  3. 目标是按时吞吐量，功耗只是副产品 vs DSA05 在交付率约束下以能耗为目标；
  4. 本文只有一个学习设备与已知协议设备共存；DSA05 所有 UE 都在学习，是完全合作的多智能体问题。
- **用 MDP 上界衡量学习方案的差距**与 DSA05 用中央上下界衡量 MARL 的做法一致，可以在方法论上引用。

## 引用建议

- 相关工作：截止期约束下基于强化学习的随机接入。
- 评价方法：用模型上界量化学习方案与最优的差距。
