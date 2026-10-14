# DRL-WNCS-2021：分布式无线网络控制系统的 DRL 传输调度

## 基本信息

| 项目 | 内容 |
|---|---|
| 标题 | Deep Reinforcement Learning for Wireless Scheduling in Distributed Networked Control |
| 作者 | Gaoyang Pang, Kang Huang, Daniel E. Quevedo, Branka Vucetic, Yonghui Li, Wanchun Liu（悉尼大学等） |
| 来源 | arXiv:2109.12562（v1 2021-09，v5 2025-05），arXiv 页面注明已被 IEEE Transactions on Cybernetics 接收 |
| 本地 PDF | [DRL-WNCS-2021.pdf](DRL-WNCS-2021.pdf)，14 页，arXiv v5 |
| 阅读范围 | 2026-10-14；摘要与引言；系统模型与算法只看了概要 |

## 研究问题与方法

- 完全分布式的无线网络控制系统（多个被控对象，传感器、控制器、执行器分布在不同位置），频率信道数有限，**联合调度上行与下行**。
- 用随机系统理论给出同时依赖控制与通信参数的稳定性充分条件；满足时存在稳定所有对象的平稳确定性调度策略。
- 把每步代价表示为有限长的可数向量状态（与每个回路自上次成功更新以来的时间有关，类似信息年龄），建成 MDP，用 DRL（DQN、DDPG、TD3）求解，并提出动作空间缩减与动作嵌入。

## 与 DSA05 的关系

- "分布式"指的是被控系统的物理分布，**调度本身是集中式的**（一个调度器为全部回路分配信道）。
- 状态中的"距上次成功的时间"刻画了连续失败的影响，是生存时间方向在控制领域的背景。
- 区别：集中式调度、没有设备自主竞争与干扰耦合。

## 引用建议

- 相关工作：控制导向的无线调度与 DRL；作为集中式参照。
